/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/initialization/DefaultInitializer.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_oc/synchronized_module/ReferenceManager.h"
#include "ocs2_sqp/SqpMpc.h"
#include "ocs2_sqp/SqpSettings.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"
#include "humanoid_common_mpc/parameter_update/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_config/mpc_parameter_update.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

/**
 * The mechanics of MpcParameterUpdaterModule, the live-tuning machinery both formulations share, with fake appliers on a
 * solver without terms: which update a solve applies, when the watched files are reloaded, how a refusing reloader or a
 * throwing applier leaves the rest to run, what a reload reports, and in which order the appliers run. What the real
 * appliers write is the formulations' tests' (humanoid_centroidal_mpc:testMpcParameterUpdaterModule,
 * testLiveUpdateEqualsFreshStart; humanoid_wb_mpc:test_wb_live_tuning).
 */
namespace ocs2::humanoid {
namespace {

constexpr size_t kInputDim = 2;

/** What the fake appliers did, in order, as "<applier>:<event>"; thread-safe. */
class EventLog {
 public:
  void record(std::string event) {
    absl::MutexLock lock(&mutex_);
    events_.push_back(std::move(event));
  }
  std::vector<std::string> events() const {
    absl::MutexLock lock(&mutex_);
    return events_;
  }
  void clear() {
    absl::MutexLock lock(&mutex_);
    events_.clear();
  }

 private:
  mutable absl::Mutex mutex_;
  std::vector<std::string> events_ ABSL_GUARDED_BY(mutex_);
};

/** An applier that records what it is called with: "<name>:apply <state_weights.scaling>" and "<name>:beforeEverySolve". */
class FakeApplier final : public HotFieldApplier {
 public:
  /** What apply() does besides recording. */
  enum class Behavior {
    kRecord,
    // Throws after recording, as an OCS2 term API does on a term of another type.
    kThrow,
  };

  FakeApplier(std::string name, std::vector<absl::string_view> fields, EventLog* absl_nonnull log, Behavior behavior)
      : name_(std::move(name)), fields_(std::move(fields)), log_(log), behavior_(behavior) {}

  absl::string_view name() const override { return name_; }
  absl::Span<const absl::string_view> fields() const override { return fields_; }
  void apply(HotUpdateTarget& target) override {
    log_->record(absl::StrCat(name_, ":apply ", target.task().state_weights.scaling));
    if (behavior_ == Behavior::kThrow) throw std::runtime_error("a term of another type");
  }
  void beforeEverySolve(SqpSolver& /*solver*/, const SwitchedModelReferenceManager* absl_nullable /*referenceManager*/) override {
    log_->record(absl::StrCat(name_, ":beforeEverySolve"));
  }

 private:
  std::string name_;
  std::vector<absl::string_view> fields_;
  EventLog* absl_nonnull log_;
  Behavior behavior_;
};

/** A solver without terms, which is all the module needs to hand its appliers a target. */
struct TermlessMpc {
  TermlessMpc() : initializer(kInputDim), mpc(mpc::Settings(), settings(), problem, initializer) {}

  static sqp::Settings settings() {
    sqp::Settings sqpSettings;
    sqpSettings.nThreads = 1;
    sqpSettings.enableLogging = false;
    return sqpSettings;
  }

  OptimalControlProblem problem;
  DefaultInitializer initializer;
  SqpMpc mpc;
};

/** A directory of the test's own, emptied. */
std::filesystem::path freshDirectory(absl::string_view name) {
  const std::filesystem::path directory = std::filesystem::path(testing::TempDir()) / std::string(name);
  std::error_code error;
  std::filesystem::remove_all(directory, error);
  std::filesystem::create_directories(directory, error);
  EXPECT_FALSE(error) << error.message();
  return directory;
}

/** Writes `text` to `path` with a modification time `seconds` after now, so that every write is seen. */
void writeWatched(const std::filesystem::path& path, absl::string_view text, int seconds) {
  {
    std::ofstream out(path, std::ios::trunc);
    out << text;
  }
  std::error_code error;
  std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(seconds), error);
  ASSERT_FALSE(error) << error.message();
}

/** Runs the pre-solve hook of `updater` once. */
void solve(MpcParameterUpdaterModule& updater) {
  const ReferenceManager referenceManager;
  updater.preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, vector_t::Zero(1), referenceManager);
}

/** Drives `updater` past its ~1 Hz poll of the watched files (every 100th pre-solve hook). */
void runFileWatch(MpcParameterUpdaterModule& updater) {
  for (size_t i = 0; i < 100; ++i) solve(updater);
}

// The command limits a reference file must give (referenceSettingsFromConfig()), but max_rotation_velocity.
constexpr char kReferenceFileLimits[] =
    "target_displacement_velocity: 0.5\n"
    "target_rotation_velocity: 0.5\n"
    "max_displacement_velocity_x: 1.0\n"
    "max_displacement_velocity_y: 0.5\n"
    "max_delta_pelvis_height: 0.1\n"
    "default_base_height: 0.9\n";

/** An update whose task file has a state_weights.scaling of `scaling`. */
mpc_config::MpcParameterUpdate updateWithScaling(scalar_t scaling) {
  mpc_config::MpcParameterUpdate update;
  update.task.state_weights.scaling = scaling;
  return update;
}

/** The fixture: a termless MPC, an event log and the options of an updater watching nothing. */
class MpcParameterUpdaterModuleMechanicsTest : public ::testing::Test {
 protected:
  /** The updater of the termless MPC with `appliers` and `options`'s files. */
  std::unique_ptr<MpcParameterUpdaterModule> makeUpdater(std::vector<std::unique_ptr<HotFieldApplier>> appliers,
                                                         MpcParameterUpdaterModule::Options options = {}) {
    options.inputDim = kInputDim;
    options.appliers = std::move(appliers);
    absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> created = MpcParameterUpdaterModule::Create(&mpc_.mpc, std::move(options));
    EXPECT_TRUE(created.ok()) << created.status();
    return created.ok() ? *std::move(created) : nullptr;
  }

  /** A recording applier `name` declaring `fields`. */
  std::unique_ptr<HotFieldApplier> fake(std::string name,
                                        std::vector<absl::string_view> fields = {},
                                        FakeApplier::Behavior behavior = FakeApplier::Behavior::kRecord) {
    return std::make_unique<FakeApplier>(std::move(name), std::move(fields), &log_, behavior);
  }

  /** The appliers `fake("first")` and `fake("second")`. */
  std::vector<std::unique_ptr<HotFieldApplier>> twoFakes() {
    std::vector<std::unique_ptr<HotFieldApplier>> appliers;
    appliers.push_back(fake("first"));
    appliers.push_back(fake("second"));
    return appliers;
  }

  /** The apply events of the log. */
  std::vector<std::string> applies() const {
    std::vector<std::string> found;
    for (const std::string& event : log_.events()) {
      if (absl::StrContains(event, ":apply")) found.push_back(event);
    }
    return found;
  }

  TermlessMpc mpc_;
  EventLog log_;
};

TEST_F(MpcParameterUpdaterModuleMechanicsTest, TheAppliedFieldsAreTheSortedUnionOfTheAppliersFields) {
  std::vector<std::unique_ptr<HotFieldApplier>> appliers;
  appliers.push_back(fake("first", {"state_weights", "multiple_shooting.g_max"}));
  appliers.push_back(fake("second", {"input_weights", "state_weights"}));
  appliers.push_back(fake("third"));
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(std::move(appliers));
  ASSERT_NE(updater, nullptr);
  EXPECT_THAT(updater->appliedFields(), testing::ElementsAre("input_weights", "multiple_shooting.g_max", "state_weights"));
}

TEST_F(MpcParameterUpdaterModuleMechanicsTest, ANullApplierIsRefused) {
  std::vector<std::unique_ptr<HotFieldApplier>> appliers = twoFakes();
  appliers.push_back(std::unique_ptr<HotFieldApplier>());
  MpcParameterUpdaterModule::Options options;
  options.appliers = std::move(appliers);
  const absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> created =
      MpcParameterUpdaterModule::Create(&mpc_.mpc, std::move(options));
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(created.status().message(), testing::HasSubstr("applier 2 of 3 is null"));
}

TEST_F(MpcParameterUpdaterModuleMechanicsTest, TheNewestEnqueuedUpdateIsTheOneTheNextSolveApplies) {
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(twoFakes());
  ASSERT_NE(updater, nullptr);
  updater->enqueueParameterUpdate(updateWithScaling(/*scaling=*/2.0));
  updater->enqueueParameterUpdate(updateWithScaling(/*scaling=*/3.0));
  EXPECT_THAT(applies(), testing::IsEmpty()) << "an enqueued update was applied before the next solve";
  solve(*updater);
  // Every applier, in order, once, with the newest file.
  EXPECT_THAT(applies(), testing::ElementsAre("first:apply 3", "second:apply 3"));
  log_.clear();
  solve(*updater);
  EXPECT_THAT(applies(), testing::IsEmpty()) << "an applied update was applied again";
}

TEST_F(MpcParameterUpdaterModuleMechanicsTest, AnUpdateEnqueuedWhileASolveAppliesOneIsAppliedByTheNextSolve) {
  // The first applier holds its apply() until another thread has enqueued an update: the enqueue may not wait for the
  // solve, and the solve applies the update it took, not the one that came in meanwhile.
  absl::Notification applying;
  absl::Notification enqueued;
  /** An applier that waits inside apply() for `enqueued`. */
  class BlockingApplier final : public HotFieldApplier {
   public:
    BlockingApplier(absl::Notification* absl_nonnull applying, absl::Notification* absl_nonnull enqueued, EventLog* absl_nonnull log)
        : applying_(applying), enqueued_(enqueued), log_(log) {}
    absl::string_view name() const override { return "blocking"; }
    absl::Span<const absl::string_view> fields() const override { return {}; }
    void apply(HotUpdateTarget& target) override {
      log_->record(absl::StrCat("blocking:apply ", target.task().state_weights.scaling));
      if (!applying_->HasBeenNotified()) applying_->Notify();
      enqueued_->WaitForNotification();
    }

   private:
    absl::Notification* absl_nonnull applying_;
    absl::Notification* absl_nonnull enqueued_;
    EventLog* absl_nonnull log_;
  };
  std::vector<std::unique_ptr<HotFieldApplier>> appliers;
  appliers.push_back(std::make_unique<BlockingApplier>(&applying, &enqueued, &log_));
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(std::move(appliers));
  ASSERT_NE(updater, nullptr);
  updater->enqueueParameterUpdate(updateWithScaling(/*scaling=*/2.0));
  std::thread producer([&updater, &applying, &enqueued]() {
    applying.WaitForNotification();
    updater->enqueueParameterUpdate(updateWithScaling(/*scaling=*/5.0));
    enqueued.Notify();
  });
  solve(*updater);
  producer.join();
  EXPECT_THAT(applies(), testing::ElementsAre("blocking:apply 2"));
  solve(*updater);
  EXPECT_THAT(applies(), testing::ElementsAre("blocking:apply 2", "blocking:apply 5")) << "the update enqueued during the solve was lost";
}

TEST_F(MpcParameterUpdaterModuleMechanicsTest, AWatchedTaskFileIsReloadedOncePerChange) {
  const std::filesystem::path taskFile = freshDirectory("updater_mechanics_task") / "task.textproto";
  writeWatched(taskFile, "state_weights { scaling: 1.5 }\n", /*seconds=*/1);
  MpcParameterUpdaterModule::Options options;
  options.taskFile = taskFile.string();
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(twoFakes(), std::move(options));
  ASSERT_NE(updater, nullptr);
  runFileWatch(*updater);
  runFileWatch(*updater);
  EXPECT_THAT(applies(), testing::IsEmpty()) << "an untouched task file was reloaded";
  writeWatched(taskFile, "state_weights { scaling: 2.5 }\n", /*seconds=*/2);
  runFileWatch(*updater);
  runFileWatch(*updater);
  runFileWatch(*updater);
  EXPECT_THAT(applies(), testing::ElementsAre("first:apply 2.5", "second:apply 2.5")) << "one change, one reload";
}

TEST_F(MpcParameterUpdaterModuleMechanicsTest, ARefusingReferenceFileReloaderDoesNotStopTheOthers) {
  const std::filesystem::path referenceFile = freshDirectory("updater_mechanics_reference") / "reference.textproto";
  writeWatched(referenceFile, absl::StrCat(kReferenceFileLimits, "max_rotation_velocity: 0.5\n"), /*seconds=*/1);
  MpcParameterUpdaterModule::Options options;
  options.referenceFile = referenceFile.string();
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(twoFakes(), std::move(options));
  ASSERT_NE(updater, nullptr);
  std::vector<scalar_t> reloaded;
  updater->addReferenceFileReloader(
      [](const ReferenceSettings& /*settings*/) { return absl::InvalidArgumentError("refused by the first"); });
  updater->addReferenceFileReloader([&reloaded](const ReferenceSettings& settings) {
    reloaded.push_back(settings.maxRotationVelocity);
    return absl::OkStatus();
  });
  writeWatched(referenceFile, absl::StrCat(kReferenceFileLimits, "max_rotation_velocity: 0.75\n"), /*seconds=*/2);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("Failed to reload"), testing::HasSubstr("refused by the first"))))
      .Times(1);
  log.StartCapturingLogs();
  runFileWatch(*updater);
  runFileWatch(*updater);
  log.StopCapturingLogs();
  EXPECT_EQ(reloaded, std::vector<scalar_t>({0.75})) << "the reloader after the refusing one did not run, or ran twice";
  EXPECT_THAT(applies(), testing::IsEmpty()) << "a reference file reached the task file's appliers";
}

TEST_F(MpcParameterUpdaterModuleMechanicsTest, AThrowingApplierIsLoggedAndTheNextOneRuns) {
  std::vector<std::unique_ptr<HotFieldApplier>> appliers;
  appliers.push_back(fake("throwing", /*fields=*/{}, FakeApplier::Behavior::kThrow));
  appliers.push_back(fake("next"));
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(std::move(appliers));
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       testing::AllOf(testing::HasSubstr("throwing"), testing::HasSubstr("stopped part-way"),
                                      testing::HasSubstr("a term of another type"))))
      .Times(1);
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr("Successfully applied"))).Times(0);
  log.StartCapturingLogs();
  updater->enqueueParameterUpdate(updateWithScaling(/*scaling=*/4.0));
  solve(*updater);
  log.StopCapturingLogs();
  EXPECT_THAT(applies(), testing::ElementsAre("throwing:apply 4", "next:apply 4"));
}

TEST_F(MpcParameterUpdaterModuleMechanicsTest, AStartUpOnlyEditChangesNoHotFieldAndIsReportedByPath) {
  MpcParameterUpdaterModule::Options options;
  options.runningTask.state_weights.scaling = 1.25;
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(twoFakes(), std::move(options));
  ASSERT_NE(updater, nullptr);
  mpc_config::MpcParameterUpdate update = updateWithScaling(/*scaling=*/1.25);
  update.task.multiple_shooting.dt = 0.02;
  update.task.contact_schedule_source = "contact_planner";
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("2 field(s)"), testing::HasSubstr("next start"),
                                      testing::HasSubstr("multiple_shooting.dt"), testing::HasSubstr("contact_schedule_source"))))
      .Times(1);
  log.StartCapturingLogs();
  updater->enqueueParameterUpdate(update);
  solve(*updater);
  log.StopCapturingLogs();
  // The appliers see the running hot values: nothing they write changes.
  EXPECT_THAT(applies(), testing::ElementsAre("first:apply 1.25", "second:apply 1.25"));

  // A hot-only edit reports nothing.
  absl::ScopedMockLog quiet(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(quiet, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr("next start"))).Times(0);
  quiet.StartCapturingLogs();
  updater->enqueueParameterUpdate(updateWithScaling(/*scaling=*/7.0));
  solve(*updater);
  quiet.StopCapturingLogs();
}

TEST_F(MpcParameterUpdaterModuleMechanicsTest, TheStartUpReportNamesWhatThisMpcReadsAndTheRobotsFieldsApart) {
  // A whole-body MPC: the centroidal model is the other formulation's, telemetry_frequency the robot process's, and
  // enable_online_tuning the GUI's own switch, which no one applies.
  MpcParameterUpdaterModule::Options options;
  options.layout.mpc = StateInputLayout::Mpc::kWholeBody;
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(twoFakes(), std::move(options));
  ASSERT_NE(updater, nullptr);
  mpc_config::MpcParameterUpdate others;
  others.task.centroidal_model = "single_rigid_body_dynamics";
  others.task.telemetry_frequency = 20.0;
  others.task.enable_online_tuning = !others.task.enable_online_tuning;
  {
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr("next start"))).Times(0);
    EXPECT_CALL(log,
                Log(absl::LogSeverity::kInfo, testing::_,
                    testing::AllOf(testing::HasSubstr("1 field(s) the robot process reads"), testing::HasSubstr("telemetry_frequency"))))
        .Times(1);
    log.StartCapturingLogs();
    updater->enqueueParameterUpdate(others);
    solve(*updater);
    log.StopCapturingLogs();
  }

  // A field the whole-body MPC reads at its start is reported, alone.
  mpc_config::MpcParameterUpdate own = others;
  own.task.costs.emplace_back("joint_torque_cost");
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("1 field(s)"), testing::HasSubstr("MPC's next start"), testing::HasSubstr("costs"),
                                      testing::Not(testing::HasSubstr("centroidal_model")))))
      .Times(1);
  log.StartCapturingLogs();
  updater->enqueueParameterUpdate(own);
  solve(*updater);
  log.StopCapturingLogs();
}

TEST_F(MpcParameterUpdaterModuleMechanicsTest, EveryApplierRunsBeforeEverySolveFirstAndInOrder) {
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(twoFakes());
  ASSERT_NE(updater, nullptr);
  solve(*updater);
  EXPECT_THAT(log_.events(), testing::ElementsAre("first:beforeEverySolve", "second:beforeEverySolve"));
  log_.clear();
  updater->enqueueParameterUpdate(updateWithScaling(/*scaling=*/2.0));
  solve(*updater);
  EXPECT_THAT(log_.events(), testing::ElementsAre("first:beforeEverySolve", "second:beforeEverySolve", "first:apply 2", "second:apply 2"));
}

TEST_F(MpcParameterUpdaterModuleMechanicsTest, WithoutAnMpcNothingIsAppliedAndNothingThrows) {
  MpcParameterUpdaterModule::Options options;
  options.appliers = twoFakes();
  absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> created =
      MpcParameterUpdaterModule::Create(/*mpcPtr=*/nullptr, std::move(options));
  ASSERT_TRUE(created.ok()) << created.status();
  (*created)->enqueueParameterUpdate(updateWithScaling(/*scaling=*/2.0));
  solve(**created);
  EXPECT_THAT(log_.events(), testing::IsEmpty());
}

}  // namespace
}  // namespace ocs2::humanoid
