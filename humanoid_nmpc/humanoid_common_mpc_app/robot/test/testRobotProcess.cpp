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
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/OperatorPayloadChecks.h"
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_common_mpc_app/robot/ConfigFileStore.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_common_mpc_app/robot/RobotProcess.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/robot/config/robot/RobotProcessSettingsFromConfig.h"
#include "humanoid_mpc_config/joint_pd_gains_file.pb.h"
#include "humanoid_mpc_config/mpc_parameter_update.pb.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/config_file_kind.nproto.h"
#include "humanoid_mpc_msgs/config_file_kind.pb.h"
#include "humanoid_mpc_msgs/config_file_save.pb.h"
#include "humanoid_mpc_msgs/config_file_save_status.pb.h"
#include "humanoid_mpc_msgs/fsm_command.pb.h"
#include "humanoid_mpc_msgs/fsm_state.pb.h"
#include "humanoid_mpc_msgs/loop_timing.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"
#include "nproto/Textproto.h"
#include "robot_ipc/Delivery.h"
#include "robot_model/ContactEstimator.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

/*
 * The robot process end to end without an MPC: a scripted controller, the headless MuJoCo backend and a loopback bus
 * with an operator on it. The operator's commands switch the mode on the realtime thread; robot/fsm_state, robot/state
 * and robot/loop_timing come out of the communication thread; controller settings from the bus and from the task file
 * reach the controller; the gantry's discontinuities reset it; the period holds; and stop() ends everything. A save on
 * operator/config_save is stored and applied by the task file's watcher, and an MPC parameter update of another
 * configuration installs nothing.
 */

namespace ocs2::humanoid {
namespace {

using robot_test::waitFor;

/** What the operator receives from the robot. */
class OperatorView {
 public:
  void onFsmState(const humanoid_mpc_msgs::FsmState& state) {
    absl::MutexLock lock(mutex_);
    fsmStates_.push_back(state);
  }
  void onState(const humanoid_mpc_msgs::RobotStateSample& sample) {
    absl::MutexLock lock(mutex_);
    samples_.push_back(sample);
  }
  void onLoopTiming(const humanoid_mpc_msgs::LoopTiming& timing) {
    absl::MutexLock lock(mutex_);
    timings_.push_back(timing);
  }
  std::optional<humanoid_mpc_msgs::FsmState> lastFsmState() const {
    absl::MutexLock lock(mutex_);
    if (fsmStates_.empty()) return std::nullopt;
    return fsmStates_.back();
  }
  std::optional<humanoid_mpc_msgs::RobotStateSample> lastSample() const {
    absl::MutexLock lock(mutex_);
    if (samples_.empty()) return std::nullopt;
    return samples_.back();
  }
  std::optional<humanoid_mpc_msgs::LoopTiming> lastTiming() const {
    absl::MutexLock lock(mutex_);
    if (timings_.empty()) return std::nullopt;
    return timings_.back();
  }
  size_t numSamples() const {
    absl::MutexLock lock(mutex_);
    return samples_.size();
  }

 private:
  mutable absl::Mutex mutex_;
  std::vector<humanoid_mpc_msgs::FsmState> fsmStates_ ABSL_GUARDED_BY(mutex_);
  std::vector<humanoid_mpc_msgs::RobotStateSample> samples_ ABSL_GUARDED_BY(mutex_);
  std::vector<humanoid_mpc_msgs::LoopTiming> timings_ ABSL_GUARDED_BY(mutex_);
};

class RobotProcessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = (std::filesystem::path(std::getenv("TEST_TMPDIR")) / "task.textproto").string();
    writeTaskFile("robot_state");

    description_ = std::make_unique<robot::model::RobotDescription>(robot_test::atlasDescription());
    controller_.setRobotJointNames(description_->getJointNames());
    RobotBackendOptions options;
    options.robotName = "drc_atlas";
    options.urdfFile = robot_test::kAtlasUrdf;
    options.mjcfFile = robot_test::kAtlasScene;
    options.initialState.emplace(*description_);
    options.initialState->setRootPositionInWorldFrame(vector3_t(0.0, 0.0, 0.95));
    const ModelSettings modelSettings =
        ModelSettings::Create(robot_test::kAtlasTask, robot_test::kAtlasUrdf, "centroidal_mpc_", /*verbose=*/false).value();
    options.contactFrameNames = modelSettings.contactNames;
    options.contactParentJointNames = modelSettings.contactParentJointNames;
    options.simulator.visualizations = std::vector<std::string>{};
    options.headless = true;
    initialState_.emplace(*options.initialState);
    absl::StatusOr<std::unique_ptr<RobotBackend>> backend = RobotBackendRegistry().create("mujoco", options);
    ASSERT_TRUE(backend.ok()) << backend.status();
    backend_ = *std::move(backend);
    backend_->registerContactEstimators(estimators_);

    robotBus_ = robot_test::createLoopbackBus("robot");
    operatorBus_ = robot_test::createLoopbackBus("operator");
    robot_test::connectBoth(*robotBus_, *operatorBus_);
    ASSERT_TRUE(operatorBus_
                    ->subscribe<humanoid_mpc_msgs::FsmState>(ipc::topics::kRobotFsmState, robot::ipc::Delivery::kAll,
                                                             [this](const humanoid_mpc_msgs::FsmState& state) { view_.onFsmState(state); })
                    .ok());
    ASSERT_TRUE(operatorBus_
                    ->subscribe<humanoid_mpc_msgs::RobotStateSample>(
                        ipc::topics::kRobotState, robot::ipc::Delivery::kAll,
                        [this](const humanoid_mpc_msgs::RobotStateSample& sample) { view_.onState(sample); })
                    .ok());
    ASSERT_TRUE(
        operatorBus_
            ->subscribe<humanoid_mpc_msgs::LoopTiming>(ipc::topics::kRobotLoopTiming, robot::ipc::Delivery::kAll,
                                                       [this](const humanoid_mpc_msgs::LoopTiming& timing) { view_.onLoopTiming(timing); })
            .ok());
  }

  void TearDown() override {
    if (process_ != nullptr) process_->stop();
    operatorBus_->stop();
  }

  void writeTaskFile(const std::string& contactEstimator) {
    std::ofstream file(taskFile_, std::ios::trunc);
    file << "contact_estimator: \"" << contactEstimator << "\"\n";
  }

  RobotProcess::Config processConfig() {
    const absl::StatusOr<humanoid_mpc_config::TaskFile> message = nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(
        "contact_estimator: \"robot_state\"\ntelemetry_sinks: \"bus\"\ntelemetry_frequency: 50\n", "test");
    EXPECT_TRUE(message.ok()) << message.status();
    mpc_config::TaskFile task;
    if (message.ok()) {
      EXPECT_TRUE(mpc_config::FromProto(*message, &task).ok());
    }
    absl::StatusOr<RobotProcessSettings> settings = robotProcessSettingsFromConfig(task);
    EXPECT_TRUE(settings.ok()) << settings.status();
    RobotProcess::Config config;
    config.controlFrequency = 500.0;
    config.settings = *settings;
    config.initialState = initialState_;
    config.taskFile = taskFile_;
    config.pdGainsFileCheckInterval = 50;
    config.loopTimingWindow = std::chrono::milliseconds(200);
    return config;
  }

  /** Publishes an FSM command until `done` holds (the subscription takes a moment to reach the publisher). */
  bool command(const std::string& name, const std::function<bool()>& done) {
    humanoid_mpc_msgs::FsmCommand message;
    message.set_command(name);
    message.set_sequence(++sequence_);
    return waitFor([&]() {
      operatorBus_->publish(ipc::topics::kOperatorFsmCommand, message).IgnoreError();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      return done();
    });
  }

  /** Publishes `message` on `topic` until `done` holds, as the GUI publishes its tuning payloads. */
  template <typename Message>
  bool publishUntil(absl::string_view topic, const Message& message, const std::function<bool()>& done) {
    return waitFor([&]() {
      operatorBus_->publish(topic, message).IgnoreError();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      return done();
    });
  }

  std::string taskFile_;
  std::unique_ptr<robot::model::RobotDescription> description_;
  std::optional<robot::model::RobotState> initialState_;
  robot::model::ContactEstimatorRegistry estimators_;
  std::unique_ptr<RobotBackend> backend_;
  std::unique_ptr<robot::ipc::Bus> robotBus_;
  std::unique_ptr<robot::ipc::Bus> operatorBus_;
  robot_test::ScriptedRobotController controller_;
  std::unique_ptr<RobotProcess> process_;
  OperatorView view_;
  uint64_t sequence_ = 0;
};

TEST_F(RobotProcessTest, RunsTheLoopAndServesTheOperator) {
  absl::StatusOr<std::unique_ptr<RobotProcess>> created =
      RobotProcess::Create(*robotBus_, *backend_, controller_, estimators_, processConfig());
  ASSERT_TRUE(created.ok()) << created.status();
  process_ = *std::move(created);
  EXPECT_EQ(controller_.contactEstimatorName(), "RobotStateContactEstimator") << "the task file's estimator is installed";
  ASSERT_FALSE(process_->start().ok()) << "the bus must run first";
  ASSERT_TRUE(robotBus_->start().ok());
  ASSERT_TRUE(operatorBus_->start().ok());

  absl::StatusOr<std::unique_ptr<RobotProcess>> again =
      RobotProcess::Create(*robotBus_, *backend_, controller_, estimators_, processConfig());
  EXPECT_EQ(again.status().code(), absl::StatusCode::kFailedPrecondition) << "a running bus takes no more subscriptions";

  ASSERT_TRUE(process_->start().ok());
  EXPECT_TRUE(controller_.started());
  ASSERT_TRUE(waitFor([&]() { return controller_.cycles() > 50; }));
  EXPECT_EQ(controller_.mode(), "ZERO_TORQUE");

  // The state is republished at 2 Hz, so a subscriber that joined late still learns it.
  ASSERT_TRUE(waitFor([&]() { return view_.lastFsmState().has_value(); }));
  EXPECT_EQ(robot_test::valueOrFail(view_.lastFsmState()).mode(), "ZERO_TORQUE");
  EXPECT_TRUE(robot_test::valueOrFail(view_.lastFsmState()).gantry_locked());
  EXPECT_TRUE(robot_test::valueOrFail(view_.lastFsmState()).mpc_healthy());

  // An FSM command over the bus reaches the controller on the realtime thread, and the state follows.
  ASSERT_TRUE(command("JOINT_PD", [&]() { return controller_.mode() == "JOINT_PD"; }));
  ASSERT_TRUE(waitFor([&]() { return view_.lastFsmState().value_or(humanoid_mpc_msgs::FsmState()).mode() == "JOINT_PD"; }));
  EXPECT_FALSE(backend_->simulator()->isZeroTorqueMode());

  // robot/state: every joint by name, at the telemetry rate.
  ASSERT_TRUE(waitFor([&]() {
    const std::optional<humanoid_mpc_msgs::RobotStateSample> sample = view_.lastSample();
    return sample.has_value() && sample->control_mode() == "JOINT_PD";
  }));
  const humanoid_mpc_msgs::RobotStateSample sample = robot_test::valueOrFail(view_.lastSample());
  ASSERT_EQ(static_cast<size_t>(sample.joint_names_size()), description_->getNumJoints());
  for (size_t joint = 0; joint < description_->getNumJoints(); ++joint) {
    EXPECT_EQ(sample.joint_names(static_cast<int>(joint)), description_->getJointName(joint));
  }
  EXPECT_EQ(sample.joint_kp(0), 100.0) << "the action the controller computed";
  EXPECT_EQ(sample.contact_flags_size(), 2);

  // robot/loop_timing, once per window, and the loop keeps its period.
  ASSERT_TRUE(waitFor([&]() { return view_.lastTiming().value_or(humanoid_mpc_msgs::LoopTiming()).cycles() > 200; }));
  const humanoid_mpc_msgs::LoopTiming timing = robot_test::valueOrFail(view_.lastTiming());
  EXPECT_DOUBLE_EQ(timing.target_period_s(), 0.002);
  EXPECT_NEAR(timing.mean_period_s(), 0.002, 0.0005);
  EXPECT_LE(timing.overruns(), timing.cycles() / 20) << "the loop overran " << timing.overruns() << " of " << timing.cycles() << " cycles";
  EXPECT_LE(timing.missed_periods(), timing.cycles() / 20) << "the loop skipped " << timing.missed_periods() << " periods";
  EXPECT_GE(timing.max_lateness_s(), 0.0);
  EXPECT_EQ(timing.telemetry_samples_dropped(), 0u);
  EXPECT_DOUBLE_EQ(timing.policy_age_s(), -1.0) << "no MPC link here";

  // Controller settings from the GUI (its whole task file) reach the controller on the realtime thread.
  humanoid_mpc_config::MpcParameterUpdate parameters;
  parameters.mutable_task()->set_contact_estimator("always_in_contact");
  parameters.mutable_task()->mutable_contact_wrench_gate()->set_ramp_time(0.04);
  parameters.set_schema_fingerprint(mpcParameterUpdateSchemaFingerprint());
  ASSERT_TRUE(publishUntil(ipc::topics::kOperatorMpcParameters, parameters,
                           [&]() { return controller_.contactEstimatorName() == "AlwaysInContactEstimator"; }));
  EXPECT_DOUBLE_EQ(controller_.gate().rampTime, 0.04);
  // And from the task file, when it is saved: the whole file, whose missing contact_wrench_gate block is the
  // instantaneous gate.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  writeTaskFile("robot_state");
  std::filesystem::last_write_time(taskFile_, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(1));
  EXPECT_TRUE(waitFor([&]() { return controller_.contactEstimatorName() == "RobotStateContactEstimator"; }));
  EXPECT_TRUE(waitFor([&]() { return controller_.gate().rampTime == ContactWrenchGate::Config{}.rampTime; }))
      << "the file without a gate kept the GUI's";

  // PD gains from the GUI (its whole gains file) and the gains file watcher.
  humanoid_mpc_config::JointPdGainsFile gains;
  gains.mutable_default_gains()->set_kp(10.0);
  gains.mutable_default_gains()->set_kd(1.0);
  EXPECT_TRUE(publishUntil(ipc::topics::kOperatorPdGains, gains, [&]() { return controller_.pdGainsDocuments() > 0; }));
  EXPECT_TRUE(waitFor([&]() { return controller_.pdGainsPolls() > 0; }));

  // The gantry: its release resets the MPC, its lock is a discontinuity that resets and holds the controller and counts
  // a controller reset in the state.
  ASSERT_TRUE(command("UNLOCK_GANTRY", [&]() { return controller_.resets() > 0; }));
  ASSERT_TRUE(command("LOCK_GANTRY", [&]() { return controller_.resetsAndHolds() > 0; }));
  ASSERT_TRUE(waitFor([&]() { return view_.lastFsmState().value_or(humanoid_mpc_msgs::FsmState()).controller_resets() == 1; }));
  EXPECT_TRUE(robot_test::valueOrFail(view_.lastFsmState()).gantry_locked());
  EXPECT_EQ(robot_test::valueOrFail(view_.lastFsmState()).mode(), "JOINT_PD");

  // The MPC's health is part of the state.
  controller_.setHealthy(false);
  EXPECT_TRUE(waitFor([&]() { return !view_.lastFsmState().value_or(humanoid_mpc_msgs::FsmState()).mpc_healthy(); }));

  // stop() ends the loop and the bus: nothing runs afterwards.
  process_->stop();
  const uint64_t cycles = controller_.cycles();
  const size_t samples = view_.numSamples();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(controller_.cycles(), cycles);
  EXPECT_EQ(view_.numSamples(), samples);
  EXPECT_FALSE(robotBus_->isRunning());
}

TEST_F(RobotProcessTest, RefusesAnUnknownContactEstimatorAndASinkItDoesNotKnow) {
  RobotProcess::Config config = processConfig();
  config.settings.contactEstimator = "magic";
  absl::StatusOr<std::unique_ptr<RobotProcess>> unknownEstimator =
      RobotProcess::Create(*robotBus_, *backend_, controller_, estimators_, config);
  EXPECT_EQ(unknownEstimator.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(unknownEstimator.status().message().find("cheater_sim"), std::string::npos) << unknownEstimator.status().message();
  // It names the file and the field the operator edits.
  EXPECT_NE(unknownEstimator.status().message().find(absl::StrCat(taskFile_, ": contact_estimator: ")), std::string::npos)
      << unknownEstimator.status().message();

  std::unique_ptr<robot::ipc::Bus> otherBus = robot_test::createLoopbackBus("robot");
  config = processConfig();
  config.settings.telemetrySinks = {"bus", "carrier_pigeon"};
  absl::StatusOr<std::unique_ptr<RobotProcess>> unknownSink = RobotProcess::Create(*otherBus, *backend_, controller_, estimators_, config);
  EXPECT_EQ(unknownSink.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(unknownSink.status().message().find("carrier_pigeon"), std::string::npos);
}

/** An estimator that breaks its contract: one flag, whatever the number of contact points. */
class OneFlagContactEstimator final : public robot::model::ContactEstimator {
 public:
  void estimateContactFlags(const robot::model::RobotState& /*robotState*/, std::vector<bool>& flags) override { flags.assign(1, true); }
  std::string getName() const override { return "OneFlagContactEstimator"; }
};

TEST_F(RobotProcessTest, RefusesAContactEstimatorWithoutOneFlagPerContactPointBeforeTheLoopCanInstallIt) {
  estimators_.add("one_flag", "one flag, whatever the robot", [] { return std::make_shared<OneFlagContactEstimator>(); });
  RobotProcess::Config config = processConfig();
  config.settings.contactEstimator = "one_flag";
  absl::StatusOr<std::unique_ptr<RobotProcess>> refused = RobotProcess::Create(*robotBus_, *backend_, controller_, estimators_, config);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(refused.status().message().find("'one_flag' reports 1 contact flags"), std::string::npos) << refused.status();

  // A swap to it while the loop runs is refused on the communication thread; the estimator in use stays. The refused
  // Create() above registered nothing on the bus.
  absl::StatusOr<std::unique_ptr<RobotProcess>> created =
      RobotProcess::Create(*robotBus_, *backend_, controller_, estimators_, processConfig());
  ASSERT_TRUE(created.ok()) << created.status();
  process_ = *std::move(created);
  ASSERT_TRUE(robotBus_->start().ok());
  ASSERT_TRUE(process_->start().ok());
  const uint64_t cyclesBefore = controller_.cycles();
  const uint64_t rejectedBefore = process_->mailbox().statistics().controllerSettingsRejected;
  writeTaskFile("one_flag");
  std::filesystem::last_write_time(taskFile_, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(1));
  // The task file is checked once a second; the loop keeps running on the estimator it had.
  ASSERT_TRUE(waitFor([&]() { return process_->mailbox().statistics().controllerSettingsRejected > rejectedBefore; }));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_GT(controller_.cycles(), cyclesBefore);
  EXPECT_EQ(controller_.contactEstimatorName(), "RobotStateContactEstimator") << "the refused estimator was installed";
}

TEST_F(RobotProcessTest, RefusesAControllerOfAnotherRobotBeforeStart) {
  robot_test::ScriptedRobotController otherRobot;
  std::vector<std::string> names = description_->getJointNames();
  std::swap(names.front(), names.back());
  otherRobot.setRobotJointNames(names);
  absl::StatusOr<std::unique_ptr<RobotProcess>> reordered =
      RobotProcess::Create(*robotBus_, *backend_, otherRobot, estimators_, processConfig());
  EXPECT_EQ(reordered.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(reordered.status().message().find("joint 0 of the controller is '" + names.front() + "'"), std::string::npos)
      << reordered.status();

  names.pop_back();
  otherRobot.setRobotJointNames(names);
  absl::StatusOr<std::unique_ptr<RobotProcess>> fewer =
      RobotProcess::Create(*robotBus_, *backend_, otherRobot, estimators_, processConfig());
  EXPECT_EQ(fewer.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(fewer.status().message().find("a robot of " + std::to_string(names.size()) + " joints"), std::string::npos) << fewer.status();
  EXPECT_EQ(otherRobot.eventSink(), nullptr) << "nothing was set up";
}

/**
 * The MuJoCo backend, recording for every cycle whether the joint action reached it and in which mode the controller
 * computed it. Realtime thread; the test reads after the loop has stopped.
 */
class RecordingBackend final : public RobotBackend {
 public:
  RecordingBackend(RobotBackend& backend, const robot_test::ScriptedRobotController& controller)
      : backend_(backend), controller_(controller) {}

  absl::string_view name() const override { return backend_.name(); }
  robot::model::RobotHWInterfaceBase& hardware() override { return backend_.hardware(); }
  const robot::model::RobotHWInterfaceBase& hardware() const override { return backend_.hardware(); }
  absl::Status initialize() override { return backend_.initialize(); }
  absl::Status start(const std::vector<int>& cores) override { return backend_.start(cores); }
  bool acceptsJointAction() const override {
    const bool accepted = backend_.acceptsJointAction();
    absl::MutexLock lock(mutex_);
    if (accepted) acceptedInModes_.push_back(controller_.mode());
    return accepted;
  }
  void readMeasuredContactForces(std::array<vector3_t, kNumContacts>& forces) override { backend_.readMeasuredContactForces(forces); }
  void enterSafeState() override {
    safeStates_.fetch_add(1);
    backend_.enterSafeState();
  }
  void registerContactEstimators(robot::model::ContactEstimatorRegistry& registry) const override {
    backend_.registerContactEstimators(registry);
  }
  robot::mujoco_sim_interface::MujocoSimInterface* absl_nullable simulator() override { return backend_.simulator(); }

  /** The controller's mode of every cycle whose action reached the backend. */
  std::vector<std::string> acceptedInModes() const {
    absl::MutexLock lock(mutex_);
    return acceptedInModes_;
  }
  uint64_t safeStates() const { return safeStates_.load(); }

 private:
  RobotBackend& backend_;
  const robot_test::ScriptedRobotController& controller_;
  mutable absl::Mutex mutex_;
  mutable std::vector<std::string> acceptedInModes_ ABSL_GUARDED_BY(mutex_);
  std::atomic<uint64_t> safeStates_{0};
};

/** A backend that is not a simulator: the MuJoCo backend behind RobotBackend's simulator(), which is nullptr. */
class NoSimulatorBackend final : public RobotBackend {
 public:
  explicit NoSimulatorBackend(RobotBackend& backend) : backend_(backend) {}

  absl::string_view name() const override { return "no_simulator"; }
  robot::model::RobotHWInterfaceBase& hardware() override { return backend_.hardware(); }
  const robot::model::RobotHWInterfaceBase& hardware() const override { return backend_.hardware(); }
  absl::Status initialize() override { return backend_.initialize(); }
  absl::Status start(const std::vector<int>& cores) override { return backend_.start(cores); }
  bool acceptsJointAction() const override { return backend_.acceptsJointAction(); }
  void readMeasuredContactForces(std::array<vector3_t, kNumContacts>& forces) override { backend_.readMeasuredContactForces(forces); }
  void enterSafeState() override { backend_.enterSafeState(); }
  void registerContactEstimators(robot::model::ContactEstimatorRegistry& registry) const override {
    backend_.registerContactEstimators(registry);
  }

 private:
  RobotBackend& backend_;
};

TEST_F(RobotProcessTest, RefusesABackendThatIsNotASimulator) {
  NoSimulatorBackend backend(*backend_);
  ASSERT_EQ(backend.simulator(), nullptr);
  absl::StatusOr<std::unique_ptr<RobotProcess>> created =
      RobotProcess::Create(*robotBus_, backend, controller_, estimators_, processConfig());
  EXPECT_EQ(created.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(created.status().message().find("'no_simulator' is not a simulator"), std::string::npos) << created.status();
  EXPECT_EQ(controller_.eventSink(), nullptr) << "the refused process hands the controller nothing";
}

TEST_F(RobotProcessTest, TheCycleThatSwitchesTheTorquesOnHandsTheBackendItsOwnAction) {
  // The cycle in which JOINT_PD switches the torques on computed the ZERO_TORQUE action (no gain, no torque): the backend
  // gets that one, at once, instead of keeping the action latched before the torques went off - here the JOINT_PD
  // action of before, in the robot an MPC action with high gains - in force for a control period.
  RecordingBackend recording(*backend_, controller_);
  absl::StatusOr<std::unique_ptr<RobotProcess>> created =
      RobotProcess::Create(*robotBus_, recording, controller_, estimators_, processConfig());
  ASSERT_TRUE(created.ok()) << created.status();
  process_ = *std::move(created);
  ASSERT_TRUE(robotBus_->start().ok());
  ASSERT_TRUE(operatorBus_->start().ok());
  ASSERT_TRUE(process_->start().ok());
  ASSERT_TRUE(waitFor([&]() { return controller_.cycles() > 20; }));
  EXPECT_TRUE(recording.acceptedInModes().empty()) << "nothing reaches the actuators while the torques are off";

  ASSERT_TRUE(command("JOINT_PD", [&]() { return controller_.mode() == "JOINT_PD"; }));
  ASSERT_TRUE(waitFor([&]() { return controller_.cycles() > 100; }));
  ASSERT_TRUE(command("ZERO_TORQUE", [&]() { return controller_.mode() == "ZERO_TORQUE"; }));
  ASSERT_TRUE(waitFor([&]() { return backend_->simulator()->isZeroTorqueMode(); }));
  const size_t acceptedBefore = recording.acceptedInModes().size();
  ASSERT_TRUE(command("JOINT_PD", [&]() { return controller_.mode() == "JOINT_PD"; }));
  ASSERT_TRUE(waitFor([&]() { return recording.acceptedInModes().size() > acceptedBefore + 5; }));
  process_->stop();

  const std::vector<std::string> accepted = recording.acceptedInModes();
  ASSERT_GT(accepted.size(), acceptedBefore);
  // Each switch-on is one cycle computed in ZERO_TORQUE whose action went out: the first JOINT_PD command's and the
  // second's. Without them the backend would have kept the old action until the next cycle.
  EXPECT_EQ(accepted.front(), "ZERO_TORQUE") << "the first switch-on";
  EXPECT_EQ(accepted[acceptedBefore], "ZERO_TORQUE") << "the second switch-on";
  for (size_t index = acceptedBefore + 1; index < accepted.size(); ++index) EXPECT_EQ(accepted[index], "JOINT_PD") << index;
  EXPECT_GE(recording.safeStates(), 1u) << "stop() puts the backend in its safe state";
  EXPECT_TRUE(backend_->simulator()->isZeroTorqueMode());
}

TEST_F(RobotProcessTest, ACycleThatThrowsStopsTheLoopAndPutsTheBackendInItsSafeState) {
  RecordingBackend recording(*backend_, controller_);
  absl::StatusOr<std::unique_ptr<RobotProcess>> created =
      RobotProcess::Create(*robotBus_, recording, controller_, estimators_, processConfig());
  ASSERT_TRUE(created.ok()) << created.status();
  process_ = *std::move(created);
  ASSERT_TRUE(robotBus_->start().ok());
  ASSERT_TRUE(operatorBus_->start().ok());
  ASSERT_TRUE(process_->start().ok());
  ASSERT_TRUE(command("JOINT_PD", [&]() { return controller_.mode() == "JOINT_PD"; }));
  ASSERT_FALSE(backend_->simulator()->isZeroTorqueMode());

  controller_.throwInNextCycle("the backend's driver failed");
  ASSERT_TRUE(waitFor([&]() { return process_->faulted(); }));
  const uint64_t cycles = controller_.cycles();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(controller_.cycles(), cycles) << "the loop ended";
  EXPECT_TRUE(backend_->simulator()->isZeroTorqueMode()) << "the actuators are off: no action of the failed controller stays";
  EXPECT_EQ(recording.safeStates(), 1u);

  const absl::Status ended = process_->runUntilShutdown([]() { return false; });
  EXPECT_EQ(ended.code(), absl::StatusCode::kInternal);
  EXPECT_NE(ended.message().find("the backend's driver failed"), std::string::npos) << ended.message();
  EXPECT_FALSE(robotBus_->isRunning()) << "runUntilShutdown() stops the process";
}

TEST_F(RobotProcessTest, TheControllerReportsThroughTheEventLog) {
  EXPECT_EQ(controller_.eventSink(), nullptr);
  absl::StatusOr<std::unique_ptr<RobotProcess>> created =
      RobotProcess::Create(*robotBus_, *backend_, controller_, estimators_, processConfig());
  ASSERT_TRUE(created.ok()) << created.status();
  process_ = *std::move(created);
  EXPECT_NE(controller_.eventSink(), nullptr) << "Create() hands the controller the process's realtime event log";
}

TEST_F(RobotProcessTest, RunUntilShutdownReturnsOkOnAShutdown) {
  absl::StatusOr<std::unique_ptr<RobotProcess>> created =
      RobotProcess::Create(*robotBus_, *backend_, controller_, estimators_, processConfig());
  ASSERT_TRUE(created.ok()) << created.status();
  process_ = *std::move(created);
  ASSERT_TRUE(robotBus_->start().ok());
  ASSERT_TRUE(process_->start().ok());
  ASSERT_TRUE(waitFor([&]() { return controller_.cycles() > 10; }));
  EXPECT_TRUE(process_->runUntilShutdown([]() { return true; }).ok());
  EXPECT_FALSE(process_->faulted());
}

TEST_F(RobotProcessTest, RunUntilShutdownReportsOnceTheLoopHasRunForTheGivenTime) {
  absl::StatusOr<std::unique_ptr<RobotProcess>> created =
      RobotProcess::Create(*robotBus_, *backend_, controller_, estimators_, processConfig());
  ASSERT_TRUE(created.ok()) << created.status();
  process_ = *std::move(created);
  ASSERT_TRUE(robotBus_->start().ok());
  ASSERT_TRUE(process_->start().ok());
  int reports = 0;
  uint64_t cyclesAtReport = 0;
  const absl::Time stopAt = absl::Now() + absl::Milliseconds(400);
  EXPECT_TRUE(process_
                  ->runUntilShutdown([&]() { return absl::Now() >= stopAt; }, absl::Milliseconds(100),
                                     [&]() {
                                       ++reports;
                                       cyclesAtReport = controller_.cycles();
                                     })
                  .ok());
  EXPECT_EQ(reports, 1);
  EXPECT_GT(cyclesAtReport, 0u) << "reported while the loop ran";
}

// The configuration identity of the robot process's task file in these tests.
constexpr char kTestTaskIdentity[] = "robot_models/test_robot/test_robot_mpc/config/mpc/task.textproto";

/** A save of the test's task file as the GUI sends it. */
humanoid_mpc_msgs::ConfigFileSave taskFileSave(const std::string& text, uint64_t sequence) {
  humanoid_mpc_msgs::ConfigFileSave save;
  save.set_kind(humanoid_mpc_msgs::CONFIG_FILE_KIND_TASK);
  save.set_config_path(kTestTaskIdentity);
  save.set_schema_fingerprint(configFileSchemaFingerprint(msgs::ConfigFileKind::kTask));
  save.set_text(text);
  save.set_sequence(sequence);
  return save;
}

TEST_F(RobotProcessTest, ASaveOnTheBusIsStoredAndTheTaskFileWatcherAppliesIt) {
  RobotProcess::Config config = processConfig();
  // The test's task file is the store's copy.
  config.configStore.task = {.path = taskFile_, .identity = kTestTaskIdentity};
  absl::StatusOr<std::unique_ptr<RobotProcess>> created = RobotProcess::Create(*robotBus_, *backend_, controller_, estimators_, config);
  ASSERT_TRUE(created.ok()) << created.status();
  process_ = *std::move(created);
  std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> answer;
  absl::Mutex answerMutex;
  ASSERT_TRUE(operatorBus_
                  ->subscribe<humanoid_mpc_msgs::ConfigFileSaveStatus>(ipc::topics::kRobotConfigSaveStatus, robot::ipc::Delivery::kAll,
                                                                       [&](const humanoid_mpc_msgs::ConfigFileSaveStatus& status) {
                                                                         absl::MutexLock lock(answerMutex);
                                                                         answer = status;
                                                                       })
                  .ok());
  ASSERT_TRUE(robotBus_->start().ok());
  ASSERT_TRUE(operatorBus_->start().ok());
  ASSERT_TRUE(process_->start().ok());
  ASSERT_TRUE(waitFor([&]() { return controller_.contactEstimatorName() == "RobotStateContactEstimator"; }));

  // The GUI repeats the save until it hears the answer, as push_robot_config does.
  const std::string saved = "contact_estimator: \"always_in_contact\"\n";
  const humanoid_mpc_msgs::ConfigFileSave save = taskFileSave(saved, /*sequence=*/1);
  ASSERT_TRUE(publishUntil(ipc::topics::kOperatorConfigSave, save, [&]() {
    absl::MutexLock lock(answerMutex);
    return answer.has_value();
  }));
  {
    absl::MutexLock lock(answerMutex);
    if (!answer.has_value()) FAIL() << "the save was not answered";
    EXPECT_EQ(answer->result(), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_SAVED) << answer->message();
    EXPECT_EQ(answer->stored_path(), taskFile_);
  }
  std::ifstream stored(taskFile_);
  EXPECT_EQ(std::string(std::istreambuf_iterator<char>(stored), std::istreambuf_iterator<char>()), saved);
  // The task file's watcher sees the new file and applies its controller-side settings, as for an edit.
  EXPECT_TRUE(waitFor([&]() { return controller_.contactEstimatorName() == "AlwaysInContactEstimator"; }));
  EXPECT_GE(process_->configStore().statistics().saved, 1u);
}

TEST_F(RobotProcessTest, AnMpcParameterUpdateOfAnotherConfigurationInstallsNothing) {
  RobotProcess::Config config = processConfig();
  config.taskFileIdentity = kTestTaskIdentity;
  absl::StatusOr<std::unique_ptr<RobotProcess>> created = RobotProcess::Create(*robotBus_, *backend_, controller_, estimators_, config);
  ASSERT_TRUE(created.ok()) << created.status();
  process_ = *std::move(created);
  ASSERT_TRUE(robotBus_->start().ok());
  ASSERT_TRUE(operatorBus_->start().ok());
  ASSERT_TRUE(process_->start().ok());

  // Another configuration of the robot (the same robot_name): refused before its controller-side settings are taken.
  humanoid_mpc_config::MpcParameterUpdate other;
  other.mutable_task()->set_contact_estimator("always_in_contact");
  other.mutable_task()->mutable_contact_wrench_gate()->set_ramp_time(0.04);
  other.set_schema_fingerprint(mpcParameterUpdateSchemaFingerprint());
  other.set_config_path("robot_models/test_robot/another_mpc/config/mpc/task.textproto");
  ASSERT_TRUE(publishUntil(ipc::topics::kOperatorMpcParameters, other,
                           [&]() { return process_->mailbox().statistics().controllerSettingsRejected > 0; }));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(controller_.contactEstimatorName(), "RobotStateContactEstimator");
  EXPECT_EQ(process_->mailbox().statistics().controllerSettingsQueued, 0u);

  // The running configuration's own update is applied.
  humanoid_mpc_config::MpcParameterUpdate own = other;
  own.set_config_path(kTestTaskIdentity);
  EXPECT_TRUE(publishUntil(ipc::topics::kOperatorMpcParameters, own,
                           [&]() { return controller_.contactEstimatorName() == "AlwaysInContactEstimator"; }));
}

}  // namespace
}  // namespace ocs2::humanoid
