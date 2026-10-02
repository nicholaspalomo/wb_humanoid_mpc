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

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include <humanoid_common_mpc/common/ModelSettings.h>
#include <robot_model/ContactEstimatorRegistry.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotState.h>

#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_common_mpc_app/robot/RobotProcess.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/fsm_command.pb.h"
#include "humanoid_mpc_msgs/fsm_state.pb.h"
#include "humanoid_mpc_msgs/loop_timing.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"
#include "humanoid_mpc_msgs/yaml_document.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"
#include "robot_ipc/Delivery.h"

/*
 * The robot process end to end without an MPC: a scripted controller, the headless MuJoCo backend and a loopback bus
 * with an operator on it. The operator's commands switch the mode on the realtime thread; robot/fsm_state, robot/state
 * and robot/loop_timing come out of the communication thread; controller settings from the bus and from the task file
 * reach the controller; the gantry's discontinuities reset it; the period holds; and stop() ends everything.
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
    taskFile_ = (std::filesystem::path(std::getenv("TEST_TMPDIR")) / "task.yaml").string();
    writeTaskFile("robot_state");

    description_ = std::make_unique<robot::model::RobotDescription>(robot_test::kAtlasUrdf);
    RobotBackendOptions options;
    options.robotName = "drc_atlas";
    options.urdfFile = robot_test::kAtlasUrdf;
    options.mjcfFile = robot_test::kAtlasScene;
    options.initialState.emplace(*description_);
    options.initialState->setRootPositionInWorldFrame(vector3_t(0.0, 0.0, 0.95));
    const ModelSettings modelSettings(robot_test::kAtlasTask, robot_test::kAtlasUrdf, "centroidal_mpc_", /*verbose=*/false);
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
    file << "contactEstimator: " << contactEstimator << "\n";
  }

  RobotProcess::Config processConfig() {
    absl::StatusOr<RobotProcessSettings> settings =
        parseRobotProcessSettings("contactEstimator: robot_state\ntelemetrySinks: [bus]\ntelemetryFrequency: 50\n", "test");
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

  bool publishYaml(absl::string_view topic, const std::string& text, const std::function<bool()>& done) {
    humanoid_mpc_msgs::YamlDocument document;
    document.set_yaml(text);
    return waitFor([&]() {
      operatorBus_->publish(topic, document).IgnoreError();
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
  EXPECT_EQ(view_.lastFsmState()->mode(), "ZERO_TORQUE");
  EXPECT_TRUE(view_.lastFsmState()->gantry_locked());
  EXPECT_TRUE(view_.lastFsmState()->mpc_healthy());

  // An FSM command over the bus reaches the controller on the realtime thread, and the state follows.
  ASSERT_TRUE(command("JOINT_PD", [&]() { return controller_.mode() == "JOINT_PD"; }));
  ASSERT_TRUE(waitFor([&]() { return view_.lastFsmState()->mode() == "JOINT_PD"; }));
  EXPECT_FALSE(backend_->simulator()->isZeroTorqueMode());

  // robot/state: every joint by name, at the telemetry rate.
  ASSERT_TRUE(waitFor([&]() {
    const std::optional<humanoid_mpc_msgs::RobotStateSample> sample = view_.lastSample();
    return sample.has_value() && sample->control_mode() == "JOINT_PD";
  }));
  const humanoid_mpc_msgs::RobotStateSample sample = *view_.lastSample();
  ASSERT_EQ(static_cast<size_t>(sample.joint_names_size()), description_->getNumJoints());
  for (size_t joint = 0; joint < description_->getNumJoints(); ++joint) {
    EXPECT_EQ(sample.joint_names(static_cast<int>(joint)), description_->getJointName(joint));
  }
  EXPECT_EQ(sample.joint_kp(0), 100.0) << "the action the controller computed";
  EXPECT_EQ(sample.contact_flags_size(), 2);

  // robot/loop_timing, once per window, and the loop keeps its period.
  ASSERT_TRUE(waitFor([&]() { return view_.lastTiming().has_value() && view_.lastTiming()->cycles() > 200; }));
  const humanoid_mpc_msgs::LoopTiming timing = *view_.lastTiming();
  EXPECT_DOUBLE_EQ(timing.target_period_s(), 0.002);
  EXPECT_NEAR(timing.mean_period_s(), 0.002, 0.0005);
  EXPECT_LE(timing.overruns(), timing.cycles() / 20) << "the loop overran " << timing.overruns() << " of " << timing.cycles() << " cycles";
  EXPECT_LE(timing.missed_periods(), timing.cycles() / 20) << "the loop skipped " << timing.missed_periods() << " periods";
  EXPECT_GE(timing.max_lateness_s(), 0.0);
  EXPECT_EQ(timing.telemetry_samples_dropped(), 0u);
  EXPECT_DOUBLE_EQ(timing.policy_age_s(), -1.0) << "no MPC link here";

  // Controller settings from the GUI reach the controller on the realtime thread.
  ASSERT_TRUE(publishYaml(ipc::topics::kOperatorMpcParameters,
                          "contactEstimator: always_in_contact\ncontact_wrench_gate:\n  rampTime: 0.04\n",
                          [&]() { return controller_.contactEstimatorName() == "AlwaysInContactEstimator"; }));
  EXPECT_DOUBLE_EQ(controller_.gate().rampTime, 0.04);
  // And from the task file, when it is saved.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  writeTaskFile("robot_state");
  std::filesystem::last_write_time(taskFile_, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(1));
  EXPECT_TRUE(waitFor([&]() { return controller_.contactEstimatorName() == "RobotStateContactEstimator"; }));

  // PD gains documents and the gains file watcher.
  EXPECT_TRUE(
      publishYaml(ipc::topics::kOperatorPdGains, "default_gains: {kp: 10, kd: 1}\n", [&]() { return controller_.pdGainsDocuments() > 0; }));
  EXPECT_TRUE(waitFor([&]() { return controller_.pdGainsPolls() > 0; }));

  // The gantry: its release resets the MPC, its lock is a discontinuity that resets and holds the controller and counts
  // a controller reset in the state.
  ASSERT_TRUE(command("UNLOCK_GANTRY", [&]() { return controller_.resets() > 0; }));
  ASSERT_TRUE(command("LOCK_GANTRY", [&]() { return controller_.resetsAndHolds() > 0; }));
  ASSERT_TRUE(waitFor([&]() { return view_.lastFsmState()->controller_resets() == 1; }));
  EXPECT_TRUE(view_.lastFsmState()->gantry_locked());
  EXPECT_EQ(view_.lastFsmState()->mode(), "JOINT_PD");

  // The MPC's health is part of the state.
  controller_.setHealthy(false);
  EXPECT_TRUE(waitFor([&]() { return !view_.lastFsmState()->mpc_healthy(); }));

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

  std::unique_ptr<robot::ipc::Bus> otherBus = robot_test::createLoopbackBus("robot");
  config = processConfig();
  config.settings.telemetrySinks = {"bus", "carrier_pigeon"};
  absl::StatusOr<std::unique_ptr<RobotProcess>> unknownSink = RobotProcess::Create(*otherBus, *backend_, controller_, estimators_, config);
  EXPECT_EQ(unknownSink.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(unknownSink.status().message().find("carrier_pigeon"), std::string::npos);
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
  void readMeasuredContactForces(std::array<vector3_t, N_CONTACTS>& forces) override { backend_.readMeasuredContactForces(forces); }
  void enterSafeState() override {
    safeStates_.fetch_add(1);
    backend_.enterSafeState();
  }
  void registerContactEstimators(robot::model::ContactEstimatorRegistry& registry) const override {
    backend_.registerContactEstimators(registry);
  }
  robot::mujoco_sim_interface::MujocoSimInterface* simulator() override { return backend_.simulator(); }

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

  controller_.throwInNextCycle("the contact estimator reported 3 contact flags");
  ASSERT_TRUE(waitFor([&]() { return process_->faulted(); }));
  const uint64_t cycles = controller_.cycles();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(controller_.cycles(), cycles) << "the loop ended";
  EXPECT_TRUE(backend_->simulator()->isZeroTorqueMode()) << "the actuators are off: no action of the failed controller stays";
  EXPECT_EQ(recording.safeStates(), 1u);

  const absl::Status ended = process_->runUntilShutdown([]() { return false; });
  EXPECT_EQ(ended.code(), absl::StatusCode::kInternal);
  EXPECT_NE(ended.message().find("3 contact flags"), std::string::npos) << ended.message();
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

}  // namespace
}  // namespace ocs2::humanoid
