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

#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc_app/robot/FsmStateMailbox.h"
#include "humanoid_common_mpc_app/robot/HardwareFallRecovery.h"
#include "humanoid_common_mpc_app/robot/HardwareFsmBridge.h"
#include "humanoid_common_mpc_app/robot/JointNamesByIndex.h"
#include "humanoid_common_mpc_app/robot/OperatorCommandMailbox.h"
#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"
#include "humanoid_common_mpc_app/robot/RobotProcess.h"
#include "humanoid_common_mpc_app/robot/RobotSupervisorRegistry.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/fsm_command.pb.h"
#include "humanoid_mpc_msgs/fsm_state.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/FakeRobotBackend.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"
#include "humanoid_state_estimator/ContactKinematics.h"
#include "humanoid_state_estimator/StateEstimatorRegistry.h"
#include "robot_ipc/Delivery.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

/*
 * The robot process on a backend that is no simulator: the `hardware`
 * supervisor (HardwareFsmBridge, HardwareFallRecovery) on the test-only
 * FakeRobotBackend, which plugs in as a hardware backend would. The FSM
 * commands switch the backend's torques and the gantry commands are ignored; a
 * tilt past the limit switches the torques off; the supervisor registry refuses
 * the simulator's supervisor for such a backend; and the whole process runs its
 * cycles over the bus on it, with either state estimator.
 */

namespace ocs2::humanoid {
namespace {

using robot_test::FakeRobotBackend;
using robot_test::waitFor;

/** [m] The base height the Atlas stands at. */
constexpr double kStandingHeight = 0.95;
/** [rad] The tilt limit of the tests (sim_max_base_tilt_angle), and a tilt past
 * it. */
constexpr double kMaxTilt = 0.5;
constexpr double kFallenTilt = 1.0;

robot::model::RobotState standingAtlas(
    const robot::model::RobotDescription& description) {
  robot::model::RobotState state(description);
  state.setRootPositionInWorldFrame(vector3_t(0.0, 0.0, kStandingHeight));
  return state;
}

std::unique_ptr<OperatorCommandMailbox> makeMailbox(
    const robot::model::RobotDescription& description,
    const robot::model::ContactEstimatorRegistry& registry) {
  OperatorCommandMailbox::Config config;
  config.jointNames = jointNamesByIndex(description);
  config.initialNominalPositions.assign(description.getNumJoints(), 0.0);
  absl::StatusOr<std::unique_ptr<OperatorCommandMailbox>> mailbox =
      OperatorCommandMailbox::Create(config, &registry);
  EXPECT_TRUE(mailbox.ok()) << mailbox.status();
  return mailbox.ok() ? *std::move(mailbox) : nullptr;
}

void send(OperatorCommandMailbox& mailbox,
          const std::string& command,
          uint64_t sequence) {
  humanoid_mpc_msgs::FsmCommand message;
  message.set_command(command);
  message.set_sequence(sequence);
  mailbox.onFsmCommand(message);
}

/** The codes the log holds, oldest first; drains it. */
std::vector<RealtimeEventCode> drainCodes(RealtimeEventLog& log) {
  std::vector<RealtimeEventCode> codes;
  log.drain(
      [&codes](const RealtimeEvent& event) { codes.push_back(event.code); });
  return codes;
}

TEST(HardwareFsmBridge,
     TheFsmCommandsSwitchTheBackendsTorquesAndTheGantryCommandsAreIgnored) {
  const robot::model::RobotDescription description =
      robot_test::atlasDescription();
  const robot::model::RobotState initial = standingAtlas(description);
  FakeRobotBackend backend(robot_test::atlasDescription(), initial);
  const robot::model::ContactEstimatorRegistry registry;
  const std::unique_ptr<OperatorCommandMailbox> mailbox =
      makeMailbox(description, registry);
  ASSERT_NE(mailbox, nullptr);
  FsmStateMailbox states;
  RealtimeEventLog log(/*capacity=*/16);
  HardwareFsmBridge bridge(description, initial, mailbox.get(), &states,
                           &backend, &log);
  msgs::FsmState state;
  ASSERT_TRUE(states.take(state));
  EXPECT_EQ(state.mode, "ZERO_TORQUE");
  EXPECT_FALSE(state.gantry_locked);

  std::string mode = "ZERO_TORQUE";
  EXPECT_FALSE(bridge.processCommands(mode)) << "no command";
  send(*mailbox, "JOINT_PD", /*sequence=*/1);
  EXPECT_TRUE(bridge.processCommands(mode));
  EXPECT_EQ(mode, "JOINT_PD");
  EXPECT_TRUE(backend.acceptsJointAction());
  EXPECT_EQ(backend.torqueEnables(), 1u);

  // The gantry is not the process's here: the command is reported and changes
  // nothing.
  send(*mailbox, "LOCK_GANTRY", /*sequence=*/2);
  EXPECT_FALSE(bridge.processCommands(mode));
  EXPECT_FALSE(bridge.isGantryLocked());
  EXPECT_EQ(mode, "JOINT_PD");

  send(*mailbox, "WB_MPC", /*sequence=*/3);
  EXPECT_TRUE(bridge.processCommands(mode));
  EXPECT_EQ(backend.torqueEnables(), 1u) << "on already: switched on once";
  send(*mailbox, "ZERO_TORQUE", /*sequence=*/4);
  EXPECT_TRUE(bridge.processCommands(mode));
  EXPECT_EQ(mode, "ZERO_TORQUE");
  EXPECT_FALSE(backend.acceptsJointAction());
  EXPECT_EQ(backend.safeStates(), 1u);
  ASSERT_TRUE(states.take(state));
  EXPECT_EQ(state.mode, "ZERO_TORQUE");
  EXPECT_FALSE(state.gantry_locked);

  EXPECT_EQ(drainCodes(log), (std::vector<RealtimeEventCode>{
                                 RealtimeEventCode::kTorquesEnabled,
                                 RealtimeEventCode::kGantryCommandIgnored,
                                 RealtimeEventCode::kModeCommanded,
                                 RealtimeEventCode::kTorquesDisabled}));

  bridge.publishControllerReset(mode);
  ASSERT_TRUE(states.take(state));
  EXPECT_EQ(state.controller_resets, 1u);
  bridge.setMpcHealthy(/*mpcHealthy=*/false, mode);
  ASSERT_TRUE(states.take(state));
  EXPECT_FALSE(state.mpc_healthy);
}

TEST(HardwareFallRecovery,
     ATiltPastTheLimitUnderTorqueSwitchesTheTorquesOffOnce) {
  const robot::model::RobotDescription description =
      robot_test::atlasDescription();
  const robot::model::RobotState upright = standingAtlas(description);
  FakeRobotBackend backend(robot_test::atlasDescription(), upright);
  RealtimeEventLog log(/*capacity=*/16);
  HardwareFallRecovery recovery(kMaxTilt, &backend, &log);
  const std::vector<scalar_t> nominal(description.getNumJoints(), 0.0);
  robot::model::RobotState fallen = upright;
  fallen.setRootRotationLocalToWorldFrame(quaternion_t(
      Eigen::AngleAxis<scalar_t>(kFallenTilt, vector3_t::UnitY())));

  std::string mode = "WB_MPC";
  EXPECT_FALSE(recovery.update(fallen, nominal, mode).modeChanged)
      << "the torques are off: nothing to switch off";
  backend.enableTorques();
  const FallRecovery::Cycle upCycle = recovery.update(upright, nominal, mode);
  EXPECT_FALSE(upCycle.modeChanged);
  EXPECT_EQ(mode, "WB_MPC");

  const FallRecovery::Cycle fell = recovery.update(fallen, nominal, mode);
  EXPECT_TRUE(fell.modeChanged);
  EXPECT_FALSE(fell.discontinuity)
      << "nothing caught the robot: the plant did not jump";
  EXPECT_FALSE(fell.baseRelocated);
  EXPECT_NEAR(fell.tilt, kFallenTilt, 1.0e-9);
  EXPECT_EQ(mode, "ZERO_TORQUE");
  EXPECT_FALSE(backend.acceptsJointAction());
  EXPECT_FALSE(recovery.update(fallen, nominal, mode).modeChanged)
      << "reported once";
  EXPECT_EQ(
      drainCodes(log),
      (std::vector<RealtimeEventCode>{RealtimeEventCode::kFallTorquesOff}));

  // A limit <= 0 disables the cut-off.
  HardwareFallRecovery disabled(/*maxBaseTiltAngle=*/0.0, &backend, &log);
  backend.enableTorques();
  mode = "JOINT_PD";
  EXPECT_FALSE(disabled.update(fallen, nominal, mode).modeChanged);
  EXPECT_TRUE(backend.acceptsJointAction());
}

TEST(RobotSupervisorRegistry,
     RefusesTheSimulatorsSupervisorWithoutASimulatorAndAnUnknownName) {
  const robot::model::RobotDescription description =
      robot_test::atlasDescription();
  const robot::model::RobotState initial = standingAtlas(description);
  FakeRobotBackend backend(robot_test::atlasDescription(), initial);
  const robot::model::ContactEstimatorRegistry registry;
  const std::unique_ptr<OperatorCommandMailbox> mailbox =
      makeMailbox(description, registry);
  ASSERT_NE(mailbox, nullptr);
  FsmStateMailbox states;
  const RobotSupervisorContext context{.backend = &backend,
                                       .description = &description,
                                       .initialState = &initial,
                                       .commands = mailbox.get(),
                                       .fsmStates = &states};
  const RobotSupervisorRegistry supervisors;
  EXPECT_EQ(supervisors.names(), (std::vector<std::string>{"sim", "hardware"}));

  const absl::StatusOr<RobotSupervisor> sim =
      supervisors.create(kSimSupervisorName, context);
  EXPECT_EQ(sim.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(sim.status().message().find("'fake' is not a simulator"),
            std::string::npos)
      << sim.status();

  const absl::StatusOr<RobotSupervisor> unknown =
      supervisors.create("teleoperator", context);
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(unknown.status().message().find("hardware ("), std::string::npos)
      << unknown.status();

  const absl::StatusOr<RobotSupervisor> hardware =
      supervisors.create(backend.supervisorName(), context);
  ASSERT_TRUE(hardware.ok()) << hardware.status();
  EXPECT_NE(hardware->fsmBridge, nullptr);
  EXPECT_NE(hardware->fallRecovery, nullptr);
}

/** What the operator receives of robot/fsm_state. */
class FsmStateView {
 public:
  void onFsmState(const humanoid_mpc_msgs::FsmState& state) {
    absl::MutexLock lock(mutex_);
    last_ = state;
  }
  std::optional<humanoid_mpc_msgs::FsmState> last() const {
    absl::MutexLock lock(mutex_);
    return last_;
  }

 private:
  mutable absl::Mutex mutex_;
  std::optional<humanoid_mpc_msgs::FsmState> last_ ABSL_GUARDED_BY(mutex_);
};

/** The robot process on the fake backend, with the operator on a loopback bus.
 */
class RobotProcessOnHardwareTest
    : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    description_ = std::make_unique<robot::model::RobotDescription>(
        robot_test::atlasDescription());
    initialState_.emplace(standingAtlas(*description_));
    backend_ = std::make_unique<FakeRobotBackend>(
        robot_test::atlasDescription(), *initialState_);
    backend_->registerContactEstimators(estimators_);
    controller_.setRobotJointNames(description_->getJointNames());
    robotBus_ = robot_test::createLoopbackBus("robot");
    operatorBus_ = robot_test::createLoopbackBus("operator");
    ASSERT_TRUE(operatorBus_
                    ->subscribe<humanoid_mpc_msgs::FsmState>(
                        ipc::topics::kRobotFsmState,
                        robot::ipc::Delivery::kLatest,
                        [this](const humanoid_mpc_msgs::FsmState& state) {
                          view_.onFsmState(state);
                        })
                    .ok());
  }

  void TearDown() override {
    if (process_ != nullptr) process_->stop();
    operatorBus_->stop();
    backend_->stop();
  }

  RobotProcess::Config processConfig(const std::string& stateEstimator) const {
    RobotProcess::Config config;
    config.controlFrequency = 500.0;
    config.initialState = initialState_;
    config.settings.contactEstimator = "robot_state";
    config.settings.telemetrySinks.clear();
    config.settings.stateEstimator = stateEstimator;
    config.settings.fallRecovery.maxBaseTiltAngle = kMaxTilt;
    const ModelSettings modelSettings =
        ModelSettings::Create(robot_test::kAtlasTask, robot_test::kAtlasUrdf,
                              "centroidal_mpc_", /*verbose=*/false)
            .value();
    config.urdfFile = robot_test::kAtlasUrdf;
    config.contactFrameNames = modelSettings.contactNames;
    config.contactParentJointNames = modelSettings.contactParentJointNames;
    // The height the contact frames stand at in the world (kalman's
    // contact_height): the Atlas stands at kStandingHeight with every joint at
    // zero, so it is that height plus the frames' height below the base at zero
    // joints, by the same leg kinematics the estimator runs.
    std::vector<estimation::ContactPointSpec> contacts;
    for (size_t contact = 0; contact < config.contactFrameNames.size();
         ++contact) {
      contacts.push_back(estimation::ContactPointSpec{
          .frameNames = {config.contactFrameNames[contact],
                         config.contactParentJointNames[contact]}});
    }
    absl::StatusOr<std::unique_ptr<estimation::ContactKinematics>> kinematics =
        estimation::ContactKinematics::Create(robot_test::kAtlasUrdf, contacts,
                                              description_->getJointNames());
    EXPECT_TRUE(kinematics.ok()) << kinematics.status();
    if (kinematics.ok()) {
      const vector_t zero = vector_t::Zero(
          static_cast<Eigen::Index>(description_->getNumJoints()));
      (*kinematics)->update(zero, zero);
      config.settings.kalmanStateEstimator.contactHeight =
          kStandingHeight + (*kinematics)->contactPositionInBase(0).z();
      EXPECT_NEAR((*kinematics)->contactPositionInBase(0).z(),
                  (*kinematics)->contactPositionInBase(1).z(), 1.0e-9)
          << "a symmetric stance";
    }
    return config;
  }

  bool command(const std::string& name, const std::function<bool()>& done) {
    humanoid_mpc_msgs::FsmCommand message;
    message.set_command(name);
    message.set_sequence(++sequence_);
    return waitFor([&]() {
      operatorBus_->publish(ipc::topics::kOperatorFsmCommand, message)
          .IgnoreError();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      return done();
    });
  }

  std::unique_ptr<robot::model::RobotDescription> description_;
  std::optional<robot::model::RobotState> initialState_;
  robot::model::ContactEstimatorRegistry estimators_;
  std::unique_ptr<FakeRobotBackend> backend_;
  std::unique_ptr<robot::ipc::Bus> robotBus_;
  std::unique_ptr<robot::ipc::Bus> operatorBus_;
  robot_test::ScriptedRobotController controller_;
  std::unique_ptr<RobotProcess> process_;
  FsmStateView view_;
  uint64_t sequence_ = 0;
};

TEST_P(RobotProcessOnHardwareTest, RunsItsCyclesOnABackendThatIsNoSimulator) {
  absl::StatusOr<std::unique_ptr<RobotProcess>> created =
      RobotProcess::Create(robotBus_.get(), backend_.get(), &controller_,
                           &estimators_, processConfig(GetParam()));
  ASSERT_TRUE(created.ok()) << created.status();
  process_ = *std::move(created);
  ASSERT_TRUE(robotBus_->start().ok());
  ASSERT_TRUE(operatorBus_->start().ok());
  ASSERT_TRUE(process_->start().ok());
  ASSERT_TRUE(waitFor([&]() {
    return controller_.cycles() > 50 && backend_->statesPublished() > 50;
  }));
  EXPECT_EQ(backend_->actionsExecuted(), 0u) << "the torques start off";

  ASSERT_TRUE(
      command("JOINT_PD", [&]() { return controller_.mode() == "JOINT_PD"; }));
  EXPECT_TRUE(backend_->acceptsJointAction());
  ASSERT_TRUE(waitFor([&]() { return backend_->actionsExecuted() > 50; }))
      << "the actions reach the drives";
  ASSERT_TRUE(waitFor([&]() {
    const std::optional<humanoid_mpc_msgs::FsmState> state = view_.last();
    return state.has_value() && state->mode() == "JOINT_PD";
  }));
  const std::optional<humanoid_mpc_msgs::FsmState> lastState = view_.last();
  if (!lastState.has_value()) {
    FAIL() << "the operator has received no FSM state";
  }
  EXPECT_FALSE(lastState->gantry_locked());

  // A standing robot stays where it stands, measured or estimated: with its
  // contact_height where the feet stand, the estimate of a robot at rest is the
  // state it was reset to, and a centimeter is far above its noise model's
  // drift.
  constexpr double kStandingTolerance = 0.01;
  if (!initialState_.has_value()) {
    FAIL() << "SetUp() builds the initial state";
  }
  EXPECT_LT((controller_.rootPosition() -
             initialState_->getRootPositionInWorldFrame())
                .norm(),
            kStandingTolerance);
  EXPECT_LT(controller_.rootLinearVelocity().norm(), kStandingTolerance);

  ASSERT_TRUE(command("ZERO_TORQUE",
                      [&]() { return controller_.mode() == "ZERO_TORQUE"; }));
  EXPECT_FALSE(backend_->acceptsJointAction());
  process_->stop();
  EXPECT_FALSE(process_->faulted());
}

INSTANTIATE_TEST_SUITE_P(
    StateEstimators,
    RobotProcessOnHardwareTest,
    ::testing::Values(std::string(estimation::kGroundTruthStateEstimatorName),
                      std::string(estimation::kKalmanStateEstimatorName)));

}  // namespace
}  // namespace ocs2::humanoid
