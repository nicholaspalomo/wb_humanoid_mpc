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
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/OperatorPayloadChecks.h"
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_common_mpc_app/robot/JointNamesByIndex.h"
#include "humanoid_common_mpc_app/robot/OperatorCommandMailbox.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "humanoid_mpc_config/joint_pd_gains_file.pb.h"
#include "humanoid_mpc_config/mpc_parameter_update.pb.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/dodgeball_throw.pb.h"
#include "humanoid_mpc_msgs/fsm_command.pb.h"
#include "humanoid_mpc_msgs/joint_targets.pb.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"
#include "nproto/Textproto.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotStateContactEstimator.h"

/*
 * The operator's commands on their way to the realtime thread: the handlers convert and resolve, the realtime side takes
 * what they left - FSM commands in order and once each, the newest posture with every joint target merged into it, the
 * newest dodgeball throw, the gantry height, and controller settings whose estimator is built and kept alive. The GUI's
 * tuning payloads are typed: the whole task file on operator/mpc_parameters,
 * the whole gains file on operator/pd_gains, a DodgeballThrow on operator/dodgeball_throw.
 */

namespace ocs2::humanoid {
namespace {

using ::testing::DoubleEq;
using ::testing::Optional;

class OperatorCommandMailboxTest : public ::testing::Test {
 protected:
  OperatorCommandMailboxTest() : description_(robot_test::atlasDescription()) {}

  std::unique_ptr<OperatorCommandMailbox> makeMailbox(OperatorCommandMailbox::Hooks hooks = OperatorCommandMailbox::Hooks(),
                                                      const std::string& robotName = "") {
    OperatorCommandMailbox::Config config;
    config.jointNames = jointNamesByIndex(description_);
    config.robotName = robotName;
    config.initialNominalPositions.assign(description_.getNumJoints(), 0.25);
    config.commandQueueCapacity = 4;
    absl::StatusOr<std::unique_ptr<OperatorCommandMailbox>> mailbox = OperatorCommandMailbox::Create(config, registry_, std::move(hooks));
    EXPECT_TRUE(mailbox.ok()) << mailbox.status();
    return *std::move(mailbox);
  }

  static humanoid_mpc_msgs::FsmCommand fsmCommand(const std::string& command, uint64_t sequence) {
    humanoid_mpc_msgs::FsmCommand message;
    message.set_command(command);
    message.set_sequence(sequence);
    return message;
  }

  /**
   * The operator/mpc_parameters message of a robot without a contact planner whose task file is `taskText`, stamped
   * with this build's schema fingerprint as the GUI stamps it.
   */
  static humanoid_mpc_config::MpcParameterUpdate parameters(absl::string_view taskText) {
    humanoid_mpc_config::MpcParameterUpdate update;
    const absl::StatusOr<humanoid_mpc_config::TaskFile> task = nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(taskText, "task");
    EXPECT_TRUE(task.ok()) << task.status();
    if (task.ok()) *update.mutable_task() = *task;
    update.set_schema_fingerprint(mpcParameterUpdateSchemaFingerprint());
    return update;
  }

  /** `message` as a sender whose schema has a field 103 more would send it: its bytes and those of that field. */
  template <typename Message>
  static Message withUnknownField(const Message& message) {
    Message received;
    EXPECT_TRUE(received.ParseFromString(message.SerializeAsString() + std::string("\xb8\x06\x01", 3)));
    return received;
  }

  /** The GUI's golden throw (test/data/dodgeball_throw.textproto). */
  static humanoid_mpc_msgs::DodgeballThrow goldenDodgeball() {
    const absl::StatusOr<humanoid_mpc_msgs::DodgeballThrow> golden = nproto::ParseTextprotoFile<humanoid_mpc_msgs::DodgeballThrow>(
        "humanoid_nmpc/humanoid_common_mpc_app/robot/test/data/dodgeball_throw.textproto");
    EXPECT_TRUE(golden.ok()) << golden.status();
    return golden.ok() ? *golden : humanoid_mpc_msgs::DodgeballThrow();
  }

  /** Every controller settings update the mailbox holds, oldest first. */
  static std::vector<ControllerSettingsUpdate> takeAll(OperatorCommandMailbox& mailbox) {
    std::vector<ControllerSettingsUpdate> updates;
    while (mailbox.takeControllerSettings([&](const ControllerSettingsUpdate& update) { updates.push_back(update); })) {
    }
    return updates;
  }

  robot::model::RobotDescription description_;
  robot::model::ContactEstimatorRegistry registry_;
};

TEST_F(OperatorCommandMailboxTest, FsmCommandsArriveInOrderAndOnce) {
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox();
  mailbox->onFsmCommand(fsmCommand("JOINT_PD", /*sequence=*/10));
  mailbox->onFsmCommand(fsmCommand("JOINT_PD", /*sequence=*/10));  // the same message again: a repeat
  mailbox->onFsmCommand(fsmCommand("ENABLE_TORQUES", /*sequence=*/11));
  mailbox->onFsmCommand(fsmCommand("DISABLE_TORQUES", /*sequence=*/12));
  mailbox->onFsmCommand(fsmCommand("LOCK_GANTRY", /*sequence=*/0));  // no sequence: never taken for a repeat
  mailbox->onFsmCommand(fsmCommand("LOCK_GANTRY", /*sequence=*/0));
  mailbox->onFsmCommand(fsmCommand("DANCE", /*sequence=*/13));
  mailbox->onFsmCommand(fsmCommand(/*command=*/"", /*sequence=*/14));

  std::vector<FsmCommandKind> kinds;
  std::vector<std::string> names;
  FsmCommandEvent event;
  while (mailbox->takeFsmCommand(event)) {
    kinds.push_back(event.kind);
    names.emplace_back(fsmCommandEventName(event));
  }
  // The queue holds four: the fifth (the second LOCK_GANTRY) is dropped and counted rather than waited for.
  const std::vector<FsmCommandKind> expected{FsmCommandKind::kJointPd, FsmCommandKind::kWbMpc, FsmCommandKind::kZeroTorque,
                                             FsmCommandKind::kLockGantry};
  EXPECT_EQ(kinds, expected);
  EXPECT_EQ(names, (std::vector<std::string>{"JOINT_PD", "ENABLE_TORQUES", "DISABLE_TORQUES", "LOCK_GANTRY"}));
  const OperatorCommandMailbox::Statistics statistics = mailbox->statistics();
  EXPECT_EQ(statistics.fsmCommandsRepeated, 1u);
  EXPECT_EQ(statistics.fsmCommandsUnknown, 1u);
  EXPECT_EQ(statistics.queueDrops, 1u);
}

/** The mode name of `command`, or "<unknown>" when it is no command. */
std::string modeName(absl::string_view command) {
  const std::optional<FsmCommandKind> kind = parseFsmCommand(command);
  return kind.has_value() ? std::string(fsmCommandModeName(*kind)) : "<unknown>";
}

TEST_F(OperatorCommandMailboxTest, TheModeNamesOfTheCommands) {
  EXPECT_EQ(modeName("MPC_ACTIVE"), "WB_MPC");
  EXPECT_EQ(modeName("GRAVITY_COMP"), "GRAVITY_COMP");
  EXPECT_EQ(modeName("SAFETY"), "SAFETY");
  EXPECT_EQ(modeName("ZERO_TORQUE"), "ZERO_TORQUE");
  EXPECT_EQ(modeName("UNLOCK_GANTRY"), "");
  EXPECT_FALSE(parseFsmCommand("joint_pd").has_value()) << "names are case-sensitive, as the ROS FSM bridge compared them";
}

TEST_F(OperatorCommandMailboxTest, JointTargetsAreMergedIntoThePostureAndTheNewestWins) {
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox();
  std::vector<double> posture(description_.getNumJoints(), 0.0);
  EXPECT_FALSE(mailbox->takeNominalPosture(posture)) << "nothing has arrived";

  const std::string first = description_.getJointName(0);
  const std::string second = description_.getJointName(1);
  humanoid_mpc_msgs::JointTargets targets;
  (*targets.mutable_positions())[first] = 0.5;
  (*targets.mutable_positions())["a_joint_of_another_robot"] = 9.0;
  mailbox->onJointTargets(targets);
  humanoid_mpc_msgs::JointTargets later;
  (*later.mutable_positions())[second] = -0.5;
  (*later.mutable_positions())[first] = std::numeric_limits<double>::quiet_NaN();
  mailbox->onJointTargets(later);

  ASSERT_TRUE(mailbox->takeNominalPosture(posture));
  EXPECT_DOUBLE_EQ(posture[0], 0.5) << "kept from the first message: a target that is not finite is refused";
  EXPECT_DOUBLE_EQ(posture[1], -0.5);
  for (size_t joint = 2; joint < posture.size(); ++joint) EXPECT_DOUBLE_EQ(posture[joint], 0.25) << joint;
  EXPECT_FALSE(mailbox->takeNominalPosture(posture)) << "taken once";
  EXPECT_EQ(mailbox->statistics().jointTargetsRejected, 1u);
}

TEST_F(OperatorCommandMailboxTest, TheNewestDodgeballIsThrownAndARefusedOneIsNot) {
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox();
  robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow throwCommand;
  humanoid_mpc_msgs::DodgeballThrow massOnly;
  massOnly.set_mass(1.0);
  mailbox->onDodgeballThrow(massOnly);
  EXPECT_FALSE(mailbox->takeDodgeballThrow(throwCommand)) << "a throw without its spawn offset, velocity and flight time";
  EXPECT_EQ(mailbox->statistics().dodgeballsRejected, 1u);

  const humanoid_mpc_msgs::DodgeballThrow golden = goldenDodgeball();
  ASSERT_TRUE(golden.has_mass());
  mailbox->onDodgeballThrow(golden);
  humanoid_mpc_msgs::DodgeballThrow heavier = golden;
  heavier.set_mass(2.5);
  mailbox->onDodgeballThrow(heavier);
  ASSERT_TRUE(mailbox->takeDodgeballThrow(throwCommand));
  EXPECT_DOUBLE_EQ(throwCommand.mass, 2.5);
  EXPECT_DOUBLE_EQ(throwCommand.flightTime, golden.flight_time());
  EXPECT_FALSE(mailbox->takeDodgeballThrow(throwCommand));
}

TEST_F(OperatorCommandMailboxTest, TheGantryHeightFollowsTheSliderClampedOnceItHasMoved) {
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox();
  EXPECT_FALSE(mailbox->desiredGantryHeight().has_value());
  humanoid_mpc_msgs::WalkingVelocityCommand command;
  command.set_desired_pelvis_height(0.9);
  mailbox->onWalkingVelocityCommand(command);
  EXPECT_THAT(mailbox->desiredGantryHeight(), Optional(DoubleEq(0.9)));
  command.set_desired_pelvis_height(4.0);
  mailbox->onWalkingVelocityCommand(command);
  EXPECT_THAT(mailbox->desiredGantryHeight(), Optional(DoubleEq(1.5)));
  command.set_desired_pelvis_height(-1.0);
  mailbox->onWalkingVelocityCommand(command);
  EXPECT_THAT(mailbox->desiredGantryHeight(), Optional(DoubleEq(0.2)));
  command.set_desired_pelvis_height(std::numeric_limits<double>::quiet_NaN());
  mailbox->onWalkingVelocityCommand(command);
  EXPECT_THAT(mailbox->desiredGantryHeight(), Optional(DoubleEq(0.2))) << "a height that is not finite is ignored";
}

TEST_F(OperatorCommandMailboxTest, ControllerSettingsArriveResolvedAndTheirEstimatorsLive) {
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox();
  mailbox->onMpcParameters(
      parameters("contact_estimator: \"always_in_contact\"\ncontact_wrench_gate { debounce_time: 0.01 ramp_time: 0.04 }\n"));
  mailbox->onMpcParameters(parameters("contact_estimator: \"no_such_estimator\"\ncontact_wrench_gate { ramp_time: 0.08 }\n"));
  mailbox->onMpcParameters(parameters("contact_estimator: \"robot_state\"\ncontact_wrench_gate { ramp_time: -1 }\n"));
  mailbox->onMpcParameters(parameters("contact_estimator: \"robot_state\"\n"));

  const std::vector<ControllerSettingsUpdate> updates = takeAll(*mailbox);
  ASSERT_EQ(updates.size(), 4u);
  EXPECT_TRUE(updates[0].hasContactEstimator);
  EXPECT_EQ(updates[0].contactEstimatorName, "always_in_contact");
  ASSERT_NE(updates[0].contactEstimator, nullptr);
  EXPECT_EQ(updates[0].contactEstimator->getName(), "AlwaysInContactEstimator");
  EXPECT_TRUE(updates[0].hasContactWrenchGate);
  EXPECT_DOUBLE_EQ(updates[0].contactWrenchGate.debounceTime, 0.01);
  EXPECT_DOUBLE_EQ(updates[0].contactWrenchGate.rampTime, 0.04);
  // An unknown estimator leaves the estimator as it is; the gate of the same file still goes through.
  EXPECT_FALSE(updates[1].hasContactEstimator);
  EXPECT_TRUE(updates[1].hasContactWrenchGate);
  EXPECT_DOUBLE_EQ(updates[1].contactWrenchGate.rampTime, 0.08);
  // A refused gate leaves the gate as it is; the estimator of the same file still goes through.
  EXPECT_TRUE(updates[2].hasContactEstimator);
  EXPECT_EQ(updates[2].contactEstimatorName, "robot_state");
  EXPECT_FALSE(updates[2].hasContactWrenchGate);
  // The payload is the whole file: a file without a contact_wrench_gate block is the instantaneous gate.
  EXPECT_TRUE(updates[3].hasContactWrenchGate);
  EXPECT_EQ(updates[3].contactWrenchGate.debounceTime, ContactWrenchGate::Config{}.debounceTime);
  EXPECT_EQ(updates[3].contactWrenchGate.rampTime, ContactWrenchGate::Config{}.rampTime);

  // One instance per name, alive for the mailbox's life: the realtime thread never frees the estimator it replaces.
  const absl::StatusOr<std::shared_ptr<robot::model::ContactEstimator>> again = mailbox->contactEstimator("always_in_contact");
  ASSERT_TRUE(again.ok());
  EXPECT_EQ(again->get(), updates[0].contactEstimator.get());
  EXPECT_GE(updates[0].contactEstimator.use_count(), 2);
  const absl::StatusOr<std::shared_ptr<robot::model::ContactEstimator>> unknown = mailbox->contactEstimator("no_such_estimator");
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(unknown.status().message().find("always_in_contact"), std::string::npos) << "the error lists the available names";
}

TEST_F(OperatorCommandMailboxTest, AnEstimatorTheCheckRefusesIsNeitherHandedOverNorKept) {
  int checks = 0;
  OperatorCommandMailbox::Hooks hooks;
  hooks.checkContactEstimator = [&checks](robot::model::ContactEstimator& /*estimator*/, absl::string_view name) {
    ++checks;
    return name == "always_in_contact" ? absl::InvalidArgumentError("it reports 1 contact flags") : absl::OkStatus();
  };
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox(std::move(hooks));
  const absl::StatusOr<std::shared_ptr<robot::model::ContactEstimator>> refused = mailbox->contactEstimator("always_in_contact");
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(refused.status().message().find("1 contact flags"), std::string::npos) << refused.status();
  EXPECT_FALSE(mailbox->contactEstimator("always_in_contact").ok()) << "a refused estimator is not kept";
  EXPECT_EQ(checks, 2);
  ASSERT_TRUE(mailbox->contactEstimator("robot_state").ok());
  ASSERT_TRUE(mailbox->contactEstimator("robot_state").ok());
  EXPECT_EQ(checks, 3) << "an accepted estimator is checked once, when it is built";

  mailbox->onMpcParameters(parameters("contact_estimator: \"always_in_contact\"\ncontact_wrench_gate { ramp_time: 0.08 }\n"));
  const std::vector<ControllerSettingsUpdate> updates = takeAll(*mailbox);
  ASSERT_EQ(updates.size(), 1u);
  EXPECT_FALSE(updates[0].hasContactEstimator) << "the refused estimator reached the realtime thread";
  EXPECT_TRUE(updates[0].hasContactWrenchGate);
  EXPECT_EQ(mailbox->statistics().controllerSettingsRejected, 1u);
}

TEST_F(OperatorCommandMailboxTest, AWholeTaskFileThatNamesNothingIsTheDefaultEstimatorAndTheInstantaneousGate) {
  const std::string defaultEstimator = mpc_config::TaskFile{}.contact_estimator;
  registry_.add(defaultEstimator, "stands in for the simulator's estimator",
                []() { return std::make_shared<robot::model::RobotStateContactEstimator>(); });
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox();
  mailbox->onMpcParameters(parameters(""));
  const std::vector<ControllerSettingsUpdate> updates = takeAll(*mailbox);
  ASSERT_EQ(updates.size(), 1u);
  EXPECT_TRUE(updates[0].hasContactEstimator);
  EXPECT_EQ(updates[0].contactEstimatorName, robot::model::ContactEstimatorRegistry::canonicalName(defaultEstimator));
  EXPECT_TRUE(updates[0].hasContactWrenchGate);
  EXPECT_EQ(updates[0].contactWrenchGate.debounceTime, ContactWrenchGate::Config{}.debounceTime);
  EXPECT_EQ(updates[0].contactWrenchGate.rampTime, ContactWrenchGate::Config{}.rampTime);
}

TEST_F(OperatorCommandMailboxTest, PdGainsGoToTheController) {
  std::vector<mpc_config::JointPdGainsFile> received;
  OperatorCommandMailbox::Hooks hooks;
  hooks.pdGains = [&](const mpc_config::JointPdGainsFile& gains) {
    received.push_back(gains);
    return absl::OkStatus();
  };
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox(std::move(hooks));
  humanoid_mpc_config::JointPdGainsFile message;
  message.mutable_default_gains()->set_kp(10.0);
  message.mutable_default_gains()->set_kd(1.0);
  humanoid_mpc_config::JointPdGainsFile::JointGains& joint = *message.add_joint_gains();
  joint.set_joint(description_.getJointName(0));
  joint.set_kp(20.0);
  mailbox->onPdGains(message);
  ASSERT_EQ(received.size(), 1u);
  EXPECT_EQ(received[0].default_gains.kp, 10.0);
  EXPECT_EQ(received[0].default_gains.kd, 1.0);
  EXPECT_FALSE(received[0].default_gains.torque_limit.has_value()) << "an absent gain is the controller's to fill in";
  ASSERT_EQ(received[0].joint_gains.size(), 1u);
  EXPECT_EQ(received[0].joint_gains[0].joint, description_.getJointName(0));
  EXPECT_EQ(received[0].joint_gains[0].kp, 20.0);
  EXPECT_EQ(mailbox->statistics().pdGainsDocuments, 1u);
}

TEST_F(OperatorCommandMailboxTest, APayloadOfAnotherSchemaVersionIsRefusedWhole) {
  int pdGains = 0;
  OperatorCommandMailbox::Hooks hooks;
  hooks.pdGains = [&pdGains](const mpc_config::JointPdGainsFile& /*gains*/) {
    ++pdGains;
    return absl::OkStatus();
  };
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox(std::move(hooks));

  humanoid_mpc_config::JointPdGainsFile gains;
  gains.mutable_default_gains()->set_kp(10.0);
  mailbox->onPdGains(withUnknownField(gains));
  EXPECT_EQ(pdGains, 0) << "a gains file with a field this build does not know reached the controller";
  EXPECT_EQ(mailbox->statistics().pdGainsRejected, 1u);
  mailbox->onPdGains(gains);
  EXPECT_EQ(pdGains, 1);

  mailbox->onMpcParameters(withUnknownField(parameters("contact_estimator: \"robot_state\"\n")));
  humanoid_mpc_config::MpcParameterUpdate unstamped = parameters("contact_estimator: \"robot_state\"\n");
  unstamped.clear_schema_fingerprint();
  mailbox->onMpcParameters(unstamped);
  EXPECT_TRUE(takeAll(*mailbox).empty()) << "nothing of a refused update is applied";
  EXPECT_EQ(mailbox->statistics().controllerSettingsRejected, 2u);

  robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow throwCommand;
  mailbox->onDodgeballThrow(withUnknownField(goldenDodgeball()));
  EXPECT_FALSE(mailbox->takeDodgeballThrow(throwCommand));
  EXPECT_EQ(mailbox->statistics().dodgeballsRejected, 1u);
}

TEST_F(OperatorCommandMailboxTest, AnUpdateOfAnotherRobotIsRefusedWhole) {
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox(OperatorCommandMailbox::Hooks(), /*robotName=*/"unitree_r1");
  mailbox->onMpcParameters(parameters("model_settings { robot_name: \"drc_atlas\" }\ncontact_estimator: \"robot_state\"\n"));
  EXPECT_TRUE(takeAll(*mailbox).empty()) << "another robot's task file reached the controller";
  EXPECT_EQ(mailbox->statistics().controllerSettingsRejected, 1u);
  mailbox->onMpcParameters(parameters("model_settings { robot_name: \"unitree_r1\" }\ncontact_estimator: \"robot_state\"\n"));
  EXPECT_EQ(takeAll(*mailbox).size(), 1u);
}

TEST_F(OperatorCommandMailboxTest, GainsTheControllerRefusesAreCounted) {
  OperatorCommandMailbox::Hooks hooks;
  hooks.pdGains = [](const mpc_config::JointPdGainsFile& /*gains*/) { return absl::InvalidArgumentError("another robot's joints"); };
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox(std::move(hooks));
  mailbox->onPdGains(humanoid_mpc_config::JointPdGainsFile());
  EXPECT_EQ(mailbox->statistics().pdGainsDocuments, 1u);
  EXPECT_EQ(mailbox->statistics().pdGainsRejected, 1u);
}

TEST_F(OperatorCommandMailboxTest, ThePostureMustMatchTheJoints) {
  OperatorCommandMailbox::Config config;
  config.jointNames = jointNamesByIndex(description_);
  config.initialNominalPositions = {0.0};
  EXPECT_EQ(OperatorCommandMailbox::Create(config, registry_).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(OperatorCommandMailboxTest, TheHandlersAreTheBusesSubscriptions) {
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox();
  std::unique_ptr<robot::ipc::Bus> robotBus = robot_test::createLoopbackBus("robot");
  std::unique_ptr<robot::ipc::Bus> operatorBus = robot_test::createLoopbackBus("operator");
  robot_test::connectBoth(*robotBus, *operatorBus);
  ASSERT_TRUE(mailbox->registerOnBus(*robotBus).ok());
  ASSERT_TRUE(robotBus->start().ok());
  ASSERT_TRUE(operatorBus->start().ok());

  // Published until it arrives (the subscription takes a moment to reach the publisher); the sequence makes every
  // copy after the first a repeat.
  FsmCommandEvent event;
  const bool arrived = robot_test::waitFor([&]() {
    operatorBus->publish(ipc::topics::kOperatorFsmCommand, fsmCommand("GRAVITY_COMP", /*sequence=*/7)).IgnoreError();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return mailbox->takeFsmCommand(event);
  });
  ASSERT_TRUE(arrived);
  EXPECT_EQ(event.kind, FsmCommandKind::kGravityComp);
  humanoid_mpc_msgs::WalkingVelocityCommand command;
  command.set_desired_pelvis_height(1.1);
  EXPECT_TRUE(robot_test::waitFor([&]() {
    operatorBus->publish(ipc::topics::kOperatorWalkingVelocityCommand, command).IgnoreError();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return mailbox->desiredGantryHeight().has_value();
  }));
  // The tuning payloads are the typed messages the GUI publishes (the bus drops a message of another type).
  const humanoid_mpc_config::MpcParameterUpdate update = parameters("contact_estimator: \"always_in_contact\"\n");
  std::vector<ControllerSettingsUpdate> updates;
  EXPECT_TRUE(robot_test::waitFor([&]() {
    operatorBus->publish(ipc::topics::kOperatorMpcParameters, update).IgnoreError();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    for (const ControllerSettingsUpdate& taken : takeAll(*mailbox)) updates.push_back(taken);
    return !updates.empty();
  }));
  ASSERT_FALSE(updates.empty());
  EXPECT_EQ(updates.front().contactEstimatorName, "always_in_contact");
  const humanoid_mpc_msgs::DodgeballThrow golden = goldenDodgeball();
  robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow throwCommand;
  EXPECT_TRUE(robot_test::waitFor([&]() {
    operatorBus->publish(ipc::topics::kOperatorDodgeballThrow, golden).IgnoreError();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return mailbox->takeDodgeballThrow(throwCommand);
  }));
  EXPECT_DOUBLE_EQ(throwCommand.mass, golden.mass());
  humanoid_mpc_config::JointPdGainsFile gains;
  gains.mutable_default_gains()->set_kp(10.0);
  EXPECT_TRUE(robot_test::waitFor([&]() {
    operatorBus->publish(ipc::topics::kOperatorPdGains, gains).IgnoreError();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return mailbox->statistics().pdGainsDocuments > 0;
  }));
  // The bus is stopped before the mailbox its handlers reach is destroyed.
  operatorBus->stop();
  robotBus->stop();
  EXPECT_FALSE(mailbox->takeFsmCommand(event)) << "the repeats were dropped";
}

}  // namespace
}  // namespace ocs2::humanoid
