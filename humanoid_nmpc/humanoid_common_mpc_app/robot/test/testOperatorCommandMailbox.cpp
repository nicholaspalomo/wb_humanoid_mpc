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

#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <robot_model/ContactEstimatorRegistry.h>
#include <robot_model/RobotDescription.h>

#include "humanoid_common_mpc_app/robot/JointNamesByIndex.h"
#include "humanoid_common_mpc_app/robot/OperatorCommandMailbox.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/fsm_command.pb.h"
#include "humanoid_mpc_msgs/joint_targets.pb.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "humanoid_mpc_msgs/yaml_document.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"

/*
 * The operator's commands on their way to the realtime thread: the handlers parse and resolve, the realtime side takes
 * what they left - FSM commands in order and once each, the newest posture with every joint target merged into it, the
 * newest dodgeball throw, the gantry height, and controller settings whose estimator is built and kept alive.
 */

namespace ocs2::humanoid {
namespace {

using robot_test::kAtlasUrdf;

class OperatorCommandMailboxTest : public ::testing::Test {
 protected:
  OperatorCommandMailboxTest() : description_(kAtlasUrdf) {}

  std::unique_ptr<OperatorCommandMailbox> makeMailbox(OperatorCommandMailbox::Hooks hooks = OperatorCommandMailbox::Hooks()) {
    OperatorCommandMailbox::Config config;
    config.jointNames = jointNamesByIndex(description_);
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

  static humanoid_mpc_msgs::YamlDocument yaml(const std::string& text) {
    humanoid_mpc_msgs::YamlDocument document;
    document.set_yaml(text);
    return document;
  }

  static std::string goldenDodgeball() {
    std::ifstream file("humanoid_nmpc/humanoid_common_mpc_app/robot/test/data/dodgeball_payload.yaml");
    std::stringstream content;
    content << file.rdbuf();
    return content.str();
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
  mailbox->onFsmCommand(fsmCommand("", /*sequence=*/14));

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

TEST_F(OperatorCommandMailboxTest, TheModeNamesOfTheCommands) {
  EXPECT_EQ(fsmCommandModeName(*parseFsmCommand("MPC_ACTIVE")), "WB_MPC");
  EXPECT_EQ(fsmCommandModeName(*parseFsmCommand("GRAVITY_COMP")), "GRAVITY_COMP");
  EXPECT_EQ(fsmCommandModeName(*parseFsmCommand("SAFETY")), "SAFETY");
  EXPECT_EQ(fsmCommandModeName(*parseFsmCommand("ZERO_TORQUE")), "ZERO_TORQUE");
  EXPECT_TRUE(fsmCommandModeName(*parseFsmCommand("UNLOCK_GANTRY")).empty());
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
  mailbox->onDodgeballThrow(yaml("dodgeball: {mass: 1.0}"));
  EXPECT_FALSE(mailbox->takeDodgeballThrow(throwCommand));
  EXPECT_EQ(mailbox->statistics().dodgeballsRejected, 1u);

  const std::string golden = goldenDodgeball();
  ASSERT_FALSE(golden.empty());
  mailbox->onDodgeballThrow(yaml(golden));
  std::string heavier = golden;
  heavier.replace(heavier.find("mass: 1.2"), std::string("mass: 1.2").size(), "mass: 2.5");
  mailbox->onDodgeballThrow(yaml(heavier));
  ASSERT_TRUE(mailbox->takeDodgeballThrow(throwCommand));
  EXPECT_DOUBLE_EQ(throwCommand.mass, 2.5);
  EXPECT_FALSE(mailbox->takeDodgeballThrow(throwCommand));
}

TEST_F(OperatorCommandMailboxTest, TheGantryHeightFollowsTheSliderClampedOnceItHasMoved) {
  std::vector<double> forwarded;
  OperatorCommandMailbox::Hooks hooks;
  hooks.walkingVelocityCommand = [&](const humanoid_mpc_msgs::WalkingVelocityCommand& command) {
    forwarded.push_back(command.desired_pelvis_height());
  };
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox(std::move(hooks));
  EXPECT_FALSE(mailbox->desiredGantryHeight().has_value());
  humanoid_mpc_msgs::WalkingVelocityCommand command;
  command.set_desired_pelvis_height(0.9);
  mailbox->onWalkingVelocityCommand(command);
  EXPECT_DOUBLE_EQ(*mailbox->desiredGantryHeight(), 0.9);
  command.set_desired_pelvis_height(4.0);
  mailbox->onWalkingVelocityCommand(command);
  EXPECT_DOUBLE_EQ(*mailbox->desiredGantryHeight(), 1.5);
  command.set_desired_pelvis_height(-1.0);
  mailbox->onWalkingVelocityCommand(command);
  EXPECT_DOUBLE_EQ(*mailbox->desiredGantryHeight(), 0.2);
  command.set_desired_pelvis_height(std::numeric_limits<double>::quiet_NaN());
  mailbox->onWalkingVelocityCommand(command);
  EXPECT_DOUBLE_EQ(*mailbox->desiredGantryHeight(), 0.2) << "a height that is not finite is ignored";
  EXPECT_EQ(forwarded.size(), 4u) << "every command reaches the other consumer (an in-process MPC)";
}

TEST_F(OperatorCommandMailboxTest, ControllerSettingsArriveResolvedAndTheirEstimatorsLive) {
  std::vector<std::string> forwarded;
  OperatorCommandMailbox::Hooks hooks;
  hooks.mpcParameters = [&](const humanoid_mpc_msgs::YamlDocument& document) { forwarded.push_back(document.yaml()); };
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox(std::move(hooks));
  mailbox->onMpcParameters(yaml("contactEstimator: always_in_contact\ncontact_wrench_gate:\n  debounceTime: 0.01\n  rampTime: 0.04\n"));
  mailbox->onMpcParameters(yaml("contactEstimator: no_such_estimator\ncontact_wrench_gate:\n  rampTime: 0.08\n"));
  mailbox->onMpcParameters(yaml("contact_wrench_gate:\n  rampTime: -1\n"));
  mailbox->onMpcParameters(yaml("mpc:\n  timeHorizon: 1.0\n"));

  std::vector<ControllerSettingsUpdate> updates;
  while (mailbox->takeControllerSettings([&](const ControllerSettingsUpdate& update) { updates.push_back(update); })) {
  }
  ASSERT_EQ(updates.size(), 2u) << "a negative gate and a document without the keys hand nothing over";
  EXPECT_TRUE(updates[0].hasContactEstimator);
  EXPECT_EQ(updates[0].contactEstimatorName, "always_in_contact");
  ASSERT_NE(updates[0].contactEstimator, nullptr);
  EXPECT_EQ(updates[0].contactEstimator->getName(), "AlwaysInContactEstimator");
  EXPECT_TRUE(updates[0].hasContactWrenchGate);
  EXPECT_DOUBLE_EQ(updates[0].contactWrenchGate.debounceTime, 0.01);
  EXPECT_DOUBLE_EQ(updates[0].contactWrenchGate.rampTime, 0.04);
  // An unknown estimator leaves the estimator as it is; the gate of the same document still goes through.
  EXPECT_FALSE(updates[1].hasContactEstimator);
  EXPECT_TRUE(updates[1].hasContactWrenchGate);
  EXPECT_DOUBLE_EQ(updates[1].contactWrenchGate.rampTime, 0.08);
  EXPECT_EQ(forwarded.size(), 4u) << "every document reaches the in-process MPC's parameter updater too";

  // One instance per name, alive for the mailbox's life: the realtime thread never frees the estimator it replaces.
  const absl::StatusOr<std::shared_ptr<robot::model::ContactEstimator>> again = mailbox->contactEstimator("always_in_contact");
  ASSERT_TRUE(again.ok());
  EXPECT_EQ(again->get(), updates[0].contactEstimator.get());
  EXPECT_GE(updates[0].contactEstimator.use_count(), 2);
  const absl::StatusOr<std::shared_ptr<robot::model::ContactEstimator>> unknown = mailbox->contactEstimator("no_such_estimator");
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(unknown.status().message().find("always_in_contact"), std::string::npos) << "the error lists the available names";
}

TEST_F(OperatorCommandMailboxTest, PdGainsGoToTheController) {
  std::vector<std::string> documents;
  OperatorCommandMailbox::Hooks hooks;
  hooks.pdGainsYaml = [&](absl::string_view text) {
    documents.emplace_back(text);
    return absl::OkStatus();
  };
  std::unique_ptr<OperatorCommandMailbox> mailbox = makeMailbox(std::move(hooks));
  mailbox->onPdGains(yaml("default_gains: {kp: 10, kd: 1}"));
  ASSERT_EQ(documents.size(), 1u);
  EXPECT_EQ(documents[0], "default_gains: {kp: 10, kd: 1}");
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
  // The bus is stopped before the mailbox its handlers reach is destroyed.
  operatorBus->stop();
  robotBus->stop();
  EXPECT_FALSE(mailbox->takeFsmCommand(event)) << "the repeats were dropped";
}

}  // namespace
}  // namespace ocs2::humanoid
