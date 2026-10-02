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

#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <robot_model/ContactEstimatorRegistry.h>

#include "humanoid_common_mpc_app/robot/FsmStateMailbox.h"
#include "humanoid_common_mpc_app/robot/JointNamesByIndex.h"
#include "humanoid_common_mpc_app/robot/OperatorCommandMailbox.h"
#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"
#include "humanoid_common_mpc_app/robot/SimFsmBridge.h"
#include "humanoid_mpc_msgs/fsm_command.pb.h"
#include "humanoid_mpc_msgs/joint_targets.pb.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "humanoid_mpc_msgs/yaml_document.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"

/*
 * The FSM of the simulated robot against the headless simulator: the commands it takes from the operator's mailbox,
 * one per cycle and in order, the torque switch and the gantry they move, the FSM state it hands to the communication
 * thread (the mode, the gantry, the controller resets the remote control re-centers its joysticks on, and the MPC's
 * health), the gantry height slider, and the joint targets.
 */

namespace ocs2::humanoid {
namespace {

using robot::mujoco_sim_interface::MujocoSimInterface;

class SimFsmBridgeTest : public ::testing::Test {
 protected:
  SimFsmBridgeTest() : sim_(robot_test::makeHeadlessAtlas(/*gantryLocked=*/true)) {
    const robot::model::RobotDescription& description = sim_->getRobotDescription();
    OperatorCommandMailbox::Config config;
    config.jointNames = jointNamesByIndex(description);
    config.initialNominalPositions.assign(description.getNumJoints(), 0.0);
    absl::StatusOr<std::unique_ptr<OperatorCommandMailbox>> mailbox = OperatorCommandMailbox::Create(config, registry_);
    EXPECT_TRUE(mailbox.ok()) << mailbox.status();
    mailbox_ = *std::move(mailbox);
    sim_->initSim();
    sim_->updateInterfaceStateFromRobot();
    bridge_ = std::make_unique<SimFsmBridge>(description, sim_->getRobotState(), *mailbox_, states_, &log_);
  }

  void send(const std::string& command) {
    humanoid_mpc_msgs::FsmCommand message;
    message.set_command(command);
    message.set_sequence(++sequence_);
    mailbox_->onFsmCommand(message);
  }

  /** The state written since the last call, or nullopt. */
  std::optional<msgs::FsmState> published() {
    msgs::FsmState state;
    if (!states_.take(state)) return std::nullopt;
    return state;
  }

  robot::model::ContactEstimatorRegistry registry_;
  std::unique_ptr<MujocoSimInterface> sim_;
  std::unique_ptr<OperatorCommandMailbox> mailbox_;
  FsmStateMailbox states_;
  RealtimeEventLog log_;
  std::unique_ptr<SimFsmBridge> bridge_;
  uint64_t sequence_ = 0;
};

TEST_F(SimFsmBridgeTest, StartsInZeroTorqueOnTheGantry) {
  const std::optional<msgs::FsmState> state = published();
  ASSERT_TRUE(state.has_value());
  EXPECT_EQ(state->mode, "ZERO_TORQUE");
  EXPECT_TRUE(state->gantry_locked);
  EXPECT_EQ(state->controller_resets, 0u);
  EXPECT_TRUE(state->mpc_healthy);
  EXPECT_TRUE(sim_->isZeroTorqueMode());
}

TEST_F(SimFsmBridgeTest, ModeCommandsSwitchTheTorquesAndPublishTheMode) {
  published();
  std::string mode = "ZERO_TORQUE";
  send("JOINT_PD");
  EXPECT_TRUE(bridge_->processCommands(mode, *sim_));
  EXPECT_EQ(mode, "JOINT_PD");
  EXPECT_FALSE(sim_->isZeroTorqueMode());
  EXPECT_EQ(published()->mode, "JOINT_PD");

  send("MPC_ACTIVE");
  EXPECT_TRUE(bridge_->processCommands(mode, *sim_));
  EXPECT_EQ(mode, "WB_MPC") << "the older name of WB_MPC";
  send("ZERO_TORQUE");
  EXPECT_TRUE(bridge_->processCommands(mode, *sim_));
  EXPECT_EQ(mode, "ZERO_TORQUE");
  EXPECT_TRUE(sim_->isZeroTorqueMode());
  send("ENABLE_TORQUES");
  EXPECT_TRUE(bridge_->processCommands(mode, *sim_));
  EXPECT_EQ(mode, "WB_MPC");
  EXPECT_FALSE(sim_->isZeroTorqueMode());
  // The torque switch was reported for the communication thread to log.
  std::vector<RealtimeEventCode> codes;
  log_.drain([&](const RealtimeEvent& event) { codes.push_back(event.code); });
  EXPECT_EQ(codes, (std::vector<RealtimeEventCode>{RealtimeEventCode::kTorquesEnabled, RealtimeEventCode::kTorquesDisabled,
                                                   RealtimeEventCode::kTorquesEnabled}));
}

TEST_F(SimFsmBridgeTest, OneCommandPerCycleInTheOrderTheyWereSent) {
  std::string mode = "ZERO_TORQUE";
  send("JOINT_PD");
  send("GRAVITY_COMP");
  send("SAFETY");
  EXPECT_TRUE(bridge_->processCommands(mode, *sim_));
  EXPECT_EQ(mode, "JOINT_PD");
  EXPECT_TRUE(bridge_->processCommands(mode, *sim_));
  EXPECT_EQ(mode, "GRAVITY_COMP");
  EXPECT_TRUE(bridge_->processCommands(mode, *sim_));
  EXPECT_EQ(mode, "SAFETY");
  EXPECT_FALSE(bridge_->processCommands(mode, *sim_));
}

TEST_F(SimFsmBridgeTest, TheGantryMovesOnlyWhenItWouldChange) {
  published();
  std::string mode = "JOINT_PD";
  send("LOCK_GANTRY");
  EXPECT_FALSE(bridge_->processCommands(mode, *sim_)) << "locked already: nothing to do";
  EXPECT_FALSE(published().has_value());
  send("UNLOCK_GANTRY");
  EXPECT_TRUE(bridge_->processCommands(mode, *sim_));
  EXPECT_FALSE(sim_->isGantryLocked());
  const std::optional<msgs::FsmState> unlocked = published();
  ASSERT_TRUE(unlocked.has_value());
  EXPECT_FALSE(unlocked->gantry_locked);
  EXPECT_EQ(unlocked->mode, "JOINT_PD");
  send("LOCK_GANTRY");
  EXPECT_TRUE(bridge_->processCommands(mode, *sim_));
  EXPECT_TRUE(sim_->isGantryLocked());
  EXPECT_TRUE(published()->gantry_locked);
}

TEST_F(SimFsmBridgeTest, TheGantryFollowsTheSliderWhileLockedAndOnlyOnceItHasMoved) {
  std::string mode = "JOINT_PD";
  const double spawnHeight = sim_->getGantryHeight();
  EXPECT_FALSE(bridge_->processCommands(mode, *sim_));
  EXPECT_DOUBLE_EQ(sim_->getGantryHeight(), spawnHeight) << "no walking command yet: the gantry stays where it is";
  humanoid_mpc_msgs::WalkingVelocityCommand command;
  command.set_desired_pelvis_height(0.7);
  mailbox_->onWalkingVelocityCommand(command);
  bridge_->processCommands(mode, *sim_);
  EXPECT_DOUBLE_EQ(sim_->getGantryHeight(), 0.7);
  // A cycle that processes a command leaves the height alone, as the ROS bridge did.
  command.set_desired_pelvis_height(0.8);
  mailbox_->onWalkingVelocityCommand(command);
  send("GRAVITY_COMP");
  bridge_->processCommands(mode, *sim_);
  EXPECT_DOUBLE_EQ(sim_->getGantryHeight(), 0.7);
  bridge_->processCommands(mode, *sim_);
  EXPECT_DOUBLE_EQ(sim_->getGantryHeight(), 0.8);
  // Unlocked, the slider does not move the gantry.
  sim_->unlockGantry();
  command.set_desired_pelvis_height(1.2);
  mailbox_->onWalkingVelocityCommand(command);
  bridge_->processCommands(mode, *sim_);
  EXPECT_DOUBLE_EQ(sim_->getGantryHeight(), 0.8);
}

TEST_F(SimFsmBridgeTest, EveryControllerResetIsCountedAndTheHealthIsPublishedOnChange) {
  published();
  bridge_->publishControllerReset("JOINT_PD", /*gantryLocked=*/true);
  bridge_->publishControllerReset("JOINT_PD", /*gantryLocked=*/true);
  const std::optional<msgs::FsmState> reset = published();
  ASSERT_TRUE(reset.has_value());
  EXPECT_EQ(reset->controller_resets, 2u) << "the newest state carries every reset so far";
  EXPECT_EQ(bridge_->controllerResets(), 2u);

  bridge_->setMpcHealthy(/*mpcHealthy=*/true, "WB_MPC", /*gantryLocked=*/false);
  EXPECT_FALSE(published().has_value()) << "healthy already: nothing to publish";
  bridge_->setMpcHealthy(/*mpcHealthy=*/false, "WB_MPC", /*gantryLocked=*/false);
  const std::optional<msgs::FsmState> unhealthy = published();
  ASSERT_TRUE(unhealthy.has_value());
  EXPECT_FALSE(unhealthy->mpc_healthy);
  EXPECT_EQ(unhealthy->controller_resets, 2u);
  bridge_->publishFsmState("WB_MPC", /*gantryLocked=*/false);
  EXPECT_FALSE(published()->mpc_healthy) << "every state carries the health";
}

TEST_F(SimFsmBridgeTest, JointTargetsMoveTheNominalPosture) {
  const std::vector<scalar_t> before = bridge_->getNominalJointPositions();
  bridge_->applyJointTargetUpdates();
  EXPECT_EQ(bridge_->getNominalJointPositions(), before) << "nothing has arrived";
  humanoid_mpc_msgs::JointTargets targets;
  const std::string joint = sim_->getRobotDescription().getJointName(3);
  (*targets.mutable_positions())[joint] = 0.42;
  mailbox_->onJointTargets(targets);
  bridge_->applyJointTargetUpdates();
  EXPECT_DOUBLE_EQ(bridge_->getNominalJointPositions()[3], 0.42);
}

TEST_F(SimFsmBridgeTest, ADodgeballIsHandedToTheSimulatorBeforeTheCommands) {
  std::ifstream file("humanoid_nmpc/humanoid_common_mpc_app/robot/test/data/dodgeball_payload.yaml");
  std::stringstream payload;
  payload << file.rdbuf();
  humanoid_mpc_msgs::YamlDocument document;
  document.set_yaml(payload.str());
  mailbox_->onDodgeballThrow(document);
  std::string mode = "JOINT_PD";
  EXPECT_FALSE(bridge_->processCommands(mode, *sim_)) << "a throw is not an FSM command";
  robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow left;
  EXPECT_FALSE(mailbox_->takeDodgeballThrow(left)) << "taken by the bridge";
}

}  // namespace
}  // namespace ocs2::humanoid
