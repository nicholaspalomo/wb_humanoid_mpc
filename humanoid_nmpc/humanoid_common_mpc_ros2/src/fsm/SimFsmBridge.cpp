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


#include "humanoid_common_mpc_ros2/fsm/SimFsmBridge.h"

#include <absl/log/log.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
SimFsmBridge::SimFsmBridge(const robot::model::RobotDescription& robotDescription,
                           const robot::model::RobotState& initState,
                           rclcpp::Node::SharedPtr nodeHandle)
    : nodeHandle_(std::move(nodeHandle)) {
  nominalJointPositions_.resize(robotDescription.getNumJoints(), 0.0);
  for (size_t i = 0; i < robotDescription.getNumJoints(); ++i) {
    nominalJointPositions_[i] = initState.getJointPosition(i);
  }

  // Initialize gantry height from the robot's spawn Z position
  desiredGantryHeight_ = initState.getRootPositionInWorldFrame().z();

  // Latched / Transient-local QoS for state topic so subscribers get current state immediately
  rclcpp::QoS stateQos(1);
  stateQos.reliable();
  stateQos.transient_local();

  fsmStatePub_ = nodeHandle_->create_publisher<std_msgs::msg::String>("/humanoid/fsm_state", stateQos);

  // Best-effort / reliable command subscriber
  rclcpp::QoS cmdQos(10);
  cmdQos.reliable();

  fsmCommandSub_ = nodeHandle_->create_subscription<std_msgs::msg::String>(
      "/humanoid/fsm_command", cmdQos,
      [this](const std_msgs::msg::String::ConstSharedPtr& msg) { fsmCommandCallback(msg); });

  // Subscribe to walking velocity command for gantry height control
  rclcpp::QoS velQos(1);
  velQos.best_effort();
  walkingVelSub_ = nodeHandle_->create_subscription<humanoid_mpc_msgs::msg::WalkingVelocityCommand>(
      "/humanoid/walking_velocity_command", velQos,
      [this](const humanoid_mpc_msgs::msg::WalkingVelocityCommand::ConstSharedPtr& msg) {
        walkingVelocityCallback(msg);
      });

  // Publish initial zero-torque + locked state
  publishFsmState("ZERO_TORQUE", true);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void SimFsmBridge::fsmCommandCallback(const std_msgs::msg::String::ConstSharedPtr& msg) {
  if (msg) {
    std::lock_guard<std::mutex> lock(commandMutex_);
    pendingCommand_ = msg->data;
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void SimFsmBridge::publishFsmState(std::string_view modeName, bool gantryLocked) const {
  if (fsmStatePub_) {
    std_msgs::msg::String msg;
    msg.data = std::string(modeName) + (gantryLocked ? ",GANTRY_LOCKED" : ",GANTRY_UNLOCKED");
    fsmStatePub_->publish(msg);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void SimFsmBridge::applyModeAction(std::string_view modeName,
                                   const robot::model::RobotDescription& robotDescription,
                                   robot::model::RobotJointAction& robotJointAction) const {
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool SimFsmBridge::processCommands(std::string& currentModeName,
                                   robot::mujoco_sim_interface::MujocoSimInterface& robotInterface) {
  std::optional<std::string> cmdOpt;
  {
    std::lock_guard<std::mutex> lock(commandMutex_);
    if (pendingCommand_.has_value()) {
      cmdOpt = std::move(pendingCommand_);
      pendingCommand_.reset();
    }
  }

  if (!cmdOpt.has_value() || cmdOpt->empty()) {
    // No FSM command pending, but still update gantry height from slider if locked
    // Only override if we've actually received a height command from the GUI
    if (robotInterface.isGantryLocked() && hasReceivedGantryHeight_.load()) {
      robotInterface.setGantryHeight(desiredGantryHeight_.load());
    }
    return false;
  }

  const std::string& cmd = *cmdOpt;

  // 1. Zero-torque mode commands
  if (cmd == "ZERO_TORQUE" || cmd == "DISABLE_TORQUES") {
    if (!robotInterface.isZeroTorqueMode()) {
      LOG(INFO) << "FSM command received: " << cmd << " - zero-torque mode.";
      robotInterface.disableTorques();
    }
    currentModeName = "ZERO_TORQUE";
    publishFsmState(currentModeName, robotInterface.isGantryLocked());
    return true;
  }

  // 2. Active torque modes (JOINT_PD, GRAVITY_COMP, WB_MPC, SAFETY, MPC_ACTIVE, ENABLE_TORQUES)
  if (cmd == "JOINT_PD" || cmd == "GRAVITY_COMP" || cmd == "WB_MPC" || cmd == "SAFETY" || cmd == "MPC_ACTIVE" ||
      cmd == "ENABLE_TORQUES") {
    if (robotInterface.isZeroTorqueMode()) {
      LOG(INFO) << "FSM command received: " << cmd << " - enabling torques.";
      robotInterface.enableTorques();
    }
    currentModeName = (cmd == "ENABLE_TORQUES" || cmd == "MPC_ACTIVE") ? "WB_MPC" : cmd;
    publishFsmState(currentModeName, robotInterface.isGantryLocked());
    return true;
  }

  // 3. Virtual gantry locking / unlocking commands
  if (cmd == "LOCK_GANTRY" && !robotInterface.isGantryLocked()) {
    LOG(INFO) << "FSM command received: Locking gantry.";
    robotInterface.lockGantry();
    publishFsmState(currentModeName, robotInterface.isGantryLocked());
    return true;
  }

  if (cmd == "UNLOCK_GANTRY" && robotInterface.isGantryLocked()) {
    LOG(INFO) << "FSM command received: Unlocking gantry.";
    robotInterface.unlockGantry();
    publishFsmState(currentModeName, robotInterface.isGantryLocked());
    return true;
  }

  return false;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void SimFsmBridge::walkingVelocityCallback(
    const humanoid_mpc_msgs::msg::WalkingVelocityCommand::ConstSharedPtr& msg) {
  desiredGantryHeight_ = std::clamp(static_cast<double>(msg->desired_pelvis_height), 0.2, 1.5);
  hasReceivedGantryHeight_.store(true);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
robot::model::RobotState createInitialSimState(const robot::model::RobotDescription& robotDescription,
                                              const ModelSettings& modelSettings,
                                              const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                              const vector_t& initMpcState) {
  robot::model::RobotState initState(robotDescription, 2);
  initState.setConfigurationToZero();

  initState.setRootPositionInWorldFrame(mpcRobotModel.getBasePosition(initMpcState));
  vector3_t baseOriEulerZyx = mpcRobotModel.getBaseOrientationEulerZYX(initMpcState);
  initState.setRootRotationLocalToWorldFrame(getQuaternionFromEulerAnglesZyx(baseOriEulerZyx));

  vector_t mpcJointAngles = mpcRobotModel.getJointAngles(initMpcState);
  std::vector<robot::joint_index_t> mpcJointIndices = robotDescription.getJointIndices(modelSettings.mpcModelJointNames);
  for (size_t i = 0; i < mpcJointIndices.size(); ++i) {
    initState.setJointPosition(mpcJointIndices[i], mpcJointAngles[i]);
  }

  return initState;
}

}  // namespace ocs2::humanoid
