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

#include <algorithm>
#include <cmath>

#include <absl/log/log.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <yaml-cpp/yaml.h>

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
SimFsmBridge::SimFsmBridge(const robot::model::RobotDescription& robotDescription,
                           const robot::model::RobotState& initState,
                           rclcpp::Node::SharedPtr nodeHandle)
    : nodeHandle_(std::move(nodeHandle)) {
  nominalJointPositions_.resize(robotDescription.getNumJoints(), 0.0);
  allJointNames_ = robotDescription.getJointNames();
  const auto& jointIdxVec = robotDescription.getJointIndices();
  allJointIndices_.assign(jointIdxVec.begin(), jointIdxVec.end());
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
      "/humanoid/fsm_command", cmdQos, [this](const std_msgs::msg::String::ConstSharedPtr& msg) { fsmCommandCallback(msg); });

  // Dodgeball throws from the GUI. RELIABLE like the FSM command and for the same reason: a throw is one event, so
  // a dropped message is a button press that did nothing rather than a value the next message corrects.
  dodgeballSub_ = nodeHandle_->create_subscription<std_msgs::msg::String>(
      "/humanoid/dodgeball_throw", cmdQos, [this](const std_msgs::msg::String::ConstSharedPtr& msg) { dodgeballCallback(msg); });

  // Subscribe to walking velocity command for gantry height control
  rclcpp::QoS velQos(1);
  velQos.best_effort();
  walkingVelSub_ = nodeHandle_->create_subscription<humanoid_mpc_msgs::msg::WalkingVelocityCommand>(
      "/humanoid/walking_velocity_command", velQos,
      [this](const humanoid_mpc_msgs::msg::WalkingVelocityCommand::ConstSharedPtr& msg) { walkingVelocityCallback(msg); });

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
void SimFsmBridge::dodgeballCallback(const std_msgs::msg::String::ConstSharedPtr& msg) {
  if (!msg) return;

  // The geometry was computed by the GUI (remote_control/tk_app/dodgeball.py) and is not recomputed here: this reads
  // the spawn offset, the launch velocity and the flight time it already worked out, in the robot's yaw frame. The
  // operator's four slider values travel in the same payload but are documentation - nothing below reads them.
  robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow command;
  try {
    const YAML::Node root = YAML::Load(msg->data);
    const YAML::Node ball = root["dodgeball"];
    if (!ball) {
      LOG(WARNING) << "Dodgeball throw ignored: the payload has no 'dodgeball' block.";
      return;
    }
    const YAML::Node offset = ball["spawnOffset"];
    const YAML::Node velocity = ball["launchVelocity"];
    if (!offset || offset.size() != 3 || !velocity || velocity.size() != 3) {
      LOG(WARNING) << "Dodgeball throw ignored: spawnOffset and launchVelocity must both be three numbers.";
      return;
    }
    for (size_t axis = 0; axis < 3; ++axis) {
      command.spawnOffset[axis] = offset[axis].as<double>();
      command.launchVelocity[axis] = velocity[axis].as<double>();
    }
    command.flightTime = ball["flightTime"] ? ball["flightTime"].as<double>() : 0.0;
    // The GUI always sends a mass; this fallback is only for a payload published by hand. It has to match the
    // registry's nominal mass, or a hand-thrown ball would weigh something the operator never asked for.
    // LINT.IfChange(dodgeball_fallback_mass)
    command.mass = ball["mass"] ? ball["mass"].as<double>() : 0.45;
    // LINT.ThenChange(//robot_runtime/mujoco_sim_interface/src/Projectile.cpp:dodgeball_properties)
  } catch (const std::exception& error) {
    LOG(WARNING) << "Dodgeball throw ignored: the payload could not be read as YAML: " << error.what();
    return;
  }

  if (!(command.mass > 0.0) || command.flightTime < 0.0) {
    LOG(WARNING) << "Dodgeball throw ignored: a ball needs a positive mass and a non-negative flight time.";
    return;
  }

  std::lock_guard<std::mutex> lock(dodgeballMutex_);
  pendingDodgeball_ = command;
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
                                   robot::model::RobotJointAction& robotJointAction) const {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t SimFsmBridge::baseTiltAngle(const quaternion_t& baseRotationLocalToWorld) {
  // The base's own vertical, expressed in the world: the third column of its rotation matrix. Its angle to the world
  // vertical is the arccosine of that column's z component, which is heading independent, so a robot that has turned
  // on the spot reads zero tilt exactly like one that has not. The clamp keeps a matrix entry that rounds just past
  // one from producing a NaN, which would compare false against the threshold and silently disable the recovery.
  const matrix3_t baseRotation = baseRotationLocalToWorld.toRotationMatrix();
  return std::acos(std::clamp(baseRotation(2, 2), scalar_t(-1.0), scalar_t(1.0)));
}

/******************************************************************************************************/
bool SimFsmBridge::recoverFromFall(const robot::model::RobotState& robotState,
                                   robot::mujoco_sim_interface::MujocoSimInterface& robotInterface,
                                   std::string& currentModeName) {
  if (maxBaseTiltAngle_ <= 0.0 || robotInterface.isGantryLocked()) return false;
  const scalar_t tilt = baseTiltAngle(robotState.getRootRotationLocalToWorldFrame());
  if (tilt <= maxBaseTiltAngle_) return false;

  LOG(INFO) << "Base tilted " << tilt << " rad past the " << maxBaseTiltAngle_
            << " rad limit — catching the robot on the gantry in JOINT_PD.";
  robotInterface.lockGantry();
  if (robotInterface.isZeroTorqueMode()) robotInterface.enableTorques();
  currentModeName = "JOINT_PD";
  publishFsmState(currentModeName, robotInterface.isGantryLocked());
  return true;
}

/******************************************************************************************************/
bool SimFsmBridge::processCommands(std::string& currentModeName, robot::mujoco_sim_interface::MujocoSimInterface& robotInterface) {
  // Handed over first and unconditionally: a throw is independent of the FSM, and the early return below fires on
  // every cycle in which no mode change is pending - which is almost all of them.
  {
    std::optional<robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow> throwOpt;
    {
      std::lock_guard<std::mutex> lock(dodgeballMutex_);
      throwOpt.swap(pendingDodgeball_);
    }
    if (throwOpt.has_value()) {
      robotInterface.throwDodgeball(*throwOpt);
    }
  }

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
  if (cmd == "JOINT_PD" || cmd == "GRAVITY_COMP" || cmd == "WB_MPC" || cmd == "SAFETY" || cmd == "MPC_ACTIVE" || cmd == "ENABLE_TORQUES") {
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
void SimFsmBridge::walkingVelocityCallback(const humanoid_mpc_msgs::msg::WalkingVelocityCommand::ConstSharedPtr& msg) {
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

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void SimFsmBridge::subscribeJointTargets(rclcpp::Node::SharedPtr node) {
  jointTargetSubscriber_.subscribe(std::move(node));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void SimFsmBridge::applyJointTargetUpdates() {
  jointTargetSubscriber_.applyPendingUpdates(nominalJointPositions_, allJointNames_, allJointIndices_);
}

}  // namespace ocs2::humanoid
