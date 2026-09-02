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

#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <string_view>
#include <humanoid_common_mpc/common/ModelSettings.h>
#include <humanoid_common_mpc/common/MpcRobotModelBase.h>
#include <mujoco_sim_interface/MujocoSimInterface.h>
#include <rclcpp/rclcpp.hpp>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>
#include <std_msgs/msg/string.hpp>

namespace ocs2::humanoid {

/**
 * @brief ROS 2-native bridge between supervisory FSM commands/state and the simulation loop.
 *
 * Encapsulates:
 * 1. Tracking and storing nominal stance positions.
 * 2. Overriding actuator commands in JOINT_PD mode to strictly track nominal posture with 0 velocity and 0 feedforward effort.
 * 3. Subscribing to ROS 2 topic `/humanoid/fsm_command` (std_msgs/msg/String) and processing mode transitions.
 * 4. Publishing ROS 2 topic `/humanoid/fsm_state` (std_msgs/msg/String) with transient-local QoS.
 * 5. Virtual gantry lock and unlock management.
 */
class SimFsmBridge {
 public:
  /**
   * @brief Construct the FSM bridge with nominal stance joint positions and ROS 2 pub/sub.
   * @param robotDescription Robot kinematic and dynamic description.
   * @param initState Initial robot state from which nominal joint positions are captured.
   * @param nodeHandle Active ROS 2 node handle for topic creation.
   */
  SimFsmBridge(const robot::model::RobotDescription& robotDescription,
               const robot::model::RobotState& initState,
               rclcpp::Node::SharedPtr nodeHandle);

  /**
   * @brief Publishes the current FSM mode and gantry state to `/humanoid/fsm_state`.
   * @param modeName Current active mode name (e.g., "ZERO_TORQUE", "JOINT_PD", "WB_MPC").
   * @param gantryLocked Whether the gantry is currently locked.
   */
  void publishFsmState(std::string_view modeName, bool gantryLocked) const;

  /**
   * @brief Applies mode-specific control overrides (e.g. JOINT_PD) to the joint action vector.
   * @param modeName Current active mode name.
   * @param robotDescription Robot description.
   * @param robotJointAction Joint action vector to be updated in-place.
   */
  void applyModeAction(std::string_view modeName,
                       const robot::model::RobotDescription& robotDescription,
                       robot::model::RobotJointAction& robotJointAction) const;

  /**
   * @brief Processes any pending ROS 2 commands received from `/humanoid/fsm_command`,
   *        applies mode/gantry changes to robotInterface, and publishes updated state.
   * @param currentModeName Reference to the current mode name string (updated if changed).
   * @param robotInterface Reference to the active MuJoCo simulation interface.
   * @return true if a command was processed, false otherwise.
   */
  bool processCommands(std::string& currentModeName, robot::mujoco_sim_interface::MujocoSimInterface& robotInterface);

  /**
   * @brief Access the captured nominal joint positions.
   */
  const std::vector<scalar_t>& getNominalJointPositions() const { return nominalJointPositions_; }

 private:
  void fsmCommandCallback(const std_msgs::msg::String::ConstSharedPtr& msg);

  rclcpp::Node::SharedPtr nodeHandle_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr fsmCommandSub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fsmStatePub_;

  std::vector<scalar_t> nominalJointPositions_;
  std::mutex commandMutex_;
  std::optional<std::string> pendingCommand_;
};

/**
 * @brief Helper function to construct initial robot::model::RobotState from MPC settings and initial MPC state.
 * @param robotDescription Robot description parsed from URDF.
 * @param modelSettings Model settings containing mpcModelJointNames.
 * @param mpcRobotModel MPC robot model for kinematics and base state extraction.
 * @param initMpcState Initial MPC state vector.
 * @return Initialized RobotState ready for simulation interface.
 */
robot::model::RobotState createInitialSimState(const robot::model::RobotDescription& robotDescription,
                                              const ModelSettings& modelSettings,
                                              const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                              const vector_t& initMpcState);

}  // namespace ocs2::humanoid
