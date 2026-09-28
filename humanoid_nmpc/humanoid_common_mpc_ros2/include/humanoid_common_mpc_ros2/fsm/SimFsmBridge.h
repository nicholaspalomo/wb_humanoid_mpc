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

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <humanoid_common_mpc/common/ModelSettings.h>
#include <humanoid_common_mpc/common/MpcRobotModelBase.h>
#include <mujoco_sim_interface/MujocoSimInterface.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>
#include <humanoid_mpc_msgs/msg/walking_velocity_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <string_view>

#include "absl/strings/string_view.h"
#include "humanoid_common_mpc_ros2/ros_comm/JointTargetSubscriber.h"

namespace ocs2::humanoid {

/**
 * The text of one `/humanoid/fsm_state` message: `<mode>,GANTRY_LOCKED|GANTRY_UNLOCKED,<controller resets>`, the last
 * field the decimal count of controller resets since the bridge started (SimFsmBridge::publishControllerReset()). The
 * remote control parses it in remote_control/fsm_state.py and re-centers its joysticks on every transition into a
 * passive mode, on every new gantry lock and on every change of the reset count.
 */
std::string formatFsmState(absl::string_view modeName, bool gantryLocked, uint64_t controllerResets);

/**
 * @brief ROS 2-native bridge between supervisory FSM commands/state and the simulation loop.
 *
 * Encapsulates:
 * 1. Tracking and storing nominal stance positions (the JOINT_PD posture, which the controllers compute the action of).
 * 2. Subscribing to ROS 2 topic `/humanoid/fsm_command` (std_msgs/msg/String) and processing mode transitions.
 * 3. Publishing ROS 2 topic `/humanoid/fsm_state` (std_msgs/msg/String) with transient-local QoS: the mode, the gantry
 *    and the number of controller resets so far (formatFsmState()).
 * 4. Virtual gantry lock and unlock commands, and the gantry height slider.
 * The fall recovery - catching a fallen robot, noticing the simulator's own resets, settling the caught robot - is
 * SimFallRecovery, which is free of ROS.
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
   * @brief Publishes the current FSM mode, the gantry state and the controller resets so far to `/humanoid/fsm_state`.
   * @param modeName Current active mode name (e.g., "ZERO_TORQUE", "JOINT_PD", "WB_MPC").
   * @param gantryLocked Whether the gantry is currently locked.
   */
  void publishFsmState(std::string_view modeName, bool gantryLocked) const;

  /**
   * @brief Counts one controller reset and publishes the state with the new count.
   *
   * A controller reset is a discontinuity of the plant after which the controller starts again from where the robot is
   * (SimFallRecovery::Cycle::discontinuity): a catch, a LOCK_GANTRY, or a reset the simulator made on its own thread.
   * The remote control re-centers its joysticks on every change of the count, which is what releases them after a reset
   * that changes neither the mode nor the gantry - the simulator putting the robot back while the gantry was already
   * locked in JOINT_PD. Call it from the control loop, the thread that publishes the state.
   */
  void publishControllerReset(std::string_view modeName, bool gantryLocked);

  /** @brief The controller resets counted so far, the last field of every state published. */
  uint64_t controllerResets() const { return controllerResets_; }

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

  /**
   * @brief Subscribe to the /joint_pd_target_positions ROS 2 topic for real-time
   *        joint target updates from the GUI slider tab.
   */
  void subscribeJointTargets(rclcpp::Node::SharedPtr node);

  /**
   * @brief Apply any pending joint target updates received via ROS topic.
   *        Call this from the control loop, ideally before reading nominalJointPositions_.
   */
  void applyJointTargetUpdates();

 private:
  void fsmCommandCallback(const std_msgs::msg::String::ConstSharedPtr& msg);
  /// Parses one YAML throw from the GUI and stages it. A malformed payload is reported and dropped: it is an
  /// operator action, not a control input, so refusing it is better than guessing at it.
  void dodgeballCallback(const std_msgs::msg::String::ConstSharedPtr& msg);
  void walkingVelocityCallback(const humanoid_mpc_msgs::msg::WalkingVelocityCommand::ConstSharedPtr& msg);

  rclcpp::Node::SharedPtr nodeHandle_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr fsmCommandSub_;
  /// Dodgeball throws from the GUI's Dodgeball tab. A YAML payload, staged here and drained by processCommands()
  /// onto the simulation interface - see MujocoSimInterface::throwDodgeball.
  /// Its topic is set, and tied to the GUI's, where the subscription is created in SimFsmBridge.cpp.
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr dodgeballSub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fsmStatePub_;
  rclcpp::Subscription<humanoid_mpc_msgs::msg::WalkingVelocityCommand>::SharedPtr walkingVelSub_;

  std::vector<scalar_t> nominalJointPositions_;
  std::vector<std::string> allJointNames_;  ///< All robot joint names (for joint target subscriber lookup)
  std::vector<size_t> allJointIndices_;     ///< All robot joint indices (parallel to allJointNames_)
  JointTargetSubscriber jointTargetSubscriber_;
  std::mutex commandMutex_;
  std::optional<std::string> pendingCommand_;
  /// Staged separately from pendingCommand_ so that throwing a ball cannot swallow a pending mode change, or the
  /// other way round: both arrive on the operator's thread and only one of each is kept.
  std::mutex dodgeballMutex_;
  std::optional<robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow> pendingDodgeball_;
  std::atomic<double> desiredGantryHeight_{0.0};      ///< Desired gantry height from walking velocity command slider.
  std::atomic<bool> hasReceivedGantryHeight_{false};  ///< True once a walking velocity message has set the height.
  uint64_t controllerResets_ = 0;                     ///< Controller resets so far (publishControllerReset()).
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
