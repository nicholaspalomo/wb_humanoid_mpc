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

#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <ocs2_ros2_msgs/msg/mpc_observation.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <humanoid_common_mpc/common/ModelSettings.h>
#include <humanoid_common_mpc/common/MpcRobotModelBase.h>
#include <ocs2_mpc/CommandData.h>
#include <ocs2_mpc/SystemObservation.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

namespace ocs2::humanoid {

/**
 * @brief Publishes comprehensive robot and MPC telemetry for visualization in PlotJuggler.
 *
 * Broadcasts:
 * - Current joint signals (positions, velocities, applied torques): /joint_states
 * - MPC target joint signals (positions, velocities, feedforward torques): /mpc/joint_targets
 * - Actual base pose and twist: /robot/base_pose, /robot/base_euler, /robot/base_twist
 * - MPC target base pose and twist: /mpc/target_base_pose, /mpc/target_base_euler, /mpc/target_base_twist
 * - MPC requested contact wrenches: /mpc/contact_wrench/left, /mpc/contact_wrench/right
 * - Measured contact forces from MuJoCo: /sensors/contact_wrench/left, /sensors/contact_wrench/right
 * - Current state-input tracked by MPC: /mpc/observation
 */
class HumanoidTelemetryPublisher {
 public:
  HumanoidTelemetryPublisher(rclcpp::Node::SharedPtr nodeHandle,
                             const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                             const ::robot::model::RobotDescription& robotDescription);

  void publish(const ::robot::model::RobotState& robotState,
               const ::robot::model::RobotJointAction& robotJointAction,
               const SystemObservation& mpcObservation,
               const vector_t& mpcPolicyInput,
               const CommandData& mpcCommand,
               const vector3_t& leftMeasuredForce,
               const vector3_t& rightMeasuredForce);

 private:
  static vector3_t quaternionToEulerZYX(const quaternion_t& q);

  rclcpp::Node::SharedPtr nodeHandle_;
  const MpcRobotModelBase<scalar_t>* mpcRobotModelPtr_;
  const ::robot::model::RobotDescription* robotDescriptionPtr_;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr jointStatePub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr mpcJointTargetPub_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr robotBasePosePub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr robotBaseEulerPub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr robotBaseTwistPub_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mpcTargetBasePosePub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr mpcTargetBaseEulerPub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr mpcTargetBaseTwistPub_;

  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr mpcContactWrenchLeftPub_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr mpcContactWrenchRightPub_;

  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr simContactWrenchLeftPub_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr simContactWrenchRightPub_;

  rclcpp::Publisher<ocs2_ros2_msgs::msg::MpcObservation>::SharedPtr mpcObservationPub_;

  std::vector<std::string> fullJointNames_;
};

}  // namespace ocs2::humanoid
