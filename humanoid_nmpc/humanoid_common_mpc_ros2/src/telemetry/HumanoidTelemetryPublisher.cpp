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

#include "humanoid_common_mpc_ros2/telemetry/HumanoidTelemetryPublisher.h"
#include "humanoid_common_mpc_ros2/telemetry/TelemetryRosHelpers.h"

#include <algorithm>
#include <cmath>

#include <humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h>
#include <ocs2_core/misc/LinearInterpolation.h>

namespace ocs2::humanoid {

HumanoidTelemetryPublisher::HumanoidTelemetryPublisher(rclcpp::Node::SharedPtr nodeHandle,
                                                       const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                       const ::robot::model::RobotDescription& robotDescription)
    : nodeHandle_(std::move(nodeHandle)), mpcRobotModelPtr_(&mpcRobotModel), robotDescriptionPtr_(&robotDescription) {
  auto qos = rclcpp::QoS(10);
  qos.best_effort();

  // Full joint names in model order
  const auto& jointNames = mpcRobotModelPtr_->modelSettings.fullJointNames;
  fullJointNames_.assign(jointNames.begin(), jointNames.end());

  // Publishers
  jointStatePub_ = nodeHandle_->create_publisher<sensor_msgs::msg::JointState>("joint_states", qos);
  mpcJointTargetPub_ = nodeHandle_->create_publisher<sensor_msgs::msg::JointState>("mpc/joint_targets", qos);

  robotBasePosePub_ = nodeHandle_->create_publisher<geometry_msgs::msg::PoseStamped>("robot/base_pose", qos);
  robotBaseEulerPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::Vector3Stamped>("robot/base_euler", qos);
  robotBaseTwistPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::TwistStamped>("robot/base_twist", qos);

  mpcTargetBasePosePub_ = nodeHandle_->create_publisher<geometry_msgs::msg::PoseStamped>("mpc/target_base_pose", qos);
  mpcTargetBaseEulerPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::Vector3Stamped>("mpc/target_base_euler", qos);
  mpcTargetBaseTwistPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::TwistStamped>("mpc/target_base_twist", qos);

  mpcContactWrenchLeftPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::WrenchStamped>("mpc/contact_wrench/left", qos);
  mpcContactWrenchRightPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::WrenchStamped>("mpc/contact_wrench/right", qos);

  simContactWrenchLeftPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::WrenchStamped>("sensors/contact_wrench/left", qos);
  simContactWrenchRightPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::WrenchStamped>("sensors/contact_wrench/right", qos);

  mpcObservationPub_ = nodeHandle_->create_publisher<ocs2_ros2_msgs::msg::MpcObservation>("mpc/observation", qos);
}

vector3_t HumanoidTelemetryPublisher::quaternionToEulerZYX(const quaternion_t& q) {
  scalar_t w = q.w();
  scalar_t x = q.x();
  scalar_t y = q.y();
  scalar_t z = q.z();

  scalar_t roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
  scalar_t pitch = std::asin(std::clamp(2.0 * (w * y - z * x), -1.0, 1.0));
  scalar_t yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));

  return vector3_t(roll, pitch, yaw);
}

void HumanoidTelemetryPublisher::publish(const ::robot::model::RobotState& robotState,
                                         const ::robot::model::RobotJointAction& robotJointAction,
                                         const SystemObservation& mpcObservation,
                                         const vector_t& mpcPolicyInput,
                                         const CommandData& mpcCommand,
                                         const vector3_t& leftMeasuredForce,
                                         const vector3_t& rightMeasuredForce) {
  const auto now = nodeHandle_->now();

  // 1. Current Robot Joint States (/joint_states)
  sensor_msgs::msg::JointState jointStateMsg;
  jointStateMsg.header.stamp = now;
  jointStateMsg.name = fullJointNames_;
  jointStateMsg.position.resize(fullJointNames_.size());
  jointStateMsg.velocity.resize(fullJointNames_.size());
  jointStateMsg.effort.resize(fullJointNames_.size());

  for (size_t i = 0; i < fullJointNames_.size(); ++i) {
    jointStateMsg.position[i] = robotState.getJointPosition(i);
    jointStateMsg.velocity[i] = robotState.getJointVelocity(i);
    const auto& actionOpt = robotJointAction.at(i);
    if (actionOpt.has_value()) {
      const auto& action = actionOpt.value();
      jointStateMsg.effort[i] = action.feed_forward_effort + action.kp * (action.q_des - jointStateMsg.position[i]) +
                                action.kd * (action.qd_des - jointStateMsg.velocity[i]);
    } else {
      jointStateMsg.effort[i] = 0.0;
    }
  }
  jointStatePub_->publish(jointStateMsg);

  // 2. MPC Target Joint States (/mpc/joint_targets)
  sensor_msgs::msg::JointState targetJointMsg;
  targetJointMsg.header.stamp = now;
  targetJointMsg.name = fullJointNames_;
  targetJointMsg.position.resize(fullJointNames_.size());
  targetJointMsg.velocity.resize(fullJointNames_.size());
  targetJointMsg.effort.resize(fullJointNames_.size());

  for (size_t i = 0; i < fullJointNames_.size(); ++i) {
    const auto& actionOpt = robotJointAction.at(i);
    if (actionOpt.has_value()) {
      targetJointMsg.position[i] = actionOpt.value().q_des;
      targetJointMsg.velocity[i] = actionOpt.value().qd_des;
      targetJointMsg.effort[i] = actionOpt.value().feed_forward_effort;
    } else {
      targetJointMsg.position[i] = robotState.getJointPosition(i);
      targetJointMsg.velocity[i] = 0.0;
      targetJointMsg.effort[i] = 0.0;
    }
  }
  mpcJointTargetPub_->publish(targetJointMsg);

  // 3. Robot Base Pose, Euler, and Twist
  const vector3_t rootPos = robotState.getRootPositionInWorldFrame();
  const quaternion_t rootQuat = robotState.getRootRotationLocalToWorldFrame();
  const vector3_t rootEuler = quaternionToEulerZYX(rootQuat);
  const vector3_t rootLinVel = rootQuat * robotState.getRootLinearVelocityInLocalFrame();
  const vector3_t rootAngVel = rootQuat * robotState.getRootAngularVelocityInLocalFrame();

  robotBasePosePub_->publish(createPoseStamped(now, "world", rootPos, rootQuat));
  robotBaseEulerPub_->publish(createVector3Stamped(now, "world", rootEuler));
  robotBaseTwistPub_->publish(createTwistStamped(now, "world", rootLinVel, rootAngVel));

  // 4. MPC Target Base Pose, Euler, and Twist
  vector3_t targetPos = rootPos;
  quaternion_t targetQuat = rootQuat;
  vector3_t targetEuler = rootEuler;
  vector3_t targetLinVel = vector3_t::Zero();
  vector3_t targetAngVel = vector3_t::Zero();

  if (!mpcCommand.mpcTargetTrajectories_.timeTrajectory.empty()) {
    scalar_t time = mpcObservation.time;
    vector_t targetState = LinearInterpolation::interpolate(time, mpcCommand.mpcTargetTrajectories_.timeTrajectory,
                                                            mpcCommand.mpcTargetTrajectories_.stateTrajectory);
    targetPos = mpcRobotModelPtr_->getBasePosition(targetState);
    const vector3_t targetEulerZyx = mpcRobotModelPtr_->getBaseOrientationEulerZYX(targetState);
    targetEuler = vector3_t(targetEulerZyx.z(), targetEulerZyx.y(), targetEulerZyx.x());
    targetQuat = getQuaternionFromEulerAnglesZyx(targetEulerZyx);

    if (!mpcCommand.mpcTargetTrajectories_.inputTrajectory.empty()) {
      vector_t targetInput = LinearInterpolation::interpolate(time, mpcCommand.mpcTargetTrajectories_.timeTrajectory,
                                                              mpcCommand.mpcTargetTrajectories_.inputTrajectory);
      if (targetInput.size() >= static_cast<int>(N_CONTACTS * CONTACT_WRENCH_DIM)) {
        targetLinVel = mpcRobotModelPtr_->getBaseComLinearVelocity(targetState);
      }
    }
  }

  mpcTargetBasePosePub_->publish(createPoseStamped(now, "world", targetPos, targetQuat));
  mpcTargetBaseEulerPub_->publish(createVector3Stamped(now, "world", targetEuler));
  mpcTargetBaseTwistPub_->publish(createTwistStamped(now, "world", targetLinVel, targetAngVel));

  // 5. Contact Wrenches: MPC Requested vs MuJoCo Measured
  if (mpcPolicyInput.size() >= static_cast<int>(N_CONTACTS * CONTACT_WRENCH_DIM)) {
    mpcContactWrenchLeftPub_->publish(
        createWrenchStamped(now, "world", mpcPolicyInput.segment<CONTACT_WRENCH_DIM>(CONTACT_LEFT_INDEX * CONTACT_WRENCH_DIM)));
    mpcContactWrenchRightPub_->publish(
        createWrenchStamped(now, "world", mpcPolicyInput.segment<CONTACT_WRENCH_DIM>(CONTACT_RIGHT_INDEX * CONTACT_WRENCH_DIM)));
  }

  simContactWrenchLeftPub_->publish(createForceWrenchStamped(now, "world", leftMeasuredForce));
  simContactWrenchRightPub_->publish(createForceWrenchStamped(now, "world", rightMeasuredForce));

  // 6. Current State-Input Tracked by MPC (/mpc/observation)
  ocs2_ros2_msgs::msg::MpcObservation obsMsg;
  obsMsg.time = mpcObservation.time;
  obsMsg.mode = mpcObservation.mode;
  if (mpcObservation.state.size() > 0) {
    obsMsg.state.value.assign(mpcObservation.state.data(), mpcObservation.state.data() + mpcObservation.state.size());
  }
  if (mpcPolicyInput.size() > 0) {
    obsMsg.input.value.assign(mpcPolicyInput.data(), mpcPolicyInput.data() + mpcPolicyInput.size());
  }
  mpcObservationPub_->publish(obsMsg);
}

}  // namespace ocs2::humanoid
