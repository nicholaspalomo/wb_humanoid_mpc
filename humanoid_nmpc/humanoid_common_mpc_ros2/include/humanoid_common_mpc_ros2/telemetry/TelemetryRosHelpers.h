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

#include <string>

#include <geometry_msgs/msg/accel.hpp>
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <geometry_msgs/msg/wrench.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include <humanoid_common_mpc/common/Types.h>

namespace ocs2::humanoid {

/**
 * @brief Helper functions to convert Eigen geometric quantities into ROS2 messages.
 */

inline geometry_msgs::msg::Point toPointMsg(const vector3_t& pos) {
  geometry_msgs::msg::Point msg;
  msg.x = pos.x();
  msg.y = pos.y();
  msg.z = pos.z();
  return msg;
}

inline geometry_msgs::msg::Vector3 toVector3Msg(const vector3_t& vec) {
  geometry_msgs::msg::Vector3 msg;
  msg.x = vec.x();
  msg.y = vec.y();
  msg.z = vec.z();
  return msg;
}

inline geometry_msgs::msg::Quaternion toQuaternionMsg(const quaternion_t& quat) {
  geometry_msgs::msg::Quaternion msg;
  msg.x = quat.x();
  msg.y = quat.y();
  msg.z = quat.z();
  msg.w = quat.w();
  return msg;
}

inline geometry_msgs::msg::Pose toPoseMsg(const vector3_t& pos, const quaternion_t& quat) {
  geometry_msgs::msg::Pose msg;
  msg.position = toPointMsg(pos);
  msg.orientation = toQuaternionMsg(quat);
  return msg;
}

inline geometry_msgs::msg::Twist toTwistMsg(const vector3_t& linVel, const vector3_t& angVel) {
  geometry_msgs::msg::Twist msg;
  msg.linear = toVector3Msg(linVel);
  msg.angular = toVector3Msg(angVel);
  return msg;
}

inline geometry_msgs::msg::Accel toAccelMsg(const vector3_t& linAccel, const vector3_t& angAccel) {
  geometry_msgs::msg::Accel msg;
  msg.linear = toVector3Msg(linAccel);
  msg.angular = toVector3Msg(angAccel);
  return msg;
}

template <typename Derived>
inline geometry_msgs::msg::Wrench toWrenchMsg(const Eigen::MatrixBase<Derived>& wrenchVec) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 6);
  geometry_msgs::msg::Wrench msg;
  msg.force.x = wrenchVec[WRENCH_FORCE_X_INDEX];
  msg.force.y = wrenchVec[WRENCH_FORCE_Y_INDEX];
  msg.force.z = wrenchVec[WRENCH_FORCE_Z_INDEX];
  msg.torque.x = wrenchVec[WRENCH_TORQUE_X_INDEX];
  msg.torque.y = wrenchVec[WRENCH_TORQUE_Y_INDEX];
  msg.torque.z = wrenchVec[WRENCH_TORQUE_Z_INDEX];
  return msg;
}

inline geometry_msgs::msg::Wrench toForceWrenchMsg(const vector3_t& force, const vector3_t& torque = vector3_t::Zero()) {
  geometry_msgs::msg::Wrench msg;
  msg.force = toVector3Msg(force);
  msg.torque = toVector3Msg(torque);
  return msg;
}

/**
 * @brief Helper functions to construct stamped ROS2 messages.
 */

inline geometry_msgs::msg::PointStamped createPointStamped(const rclcpp::Time& stamp, const std::string& frameId, const vector3_t& pos) {
  geometry_msgs::msg::PointStamped msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frameId;
  msg.point = toPointMsg(pos);
  return msg;
}

inline geometry_msgs::msg::Vector3Stamped createVector3Stamped(const rclcpp::Time& stamp,
                                                               const std::string& frameId,
                                                               const vector3_t& vec) {
  geometry_msgs::msg::Vector3Stamped msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frameId;
  msg.vector = toVector3Msg(vec);
  return msg;
}

inline geometry_msgs::msg::PoseStamped createPoseStamped(const rclcpp::Time& stamp,
                                                         const std::string& frameId,
                                                         const vector3_t& pos,
                                                         const quaternion_t& quat) {
  geometry_msgs::msg::PoseStamped msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frameId;
  msg.pose = toPoseMsg(pos, quat);
  return msg;
}

inline geometry_msgs::msg::TwistStamped createTwistStamped(const rclcpp::Time& stamp,
                                                           const std::string& frameId,
                                                           const vector3_t& linVel,
                                                           const vector3_t& angVel) {
  geometry_msgs::msg::TwistStamped msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frameId;
  msg.twist = toTwistMsg(linVel, angVel);
  return msg;
}

inline geometry_msgs::msg::AccelStamped createAccelStamped(const rclcpp::Time& stamp,
                                                           const std::string& frameId,
                                                           const vector3_t& linAccel,
                                                           const vector3_t& angAccel) {
  geometry_msgs::msg::AccelStamped msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frameId;
  msg.accel = toAccelMsg(linAccel, angAccel);
  return msg;
}

template <typename Derived>
inline geometry_msgs::msg::WrenchStamped createWrenchStamped(const rclcpp::Time& stamp,
                                                             const std::string& frameId,
                                                             const Eigen::MatrixBase<Derived>& wrenchVec) {
  geometry_msgs::msg::WrenchStamped msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frameId;
  msg.wrench = toWrenchMsg(wrenchVec);
  return msg;
}

inline geometry_msgs::msg::WrenchStamped createForceWrenchStamped(const rclcpp::Time& stamp,
                                                                  const std::string& frameId,
                                                                  const vector3_t& force,
                                                                  const vector3_t& torque = vector3_t::Zero()) {
  geometry_msgs::msg::WrenchStamped msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frameId;
  msg.wrench = toForceWrenchMsg(force, torque);
  return msg;
}

}  // namespace ocs2::humanoid
