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

#include "humanoid_common_mpc_app/robot/TelemetrySampler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {

msgs::RobotStateSample TelemetrySampler::prototype(
    const std::vector<std::string>& jointNames) {
  const Eigen::Index numJoints = static_cast<Eigen::Index>(jointNames.size());
  msgs::RobotStateSample sample;
  sample.joint_names = jointNames;
  sample.joint_positions = Eigen::VectorXd::Zero(numJoints);
  sample.joint_velocities = Eigen::VectorXd::Zero(numJoints);
  sample.joint_measured_efforts = Eigen::VectorXd::Zero(numJoints);
  sample.joint_position_targets = Eigen::VectorXd::Zero(numJoints);
  sample.joint_velocity_targets = Eigen::VectorXd::Zero(numJoints);
  sample.joint_kp = Eigen::VectorXd::Zero(numJoints);
  sample.joint_kd = Eigen::VectorXd::Zero(numJoints);
  sample.joint_feed_forward_efforts = Eigen::VectorXd::Zero(numJoints);
  sample.contact_flags.assign(kNumContacts, false);
  sample.meas_wrench_contacts_ewrt_world.assign(kNumContacts, msgs::Wrench());
  return sample;
}

TelemetrySampler::TelemetrySampler(const Config& config)
    : decimation_(std::max<size_t>(1, config.decimation)),
      numJoints_(config.jointNames.size()),
      queue_(std::max<size_t>(1, config.capacity),
             prototype(config.jointNames)) {}

bool TelemetrySampler::sample(
    const robot::model::RobotState& robotState,
    const robot::model::RobotJointAction& jointAction,
    absl::string_view controlMode,
    const contact_flag_t& measuredContactFlags,
    const std::array<vector3_t, kNumContacts>& measForceFeetEwrtWorld) {
  if (++cycle_ < decimation_) {
    return false;
  }
  cycle_ = 0;
  samplesTaken_.fetch_add(1, std::memory_order_relaxed);
  queue_.tryPushInPlace([&](msgs::RobotStateSample& slot) {
    slot.time = robotState.getTime();
    // Every control mode name fits std::string's small-string buffer, so this
    // copies without allocating.
    slot.control_mode.assign(controlMode.data(), controlMode.size());
    const vector3_t posBaseRtWorldEwrtWorld =
        robotState.getPosBaseRtWorldEwrtWorld();
    slot.pos_base_rt_world_ewrt_world = {.x = posBaseRtWorldEwrtWorld.x(),
                                         .y = posBaseRtWorldEwrtWorld.y(),
                                         .z = posBaseRtWorldEwrtWorld.z()};
    const quaternion_t quatBaseRtWorld = robotState.getQuatBaseRtWorld();
    slot.quat_base_rt_world = {.w = quatBaseRtWorld.w(),
                               .x = quatBaseRtWorld.x(),
                               .y = quatBaseRtWorld.y(),
                               .z = quatBaseRtWorld.z()};
    const vector3_t linvelBaseRtWorldEwrtBase =
        robotState.getLinvelBaseRtWorldEwrtBase();
    slot.linvel_base_rt_world_ewrt_base = {.x = linvelBaseRtWorldEwrtBase.x(),
                                           .y = linvelBaseRtWorldEwrtBase.y(),
                                           .z = linvelBaseRtWorldEwrtBase.z()};
    const vector3_t angvelBaseRtWorldEwrtBase =
        robotState.getAngvelBaseRtWorldEwrtBase();
    slot.angvel_base_rt_world_ewrt_base = {.x = angvelBaseRtWorldEwrtBase.x(),
                                           .y = angvelBaseRtWorldEwrtBase.y(),
                                           .z = angvelBaseRtWorldEwrtBase.z()};
    for (size_t joint = 0; joint < numJoints_; ++joint) {
      const Eigen::Index row = static_cast<Eigen::Index>(joint);
      const std::optional<robot::model::JointAction>& action =
          jointAction[joint];  // joint < numJoints_, the description's
      slot.joint_positions[row] = robotState.getJointPosition(joint);
      slot.joint_velocities[row] = robotState.getJointVelocity(joint);
      // The RobotState carries no measured effort yet (nothing fills
      // JointState::measuredEffort).
      slot.joint_measured_efforts[row] = 0.0;
      // robot_state_sample.proto: a joint without an action has a NaN position
      // target, which RobotStateDecoder reads as "no action" (its applied
      // effort is then zero, its target drawn at the measured position). Its
      // gains, velocity target and feedforward read as zero.
      slot.joint_position_targets[row] =
          action.has_value() ? action->q_des
                             : std::numeric_limits<double>::quiet_NaN();
      slot.joint_velocity_targets[row] =
          action.has_value() ? action->qd_des : 0.0;
      slot.joint_kp[row] = action.has_value() ? action->kp : 0.0;
      slot.joint_kd[row] = action.has_value() ? action->kd : 0.0;
      slot.joint_feed_forward_efforts[row] =
          action.has_value() ? action->feed_forward_effort : 0.0;
    }
    for (size_t contact = 0; contact < kNumContacts; ++contact) {
      slot.contact_flags[contact] = measuredContactFlags[contact];
      msgs::Wrench& measWrenchContactEwrtWorld =
          slot.meas_wrench_contacts_ewrt_world[contact];
      // NOLINTBEGIN(frame-naming): the components of a Wrench message, whose frames its field names
      measWrenchContactEwrtWorld.force = {
          .x = measForceFeetEwrtWorld[contact].x(),
          .y = measForceFeetEwrtWorld[contact].y(),
          .z = measForceFeetEwrtWorld[contact].z()};
      measWrenchContactEwrtWorld.torque = msgs::Vector3();
      // NOLINTEND(frame-naming)
    }
  });
  return true;
}

size_t TelemetrySampler::drain(
    absl::FunctionRef<void(const msgs::RobotStateSample& sample)> consumer) {
  size_t drained = 0;
  while (queue_.tryPopInPlace(
      [&](const msgs::RobotStateSample& sample) { consumer(sample); })) {
    ++drained;
  }
  return drained;
}

}  // namespace ocs2::humanoid
