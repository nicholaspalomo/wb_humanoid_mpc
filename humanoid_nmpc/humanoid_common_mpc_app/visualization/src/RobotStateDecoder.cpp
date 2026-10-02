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

#include "humanoid_common_mpc_app/visualization/RobotStateDecoder.h"

#include <cmath>
#include <string>
#include <vector>

#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc_app/visualization/EulerAngles.h"

namespace ocs2::humanoid::visualization {

namespace {

/** A quaternion whose norm is this far from 1 is not a rotation up to rounding. */
constexpr scalar_t kQuaternionNormTolerance = 1e-3;

/** A joint array of the sample: aligned with the joint names, or (`optional`) empty. */
absl::Status checkJointArray(const Eigen::VectorXd& values, size_t jointCount, const char* field, bool optional) {
  const size_t size = static_cast<size_t>(values.size());
  if (size == jointCount || (optional && size == 0)) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat("robot/state: ", field, " has ", size, " values for ", jointCount, " joint names",
                                                 optional ? " (send none, or one per joint)." : "."));
}

/** values[index], or 0 for an array that was not sent. */
scalar_t valueOrZero(const Eigen::VectorXd& values, int index) {
  return values.size() == 0 ? 0.0 : values[index];
}

vector3_t toVector3(const msgs::Vector3& vector) {
  return vector3_t(vector.x, vector.y, vector.z);
}

/** True when joint `index` of `sample` had an action this cycle (see RobotStateDecoder). */
bool hasAction(const msgs::RobotStateSample& sample, int index) {
  return sample.joint_position_targets.size() > 0 && std::isfinite(sample.joint_position_targets[index]);
}

/** The effort applied to joint `index` of `sample`: feed-forward + kp (q_des - q) + kd (qd_des - qd); 0 without action. */
scalar_t appliedEffort(const msgs::RobotStateSample& sample, int index) {
  if (!hasAction(sample, index)) {
    return 0.0;
  }
  return valueOrZero(sample.joint_feed_forward_efforts, index) +
         valueOrZero(sample.joint_kp, index) * (sample.joint_position_targets[index] - sample.joint_positions[index]) +
         valueOrZero(sample.joint_kd, index) * (valueOrZero(sample.joint_velocity_targets, index) - sample.joint_velocities[index]);
}

/** For each of `names`, its index in `sampleIndices`, or -1. */
void lookUpJoints(const absl::flat_hash_map<std::string, int>& sampleIndices,
                  const std::vector<std::string>& names,
                  std::vector<int>* indices) {
  indices->assign(names.size(), -1);
  for (size_t joint = 0; joint < names.size(); ++joint) {
    const absl::flat_hash_map<std::string, int>::const_iterator found = sampleIndices.find(names[joint]);
    if (found != sampleIndices.end()) {
      (*indices)[joint] = found->second;
    }
  }
}

}  // namespace

RobotStateDecoder::RobotStateDecoder(const ModelSettings& modelSettings)
    : fullJointNames_(modelSettings.fullJointNames), mpcJointNames_(modelSettings.mpcModelJointNames) {}

void RobotStateDecoder::updateJointIndices(const std::vector<std::string>& sampleJointNames) {
  if (hasJointIndices_ && sampleJointNames == sampleJointNames_) {
    return;
  }
  absl::flat_hash_map<std::string, int> sampleIndices;
  for (size_t index = 0; index < sampleJointNames.size(); ++index) {
    sampleIndices.emplace(sampleJointNames[index], static_cast<int>(index));
  }
  lookUpJoints(sampleIndices, fullJointNames_, &fullJointToSample_);
  lookUpJoints(sampleIndices, mpcJointNames_, &mpcJointToSample_);
  sampleJointNames_ = sampleJointNames;
  hasJointIndices_ = true;
}

absl::Status RobotStateDecoder::decode(const msgs::RobotStateSample& sample, DecodedRobotState* state) {
  const size_t jointCount = sample.joint_names.size();
  if (absl::Status status = checkJointArray(sample.joint_positions, jointCount, "joint_positions", /*optional=*/false); !status.ok()) {
    return status;
  }
  if (absl::Status status = checkJointArray(sample.joint_velocities, jointCount, "joint_velocities", /*optional=*/false); !status.ok()) {
    return status;
  }
  const std::array<std::pair<const Eigen::VectorXd*, const char*>, 5> actionArrays = {{
      {&sample.joint_position_targets, "joint_position_targets"},
      {&sample.joint_velocity_targets, "joint_velocity_targets"},
      {&sample.joint_kp, "joint_kp"},
      {&sample.joint_kd, "joint_kd"},
      {&sample.joint_feed_forward_efforts, "joint_feed_forward_efforts"},
  }};
  for (const std::pair<const Eigen::VectorXd*, const char*>& actionArray : actionArrays) {
    if (absl::Status status = checkJointArray(*actionArray.first, jointCount, actionArray.second, /*optional=*/true); !status.ok()) {
      return status;
    }
  }
  const msgs::Quaternion& orientation = sample.base_orientation_world;
  quaternion_t quaternion(orientation.w, orientation.x, orientation.y, orientation.z);
  const scalar_t norm = quaternion.norm();
  if (!std::isfinite(norm) || std::abs(norm - 1.0) > kQuaternionNormTolerance) {
    return absl::InvalidArgumentError(
        absl::StrCat("robot/state: base_orientation_world has the norm ", norm, "; it must be a unit quaternion."));
  }
  quaternion.normalize();
  updateJointIndices(sample.joint_names);

  state->time = sample.time;
  state->basePosition = toVector3(sample.base_position_world);
  state->baseOrientation = quaternion;
  state->baseRotation = quaternion.toRotationMatrix();
  state->baseLinearVelocity = state->baseRotation * toVector3(sample.base_linear_velocity_local);
  state->baseAngularVelocity = state->baseRotation * toVector3(sample.base_angular_velocity_local);
  state->baseEulerAnglesZyx = eulerAnglesZyxFromRotation(state->baseRotation);

  const size_t mpcJointCount = mpcJointNames_.size();
  state->generalizedCoordinates.setZero(FLOATING_BASE_DIM + mpcJointCount);
  state->generalizedVelocities.setZero(FLOATING_BASE_DIM + mpcJointCount);
  state->generalizedForces.setZero(FLOATING_BASE_DIM + mpcJointCount);
  state->generalizedCoordinates.head<BASE_TRANSLATION_DIM>() = state->basePosition;
  state->generalizedCoordinates.segment<BASE_ROTATION_DIM>(BASE_TRANSLATION_DIM) = state->baseEulerAnglesZyx;
  state->generalizedVelocities.head<BASE_TRANSLATION_DIM>() = state->baseLinearVelocity;
  state->generalizedVelocities.segment<BASE_ROTATION_DIM>(BASE_TRANSLATION_DIM) =
      getEulerAnglesZyxDerivativesFromGlobalAngularVelocity<scalar_t>(state->baseEulerAnglesZyx, state->baseAngularVelocity);
  for (size_t joint = 0; joint < mpcJointCount; ++joint) {
    const int index = mpcJointToSample_[joint];
    if (index < 0) {
      continue;
    }
    state->generalizedCoordinates[JOINT_COORDINATE_OFFSET + joint] = sample.joint_positions[index];
    state->generalizedVelocities[JOINT_COORDINATE_OFFSET + joint] = sample.joint_velocities[index];
    state->generalizedForces[JOINT_COORDINATE_OFFSET + joint] = appliedEffort(sample, index);
  }

  const size_t fullJointCount = fullJointNames_.size();
  state->jointPositions.setZero(fullJointCount);
  state->jointVelocities.setZero(fullJointCount);
  state->jointEfforts.setZero(fullJointCount);
  state->jointPositionTargets.setZero(fullJointCount);
  state->jointVelocityTargets.setZero(fullJointCount);
  state->jointFeedForwardEfforts.setZero(fullJointCount);
  for (size_t joint = 0; joint < fullJointCount; ++joint) {
    const int index = fullJointToSample_[joint];
    if (index < 0) {
      continue;
    }
    state->jointPositions[joint] = sample.joint_positions[index];
    state->jointVelocities[joint] = sample.joint_velocities[index];
    state->jointEfforts[joint] = appliedEffort(sample, index);
    if (hasAction(sample, index)) {
      state->jointPositionTargets[joint] = sample.joint_position_targets[index];
      state->jointVelocityTargets[joint] = valueOrZero(sample.joint_velocity_targets, index);
      state->jointFeedForwardEfforts[joint] = valueOrZero(sample.joint_feed_forward_efforts, index);
    } else {
      state->jointPositionTargets[joint] = sample.joint_positions[index];
    }
  }

  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    vector6_t& wrench = state->measuredContactWrenches[contact];
    wrench.setZero();
    if (contact < sample.measured_contact_wrenches.size()) {
      wrench.head<3>() = toVector3(sample.measured_contact_wrenches[contact].force);
      wrench.tail<3>() = toVector3(sample.measured_contact_wrenches[contact].torque);
    }
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::visualization
