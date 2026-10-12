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
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/orientation/EulerBoundary.h"

namespace ocs2::humanoid::visualization {

namespace {

/** A quaternion whose norm is this far from 1 is not a rotation up to rounding. */
constexpr scalar_t kQuaternionNormTolerance = 1.0e-3;

/** A joint array of the sample: aligned with the joint names, or (`optional`) empty. */
absl::Status checkJointArray(const Eigen::VectorXd& values, size_t jointCount, const char* absl_nonnull field, bool optional) {
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
                  std::vector<int>* absl_nonnull indices) {
  indices->assign(names.size(), -1);
  for (size_t joint = 0; joint < names.size(); ++joint) {
    const absl::flat_hash_map<std::string, int>::const_iterator found = sampleIndices.find(names[joint]);
    if (found != sampleIndices.end()) {
      (*indices)[joint] = found->second;
    }
  }
}

}  // namespace

vector3_t eulerZyxRatesForDisplay(const vector3_t& eulerZyx, const vector3_t& angularVelocityLocal) {
  // The not-form is also true for a NaN pitch.
  if (!(std::abs(std::cos(eulerZyx(kBaseRotPitchIndex))) >= kDisplayEulerRateMinCosPitch)) {
    return vector3_t::Constant(std::numeric_limits<scalar_t>::quiet_NaN());
  }
  return eulerZyxRatesFromLocalAngularVelocity(eulerZyx, angularVelocityLocal);
}

RobotStateDecoder::RobotStateDecoder(const MpcRobotModelBase<scalar_t>& mpcRobotModel)
    : robotModel_(mpcRobotModel.clone()),
      fullJointNames_(mpcRobotModel.modelSettings.fullJointNames),
      mpcJointNames_(mpcRobotModel.modelSettings.mpcModelJointNames),
      mpcJointPositions_(vector_t::Zero(static_cast<Eigen::Index>(mpcJointNames_.size()))),
      mpcJointVelocities_(vector_t::Zero(static_cast<Eigen::Index>(mpcJointNames_.size()))) {}

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

absl::Status RobotStateDecoder::decode(const msgs::RobotStateSample& sample, DecodedRobotState* absl_nonnull state) {
  const size_t jointCount = sample.joint_names.size();
  if (absl::Status status = checkJointArray(sample.joint_positions, jointCount, "joint_positions", /*optional=*/false); !status.ok()) {
    return status;
  }
  if (absl::Status status = checkJointArray(sample.joint_velocities, jointCount, "joint_velocities", /*optional=*/false); !status.ok()) {
    return status;
  }
  const std::array<std::pair<const Eigen::VectorXd* absl_nonnull, const char* absl_nonnull>, 5> actionArrays = {{
      {&sample.joint_position_targets, "joint_position_targets"},
      {&sample.joint_velocity_targets, "joint_velocity_targets"},
      {&sample.joint_kp, "joint_kp"},
      {&sample.joint_kd, "joint_kd"},
      {&sample.joint_feed_forward_efforts, "joint_feed_forward_efforts"},
  }};
  for (const std::pair<const Eigen::VectorXd* absl_nonnull, const char* absl_nonnull>& actionArray : actionArrays) {
    if (absl::Status status = checkJointArray(*actionArray.first, jointCount, actionArray.second, /*optional=*/true); !status.ok()) {
      return status;
    }
  }
  const msgs::Quaternion& orientation = sample.base_orientation_world;
  // From the coefficients in Eigen's storage order (x, y, z, w), not its scalar-first constructor.
  quaternion_t quaternion(vector4_t(orientation.x, orientation.y, orientation.z, orientation.w));
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
  state->baseAngularVelocityLocal = toVector3(sample.base_angular_velocity_local);
  state->baseAngularVelocity = state->baseRotation * state->baseAngularVelocityLocal;
  // The display values: the MPC's q below keeps the quaternion.
  state->baseEulerAnglesZyx = eulerZyxFromQuaternion(quaternion.coeffs());
  state->baseEulerZyxRates = eulerZyxRatesForDisplay(state->baseEulerAnglesZyx, state->baseAngularVelocityLocal);

  const size_t mpcJointCount = mpcJointNames_.size();
  state->generalizedForces.setZero(robotModel_->getVelocityDim());
  for (size_t joint = 0; joint < mpcJointCount; ++joint) {
    const int index = mpcJointToSample_[joint];
    if (index < 0) {
      mpcJointPositions_[joint] = 0.0;
      mpcJointVelocities_[joint] = 0.0;
      continue;
    }
    mpcJointPositions_[joint] = sample.joint_positions[index];
    mpcJointVelocities_[joint] = sample.joint_velocities[index];
    state->generalizedForces[kJointVelocityOffset + joint] = appliedEffort(sample, index);
  }
  // The measured base and joints in the MPC's generalized coordinates, as the MRT controllers convert them.
  const vector4_t orientationCoefficients = quaternion.coeffs();
  state->generalizedCoordinates.resize(robotModel_->getConfigurationDim());
  state->generalizedVelocities.resize(robotModel_->getVelocityDim());
  robotModel_->fillGeneralizedCoordinatesFromMeasurement(state->basePosition, orientationCoefficients, mpcJointPositions_,
                                                         state->generalizedCoordinates);
  robotModel_->fillGeneralizedVelocitiesFromMeasurement(orientationCoefficients, toVector3(sample.base_linear_velocity_local),
                                                        state->baseAngularVelocityLocal, mpcJointVelocities_, state->generalizedVelocities);

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

  for (size_t contact = 0; contact < kNumContacts; ++contact) {
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
