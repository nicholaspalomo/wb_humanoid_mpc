/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include "robot_model/RobotState.h"

#include <algorithm>
#include <optional>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"

namespace robot::model {

RobotState::RobotState(const RobotDescription& robotDescription, size_t contactSize)
    : jointStateMap_(robotDescription),
      rootPosition_(vector3_t::Zero()),
      rootLinearVelocity_(vector3_t::Zero()),
      rootOrientation_(quaternion_t::Identity()),
      rootAngularVelocity_(vector3_t::Zero()),
      contactFlags_(contactSize) {
  for (joint_index_t idx : robotDescription.getJointIndices()) {
    jointStateMap_[idx].emplace();
  }

  setConfigurationToZero();

  std::fill(contactFlags_.begin(), contactFlags_.end(), true);  // Assume robot is in contact
}

const JointState* absl_nullable RobotState::findJoint(size_t jointId) const {
  if (!jointStateMap_.inRange(jointId)) return nullptr;
  const std::optional<JointState>& joint = jointStateMap_[jointId];
  return joint.has_value() ? &*joint : nullptr;
}

JointState* absl_nullable RobotState::findJoint(size_t jointId) {
  if (!jointStateMap_.inRange(jointId)) return nullptr;
  std::optional<JointState>& joint = jointStateMap_[jointId];
  return joint.has_value() ? &*joint : nullptr;
}

void RobotState::setJointPosition(size_t jointId, scalar_t jointPosition) {
  JointState* absl_nullable const joint = findJoint(jointId);
  if (joint != nullptr) {
    joint->position = jointPosition;
  }
}

scalar_t RobotState::getCheckedJointPosition(size_t jointId) const {
  const JointState* absl_nullable const joint = findJoint(jointId);
  ABSL_CHECK(joint != nullptr) << "RobotState: no joint " << jointId;
  return joint->position;
}

void RobotState::setJointVelocity(size_t jointId, scalar_t jointVelocity) {
  JointState* absl_nullable const joint = findJoint(jointId);
  if (joint != nullptr) {
    joint->velocity = jointVelocity;
  }
}

scalar_t RobotState::getCheckedJointVelocity(size_t jointId) const {
  const JointState* absl_nullable const joint = findJoint(jointId);
  ABSL_CHECK(joint != nullptr) << "RobotState: no joint " << jointId;
  return joint->velocity;
}

void RobotState::setConfigurationToZero() {
  rootPosition_.setZero();
  rootOrientation_.setIdentity();
  rootLinearVelocity_.setZero();
  rootAngularVelocity_.setZero();

  for (JointState& joint : jointStateMap_) {
    joint.position = 0.0;
    joint.velocity = 0.0;
    joint.measuredEffort = 0.0;
  }

  std::fill(contactFlags_.begin(), contactFlags_.end(), true);  // Assume robot is in contact
}

}  // namespace robot::model
