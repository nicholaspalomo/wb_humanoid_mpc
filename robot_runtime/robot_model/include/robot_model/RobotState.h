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

#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"

#include "robot_core/Types.h"
#include "robot_model/JointIDMap.h"
#include "robot_model/RobotDescription.h"

namespace robot::model {

/** The measured state of one joint. */
struct JointState {
  scalar_t position = 0.0;
  scalar_t velocity = 0.0;
  scalar_t measuredEffort = 0.0;

  JointState() = default;
};

/**
 * The state of a floating-base robot: the root's pose and twist, the state of every joint of a robot description by
 * joint index, and a contact flag per contact point. Copying one into another of the same robot allocates nothing (the
 * realtime loop does it every cycle). Not thread-safe.
 */
class RobotState {
 public:
  explicit RobotState(const RobotDescription& robotDescription, size_t contactSize = 2);

  // orientation of the root joint wrt world frame, corresponds to the passive
  // rotation from local to world R_l_to_w
  quaternion_t getRootRotationLocalToWorldFrame() const { return rootOrientation_; }
  vector3_t getRootPositionInWorldFrame() const { return rootPosition_; }

  vector3_t getRootLinearVelocityInLocalFrame() const { return rootLinearVelocity_; }
  vector3_t getRootAngularVelocityInLocalFrame() const { return rootAngularVelocity_; }

  // orientation of the root joint wrt world frame, corresponds to the passive
  // rotation from local to world R_l_to_w
  void setRootRotationLocalToWorldFrame(const quaternion_t& orientation) { rootOrientation_ = orientation; }
  void setRootPositionInWorldFrame(const vector3_t& position) { rootPosition_ = position; }

  void setRootLinearVelocityInLocalFrame(const vector3_t& linearVelocity) { rootLinearVelocity_ = linearVelocity; }
  void setRootAngularVelocityInLocalFrame(const vector3_t& angularVelocity) { rootAngularVelocity_ = angularVelocity; }

  /** True when `jointId` is a joint of the robot description. */
  bool hasJoint(size_t jointId) const { return findJoint(jointId) != nullptr; }

  /** Sets the position of joint `jointId`; an id that is not a joint (hasJoint()) is ignored. */
  void setJointPosition(size_t jointId, scalar_t jointPosition);
  /**
   * The position of joint `jointId`, without a check, as std::vector's operator[]: `jointId` must be a joint of the
   * robot description (hasJoint(), every index below its getNumJoints()). The control thread's form: code on the
   * realtime path validates its ids once, when it is built, and nothing there may end the process. Setup code and tests
   * use getCheckedJointPosition().
   */
  scalar_t getJointPosition(size_t jointId) const {
    return jointStateMap_[jointId]->position;  // NOLINT(bugprone-unchecked-optional-access): the precondition above.
  }
  /** getJointPosition(), checked: an id that is not a joint ends the process. Not for the realtime path. */
  scalar_t getCheckedJointPosition(size_t jointId) const;
  /** As setJointPosition(), for the velocity. */
  void setJointVelocity(size_t jointId, scalar_t jointVelocity);
  /** As getJointPosition(), for the velocity: unchecked, `jointId` must be a joint. */
  scalar_t getJointVelocity(size_t jointId) const {
    return jointStateMap_[jointId]->velocity;  // NOLINT(bugprone-unchecked-optional-access): the precondition above.
  }
  /** As getCheckedJointPosition(), for the velocity. */
  scalar_t getCheckedJointVelocity(size_t jointId) const;

  //  Get a vector_t of joint positions given a vector of joint IDs
  template <typename E>
  vector_t getJointPositions(const std::vector<E>& jointIds, scalar_t defaultValue = std::numeric_limits<scalar_t>::quiet_NaN()) const {
    return jointStateMap_.toVector(jointIds, &RobotState::positionOf, defaultValue);
  }

  //  Get a vector_t of joint velocities given a vector of joint IDs
  template <typename E>
  vector_t getJointVelocities(const std::vector<E>& jointIds, scalar_t defaultValue = std::numeric_limits<scalar_t>::quiet_NaN()) const {
    return jointStateMap_.toVector(jointIds, &RobotState::velocityOf, defaultValue);
  }

  //  The joint positions into `positions`, resized to jointIds.size(): no allocation once it has that size (the
  //  control thread's form).
  template <typename E>
  void getJointPositions(const std::vector<E>& jointIds,
                         vector_t& positions,
                         scalar_t defaultValue = std::numeric_limits<scalar_t>::quiet_NaN()) const {
    jointStateMap_.writeVector(jointIds, &RobotState::positionOf, positions, defaultValue);
  }

  //  The joint velocities into `velocities`, as getJointPositions() above.
  template <typename E>
  void getJointVelocities(const std::vector<E>& jointIds,
                          vector_t& velocities,
                          scalar_t defaultValue = std::numeric_limits<scalar_t>::quiet_NaN()) const {
    jointStateMap_.writeVector(jointIds, &RobotState::velocityOf, velocities, defaultValue);
  }

  /** The flag of contact point `index`, which must be below getContactFlags().size(); another index ends the process. */
  bool getContactFlag(size_t index) const {
    ABSL_CHECK_LT(index, contactFlags_.size()) << "RobotState: no contact point " << index;
    return contactFlags_[index];
  }

  /** Sets the flag of contact point `index`, as getContactFlag(). */
  void setContactFlag(size_t index, bool contactFlag) {
    ABSL_CHECK_LT(index, contactFlags_.size()) << "RobotState: no contact point " << index;
    contactFlags_[index] = contactFlag;
  }

  const std::vector<bool>& getContactFlags() const { return contactFlags_; }

  scalar_t getTime() const { return time_; }

  void setTime(scalar_t time) { time_ = time; }

  void setConfigurationToZero();

 private:
  /** The state of joint `jointId`, or nullptr when it is not a joint. */
  const JointState* absl_nullable findJoint(size_t jointId) const;
  JointState* absl_nullable findJoint(size_t jointId);

  static scalar_t positionOf(const JointState& jointState) { return jointState.position; }
  static scalar_t velocityOf(const JointState& jointState) { return jointState.velocity; }

  JointIdMap<JointState> jointStateMap_;

  scalar_t time_ = 0.0;

  vector3_t rootPosition_;
  vector3_t rootLinearVelocity_;
  quaternion_t rootOrientation_;
  vector3_t rootAngularVelocity_;
  std::vector<bool> contactFlags_;
};

}  // namespace robot::model
