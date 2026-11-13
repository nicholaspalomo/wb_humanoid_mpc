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

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/common/Types.h"

namespace ocs2::legged {

/**
 * What the state and input costs regularize the leg joints toward
 * (leg_joint_reference of the task file, by name, legJointReferenceFromName()).
 */
enum class LegJointReference {
  /** The target's joint positions and zero joint velocities (the default). */
  kDefaultPosture,
  /**
   * The leg joints that move each contact relative to the base as the
   * references of the plan move it: a stance contact still in the world where
   * it landed, under the commanded base twist; a swing contact along its swing
   * reference (FootMotionLegReference).
   */
  kFootMotion,
};

/** The names legJointReferenceFromName() accepts, in the enum's order. */
std::vector<std::string> legJointReferenceNames();

/** The name of `reference`, the inverse of legJointReferenceFromName(). */
absl::string_view legJointReferenceName(LegJointReference reference);

/** The reference named `name`; InvalidArgument naming the accepted names for
 * any other. */
absl::StatusOr<LegJointReference> legJointReferenceFromName(
    absl::string_view name);

/**
 * The leg maps of the foot_motion leg joint reference
 * (LegJointReference::kFootMotion): for each contact of a set of 3-DoF point
 * contacts on separate legs, the joints of its leg at the nominal posture,
 * where the contact then stands relative to the base, and the leg's Jacobian
 * and its derivatives there. setLegJointPositions() and setLegJointVelocities()
 * map a contact position and velocity relative to the base to the leg's joint
 * positions and velocities through them, to second order about the nominal
 * posture: a first-order map would put a foot displaced along the stride by d
 * about d^2 / (2 l) too high for a leg of length l, which lowers the base by
 * as much, 8 mm on ANYmal B at 0.5 m/s.
 *
 * The map is the nominal posture's, not that of the state a node is at, so that
 * the references do not depend on the state and the costs' Gauss-Newton
 * gradients, which hold the references constant, stay exact.
 *
 * Usage: Create() once at start-up; then the two setters per node. A value
 * type; its const functions are safe to call concurrently.
 */
class FootMotionLegReference {
 public:
  /**
   * Derives the maps of `mpcRobotModel`'s contact set at `nominalState` from
   * `pinocchioInterface` (its generalized velocity the base's 6 entries and
   * then the MPC joints in their order). The leg of a contact is the joints
   * whose columns of the contact's position Jacobian are not zero there.
   * InvalidArgument when a contact is not a 3-DoF point contact, when two
   * contacts share a joint (a foot with several contacts), or when a leg's
   * Jacobian is singular at the nominal posture (its smallest singular value
   * below kMinSingularValue).
   */
  static absl::StatusOr<FootMotionLegReference> Create(
      PinocchioInterface pinocchioInterface,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      const vector_t& nominalState);

  /** [m/rad] The smallest singular value a leg's Jacobian may have. */
  static constexpr scalar_t kMinSingularValue = 1.0e-3;

  FootMotionLegReference(const FootMotionLegReference&) = default;
  FootMotionLegReference& operator=(const FootMotionLegReference&) = default;
  FootMotionLegReference(FootMotionLegReference&&) noexcept = default;
  FootMotionLegReference& operator=(FootMotionLegReference&&) noexcept =
      default;
  ~FootMotionLegReference() = default;

  /** The contacts of the set, in contact order. */
  size_t numContacts() const { return legs_.size(); }

  /** [m] Where contact `contactIndex` stands relative to the base at the
   * nominal posture, in the heading frame. `contactIndex` < numContacts(). */
  const vector3_t& getNomPosContactRtBaseEwrtHeading(
      size_t contactIndex) const {
    return legs_[contactIndex].nomPosContactRtBaseEwrtHeading;
  }

  /**
   * Writes the joint positions of contact `contactIndex`'s leg that put the
   * contact at `posContactRtBaseEwrtHeading` [m] into their entries of
   * `jointPositions` (the MPC joints, getJointDim() entries); the other
   * entries are left as they are. `contactIndex` < numContacts().
   */
  void setLegJointPositions(size_t contactIndex,
                            const vector3_t& posContactRtBaseEwrtHeading,
                            Eigen::Ref<vector_t> jointPositions) const;

  /**
   * Writes the joint velocities of contact `contactIndex`'s leg that move the
   * contact, at `posContactRtBaseEwrtHeading` [m], at
   * `linvelContactRtBaseEwrtHeading` [m/s] into their entries of
   * `jointVelocities` (the MPC joints); the other entries are left as they
   * are. The minimum-norm rates of the Jacobian at the posture
   * setLegJointPositions() gives for that position. `contactIndex` <
   * numContacts().
   */
  void setLegJointVelocities(size_t contactIndex,
                             const vector3_t& posContactRtBaseEwrtHeading,
                             const vector3_t& linvelContactRtBaseEwrtHeading,
                             Eigen::Ref<vector_t> jointVelocities) const;

  /**
   * [m/s] The velocity relative to the base, in the heading frame, of a contact
   * that stays still in the world while the base moves at
   * `cmdLinvelBaseRtWorldEwrtHeadingXy` [m/s] and turns at
   * `cmdRateYawBaseRtWorld` [rad/s], the contact at
   * `posContactRtBaseEwrtHeading` [m]: -(v + omega e_z x r).
   */
  static vector3_t computeStanceLinvelContactRtBaseEwrtHeading(
      const vector2_t& cmdLinvelBaseRtWorldEwrtHeadingXy,
      scalar_t cmdRateYawBaseRtWorld,
      const vector3_t& posContactRtBaseEwrtHeading);

 private:
  /** One contact's leg. */
  struct Leg {
    /** The leg's joints, as indices of the MPC joints. */
    std::vector<size_t> jointIndices;
    /** [rad] The leg's joints at the nominal posture, in jointIndices' order.
     */
    vector_t nomJointPositions;
    /** [m/rad] The leg's 3 x jointIndices.size() Jacobian, in the heading
     * frame, and its pseudo-inverse. */
    matrix_t jacobian;
    matrix_t inverseJacobian;
    /** [m/rad^2] d jacobian / dq_j, one per joint of the leg. */
    std::vector<matrix_t> jacobianDerivatives;
    vector3_t nomPosContactRtBaseEwrtHeading = vector3_t::Zero();
  };

  /** [rad] The displacement of `leg`'s joints from the nominal posture that
   * moves its contact by `errPosContactRtBaseEwrtHeading` [m], to second order.
   */
  static vector_t computeLegJointDisplacement(
      const Leg& leg, const vector3_t& errPosContactRtBaseEwrtHeading);

  explicit FootMotionLegReference(std::vector<Leg> legs)
      : legs_(std::move(legs)) {}

  std::vector<Leg> legs_;
};

}  // namespace ocs2::legged
