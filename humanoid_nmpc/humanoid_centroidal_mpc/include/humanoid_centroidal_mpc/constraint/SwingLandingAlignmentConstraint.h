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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cstddef>
#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "ocs2_core/constraint/StateConstraintCppAd.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Lands one 6-DoF contact flat on the patch it swings to (the landing
 * alignment of the soft constraint contact_swing; design section 7.2,
 * humanoid_nmpc/docs/multi_contact_planning/README.md):
 *
 *   g(x) = z_c(q) x n,
 *
 * three rows (of rank two) with z_c the world z axis of the contact's frame -
 * a 6-DoF contact frame's z axis is its pad normal - and n the outward normal
 * of the patch of its next touch-down (getContactPatchFrame()), a parameter of
 * the tape. |g| is the sine of the angle between the pad and the patch, so a
 * quadratic penalty of weight w costs (w / 2) sin^2 of it. contact_swing's
 * servo moves only the pad's center: without this term a sole can reach its
 * patch pitched, its toe or heel first. Active over the last `window` seconds
 * of a swing, by the reference manager's mode schedule: while the contact is
 * not in contact and touches down within the window.
 *
 * Like every OCS2 term it is cloned for each solver thread, and a single
 * instance is not thread-safe. The reference manager and the robot model must
 * outlive it.
 */
class SwingLandingAlignmentConstraint final : public StateConstraintCppAd {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * The alignment of 6-DoF contact `contactIndex` (its frame
   * ModelSettings::contactNames[contactIndex]) over the last `window` [s],
   * positive, of its swings, taped as the CppAD library `libraryName` in the
   * model settings' folder. `referenceManager` and `mpcRobotModel` are
   * retained.
   */
  SwingLandingAlignmentConstraint(
      const SwitchedModelReferenceManager& referenceManager,
      const PinocchioInterface& pinocchioInterface,
      const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
      size_t contactIndex,
      scalar_t window,
      const std::string& libraryName,
      const ModelSettings& modelSettings);
  ~SwingLandingAlignmentConstraint() override = default;
  SwingLandingAlignmentConstraint& operator=(
      const SwingLandingAlignmentConstraint&) = delete;
  SwingLandingAlignmentConstraint(SwingLandingAlignmentConstraint&&) = delete;
  SwingLandingAlignmentConstraint& operator=(
      SwingLandingAlignmentConstraint&&) = delete;
  SwingLandingAlignmentConstraint* absl_nonnull clone() const override {
    return new SwingLandingAlignmentConstraint(*this);
  }

  /** Not in contact at `time`, and touching down within the window after it.
   */
  bool isActive(scalar_t time) const override;
  size_t getNumConstraints(scalar_t /*time*/) const override {
    return kPositionDim;
  }
  /** The normal of the patch the contact swings to at `time`. */
  vector_t getParameters(scalar_t time,
                         const PreComputation& preComputation) const override;

  /** [s] The time from `time` to the contact's next touch-down by the
   * reference manager's mode schedule, or std::nullopt when it is in contact
   * at `time` or does not touch down again. */
  std::optional<scalar_t> timeToTouchDown(scalar_t time) const;

 private:
  SwingLandingAlignmentConstraint(const SwingLandingAlignmentConstraint& other);

  ad_vector_t constraintFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& parameters) const override;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  const MpcRobotModelBase<ad_scalar_t>* absl_nonnull mpcRobotModelPtr_;
  const size_t contactIndex_;
  const size_t numContacts_;
  const std::string frameName_;
  const scalar_t window_;
};

}  // namespace ocs2::humanoid
