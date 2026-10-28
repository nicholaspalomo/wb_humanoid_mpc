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
#include <memory>

#include "absl/base/nullability.h"
#include "ocs2_core/constraint/StateInputConstraint.h"
#include "ocs2_robotic_tools/end_effector/EndEffectorKinematics.h"

#include "humanoid_common_mpc/constraint/EndEffectorKinematicsLinearVelConstraint.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * The swing servo of one contact of any kind on the generalized swing planner
 * (the soft constraint contact_swing; design section 7.2,
 * humanoid_nmpc/docs/multi_contact_planning/README.md):
 *
 *   g = v - v_ref + k (p - p_ref) = 0,
 *
 * three rows on the world position p and linear velocity v of the contact's
 * frame, against the swing reference (p_ref, v_ref) the pre-computation reads
 * from the reference manager's ContactSwingPlanner
 * (HumanoidPreComputation::getContactSwingReference()), with the position gain
 * k [1/s]. Active while the contact swings with a planned touch-down: not in
 * contact, and the planner has a reference for it. Wrapped in a quadratic
 * penalty it is the swing tracking of a knee, a hand or a torso pad as well as
 * a foot.
 *
 * Not thread-safe: its end-effector kinematics evaluate on the Pinocchio data
 * of one worker's PreComputation, so the solver gives each worker thread its
 * own clone(). The reference manager must outlive it.
 */
class ContactSwingConstraint final : public StateInputConstraint {
 public:
  /** The three rows of the residual. */
  static constexpr size_t kNumConstraints = 3;

  /**
   * The servo of contact `contactIndex` of the model's contact set, on
   * `endEffectorKinematics` (cloned; the kinematics of that contact's frame)
   * with the position gain `positionGain` [1/s], not negative.
   * `referenceManager` is retained.
   */
  ContactSwingConstraint(
      const SwitchedModelReferenceManager& referenceManager,
      const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
      size_t contactIndex,
      scalar_t positionGain);
  ~ContactSwingConstraint() override = default;
  ContactSwingConstraint* absl_nonnull clone() const override {
    return new ContactSwingConstraint(*this);
  }
  // Copied only by clone(), whose copy constructor is private.
  ContactSwingConstraint& operator=(const ContactSwingConstraint&) = delete;
  ContactSwingConstraint(ContactSwingConstraint&&) = delete;
  ContactSwingConstraint& operator=(ContactSwingConstraint&&) = delete;

  /** Not in contact at `time`, with a swing reference there. */
  bool isActive(scalar_t time) const override;
  void setActive(bool isActive) override { isActive_ = isActive; }
  bool getActive() const override { return isActive_; }
  size_t getNumConstraints(scalar_t /*time*/) const override {
    return kNumConstraints;
  }
  scalar_t getPositionGain() const { return positionGain_; }

  /** The residual at the pre-computation's swing reference; zero rows, and a
   * zero linearization, when it has none for the contact. */
  vector_t getValue(scalar_t time,
                    const vector_t& state,
                    const vector_t& input,
                    const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const PreComputation& preComp) const override;

 private:
  ContactSwingConstraint(const ContactSwingConstraint& rhs);

  /** Writes the coefficients of the pre-computation's reference into the
   * linear constraint; false when it has no reference for the contact. */
  bool configureForEvaluation(const PreComputation& preComp) const;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  std::unique_ptr<EndEffectorKinematicsLinearVelConstraint>
      eeLinearConstraintPtr_;
  const size_t contactIndex_;
  const scalar_t positionGain_;  // [1/s] not negative
  bool isActive_ = true;
};

}  // namespace ocs2::humanoid
