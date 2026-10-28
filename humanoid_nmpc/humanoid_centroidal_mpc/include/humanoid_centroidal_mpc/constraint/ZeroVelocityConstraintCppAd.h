/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include <memory>

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/constraint/StateInputConstraint.h"

#include "humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * The stance-foot (zero_velocity) constraint of the centroidal MPC: an
 * EndEffectorKinematicsTwistConstraint Ax * pose + Av * twist + b = 0 of one
 * contact frame, active while the reference manager's mode schedule has the
 * foot in contact. Its rows are the 6 of a 6-DoF (surface) contact - the linear
 * velocity and the rate of the orientation error with respect to the ground
 * plane, with the yaw rate about the contact normal - or the first 3, the
 * translation alone (numConstraints), which is all a 3-DoF (point) contact
 * has. The rows are written in the frame of the patch the contact touches
 * (SwitchedModelReferenceManager::getContactPatchFrame(), a parameter of the
 * contact phase read on every evaluation). On a world-aligned patch, b_z is
 * set to -Ax_zz times the foot's reference height of the
 * HumanoidPreComputation, so that the z position gain pulls the sole to the
 * ground the swing planner was built on - the rows every shipped robot has
 * always had; on any other patch, or for a contact the swing planner does not
 * plan (a contact set other than the biped's), to -Ax_zz times the patch's
 * normal offset. The gains come from footTwistConfig()
 * (ZeroVelocityGainsApplier.h). Not thread-safe; the solver clones it per
 * worker.
 */
class ZeroVelocityConstraintCppAd final : public StateInputConstraint {
 public:
  /** What the contact can be held by: the six rows of a surface (6-DoF)
   * contact, or the three translation rows of a point (3-DoF) contact. */
  enum class ContactKind { kSurface, kPoint };

  /**
   * Constructor
   * @param [in] referenceManager : Switched model ReferenceManager
   * @param [in] endEffectorKinematics: The kinematic interface to the target
   * end-effector.
   * @param [in] contactPointIndex : The index of the contact in the reference
   * manager's contact flags.
   * @param [in] numConstraints : Number of constraint rows (3 =
   * translation-only, 6 = full pose of a 6-DoF contact, the default).
   * @param [in] config: The constraint coefficients
   * @param [in] contactKind: A point contact keeps three rows whatever
   * numConstraints and a reconfiguration ask for
   * (configureZeroVelocityConstraint()).
   */
  ZeroVelocityConstraintCppAd(
      const SwitchedModelReferenceManager& referenceManager,
      const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
      size_t contactPointIndex,
      size_t numConstraints = kPoseTangentDim,
      EndEffectorKinematicsTwistConstraint::Config config =
          EndEffectorKinematicsTwistConstraint::Config(),
      ContactKind contactKind = ContactKind::kSurface);

  ContactKind getContactKind() const { return contactKind_; }

  ~ZeroVelocityConstraintCppAd() override = default;
  ZeroVelocityConstraintCppAd* absl_nonnull clone() const override {
    return new ZeroVelocityConstraintCppAd(*this);
  }
  // Copied only by clone(), whose copy constructor is private; never assigned
  // or moved.
  ZeroVelocityConstraintCppAd& operator=(const ZeroVelocityConstraintCppAd&) =
      delete;
  ZeroVelocityConstraintCppAd(ZeroVelocityConstraintCppAd&&) = delete;
  ZeroVelocityConstraintCppAd& operator=(ZeroVelocityConstraintCppAd&&) =
      delete;

  bool isActive(scalar_t time) const override;
  void setActive(bool isActive) override { isActive_ = isActive; }
  bool getActive() const override { return isActive_; }
  /** Access the inner twist constraint to allow runtime config updates (e.g.
   * foot error gains). */
  EndEffectorKinematicsTwistConstraint& getTwistConstraint() {
    return *eeTwistConstraintPtr_;
  }
  size_t getNumConstraints(scalar_t time) const override {
    return eeTwistConstraintPtr_->getNumConstraints(time);
  }
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
  ZeroVelocityConstraintCppAd(const ZeroVelocityConstraintCppAd& rhs);

  /** Puts the twist constraint on the patch contact `contactPointIndex_`
   * touches at `time`, with b_z its reference height (the class comment). */
  void configureForEvaluation(scalar_t time,
                              const PreComputation& preComp) const;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  std::unique_ptr<EndEffectorKinematicsTwistConstraint> eeTwistConstraintPtr_;
  const size_t contactPointIndex_;
  const ContactKind contactKind_;
  bool isActive_ = true;
};

}  // namespace ocs2::humanoid
