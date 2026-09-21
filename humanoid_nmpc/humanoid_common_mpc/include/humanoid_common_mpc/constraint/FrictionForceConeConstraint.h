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

#include <ocs2_core/constraint/StateInputConstraint.h>
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Implements the constraint h(t,x,u) >= 0
 *
 * frictionCoefficient * (Fz + gripperForce) - sqrt(Fx * Fx + Fy * Fy + regularization) >= 0
 *
 * The gripper force shifts the origin of the friction cone down in z-direction by the amount of gripping force available. This makes it
 * possible to produce tangential forces without applying a regular normal force on that foot, or to "pull" on the foot with magnitude up to
 * the gripping force.
 *
 * The regularization prevents the constraint gradient / hessian to go to infinity when Fx = Fz = 0. It also creates a parabolic safety
 * margin to the friction cone. For example: when Fx = Fy = 0 the constraint zero-crossing will be at Fz = 1/frictionCoefficient *
 * sqrt(regularization) instead of Fz = 0
 *
 */
class FrictionForceConeConstraint final : public StateInputConstraint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * frictionCoefficient: The coefficient of friction.
   * regularization: A positive number to regulize the friction constraint. refer to the FrictionForceConeConstraint documentation.
   * gripperForce: Gripper force in normal direction.
   * hessianDiagonalShift: The Hessian shift to assure a strictly-convex quadratic constraint approximation.
   */
  struct Config {
    explicit Config(scalar_t frictionCoefficientParam = 0.7,
                    scalar_t regularizationParam = 25.0,
                    scalar_t gripperForceParam = 0.0,
                    scalar_t hessianDiagonalShiftParam = 1e-6)
        : frictionCoefficient(frictionCoefficientParam),
          regularization(regularizationParam),
          gripperForce(gripperForceParam),
          hessianDiagonalShift(hessianDiagonalShiftParam) {
      assert(frictionCoefficient > 0.0);
      assert(regularization > 0.0);
      assert(hessianDiagonalShift >= 0.0);
    }

    scalar_t frictionCoefficient;
    scalar_t regularization;
    scalar_t gripperForce;
    scalar_t hessianDiagonalShift;
  };

  /**
   * Constructor
   * @param [in] referenceManager : Switched model ReferenceManager.
   * @param [in] config : Friction model settings.
   * @param [in] contactPointIndex : The 3 DoF contact index.
   * @param [in] info : The centroidal model information.
   * @param [in] scheduleGated : When true (the historical behaviour) the term switches itself off while the mode
   *             schedule calls this foot a swing foot, which is only sound because the hard `zero_wrench` constraint
   *             has already pinned its wrench to zero. When false - the contact-implicit formulation, which removes
   *             `zero_wrench` - the cone is enforced at every node. Two things then have to change, or an always-on
   *             cone reports a permanent violation on a foot in flight: `gripperForce` is dropped, because a foot in
   *             flight has no adhesion to offer, and the constant `sqrt(regularization)` is added back to the value so
   *             that the zero force sits exactly on the cone rather than `sqrt(regularization)` inside it. Neither
   *             touches the gradient or the Hessian, so the smoothing the regularization exists for is untouched and
   *             only its parabolic safety margin - a statement about a loaded foot - is removed.
   *             See humanoid_nmpc/docs/contact_implicit_mpc/README.md.
   */
  FrictionForceConeConstraint(const SwitchedModelReferenceManager& referenceManager,
                              Config config,
                              size_t contactPointIndex,
                              const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                              bool scheduleGated = true);

  /** Whether this term is gated on the mode schedule's contact flag; see the constructor. */
  bool isScheduleGated() const { return scheduleGated_; }

  /** The adhesion a non-gated cone drops, so that a caller can state what it asked for. */
  static Config withoutAdhesion(Config config);

  /** The configuration this term ended up with, after any un-gating adjustment; exposed for the tests. */
  const Config& getConfig() const { return config_; }

  /**
   * The 3 x getContactInputDim() block of d(contact force) / d(input) this term linearises through; for the tests.
   *
   * It is the identity for a wrench-space model and the force rows of `B_local` under BasisInputsModelDecorator, and
   * pinning it is what keeps the cone's Jacobian tied to the parameterization actually in use.
   */
  const matrix_t& getContactForceInputJacobian() const { return contactForceInputJacobian_; }

  ~FrictionForceConeConstraint() override = default;
  FrictionForceConeConstraint* clone() const override { return new FrictionForceConeConstraint(*this); }

  bool isActive(scalar_t time) const override;
  void setActive(bool active) override { isActive_ = active; }
  bool getActive() const override { return isActive_; }
  size_t getNumConstraints(scalar_t time) const override { return 1; };
  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const vector_t& input,
                                                           const PreComputation& preComp) const override;
  VectorFunctionQuadraticApproximation getQuadraticApproximation(scalar_t time,
                                                                 const vector_t& state,
                                                                 const vector_t& input,
                                                                 const PreComputation& preComp) const override;

  /** Sets the estimated terrain normal expressed in the world frame. */
  void setSurfaceNormalInWorld(const vector3_t& surfaceNormalInWorld);

 private:
  struct LocalForceDerivatives {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 3 x contactInputDim_: derivative of the local-frame force w.r.t. THIS CONTACT'S INPUT BLOCK. It used to be the
    // 3x3 rotation alone, which silently assumed the block held the three force components; see the member below.
    matrix_t dF_du;
  };

  struct ConeLocalDerivatives {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vector3_t dCone_dF;    // derivative w.r.t local force
    matrix3_t d2Cone_dF2;  // second derivative w.r.t local force
  };

  struct ConeDerivatives {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    matrix_t dCone_du;    // 1 x contactInputDim_
    matrix_t d2Cone_du2;  // contactInputDim_ x contactInputDim_
  };

  FrictionForceConeConstraint(const FrictionForceConeConstraint& other);
  vector_t coneConstraint(const vector3_t& localForces) const;
  LocalForceDerivatives computeLocalForceDerivatives() const;
  ConeLocalDerivatives computeConeLocalDerivatives(const vector3_t& localForces) const;
  ConeDerivatives computeConeConstraintDerivatives(const ConeLocalDerivatives& coneLocalDerivatives,
                                                   const LocalForceDerivatives& localForceDerivatives) const;

  matrix_t frictionConeInputDerivative(size_t inputDim, const ConeDerivatives& coneDerivatives) const;
  matrix_t frictionConeSecondDerivativeInput(size_t inputDim, const ConeDerivatives& coneDerivatives) const;
  matrix_t frictionConeSecondDerivativeState(size_t stateDim, const ConeDerivatives& coneDerivatives) const;

  const SwitchedModelReferenceManager* referenceManagerPtr_;
  const MpcRobotModelBase<scalar_t>* mpcRobotModelPtr_;

  const Config config_;
  const size_t contactPointIndex_;

  // WHERE THIS CONTACT'S INPUTS ARE, AND HOW ITS FORCE DEPENDS ON THEM, read off the model at construction rather
  // than assumed. The wrench-space CentroidalMpcRobotModel stores the three force components directly in the input,
  // so d(force)/d(input) over the block is an identity and the derivative could be written as a 3x3 at the force
  // start index. Under BasisInputsModelDecorator - which the shipped DRC Atlas runs - the same contact occupies a
  // wider block of basis scalings and the force is `B_local * lambda`, so that 3x3 write lands on the first three
  // scalings and is wrong in every entry: getValue() evaluated the cone on `B_local * lambda` while the solver was
  // handed the Jacobian of a different function. See contactForceInputJacobian().
  const size_t contactInputStart_;
  const size_t contactInputDim_;
  const matrix_t contactForceInputJacobian_;  // 3 x contactInputDim_

  // rotation world to terrain
  matrix3_t t_R_w = matrix3_t::Identity();

  bool isActive_ = true;
  // Fixed by the formulation at load time rather than tuned, so it is const and the parallel solve reads it without
  // synchronisation. It has to survive the copy the SQP solver makes of the whole problem per worker thread.
  const bool scheduleGated_;
  // sqrt(regularization) when the term is not schedule gated, 0 otherwise; added to the cone value so that the zero
  // force lies exactly on the cone. See the constructor.
  const scalar_t coneValueOffset_;
};

}  // namespace ocs2::humanoid
