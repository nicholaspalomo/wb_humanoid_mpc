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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/constraint/StateInputConstraint.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/multibody/fwd.hpp"

#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::legged {

/**
 * Implements the friction cone of one contact as h(t,x,u) >= 0, in one of two
 * forms chosen by `scheduleGated`.
 *
 * The force F below is the contact's world-frame force written in the frame of
 * the patch the contact touches, p_R_w F_w, with the rotation p_R_w of
 * SwitchedModelReferenceManager::getContactPatchFrame(), a parameter of the
 * contact phase. On the world-aligned patch of flat ground p_R_w is the
 * identity and the cone is the world-z cone it has always been.
 *
 * F_w is the model's getForceContactEwrtParam() rotated out of the frame
 * getContactInputFrame() reports. A wrench-space model stores F_w in the input,
 * so F = p_R_w F_param and the state Jacobian is zero. Under
 * BasisInputsModelDecorator the input stores scalings of a basis fixed to the
 * foot, F_param is the force in contact axes, and F = p_R_w w_R_c(q) F_param
 * with the contact frame's orientation w_R_c read off the state through
 * Pinocchio; the cone then has a state Jacobian through w_R_c, taken with the
 * contact frame's LOCAL Jacobian as ContactWrenchConeConstraint does. Rotating
 * F_param by p_R_w alone, as this term once did, checked the cone in axes that
 * turn with the foot: a vertical world force on a foot pitched by 0.3 rad read
 * as 177 N of friction.
 *
 * SCHEDULE GATED (the historical form, one row, active only while the mode
 * schedule calls the foot a stance foot):
 *
 *   frictionCoefficient * (Fz + gripForceContactEwrtSurfaceZ) - sqrt(Fx * Fx +
 * Fy * Fy + regularization) >= 0
 *
 * The gripper force shifts the origin of the friction cone down in z-direction
 * by the amount of gripping force available. This makes it possible to produce
 * tangential forces without applying a regular normal force on that foot, or to
 * "pull" on the foot with magnitude up to the gripping force.
 *
 * The regularization prevents the constraint gradient / hessian to go to
 * infinity when Fx = Fz = 0. It also creates a parabolic safety margin to the
 * friction cone. For example: when Fx = Fy = 0 the constraint zero-crossing
 * will be at Fz = 1/frictionCoefficient * sqrt(regularization) instead of Fz =
 * 0. This is an INNER approximation of the Coulomb cone, and that is sound for
 * a foot the schedule has declared loaded.
 *
 * NOT SCHEDULE GATED (the contact-implicit formulation, two rows, active at
 * every node):
 *
 *   row 0:  sqrt(mu^2 Fz^2 + regularization) - sqrt(Fx^2 + Fy^2 +
 * regularization) >= 0 row 1:  Fz >= 0
 *
 * An always-active cone is evaluated on every foot in flight, i.e. at the ZERO
 * force, and the gated form reads -sqrt(regularization) there: a permanent
 * violation that the penalty would buy off by inventing a normal force. The
 * zero force has to lie exactly on the boundary. Adding sqrt(regularization)
 * back to the gated row achieves that, and this term used to do exactly that -
 * but it turns the inner approximation into an OUTER one. Its feasible set is
 * then |F_t| <= sqrt(mu^2 Fz^2 + 2 mu Fz sqrt(regularization)), strictly larger
 * than |F_t| <= mu Fz for every Fz > 0, by up to sqrt(regularization) = 5 N at
 * the default; at Fz = 1 N and mu = 0.4 it admitted 2 N of friction, an
 * effective coefficient five times the configured one, on exactly the lightly
 * loaded touch-down and lift-off feet this formulation exists to handle.
 *
 * The two rows above are EXACT instead: row 0 is non-negative if and only if
 * |F_t| <= mu |Fz| (both square roots are monotone in their argument, and the
 * same regularization sits under both), and row 1 removes the mirror-image cone
 * below the origin, so together they are the Coulomb cone and nothing else, for
 * every force. Both rows are smooth, both are exactly zero at the zero force,
 * and the regularization still keeps the gradient finite there - it only shapes
 * how steeply a violation near the apex is priced, not where the boundary lies.
 * No single smooth row can do this: a smooth function that is zero at the apex
 * of a pointed cone and non-negative only inside it must have a zero gradient
 * there, and then its second-order part is even in F and cannot tell the cone
 * from its mirror image.
 *
 * Row 0 is convex in Fz, so its Hessian is indefinite. The un-gated term
 * therefore declares ConstraintOrder::Linear and the soft-constraint wrapper
 * linearizes it (a Gauss-Newton approximation of the penalty), which is
 * positive semi-definite whatever the penalty; getQuadraticApproximation()
 * still returns the exact second derivatives, minus the Hessian shift, for
 * anyone who asks. The gated term keeps ConstraintOrder::Quadratic: its row is
 * concave, so its Hessian can only add curvature to the penalty. With a
 * contact-frame input the input Hessian and the input-state cross term are
 * exact, and the state Hessian keeps the cone's curvature in F, J_x^T
 * d2Cone_dF2 J_x, but drops the second derivative of w_R_c(q) F_param.
 *
 * Not thread-safe; the solver gives each worker thread its own clone().
 */
class FrictionForceConeConstraint final : public StateInputConstraint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * frictionCoefficient: The coefficient of friction.
   * regularization: A positive number to regulize the friction constraint.
   * refer to the FrictionForceConeConstraint documentation.
   * gripForceContactEwrtSurfaceZ: Gripper force in normal direction.
   * hessianDiagonalShift: The Hessian shift to assure a strictly-convex
   * quadratic constraint approximation. It is subtracted from the diagonal of
   * the input Hessian and of the state Hessian, which is in the ambient state;
   * the solver pulls the state Hessian back to the tangent, E^T H E, where the
   * quaternion block's E (G(xi) / 2 at a unit xi, E^T E = I / 4) leaves a
   * quarter of the shift on the three rotation rows.
   */
  struct Config {
    static constexpr scalar_t kDefaultFrictionCoefficient = 0.7;
    /** [N^2] the regularization under the square roots, sqrt(regularization) =
     * 5 N. */
    static constexpr scalar_t kDefaultRegularization = 25.0;
    static constexpr scalar_t kDefaultHessianDiagonalShift = 1.0e-6;

    explicit Config(
        scalar_t frictionCoefficientParam = kDefaultFrictionCoefficient,
        scalar_t regularizationParam = kDefaultRegularization,
        scalar_t gripForceContactEwrtSurfaceZ = 0.0,
        scalar_t hessianDiagonalShiftParam = kDefaultHessianDiagonalShift)
        : frictionCoefficient(frictionCoefficientParam),
          regularization(regularizationParam),
          gripForceContactEwrtSurfaceZ(gripForceContactEwrtSurfaceZ),
          hessianDiagonalShift(hessianDiagonalShiftParam) {
      ABSL_CHECK_GT(frictionCoefficient, 0.0)
          << "FrictionForceConeConstraint: frictionCoefficient must be "
             "positive";
      ABSL_CHECK_GT(regularization, 0.0)
          << "FrictionForceConeConstraint: regularization must be positive";
      ABSL_CHECK_GE(hessianDiagonalShift, 0.0)
          << "FrictionForceConeConstraint: hessianDiagonalShift must be "
             "non-negative";
    }

    scalar_t frictionCoefficient;
    scalar_t regularization;
    scalar_t gripForceContactEwrtSurfaceZ;
    scalar_t hessianDiagonalShift;
  };

  /**
   * Constructor
   * @param [in] referenceManager : Switched model ReferenceManager.
   * @param [in] config : Friction model settings.
   * @param [in] contactPointIndex : The 3 DoF contact index.
   * @param [in] mpcRobotModel : The robot model whose input the term reads;
   * retained, it must outlive the term.
   * @param [in] pinocchioInterface : The Pinocchio model the contact frame's
   * orientation and Jacobian are taken from when the model's input is in
   * contact axes; retained, it must outlive the term. Its velocity dimension
   * must equal the model's then, which the constructor checks and Create()
   * returns as a Status.
   * @param [in] scheduleGated : When true (the historical behavior) the term
   * switches itself off while the mode schedule calls this foot a swing foot,
   * which is only sound because the hard `zero_wrench` constraint has already
   * pinned its wrench to zero. When false - the contact-implicit formulation,
   * which removes `zero_wrench` - the cone is enforced at every node, in the
   * exact two-row form of the class comment, whose boundary passes through the
   * zero force; `gripForceContactEwrtSurfaceZ` is dropped as well, because a
   * foot in flight has no adhesion to offer. See
   * legged_nmpc/docs/contact_implicit_mpc/README.md.
   */
  FrictionForceConeConstraint(
      const SwitchedModelReferenceManager& referenceManager,
      Config config,
      size_t contactPointIndex,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      const PinocchioInterface& pinocchioInterface,
      bool scheduleGated = true);

  /** Returns the term of the constructor, or InvalidArgument where the
   * constructor would abort: a contact-frame input with a Pinocchio model
   * whose velocity dimension differs from the robot model's. */
  static absl::StatusOr<std::unique_ptr<FrictionForceConeConstraint>> Create(
      const SwitchedModelReferenceManager& referenceManager,
      Config config,
      size_t contactPointIndex,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      const PinocchioInterface& pinocchioInterface,
      bool scheduleGated = true);

  /** Whether this term is gated on the mode schedule's contact flag; see the
   * constructor. */
  bool isScheduleGated() const { return scheduleGated_; }

  /** The adhesion a non-gated cone drops, so that a caller can state what it
   * asked for. */
  static Config withoutAdhesion(Config config);

  /** The configuration this term ended up with, after any un-gating adjustment;
   * exposed for the tests. */
  const Config& getConfig() const { return config_; }

  /**
   * The 3 x getContactInputDim() block of d(contact force) / d(input) this term
   * linearizes through; for the tests.
   *
   * It is the identity for a wrench-space model and the force rows of `B_local`
   * under BasisInputsModelDecorator, and pinning it is what keeps the cone's
   * Jacobian tied to the parameterization actually in use.
   */
  const matrix_t& getInputJacobianForceContactEwrtParam() const {
    return inputJacobianForceContactEwrtParam_;
  }

  /** The axes of the model's input force: under ContactInputFrame::kWorld the
   * term reads only the input, and its state Hessian is the shift alone. */
  ContactInputFrame getContactInputFrame() const { return contactInputFrame_; }

  ~FrictionForceConeConstraint() override = default;
  FrictionForceConeConstraint& operator=(const FrictionForceConeConstraint&) =
      delete;
  FrictionForceConeConstraint(FrictionForceConeConstraint&&) = delete;
  FrictionForceConeConstraint& operator=(FrictionForceConeConstraint&&) =
      delete;
  FrictionForceConeConstraint* absl_nonnull clone() const override {
    return new FrictionForceConeConstraint(*this);
  }

  bool isActive(scalar_t time) const override;
  void setActive(bool active) override { isActive_ = active; }
  bool getActive() const override { return isActive_; }
  /** One row when schedule gated, two when not: the friction row, then Fz >= 0.
   * See the class comment. */
  size_t getNumConstraints(scalar_t /*time*/) const override {
    return scheduleGated_ ? kNumGatedConstraints : kNumUngatedConstraints;
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
  VectorFunctionQuadraticApproximation getQuadraticApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const PreComputation& preComp) const override;

 private:
  /** The rows of the schedule-gated form (the friction row) and of the un-gated
   * one (the friction row and Fz >= 0). */
  static constexpr int kNumGatedConstraints = 1;
  static constexpr int kNumUngatedConstraints = 2;

  struct LocalForceDerivatives {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 3 x contactInputDim_: derivative of the local-frame force w.r.t. THIS
    // CONTACT'S INPUT BLOCK. It used to be the 3x3 rotation alone, which
    // silently assumed the block held the three force components; see the
    // member below.
    matrix_t dF_du;
  };

  struct ConeLocalDerivatives {
    matrix_t dCone_dF;  // numRows x 3: derivative of every row w.r.t. the local
                        // force
    std::vector<matrix3_t> d2Cone_dF2;  // one 3 x 3 second derivative w.r.t.
                                        // the local force per row
  };

  struct ConeDerivatives {
    matrix_t dCone_du;  // numRows x contactInputDim_
    std::vector<matrix_t>
        d2Cone_du2;  // one contactInputDim_ x contactInputDim_ per row
  };

  /**
   * The force in surface axes at one (state, input), and how it moves with the
   * state. Under a world-frame input `rotmatParamRtSurface` is p_R_w and the
   * two Jacobians are empty. Under a contact-frame input it is p_R_w
   * w_R_c(q), `angjacContactRtWorldEwrtContact` is the contact frame's LOCAL
   * angular Jacobian taken to the state by getPinocchioTangentJacobian(), and
   * `stateJacobian` = -rotmatParamRtSurface [F_param]x
   * angjacContactRtWorldEwrtContact.
   */
  struct ForceContactEwrtSurface {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vector3_t value = vector3_t::Zero();
    matrix3_t rotmatParamRtSurface = matrix3_t::Identity();
    matrix_t angjacContactRtWorldEwrtContact;  // 3 x stateDim, or empty
    matrix_t stateJacobian;                    // 3 x stateDim, or empty
  };

  FrictionForceConeConstraint(const FrictionForceConeConstraint& rhs);
  /** Returns InvalidArgument when the contact-frame path cannot take the
   * Pinocchio Jacobian to the state; OK for a world-frame input. */
  static absl::Status checkPinocchioVelocityDim(
      const PinocchioInterface& pinocchioInterface,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      size_t contactPointIndex);
  /** Evaluates F at (time, state, input), with its state Jacobians when
   * `withStateJacobian` and the input is in contact axes. */
  ForceContactEwrtSurface computeForceContactEwrtSurface(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      bool withStateJacobian) const;
  /** Returns the part of d(dCone_du)/d(state) of one row, contactInputDim_ x
   * stateDim, that comes from the input Jacobian p_R_w w_R_c(q) B turning
   * with the state; `dCone_dF` is that row's 1 x 3 force gradient. */
  matrix_t computeInputJacobianStateDerivative(
      const matrix_t& dCone_dF,
      const ForceContactEwrtSurface& forceContactEwrtSurface) const;
  vector_t coneConstraint(const vector3_t& forceContactEwrtSurface) const;
  /** The rotation p_R_w of the patch the contact touches at `time`
   * (SwitchedModelReferenceManager::getContactPatchFrame()). */
  matrix3_t getRotmatWorldRtSurface(scalar_t time) const;
  LocalForceDerivatives computeInputJacobianForceContactEwrtSurface(
      const matrix3_t& rotmatParamRtSurface) const;
  ConeLocalDerivatives computeConeLocalDerivatives(
      const vector3_t& forceContactEwrtSurface) const;
  ConeDerivatives computeConeConstraintDerivatives(
      const ConeLocalDerivatives& coneLocalDerivatives,
      const LocalForceDerivatives& inputJacobianForceContactEwrtSurface) const;

  matrix_t frictionConeInputDerivative(
      size_t inputDim, const ConeDerivatives& coneDerivatives) const;
  matrix_t frictionConeSecondDerivativeInput(size_t inputDim,
                                             const matrix_t& d2Cone_du2) const;
  matrix_t frictionConeSecondDerivativeState(size_t stateDim) const;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModelPtr_;
  const PinocchioInterface* absl_nonnull pinocchioInterfacePtr_;

  const Config config_;
  const size_t contactPointIndex_;

  // WHERE THIS CONTACT'S INPUTS ARE, AND HOW ITS FORCE DEPENDS ON THEM, read
  // off the model at construction rather than assumed. The wrench-space
  // CentroidalMpcRobotModel stores the three force components directly in the
  // input, so d(force)/d(input) over the block is an identity and the
  // derivative could be written as a 3x3 at the force start index. Under
  // BasisInputsModelDecorator - the parameterization the shipped DRC Atlas
  // runs - the same
  // contact occupies a wider block of basis scalings and the force is `B_local
  // * lambda`, so that 3x3 write lands on the first three scalings and is wrong
  // in every entry: getValue() evaluated the cone on `B_local * lambda` while
  // the solver was handed the Jacobian of a different function. See
  // inputJacobianForceContactEwrtParam(). That force is also in contact axes,
  // which contactInputFrame_ records.
  const size_t contactInputStart_;
  const size_t contactInputDim_;
  const matrix_t inputJacobianForceContactEwrtParam_;  // 3 x contactInputDim_
  // The axes of getForceContactEwrtParam(), read off the model once.
  const ContactInputFrame contactInputFrame_;
  // The contact's Pinocchio frame, read only under ContactInputFrame::kContact.
  const pinocchio::FrameIndex contactFrameIndex_;

  bool isActive_ = true;
  // Fixed by the formulation at load time rather than tuned, so it is const and
  // the parallel solve reads it without synchronization. It has to survive the
  // copy the SQP solver makes of the whole problem per worker thread.
  const bool scheduleGated_;
};

}  // namespace ocs2::legged
