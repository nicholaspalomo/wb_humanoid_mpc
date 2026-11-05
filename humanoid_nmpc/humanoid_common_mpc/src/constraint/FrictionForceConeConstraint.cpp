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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/constraint/ConstraintOrder.h"
#include "ocs2_core/constraint/StateInputConstraint.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactInputJacobian.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/PinocchioFrameConversions.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

namespace {

// The largest force Jacobian entry outside a contact's input block that still
// counts as zero.
constexpr scalar_t kOutsideBlockTolerance = 1.0e-12;

/** The rows of the cone: the friction row of both forms, then the un-gated
 * form's Fz >= 0. */
constexpr int kFrictionRow = 0;
constexpr int kNormalForceRowIndex = 1;

/** [v]x, with [v]x w = v x w. */
matrix3_t skewSymmetric(const vector3_t& v) {
  matrix3_t skew;
  skew << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return skew;
}

/**
 * This contact's columns of the force Jacobian, with the contract they rest on
 * verified rather than assumed.
 *
 * MpcRobotModelBase declares that a contact's inputs occupy
 * `[getContactWrenchStartIndices, + getContactInputDim)`. A model whose contact
 * force also depended on inputs outside that block would lose those columns
 * here, so the block is checked against the full probe once, at construction,
 * where the cost is irrelevant.
 */
matrix_t computeBlockInputJacobianForceContactEwrtParam(
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    size_t contactPointIndex) {
  const Eigen::Index start = static_cast<Eigen::Index>(
      mpcRobotModel.getContactWrenchStartIndices(contactPointIndex));
  const Eigen::Index width = static_cast<Eigen::Index>(
      mpcRobotModel.getContactInputDim(contactPointIndex));
  const matrix_t jacobian =
      inputJacobianForceContactEwrtParam(mpcRobotModel, contactPointIndex);

  matrix_t outsideTheBlock = jacobian;
  outsideTheBlock.middleCols(start, width).setZero();
  CHECK(outsideTheBlock.isZero(kOutsideBlockTolerance))
      << "[FrictionForceConeConstraint] the force of contact "
      << contactPointIndex << " depends on inputs outside its own block ["
      << start << ", " << start + width
      << "), which the cone's Jacobian would drop";
  return jacobian.middleCols(start, width);
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
FrictionForceConeConstraint::FrictionForceConeConstraint(
    const SwitchedModelReferenceManager& referenceManager,
    Config config,
    size_t contactPointIndex,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const PinocchioInterface& pinocchioInterface,
    bool scheduleGated)
    // The un-gated friction row is convex in Fz, so it is linearized rather
    // than handed to the penalty with its indefinite Hessian; see the class
    // comment.
    : StateInputConstraint(scheduleGated ? ConstraintOrder::Quadratic
                                         : ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      mpcRobotModelPtr_(&mpcRobotModel),
      pinocchioInterfacePtr_(&pinocchioInterface),
      config_(scheduleGated ? config : withoutAdhesion(config)),
      contactPointIndex_(contactPointIndex),
      contactInputStart_(
          mpcRobotModel.getContactWrenchStartIndices(contactPointIndex)),
      contactInputDim_(mpcRobotModel.getContactInputDim(contactPointIndex)),
      inputJacobianForceContactEwrtParam_(
          computeBlockInputJacobianForceContactEwrtParam(mpcRobotModel,
                                                         contactPointIndex)),
      contactInputFrame_(mpcRobotModel.getContactInputFrame(contactPointIndex)),
      contactFrameIndex_(getContactFrameIndex(
          pinocchioInterface, mpcRobotModel, contactPointIndex)),
      scheduleGated_(scheduleGated) {
  // The documented guard for the callers that construct directly; Create()
  // returns the same refusal as a Status.
  const absl::Status velocityDim = checkPinocchioVelocityDim(
      pinocchioInterface, mpcRobotModel, contactPointIndex);
  CHECK(velocityDim.ok()) << velocityDim.message();
}

absl::StatusOr<std::unique_ptr<FrictionForceConeConstraint>>
FrictionForceConeConstraint::Create(
    const SwitchedModelReferenceManager& referenceManager,
    Config config,
    size_t contactPointIndex,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const PinocchioInterface& pinocchioInterface,
    bool scheduleGated) {
  RETURN_IF_ERROR(checkPinocchioVelocityDim(pinocchioInterface, mpcRobotModel,
                                            contactPointIndex));
  return std::make_unique<FrictionForceConeConstraint>(
      referenceManager, config, contactPointIndex, mpcRobotModel,
      pinocchioInterface, scheduleGated);
}

absl::Status FrictionForceConeConstraint::checkPinocchioVelocityDim(
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    size_t contactPointIndex) {
  // A world-frame input never reads the Pinocchio model.
  if (mpcRobotModel.getContactInputFrame(contactPointIndex) ==
      ContactInputFrame::kWorld) {
    return absl::OkStatus();
  }
  const int pinocchioVelocityDim = pinocchioInterface.getModel().nv;
  if (static_cast<int64_t>(mpcRobotModel.getVelocityDim()) !=
      static_cast<int64_t>(pinocchioVelocityDim)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[FrictionForceConeConstraint] contact ", contactPointIndex,
        " has its input force in contact axes, whose orientation depends on "
        "the state, but the robot model's Pinocchio velocity has ",
        mpcRobotModel.getVelocityDim(), " entries and the Pinocchio model's ",
        pinocchioVelocityDim,
        ": the contact frame's Jacobian cannot be taken to the state."));
  }
  return absl::OkStatus();
}

FrictionForceConeConstraint::Config
FrictionForceConeConstraint::withoutAdhesion(Config config) {
  if (config.gripForceContactEwrtSurfaceZ > 0.0) {
    LOG(WARNING) << "[FrictionForceConeConstraint] the contact-implicit "
                    "formulation drops "
                 << "the gripper force of its Config ("
                 << config.gripForceContactEwrtSurfaceZ
                 << " N): an always-active cone cannot credit a foot in flight "
                    "with adhesion.";
  }
  config.gripForceContactEwrtSurfaceZ = 0.0;
  return config;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

// Copies every member, isActive_ included: a hand-written copy that dropped it
// made every per-thread copy the SQP solver makes of the problem revert a
// deactivated cone to active.
FrictionForceConeConstraint::FrictionForceConeConstraint(
    const FrictionForceConeConstraint& rhs) = default;

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool FrictionForceConeConstraint::isActive(scalar_t time) const {
  if (!isActive_) return false;
  // Under the contact-implicit formulation the mode schedule no longer decides
  // which foot carries load, so it cannot be allowed to decide which foot's
  // force is bounded either; see contactConstraintsAreScheduleGated().
  if (!scheduleGated_) return true;
  return referenceManagerPtr_->isInContact(time, contactPointIndex_);
}

matrix3_t FrictionForceConeConstraint::getRotmatWorldRtSurface(
    scalar_t time) const {
  return referenceManagerPtr_->getContactPatchFrame(contactPointIndex_, time)
      .rotmatWorldRtSurface;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
FrictionForceConeConstraint::ForceContactEwrtSurface
FrictionForceConeConstraint::computeForceContactEwrtSurface(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    bool withStateJacobian) const {
  ForceContactEwrtSurface forceContactEwrtSurface;
  const vector3_t forceContactEwrtParam =
      mpcRobotModelPtr_->getForceContactEwrtParam(input, contactPointIndex_);
  const matrix3_t rotmatWorldRtSurface = getRotmatWorldRtSurface(time);
  if (contactInputFrame_ == ContactInputFrame::kWorld) {
    // The input stores the world-frame force: F = p_R_w F_param, and nothing
    // depends on the state.
    forceContactEwrtSurface.rotmatParamRtSurface = rotmatWorldRtSurface;
    forceContactEwrtSurface.value =
        rotmatWorldRtSurface * forceContactEwrtParam;
    return forceContactEwrtSurface;
  }

  // The input stores the force in contact axes: F = p_R_w w_R_c(q) F_param.
  const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
  pinocchio::Data data = pinocchioInterfacePtr_->getData();
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  updateFramePlacements(q, model, data);
  const matrix3_t rotmatContactRtWorld =
      getRotmatLinkRtWorld(data, contactFrameIndex_);
  forceContactEwrtSurface.rotmatParamRtSurface =
      rotmatWorldRtSurface * rotmatContactRtWorld;
  forceContactEwrtSurface.value =
      forceContactEwrtSurface.rotmatParamRtSurface * forceContactEwrtParam;
  if (!withStateJacobian) return forceContactEwrtSurface;

  // THE STATE DERIVATIVE, through w_R_c. With w_R_c' = w_R_c [omega_c]x, a
  // fixed contact-axes vector moves in world axes as w_R_c [omega_c]x F_param
  // = -w_R_c [F_param]x omega_c, and omega_c is the angular half of the contact
  // frame's LOCAL Jacobian times the Pinocchio velocity, which
  // getPinocchioTangentJacobian() takes to the state.
  matrix_t jacContactRtWorldEwrtContact = matrix_t::Zero(kTwistDim, model.nv);
  pinocchio::computeFrameJacobian(model, data, q, contactFrameIndex_,
                                  pinocchio::LOCAL,
                                  jacContactRtWorldEwrtContact);
  forceContactEwrtSurface.angjacContactRtWorldEwrtContact =
      jacContactRtWorldEwrtContact.bottomRows<kRotationTangentDim>() *
      mpcRobotModelPtr_->getPinocchioTangentJacobian(state);
  forceContactEwrtSurface.stateJacobian =
      -forceContactEwrtSurface.rotmatParamRtSurface *
      skewSymmetric(forceContactEwrtParam) *
      forceContactEwrtSurface.angjacContactRtWorldEwrtContact;
  return forceContactEwrtSurface;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t FrictionForceConeConstraint::getValue(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  return coneConstraint(computeForceContactEwrtSurface(
                            time, state, input, /*withStateJacobian=*/false)
                            .value);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation
FrictionForceConeConstraint::getLinearApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  const ForceContactEwrtSurface forceContactEwrtSurface =
      computeForceContactEwrtSurface(time, state, input,
                                     /*withStateJacobian=*/true);

  const LocalForceDerivatives inputJacobianForceContactEwrtSurface =
      computeInputJacobianForceContactEwrtSurface(
          forceContactEwrtSurface.rotmatParamRtSurface);
  const ConeLocalDerivatives coneLocalDerivatives =
      computeConeLocalDerivatives(forceContactEwrtSurface.value);
  const ConeDerivatives coneDerivatives = computeConeConstraintDerivatives(
      coneLocalDerivatives, inputJacobianForceContactEwrtSurface);

  VectorFunctionLinearApproximation linearApproximation;
  linearApproximation.f = coneConstraint(forceContactEwrtSurface.value);
  if (contactInputFrame_ == ContactInputFrame::kWorld) {
    linearApproximation.dfdx =
        matrix_t::Zero(linearApproximation.f.size(), state.size());
  } else {
    linearApproximation.dfdx =
        coneLocalDerivatives.dCone_dF * forceContactEwrtSurface.stateJacobian;
  }
  linearApproximation.dfdu =
      frictionConeInputDerivative(input.size(), coneDerivatives);
  return linearApproximation;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionQuadraticApproximation
FrictionForceConeConstraint::getQuadraticApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  const ForceContactEwrtSurface forceContactEwrtSurface =
      computeForceContactEwrtSurface(time, state, input,
                                     /*withStateJacobian=*/true);

  const LocalForceDerivatives inputJacobianForceContactEwrtSurface =
      computeInputJacobianForceContactEwrtSurface(
          forceContactEwrtSurface.rotmatParamRtSurface);
  const ConeLocalDerivatives coneLocalDerivatives =
      computeConeLocalDerivatives(forceContactEwrtSurface.value);
  const ConeDerivatives coneDerivatives = computeConeConstraintDerivatives(
      coneLocalDerivatives, inputJacobianForceContactEwrtSurface);

  VectorFunctionQuadraticApproximation quadraticApproximation;
  quadraticApproximation.f = coneConstraint(forceContactEwrtSurface.value);
  quadraticApproximation.dfdu =
      frictionConeInputDerivative(input.size(), coneDerivatives);
  if (contactInputFrame_ == ContactInputFrame::kWorld) {
    quadraticApproximation.dfdx =
        matrix_t::Zero(quadraticApproximation.f.size(), state.size());
    for (const matrix_t& d2Cone_du2 : coneDerivatives.d2Cone_du2) {
      quadraticApproximation.dfdxx.emplace_back(
          frictionConeSecondDerivativeState(state.size()));
      quadraticApproximation.dfduu.emplace_back(
          frictionConeSecondDerivativeInput(input.size(), d2Cone_du2));
      quadraticApproximation.dfdux.emplace_back(
          matrix_t::Zero(input.size(), state.size()));
    }
    return quadraticApproximation;
  }

  const matrix_t& stateJacobian = forceContactEwrtSurface.stateJacobian;
  const matrix_t& inputJacobian = inputJacobianForceContactEwrtSurface.dF_du;
  quadraticApproximation.dfdx = coneLocalDerivatives.dCone_dF * stateJacobian;
  for (size_t row = 0; row < coneDerivatives.d2Cone_du2.size(); ++row) {
    const matrix3_t& d2Cone_dF2 = coneLocalDerivatives.d2Cone_dF2[row];
    // The state Hessian keeps the cone's curvature in F and drops that of
    // w_R_c(q) F_param; see the class comment.
    matrix_t dfdxx = frictionConeSecondDerivativeState(state.size());
    dfdxx.noalias() += stateJacobian.transpose() * d2Cone_dF2 * stateJacobian;
    quadraticApproximation.dfdxx.emplace_back(std::move(dfdxx));
    quadraticApproximation.dfduu.emplace_back(frictionConeSecondDerivativeInput(
        input.size(), coneDerivatives.d2Cone_du2[row]));
    // Exact: the cone's curvature in F, and the input Jacobian turning with
    // w_R_c(q).
    matrix_t dfdux = matrix_t::Zero(input.size(), state.size());
    dfdux.middleRows(static_cast<Eigen::Index>(contactInputStart_),
                     static_cast<Eigen::Index>(contactInputDim_)) =
        inputJacobian.transpose() * d2Cone_dF2 * stateJacobian +
        computeInputJacobianStateDerivative(
            coneLocalDerivatives.dCone_dF.row(static_cast<Eigen::Index>(row)),
            forceContactEwrtSurface);
    quadraticApproximation.dfdux.emplace_back(std::move(dfdux));
  }
  return quadraticApproximation;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
matrix_t FrictionForceConeConstraint::computeInputJacobianStateDerivative(
    const matrix_t& dCone_dF,
    const ForceContactEwrtSurface& forceContactEwrtSurface) const {
  // Column j of the input Jacobian is p_R_w w_R_c(q) b_j, with b_j column j of
  // inputJacobianForceContactEwrtParam_, and it moves with the state as
  // -p_R_w w_R_c [b_j]x angjac, exactly as the force does.
  const matrix_t coneRowGradientInParamAxes =
      dCone_dF * forceContactEwrtSurface.rotmatParamRtSurface;  // 1 x 3
  matrix_t derivative = matrix_t::Zero(
      static_cast<Eigen::Index>(contactInputDim_),
      forceContactEwrtSurface.angjacContactRtWorldEwrtContact.cols());
  for (Eigen::Index column = 0;
       column < inputJacobianForceContactEwrtParam_.cols(); ++column) {
    const vector3_t inputJacobianColumn =
        inputJacobianForceContactEwrtParam_.col(column);
    derivative.row(column).noalias() =
        -coneRowGradientInParamAxes * skewSymmetric(inputJacobianColumn) *
        forceContactEwrtSurface.angjacContactRtWorldEwrtContact;
  }
  return derivative;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
FrictionForceConeConstraint::LocalForceDerivatives
FrictionForceConeConstraint::computeInputJacobianForceContactEwrtSurface(
    const matrix3_t& rotmatParamRtSurface) const {
  LocalForceDerivatives inputJacobianForceContactEwrtSurface{};
  // The cone is evaluated on `rotmatParamRtSurface *
  // getForceContactEwrtParam(input)`, so its derivative is that same rotation
  // applied to the model's own force Jacobian
  // - NOT the rotation on its own, which is only the answer when the input
  // happens to store the force directly.
  inputJacobianForceContactEwrtSurface.dF_du.noalias() =
      rotmatParamRtSurface * inputJacobianForceContactEwrtParam_;
  return inputJacobianForceContactEwrtSurface;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
FrictionForceConeConstraint::ConeLocalDerivatives
FrictionForceConeConstraint::computeConeLocalDerivatives(
    const vector3_t& forceContactEwrtSurface) const {
  const scalar_t F_x_square =
      forceContactEwrtSurface.x() * forceContactEwrtSurface.x();
  const scalar_t F_y_square =
      forceContactEwrtSurface.y() * forceContactEwrtSurface.y();
  const scalar_t F_tangent_square =
      F_x_square + F_y_square + config_.regularization;
  const scalar_t F_tangent_norm = std::sqrt(F_tangent_square);
  const scalar_t F_tangent_square_pow32 =
      F_tangent_norm * F_tangent_square;  // = F_tangent_square ^ (3/2)

  ConeLocalDerivatives coneDerivatives;
  const Eigen::Index numRows =
      scheduleGated_ ? kNumGatedConstraints : kNumUngatedConstraints;
  coneDerivatives.dCone_dF = matrix_t::Zero(numRows, kForceDim);
  coneDerivatives.d2Cone_dF2.assign(static_cast<size_t>(numRows),
                                    matrix3_t::Zero());

  // Row 0, the friction row. Its tangential half, -sqrt(Fx^2 + Fy^2 +
  // regularization), is common to both forms.
  coneDerivatives.dCone_dF(kFrictionRow, kXAxisIndex) =
      -forceContactEwrtSurface.x() / F_tangent_norm;
  coneDerivatives.dCone_dF(kFrictionRow, kYAxisIndex) =
      -forceContactEwrtSurface.y() / F_tangent_norm;
  matrix3_t& frictionHessian = coneDerivatives.d2Cone_dF2[kFrictionRow];
  frictionHessian(kXAxisIndex, kXAxisIndex) =
      -(F_y_square + config_.regularization) / F_tangent_square_pow32;
  frictionHessian(kXAxisIndex, kYAxisIndex) = forceContactEwrtSurface.x() *
                                              forceContactEwrtSurface.y() /
                                              F_tangent_square_pow32;
  frictionHessian(kYAxisIndex, kXAxisIndex) =
      frictionHessian(kXAxisIndex, kYAxisIndex);
  frictionHessian(kYAxisIndex, kYAxisIndex) =
      -(F_x_square + config_.regularization) / F_tangent_square_pow32;

  if (scheduleGated_) {
    // mu * (Fz + gripForceContactEwrtSurfaceZ): linear in Fz.
    coneDerivatives.dCone_dF(kFrictionRow, kZAxisIndex) =
        config_.frictionCoefficient;
  } else {
    // sqrt(mu^2 Fz^2 + regularization): its slope runs from 0 at the unloaded
    // foot to mu under load, and its curvature mu^2 regularization / (mu^2 Fz^2
    // + regularization)^(3/2) is positive - the reason this form is linearized.
    const scalar_t muSquare =
        config_.frictionCoefficient * config_.frictionCoefficient;
    const scalar_t F_friction_bound_square =
        muSquare * forceContactEwrtSurface.z() * forceContactEwrtSurface.z() +
        config_.regularization;
    const scalar_t F_friction_bound = std::sqrt(F_friction_bound_square);
    coneDerivatives.dCone_dF(kFrictionRow, kZAxisIndex) =
        muSquare * forceContactEwrtSurface.z() / F_friction_bound;
    frictionHessian(kZAxisIndex, kZAxisIndex) =
        muSquare * config_.regularization /
        (F_friction_bound * F_friction_bound_square);
    // Row 1, Fz >= 0: linear.
    coneDerivatives.dCone_dF(kNormalForceRowIndex, kZAxisIndex) = 1.0;
  }
  return coneDerivatives;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t FrictionForceConeConstraint::coneConstraint(
    const vector3_t& forceContactEwrtSurface) const {
  const scalar_t F_tangent_square =
      forceContactEwrtSurface.x() * forceContactEwrtSurface.x() +
      forceContactEwrtSurface.y() * forceContactEwrtSurface.y() +
      config_.regularization;
  const scalar_t F_tangent_norm = std::sqrt(F_tangent_square);
  if (scheduleGated_) {
    return (vector_t(kNumGatedConstraints)
            << config_.frictionCoefficient *
                       (forceContactEwrtSurface.z() +
                        config_.gripForceContactEwrtSurfaceZ) -
                   F_tangent_norm)
        .finished();
  }
  // The exact Coulomb cone, as two rows that are both zero at the zero force;
  // see the class comment for why the gated row with sqrt(regularization) added
  // back - which this term used to evaluate - admitted more friction than mu.
  const scalar_t muFz =
      config_.frictionCoefficient * forceContactEwrtSurface.z();
  const scalar_t F_friction_bound =
      std::sqrt(muFz * muFz + config_.regularization);
  return (vector_t(kNumUngatedConstraints) << F_friction_bound - F_tangent_norm,
          forceContactEwrtSurface.z())
      .finished();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
FrictionForceConeConstraint::ConeDerivatives
FrictionForceConeConstraint::computeConeConstraintDerivatives(
    const ConeLocalDerivatives& coneLocalDerivatives,
    const LocalForceDerivatives& inputJacobianForceContactEwrtSurface) const {
  ConeDerivatives coneDerivatives;
  // First order derivatives
  coneDerivatives.dCone_du.noalias() =
      coneLocalDerivatives.dCone_dF *
      inputJacobianForceContactEwrtSurface.dF_du;

  // Second order derivatives, one per row
  coneDerivatives.d2Cone_du2.reserve(coneLocalDerivatives.d2Cone_dF2.size());
  for (const matrix3_t& d2Cone_dF2 : coneLocalDerivatives.d2Cone_dF2) {
    coneDerivatives.d2Cone_du2.emplace_back(
        inputJacobianForceContactEwrtSurface.dF_du.transpose() * d2Cone_dF2 *
        inputJacobianForceContactEwrtSurface.dF_du);
  }
  return coneDerivatives;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
matrix_t FrictionForceConeConstraint::frictionConeInputDerivative(
    size_t inputDim, const ConeDerivatives& coneDerivatives) const {
  matrix_t dhdu = matrix_t::Zero(coneDerivatives.dCone_du.rows(), inputDim);
  dhdu.middleCols(static_cast<Eigen::Index>(contactInputStart_),
                  static_cast<Eigen::Index>(contactInputDim_)) =
      coneDerivatives.dCone_du;
  return dhdu;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
matrix_t FrictionForceConeConstraint::frictionConeSecondDerivativeInput(
    size_t inputDim, const matrix_t& d2Cone_du2) const {
  matrix_t ddhdudu = matrix_t::Zero(inputDim, inputDim);
  ddhdudu.block(static_cast<Eigen::Index>(contactInputStart_),
                static_cast<Eigen::Index>(contactInputStart_),
                static_cast<Eigen::Index>(contactInputDim_),
                static_cast<Eigen::Index>(contactInputDim_)) = d2Cone_du2;
  ddhdudu.diagonal().array() -= config_.hessianDiagonalShift;
  return ddhdudu;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
matrix_t FrictionForceConeConstraint::frictionConeSecondDerivativeState(
    size_t stateDim) const {
  matrix_t ddhdxdx = matrix_t::Zero(stateDim, stateDim);
  ddhdxdx.diagonal().array() -= config_.hessianDiagonalShift;
  return ddhdxdx;
}

}  // namespace ocs2::humanoid
