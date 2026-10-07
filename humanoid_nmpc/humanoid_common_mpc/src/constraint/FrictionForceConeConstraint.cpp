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

#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"

#include <cmath>

#include "absl/log/check.h"
#include "absl/log/log.h"

#include "humanoid_common_mpc/contact/ContactInputJacobian.h"

namespace ocs2::humanoid {

namespace {

/**
 * This contact's columns of the force Jacobian, with the contract they rest on verified rather than assumed.
 *
 * MpcRobotModelBase declares that a contact's inputs occupy `[getContactWrenchStartIndices, + getContactInputDim)`.
 * A model whose contact force also depended on inputs outside that block would lose those columns here, so the block
 * is checked against the full probe once, at construction, where the cost is irrelevant.
 */
matrix_t contactBlockOfForceJacobian(const MpcRobotModelBase<scalar_t>& mpcRobotModel, size_t contactPointIndex) {
  const Eigen::Index start = static_cast<Eigen::Index>(mpcRobotModel.getContactWrenchStartIndices(contactPointIndex));
  const Eigen::Index width = static_cast<Eigen::Index>(mpcRobotModel.getContactInputDim(contactPointIndex));
  const matrix_t jacobian = contactForceInputJacobian(mpcRobotModel, contactPointIndex);

  matrix_t outsideTheBlock = jacobian;
  outsideTheBlock.middleCols(start, width).setZero();
  CHECK(outsideTheBlock.isZero(1.0e-12)) << "[FrictionForceConeConstraint] the force of contact " << contactPointIndex
                                         << " depends on inputs outside its own block [" << start << ", " << start + width
                                         << "), which the cone's Jacobian would drop";
  return jacobian.middleCols(start, width);
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
FrictionForceConeConstraint::FrictionForceConeConstraint(const SwitchedModelReferenceManager& referenceManager,
                                                         Config config,
                                                         size_t contactPointIndex,
                                                         const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                         bool scheduleGated)
    // The un-gated friction row is convex in Fz, so it is linearized rather than handed to the penalty with its
    // indefinite Hessian; see the class comment.
    : StateInputConstraint(scheduleGated ? ConstraintOrder::Quadratic : ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      mpcRobotModelPtr_(&mpcRobotModel),
      config_(scheduleGated ? config : withoutAdhesion(config)),
      contactPointIndex_(contactPointIndex),
      contactInputStart_(mpcRobotModel.getContactWrenchStartIndices(contactPointIndex)),
      contactInputDim_(mpcRobotModel.getContactInputDim(contactPointIndex)),
      contactForceInputJacobian_(contactBlockOfForceJacobian(mpcRobotModel, contactPointIndex)),
      scheduleGated_(scheduleGated) {}

FrictionForceConeConstraint::Config FrictionForceConeConstraint::withoutAdhesion(Config config) {
  if (config.gripperForce > 0.0) {
    LOG(WARNING) << "[FrictionForceConeConstraint] the contact-implicit formulation drops " << "the gripper force of its Config ("
                 << config.gripperForce << " N): an always-active cone cannot credit a foot in flight with adhesion.";
  }
  config.gripperForce = 0.0;
  return config;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

FrictionForceConeConstraint::FrictionForceConeConstraint(const FrictionForceConeConstraint& rhs)
    : StateInputConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      mpcRobotModelPtr_(rhs.mpcRobotModelPtr_),
      config_(rhs.config_),
      contactPointIndex_(rhs.contactPointIndex_),
      contactInputStart_(rhs.contactInputStart_),
      contactInputDim_(rhs.contactInputDim_),
      contactForceInputJacobian_(rhs.contactForceInputJacobian_),
      // isActive_ was dropped here, so every per-thread copy the SQP solver makes of the problem silently reverted a
      // deactivated cone to active.
      isActive_(rhs.isActive_),
      scheduleGated_(rhs.scheduleGated_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool FrictionForceConeConstraint::isActive(scalar_t time) const {
  if (!isActive_) return false;
  // Under the contact-implicit formulation the mode schedule no longer decides which foot carries load, so it cannot
  // be allowed to decide which foot's force is bounded either; see contactConstraintsAreScheduleGated().
  if (!scheduleGated_) return true;
  return referenceManagerPtr_->getContactFlags(time)[contactPointIndex_];
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t FrictionForceConeConstraint::getValue(scalar_t /*time*/,
                                               const vector_t& /*state*/,
                                               const vector_t& input,
                                               const PreComputation& /*preComp*/) const {
  const vector3_t contactForce = mpcRobotModelPtr_->getContactForce(input, contactPointIndex_);
  const vector3_t localForce = t_R_w_ * contactForce;
  return coneConstraint(localForce);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation FrictionForceConeConstraint::getLinearApproximation(scalar_t /*time*/,
                                                                                      const vector_t& state,
                                                                                      const vector_t& input,
                                                                                      const PreComputation& /*preComp*/) const {
  const vector3_t contactForce = mpcRobotModelPtr_->getContactForce(input, contactPointIndex_);
  const vector3_t localForce = t_R_w_ * contactForce;

  const LocalForceDerivatives localForceDerivatives = computeLocalForceDerivatives();
  const ConeLocalDerivatives coneLocalDerivatives = computeConeLocalDerivatives(localForce);
  const ConeDerivatives coneDerivatives = computeConeConstraintDerivatives(coneLocalDerivatives, localForceDerivatives);

  VectorFunctionLinearApproximation linearApproximation;
  linearApproximation.f = coneConstraint(localForce);
  linearApproximation.dfdx = matrix_t::Zero(linearApproximation.f.size(), state.size());
  linearApproximation.dfdu = frictionConeInputDerivative(input.size(), coneDerivatives);
  return linearApproximation;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionQuadraticApproximation FrictionForceConeConstraint::getQuadraticApproximation(scalar_t /*time*/,
                                                                                            const vector_t& state,
                                                                                            const vector_t& input,
                                                                                            const PreComputation& /*preComp*/) const {
  const vector3_t contactForce = mpcRobotModelPtr_->getContactForce(input, contactPointIndex_);
  const vector3_t localForce = t_R_w_ * contactForce;

  const LocalForceDerivatives localForceDerivatives = computeLocalForceDerivatives();
  const ConeLocalDerivatives coneLocalDerivatives = computeConeLocalDerivatives(localForce);
  const ConeDerivatives coneDerivatives = computeConeConstraintDerivatives(coneLocalDerivatives, localForceDerivatives);

  VectorFunctionQuadraticApproximation quadraticApproximation;
  quadraticApproximation.f = coneConstraint(localForce);
  quadraticApproximation.dfdx = matrix_t::Zero(quadraticApproximation.f.size(), state.size());
  quadraticApproximation.dfdu = frictionConeInputDerivative(input.size(), coneDerivatives);
  for (const matrix_t& d2Cone_du2 : coneDerivatives.d2Cone_du2) {
    quadraticApproximation.dfdxx.emplace_back(frictionConeSecondDerivativeState(state.size()));
    quadraticApproximation.dfduu.emplace_back(frictionConeSecondDerivativeInput(input.size(), d2Cone_du2));
    quadraticApproximation.dfdux.emplace_back(matrix_t::Zero(input.size(), state.size()));
  }
  return quadraticApproximation;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
FrictionForceConeConstraint::LocalForceDerivatives FrictionForceConeConstraint::computeLocalForceDerivatives() const {
  LocalForceDerivatives localForceDerivatives{};
  // The cone is evaluated on `t_R_w * getContactForce(input)`, so its derivative is that same rotation applied to the
  // model's own force Jacobian - NOT the rotation on its own, which is only the answer when the input happens to
  // store the force directly.
  localForceDerivatives.dF_du.noalias() = t_R_w_ * contactForceInputJacobian_;
  return localForceDerivatives;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
FrictionForceConeConstraint::ConeLocalDerivatives FrictionForceConeConstraint::computeConeLocalDerivatives(
    const vector3_t& localForces) const {
  const scalar_t F_x_square = localForces.x() * localForces.x();
  const scalar_t F_y_square = localForces.y() * localForces.y();
  const scalar_t F_tangent_square = F_x_square + F_y_square + config_.regularization;
  const scalar_t F_tangent_norm = std::sqrt(F_tangent_square);
  const scalar_t F_tangent_square_pow32 = F_tangent_norm * F_tangent_square;  // = F_tangent_square ^ (3/2)

  ConeLocalDerivatives coneDerivatives;
  const Eigen::Index numRows = scheduleGated_ ? 1 : 2;
  coneDerivatives.dCone_dF = matrix_t::Zero(numRows, 3);
  coneDerivatives.d2Cone_dF2.assign(static_cast<size_t>(numRows), matrix3_t::Zero());

  // Row 0, the friction row. Its tangential half, -sqrt(Fx^2 + Fy^2 + regularization), is common to both forms.
  coneDerivatives.dCone_dF(0, 0) = -localForces.x() / F_tangent_norm;
  coneDerivatives.dCone_dF(0, 1) = -localForces.y() / F_tangent_norm;
  matrix3_t& frictionHessian = coneDerivatives.d2Cone_dF2[0];
  frictionHessian(0, 0) = -(F_y_square + config_.regularization) / F_tangent_square_pow32;
  frictionHessian(0, 1) = localForces.x() * localForces.y() / F_tangent_square_pow32;
  frictionHessian(1, 0) = frictionHessian(0, 1);
  frictionHessian(1, 1) = -(F_x_square + config_.regularization) / F_tangent_square_pow32;

  if (scheduleGated_) {
    // mu * (Fz + gripperForce): linear in Fz.
    coneDerivatives.dCone_dF(0, 2) = config_.frictionCoefficient;
  } else {
    // sqrt(mu^2 Fz^2 + regularization): its slope runs from 0 at the unloaded foot to mu under load, and its curvature
    // mu^2 regularization / (mu^2 Fz^2 + regularization)^(3/2) is positive - the reason this form is linearized.
    const scalar_t muSquare = config_.frictionCoefficient * config_.frictionCoefficient;
    const scalar_t F_normal_square = muSquare * localForces.z() * localForces.z() + config_.regularization;
    const scalar_t F_normal_norm = std::sqrt(F_normal_square);
    coneDerivatives.dCone_dF(0, 2) = muSquare * localForces.z() / F_normal_norm;
    frictionHessian(2, 2) = muSquare * config_.regularization / (F_normal_norm * F_normal_square);
    // Row 1, Fz >= 0: linear.
    coneDerivatives.dCone_dF(1, 2) = 1.0;
  }
  return coneDerivatives;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t FrictionForceConeConstraint::coneConstraint(const vector3_t& localForces) const {
  const scalar_t F_tangent_square = localForces.x() * localForces.x() + localForces.y() * localForces.y() + config_.regularization;
  const scalar_t F_tangent_norm = std::sqrt(F_tangent_square);
  if (scheduleGated_) {
    return (vector_t(1) << config_.frictionCoefficient * (localForces.z() + config_.gripperForce) - F_tangent_norm).finished();
  }
  // The exact Coulomb cone, as two rows that are both zero at the zero force; see the class comment for why the gated
  // row with sqrt(regularization) added back - which this term used to evaluate - admitted more friction than mu.
  const scalar_t muFz = config_.frictionCoefficient * localForces.z();
  const scalar_t F_normal_norm = std::sqrt(muFz * muFz + config_.regularization);
  return (vector_t(2) << F_normal_norm - F_tangent_norm, localForces.z()).finished();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
FrictionForceConeConstraint::ConeDerivatives FrictionForceConeConstraint::computeConeConstraintDerivatives(
    const ConeLocalDerivatives& coneLocalDerivatives, const LocalForceDerivatives& localForceDerivatives) const {
  ConeDerivatives coneDerivatives;
  // First order derivatives
  coneDerivatives.dCone_du.noalias() = coneLocalDerivatives.dCone_dF * localForceDerivatives.dF_du;

  // Second order derivatives, one per row
  coneDerivatives.d2Cone_du2.reserve(coneLocalDerivatives.d2Cone_dF2.size());
  for (const matrix3_t& d2Cone_dF2 : coneLocalDerivatives.d2Cone_dF2) {
    coneDerivatives.d2Cone_du2.emplace_back(localForceDerivatives.dF_du.transpose() * d2Cone_dF2 * localForceDerivatives.dF_du);
  }
  return coneDerivatives;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
matrix_t FrictionForceConeConstraint::frictionConeInputDerivative(size_t inputDim, const ConeDerivatives& coneDerivatives) const {
  matrix_t dhdu = matrix_t::Zero(coneDerivatives.dCone_du.rows(), inputDim);
  dhdu.middleCols(static_cast<Eigen::Index>(contactInputStart_), static_cast<Eigen::Index>(contactInputDim_)) = coneDerivatives.dCone_du;
  return dhdu;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
matrix_t FrictionForceConeConstraint::frictionConeSecondDerivativeInput(size_t inputDim, const matrix_t& d2Cone_du2) const {
  matrix_t ddhdudu = matrix_t::Zero(inputDim, inputDim);
  ddhdudu.block(static_cast<Eigen::Index>(contactInputStart_), static_cast<Eigen::Index>(contactInputStart_),
                static_cast<Eigen::Index>(contactInputDim_), static_cast<Eigen::Index>(contactInputDim_)) = d2Cone_du2;
  ddhdudu.diagonal().array() -= config_.hessianDiagonalShift;
  return ddhdudu;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
matrix_t FrictionForceConeConstraint::frictionConeSecondDerivativeState(size_t stateDim) const {
  matrix_t ddhdxdx = matrix_t::Zero(stateDim, stateDim);
  ddhdxdx.diagonal().array() -= config_.hessianDiagonalShift;
  return ddhdxdx;
}

}  // namespace ocs2::humanoid
