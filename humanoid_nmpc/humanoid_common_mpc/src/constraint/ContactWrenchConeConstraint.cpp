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

#include <pinocchio/fwd.hpp>

#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/PinocchioFrameConversions.h"

#include <cmath>

namespace ocs2::humanoid {

namespace {

static constexpr size_t kExtraConstraintsCount = 7;
static constexpr size_t kWrenchForceDim = 3;
static constexpr size_t kWrenchMomentDim = 3;

static constexpr size_t kForceXIdx = 0;
static constexpr size_t kForceYIdx = 1;
static constexpr size_t kForceZIdx = 2;

static constexpr size_t kMomentXIdx = 0;
static constexpr size_t kMomentYIdx = 1;
static constexpr size_t kMomentZIdx = 2;

static constexpr scalar_t kPatchOffsetZeroThreshold = 1e-9;
static constexpr scalar_t kHalf = 0.5;

}  // namespace

ContactWrenchConeConstraint::ContactWrenchConeConstraint(const SwitchedModelReferenceManager& referenceManager,
                                                         const ContactRectangle& contactRectangle,
                                                         size_t contactPointIndex,
                                                         const PinocchioInterface& pinocchioInterface,
                                                         const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                         Config config)
    : StateInputConstraint(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      pinocchioInterfacePtr_(&pinocchioInterface),
      mpcRobotModelPtr_(mpcRobotModel.clone()),
      contactRectangle_(contactRectangle),
      contactPointIndex_(contactPointIndex),
      config_(std::move(config)) {
  initializeLocalConstraintMatrix();
}

ContactWrenchConeConstraint::ContactWrenchConeConstraint(const ContactWrenchConeConstraint& other)
    : StateInputConstraint(other),
      referenceManagerPtr_(other.referenceManagerPtr_),
      pinocchioInterfacePtr_(other.pinocchioInterfacePtr_),
      mpcRobotModelPtr_(other.mpcRobotModelPtr_->clone()),
      contactRectangle_(other.contactRectangle_),
      contactPointIndex_(other.contactPointIndex_),
      config_(other.config_),
      numConstraints_(other.numConstraints_),
      isActive_(other.isActive_),
      A_f_local_(other.A_f_local_),
      A_tau_local_(other.A_tau_local_),
      b_local_(other.b_local_) {}

bool ContactWrenchConeConstraint::isActive(scalar_t time) const {
  if (!isActive_) {
    return false;
  }
  return referenceManagerPtr_->getContactFlags(time)[contactPointIndex_];
}

void ContactWrenchConeConstraint::initializeLocalConstraintMatrix() {
  numConstraints_ = config_.numBasisVectors + kExtraConstraintsCount;
  A_f_local_ = matrix_t::Zero(numConstraints_, kWrenchForceDim);
  A_tau_local_ = matrix_t::Zero(numConstraints_, kWrenchMomentDim);
  b_local_ = vector_t::Zero(numConstraints_);

  size_t constraintIdx = 0;
  const scalar_t effectiveGripperNormal = config_.frictionCoefficient * config_.gripperForce;

  // 1. Friction cone approximation with numBasisVectors basis vectors:
  // mu * (Fz + F_grip) - (cos(theta_k) * Fx + sin(theta_k) * Fy) >= 0
  const size_t numBasisVectors = config_.numBasisVectors;
  const scalar_t angleStep = 2.0 * M_PI / static_cast<scalar_t>(numBasisVectors);
  for (size_t k = 0; k < numBasisVectors; ++k) {
    const scalar_t theta_k = k * angleStep;
    const scalar_t cos_k = std::cos(theta_k);
    const scalar_t sin_k = std::sin(theta_k);
    A_f_local_(constraintIdx, kForceXIdx) = -cos_k;
    A_f_local_(constraintIdx, kForceYIdx) = -sin_k;
    A_f_local_(constraintIdx, kForceZIdx) = config_.frictionCoefficient;
    b_local_(constraintIdx) = effectiveGripperNormal;
    constraintIdx++;
  }

  // 2. Normal force limit: Fz - minNormalForce >= 0
  A_f_local_(constraintIdx, kForceZIdx) = 1.0;
  b_local_(constraintIdx) = -config_.minNormalForce;
  constraintIdx++;

  // 3. Center of Pressure (CoP) / Moment constraints (Bounds relative to local foot contact frame)
  const PolygonBounds& bounds = contactRectangle_.getBounds();
  // tau_x - y_min * Fz >= 0
  A_f_local_(constraintIdx, kForceZIdx) = -bounds.y_min;
  A_tau_local_(constraintIdx, kMomentXIdx) = 1.0;
  constraintIdx++;

  // -tau_x + y_max * Fz >= 0
  A_f_local_(constraintIdx, kForceZIdx) = bounds.y_max;
  A_tau_local_(constraintIdx, kMomentXIdx) = -1.0;
  constraintIdx++;

  // -tau_y - x_min * Fz >= 0
  A_f_local_(constraintIdx, kForceZIdx) = -bounds.x_min;
  A_tau_local_(constraintIdx, kMomentYIdx) = -1.0;
  constraintIdx++;

  // tau_y + x_max * Fz >= 0
  A_f_local_(constraintIdx, kForceZIdx) = bounds.x_max;
  A_tau_local_(constraintIdx, kMomentYIdx) = 1.0;
  constraintIdx++;

  // 4. Torsional yaw friction moment about the contact patch center
  vector3_t offset;
  if (config_.patchOffset.isZero(kPatchOffsetZeroThreshold)) {
    offset = vector3_t(kHalf * (bounds.x_min + bounds.x_max), kHalf * (bounds.y_min + bounds.y_max), 0.0);
  } else {
    offset = config_.patchOffset;
  }

  const scalar_t effectiveTorsionalGripper = config_.torsionalFrictionCoefficient * config_.gripperForce;

  // mu_torsion * (Fz + F_grip) + tau_patch_z >= 0
  // tau_patch_z = tau_z - (offset.x * Fy - offset.y * Fx) = offset.y * Fx - offset.x * Fy + tau_z
  A_f_local_(constraintIdx, kForceXIdx) = offset.y();
  A_f_local_(constraintIdx, kForceYIdx) = -offset.x();
  A_f_local_(constraintIdx, kForceZIdx) = config_.torsionalFrictionCoefficient;
  A_tau_local_(constraintIdx, kMomentZIdx) = 1.0;
  b_local_(constraintIdx) = effectiveTorsionalGripper;
  constraintIdx++;

  // mu_torsion * (Fz + F_grip) - tau_patch_z >= 0
  // -tau_patch_z = -offset.y * Fx + offset.x * Fy - tau_z
  A_f_local_(constraintIdx, kForceXIdx) = -offset.y();
  A_f_local_(constraintIdx, kForceYIdx) = offset.x();
  A_f_local_(constraintIdx, kForceZIdx) = config_.torsionalFrictionCoefficient;
  A_tau_local_(constraintIdx, kMomentZIdx) = -1.0;
  b_local_(constraintIdx) = effectiveTorsionalGripper;
  constraintIdx++;
}

vector_t ContactWrenchConeConstraint::getValue(scalar_t time,
                                               const vector_t& state,
                                               const vector_t& input,
                                               const PreComputation& preComp) const {
  const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
  pinocchio::Data data = pinocchioInterfacePtr_->getData();
  updateFramePlacements(mpcRobotModelPtr_->getGeneralizedCoordinates(state), model, data);
  const pinocchio::FrameIndex frameID = getContactFrameIndex(*pinocchioInterfacePtr_, *mpcRobotModelPtr_, contactPointIndex_);

  const matrix3_t w_R_l = getRotationMatrixLocalToWorld(data, frameID);
  const matrix3_t l_R_w = w_R_l.transpose();

  const vector3_t forceInWorld = mpcRobotModelPtr_->getContactForce(input, contactPointIndex_);
  const vector3_t momentInWorld = mpcRobotModelPtr_->getContactMoment(input, contactPointIndex_);

  const vector3_t forceInLocal = l_R_w * forceInWorld;
  const vector3_t momentInLocal = l_R_w * momentInWorld;

  return A_f_local_ * forceInLocal + A_tau_local_ * momentInLocal + b_local_;
}

VectorFunctionLinearApproximation ContactWrenchConeConstraint::getLinearApproximation(scalar_t time,
                                                                                      const vector_t& state,
                                                                                      const vector_t& input,
                                                                                      const PreComputation& preComp) const {
  const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
  pinocchio::Data data = pinocchioInterfacePtr_->getData();
  updateFramePlacements(mpcRobotModelPtr_->getGeneralizedCoordinates(state), model, data);
  const pinocchio::FrameIndex frameID = getContactFrameIndex(*pinocchioInterfacePtr_, *mpcRobotModelPtr_, contactPointIndex_);

  const matrix3_t w_R_l = getRotationMatrixLocalToWorld(data, frameID);
  const matrix3_t l_R_w = w_R_l.transpose();

  const matrix_t A_f_world = A_f_local_ * l_R_w;
  const matrix_t A_tau_world = A_tau_local_ * l_R_w;

  const vector3_t forceInWorld = mpcRobotModelPtr_->getContactForce(input, contactPointIndex_);
  const vector3_t momentInWorld = mpcRobotModelPtr_->getContactMoment(input, contactPointIndex_);

  VectorFunctionLinearApproximation linearApproximation;
  linearApproximation.f = A_f_world * forceInWorld + A_tau_world * momentInWorld + b_local_;
  linearApproximation.dfdx = matrix_t::Zero(numConstraints_, mpcRobotModelPtr_->getStateDim());
  linearApproximation.dfdu = matrix_t::Zero(numConstraints_, mpcRobotModelPtr_->getInputDim());

  const size_t forceStartIdx = mpcRobotModelPtr_->getContactForceStartIndices(contactPointIndex_);
  const size_t momentStartIdx = mpcRobotModelPtr_->getContactMomentStartIndices(contactPointIndex_);

  linearApproximation.dfdu.block(0, forceStartIdx, numConstraints_, kWrenchForceDim) = A_f_world;
  linearApproximation.dfdu.block(0, momentStartIdx, numConstraints_, kWrenchMomentDim) = A_tau_world;

  return linearApproximation;
}

VectorFunctionQuadraticApproximation ContactWrenchConeConstraint::getQuadraticApproximation(scalar_t time,
                                                                                            const vector_t& state,
                                                                                            const vector_t& input,
                                                                                            const PreComputation& preComp) const {
  const VectorFunctionLinearApproximation linearApprox = getLinearApproximation(time, state, input, preComp);
  VectorFunctionQuadraticApproximation quadraticApproximation;
  quadraticApproximation.f = linearApprox.f;
  quadraticApproximation.dfdx = linearApprox.dfdx;
  quadraticApproximation.dfdu = linearApprox.dfdu;
  quadraticApproximation.dfdxx.resize(numConstraints_, matrix_t::Zero(mpcRobotModelPtr_->getStateDim(), mpcRobotModelPtr_->getStateDim()));
  quadraticApproximation.dfduu.resize(numConstraints_, matrix_t::Zero(mpcRobotModelPtr_->getInputDim(), mpcRobotModelPtr_->getInputDim()));
  quadraticApproximation.dfdux.resize(numConstraints_, matrix_t::Zero(mpcRobotModelPtr_->getInputDim(), mpcRobotModelPtr_->getStateDim()));
  return quadraticApproximation;
}

}  // namespace ocs2::humanoid
