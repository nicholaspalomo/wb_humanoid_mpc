/******************************************************************************
Copyright (c) 2025, Nicholas Palomo and Manuel Yves Galliker. All rights
reserved.

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

#include "humanoid_common_mpc/constraint/FrictionForceConeLinearConstraint.h"

namespace ocs2::humanoid {

constexpr size_t kForceSize = 3;

FrictionForceConeLinearConstraint::FrictionForceConeLinearConstraint(
    const SwitchedModelReferenceManager& referenceManager, const Config& config,
    size_t contactPointIndex, const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    PreComputationCallback callback)
    : StateInputConstraint(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      config_(config),
      mpcRobotModelPtr_(&mpcRobotModel),
      contactPointIndex_(contactPointIndex),
      preCompCallback_(callback) {}

bool FrictionForceConeLinearConstraint::isActive(scalar_t time) const {
  return referenceManagerPtr_->getContactFlags(time)[contactPointIndex_];
}

size_t FrictionForceConeLinearConstraint::getNumConstraints(scalar_t) const {
  return config_.numBasisVectors + 1;
}

matrix_t FrictionForceConeLinearConstraint::getBasisVectors(
    scalar_t time) const {
  size_t numBasisVectors = config_.numBasisVectors;
  matrix_t basisVectors(getNumConstraints(time), kForceSize);
  basisVectors.setZero();
  for (size_t i = 0; i < numBasisVectors; i++) {
    scalar_t theta = i * 2 * M_PI / numBasisVectors;
    basisVectors(i, 0) = cos(theta);
    basisVectors(i, 1) = sin(theta);
    basisVectors(i, 2) = -config_.frictionCoefficient;
  }

  basisVectors(numBasisVectors, 2) = 1.0;
  return basisVectors;
}

vector_t FrictionForceConeLinearConstraint::getValue(
    scalar_t time, const vector_t& state, const vector_t& input,
    const PreComputation& preComp) const {
  matrix_t basisVectors = getBasisVectors(time);
  const vector3_t& forcesInWorldFrame =
      mpcRobotModelPtr_->getContactForce(input, contactPointIndex_);
  const matrix3_t& t_R_w = preCompCallback_(state, input, preComp);
  const vector3_t& localForce = t_R_w * forcesInWorldFrame;
  vector_t f(basisVectors.rows());
  f.setZero();
  size_t numBasisVectors = config_.numBasisVectors;
  f(numBasisVectors) = -config_.minimumNormalForce;
  return f - basisVectors * localForce;
}

VectorFunctionLinearApproximation
FrictionForceConeLinearConstraint::getLinearApproximation(
    scalar_t time, const vector_t& state, const vector_t& input,
    const PreComputation& preComp) const {
  size_t numConstraints = getNumConstraints(time);
  VectorFunctionLinearApproximation linearApproximation =
      VectorFunctionLinearApproximation::Zero(numConstraints, state.size(),
                                              input.size());
  linearApproximation.f = getValue(time, state, input, preComp);
  const matrix3_t& t_R_w = preCompCallback_(state, input, preComp);
  linearApproximation.dfdu.block(
      0, mpcRobotModelPtr_->getContactForceStartIndices(contactPointIndex_),
      numConstraints, kForceSize) = -getBasisVectors(time) * t_R_w;
  return linearApproximation;
}

}  // namespace ocs2::humanoid
