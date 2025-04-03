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

constexpr scalar_t kForceSize = 3;

FrictionForceConeLinearConstraint::FrictionForceConeLinearConstraint(
    const SwitchedModelReferenceManager& referenceManager, const Config& config,
    size_t contactPointIndex, const MpcRobotModelBase<scalar_t>& mpcRobotModel)
    : StateInputConstraint(ConstraintOrder::Quadratic),
      referenceManagerPtr_(&referenceManager),
      config_(config),
      mpcRobotModelPtr_(&mpcRobotModel),
      contactPointIndex_(contactPointIndex) {}
{}

FrictionForceConeLinearConstraint::FrictionForceConeLinearConstraint(
    const FrictionForceConeLinearConstraint& rhs)
    : StateInputConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      config_(rhs.config_),
      mpcRobotModelPtr_(rhs.mpcRobotModelPtr_),
      contactPointIndex_(rhs.contactPointIndex_) {}

bool FrictionForceConeLinearConstraint::isActive(scalar_t time) const {
  return referenceManagerPtr_->getContactFlags(time)[contactPointIndex_];
}

size_t FrictionForceConeLinearConstraint::getNumConstraints(
    scalar_t time) const {
  return config_.numBasisVectors;
}

vector_t FrictionForceConeLinearConstraint::getValue(
    scalar_t time, const vector_t& state, const vector_t& input,
    const PreComputation& preComp) const {
  const vector3_t& forcesInWorldFrame =
      mpcRobotModelPtr_->getContactForce(input, contactPointIndex_);
  const matrix3_t& t_R_w =
      pinocchioInterfacePtr_->getRotationMatrixLocalToWorld(
          preComp.pinocchioData,
          contactPointIndex_);  // TODO: Pass in precomputation callback to
                                // class constructor
  const vector3_t& localForce = t_R_w * forcesInWorldFrame;

  size_t numBasisVectors = config_.numBasisVectors;
  matrix_t basisVectors(numBasisVectors, kForceSize);
  basisVectors.setZero();
  for (int i = 0; i < numBasisVectors; i++) {
    scalar_t theta = i * 2 * M_PI / numBasisVectors;
    basisVectors(i, 0) = cos(theta);
    basisVectors(i, 1) = sin(theta);
    basisVectors(i, 2) = -config_.frictionCoefficient;
  }

  return basisVectors * localForce;
}

VectorFunctionLinearApproximation
FrictionForceConeLinearConstraint::getLinearApproximation(
    scalar_t time, const vector_t& state, const vector_t& input,
    const PreComputation& preComp) const {
  numBasisVectors = config_.numBasisVectors;
  VectorFunctionLinearApproximation linearApproximation;
  linearApproximation.f = getValue(time, state, input, preComp);
  linearApproximation.dfdx = matrix_t::Zero(numBasisVectors, state.size());
  const matrix3_t& t_R_w =
      pinocchioInterfacePtr_->getRotationMatrixLocalToWorld(
          preComp.pinocchioData,
          contactPointIndex_);  // TODO: Pass in precomputation callback to
                                // class constructor
  linearApproximation.dfdu.setZero(numBasisVectors, input.size());
  linearApproximation.dfdu.middleCols(
      mpcRobotModelPtr_->getContactForceStartIndices(contactPointIndex_),
      kForceSize) = basisVectors * t_R_w;
  return linearApproximation;
}

}  // namespace ocs2::humanoid
