/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"

#include <cmath>

#include "absl/log/check.h"

namespace ocs2::humanoid {

vector_t normalContactForceRow(const MpcRobotModelBase<scalar_t>& mpcRobotModel, size_t contactPointIndex) {
  // The contact force is linear in the input for every parameterization the model offers (the wrench block itself, or
  // the basis scalings of BasisInputsModelDecorator), so the row is read off once by probing the unit inputs of this
  // contact's block. Every other input, the joint velocities and the other foot, probes to zero.
  const size_t inputDim = mpcRobotModel.getInputDim();
  vector_t row = vector_t::Zero(inputDim);
  const vector_t zeroInput = vector_t::Zero(inputDim);
  const scalar_t offset = mpcRobotModel.getContactForce(zeroInput, contactPointIndex)(2);
  CHECK_LE(std::abs(offset), 1e-12) << "[ContactComplementarityConstraint] the contact force must be linear in the input";
  for (size_t index = 0; index < inputDim; ++index) {
    vector_t probe = vector_t::Zero(inputDim);
    probe(index) = 1.0;
    row(index) = mpcRobotModel.getContactForce(probe, contactPointIndex)(2);
  }
  return row;
}

ContactComplementarityConstraint::ContactComplementarityConstraint(const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
                                                                   const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                   size_t contactPointIndex,
                                                                   scalar_t terrainHeight)
    : StateInputConstraint(ConstraintOrder::Linear),
      endEffectorKinematicsPtr_(endEffectorKinematics.clone()),
      contactPointIndex_(contactPointIndex),
      terrainHeight_(terrainHeight),
      normalForceRow_(normalContactForceRow(mpcRobotModel, contactPointIndex)) {
  CHECK_EQ(endEffectorKinematicsPtr_->getIds().size(), 1U)
      << "[ContactComplementarityConstraint] expects exactly one end-effector, the contact frame of this foot";
}

ContactComplementarityConstraint::ContactComplementarityConstraint(const ContactComplementarityConstraint& rhs)
    : StateInputConstraint(rhs),
      endEffectorKinematicsPtr_(rhs.endEffectorKinematicsPtr_->clone()),
      contactPointIndex_(rhs.contactPointIndex_),
      terrainHeight_(rhs.terrainHeight_),
      normalForceRow_(rhs.normalForceRow_) {}

vector_t ContactComplementarityConstraint::getValue(scalar_t time,
                                                    const vector_t& state,
                                                    const vector_t& input,
                                                    const PreComputation& preComp) const {
  const scalar_t height = endEffectorKinematicsPtr_->getPosition(state).front()(2) - terrainHeight_;
  const scalar_t normalForce = normalForceRow_.dot(input);
  return (vector_t(1) << normalForce * height).finished();
}

VectorFunctionLinearApproximation ContactComplementarityConstraint::getLinearApproximation(scalar_t time,
                                                                                           const vector_t& state,
                                                                                           const vector_t& input,
                                                                                           const PreComputation& preComp) const {
  const VectorFunctionLinearApproximation position = endEffectorKinematicsPtr_->getPositionLinearApproximation(state).front();
  const scalar_t height = position.f(2) - terrainHeight_;
  const scalar_t normalForce = normalForceRow_.dot(input);

  VectorFunctionLinearApproximation approximation;
  approximation.f = (vector_t(1) << normalForce * height).finished();
  approximation.dfdx = normalForce * position.dfdx.row(2);
  approximation.dfdu = height * normalForceRow_.transpose();
  return approximation;
}

}  // namespace ocs2::humanoid
