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

#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"

#include "absl/log/check.h"

namespace ocs2::humanoid {

GroundPenetrationConstraint::GroundPenetrationConstraint(const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
                                                         scalar_t terrainHeight)
    : StateConstraint(ConstraintOrder::Linear), endEffectorKinematicsPtr_(endEffectorKinematics.clone()), terrainHeight_(terrainHeight) {
  CHECK_EQ(endEffectorKinematicsPtr_->getIds().size(), 1U)
      << "[GroundPenetrationConstraint] expects exactly one end-effector, the contact frame of this foot";
}

GroundPenetrationConstraint::GroundPenetrationConstraint(const GroundPenetrationConstraint& rhs)
    : StateConstraint(rhs), endEffectorKinematicsPtr_(rhs.endEffectorKinematicsPtr_->clone()), terrainHeight_(rhs.terrainHeight_) {}

vector_t GroundPenetrationConstraint::getValue(scalar_t time, const vector_t& state, const PreComputation& preComp) const {
  const scalar_t height = endEffectorKinematicsPtr_->getPosition(state).front()(2) - terrainHeight_;
  return (vector_t(1) << height).finished();
}

VectorFunctionLinearApproximation GroundPenetrationConstraint::getLinearApproximation(scalar_t time,
                                                                                      const vector_t& state,
                                                                                      const PreComputation& preComp) const {
  const VectorFunctionLinearApproximation position = endEffectorKinematicsPtr_->getPositionLinearApproximation(state).front();
  VectorFunctionLinearApproximation approximation;
  approximation.f = (vector_t(1) << position.f(2) - terrainHeight_).finished();
  approximation.dfdx = position.dfdx.row(2);
  return approximation;
}

}  // namespace ocs2::humanoid
