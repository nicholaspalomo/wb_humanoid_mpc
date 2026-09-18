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

#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"

#include "absl/log/check.h"
#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"

namespace ocs2::humanoid {

ForceWeightedSlipConstraint::ForceWeightedSlipConstraint(const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
                                                         const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                         size_t contactPointIndex)
    : StateInputConstraint(ConstraintOrder::Linear),
      endEffectorKinematicsPtr_(endEffectorKinematics.clone()),
      contactPointIndex_(contactPointIndex),
      normalForceRow_(normalContactForceRow(mpcRobotModel, contactPointIndex)) {
  CHECK_EQ(endEffectorKinematicsPtr_->getIds().size(), 1U)
      << "[ForceWeightedSlipConstraint] expects exactly one end-effector, the contact frame of this foot";
}

ForceWeightedSlipConstraint::ForceWeightedSlipConstraint(const ForceWeightedSlipConstraint& rhs)
    : StateInputConstraint(rhs),
      endEffectorKinematicsPtr_(rhs.endEffectorKinematicsPtr_->clone()),
      contactPointIndex_(rhs.contactPointIndex_),
      normalForceRow_(rhs.normalForceRow_) {}

vector_t ForceWeightedSlipConstraint::getValue(scalar_t time,
                                               const vector_t& state,
                                               const vector_t& input,
                                               const PreComputation& preComp) const {
  const vector3_t velocity = endEffectorKinematicsPtr_->getVelocity(state, input).front();
  const vector3_t angularVelocity = endEffectorKinematicsPtr_->getAngularVelocity(state, input).front();
  const scalar_t normalForce = normalForceRow_.dot(input);
  vector_t value(3);
  value << normalForce * velocity.head<2>(), normalForce * angularVelocity(2);
  return value;
}

VectorFunctionLinearApproximation ForceWeightedSlipConstraint::getLinearApproximation(scalar_t time,
                                                                                      const vector_t& state,
                                                                                      const vector_t& input,
                                                                                      const PreComputation& preComp) const {
  const VectorFunctionLinearApproximation velocity = endEffectorKinematicsPtr_->getVelocityLinearApproximation(state, input).front();
  const VectorFunctionLinearApproximation angularVelocity =
      endEffectorKinematicsPtr_->getAngularVelocityLinearApproximation(state, input).front();
  const scalar_t normalForce = normalForceRow_.dot(input);

  // The constrained components of the twist: the two tangential linear velocities and the spin about the normal.
  vector_t twist(3);
  twist << velocity.f.head<2>(), angularVelocity.f(2);
  matrix_t dTwistdx(3, velocity.dfdx.cols());
  dTwistdx << velocity.dfdx.topRows<2>(), angularVelocity.dfdx.row(2);
  matrix_t dTwistdu(3, velocity.dfdu.cols());
  dTwistdu << velocity.dfdu.topRows<2>(), angularVelocity.dfdu.row(2);

  VectorFunctionLinearApproximation approximation;
  approximation.f = normalForce * twist;
  approximation.dfdx = normalForce * dTwistdx;
  // d(f_n v) / du = f_n dv/du + v (df_n/du), the second term being the constant normal-force row.
  approximation.dfdu = normalForce * dTwistdu + twist * normalForceRow_.transpose();
  return approximation;
}

}  // namespace ocs2::humanoid
