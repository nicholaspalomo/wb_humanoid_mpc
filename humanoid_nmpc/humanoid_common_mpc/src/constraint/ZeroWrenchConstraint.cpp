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

#include "humanoid_common_mpc/constraint/ZeroWrenchConstraint.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ZeroWrenchConstraint::ZeroWrenchConstraint(const SwitchedModelReferenceManager& referenceManager,
                                           size_t contactPointIndex,
                                           const MpcRobotModelBase<scalar_t>& mpcRobotModel)
    : StateInputConstraint(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      contactPointIndex_(contactPointIndex),
      mpcRobotModelPtr_(&mpcRobotModel) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ZeroWrenchConstraint::ZeroWrenchConstraint(const ZeroWrenchConstraint& rhs)
    : StateInputConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      contactPointIndex_(rhs.contactPointIndex_),
      mpcRobotModelPtr_(rhs.mpcRobotModelPtr_->clone()) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool ZeroWrenchConstraint::isActive(scalar_t time) const {
  if (!isActive_) return false;
  return !referenceManagerPtr_->getContactFlags(time)[contactPointIndex_];
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t ZeroWrenchConstraint::getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const {
  return mpcRobotModelPtr_->getContactWrench(input, contactPointIndex_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation ZeroWrenchConstraint::getLinearApproximation(scalar_t time,
                                                                               const vector_t& state,
                                                                               const vector_t& input,
                                                                               const PreComputation& preComp) const {
  VectorFunctionLinearApproximation approx;
  approx.f = getValue(time, state, input, preComp);
  approx.dfdx = matrix_t::Zero(n_constraints, mpcRobotModelPtr_->getStateDim());
  approx.dfdu = matrix_t::Zero(n_constraints, mpcRobotModelPtr_->getInputDim());

  // Use the model's start index — this correctly handles both wrench-space
  // models (identity Jacobian at wrench columns) and basis-vector models
  // (B matrix at λ columns).
  const size_t colStart = mpcRobotModelPtr_->getContactWrenchStartIndices(contactPointIndex_);
  const size_t nextBlockStart = (contactPointIndex_ + 1 < N_CONTACTS)
                                    ? mpcRobotModelPtr_->getContactWrenchStartIndices(contactPointIndex_ + 1)
                                    : mpcRobotModelPtr_->getJointVelocitiesStartindex();
  const size_t colSpan = nextBlockStart - colStart;

  // For wrench-space: colSpan == 6, and the Jacobian is identity.
  // For basis-vector: colSpan == numBasisPerFoot, and the Jacobian is B.
  if (colSpan == static_cast<size_t>(n_constraints)) {
    // Wrench-space model: d(wrench)/d(wrench_input) = I
    approx.dfdu.middleCols(colStart, colSpan).setIdentity();
  } else {
    // Basis-vector model: d(B * λ)/d(λ) = B — compute via finite difference of getContactWrench
    // Each basis scalar λ_j contributes column B[:, j] to the Jacobian.
    for (size_t j = 0; j < colSpan; ++j) {
      vector_t perturbedInput = vector_t::Zero(mpcRobotModelPtr_->getInputDim());
      perturbedInput(colStart + j) = 1.0;
      approx.dfdu.col(colStart + j) = mpcRobotModelPtr_->getContactWrench(perturbedInput, contactPointIndex_);
    }
  }

  return approx;
}

}  // namespace ocs2::humanoid
