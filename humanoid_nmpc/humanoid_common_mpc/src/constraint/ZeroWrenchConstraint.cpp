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
      mpcRobotModelPtr_(&mpcRobotModel) {
  // The contact block holds six wrench components for a wrench-space model and numBasisPerFoot scalings for a
  // basis-vector model; the model reports its own size rather than it being inferred from the input layout.
  contactBlockStart_ = mpcRobotModel.getContactWrenchStartIndices(contactPointIndex);
  numConstraints_ = mpcRobotModel.getContactInputDim(contactPointIndex);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ZeroWrenchConstraint::ZeroWrenchConstraint(const ZeroWrenchConstraint& rhs)
    : StateInputConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      contactPointIndex_(rhs.contactPointIndex_),
      // The model is borrowed, exactly as in the primary constructor. Cloning here would leak, since the member is a
      // raw non-owning pointer; the model outlives every constraint and its accessors are const and thread safe.
      mpcRobotModelPtr_(rhs.mpcRobotModelPtr_),
      contactBlockStart_(rhs.contactBlockStart_),
      numConstraints_(rhs.numConstraints_),
      isActive_(rhs.isActive_) {}

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
  // The contact block itself, not the wrench reconstructed from it: see the class documentation.
  return input.segment(contactBlockStart_, numConstraints_);
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
  approx.dfdx = matrix_t::Zero(numConstraints_, mpcRobotModelPtr_->getStateDim());
  approx.dfdu = matrix_t::Zero(numConstraints_, mpcRobotModelPtr_->getInputDim());
  approx.dfdu.middleCols(contactBlockStart_, numConstraints_).setIdentity();
  return approx;
}

}  // namespace ocs2::humanoid
