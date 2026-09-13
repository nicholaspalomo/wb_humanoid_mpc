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

#pragma once

#include <ocs2_core/constraint/StateInputConstraint.h>
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Forces the contact block of the input to zero while a foot is not in contact.
 *
 * The constrained block is whatever the model parameterizes the contact with: the six wrench components for a
 * wrench-space model, or the basis-vector scalings lambda for BasisInputsModelDecorator. Constraining lambda directly
 * rather than the wrench B * lambda matters: B has only six independent rows, so constraining the wrench would leave
 * the null space of B (numBasisPerFoot - 6 directions) completely free during swing, held down by nothing but the
 * small input regularization. Since the wrench is zero exactly when lambda is zero, constraining the block itself is
 * equivalent, full rank and better conditioned for the equality-constraint projection.
 */
class ZeroWrenchConstraint final : public StateInputConstraint {
 public:
  /*
   * Constructor
   * @param [in] referenceManager : Switched model ReferenceManager.
   * @param [in] contactPointIndex : The 3 DoF contact index.
   */
  ZeroWrenchConstraint(const SwitchedModelReferenceManager& referenceManager,
                       size_t contactPointIndex,
                       const MpcRobotModelBase<scalar_t>& mpcRobotModel);

  ~ZeroWrenchConstraint() override = default;
  ZeroWrenchConstraint* clone() const override { return new ZeroWrenchConstraint(*this); }

  bool isActive(scalar_t time) const override;
  void setActive(bool isActive) override { isActive_ = isActive; }
  bool getActive() const override { return isActive_; }
  size_t getNumConstraints(scalar_t time) const override { return numConstraints_; }
  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const vector_t& input,
                                                           const PreComputation& preComp) const override;

 private:
  ZeroWrenchConstraint(const ZeroWrenchConstraint& rhs);
  const MpcRobotModelBase<scalar_t>* mpcRobotModelPtr_;
  const SwitchedModelReferenceManager* referenceManagerPtr_;
  const size_t contactPointIndex_;
  size_t contactBlockStart_;  ///< first input index of this contact's block
  size_t numConstraints_;     ///< size of that block: 6 wrench components, or numBasisPerFoot scalings
  bool isActive_ = true;
};

}  // namespace ocs2::humanoid
