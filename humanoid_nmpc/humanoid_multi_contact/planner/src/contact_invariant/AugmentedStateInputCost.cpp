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

#include "humanoid_multi_contact/planner/contact_invariant/AugmentedStateInputCost.h"

#include <memory>
#include <utility>

#include "absl/log/absl_check.h"
#include "ocs2_core/PreComputation.h"

#include "humanoid_multi_contact/planner/contact_invariant/AugmentedPreComputation.h"

namespace ocs2::humanoid::multi_contact {

AugmentedStateInputCost::AugmentedStateInputCost(
    const ContactInvariantLayout& layout,
    std::unique_ptr<StateInputCost> physical)
    : layout_(layout), physical_(std::move(physical)) {
  ABSL_CHECK(physical_ != nullptr);
}

AugmentedStateInputCost::AugmentedStateInputCost(
    const AugmentedStateInputCost& other)
    : StateInputCost(other),
      layout_(other.layout_),
      physical_(other.physical_->clone()) {}

scalar_t AugmentedStateInputCost::getValue(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const TargetTrajectories& targetTrajectories,
    const PreComputation& preComp) const {
  const AugmentedPreComputation& augmented =
      cast<AugmentedPreComputation>(preComp);
  return physical_->getValue(time, state.head(layout_.physicalStateDim),
                             input.head(layout_.physicalInputDim),
                             targetTrajectories, augmented.physical());
}

ScalarFunctionQuadraticApproximation
AugmentedStateInputCost::getQuadraticApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const TargetTrajectories& targetTrajectories,
    const PreComputation& preComp) const {
  const AugmentedPreComputation& augmented =
      cast<AugmentedPreComputation>(preComp);
  const int n = layout_.physicalStateDim;
  const int m = layout_.physicalInputDim;
  const ScalarFunctionQuadraticApproximation physical =
      physical_->getQuadraticApproximation(time, state.head(n), input.head(m),
                                           targetTrajectories,
                                           augmented.physical());
  ScalarFunctionQuadraticApproximation result;
  result.setZero(layout_.stateDim(), layout_.inputDim());
  result.f = physical.f;
  result.dfdx.head(n) = physical.dfdx;
  result.dfdu.head(m) = physical.dfdu;
  result.dfdxx.topLeftCorner(n, n) = physical.dfdxx;
  result.dfduu.topLeftCorner(m, m) = physical.dfduu;
  result.dfdux.topLeftCorner(m, n) = physical.dfdux;
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
