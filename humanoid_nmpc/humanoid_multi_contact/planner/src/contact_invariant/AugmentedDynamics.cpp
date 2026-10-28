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

#include "humanoid_multi_contact/planner/contact_invariant/AugmentedDynamics.h"

#include <memory>
#include <utility>

#include "absl/log/absl_check.h"

namespace ocs2::humanoid::multi_contact {

AugmentedDynamics::AugmentedDynamics(
    const ContactInvariantLayout& layout,
    std::unique_ptr<SystemDynamicsBase> physical)
    : layout_(layout), physical_(std::move(physical)) {
  ABSL_CHECK(physical_ != nullptr);
}

AugmentedDynamics::AugmentedDynamics(const AugmentedDynamics& other)
    : SystemDynamicsBase(other),
      layout_(other.layout_),
      physical_(other.physical_->clone()) {}

vector_t AugmentedDynamics::computeFlowMap(scalar_t time,
                                           const vector_t& state,
                                           const vector_t& input,
                                           const PreComputation& /*preComp*/) {
  vector_t flow(layout_.stateDim());
  flow.head(layout_.physicalStateDim) =
      physical_->computeFlowMap(time, state.head(layout_.physicalStateDim),
                                input.head(layout_.physicalInputDim));
  flow.tail(layout_.numCandidates()) = input.tail(layout_.numCandidates());
  return flow;
}

VectorFunctionLinearApproximation AugmentedDynamics::linearApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) {
  const VectorFunctionLinearApproximation physical =
      physical_->linearApproximation(time, state.head(layout_.physicalStateDim),
                                     input.head(layout_.physicalInputDim));
  const int n = layout_.physicalStateDim;
  const int m = layout_.physicalInputDim;
  const int k = layout_.numCandidates();
  VectorFunctionLinearApproximation result;
  result.setZero(layout_.stateDim(), layout_.stateDim(), layout_.inputDim());
  result.f.head(n) = physical.f;
  result.f.tail(k) = input.tail(k);
  result.dfdx.topLeftCorner(n, n) = physical.dfdx;
  result.dfdu.topLeftCorner(n, m) = physical.dfdu;
  result.dfdu.bottomRightCorner(k, k).setIdentity();
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
