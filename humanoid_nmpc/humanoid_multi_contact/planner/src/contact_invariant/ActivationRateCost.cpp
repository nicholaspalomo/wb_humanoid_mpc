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

#include "humanoid_multi_contact/planner/contact_invariant/ActivationRateCost.h"

namespace ocs2::humanoid::multi_contact {

ActivationRateCost::ActivationRateCost(const ContactInvariantLayout& layout,
                                       scalar_t weight)
    : layout_(layout), weight_(weight) {}

scalar_t ActivationRateCost::getValue(
    scalar_t /*time*/,
    const vector_t& /*state*/,
    const vector_t& input,
    const TargetTrajectories& /*targetTrajectories*/,
    const PreComputation& /*preComp*/) const {
  return weight_ * input.tail(layout_.numCandidates()).squaredNorm();
}

ScalarFunctionQuadraticApproximation
ActivationRateCost::getQuadraticApproximation(
    scalar_t /*time*/,
    const vector_t& /*state*/,
    const vector_t& input,
    const TargetTrajectories& /*targetTrajectories*/,
    const PreComputation& /*preComp*/) const {
  const int k = layout_.numCandidates();
  ScalarFunctionQuadraticApproximation result;
  result.setZero(layout_.stateDim(), layout_.inputDim());
  result.f = weight_ * input.tail(k).squaredNorm();
  result.dfdu.tail(k) = 2.0 * weight_ * input.tail(k);
  result.dfduu.bottomRightCorner(k, k).diagonal().setConstant(2.0 * weight_);
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
