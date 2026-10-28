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

#include "humanoid_multi_contact/planner/contact_invariant/ActivationBoundsConstraint.h"

#include <cstddef>

namespace ocs2::humanoid::multi_contact {
namespace {

// A lower and an upper bound per contact variable.
constexpr int kBoundsPerVariable = 2;

}  // namespace

ActivationBoundsConstraint::ActivationBoundsConstraint(
    const ContactInvariantLayout& layout, scalar_t minimumSupport)
    : StateConstraint(ConstraintOrder::Linear),
      layout_(layout),
      minimumSupport_(minimumSupport) {}

size_t ActivationBoundsConstraint::getNumConstraints(scalar_t /*time*/) const {
  return static_cast<size_t>(kBoundsPerVariable * layout_.numCandidates() +
                             (minimumSupport_ > 0.0 ? 1 : 0));
}

vector_t ActivationBoundsConstraint::getValue(
    scalar_t time,
    const vector_t& state,
    const PreComputation& /*preComp*/) const {
  const int k = layout_.numCandidates();
  vector_t value(getNumConstraints(time));
  value.head(k) = state.tail(k);
  value.segment(k, k) = vector_t::Ones(k) - state.tail(k);
  if (minimumSupport_ > 0.0) {
    value(kBoundsPerVariable * k) = state.tail(k).sum() - minimumSupport_;
  }
  return value;
}

VectorFunctionLinearApproximation
ActivationBoundsConstraint::getLinearApproximation(
    scalar_t time, const vector_t& state, const PreComputation& preComp) const {
  const int k = layout_.numCandidates();
  VectorFunctionLinearApproximation result;
  result.setZero(static_cast<int>(getNumConstraints(time)), layout_.stateDim());
  result.f = getValue(time, state, preComp);
  result.dfdx.block(0, layout_.physicalStateDim, k, k).setIdentity();
  result.dfdx.block(k, layout_.physicalStateDim, k, k) =
      -matrix_t::Identity(k, k);
  if (minimumSupport_ > 0.0) {
    result.dfdx.block(kBoundsPerVariable * k, layout_.physicalStateDim, 1, k)
        .setOnes();
  }
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
