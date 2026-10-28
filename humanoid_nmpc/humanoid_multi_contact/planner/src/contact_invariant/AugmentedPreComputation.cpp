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

#include "humanoid_multi_contact/planner/contact_invariant/AugmentedPreComputation.h"

#include <memory>
#include <utility>

#include "absl/log/absl_check.h"

namespace ocs2::humanoid::multi_contact {

AugmentedPreComputation::AugmentedPreComputation(
    const ContactInvariantLayout& layout,
    std::unique_ptr<PreComputation> physical,
    std::unique_ptr<ContactFeatureFunction> features)
    : layout_(layout),
      physical_(std::move(physical)),
      featureFunction_(std::move(features)) {
  ABSL_CHECK(physical_ != nullptr);
  ABSL_CHECK(featureFunction_ != nullptr);
}

AugmentedPreComputation::AugmentedPreComputation(
    const AugmentedPreComputation& other)
    : PreComputation(other),
      layout_(other.layout_),
      physical_(other.physical_->clone()),
      featureFunction_(
          std::make_unique<ContactFeatureFunction>(*other.featureFunction_)),
      features_(other.features_) {}

void AugmentedPreComputation::request(RequestSet request,
                                      scalar_t t,
                                      const vector_t& x,
                                      const vector_t& u) {
  const vector_t state = x.head(layout_.physicalStateDim);
  const vector_t input = u.head(layout_.physicalInputDim);
  physical_->request(request, t, state, input);
  if (!request.containsAny(Request::Cost + Request::SoftConstraint +
                           Request::Constraint)) {
    return;
  }
  if (request.contains(Request::Approximation)) {
    const VectorFunctionLinearApproximation physical =
        featureFunction_->linearApproximation(state, input);
    features_.f = physical.f;
    features_.dfdx.setZero(physical.f.size(), layout_.stateDim());
    features_.dfdx.leftCols(layout_.physicalStateDim) = physical.dfdx;
    features_.dfdu.setZero(physical.f.size(), layout_.inputDim());
    features_.dfdu.leftCols(layout_.physicalInputDim) = physical.dfdu;
  } else {
    features_.f = featureFunction_->value(state, input);
  }
}

void AugmentedPreComputation::requestPreJump(RequestSet request,
                                             scalar_t t,
                                             const vector_t& x) {
  physical_->requestPreJump(request, t, x.head(layout_.physicalStateDim));
}

void AugmentedPreComputation::requestFinal(RequestSet request,
                                           scalar_t t,
                                           const vector_t& x) {
  physical_->requestFinal(request, t, x.head(layout_.physicalStateDim));
}

}  // namespace ocs2::humanoid::multi_contact
