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

#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureConstraint.h"

#include <utility>
#include <vector>

#include "ocs2_core/PreComputation.h"

#include "humanoid_multi_contact/planner/contact_invariant/AugmentedPreComputation.h"

namespace ocs2::humanoid::multi_contact {

ContactFeatureConstraint::ContactFeatureConstraint(std::vector<int> rows)
    : StateInputConstraint(ConstraintOrder::Linear), rows_(std::move(rows)) {}

vector_t ContactFeatureConstraint::getValue(
    scalar_t /*time*/,
    const vector_t& /*state*/,
    const vector_t& /*input*/,
    const PreComputation& preComp) const {
  const vector_t& features =
      cast<AugmentedPreComputation>(preComp).features().f;
  vector_t value(rows_.size());
  for (size_t k = 0; k < rows_.size(); ++k) value(k) = features(rows_[k]);
  return value;
}

VectorFunctionLinearApproximation
ContactFeatureConstraint::getLinearApproximation(
    scalar_t /*time*/,
    const vector_t& /*state*/,
    const vector_t& /*input*/,
    const PreComputation& preComp) const {
  const VectorFunctionLinearApproximation& features =
      cast<AugmentedPreComputation>(preComp).features();
  const int numRows = static_cast<int>(rows_.size());
  VectorFunctionLinearApproximation result;
  result.setZero(numRows, static_cast<int>(features.dfdx.cols()),
                 static_cast<int>(features.dfdu.cols()));
  for (int k = 0; k < numRows; ++k) {
    result.f(k) = features.f(rows_[k]);
    result.dfdx.row(k) = features.dfdx.row(rows_[k]);
    result.dfdu.row(k) = features.dfdu.row(rows_[k]);
  }
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
