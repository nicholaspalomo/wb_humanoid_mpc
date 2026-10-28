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

#pragma once

#include <cstddef>
#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/constraint/StateInputConstraint.h"

namespace ocs2::humanoid::multi_contact {

/**
 * Some rows of the contact features of an AugmentedPreComputation as an
 * inequality h >= 0 (ContactFeatureRows.h): the friction cones of the point
 * candidates in the blended patch frame, or the clearance of every pad's
 * geometry to the terrain (design section 4.2, "hard constraints"). Wrapped
 * in a relaxed barrier (StateInputSoftConstraint), as the MPC's cones are:
 * OCS2's SQP hands HPIPM equality constraints only. Linear approximations
 * from the features' Jacobians. A value type.
 */
class ContactFeatureConstraint final : public StateInputConstraint {
 public:
  /** The feature rows `rows`, in that order. */
  explicit ContactFeatureConstraint(std::vector<int> rows);
  ~ContactFeatureConstraint() override = default;
  ContactFeatureConstraint* absl_nonnull clone() const override {
    return new ContactFeatureConstraint(*this);
  }
  ContactFeatureConstraint& operator=(const ContactFeatureConstraint&) = delete;
  ContactFeatureConstraint(ContactFeatureConstraint&&) = delete;
  ContactFeatureConstraint& operator=(ContactFeatureConstraint&&) = delete;

  size_t getNumConstraints(scalar_t /*time*/) const override {
    return rows_.size();
  }
  vector_t getValue(scalar_t time,
                    const vector_t& state,
                    const vector_t& input,
                    const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const PreComputation& preComp) const override;

 private:
  ContactFeatureConstraint(const ContactFeatureConstraint& other) = default;

  std::vector<int> rows_;
};

}  // namespace ocs2::humanoid::multi_contact
