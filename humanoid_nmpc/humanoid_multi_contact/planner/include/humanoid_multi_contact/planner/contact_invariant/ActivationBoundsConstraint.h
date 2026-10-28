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

#include "absl/base/nullability.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/constraint/StateConstraint.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantLayout.h"

namespace ocs2::humanoid::multi_contact {

/**
 * 0 <= c_i <= 1 for every contact variable (design section 4.2): the rows
 * c_i and then 1 - c_i, h >= 0, linear in the augmented state; with a
 * positive minimum support, one more row sum_i c_i - minimumSupport >= 0: the
 * robot always claims some contact, so that no solve buys its motion with a
 * flight phase, which the quasi-static policy of design section 2 excludes.
 * Wrapped in a relaxed barrier (StateSoftConstraint). A value type.
 */
class ActivationBoundsConstraint final : public StateConstraint {
 public:
  /** Keeps a copy of `layout`; no support row when `minimumSupport` is not
   * positive. */
  ActivationBoundsConstraint(const ContactInvariantLayout& layout,
                             scalar_t minimumSupport);
  ~ActivationBoundsConstraint() override = default;
  ActivationBoundsConstraint* absl_nonnull clone() const override {
    return new ActivationBoundsConstraint(*this);
  }
  ActivationBoundsConstraint& operator=(const ActivationBoundsConstraint&) =
      delete;
  ActivationBoundsConstraint(ActivationBoundsConstraint&&) = delete;
  ActivationBoundsConstraint& operator=(ActivationBoundsConstraint&&) = delete;

  size_t getNumConstraints(scalar_t time) const override;
  vector_t getValue(scalar_t time,
                    const vector_t& state,
                    const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(
      scalar_t time,
      const vector_t& state,
      const PreComputation& preComp) const override;

 private:
  ActivationBoundsConstraint(const ActivationBoundsConstraint& other) = default;

  ContactInvariantLayout layout_;
  scalar_t minimumSupport_ = 0.0;
};

}  // namespace ocs2::humanoid::multi_contact
