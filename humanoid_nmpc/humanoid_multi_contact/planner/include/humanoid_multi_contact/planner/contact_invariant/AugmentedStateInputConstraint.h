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

#include <memory>

#include "absl/base/nullability.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/constraint/StateInputConstraint.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantLayout.h"

namespace ocs2::humanoid::multi_contact {

/**
 * A state-input constraint of the physical problem in a contact-invariant
 * problem (the zero wrench of a candidate the scenario does not allow):
 * evaluated on the physical part of the augmented state and input, with the
 * physical pre-computation, its derivatives padded with zero columns for the
 * contact variables and their rates. Linear approximations only.
 *
 * Cloned per solver worker; a single instance is not thread-safe.
 */
class AugmentedStateInputConstraint final : public StateInputConstraint {
 public:
  /** Keeps a copy of `layout`. */
  AugmentedStateInputConstraint(const ContactInvariantLayout& layout,
                                std::unique_ptr<StateInputConstraint> physical);
  ~AugmentedStateInputConstraint() override = default;
  AugmentedStateInputConstraint* absl_nonnull clone() const override {
    return new AugmentedStateInputConstraint(*this);
  }
  AugmentedStateInputConstraint& operator=(
      const AugmentedStateInputConstraint&) = delete;
  AugmentedStateInputConstraint(AugmentedStateInputConstraint&&) = delete;
  AugmentedStateInputConstraint& operator=(AugmentedStateInputConstraint&&) =
      delete;

  bool isActive(scalar_t time) const override {
    return physical_->isActive(time);
  }
  size_t getNumConstraints(scalar_t time) const override {
    return physical_->getNumConstraints(time);
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
  AugmentedStateInputConstraint(const AugmentedStateInputConstraint& other);

  ContactInvariantLayout layout_;
  std::unique_ptr<StateInputConstraint> physical_;
};

}  // namespace ocs2::humanoid::multi_contact
