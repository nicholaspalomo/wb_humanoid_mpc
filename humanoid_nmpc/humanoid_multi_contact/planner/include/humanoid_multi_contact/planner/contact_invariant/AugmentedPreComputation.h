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
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureFunction.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantLayout.h"

namespace ocs2::humanoid::multi_contact {

/**
 * The pre-computation of a contact-invariant problem: the physical problem's
 * own pre-computation, requested with the physical part of the augmented
 * state and input, and the contact features (ContactFeatureFunction) of every
 * candidate at that state and input - their values for a cost or constraint
 * request, with their Jacobians (with respect to the augmented state and
 * input, zero in the activation columns) when the request asks for an
 * approximation. The final node requests only the physical pre-computation.
 *
 * Cloned per solver worker; a single instance is not thread-safe.
 */
class AugmentedPreComputation final : public PreComputation {
 public:
  /** Keeps a copy of `layout`. */
  AugmentedPreComputation(const ContactInvariantLayout& layout,
                          std::unique_ptr<PreComputation> physical,
                          std::unique_ptr<ContactFeatureFunction> features);
  ~AugmentedPreComputation() override = default;
  AugmentedPreComputation* absl_nonnull clone() const override {
    return new AugmentedPreComputation(*this);
  }
  AugmentedPreComputation& operator=(const AugmentedPreComputation&) = delete;
  AugmentedPreComputation(AugmentedPreComputation&&) = delete;
  AugmentedPreComputation& operator=(AugmentedPreComputation&&) = delete;

  void request(RequestSet request,
               scalar_t t,
               const vector_t& x,
               const vector_t& u) override;
  void requestPreJump(RequestSet request,
                      scalar_t t,
                      const vector_t& x) override;
  void requestFinal(RequestSet request, scalar_t t, const vector_t& x) override;

  /** The physical problem's pre-computation, as of the last request. */
  const PreComputation& physical() const { return *physical_; }
  /** The features of the last intermediate request: `f`, and, after an
   * approximation request, `dfdx` and `dfdu` over the augmented state and
   * input. */
  const VectorFunctionLinearApproximation& features() const {
    return features_;
  }
  const ContactInvariantLayout& layout() const { return layout_; }

 private:
  AugmentedPreComputation(const AugmentedPreComputation& other);

  ContactInvariantLayout layout_;
  std::unique_ptr<PreComputation> physical_;
  std::unique_ptr<ContactFeatureFunction> featureFunction_;
  VectorFunctionLinearApproximation features_;
};

}  // namespace ocs2::humanoid::multi_contact
