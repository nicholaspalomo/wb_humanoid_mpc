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
#include "ocs2_core/cost/StateCost.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantLayout.h"

namespace ocs2::humanoid::multi_contact {

/**
 * A state cost (or state soft constraint, or final cost) of the physical
 * problem in a contact-invariant problem: evaluated on the physical part of
 * the augmented state, with the physical pre-computation of the
 * AugmentedPreComputation it is given, its derivatives padded with zeros in
 * the activation rows and columns.
 *
 * Cloned per solver worker; a single instance is not thread-safe.
 */
class AugmentedStateCost final : public StateCost {
 public:
  /** Keeps a copy of `layout`. */
  AugmentedStateCost(const ContactInvariantLayout& layout,
                     std::unique_ptr<StateCost> physical);
  ~AugmentedStateCost() override = default;
  AugmentedStateCost* absl_nonnull clone() const override {
    return new AugmentedStateCost(*this);
  }
  AugmentedStateCost& operator=(const AugmentedStateCost&) = delete;
  AugmentedStateCost(AugmentedStateCost&&) = delete;
  AugmentedStateCost& operator=(AugmentedStateCost&&) = delete;

  bool isActive(scalar_t time) const override {
    return physical_->isActive(time);
  }
  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComp) const override;
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(
      scalar_t time,
      const vector_t& state,
      const TargetTrajectories& targetTrajectories,
      const PreComputation& preComp) const override;

 private:
  AugmentedStateCost(const AugmentedStateCost& other);

  ContactInvariantLayout layout_;
  std::unique_ptr<StateCost> physical_;
};

}  // namespace ocs2::humanoid::multi_contact
