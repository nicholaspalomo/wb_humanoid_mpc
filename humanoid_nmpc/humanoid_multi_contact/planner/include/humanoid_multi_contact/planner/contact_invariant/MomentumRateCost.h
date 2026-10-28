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

#include "absl/base/nullability.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/cost/StateInputCost.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantLayout.h"

namespace ocs2::humanoid::multi_contact {

/**
 * The quasi-static terms of design section 4.2, |c_ddot|^2 and |L_dot|^2:
 *
 *   L = sum_k w_k hdot_k^2
 *
 * over the six momentum-rate rows of the contact features of an
 * AugmentedPreComputation (ContactFeatureRows.h: the CoM's acceleration and
 * the rate of the angular momentum about it over the mass), with the
 * Gauss-Newton Hessian 2 J' W J. A value type.
 */
class MomentumRateCost final : public StateInputCost {
 public:
  /** The weights of the three linear, then the three angular rows. Keeps a
   * copy of `layout`. */
  MomentumRateCost(const ContactInvariantLayout& layout,
                   scalar_t linearWeight,
                   scalar_t angularWeight);
  ~MomentumRateCost() override = default;
  MomentumRateCost* absl_nonnull clone() const override {
    return new MomentumRateCost(*this);
  }
  MomentumRateCost& operator=(const MomentumRateCost&) = delete;
  MomentumRateCost(MomentumRateCost&&) = delete;
  MomentumRateCost& operator=(MomentumRateCost&&) = delete;

  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const vector_t& input,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComp) const override;
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const TargetTrajectories& targetTrajectories,
      const PreComputation& preComp) const override;

 private:
  MomentumRateCost(const MomentumRateCost& other) = default;

  ContactInvariantLayout layout_;
  vector_t weights_;  // one per momentum-rate row
};

}  // namespace ocs2::humanoid::multi_contact
