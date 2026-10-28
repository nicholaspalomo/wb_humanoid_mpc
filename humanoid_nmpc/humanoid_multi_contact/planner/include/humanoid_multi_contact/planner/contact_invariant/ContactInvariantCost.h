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

/** The weights of ContactInvariantCost. Passive data. */
struct ContactInvariantWeights {
  // The weight of the whole term, raised by the continuation.
  scalar_t weight = 0.0;
  // [s^2/m^2 relative to 1/m^2] w_slip of the tangential velocity, and the
  // weight of the tilt (dimensionless) against the squared distance [m^2].
  scalar_t slipWeight = 0.0;
  scalar_t tiltWeight = 0.0;
};

/**
 * The contact-invariant cost of design section 4.2 (Mordatch, Todorov and
 * Popovic's L_CI):
 *
 *   L = weight sum_i c_i (d_i^2 + w_slip |v_t,i|^2 + w_tilt tilt_i^2),
 *
 * with c_i the candidate's contact variable (clamped at 0 from below, where
 * the bound constraint keeps it; linear above 1, so that a c the force gate
 * no longer rewards always pays) and the distance, slip and tilt rows of the
 * contact features of an AugmentedPreComputation (ContactFeatureRows.h): a
 * candidate that claims contact pays for being away from a surface, sliding
 * on it or misaligned with it.
 *
 * The term is not convex in (c_i, e_i): its exact Hessian in those variables
 * is projected onto the positive semidefinite cone
 * (projectToPositiveSemidefinite()) and chained through the features'
 * Jacobians (Gauss-Newton in e). Cloned per solver worker.
 */
class ContactInvariantCost final : public StateInputCost {
 public:
  /** Keeps a copy of `layout`. */
  ContactInvariantCost(const ContactInvariantLayout& layout,
                       const ContactInvariantWeights& weights);
  ~ContactInvariantCost() override = default;
  ContactInvariantCost* absl_nonnull clone() const override {
    return new ContactInvariantCost(*this);
  }
  ContactInvariantCost& operator=(const ContactInvariantCost&) = delete;
  ContactInvariantCost(ContactInvariantCost&&) = delete;
  ContactInvariantCost& operator=(ContactInvariantCost&&) = delete;

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

  /** The weighted residual r_i = d^2 + w_slip |v_t|^2 + w_tilt tilt^2 of
   * candidate `candidate` in `features`. */
  scalar_t residual(const vector_t& features, int candidate) const;

 private:
  ContactInvariantCost(const ContactInvariantCost& other) = default;

  /** diag(1, w_slip, w_slip, w_slip, w_tilt), the weight of e_i. */
  vector_t residualWeights() const;

  ContactInvariantLayout layout_;
  ContactInvariantWeights weights_;
};

}  // namespace ocs2::humanoid::multi_contact
