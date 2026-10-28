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

/** The parameters of ContactForceGateCost. Passive data. */
struct ForceGateWeights {
  // k0, raised by the continuation, and k1 > 0, the gate's stiffness: the
  // smaller, the more a candidate with c = 0 pays for a force.
  scalar_t k0 = 0.0;
  scalar_t k1 = 0.0;
  // [N] The force the forces are measured in (the robot's weight), and [m]
  // the length a surface candidate's moment is divided by on top of it.
  scalar_t forceScale = 1.0;
  scalar_t momentArm = 1.0;
};

/**
 * The force gate of design section 4.2 (the paper's L_physics force
 * penalty):
 *
 *   L = sum_i k0 |w_i|^2 / (c_i^2 + k1),
 *
 * with w_i candidate i's force over forceScale (and a surface candidate's
 * moment over forceScale times momentArm) and c_i its contact variable,
 * clamped into [0, 1]: a candidate that does not claim contact cannot push,
 * and a c beyond 1 buys nothing. Smooth and bounded; stiff when k1 is small.
 *
 * Its exact Hessian in (c_i, w_i) is projected onto the positive
 * semidefinite cone (projectToPositiveSemidefinite()). Cloned per solver
 * worker.
 */
class ContactForceGateCost final : public StateInputCost {
 public:
  /** Keeps a copy of `layout`. */
  ContactForceGateCost(const ContactInvariantLayout& layout,
                       const ForceGateWeights& weights);
  ~ContactForceGateCost() override = default;
  ContactForceGateCost* absl_nonnull clone() const override {
    return new ContactForceGateCost(*this);
  }
  ContactForceGateCost& operator=(const ContactForceGateCost&) = delete;
  ContactForceGateCost(ContactForceGateCost&&) = delete;
  ContactForceGateCost& operator=(ContactForceGateCost&&) = delete;

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
  ContactForceGateCost(const ContactForceGateCost& other) = default;

  /** The scaled wrench w_i of candidate `candidate` in `input`. */
  vector_t scaledWrench(const vector_t& input, int candidate) const;
  /** The scale of each entry of candidate `candidate`'s wrench. */
  vector_t wrenchScales(int candidate) const;

  ContactInvariantLayout layout_;
  ForceGateWeights weights_;
};

}  // namespace ocs2::humanoid::multi_contact
