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
#include "ocs2_core/dynamics/SystemDynamicsBase.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantLayout.h"

namespace ocs2::humanoid::multi_contact {

/**
 * The dynamics of a contact-invariant problem (design section 4.2): the
 * physical dynamics on the physical part of the augmented state and input,
 * and c_dot = u_c for the contact variables. Jumps leave the contact
 * variables unchanged.
 *
 * Cloned per solver worker; a single instance is not thread-safe.
 */
class AugmentedDynamics final : public SystemDynamicsBase {
 public:
  /** Keeps a copy of `layout`. */
  AugmentedDynamics(const ContactInvariantLayout& layout,
                    std::unique_ptr<SystemDynamicsBase> physical);
  ~AugmentedDynamics() override = default;
  AugmentedDynamics* absl_nonnull clone() const override {
    return new AugmentedDynamics(*this);
  }
  AugmentedDynamics& operator=(const AugmentedDynamics&) = delete;
  AugmentedDynamics(AugmentedDynamics&&) = delete;
  AugmentedDynamics& operator=(AugmentedDynamics&&) = delete;

  vector_t computeFlowMap(scalar_t time,
                          const vector_t& state,
                          const vector_t& input,
                          const PreComputation& preComp) override;
  VectorFunctionLinearApproximation linearApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const PreComputation& preComp) override;

 private:
  AugmentedDynamics(const AugmentedDynamics& other);

  ContactInvariantLayout layout_;
  std::unique_ptr<SystemDynamicsBase> physical_;
};

}  // namespace ocs2::humanoid::multi_contact
