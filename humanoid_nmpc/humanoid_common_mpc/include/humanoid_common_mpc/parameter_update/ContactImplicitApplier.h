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

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/Types.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Applies contact_implicit to the terms of the contact-implicit formulation (humanoid_nmpc/docs/contact_implicit_mpc/
 * README.md): the weights and references of the complementarity, slip and penetration terms of every foot, checked as
 * at start-up and refused as a whole, so that the terms never run on half of an edit. They are absent unless the
 * formulation is listed, which is the normal case.
 *
 * It also keeps the complementarity and penetration terms on the ground the reference manager applied: before every
 * solve (beforeEverySolve(), which runs right after the reference manager's own preSolverRun()), the terms move onto
 * getAppliedTerrainHeight() when it has moved, so a reloaded terrain_height (ReferenceManagerApplier) reaches them in
 * the very solve whose swing trajectories and landing targets were built on it. Without a reference manager there are
 * no references to agree with, and ReferenceManagerApplier sets the terms at once (setContactImplicitTermsTerrainHeight()).
 *
 * Not thread-safe (HotFieldApplier).
 */
class ContactImplicitApplier final : public HotFieldApplier {
 public:
  /** The fields it applies: contact_implicit. */
  static absl::Span<const absl::string_view> staticFields();

  /** The applier of a problem whose contacts are `contactNames`, in input order. */
  explicit ContactImplicitApplier(std::vector<std::string> contactNames) : contactNames_(std::move(contactNames)) {}

  absl::string_view name() const override { return "ContactImplicitApplier"; }
  absl::Span<const absl::string_view> fields() const final { return staticFields(); }
  void apply(HotUpdateTarget& target) override;

  /**
   * Moves the complementarity and penetration terms of every worker's problem of `solver` onto the ground
   * `referenceManager` applied (nothing without one), when it differs from the ground they were last set to.
   */
  void beforeEverySolve(SqpSolver& solver, const SwitchedModelReferenceManager* absl_nullable referenceManager) override;

 private:
  std::vector<std::string> contactNames_;
  /// [m] The ground the terms were last set to by beforeEverySolve(); empty until the first one with a reference manager.
  std::optional<scalar_t> termsTerrainHeight_;
};

/**
 * Sets the terrain height of the complementarity and penetration terms of every foot of `contactNames` in every problem
 * of `problems` to `terrainHeight`: the two terms are a pair - one says a foot may not carry load above the ground, the
 * other that it may not go below it - so they always move together. A problem without them is left alone.
 */
void setContactImplicitTermsTerrainHeight(std::vector<OptimalControlProblem>& problems,
                                          const std::vector<std::string>& contactNames,
                                          scalar_t terrainHeight);

}  // namespace ocs2::humanoid
