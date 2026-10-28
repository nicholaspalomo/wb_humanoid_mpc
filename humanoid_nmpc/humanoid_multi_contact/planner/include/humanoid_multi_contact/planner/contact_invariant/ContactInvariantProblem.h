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

#include "absl/status/statusor.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h"
#include "ocs2_core/penalties/penalties/SquaredHingePenalty.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureFunction.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactForceGateCost.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantCost.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantLayout.h"

namespace ocs2::humanoid::multi_contact {

/** The names of the contact-invariant terms in the problem's collections. */
inline constexpr char kContactInvariantTerm[] = "contactInvariant";
inline constexpr char kForceGateTerm[] = "contactForceGate";
inline constexpr char kActivationRateTerm[] = "contactActivationRate";
inline constexpr char kMomentumRateTerm[] = "momentumRate";
inline constexpr char kActivationBoundsTerm[] = "contactActivationBounds";
inline constexpr char kFrictionConeTerm[] = "blendedFrictionCones";
inline constexpr char kPadClearanceTerm[] = "padClearance";

/** The weights of one solve of the continuation (design section 4.2).
 * Passive data. */
struct ContactInvariantStageWeights {
  ContactInvariantWeights contactInvariant;
  ForceGateWeights forceGate;
  // |u_c|^2.
  scalar_t activationRateWeight = 0.0;
  // [s^4/m^2] |c_ddot|^2 and |L_dot / m|^2 (the quasi-static terms).
  scalar_t linearMomentumRateWeight = 0.0;
  scalar_t angularMomentumRateWeight = 0.0;
  // The least sum of the contact variables (ActivationBoundsConstraint); 0:
  // no bound, flight phases allowed.
  scalar_t minimumSupport = 0.0;
};

/** The penalties of the inequality terms. Passive data. */
struct ContactInvariantBarriers {
  // A squared hinge: no push inside [0, 1], where c is free to fall to 0.
  SquaredHingePenalty::Config activationBounds;
  RelaxedBarrierPenalty::Config frictionCone;
  RelaxedBarrierPenalty::Config padClearance;
};

/**
 * The contact-invariant problem of design section 4.2 on a physical problem:
 * every term of `physical` (copied), wrapped to read the physical part of the
 * augmented state and input (AugmentedStateInputCost, AugmentedStateCost,
 * AugmentedStateInputConstraint), its dynamics augmented with c_dot = u_c
 * (AugmentedDynamics) and its pre-computation with the contact features of
 * `features` (AugmentedPreComputation), plus
 *  - the contact-invariant cost (ContactInvariantCost),
 *  - the force gate (ContactForceGateCost),
 *  - |u_c|^2 (ActivationRateCost) and the quasi-static terms
 *    (MomentumRateCost),
 *  - 0 <= c <= 1 (ActivationBoundsConstraint) under a squared hinge, the
 *    friction cones of the point candidates in the blended patch frame and
 *    the clearance of every pad's geometry (ContactFeatureConstraint) under
 *    relaxed barriers;
 * on `augmentedManifold` (nullptr: Euclidean), the physical state's manifold
 * with a Euclidean block for the contact variables.
 *
 * InvalidArgument when `physical` has state-only equality or inequality
 * constraints, which no term wraps, or when the layout does not match the
 * features.
 */
absl::StatusOr<OptimalControlProblem> augmentProblem(
    const OptimalControlProblem& physical,
    const ContactInvariantLayout& layout,
    const ContactFeatureFunction& features,
    std::shared_ptr<const StateManifold> augmentedManifold,
    const ContactInvariantStageWeights& weights,
    const ContactInvariantBarriers& barriers);

}  // namespace ocs2::humanoid::multi_contact
