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

#include <array>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"

namespace ocs2::humanoid {

/** How a tracked plan plays (multi_contact_plan.playback; design section 7.3,
 * humanoid_nmpc/docs/multi_contact_planning/README.md). */
enum class PlanPlayback {
  // From its first knot to its last, then holding its last state.
  kOnce,
  // Forward, a dwell at its end, backward in time to its start, a dwell there,
  // and again, forever (forwardAndReverseCycle()).
  kForwardAndReverse,
};

/** The names of PlanPlayback, in its order. */
// LINT.IfChange(plan_playback_names)
inline constexpr std::array<absl::string_view, 2> kPlanPlaybackNames = {
    "once", "forward_and_reverse"};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/multi_contact_plan_config.proto:plan_playback_names)

/** The playback named `name`; InvalidArgument naming the field and the names
 * for any other. */
absl::StatusOr<PlanPlayback> planPlaybackFromName(absl::string_view name);

/**
 * One period of the "forward_and_reverse" playback of `plan` (t0 its first
 * knot's time, tN its last's, d = `dwell` [s]): the plan itself over [t0, tN];
 * a dwell standing at its last configuration, at rest, until tN + d; the plan
 * in reverse time over [tN + d, 2 tN + d - t0]; a dwell standing at its first
 * configuration, at rest; and the plan's first knot again at t0 + T, the
 * period T = 2 (tN - t0) + 2 d. A plan that plays this cycle periodically is
 * continuous at both seams.
 *
 * The reversed knot of knot i is at tau = 2 tN + d - t_i with the same base
 * pose and joint positions, the negated joint velocities and centroidal
 * momenta, and the active contacts and contact wrenches of knot i - 1 (of
 * itself for the first): a knot's stance and wrenches hold until the next
 * knot, so in reverse time until the knot before, and the reversed mode
 * schedule is the exact mirror of the plan's. Reversing time keeps every
 * configuration q and negates every velocity, so the momentum rate is the
 * same at every mirrored instant - the linear one m c_ddot because c_ddot is
 * even in time, the angular one A(q) q_ddot + A_dot q_dot because q_ddot is
 * even and both A_dot and q_dot are odd - and the same wrenches realize it.
 * The contact wrench cone, the static margin, the joint position limits and
 * the clearances of the reversed half are therefore the plan's own (the
 * certifier's central differences of the momenta agree exactly). Each dwell
 * holds one knot at rest in its middle, with the wrenches of the knot it
 * stands at, so that the momenta of a plan that does not end at rest return
 * to zero there and then to the negated end momenta.
 *
 * Every contact phase [s, e] is kept and mirrored to [tau(e), tau(s)], where
 * the reversed schedule has the contact down; a phase that lasts to tN merges
 * with its mirror through the top dwell, and the mirror of a phase from t0
 * lasts to t0 + T, through the bottom dwell. The scenario, hashes, joint names
 * and planner are the plan's; the cycle carries no certifier report.
 *
 * @return InvalidArgument for a plan without knots or a dwell that is not
 * finite and positive.
 */
absl::StatusOr<multi_contact::config::MultiContactPlan> forwardAndReverseCycle(
    const multi_contact::config::MultiContactPlan& plan, scalar_t dwell);

}  // namespace ocs2::humanoid
