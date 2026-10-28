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

#include "humanoid_centroidal_mpc/multi_contact/PlanPlayback.h"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace ocs2::humanoid {
namespace {

using Plan = multi_contact::config::MultiContactPlan;

/** [s] A phase that ends this close to the plan's last knot lasts to it, and
 * one that starts this close to the first knot lasts from it. */
constexpr scalar_t kPhaseEndTolerance = 1.0e-6;

/**
 * `knot` at `time` with its velocities negated, in the stance and with the
 * wrenches of `before`, the knot before it (itself for the first): the knot
 * of the plan in reverse time. A knot's stance and wrenches hold until the
 * next knot, so in reverse time they hold from `knot` back to `before`.
 */
Plan::Knot reversedKnot(const Plan::Knot& knot,
                        const Plan::Knot& before,
                        scalar_t time) {
  Plan::Knot reversed = knot;
  reversed.time = time;
  reversed.active_contacts = before.active_contacts;
  reversed.contact_wrenches = before.contact_wrenches;
  reversed.joint_velocities = -knot.joint_velocities;
  reversed.linear_momentum = -knot.linear_momentum;
  reversed.angular_momentum = -knot.angular_momentum;
  return reversed;
}

/** `knot` at `time`, at rest: its velocities and momenta zero (each of the
 * size it has). */
Plan::Knot restingKnot(const Plan::Knot& knot, scalar_t time) {
  Plan::Knot resting = knot;
  resting.time = time;
  resting.joint_velocities.setZero();
  resting.linear_momentum.setZero();
  resting.angular_momentum.setZero();
  return resting;
}

/** `phase` over [startTime, endTime]. */
Plan::ContactPhase phaseOver(const Plan::ContactPhase& phase,
                             scalar_t startTime,
                             scalar_t endTime) {
  Plan::ContactPhase copy = phase;
  copy.start_time = startTime;
  copy.end_time = endTime;
  return copy;
}

}  // namespace

absl::StatusOr<PlanPlayback> planPlaybackFromName(absl::string_view name) {
  if (name == kPlanPlaybackNames[0]) return PlanPlayback::kOnce;
  if (name == kPlanPlaybackNames[1]) return PlanPlayback::kForwardAndReverse;
  return absl::InvalidArgumentError(
      absl::StrCat("multi_contact_plan.playback: '", name, "' is none of ",
                   absl::StrJoin(kPlanPlaybackNames, ", ")));
}

absl::StatusOr<Plan> forwardAndReverseCycle(const Plan& plan, scalar_t dwell) {
  if (plan.knots.empty()) {
    return absl::InvalidArgumentError(
        "[forwardAndReverseCycle] the plan has no knots.");
  }
  if (!std::isfinite(dwell) || !(dwell > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("multi_contact_plan.playback_dwell is ", dwell,
                     " s, but it must be finite and positive."));
  }
  const Plan::Knot& first = plan.knots.front();
  const Plan::Knot& last = plan.knots.back();
  const scalar_t start = first.time;
  const scalar_t end = last.time;
  // The reversed plan's time of plan time t, and the cycle's end.
  const scalar_t mirror = 2.0 * end + dwell;
  const scalar_t period = 2.0 * (end - start) + 2.0 * dwell;
  const scalar_t cycleEnd = start + period;

  Plan cycle = plan;
  cycle.report.reset();
  cycle.knots.clear();
  for (const Plan::Knot& knot : plan.knots) cycle.knots.push_back(knot);
  cycle.knots.push_back(restingKnot(last, end + 0.5 * dwell));
  for (size_t i = plan.knots.size(); i-- > 0;) {
    const Plan::Knot& knot = plan.knots[i];
    cycle.knots.push_back(
        reversedKnot(knot, plan.knots[i == 0 ? 0 : i - 1], mirror - knot.time));
  }
  cycle.knots.push_back(restingKnot(first, cycleEnd - 0.5 * dwell));
  Plan::Knot again = first;
  again.time = cycleEnd;
  cycle.knots.push_back(std::move(again));

  // A phase [s, e] in reverse time is [tau(e), tau(s)], the stances of the
  // reversed knots its mirror exactly.
  cycle.phases.clear();
  for (const Plan::ContactPhase& phase : plan.phases) {
    const bool fromStart = phase.start_time <= start + kPhaseEndTolerance;
    const bool toEnd = phase.end_time >= end - kPhaseEndTolerance;
    const scalar_t mirroredEnd =
        fromStart ? cycleEnd : mirror - phase.start_time;
    if (toEnd) {
      cycle.phases.push_back(phaseOver(phase, phase.start_time, mirroredEnd));
      continue;
    }
    cycle.phases.push_back(phase);
    cycle.phases.push_back(
        phaseOver(phase, mirror - phase.end_time, mirroredEnd));
  }
  return cycle;
}

}  // namespace ocs2::humanoid
