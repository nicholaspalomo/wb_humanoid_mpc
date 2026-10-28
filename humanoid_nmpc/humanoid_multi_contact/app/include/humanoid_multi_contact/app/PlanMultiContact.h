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
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/planner/MultiContactPlannerRegistry.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"

namespace ocs2::humanoid::multi_contact {

/**
 * Plans the scenario at `scenarioPath` (loadScenario()) with the planner
 * `plannerName` of `registry`, or the scenario's own planner when it is empty,
 * and returns the certified plan (MultiContactPlanner::plan()). The errors of
 * the loader, the registry and the planner.
 */
absl::StatusOr<config::MultiContactPlan> planScenarioFile(
    absl::string_view scenarioPath,
    absl::string_view plannerName,
    const MultiContactPlannerRegistry& registry);

/** What a plan does, for its report. Passive data. */
struct PlanSummary {
  // The number of stances: the contact changes plus one.
  int stances = 0;
  int knots = 0;
  // [s]
  double duration = 0.0;
  // The candidates the plan uses, in contact-set order.
  std::vector<std::string> contactsUsed;
  // [m] The smallest static margin of a knot; std::nullopt when none has one.
  std::optional<double> minStaticMargin;
  // The largest torque utilization of a knot, and its joint.
  std::optional<double> maxTorqueUtilization;
  std::string maxTorqueJoint;
  // Whether the certifier judged the torques (joint limits
  // "position_and_effort"), or reported them for information only.
  bool torquesJudged = true;
  // [m] The smallest collision-sphere clearance of a knot.
  std::optional<double> minSphereClearance;
  bool certified = false;
};

/** The summary of `plan`, from its knots and its report; InvalidArgument for a
 * mask beyond `contactSet`. */
absl::StatusOr<PlanSummary> summarizePlan(const config::MultiContactPlan& plan,
                                          const ContactSet& contactSet);

/** The summary as one paragraph of text. */
std::string formatPlanSummary(const PlanSummary& summary);

}  // namespace ocs2::humanoid::multi_contact
