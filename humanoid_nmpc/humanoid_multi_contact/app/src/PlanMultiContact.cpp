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

#include "humanoid_multi_contact/app/PlanMultiContact.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/planner/MultiContactPlanner.h"

namespace ocs2::humanoid::multi_contact {
namespace {

/** "0.0123 m", or "-" for an absent value. */
std::string optionalValue(std::optional<double> value, absl::string_view unit) {
  return value.has_value() ? absl::StrFormat("%.4f%s", *value, unit) : "-";
}

}  // namespace

absl::StatusOr<config::MultiContactPlan> planScenarioFile(
    absl::string_view scenarioPath,
    absl::string_view plannerName,
    const MultiContactPlannerRegistry& registry) {
  ASSIGN_OR_RETURN(const LoadedScenario loaded, loadScenario(scenarioPath));
  ASSIGN_OR_RETURN(
      std::unique_ptr<MultiContactPlanner> planner,
      registry.create(plannerName.empty()
                          ? absl::string_view(loaded.scenario.planner)
                          : plannerName));
  return planner->plan(loaded);
}

absl::StatusOr<PlanSummary> summarizePlan(const config::MultiContactPlan& plan,
                                          const ContactSet& contactSet) {
  PlanSummary summary;
  summary.knots = static_cast<int>(plan.knots.size());
  ContactMask used = 0;
  for (size_t k = 0; k < plan.knots.size(); ++k) {
    used |= plan.knots[k].active_contacts;
    if (k == 0 ||
        plan.knots[k].active_contacts != plan.knots[k - 1].active_contacts) {
      ++summary.stances;
    }
  }
  if (!plan.knots.empty()) {
    summary.duration = plan.knots.back().time - plan.knots.front().time;
  }
  ASSIGN_OR_RETURN(summary.contactsUsed, contactSet.namesOf(used));
  if (!plan.report.has_value()) return summary;
  summary.certified = plan.report->passed;
  summary.torquesJudged = plan.report->torque_margin.has_value();
  for (const config::CertifierReport::Knot& knot : plan.report->knots) {
    if (knot.static_margin.has_value()) {
      summary.minStaticMargin =
          std::min(summary.minStaticMargin.value_or(*knot.static_margin),
                   *knot.static_margin);
    }
    if (knot.torque_utilization.has_value() &&
        (!summary.maxTorqueUtilization.has_value() ||
         *knot.torque_utilization > *summary.maxTorqueUtilization)) {
      summary.maxTorqueUtilization = knot.torque_utilization;
      summary.maxTorqueJoint = knot.max_torque_joint;
    }
    if (knot.min_sphere_clearance.has_value()) {
      summary.minSphereClearance = std::min(
          summary.minSphereClearance.value_or(*knot.min_sphere_clearance),
          *knot.min_sphere_clearance);
    }
  }
  return summary;
}

std::string formatPlanSummary(const PlanSummary& summary) {
  return absl::StrFormat(
      "%s: %d stances, %d knots, %.1f s; contacts used: %s; min static margin "
      "%s; max torque utilization %s (%s%s); min sphere clearance %s",
      summary.certified ? "certified" : "NOT certified", summary.stances,
      summary.knots, summary.duration,
      absl::StrJoin(summary.contactsUsed, ", "),
      optionalValue(summary.minStaticMargin, " m"),
      optionalValue(summary.maxTorqueUtilization, /*unit=*/""),
      summary.maxTorqueJoint,
      summary.torquesJudged ? "" : ", not judged: position limits only",
      optionalValue(summary.minSphereClearance, " m"));
}

}  // namespace ocs2::humanoid::multi_contact
