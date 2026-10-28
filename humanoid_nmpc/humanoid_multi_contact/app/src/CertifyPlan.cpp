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

#include "humanoid_multi_contact/app/CertifyPlan.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/plan/PlanFile.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"
#include "humanoid_multi_contact/stability/StabilityCertifier.h"

namespace ocs2::humanoid::multi_contact {
namespace {

/** `from` + `fraction` (`to` - `from`); `from` when the two differ in size.
 */
Eigen::VectorXd lerp(const Eigen::VectorXd& from,
                     const Eigen::VectorXd& to,
                     double fraction) {
  if (from.size() != to.size()) return from;
  return from + fraction * (to - from);
}

/** The knot at `fraction` of the way from `from` to `to` (planBetweenKnots()).
 */
config::MultiContactPlan::Knot knotBetween(
    const config::MultiContactPlan::Knot& from,
    const config::MultiContactPlan::Knot& to,
    double fraction) {
  config::MultiContactPlan::Knot knot = from;
  knot.time = from.time + fraction * (to.time - from.time);
  knot.base_position = lerp(from.base_position, to.base_position, fraction);
  if (from.base_orientation_xyzw.size() == kQuaternionCoefficients &&
      to.base_orientation_xyzw.size() == kQuaternionCoefficients) {
    knot.base_orientation_xyzw = quaternionSlerp<double>(
        Eigen::Vector4d(from.base_orientation_xyzw),
        Eigen::Vector4d(to.base_orientation_xyzw), fraction);
  }
  knot.joint_positions =
      lerp(from.joint_positions, to.joint_positions, fraction);
  knot.joint_velocities =
      lerp(from.joint_velocities, to.joint_velocities, fraction);
  knot.linear_momentum =
      lerp(from.linear_momentum, to.linear_momentum, fraction);
  knot.angular_momentum =
      lerp(from.angular_momentum, to.angular_momentum, fraction);
  if (from.contact_wrenches.size() != to.contact_wrenches.size()) return knot;
  for (size_t c = 0; c < knot.contact_wrenches.size(); ++c) {
    if (from.contact_wrenches[c].contact != to.contact_wrenches[c].contact) {
      knot.contact_wrenches = from.contact_wrenches;
      return knot;
    }
    knot.contact_wrenches[c].force = lerp(
        from.contact_wrenches[c].force, to.contact_wrenches[c].force, fraction);
    knot.contact_wrenches[c].torque =
        lerp(from.contact_wrenches[c].torque, to.contact_wrenches[c].torque,
             fraction);
  }
  return knot;
}

/** "0.0123", or "-" for an absent value. */
std::string optionalValue(std::optional<double> value) {
  return value.has_value() ? absl::StrFormat("%.4f", *value) : "-";
}

}  // namespace

config::MultiContactPlan planBetweenKnots(const config::MultiContactPlan& plan,
                                          int samples) {
  config::MultiContactPlan dense = plan;
  dense.report = std::nullopt;
  dense.knots.clear();
  for (size_t k = 0; k < plan.knots.size(); ++k) {
    dense.knots.push_back(plan.knots[k]);
    if (k + 1 == plan.knots.size()) break;
    for (int i = 1; i <= samples; ++i) {
      dense.knots.push_back(
          knotBetween(plan.knots[k], plan.knots[k + 1],
                      static_cast<double>(i) / (samples + 1)));
    }
  }
  return dense;
}

absl::StatusOr<config::MultiContactPlan> certifyPlanFile(
    absl::string_view scenarioPath,
    absl::string_view planPath,
    int samplesBetweenKnots) {
  if (samplesBetweenKnots < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("the samples between knots must not be negative, not ",
                     samplesBetweenKnots));
  }
  ASSIGN_OR_RETURN(const LoadedScenario loaded, loadScenario(scenarioPath));
  ASSIGN_OR_RETURN(config::MultiContactPlan plan,
                   loadPlanFile(planPath, loaded.contactSetConfig,
                                loaded.scenario.terrain_scene));
  if (samplesBetweenKnots > 0) {
    plan = planBetweenKnots(plan, samplesBetweenKnots);
  }
  ASSIGN_OR_RETURN(const std::unique_ptr<StabilityCertifier> certifier,
                   makeCertifier(loaded));
  ASSIGN_OR_RETURN(plan.report, certifier->certify(plan));
  return plan;
}

std::string formatReport(const config::CertifierReport& report) {
  // The torque margin, or the joint limits that judge no torque.
  const std::string torques =
      report.torque_margin.has_value()
          ? absl::StrFormat("torque margin %.2f", *report.torque_margin)
          : absl::StrFormat("torques not judged: joint limits %s",
                            report.joint_limits);
  std::string text =
      absl::StrFormat("%s: %d of %d knots fail (static margin %s m, %s)\n",
                      report.passed ? "PASSED" : "FAILED", report.failed_knots,
                      static_cast<int>(report.knots.size()),
                      optionalValue(report.static_margin), torques);
  for (const config::CertifierReport::Knot& knot : report.knots) {
    absl::StrAppend(
        &text,
        absl::StrFormat(
            "  t=%7.3f %s [%s] rho_s=%s torque=%s (%s) cwc=%s pad=%.4f%s\n",
            knot.time, knot.passed ? "ok  " : "FAIL",
            absl::StrJoin(knot.active_contacts, ","),
            optionalValue(knot.static_margin),
            optionalValue(knot.torque_utilization), knot.max_torque_joint,
            knot.momentum_rate_feasible ? "in" : "OUT",
            knot.max_active_contact_distance,
            knot.failures.empty()
                ? ""
                : absl::StrCat(" -- ", absl::StrJoin(knot.failures, "; "))));
  }
  return text;
}

}  // namespace ocs2::humanoid::multi_contact
