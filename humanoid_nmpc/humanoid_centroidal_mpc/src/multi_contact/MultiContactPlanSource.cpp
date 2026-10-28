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

#include "humanoid_centroidal_mpc/multi_contact/MultiContactPlanSource.h"

#include <cmath>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/plan/PlanFile.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid {
namespace {

/** `path` when a file of that name is in the working directory, otherwise
 * the runfiles' file of the repository-relative path. */
absl::StatusOr<std::string> resolvePlanInput(absl::string_view field,
                                             absl::string_view path) {
  if (path.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[MultiContactPlanSource] multi_contact_plan.", field,
                     " is empty: name the file (or pass --multi_contact_",
                     field == "scenario_file" ? "scenario" : "plan", ")."));
  }
  std::error_code error;
  if (std::filesystem::is_regular_file(std::filesystem::path(path), error)) {
    return std::string(path);
  }
  absl::StatusOr<std::string> resolved = robot::resolveResourcePath(path);
  if (!resolved.ok() &&
      resolved.status().code() == absl::StatusCode::kNotFound) {
    return absl::NotFoundError(absl::StrCat(
        "[MultiContactPlanSource] multi_contact_plan.", field, " '", path,
        "' is neither a file of the working directory nor of the binary's "
        "runfiles",
        field == "plan_file"
            ? "; plan the scenario first (make plan-drc-atlas-<scenario>, "
              "humanoid_nmpc/docs/multi_contact_planning/README.md) or name "
              "another plan with --multi_contact_plan"
            : "",
        ". ", resolved.status().message()));
  }
  return resolved;
}

/** The values a field may take. */
enum class Range { kPositive, kNotNegative };

/** InvalidArgument naming `<block>.<field>` unless `value` is finite and in
 * `range`. */
absl::Status checkValue(absl::string_view block,
                        absl::string_view field,
                        scalar_t value,
                        Range range) {
  const bool positive = range == Range::kPositive;
  const bool inRange =
      std::isfinite(value) && (positive ? value > 0.0 : value >= 0.0);
  if (inRange) return absl::OkStatus();
  return absl::InvalidArgumentError(
      absl::StrCat("multi_contact_plan.", block, ".", field, " is ", value,
                   ", but it must be finite and ",
                   positive ? "positive." : "not negative."));
}

}  // namespace

absl::StatusOr<MultiContactPlanSource> loadMultiContactPlanSource(
    absl::string_view scenarioFile, absl::string_view planFile) {
  ASSIGN_OR_RETURN(const std::string scenarioPath,
                   resolvePlanInput("scenario_file", scenarioFile));
  ASSIGN_OR_RETURN(const std::string planPath,
                   resolvePlanInput("plan_file", planFile));
  ASSIGN_OR_RETURN(multi_contact::LoadedScenario scenario,
                   multi_contact::loadScenario(scenarioPath));
  ASSIGN_OR_RETURN(
      multi_contact::config::MultiContactPlan plan,
      multi_contact::loadPlanFile(planPath, scenario.contactSetConfig,
                                  scenario.scenario.terrain_scene));
  return MultiContactPlanSource{.scenario = std::move(scenario),
                                .plan = std::move(plan)};
}

absl::StatusOr<PlannedMotionSettings> plannedMotionSettingsFromConfig(
    const mpc_config::MultiContactPlanConfig& config) {
  PlannedMotionSettings settings;
  const mpc_config::MultiContactPlanConfig::ClockHold& hold = config.clock_hold;
  RETURN_IF_ERROR(
      checkValue("clock_hold", "lead", hold.lead, Range::kPositive));
  RETURN_IF_ERROR(checkValue("clock_hold", "contact_tolerance",
                             hold.contact_tolerance, Range::kPositive));
  RETURN_IF_ERROR(checkValue("clock_hold", "com_tolerance", hold.com_tolerance,
                             Range::kPositive));
  settings.clockHoldLead = hold.lead;
  settings.contactTolerance = hold.contact_tolerance;
  settings.comTolerance = hold.com_tolerance;

  RETURN_IF_ERROR(checkValue("swing", "lift_height", config.swing.lift_height,
                             Range::kPositive));
  RETURN_IF_ERROR(checkValue("swing", "clearance", config.swing.clearance,
                             Range::kNotNegative));
  settings.swing.liftHeight = config.swing.lift_height;
  settings.swing.clearance = config.swing.clearance;

  RETURN_IF_ERROR(checkValue("contact_swing", "weight",
                             config.contact_swing.weight, Range::kPositive));
  RETURN_IF_ERROR(checkValue("contact_swing", "position_gain",
                             config.contact_swing.position_gain,
                             Range::kNotNegative));
  settings.contactSwingWeight = config.contact_swing.weight;
  settings.contactSwingPositionGain = config.contact_swing.position_gain;
  RETURN_IF_ERROR(checkValue("contact_swing", "landing_alignment_weight",
                             config.contact_swing.landing_alignment_weight,
                             Range::kNotNegative));
  RETURN_IF_ERROR(checkValue("contact_swing", "landing_window",
                             config.contact_swing.landing_window,
                             Range::kPositive));
  settings.landingAlignmentWeight =
      config.contact_swing.landing_alignment_weight;
  settings.landingWindow = config.contact_swing.landing_window;

  const mpc_config::MultiContactPlanConfig::TerrainNonPenetration& barrier =
      config.terrain_non_penetration;
  RETURN_IF_ERROR(checkValue("terrain_non_penetration", "mu", barrier.mu,
                             Range::kPositive));
  RETURN_IF_ERROR(checkValue("terrain_non_penetration", "delta", barrier.delta,
                             Range::kPositive));
  RETURN_IF_ERROR(checkValue("terrain_non_penetration", "margin",
                             barrier.margin, Range::kNotNegative));
  RETURN_IF_ERROR(checkValue("multi_contact_plan", "arc_swing_duration_limit",
                             config.arc_swing_duration_limit,
                             Range::kNotNegative));
  settings.arcSwingDurationLimit = config.arc_swing_duration_limit;
  settings.nonPenetrationMu = barrier.mu;
  settings.nonPenetrationDelta = barrier.delta;
  settings.nonPenetrationMargin = barrier.margin;
  if (config.force_reference == kForceReferenceNames[0]) {
    settings.forceReference = ForceReference::kWeightShare;
  } else if (config.force_reference == kForceReferenceNames[1]) {
    settings.forceReference = ForceReference::kPlannedWrenches;
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "multi_contact_plan.force_reference: '", config.force_reference,
        "' is none of ", absl::StrJoin(kForceReferenceNames, ", ")));
  }
  ASSIGN_OR_RETURN(settings.playback, planPlaybackFromName(config.playback));
  if (settings.playback == PlanPlayback::kForwardAndReverse &&
      (!std::isfinite(config.playback_dwell) ||
       !(config.playback_dwell > 0.0))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "multi_contact_plan.playback_dwell is ", config.playback_dwell,
        ", but under playback \"forward_and_reverse\" it must be finite and "
        "positive."));
  }
  settings.playbackDwell = config.playback_dwell;
  return settings;
}

}  // namespace ocs2::humanoid
