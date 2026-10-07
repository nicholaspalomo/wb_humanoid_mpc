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

#include "humanoid_common_mpc/config/contact_planning/ContactPlanningFromConfig.h"

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.h"

namespace ocs2::humanoid {

namespace {

using File = mpc_config::ContactPlanningFile;

/** The list of `file` that holds the terms of `kind`. */
const std::vector<std::string>& termList(const File& file, TermKind kind) {
  switch (kind) {
    case TermKind::kModelBlock:
      return file.dynamics;
    case TermKind::kCost:
      return file.costs;
    case TermKind::kSoftConstraint:
      return file.soft_constraints;
    case TermKind::kHardConstraint:
      return file.hard_constraints;
    case TermKind::kLogicRule:
      return file.logic_rules;
    case TermKind::kAssignmentCost:
      return file.assignment_costs;
    case TermKind::kSearchStage:
      return file.search;
    case TermKind::kExecutionRule:
      return file.execution;
  }
  return file.execution;
}

/** The formulation of `file`: its eight lists as they are, an absent one empty; InvalidArgument for an empty name. */
absl::StatusOr<ContactPlanningFormulation> formulationFromConfig(const File& file) {
  ContactPlanningFormulation formulation;
  for (const TermKind kind : allTermKinds()) {
    const std::vector<std::string>& names = termList(file, kind);
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i].empty()) {
        return absl::InvalidArgumentError(
            absl::StrCat("[ContactPlanningConfig] ", termKindName(kind), "[", i,
                         "] is empty: every entry of a term list names a term (ContactPlanningFormulation.h)"));
      }
    }
    formulation.list(kind) = names;
  }
  return formulation;
}

/** The penalty of a soft constraint whose block has `slack`: the shared penalty with the halves the block sets. */
std::optional<SlackPenalty> termSlack(std::optional<File::TermSlackConfig> slack, const SlackPenalty& shared) {
  if (!slack.has_value()) return std::nullopt;
  SlackPenalty penalty = shared;
  if (slack->quadratic.has_value()) penalty.quadratic = *slack->quadratic;
  if (slack->linear.has_value()) penalty.linear = *slack->linear;
  return penalty;
}

/** Whether planner.threading `name` runs the planner on a worker thread; InvalidArgument listing the names otherwise. */
absl::StatusOr<bool> runsInBackgroundThread(absl::string_view name) {
  if (name == kBackgroundThreadPlannerThreading) return true;
  if (name == kPreSolveHookPlannerThreading) return false;
  return absl::InvalidArgumentError(
      absl::StrCat("[ContactPlanningConfig] planner.threading is '", name,
                   "', which is not where a planner runs; valid names are: ", absl::StrJoin(plannerThreadingNames(), ", "), "."));
}

/** Whether terminal_dcm.target `name` tracks the commanded velocity; InvalidArgument listing the names otherwise. */
absl::StatusOr<bool> tracksCommandedVelocity(absl::string_view name) {
  if (name == kRestTerminalDcmTarget) return false;
  if (name == kCommandedVelocityTerminalDcmTarget) return true;
  return absl::InvalidArgumentError(
      absl::StrCat("[ContactPlanningConfig] terminal_dcm.target is '", name,
                   "', which is not a target of the terminal DCM; valid names are: ", absl::StrJoin(terminalDcmTargetNames(), ", "), "."));
}

/**
 * The LIP height of shared.com_height: unset, for the model's pendulum, where the file leaves it out; the height where
 * it gives a positive one; InvalidArgument otherwise, also for the 0 that used to stand for the model's.
 */
absl::StatusOr<std::optional<scalar_t>> comHeightFromConfig(std::optional<double> comHeight) {
  if (!comHeight.has_value()) return std::nullopt;
  if (!std::isfinite(*comHeight) || *comHeight <= 0.0) {
    return absl::InvalidArgumentError(absl::StrCat("[ContactPlanningConfig] shared.com_height is ", *comHeight,
                                                   ", which is not a pendulum height: give a positive height, or leave shared.com_height ",
                                                   "out for the model's center of mass above its feet at the task file's initial_state."));
  }
  return comHeight;
}

absl::StatusOr<PlannerSettings> plannerFromConfig(const File::PlannerConfig& from) {
  PlannerSettings to;
  to.type = from.type;
  to.dt = from.dt;
  to.numNodes = from.num_nodes;
  to.commitTime = from.commit_time;
  to.maxCommitExtension = from.max_commit_extension;
  to.maxBranchAndBoundNodes = from.max_branch_and_bound_nodes;
  to.maxSolveTime = from.max_solve_time;
  to.maxQpIterations = from.max_qp_iterations;
  ASSIGN_OR_RETURN(to.runInBackgroundThread, runsInBackgroundThread(from.threading));
  to.planningFrequency = from.planning_frequency;
  to.verbose = from.verbose;
  to.logPlans = from.log_plans;
  return to;
}

absl::StatusOr<SharedParameters> sharedFromConfig(const File::SharedConfig& from) {
  SharedParameters to;
  to.gravity = from.gravity;
  ASSIGN_OR_RETURN(to.comHeight, comHeightFromConfig(from.com_height));
  to.bigM = from.big_m;
  to.slackPenalty.quadratic = from.slack_penalty.quadratic;
  to.slackPenalty.linear = from.slack_penalty.linear;
  to.gaitLimits.minSwingDuration = from.gait_limits.min_swing_duration;
  to.gaitLimits.maxSwingDuration = from.gait_limits.max_swing_duration;
  to.gaitLimits.minContactDuration = from.gait_limits.min_contact_duration;
  to.gaitLimits.maxContactDuration = from.gait_limits.max_contact_duration;
  to.gaitLimits.minDoubleSupportDuration = from.gait_limits.min_double_support_duration;
  return to;
}

HlipParameters hlipFromConfig(const File::HlipConfig& from) {
  HlipParameters to;
  to.sspDuration = from.ssp_duration;
  to.dspDuration = from.dsp_duration;
  to.stepWidth = from.step_width;
  to.maxStepLength = from.max_step_length;
  to.maxStepWidth = from.max_step_width;
  to.minStepWidth = from.min_step_width;
  to.blend.sharpness = from.blend.sharpness;
  to.blend.threshold = from.blend.threshold;
  to.blend.maxCommandedVelocityX = from.blend.max_commanded_velocity_x;
  to.blend.maxCommandedVelocityY = from.blend.max_commanded_velocity_y;
  to.blend.maxCommandedYawRate = from.blend.max_commanded_yaw_rate;
  to.blend.maxComVelocityX = from.blend.max_com_velocity_x;
  to.blend.maxComVelocityY = from.blend.max_com_velocity_y;
  return to;
}

/** The cost blocks, in the order of the file; InvalidArgument for a terminal_dcm.target that is none of its names. */
absl::Status copyCosts(const File& from, ContactPlanningConfig& to) {
  to.regularization.state = from.regularization.state;
  to.regularization.input = from.regularization.input;
  to.previousFootholdConsistency.weight = from.previous_foothold_consistency.weight;
  to.velocityTracking.weight = from.velocity_tracking.weight;
  to.stepWidth.weight = from.step_width.weight;
  to.stepWidth.nominalStepWidth = from.step_width.nominal_step_width;
  to.headingRateTracking.weight = from.heading_rate_tracking.weight;
  to.headingTracking.weight = from.heading_tracking.weight;
  to.footYawTracking.weight = from.foot_yaw_tracking.weight;
  to.yawTorqueRegularization.weight = from.yaw_torque_regularization.weight;
  to.footYawRegularization.weight = from.foot_yaw_regularization.weight;
  to.zmpRegularization.weight = from.zmp_regularization.weight;
  to.footholdRegularization.weight = from.foothold_regularization.weight;
  to.stepLength.weight = from.step_length.weight;
  to.terminalDcm.weight = from.terminal_dcm.weight;
  ASSIGN_OR_RETURN(to.terminalDcm.trackCommandedVelocity, tracksCommandedVelocity(from.terminal_dcm.target));
  return absl::OkStatus();
}

/** The soft-constraint blocks; their slack falls back on `to.shared.slackPenalty`, so the shared block comes first. */
void copySoftConstraints(const File& from, ContactPlanningConfig& to) {
  const SlackPenalty& shared = to.shared.slackPenalty;
  to.zmpSupportRegion.halfWidthX = from.zmp_support_region.half_width_x;
  to.zmpSupportRegion.halfWidthY = from.zmp_support_region.half_width_y;
  to.zmpSupportRegion.slack = termSlack(from.zmp_support_region.slack, shared);
  to.reachability.reachX = from.reachability.reach_x;
  to.reachability.reachYInner = from.reachability.reach_y_inner;
  to.reachability.reachYOuter = from.reachability.reach_y_outer;
  to.reachability.slack = termSlack(from.reachability.slack, shared);
  to.footSeparation.maxStepLength = from.foot_separation.max_step_length;
  to.footSeparation.minStepWidth = from.foot_separation.min_step_width;
  to.footSeparation.maxStepWidth = from.foot_separation.max_step_width;
  to.footSeparation.slack = termSlack(from.foot_separation.slack, shared);
  to.hipYawRange.slack = termSlack(from.hip_yaw_range.slack, shared);
}

/** The blocks of the assignment costs, the search stages and the execution rules. */
void copyBinaryTerms(const File& from, ContactPlanningConfig& to) {
  to.contactSwitch.cost = from.contact_switch.cost;
  to.planConsistency.cost = from.plan_consistency.cost;
  to.doubleSupportPenalty.cost = from.double_support_penalty.cost;
  to.diving.maxDiveIterations = from.diving.max_dive_iterations;
  to.eventShiftLocalSearch.iterations = from.event_shift_local_search.iterations;
  to.eventShiftLocalSearch.maxTime = from.event_shift_local_search.max_time;
  to.cadenceStretch.samples = from.cadence_stretch.samples;
  to.cadenceStretch.maxStretch = from.cadence_stretch.max_stretch;
  to.headingRelinearization.passes = from.heading_relinearization.passes;
  to.phaseResetting.earlyTouchdownMinSwingRatio = from.phase_resetting.early_touchdown_min_swing_ratio;
  to.phaseResetting.earlyTouchdownMinContactDuration = from.phase_resetting.early_touchdown_min_contact_duration;
  to.phaseResetting.earlyTouchdownMinAdvance = from.phase_resetting.early_touchdown_min_advance;
  to.phaseResetting.maxLateTouchdownExtension = from.phase_resetting.max_late_touchdown_extension;
  to.phaseResetting.lateTouchdownExtensionStep = from.phase_resetting.late_touchdown_extension_step;
  to.phaseResetting.lateTouchdownSearchVelocity = from.phase_resetting.late_touchdown_search_velocity;
  to.energyCadenceModulation.gain = from.energy_cadence_modulation.gain;
  to.energyCadenceModulation.deadband = from.energy_cadence_modulation.deadband;
  to.dcmStepAdjustment.gain = from.dcm_step_adjustment.gain;
  to.dcmStepAdjustment.maxOffset = from.dcm_step_adjustment.max_offset;
}

}  // namespace

std::vector<std::string> plannerThreadingNames() {
  return {std::string(kBackgroundThreadPlannerThreading), std::string(kPreSolveHookPlannerThreading)};
}

std::vector<std::string> terminalDcmTargetNames() {
  return {std::string(kRestTerminalDcmTarget), std::string(kCommandedVelocityTerminalDcmTarget)};
}

absl::StatusOr<ContactPlanningConfig> contactPlanningConfigFromConfig(const mpc_config::ContactPlanningFile& file,
                                                                      ContactPlanningValidation validation) {
  ContactPlanningConfig config;
  ASSIGN_OR_RETURN(config.planner, plannerFromConfig(file.planner));
  ASSIGN_OR_RETURN(config.shared, sharedFromConfig(file.shared));
  ASSIGN_OR_RETURN(config.formulation, formulationFromConfig(file));
  config.hlip = hlipFromConfig(file.hlip);
  RETURN_IF_ERROR(copyCosts(file, config));
  copySoftConstraints(file, config);
  copyBinaryTerms(file, config);
  if (validation == ContactPlanningValidation::kValidate) {
    RETURN_IF_ERROR(config.validateStatus());
  }
  return config;
}

absl::StatusOr<ContactPlanningConfig> contactPlanningConfigFromOptionalFile(const mpc_config::ContactPlanningFile* absl_nullable file,
                                                                            ContactPlanningValidation validation) {
  if (file != nullptr) {
    return contactPlanningConfigFromConfig(*file, validation);
  }
  ContactPlanningConfig config;
  if (validation == ContactPlanningValidation::kValidate) {
    RETURN_IF_ERROR(config.validateStatus());
  }
  return config;
}

}  // namespace ocs2::humanoid
