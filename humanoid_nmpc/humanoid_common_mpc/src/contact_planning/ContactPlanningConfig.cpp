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

#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/strip.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"

namespace ocs2::humanoid {

namespace {

// [s] the slack of the validation's comparisons of two configured durations, so
// that a commit window that matches the solve budget, or a swing time scale
// that matches the shortest swing, passes. The event times of a plan are
// compared with kEventTimeTolerance (ContactScheduleAdaptation.h) instead.
constexpr scalar_t kTimeComparisonTolerance = 1.0e-9;
// The fewest nodes a planning horizon has.
constexpr int kMinNumNodes = 2;
// The entries of the tables of validateStatus(): the weights and costs that
// must not be negative, and the command ranges of the H-LIP blend that must be
// positive.
constexpr int kNonNegativeWeightKeys = 20;
constexpr int kBlendRangeKeys = 5;

absl::Status invalidConfig(absl::string_view message) {
  return absl::InvalidArgumentError(
      absl::StrCat("[ContactPlanningConfig] ", message));
}

/** "`key` (value) must be `requirement`", the format of every single-key
 * rejection. */
absl::Status keyMustBe(absl::string_view key,
                       scalar_t value,
                       absl::string_view requirement) {
  return invalidConfig(
      absl::StrCat(key, " (", value, ") must be ", requirement));
}

/** The first rejection among the slack blocks of the soft constraints, or OK.
 */
absl::Status checkSlack(std::optional<SlackPenalty> slack,
                        absl::string_view term) {
  if (!slack.has_value()) return absl::OkStatus();
  if (slack->quadratic < 0.0)
    return keyMustBe(absl::StrCat(term, ".slack.quadratic"), slack->quadratic,
                     "non-negative");
  if (slack->linear < 0.0)
    return keyMustBe(absl::StrCat(term, ".slack.linear"), slack->linear,
                     "non-negative");
  return absl::OkStatus();
}

/** Whether the mixed-integer planner runs the event-shift local search after
 * its branch-and-bound. */
bool lipMiqpRunsLocalSearch(const ContactPlanningConfig& config) {
  return config.formulation.hasSearchStage(term::kEventShiftLocalSearch) &&
         config.eventShiftLocalSearch.iterations > 0;
}

/**
 * Worst-case time one mixed-integer plan may take [s]: the branch-and-bound's
 * anytime limit plus the local search's own budget when that stage is listed.
 * It is how long a synchronous planner blocks a solve, and the least a plan's
 * commit window has to cover when it is planned on the worker thread.
 */
scalar_t lipMiqpSolveBudget(const ContactPlanningConfig& config) {
  return config.planner.maxSolveTime +
         (lipMiqpRunsLocalSearch(config) ? config.eventShiftLocalSearch.maxTime
                                         : 0.0);
}

/** The keys lipMiqpSolveBudget() adds up, so that a message names exactly the
 * ones that make up its number. */
std::string lipMiqpSolveBudgetKeys(const ContactPlanningConfig& config) {
  return lipMiqpRunsLocalSearch(config)
             ? "planner.max_solve_time + event_shift_local_search.max_time"
             : "planner.max_solve_time";
}

/** The commit-boundary stall of an instantaneous exchange of support, which
 * both planners can reach. */
std::string commitStallWarning(absl::string_view zeroDoubleSupportKey,
                               scalar_t smallestCap) {
  return absl::StrCat(zeroDoubleSupportKey,
                      " is 0 and planner.max_commit_extension <= 0 (no cap). "
                      "Once the double supports are gone every swing starts "
                      "exactly where the previous one ends, so nothing stops "
                      "the commit boundary from being extended to the end of "
                      "the stepping region: no plan reaches past it, none is "
                      "merged and the robot stops stepping. Set "
                      "planner.max_commit_extension to at least ",
                      smallestCap, " s.");
}

void appendLipMiqpWarnings(const ContactPlanningConfig& config,
                           std::vector<std::string>& out) {
  const PlannerSettings& p = config.planner;
  const scalar_t budget = lipMiqpSolveBudget(config);
  const std::string budgetKeys = lipMiqpSolveBudgetKeys(config);
  // The branch-and-bound takes tens of milliseconds and may take its whole
  // budget. Planned in the pre-solve hook it stalls every MPC solve by that
  // much; the integration tests do it on purpose, which is why this is not an
  // error.
  if (!p.runInBackgroundThread) {
    out.push_back(absl::StrCat(
        "planner.type: lip_miqp with planner.threading: pre_solve_hook. The "
        "mixed-integer search then runs "
        "inside the MPC's pre-solve hook and blocks every solve for up to ",
        budgetKeys, " = ", budget,
        " s. Set planner.threading: background_thread."));
  } else if (p.commitTime < budget - kTimeComparisonTolerance) {
    // On the worker thread a plan reaches the solver no earlier than its solve
    // time after the snapshot it was made from, and one whose commit boundary
    // has passed by then is dropped as stale (activatePendingPlan). A commit
    // window shorter than the budget therefore throws away exactly the plans
    // that needed the budget.
    out.push_back(absl::StrCat(
        "planner.commit_time (", p.commitTime,
        " s) is shorter than the solve budget ", budgetKeys, " (", budget,
        " s): a plan that uses its budget arrives after its own commit "
        "boundary and is dropped as stale, and "
        "while plans are being dropped the executed schedule runs out and the "
        "robot stops stepping. Raise "
        "planner.commit_time or lower ",
        budgetKeys, "."));
  }
  if (config.shared.gaitLimits.minDoubleSupportDuration <= 0.0 &&
      p.maxCommitExtension <= 0.0) {
    out.push_back(
        commitStallWarning("shared.gait_limits.min_double_support_duration",
                           config.shared.gaitLimits.maxSwingDuration));
  }
}

void appendHlipWarnings(const ContactPlanningConfig& config,
                        std::vector<std::string>& out) {
  const PlannerSettings& p = config.planner;
  const HlipParameters& h = config.hlip;

  // planned_com_override is the paper's reference plumbing (equations 11-14),
  // not a heuristic. Without it the whole-body MPC tracks the operator's
  // straight-line center of mass while the footholds were placed for a lateral
  // orbit; the planner reads back a center of mass with no lateral velocity,
  // narrows the step and the robot falls (H-LIP README section 3c). The rule
  // cannot be the library default because that default is shared with lip_miqp.
  if (!config.formulation.hasExecutionRule(term::kPlannedComOverride)) {
    out.push_back(
        absl::StrCat("planner.type: hlip without ", term::kPlannedComOverride,
                     " in the execution list. The H-LIP orbit needs the center "
                     "of mass to fall towards the swing foot; without "
                     "the rule the whole-body MPC tracks the operator's "
                     "straight-line reference instead, the planner reads back "
                     "a center of mass with no lateral velocity, narrows the "
                     "step to hlip.min_step_width and the robot sidesteps "
                     "and falls. Add ",
                     term::kPlannedComOverride, " to `execution`."));
  }

  // The H-LIP deadbeat step is a feedback law on the measured state, and a plan
  // costs microseconds because nothing is solved. Running it on a background
  // thread at a fraction of the MPC rate therefore buys nothing and costs the
  // one thing the law depends on: a plan made at 10 Hz is made from a snapshot
  // up to 100 ms old, which is two fifths of a 0.25 s single support. This is a
  // warning rather than an error because the threading is the operator's call.
  if (p.runInBackgroundThread) {
    out.push_back(absl::StrCat(
        "planner.type: hlip with planner.threading: background_thread. The "
        "closed-form H-LIP planner costs microseconds, so "
        "the background thread only adds latency to a feedback law: its plan "
        "is made at most planner.planning_frequency (",
        p.planningFrequency,
        " Hz) times a second, from a snapshot up to a planning period old (on "
        "either path a plan is activated one MPC "
        "cycle after it is made). Set planner.threading: pre_solve_hook to "
        "plan in the pre-solve hook at the MPC rate."));
  }

  if (h.dspDuration <= 0.0 && p.maxCommitExtension <= 0.0) {
    out.push_back(commitStallWarning(
        "hlip.dsp_duration",
        std::max(h.sspDuration, config.shared.gaitLimits.maxSwingDuration)));
  }

  // Everything below evaluates the H-LIP itself, which HlipModel::Create()
  // refuses to build on a cadence or a pendulum that validateStatus() rejects,
  // so an invalid configuration stops here.
  const scalar_t stepDuration = h.sspDuration + h.dspDuration;
  if (h.sspDuration <= 0.0 || h.dspDuration < 0.0 ||
      !(config.pendulumHeight() > 0.0) || config.shared.gravity <= 0.0)
    return;

  // A robot standing still has its center of mass half a step width from the
  // stance foot with no lateral velocity, so the first single support ends far
  // outside the orbit and the deadbeat law asks for a wide first step. If that
  // step does not fit hlip.maxStepWidth it is clipped and the step is no longer
  // deadbeat: depending on how much is cut the residual either dies out over a
  // few steps or grows by cosh(omega sspDuration) every step until the gait
  // locks into a wide/narrow limit cycle that falls (H-LIP README section 3b).
  // Which of the two happens is what the start-up summary rolls out
  // (HlipContactPlanner::startUpLateralWidths); a first step without margin is
  // reported either way, because the summary is the only place that said so and
  // nothing checked it.
  const absl::StatusOr<scalar_t> startUpStep =
      HlipContactPlanner::startUpLateralStep(config);
  if (startUpStep.ok() && *startUpStep > h.maxStepWidth) {
    out.push_back(absl::StrCat(
        "the first step out of a standstill needs ", *startUpStep,
        " m of lateral step (HlipContactPlanner::startUpLateralStep) against "
        "hlip.max_step_width (",
        h.maxStepWidth,
        " m): it is clipped, which costs the deadbeat property, and depending "
        "on how much is cut the gait either "
        "recovers over a few steps or locks into an alternating wide/narrow "
        "limit cycle that falls (the planner's "
        "start-up summary says which). Shorten hlip.ssp_duration (or "
        "hlip.dsp_duration), or raise hlip.max_step_width."));
  }

  // The step width bounds are checked by validateStatus() against
  // hlip.stepWidth alone, which is the nominal orbit at a ZERO lateral command.
  // That is not the orbit the planner walks: HlipContactPlanner::deadbeatStep
  // builds the period-two orbit at +-stepWidth + v_y,cmd * (sspDuration +
  // dspDuration) and then clips the placed foot into [minStepWidth,
  // maxStepWidth], so a sustained sidestep moves BOTH sides of the orbit by the
  // drift of one step. The configuration is only self-consistent when the
  // narrowed side still clears the self-collision margin and the widened side
  // still fits the reach, at the largest lateral command the blend is
  // normalized for. Otherwise the narrow step is clipped on every cycle for as
  // long as the command is held, ContactPlan::describe() prints CLIPPED-STEPS
  // forever and the one signal documented to mean "the legs cannot deliver this
  // command at this cadence" becomes a standing false positive that hides
  // genuine clipping, while the realized lateral rate runs below the
  // command.
  const scalar_t lateralDrift = h.blend.maxCommandedVelocityY * stepDuration;
  if (h.stepWidth - lateralDrift < h.minStepWidth) {
    out.push_back(absl::StrCat(
        "a sustained sidestep at "
        "hlip.blend.cmd_max_linvel_base_rt_world_ewrt_heading_y (",
        h.blend.maxCommandedVelocityY, " m/s) drifts ", lateralDrift,
        " m over a step of ", stepDuration,
        " s, so the step that places the trailing foot is planned at "
        "hlip.step_width - drift = ",
        h.stepWidth - lateralDrift, " m and clipped to hlip.min_step_width (",
        h.minStepWidth,
        " m) on every occurrence. Raise hlip.step_width, or lower "
        "hlip.blend.cmd_max_linvel_base_rt_world_ewrt_heading_y below ",
        (h.stepWidth - h.minStepWidth) / stepDuration,
        " m/s, or shorten the step."));
  }
  if (h.stepWidth + lateralDrift > h.maxStepWidth) {
    out.push_back(absl::StrCat(
        "a sustained sidestep at "
        "hlip.blend.cmd_max_linvel_base_rt_world_ewrt_heading_y (",
        h.blend.maxCommandedVelocityY, " m/s) drifts ", lateralDrift,
        " m over a step of ", stepDuration,
        " s, so the step that places the leading foot is planned at "
        "hlip.step_width + drift = ",
        h.stepWidth + lateralDrift, " m and clipped to hlip.max_step_width (",
        h.maxStepWidth,
        " m) on every occurrence. Raise hlip.max_step_width, or lower "
        "hlip.blend.cmd_max_linvel_base_rt_world_ewrt_heading_y below ",
        (h.maxStepWidth - h.stepWidth) / stepDuration,
        " m/s, or shorten the step."));
  }
}

/** The `planner` keys: the first key that does not hold, as an InvalidArgument
 * naming it. */
absl::Status validatePlanner(const ContactPlanningConfig& config) {
  const PlannerSettings& p = config.planner;
  const GaitLimits& g = config.shared.gaitLimits;
  if (p.dt <= 0.0) return keyMustBe("planner.dt", p.dt, "positive");
  if (p.numNodes < kMinNumNodes)
    return keyMustBe("planner.num_nodes", p.numNodes,
                     absl::StrCat("at least ", kMinNumNodes));
  // planner.type selects the implementation ContactPlannerFactory builds, and
  // every consumer of this configuration treats "validation passed" as "this
  // configuration can be applied": loadContactPlanningConfig and
  // ContactPlanningReferenceManager::setConfig delegate their rejection here,
  // and the parameter updater reports a reload that raised nothing as applied.
  // An unknown name used to be noticed only by the factory, one layer too late:
  // by then ContactPlannerModule had already stored the whole new parameter
  // set, rebuilt the execution rules and started or stopped the background
  // worker. Rejecting the name here makes the reload fail atomically with the
  // previous configuration still in force, and the message lists the planners
  // that exist.
  if (canonicalPlannerName(p.type).empty()) {
    return invalidConfig(absl::StrCat(
        "unknown planner.type '", p.type,
        "'; supported: ", absl::StrJoin(knownPlannerNames(), ", ")));
  }
  if (p.commitTime < 0.0)
    return keyMustBe("planner.commit_time", p.commitTime, "non-negative");
  if (config.commitNodes() >= p.numNodes) {
    return invalidConfig(absl::StrCat(
        "planner.commit_time (", p.commitTime,
        " s) must be shorter than the planning horizon planner.num_nodes * ",
        "planner.dt (", config.horizon(), " s)"));
  }
  if (p.maxCommitExtension > 0.0 && p.maxCommitExtension < g.maxSwingDuration) {
    return invalidConfig(absl::StrCat(
        "planner.max_commit_extension (", p.maxCommitExtension,
        " s) must be 0 (no cap) or at least "
        "shared.gait_limits.max_swing_duration (",
        g.maxSwingDuration, " s), so a whole swing still fits in it"));
  }
  if (p.maxBranchAndBoundNodes < 1)
    return keyMustBe("planner.max_branch_and_bound_nodes",
                     p.maxBranchAndBoundNodes, "at least 1");
  if (p.maxSolveTime <= 0.0)
    return keyMustBe("planner.max_solve_time", p.maxSolveTime, "positive");
  if (p.maxQpIterations < 1)
    return keyMustBe("planner.max_qp_iterations", p.maxQpIterations,
                     "at least 1");
  if (p.planningFrequency <= 0.0)
    return keyMustBe("planner.planning_frequency", p.planningFrequency,
                     "positive");
  return absl::OkStatus();
}

/** The `shared` keys: the first key that does not hold, as an InvalidArgument
 * naming it. */
absl::Status validateShared(const ContactPlanningConfig& config) {
  const SharedParameters& s = config.shared;
  const GaitLimits& g = s.gaitLimits;
  if (s.gravity <= 0.0)
    return keyMustBe("shared.gravity", s.gravity, "positive");
  if (!s.comHeight.has_value()) {
    return invalidConfig(
        "shared.com_height is unset: a file that leaves it out gets the "
        "model's pendulum (ContactPlanningModelParameters::applyTo()) "
        "before validation, and a configuration without a model sets a "
        "positive height");
  }
  if (!(*s.comHeight > 0.0))
    return keyMustBe("shared.com_height", *s.comHeight, "positive");
  if (g.minSwingDuration <= 0.0)
    return keyMustBe("shared.gait_limits.min_swing_duration",
                     g.minSwingDuration, "positive");
  if (g.maxSwingDuration < g.minSwingDuration) {
    return invalidConfig(absl::StrCat(
        "shared.gait_limits.max_swing_duration (", g.maxSwingDuration,
        " s) must be at least shared.gait_limits.min_swing_duration (",
        g.minSwingDuration, " s)"));
  }
  if (g.minContactDuration <= 0.0)
    return keyMustBe("shared.gait_limits.min_contact_duration",
                     g.minContactDuration, "positive");
  if (g.maxContactDuration > 0.0 &&
      g.maxContactDuration < g.minContactDuration) {
    return invalidConfig(absl::StrCat(
        "shared.gait_limits.max_contact_duration (", g.maxContactDuration,
        " s) must be 0 (no limit) or at least "
        "shared.gait_limits.min_contact_duration (",
        g.minContactDuration, " s)"));
  }
  if (g.minDoubleSupportDuration < 0.0) {
    return keyMustBe("shared.gait_limits.min_double_support_duration",
                     g.minDoubleSupportDuration, "non-negative");
  }
  if (s.slackPenalty.quadratic < 0.0)
    return keyMustBe("shared.slack_penalty.quadratic", s.slackPenalty.quadratic,
                     "non-negative");
  if (s.slackPenalty.linear < 0.0)
    return keyMustBe("shared.slack_penalty.linear", s.slackPenalty.linear,
                     "non-negative");
  return absl::OkStatus();
}

/** The geometry of the mixed-integer terms: the first key that does not hold,
 * as an InvalidArgument naming it. */
absl::Status validateMixedIntegerGeometry(const ContactPlanningConfig& config) {
  const SharedParameters& s = config.shared;
  if (config.zmpSupportRegion.halfWidthX <= 0.0) {
    return keyMustBe("zmp_support_region.half_width_x",
                     config.zmpSupportRegion.halfWidthX, "positive");
  }
  if (config.zmpSupportRegion.halfWidthY <= 0.0) {
    return keyMustBe("zmp_support_region.half_width_y",
                     config.zmpSupportRegion.halfWidthY, "positive");
  }
  if (config.footSeparation.minStepWidth <= 0.0) {
    return keyMustBe("foot_separation.min_step_width",
                     config.footSeparation.minStepWidth, "positive");
  }
  if (config.footSeparation.maxStepWidth < config.footSeparation.minStepWidth) {
    return invalidConfig(absl::StrCat(
        "foot_separation.max_step_width (", config.footSeparation.maxStepWidth,
        " m) must be at least foot_separation.min_step_width (",
        config.footSeparation.minStepWidth, " m)"));
  }
  if (config.stepWidth.nominalStepWidth < config.footSeparation.minStepWidth ||
      config.stepWidth.nominalStepWidth > config.footSeparation.maxStepWidth) {
    return invalidConfig(absl::StrCat(
        "step_width.nominal_step_width (", config.stepWidth.nominalStepWidth,
        " m) must lie within [foot_separation.min_step_width, "
        "foot_separation.max_step_width] = [",
        config.footSeparation.minStepWidth, ", ",
        config.footSeparation.maxStepWidth, "] m"));
  }
  if (config.footSeparation.maxStepLength <= 0.0) {
    return keyMustBe("foot_separation.max_step_length",
                     config.footSeparation.maxStepLength, "positive");
  }
  if (config.reachability.reachX <= 0.0)
    return keyMustBe("reachability.reach_x", config.reachability.reachX,
                     "positive");
  if (config.reachability.reachYOuter <= config.reachability.reachYInner) {
    return invalidConfig(absl::StrCat(
        "reachability.reach_y_outer (", config.reachability.reachYOuter,
        " m) must exceed reachability.reach_y_inner (",
        config.reachability.reachYInner, " m)"));
  }
  if (s.bigM <= config.footSeparation.maxStepLength) {
    return invalidConfig(
        absl::StrCat("shared.big_m (", s.bigM,
                     " m) must exceed foot_separation.max_step_length (",
                     config.footSeparation.maxStepLength, " m)"));
  }
  // The big-M must also cover the LATERAL separation of the feet, and that, not
  // the heading axis guarded above, is the requirement that actually binds. In
  // double support both contact binaries are one, so the +M c_i - M c_other
  // terms of a single-support ZMP box cancel and that box relaxes to only
  // |e_y'(zmp - p_i)| <= halfWidthY + M, while the double-support rows open the
  // whole strip between the feet, e_y'p_R - halfWidthY <= e_y'zmp <= e_y'p_L +
  // halfWidthY (ZmpSupportRegionConstraint::addRows). Intersecting the two, the
  // opposite foot's relaxed row binds as soon as M is smaller than the
  // separation w = e_y'(p_L - p_R), which foot_separation.maxStepWidth bounds
  // from above; the half-width cancels, so the exact condition is bigM >=
  // maxStepWidth. Below it the ZMP is pulled towards one foot in double support
  // and every weight transfer past that point pays slack for a disjunction that
  // was supposed to be switched off.
  if (s.bigM < config.footSeparation.maxStepWidth) {
    return invalidConfig(
        absl::StrCat("shared.big_m (", s.bigM,
                     " m) must be at least foot_separation.max_step_width (",
                     config.footSeparation.maxStepWidth,
                     " m), otherwise the relaxed single-support ZMP box clips "
                     "the double-support region laterally"));
  }
  for (const absl::Status& slack :
       {checkSlack(config.zmpSupportRegion.slack, term::kZmpSupportRegion),
        checkSlack(config.reachability.slack, term::kReachability),
        checkSlack(config.footSeparation.slack, term::kFootSeparation),
        checkSlack(config.hipYawRange.slack, term::kHipYawRange)}) {
    if (!slack.ok()) return slack;
  }
  return absl::OkStatus();
}

/** The weights and costs, one key per message: the first key that does not
 * hold, as an InvalidArgument naming it. */
absl::Status validateWeights(const ContactPlanningConfig& config) {
  const std::array<std::pair<const char* absl_nonnull, scalar_t>,
                   kNonNegativeWeightKeys>
      nonNegative{{
          {"regularization.state", config.regularization.state},
          {"regularization.input", config.regularization.input},
          {"previous_foothold_consistency.weight",
           config.previousFootholdConsistency.weight},
          {"velocity_tracking.weight", config.velocityTracking.weight},
          {"step_width.weight", config.stepWidth.weight},
          {"heading_rate_tracking.weight", config.headingRateTracking.weight},
          {"heading_tracking.weight", config.headingTracking.weight},
          {"foot_yaw_tracking.weight", config.footYawTracking.weight},
          {"yaw_torque_regularization.weight",
           config.yawTorqueRegularization.weight},
          {"foot_yaw_regularization.weight",
           config.footYawRegularization.weight},
          {"zmp_regularization.weight", config.zmpRegularization.weight},
          {"foothold_regularization.weight",
           config.footholdRegularization.weight},
          {"step_length.weight", config.stepLength.weight},
          {"terminal_dcm.weight", config.terminalDcm.weight},
          {"contact_switch.cost", config.contactSwitch.cost},
          {"plan_consistency.cost", config.planConsistency.cost},
          {"double_support_penalty.cost", config.doubleSupportPenalty.cost},
          {"event_shift_local_search.max_time",
           config.eventShiftLocalSearch.maxTime},
          {"energy_cadence_modulation.deadband",
           config.energyCadenceModulation.deadband},
          {"energy_cadence_modulation.gain",
           config.energyCadenceModulation.gain},
      }};
  for (const std::pair<const char* absl_nonnull, scalar_t>& entry :
       nonNegative) {
    if (entry.second < 0.0)
      return keyMustBe(entry.first, entry.second, "non-negative");
  }
  return absl::OkStatus();
}

/** The parameters of the search stages: the first key that does not hold, as an
 * InvalidArgument naming it. */
absl::Status validateSearchStages(const ContactPlanningConfig& config) {
  if (config.eventShiftLocalSearch.iterations < 0) {
    return keyMustBe("event_shift_local_search.iterations",
                     config.eventShiftLocalSearch.iterations, "non-negative");
  }
  if (config.diving.maxDiveIterations < 1)
    return keyMustBe("diving.max_dive_iterations",
                     config.diving.maxDiveIterations, "at least 1");
  if (config.cadenceStretch.samples < 0)
    return keyMustBe("cadence_stretch.samples", config.cadenceStretch.samples,
                     "non-negative");
  if (config.cadenceStretch.samples > 0 &&
      config.cadenceStretch.maxStretch < 1.0) {
    return keyMustBe("cadence_stretch.max_stretch",
                     config.cadenceStretch.maxStretch,
                     "at least 1: a stretch below 1 shrinks the commit window "
                     "and the horizon");
  }
  if (config.headingRelinearization.passes < 0 ||
      config.headingRelinearization.passes >
          HeadingRelinearizationParameters::kMaxPasses) {
    return keyMustBe(
        "heading_relinearization.passes", config.headingRelinearization.passes,
        absl::StrCat("in [0, ", HeadingRelinearizationParameters::kMaxPasses,
                     "]"));
  }
  return absl::OkStatus();
}

/** The parameters of the execution rules: the first key that does not hold, as
 * an InvalidArgument naming it. */
absl::Status validateExecutionRules(const ContactPlanningConfig& config) {
  const PhaseResettingParameters& r = config.phaseResetting;
  if (r.earlyTouchdownMinSwingRatio < 0.0 ||
      r.earlyTouchdownMinSwingRatio > 1.0) {
    return keyMustBe("phase_resetting.early_touchdown_min_swing_ratio",
                     r.earlyTouchdownMinSwingRatio, "in [0, 1]");
  }
  if (r.earlyTouchdownMinContactDuration < 0.0) {
    return keyMustBe("phase_resetting.early_touchdown_min_contact_duration",
                     r.earlyTouchdownMinContactDuration, "non-negative");
  }
  if (r.earlyTouchdownMinAdvance < 0.0) {
    return keyMustBe("phase_resetting.early_touchdown_min_advance",
                     r.earlyTouchdownMinAdvance, "non-negative");
  }
  if (r.maxLateTouchdownExtension < 0.0) {
    return keyMustBe("phase_resetting.max_late_touchdown_extension",
                     r.maxLateTouchdownExtension, "non-negative");
  }
  if (r.lateTouchdownExtensionStep <= 0.0) {
    return keyMustBe("phase_resetting.late_touchdown_extension_step",
                     r.lateTouchdownExtensionStep, "positive");
  }
  if (r.lateTouchdownSearchVelocity < 0.0) {
    return keyMustBe(
        "phase_resetting.late_touchdown_search_speed_foot_rt_world",
        r.lateTouchdownSearchVelocity, "non-negative");
  }
  if (config.dcmStepAdjustment.gain < 0.0)
    return keyMustBe("dcm_step_adjustment.gain", config.dcmStepAdjustment.gain,
                     "non-negative");
  if (config.dcmStepAdjustment.maxOffset < 0.0) {
    return keyMustBe("dcm_step_adjustment.max_offset",
                     config.dcmStepAdjustment.maxOffset, "non-negative");
  }
  return absl::OkStatus();
}

/** The `hlip` keys: the first key that does not hold, as an InvalidArgument
 * naming it. */
absl::Status validateHlip(const ContactPlanningConfig& config) {
  const HlipParameters& h = config.hlip;
  if (h.sspDuration <= 0.0)
    return keyMustBe("hlip.ssp_duration", h.sspDuration, "positive");
  if (h.dspDuration < 0.0)
    return keyMustBe("hlip.dsp_duration", h.dspDuration, "non-negative");
  if (h.stepWidth <= 0.0)
    return keyMustBe("hlip.step_width", h.stepWidth, "positive");
  if (h.maxStepLength <= 0.0)
    return keyMustBe("hlip.max_step_length", h.maxStepLength, "positive");
  if (h.minStepWidth <= 0.0)
    return keyMustBe("hlip.min_step_width", h.minStepWidth, "positive");
  if (h.maxStepWidth < h.minStepWidth) {
    return invalidConfig(absl::StrCat(
        "hlip.max_step_width (", h.maxStepWidth,
        " m) must be at least hlip.min_step_width (", h.minStepWidth, " m)"));
  }
  if (h.stepWidth < h.minStepWidth || h.stepWidth > h.maxStepWidth) {
    return invalidConfig(absl::StrCat(
        "hlip.step_width (", h.stepWidth,
        " m) must lie within [hlip.min_step_width, hlip.max_step_width] = [",
        h.minStepWidth, ", ", h.maxStepWidth, "] m"));
  }
  if (h.blend.sharpness <= 0.0)
    return keyMustBe("hlip.blend.sharpness", h.blend.sharpness, "positive");
  // rho_2. The activity phi is a sum of squared ratios and so is never
  // negative, and isWalking compares it with
  // `>=`, so a threshold of zero declares the robot to be walking at rest:
  // standing becomes unreachable and the gait marches in place forever.
  if (h.blend.threshold <= 0.0) {
    return keyMustBe("hlip.blend.threshold", h.blend.threshold,
                     "positive, or the blend can never reach standing");
  }
  const std::array<std::pair<const char* absl_nonnull, scalar_t>,
                   kBlendRangeKeys>
      blendRanges{{
          {"hlip.blend.cmd_max_linvel_base_rt_world_ewrt_heading_x",
           h.blend.maxCommandedVelocityX},
          {"hlip.blend.cmd_max_linvel_base_rt_world_ewrt_heading_y",
           h.blend.maxCommandedVelocityY},
          {"hlip.blend.cmd_max_angvel_base_rt_world_ewrt_world_z",
           h.blend.maxCommandedYawRate},
          {"hlip.blend.max_linvel_com_rt_world_ewrt_heading_x",
           h.blend.maxComVelocityX},
          {"hlip.blend.max_linvel_com_rt_world_ewrt_heading_y",
           h.blend.maxComVelocityY},
      }};
  for (const std::pair<const char* absl_nonnull, scalar_t>& entry :
       blendRanges) {
    if (entry.second <= 0.0)
      return keyMustBe(entry.first, entry.second, "positive");
  }
  return absl::OkStatus();
}

/** The values derived from the model, not file keys: the first key that does
 * not hold, as an InvalidArgument naming it. */
absl::Status validateModelDerived(const ContactPlanningConfig& config) {
  if (config.yawTorqueBudget.torsionalFrictionTorque < 0.0) {
    return keyMustBe(
        "yaw_torque_budget torsional friction torque (derived from "
        "task.textproto contacts.contact_wrench_cone_soft_constraint)",
        config.yawTorqueBudget.torsionalFrictionTorque, "non-negative");
  }
  if (config.yawTorqueBudget.doubleSupportYawCouple < 0.0) {
    return keyMustBe(
        "yaw_torque_budget double-support yaw couple (derived from "
        "task.textproto contacts.contact_wrench_cone_soft_constraint)",
        config.yawTorqueBudget.doubleSupportYawCouple, "non-negative");
  }
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    const bool unset = config.hipYawRange.lower[foot] == 0.0 &&
                       config.hipYawRange.upper[foot] == 0.0;
    if (!unset && !(config.hipYawRange.lower[foot] < 0.0 &&
                    config.hipYawRange.upper[foot] > 0.0)) {
      return invalidConfig(absl::StrCat("hip_yaw_range of foot ", foot,
                                        " (derived from the URDF's hip yaw "
                                        "joint limits) must satisfy lower (",
                                        config.hipYawRange.lower[foot],
                                        ") < 0 < upper (",
                                        config.hipYawRange.upper[foot], ")"));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ContactPlanningConfig::validateStatus() const {
  RETURN_IF_ERROR(validatePlanner(*this));
  RETURN_IF_ERROR(validateShared(*this));
  RETURN_IF_ERROR(validateMixedIntegerGeometry(*this));
  RETURN_IF_ERROR(validateWeights(*this));
  RETURN_IF_ERROR(validateSearchStages(*this));
  RETURN_IF_ERROR(validateExecutionRules(*this));
  RETURN_IF_ERROR(validateHlip(*this));
  RETURN_IF_ERROR(validateModelDerived(*this));
  RETURN_IF_ERROR(formulation.validateStatus());

  // Combinations that are legal but documented to fall or block are reported
  // rather than rejected; see warnings().
  for (const std::string& warning : warnings()) {
    LOG(WARNING) << "[ContactPlanningConfig] " << warning;
  }
  return absl::OkStatus();
}

std::vector<std::string> ContactPlanningConfig::warnings() const {
  std::vector<std::string> out;
  const std::string plannerName = canonicalPlannerName(planner.type);
  if (plannerName == planner::kLipMiqp) appendLipMiqpWarnings(*this, out);
  if (plannerName == planner::kHlip) appendHlipWarnings(*this, out);
  return out;
}

scalar_t ContactPlanningConfig::shortestPlannedSwingDuration() const {
  if (canonicalPlannerName(planner.type) == planner::kHlip)
    return hlip.sspDuration;
  return static_cast<scalar_t>(minSwingNodes()) * planner.dt;
}

std::optional<std::string> ContactPlanningConfig::swingTimeScaleWarning(
    scalar_t swingTimeScale) const {
  const scalar_t shortest = shortestPlannedSwingDuration();
  if (swingTimeScale <= shortest + kTimeComparisonTolerance)
    return std::nullopt;
  const bool hlipPlanner = canonicalPlannerName(planner.type) == planner::kHlip;
  return absl::StrCat(
      "task.textproto swing_trajectory_config.swing_time_scale (",
      swingTimeScale, " s) exceeds the shortest swing the planner emits, ",
      hlipPlanner
          ? "hlip.ssp_duration"
          : "shared.gait_limits.min_swing_duration rounded up to planner.dt",
      " (", shortest,
      " s): every such swing is scaled down in height and velocity to ",
      shortest / swingTimeScale,
      " of its nominal and lands short and low. Lower swing_time_scale to at "
      "most ",
      shortest, " s or lengthen the swing.");
}

}  // namespace ocs2::humanoid
