/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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
#include <exception>
#include <filesystem>
#include <system_error>
#include <utility>

#include <boost/property_tree/ptree.hpp>

#include <ocs2_core/misc/LoadData.h>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"

namespace ocs2::humanoid {

namespace {

absl::Status invalidConfig(absl::string_view message) {
  return absl::InvalidArgumentError(absl::StrCat("[ContactPlanningConfig] ", message));
}

/** "`key` (value) must be `requirement`", the shape of every single-key rejection. */
absl::Status keyMustBe(absl::string_view key, scalar_t value, absl::string_view requirement) {
  return invalidConfig(absl::StrCat(key, " (", value, ") must be ", requirement));
}

/** The first rejection among the slack blocks of the soft constraints, or OK. */
absl::Status checkSlack(const std::optional<SlackPenalty>& slack, absl::string_view term) {
  if (!slack.has_value()) return absl::OkStatus();
  if (slack->quadratic < 0.0) return keyMustBe(absl::StrCat(term, ".slack.quadratic"), slack->quadratic, "non-negative");
  if (slack->linear < 0.0) return keyMustBe(absl::StrCat(term, ".slack.linear"), slack->linear, "non-negative");
  return absl::OkStatus();
}

/** Whether the mixed-integer planner runs the event-shift local search after its branch-and-bound. */
bool lipMiqpRunsLocalSearch(const ContactPlanningConfig& config) {
  return config.formulation.hasSearchStage(term::kEventShiftLocalSearch) && config.eventShiftLocalSearch.iterations > 0;
}

/**
 * Worst-case time one mixed-integer plan may take [s]: the branch-and-bound's anytime limit plus the local search's
 * own budget when that stage is listed. It is how long a synchronous planner blocks a solve, and the least a plan's
 * commit window has to cover when it is planned on the worker thread.
 */
scalar_t lipMiqpSolveBudget(const ContactPlanningConfig& config) {
  return config.planner.maxSolveTime + (lipMiqpRunsLocalSearch(config) ? config.eventShiftLocalSearch.maxTime : 0.0);
}

/** The keys lipMiqpSolveBudget() adds up, so that a message names exactly the ones that make up its number. */
std::string lipMiqpSolveBudgetKeys(const ContactPlanningConfig& config) {
  return lipMiqpRunsLocalSearch(config) ? "planner.maxSolveTime + event_shift_local_search.maxTime" : "planner.maxSolveTime";
}

/** The commit-boundary stall of an instantaneous exchange of support, which both planners can reach. */
std::string commitStallWarning(absl::string_view zeroDoubleSupportKey, scalar_t smallestCap) {
  return absl::StrCat(zeroDoubleSupportKey,
                      " is 0 and planner.maxCommitExtension <= 0 (no cap). Once the double supports are gone every swing starts "
                      "exactly where the previous one ends, so nothing stops the commit boundary from being extended to the end of "
                      "the stepping region: no plan reaches past it, none is merged and the robot stops stepping. Set "
                      "planner.maxCommitExtension to at least ",
                      smallestCap, " s.");
}

void appendLipMiqpWarnings(const ContactPlanningConfig& config, std::vector<std::string>& out) {
  const PlannerSettings& p = config.planner;
  const scalar_t budget = lipMiqpSolveBudget(config);
  const std::string budgetKeys = lipMiqpSolveBudgetKeys(config);
  // The branch-and-bound takes tens of milliseconds and may take its whole budget. Planned in the pre-solve hook it
  // stalls every MPC solve by that much; the integration tests do it on purpose, which is why this is not an error.
  if (!p.runInBackgroundThread) {
    out.push_back(
        absl::StrCat("planner.type: lip_miqp with planner.runInBackgroundThread: false. The mixed-integer search then runs "
                     "inside the MPC's pre-solve hook and blocks every solve for up to ",
                     budgetKeys, " = ", budget, " s. Set planner.runInBackgroundThread: true."));
  } else if (p.commitTime < budget - 1e-9) {
    // On the worker thread a plan reaches the solver no earlier than its solve time after the snapshot it was made
    // from, and one whose commit boundary has passed by then is dropped as stale (activatePendingPlan). A commit
    // window shorter than the budget therefore throws away exactly the plans that needed the budget.
    out.push_back(absl::StrCat("planner.commitTime (", p.commitTime, " s) is shorter than the solve budget ", budgetKeys, " (", budget,
                               " s): a plan that uses its budget arrives after its own commit boundary and is dropped as stale, and "
                               "while plans are being dropped the executed schedule runs out and the robot stops stepping. Raise "
                               "planner.commitTime or lower ",
                               budgetKeys, "."));
  }
  if (config.shared.gaitLimits.minDoubleSupportDuration <= 0.0 && p.maxCommitExtension <= 0.0) {
    out.push_back(commitStallWarning("shared.gait_limits.minDoubleSupportDuration", config.shared.gaitLimits.maxSwingDuration));
  }
}

void appendHlipWarnings(const ContactPlanningConfig& config, std::vector<std::string>& out) {
  const PlannerSettings& p = config.planner;
  const HlipParameters& h = config.hlip;

  // planned_com_override is the paper's reference plumbing (equations 11-14), not a heuristic. Without it the
  // whole-body MPC tracks the operator's straight-line center of mass while the footholds were placed for a lateral
  // orbit; the planner reads back a center of mass with no lateral velocity, narrows the step and the robot falls
  // (H-LIP README section 3c). The rule cannot be the library default because that default is shared with lip_miqp.
  if (!config.formulation.hasExecutionRule(term::kPlannedComOverride)) {
    out.push_back(absl::StrCat("planner.type: hlip without ", term::kPlannedComOverride,
                               " in the execution list. The H-LIP orbit needs the center of mass to fall towards the swing foot; without "
                               "the rule the whole-body MPC tracks the operator's straight-line reference instead, the planner reads back "
                               "a center of mass with no lateral velocity, narrows the step to hlip.minStepWidth and the robot sidesteps "
                               "and falls. Add ",
                               term::kPlannedComOverride, " to `execution`."));
  }

  // The H-LIP deadbeat step is a feedback law on the measured state, and a plan costs microseconds because nothing is
  // solved. Running it on a background thread at a fraction of the MPC rate therefore buys nothing and costs the one
  // thing the law depends on: a plan made at 10 Hz is made from a snapshot up to 100 ms old, which is two fifths of a
  // 0.25 s single support. This is a warning rather than an error because the threading is the operator's
  // call.
  if (p.runInBackgroundThread) {
    out.push_back(
        absl::StrCat("planner.type: hlip with planner.runInBackgroundThread: true. The closed-form H-LIP planner costs microseconds, so "
                     "the background thread only adds latency to a feedback law: its plan is made at most planner.planningFrequency (",
                     p.planningFrequency,
                     " Hz) times a second, from a snapshot up to a planning period old (on either path a plan is activated one MPC "
                     "cycle after it is made). Set planner.runInBackgroundThread: false to plan in the pre-solve hook at the MPC rate."));
  }

  if (h.dspDuration <= 0.0 && p.maxCommitExtension <= 0.0) {
    out.push_back(commitStallWarning("hlip.dspDuration", std::max(h.sspDuration, config.shared.gaitLimits.maxSwingDuration)));
  }

  // Everything below evaluates the H-LIP itself, whose constructor CHECK-fails on a cadence or a pendulum that
  // validateStatus() rejects, so an invalid configuration stops here instead of aborting the process.
  const scalar_t stepDuration = h.sspDuration + h.dspDuration;
  if (h.sspDuration <= 0.0 || h.dspDuration < 0.0 || config.shared.comHeight <= 0.0 || config.shared.gravity <= 0.0) return;

  // A robot standing still has its center of mass half a step width from the stance foot with no lateral velocity,
  // so the first single support ends far outside the orbit and the deadbeat law asks for a wide first step. If that
  // step does not fit hlip.maxStepWidth it is clipped and the step is no longer deadbeat: depending on how much is cut
  // the residual either dies out over a few steps or grows by cosh(omega sspDuration) every step until the gait locks
  // into a wide/narrow limit cycle that falls (H-LIP README section 3b). Which of the two happens is what the start-up
  // summary rolls out (HlipContactPlanner::startUpLateralWidths); a first step without margin is reported either way,
  // because the summary is the only place that said so and nothing checked it.
  const scalar_t startUpStep = HlipContactPlanner::startUpLateralStep(config);
  if (startUpStep > h.maxStepWidth) {
    out.push_back(
        absl::StrCat("the first step out of a standstill needs ", startUpStep,
                     " m of lateral step (HlipContactPlanner::startUpLateralStep) against hlip.maxStepWidth (", h.maxStepWidth,
                     " m): it is clipped, which costs the deadbeat property, and depending on how much is cut the gait either "
                     "recovers over a few steps or locks into an alternating wide/narrow limit cycle that falls (the planner's "
                     "start-up summary says which). Shorten hlip.sspDuration (or hlip.dspDuration), or raise hlip.maxStepWidth."));
  }

  // The step width bounds are checked by validateStatus() against hlip.stepWidth alone, which is the nominal orbit at a
  // ZERO lateral command. That is not the orbit the planner walks: HlipContactPlanner::deadbeatStep builds the
  // period-two orbit at +-stepWidth + v_y,cmd * (sspDuration + dspDuration) and then clips the placed foot into
  // [minStepWidth, maxStepWidth], so a sustained sidestep moves BOTH sides of the orbit by the drift of one step. The
  // configuration is only self-consistent when the narrowed side still clears the self-collision margin and the widened
  // side still fits the reach, at the largest lateral command the blend is normalized for. Otherwise the narrow step
  // is clipped on every cycle for as long as the command is held, ContactPlan::describe() prints CLIPPED-STEPS forever
  // and the one signal documented to mean "the legs cannot deliver this command at this cadence" becomes a standing
  // false positive that hides genuine clipping, while the realized lateral rate quietly runs below the command.
  const scalar_t lateralDrift = h.blend.maxCommandedVelocityY * stepDuration;
  if (h.stepWidth - lateralDrift < h.minStepWidth) {
    out.push_back(absl::StrCat("a sustained sidestep at hlip.blend.maxCommandedVelocityY (", h.blend.maxCommandedVelocityY, " m/s) drifts ",
                               lateralDrift, " m over a step of ", stepDuration,
                               " s, so the step that places the trailing foot is planned at hlip.stepWidth - drift = ",
                               h.stepWidth - lateralDrift, " m and clipped to hlip.minStepWidth (", h.minStepWidth,
                               " m) on every occurrence. Raise hlip.stepWidth, or lower hlip.blend.maxCommandedVelocityY below ",
                               (h.stepWidth - h.minStepWidth) / stepDuration, " m/s, or shorten the step."));
  }
  if (h.stepWidth + lateralDrift > h.maxStepWidth) {
    out.push_back(absl::StrCat("a sustained sidestep at hlip.blend.maxCommandedVelocityY (", h.blend.maxCommandedVelocityY, " m/s) drifts ",
                               lateralDrift, " m over a step of ", stepDuration,
                               " s, so the step that places the leading foot is planned at hlip.stepWidth + drift = ",
                               h.stepWidth + lateralDrift, " m and clipped to hlip.maxStepWidth (", h.maxStepWidth,
                               " m) on every occurrence. Raise hlip.maxStepWidth, or lower hlip.blend.maxCommandedVelocityY below ",
                               (h.maxStepWidth - h.stepWidth) / stepDuration, " m/s, or shorten the step."));
  }
}

}  // namespace

absl::Status ContactPlanningConfig::validateStatus() const {
  const PlannerSettings& p = planner;
  const SharedParameters& s = shared;
  const GaitLimits& g = s.gaitLimits;

  // ---- planner ----
  if (p.dt <= 0.0) return keyMustBe("planner.dt", p.dt, "positive");
  if (p.numNodes < 2) return keyMustBe("planner.numNodes", p.numNodes, "at least 2");
  // planner.type selects the implementation ContactPlannerFactory builds, and every consumer of this configuration
  // treats "validation passed" as "this configuration can be applied": loadContactPlanningConfig and
  // ContactPlanningReferenceManager::setConfig delegate their rejection here, and the parameter updater reports a
  // reload that raised nothing as applied. An unknown name used to be noticed only by the factory, one layer too
  // late: by then ContactPlannerModule had already stored the whole new parameter set, rebuilt the execution rules and
  // started or stopped the background worker. Rejecting the name here makes the reload fail atomically with the
  // previous configuration still in force, and the message lists the planners that exist.
  if (canonicalPlannerName(p.type).empty()) {
    return invalidConfig(absl::StrCat("unknown planner.type '", p.type, "'; supported: ", absl::StrJoin(knownPlannerNames(), ", ")));
  }
  if (p.commitTime < 0.0) return keyMustBe("planner.commitTime", p.commitTime, "non-negative");
  if (commitNodes() >= p.numNodes) {
    return invalidConfig(absl::StrCat("planner.commitTime (", p.commitTime,
                                      " s) must be shorter than the planning horizon planner.numNodes * ", "planner.dt (", horizon(),
                                      " s)"));
  }
  if (p.maxCommitExtension > 0.0 && p.maxCommitExtension < g.maxSwingDuration) {
    return invalidConfig(absl::StrCat("planner.maxCommitExtension (", p.maxCommitExtension,
                                      " s) must be 0 (no cap) or at least shared.gait_limits.maxSwingDuration (", g.maxSwingDuration,
                                      " s), so a whole swing still fits in it"));
  }
  if (p.maxBranchAndBoundNodes < 1) return keyMustBe("planner.maxBranchAndBoundNodes", p.maxBranchAndBoundNodes, "at least 1");
  if (p.maxSolveTime <= 0.0) return keyMustBe("planner.maxSolveTime", p.maxSolveTime, "positive");
  if (p.maxQpIterations < 1) return keyMustBe("planner.maxQpIterations", p.maxQpIterations, "at least 1");
  if (p.planningFrequency <= 0.0) return keyMustBe("planner.planningFrequency", p.planningFrequency, "positive");

  // ---- shared ----
  if (s.gravity <= 0.0) return keyMustBe("shared.gravity", s.gravity, "positive");
  if (s.comHeight <= 0.0) return keyMustBe("shared.comHeight", s.comHeight, "positive (0 is resolved from the model before validation)");
  if (g.minSwingDuration <= 0.0) return keyMustBe("shared.gait_limits.minSwingDuration", g.minSwingDuration, "positive");
  if (g.maxSwingDuration < g.minSwingDuration) {
    return invalidConfig(absl::StrCat("shared.gait_limits.maxSwingDuration (", g.maxSwingDuration,
                                      " s) must be at least shared.gait_limits.minSwingDuration (", g.minSwingDuration, " s)"));
  }
  if (g.minContactDuration <= 0.0) return keyMustBe("shared.gait_limits.minContactDuration", g.minContactDuration, "positive");
  if (g.maxContactDuration > 0.0 && g.maxContactDuration < g.minContactDuration) {
    return invalidConfig(absl::StrCat("shared.gait_limits.maxContactDuration (", g.maxContactDuration,
                                      " s) must be 0 (no limit) or at least shared.gait_limits.minContactDuration (", g.minContactDuration,
                                      " s)"));
  }
  if (g.minDoubleSupportDuration < 0.0) {
    return keyMustBe("shared.gait_limits.minDoubleSupportDuration", g.minDoubleSupportDuration, "non-negative");
  }
  if (s.slackPenalty.quadratic < 0.0) return keyMustBe("shared.slack_penalty.quadratic", s.slackPenalty.quadratic, "non-negative");
  if (s.slackPenalty.linear < 0.0) return keyMustBe("shared.slack_penalty.linear", s.slackPenalty.linear, "non-negative");

  // ---- geometry of the mixed-integer terms ----
  if (zmpSupportRegion.halfWidthX <= 0.0) return keyMustBe("zmp_support_region.halfWidthX", zmpSupportRegion.halfWidthX, "positive");
  if (zmpSupportRegion.halfWidthY <= 0.0) return keyMustBe("zmp_support_region.halfWidthY", zmpSupportRegion.halfWidthY, "positive");
  if (footSeparation.minStepWidth <= 0.0) return keyMustBe("foot_separation.minStepWidth", footSeparation.minStepWidth, "positive");
  if (footSeparation.maxStepWidth < footSeparation.minStepWidth) {
    return invalidConfig(absl::StrCat("foot_separation.maxStepWidth (", footSeparation.maxStepWidth,
                                      " m) must be at least foot_separation.minStepWidth (", footSeparation.minStepWidth, " m)"));
  }
  if (stepWidth.nominalStepWidth < footSeparation.minStepWidth || stepWidth.nominalStepWidth > footSeparation.maxStepWidth) {
    return invalidConfig(absl::StrCat("step_width.nominalStepWidth (", stepWidth.nominalStepWidth,
                                      " m) must lie within [foot_separation.minStepWidth, foot_separation.maxStepWidth] = [",
                                      footSeparation.minStepWidth, ", ", footSeparation.maxStepWidth, "] m"));
  }
  if (footSeparation.maxStepLength <= 0.0) return keyMustBe("foot_separation.maxStepLength", footSeparation.maxStepLength, "positive");
  if (reachability.reachX <= 0.0) return keyMustBe("reachability.reachX", reachability.reachX, "positive");
  if (reachability.reachYOuter <= reachability.reachYInner) {
    return invalidConfig(absl::StrCat("reachability.reachYOuter (", reachability.reachYOuter, " m) must exceed reachability.reachYInner (",
                                      reachability.reachYInner, " m)"));
  }
  if (s.bigM <= footSeparation.maxStepLength) {
    return invalidConfig(
        absl::StrCat("shared.bigM (", s.bigM, " m) must exceed foot_separation.maxStepLength (", footSeparation.maxStepLength, " m)"));
  }
  // The big-M must also cover the LATERAL separation of the feet, and that, not the heading axis guarded above, is the
  // requirement that actually binds. In double support both contact binaries are one, so the +M c_i - M c_other terms
  // of a single-support ZMP box cancel and that box relaxes to only |e_y'(zmp - p_i)| <= halfWidthY + M, while the
  // double-support rows open the whole strip between the feet, e_y'p_R - halfWidthY <= e_y'zmp <= e_y'p_L + halfWidthY
  // (ZmpSupportRegionConstraint::addRows). Intersecting the two, the opposite foot's relaxed row binds as soon as M is
  // smaller than the separation w = e_y'(p_L - p_R), which foot_separation.maxStepWidth bounds from above; the
  // half-width cancels, so the exact condition is bigM >= maxStepWidth. Below it the ZMP is pulled towards one foot in
  // double support and every weight transfer past that point pays slack for a disjunction that was supposed to be
  // switched off.
  if (s.bigM < footSeparation.maxStepWidth) {
    return invalidConfig(absl::StrCat("shared.bigM (", s.bigM, " m) must be at least foot_separation.maxStepWidth (",
                                      footSeparation.maxStepWidth,
                                      " m), otherwise the relaxed single-support ZMP box clips the double-support region laterally"));
  }
  for (const absl::Status& slack :
       {checkSlack(zmpSupportRegion.slack, term::kZmpSupportRegion), checkSlack(reachability.slack, term::kReachability),
        checkSlack(footSeparation.slack, term::kFootSeparation), checkSlack(hipYawRange.slack, term::kHipYawRange)}) {
    if (!slack.ok()) return slack;
  }

  // ---- weights and costs, one key per message ----
  const std::array<std::pair<const char*, scalar_t>, 20> nonNegative{{
      {"regularization.state", regularization.state},
      {"regularization.input", regularization.input},
      {"previous_foothold_consistency.weight", previousFootholdConsistency.weight},
      {"velocity_tracking.weight", velocityTracking.weight},
      {"step_width.weight", stepWidth.weight},
      {"heading_rate_tracking.weight", headingRateTracking.weight},
      {"heading_tracking.weight", headingTracking.weight},
      {"foot_yaw_tracking.weight", footYawTracking.weight},
      {"yaw_torque_regularization.weight", yawTorqueRegularization.weight},
      {"foot_yaw_regularization.weight", footYawRegularization.weight},
      {"zmp_regularization.weight", zmpRegularization.weight},
      {"foothold_regularization.weight", footholdRegularization.weight},
      {"step_length.weight", stepLength.weight},
      {"terminal_dcm.weight", terminalDcm.weight},
      {"contact_switch.cost", contactSwitch.cost},
      {"plan_consistency.cost", planConsistency.cost},
      {"double_support_penalty.cost", doubleSupportPenalty.cost},
      {"event_shift_local_search.maxTime", eventShiftLocalSearch.maxTime},
      {"energy_cadence_modulation.deadband", energyCadenceModulation.deadband},
      {"energy_cadence_modulation.gain", energyCadenceModulation.gain},
  }};
  for (const std::pair<const char*, scalar_t>& entry : nonNegative) {
    if (entry.second < 0.0) return keyMustBe(entry.first, entry.second, "non-negative");
  }

  // ---- search stages ----
  if (eventShiftLocalSearch.iterations < 0) {
    return keyMustBe("event_shift_local_search.iterations", eventShiftLocalSearch.iterations, "non-negative");
  }
  if (diving.maxDiveIterations < 1) return keyMustBe("diving.maxDiveIterations", diving.maxDiveIterations, "at least 1");
  if (cadenceStretch.samples < 0) return keyMustBe("cadence_stretch.samples", cadenceStretch.samples, "non-negative");
  if (cadenceStretch.samples > 0 && cadenceStretch.maxStretch < 1.0) {
    return keyMustBe("cadence_stretch.maxStretch", cadenceStretch.maxStretch,
                     "at least 1: a stretch below 1 shrinks the commit window and the horizon");
  }
  if (headingRelinearization.passes < 0 || headingRelinearization.passes > 5) {
    return keyMustBe("heading_relinearization.passes", headingRelinearization.passes, "in [0, 5]");
  }

  // ---- execution rules ----
  const PhaseResettingParameters& r = phaseResetting;
  if (r.earlyTouchdownMinSwingRatio < 0.0 || r.earlyTouchdownMinSwingRatio > 1.0) {
    return keyMustBe("phase_resetting.earlyTouchdownMinSwingRatio", r.earlyTouchdownMinSwingRatio, "in [0, 1]");
  }
  if (r.earlyTouchdownMinContactDuration < 0.0) {
    return keyMustBe("phase_resetting.earlyTouchdownMinContactDuration", r.earlyTouchdownMinContactDuration, "non-negative");
  }
  if (r.earlyTouchdownMinAdvance < 0.0)
    return keyMustBe("phase_resetting.earlyTouchdownMinAdvance", r.earlyTouchdownMinAdvance, "non-negative");
  if (r.maxLateTouchdownExtension < 0.0) {
    return keyMustBe("phase_resetting.maxLateTouchdownExtension", r.maxLateTouchdownExtension, "non-negative");
  }
  if (r.lateTouchdownExtensionStep <= 0.0) {
    return keyMustBe("phase_resetting.lateTouchdownExtensionStep", r.lateTouchdownExtensionStep, "positive");
  }
  if (r.lateTouchdownSearchVelocity < 0.0) {
    return keyMustBe("phase_resetting.lateTouchdownSearchVelocity", r.lateTouchdownSearchVelocity, "non-negative");
  }
  if (dcmStepAdjustment.gain < 0.0) return keyMustBe("dcm_step_adjustment.gain", dcmStepAdjustment.gain, "non-negative");
  if (dcmStepAdjustment.maxOffset < 0.0) return keyMustBe("dcm_step_adjustment.maxOffset", dcmStepAdjustment.maxOffset, "non-negative");

  // ---- hlip ----
  const HlipParameters& h = hlip;
  if (h.sspDuration <= 0.0) return keyMustBe("hlip.sspDuration", h.sspDuration, "positive");
  if (h.dspDuration < 0.0) return keyMustBe("hlip.dspDuration", h.dspDuration, "non-negative");
  if (h.stepWidth <= 0.0) return keyMustBe("hlip.stepWidth", h.stepWidth, "positive");
  if (h.maxStepLength <= 0.0) return keyMustBe("hlip.maxStepLength", h.maxStepLength, "positive");
  if (h.minStepWidth <= 0.0) return keyMustBe("hlip.minStepWidth", h.minStepWidth, "positive");
  if (h.maxStepWidth < h.minStepWidth) {
    return invalidConfig(
        absl::StrCat("hlip.maxStepWidth (", h.maxStepWidth, " m) must be at least hlip.minStepWidth (", h.minStepWidth, " m)"));
  }
  if (h.stepWidth < h.minStepWidth || h.stepWidth > h.maxStepWidth) {
    return invalidConfig(absl::StrCat("hlip.stepWidth (", h.stepWidth, " m) must lie within [hlip.minStepWidth, hlip.maxStepWidth] = [",
                                      h.minStepWidth, ", ", h.maxStepWidth, "] m"));
  }
  if (h.blend.sharpness <= 0.0) return keyMustBe("hlip.blend.sharpness", h.blend.sharpness, "positive");
  // rho_2. The activity phi is a sum of squared ratios and so is never negative, and isWalking compares it with
  // `>=`, so a threshold of zero declares the robot to be walking at rest: standing becomes unreachable and the gait
  // marches in place forever.
  if (h.blend.threshold <= 0.0)
    return keyMustBe("hlip.blend.threshold", h.blend.threshold, "positive, or the blend can never reach standing");
  const std::array<std::pair<const char*, scalar_t>, 5> blendRanges{{
      {"hlip.blend.maxCommandedVelocityX", h.blend.maxCommandedVelocityX},
      {"hlip.blend.maxCommandedVelocityY", h.blend.maxCommandedVelocityY},
      {"hlip.blend.maxCommandedYawRate", h.blend.maxCommandedYawRate},
      {"hlip.blend.maxComVelocityX", h.blend.maxComVelocityX},
      {"hlip.blend.maxComVelocityY", h.blend.maxComVelocityY},
  }};
  for (const std::pair<const char*, scalar_t>& entry : blendRanges) {
    if (entry.second <= 0.0) return keyMustBe(entry.first, entry.second, "positive");
  }

  // ---- derived from the model, not file keys ----
  if (yawTorqueBudget.torsionalFrictionTorque < 0.0) {
    return keyMustBe("yaw_torque_budget torsional friction torque (derived from task.yaml contactWrenchConeSoftConstraint)",
                     yawTorqueBudget.torsionalFrictionTorque, "non-negative");
  }
  if (yawTorqueBudget.doubleSupportYawCouple < 0.0) {
    return keyMustBe("yaw_torque_budget double-support yaw couple (derived from task.yaml contactWrenchConeSoftConstraint)",
                     yawTorqueBudget.doubleSupportYawCouple, "non-negative");
  }
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const bool unset = hipYawRange.lower[foot] == 0.0 && hipYawRange.upper[foot] == 0.0;
    if (!unset && !(hipYawRange.lower[foot] < 0.0 && hipYawRange.upper[foot] > 0.0)) {
      return invalidConfig(absl::StrCat("hip_yaw_range of foot ", foot,
                                        " (derived from the URDF's hip yaw joint limits) must satisfy lower (", hipYawRange.lower[foot],
                                        ") < 0 < upper (", hipYawRange.upper[foot], ")"));
    }
  }

  RETURN_IF_ERROR(formulation.validateStatus());

  // Combinations that are legal but documented to fall or block are reported rather than rejected; see warnings().
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
  if (canonicalPlannerName(planner.type) == planner::kHlip) return hlip.sspDuration;
  return static_cast<scalar_t>(minSwingNodes()) * planner.dt;
}

std::optional<std::string> ContactPlanningConfig::swingTimeScaleWarning(scalar_t swingTimeScale) const {
  const scalar_t shortest = shortestPlannedSwingDuration();
  if (swingTimeScale <= shortest + 1e-9) return std::nullopt;
  const bool hlipPlanner = canonicalPlannerName(planner.type) == planner::kHlip;
  return absl::StrCat("task.yaml swing_trajectory_config.swingTimeScale (", swingTimeScale,
                      " s) exceeds the shortest swing the planner emits, ",
                      hlipPlanner ? "hlip.sspDuration" : "shared.gait_limits.minSwingDuration rounded up to planner.dt", " (", shortest,
                      " s): every such swing is scaled down in height and velocity to ", shortest / swingTimeScale,
                      " of its nominal and lands short and low. Lower swingTimeScale to at most ", shortest, " s or lengthen the swing.");
}

std::string resolveContactPlanningConfigFile(const std::string& taskFile) {
  const std::filesystem::path sibling = std::filesystem::path(taskFile).parent_path() / kContactPlanningConfigFileName;
  std::error_code ec;
  if (std::filesystem::is_regular_file(sibling, ec)) return sibling.string();
  return taskFile;
}

namespace {

using boost::property_tree::ptree;

/** Keys of the previous, flat layout: their presence directly under the block identifies a file that was not migrated. */
constexpr std::array<const char*, 61> kLegacyKeys{"dt",
                                                  "numNodes",
                                                  "commitTime",
                                                  "comHeight",
                                                  "gravity",
                                                  "minSwingDuration",
                                                  "maxSwingDuration",
                                                  "minContactDuration",
                                                  "maxContactDuration",
                                                  "enforceAlternatingFeet",
                                                  "minDoubleSupportDuration",
                                                  "zmpHalfWidthX",
                                                  "zmpHalfWidthY",
                                                  "nominalStepWidth",
                                                  "minStepWidth",
                                                  "maxStepWidth",
                                                  "maxStepLength",
                                                  "reachX",
                                                  "reachYInner",
                                                  "reachYOuter",
                                                  "bigM",
                                                  "velocityTrackingWeight",
                                                  "zmpRegularizationWeight",
                                                  "footholdRegularizationWeight",
                                                  "stepWidthWeight",
                                                  "contactSwitchCost",
                                                  "planConsistencyCost",
                                                  "previousFootholdWeight",
                                                  "terminalDcmWeight",
                                                  "constraintSlackWeight",
                                                  "constraintSlackLinearWeight",
                                                  "maxBranchAndBoundNodes",
                                                  "maxSolveTime",
                                                  "maxQpIterations",
                                                  "localSearchIterations",
                                                  "localSearchMaxTime",
                                                  "verbose",
                                                  "runInBackgroundThread",
                                                  "planningFrequency",
                                                  "enablePhaseResetting",
                                                  "earlyTouchdownMinSwingRatio",
                                                  "earlyTouchdownMinContactDuration",
                                                  "maxLateTouchdownExtension",
                                                  "lateTouchdownExtensionStep",
                                                  "lateTouchdownSearchVelocity",
                                                  "enableDcmStepAdjustment",
                                                  "dcmAdjustmentGain",
                                                  "dcmAdjustmentMaxOffset",
                                                  "enableEnergyCadenceModulation",
                                                  "energyCadenceGain",
                                                  "energyCadenceDeadband",
                                                  "useAcomDynamics",
                                                  "headingRateTrackingWeight",
                                                  "headingTrackingWeight",
                                                  "yawTorqueWeight",
                                                  "footYawTrackingWeight",
                                                  "footYawRegularizationWeight",
                                                  "headingLinearizationPasses",
                                                  "planHeadingOverridesTarget",
                                                  "torsionalFrictionTorque",
                                                  "doubleSupportYawCouple"};

/** Keys of the structured layout that identify it (the block names besides the term blocks). */
constexpr std::array<const char*, 11> kStructuredKeys{"planner",          "shared",           "hlip",        "dynamics",         "costs",
                                                      "soft_constraints", "hard_constraints", "logic_rules", "assignment_costs", "search",
                                                      "execution"};

/** The keys of `keys` present directly under `block`, in the order of `keys`. */
template <size_t N>
std::vector<std::string> presentKeys(const ptree& block, const std::array<const char*, N>& keys) {
  std::vector<std::string> present;
  for (const char* key : keys) {
    if (block.get_child_optional(key)) present.emplace_back(key);
  }
  return present;
}

bool hasAnyTermBlock(const ptree& block) {
  for (const TermKind kind : allTermKinds()) {
    for (const std::string& name : knownTermNames(kind)) {
      if (block.get_child_optional(name)) return true;
    }
  }
  return false;
}

/**
 * Reads the keys of one file into a configuration. The first value that does not parse is kept as the loader's status,
 * naming its key; loadPtreeValue itself only reports the type it failed to convert to.
 */
class KeyLoader {
 public:
  KeyLoader(const ptree& pt, absl::string_view prefix, bool verbose) : pt_(pt), prefix_(prefix), verbose_(verbose) {}

  /** Overwrites `value` with `key` when the file has it; a missing key keeps the value it had. */
  template <typename T>
  void operator()(T& value, absl::string_view key) {
    const std::string path = absl::StrCat(prefix_, key);
    try {
      loadData::loadPtreeValue(pt_, value, path, verbose_);
    } catch (const boost::property_tree::ptree_error& error) {
      if (status_.ok()) status_ = absl::InvalidArgumentError(absl::StrCat(path, " cannot be read: ", error.what()));
    }
  }

  /**
   * The `slack` block of one term, if it has one. `sharedDefault` is `shared.slack_penalty` as already loaded from the
   * file, and it seeds the pair: a key that is absent leaves its destination untouched, so a block that supplies only
   * one of the two numbers must inherit the other from the shared default. Seeding from a default-constructed
   * SlackPenalty instead silently substituted the hard-coded struct value (1e4 / 100) for the half the operator
   * omitted.
   */
  void slack(absl::string_view term, const SlackPenalty& sharedDefault, std::optional<SlackPenalty>& slack) {
    if (!pt_.get_child_optional(absl::StrCat(prefix_, term, ".slack"))) return;
    SlackPenalty penalty = sharedDefault;
    (*this)(penalty.quadratic, absl::StrCat(term, ".slack.quadratic"));
    (*this)(penalty.linear, absl::StrCat(term, ".slack.linear"));
    slack = penalty;
  }

  const absl::Status& status() const { return status_; }

 private:
  const ptree& pt_;
  const std::string prefix_;
  const bool verbose_;
  absl::Status status_;
};

/** Replaces `target` with the YAML sequence under `key` when the block has that key; an absent key keeps the default. */
void loadList(const ptree& block, const char* key, std::vector<std::string>& target) {
  const boost::optional<const ptree&> child = block.get_child_optional(key);
  if (!child) return;
  std::vector<std::string> list;
  for (const ptree::value_type& item : *child) {
    const std::string value = item.second.data();
    if (!value.empty()) list.push_back(value);
  }
  target = std::move(list);
}

absl::Status loadStructured(const ptree& pt, const ptree& block, absl::string_view prefix, ContactPlanningConfig& config, bool verbose) {
  KeyLoader load(pt, prefix, verbose);
  // LINT.IfChange(contact_planning_keys)
  PlannerSettings& p = config.planner;
  load(p.dt, "planner.dt");
  load(p.numNodes, "planner.numNodes");
  load(p.commitTime, "planner.commitTime");
  load(p.maxCommitExtension, "planner.maxCommitExtension");
  load(p.maxBranchAndBoundNodes, "planner.maxBranchAndBoundNodes");
  load(p.maxSolveTime, "planner.maxSolveTime");
  load(p.maxQpIterations, "planner.maxQpIterations");
  load(p.runInBackgroundThread, "planner.runInBackgroundThread");
  load(p.planningFrequency, "planner.planningFrequency");
  load(p.verbose, "planner.verbose");
  load(p.logPlans, "planner.logPlans");
  load(p.type, "planner.type");

  SharedParameters& s = config.shared;
  load(s.gravity, "shared.gravity");
  load(s.comHeight, "shared.comHeight");
  load(s.bigM, "shared.bigM");
  load(s.slackPenalty.quadratic, "shared.slack_penalty.quadratic");
  load(s.slackPenalty.linear, "shared.slack_penalty.linear");
  load(s.gaitLimits.minSwingDuration, "shared.gait_limits.minSwingDuration");
  load(s.gaitLimits.maxSwingDuration, "shared.gait_limits.maxSwingDuration");
  load(s.gaitLimits.minContactDuration, "shared.gait_limits.minContactDuration");
  load(s.gaitLimits.maxContactDuration, "shared.gait_limits.maxContactDuration");
  load(s.gaitLimits.minDoubleSupportDuration, "shared.gait_limits.minDoubleSupportDuration");

  HlipParameters& h = config.hlip;
  load(h.sspDuration, "hlip.sspDuration");
  load(h.dspDuration, "hlip.dspDuration");
  load(h.stepWidth, "hlip.stepWidth");
  load(h.maxStepLength, "hlip.maxStepLength");
  load(h.maxStepWidth, "hlip.maxStepWidth");
  load(h.minStepWidth, "hlip.minStepWidth");
  load(h.blend.sharpness, "hlip.blend.sharpness");
  load(h.blend.threshold, "hlip.blend.threshold");
  load(h.blend.maxCommandedVelocityX, "hlip.blend.maxCommandedVelocityX");
  load(h.blend.maxCommandedVelocityY, "hlip.blend.maxCommandedVelocityY");
  load(h.blend.maxCommandedYawRate, "hlip.blend.maxCommandedYawRate");
  load(h.blend.maxComVelocityX, "hlip.blend.maxComVelocityX");
  load(h.blend.maxComVelocityY, "hlip.blend.maxComVelocityY");

  ContactPlanningFormulation& f = config.formulation;
  for (const TermKind kind : allTermKinds()) {
    loadList(block, termKindName(kind).c_str(), f.list(kind));
  }

  load(config.regularization.state, absl::StrCat(term::kRegularization, ".state"));
  load(config.regularization.input, absl::StrCat(term::kRegularization, ".input"));
  load(config.previousFootholdConsistency.weight, absl::StrCat(term::kPreviousFootholdConsistency, ".weight"));
  load(config.velocityTracking.weight, absl::StrCat(term::kVelocityTracking, ".weight"));
  load(config.stepWidth.weight, absl::StrCat(term::kStepWidth, ".weight"));
  load(config.stepWidth.nominalStepWidth, absl::StrCat(term::kStepWidth, ".nominalStepWidth"));
  load(config.headingRateTracking.weight, absl::StrCat(term::kHeadingRateTracking, ".weight"));
  load(config.headingTracking.weight, absl::StrCat(term::kHeadingTracking, ".weight"));
  load(config.footYawTracking.weight, absl::StrCat(term::kFootYawTracking, ".weight"));
  load(config.yawTorqueRegularization.weight, absl::StrCat(term::kYawTorqueRegularization, ".weight"));
  load(config.footYawRegularization.weight, absl::StrCat(term::kFootYawRegularization, ".weight"));
  load(config.zmpRegularization.weight, absl::StrCat(term::kZmpRegularization, ".weight"));
  load(config.footholdRegularization.weight, absl::StrCat(term::kFootholdRegularization, ".weight"));
  load(config.stepLength.weight, absl::StrCat(term::kStepLength, ".weight"));
  load(config.terminalDcm.weight, absl::StrCat(term::kTerminalDcm, ".weight"));
  load(config.terminalDcm.trackCommandedVelocity, absl::StrCat(term::kTerminalDcm, ".trackCommandedVelocity"));
  load(config.zmpSupportRegion.halfWidthX, absl::StrCat(term::kZmpSupportRegion, ".halfWidthX"));
  load(config.zmpSupportRegion.halfWidthY, absl::StrCat(term::kZmpSupportRegion, ".halfWidthY"));
  load.slack(term::kZmpSupportRegion, s.slackPenalty, config.zmpSupportRegion.slack);
  load(config.reachability.reachX, absl::StrCat(term::kReachability, ".reachX"));
  load(config.reachability.reachYInner, absl::StrCat(term::kReachability, ".reachYInner"));
  load(config.reachability.reachYOuter, absl::StrCat(term::kReachability, ".reachYOuter"));
  load.slack(term::kReachability, s.slackPenalty, config.reachability.slack);
  load(config.footSeparation.maxStepLength, absl::StrCat(term::kFootSeparation, ".maxStepLength"));
  load(config.footSeparation.minStepWidth, absl::StrCat(term::kFootSeparation, ".minStepWidth"));
  load(config.footSeparation.maxStepWidth, absl::StrCat(term::kFootSeparation, ".maxStepWidth"));
  load.slack(term::kFootSeparation, s.slackPenalty, config.footSeparation.slack);
  load.slack(term::kHipYawRange, s.slackPenalty, config.hipYawRange.slack);
  load(config.contactSwitch.cost, absl::StrCat(term::kContactSwitch, ".cost"));
  load(config.doubleSupportPenalty.cost, absl::StrCat(term::kDoubleSupportPenalty, ".cost"));
  load(config.planConsistency.cost, absl::StrCat(term::kPlanConsistency, ".cost"));
  load(config.diving.maxDiveIterations, absl::StrCat(term::kDiving, ".maxDiveIterations"));
  load(config.eventShiftLocalSearch.iterations, absl::StrCat(term::kEventShiftLocalSearch, ".iterations"));
  load(config.eventShiftLocalSearch.maxTime, absl::StrCat(term::kEventShiftLocalSearch, ".maxTime"));
  load(config.cadenceStretch.samples, absl::StrCat(term::kCadenceStretch, ".samples"));
  load(config.cadenceStretch.maxStretch, absl::StrCat(term::kCadenceStretch, ".maxStretch"));
  load(config.headingRelinearization.passes, absl::StrCat(term::kHeadingRelinearization, ".passes"));
  load(config.phaseResetting.earlyTouchdownMinSwingRatio, absl::StrCat(term::kPhaseResetting, ".earlyTouchdownMinSwingRatio"));
  load(config.phaseResetting.earlyTouchdownMinContactDuration, absl::StrCat(term::kPhaseResetting, ".earlyTouchdownMinContactDuration"));
  load(config.phaseResetting.maxLateTouchdownExtension, absl::StrCat(term::kPhaseResetting, ".maxLateTouchdownExtension"));
  load(config.phaseResetting.lateTouchdownExtensionStep, absl::StrCat(term::kPhaseResetting, ".lateTouchdownExtensionStep"));
  load(config.phaseResetting.earlyTouchdownMinAdvance, absl::StrCat(term::kPhaseResetting, ".earlyTouchdownMinAdvance"));
  load(config.phaseResetting.lateTouchdownSearchVelocity, absl::StrCat(term::kPhaseResetting, ".lateTouchdownSearchVelocity"));
  load(config.energyCadenceModulation.gain, absl::StrCat(term::kEnergyCadenceModulation, ".gain"));
  load(config.energyCadenceModulation.deadband, absl::StrCat(term::kEnergyCadenceModulation, ".deadband"));
  load(config.dcmStepAdjustment.gain, absl::StrCat(term::kDcmStepAdjustment, ".gain"));
  load(config.dcmStepAdjustment.maxOffset, absl::StrCat(term::kDcmStepAdjustment, ".maxOffset"));
  // Each robot's file is split into several blocks, because ifttt-lint cannot nest the small cross-file pairs (the
  // pendulum against task.yaml's DCM cost, the cadences against swingTimeScale) inside one file-wide block. Every block
  // that holds keys read here is named, so a key added above has to be documented in each robot's file.
  // clang-format off
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/contact_planning.yaml:contact_planning_config, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/contact_planning.yaml:atlas_pendulum, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/contact_planning.yaml:atlas_gait_cadence, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/contact_planning.yaml:atlas_hlip_cadence, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/contact_planning.yaml:contact_planning_config_tail, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/contact_planning.yaml:contact_planning_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/contact_planning.yaml:sa01_pendulum, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/contact_planning.yaml:sa01_gait_cadence, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/contact_planning.yaml:sa01_hlip_cadence, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/contact_planning.yaml:contact_planning_config_tail)
  // clang-format on
  return load.status();
}

}  // namespace

absl::StatusOr<ContactPlanningConfig> loadContactPlanningConfigStatus(absl::string_view yamlFile,
                                                                      absl::string_view prefix,
                                                                      bool verbose,
                                                                      bool validate) {
  const std::string file(yamlFile);
  ptree pt;
  try {
    loadData::readPropertyTree(file, pt);
  } catch (const std::exception& error) {
    return invalidConfig(absl::StrCat("cannot read ", file, ": ", error.what()));
  }
  ContactPlanningConfig config;
  if (verbose) {
    LOG(INFO) << "\n #### Contact Planning Config:";
    LOG(INFO) << "\n #### =============================================================================\n";
  }
  // "contact_planning." -> "contact_planning"
  const std::string blockKey = prefix.empty() ? std::string() : std::string(prefix.substr(0, prefix.size() - 1));
  boost::optional<const ptree&> block;
  if (blockKey.empty()) {
    block = pt;
  } else {
    const boost::optional<const ptree&> child = std::as_const(pt).get_child_optional(blockKey);
    if (child) block = *child;
  }
  if (block) {
    const std::vector<std::string> legacy = presentKeys(*block, kLegacyKeys);
    if (!legacy.empty()) {
      const bool structured = !presentKeys(*block, kStructuredKeys).empty() || hasAnyTermBlock(*block);
      return invalidConfig(absl::StrCat(
          file, structured ? " mixes the structured contact_planning layout with" : " uses",
          " keys of the flat layout of the previous planner, which is no longer read: ", blockKey, ".{", absl::StrJoin(legacy, ", "),
          "}. Migrate the block to the structured layout: `planner` / `shared` blocks, the term lists (dynamics, costs, soft_constraints, "
          "hard_constraints, logic_rules, assignment_costs, search, execution) and one parameter block per term, as in the DRC Atlas "
          "contact_planning.yaml; the flags became list entries (useAcomDynamics -> heading_double_integrator and its terms, "
          "enablePhaseResetting -> phase_resetting, ...)."));
    }
    const absl::Status loaded = loadStructured(pt, *block, prefix, config, verbose);
    if (!loaded.ok()) return invalidConfig(absl::StrCat(file, ": ", loaded.message()));
  }
  if (verbose) {
    LOG(INFO) << " #### =============================================================================";
  }
  if (validate) {
    RETURN_IF_ERROR(config.validateStatus());
  }
  return config;
}

}  // namespace ocs2::humanoid
