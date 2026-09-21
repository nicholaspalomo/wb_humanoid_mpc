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

#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <boost/property_tree/ptree.hpp>

#include <ocs2_core/misc/LoadData.h>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"

namespace ocs2::humanoid {

void ContactPlanningConfig::validate() const {
  const auto fail = [](const std::string& what) { throw std::invalid_argument("[ContactPlanningConfig] " + what); };
  const PlannerSettings& p = planner;
  const SharedParameters& s = shared;
  const GaitLimits& g = s.gaitLimits;
  if (p.dt <= 0.0) fail("planner.dt must be positive");
  if (p.numNodes < 2) fail("planner.numNodes must be at least 2");
  // planner.type selects the implementation ContactPlannerFactory builds, and every consumer of this configuration
  // treats "validate() did not throw" as "this configuration can be applied": loadContactPlanningConfig and
  // ContactPlanningReferenceManager::setConfig delegate their rejection here, and the parameter updater reports a
  // reload that raised nothing as applied. Until now an unknown name was noticed only by the factory, one layer too
  // late: by then ContactPlannerModule had already stored the whole new parameter set, rebuilt the execution rules and
  // started or stopped the background worker, and the factory's failed status was logged and discarded, so the new
  // parameters were handed to the planner the configuration no longer names and nothing retried until the next reload.
  // Rejecting the name here makes the reload fail atomically with the previous configuration still in force, and the
  // message lists the planners that exist so that a typo in the task file carries its own fix.
  const std::string plannerName = canonicalPlannerName(p.type);
  if (plannerName.empty()) {
    fail(absl::StrCat("unknown planner.type '", p.type, "'; supported: ", absl::StrJoin(knownPlannerNames(), ", ")));
  }
  if (s.comHeight <= 0.0 || s.gravity <= 0.0) fail("shared.comHeight and shared.gravity must be positive");
  if (g.minSwingDuration <= 0.0 || g.maxSwingDuration < g.minSwingDuration) fail("need 0 < minSwingDuration <= maxSwingDuration");
  if (g.minContactDuration <= 0.0) fail("minContactDuration must be positive");
  if (g.maxContactDuration > 0.0 && g.maxContactDuration < g.minContactDuration) fail("maxContactDuration must be >= minContactDuration");
  if (g.minDoubleSupportDuration < 0.0) fail("minDoubleSupportDuration must be non-negative");
  if (zmpSupportRegion.halfWidthX <= 0.0 || zmpSupportRegion.halfWidthY <= 0.0) fail("ZMP half widths must be positive");
  if (footSeparation.minStepWidth <= 0.0 || footSeparation.maxStepWidth < footSeparation.minStepWidth) {
    fail("need 0 < minStepWidth <= maxStepWidth");
  }
  if (stepWidth.nominalStepWidth < footSeparation.minStepWidth || stepWidth.nominalStepWidth > footSeparation.maxStepWidth) {
    fail("nominalStepWidth must lie within the step width bounds");
  }
  if (footSeparation.maxStepLength <= 0.0 || reachability.reachX <= 0.0) fail("maxStepLength and reachX must be positive");
  if (reachability.reachYOuter <= reachability.reachYInner) fail("reachYOuter must exceed reachYInner");
  if (s.bigM <= footSeparation.maxStepLength) fail("shared.bigM must exceed foot_separation.maxStepLength");
  // The big-M must also cover the LATERAL separation of the feet, and that, not the heading axis guarded above, is the
  // requirement that actually binds. In double support both contact binaries are one, so the +M c_i - M c_other terms
  // of a single-support ZMP box cancel and that box relaxes to only |e_y'(zmp - p_i)| <= halfWidthY + M, while the
  // double-support rows open the whole strip between the feet, e_y'p_R - halfWidthY <= e_y'zmp <= e_y'p_L + halfWidthY
  // (ZmpSupportRegionConstraint::addRows). Intersecting the two, the opposite foot's relaxed row binds as soon as M is
  // smaller than the separation w = e_y'(p_L - p_R), which foot_separation.maxStepWidth bounds from above; the
  // half-width cancels, so the exact condition is bigM >= maxStepWidth. Below it the ZMP is pulled towards one foot in
  // double support and every weight transfer past that point pays slack for a disjunction that was supposed to be
  // switched off. The previous check constrained bigM only against maxStepLength, of which the heading axis needs no
  // more than half, so it was neither necessary nor sufficient for the disjunction it claimed to guard: an operator
  // following the shipped files' advice to keep bigM "just above maxStepLength" while tightening maxStepLength below
  // the step width passed validation and then quietly paid slack on every double-support node.
  if (s.bigM < footSeparation.maxStepWidth) {
    fail(absl::StrCat("shared.bigM (", s.bigM, " m) must be at least foot_separation.maxStepWidth (", footSeparation.maxStepWidth,
                      " m), otherwise the relaxed single-support ZMP box clips the double-support region laterally"));
  }
  if (p.commitTime < 0.0) fail("planner.commitTime must be non-negative");
  if (cadenceStretch.samples < 0) fail("cadence_stretch.samples must be non-negative");
  if (cadenceStretch.samples > 0 && cadenceStretch.maxStretch < 1.0) {
    fail("cadence_stretch.maxStretch must be >= 1: a stretch below 1 shrinks the commit window and the horizon");
  }
  if (p.maxCommitExtension > 0.0 && p.maxCommitExtension < g.maxSwingDuration) {
    fail("planner.maxCommitExtension must be 0 (no cap) or at least maxSwingDuration, so a whole swing still fits in it");
  }
  if (s.slackPenalty.quadratic < 0.0 || s.slackPenalty.linear < 0.0) fail("shared.slack_penalty must be non-negative");
  const auto checkSlack = [&](const std::optional<SlackPenalty>& slack, const char* term) {
    if (slack.has_value() && (slack->quadratic < 0.0 || slack->linear < 0.0)) fail(std::string(term) + ".slack must be non-negative");
  };
  checkSlack(zmpSupportRegion.slack, "zmp_support_region");
  checkSlack(reachability.slack, "reachability");
  checkSlack(footSeparation.slack, "foot_separation");
  checkSlack(hipYawRange.slack, "hip_yaw_range");
  if (regularization.state < 0.0 || regularization.input < 0.0) fail("regularization terms must be non-negative");
  if (planConsistency.cost < 0.0 || previousFootholdConsistency.weight < 0.0) fail("plan consistency terms must be non-negative");
  if (contactSwitch.cost < 0.0) fail("contact_switch.cost must be non-negative");
  if (doubleSupportPenalty.cost < 0.0) fail("double_support_penalty.cost must be non-negative");
  if (velocityTracking.weight < 0.0 || stepWidth.weight < 0.0 || zmpRegularization.weight < 0.0 || footholdRegularization.weight < 0.0 ||
      stepLength.weight < 0.0 || terminalDcm.weight < 0.0) {
    fail("cost weights must be non-negative");
  }
  if (p.maxBranchAndBoundNodes < 1 || p.maxSolveTime <= 0.0 || p.maxQpIterations < 1) fail("invalid solver limits");
  if (eventShiftLocalSearch.iterations < 0 || eventShiftLocalSearch.maxTime < 0.0) fail("invalid local search limits");
  if (diving.maxDiveIterations < 1) fail("diving.maxDiveIterations must be at least 1");
  if (p.planningFrequency <= 0.0) fail("planner.planningFrequency must be positive");
  if (commitNodes() >= p.numNodes) fail("planner.commitTime must be shorter than the planning horizon");
  const PhaseResettingParameters& r = phaseResetting;
  if (r.earlyTouchdownMinSwingRatio < 0.0 || r.earlyTouchdownMinSwingRatio > 1.0) fail("earlyTouchdownMinSwingRatio must be in [0, 1]");
  if (r.earlyTouchdownMinContactDuration < 0.0) fail("earlyTouchdownMinContactDuration must be non-negative");
  if (r.earlyTouchdownMinAdvance < 0.0) fail("earlyTouchdownMinAdvance must be non-negative");
  if (r.maxLateTouchdownExtension < 0.0) fail("maxLateTouchdownExtension must be non-negative");
  if (r.lateTouchdownExtensionStep <= 0.0) fail("lateTouchdownExtensionStep must be positive");
  if (r.lateTouchdownSearchVelocity < 0.0) fail("lateTouchdownSearchVelocity must be non-negative");
  if (dcmStepAdjustment.gain < 0.0) fail("dcm_step_adjustment.gain must be non-negative");
  if (dcmStepAdjustment.maxOffset < 0.0) fail("dcm_step_adjustment.maxOffset must be non-negative");
  if (energyCadenceModulation.gain < 0.0) fail("energy_cadence_modulation.gain must be non-negative");
  const HlipParameters& h = hlip;
  if (h.sspDuration <= 0.0) fail("hlip.sspDuration must be positive");
  if (h.dspDuration < 0.0) fail("hlip.dspDuration must be non-negative");
  if (h.stepWidth <= 0.0) fail("hlip.stepWidth must be positive");
  if (h.maxStepLength <= 0.0) fail("hlip.maxStepLength must be positive");
  if (h.minStepWidth <= 0.0 || h.maxStepWidth < h.minStepWidth) fail("need 0 < hlip.minStepWidth <= hlip.maxStepWidth");
  if (h.stepWidth < h.minStepWidth || h.stepWidth > h.maxStepWidth) fail("hlip.stepWidth must lie within the hlip step width bounds");
  if (h.blend.sharpness <= 0.0) fail("hlip.blend.sharpness must be positive");
  // rho_2. The activity phi is a sum of squared ratios and so is never negative, and isWalking compares it with
  // `>=`, so a threshold of zero declares the robot to be walking at rest: standing becomes unreachable and the gait
  // marches in place forever. Nothing checked this, while its neighbour rho_1 was checked right above.
  if (h.blend.threshold <= 0.0) fail("hlip.blend.threshold must be positive, or the blend can never reach standing");
  if (h.blend.maxCommandedVelocityX <= 0.0 || h.blend.maxCommandedVelocityY <= 0.0 || h.blend.maxCommandedYawRate <= 0.0 ||
      h.blend.maxComVelocityX <= 0.0 || h.blend.maxComVelocityY <= 0.0) {
    fail("every hlip.blend command threshold must be positive");
  }
  if (energyCadenceModulation.deadband < 0.0) fail("energy_cadence_modulation.deadband must be non-negative");
  if (yawTorqueBudget.torsionalFrictionTorque < 0.0 || yawTorqueBudget.doubleSupportYawCouple < 0.0) fail("yaw torque limits must be >= 0");
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const bool unset = hipYawRange.lower[foot] == 0.0 && hipYawRange.upper[foot] == 0.0;
    if (!unset && !(hipYawRange.lower[foot] < 0.0 && hipYawRange.upper[foot] > 0.0)) fail("foot yaw bounds must be lower < 0 < upper");
  }
  if (headingRateTracking.weight < 0.0 || headingTracking.weight < 0.0 || yawTorqueRegularization.weight < 0.0 ||
      footYawTracking.weight < 0.0 || footYawRegularization.weight < 0.0) {
    fail("heading model weights must be >= 0");
  }
  if (headingRelinearisation.passes < 0 || headingRelinearisation.passes > 5) fail("heading_relinearisation.passes must be in [0, 5]");
  // Combinations that are legal but self-inconsistent are reported rather than rejected: they cost performance, not
  // correctness, and rejecting them would stop a shipped robot's file from loading over a few millimetres of step
  // width. They live in warnings() so that a test can assert on them; validate() is the one caller that emits them.
  for (const std::string& warning : warnings()) {
    LOG(WARNING) << "[ContactPlanningConfig] " << warning;
  }
  formulation.validate();
}

std::vector<std::string> ContactPlanningConfig::warnings() const {
  std::vector<std::string> out;
  const PlannerSettings& p = planner;
  const HlipParameters& h = hlip;
  // Every finding below is about the `hlip` block, which only the closed-form planner reads: warning a lip_miqp
  // operator about keys nothing in their formulation looks at would be noise.
  if (canonicalPlannerName(p.type) != planner::kHlip) return out;

  // The H-LIP deadbeat step is a feedback law on the measured state, and a plan costs microseconds because nothing is
  // solved. Running it on a background thread at a fraction of the MPC rate therefore buys nothing and costs the one
  // thing the law depends on: a plan posted at 10 Hz and handed over a cycle later is up to 100 ms stale, which is two
  // fifths of a 0.25 s single support. This is a warning rather than an error because the threading is the operator's
  // call and the mixed-integer planner genuinely needs the background thread.
  if (p.runInBackgroundThread) {
    out.push_back(
        absl::StrCat("planner.type: hlip with planner.runInBackgroundThread: true. The closed-form H-LIP planner costs microseconds, so "
                     "the background thread only adds latency to a feedback law: its plan reaches the solver a cycle late and at most "
                     "planner.planningFrequency (",
                     p.planningFrequency,
                     " Hz) times a second. Set planner.runInBackgroundThread: false to plan in the pre-solve hook at the MPC rate."));
  }

  // The step width bounds are checked above against hlip.stepWidth alone, which is the nominal orbit at a ZERO lateral
  // command. That is not the orbit the planner walks: HlipContactPlanner::deadbeatStep builds the period-two orbit at
  // +-stepWidth + v_y,cmd * (sspDuration + dspDuration) and then clips the placed foot into [minStepWidth,
  // maxStepWidth], so a sustained sidestep moves BOTH sides of the orbit by the drift of one step. The configuration is
  // only self-consistent when the narrowed side still clears the self-collision margin and the widened side still fits
  // the reach, at the largest lateral command the blend is normalised for. Nothing checked this before, and the
  // consequence is not a divergence - the widening side keeps its margin and reabsorbs the error - but a permanently
  // clipped narrow step: plan.numClippedSteps is non-zero on every cycle for as long as the command is held, so
  // ContactPlan::describe() prints CLIPPED-STEPS forever and the one signal documented to mean "the legs cannot deliver
  // this command at this cadence" becomes a standing false positive that hides genuine clipping, while the realised
  // lateral rate quietly runs below the command.
  const scalar_t stepDuration = h.sspDuration + h.dspDuration;
  if (stepDuration <= 0.0) return out;  // an invalid cadence; validate() has already reported it
  const scalar_t lateralDrift = h.blend.maxCommandedVelocityY * stepDuration;
  if (h.stepWidth - lateralDrift < h.minStepWidth) {
    out.push_back(absl::StrCat("a sustained sidestep at hlip.blend.maxCommandedVelocityY (", h.blend.maxCommandedVelocityY, " m/s) drifts ",
                               lateralDrift, " m over a step of ", stepDuration,
                               " s, so the step that places the "
                               "trailing foot is planned at hlip.stepWidth - drift = ",
                               h.stepWidth - lateralDrift, " m and clipped to hlip.minStepWidth (", h.minStepWidth,
                               " m) on every occurrence. Raise hlip.stepWidth, or lower hlip.blend.maxCommandedVelocityY below ",
                               (h.stepWidth - h.minStepWidth) / stepDuration, " m/s, or shorten the step."));
  }
  if (h.stepWidth + lateralDrift > h.maxStepWidth) {
    out.push_back(absl::StrCat("a sustained sidestep at hlip.blend.maxCommandedVelocityY (", h.blend.maxCommandedVelocityY, " m/s) drifts ",
                               lateralDrift, " m over a step of ", stepDuration,
                               " s, so the step that places the "
                               "leading foot is planned at hlip.stepWidth + drift = ",
                               h.stepWidth + lateralDrift, " m and clipped to hlip.maxStepWidth (", h.maxStepWidth,
                               " m) on every occurrence. Raise hlip.maxStepWidth, or lower hlip.blend.maxCommandedVelocityY below ",
                               (h.maxStepWidth - h.stepWidth) / stepDuration, " m/s, or shorten the step."));
  }
  return out;
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

bool hasAnyKey(const ptree& block, const char* const* keys, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (block.get_child_optional(keys[i])) return true;
  }
  return false;
}

bool hasAnyTermBlock(const ptree& block) {
  static const std::array<TermKind, 8> kinds{TermKind::MODEL_BLOCK,     TermKind::COST,          TermKind::SOFT_CONSTRAINT,
                                             TermKind::HARD_CONSTRAINT, TermKind::LOGIC_RULE,    TermKind::ASSIGNMENT_COST,
                                             TermKind::SEARCH_STAGE,    TermKind::EXECUTION_RULE};
  for (const TermKind kind : kinds) {
    for (const std::string& name : knownTermNames(kind)) {
      if (block.get_child_optional(name)) return true;
    }
  }
  return false;
}

/** A YAML sequence under `key`, or empty when the key is absent (`present` tells the two apart). */
std::vector<std::string> readList(const ptree& block, const char* key, bool& present) {
  std::vector<std::string> list;
  const auto child = block.get_child_optional(key);
  present = child.has_value();
  if (!present) return list;
  for (const auto& item : *child) {
    const std::string value = item.second.data();
    if (!value.empty()) list.push_back(value);
  }
  return list;
}

/**
 * The `slack` block of one term, if it has one. `sharedDefault` is `shared.slack_penalty` as already loaded from the
 * file, and it seeds the pair: loadPtreeValue leaves its destination untouched when the key is absent, so a block that
 * supplies only one of the two numbers must inherit the other from the shared default. Seeding from a
 * default-constructed SlackPenalty instead - which is what this function did - silently substituted the hard-coded
 * struct value (1e4 / 100) for the half the operator omitted, so a file that retuned shared.slack_penalty and then
 * overrode one number on one term ran that term on a penalty nobody wrote and that is not the shared default, against
 * what LipConstraintBase and the planner's documentation promise ("the shared default otherwise").
 */
void readSlack(
    const ptree& pt, const std::string& prefix, const SlackPenalty& sharedDefault, std::optional<SlackPenalty>& slack, bool verbose) {
  const boost::optional<const ptree&> child = pt.get_child_optional(prefix + "slack");
  if (!child) return;
  SlackPenalty penalty = sharedDefault;
  loadData::loadPtreeValue(pt, penalty.quadratic, prefix + "slack.quadratic", verbose);
  loadData::loadPtreeValue(pt, penalty.linear, prefix + "slack.linear", verbose);
  slack = penalty;
}

void loadStructured(const ptree& pt, const ptree& block, const std::string& prefix, ContactPlanningConfig& config, bool verbose) {
  const auto load = [&](auto& value, const std::string& key) { loadData::loadPtreeValue(pt, value, prefix + key, verbose); };
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
  bool present = false;
  const auto list = [&](const char* key, std::vector<std::string>& target) {
    std::vector<std::string> values = readList(block, key, present);
    if (present) target = std::move(values);
  };
  list("dynamics", f.dynamics);
  list("costs", f.costs);
  list("soft_constraints", f.softConstraints);
  list("hard_constraints", f.hardConstraints);
  list("logic_rules", f.logicRules);
  list("assignment_costs", f.assignmentCosts);
  list("search", f.search);
  list("execution", f.execution);

  load(config.regularization.state, std::string(term::kRegularization) + ".state");
  load(config.regularization.input, std::string(term::kRegularization) + ".input");
  load(config.previousFootholdConsistency.weight, std::string(term::kPreviousFootholdConsistency) + ".weight");
  load(config.velocityTracking.weight, std::string(term::kVelocityTracking) + ".weight");
  load(config.stepWidth.weight, std::string(term::kStepWidth) + ".weight");
  load(config.stepWidth.nominalStepWidth, std::string(term::kStepWidth) + ".nominalStepWidth");
  load(config.headingRateTracking.weight, std::string(term::kHeadingRateTracking) + ".weight");
  load(config.headingTracking.weight, std::string(term::kHeadingTracking) + ".weight");
  load(config.footYawTracking.weight, std::string(term::kFootYawTracking) + ".weight");
  load(config.yawTorqueRegularization.weight, std::string(term::kYawTorqueRegularization) + ".weight");
  load(config.footYawRegularization.weight, std::string(term::kFootYawRegularization) + ".weight");
  load(config.zmpRegularization.weight, std::string(term::kZmpRegularization) + ".weight");
  load(config.footholdRegularization.weight, std::string(term::kFootholdRegularization) + ".weight");
  load(config.stepLength.weight, std::string(term::kStepLength) + ".weight");
  load(config.terminalDcm.weight, std::string(term::kTerminalDcm) + ".weight");
  load(config.terminalDcm.trackCommandedVelocity, std::string(term::kTerminalDcm) + ".trackCommandedVelocity");
  load(config.zmpSupportRegion.halfWidthX, std::string(term::kZmpSupportRegion) + ".halfWidthX");
  load(config.zmpSupportRegion.halfWidthY, std::string(term::kZmpSupportRegion) + ".halfWidthY");
  readSlack(pt, prefix + term::kZmpSupportRegion + ".", s.slackPenalty, config.zmpSupportRegion.slack, verbose);
  load(config.reachability.reachX, std::string(term::kReachability) + ".reachX");
  load(config.reachability.reachYInner, std::string(term::kReachability) + ".reachYInner");
  load(config.reachability.reachYOuter, std::string(term::kReachability) + ".reachYOuter");
  readSlack(pt, prefix + term::kReachability + ".", s.slackPenalty, config.reachability.slack, verbose);
  load(config.footSeparation.maxStepLength, std::string(term::kFootSeparation) + ".maxStepLength");
  load(config.footSeparation.minStepWidth, std::string(term::kFootSeparation) + ".minStepWidth");
  load(config.footSeparation.maxStepWidth, std::string(term::kFootSeparation) + ".maxStepWidth");
  readSlack(pt, prefix + term::kFootSeparation + ".", s.slackPenalty, config.footSeparation.slack, verbose);
  readSlack(pt, prefix + term::kHipYawRange + ".", s.slackPenalty, config.hipYawRange.slack, verbose);
  load(config.contactSwitch.cost, std::string(term::kContactSwitch) + ".cost");
  load(config.doubleSupportPenalty.cost, std::string(term::kDoubleSupportPenalty) + ".cost");
  load(config.planConsistency.cost, std::string(term::kPlanConsistency) + ".cost");
  load(config.diving.maxDiveIterations, std::string(term::kDiving) + ".maxDiveIterations");
  load(config.eventShiftLocalSearch.iterations, std::string(term::kEventShiftLocalSearch) + ".iterations");
  load(config.eventShiftLocalSearch.maxTime, std::string(term::kEventShiftLocalSearch) + ".maxTime");
  load(config.cadenceStretch.samples, std::string(term::kCadenceStretch) + ".samples");
  load(config.cadenceStretch.maxStretch, std::string(term::kCadenceStretch) + ".maxStretch");
  load(config.headingRelinearisation.passes, std::string(term::kHeadingRelinearisation) + ".passes");
  load(config.phaseResetting.earlyTouchdownMinSwingRatio, std::string(term::kPhaseResetting) + ".earlyTouchdownMinSwingRatio");
  load(config.phaseResetting.earlyTouchdownMinContactDuration, std::string(term::kPhaseResetting) + ".earlyTouchdownMinContactDuration");
  load(config.phaseResetting.maxLateTouchdownExtension, std::string(term::kPhaseResetting) + ".maxLateTouchdownExtension");
  load(config.phaseResetting.lateTouchdownExtensionStep, std::string(term::kPhaseResetting) + ".lateTouchdownExtensionStep");
  load(config.phaseResetting.earlyTouchdownMinAdvance, std::string(term::kPhaseResetting) + ".earlyTouchdownMinAdvance");
  load(config.phaseResetting.lateTouchdownSearchVelocity, std::string(term::kPhaseResetting) + ".lateTouchdownSearchVelocity");
  load(config.energyCadenceModulation.gain, std::string(term::kEnergyCadenceModulation) + ".gain");
  load(config.energyCadenceModulation.deadband, std::string(term::kEnergyCadenceModulation) + ".deadband");
  load(config.dcmStepAdjustment.gain, std::string(term::kDcmStepAdjustment) + ".gain");
  load(config.dcmStepAdjustment.maxOffset, std::string(term::kDcmStepAdjustment) + ".maxOffset");
  // Both shipped robots open a contact_planning_config block that points back at this label, so both are named here: a
  // key added above has to be documented in each of them. Only the DRC Atlas file used to be named, and the EngineAI
  // SA01 file was left to drift out of sync with the loader unnoticed - exactly the failure the directive pair exists
  // to prevent. The third target this directive carried, the tuning GUI's contact_planning_gui_keys, was deleted
  // together with the GUI's hard-coded key list (the GUI renders the parameters generically now), so it pointed at a
  // label that no longer exists anywhere, which a label-aware IFTTT checker reports as an error on any edit here.
  // clang-format off
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/contact_planning.yaml:contact_planning_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/contact_planning.yaml:contact_planning_config)
  // clang-format on
}

}  // namespace

ContactPlanningConfig loadContactPlanningConfig(const std::string& yamlFile, const std::string& prefix, bool verbose, bool validate) {
  ptree pt;
  loadData::readPropertyTree(yamlFile, pt);
  ContactPlanningConfig config;
  if (verbose) {
    LOG(INFO) << "\n #### Contact Planning Config:";
    LOG(INFO) << "\n #### =============================================================================\n";
  }
  const std::string blockKey =
      prefix.empty() ? std::string() : prefix.substr(0, prefix.size() - 1);  // "contact_planning." -> "contact_planning"
  boost::optional<const ptree&> block;
  if (blockKey.empty()) {
    block = pt;
  } else if (const auto child = pt.get_child_optional(blockKey)) {
    block = *child;
  }
  if (block) {
    const bool structured = hasAnyKey(*block, kStructuredKeys.data(), kStructuredKeys.size()) || hasAnyTermBlock(*block);
    const bool legacy = hasAnyKey(*block, kLegacyKeys.data(), kLegacyKeys.size());
    if (legacy) {
      throw std::invalid_argument(
          "[ContactPlanningConfig] " + yamlFile + (structured ? " mixes the structured contact_planning layout with" : " uses") +
          " keys of the flat layout of the previous planner (dt, velocityTrackingWeight, useAcomDynamics, ...), which is no longer read. "
          "Migrate the block to the structured layout: `planner` / `shared` blocks, the term lists (dynamics, costs, soft_constraints, "
          "hard_constraints, logic_rules, assignment_costs, search, execution) and one parameter block per term, as in the DRC Atlas "
          "contact_planning.yaml; the flags became list entries (useAcomDynamics -> heading_double_integrator and its terms, "
          "enablePhaseResetting -> phase_resetting, ...).");
    }
    loadStructured(pt, *block, prefix, config, verbose);
  }
  if (verbose) {
    LOG(INFO) << " #### =============================================================================";
  }
  if (validate) config.validate();
  return config;
}

}  // namespace ocs2::humanoid
