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

#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <utility>

#include <ocs2_core/Types.h>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"

namespace ocs2::humanoid {

/**
 * Configuration of the mixed-integer contact planner: the `contact_planning` block of `contact_planning.yaml`.
 *
 * The block mirrors the structure of the planner. `planner` holds what is a property of the planner itself (grid,
 * commit window, solver budget, threading), `shared` the parameters read by more than one term, `formulation` the term
 * lists, and then one parameter block per term, named as the term (ContactPlanningFormulation.h). A term reads its own
 * block in ContactPlanningTerm::configure(), which is also the hot-reload path: the parameter updater re-parses the
 * file and hands the whole configuration to the planner and the reference manager, and every term picks up its block.
 *
 * Geometry is expressed in the yaw-aligned frame of the base at planning time (x forward, y left).
 */

/** Penalty of the slacks of a soft constraint, 0.5 Z s^2 + z s per slack (HPIPM soft constraints). */
struct SlackPenalty {
  scalar_t quadratic = 1.0e4;
  scalar_t linear = 100.0;
};

/** Phase duration limits [s], read by the logic rules, the cadence modulation and the node conversions. */
struct GaitLimits {
  scalar_t minSwingDuration = 0.3;
  scalar_t maxSwingDuration = 0.6;
  scalar_t minContactDuration = 0.15;
  scalar_t maxContactDuration = 0.0;        // <= 0 disables the limit (standing is allowed indefinitely)
  scalar_t minDoubleSupportDuration = 0.1;  // after a touch-down the other foot stays down at least this long (0 disables)
};

/** Properties of the planner, not of a term. */
struct PlannerSettings {
  scalar_t dt = 0.1;           // [s] planner node duration
  int numNodes = 12;           // planning horizon = numNodes * dt, should cover the MPC horizon
  scalar_t commitTime = 0.25;  // [s] contacts within this window keep the applied schedule (must cover planner latency)
  // [s] cap on how far past `commitTime` the commit boundary may be extended to reach a swing's touch-down.
  // <= 0: no cap (the historical behaviour). The extension walks the phases that overlap the window, and a swing whose
  // touch-down lies beyond it pushes the boundary out to that touch-down; the walk then covers the phases overlapping
  // the extended boundary as well. In a gait that exchanges support in one instant every swing starts exactly where
  // the previous one ends, so nothing stops that walk: the boundary runs to the end of the stepping region, the plan
  // no longer reaches past it (ContactPlanningReferenceManager's planUsable needs endTime() > commitTime + dt), no plan
  // is ever merged and the robot stops stepping. A double support of any length breaks the chain, which is why the
  // failure only appears once the double supports are gone. Capping the extension keeps replanning alive; the cost is
  // that a boundary cut short can land inside a swing that is already in flight, handing a later plan the authority to
  // re-time it, which is exactly what the extension exists to prevent. Keep the cap above maxSwingDuration so that a
  // single swing still fits inside it.
  scalar_t maxCommitExtension = 0.0;
  int maxBranchAndBoundNodes = 200;
  scalar_t maxSolveTime = 0.1;  // [s]
  int maxQpIterations = 60;
  bool runInBackgroundThread = true;  // false: plan synchronously inside the MPC's pre-solve hook
  scalar_t planningFrequency = 10.0;  // [Hz] upper bound on the background planning rate
  bool verbose = false;
  bool logPlans = false;  // one line per plan: search statistics, phase durations, step lengths (describeContactPlan)
};

/** Parameters read by more than one term. */
struct SharedParameters {
  scalar_t gravity = 9.81;    // [m/s^2]
  scalar_t comHeight = 0.85;  // [m] LIP height, omega = sqrt(g / comHeight); 0 = from the model
  scalar_t bigM = 1.0;        // [m] big-M of the ZMP / foothold disjunctions (must exceed foot_separation.maxStepLength)
  SlackPenalty slackPenalty;  // default of every soft constraint without a penalty of its own
  GaitLimits gaitLimits;
};

// ---- per-term parameter blocks, named as the terms ----

struct RegularizationParameters {
  scalar_t state = 1.0e-8;  // added to the diagonal of Q at every node
  scalar_t input = 1.0e-6;  // added to the diagonal of R at every running node
};
struct PreviousFootholdConsistencyParameters {
  scalar_t weight = 5.0;  // ||p_foot - p_foot,previous plan||^2 per node, damps foothold jitter
};
struct VelocityTrackingParameters {
  scalar_t weight = 20.0;  // ||v_com - v_cmd||^2 per node
};
struct StepWidthParameters {
  scalar_t weight = 10.0;            // (step width - nominalStepWidth)^2 per node
  scalar_t nominalStepWidth = 0.25;  // [m] lateral distance left foot - right foot the planner is drawn to
};
struct HeadingRateTrackingParameters {
  scalar_t weight = 20.0;  // (heading rate - commanded yaw rate)^2 per node
};
struct HeadingTrackingParameters {
  scalar_t weight = 5.0;  // (heading - commanded heading)^2 per node
};
struct FootYawTrackingParameters {
  scalar_t weight = 5.0;  // (foot yaw - nominal heading)^2 per node
};
struct YawTorqueRegularizationParameters {
  scalar_t weight = 1.0e-3;  // yaw torque^2 per node
};
struct FootYawRegularizationParameters {
  scalar_t weight = 1.0;  // (foot yaw displacement)^2 per node
};
struct ZmpRegularizationParameters {
  scalar_t weight = 2.0;  // ||zmp - com||^2 per node, keeps the ZMP under the CoM when possible
};
struct FootholdRegularizationParameters {
  scalar_t weight = 5.0;  // ||foot displacement||^2 per node, prefers short steps
};
struct StepLengthParameters {
  scalar_t weight = 0.0;  // ||dp_swing - d_nom||^2 per running node, d_nom from v_cmd at the nominal cadence (StepLengthCost)
};
struct TerminalDcmParameters {
  scalar_t weight = 200.0;  // ||DCM_N - zmp_{N-1}||^2, terminal capturability
  // false: the DCM is drawn onto the last ZMP, i.e. the plan comes to rest at the end of the horizon, which shortens
  // the steps in the horizon at speed. true: the DCM is drawn to zmp + v_cmd / omega, the offset of a CoM over the foot
  // that keeps moving at the commanded velocity, so the plan is asked to keep walking, not to stop.
  bool trackCommandedVelocity = false;
};
struct ZmpSupportRegionParameters {
  // ZMP support region half-widths. Single support: a box of these half-widths around the stance foot. Double support:
  // along the heading a box of half-width halfWidthX around the midpoint of the feet, laterally the strip between the
  // right foot minus and the left foot plus halfWidthY. 0 = from the sole's footprint (model parameters).
  scalar_t halfWidthX = 0.08;  // [m] along the heading
  scalar_t halfWidthY = 0.04;  // [m] lateral
  std::optional<SlackPenalty> slack;
};
struct ReachabilityParameters {
  scalar_t reachX = 0.45;       // foot x offset from the CoM must stay within [-reachX, reachX]
  scalar_t reachYInner = 0.05;  // left foot y - CoM y >= reachYInner (mirrored for the right foot)
  scalar_t reachYOuter = 0.45;  // left foot y - CoM y <= reachYOuter (mirrored for the right foot)
  std::optional<SlackPenalty> slack;
};
struct FootSeparationParameters {
  scalar_t maxStepLength = 0.5;  // |x_left - x_right| bound
  scalar_t minStepWidth = 0.15;  // self-collision margin
  scalar_t maxStepWidth = 0.45;
  std::optional<SlackPenalty> slack;
};
/** Hip yaw range per foot, derived from the model (ContactPlanningModelParameters), not a file key. */
struct HipYawRangeParameters {
  feet_array_t<scalar_t> lower = makeFeetArray(0.0);  // [rad] bounds on foot yaw - heading
  feet_array_t<scalar_t> upper = makeFeetArray(0.0);
  std::optional<SlackPenalty> slack;
};
/** Yaw torque the ground can carry, derived from the model and the wrench cone, not a file key. */
struct YawTorqueBudgetParameters {
  scalar_t torsionalFrictionTorque = 0.0;  // [N m] |yaw torque| one stance foot carries
  scalar_t doubleSupportYawCouple = 0.0;   // [N m] additional |yaw torque| from the couple of two stance feet
};
struct ContactSwitchParameters {
  scalar_t cost = 0.2;  // linear cost per lift-off / touch-down event
};
struct PlanConsistencyParameters {
  scalar_t cost = 0.5;  // cost per node whose contact differs from the previous plan (hysteresis)
};
struct DoubleSupportPenaltyParameters {
  // Cost per node at which both feet are in contact, which buys the weight transfer of a step against the double
  // support the ZMP terms would otherwise pay for. The term cannot tell a transfer apart from standing, so it also
  // prices standing still (every node of it): above ~0.2 with the Atlas gait limits the planner would rather step in
  // place than stand, and at 5.0 it marches continuously at a zero velocity command. Keep it well under that; 0.1
  // removes the transition double support at walking speed and leaves standing alone.
  scalar_t cost = 0.1;
};
struct DivingParameters {
  int maxDiveIterations = 64;
};
struct EventShiftLocalSearchParameters {
  int iterations = 10;      // rounds of event-shift local search after the branch-and-bound (0 disables)
  scalar_t maxTime = 0.05;  // [s] time budget of the local search
};
struct HeadingRelinearisationParameters {
  int passes = 1;  // re-linearisations of the heading frame at the incumbent (0 disables)
};
struct PhaseResettingParameters {
  scalar_t earlyTouchdownMinSwingRatio = 0.25;       // contact during this initial fraction of the nominal swing is ignored (scuffing)
  scalar_t earlyTouchdownMinContactDuration = 0.02;  // [s] contact must persist this long before the swing is ended (debounce)
  // [s] a touch-down closer than this to its scheduled time is executed as planned instead of ending the swing early.
  // Ending it early puts the foot in contact from the measured contact until its scheduled touch-down, and when the
  // gait exchanges support in a single instant (minDoubleSupportDuration: 0) that scheduled touch-down is the other
  // foot's lift-off, so the truncation opens a double support exactly as long as the foot was early. Landing within a
  // few milliseconds of the plan is the common case, not the exception, so this also bounds the shortest double
  // support the rule can create. Keep it above earlyTouchdownMinContactDuration, or the debounce alone already pushes
  // every truncation into that window. 0 restores the unguarded behaviour.
  scalar_t earlyTouchdownMinAdvance = 0.04;
  scalar_t maxLateTouchdownExtension = 0.15;    // [s] total extension budget of a swing past its planned touch-down
  scalar_t lateTouchdownExtensionStep = 0.05;   // [s] the touch-down is pushed this far ahead of the current time per cycle
  scalar_t lateTouchdownSearchVelocity = 0.05;  // [m/s] the foot height target descends at this rate during the extension
};
struct EnergyCadenceModulationParameters {
  scalar_t gain = 0.01;     // [s/J] touch-down shift = -gain * (E - E_pred), E = m (v^2 - w^2 x^2) / 2, full model mass
  scalar_t deadband = 0.0;  // [J] deviations within this band re-time nothing, beyond it the shift is measured from the band's edge
};
struct DcmStepAdjustmentParameters {
  scalar_t gain = 0.5;        // gain on the closed-form LIP step adjustment (1 = exact compensation of the increment)
  scalar_t maxOffset = 0.05;  // [m] bound on the landing target offset (also clipped to the reachable region)
};

struct ContactPlanningConfig {
  PlannerSettings planner;
  SharedParameters shared;
  ContactPlanningFormulation formulation;

  RegularizationParameters regularization;
  PreviousFootholdConsistencyParameters previousFootholdConsistency;
  VelocityTrackingParameters velocityTracking;
  StepWidthParameters stepWidth;
  HeadingRateTrackingParameters headingRateTracking;
  HeadingTrackingParameters headingTracking;
  FootYawTrackingParameters footYawTracking;
  YawTorqueRegularizationParameters yawTorqueRegularization;
  FootYawRegularizationParameters footYawRegularization;
  ZmpRegularizationParameters zmpRegularization;
  FootholdRegularizationParameters footholdRegularization;
  StepLengthParameters stepLength;
  TerminalDcmParameters terminalDcm;
  ZmpSupportRegionParameters zmpSupportRegion;
  ReachabilityParameters reachability;
  FootSeparationParameters footSeparation;
  HipYawRangeParameters hipYawRange;
  YawTorqueBudgetParameters yawTorqueBudget;
  ContactSwitchParameters contactSwitch;
  PlanConsistencyParameters planConsistency;
  DoubleSupportPenaltyParameters doubleSupportPenalty;
  DivingParameters diving;
  EventShiftLocalSearchParameters eventShiftLocalSearch;
  HeadingRelinearisationParameters headingRelinearisation;
  PhaseResettingParameters phaseResetting;
  EnergyCadenceModulationParameters energyCadenceModulation;
  DcmStepAdjustmentParameters dcmStepAdjustment;

  static constexpr scalar_t kDefaultFootYawOffset = 0.5;  // [rad] symmetric fallback when no hip yaw joint can be identified

  // ---- conveniences over the blocks ----
  bool usesHeadingModel() const { return formulation.usesHeadingModel(); }
  void setHeadingModel(bool on) { formulation.setHeadingModel(on); }

  scalar_t omega() const { return std::sqrt(shared.gravity / shared.comHeight); }
  scalar_t horizon() const { return planner.dt * static_cast<scalar_t>(planner.numNodes); }

  /** Bounds on (foot yaw - heading) of a foot. */
  std::pair<scalar_t, scalar_t> footYawOffsetBounds(size_t foot) const { return {hipYawRange.lower[foot], hipYawRange.upper[foot]}; }
  /** Symmetric foot yaw bounds, for tests and for robots without a model-derived value. */
  void setSymmetricFootYawOffset(scalar_t offset) {
    hipYawRange.lower = makeFeetArray(-offset);
    hipYawRange.upper = makeFeetArray(offset);
  }
  /** True once the model-derived parameters of the heading model have been applied. */
  bool hasModelParameters() const {
    // The foot yaw bounds are the marker: derived bounds always straddle zero, and a zero torque limit is a legitimate
    // derived value (a frictionless sole), not a missing one.
    if (yawTorqueBudget.torsionalFrictionTorque < 0.0 || yawTorqueBudget.doubleSupportYawCouple < 0.0) return false;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (!(hipYawRange.lower[foot] < 0.0 && hipYawRange.upper[foot] > 0.0)) return false;
    }
    return true;
  }

  // Duration limits in planner nodes (conservative rounding).
  int minSwingNodes() const { return std::max(1, static_cast<int>(std::ceil(shared.gaitLimits.minSwingDuration / planner.dt - 1e-9))); }
  int maxSwingNodes() const {
    return std::max(minSwingNodes(), static_cast<int>(std::floor(shared.gaitLimits.maxSwingDuration / planner.dt + 1e-9)));
  }
  int minContactNodes() const { return std::max(1, static_cast<int>(std::ceil(shared.gaitLimits.minContactDuration / planner.dt - 1e-9))); }
  int minDoubleSupportNodes() const {
    if (shared.gaitLimits.minDoubleSupportDuration <= 0.0) return 0;
    return static_cast<int>(std::ceil(shared.gaitLimits.minDoubleSupportDuration / planner.dt - 1e-9));
  }
  int maxContactNodes() const {
    if (shared.gaitLimits.maxContactDuration <= 0.0) return 0;
    return std::max(minContactNodes(), static_cast<int>(std::floor(shared.gaitLimits.maxContactDuration / planner.dt + 1e-9)));
  }
  int commitNodes() const { return std::max(0, static_cast<int>(std::ceil(planner.commitTime / planner.dt - 1e-9))); }

  /** Throws std::invalid_argument if the configuration is inconsistent (every block, and the formulation). */
  void validate() const;
};

/** Name of the planner's own configuration file, expected in the directory of the robot's task file. */
inline constexpr const char* kContactPlanningConfigFileName = "contact_planning.yaml";

/**
 * The file the contact planning configuration is read from for a given task file: `contact_planning.yaml` in the task
 * file's directory when it exists, otherwise the task file itself (a `contact_planning` block inside it, the layout from
 * before the planner had its own file). The `useContactPlanning` switch stays in the task file with the other model
 * settings; everything the planner is tuned with lives in its own file.
 */
std::string resolveContactPlanningConfigFile(const std::string& taskFile);

/**
 * Loads the configuration from a YAML file: the planner's own contact_planning.yaml, or a task file with the block
 * inline (see resolveContactPlanningConfigFile). Missing keys keep their defaults.
 *
 * The block is the structured layout of this header (`planner`, `shared`, the term lists, one block per term). A file
 * with keys of the flat layout of the previous planner (every key directly under `contact_planning`, `useAcomDynamics`,
 * `enablePhaseResetting`, ... as booleans) is rejected with a message that says how to migrate it.
 *
 * With `validate` false the values that may be 0 for "derive from the model" are accepted as they are; call validate()
 * after ContactPlanningModelParameters::applyTo().
 */
ContactPlanningConfig loadContactPlanningConfig(const std::string& yamlFile,
                                                const std::string& prefix = "contact_planning.",
                                                bool verbose = false,
                                                bool validate = true);

}  // namespace ocs2::humanoid
