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

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"

namespace ocs2::humanoid {

/**
 * Configuration of the contact planner that `planner.type` selects (ContactPlannerFactory: `hlip`, the closed-form
 * H-LIP planner, or `lip_miqp`, the mixed-integer program): the robot's `contact_planning.textproto`
 * (contactPlanningConfigFromConfig()).
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

/**
 * Properties of the planner, not of a term.
 *
 * The defaults of this header are ONE coherent configuration, and it is the mixed-integer planner's: a coarse grid
 * (dt 0.1 x 12 nodes), a worker thread at 10 Hz, the point-mass formulation without execution rules. They are what a
 * robot gets when its file omits keys, or when it has no contact_planning block at all, and
 * ContactPlanningConfig::warnings() is empty for them (a test holds that). The type used to default to `hlip` while
 * every other default stayed the mixed-integer one, which put together exactly the configurations the H-LIP README
 * documents as falling: no planned_com_override, a 100 ms stale plan from the worker thread, and a 0.35 s single
 * support whose first step out of a standstill does not fit the step-width clip. A robot that wants `hlip` says so in
 * its file together with the settings that planner needs, as both shipped contact_planning.textproto files do.
 */
struct PlannerSettings {
  // Implementation that plans the contacts, resolved by ContactPlannerFactory. `hlip` is the closed-form
  // reduced-order stepper of arXiv:2502.15630 (humanoid_nmpc/docs/hlip_contact_planner/README.md); `lip_miqp` is the
  // mixed-integer program of LipContactPlanner. Only the blocks the selected planner reads are used: `hlip` ignores
  // the term lists of `formulation` (except whether `dynamics` lists the heading block, and `execution`, which the
  // reference manager reads under either planner) and every per-term block, and `lip_miqp` ignores the `hlip` block.
  std::string type = "lip_miqp";
  scalar_t dt = 0.1;           // [s] planner node duration
  int numNodes = 12;           // planning horizon = numNodes * dt, should cover the MPC horizon
  scalar_t commitTime = 0.25;  // [s] contacts within this window keep the applied schedule (must cover planner latency)
  // [s] cap on how far past `commitTime` the commit boundary may be extended to reach a swing's touch-down.
  // <= 0: no cap (the historical behavior). The extension walks the phases that overlap the window, and a swing whose
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
  // planner.threading "background_thread"; false ("pre_solve_hook"): plan synchronously inside the MPC's pre-solve hook.
  bool runInBackgroundThread = true;
  scalar_t planningFrequency = 10.0;  // [Hz] upper bound on the background planning rate
  bool verbose = false;
  bool logPlans = false;  // one line per plan: search statistics, phase durations, step lengths (describeContactPlan)
};

/** Parameters read by more than one term. */
struct SharedParameters {
  scalar_t gravity = 9.81;  // [m/s^2]
  // [m] LIP height, omega = sqrt(g / comHeight). Unset = from the model: its center of mass above its feet at the task
  // file's initialState (computeComHeightAboveFeet, which an unset DCM terminal cost height resolves to as well), filled
  // in by ContactPlanningModelParameters::applyTo before validation, at start-up and on every hot reload. It is also
  // the library default, so a robot file that leaves shared.com_height out (as both shipped files do) plans on its own
  // model's pendulum - the one its DCM terminal cost derives - rather than on a fixed height that belongs to no robot
  // (it was 0.85 m, the height the Atlas was once hand-set to). A configuration with no model to derive it from - a
  // planner unit test - must set a positive height itself: validateStatus() refuses an unset one.
  std::optional<scalar_t> comHeight;
  // [m] big-M of the ZMP / foothold disjunctions. Two bounds, not one: it must EXCEED foot_separation.maxStepLength,
  // which the foothold displacement bound needs, and it must be at least foot_separation.maxStepWidth, because that is
  // what it takes for a single-support ZMP box to actually switch off in double support (the lateral axis is the one
  // that binds there; see the derivation next to the check in ContactPlanningConfig::validateStatus()).
  scalar_t bigM = 1.0;
  SlackPenalty slackPenalty;  // default of every soft constraint without a penalty of its own
  GaitLimits gaitLimits;
};

/**
 * Blend between standing and walking, equations (15) - (17) of arXiv:2502.15630.
 *
 * The activity phi is the squared norm of the commanded and the measured planar velocity, each component divided by
 * the largest value that component is expected to take, and the blend weight is alpha = tanh(sharpness (phi -
 * threshold)) / 2 + 1/2. The planner stands while alpha is below a half and steps above it, and scales the commanded
 * velocity it plans for by alpha, so that the step length grows out of standing instead of jumping to its full value
 * at the first non-zero command. HlipContactPlanner feeds the blend the measured center-of-mass velocity with its
 * lateral component taken relative to the sway of stepping in place (HlipContactPlanner::blendVelocity), so the
 * zero-command orbit's own lateral velocity (0.132 m/s at the Atlas cadence on its 1.0805 m pendulum) does not count as
 * motion.
 *
 * The paper's phi also contains the commanded center of mass height divided by a minimum height. That term is an
 * absolute height, not a deviation, so it alone exceeds one for any upright robot and would saturate alpha at one; it
 * is left out here. Every remaining threshold is a command range, not a tuning weight.
 *
 * Dropping that term also moves where phi sits, so `sharpness` and `threshold` are not the paper's 5.0 and 0.5. With
 * the defaults below the half point sits at sqrt(threshold), 14 % of a component's range (0.099 m/s of the 0.7 m/s
 * forward range); alpha is about 0.83 at a fifth of the range and passes 0.99 only near 28 % (testHlipStandingBlend pins
 * these numbers). That is what "walk when asked to walk, stand when asked to stand" means for a humanoid: at a tenth of
 * full stick the robot is being asked to move, not to hold station.
 */
struct HlipBlendParameters {
  scalar_t sharpness = 40.0;             // rho_1
  scalar_t threshold = 0.02;             // rho_2
  scalar_t maxCommandedVelocityX = 0.7;  // [m/s]
  // [m/s]. This is a command range, but it is not free of the step geometry: the planner plans the period-two orbit at
  // +-hlip.stepWidth + v_y (sspDuration + dspDuration), so at the largest lateral command the narrow side of the orbit
  // must still clear hlip.minStepWidth and the wide side must still fit hlip.maxStepWidth
  // (ContactPlanningConfig::warnings() checks both and says which key to move). Against the defaults below, a 0.30 s
  // step, 0.25 m/s leaves 0.175 m on the narrow side and asks 0.325 m on the wide one, both inside the clips.
  scalar_t maxCommandedVelocityY = 0.25;
  scalar_t maxCommandedYawRate = 0.61;  // [rad/s] (35 deg/s)
  scalar_t maxComVelocityX = 0.5;       // [m/s]
  scalar_t maxComVelocityY = 0.4;       // [m/s]
};

/**
 * The closed-form H-LIP contact planner (`planner.type: hlip`), the reduced-order layer of arXiv:2502.15630.
 *
 * The gait is a fixed cadence: single support of `sspDuration` alternating between the feet, separated by a double
 * support of `dspDuration`. The footholds come from the deadbeat step-to-step controller of the H-LIP, which has no
 * gain to tune (HlipModel). The only quantities below that are neither a cadence nor a command range are the step
 * bounds, which exist so that a foothold the deadbeat law asks for outside the leg's reach is clipped instead of
 * handed to the whole-body MPC.
 *
 * The cadence defaults are the one validated in simulation on the DRC Atlas, not the paper's. The figures below are
 * for a 0.85 m pendulum (shared.comHeight 0.85), the one the Atlas was hand-set to when that cadence fell (on its
 * model's 1.0805 m pendulum, which both shipped files now resolve, each demand is smaller: 0.45, 0.47 and 0.39 m, and
 * 0.35 s / 0.05 s recovers instead of locking). The previous 0.35 s with an instantaneous exchange
 * (dspDuration 0) asked 0.49 m of the first step out of a standstill against the 0.45 m maxStepWidth below, so that
 * step was clipped; the reduced model still recovers from that particular cut, but the same 0.35 s with the 0.05 s
 * double support the whole-body MPC needs asks 0.52 m and locks into README section 3b's "cadence that fell". A zero
 * double support with an uncapped planner.maxCommitExtension is also the chain that stalls the commit boundary. At
 * 0.25 / 0.05 the first step needs 0.42 m and fits. ContactPlanningConfig::warnings() reports a first step that does
 * not fit and the uncapped instantaneous exchange for any configuration that reintroduces them.
 */
struct HlipParameters {
  scalar_t sspDuration = 0.25;   // [s] single support duration
  scalar_t dspDuration = 0.05;   // [s] double support duration; the paper's 0 exchanges support in one instant
  scalar_t stepWidth = 0.25;     // [m] lateral distance between the feet of the nominal period-two orbit
  scalar_t maxStepLength = 0.5;  // [m] clip on the planned step along the heading
  scalar_t maxStepWidth = 0.45;  // [m] clip on the lateral distance between the feet
  scalar_t minStepWidth = 0.15;  // [m] self-collision margin on that distance
  HlipBlendParameters blend;
};

// ---- per-term parameter blocks, named as the terms ----

/** The parameters of the `regularization` cost (the `regularization` block of contact_planning). Passive data. */
struct RegularizationParameters {
  scalar_t state = 1.0e-8;  // added to the diagonal of Q at every node
  scalar_t input = 1.0e-6;  // added to the diagonal of R at every running node
};
/** The parameters of the `previous_foothold_consistency` cost (the `previous_foothold_consistency` block of contact_planning). Passive
 * data. */
struct PreviousFootholdConsistencyParameters {
  scalar_t weight = 5.0;  // ||p_foot - p_foot,previous plan||^2 per node, damps foothold jitter
};
/** The parameters of the `velocity_tracking` cost (the `velocity_tracking` block of contact_planning). Passive data. */
struct VelocityTrackingParameters {
  scalar_t weight = 20.0;  // ||v_com - v_cmd||^2 per node
};
/** The parameters of the `step_width` cost (the `step_width` block of contact_planning). Passive data. */
struct StepWidthParameters {
  scalar_t weight = 10.0;            // (step width - nominalStepWidth)^2 per node
  scalar_t nominalStepWidth = 0.25;  // [m] lateral distance left foot - right foot the planner is drawn to
};
/** The parameters of the `heading_rate_tracking` cost (the `heading_rate_tracking` block of contact_planning). Passive data. */
struct HeadingRateTrackingParameters {
  scalar_t weight = 20.0;  // (heading rate - commanded yaw rate)^2 per node
};
/** The parameters of the `heading_tracking` cost (the `heading_tracking` block of contact_planning). Passive data. */
struct HeadingTrackingParameters {
  scalar_t weight = 5.0;  // (heading - commanded heading)^2 per node
};
/** The parameters of the `foot_yaw_tracking` cost (the `foot_yaw_tracking` block of contact_planning). Passive data. */
struct FootYawTrackingParameters {
  scalar_t weight = 5.0;  // (foot yaw - nominal heading)^2 per node
};
/** The parameters of the `yaw_torque_regularization` cost (the `yaw_torque_regularization` block of contact_planning). Passive data. */
struct YawTorqueRegularizationParameters {
  scalar_t weight = 1.0e-3;  // yaw torque^2 per node
};
/** The parameters of the `foot_yaw_regularization` cost (the `foot_yaw_regularization` block of contact_planning). Passive data. */
struct FootYawRegularizationParameters {
  scalar_t weight = 1.0;  // (foot yaw displacement)^2 per node
};
/** The parameters of the `zmp_regularization` cost (the `zmp_regularization` block of contact_planning). Passive data. */
struct ZmpRegularizationParameters {
  scalar_t weight = 2.0;  // ||zmp - com||^2 per node, keeps the ZMP under the CoM when possible
};
/** The parameters of the `foothold_regularization` cost (the `foothold_regularization` block of contact_planning). Passive data. */
struct FootholdRegularizationParameters {
  scalar_t weight = 5.0;  // ||foot displacement||^2 per node, prefers short steps
};
/** The parameters of the `step_length` cost (the `step_length` block of contact_planning). Passive data. */
struct StepLengthParameters {
  scalar_t weight = 0.0;  // ||dp_swing - d_nom||^2 per running node, d_nom from v_cmd at the nominal cadence (StepLengthCost)
};
/** The parameters of the `terminal_dcm` cost (the `terminal_dcm` block of contact_planning). Passive data. */
struct TerminalDcmParameters {
  scalar_t weight = 200.0;  // ||DCM_N - zmp_{N-1}||^2, terminal capturability
  // false (terminal_dcm.target "rest"): the DCM is drawn onto the last ZMP, i.e. the plan comes to rest at the end of
  // the horizon, which shortens the steps in the horizon at speed. true ("commanded_velocity"): the terminal DCM xi_N is
  // drawn to zmp_{N-1} + v_cmd / omega, the offset of a CoM over the foot that keeps moving at the commanded velocity,
  // so the plan is asked to keep walking, not to stop.
  bool trackCommandedVelocity = false;
};
/** The parameters of the `zmp_support_region` soft constraint (the `zmp_support_region` block of contact_planning). Passive data. */
struct ZmpSupportRegionParameters {
  // ZMP support region half-widths. Single support: a box of these half-widths around the stance foot. Double support:
  // along the heading a box of half-width halfWidthX around the midpoint of the feet, laterally the strip between the
  // right foot minus and the left foot plus halfWidthY. 0 = from the sole's footprint (model parameters).
  scalar_t halfWidthX = 0.08;  // [m] along the heading
  scalar_t halfWidthY = 0.04;  // [m] lateral
  std::optional<SlackPenalty> slack;
};
/** The parameters of the `reachability` soft constraint (the `reachability` block of contact_planning). Passive data. */
struct ReachabilityParameters {
  scalar_t reachX = 0.45;       // foot x offset from the CoM must stay within [-reachX, reachX]
  scalar_t reachYInner = 0.05;  // left foot y - CoM y >= reachYInner (mirrored for the right foot)
  scalar_t reachYOuter = 0.45;  // left foot y - CoM y <= reachYOuter (mirrored for the right foot)
  std::optional<SlackPenalty> slack;
};
/** The parameters of the `foot_separation` soft constraint (the `foot_separation` block of contact_planning). Passive data. */
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
/** The parameters of the `contact_switch` assignment cost (the `contact_switch` block of contact_planning). Passive data. */
struct ContactSwitchParameters {
  scalar_t cost = 0.2;  // linear cost per lift-off / touch-down event
};
/** The parameters of the `plan_consistency` assignment cost (the `plan_consistency` block of contact_planning). Passive data. */
struct PlanConsistencyParameters {
  scalar_t cost = 0.5;  // cost per node whose contact differs from the previous plan (hysteresis)
};
/** The parameters of the `double_support_penalty` assignment cost (the `double_support_penalty` block of contact_planning). Passive data.
 */
struct DoubleSupportPenaltyParameters {
  // Cost per node at which both feet are in contact, which buys the weight transfer of a step against the double
  // support the ZMP terms would otherwise pay for. The term cannot tell a transfer apart from standing, so it also
  // prices standing still (every node of it): above ~0.2 with the Atlas gait limits the planner would rather step in
  // place than stand, and at 5.0 it marches continuously at a zero velocity command. Keep it well under that; 0.1
  // removes the transition double support at walking speed and leaves standing alone.
  scalar_t cost = 0.1;
};
/** The parameters of the `diving` search stage (the `diving` block of contact_planning). Passive data. */
struct DivingParameters {
  int maxDiveIterations = 64;
};
/** The parameters of the `event_shift_local_search` search stage (the `event_shift_local_search` block of contact_planning). Passive data.
 */
struct EventShiftLocalSearchParameters {
  int iterations = 10;      // rounds of event-shift local search after the branch-and-bound (0 disables)
  scalar_t maxTime = 0.05;  // [s] time budget of the local search
};
/**
 * `cadence_stretch`: the cadence the planner can express is quantized by the node grid, because a phase lasts a whole
 * number of nodes. minSwingNodes() = ceil(minSwingDuration / dt) and maxSwingNodes() = floor(maxSwingDuration / dt), so
 * at dt 0.1 with swing limits [0.4, 0.5] the swing is 4 or 5 nodes and nothing between - a 25% jump in the step the
 * commanded speed needs. This stage recovers the interval without leaving the grid: under a FIXED contact pattern the
 * cadence simply is the grid scale, so stretching the node duration to s * dt re-times every phase together. Every term
 * is already an exact function of the node duration (the LIP block is cosh/sinh of omega dt, the heading block is
 * linear in dt, step_length's nominal displacement is v dt ratio, terminal_dcm's gain is exp(2 omega dt)), and
 * ContactPlan carries its own dt, so a stretched plan needs no new representation.
 *
 * The stretch is bounded below by 1: s < 1 shrinks the committed window below commitTime and the horizon below
 * mpc.timeHorizon, which makes the merge pad the tail with STANCE and throws away the last steps' anticipation. And
 * above by the commit window, since the boundary itself is not stretched.
 */
struct CadenceStretchParameters {
  int samples = 0;  // stretch candidates evaluated after the branch-and-bound (0 disables the stage)
  // Upper bound on s; the admissible range is also clipped by the gait limits and by the commit window
  // (CadenceStretchStage::commitWindowStretch: the last committed node must stay the one live at the commit boundary).
  scalar_t maxStretch = 1.25;
};
/** The parameters of the `heading_relinearization` search stage (the `heading_relinearization` block of contact_planning). Passive data. */
struct HeadingRelinearizationParameters {
  int passes = 1;  // re-linearizations of the heading frame at the incumbent (0 disables)
};
/** The parameters of the `phase_resetting` execution rule (the `phase_resetting` block of contact_planning). Passive data. */
struct PhaseResettingParameters {
  scalar_t earlyTouchdownMinSwingRatio = 0.25;       // contact during this initial fraction of the nominal swing is ignored (scuffing)
  scalar_t earlyTouchdownMinContactDuration = 0.02;  // [s] contact must persist this long before the swing is ended (debounce)
  // [s] a touch-down closer than this to its scheduled time is executed as planned instead of ending the swing early.
  // Ending it early puts the foot in contact from the measured contact until its scheduled touch-down, and when the
  // gait exchanges support in a single instant (minDoubleSupportDuration: 0) that scheduled touch-down is the other
  // foot's lift-off, so the truncation opens a double support exactly as long as the foot was early. Landing within a
  // few milliseconds of the plan is the common case, not the exception, so this also bounds the shortest double
  // support the rule can create. Keep it above earlyTouchdownMinContactDuration, or the debounce alone already pushes
  // every truncation into that window. 0 restores the unguarded behavior.
  scalar_t earlyTouchdownMinAdvance = 0.04;
  scalar_t maxLateTouchdownExtension = 0.15;    // [s] total extension budget of a swing past its planned touch-down
  scalar_t lateTouchdownExtensionStep = 0.05;   // [s] the touch-down is pushed this far ahead of the current time per cycle
  scalar_t lateTouchdownSearchVelocity = 0.05;  // [m/s] the foot height target descends at this rate during the extension
};
/** The parameters of the `energy_cadence_modulation` execution rule (the `energy_cadence_modulation` block of contact_planning). Passive
 * data. */
struct EnergyCadenceModulationParameters {
  scalar_t gain = 0.01;     // [s/J] touch-down shift = -gain * (E - E_pred), E = m (v^2 - w^2 x^2) / 2, full model mass
  scalar_t deadband = 0.0;  // [J] deviations within this band re-time nothing, beyond it the shift is measured from the band's edge
};
/** The parameters of the `dcm_step_adjustment` execution rule (the `dcm_step_adjustment` block of contact_planning). Passive data. */
struct DcmStepAdjustmentParameters {
  scalar_t gain = 0.5;        // gain on the closed-form LIP step adjustment (1 = exact compensation of the increment)
  scalar_t maxOffset = 0.05;  // [m] bound on the landing target offset (also clipped to the reachable region)
};

/**
 * The contact planner's whole configuration: the planner settings, the shared parameters, the formulation (the term
 * lists) and one parameter block per term, as contactPlanningConfigFromConfig() converts them from contact_planning.textproto.
 * A value type; validateStatus() checks it. Not synchronized: each thread works on its own copy.
 */
struct ContactPlanningConfig {
  PlannerSettings planner;
  SharedParameters shared;
  ContactPlanningFormulation formulation;

  HlipParameters hlip;

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
  CadenceStretchParameters cadenceStretch;
  HeadingRelinearizationParameters headingRelinearization;
  PhaseResettingParameters phaseResetting;
  EnergyCadenceModulationParameters energyCadenceModulation;
  DcmStepAdjustmentParameters dcmStepAdjustment;

  static constexpr scalar_t kDefaultFootYawOffset = 0.5;  // [rad] symmetric fallback when no hip yaw joint can be identified

  // ---- conveniences over the blocks ----
  bool usesHeadingModel() const { return formulation.usesHeadingModel(); }
  void setHeadingModel(bool on) { formulation.setHeadingModel(on); }

  /** sqrt(gravity / comHeight); NaN while shared.comHeight is unset (validateStatus() refuses that). */
  scalar_t omega() const { return std::sqrt(shared.gravity / pendulumHeight()); }
  /** [m] shared.comHeight; NaN while it is unset (validateStatus() refuses that), for the code that runs validated. */
  scalar_t pendulumHeight() const { return shared.comHeight.value_or(std::numeric_limits<scalar_t>::quiet_NaN()); }
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
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      if (!(hipYawRange.lower[foot] < 0.0 && hipYawRange.upper[foot] > 0.0)) return false;
    }
    return true;
  }

  // Duration limits in planner nodes (conservative rounding).
  int minSwingNodes() const { return std::max(1, static_cast<int>(std::ceil(shared.gaitLimits.minSwingDuration / planner.dt - 1.0e-9))); }
  int maxSwingNodes() const {
    return std::max(minSwingNodes(), static_cast<int>(std::floor(shared.gaitLimits.maxSwingDuration / planner.dt + 1.0e-9)));
  }
  int minContactNodes() const {
    return std::max(1, static_cast<int>(std::ceil(shared.gaitLimits.minContactDuration / planner.dt - 1.0e-9)));
  }
  int minDoubleSupportNodes() const {
    if (shared.gaitLimits.minDoubleSupportDuration <= 0.0) return 0;
    return static_cast<int>(std::ceil(shared.gaitLimits.minDoubleSupportDuration / planner.dt - 1.0e-9));
  }
  int maxContactNodes() const {
    if (shared.gaitLimits.maxContactDuration <= 0.0) return 0;
    return std::max(minContactNodes(), static_cast<int>(std::floor(shared.gaitLimits.maxContactDuration / planner.dt + 1.0e-9)));
  }
  int commitNodes() const { return std::max(0, static_cast<int>(std::ceil(planner.commitTime / planner.dt - 1.0e-9))); }

  /**
   * OK, or InvalidArgument for the first inconsistent key (every block, and the formulation). Each message names the
   * key to change with its path inside the `contact_planning` block (`foothold_regularization.weight`,
   * `planner.maxSolveTime`, `hlip.minStepWidth`, ...) and the value it read. On success the warnings() are logged with
   * LOG(WARNING).
   */
  absl::Status validateStatus() const;

  /**
   * Combinations that are legal but documented to fall or to block, one message per finding naming the keys to move,
   * empty when there is nothing to say. Per planner:
   *  - hlip: planned_com_override missing from `execution` (the robot sidesteps and falls, H-LIP README 3c); a first
   *    step out of a standstill that does not fit hlip.maxStepWidth (README 3b); a background thread (a stale feedback
   *    law, README 6); a sustained sidestep that is clipped on every step; an instantaneous exchange of support with
   *    an uncapped commit extension (the planner stalls, PlannerSettings::maxCommitExtension).
   *  - lip_miqp: a synchronous planner (the branch-and-bound blocks every MPC solve); a commit window shorter than the
   *    solve budget (plans arrive stale and are dropped); a zero minimum double support with an uncapped commit
   *    extension (the same stall).
   *
   * These are warnings, not errors, because the integration tests run some of them on purpose (a synchronous
   * lip_miqp) and because a shipped robot's file must not stop loading over a few millimeters of step width.
   * validateStatus() is what emits them; they are returned rather than only logged so that a test can assert on them.
   */
  std::vector<std::string> warnings() const;

  /**
   * The shortest swing the configured planner emits [s]: hlip.sspDuration under `hlip`, whose every swing lasts exactly
   * that, and shared.gait_limits.minSwingDuration rounded up to whole nodes under `lip_miqp`.
   */
  scalar_t shortestPlannedSwingDuration() const;

  /**
   * A warning when the task file's swing_trajectory_config.swing_time_scale exceeds shortestPlannedSwingDuration(), empty
   * otherwise. SwingTrajectoryPlanner scales every swing shorter than swingTimeScale down in height and velocity by
   * duration / swingTimeScale, so such a configuration lands every step short and low without any other sign. The
   * two keys live in different files and a GUI reload can move either one, so, in addition to the LINT pair between
   * the files, the reference manager checks this at run time, where it knows both: at construction, on every reload
   * of this configuration, and at the first solver run after the swing trajectory planner's swingTimeScale changed.
   */
  std::optional<std::string> swingTimeScaleWarning(scalar_t swingTimeScale) const;
};

}  // namespace ocs2::humanoid
