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
#include <string>
#include <utility>

#include <ocs2_core/Types.h>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * Parameters of the mixed-integer contact planner (task.yaml block `contact_planning`).
 *
 * Geometry is expressed in the yaw-aligned frame of the base at planning time (x forward, y left).
 */
struct ContactPlanningConfig {
  // Horizon and timing
  scalar_t dt = 0.1;           // [s] planner node duration
  int numNodes = 12;           // planning horizon = numNodes * dt, should cover the MPC horizon
  scalar_t commitTime = 0.25;  // [s] contacts within this window keep the applied schedule (must cover planner latency)
  scalar_t comHeight = 0.85;   // [m] LIP height, omega = sqrt(g / comHeight)
  scalar_t gravity = 9.81;     // [m/s^2]

  // Phase duration limits [s]
  scalar_t minSwingDuration = 0.3;
  scalar_t maxSwingDuration = 0.6;
  scalar_t minContactDuration = 0.15;
  scalar_t maxContactDuration = 0.0;        // <= 0 disables the limit (standing is allowed indefinitely)
  bool enforceAlternatingFeet = true;       // a foot may not swing twice in a row (biped gait structure)
  scalar_t minDoubleSupportDuration = 0.1;  // [s] after a touch-down the other foot stays down at least this long (0 disables)

  // Support geometry [m], yaw frame
  // ZMP support region half-widths. Single support: a box of these half-widths around the stance foot. Double support:
  // along the heading a box of half-width zmpHalfWidthX around the midpoint of the feet, laterally the strip between the
  // right foot minus and the left foot plus zmpHalfWidthY (LipContactPlanner, ZMP rows).
  scalar_t zmpHalfWidthX = 0.08;     // [m] along the heading
  scalar_t zmpHalfWidthY = 0.04;     // [m] lateral
  scalar_t nominalStepWidth = 0.25;  // lateral distance left foot - right foot the planner is drawn to
  scalar_t minStepWidth = 0.15;      // self-collision margin
  scalar_t maxStepWidth = 0.45;
  scalar_t maxStepLength = 0.5;  // |x_left - x_right| bound
  scalar_t reachX = 0.45;        // foot x offset from the CoM must stay within [-reachX, reachX]
  scalar_t reachYInner = 0.05;   // left foot y - CoM y >= reachYInner (mirrored for the right foot)
  scalar_t reachYOuter = 0.45;   // left foot y - CoM y <= reachYOuter (mirrored for the right foot)
  scalar_t bigM = 1.0;           // [m] big-M for the ZMP / foothold disjunctions (must exceed maxStepLength)

  // Objective weights
  scalar_t velocityTrackingWeight = 20.0;        // ||v_com - v_cmd||^2 per node
  scalar_t zmpRegularizationWeight = 2.0;        // ||zmp - com||^2 per node, keeps the ZMP under the CoM when possible
  scalar_t footholdRegularizationWeight = 5.0;   // ||foot displacement||^2 per node, prefers short steps
  scalar_t stepWidthWeight = 10.0;               // (step width - nominalStepWidth)^2 per node
  scalar_t contactSwitchCost = 0.2;              // linear cost per lift-off / touch-down event
  scalar_t planConsistencyCost = 0.5;            // cost per node whose contact differs from the previous plan (hysteresis)
  scalar_t previousFootholdWeight = 5.0;         // ||p_foot - p_foot,previous plan||^2 per node, damps foothold jitter
  scalar_t terminalDcmWeight = 200.0;            // ||DCM_N - zmp_{N-1}||^2, terminal capturability
  scalar_t constraintSlackWeight = 1.0e4;        // quadratic penalty on the soft ZMP / reachability slacks
  scalar_t constraintSlackLinearWeight = 100.0;  // linear penalty on the same slacks

  // Mixed-integer solver
  int maxBranchAndBoundNodes = 200;
  scalar_t maxSolveTime = 0.1;  // [s]
  int maxQpIterations = 60;
  int localSearchIterations = 10;      // rounds of event-shift local search after the branch-and-bound (0 disables)
  scalar_t localSearchMaxTime = 0.05;  // [s] time budget of the local search
  bool verbose = false;

  // Runtime integration
  bool runInBackgroundThread = true;  // false: plan synchronously inside the MPC's pre-solve hook
  scalar_t planningFrequency = 10.0;  // [Hz] upper bound on the background planning rate

  // Adaptive execution of the schedule between plans (ContactScheduleAdaptation.h), generic in the number of feet.
  //
  // All three features change the closed-loop behaviour of the controller and are therefore opt-in. With them disabled
  // the reference manager merges plans exactly as it did before they existed, which is the behaviour every gait is
  // tuned against. Enable one at a time and validate it in simulation.
  bool enablePhaseResetting = false;                 // switch a swing foot to contact when it touches down early, extend when late
  scalar_t earlyTouchdownMinSwingRatio = 0.25;       // contact during this initial fraction of the nominal swing is ignored (scuffing)
  scalar_t earlyTouchdownMinContactDuration = 0.02;  // [s] contact must persist this long before the swing is ended (debounce)
  scalar_t maxLateTouchdownExtension = 0.15;         // [s] total extension budget of a swing past its planned touch-down
  scalar_t lateTouchdownExtensionStep = 0.05;        // [s] the touch-down is pushed this far ahead of the current time per cycle
  scalar_t lateTouchdownSearchVelocity = 0.05;       // [m/s] the foot height target descends at this rate during the extension

  // Closed-form capture-point step adjustment and orbital-energy cadence modulation. Both measure the deviation of the
  // measured centre of mass from the NMPC's own predicted trajectory (handed over after every solve), not from the
  // planner's Linear Inverted Pendulum: the whole-body controller chooses a different ZMP than the reduced model by
  // design, so a LIP reference would report that design difference as a "disturbance", amplified by exp(omega * plan
  // age) and again by exp(omega * time to touch-down), and saturate the bound every step. Against the NMPC's prediction
  // the error is zero while the robot does what the controller expects and non-zero only under a real disturbance.
  // dcmAdjustmentMaxOffset remains the safety bound.
  bool enableDcmStepAdjustment = false;    // move the landing target by the DCM error propagated to touch-down
  scalar_t dcmAdjustmentGain = 0.5;        // gain on the closed-form LIP step adjustment (1 = exact compensation)
  scalar_t dcmAdjustmentMaxOffset = 0.05;  // [m] bound on the landing target offset (also clipped to reachX / reachY*)

  bool enableEnergyCadenceModulation = false;  // move the touch-down of the swing in flight by the LIP orbital energy error
  scalar_t energyCadenceGain = 0.01;           // [s/J] touch-down shift = -gain * (E - E_plan), E = m (v^2 - w^2 x^2) / 2

  // Heading model. Off: the point-mass LIP, whose foothold frame is the base yaw at planning time and cannot express a
  // turn. On: the LIP for the centre of mass plus the whole-body heading (the angular centre of mass, ACoM, when the
  // robot has one, the base yaw otherwise) and its angular momentum about the vertical, driven by the yaw torques the
  // stance feet can carry: torsional friction on every stance foot and, in double support, the friction couple of the
  // two feet. Every foot gets a yaw that is pinned while the foot is in contact and must stay within hip range of the
  // heading. The foothold frame becomes the planned heading per node: the constraints are linearised around the
  // previous plan and re-solved at the incumbent (successive linearisation), which keeps the relaxations convex.
  // Turning in place becomes a stepping decision of the planner, bounded by the ground torques. Changes the closed-loop
  // behaviour: opt-in.
  bool useAcomDynamics = false;
  // Properties of the robot and the ground, not task-file keys: derived from the model and the wrench cone by
  // deriveContactPlanningModelParameters() and applied to every configuration (ContactPlanningModelParameters::applyTo).
  // The heading model refuses to plan until they are set. The yaw inertia is taken from the model at every plan.
  scalar_t torsionalFrictionTorque = 0.0;                          // [N m] |yaw torque| one stance foot carries
  scalar_t doubleSupportYawCouple = 0.0;                           // [N m] additional |yaw torque| from the couple of two stance feet
  feet_array_t<scalar_t> footYawOffsetLower = makeFeetArray(0.0);  // [rad] bounds on foot yaw - heading (hip yaw limits)
  feet_array_t<scalar_t> footYawOffsetUpper = makeFeetArray(0.0);
  static constexpr scalar_t kDefaultFootYawOffset = 0.5;  // [rad] symmetric fallback when no hip yaw joint can be identified
  scalar_t headingRateTrackingWeight = 20.0;              // (heading rate - commanded yaw rate)^2 per node
  scalar_t headingTrackingWeight = 5.0;                   // (heading - commanded heading)^2 per node
  scalar_t yawTorqueWeight = 1.0e-3;                      // yaw torque^2 per node
  scalar_t footYawTrackingWeight = 5.0;                   // (foot yaw - heading)^2 per node
  scalar_t footYawRegularizationWeight = 1.0;             // (foot yaw displacement)^2 per node
  int headingLinearizationPasses = 1;                     // re-linearisations of the heading frame at the incumbent (0 disables)
  bool planHeadingOverridesTarget = true;                 // the plan's heading replaces the commanded yaw in the MPC target trajectory

  /** Bounds on (foot yaw - heading) of a foot. */
  std::pair<scalar_t, scalar_t> footYawOffsetBounds(size_t foot) const { return {footYawOffsetLower[foot], footYawOffsetUpper[foot]}; }
  /** Symmetric foot yaw bounds, for tests and for robots without a model-derived value. */
  void setSymmetricFootYawOffset(scalar_t offset) {
    footYawOffsetLower = makeFeetArray(-offset);
    footYawOffsetUpper = makeFeetArray(offset);
  }
  /** True once the model-derived parameters of the heading model have been applied. */
  bool hasModelParameters() const {
    // The foot yaw bounds are the marker: derived bounds always straddle zero, and a zero torque limit is a legitimate
    // derived value (a frictionless sole), not a missing one.
    if (torsionalFrictionTorque < 0.0 || doubleSupportYawCouple < 0.0) return false;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (!(footYawOffsetLower[foot] < 0.0 && footYawOffsetUpper[foot] > 0.0)) return false;
    }
    return true;
  }

  scalar_t omega() const { return std::sqrt(gravity / comHeight); }
  scalar_t horizon() const { return dt * static_cast<scalar_t>(numNodes); }

  // Duration limits in planner nodes (conservative rounding).
  int minSwingNodes() const { return std::max(1, static_cast<int>(std::ceil(minSwingDuration / dt - 1e-9))); }
  int maxSwingNodes() const { return std::max(minSwingNodes(), static_cast<int>(std::floor(maxSwingDuration / dt + 1e-9))); }
  int minContactNodes() const { return std::max(1, static_cast<int>(std::ceil(minContactDuration / dt - 1e-9))); }
  int minDoubleSupportNodes() const {
    if (minDoubleSupportDuration <= 0.0) return 0;
    return static_cast<int>(std::ceil(minDoubleSupportDuration / dt - 1e-9));
  }
  int maxContactNodes() const {
    if (maxContactDuration <= 0.0) return 0;
    return std::max(minContactNodes(), static_cast<int>(std::floor(maxContactDuration / dt + 1e-9)));
  }
  int commitNodes() const { return std::max(0, static_cast<int>(std::ceil(commitTime / dt - 1e-9))); }

  /** Throws std::invalid_argument if the configuration is inconsistent. */
  void validate() const;
};

/**
 * Loads the configuration from a task file. Missing keys keep their defaults. With `validate` false the values that may
 * be 0 for "derive from the model" are accepted as they are; call validate() after ContactPlanningModelParameters::applyTo().
 */
ContactPlanningConfig loadContactPlanningConfig(const std::string& taskFile,
                                                const std::string& prefix = "contact_planning.",
                                                bool verbose = false,
                                                bool validate = true);

}  // namespace ocs2::humanoid
