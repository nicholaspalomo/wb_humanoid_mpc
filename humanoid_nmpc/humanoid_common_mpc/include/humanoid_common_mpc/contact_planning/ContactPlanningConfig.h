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

#include <cmath>
#include <string>

#include <ocs2_core/Types.h>

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
  scalar_t zmpHalfWidthX = 0.08;     // ZMP box half-width around the anchor foot (along x)
  scalar_t zmpHalfWidthY = 0.04;     // ZMP box half-width around the anchor foot (along y)
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

/** Loads the configuration from a task file. Missing keys keep their defaults. */
ContactPlanningConfig loadContactPlanningConfig(const std::string& taskFile,
                                                const std::string& prefix = "contact_planning.",
                                                bool verbose = false);

}  // namespace ocs2::humanoid
