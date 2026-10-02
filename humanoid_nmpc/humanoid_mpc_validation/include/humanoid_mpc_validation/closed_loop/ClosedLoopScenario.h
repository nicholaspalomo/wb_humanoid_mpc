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

#include <string>
#include <vector>

#include <Eigen/Core>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::validation {

/** One piece of a scenario: the operator holds this velocity for `duration`. */
struct CommandSegment {
  double duration = 0.0;         ///< [s]
  double forwardVelocity = 0.0;  ///< [m/s] along the heading
  double lateralVelocity = 0.0;  ///< [m/s] to the left of the heading
  double yawRate = 0.0;          ///< [rad/s] counterclockwise seen from above
};

/**
 * A closed-loop scenario of the quaternion design's section 4.5: what the operator commands once the robot stands on its
 * own. Before the commands, the lockstep runner settles the robot in JOINT_PD on the gantry, enters WB_MPC, releases the
 * gantry and lets the robot stand for `standingTime`; the metrics are evaluated over the commands only.
 */
struct ClosedLoopScenario {
  std::string name;
  std::string description;
  std::vector<CommandSegment> segments;
  double standingTime = 2.0;  ///< [s] standing still between the gantry release and the first command
};

// LINT.IfChange(scenario_names)
/**
 * The scenarios, by name:
 *  - standing: 10 s at rest;
 *  - walk_0p5: 0.5 m/s forward for 15 s, then 2 s at rest;
 *  - walk_0p3: the same at 0.3 m/s, the slower variant section 4.5 allows for a robot limited at 0.5 m/s;
 *  - lateral_0p2: 0.2 m/s to the left for 10 s, then 2 s at rest;
 *  - turn_in_place_720: +0.5 rad/s through 720 degrees and -0.5 rad/s back, then 2 s at rest;
 *  - turn_1radps: 1 rad/s for 12.6 s, then 2 s at rest;
 *  - arc: 0.3 m/s forward while turning at 0.3 rad/s for 15 s, then 2 s at rest;
 *  - smoke: 1.5 s of 0.3 m/s and 0.2 rad/s after 0.5 s of standing, cheap enough for the determinism test.
 */
const std::vector<ClosedLoopScenario>& closedLoopScenarios();
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/README.md:scenario_names)

/** The scenario named `name`; NotFound listing the names when there is none. */
absl::StatusOr<ClosedLoopScenario> findClosedLoopScenario(absl::string_view name);

/** [s] The sum of the segments' durations. */
double getCommandDuration(const ClosedLoopScenario& scenario);

/** [rad] The largest heading change the yaw-rate commands integrate to (0 for a scenario that never turns left). */
double getCommandedHeadingPeak(const ClosedLoopScenario& scenario);

/**
 * (forward velocity, lateral velocity, yaw rate) commanded `timeSinceStart` seconds after the first command: the
 * segment holding that time (a segment holds [start, start + duration)), zero before the first and after the last.
 */
Eigen::Vector3d commandAt(const ClosedLoopScenario& scenario, double timeSinceStart);

/** What the base-controller GUI publishes for a physical command (humanoid_mpc_msgs/WalkingVelocityCommand). */
struct GuiVelocityCommand {
  /// (linear_velocity_x, linear_velocity_y, desired_pelvis_height, angular_velocity_z) as the message carries them:
  /// the stick positions in [-1, 1], which ProceduralMpcMotionManager scales by the command limits of reference.yaml,
  /// and the pelvis height above the ground the GUI's slider sends.
  Eigen::Vector4d message;
  /// The physical command needed a stick beyond [-1, 1] and was cut there: the robot cannot be asked for it.
  bool saturated = false;
};

/**
 * The GUI message that commands `command` (forward, lateral, yaw rate) on a robot with `commandLimits`
 * (maxDisplacementVelocityX, maxDisplacementVelocityY, maxRotationVelocity of reference.yaml), at `pelvisHeight` (the
 * GUI sends reference.yaml's defaultBaseHeight). The sticks are clamped as the controller clamps a received message.
 */
GuiVelocityCommand toGuiVelocityCommand(const Eigen::Vector3d& command, const Eigen::Vector3d& commandLimits, double pelvisHeight);

/**
 * The clamp the MPC node applies to a received message, its own node::clampWalkingVelocityCommand()
 * (WalkingVelocityCommandConversions.h): sticks to [-1, 1], pelvis height to [0.2, 1.0] m.
 */
Eigen::Vector4d clampGuiVelocityMessage(const Eigen::Vector4d& message);

}  // namespace ocs2::humanoid::validation
