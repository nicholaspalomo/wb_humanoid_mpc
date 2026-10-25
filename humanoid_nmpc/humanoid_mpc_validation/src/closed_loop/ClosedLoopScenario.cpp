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

#include "humanoid_mpc_validation/closed_loop/ClosedLoopScenario.h"

#include <cmath>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc_app/node/WalkingVelocityCommandConversions.h"

namespace ocs2::humanoid::validation {
namespace {

/** [s] Standing still after every motion, so that the metrics include the stop.
 */
constexpr double kStopDuration = 2.0;

// The commands of each scenario (closedLoopScenarios()): durations [s],
// velocities [m/s] and yaw rates [rad/s].
constexpr double kStandingDuration = 10.0;
constexpr double kWalkDuration = 15.0;
constexpr double kWalkVelocity = 0.5;
constexpr double kSlowWalkVelocity =
    0.3;  // the slower walking variant of the design's section 4.5
constexpr double kLateralWalkDuration = 10.0;
constexpr double kLateralVelocity = 0.2;
constexpr double kTurnInPlaceAngle = 4.0 * M_PI;  // [rad] 720 degrees
constexpr double kTurnInPlaceYawRate = 0.5;
constexpr double kTurnDuration = 12.6;
constexpr double kTurnYawRate = 1.0;
constexpr double kArcDuration = 15.0;
constexpr double kArcVelocity = 0.3;
constexpr double kArcYawRate = 0.3;
constexpr double kSmokeDuration = 1.5;
constexpr double kSmokeVelocity = 0.3;
constexpr double kSmokeYawRate = 0.2;
constexpr double kSmokeStandingTime = 0.5;

/** At rest for `duration` seconds. */
CommandSegment rest(double duration) {
  return CommandSegment{.duration = duration};
}

std::vector<ClosedLoopScenario> makeScenarios() {
  // 720 degrees at 0.5 rad/s.
  const double turnDuration = kTurnInPlaceAngle / kTurnInPlaceYawRate;
  std::vector<ClosedLoopScenario> scenarios;
  scenarios.push_back({"standing",
                       absl::StrCat("at rest for ", kStandingDuration, " s"),
                       {rest(kStandingDuration)}});
  scenarios.push_back({"walk_0p5",
                       absl::StrCat(kWalkVelocity, " m/s forward for ",
                                    kWalkDuration, " s, then at rest"),
                       {CommandSegment{.duration = kWalkDuration,
                                       .forwardVelocity = kWalkVelocity},
                        rest(kStopDuration)}});
  scenarios.push_back(
      {"walk_0p3",
       absl::StrCat(kSlowWalkVelocity, " m/s forward for ", kWalkDuration,
                    " s, then at rest (the slower walking variant)"),
       {CommandSegment{.duration = kWalkDuration,
                       .forwardVelocity = kSlowWalkVelocity},
        rest(kStopDuration)}});
  scenarios.push_back({"lateral_0p2",
                       absl::StrCat(kLateralVelocity, " m/s to the left for ",
                                    kLateralWalkDuration, " s, then at rest"),
                       {CommandSegment{.duration = kLateralWalkDuration,
                                       .lateralVelocity = kLateralVelocity},
                        rest(kStopDuration)}});
  scenarios.push_back(
      {"turn_in_place_720",
       absl::StrCat("+", kTurnInPlaceYawRate,
                    " rad/s in place through 720 degrees, -",
                    kTurnInPlaceYawRate, " rad/s back, then at rest"),
       {CommandSegment{.duration = turnDuration,
                       .yawRate = kTurnInPlaceYawRate},
        CommandSegment{.duration = turnDuration,
                       .yawRate = -kTurnInPlaceYawRate},
        rest(kStopDuration)}});
  scenarios.push_back(
      {"turn_1radps",
       absl::StrCat(kTurnYawRate, " rad/s in place for ", kTurnDuration,
                    " s, then at rest"),
       {CommandSegment{.duration = kTurnDuration, .yawRate = kTurnYawRate},
        rest(kStopDuration)}});
  scenarios.push_back(
      {"arc",
       absl::StrCat(kArcVelocity, " m/s forward while turning at ", kArcYawRate,
                    " rad/s for ", kArcDuration, " s, then at rest"),
       {CommandSegment{.duration = kArcDuration,
                       .forwardVelocity = kArcVelocity,
                       .yawRate = kArcYawRate},
        rest(kStopDuration)}});
  ClosedLoopScenario smoke{
      .name = "smoke",
      .description = absl::StrCat(
          kSmokeVelocity, " m/s forward while turning at ", kSmokeYawRate,
          " rad/s for ", kSmokeDuration, " s after ", kSmokeStandingTime,
          " s of standing (the determinism test)"),
      .segments = {CommandSegment{.duration = kSmokeDuration,
                                  .forwardVelocity = kSmokeVelocity,
                                  .yawRate = kSmokeYawRate}}};
  smoke.standingTime = kSmokeStandingTime;
  scenarios.push_back(smoke);
  return scenarios;
}

}  // namespace

const std::vector<ClosedLoopScenario>& closedLoopScenarios() {
  static const absl::NoDestructor<std::vector<ClosedLoopScenario>> kScenarios(
      makeScenarios());
  return *kScenarios;
}

absl::StatusOr<ClosedLoopScenario> findClosedLoopScenario(
    absl::string_view name) {
  std::string names;
  for (const ClosedLoopScenario& scenario : closedLoopScenarios()) {
    if (scenario.name == name) return scenario;
    absl::StrAppend(&names, names.empty() ? "" : ", ", scenario.name);
  }
  return absl::NotFoundError(
      absl::StrCat("[findClosedLoopScenario] no scenario '", name,
                   "'; the scenarios are ", names));
}

double getCommandDuration(const ClosedLoopScenario& scenario) {
  double duration = 0.0;
  for (const CommandSegment& piece : scenario.segments)
    duration += piece.duration;
  return duration;
}

Eigen::Vector3d commandAt(const ClosedLoopScenario& scenario,
                          double timeSinceStart) {
  if (timeSinceStart < 0.0) return Eigen::Vector3d::Zero();
  double start = 0.0;
  for (const CommandSegment& piece : scenario.segments) {
    if (timeSinceStart < start + piece.duration)
      return Eigen::Vector3d(piece.forwardVelocity, piece.lateralVelocity,
                             piece.yawRate);
    start += piece.duration;
  }
  return Eigen::Vector3d::Zero();
}

Eigen::Vector4d clampGuiVelocityMessage(const Eigen::Vector4d& message) {
  return node::clampWalkingVelocityCommand(WalkingVelocityCommand(message))
      .toVector();
}

GuiVelocityCommand toGuiVelocityCommand(const Eigen::Vector3d& command,
                                        const Eigen::Vector3d& commandLimits,
                                        double pelvisHeight) {
  const Eigen::Vector4d sticks(
      command(kCommandForwardVelocityIndex) /
          commandLimits(kCommandForwardVelocityIndex),
      command(kCommandLateralVelocityIndex) /
          commandLimits(kCommandLateralVelocityIndex),
      pelvisHeight,
      command(kCommandYawRateIndex) / commandLimits(kCommandYawRateIndex));
  GuiVelocityCommand gui;
  gui.message = clampGuiVelocityMessage(sticks);
  // A stick beyond its range is cut; a few ulps of the division are not a
  // saturation.
  constexpr double kTolerance = 1.0e-12;
  // The sticks of the linear velocity and of the yaw rate, at their entries of
  // the message (a WalkingVelocityCommand).
  gui.saturated =
      std::abs(sticks(WalkingVelocityCommand::kLinearVelocityXIndex)) >
          1.0 + kTolerance ||
      std::abs(sticks(WalkingVelocityCommand::kLinearVelocityYIndex)) >
          1.0 + kTolerance ||
      std::abs(sticks(WalkingVelocityCommand::kAngularVelocityZIndex)) >
          1.0 + kTolerance;
  return gui;
}

}  // namespace ocs2::humanoid::validation
