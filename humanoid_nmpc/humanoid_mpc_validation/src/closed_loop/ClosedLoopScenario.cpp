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

#include <algorithm>
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

/** [s] Standing still after every motion, so that the metrics include the stop. */
constexpr double kStopDuration = 2.0;

/** At rest for `duration` seconds. */
CommandSegment rest(double duration) {
  return CommandSegment{.duration = duration};
}

std::vector<ClosedLoopScenario> makeScenarios() {
  // 720 degrees at 0.5 rad/s.
  const double turnDuration = 4.0 * M_PI / 0.5;
  std::vector<ClosedLoopScenario> scenarios;
  scenarios.push_back({"standing", "at rest for 10 s", {rest(/*duration=*/10.0)}});
  scenarios.push_back({"walk_0p5",
                       "0.5 m/s forward for 15 s, then at rest",
                       {CommandSegment{.duration = 15.0, .forwardVelocity = 0.5}, rest(kStopDuration)}});
  scenarios.push_back({"walk_0p3",
                       "0.3 m/s forward for 15 s, then at rest (the slower walking variant)",
                       {CommandSegment{.duration = 15.0, .forwardVelocity = 0.3}, rest(kStopDuration)}});
  scenarios.push_back({"lateral_0p2",
                       "0.2 m/s to the left for 10 s, then at rest",
                       {CommandSegment{.duration = 10.0, .lateralVelocity = 0.2}, rest(kStopDuration)}});
  scenarios.push_back({"turn_in_place_720",
                       "+0.5 rad/s in place through 720 degrees, -0.5 rad/s back, then at rest",
                       {CommandSegment{.duration = turnDuration, .yawRate = 0.5}, CommandSegment{.duration = turnDuration, .yawRate = -0.5},
                        rest(kStopDuration)}});
  scenarios.push_back({"turn_1radps",
                       "1 rad/s in place for 12.6 s, then at rest",
                       {CommandSegment{.duration = 12.6, .yawRate = 1.0}, rest(kStopDuration)}});
  scenarios.push_back({"arc",
                       "0.3 m/s forward while turning at 0.3 rad/s for 15 s, then at rest",
                       {CommandSegment{.duration = 15.0, .forwardVelocity = 0.3, .yawRate = 0.3}, rest(kStopDuration)}});
  ClosedLoopScenario smoke{
      .name = "smoke",
      .description = "0.3 m/s forward while turning at 0.2 rad/s for 1.5 s after 0.5 s of standing (the determinism test)",
      .segments = {CommandSegment{.duration = 1.5, .forwardVelocity = 0.3, .yawRate = 0.2}}};
  smoke.standingTime = 0.5;
  scenarios.push_back(smoke);
  return scenarios;
}

}  // namespace

const std::vector<ClosedLoopScenario>& closedLoopScenarios() {
  static const absl::NoDestructor<std::vector<ClosedLoopScenario>> kScenarios(makeScenarios());
  return *kScenarios;
}

absl::StatusOr<ClosedLoopScenario> findClosedLoopScenario(absl::string_view name) {
  std::string names;
  for (const ClosedLoopScenario& scenario : closedLoopScenarios()) {
    if (scenario.name == name) return scenario;
    absl::StrAppend(&names, names.empty() ? "" : ", ", scenario.name);
  }
  return absl::NotFoundError(absl::StrCat("[findClosedLoopScenario] no scenario '", name, "'; the scenarios are ", names));
}

double getCommandDuration(const ClosedLoopScenario& scenario) {
  double duration = 0.0;
  for (const CommandSegment& piece : scenario.segments) duration += piece.duration;
  return duration;
}

double getCommandedHeadingPeak(const ClosedLoopScenario& scenario) {
  double heading = 0.0;
  double peak = 0.0;
  for (const CommandSegment& piece : scenario.segments) {
    heading += piece.yawRate * piece.duration;
    peak = std::max(peak, heading);
  }
  return peak;
}

Eigen::Vector3d commandAt(const ClosedLoopScenario& scenario, double timeSinceStart) {
  if (timeSinceStart < 0.0) return Eigen::Vector3d::Zero();
  double start = 0.0;
  for (const CommandSegment& piece : scenario.segments) {
    if (timeSinceStart < start + piece.duration) return Eigen::Vector3d(piece.forwardVelocity, piece.lateralVelocity, piece.yawRate);
    start += piece.duration;
  }
  return Eigen::Vector3d::Zero();
}

Eigen::Vector4d clampGuiVelocityMessage(const Eigen::Vector4d& message) {
  return node::clampWalkingVelocityCommand(WalkingVelocityCommand(message)).toVector();
}

GuiVelocityCommand toGuiVelocityCommand(const Eigen::Vector3d& command, const Eigen::Vector3d& commandLimits, double pelvisHeight) {
  const Eigen::Vector4d sticks(command(0) / commandLimits(0), command(1) / commandLimits(1), pelvisHeight, command(2) / commandLimits(2));
  GuiVelocityCommand gui;
  gui.message = clampGuiVelocityMessage(sticks);
  // A stick beyond its range is cut; a few ulps of the division are not a saturation.
  constexpr double kTolerance = 1.0e-12;
  gui.saturated =
      std::abs(sticks(0)) > 1.0 + kTolerance || std::abs(sticks(1)) > 1.0 + kTolerance || std::abs(sticks(3)) > 1.0 + kTolerance;
  return gui;
}

}  // namespace ocs2::humanoid::validation
