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

#include "humanoid_common_mpc/locomotion_heuristics/foothold/HighSpeedTurningHeuristic.h"

#include <cmath>

#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

namespace ocs2::humanoid {

namespace {

/** Rotates a (forward, left) offset of the base's yaw frame into the world. */
vector2_t toWorld(const vector2_t& inBaseFrame, scalar_t yaw) {
  const scalar_t cosYaw = std::cos(yaw);
  const scalar_t sinYaw = std::sin(yaw);
  return vector2_t(cosYaw * inBaseFrame.x() - sinYaw * inBaseFrame.y(), sinYaw * inBaseFrame.x() + cosYaw * inBaseFrame.y());
}

}  // namespace

absl::Status HighSpeedTurningHeuristic::configure(const LocomotionHeuristicConfig& config,
                                                  const LocomotionHeuristicModelParameters& /*model*/) {
  parameters_ = config.highSpeedTurning;
  return absl::OkStatus();
}

vector2_t HighSpeedTurningHeuristic::offset(const FootholdHeuristicContext& context) const {
  // H_r(pdot x omega) = a1 (pdot x omega) + a0, Bledt Table C.2.
  //
  // The full cross product of a 3D velocity and a 3D angular rate has three components, but Bledt's own reduction
  // (equation 4.27) drops the vertical velocity and the roll and pitch rates as negligible during regular
  // locomotion, which leaves omega = psidot e_z and a horizontal velocity. Then
  //
  //     v x omega = (v_x, v_y, 0) x (0, 0, psidot) = (v_y psidot, -v_x psidot, 0).
  //
  // Evaluated in the BASE frame, so that "forward" and "lateral" below mean the robot's own, and the whole offset is
  // rotated into the world afterwards. This is the term that vanishes at zero speed however fast the robot spins,
  // which is what distinguishes it from in_place_turning.
  const vector2_t commandedInBaseFrame = toWorld(context.commandedVelocity, -context.measuredBaseYaw);
  const scalar_t crossForward = commandedInBaseFrame.y() * context.commandedYawRate;
  const scalar_t crossLateral = -commandedInBaseFrame.x() * context.commandedYawRate;
  const vector2_t offsetInBaseFrame(parameters_.forwardPerCrossTerm * crossForward + parameters_.forwardOffset,
                                    parameters_.lateralPerCrossTerm * crossLateral + context.side * parameters_.lateralOffset);
  return toWorld(offsetInBaseFrame, context.measuredBaseYaw);
}

std::string HighSpeedTurningHeuristic::describe() const {
  return absl::StrCat("high_speed_turning: d_forward = ", parameters_.forwardPerCrossTerm, " * (v_y * psidot) + ",
                      parameters_.forwardOffset, " [m], d_lateral = ", parameters_.lateralPerCrossTerm, " * (-v_x * psidot) + side * ",
                      parameters_.lateralOffset, " [m]");
}

}  // namespace ocs2::humanoid
