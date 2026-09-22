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

#include "humanoid_common_mpc/locomotion_heuristics/foothold/TranslationalSteppingHeuristic.h"

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

absl::Status TranslationalSteppingHeuristic::configure(const LocomotionHeuristicConfig& config,
                                                       const LocomotionHeuristicModelParameters& /*model*/) {
  parameters_ = config.translationalStepping;
  return absl::OkStatus();
}

vector2_t TranslationalSteppingHeuristic::offset(const FootholdHeuristicContext& context) const {
  // H_r(pdot) = a1 pdot + a0, Bledt Table C.2, in the base's yaw frame. The commanded velocity is in the world, so it
  // is rotated into the base frame, the affine law is applied there, and the result is rotated back - which is not
  // the same as applying the law in the world, because the forward and lateral coefficients differ and the robot's
  // forward is not the world's x.
  const vector2_t commandedInBaseFrame = toWorld(context.commandedVelocity, -context.measuredBaseYaw);
  const vector2_t offsetInBaseFrame(
      parameters_.forwardPerForwardVelocity * commandedInBaseFrame.x() + parameters_.forwardOffset,
      // The lateral constant is signed by the foot's own side, so one number in the task file widens the stance
      // instead of shifting the whole robot to the left. The velocity-proportional part is NOT signed: both feet
      // step the same way when the robot is asked to move sideways.
      parameters_.lateralPerLateralVelocity * commandedInBaseFrame.y() + context.side * parameters_.lateralOffset);
  return toWorld(offsetInBaseFrame, context.measuredBaseYaw);
}

std::string TranslationalSteppingHeuristic::describe() const {
  return absl::StrCat("translational_stepping: d_forward = ", parameters_.forwardPerForwardVelocity, " * v_x + ", parameters_.forwardOffset,
                      " [m], d_lateral = ", parameters_.lateralPerLateralVelocity, " * v_y + side * ", parameters_.lateralOffset, " [m]");
}

}  // namespace ocs2::humanoid
