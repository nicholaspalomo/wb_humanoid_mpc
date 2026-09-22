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

#include "humanoid_common_mpc/locomotion_heuristics/base_pose/HeightCompensationHeuristic.h"

#include <algorithm>

#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

namespace ocs2::humanoid {

absl::Status HeightCompensationHeuristic::configure(const LocomotionHeuristicConfig& config,
                                                    const LocomotionHeuristicModelParameters& /*model*/) {
  parameters_ = config.heightCompensation;
  return absl::OkStatus();
}

BasePoseOffset HeightCompensationHeuristic::offset(const BasePoseHeuristicContext& context) const {
  BasePoseOffset result;
  // H_z(v) = a2 v^2 + a1 v + a0 in the MAGNITUDE of the commanded horizontal velocity. The magnitude rather than the
  // signed forward component, because crouching to walk forwards and rising to walk backwards is not a thing any
  // legged system does, and an odd polynomial in a signed speed would do exactly that.
  const scalar_t speed = context.commandedVelocityInBaseFrame.norm();
  const scalar_t height = parameters_.heightPerSpeedSquared * speed * speed + parameters_.heightPerSpeed * speed + parameters_.heightOffset;
  // Clamped for the same reason as the orientation tilt, and with a sharper consequence: this channel is also where
  // adaptToCurrentGroundHeight() writes the terrain, and a runaway offset added on top of it would look like the
  // ground moving rather than like a bad coefficient.
  result.height = std::clamp(height, -parameters_.maximumHeightOffset, parameters_.maximumHeightOffset);
  return result;
}

std::string HeightCompensationHeuristic::describe() const {
  return absl::StrCat("height_compensation: dz = ", parameters_.heightPerSpeedSquared, " * |v|^2 + ", parameters_.heightPerSpeed,
                      " * |v| + ", parameters_.heightOffset, " [m], clamped to +/- ", parameters_.maximumHeightOffset, " m");
}

}  // namespace ocs2::humanoid
