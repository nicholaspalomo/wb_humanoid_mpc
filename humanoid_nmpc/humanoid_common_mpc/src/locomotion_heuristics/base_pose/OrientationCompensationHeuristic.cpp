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

#include "humanoid_common_mpc/locomotion_heuristics/base_pose/OrientationCompensationHeuristic.h"

#include <algorithm>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/locomotion_heuristics/BasePoseHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

namespace ocs2::humanoid {

absl::Status OrientationCompensationHeuristic::configure(
    const LocomotionHeuristicConfig& config,
    const LocomotionHeuristicModelParameters& /*model*/) {
  parameters_ = config.orientationCompensationParameters;
  return absl::OkStatus();
}

BasePoseOffset OrientationCompensationHeuristic::offset(
    const BasePoseHeuristicContext& context) const {
  BasePoseOffset result;
  // H_Theta(pdot) = a1 pdot + a0, evaluated on the two components separately:
  // roll against the LATERAL command and pitch against the FORWARD one, which
  // is the pairing Bledt's regressions found (equations 4.11 and 4.12). The
  // velocity is already in the base's own yaw frame, so these are the robot's
  // forward and left, not the world's.
  result.offsetEulerzyxBaseRtWorldRoll =
      parameters_.rollPerLateralVelocityGain *
          context.cmdLinvelBaseRtWorldEwrtHeadingXy.y() +
      parameters_.offsetEulerzyxBaseRtWorldRoll;
  result.offsetEulerzyxBaseRtWorldPitch =
      parameters_.pitchPerForwardVelocityGain *
          context.cmdLinvelBaseRtWorldEwrtHeadingXy.x() +
      parameters_.offsetEulerzyxBaseRtWorldPitch;
  // The clamp is not defensive trimming: the command this reads is the
  // operator's, filtered but not rate limited, and an affine law applied to it
  // has no bound of its own. A reference tilt the legs cannot reach is worse
  // than no reference at all, because the state cost pulls towards it at every
  // node and the solver spends its iterations on a target it cannot have.
  result.offsetEulerzyxBaseRtWorldRoll =
      std::clamp(result.offsetEulerzyxBaseRtWorldRoll,
                 -parameters_.maxOffsetEulerzyxBaseRtWorld,
                 parameters_.maxOffsetEulerzyxBaseRtWorld);
  result.offsetEulerzyxBaseRtWorldPitch =
      std::clamp(result.offsetEulerzyxBaseRtWorldPitch,
                 -parameters_.maxOffsetEulerzyxBaseRtWorld,
                 parameters_.maxOffsetEulerzyxBaseRtWorld);
  return result;
}

std::string OrientationCompensationHeuristic::describe() const {
  return absl::StrCat(
      "orientation_compensation: roll = ",
      parameters_.rollPerLateralVelocityGain, " * v_y + ",
      parameters_.offsetEulerzyxBaseRtWorldRoll,
      " [rad], pitch = ", parameters_.pitchPerForwardVelocityGain, " * v_x + ",
      parameters_.offsetEulerzyxBaseRtWorldPitch,
      " [rad], both clamped to +/- ", parameters_.maxOffsetEulerzyxBaseRtWorld,
      " rad");
}

}  // namespace ocs2::humanoid
