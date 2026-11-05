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
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/locomotion_heuristics/FootholdHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

namespace ocs2::humanoid {

namespace {

/** Rotates `planarVector` by `angle` [rad] about the vertical: the components
 * in the heading frame into the world's for angle = yaw_base_rt_world, and back
 * for -yaw_base_rt_world. */
vector2_t rotateAboutVertical(const vector2_t& planarVector, scalar_t angle) {
  const scalar_t cosAngle = std::cos(angle);
  const scalar_t sinAngle = std::sin(angle);
  return vector2_t(cosAngle * planarVector.x() - sinAngle * planarVector.y(),
                   sinAngle * planarVector.x() + cosAngle * planarVector.y());
}

}  // namespace

absl::Status HighSpeedTurningHeuristic::configure(
    const LocomotionHeuristicConfig& config,
    const LocomotionHeuristicModelParameters& /*model*/) {
  parameters_ = config.highSpeedTurning;
  return absl::OkStatus();
}

vector2_t HighSpeedTurningHeuristic::computeOffsetPosFootRtWorldEwrtWorldXy(
    const FootholdHeuristicContext& context) const {
  // H_r(pdot x omega) = a1 (pdot x omega) + a0, Bledt Table C.2.
  //
  // The full cross product of a 3D velocity and a 3D angular rate has three
  // components, but Bledt's own reduction (equation 4.27) drops the vertical
  // velocity and the roll and pitch rates as negligible during regular
  // locomotion, which leaves omega = psidot e_z and a horizontal velocity. Then
  //
  //     v x omega = (v_x, v_y, 0) x (0, 0, psidot) = (v_y psidot, -v_x psidot,
  //     0).
  //
  // Evaluated in the heading frame, so that "forward" and "lateral" below mean
  // the robot's own, and the whole offset is rotated into the world afterwards.
  // This is the term that vanishes at zero speed however fast the robot spins,
  // which is what distinguishes it from in_place_turning.
  const vector2_t cmdLinvelBaseRtWorldEwrtHeadingXy = rotateAboutVertical(
      context.cmdLinvelBaseRtWorldEwrtWorldXy, -context.yawBaseRtWorld);
  const scalar_t crossForward =
      cmdLinvelBaseRtWorldEwrtHeadingXy.y() * context.cmdRateYawBaseRtWorld;
  const scalar_t crossLateral =
      -cmdLinvelBaseRtWorldEwrtHeadingXy.x() * context.cmdRateYawBaseRtWorld;
  const vector2_t offsetPosFootRtWorldEwrtHeadingXy(
      parameters_.forwardPerCrossTerm * crossForward +
          parameters_.offsetPosFootRtWorldEwrtHeadingX,
      parameters_.lateralPerCrossTerm * crossLateral +
          context.side * parameters_.offsetPosFootRtWorldEwrtHeadingY);
  return rotateAboutVertical(offsetPosFootRtWorldEwrtHeadingXy,
                             context.yawBaseRtWorld);
}

std::string HighSpeedTurningHeuristic::describe() const {
  return absl::StrCat(
      "high_speed_turning: d_forward = ", parameters_.forwardPerCrossTerm,
      " * (v_y * psidot) + ", parameters_.offsetPosFootRtWorldEwrtHeadingX,
      " [m], d_lateral = ", parameters_.lateralPerCrossTerm,
      " * (-v_x * psidot) + side * ",
      parameters_.offsetPosFootRtWorldEwrtHeadingY, " [m]");
}

}  // namespace ocs2::humanoid
