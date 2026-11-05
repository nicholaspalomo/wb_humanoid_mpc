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

absl::Status TranslationalSteppingHeuristic::configure(
    const LocomotionHeuristicConfig& config,
    const LocomotionHeuristicModelParameters& /*model*/) {
  parameters_ = config.translationalStepping;
  return absl::OkStatus();
}

vector2_t
TranslationalSteppingHeuristic::computeOffsetPosFootRtWorldEwrtWorldXy(
    const FootholdHeuristicContext& context) const {
  // H_r(pdot) = a1 pdot + a0, Bledt Table C.2, in the base's yaw frame. The
  // commanded velocity is in the world, so it is rotated into the heading
  // frame, the affine law is applied there, and the result is rotated back -
  // which is not the same as applying the law in the world, because the forward
  // and lateral coefficients differ and the robot's forward is not the world's
  // x.
  //
  // a1 is Bledt's constant plus a term proportional to the stance about to
  // begin. His a1 was regressed at one gait; this robot's gait scheduler
  // changes the stance duration with the commanded speed, and Raibert's rule -
  // the foot half a stance of travel ahead of the hip it lands under - is a1 =
  // T_stance / 2. With the stance fraction at 0.5 and the constant at 0 the
  // lead is right on every gait the scheduler moves between; with the fraction
  // at 0 it is Bledt's.
  const vector2_t cmdLinvelBaseRtWorldEwrtHeadingXy = rotateAboutVertical(
      context.cmdLinvelBaseRtWorldEwrtWorldXy, -context.yawBaseRtWorld);
  const scalar_t forwardGain =
      parameters_.forwardPerForwardVelocityGain +
      parameters_.forwardStanceFraction * context.stanceDuration;
  const scalar_t lateralGain =
      parameters_.lateralPerLateralVelocityGain +
      parameters_.lateralStanceFraction * context.stanceDuration;
  const vector2_t offsetPosFootRtWorldEwrtHeadingXy(
      forwardGain * cmdLinvelBaseRtWorldEwrtHeadingXy.x() +
          parameters_.offsetPosFootRtWorldEwrtHeadingX,
      // The lateral constant is signed by the foot's own side, so one number in
      // the task file widens the stance instead of shifting the whole robot to
      // the left. The velocity-proportional part is NOT signed: both feet step
      // the same way when the robot is asked to move sideways.
      lateralGain * cmdLinvelBaseRtWorldEwrtHeadingXy.y() +
          context.side * parameters_.offsetPosFootRtWorldEwrtHeadingY);
  return rotateAboutVertical(offsetPosFootRtWorldEwrtHeadingXy,
                             context.yawBaseRtWorld);
}

std::string TranslationalSteppingHeuristic::describe() const {
  return absl::StrCat(
      "translational_stepping: d_forward = (",
      parameters_.forwardPerForwardVelocityGain, " + ",
      parameters_.forwardStanceFraction, " * T_stance) * v_x + ",
      parameters_.offsetPosFootRtWorldEwrtHeadingX, " [m], d_lateral = (",
      parameters_.lateralPerLateralVelocityGain, " + ",
      parameters_.lateralStanceFraction, " * T_stance) * v_y + side * ",
      parameters_.offsetPosFootRtWorldEwrtHeadingY, " [m]");
}

}  // namespace ocs2::humanoid
