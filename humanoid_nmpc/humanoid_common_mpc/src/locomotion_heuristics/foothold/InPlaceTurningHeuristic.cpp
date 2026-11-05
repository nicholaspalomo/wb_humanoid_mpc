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

#include "humanoid_common_mpc/locomotion_heuristics/foothold/InPlaceTurningHeuristic.h"

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

absl::Status InPlaceTurningHeuristic::configure(
    const LocomotionHeuristicConfig& config,
    const LocomotionHeuristicModelParameters& /*model*/) {
  parameters_ = config.inPlaceTurning;
  return absl::OkStatus();
}

vector2_t InPlaceTurningHeuristic::computeOffsetPosFootRtWorldEwrtWorldXy(
    const FootholdHeuristicContext& context) const {
  // H_r(psidot) = a1 psidot + a0, Bledt Table C.2, in the base's yaw frame.
  //
  // WHICH WAY EACH FOOT GOES. A foot at position r relative to the base moves
  // at omega x r, so with omega = psidot e_z the LEFT foot, which sits at r =
  // (0, +d, 0), moves at
  //
  //     (0, 0, psidot) x (0, d, 0) = (-psidot d, 0, 0),
  //
  // i.e. BACKWARDS for a positive (counter-clockwise) yaw rate, while the right
  // foot at (0, -d, 0) moves forwards. That is how a turn in place is actually
  // done: to turn left you swing the right foot forward and around it and the
  // left foot back. The fore-aft displacements of the two feet are therefore
  // equal and opposite, which is why this term is signed per foot and the
  // translational one is not.
  //
  // The sign below carries the MINUS so that a POSITIVE forwardPerYawRate means
  // "place each foot further along the direction its own hip is traveling",
  // i.e. lead the hips into the turn - which is what the heuristic is for
  // (figure 4-10: without it the feet trail the hips until they run out of
  // workspace). Leaving the minus out would make a positive coefficient drive
  // the feet to trail the hips by twice the intended amount, which is the very
  // failure being fixed.
  //
  // The lateral RATE term is not signed per foot: it moves the whole stance
  // sideways, which is what a turn with a radius does. The lateral CONSTANT is
  // signed, and that one opens or closes the stance.
  //
  // As in translational_stepping, a1 is Bledt's constant plus a term
  // proportional to the stance about to begin: the hip travels its lever arm
  // times psidot, and Raibert's lead is half a stance of that travel.
  const scalar_t forwardGain =
      parameters_.forwardPerYawRate +
      parameters_.forwardStanceLever * context.stanceDuration;
  const vector2_t offsetPosFootRtWorldEwrtHeadingXy(
      -context.side * forwardGain * context.cmdRateYawBaseRtWorld +
          parameters_.offsetPosFootRtWorldEwrtHeadingX,
      parameters_.lateralPerYawRate * context.cmdRateYawBaseRtWorld +
          context.side * parameters_.offsetPosFootRtWorldEwrtHeadingY);
  return rotateAboutVertical(offsetPosFootRtWorldEwrtHeadingXy,
                             context.yawBaseRtWorld);
}

std::string InPlaceTurningHeuristic::describe() const {
  return absl::StrCat(
      "in_place_turning: d_forward = -side * (", parameters_.forwardPerYawRate,
      " + ", parameters_.forwardStanceLever, " * T_stance) * psidot + ",
      parameters_.offsetPosFootRtWorldEwrtHeadingX,
      " [m], d_lateral = ", parameters_.lateralPerYawRate,
      " * psidot + side * ", parameters_.offsetPosFootRtWorldEwrtHeadingY,
      " [m]");
}

}  // namespace ocs2::humanoid
