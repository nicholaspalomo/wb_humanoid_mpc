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

#include "humanoid_common_mpc/locomotion_heuristics/foothold/HipCenteredSteppingHeuristic.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/locomotion_heuristics/FootholdHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

namespace ocs2::humanoid {

absl::Status HipCenteredSteppingHeuristic::configure(
    const LocomotionHeuristicConfig& config,
    const LocomotionHeuristicModelParameters& model) {
  parameters_ = config.hipCenteredStepping;
  posHipsRtBaseEwrtHeadingXy_ = model.posHipsRtBaseEwrtHeadingXy;
  return absl::OkStatus();
}

vector2_t HipCenteredSteppingHeuristic::computeOffsetPosFootRtWorldEwrtWorldXy(
    const FootholdHeuristicContext& context) const {
  // H_r(Theta) = PTP(R(Theta) r_hip): the hip's position relative to the base
  // in the heading frame,
  // rotated into the world by the base yaw predicted at the touch-down and
  // dropped onto the ground. R(Theta) is the full body rotation in the
  // dissertation and R_z(yaw) here - a choice of this heuristic, not a
  // simplification of the control model, which carries the full base rotation:
  // only the heading at the touch-down is predicted, and the roll and pitch of
  // a walking base move a ground-projected hip offset by a second-order amount.
  // The hip of the landing contact's leg. A leg the model parameters have no
  // hip for keeps the predicted base as its anchor rather than reading past
  // the end.
  if (context.legIndex >= posHipsRtBaseEwrtHeadingXy_.size()) {
    return vector2_t::Zero();
  }
  const vector2_t posHipRtBaseEwrtHeadingXy =
      posHipsRtBaseEwrtHeadingXy_[context.legIndex];
  const scalar_t scaledPosHipRtBaseEwrtHeadingX =
      parameters_.longitudinalScale * posHipRtBaseEwrtHeadingXy.x();
  const scalar_t scaledPosHipRtBaseEwrtHeadingY =
      parameters_.lateralScale * posHipRtBaseEwrtHeadingXy.y();
  const scalar_t cosYaw = std::cos(context.yawBaseRtWorld);
  const scalar_t sinYaw = std::sin(context.yawBaseRtWorld);
  const vector2_t scaledPosHipRtWorldEwrtWorldXy(
      context.posBaseRtWorldEwrtWorldXy.x() +
          cosYaw * scaledPosHipRtBaseEwrtHeadingX -
          sinYaw * scaledPosHipRtBaseEwrtHeadingY,
      context.posBaseRtWorldEwrtWorldXy.y() +
          sinYaw * scaledPosHipRtBaseEwrtHeadingX +
          cosYaw * scaledPosHipRtBaseEwrtHeadingY);
  // This heuristic MOVES THE ANCHOR:
  // SwitchedModelReferenceManager::getNomPosFootRtWorldEwrtWorldXy() adds the
  // offsets of a list that contains it to the predicted base rather than to the
  // step-width anchor. Everything here is expressed relative to that predicted
  // base, so what it has to return is exactly the hip position relative to it.
  return vector2_t(scaledPosHipRtWorldEwrtWorldXy -
                   context.posBaseRtWorldEwrtWorldXy);
}

std::string HipCenteredSteppingHeuristic::describe() const {
  std::vector<std::string> hips;
  for (size_t leg = 0; leg < posHipsRtBaseEwrtHeadingXy_.size(); ++leg) {
    hips.push_back(absl::StrCat("leg ", leg, " (",
                                posHipsRtBaseEwrtHeadingXy_[leg].x(), ", ",
                                posHipsRtBaseEwrtHeadingXy_[leg].y(), ") m"));
  }
  return absl::StrCat(
      "hip_centered_stepping: foot under the hip at (",
      parameters_.longitudinalScale, " * x_hip, ", parameters_.lateralScale,
      " * y_hip) of the measured base; hips ", absl::StrJoin(hips, ", "));
}

}  // namespace ocs2::humanoid
