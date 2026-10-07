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
#include <string>

#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

namespace ocs2::humanoid {

absl::Status HipCenteredSteppingHeuristic::configure(const LocomotionHeuristicConfig& config,
                                                     const LocomotionHeuristicModelParameters& model) {
  parameters_ = config.hipCenteredStepping;
  hipPositionInBaseFrame_ = model.hipPositionInBaseFrame;
  return absl::OkStatus();
}

vector2_t HipCenteredSteppingHeuristic::offset(const FootholdHeuristicContext& context) const {
  // H_r(Theta) = PTP(R(Theta) r_hip): the hip's position in the base frame, rotated into the world by the base yaw
  // predicted at the touch-down and dropped onto the ground. R(Theta) is the full body rotation in the dissertation and R_z(yaw) here,
  // which is the same simplification the control model itself makes for its orientation dynamics (section 3.2.1):
  // the roll and pitch of a walking base are small, and PTP() would flatten their contribution anyway.
  const vector2_t hip = hipPositionInBaseFrame_[context.contactIndex];
  const scalar_t longitudinal = parameters_.longitudinalScale * hip.x();
  const scalar_t lateral = parameters_.lateralScale * hip.y();
  const scalar_t cosYaw = std::cos(context.baseYaw);
  const scalar_t sinYaw = std::sin(context.baseYaw);
  const vector2_t hipInWorld(context.basePosition.x() + cosYaw * longitudinal - sinYaw * lateral,
                             context.basePosition.y() + sinYaw * longitudinal + cosYaw * lateral);
  // This heuristic MOVES THE ANCHOR: SwitchedModelReferenceManager::nominalFoothold() adds the offsets of a list that
  // contains it to the predicted base rather than to the stance-foot anchor. Everything here is expressed relative to
  // that predicted base, so what it has to return is exactly the hip position relative to it.
  return vector2_t(hipInWorld - context.basePosition);
}

std::string HipCenteredSteppingHeuristic::describe() const {
  return absl::StrCat("hip_centered_stepping: foot under the hip at (", parameters_.longitudinalScale, " * x_hip, ",
                      parameters_.lateralScale, " * y_hip) of the measured base; left hip = (",
                      hipPositionInBaseFrame_[kContactLeftIndex].x(), ", ", hipPositionInBaseFrame_[kContactLeftIndex].y(), ") m");
}

}  // namespace ocs2::humanoid
