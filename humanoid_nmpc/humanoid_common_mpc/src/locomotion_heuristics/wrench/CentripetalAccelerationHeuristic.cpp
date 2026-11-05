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

#include "humanoid_common_mpc/locomotion_heuristics/wrench/CentripetalAccelerationHeuristic.h"

#include <cstddef>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"
#include "humanoid_common_mpc/locomotion_heuristics/WrenchHeuristic.h"

namespace ocs2::humanoid {

absl::Status CentripetalAccelerationHeuristic::configure(
    const LocomotionHeuristicConfig& config,
    const LocomotionHeuristicModelParameters& model) {
  parameters_ = config.centripetalAccelerationParameters;
  totalMass_ = model.totalMass;
  totalWeight_ = model.totalWeight;
  return absl::OkStatus();
}

vector3_t CentripetalAccelerationHeuristic::computeOffsetForceFootEwrtWorld(
    const WrenchHeuristicContext& context, size_t /*contactIndex*/) const {
  // H_f(Theta x pdot) = m omega x pdot, Bledt Table C.1.
  //
  // With omega = psidot e_z and a horizontal velocity,
  //
  //     omega x v = (0, 0, psidot) x (v_x, v_y, 0) = (-psidot v_y, psidot v_x,
  //     0),
  //
  // which points towards the center of the turn and has magnitude m |psidot|
  // |v| - the centripetal force of a body of mass m on a circle of radius |v| /
  // |psidot|. The feet share it equally, matching how the existing reference
  // shares the weight.
  if (context.numStanceFeet == 0) return vector3_t::Zero();

  const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy =
      context.cmdLinvelBaseRtWorldEwrtWorldXy;
  const scalar_t cmdRateYawBaseRtWorld = context.cmdRateYawBaseRtWorld;
  vector3_t forceComEwrtWorld(
      -totalMass_ * cmdRateYawBaseRtWorld * cmdLinvelBaseRtWorldEwrtWorldXy.y(),
      totalMass_ * cmdRateYawBaseRtWorld * cmdLinvelBaseRtWorldEwrtWorldXy.x(),
      0.0);
  forceComEwrtWorld *= parameters_.scale;

  // A foot cannot pull sideways harder than friction allows, and a reference
  // outside the friction cone is one the wrench-cone barrier will fight at
  // every node. The clamp is on the TOTAL horizontal force, before it is
  // shared: the feet share the normal load W / n_stance as they share this
  // force, so each foot's friction budget is mu W / n_stance and a total clamp
  // of r W leaves each foot (r W / n_stance) / (mu W / n_stance) = r / mu of
  // its own budget in single and in double support alike. A clamp on each
  // foot's share would allow twice the fraction in double support that it
  // allows in single support. The default clamp is a fraction of body weight
  // rather than an absolute force so that it scales across the robots in this
  // repository, whose masses differ by a factor of five.
  const scalar_t limit = getMaxForceComEwrtWorldXy();
  const scalar_t magnitude = forceComEwrtWorld.norm();
  if (limit > 0.0 && magnitude > limit) forceComEwrtWorld *= limit / magnitude;
  return forceComEwrtWorld / static_cast<scalar_t>(context.numStanceFeet);
}

scalar_t CentripetalAccelerationHeuristic::getMaxForceComEwrtWorldXy() const {
  return parameters_.maxForceComEwrtWorldXy > 0.0
             ? parameters_.maxForceComEwrtWorldXy
             : parameters_.maxWnormForceComEwrtWorldXy * totalWeight_;
}

std::string CentripetalAccelerationHeuristic::describe() const {
  return absl::StrCat("centripetal_acceleration: f_xy = ", parameters_.scale,
                      " * ", totalMass_,
                      " kg * (omega x v) / n_stance [N], the total over the "
                      "stance feet clamped to ",
                      getMaxForceComEwrtWorldXy(), " N");
}

}  // namespace ocs2::humanoid
