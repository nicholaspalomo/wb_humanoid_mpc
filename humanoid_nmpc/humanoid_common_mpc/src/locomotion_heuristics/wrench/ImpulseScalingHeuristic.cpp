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

#include "humanoid_common_mpc/locomotion_heuristics/wrench/ImpulseScalingHeuristic.h"

#include <algorithm>

#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

namespace ocs2::humanoid {

absl::Status ImpulseScalingHeuristic::configure(const LocomotionHeuristicConfig& config,
                                                const LocomotionHeuristicModelParameters& /*model*/) {
  parameters_ = config.impulseScaling;
  return absl::OkStatus();
}

vector3_t ImpulseScalingHeuristic::forceOffset(const WrenchHeuristicContext& context, size_t contactIndex) const {
  // H_f(Phi) = m g / (F beta), Bledt Table C.1, expressed as the OFFSET from the reference that already exists.
  //
  // F is the number of FEET THE ROBOT HAS - a constant - and not the number in contact at this instant. That
  // distinction is the whole content of the heuristic and it is easy to get wrong. The existing reference is
  //
  //     f_z,now = W / n_stance(t),
  //
  // which always sums to exactly W across the stance feet, at every instant. Bledt's reference is
  //
  //     f_z,ref = W / (F beta),
  //
  // the same value on every stance foot whatever n_stance(t) happens to be, so its total varies over the cycle -
  // below W in single support, above it in double support - and averages, over one cycle,
  //
  //     <n_stance> * W / (F beta) = (F beta) * W / (F beta) = W,
  //
  // because the mean number of feet on the ground IS F times the duty factor. THAT is the impulse budget the
  // heuristic exists to satisfy. Scaling W / n_stance(t) by 1/beta instead would put W/beta on the ground at every
  // instant and deliver W T / beta over the cycle, i.e. it would regularise the solver towards accelerating the
  // centre of mass upwards for ever.
  //
  // Two sanity checks fall straight out. Standing (beta = 1, n_stance = F) gives W/F on each foot, which is the
  // existing reference, so the offset is zero. A pure alternating single support (beta = 1/2, n_stance = 1) gives
  // W on the one stance foot, which is again the existing reference. The correction therefore bites exactly where
  // the two disagree - a gait with double support - and there it asks the feet to push harder than static
  // equilibrium because a single-support phase is coming in which one of them will carry everything.
  if (context.numStanceFeet == 0) return vector3_t::Zero();

  const scalar_t numFeet = static_cast<scalar_t>(context.contactFlags.size());
  const scalar_t dutyFactor = std::max(context.stanceDutyFactor[contactIndex], parameters_.minimumDutyFactor);
  const scalar_t baseline = context.totalWeight / static_cast<scalar_t>(context.numStanceFeet);
  const scalar_t target = context.totalWeight / (numFeet * dutyFactor);

  // `scale` blends between the two rather than switching, so that it can be swept from 0 without a discontinuity.
  scalar_t reference = baseline + parameters_.scale * (target - baseline);
  // Clamped ABOVE by what the legs and the friction cone can deliver, and BELOW at zero because a contact force
  // reference that pulls the robot down through the floor is not a reference any foot can track. The lower clamp is
  // what makes a negative `scale` - which validate() also rejects - harmless rather than inverted.
  reference = std::clamp(reference, 0.0, parameters_.maximumForceRatio * baseline);
  return vector3_t(0.0, 0.0, reference - baseline);
}

std::string ImpulseScalingHeuristic::describe() const {
  return absl::StrCat("impulse_scaling: f_z *= ", parameters_.scale, " / beta + ", 1.0 - parameters_.scale,
                      ", beta >= ", parameters_.minimumDutyFactor, ", ratio clamped to ", parameters_.maximumForceRatio);
}

}  // namespace ocs2::humanoid
