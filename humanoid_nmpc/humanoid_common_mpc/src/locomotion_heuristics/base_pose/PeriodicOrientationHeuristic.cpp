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

#include "humanoid_common_mpc/locomotion_heuristics/base_pose/PeriodicOrientationHeuristic.h"

#include <cmath>

#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

namespace ocs2::humanoid {

absl::Status PeriodicOrientationHeuristic::configure(const LocomotionHeuristicConfig& config,
                                                     const LocomotionHeuristicModelParameters& /*model*/) {
  parameters_ = config.periodicOrientation;
  return absl::OkStatus();
}

BasePoseOffset PeriodicOrientationHeuristic::offset(const BasePoseHeuristicContext& context) const {
  BasePoseOffset result;
  // H_Theta(Phi) = b1 sin(c1 Phi + d1), Bledt equation 4.13. Phi is the gait phase in [0, 1), so c1 = 2 pi is one
  // period per left-right cycle and c1 = 4 pi is one per step. No clamp: a sinusoid is already bounded by its own
  // amplitude, and that amplitude is the coefficient the operator sets.
  result.roll = parameters_.rollAmplitude * std::sin(parameters_.rollPhaseRate * context.gaitPhase + parameters_.rollPhaseOffset);
  result.pitch = parameters_.pitchAmplitude * std::sin(parameters_.pitchPhaseRate * context.gaitPhase + parameters_.pitchPhaseOffset);
  return result;
}

std::string PeriodicOrientationHeuristic::describe() const {
  return absl::StrCat("periodic_orientation: roll = ", parameters_.rollAmplitude, " * sin(", parameters_.rollPhaseRate, " * phi + ",
                      parameters_.rollPhaseOffset, ") [rad], pitch = ", parameters_.pitchAmplitude, " * sin(", parameters_.pitchPhaseRate,
                      " * phi + ", parameters_.pitchPhaseOffset, ") [rad]");
}

}  // namespace ocs2::humanoid
