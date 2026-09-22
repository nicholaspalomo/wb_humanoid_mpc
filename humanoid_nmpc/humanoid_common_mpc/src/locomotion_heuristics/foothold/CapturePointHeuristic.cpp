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

#include "humanoid_common_mpc/locomotion_heuristics/foothold/CapturePointHeuristic.h"

#include <cmath>

#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

namespace ocs2::humanoid {

absl::Status CapturePointHeuristic::configure(const LocomotionHeuristicConfig& config, const LocomotionHeuristicModelParameters& model) {
  parameters_ = config.capturePoint;
  nominalComHeight_ = model.nominalComHeight;
  return absl::OkStatus();
}

vector2_t CapturePointHeuristic::offset(const FootholdHeuristicContext& context) const {
  // H_r(pdot, Phi) = gain sqrt(p_z / g) (pdot - pdot_d), Bledt Table C.1. sqrt(p_z/g) is 1/omega, the time constant
  // of a linear inverted pendulum of height p_z, so the offset is "how far the centre of mass will have carried past
  // the foot by the time the pendulum has done its work" - the capture point of Pratt et al. applied to the velocity
  // ERROR rather than to the velocity, which is what makes it silent while the robot is tracking its command.
  //
  // The height is preferred from the measurement and falls back to the nominal: a robot recovering from a push is
  // exactly when the measurement is worth having, and exactly when it can also be nonsense, so a non-positive
  // measurement is replaced rather than trusted.
  scalar_t comHeight = parameters_.comHeightOverride > 0.0 ? parameters_.comHeightOverride : context.comHeight;
  if (!(comHeight > 0.0)) comHeight = nominalComHeight_;
  if (!(comHeight > 0.0)) return vector2_t::Zero();

  const scalar_t timeConstant = std::sqrt(comHeight / parameters_.gravity);
  const vector2_t velocityError = context.measuredVelocity - context.commandedVelocity;
  vector2_t result = parameters_.gain * timeConstant * velocityError;
  // Clamped in MAGNITUDE rather than per axis, so that the direction of the step into the error is preserved. A
  // per-axis clamp would rotate the offset towards the diagonal exactly when it is largest, which is when the robot
  // most needs the foot to go where the push came from.
  const scalar_t magnitude = result.norm();
  if (parameters_.maximumOffset > 0.0 && magnitude > parameters_.maximumOffset) {
    result *= parameters_.maximumOffset / magnitude;
  }
  return result;
}

std::string CapturePointHeuristic::describe() const {
  return absl::StrCat("capture_point: dr = ", parameters_.gain, " * sqrt(z/", parameters_.gravity,
                      ") * (v_measured - v_commanded) [m], z = ",
                      parameters_.comHeightOverride > 0.0 ? absl::StrCat(parameters_.comHeightOverride, " m (fixed)")
                                                          : absl::StrCat("measured, nominally ", nominalComHeight_, " m"),
                      ", clamped to ", parameters_.maximumOffset, " m");
}

}  // namespace ocs2::humanoid
