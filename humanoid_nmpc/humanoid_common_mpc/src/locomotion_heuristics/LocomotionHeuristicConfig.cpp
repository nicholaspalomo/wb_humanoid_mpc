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

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

namespace {

/** Rejects a parameter that is outside the range its formula is defined on. */
absl::Status requirePositive(scalar_t value, absl::string_view key) {
  if (value <= 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[LocomotionHeuristicConfig] locomotion_heuristics.", key, " must be positive, got ", value, "."));
  }
  return absl::OkStatus();
}

absl::Status requireNonNegative(scalar_t value, absl::string_view key) {
  if (value < 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[LocomotionHeuristicConfig] locomotion_heuristics.", key, " must not be negative, got ", value, "."));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status LocomotionHeuristicConfig::validate() const {
  RETURN_IF_ERROR(formulation.validate());

  // Only the keys a wrong value would make MEANINGLESS are checked, and only against the range their own formula
  // needs. Every coefficient of Table C.2 is deliberately absent from this list: a fitted number has no admissible
  // range at all and is as legitimately negative as positive, which is the whole point of fitting it rather than
  // designing it.
  RETURN_IF_ERROR(requirePositive(orientationCompensation.maximumTilt, "orientation_compensation.maximum_tilt"));
  RETURN_IF_ERROR(requireNonNegative(heightCompensation.maximumHeightOffset, "height_compensation.maximum_height_offset"));
  RETURN_IF_ERROR(requireNonNegative(capturePoint.comHeightOverride, "capture_point.com_height_override"));
  RETURN_IF_ERROR(requirePositive(capturePoint.gravity, "capture_point.gravity"));
  // Positive, not merely non-negative: this is the clamp that stops a bad velocity estimate throwing the landing
  // target out of the leg's reach, and a value of zero would remove it rather than tighten it.
  RETURN_IF_ERROR(requirePositive(capturePoint.maximumOffset, "capture_point.maximum_offset"));
  // A negative scale or gain INVERTS the heuristic rather than reducing it: the hip landmark mirrored through the base,
  // or the foot stepped against the velocity error, which accelerates the fall the capture point exists to stop.
  RETURN_IF_ERROR(requireNonNegative(hipCenteredStepping.lateralScale, "hip_centered_stepping.lateral_scale"));
  RETURN_IF_ERROR(requireNonNegative(hipCenteredStepping.longitudinalScale, "hip_centered_stepping.longitudinal_scale"));
  RETURN_IF_ERROR(requireNonNegative(capturePoint.gain, "capture_point.gain"));
  // A negative blend would invert the heuristic rather than reduce it: the reference would move AWAY from Bledt's
  // value as the knob is turned down. Both scales are therefore floors at zero.
  RETURN_IF_ERROR(requireNonNegative(impulseScaling.scale, "impulse_scaling.scale"));
  RETURN_IF_ERROR(requireNonNegative(centripetalAcceleration.scale, "centripetal_acceleration.scale"));
  RETURN_IF_ERROR(requireNonNegative(centripetalAcceleration.maximumForce, "centripetal_acceleration.maximum_force"));
  RETURN_IF_ERROR(
      requireNonNegative(centripetalAcceleration.maximumForceRatioOfWeight, "centripetal_acceleration.maximum_force_ratio_of_weight"));

  // beta is a fraction of a gait cycle, so a clamp outside (0, 1] cannot be reached, and one at 0 lets 1/beta run
  // away at the onset of a flight phase - which is the failure this clamp exists to prevent.
  if (impulseScaling.minimumDutyFactor <= 0.0 || impulseScaling.minimumDutyFactor > 1.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[LocomotionHeuristicConfig] locomotion_heuristics.impulse_scaling.minimum_duty_factor must lie in (0, 1], got ",
                     impulseScaling.minimumDutyFactor,
                     ". It is the floor the stance duty factor is clamped to before W / (F beta) is formed, and that is unbounded "
                     "as a flight phase opens."));
  }
  if (impulseScaling.maximumForceRatio < 1.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[LocomotionHeuristicConfig] locomotion_heuristics.impulse_scaling.maximum_force_ratio must be at least 1, got ",
                     impulseScaling.maximumForceRatio,
                     ". It clamps the scaled reference as a multiple of weight compensation, and a value below 1 would ask the "
                     "stance feet to carry less than the robot weighs even while standing."));
  }
  // `maximum_force: 0` is the documented way to say "use the ratio instead", so exactly one of the two has to be
  // positive; both at zero would remove the clamp entirely rather than select a default.
  if (centripetalAcceleration.maximumForce <= 0.0 && centripetalAcceleration.maximumForceRatioOfWeight <= 0.0) {
    return absl::InvalidArgumentError(
        "[LocomotionHeuristicConfig] locomotion_heuristics.centripetal_acceleration needs a force clamp: set "
        "maximum_force to a positive value in newtons, or leave it at 0 and set maximum_force_ratio_of_weight to a "
        "positive fraction of body weight. With both at zero nothing bounds a horizontal force reference that no "
        "friction cone would admit.");
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid
