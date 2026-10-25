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

#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"

#include <array>
#include <cmath>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigValueChecks.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/swing_trajectory_config.nproto.h"

namespace ocs2::humanoid {
namespace {

using SwingConfig = mpc_config::SwingTrajectoryConfig;
using SwingSettings = SwingTrajectoryPlanner::Config;

/** A field of the swing_trajectory_config block and the planner setting it
 * becomes; `name` is a string literal. */
struct SwingField {
  const char* absl_nonnull name;
  double SwingConfig::*absl_nonnull field;
  scalar_t SwingSettings::*absl_nonnull setting;
};

// Every field of swing_trajectory_config.proto, in its order, with the planner
// setting it becomes.
constexpr std::array<SwingField, 11> kSwingFields = {{
    {.name = "lift_off_velocity",
     .field = &SwingConfig::lift_off_velocity,
     .setting = &SwingSettings::liftOffVelocity},
    {.name = "touch_down_velocity",
     .field = &SwingConfig::touch_down_velocity,
     .setting = &SwingSettings::touchDownVelocity},
    {.name = "swing_height",
     .field = &SwingConfig::swing_height,
     .setting = &SwingSettings::swingHeight},
    {.name = "swing_time_scale",
     .field = &SwingConfig::swing_time_scale,
     .setting = &SwingSettings::swingTimeScale},
    {.name = "touch_down_height_offset",
     .field = &SwingConfig::touch_down_height_offset,
     .setting = &SwingSettings::touchDownHeightOffset},
    {.name = "impact_proximity_factor_lift_off_velocity",
     .field = &SwingConfig::impact_proximity_factor_lift_off_velocity,
     .setting = &SwingSettings::impactProximityFactorLiftOffVelocity},
    {.name = "impact_proximity_factor_touch_down_velocity",
     .field = &SwingConfig::impact_proximity_factor_touch_down_velocity,
     .setting = &SwingSettings::impactProximityFactorTouchDownVelocity},
    {.name = "impact_proximity_factor_mid_point_value",
     .field = &SwingConfig::impact_proximity_factor_mid_point_value,
     .setting = &SwingSettings::impactProximityFactorMidPointValue},
    {.name = "swing_pitch_angle",
     .field = &SwingConfig::swing_pitch_angle,
     .setting = &SwingSettings::swingPitchAngle},
    {.name = "swing_pitch_rise_fraction",
     .field = &SwingConfig::swing_pitch_rise_fraction,
     .setting = &SwingSettings::swingPitchRiseFraction},
    {.name = "swing_pitch_fall_fraction",
     .field = &SwingConfig::swing_pitch_fall_fraction,
     .setting = &SwingSettings::swingPitchFallFraction},
}};

/**
 * OK when the finite fields of `config` lie in the ranges the schema documents:
 * swing_time_scale > 0 (the planner divides a swing's duration by it, and a
 * negative one flips the scaling of the swing height), swing_height >= 0 (a
 * negative apex sends the foot below the ground), the lift-off slope of the
 * impact-proximity factor <= 0 and its touch-down slope >= 0, its mid-swing
 * value and the pitch ramp fractions in [0, 1].
 */
absl::Status checkSwingRanges(const SwingConfig& config) {
  constexpr absl::string_view kBlock = "swing_trajectory_config.";
  RETURN_IF_ERROR(requirePositive(absl::StrCat(kBlock, "swing_time_scale"),
                                  config.swing_time_scale));
  RETURN_IF_ERROR(requireNonNegative(absl::StrCat(kBlock, "swing_height"),
                                     config.swing_height));
  if (!(config.impact_proximity_factor_lift_off_velocity <= 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat(kBlock, "impact_proximity_factor_lift_off_velocity: ",
                     config.impact_proximity_factor_lift_off_velocity,
                     " is not 0 or negative"));
  }
  RETURN_IF_ERROR(requireNonNegative(
      absl::StrCat(kBlock, "impact_proximity_factor_touch_down_velocity"),
      config.impact_proximity_factor_touch_down_velocity));
  for (const SwingField& entry : kSwingFields) {
    const absl::string_view name = entry.name;
    if (name != "impact_proximity_factor_mid_point_value" &&
        name != "swing_pitch_rise_fraction" &&
        name != "swing_pitch_fall_fraction") {
      continue;
    }
    const double value = config.*entry.field;
    if (!(value >= 0.0 && value <= 1.0)) {
      return absl::InvalidArgumentError(
          absl::StrCat(kBlock, name, ": ", value, " is not in [0, 1]"));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<SwingTrajectoryPlanner::Config>
swingTrajectorySettingsFromConfig(
    const mpc_config::SwingTrajectoryConfig& config) {
  SwingTrajectoryPlanner::Config settings;
  for (const SwingField& entry : kSwingFields) {
    const double value = config.*entry.field;
    if (!std::isfinite(value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("swing_trajectory_config.", entry.name, " is ", value,
                       ", but every value of the block must be finite."));
    }
    settings.*entry.setting = value;
  }
  RETURN_IF_ERROR(checkSwingRanges(config));
  return settings;
}

}  // namespace ocs2::humanoid
