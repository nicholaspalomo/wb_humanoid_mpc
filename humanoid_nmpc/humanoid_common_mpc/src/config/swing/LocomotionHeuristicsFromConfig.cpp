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

#include "humanoid_common_mpc/config/swing/LocomotionHeuristicsFromConfig.h"

#include <array>
#include <cmath>
#include <cstddef>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"
#include "humanoid_mpc_config/locomotion_heuristics_config.nproto.h"

namespace ocs2::humanoid {
namespace {

using HeuristicsConfig = mpc_config::LocomotionHeuristicsConfig;

/**
 * A field of a heuristic's parameter block and the member of the heuristic's parameter struct it becomes; `name` is a
 * string literal.
 */
template <typename Block, typename Parameters>
struct Coefficient {
  const char* absl_nonnull name;
  double Block::*absl_nonnull field;
  scalar_t Parameters::*absl_nonnull parameter;
};

/** The parameters of the heuristic `heuristic` from its block, field by field; InvalidArgument for a value that is not finite. */
template <typename Block, typename Parameters, size_t kCount>
absl::StatusOr<Parameters> convertBlock(absl::string_view heuristic,
                                        const Block& block,
                                        const std::array<Coefficient<Block, Parameters>, kCount>& coefficients) {
  Parameters parameters;
  for (const Coefficient<Block, Parameters>& coefficient : coefficients) {
    const double value = block.*coefficient.field;
    if (!std::isfinite(value)) {
      return absl::InvalidArgumentError(absl::StrCat("locomotion_heuristics.", heuristic, ".", coefficient.name, " is ", value,
                                                     ", but every parameter of the block must be finite."));
    }
    parameters.*coefficient.parameter = value;
  }
  return parameters;
}

// Every field of the parameter blocks of locomotion_heuristics_config.proto, in its order, with the member it becomes.
using OrientationCompensationBlock = HeuristicsConfig::OrientationCompensation;
using PeriodicOrientationBlock = HeuristicsConfig::PeriodicOrientation;
using HeightCompensationBlock = HeuristicsConfig::HeightCompensation;
using HipCenteredSteppingBlock = HeuristicsConfig::HipCenteredStepping;
using CapturePointBlock = HeuristicsConfig::CapturePoint;
using TranslationalSteppingBlock = HeuristicsConfig::TranslationalStepping;
using InPlaceTurningBlock = HeuristicsConfig::InPlaceTurning;
using HighSpeedTurningBlock = HeuristicsConfig::HighSpeedTurning;
using ImpulseScalingBlock = HeuristicsConfig::ImpulseScaling;
using CentripetalAccelerationBlock = HeuristicsConfig::CentripetalAcceleration;

constexpr std::array<Coefficient<OrientationCompensationBlock, OrientationCompensationParameters>, 5> kOrientationCompensationCoefficients =
    {{
        {.name = "roll_per_lateral_velocity",
         .field = &OrientationCompensationBlock::roll_per_lateral_velocity,
         .parameter = &OrientationCompensationParameters::rollPerLateralVelocity},
        {.name = "roll_offset",
         .field = &OrientationCompensationBlock::roll_offset,
         .parameter = &OrientationCompensationParameters::rollOffset},
        {.name = "pitch_per_forward_velocity",
         .field = &OrientationCompensationBlock::pitch_per_forward_velocity,
         .parameter = &OrientationCompensationParameters::pitchPerForwardVelocity},
        {.name = "pitch_offset",
         .field = &OrientationCompensationBlock::pitch_offset,
         .parameter = &OrientationCompensationParameters::pitchOffset},
        {.name = "maximum_tilt",
         .field = &OrientationCompensationBlock::maximum_tilt,
         .parameter = &OrientationCompensationParameters::maximumTilt},
    }};

constexpr std::array<Coefficient<PeriodicOrientationBlock, PeriodicOrientationParameters>, 6> kPeriodicOrientationCoefficients = {{
    {.name = "roll_amplitude",
     .field = &PeriodicOrientationBlock::roll_amplitude,
     .parameter = &PeriodicOrientationParameters::rollAmplitude},
    {.name = "roll_phase_rate",
     .field = &PeriodicOrientationBlock::roll_phase_rate,
     .parameter = &PeriodicOrientationParameters::rollPhaseRate},
    {.name = "roll_phase_offset",
     .field = &PeriodicOrientationBlock::roll_phase_offset,
     .parameter = &PeriodicOrientationParameters::rollPhaseOffset},
    {.name = "pitch_amplitude",
     .field = &PeriodicOrientationBlock::pitch_amplitude,
     .parameter = &PeriodicOrientationParameters::pitchAmplitude},
    {.name = "pitch_phase_rate",
     .field = &PeriodicOrientationBlock::pitch_phase_rate,
     .parameter = &PeriodicOrientationParameters::pitchPhaseRate},
    {.name = "pitch_phase_offset",
     .field = &PeriodicOrientationBlock::pitch_phase_offset,
     .parameter = &PeriodicOrientationParameters::pitchPhaseOffset},
}};

constexpr std::array<Coefficient<HeightCompensationBlock, HeightCompensationParameters>, 4> kHeightCompensationCoefficients = {{
    {.name = "height_per_speed_squared",
     .field = &HeightCompensationBlock::height_per_speed_squared,
     .parameter = &HeightCompensationParameters::heightPerSpeedSquared},
    {.name = "height_per_speed",
     .field = &HeightCompensationBlock::height_per_speed,
     .parameter = &HeightCompensationParameters::heightPerSpeed},
    {.name = "height_offset", .field = &HeightCompensationBlock::height_offset, .parameter = &HeightCompensationParameters::heightOffset},
    {.name = "maximum_height_offset",
     .field = &HeightCompensationBlock::maximum_height_offset,
     .parameter = &HeightCompensationParameters::maximumHeightOffset},
}};

constexpr std::array<Coefficient<HipCenteredSteppingBlock, HipCenteredSteppingParameters>, 2> kHipCenteredSteppingCoefficients = {{
    {.name = "lateral_scale", .field = &HipCenteredSteppingBlock::lateral_scale, .parameter = &HipCenteredSteppingParameters::lateralScale},
    {.name = "longitudinal_scale",
     .field = &HipCenteredSteppingBlock::longitudinal_scale,
     .parameter = &HipCenteredSteppingParameters::longitudinalScale},
}};

constexpr std::array<Coefficient<CapturePointBlock, CapturePointParameters>, 4> kCapturePointCoefficients = {{
    {.name = "gain", .field = &CapturePointBlock::gain, .parameter = &CapturePointParameters::gain},
    {.name = "com_height_override",
     .field = &CapturePointBlock::com_height_override,
     .parameter = &CapturePointParameters::comHeightOverride},
    {.name = "gravity", .field = &CapturePointBlock::gravity, .parameter = &CapturePointParameters::gravity},
    {.name = "maximum_offset", .field = &CapturePointBlock::maximum_offset, .parameter = &CapturePointParameters::maximumOffset},
}};

constexpr std::array<Coefficient<TranslationalSteppingBlock, TranslationalSteppingParameters>, 6> kTranslationalSteppingCoefficients = {{
    {.name = "forward_per_forward_velocity",
     .field = &TranslationalSteppingBlock::forward_per_forward_velocity,
     .parameter = &TranslationalSteppingParameters::forwardPerForwardVelocity},
    {.name = "forward_stance_fraction",
     .field = &TranslationalSteppingBlock::forward_stance_fraction,
     .parameter = &TranslationalSteppingParameters::forwardStanceFraction},
    {.name = "forward_offset",
     .field = &TranslationalSteppingBlock::forward_offset,
     .parameter = &TranslationalSteppingParameters::forwardOffset},
    {.name = "lateral_per_lateral_velocity",
     .field = &TranslationalSteppingBlock::lateral_per_lateral_velocity,
     .parameter = &TranslationalSteppingParameters::lateralPerLateralVelocity},
    {.name = "lateral_stance_fraction",
     .field = &TranslationalSteppingBlock::lateral_stance_fraction,
     .parameter = &TranslationalSteppingParameters::lateralStanceFraction},
    {.name = "lateral_offset",
     .field = &TranslationalSteppingBlock::lateral_offset,
     .parameter = &TranslationalSteppingParameters::lateralOffset},
}};

constexpr std::array<Coefficient<InPlaceTurningBlock, InPlaceTurningParameters>, 5> kInPlaceTurningCoefficients = {{
    {.name = "forward_per_yaw_rate",
     .field = &InPlaceTurningBlock::forward_per_yaw_rate,
     .parameter = &InPlaceTurningParameters::forwardPerYawRate},
    {.name = "forward_stance_lever",
     .field = &InPlaceTurningBlock::forward_stance_lever,
     .parameter = &InPlaceTurningParameters::forwardStanceLever},
    {.name = "forward_offset", .field = &InPlaceTurningBlock::forward_offset, .parameter = &InPlaceTurningParameters::forwardOffset},
    {.name = "lateral_per_yaw_rate",
     .field = &InPlaceTurningBlock::lateral_per_yaw_rate,
     .parameter = &InPlaceTurningParameters::lateralPerYawRate},
    {.name = "lateral_offset", .field = &InPlaceTurningBlock::lateral_offset, .parameter = &InPlaceTurningParameters::lateralOffset},
}};

constexpr std::array<Coefficient<HighSpeedTurningBlock, HighSpeedTurningParameters>, 4> kHighSpeedTurningCoefficients = {{
    {.name = "forward_per_cross_term",
     .field = &HighSpeedTurningBlock::forward_per_cross_term,
     .parameter = &HighSpeedTurningParameters::forwardPerCrossTerm},
    {.name = "forward_offset", .field = &HighSpeedTurningBlock::forward_offset, .parameter = &HighSpeedTurningParameters::forwardOffset},
    {.name = "lateral_per_cross_term",
     .field = &HighSpeedTurningBlock::lateral_per_cross_term,
     .parameter = &HighSpeedTurningParameters::lateralPerCrossTerm},
    {.name = "lateral_offset", .field = &HighSpeedTurningBlock::lateral_offset, .parameter = &HighSpeedTurningParameters::lateralOffset},
}};

constexpr std::array<Coefficient<ImpulseScalingBlock, ImpulseScalingParameters>, 3> kImpulseScalingCoefficients = {{
    {.name = "scale", .field = &ImpulseScalingBlock::scale, .parameter = &ImpulseScalingParameters::scale},
    {.name = "minimum_duty_factor",
     .field = &ImpulseScalingBlock::minimum_duty_factor,
     .parameter = &ImpulseScalingParameters::minimumDutyFactor},
    {.name = "maximum_force_ratio",
     .field = &ImpulseScalingBlock::maximum_force_ratio,
     .parameter = &ImpulseScalingParameters::maximumForceRatio},
}};

constexpr std::array<Coefficient<CentripetalAccelerationBlock, CentripetalAccelerationParameters>, 3> kCentripetalAccelerationCoefficients =
    {{
        {.name = "scale", .field = &CentripetalAccelerationBlock::scale, .parameter = &CentripetalAccelerationParameters::scale},
        {.name = "maximum_force",
         .field = &CentripetalAccelerationBlock::maximum_force,
         .parameter = &CentripetalAccelerationParameters::maximumForce},
        {.name = "maximum_force_ratio_of_weight",
         .field = &CentripetalAccelerationBlock::maximum_force_ratio_of_weight,
         .parameter = &CentripetalAccelerationParameters::maximumForceRatioOfWeight},
    }};
}  // namespace

absl::StatusOr<LocomotionHeuristicConfig> locomotionHeuristicConfigFromConfig(const mpc_config::LocomotionHeuristicsConfig& config) {
  LocomotionHeuristicConfig heuristics;
  heuristics.formulation.basePose = config.base_pose;
  heuristics.formulation.foothold = config.foothold;
  heuristics.formulation.wrench = config.wrench;
  ASSIGN_OR_RETURN(heuristics.orientationCompensation, convertBlock(heuristic::kOrientationCompensation, config.orientation_compensation,
                                                                    kOrientationCompensationCoefficients));
  ASSIGN_OR_RETURN(heuristics.periodicOrientation,
                   convertBlock(heuristic::kPeriodicOrientation, config.periodic_orientation, kPeriodicOrientationCoefficients));
  ASSIGN_OR_RETURN(heuristics.heightCompensation,
                   convertBlock(heuristic::kHeightCompensation, config.height_compensation, kHeightCompensationCoefficients));
  ASSIGN_OR_RETURN(heuristics.hipCenteredStepping,
                   convertBlock(heuristic::kHipCenteredStepping, config.hip_centered_stepping, kHipCenteredSteppingCoefficients));
  ASSIGN_OR_RETURN(heuristics.capturePoint, convertBlock(heuristic::kCapturePoint, config.capture_point, kCapturePointCoefficients));
  ASSIGN_OR_RETURN(heuristics.translationalStepping,
                   convertBlock(heuristic::kTranslationalStepping, config.translational_stepping, kTranslationalSteppingCoefficients));
  ASSIGN_OR_RETURN(heuristics.inPlaceTurning,
                   convertBlock(heuristic::kInPlaceTurning, config.in_place_turning, kInPlaceTurningCoefficients));
  ASSIGN_OR_RETURN(heuristics.highSpeedTurning,
                   convertBlock(heuristic::kHighSpeedTurning, config.high_speed_turning, kHighSpeedTurningCoefficients));
  ASSIGN_OR_RETURN(heuristics.impulseScaling,
                   convertBlock(heuristic::kImpulseScaling, config.impulse_scaling, kImpulseScalingCoefficients));
  ASSIGN_OR_RETURN(heuristics.centripetalAcceleration, convertBlock(heuristic::kCentripetalAcceleration, config.centripetal_acceleration,
                                                                    kCentripetalAccelerationCoefficients));
  RETURN_IF_ERROR(heuristics.validate());
  return heuristics;
}

}  // namespace ocs2::humanoid
