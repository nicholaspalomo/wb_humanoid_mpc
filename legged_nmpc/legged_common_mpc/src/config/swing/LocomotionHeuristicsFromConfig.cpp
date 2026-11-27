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

#include "legged_common_mpc/config/swing/LocomotionHeuristicsFromConfig.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "legged_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"
#include "legged_mpc_config/locomotion_heuristics_config.nproto.h"

namespace ocs2::legged {
namespace {

using HeuristicsConfig = mpc_config::LocomotionHeuristicsConfig;

/**
 * A field of a heuristic's parameter block and the member of the heuristic's
 * parameter struct it becomes; `name` is a string literal.
 */
template <typename Block, typename Parameters>
struct Coefficient {
  const char* absl_nonnull name;
  double Block::*absl_nonnull field;
  scalar_t Parameters::*absl_nonnull parameter;
};

/** The parameters of the heuristic `heuristic` from its block, field by field;
 * InvalidArgument for a value that is not finite. */
template <typename Block, typename Parameters, size_t kCount>
absl::StatusOr<Parameters> convertBlock(
    absl::string_view heuristic,
    const Block& block,
    const std::array<Coefficient<Block, Parameters>, kCount>& coefficients) {
  Parameters parameters;
  for (const Coefficient<Block, Parameters>& coefficient : coefficients) {
    const double value = block.*coefficient.field;
    if (!std::isfinite(value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("locomotion_heuristics.", parameterBlockName(heuristic),
                       ".", coefficient.name, " is ", value,
                       ", but every parameter of the block must be finite."));
    }
    parameters.*coefficient.parameter = value;
  }
  return parameters;
}

// Every field of the parameter blocks of locomotion_heuristics_config.proto, in
// its order, with the member it becomes.
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

constexpr std::array<Coefficient<OrientationCompensationBlock,
                                 OrientationCompensationParameters>,
                     5>
    kOrientationCompensationCoefficients = {{
        {.name = "roll_per_lateral_velocity_gain",
         .field = &OrientationCompensationBlock::roll_per_lateral_velocity_gain,
         .parameter =
             &OrientationCompensationParameters::rollPerLateralVelocityGain},
        {.name = "offset_eulerzyx_base_rt_world_roll",
         .field =
             &OrientationCompensationBlock::offset_eulerzyx_base_rt_world_roll,
         .parameter =
             &OrientationCompensationParameters::offsetEulerzyxBaseRtWorldRoll},
        {.name = "pitch_per_forward_velocity_gain",
         .field =
             &OrientationCompensationBlock::pitch_per_forward_velocity_gain,
         .parameter =
             &OrientationCompensationParameters::pitchPerForwardVelocityGain},
        {.name = "offset_eulerzyx_base_rt_world_pitch",
         .field =
             &OrientationCompensationBlock::offset_eulerzyx_base_rt_world_pitch,
         .parameter = &OrientationCompensationParameters::
                          offsetEulerzyxBaseRtWorldPitch},
        {.name = "max_offset_eulerzyx_base_rt_world",
         .field =
             &OrientationCompensationBlock::max_offset_eulerzyx_base_rt_world,
         .parameter =
             &OrientationCompensationParameters::maxOffsetEulerzyxBaseRtWorld},
    }};

constexpr std::array<
    Coefficient<PeriodicOrientationBlock, PeriodicOrientationParameters>,
    6>
    kPeriodicOrientationCoefficients = {{
        {.name = "amplitude_eulerzyx_base_rt_world_roll",
         .field =
             &PeriodicOrientationBlock::amplitude_eulerzyx_base_rt_world_roll,
         .parameter =
             &PeriodicOrientationParameters::amplitudeEulerzyxBaseRtWorldRoll},
        {.name = "roll_phase_rate",
         .field = &PeriodicOrientationBlock::roll_phase_rate,
         .parameter = &PeriodicOrientationParameters::rollPhaseRate},
        {.name = "roll_phase_offset",
         .field = &PeriodicOrientationBlock::roll_phase_offset,
         .parameter = &PeriodicOrientationParameters::rollPhaseOffset},
        {.name = "amplitude_eulerzyx_base_rt_world_pitch",
         .field =
             &PeriodicOrientationBlock::amplitude_eulerzyx_base_rt_world_pitch,
         .parameter =
             &PeriodicOrientationParameters::amplitudeEulerzyxBaseRtWorldPitch},
        {.name = "pitch_phase_rate",
         .field = &PeriodicOrientationBlock::pitch_phase_rate,
         .parameter = &PeriodicOrientationParameters::pitchPhaseRate},
        {.name = "pitch_phase_offset",
         .field = &PeriodicOrientationBlock::pitch_phase_offset,
         .parameter = &PeriodicOrientationParameters::pitchPhaseOffset},
    }};

constexpr std::array<
    Coefficient<HeightCompensationBlock, HeightCompensationParameters>,
    4>
    kHeightCompensationCoefficients = {{
        {.name = "height_per_speed_squared_gain",
         .field = &HeightCompensationBlock::height_per_speed_squared_gain,
         .parameter = &HeightCompensationParameters::heightPerSpeedSquaredGain},
        {.name = "height_per_speed_gain",
         .field = &HeightCompensationBlock::height_per_speed_gain,
         .parameter = &HeightCompensationParameters::heightPerSpeedGain},
        {.name = "offset_pos_base_rt_world_ewrt_world_z",
         .field =
             &HeightCompensationBlock::offset_pos_base_rt_world_ewrt_world_z,
         .parameter =
             &HeightCompensationParameters::offsetPosBaseRtWorldEwrtWorldZ},
        {.name = "max_offset_pos_base_rt_world_ewrt_world_z",
         .field = &HeightCompensationBlock::
                      max_offset_pos_base_rt_world_ewrt_world_z,
         .parameter =
             &HeightCompensationParameters::maxOffsetPosBaseRtWorldEwrtWorldZ},
    }};

constexpr std::array<
    Coefficient<HipCenteredSteppingBlock, HipCenteredSteppingParameters>,
    2>
    kHipCenteredSteppingCoefficients = {{
        {.name = "lateral_scale",
         .field = &HipCenteredSteppingBlock::lateral_scale,
         .parameter = &HipCenteredSteppingParameters::lateralScale},
        {.name = "longitudinal_scale",
         .field = &HipCenteredSteppingBlock::longitudinal_scale,
         .parameter = &HipCenteredSteppingParameters::longitudinalScale},
    }};

// override_pos_com_rt_feetmean_ewrt_world_z is optional (optionalPositive()).
constexpr std::array<Coefficient<CapturePointBlock, CapturePointParameters>, 2>
    kCapturePointCoefficients = {{
        {.name = "gain",
         .field = &CapturePointBlock::gain,
         .parameter = &CapturePointParameters::gain},
        {.name = "max_norm_offset_pos_foot_rt_world_ewrt_world_xy",
         .field = &CapturePointBlock::
                      max_norm_offset_pos_foot_rt_world_ewrt_world_xy,
         .parameter =
             &CapturePointParameters::maxNormOffsetPosFootRtWorldEwrtWorldXy},
    }};

constexpr std::array<
    Coefficient<TranslationalSteppingBlock, TranslationalSteppingParameters>,
    6>
    kTranslationalSteppingCoefficients = {{
        {.name = "forward_per_forward_velocity_gain",
         .field =
             &TranslationalSteppingBlock::forward_per_forward_velocity_gain,
         .parameter =
             &TranslationalSteppingParameters::forwardPerForwardVelocityGain},
        {.name = "forward_stance_fraction",
         .field = &TranslationalSteppingBlock::forward_stance_fraction,
         .parameter = &TranslationalSteppingParameters::forwardStanceFraction},
        {.name = "offset_pos_foot_rt_world_ewrt_heading_x",
         .field = &TranslationalSteppingBlock::
                      offset_pos_foot_rt_world_ewrt_heading_x,
         .parameter = &TranslationalSteppingParameters::
                          offsetPosFootRtWorldEwrtHeadingX},
        {.name = "lateral_per_lateral_velocity_gain",
         .field =
             &TranslationalSteppingBlock::lateral_per_lateral_velocity_gain,
         .parameter =
             &TranslationalSteppingParameters::lateralPerLateralVelocityGain},
        {.name = "lateral_stance_fraction",
         .field = &TranslationalSteppingBlock::lateral_stance_fraction,
         .parameter = &TranslationalSteppingParameters::lateralStanceFraction},
        {.name = "offset_pos_foot_rt_world_ewrt_heading_y",
         .field = &TranslationalSteppingBlock::
                      offset_pos_foot_rt_world_ewrt_heading_y,
         .parameter = &TranslationalSteppingParameters::
                          offsetPosFootRtWorldEwrtHeadingY},
    }};

constexpr std::array<Coefficient<InPlaceTurningBlock, InPlaceTurningParameters>,
                     5>
    kInPlaceTurningCoefficients = {{
        {.name = "forward_per_yaw_rate",
         .field = &InPlaceTurningBlock::forward_per_yaw_rate,
         .parameter = &InPlaceTurningParameters::forwardPerYawRate},
        {.name = "forward_stance_lever",
         .field = &InPlaceTurningBlock::forward_stance_lever,
         .parameter = &InPlaceTurningParameters::forwardStanceLever},
        {.name = "offset_pos_foot_rt_world_ewrt_heading_x",
         .field = &InPlaceTurningBlock::offset_pos_foot_rt_world_ewrt_heading_x,
         .parameter =
             &InPlaceTurningParameters::offsetPosFootRtWorldEwrtHeadingX},
        {.name = "lateral_per_yaw_rate",
         .field = &InPlaceTurningBlock::lateral_per_yaw_rate,
         .parameter = &InPlaceTurningParameters::lateralPerYawRate},
        {.name = "offset_pos_foot_rt_world_ewrt_heading_y",
         .field = &InPlaceTurningBlock::offset_pos_foot_rt_world_ewrt_heading_y,
         .parameter =
             &InPlaceTurningParameters::offsetPosFootRtWorldEwrtHeadingY},
    }};

constexpr std::array<
    Coefficient<HighSpeedTurningBlock, HighSpeedTurningParameters>,
    4>
    kHighSpeedTurningCoefficients = {{
        {.name = "forward_per_cross_term",
         .field = &HighSpeedTurningBlock::forward_per_cross_term,
         .parameter = &HighSpeedTurningParameters::forwardPerCrossTerm},
        {.name = "offset_pos_foot_rt_world_ewrt_heading_x",
         .field =
             &HighSpeedTurningBlock::offset_pos_foot_rt_world_ewrt_heading_x,
         .parameter =
             &HighSpeedTurningParameters::offsetPosFootRtWorldEwrtHeadingX},
        {.name = "lateral_per_cross_term",
         .field = &HighSpeedTurningBlock::lateral_per_cross_term,
         .parameter = &HighSpeedTurningParameters::lateralPerCrossTerm},
        {.name = "offset_pos_foot_rt_world_ewrt_heading_y",
         .field =
             &HighSpeedTurningBlock::offset_pos_foot_rt_world_ewrt_heading_y,
         .parameter =
             &HighSpeedTurningParameters::offsetPosFootRtWorldEwrtHeadingY},
    }};

constexpr std::array<Coefficient<ImpulseScalingBlock, ImpulseScalingParameters>,
                     3>
    kImpulseScalingCoefficients = {{
        {.name = "scale",
         .field = &ImpulseScalingBlock::scale,
         .parameter = &ImpulseScalingParameters::scale},
        {.name = "minimum_duty_factor",
         .field = &ImpulseScalingBlock::minimum_duty_factor,
         .parameter = &ImpulseScalingParameters::minimumDutyFactor},
        {.name = "maximum_weight_compensation_ratio",
         .field = &ImpulseScalingBlock::maximum_weight_compensation_ratio,
         .parameter =
             &ImpulseScalingParameters::maximumWeightCompensationRatio},
    }};

// max_force_com_ewrt_world_xy is optional (optionalPositive()).
constexpr std::array<Coefficient<CentripetalAccelerationBlock,
                                 CentripetalAccelerationParameters>,
                     2>
    kCentripetalAccelerationCoefficients = {{
        {.name = "scale",
         .field = &CentripetalAccelerationBlock::scale,
         .parameter = &CentripetalAccelerationParameters::scale},
        {.name = "max_wnorm_force_com_ewrt_world_xy",
         .field =
             &CentripetalAccelerationBlock::max_wnorm_force_com_ewrt_world_xy,
         .parameter =
             &CentripetalAccelerationParameters::maxWnormForceComEwrtWorldXy},
    }};
/**
 * The parameter of the optional field `name` of the block `heuristic`:
 * `absent`, the parameter struct's value for "not set", when the file leaves it
 * out; otherwise the value, which must be finite and positive (the schema has
 * no sentinel: leaving the field out is how a file says "not set").
 */
absl::StatusOr<scalar_t> optionalPositive(absl::string_view heuristic,
                                          absl::string_view name,
                                          std::optional<double> value,
                                          scalar_t absent) {
  if (!value.has_value()) return absent;
  // Negated, so that NaN is refused too.
  if (!std::isfinite(*value) || !(*value > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "locomotion_heuristics.", parameterBlockName(heuristic), ".", name,
        " is ", *value,
        ", but it must be finite and positive; leave it out for the default."));
  }
  return *value;
}

}  // namespace

absl::StatusOr<LocomotionHeuristicConfig> locomotionHeuristicConfigFromConfig(
    const mpc_config::LocomotionHeuristicsConfig& config) {
  LocomotionHeuristicConfig heuristics;
  heuristics.formulation.basePoseHeuristics = config.base_pose_heuristics;
  heuristics.formulation.footholdHeuristics = config.foothold_heuristics;
  heuristics.formulation.wrenchHeuristics = config.wrench_heuristics;
  ASSIGN_OR_RETURN(heuristics.orientationCompensationParameters,
                   convertBlock(heuristic::kOrientationCompensationName,
                                config.orientation_compensation_parameters,
                                kOrientationCompensationCoefficients));
  ASSIGN_OR_RETURN(heuristics.periodicOrientationParameters,
                   convertBlock(heuristic::kPeriodicOrientationName,
                                config.periodic_orientation_parameters,
                                kPeriodicOrientationCoefficients));
  ASSIGN_OR_RETURN(
      heuristics.heightCompensation,
      convertBlock(heuristic::kHeightCompensation, config.height_compensation,
                   kHeightCompensationCoefficients));
  ASSIGN_OR_RETURN(heuristics.hipCenteredStepping,
                   convertBlock(heuristic::kHipCenteredStepping,
                                config.hip_centered_stepping,
                                kHipCenteredSteppingCoefficients));
  ASSIGN_OR_RETURN(heuristics.capturePoint,
                   convertBlock(heuristic::kCapturePoint, config.capture_point,
                                kCapturePointCoefficients));
  // Absent: the center-of-mass height measured at each solve
  // (CapturePointParameters::overridePosComRtFeetmeanEwrtWorldZ 0).
  ASSIGN_OR_RETURN(
      heuristics.capturePoint.overridePosComRtFeetmeanEwrtWorldZ,
      optionalPositive(
          heuristic::kCapturePoint, "override_pos_com_rt_feetmean_ewrt_world_z",
          config.capture_point.override_pos_com_rt_feetmean_ewrt_world_z,
          /*absent=*/0.0));
  ASSIGN_OR_RETURN(heuristics.translationalStepping,
                   convertBlock(heuristic::kTranslationalStepping,
                                config.translational_stepping,
                                kTranslationalSteppingCoefficients));
  ASSIGN_OR_RETURN(
      heuristics.inPlaceTurning,
      convertBlock(heuristic::kInPlaceTurning, config.in_place_turning,
                   kInPlaceTurningCoefficients));
  ASSIGN_OR_RETURN(
      heuristics.highSpeedTurning,
      convertBlock(heuristic::kHighSpeedTurning, config.high_speed_turning,
                   kHighSpeedTurningCoefficients));
  ASSIGN_OR_RETURN(
      heuristics.impulseScaling,
      convertBlock(heuristic::kImpulseScaling, config.impulse_scaling,
                   kImpulseScalingCoefficients));
  ASSIGN_OR_RETURN(heuristics.centripetalAccelerationParameters,
                   convertBlock(heuristic::kCentripetalAccelerationName,
                                config.centripetal_acceleration_parameters,
                                kCentripetalAccelerationCoefficients));
  // Absent: the clamp is max_wnorm_force_com_ewrt_world_xy
  // (CentripetalAccelerationParameters::maxForceComEwrtWorldXy 0).
  ASSIGN_OR_RETURN(
      heuristics.centripetalAccelerationParameters.maxForceComEwrtWorldXy,
      optionalPositive(heuristic::kCentripetalAccelerationName,
                       "max_force_com_ewrt_world_xy",
                       config.centripetal_acceleration_parameters
                           .max_force_com_ewrt_world_xy,
                       /*absent=*/0.0));
  RETURN_IF_ERROR(heuristics.validate());
  return heuristics;
}

}  // namespace ocs2::legged
