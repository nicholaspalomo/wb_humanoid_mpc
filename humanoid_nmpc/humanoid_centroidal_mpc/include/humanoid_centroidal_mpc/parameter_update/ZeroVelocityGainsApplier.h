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

#pragma once

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_centroidal_mpc/constraint/ZeroVelocityConstraintCppAd.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"

namespace ocs2::humanoid {

/**
 * Applies the foot constraint's error gains and its stance constraint to the twist constraint of every foot's
 * zero_velocity term, hard (an equality constraint) or soft (inside a StateInputSoftConstraint): its rows (6 with the
 * orientation, 3 without), the yaw-rate row about the contact normal and the gains of footTwistConfig(). The
 * pre-computation's copy of position_error_gain_z is FootConstraintSoftWeightsApplier's. Not thread-safe
 * (HotFieldApplier).
 */
class ZeroVelocityGainsApplier final : public HotFieldApplier {
 public:
  /**
   * The fields it applies: model_settings.foot_constraint.{position_error_gain_z, orientation_error_gain,
   * linear_velocity_error_gain_z, linear_velocity_error_gain_xy, angular_velocity_error_gain, stance_constraint}.
   */
  static absl::Span<const absl::string_view> staticFields();

  ZeroVelocityGainsApplier() = default;

  absl::string_view name() const override { return "ZeroVelocityGainsApplier"; }
  absl::Span<const absl::string_view> fields() const final { return staticFields(); }
  void apply(HotUpdateTarget& target) override;
};

/** The Ax/Av gains of the zero_velocity twist constraint of `gains` (mirrors CentroidalMpcInterface::getStanceFootConstraint). */
EndEffectorKinematicsTwistConstraint::Config footTwistConfig(const ModelSettings::FootConstraintConfig& gains);

/**
 * Applies `gains` to the twist constraint of the zero_velocity term `constraint`: its rows (6 with the orientation, 3
 * without), the yaw-rate row about the contact normal and the error gains of `twistConfig` (footTwistConfig(gains)).
 */
void configureZeroVelocityConstraint(ZeroVelocityConstraintCppAd& constraint,
                                     const ModelSettings::FootConstraintConfig& gains,
                                     const EndEffectorKinematicsTwistConstraint::Config& twistConfig);

}  // namespace ocs2::humanoid
