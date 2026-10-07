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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_centroidal_mpc/parameter_update/ZeroVelocityGainsApplier.h"

#include <array>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/misc/Numerics.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(zero_velocity_gains_fields)
constexpr std::array<absl::string_view, 6> kFields = {
    "model_settings.foot_constraint.position_error_gain_z",        "model_settings.foot_constraint.orientation_error_gain",
    "model_settings.foot_constraint.linear_velocity_error_gain_z", "model_settings.foot_constraint.linear_velocity_error_gain_xy",
    "model_settings.foot_constraint.angular_velocity_error_gain",  "model_settings.foot_constraint.stance_constraint",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/model_settings_config.proto)

}  // namespace

absl::Span<const absl::string_view> ZeroVelocityGainsApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

EndEffectorKinematicsTwistConstraint::Config footTwistConfig(const ModelSettings::FootConstraintConfig& gains) {
  EndEffectorKinematicsTwistConstraint::Config config;
  config.b.setZero(6);
  config.Ax.setZero(6, 6);
  config.Av.setZero(6, 6);
  if (!numerics::almost_eq(gains.positionErrorGain_z, /*y=*/0.0)) {
    config.Ax(2, 2) = gains.positionErrorGain_z;
  }
  if (!numerics::almost_eq(gains.orientationErrorGain, /*y=*/0.0)) {
    config.Ax.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * gains.orientationErrorGain;
  }
  config.Av(0, 0) = gains.linearVelocityErrorGain_xy;
  config.Av(1, 1) = gains.linearVelocityErrorGain_xy;
  config.Av(2, 2) = gains.linearVelocityErrorGain_z;
  config.Av(3, 3) = gains.angularVelocityErrorGain;
  config.Av(4, 4) = gains.angularVelocityErrorGain;
  config.Av(5, 5) = gains.angularVelocityErrorGain;
  return config;
}

void configureZeroVelocityConstraint(ZeroVelocityConstraintCppAd& constraint,
                                     const ModelSettings::FootConstraintConfig& gains,
                                     const EndEffectorKinematicsTwistConstraint::Config& twistConfig) {
  EndEffectorKinematicsTwistConstraint& twist = constraint.getTwistConstraint();
  twist.setNumConstraints(gains.constrainOrientation ? 6 : 3);
  twist.setConstrainYawRateAboutNormal(gains.constrainYawRateAboutContactNormal);
  twist.configure(EndEffectorKinematicsTwistConstraint::Config(twistConfig));
}

void ZeroVelocityGainsApplier::apply(HotUpdateTarget& target) {
  const std::optional<ModelSettings::FootConstraintConfig>& gains = target.footConstraint();
  if (!gains.has_value()) return;
  // A zero_velocity term is hard (the equality constraints), soft (wrapped in a StateInputSoftConstraint) or not listed;
  // whichever of the two the problem does not carry is skipped.
  const EndEffectorKinematicsTwistConstraint::Config twistConfig = footTwistConfig(*gains);
  for (OptimalControlProblem& ocp : target.problems()) {
    for (const std::string& footName : target.contactNames()) {
      const std::string termName = zeroVelocityTermName(footName);
      updateTermIfPresent<ZeroVelocityConstraintCppAd>(*ocp.equalityConstraintPtr, termName, [&](ZeroVelocityConstraintCppAd& constraint) {
        configureZeroVelocityConstraint(constraint, *gains, twistConfig);
      });
      updateTermIfPresent<StateInputSoftConstraint>(*ocp.softConstraintPtr, termName, [&](StateInputSoftConstraint& softCon) {
        configureZeroVelocityConstraint(softCon.get<ZeroVelocityConstraintCppAd>(), *gains, twistConfig);
      });
    }
  }
}

}  // namespace ocs2::humanoid
