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

#include "humanoid_wb_mpc/parameter_update/StanceFootAccelerationGainsApplier.h"

#include <array>
#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"
#include "humanoid_wb_mpc/WBMpcPreComputation.h"
#include "humanoid_wb_mpc/constraint/EndEffectorDynamicsAccelerationsConstraint.h"
#include "humanoid_wb_mpc/constraint/ZeroAccelerationConstraintCppAd.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(stance_foot_acceleration_gains_fields)
constexpr std::array<absl::string_view, 8> kFields = {
    "model_settings.foot_constraint.position_error_gain_z",
    "model_settings.foot_constraint.orientation_error_gain",
    "model_settings.foot_constraint.linear_velocity_error_gain_z",
    "model_settings.foot_constraint.linear_velocity_error_gain_xy",
    "model_settings.foot_constraint.angular_velocity_error_gain",
    "model_settings.foot_constraint.linear_acceleration_error_gain_z",
    "model_settings.foot_constraint.linear_acceleration_error_gain_xy",
    "model_settings.foot_constraint.angular_acceleration_error_gain",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/model_settings_config.proto)

}  // namespace

absl::Span<const absl::string_view> StanceFootAccelerationGainsApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

void StanceFootAccelerationGainsApplier::apply(HotUpdateTarget& target) {
  const std::optional<ModelSettings::FootConstraintConfig>& gains = target.footConstraint();
  if (!gains.has_value()) return;
  const EndEffectorDynamicsAccelerationsConstraint::Config stanceConfig = stanceFootAccelerationConstraintConfig(*gains);
  const WBMpcPreComputation::SwingFootGains swingGains{.linearVelocityErrorGainZ = gains->linearVelocityErrorGain_z,
                                                       .linearAccelerationErrorGainZ = gains->linearAccelerationErrorGain_z};
  for (OptimalControlProblem& problem : target.problems()) {
    // The stance foot's zero_velocity term, hard or soft: the interface builds one of the two per foot, or neither.
    for (const std::string& footName : target.contactNames()) {
      const std::string term = zeroVelocityTermName(footName);
      updateTermIfPresent<ZeroAccelerationConstraintCppAd>(
          *problem.equalityConstraintPtr, term,
          [&stanceConfig](ZeroAccelerationConstraintCppAd& constraint) { constraint.configure(stanceConfig); });
      updateTermIfPresent<StateInputSoftConstraint>(*problem.softConstraintPtr, term, [&stanceConfig](StateInputSoftConstraint& soft) {
        soft.get<ZeroAccelerationConstraintCppAd>().configure(stanceConfig);
      });
    }
    // The swing foot's normal-motion coefficients, which every worker's pre-computation derives itself.
    // NOLINTNEXTLINE(rtti): OCS2 holds the pre-computation as a PreComputation; the whole-body problem's is a WBMpcPreComputation.
    WBMpcPreComputation* absl_nullable preComputationPtr = dynamic_cast<WBMpcPreComputation*>(problem.preComputationPtr.get());
    if (preComputationPtr == nullptr) {
      target.reportNotApplied("model_settings.foot_constraint (the swing foot's gains)",
                              absl::FailedPreconditionError("the problem's pre-computation is not the whole-body MPC's"));
      return;
    }
    preComputationPtr->setSwingFootGains(swingGains);
  }
}

}  // namespace ocs2::humanoid
