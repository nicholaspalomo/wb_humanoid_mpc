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

#include "humanoid_common_mpc/parameter_update/FootConstraintSoftWeightsApplier.h"

#include <array>
#include <optional>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/HumanoidPreComputation.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(foot_constraint_soft_weights_fields)
constexpr std::array<absl::string_view, 3> kFields = {
    "model_settings.foot_constraint.soft_constraint_weight",
    "model_settings.foot_constraint.normal_velocity_soft_constraint_weight",
    "model_settings.foot_constraint.position_error_gain_z",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/model_settings_config.proto)

}  // namespace

absl::Span<const absl::string_view> FootConstraintSoftWeightsApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

void FootConstraintSoftWeightsApplier::apply(HotUpdateTarget& target) {
  const std::optional<ModelSettings::FootConstraintConfig>& gains = target.footConstraint();
  if (!gains.has_value()) return;
  // No soft zero_velocity: it is listed as a hard constraint (the shipped Atlas), or not at all (the contact-implicit
  // formulation). No soft normal_velocity in every schedule-gated configuration.
  setSoftTermWeight(target.problems(), target.contactNames(), kZeroVelocityTermSuffix, gains->softConstraintWeight);
  setSoftTermWeight(target.problems(), target.contactNames(), contact_term::kNormalVelocitySoft, gains->normalVelocitySoftConstraintWeight);

  // position_error_gain_z has to reach the PRE-COMPUTATION as well as the zero_velocity twist config, and that is not a
  // refinement: under the contact-implicit formulation zero_velocity is not built at all, so the twist gain is written
  // into a term that does not exist while the term that does - the soft normal-velocity servo - reads it from here.
  // Each worker thread owns its own PreComputation, so writing it per problem is also what keeps this race-free;
  // mutating the shared ModelSettings would not be.
  for (OptimalControlProblem& ocp : target.problems()) {
    // NOLINTNEXTLINE(rtti): OCS2 holds the pre-computation as a PreComputation, and a problem may carry another kind.
    HumanoidPreComputation* absl_nullable preComputationPtr = dynamic_cast<HumanoidPreComputation*>(ocp.preComputationPtr.get());
    if (preComputationPtr != nullptr) {
      preComputationPtr->setNormalVelocityPositionErrorGain(gains->positionErrorGain_z);
    }
  }
}

}  // namespace ocs2::humanoid
