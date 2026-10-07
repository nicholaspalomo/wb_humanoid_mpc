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

#include "humanoid_centroidal_mpc/parameter_update/BasisInputsCostApplier.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/cost/QuadraticStateInputCost.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(basis_inputs_cost_fields)
constexpr std::array<absl::string_view, 4> kFields = {
    "input_weights",
    "contacts.basis_scaling_regularization",
    "contacts.basis_regularization",
    "contacts.basis_non_negativity_barrier",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto, //humanoid_nmpc/humanoid_mpc_config/contacts_config.proto)

/** The reloaded basis-space input cost and the transform it was built with. */
struct BasisSpaceInputCost {
  matrix_t R;
  BasisInputsCostTransformConfig transform;
};

/** input_weights of `target`'s file in basis space, transformed with `running` and the reloaded regularization. */
absl::StatusOr<BasisSpaceInputCost> basisSpaceInputCost(const HotUpdateTarget& target, const BasisInputsCostTransformConfig& running) {
  ASSIGN_OR_RETURN(const matrix_t R, inputWeightsFromConfig(target.task().input_weights, target.layout(), "input_weights"));
  BasisInputsCostTransformConfig candidate = running;
  ASSIGN_OR_RETURN(const BasisRegularizationSettings regularization, basisRegularizationFromConfig(target.task().contacts));
  candidate.lambdaRegularization = regularization.basisScalingRegularization;
  candidate.regularization = regularization.basisRegularization;
  if (static_cast<size_t>(R.rows()) != candidate.wrenchInputDim) {
    return absl::InvalidArgumentError(
        absl::StrCat("input_weights has ", R.rows(), " rows, but the wrench-space input has ", candidate.wrenchInputDim, "."));
  }
  // An unknown regularization name, a negative weight, or a lambda block that is not positive definite (a zero weight,
  // or null_space with a contact wrench direction R does not weigh) would install an R the QP cannot solve with.
  RETURN_IF_ERROR(validateBasisInputsCostTransformConfig(candidate));
  matrix_t R_basis = transformWrenchInputCostToBasisSpace(R, candidate);
  RETURN_IF_ERROR(checkLambdaBlockPositiveDefinite(R_basis, candidate.numBasisInputs));
  return BasisSpaceInputCost{.R = std::move(R_basis), .transform = std::move(candidate)};
}

}  // namespace

absl::Span<const absl::string_view> BasisInputsCostApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

absl::StatusOr<std::unique_ptr<BasisInputsCostApplier>> BasisInputsCostApplier::Create(
    std::optional<BasisInputsCostTransformConfig> basisCostTransform, size_t inputDim) {
  if (basisCostTransform.has_value()) {
    const BasisInputsCostTransformConfig& cfg = *basisCostTransform;
    if (inputDim != cfg.basisInputDim()) {
      return absl::InvalidArgumentError(absl::StrCat("[BasisInputsCostApplier] inputDim (", inputDim,
                                                     ") must equal the basis-space input dimension of the cost transform (",
                                                     cfg.basisInputDim(), ")."));
    }
    if (static_cast<size_t>(cfg.basisToWrenchMap.rows()) != cfg.wrenchInputDim) {
      return absl::InvalidArgumentError(absl::StrCat("[BasisInputsCostApplier] basisToWrenchMap has ", cfg.basisToWrenchMap.rows(),
                                                     " rows but wrenchInputDim is ", cfg.wrenchInputDim, "."));
    }
    if (cfg.numBasisInputs > inputDim) {
      return absl::InvalidArgumentError(
          absl::StrCat("[BasisInputsCostApplier] numBasisInputs (", cfg.numBasisInputs, ") exceeds inputDim (", inputDim, ")."));
    }
    RETURN_IF_ERROR(validateBasisInputsCostTransformConfig(cfg));
  }
  return absl::WrapUnique(new BasisInputsCostApplier(std::move(basisCostTransform)));
}

void BasisInputsCostApplier::apply(HotUpdateTarget& target) {
  if (!basisCostTransform_.has_value()) return;
  std::optional<BasisSpaceInputCost> inputCost =
      convertedOrReported(basisSpaceInputCost(target, *basisCostTransform_), target.source(), "input_weights");
  // The lambda >= 0 barrier of the basis-vector contact inputs, which CentroidalMpcInterface builds for every contact.
  const std::optional<PieceWisePolynomialBarrierPenalty::Config> barrier = convertedOrReported(
      basisNonNegativityBarrierFromConfig(target.task().contacts), target.source(), "contacts.basis_non_negativity_barrier");

  const matrix_t zeroQ = matrix_t::Zero(target.stateDim(), target.stateDim());
  for (OptimalControlProblem& ocp : target.problems()) {
    if (inputCost.has_value()) {
      // Q of stateInputQuadraticCost is QuadraticCostWeightsApplier's, which kept the running R in it.
      updateTermIfPresent<QuadraticStateInputCost>(*ocp.costPtr, kStateInputQuadraticCostTerm, [&](QuadraticStateInputCost& cost) {
        matrix_t runningQ;
        matrix_t runningR;
        matrix_t runningP;
        cost.getGains(runningQ, runningR, runningP);
        cost.setGains(runningQ, inputCost->R, runningP);
      });
      updateTermIfPresent<QuadraticStateInputCost>(*ocp.costPtr, kInputQuadraticCostTerm,
                                                   [&](QuadraticStateInputCost& cost) { cost.setGains(zeroQ, inputCost->R); });
    }
    if (barrier.has_value()) {
      for (const std::string& footName : target.contactNames()) {
        updateTermIfPresent<BasisScalingNonNegativityConstraint>(
            *ocp.costPtr, basisNonNegativityTermName(footName),
            [&](BasisScalingNonNegativityConstraint& term) { term.setBarrierPenalty(*barrier); });
      }
    }
  }
  if (inputCost.has_value()) basisCostTransform_ = std::move(inputCost->transform);
}

}  // namespace ocs2::humanoid
