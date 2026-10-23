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

#include "humanoid_common_mpc/parameter_update/QuadraticCostWeightsApplier.h"

#include <array>
#include <optional>
#include <string>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/misc/Collection.h"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/InputQuadraticCost.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateCost.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateInputCost.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(quadratic_cost_weights_fields)
constexpr std::array<absl::string_view, 4> kFields = {
    "state_weights",
    "input_weights",
    "final_state_weights",
    "terminal_cost_scaling",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto)

/** `weights` (`field`), refused unless it has the rows of the task file's tuning layout. */
absl::StatusOr<matrix_t> checkedTuningWeights(absl::StatusOr<matrix_t> weights, const HotUpdateTarget& target, absl::string_view field) {
  if (!weights.ok()) return weights;
  if (static_cast<size_t>(weights->rows()) != target.tuningDim() || static_cast<size_t>(weights->cols()) != target.tuningDim()) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, " is ", weights->rows(), " x ", weights->cols(), ", but the tuning layout has ", target.tuningDim(), " rows."));
  }
  return weights;
}

/** R of input_weights, refused unless it has the problem's input dimension. */
absl::StatusOr<matrix_t> inputWeights(const HotUpdateTarget& target) {
  absl::StatusOr<matrix_t> R = inputWeightsFromConfig(target.task().input_weights, target.layout(), "input_weights");
  if (!R.ok()) return R;
  if (static_cast<size_t>(R->rows()) != target.inputDim()) {
    return absl::InvalidArgumentError(
        absl::StrCat("input_weights has ", R->rows(), " rows, but the problem has ", target.inputDim(), " inputs."));
  }
  return R;
}

/**
 * Hands the term `name` of `collection`, when the problem carries it, to `set`, and reports a refusal of its setter as
 * the failed update of the term (reportFailedTermUpdate()): the setter compares the weights with the running term's own
 * dimensions. A term of another type is reported by updateTermIfPresent().
 */
template <typename Derived, typename Term>
void setWeightsIfPresent(Collection<Term>& collection, const std::string& name, absl::FunctionRef<absl::Status(Derived&)> set) {
  absl::Status refused = absl::OkStatus();
  updateTermIfPresent<Derived>(collection, name, [&](Derived& term) { refused = set(term); });
  if (!refused.ok()) reportFailedTermUpdate(name, refused);
}

}  // namespace

absl::Span<const absl::string_view> QuadraticCostWeightsApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

void QuadraticCostWeightsApplier::apply(HotUpdateTarget& target) {
  const mpc_config::TaskFile& task = target.task();
  const OptimalControlProblem& running = target.runningProblem();
  // The factory zeroes the base-pose block of both the running and the terminal state cost when the problem carries the
  // CoM + ACoM cost, so a reload must do the same or a slider drag silently re-introduces base-pose tracking. Read off
  // the running problem, never the reloaded file.
  const BasePoseWeights basePose = carriesTerm(*running.stateCostPtr, std::string(ComAndAcomTrackingCost::kRunningTermName))
                                       ? BasePoseWeights::kZeroedForComAndAcomTracking
                                       : BasePoseWeights::kKept;
  const bool quadraticTerminalCost = carriesTerm(*running.finalCostPtr, kTerminalCostTerm);

  // The conversions of the start-up factory (HumanoidCostConstraintFactory), so that a reload and a fresh start hand the
  // costs' TuningLayoutDeviation the same Euler-written rows.
  const std::optional<matrix_t> Q = convertedOrReported(
      checkedTuningWeights(trackingStateWeightsFromConfig(task.state_weights, target.layout(), "state_weights", basePose), target,
                           "state_weights"),
      target.source(), "state_weights");
  std::optional<matrix_t> R;
  if (inputCost_ == InputCost::kInputWeights) {
    R = convertedOrReported(inputWeights(target), target.source(), "input_weights");
  }
  std::optional<matrix_t> Q_final;
  if (quadraticTerminalCost) {
    if (const std::optional<scalar_t>& scaling = target.terminalCostScaling(); scaling.has_value()) {
      Q_final = convertedOrReported(checkedTuningWeights(terminalStateWeightsFromConfig(task.final_state_weights, *scaling, target.layout(),
                                                                                        "final_state_weights", basePose),
                                                         target, "final_state_weights"),
                                    target.source(), "final_state_weights");
    }
  }

  for (OptimalControlProblem& ocp : target.problems()) {
    // Each term is absent when the task file does not list it; whichever of Q and R was refused keeps its running value.
    if (Q.has_value()) {
      setWeightsIfPresent<TuningQuadraticStateInputCost>(*ocp.costPtr, kStateInputQuadraticCostTerm,
                                                         [&](TuningQuadraticStateInputCost& cost) { return cost.setStateWeights(*Q); });
      setWeightsIfPresent<TuningQuadraticStateInputCost>(*ocp.costPtr, kStateQuadraticCostTerm,
                                                         [&](TuningQuadraticStateInputCost& cost) { return cost.setStateWeights(*Q); });
    }
    if (R.has_value()) {
      setWeightsIfPresent<TuningQuadraticStateInputCost>(*ocp.costPtr, kStateInputQuadraticCostTerm,
                                                         [&](TuningQuadraticStateInputCost& cost) { return cost.setInputWeights(*R); });
      setWeightsIfPresent<InputQuadraticCost>(*ocp.costPtr, kInputQuadraticCostTerm,
                                              [&](InputQuadraticCost& cost) { return cost.setInputWeights(*R); });
    }
    if (Q_final.has_value()) {
      setWeightsIfPresent<TuningQuadraticStateCost>(*ocp.finalCostPtr, kTerminalCostTerm,
                                                    [&](TuningQuadraticStateCost& cost) { return cost.setStateWeights(*Q_final); });
    }
  }
}

}  // namespace ocs2::humanoid
