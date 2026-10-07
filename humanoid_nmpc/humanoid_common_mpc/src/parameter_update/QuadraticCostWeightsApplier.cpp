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
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/cost/QuadraticStateCost.h"
#include "ocs2_core/cost/QuadraticStateInputCost.h"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
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

/** What stateWeights() does with the base-pose block of a state weight matrix. */
enum class BasePoseWeights {
  kKept,
  // Zeroed: the problem carries the CoM + ACoM cost, which tracks the base pose instead.
  kZeroedForComAndAcomTracking,
};

/**
 * The state weight matrix of `weights` (`field`), refused unless it has the problem's state dimension, with its
 * base-pose block as `basePose` says.
 */
absl::StatusOr<matrix_t> stateWeights(const mpc_config::StateWeights& weights,
                                      const HotUpdateTarget& target,
                                      absl::string_view field,
                                      BasePoseWeights basePose) {
  absl::StatusOr<matrix_t> Q = stateWeightsFromConfig(weights, target.layout(), field);
  if (!Q.ok()) return Q;
  if (static_cast<size_t>(Q->rows()) != target.stateDim() || static_cast<size_t>(Q->cols()) != target.stateDim()) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, " is ", Q->rows(), " x ", Q->cols(), ", but the problem has ", target.stateDim(), " states."));
  }
  // The factory zeroes the base-pose block of both the running and the terminal state cost when the problem carries the
  // CoM + ACoM cost, so a reload must do the same or a slider drag silently re-introduces base-pose tracking.
  if (basePose == BasePoseWeights::kZeroedForComAndAcomTracking) ComAndAcomTrackingCost::zeroBasePoseWeights(*Q);
  return Q;
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

}  // namespace

absl::Span<const absl::string_view> QuadraticCostWeightsApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

void QuadraticCostWeightsApplier::apply(HotUpdateTarget& target) {
  const mpc_config::TaskFile& task = target.task();
  const OptimalControlProblem& running = target.runningProblem();
  const BasePoseWeights basePose = carriesTerm(*running.stateCostPtr, std::string(ComAndAcomTrackingCost::kRunningTermName))
                                       ? BasePoseWeights::kZeroedForComAndAcomTracking
                                       : BasePoseWeights::kKept;
  const bool quadraticTerminalCost = carriesTerm(*running.finalCostPtr, kTerminalCostTerm);

  const std::optional<matrix_t> Q =
      convertedOrReported(stateWeights(task.state_weights, target, "state_weights", basePose), target.source(), "state_weights");
  std::optional<matrix_t> R;
  if (inputCost_ == InputCost::kInputWeights) {
    R = convertedOrReported(inputWeights(target), target.source(), "input_weights");
  }
  std::optional<matrix_t> Q_final;
  if (quadraticTerminalCost) {
    if (const std::optional<scalar_t>& scaling = target.terminalCostScaling(); scaling.has_value()) {
      Q_final = convertedOrReported(stateWeights(task.final_state_weights, target, "final_state_weights", basePose), target.source(),
                                    "final_state_weights");
      if (Q_final.has_value()) *Q_final *= *scaling;
    }
  }

  const matrix_t zeroR = matrix_t::Zero(target.inputDim(), target.inputDim());
  const matrix_t zeroQ = matrix_t::Zero(target.stateDim(), target.stateDim());
  for (OptimalControlProblem& ocp : target.problems()) {
    // Each term is absent when the task file does not list it; whichever of Q and R was refused keeps its running value.
    updateTermIfPresent<QuadraticStateInputCost>(*ocp.costPtr, kStateInputQuadraticCostTerm, [&](QuadraticStateInputCost& cost) {
      matrix_t runningQ;
      matrix_t runningR;
      matrix_t runningP;
      cost.getGains(runningQ, runningR, runningP);
      cost.setGains(Q.has_value() ? *Q : runningQ, R.has_value() ? *R : runningR, runningP);
    });
    if (Q.has_value()) {
      updateTermIfPresent<QuadraticStateInputCost>(*ocp.costPtr, kStateQuadraticCostTerm,
                                                   [&](QuadraticStateInputCost& cost) { cost.setGains(*Q, zeroR); });
    }
    if (R.has_value()) {
      updateTermIfPresent<QuadraticStateInputCost>(*ocp.costPtr, kInputQuadraticCostTerm,
                                                   [&](QuadraticStateInputCost& cost) { cost.setGains(zeroQ, *R); });
    }
    if (Q_final.has_value()) {
      updateTermIfPresent<QuadraticStateCost>(*ocp.finalCostPtr, kTerminalCostTerm,
                                              [&](QuadraticStateCost& cost) { cost.setGains(*Q_final); });
    }
  }
}

}  // namespace ocs2::humanoid
