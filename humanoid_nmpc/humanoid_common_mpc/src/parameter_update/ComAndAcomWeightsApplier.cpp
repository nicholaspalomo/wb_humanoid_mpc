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

#include "humanoid_common_mpc/parameter_update/ComAndAcomWeightsApplier.h"

#include <array>
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(com_and_acom_weights_fields)
constexpr std::array<absl::string_view, 2> kFields = {
    "com_weights",
    "acom_weights",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto)

/** Sets the CoM and ACoM weights of the term `termName` of `collection`; a refused matrix or a mistyped term is returned. */
template <typename Term>
absl::Status setComAndAcomWeights(Collection<Term>& collection, const std::string& termName, const matrix_t& com, const matrix_t& acom) {
  // A refused weight matrix comes back from setWeights(), a missing or mistyped term from updateTerm().
  absl::Status weightsSet;
  absl::Status status = updateTerm<ComAndAcomTrackingCost>(collection, termName,
                                                           [&](ComAndAcomTrackingCost& cost) { weightsSet = cost.setWeights(com, acom); });
  if (status.ok()) status = weightsSet;
  return status;
}

}  // namespace

absl::Span<const absl::string_view> ComAndAcomWeightsApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

void ComAndAcomWeightsApplier::apply(HotUpdateTarget& target) {
  const std::string runningTermName(ComAndAcomTrackingCost::kRunningTermName);
  if (!carriesTerm(*target.runningProblem().stateCostPtr, runningTermName)) return;
  // The terminal instance is weighted like final_state_weights, as at start-up; without the scaling it keeps its weights.
  const std::optional<scalar_t>& terminalCostScaling = target.terminalCostScaling();
  const mpc_config::TaskFile& task = target.task();
  const absl::StatusOr<matrix_t> com = comWeightsFromConfig(task.com_weights, "com_weights");
  const absl::StatusOr<matrix_t> acom = acomWeightsFromConfig(task.acom_weights, "acom_weights");
  if (!com.ok() || !acom.ok()) {
    target.reportNotApplied("the CoM + ACoM tracking weights", com.ok() ? acom.status() : com.status());
    return;
  }
  const std::string terminalTermName(ComAndAcomTrackingCost::kTerminalTermName);
  for (OptimalControlProblem& ocp : target.problems()) {
    absl::Status status = setComAndAcomWeights(*ocp.stateCostPtr, runningTermName, *com, *acom);
    if (status.ok() && terminalCostScaling.has_value() && carriesTerm(*ocp.finalCostPtr, terminalTermName)) {
      status = setComAndAcomWeights(*ocp.finalCostPtr, terminalTermName, *com * *terminalCostScaling, *acom * *terminalCostScaling);
    }
    if (!status.ok()) {
      LOG(WARNING) << "[MpcParameterUpdaterModule] Failed to update the CoM + ACoM tracking cost: " << status.message();
    }
  }
}

}  // namespace ocs2::humanoid
