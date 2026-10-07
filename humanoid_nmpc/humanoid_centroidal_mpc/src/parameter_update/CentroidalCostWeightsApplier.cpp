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

#include "humanoid_centroidal_mpc/parameter_update/CentroidalCostWeightsApplier.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_centroidal_mpc/config/costs/DcmTerminalCostFromConfig.h"
#include "humanoid_centroidal_mpc/config/costs/IcpCostFromConfig.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(centroidal_cost_weights_fields)
constexpr std::array<absl::string_view, 3> kFields = {
    "task_space_foot_cost",
    "icp_cost_weights",
    "dcm_terminal_cost",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto)

/** Whether `ocp` carries the foot cost of any foot of `contactNames`. */
bool carriesFootCost(const OptimalControlProblem& ocp, const std::vector<std::string>& contactNames) {
  for (const std::string& footName : contactNames) {
    if (carriesTerm(*ocp.costPtr, taskSpaceKinematicsCostName(footName))) return true;
  }
  return false;
}

}  // namespace

absl::Span<const absl::string_view> CentroidalCostWeightsApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

void CentroidalCostWeightsApplier::apply(HotUpdateTarget& target) {
  const mpc_config::TaskFile& task = target.task();
  const OptimalControlProblem& running = target.runningProblem();
  std::optional<TaskSpaceFootCostSettings> footCost;
  if (carriesFootCost(running, target.contactNames())) {
    footCost = convertedOrReported(taskSpaceFootCostFromConfig(task.task_space_foot_cost), target.source(), "task_space_foot_cost");
  }
  std::optional<vector2_t> icpWeights;
  if (carriesTerm(*running.costPtr, kIcpCostTerm)) {
    icpWeights = convertedOrReported(icpCostWeightsFromConfig(task.icp_cost_weights), target.source(), "icp_cost_weights");
  }
  std::optional<DcmTerminalCost::Config> dcmTerminalCost;
  if (carriesTerm(*running.finalCostPtr, DcmTerminalCost::kTermName)) {
    dcmTerminalCost = convertedOrReported(dcmTerminalCostConfigFromConfig(task.dcm_terminal_cost), target.source(), "dcm_terminal_cost");
  }

  bool dcmRefusalLogged = false;
  for (OptimalControlProblem& ocp : target.problems()) {
    if (footCost.has_value()) {
      const vector12_t weights = footCost->weights.toVector();
      for (const std::string& footName : target.contactNames()) {
        updateTermIfPresent<CentroidalMpcEndEffectorFootCost>(*ocp.costPtr, taskSpaceKinematicsCostName(footName),
                                                              [&](CentroidalMpcEndEffectorFootCost& cost) {
                                                                cost.setWeights(weights);
                                                                cost.setActiveInStance(footCost->activeInStance);
                                                              });
      }
    }
    if (icpWeights.has_value()) {
      updateTermIfPresent<ICPCost>(*ocp.costPtr, kIcpCostTerm, [&](ICPCost& cost) { cost.setWeights(*icpWeights); });
    }
    if (dcmTerminalCost.has_value()) {
      absl::Status applied = absl::OkStatus();
      updateTermIfPresent<DcmTerminalCost>(*ocp.finalCostPtr, DcmTerminalCost::kTermName,
                                           [&](DcmTerminalCost& cost) { applied = cost.setConfig(*dcmTerminalCost); });
      if (!applied.ok() && !dcmRefusalLogged) {
        LOG(WARNING) << "[MpcParameterUpdaterModule] dcm_terminal_cost not applied, the running cost is kept: " << applied.message();
        dcmRefusalLogged = true;
      }
    }
  }
}

}  // namespace ocs2::humanoid
