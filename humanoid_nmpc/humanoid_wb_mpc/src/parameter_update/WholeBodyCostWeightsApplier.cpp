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

#include "humanoid_wb_mpc/parameter_update/WholeBodyCostWeightsApplier.h"

#include <array>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"
#include "humanoid_wb_mpc/config/costs/EndEffectorDynamicsWeightsFromConfig.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsCostHelpers.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsFootCost.h"
#include "humanoid_wb_mpc/cost/JointTorqueCostCppAd.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(whole_body_cost_weights_fields)
constexpr std::array<absl::string_view, 2> kFields = {
    "task_space_foot_cost",
    "joint_torque_weights",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto)

constexpr char kFootCostField[] = "task_space_foot_cost";
constexpr char kJointTorqueField[] = "joint_torque_weights";

}  // namespace

absl::Span<const absl::string_view> WholeBodyCostWeightsApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

void WholeBodyCostWeightsApplier::apply(HotUpdateTarget& target) {
  applyFootCostWeights(target);
  applyJointTorqueWeights(target);
}

void WholeBodyCostWeightsApplier::applyFootCostWeights(HotUpdateTarget& target) {
  bool carried = false;
  for (const std::string& footName : target.contactNames()) {
    carried = carried || carriesTerm(*target.runningProblem().costPtr, EndEffectorDynamicsFootCost::termName(footName));
  }
  if (!carried) return;
  const absl::StatusOr<EndEffectorDynamicsWeights> weights = wholeBodyFootCostWeightsFromConfig(target.task().task_space_foot_cost);
  if (!weights.ok()) {
    target.reportNotApplied(kFootCostField, weights.status());
    return;
  }
  for (OptimalControlProblem& problem : target.problems()) {
    for (const std::string& footName : target.contactNames()) {
      updateTermIfPresent<EndEffectorDynamicsFootCost>(*problem.costPtr, EndEffectorDynamicsFootCost::termName(footName),
                                                       [&weights](EndEffectorDynamicsFootCost& cost) { cost.setWeights(*weights); });
    }
  }
}

void WholeBodyCostWeightsApplier::applyJointTorqueWeights(HotUpdateTarget& target) {
  if (!carriesTerm(*target.runningProblem().costPtr, JointTorqueCostCppAd::kTermName)) return;
  const absl::StatusOr<vector_t> weights =
      jointTorqueWeightsFromConfig(target.task().joint_torque_weights, target.layout(), kJointTorqueField);
  if (!weights.ok()) {
    target.reportNotApplied(kJointTorqueField, weights.status());
    return;
  }
  for (OptimalControlProblem& problem : target.problems()) {
    absl::Status written;
    updateTermIfPresent<JointTorqueCostCppAd>(*problem.costPtr, JointTorqueCostCppAd::kTermName,
                                              [&weights, &written](JointTorqueCostCppAd& cost) { written = cost.setWeights(*weights); });
    // Every worker's cost has the size of the first: a refusal refuses them all, before any was written.
    if (!written.ok()) {
      target.reportNotApplied(kJointTorqueField, written);
      return;
    }
  }
}

}  // namespace ocs2::humanoid
