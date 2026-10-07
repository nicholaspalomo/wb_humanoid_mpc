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

#include "humanoid_common_mpc/parameter_update/TrackingCostWeightsApplier.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(tracking_cost_weights_fields)
constexpr std::array<absl::string_view, 3> kFields = {
    "task_space_costs",
    "left_leg_torque_cost",
    "right_leg_torque_cost",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto)

/** The weights of a named term: a task-space link cost or a leg's torque cost. */
struct TermWeights {
  std::string term;
  vector_t weights;
};

/**
 * The terms of the link costs the running problem carries, sorted: its task-space kinematics terms that are not a foot's
 * (the centroidal foot cost shares the name pattern).
 */
std::vector<std::string> carriedLinkCostTerms(const HotUpdateTarget& target) {
  std::vector<std::string> footTerms;
  for (const std::string& footName : target.contactNames()) footTerms.push_back(taskSpaceKinematicsCostName(footName));
  const std::string suffix = taskSpaceKinematicsCostName(/*name=*/"");
  std::vector<std::string> terms;
  for (const std::pair<const std::string, size_t>& term : target.runningProblem().costPtr->getTermNameMap()) {
    if (!absl::EndsWith(term.first, suffix)) continue;
    if (std::find(footTerms.begin(), footTerms.end(), term.first) != footTerms.end()) continue;
    terms.push_back(term.first);
  }
  // The collection's map is unordered; the reports follow the names.
  std::sort(terms.begin(), terms.end());
  return terms;
}

bool namesTerm(const std::vector<TermWeights>& entries, const std::string& term) {
  return std::any_of(entries.begin(), entries.end(), [&term](const TermWeights& entry) { return entry.term == term; });
}

/**
 * Reports what of the reloaded task_space_costs `reloaded` a reload cannot apply, because it would assemble another
 * problem: an entry whose link cost the running problem does not carry (added or renamed), and a link cost the running
 * problem carries that no entry names any more (removed or renamed). Both take effect at the next start, as the start-up
 * report says too (task_space_costs[i].name). A problem assembled without link costs (its costs do not list
 * task_space_torso_cost) has nothing to compare: the entries are read by no term either way.
 */
void reportStructuralChanges(const HotUpdateTarget& target, const std::vector<TermWeights>& reloaded) {
  const std::vector<std::string> carried = carriedLinkCostTerms(target);
  if (carried.empty()) return;
  for (const TermWeights& entry : reloaded) {
    if (std::find(carried.begin(), carried.end(), entry.term) != carried.end()) continue;
    target.reportNotApplied(absl::StrCat("the task_space_costs entry of ", entry.term),
                            absl::FailedPreconditionError("the running problem carries no such link cost; an added or renamed entry "
                                                          "takes effect at the next start"));
  }
  for (const std::string& term : carried) {
    if (namesTerm(reloaded, term)) continue;
    target.reportNotApplied(absl::StrCat("the removal of the link cost ", term),
                            absl::FailedPreconditionError("the reloaded task_space_costs names it no more; a removed or renamed entry "
                                                          "takes effect at the next start"));
  }
}

/**
 * The weights of task_space_costs, by term, with what of them a reload cannot apply reported (reportStructuralChanges());
 * empty for an empty block, and for a refused one, whose refusal is reported.
 */
std::vector<TermWeights> linkCostWeights(const HotUpdateTarget& target) {
  std::vector<TermWeights> weights;
  if (target.task().task_space_costs.empty()) {
    reportStructuralChanges(target, weights);
    return weights;
  }
  const std::optional<std::vector<TaskSpaceLinkCostSettings>> linkCosts =
      convertedOrReported(taskSpaceLinkCostsFromConfig(target.task().task_space_costs), target.source(), "task_space_costs");
  if (!linkCosts.has_value()) return weights;
  for (const TaskSpaceLinkCostSettings& linkCost : *linkCosts) {
    weights.push_back(TermWeights{.term = taskSpaceKinematicsCostName(linkCost.name), .weights = linkCost.weights.toVector()});
  }
  reportStructuralChanges(target, weights);
  return weights;
}

/** The weights of the leg torque costs the running problem carries: the left leg's for the first contact, the right's for the second. */
std::vector<TermWeights> legTorqueWeights(const HotUpdateTarget& target) {
  std::vector<TermWeights> weights;
  const std::vector<std::string>& contactNames = target.contactNames();
  for (size_t i = 0; i < contactNames.size(); ++i) {
    const std::string term = externalTorqueCostName(contactNames[i]);
    if (!carriesTerm(*target.runningProblem().costPtr, term)) continue;
    const bool left = i == 0;
    const absl::string_view field = left ? "left_leg_torque_cost" : "right_leg_torque_cost";
    if (const std::optional<ExternalTorqueQuadraticCostAD::Config> config =
            convertedOrReported(legTorqueCostFromConfig(left ? target.task().left_leg_torque_cost : target.task().right_leg_torque_cost,
                                                        target.layout(), field),
                                target.source(), field)) {
      weights.push_back(TermWeights{.term = term, .weights = config->weights});
    }
  }
  return weights;
}

}  // namespace

absl::Span<const absl::string_view> TrackingCostWeightsApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

void TrackingCostWeightsApplier::apply(HotUpdateTarget& target) {
  const std::vector<TermWeights> linkCosts = linkCostWeights(target);
  const std::vector<TermWeights> legTorques = legTorqueWeights(target);
  for (OptimalControlProblem& ocp : target.problems()) {
    for (const TermWeights& update : linkCosts) {
      updateTermIfPresent<EndEffectorKinematicsQuadraticCost>(
          *ocp.costPtr, update.term, [&](EndEffectorKinematicsQuadraticCost& cost) { cost.setWeights(vector12_t(update.weights)); });
    }
    for (const TermWeights& update : legTorques) {
      updateTermIfPresent<ExternalTorqueQuadraticCostAD>(*ocp.costPtr, update.term,
                                                         [&](ExternalTorqueQuadraticCostAD& cost) { cost.setWeights(update.weights); });
    }
  }
}

}  // namespace ocs2::humanoid
