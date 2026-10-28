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

#include "humanoid_multi_contact/planner/StanceGraphThenContactInvariantPlanner.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/planner/contact_invariant/HardContactRefinement.h"
#include "humanoid_multi_contact/planner/contact_invariant/PlanSeeding.h"
#include "humanoid_multi_contact/planner/contact_invariant/PlanTrajectory.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"

namespace ocs2::humanoid::multi_contact {

StanceGraphThenContactInvariantOptions::
    StanceGraphThenContactInvariantOptions() {
  // The stairs' seed: the certifier's tolerances, which solve 3 re-solves.
  stanceGraph.contactBreak = ContactBreak::kAtSecondKnot;
  stanceGraph.staticMarginBuffer = kSeedStaticMarginBuffer;
  stanceGraph.ik.penetrationTolerance = kDefaultReachTolerance;
  stanceGraph.motion.sphereTolerance = kDefaultContactDistanceTolerance;
  // Solve 3 re-solves the motion between the knots too.
  stanceGraph.motion.sweep = std::nullopt;
  stanceGraph.motion.pointTouchdownPostureLead = 1.0;
  stanceGraph.retimingScope = RetimingScope::kWholePlan;
}

StanceGraphThenContactInvariantPlanner::StanceGraphThenContactInvariantPlanner(
    const StanceGraphThenContactInvariantOptions& options,
    std::unique_ptr<StanceGraphPlanner> stanceGraph)
    : options_(options), stanceGraph_(std::move(stanceGraph)) {}

absl::StatusOr<std::unique_ptr<StanceGraphThenContactInvariantPlanner>>
StanceGraphThenContactInvariantPlanner::Create(
    const StanceGraphThenContactInvariantOptions& options) {
  if (!(options.minimumStance > 0.0)) {
    return absl::InvalidArgumentError(
        "stance_graph_then_contact_invariant: the minimum stance must be "
        "positive");
  }
  // Planner A's options are checked by its own factory.
  RETURN_IF_ERROR(
      ContactInvariantPlanner::Create(options.contactInvariant).status());
  ASSIGN_OR_RETURN(std::unique_ptr<StanceGraphPlanner> stanceGraph,
                   StanceGraphPlanner::Create(options.stanceGraph));
  return absl::WrapUnique(new StanceGraphThenContactInvariantPlanner(
      options, std::move(stanceGraph)));
}

absl::StatusOr<config::MultiContactPlan>
StanceGraphThenContactInvariantPlanner::plan(const LoadedScenario& loaded) {
  const absl::Time started = absl::Now();
  absl::StatusOr<config::MultiContactPlan> sequence =
      stanceGraph_->plan(loaded);
  if (!sequence.ok()) {
    return absl::Status(
        sequence.status().code(),
        absl::StrCat(kStanceGraphThenContactInvariantPlannerName,
                     ": planner B found no stance sequence to initialize "
                     "planner A with: ",
                     sequence.status().message()));
  }
  const double duration = loaded.scenario.duration;
  LOG(INFO) << absl::StrFormat(
      "%s: planner B's sequence for '%s': %d knots over %.1f s, retimed to "
      "%.1f s",
      kStanceGraphThenContactInvariantPlannerName, loaded.scenario.name,
      static_cast<int>(sequence->knots.size()), sequence->knots.back().time,
      duration);
  ASSIGN_OR_RETURN(
      const config::MultiContactPlan retimed,
      retimedPlan(*sequence, duration, options_.minimumStance,
                  options_.contactInvariant.thresholding.changeQuantum));
  ASSIGN_OR_RETURN(const CentroidalMpcConfig config,
                   plannerMpcConfigOf(loaded));
  ASSIGN_OR_RETURN(
      std::unique_ptr<CentroidalMpcInterface> models,
      CentroidalMpcInterface::CreateControllerModels(config, loaded.urdfPath));
  ASSIGN_OR_RETURN(
      std::unique_ptr<RobotStanceModel> stanceModel,
      RobotStanceModel::Create(loaded.urdfPath, loaded.contactSet));
  const RefinementOptions& refinementOptions =
      options_.contactInvariant.refinement;
  const std::vector<std::string>& contactNames =
      models->modelSettings().contactNames;
  const double mass = models->getCentroidalModelInfo().robotMass;
  ASSIGN_OR_RETURN(
      const MpcTrajectory guess,
      mpcTrajectoryOfPlan(retimed, models->getMpcRobotModel(), contactNames,
                          loaded.contactSet, mass,
                          knotGrid(duration, refinementOptions.knotSpacing)));
  ASSIGN_OR_RETURN(
      const config::MultiContactPlan seed,
      planOfTrajectory(
          guess,
          PlanConversion{.scenario = &loaded,
                         .model = &models->getMpcRobotModel(),
                         .contactNames = contactNames,
                         .mass = mass,
                         .stanceModel = stanceModel.get(),
                         .planner = std::string(
                             kStanceGraphThenContactInvariantPlannerName)}));
  ASSIGN_OR_RETURN(std::unique_ptr<HardContactRefinement> refinement,
                   HardContactRefinement::Create(
                       loaded, config, loaded.urdfPath, refinementOptions));
  ASSIGN_OR_RETURN(config::MultiContactPlan plan,
                   refinedPlan(*refinement, seed, guess, loaded,
                               kStanceGraphThenContactInvariantPlannerName));
  LOG(INFO) << absl::StrFormat(
      "%s: '%s' planned in %.1f s: %d knots, %.1f s of motion",
      kStanceGraphThenContactInvariantPlannerName, loaded.scenario.name,
      absl::ToDoubleSeconds(absl::Now() - started),
      static_cast<int>(plan.knots.size()), plan.knots.back().time);
  return plan;
}

}  // namespace ocs2::humanoid::multi_contact
