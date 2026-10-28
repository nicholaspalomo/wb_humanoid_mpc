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

#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantContinuation.h"

#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/planner/contact_invariant/AugmentedReferenceManager.h"

namespace ocs2::humanoid::multi_contact {
namespace {

/** The warm start of a solve: `guess` as a primal solution. */
PrimalSolution primalSolutionOf(const GuessTrajectory& guess) {
  PrimalSolution solution;
  solution.timeTrajectory_ = guess.times;
  solution.stateTrajectory_ = guess.states;
  solution.inputTrajectory_ = guess.inputs;
  return solution;
}

/** The trajectory of `solution`, with an input at every time. */
GuessTrajectory trajectoryOf(const PrimalSolution& solution) {
  GuessTrajectory trajectory;
  trajectory.times = solution.timeTrajectory_;
  trajectory.states = solution.stateTrajectory_;
  trajectory.inputs = solution.inputTrajectory_;
  while (trajectory.inputs.size() < trajectory.times.size() &&
         !trajectory.inputs.empty()) {
    trajectory.inputs.push_back(trajectory.inputs.back());
  }
  return trajectory;
}

/** One stage: the problem of its weights, solved from `guess`. */
absl::StatusOr<StageSolution> solveStage(const ContinuationProblem& problem,
                                         const GuessTrajectory& guess,
                                         const ContinuationStage& stage) {
  ASSIGN_OR_RETURN(const OptimalControlProblem ocp,
                   augmentProblem(*problem.physical, problem.layout,
                                  *problem.features, problem.augmentedManifold,
                                  stage.weights, problem.barriers));
  sqp::Settings settings = problem.settings;
  settings.sqpIteration = static_cast<size_t>(stage.sqpIterations);
  const TrajectoryInitializer initializer(guess, problem.augmentedManifold);
  const absl::Time started = absl::Now();
  StageSolution result;
  result.stage = stage.name;
  // OCS2's SQP throws when a QP has no solution; converted at once.
  try {  // NOLINT(exceptions): OCS2's solver throws.
    SqpSolver solver(settings, ocp, initializer);
    solver.setReferenceManager(std::make_shared<AugmentedReferenceManager>(
        problem.physicalReferenceManager, problem.layout.physicalStateDim));
    solver.run(problem.initTime, guess.states.front(), problem.initMode,
               problem.finalTime, primalSolutionOf(guess));
    result.trajectory = trajectoryOf(solver.primalSolution(problem.finalTime));
    result.performance = solver.getPerformanceIndeces();
    result.iterations = static_cast<int>(solver.getNumIterations());
  } catch (const std::exception& error) {  // NOLINT(exceptions): see above.
    return absl::InternalError(absl::StrCat("contact-invariant stage '",
                                            stage.name, "': ", error.what()));
  }
  result.seconds = absl::ToDoubleSeconds(absl::Now() - started);
  LOG(INFO) << absl::StrFormat(
      "contact_invariant: stage '%s': %d iterations in %.1f s, cost %.4g, "
      "dynamics defect %.3g, equality defect %.3g",
      stage.name, result.iterations, result.seconds, result.performance.cost,
      result.performance.dynamicsViolationSSE,
      result.performance.equalityConstraintsSSE);
  return result;
}

}  // namespace

absl::StatusOr<std::vector<StageSolution>> solveContinuation(
    const ContinuationProblem& problem,
    const GuessTrajectory& guess,
    const std::vector<ContinuationStage>& stages) {
  std::vector<StageSolution> solutions;
  GuessTrajectory current = guess;
  for (const ContinuationStage& stage : stages) {
    ASSIGN_OR_RETURN(StageSolution solution,
                     solveStage(problem, current, stage));
    current = solution.trajectory;
    solutions.push_back(std::move(solution));
  }
  return solutions;
}

}  // namespace ocs2::humanoid::multi_contact
