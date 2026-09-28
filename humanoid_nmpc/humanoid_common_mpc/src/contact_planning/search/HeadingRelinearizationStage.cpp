/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include "humanoid_common_mpc/contact_planning/search/HeadingRelinearizationStage.h"

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {

std::string HeadingRelinearizationStage::describe() const {
  return absl::StrCat("re-linearize the foothold frame at the incumbent's heading and re-solve with the contacts fixed, ", passes_,
                      " pass(es)");
}

void HeadingRelinearizationStage::configure(const ContactPlanningConfig& config) {
  passes_ = config.headingRelinearization.passes;
  // The plan's own time budget: the branch-and-bound's plus the local search's.
  timeBudget_ = config.planner.maxSolveTime + config.eventShiftLocalSearch.maxTime;
}

void HeadingRelinearizationStage::afterSearch(SearchRun& run) const {
  MiqpResult& result = *run.result;
  if (!run.layout->hasHeading || !result.hasIncumbent || passes_ <= 0) return;
  for (int pass = 0; pass < passes_; ++pass) {
    if (run.elapsedSeconds() > timeBudget_) break;
    try {
      OcpQpProblem relinearized = run.assembleWithNominal(nominalFromSolution(*run.layout, result.solution.x));
      OcpQpSolution solution;
      scalar_t objective = 0.0;
      ++run.statistics->numHeadingRelinearizations;
      if (!run.miqp->solveFixed(relinearized, *run.binaries, result.assignment, *run.propagate, *run.assignmentCost, solution, objective)) {
        break;
      }
      run.statistics->totalQpIterations += solution.iterations;
      *run.problem = std::move(relinearized);
      result.solution = std::move(solution);
      result.incumbentObjective = objective;
    } catch (const std::exception& e) {
      LOG(ERROR) << "[LipContactPlanner] heading re-linearization failure: " << e.what();
      break;
    }
  }
}

}  // namespace ocs2::humanoid
