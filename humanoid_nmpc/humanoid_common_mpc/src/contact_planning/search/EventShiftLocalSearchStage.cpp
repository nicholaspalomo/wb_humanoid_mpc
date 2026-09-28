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

#include "humanoid_common_mpc/contact_planning/search/EventShiftLocalSearchStage.h"

#include <chrono>
#include <functional>
#include <set>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"
#include "humanoid_common_mpc/contact_planning/search/CadenceStretchStage.h"

namespace ocs2::humanoid {

std::string EventShiftLocalSearchStage::describe() const {
  return absl::StrCat("shift every contact event of the incumbent by one node while it improves, ", params_.iterations, " rounds, ",
                      params_.maxTime, " s");
}

void EventShiftLocalSearchStage::configure(const ContactPlanningConfig& config) {
  params_ = config.eventShiftLocalSearch;
}

void EventShiftLocalSearchStage::afterSearch(SearchRun& run) const {
  const SearchRun::Clock::time_point start = SearchRun::Clock::now();
  const std::function<scalar_t()> elapsed = [&start]() { return std::chrono::duration<scalar_t>(SearchRun::Clock::now() - start).count(); };
  MiqpResult& result = *run.result;
  SearchStatistics& statistics = *run.statistics;
  const scalar_t timeBudget = params_.maxTime;
  if (!result.hasIncumbent || params_.iterations <= 0 || timeBudget <= 0.0) return;
  const int N = run.config->planner.numNodes;
  const MiqpAssignment& initial = *run.initialAssignment;
  // The grid the incumbent is on: planner.dt unless a stage listed before this one re-timed it (cadence_stretch).
  const scalar_t stretch = run.chosenDt > 0.0 ? run.chosenDt / run.config->planner.dt : 1.0;

  std::set<MiqpAssignment> evaluated;
  evaluated.insert(result.assignment);
  for (int round = 0; round < params_.iterations; ++round) {
    if (elapsed() > timeBudget) break;
    const MiqpAssignment base = result.assignment;
    bool improved = false;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      for (int k = 1; k < N; ++k) {
        const size_t index = static_cast<size_t>(ContactLogicState::contactBinaryIndex(k, foot));
        const size_t previousIndex = static_cast<size_t>(ContactLogicState::contactBinaryIndex(k - 1, foot));
        if (base[index] == base[previousIndex]) continue;  // no transition at k
        // Move the transition one node earlier (the previous node takes the new value) or later (this node keeps the old).
        for (const int shift : {-1, +1}) {
          MiqpAssignment candidate = base;
          if (shift < 0) {
            if (initial[previousIndex] != kMiqpFree) continue;
            candidate[previousIndex] = base[index];
          } else {
            if (initial[index] != kMiqpFree) continue;
            candidate[index] = base[previousIndex];
          }
          if (!(*run.propagate)(candidate)) continue;
          // The propagation checks the gait limits on the planner.dt grid, but after a cadence_stretch listed earlier
          // the problem being searched - and the plan to be emitted - is on the stretched grid, where a phase lasts
          // chosenDt per node. A candidate that moves a touch-down one node later is then legal in nodes and too long
          // in seconds (a five-node swing at dt 0.1 against a 0.5 s maximum passes propagation and executes for
          // 0.625 s at a stretch of 1.25), so the candidate has to admit the stretch the incumbent was emitted with.
          if (stretch > 1.0 && CadenceStretchStage::admissibleStretch(*run.config, candidate, N, stretch, run.input) < stretch - 1e-9) {
            continue;
          }
          if (!evaluated.insert(candidate).second) continue;
          if (elapsed() > timeBudget) break;
          OcpQpSolution solution;
          scalar_t objective = 0.0;
          ++statistics.numLocalSearchQps;
          if (!run.miqp->solveFixed(*run.problem, *run.binaries, candidate, *run.propagate, *run.assignmentCost, solution, objective))
            continue;
          statistics.totalQpIterations += solution.iterations;
          if (objective < result.incumbentObjective - 1e-6) {
            result.incumbentObjective = objective;
            result.solution = solution;
            result.assignment = candidate;
            improved = true;
            statistics.localSearchImproved = true;
            if (run.verbose) {
              LOG(INFO) << "[LipContactPlanner] local search improved the objective to " << objective;
            }
          }
        }
      }
    }
    if (!improved) break;
  }
  statistics.localSearchTime = elapsed();
}

}  // namespace ocs2::humanoid
