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

#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"

#include "absl/log/log.h"

namespace ocs2::humanoid {

std::string EventShiftLocalSearchStage::describe() const {
  std::ostringstream out;
  out << "shift every contact event of the incumbent by one node while it improves, " << params_.iterations << " rounds, "
      << params_.maxTime << " s";
  return out.str();
}

void EventShiftLocalSearchStage::configure(const ContactPlanningConfig& config) {
  if (config.eventShiftLocalSearch.iterations < 0 || config.eventShiftLocalSearch.maxTime < 0.0) {
    throw std::invalid_argument("[event_shift_local_search] invalid local search limits");
  }
  params_ = config.eventShiftLocalSearch;
}

void EventShiftLocalSearchStage::afterSearch(SearchRun& run) const {
  const auto start = SearchRun::Clock::now();
  const auto elapsed = [&]() { return std::chrono::duration<scalar_t>(SearchRun::Clock::now() - start).count(); };
  MiqpResult& result = *run.result;
  SearchStatistics& statistics = *run.statistics;
  const scalar_t timeBudget = params_.maxTime;
  if (!result.hasIncumbent || params_.iterations <= 0 || timeBudget <= 0.0) return;
  const int N = run.config->planner.numNodes;
  const MiqpAssignment& initial = *run.initialAssignment;

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
