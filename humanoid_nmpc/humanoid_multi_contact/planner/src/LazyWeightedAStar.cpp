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

#include "humanoid_multi_contact/planner/LazyWeightedAStar.h"

#include <algorithm>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid::multi_contact {
namespace {

/** One entry of the open list. */
struct OpenEntry {
  double priority = 0.0;
  double tieBreak = 0.0;
  // Increasing with insertion: the earlier entry first among equals.
  int order = 0;
  int node = 0;
  // Whether the edge into `node` has been validated.
  bool validated = false;
};

/** The order of std::priority_queue: the entry with the smallest priority
 * first. */
struct LaterEntry {
  bool operator()(const OpenEntry& first, const OpenEntry& second) const {
    if (first.priority != second.priority) {
      return first.priority > second.priority;
    }
    if (first.tieBreak != second.tieBreak) {
      return first.tieBreak > second.tieBreak;
    }
    return first.order > second.order;
  }
};

/** The path from the start to `goal` through the parents. */
std::vector<int> pathTo(int goal,
                        const absl::flat_hash_map<int, int>& parents) {
  std::vector<int> path = {goal};
  for (absl::flat_hash_map<int, int>::const_iterator parent =
           parents.find(goal);
       parent != parents.end(); parent = parents.find(path.back())) {
    path.push_back(parent->second);
  }
  std::reverse(path.begin(), path.end());
  return path;
}

}  // namespace

absl::StatusOr<LazySearchResult> lazyWeightedAStar(
    LazySearchProblem& problem, int start, const LazySearchOptions& options) {
  if (!(options.heuristicWeight >= 1.0) || options.maxValidations < 0) {
    return absl::InvalidArgumentError(
        "the heuristic weight must be at least 1 and the validations not "
        "negative");
  }
  const absl::Time deadline = options.timeBudget.has_value()
                                  ? absl::Now() + *options.timeBudget
                                  : absl::InfiniteFuture();
  LazySearchResult result;
  absl::flat_hash_map<int, double> costs = {{start, 0.0}};
  absl::flat_hash_map<int, int> parents;
  absl::flat_hash_set<std::string> closed;
  std::priority_queue<OpenEntry, std::vector<OpenEntry>, LaterEntry> open;
  int order = 0;
  open.push(
      OpenEntry{.priority = options.heuristicWeight * problem.heuristic(start),
                .order = order++,
                .node = start,
                .validated = true});
  while (!open.empty()) {
    const OpenEntry entry = open.top();
    open.pop();
    if (!entry.validated) {
      if (closed.contains(problem.key(entry.node))) continue;
      if (result.validations >= options.maxValidations) {
        result.stopReason = "validations";
        return result;
      }
      if (absl::Now() > deadline) {
        result.stopReason = "time";
        return result;
      }
      ++result.validations;
      ASSIGN_OR_RETURN(const std::optional<double> edgeCost,
                       problem.validate(entry.node));
      if (!edgeCost.has_value()) continue;
      const double cost = costs[parents[entry.node]] + *edgeCost;
      costs[entry.node] = cost;
      open.push(OpenEntry{.priority = cost + options.heuristicWeight *
                                                 problem.heuristic(entry.node),
                          .tieBreak = entry.tieBreak,
                          .order = order++,
                          .node = entry.node,
                          .validated = true});
      continue;
    }
    if (!closed.insert(problem.key(entry.node)).second) continue;
    ASSIGN_OR_RETURN(const bool goal, problem.isGoal(entry.node));
    if (goal) {
      result.found = true;
      result.path = pathTo(entry.node, parents);
      result.cost = costs[entry.node];
      return result;
    }
    ++result.expansions;
    const double cost = costs[entry.node];
    for (const SearchCandidate& candidate : problem.candidates(entry.node)) {
      parents[candidate.node] = entry.node;
      open.push(OpenEntry{.priority = cost + candidate.estimatedCost +
                                      options.heuristicWeight *
                                          problem.heuristic(candidate.node),
                          .tieBreak = candidate.tieBreak,
                          .order = order++,
                          .node = candidate.node,
                          .validated = false});
    }
  }
  result.stopReason = "exhausted";
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
