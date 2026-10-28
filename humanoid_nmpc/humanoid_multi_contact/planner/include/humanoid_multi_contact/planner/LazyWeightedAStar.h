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

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/time/time.h"

namespace ocs2::humanoid::multi_contact {

/** A successor of an expanded node before it is validated. Passive data. */
struct SearchCandidate {
  // The problem's id of the new node.
  int node = 0;
  // A lower estimate of the edge's cost, known without validating it.
  double estimatedCost = 0.0;
  // Orders candidates of equal priority, smaller first.
  double tieBreak = 0.0;
};

/**
 * A graph whose edges are expensive to check, for lazyWeightedAStar(): the
 * candidates of a node are cheap to list, and only the ones the search pops
 * are validated (lazy evaluation). The problem owns the nodes and their states;
 * the search keeps their costs.
 */
class LazySearchProblem {
 public:
  virtual ~LazySearchProblem() = default;

  /** The candidate successors of the validated node `node`. */
  virtual std::vector<SearchCandidate> candidates(int node) = 0;

  /** Validates the candidate `node`, whose parent the search has validated:
   * the edge's true cost, at least its estimate, or std::nullopt when the edge
   * is infeasible. */
  virtual absl::StatusOr<std::optional<double>> validate(int node) = 0;

  /** The estimated cost from `node` to a goal, not negative. */
  virtual double heuristic(int node) = 0;

  /** The state of `node` for duplicate detection: of the nodes with one key,
   * the first one popped validated is expanded and the others are dropped. */
  virtual std::string key(int node) = 0;

  /** Whether the validated node `node` is a goal; may do work, and is asked
   * once per node. */
  virtual absl::StatusOr<bool> isGoal(int node) = 0;
};

/** The defaults of LazySearchOptions; its fields say what each is. */
inline constexpr int kDefaultMaxValidations = 1000;

/** The settings of lazyWeightedAStar(). */
struct LazySearchOptions {
  // epsilon of f = g + epsilon h: 1 for A*, larger for a greedier search whose
  // path costs at most epsilon times the optimum with a consistent heuristic.
  double heuristicWeight = 1.0;
  // The most candidates validated before the search gives up.
  int maxValidations = kDefaultMaxValidations;
  // The longest the search runs; std::nullopt: no limit.
  std::optional<absl::Duration> timeBudget;
};

/** What lazyWeightedAStar() found. Passive data. */
struct LazySearchResult {
  bool found = false;
  // From the start node to the goal node, when found.
  std::vector<int> path;
  double cost = 0.0;
  int validations = 0;
  int expansions = 0;
  // Why the search stopped without a goal: "exhausted", "validations" or
  // "time".
  std::string stopReason;
};

/**
 * Weighted A* with lazy edge evaluation (Cohen, Phillips and Likhachev's
 * lazy weighted A*): a node's candidates enter the open list with their
 * estimated costs, and an edge is validated only when its candidate is popped;
 * a valid one re-enters with its true cost. The start node must be valid. The
 * errors of the problem's methods are returned as they are.
 */
absl::StatusOr<LazySearchResult> lazyWeightedAStar(
    LazySearchProblem& problem, int start, const LazySearchOptions& options);

}  // namespace ocs2::humanoid::multi_contact
