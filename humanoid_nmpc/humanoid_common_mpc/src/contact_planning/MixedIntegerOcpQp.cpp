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

#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

namespace {

// A relaxed binary at or above this value rounds to 1, below it to 0.
constexpr scalar_t kRoundUpThreshold = 0.5;

struct Node {
  MiqpAssignment assignment;
  scalar_t parentBound;
};

using Clock = std::chrono::steady_clock;

scalar_t elapsedSeconds(const Clock::time_point& start) {
  return std::chrono::duration<scalar_t>(Clock::now() - start).count();
}

bool isComplete(const MiqpAssignment& assignment) {
  return std::none_of(assignment.begin(), assignment.end(),
                      [](int8_t v) { return v == kMiqpFree; });
}

}  // namespace

MixedIntegerOcpQp::MixedIntegerOcpQp(OcpQpHpipmSolver::Settings qpSettings,
                                     MiqpSettings settings)
    : qpSolver_(qpSettings), settings_(settings) {}

absl::StatusOr<std::vector<MixedIntegerOcpQp::BinaryLocation>>
MixedIntegerOcpQp::locateBinaries(
    const OcpQpProblem& problem,
    const std::vector<MiqpBinaryVariable>& binaries) const {
  std::vector<BinaryLocation> locations;
  locations.reserve(binaries.size());
  for (const MiqpBinaryVariable& binary : binaries) {
    if (binary.stage < 0 || binary.stage >= problem.numStages()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[MixedIntegerOcpQp] binary variable stage out of range: ",
          binary.stage));
    }
    const OcpQpStage& stage = problem.stages[binary.stage];
    const std::vector<int>::const_iterator it =
        std::find(stage.idxbu.begin(), stage.idxbu.end(), binary.inputIndex);
    if (it == stage.idxbu.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("[MixedIntegerOcpQp] binary input ", binary.inputIndex,
                       " of stage ", binary.stage, " has no box constraint"));
    }
    locations.push_back(
        {binary.stage, static_cast<int>(it - stage.idxbu.begin())});
  }
  return locations;
}

void MixedIntegerOcpQp::applyAssignment(
    OcpQpProblem& problem,
    const std::vector<BinaryLocation>& locations,
    const MiqpAssignment& assignment) const {
  for (size_t i = 0; i < locations.size(); ++i) {
    OcpQpStage& stage = problem.stages[locations[i].stage];
    const int boxIndex = locations[i].boxIndex;
    if (assignment[i] == kMiqpFree) {
      stage.lbu(boxIndex) = 0.0;
      stage.ubu(boxIndex) = 1.0;
    } else {
      const scalar_t value = static_cast<scalar_t>(assignment[i]);
      stage.lbu(boxIndex) = value;
      stage.ubu(boxIndex) = value;
    }
  }
}

int MixedIntegerOcpQp::firstFractional(
    const OcpQpSolution& solution,
    const std::vector<MiqpBinaryVariable>& binaries,
    const MiqpAssignment& assignment,
    scalar_t& fractionalValue) const {
  for (size_t i = 0; i < binaries.size(); ++i) {
    if (assignment[i] != kMiqpFree) {
      continue;
    }
    const scalar_t value =
        solution.u[binaries[i].stage](binaries[i].inputIndex);
    if (std::abs(value - std::round(value)) > settings_.integralityTol) {
      fractionalValue = value;
      return static_cast<int>(i);
    }
  }
  return -1;
}

absl::StatusOr<bool> MixedIntegerOcpQp::solveFixed(
    OcpQpProblem& problem,
    const std::vector<MiqpBinaryVariable>& binaries,
    MiqpAssignment assignment,
    const MiqpPropagateFn& propagate,
    const MiqpAssignmentCostFn& assignmentCost,
    OcpQpSolution& solution,
    scalar_t& objective) {
  objective = std::numeric_limits<scalar_t>::infinity();
  if (assignment.size() != binaries.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[MixedIntegerOcpQp] the assignment has ", assignment.size(),
        " entries for ", binaries.size(), " binaries"));
  }
  if (propagate && !propagate(assignment)) {
    return false;
  }
  if (!isComplete(assignment)) {
    return false;
  }
  ASSIGN_OR_RETURN(const std::vector<BinaryLocation> locations,
                   locateBinaries(problem, binaries));
  applyAssignment(problem, locations, assignment);
  ASSIGN_OR_RETURN(solution, qpSolver_.solve(problem));
  if (!solution.success()) {
    return false;
  }
  objective =
      solution.objective + (assignmentCost ? assignmentCost(assignment) : 0.0);
  return true;
}

/**
 * One branch-and-bound search of MixedIntegerOcpQp::solve(): the problem, its
 * binaries and hooks, the result found so far and the first error of the QP
 * solver. It lives for one solve() and refers to the solve()'s arguments, which
 * outlive it. Not thread-safe.
 */
class MixedIntegerOcpQp::Search {
 public:
  Search(MixedIntegerOcpQp* absl_nonnull owner,
         OcpQpProblem* absl_nonnull problem,
         const std::vector<MiqpBinaryVariable>* absl_nonnull binaries,
         std::vector<BinaryLocation> locations,
         const MiqpPropagateFn* absl_nonnull propagate,
         const MiqpAssignmentCostFn* absl_nonnull assignmentCost)
      : owner_(owner),
        problem_(problem),
        binaries_(binaries),
        locations_(std::move(locations)),
        propagate_(propagate),
        assignmentCost_(assignmentCost),
        startTime_(Clock::now()) {
    result_.incumbentObjective = std::numeric_limits<scalar_t>::infinity();
  }

  Search(const Search&) = delete;
  Search& operator=(const Search&) = delete;
  ~Search() = default;

  /** Runs the whole search from `initialAssignment` and returns its result, or
   * the first error of the QP solver. */
  absl::StatusOr<MiqpResult> run(
      const MiqpAssignment& initialAssignment,
      const MiqpAssignment* absl_nullable warmStart) {
    // Root assignment.
    MiqpAssignment root = initialAssignment;
    if (!runPropagation(root)) {
      result_.solveTime = elapsedSeconds(startTime_);
      result_.optimal = true;  // provably infeasible
      return result_;
    }
    tryWarmStart(root, warmStart);
    const bool exhausted = searchBestFirst(std::move(root));

    RETURN_IF_ERROR(qpError_);
    result_.optimal = exhausted && !result_.nodeLimitHit &&
                      !result_.timeLimitHit &&
                      result_.numFailedRelaxations == 0;
    result_.solveTime = elapsedSeconds(startTime_);
    if (result_.hasIncumbent) {
      // Leave the problem's binary bounds at the incumbent so that the caller
      // can inspect it.
      owner_->applyAssignment(*problem_, locations_, result_.assignment);
    }
    return result_;
  }

 private:
  /** Orders the open nodes of the best-first search: the node with the lowest
   * parent bound is on top. */
  struct WorseBound {
    bool operator()(const Node& a, const Node& b) const {
      return a.parentBound > b.parentBound;
    }
  };
  using OpenNodes = std::priority_queue<Node, std::vector<Node>, WorseBound>;

  bool runPropagation(MiqpAssignment& assignment) const {
    return !*propagate_ || (*propagate_)(assignment);
  }
  scalar_t logicalCost(const MiqpAssignment& assignment) const {
    return *assignmentCost_ ? (*assignmentCost_)(assignment) : 0.0;
  }

  /** Whether the search has to stop: a QP error, or the node or time limit
   * (which it records). */
  bool limitsHit() {
    if (!qpError_.ok()) return true;
    if (result_.numNodes >= owner_->settings_.maxNodes) {
      result_.nodeLimitHit = true;
      return true;
    }
    if (elapsedSeconds(startTime_) > owner_->settings_.maxSolveTime) {
      result_.timeLimitHit = true;
      return true;
    }
    return false;
  }

  /** Solves the relaxation of an assignment into `solution`. Returns false if
   * the QP failed (treated as infeasible). */
  bool solveRelaxation(const MiqpAssignment& assignment,
                       OcpQpSolution& solution) {
    owner_->applyAssignment(*problem_, locations_, assignment);
    absl::StatusOr<OcpQpSolution> solved = owner_->qpSolver_.solve(*problem_);
    if (!solved.ok()) {
      qpError_ = solved.status();
      return false;
    }
    solution = *std::move(solved);
    ++result_.numNodes;
    result_.totalQpIterations += solution.iterations;
    if (owner_->settings_.verbose) {
      int numFixed = 0;
      for (const int8_t v : assignment) numFixed += (v != kMiqpFree);
      LOG(INFO) << "[MixedIntegerOcpQp] relaxation " << result_.numNodes
                << ": fixed " << numFixed << "/" << assignment.size()
                << " status " << static_cast<int>(solution.status)
                << " iterations " << solution.iterations << " objective "
                << solution.objective;
    }
    if (!solution.success()) {
      // HPIPM does not certify infeasibility: a relaxation it did not solve
      // (iteration limit, step failure) is dropped with its subtree unproven,
      // which is counted separately and keeps the result from being reported as
      // optimal.
      ++result_.numFailedRelaxations;
      return false;
    }
    return true;
  }

  /** Takes a complete assignment and its relaxation as the incumbent when it
   * improves on it. */
  void updateIncumbent(const MiqpAssignment& assignment,
                       const OcpQpSolution& solution) {
    const scalar_t objective = solution.objective + logicalCost(assignment);
    if (objective < result_.incumbentObjective) {
      result_.hasIncumbent = true;
      result_.incumbentObjective = objective;
      result_.solution = solution;
      result_.assignment = assignment;
      if (owner_->settings_.verbose) {
        LOG(INFO) << "[MixedIntegerOcpQp] incumbent " << objective << " after "
                  << result_.numNodes << " relaxations";
      }
    }
  }

  /**
   * Diving heuristic: fix every free binary that is integral in the relaxation,
   * round the first fractional one, propagate, re-solve, and repeat until the
   * assignment is complete. Only used to find incumbents, never to prune.
   */
  void dive(MiqpAssignment assignment, OcpQpSolution relaxation) {
    const std::vector<MiqpBinaryVariable>& binaries = *binaries_;
    for (int iteration = 0;
         iteration < owner_->settings_.maxDiveIterations && !limitsHit();
         ++iteration) {
      bool rounded = false;
      for (size_t i = 0; i < binaries.size(); ++i) {
        if (assignment[i] != kMiqpFree) continue;
        const scalar_t value =
            relaxation.u[binaries[i].stage](binaries[i].inputIndex);
        if (std::abs(value - std::round(value)) <=
            owner_->settings_.integralityTol) {
          assignment[i] = static_cast<int8_t>(std::round(value));
        } else if (!rounded) {
          assignment[i] =
              static_cast<int8_t>(value >= kRoundUpThreshold ? 1 : 0);
          rounded = true;
        }
      }
      if (!runPropagation(assignment)) return;
      if (!solveRelaxation(assignment, relaxation)) return;
      if (isComplete(assignment)) {
        updateIncumbent(assignment, relaxation);
        return;
      }
      scalar_t unused = 0.0;
      if (owner_->firstFractional(relaxation, binaries, assignment, unused) <
          0) {
        // Integral without being complete: the remaining free binaries take
        // their values on the next round.
        continue;
      }
    }
  }

  /**
   * Warm start: tries the provided complete assignment first. Where it
   * contradicts the root fixings (e.g. a committed schedule that moved on since
   * the warm start was made) the fixings win and the rest of the warm start is
   * kept.
   */
  void tryWarmStart(const MiqpAssignment& root,
                    const MiqpAssignment* absl_nullable warmStart) {
    if (warmStart == nullptr || warmStart->size() != binaries_->size()) return;
    MiqpAssignment warm = root;
    for (size_t i = 0; i < warm.size(); ++i) {
      if (warm[i] == kMiqpFree) warm[i] = (*warmStart)[i];
    }
    if (runPropagation(warm) && isComplete(warm)) {
      OcpQpSolution solution;
      if (solveRelaxation(warm, solution)) {
        updateIncumbent(warm, solution);
      }
    }
  }

  /**
   * Best-first search over the open nodes (lowest parent bound first) with
   * depth-first diving: after branching, the preferred child is processed
   * immediately, the other one is queued. Returns whether the open nodes were
   * exhausted.
   */
  bool searchBestFirst(MiqpAssignment root) {
    OpenNodes open;
    open.push({.assignment = std::move(root),
               .parentBound = -std::numeric_limits<scalar_t>::infinity()});
    while (!open.empty() && !limitsHit()) {
      std::optional<Node> current = open.top();
      open.pop();
      while (current.has_value() && !limitsHit()) {
        Node node = std::move(*current);
        current = expand(std::move(node), open);
      }
    }
    return open.empty();
  }

  /**
   * Solves the relaxation of `node` and branches on it: queues one child in
   * `open` and returns the other to dive into, or nothing when the node is
   * pruned, infeasible or closed.
   */
  std::optional<Node> expand(Node node, OpenNodes& open) {
    const MiqpSettings& settings = owner_->settings_;
    if (node.parentBound >= result_.incumbentObjective - settings.absoluteGap) {
      ++result_.numPrunedByBound;
      return std::nullopt;
    }

    OcpQpSolution relaxation;
    const bool solved = solveRelaxation(node.assignment, relaxation);
    const bool wasRoot = isRoot_;
    if (isRoot_) {
      isRoot_ = false;
      result_.rootRelaxationSolved = solved;
      result_.rootBound =
          solved ? relaxation.objective + logicalCost(node.assignment)
                 : std::numeric_limits<scalar_t>::infinity();
    }
    if (!solved) return std::nullopt;
    const scalar_t bound = relaxation.objective + logicalCost(node.assignment);
    if (bound >= result_.incumbentObjective - settings.absoluteGap) {
      ++result_.numPrunedByBound;
      return std::nullopt;
    }

    const std::optional<Branch> branch =
        chooseBranch(node.assignment, relaxation, bound);
    if (!branch.has_value()) return std::nullopt;

    if (settings.useDivingHeuristic && wasRoot) {
      dive(node.assignment, relaxation);
    }

    // Branch: continue with the rounding direction, queue the other child.
    const int8_t preferred = static_cast<int8_t>(
        branch->fractionalValue >= kRoundUpThreshold ? 1 : 0);
    Node other{.assignment = node.assignment, .parentBound = bound};
    other.assignment[branch->index] = static_cast<int8_t>(1 - preferred);
    if (runPropagation(other.assignment)) {
      open.push(std::move(other));
    } else {
      ++result_.numInfeasible;
    }
    Node next{.assignment = std::move(node.assignment), .parentBound = bound};
    next.assignment[branch->index] = preferred;
    if (runPropagation(next.assignment)) return next;
    ++result_.numInfeasible;
    return std::nullopt;
  }

  /** The binary to branch on and its relaxed value. */
  struct Branch {
    int index = 0;
    scalar_t fractionalValue = 0.0;
  };

  /**
   * The binary a node with relaxation `relaxation` and bound `bound` branches
   * on: its first fractional one, or, when every binary is integral, its first
   * free one unless the integral completion closes the subtree. Nothing when
   * the node is closed or has no free binary left.
   */
  std::optional<Branch> chooseBranch(const MiqpAssignment& assignment,
                                     const OcpQpSolution& relaxation,
                                     scalar_t bound) {
    const std::vector<MiqpBinaryVariable>& binaries = *binaries_;
    scalar_t fractionalValue = 0.0;
    const int fractional = owner_->firstFractional(relaxation, binaries,
                                                   assignment, fractionalValue);
    if (fractional >= 0)
      return Branch{.index = fractional, .fractionalValue = fractionalValue};

    // Every binary is integral. The free ones took integral values by
    // themselves, but only the propagation rules (which the QP does not know)
    // decide whether that combination is admissible.
    MiqpAssignment complete = assignment;
    for (size_t i = 0; i < binaries.size(); ++i) {
      if (complete[i] == kMiqpFree) {
        complete[i] = static_cast<int8_t>(std::round(
            relaxation.u[binaries[i].stage](binaries[i].inputIndex)));
      }
    }
    if (runPropagation(complete)) {
      updateIncumbent(complete, relaxation);
      // CLOSING THE SUBTREE HERE IS ONLY SOUND WHEN THIS COMPLETION ATTAINS THE
      // NODE'S OWN LOWER BOUND.
      //
      // The node's bound is `relaxation.objective +
      // logicalCost(node.assignment)`, and the contract on MiqpAssignmentCostFn
      // (MixedIntegerOcpQp.h) is that the cost is EXACT on a complete
      // assignment but only a LOWER BOUND on a partial one. So a sibling
      // completion of this same node can have a worse QP objective and a
      // smaller assignment cost, and a strictly smaller total - the relaxation
      // being integral says nothing about the part of the objective the QP
      // cannot see. Breaking unconditionally discarded it unexplored, and
      // nothing counted it, so the search still reported `optimal`.
      //
      // When no assignment cost is supplied, or the node's assignment was
      // already complete, the two agree exactly and this test always passes:
      // the fast path is preserved bit for bit.
      if (relaxation.objective + logicalCost(complete) <=
          bound + owner_->settings_.absoluteGap) {
        ++result_.numPrunedByBound;
        return std::nullopt;
      }
      // Otherwise fall through and branch - the subtree may still hold
      // something better than what we just took.
    }
    // Branch on the first free variable instead.
    for (size_t i = 0; i < binaries.size(); ++i) {
      if (assignment[i] == kMiqpFree) {
        return Branch{.index = static_cast<int>(i),
                      .fractionalValue = relaxation.u[binaries[i].stage](
                          binaries[i].inputIndex)};
      }
    }
    return std::nullopt;
  }

  MixedIntegerOcpQp* absl_nonnull owner_;
  OcpQpProblem* absl_nonnull problem_;
  const std::vector<MiqpBinaryVariable>* absl_nonnull binaries_;
  std::vector<BinaryLocation> locations_;
  const MiqpPropagateFn* absl_nonnull
  propagate_;  // may hold no target: no propagation
  const MiqpAssignmentCostFn* absl_nonnull
  assignmentCost_;  // may hold no target: no assignment cost
  Clock::time_point startTime_;
  MiqpResult result_;
  // The first problem the QP solver rejects (inconsistent data, memory). Every
  // relaxation solves the same problem structure, so once one is rejected the
  // search stops and returns this instead of a result.
  absl::Status qpError_ = absl::OkStatus();
  bool isRoot_ = true;  // until the first relaxation has been solved
};

absl::StatusOr<MiqpResult> MixedIntegerOcpQp::solve(
    OcpQpProblem& problem,
    const std::vector<MiqpBinaryVariable>& binaries,
    const MiqpAssignment& initialAssignment,
    const MiqpPropagateFn& propagate,
    const MiqpAssignment* absl_nullable warmStart,
    const MiqpAssignmentCostFn& assignmentCost) {
  if (initialAssignment.size() != binaries.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[MixedIntegerOcpQp] the initial assignment has ",
                     initialAssignment.size(), " entries for ", binaries.size(),
                     " binaries"));
  }
  ASSIGN_OR_RETURN(std::vector<BinaryLocation> locations,
                   locateBinaries(problem, binaries));
  Search search(this, &problem, &binaries, std::move(locations), &propagate,
                &assignmentCost);
  return search.run(initialAssignment, warmStart);
}

}  // namespace ocs2::humanoid
