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

#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>

namespace ocs2::humanoid {

namespace {

struct Node {
  MiqpAssignment assignment;
  scalar_t parentBound;
};

using Clock = std::chrono::steady_clock;

scalar_t elapsedSeconds(const Clock::time_point& start) {
  return std::chrono::duration<scalar_t>(Clock::now() - start).count();
}

bool isComplete(const MiqpAssignment& assignment) {
  return std::none_of(assignment.begin(), assignment.end(), [](std::int8_t v) { return v == kMiqpFree; });
}

}  // namespace

MixedIntegerOcpQp::MixedIntegerOcpQp(OcpQpHpipmSolver::Settings qpSettings, MiqpSettings settings)
    : qpSolver_(qpSettings), settings_(settings) {}

std::vector<MixedIntegerOcpQp::BinaryLocation> MixedIntegerOcpQp::locateBinaries(const OcpQpProblem& problem,
                                                                                 const std::vector<MiqpBinaryVariable>& binaries) const {
  std::vector<BinaryLocation> locations;
  locations.reserve(binaries.size());
  for (const MiqpBinaryVariable& binary : binaries) {
    if (binary.stage < 0 || binary.stage >= problem.numStages()) {
      throw std::invalid_argument("[MixedIntegerOcpQp] binary variable stage out of range: " + std::to_string(binary.stage));
    }
    const OcpQpStage& stage = problem.stages[binary.stage];
    const auto it = std::find(stage.idxbu.begin(), stage.idxbu.end(), binary.inputIndex);
    if (it == stage.idxbu.end()) {
      throw std::invalid_argument("[MixedIntegerOcpQp] binary input " + std::to_string(binary.inputIndex) + " of stage " +
                                  std::to_string(binary.stage) + " has no box constraint");
    }
    locations.push_back({binary.stage, static_cast<int>(it - stage.idxbu.begin())});
  }
  return locations;
}

void MixedIntegerOcpQp::applyAssignment(OcpQpProblem& problem,
                                        const std::vector<BinaryLocation>& locations,
                                        const MiqpAssignment& assignment) const {
  for (std::size_t i = 0; i < locations.size(); ++i) {
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

int MixedIntegerOcpQp::firstFractional(const OcpQpSolution& solution,
                                       const std::vector<MiqpBinaryVariable>& binaries,
                                       const MiqpAssignment& assignment,
                                       scalar_t& fractionalValue) const {
  for (std::size_t i = 0; i < binaries.size(); ++i) {
    if (assignment[i] != kMiqpFree) {
      continue;
    }
    const scalar_t value = solution.u[binaries[i].stage](binaries[i].inputIndex);
    if (std::abs(value - std::round(value)) > settings_.integralityTol) {
      fractionalValue = value;
      return static_cast<int>(i);
    }
  }
  return -1;
}

bool MixedIntegerOcpQp::solveFixed(OcpQpProblem& problem,
                                   const std::vector<MiqpBinaryVariable>& binaries,
                                   MiqpAssignment assignment,
                                   const MiqpPropagateFn& propagate,
                                   const MiqpAssignmentCostFn& assignmentCost,
                                   OcpQpSolution& solution,
                                   scalar_t& objective) {
  objective = std::numeric_limits<scalar_t>::infinity();
  if (assignment.size() != binaries.size()) {
    throw std::invalid_argument("[MixedIntegerOcpQp] assignment size does not match the number of binaries");
  }
  if (propagate && !propagate(assignment)) {
    return false;
  }
  if (!isComplete(assignment)) {
    return false;
  }
  const std::vector<BinaryLocation> locations = locateBinaries(problem, binaries);
  applyAssignment(problem, locations, assignment);
  solution = qpSolver_.solve(problem);
  if (!solution.success()) {
    return false;
  }
  objective = solution.objective + (assignmentCost ? assignmentCost(assignment) : 0.0);
  return true;
}

MiqpResult MixedIntegerOcpQp::solve(OcpQpProblem& problem,
                                    const std::vector<MiqpBinaryVariable>& binaries,
                                    const MiqpAssignment& initialAssignment,
                                    const MiqpPropagateFn& propagate,
                                    const MiqpAssignment* warmStart,
                                    const MiqpAssignmentCostFn& assignmentCost) {
  const auto startTime = Clock::now();
  MiqpResult result;
  result.incumbentObjective = std::numeric_limits<scalar_t>::infinity();

  if (initialAssignment.size() != binaries.size()) {
    throw std::invalid_argument("[MixedIntegerOcpQp] initialAssignment size does not match the number of binaries");
  }
  const std::vector<BinaryLocation> locations = locateBinaries(problem, binaries);

  const auto runPropagation = [&](MiqpAssignment& assignment) -> bool { return !propagate || propagate(assignment); };
  const auto logicalCost = [&](const MiqpAssignment& assignment) -> scalar_t { return assignmentCost ? assignmentCost(assignment) : 0.0; };

  // Merges fixed entries of `fixings` into `assignment`; false when they contradict existing fixings.
  const auto mergeFixings = [](MiqpAssignment& assignment, const MiqpAssignment& fixings) -> bool {
    for (std::size_t i = 0; i < assignment.size(); ++i) {
      if (fixings[i] == kMiqpFree) continue;
      if (assignment[i] != kMiqpFree && assignment[i] != fixings[i]) return false;
      assignment[i] = fixings[i];
    }
    return true;
  };

  const auto limitsHit = [&]() {
    if (result.numNodes >= settings_.maxNodes) {
      result.nodeLimitHit = true;
      return true;
    }
    if (elapsedSeconds(startTime) > settings_.maxSolveTime) {
      result.timeLimitHit = true;
      return true;
    }
    return false;
  };

  // Solves the relaxation of an assignment. Returns false if the QP failed (treated as infeasible).
  const auto solveRelaxation = [&](const MiqpAssignment& assignment, OcpQpSolution& solution) -> bool {
    applyAssignment(problem, locations, assignment);
    solution = qpSolver_.solve(problem);
    ++result.numNodes;
    result.totalQpIterations += solution.iterations;
    if (settings_.verbose) {
      int numFixed = 0;
      for (const std::int8_t v : assignment) numFixed += (v != kMiqpFree);
      std::cout << "[MixedIntegerOcpQp] relaxation " << result.numNodes << ": fixed " << numFixed << "/" << assignment.size() << " status "
                << static_cast<int>(solution.status) << " iterations " << solution.iterations << " objective " << solution.objective
                << std::endl;
    }
    if (!solution.success()) {
      ++result.numInfeasible;
      return false;
    }
    return true;
  };

  const auto updateIncumbent = [&](const MiqpAssignment& assignment, const OcpQpSolution& solution) {
    const scalar_t objective = solution.objective + logicalCost(assignment);
    if (objective < result.incumbentObjective) {
      result.hasIncumbent = true;
      result.incumbentObjective = objective;
      result.solution = solution;
      result.assignment = assignment;
      if (settings_.verbose) {
        std::cout << "[MixedIntegerOcpQp] incumbent " << objective << " after " << result.numNodes << " relaxations" << std::endl;
      }
    }
  };

  // Diving heuristic: fix every free binary that is integral in the relaxation, round the first fractional one, propagate,
  // re-solve, and repeat until the assignment is complete. Only used to find incumbents, never to prune.
  const auto dive = [&](MiqpAssignment assignment, OcpQpSolution relaxation) {
    for (int iteration = 0; iteration < settings_.maxDiveIterations && !limitsHit(); ++iteration) {
      bool rounded = false;
      for (std::size_t i = 0; i < binaries.size(); ++i) {
        if (assignment[i] != kMiqpFree) continue;
        const scalar_t value = relaxation.u[binaries[i].stage](binaries[i].inputIndex);
        if (std::abs(value - std::round(value)) <= settings_.integralityTol) {
          assignment[i] = static_cast<std::int8_t>(std::round(value));
        } else if (!rounded) {
          assignment[i] = static_cast<std::int8_t>(value >= 0.5 ? 1 : 0);
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
      if (firstFractional(relaxation, binaries, assignment, unused) < 0) {
        // Integral without being complete: the remaining free binaries take their values on the next round.
        continue;
      }
    }
  };

  // Root assignment.
  MiqpAssignment root = initialAssignment;
  if (!runPropagation(root)) {
    result.solveTime = elapsedSeconds(startTime);
    result.optimal = true;  // provably infeasible
    return result;
  }

  // Warm start: try the provided complete assignment first.
  if (warmStart != nullptr && warmStart->size() == binaries.size()) {
    MiqpAssignment warm = root;
    if (mergeFixings(warm, *warmStart) && runPropagation(warm) && isComplete(warm)) {
      OcpQpSolution solution;
      if (solveRelaxation(warm, solution)) {
        updateIncumbent(warm, solution);
      }
    }
  }

  // Best-first search over the open nodes (lowest parent bound first) with depth-first diving: after branching, the
  // preferred child is processed immediately, the other one is queued.
  const auto worseBound = [](const Node& a, const Node& b) { return a.parentBound > b.parentBound; };
  std::priority_queue<Node, std::vector<Node>, decltype(worseBound)> open(worseBound);
  open.push({root, -std::numeric_limits<scalar_t>::infinity()});
  bool isRoot = true;

  while (!open.empty() && !limitsHit()) {
    std::optional<Node> current = open.top();
    open.pop();

    while (current.has_value() && !limitsHit()) {
      Node node = std::move(*current);
      current.reset();

      if (node.parentBound >= result.incumbentObjective - settings_.absoluteGap) {
        ++result.numPrunedByBound;
        break;
      }

      OcpQpSolution relaxation;
      const bool solved = solveRelaxation(node.assignment, relaxation);
      const bool wasRoot = isRoot;
      if (isRoot) {
        isRoot = false;
        result.rootRelaxationSolved = solved;
        result.rootBound = solved ? relaxation.objective + logicalCost(node.assignment) : std::numeric_limits<scalar_t>::infinity();
      }
      if (!solved) {
        break;
      }
      const scalar_t bound = relaxation.objective + logicalCost(node.assignment);
      if (bound >= result.incumbentObjective - settings_.absoluteGap) {
        ++result.numPrunedByBound;
        break;
      }

      scalar_t fractionalValue = 0.0;
      int branchIndex = firstFractional(relaxation, binaries, node.assignment, fractionalValue);
      if (branchIndex < 0) {
        // Every binary is integral. The free ones took integral values by themselves, but only the propagation rules
        // (which the QP does not know) decide whether that combination is admissible.
        MiqpAssignment complete = node.assignment;
        for (std::size_t i = 0; i < binaries.size(); ++i) {
          if (complete[i] == kMiqpFree) {
            complete[i] = static_cast<std::int8_t>(std::round(relaxation.u[binaries[i].stage](binaries[i].inputIndex)));
          }
        }
        if (runPropagation(complete)) {
          updateIncumbent(complete, relaxation);
          break;
        }
        // Inadmissible: branch on the first free variable instead.
        for (std::size_t i = 0; i < binaries.size(); ++i) {
          if (node.assignment[i] == kMiqpFree) {
            branchIndex = static_cast<int>(i);
            fractionalValue = relaxation.u[binaries[i].stage](binaries[i].inputIndex);
            break;
          }
        }
        if (branchIndex < 0) {
          break;
        }
      }

      if (settings_.useDivingHeuristic && wasRoot) {
        dive(node.assignment, relaxation);
      }

      // Branch: continue with the rounding direction, queue the other child.
      const std::int8_t preferred = static_cast<std::int8_t>(fractionalValue >= 0.5 ? 1 : 0);
      Node other{node.assignment, bound};
      other.assignment[branchIndex] = static_cast<std::int8_t>(1 - preferred);
      if (runPropagation(other.assignment)) {
        open.push(std::move(other));
      } else {
        ++result.numInfeasible;
      }
      Node next{std::move(node.assignment), bound};
      next.assignment[branchIndex] = preferred;
      if (runPropagation(next.assignment)) {
        current = std::move(next);
      } else {
        ++result.numInfeasible;
      }
    }
  }

  result.optimal = open.empty() && !result.nodeLimitHit && !result.timeLimitHit;
  result.solveTime = elapsedSeconds(startTime);
  if (result.hasIncumbent) {
    // Leave the problem's binary bounds at the incumbent so that the caller can inspect it.
    applyAssignment(problem, locations, result.assignment);
  }
  return result;
}

}  // namespace ocs2::humanoid
