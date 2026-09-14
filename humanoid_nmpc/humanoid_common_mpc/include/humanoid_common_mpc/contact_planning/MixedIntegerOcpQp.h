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

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <ocs2_core/Types.h>

#include "humanoid_common_mpc/contact_planning/OcpQpHpipm.h"

namespace ocs2::humanoid {

/** An input entry of an OCP-QP stage that is restricted to {0, 1}. It must appear in the stage's input box constraints. */
struct MiqpBinaryVariable {
  int stage = 0;
  int inputIndex = 0;
};

/** Value of a binary variable in a (partial) assignment: -1 free, 0 or 1 fixed. */
using MiqpAssignment = std::vector<std::int8_t>;
constexpr std::int8_t kMiqpFree = -1;

/**
 * Logical propagation hook. Receives a partial assignment, may fix further variables that are implied by the fixed ones, and
 * returns false when the partial assignment is provably infeasible. The branch-and-bound calls it on every node before the
 * relaxation, and on every complete assignment before it is accepted as an incumbent, so rules that are not part of the QP
 * (pure logic on the binaries) are enforced exactly through this hook.
 */
using MiqpPropagateFn = std::function<bool(MiqpAssignment& assignment)>;

/**
 * Optional cost on the binaries that is not part of the QP (e.g. a price per contact switch). For a complete assignment it
 * must return the exact cost, for a partial one a lower bound (typically: only count what is decided).
 */
using MiqpAssignmentCostFn = std::function<scalar_t(const MiqpAssignment& assignment)>;

struct MiqpSettings {
  int maxNodes = 300;              // branch-and-bound node limit (QP relaxations solved, heuristics included)
  scalar_t maxSolveTime = 0.1;     // [s] wall-clock limit; the best incumbent so far is returned when exceeded
  scalar_t integralityTol = 1e-4;  // |v - round(v)| below this counts as integral
  scalar_t absoluteGap = 1e-6;     // nodes whose bound is within this gap of the incumbent are pruned
  bool useDivingHeuristic = true;  // after the root relaxation, dive (fix integral values, round the rest) for an early incumbent
  int maxDiveIterations = 64;
  bool verbose = false;
};

struct MiqpResult {
  bool hasIncumbent = false;
  OcpQpSolution solution;             // trajectories of the incumbent (all binaries integral)
  MiqpAssignment assignment;          // the incumbent's binary values
  scalar_t incumbentObjective = 0.0;  // QP objective + assignment cost of the incumbent
  scalar_t rootBound = 0.0;           // bound of the root relaxation (when it succeeded)
  bool rootRelaxationSolved = false;
  int numNodes = 0;           // relaxations solved, including root, heuristic and warm-start solves
  int totalQpIterations = 0;  // interior point iterations summed over all relaxations
  int numPrunedByBound = 0;
  int numInfeasible = 0;
  bool nodeLimitHit = false;
  bool timeLimitHit = false;
  bool optimal = false;      // the search tree was exhausted (up to the tolerances)
  scalar_t solveTime = 0.0;  // [s]
};

/**
 * Branch-and-bound for OCP-QPs with binary input variables.
 *
 * The continuous relaxation of every node is solved with HPIPM on the structured OCP form, so a node costs one Riccati-based
 * interior point solve. Open nodes are processed best-bound first with depth-first diving into the rounding direction.
 * Branching always picks the first fractional binary in the order given by the caller, which for contact planning is time
 * order: early contact decisions are settled first and the propagation hook (phase durations, alternation) then fixes long
 * runs of later variables. A warm-start assignment (e.g. the previous plan) and a diving heuristic provide incumbents early.
 *
 * Note that for disjunctive constraints (big-M) the relaxation bounds are weak, so the search is effectively an enumeration
 * of the admissible assignments with pruning by propagation; the node and time limits make it an anytime method.
 */
class MixedIntegerOcpQp {
 public:
  MixedIntegerOcpQp(OcpQpHpipmSolver::Settings qpSettings, MiqpSettings settings);

  /**
   * @param problem            The OCP-QP. The binaries' box bounds are overwritten by the search.
   * @param binaries           Binary variables in branching order.
   * @param initialAssignment  Fixings that hold for the whole search (e.g. the committed schedule); free entries are -1.
   * @param propagate          Logical propagation hook, may be empty.
   * @param warmStart          Optional complete assignment tried first to obtain an incumbent, may be nullptr.
   * @param assignmentCost     Optional logical cost on the binaries, may be empty.
   */
  MiqpResult solve(OcpQpProblem& problem,
                   const std::vector<MiqpBinaryVariable>& binaries,
                   const MiqpAssignment& initialAssignment,
                   const MiqpPropagateFn& propagate,
                   const MiqpAssignment* warmStart = nullptr,
                   const MiqpAssignmentCostFn& assignmentCost = nullptr);

  /**
   * Solves the QP with every binary fixed to `assignment` (which must be complete after propagation). Returns false if the
   * propagation rejects the assignment or the QP fails; `objective` then is +inf. Used for local search around an incumbent.
   */
  bool solveFixed(OcpQpProblem& problem,
                  const std::vector<MiqpBinaryVariable>& binaries,
                  MiqpAssignment assignment,
                  const MiqpPropagateFn& propagate,
                  const MiqpAssignmentCostFn& assignmentCost,
                  OcpQpSolution& solution,
                  scalar_t& objective);

  const MiqpSettings& getSettings() const { return settings_; }
  void setSettings(const MiqpSettings& settings) { settings_ = settings; }

 private:
  struct BinaryLocation {
    int stage;
    int boxIndex;  // index into the stage's idxbu / lbu / ubu
  };

  std::vector<BinaryLocation> locateBinaries(const OcpQpProblem& problem, const std::vector<MiqpBinaryVariable>& binaries) const;
  void applyAssignment(OcpQpProblem& problem, const std::vector<BinaryLocation>& locations, const MiqpAssignment& assignment) const;
  int firstFractional(const OcpQpSolution& solution,
                      const std::vector<MiqpBinaryVariable>& binaries,
                      const MiqpAssignment& assignment,
                      scalar_t& fractionalValue) const;

  OcpQpHpipmSolver qpSolver_;
  MiqpSettings settings_;
};

}  // namespace ocs2::humanoid
