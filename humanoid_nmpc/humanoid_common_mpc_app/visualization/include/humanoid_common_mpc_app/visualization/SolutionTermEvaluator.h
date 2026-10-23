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

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

namespace ocs2::humanoid::visualization {

/** The values of one MPC solution that the mpc_solution telemetry groups plot (SolutionTermEvaluator). */
struct SolutionTerms {
  /** The solver's performance index of the solution, over the whole horizon (OCS2's PerformanceIndex). */
  PerformanceIndex performance;
  /**
   * Per SolutionTermEvaluator::costNames(): an intermediate term's value L_i(t0, x0, u0) at the solution's first node (a
   * cost rate, per second; 0 while the term is inactive), a final term's Phi_i(tf, xf) at its last node; NaN for a term
   * that the problem no longer has.
   */
  vector_t costs;
  /**
   * Per SolutionTermEvaluator::constraintNames(): the 2-norm of the term's violation at the same node as the costs -
   * |g| for an equality g = 0, |min(g, 0)| for an inequality g >= 0 (OCS2's convention); 0 while the term is inactive,
   * NaN for a term that the problem no longer has.
   */
  vector_t constraintViolations;
};

/**
 * Evaluates every cost and constraint term of the MPC's optimal control problem, by name, at the first and the last
 * node of a solution: what each term contributes, which the solver's PerformanceIndex only sums.
 *
 * It evaluates the solver's own problem - the one the solver solved, with the weights the parameter updater wrote into
 * it - so the values are the solver's. It therefore runs on the solver thread, between two solves, when no worker of
 * the solver runs: the MPC node calls evaluate() from the post-solve observer (VisualizationPublisher::
 * postSolveObserver()), after the policy is published, and never on the robot's realtime thread. One precomputation
 * request and one evaluation per term at each of the two nodes: tens of microseconds for the shipped problems.
 *
 * The term names are read once, in Create(), from the problem's collections, in the order the terms were added; a term
 * is looked up by its name at every evaluation, so a term that a reload removed reads NaN and never a dangling term.
 * Not thread-safe; evaluate() writes the problem's precomputation.
 */
class SolutionTermEvaluator {
 public:
  /**
   * Retains `problem` (the solver's first worker's, SqpSolver::getOcpDefinitions().front()), which must outlive the
   * evaluator. InvalidArgument when a collection of the problem is missing.
   */
  static absl::StatusOr<std::unique_ptr<SolutionTermEvaluator>> Create(OptimalControlProblem* absl_nonnull problem);

  SolutionTermEvaluator(const SolutionTermEvaluator&) = delete;
  SolutionTermEvaluator& operator=(const SolutionTermEvaluator&) = delete;
  ~SolutionTermEvaluator() = default;

  /**
   * "<collection>/<term>": intermediate/, state/, soft/, state_soft/ (the intermediate costs and soft constraints, at
   * the first node), final/, final_soft/ (at the last node).
   */
  const std::vector<std::string>& costNames() const { return costNames_; }
  /**
   * "<collection>/<term>": equality/, state_equality/, inequality/, state_inequality/ (at the first node),
   * final_equality/, final_inequality/ (at the last node).
   */
  const std::vector<std::string>& constraintNames() const { return constraintNames_; }

  /**
   * Writes the terms of `solution`, whose costs track `target`, and `performance` into `terms`, sizing its vectors on
   * the first call (later calls do not allocate). InvalidArgument, with `terms` untouched, when the solution has no node
   * or no input at its first node. The terms may throw (OCS2, CppAD, Pinocchio): the caller isolates them.
   */
  absl::Status evaluate(const PrimalSolution& solution,
                        const TargetTrajectories& target,
                        const PerformanceIndex& performance,
                        SolutionTerms* absl_nonnull terms);

 private:
  /** The collections of the problem's costs (and soft constraints), in the order of the names. */
  enum class CostCollection { kStateInput, kState, kSoftStateInput, kSoftState, kFinal, kFinalSoft };
  /** The collections of the problem's constraints, in the order of the names. */
  enum class ConstraintCollection {
    kStateInputEquality,
    kStateEquality,
    kStateInputInequality,
    kStateInequality,
    kFinalEquality,
    kFinalInequality
  };

  struct CostTerm {
    CostCollection collection;
    std::string term;
  };
  struct ConstraintTerm {
    ConstraintCollection collection;
    std::string term;
  };

  explicit SolutionTermEvaluator(OptimalControlProblem* absl_nonnull problem) : problem_(problem) {}

  /** Appends the terms `names` of `collection`, named "<prefix>/<name>". */
  void addCostTerms(CostCollection collection, absl::string_view prefix, const std::vector<std::string>& names);
  void addConstraintTerms(ConstraintCollection collection, absl::string_view prefix, const std::vector<std::string>& names);
  scalar_t costValue(const CostTerm& term, scalar_t time, const vector_t& state, const vector_t& input, const TargetTrajectories& target);
  scalar_t constraintViolation(const ConstraintTerm& term, scalar_t time, const vector_t& state, const vector_t& input);

  OptimalControlProblem* absl_nonnull problem_;
  std::vector<CostTerm> costTerms_;
  std::vector<std::string> costNames_;
  std::vector<ConstraintTerm> constraintTerms_;
  std::vector<std::string> constraintNames_;
};

}  // namespace ocs2::humanoid::visualization
