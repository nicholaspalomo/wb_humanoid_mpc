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

#include "humanoid_common_mpc_app/visualization/SolutionTermEvaluator.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/ComputationRequest.h"
#include "ocs2_core/misc/Collection.h"

namespace ocs2::humanoid::visualization {

namespace {

constexpr scalar_t kNotANumber = std::numeric_limits<scalar_t>::quiet_NaN();

/** The term names of `collection`, in the order the terms were added (their index), not the hash map's. */
template <typename T>
std::vector<std::string> termNamesInOrder(const Collection<T>& collection) {
  // NOLINTNEXTLINE(totw-unordered-container): OCS2's Collection::getTermNameMap() returns one; sorted by index below.
  const std::unordered_map<std::string, size_t> nameMap = collection.getTermNameMap();
  std::vector<std::pair<size_t, std::string>> indexed;
  indexed.reserve(nameMap.size());
  for (const std::pair<const std::string, size_t>& entry : nameMap) {
    indexed.emplace_back(entry.second, entry.first);
  }
  std::sort(indexed.begin(), indexed.end());
  std::vector<std::string> names;
  names.reserve(indexed.size());
  for (std::pair<size_t, std::string>& entry : indexed) {
    names.push_back(std::move(entry.second));
  }
  return names;
}

/** The term `name` of `collection`, or null when the collection has no term of that name. */
template <typename T>
T* absl_nullable findTerm(Collection<T>& collection, const std::string& name) {
  size_t index = 0;
  if (!collection.getTermIndex(name, index)) return nullptr;
  return &collection.get(name);
}

/** The 2-norm of the violation of an equality g = 0 (all of g) or an inequality g >= 0 (its negative part). */
scalar_t violationNorm(const vector_t& value, bool inequality) {
  return inequality ? value.cwiseMin(0.0).norm() : value.norm();
}

}  // namespace

absl::StatusOr<std::unique_ptr<SolutionTermEvaluator>> SolutionTermEvaluator::Create(OptimalControlProblem* absl_nonnull problem) {
  const OptimalControlProblem& ocp = *problem;
  if (ocp.costPtr == nullptr || ocp.stateCostPtr == nullptr || ocp.softConstraintPtr == nullptr || ocp.stateSoftConstraintPtr == nullptr ||
      ocp.finalCostPtr == nullptr || ocp.finalSoftConstraintPtr == nullptr || ocp.equalityConstraintPtr == nullptr ||
      ocp.stateEqualityConstraintPtr == nullptr || ocp.inequalityConstraintPtr == nullptr || ocp.stateInequalityConstraintPtr == nullptr ||
      ocp.finalEqualityConstraintPtr == nullptr || ocp.finalInequalityConstraintPtr == nullptr || ocp.preComputationPtr == nullptr) {
    return absl::InvalidArgumentError("the optimal control problem lacks a cost or constraint collection, or its precomputation.");
  }
  std::unique_ptr<SolutionTermEvaluator> evaluator = absl::WrapUnique(new SolutionTermEvaluator(problem));
  evaluator->addCostTerms(CostCollection::kStateInput, "intermediate", termNamesInOrder(*ocp.costPtr));
  evaluator->addCostTerms(CostCollection::kState, "state", termNamesInOrder(*ocp.stateCostPtr));
  evaluator->addCostTerms(CostCollection::kSoftStateInput, "soft", termNamesInOrder(*ocp.softConstraintPtr));
  evaluator->addCostTerms(CostCollection::kSoftState, "state_soft", termNamesInOrder(*ocp.stateSoftConstraintPtr));
  evaluator->addCostTerms(CostCollection::kFinal, "final", termNamesInOrder(*ocp.finalCostPtr));
  evaluator->addCostTerms(CostCollection::kFinalSoft, "final_soft", termNamesInOrder(*ocp.finalSoftConstraintPtr));
  evaluator->addConstraintTerms(ConstraintCollection::kStateInputEquality, "equality", termNamesInOrder(*ocp.equalityConstraintPtr));
  evaluator->addConstraintTerms(ConstraintCollection::kStateEquality, "state_equality", termNamesInOrder(*ocp.stateEqualityConstraintPtr));
  evaluator->addConstraintTerms(ConstraintCollection::kStateInputInequality, "inequality", termNamesInOrder(*ocp.inequalityConstraintPtr));
  evaluator->addConstraintTerms(ConstraintCollection::kStateInequality, "state_inequality",
                                termNamesInOrder(*ocp.stateInequalityConstraintPtr));
  evaluator->addConstraintTerms(ConstraintCollection::kFinalEquality, "final_equality", termNamesInOrder(*ocp.finalEqualityConstraintPtr));
  evaluator->addConstraintTerms(ConstraintCollection::kFinalInequality, "final_inequality",
                                termNamesInOrder(*ocp.finalInequalityConstraintPtr));
  return evaluator;
}

void SolutionTermEvaluator::addCostTerms(CostCollection collection, absl::string_view prefix, const std::vector<std::string>& names) {
  for (const std::string& name : names) {
    costTerms_.push_back(CostTerm{.collection = collection, .term = name});
    costNames_.push_back(absl::StrCat(prefix, "/", name));
  }
}

void SolutionTermEvaluator::addConstraintTerms(ConstraintCollection collection,
                                               absl::string_view prefix,
                                               const std::vector<std::string>& names) {
  for (const std::string& name : names) {
    constraintTerms_.push_back(ConstraintTerm{.collection = collection, .term = name});
    constraintNames_.push_back(absl::StrCat(prefix, "/", name));
  }
}

scalar_t SolutionTermEvaluator::costValue(
    const CostTerm& term, scalar_t time, const vector_t& state, const vector_t& input, const TargetTrajectories& target) {
  const PreComputation& preComputation = *problem_->preComputationPtr;
  switch (term.collection) {
    case CostCollection::kStateInput:
    case CostCollection::kSoftStateInput: {
      StateInputCostCollection& collection =
          term.collection == CostCollection::kStateInput ? *problem_->costPtr : *problem_->softConstraintPtr;
      const StateInputCost* absl_nullable cost = findTerm(collection, term.term);
      if (cost == nullptr) return kNotANumber;
      return cost->isActive(time) ? cost->getValue(time, state, input, target, preComputation) : 0.0;
    }
    case CostCollection::kState:
    case CostCollection::kSoftState:
    case CostCollection::kFinal:
    case CostCollection::kFinalSoft: {
      StateCostCollection* absl_nonnull collection = problem_->stateCostPtr.get();
      if (term.collection == CostCollection::kSoftState) collection = problem_->stateSoftConstraintPtr.get();
      if (term.collection == CostCollection::kFinal) collection = problem_->finalCostPtr.get();
      if (term.collection == CostCollection::kFinalSoft) collection = problem_->finalSoftConstraintPtr.get();
      const StateCost* absl_nullable cost = findTerm(*collection, term.term);
      if (cost == nullptr) return kNotANumber;
      return cost->isActive(time) ? cost->getValue(time, state, target, preComputation) : 0.0;
    }
  }
  return kNotANumber;
}

scalar_t SolutionTermEvaluator::constraintViolation(const ConstraintTerm& term,
                                                    scalar_t time,
                                                    const vector_t& state,
                                                    const vector_t& input) {
  const PreComputation& preComputation = *problem_->preComputationPtr;
  switch (term.collection) {
    case ConstraintCollection::kStateInputEquality:
    case ConstraintCollection::kStateInputInequality: {
      const bool inequality = term.collection == ConstraintCollection::kStateInputInequality;
      StateInputConstraintCollection& collection = inequality ? *problem_->inequalityConstraintPtr : *problem_->equalityConstraintPtr;
      const StateInputConstraint* absl_nullable constraint = findTerm(collection, term.term);
      if (constraint == nullptr) return kNotANumber;
      return constraint->isActive(time) ? violationNorm(constraint->getValue(time, state, input, preComputation), inequality) : 0.0;
    }
    case ConstraintCollection::kStateEquality:
    case ConstraintCollection::kStateInequality:
    case ConstraintCollection::kFinalEquality:
    case ConstraintCollection::kFinalInequality: {
      const bool inequality =
          term.collection == ConstraintCollection::kStateInequality || term.collection == ConstraintCollection::kFinalInequality;
      StateConstraintCollection* absl_nonnull collection = problem_->stateEqualityConstraintPtr.get();
      if (term.collection == ConstraintCollection::kStateInequality) collection = problem_->stateInequalityConstraintPtr.get();
      if (term.collection == ConstraintCollection::kFinalEquality) collection = problem_->finalEqualityConstraintPtr.get();
      if (term.collection == ConstraintCollection::kFinalInequality) collection = problem_->finalInequalityConstraintPtr.get();
      const StateConstraint* absl_nullable constraint = findTerm(*collection, term.term);
      if (constraint == nullptr) return kNotANumber;
      return constraint->isActive(time) ? violationNorm(constraint->getValue(time, state, preComputation), inequality) : 0.0;
    }
  }
  return kNotANumber;
}

absl::Status SolutionTermEvaluator::evaluate(const PrimalSolution& solution,
                                             const TargetTrajectories& target,
                                             const PerformanceIndex& performance,
                                             SolutionTerms* absl_nonnull terms) {
  if (solution.timeTrajectory_.empty() || solution.stateTrajectory_.size() != solution.timeTrajectory_.size() ||
      solution.inputTrajectory_.empty()) {
    return absl::InvalidArgumentError("the solution has no node, or no input at its first node.");
  }
  terms->performance = performance;
  terms->costs.resize(static_cast<Eigen::Index>(costTerms_.size()));
  terms->constraintViolations.resize(static_cast<Eigen::Index>(constraintTerms_.size()));

  // The intermediate terms at the first node, then the final ones at the last: each after its own precomputation
  // request, as the solver's metrics are computed (multiple_shooting::computeIntermediateMetrics(), computeTerminalMetrics()).
  constexpr RequestSet kRequest = Request::Cost + Request::SoftConstraint + Request::Constraint;
  const scalar_t startTime = solution.timeTrajectory_.front();
  const vector_t& startState = solution.stateTrajectory_.front();
  const vector_t& startInput = solution.inputTrajectory_.front();
  problem_->preComputationPtr->request(kRequest, startTime, startState, startInput);
  for (size_t index = 0; index < costTerms_.size(); ++index) {
    const CostCollection collection = costTerms_[index].collection;
    if (collection != CostCollection::kFinal && collection != CostCollection::kFinalSoft) {
      terms->costs[static_cast<Eigen::Index>(index)] = costValue(costTerms_[index], startTime, startState, startInput, target);
    }
  }
  for (size_t index = 0; index < constraintTerms_.size(); ++index) {
    const ConstraintCollection collection = constraintTerms_[index].collection;
    if (collection != ConstraintCollection::kFinalEquality && collection != ConstraintCollection::kFinalInequality) {
      terms->constraintViolations[static_cast<Eigen::Index>(index)] =
          constraintViolation(constraintTerms_[index], startTime, startState, startInput);
    }
  }

  const scalar_t finalTime = solution.timeTrajectory_.back();
  const vector_t& finalState = solution.stateTrajectory_.back();
  problem_->preComputationPtr->requestFinal(kRequest, finalTime, finalState);
  for (size_t index = 0; index < costTerms_.size(); ++index) {
    const CostCollection collection = costTerms_[index].collection;
    if (collection == CostCollection::kFinal || collection == CostCollection::kFinalSoft) {
      terms->costs[static_cast<Eigen::Index>(index)] = costValue(costTerms_[index], finalTime, finalState, startInput, target);
    }
  }
  for (size_t index = 0; index < constraintTerms_.size(); ++index) {
    const ConstraintCollection collection = constraintTerms_[index].collection;
    if (collection == ConstraintCollection::kFinalEquality || collection == ConstraintCollection::kFinalInequality) {
      terms->constraintViolations[static_cast<Eigen::Index>(index)] =
          constraintViolation(constraintTerms_[index], finalTime, finalState, startInput);
    }
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::visualization
