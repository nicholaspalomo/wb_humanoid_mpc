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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "tools/config_dump/ProblemDump.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/ComputationRequest.h"
#include "ocs2_core/cost/StateCostCollection.h"
#include "ocs2_core/cost/StateInputCostCollection.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/approximate_model/LinearQuadraticApproximator.h"

#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid::config_dump {
namespace {

// The points of the horizon the problem is approximated at: in the first double support, in the left swing and in
// the right swing of the walking schedule below.
constexpr std::array<scalar_t, 3> kTimes = {0.05, 0.3, 0.8};
constexpr scalar_t kFinalTime = 1.5;

// The walking schedule: its event times [s], between stance, left swing, stance, right swing, stance, left swing and
// stance.
constexpr std::array<scalar_t, 6> kWalkingEventTimes = {0.2, 0.55, 0.65, 1.0, 1.1, 1.45};
/** [m] How far ahead of the initial state the target is, along the world x axis. */
constexpr scalar_t kTargetDistance = 0.8;
/** [s] When the target is reached. */
constexpr scalar_t kTargetTime = 2.0;

// The perturbations of the points' states and inputs and of the final state, smooth in the element index i and the
// point index p: amplitude * sin(element phase * i + point phase * p) for a state, and mean + amplitude *
// cos(element phase * i + p) for an input.
constexpr scalar_t kStatePerturbationAmplitude = 0.01;
constexpr scalar_t kStatePerturbationElementPhase = 0.7;
constexpr scalar_t kStatePerturbationPointPhase = 1.3;
constexpr scalar_t kInputMean = 5.0;
constexpr scalar_t kInputPerturbationAmplitude = 3.0;
constexpr scalar_t kInputPerturbationElementPhase = 0.4;
constexpr scalar_t kFinalStatePerturbationAmplitude = 0.02;
constexpr scalar_t kFinalStatePerturbationElementPhase = 0.3;

/** The names of `collection`'s terms, sorted. */
template <typename CollectionT>
std::vector<std::string> termNames(const CollectionT& collection) {
  std::vector<std::string> names;
  for (const std::pair<const std::string, size_t>& entry : collection.getTermNameMap()) {
    names.push_back(entry.first);
  }
  std::sort(names.begin(), names.end());
  return names;
}

void dumpQuadratic(ValueDump& dump, absl::string_view path, const ScalarFunctionQuadraticApproximation& approximation) {
  dump.addDouble(joinPath(path, "f"), approximation.f);
  dump.addFingerprint(joinPath(path, "dfdx"), approximation.dfdx);
  dump.addDiagonal(joinPath(path, "dfdxx.diagonal"), approximation.dfdxx);
  dump.addFingerprint(joinPath(path, "dfdxx"), approximation.dfdxx);
  if (approximation.dfdu.size() > 0) {
    dump.addFingerprint(joinPath(path, "dfdu"), approximation.dfdu);
    dump.addDiagonal(joinPath(path, "dfduu.diagonal"), approximation.dfduu);
    dump.addFingerprint(joinPath(path, "dfduu"), approximation.dfduu);
    dump.addFingerprint(joinPath(path, "dfdux"), approximation.dfdux);
  }
}

void dumpLinear(ValueDump& dump, absl::string_view path, const VectorFunctionLinearApproximation& approximation) {
  dump.addFingerprint(joinPath(path, "f"), approximation.f);
  dump.addFingerprint(joinPath(path, "dfdx"), approximation.dfdx);
  if (approximation.dfdu.size() > 0) {
    dump.addFingerprint(joinPath(path, "dfdu"), approximation.dfdu);
  }
}

/** Every term of the state-input cost collection `collection` at (t, x, u): whether it is active, and its approximation. */
void dumpStateInputTerms(ValueDump& dump,
                         absl::string_view path,
                         StateInputCostCollection& collection,
                         scalar_t time,
                         const vector_t& state,
                         const vector_t& input,
                         const OptimalControlProblem& problem) {
  for (const std::string& name : termNames(collection)) {
    StateInputCost& term = collection.get(name);
    const std::string termPath = joinPath(path, name);
    dump.addBool(joinPath(termPath, "active"), term.isActive(time));
    if (term.isActive(time)) {
      dumpQuadratic(dump, termPath,
                    term.getQuadraticApproximation(time, state, input, *problem.targetTrajectoriesPtr, *problem.preComputationPtr));
    }
  }
}

/** Every term of the state cost collection `collection` at (t, x). */
void dumpStateTerms(ValueDump& dump,
                    absl::string_view path,
                    StateCostCollection& collection,
                    scalar_t time,
                    const vector_t& state,
                    const OptimalControlProblem& problem) {
  for (const std::string& name : termNames(collection)) {
    StateCost& term = collection.get(name);
    const std::string termPath = joinPath(path, name);
    dump.addBool(joinPath(termPath, "active"), term.isActive(time));
    if (term.isActive(time)) {
      dumpQuadratic(dump, termPath,
                    term.getQuadraticApproximation(time, state, *problem.targetTrajectoriesPtr, *problem.preComputationPtr));
    }
  }
}

/** The state of point `point`: the initial state, perturbed element by element. */
vector_t perturbedState(const vector_t& initialState, size_t point) {
  vector_t state = initialState;
  for (Eigen::Index i = 0; i < state.size(); ++i) {
    state(i) += kStatePerturbationAmplitude * std::sin(kStatePerturbationElementPhase * static_cast<scalar_t>(i) +
                                                       kStatePerturbationPointPhase * static_cast<scalar_t>(point));
  }
  return state;
}

/** The input of point `point`: positive, so that no contact force is at a barrier's kink. */
vector_t perturbedInput(size_t inputDim, size_t point) {
  vector_t input(static_cast<Eigen::Index>(inputDim));
  for (Eigen::Index i = 0; i < input.size(); ++i) {
    input(i) = kInputMean + kInputPerturbationAmplitude *
                                std::cos(kInputPerturbationElementPhase * static_cast<scalar_t>(i) + static_cast<scalar_t>(point));
  }
  return input;
}

}  // namespace

void dumpProblem(ValueDump& dump,
                 absl::string_view path,
                 const OptimalControlProblem& problem,
                 SwitchedModelReferenceManager& referenceManager,
                 const vector_t& initialState,
                 const MpcRobotModelBase<scalar_t>& robotModel) {
  // A copy, whose terms can be evaluated one by one (Collection::get() is non-const).
  OptimalControlProblem copy(problem);
  dump.addStrings(joinPath(path, "costPtr"), termNames(*copy.costPtr));
  dump.addStrings(joinPath(path, "stateCostPtr"), termNames(*copy.stateCostPtr));
  dump.addStrings(joinPath(path, "finalCostPtr"), termNames(*copy.finalCostPtr));
  dump.addStrings(joinPath(path, "softConstraintPtr"), termNames(*copy.softConstraintPtr));
  dump.addStrings(joinPath(path, "stateSoftConstraintPtr"), termNames(*copy.stateSoftConstraintPtr));
  dump.addStrings(joinPath(path, "finalSoftConstraintPtr"), termNames(*copy.finalSoftConstraintPtr));
  dump.addStrings(joinPath(path, "equalityConstraintPtr"), termNames(*copy.equalityConstraintPtr));
  dump.addStrings(joinPath(path, "stateEqualityConstraintPtr"), termNames(*copy.stateEqualityConstraintPtr));
  dump.addStrings(joinPath(path, "inequalityConstraintPtr"), termNames(*copy.inequalityConstraintPtr));

  // A walking schedule and a target 0.8 m ahead, as the MPC loop sets them before a solve.
  const ModeSchedule schedule(std::vector<scalar_t>(kWalkingEventTimes.begin(), kWalkingEventTimes.end()),
                              {ModeNumber::kStance, ModeNumber::kLf, ModeNumber::kStance, ModeNumber::kRf, ModeNumber::kStance,
                               ModeNumber::kLf, ModeNumber::kStance});
  vector_t targetState = initialState;
  vector3_t targetBasePosition = robotModel.getBasePosition(targetState);
  targetBasePosition.x() += kTargetDistance;
  robotModel.setBasePosition(targetState, targetBasePosition);
  const size_t inputDim = robotModel.getInputDim();
  const vector_t zeroInput = vector_t::Zero(static_cast<Eigen::Index>(inputDim));
  referenceManager.getGaitSchedule()->updateModeSchedule(schedule);
  referenceManager.setTargetTrajectories(TargetTrajectories({0.0, kTargetTime}, {initialState, targetState}, {zeroInput, zeroInput}));
  referenceManager.preSolverRun(/*initTime=*/0.0, /*finalTime=*/kFinalTime, initialState, ModeNumber::kStance);
  copy.targetTrajectoriesPtr = &referenceManager.getTargetTrajectories();

  for (size_t point = 0; point < kTimes.size(); ++point) {
    const scalar_t time = kTimes[point];
    const vector_t state = perturbedState(initialState, point);
    const vector_t input = perturbedInput(inputDim, point);
    const std::string at = joinPath(path, absl::StrCat("t=", time));
    dump.addMatrix(joinPath(at, "desiredState"), referenceManager.getDesiredState(*copy.targetTrajectoriesPtr, state, time));
    // The whole problem first: it requests every precomputation the terms read at (t, x, u).
    const ModelData model = approximateIntermediateLQ(copy, time, state, input);
    dumpQuadratic(dump, joinPath(at, "lq.cost"), model.cost);
    dumpLinear(dump, joinPath(at, "lq.dynamics"), model.dynamics);
    dumpLinear(dump, joinPath(at, "lq.stateInputEqConstraint"), model.stateInputEqConstraint);
    dumpStateInputTerms(dump, joinPath(at, "cost"), *copy.costPtr, time, state, input, copy);
    dumpStateInputTerms(dump, joinPath(at, "softConstraint"), *copy.softConstraintPtr, time, state, input, copy);
    dumpStateTerms(dump, joinPath(at, "stateCost"), *copy.stateCostPtr, time, state, copy);
    dumpStateTerms(dump, joinPath(at, "stateSoftConstraint"), *copy.stateSoftConstraintPtr, time, state, copy);
  }

  vector_t finalState = initialState;
  for (Eigen::Index i = 0; i < finalState.size(); ++i) {
    finalState(i) += kFinalStatePerturbationAmplitude * std::cos(kFinalStatePerturbationElementPhase * static_cast<scalar_t>(i));
  }
  const std::string at = joinPath(path, "final");
  const ModelData model = approximateFinalLQ(copy, kFinalTime, finalState);
  dumpQuadratic(dump, joinPath(at, "lq.cost"), model.cost);
  dumpStateTerms(dump, joinPath(at, "finalCost"), *copy.finalCostPtr, kFinalTime, finalState, copy);
  dumpStateTerms(dump, joinPath(at, "finalSoftConstraint"), *copy.finalSoftConstraintPtr, kFinalTime, finalState, copy);
}

}  // namespace ocs2::humanoid::config_dump
