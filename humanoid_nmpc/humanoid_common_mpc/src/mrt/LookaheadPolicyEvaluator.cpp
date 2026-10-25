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

#include "humanoid_common_mpc/mrt/LookaheadPolicyEvaluator.h"

#include <cstddef>

#include "absl/base/nullability.h"

namespace ocs2::humanoid {

LookaheadPolicyEvaluator::LookaheadPolicyEvaluator(
    const ipc::ModelDimensions& dimensions,
    const StateManifold* absl_nullable stateManifold)
    : evaluator_(dimensions, stateManifold),
      planStateNow_(
          vector_t::Zero(static_cast<Eigen::Index>(dimensions.stateDim))),
      inputAtMeasurement_(
          vector_t::Zero(static_cast<Eigen::Index>(dimensions.inputDim))),
      inputOnPlanNow_(
          vector_t::Zero(static_cast<Eigen::Index>(dimensions.inputDim))),
      lookaheadPlanState_(
          vector_t::Zero(static_cast<Eigen::Index>(dimensions.stateDim))),
      lookaheadInputOnPlan_(
          vector_t::Zero(static_cast<Eigen::Index>(dimensions.inputDim))),
      discardedState_(
          vector_t::Zero(static_cast<Eigen::Index>(dimensions.stateDim))) {}

LookaheadPolicyEvaluator::Outcome LookaheadPolicyEvaluator::evaluate(
    const PrimalSolution& policy,
    scalar_t measurementTime,
    scalar_t lookahead,
    const vector_t& measuredState,
    vector_t& plannedState,
    vector_t& input,
    size_t& mode) {
  size_t discardedMode = 0;
  // At the measurement: x*(t) and u(t, x). The first evaluation checks the
  // policy and the state; the others evaluate the same policy on states of the
  // same size, so their outcome is the same.
  Outcome outcome =
      evaluator_.evaluate(policy, measurementTime, measuredState, planStateNow_,
                          inputAtMeasurement_, discardedMode);
  if (outcome != Outcome::kEvaluated) return outcome;
  // u(t, x*(t)): the same law on the plan's own state at t.
  outcome =
      evaluator_.evaluate(policy, measurementTime, planStateNow_,
                          discardedState_, inputOnPlanNow_, discardedMode);
  if (outcome != Outcome::kEvaluated) return outcome;
  // x*(t + delta) (the state argument only enters the input, which the next
  // evaluation recomputes), then the feedforward u(t + delta, x*(t + delta)) on
  // it.
  const scalar_t lookaheadTime = measurementTime + lookahead;
  size_t lookaheadMode = 0;
  outcome = evaluator_.evaluate(policy, lookaheadTime, planStateNow_,
                                lookaheadPlanState_, lookaheadInputOnPlan_,
                                lookaheadMode);
  if (outcome != Outcome::kEvaluated) return outcome;
  outcome = evaluator_.evaluate(policy, lookaheadTime, lookaheadPlanState_,
                                discardedState_, lookaheadInputOnPlan_,
                                lookaheadMode);
  if (outcome != Outcome::kEvaluated) return outcome;

  plannedState = lookaheadPlanState_;
  input = lookaheadInputOnPlan_ + (inputAtMeasurement_ - inputOnPlanNow_);
  mode = lookaheadMode;
  return Outcome::kEvaluated;
}

}  // namespace ocs2::humanoid
