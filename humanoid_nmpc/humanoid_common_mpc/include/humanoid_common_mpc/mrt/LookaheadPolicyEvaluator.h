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

#include <cstddef>

#include "absl/base/nullability.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_mpc_ipc/RealtimePolicyEvaluator.h"

namespace ocs2::humanoid {

/**
 * Evaluates the policy in use for an MRT joint controller that commands it a
 * lookahead delta ahead of its measurement: the measurement x is taken at time
 * t, and the command reaches the joints about one control period later.
 *
 * The plan is read at t + delta - the planned state x*(t + delta), which gives
 * the joint targets, the mode, and the feedforward input u(t + delta, x*(t +
 * delta)) - while the feedback correction is formed where the measurement is,
 * at t, against the plan at t:
 *   input = u(t + delta, x*(t + delta)) + [u(t, x) - u(t, x*(t))].
 * For a linear policy u(t, x) = u_ff(t) + K(t) (x (-) x*(t)) that is u_ff(t +
 * delta) + K(t) (x (-) x*(t)). Evaluating the whole law at t + delta on the
 * state measured at t, as the controllers did, gives u_ff(t + delta) + K(t +
 * delta) (x (-) x*(t + delta)) instead: on a moving plan a robot exactly on it
 * gets the feedback -K (x*(t + delta) (-) x*(t)) ~ -K xdot* delta, a correction
 * towards where the plan will be rather than none. Here a measurement exactly
 * on the plan (x equal to x*(t), bit for bit) gets exactly the feedforward: the
 * two evaluations at t are the same arithmetic on the same doubles. A
 * FeedforwardController's input is u_ff(t + delta).
 *
 * Every evaluation is the RealtimePolicyEvaluator's (four of them, on buffers
 * allocated once by the constructor), so evaluate() allocates nothing while its
 * outputs have the model's sizes, logs nothing and takes no lock. Not
 * thread-safe: one per control thread.
 */
class LookaheadPolicyEvaluator {
 public:
  using Outcome = ipc::RealtimePolicyEvaluator::Outcome;

  /**
   * Allocates the buffers for a model of `dimensions` whose state lives on
   * `stateManifold` (null: a flat state), which is retained and must outlive
   * the evaluator (RealtimePolicyEvaluator's contract).
   */
  LookaheadPolicyEvaluator(const ipc::ModelDimensions& dimensions,
                           const StateManifold* absl_nullable stateManifold);

  /**
   * Writes x*(measurementTime + lookahead) into `plannedState`, the input of
   * the class comment into `input` and the mode at measurementTime + lookahead
   * into `mode`, for the state `measuredState` measured at `measurementTime`.
   * Unless the outcome is kEvaluated (RealtimePolicyEvaluator::evaluate()'s
   * outcome for `policy`), the outputs are left as they were.
   */
  Outcome evaluate(const PrimalSolution& policy,
                   scalar_t measurementTime,
                   scalar_t lookahead,
                   const vector_t& measuredState,
                   vector_t& plannedState,
                   vector_t& input,
                   size_t& mode);

 private:
  ipc::RealtimePolicyEvaluator evaluator_;
  vector_t planStateNow_;          ///< x*(t)
  vector_t inputAtMeasurement_;    ///< u(t, x)
  vector_t inputOnPlanNow_;        ///< u(t, x*(t))
  vector_t lookaheadPlanState_;    ///< x*(t + delta)
  vector_t lookaheadInputOnPlan_;  ///< u(t + delta, x*(t + delta))
  vector_t discardedState_;  ///< the planned state of the evaluations whose
                             ///< state output is not needed
};

}  // namespace ocs2::humanoid
