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
#include "ocs2_core/control/LinearController.h"
#include "ocs2_core/control/ManifoldLinearController.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/misc/LinearInterpolation.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_mpc_ipc/MpcMessageConversions.h"

namespace ocs2::humanoid::ipc {

/**
 * Evaluates the policy in use on the realtime thread: the nominal state, the input and the mode that
 * MRT_BASE::evaluatePolicy() computes, bit for bit, without a heap allocation and without logging.
 *
 * MRT_BASE::evaluatePolicy() is OCS2's (lib/ocs2), and does both: ControllerBase::computeInput() and
 * LinearInterpolation::interpolate() return by value, a LinearController's interpolated gain matrix included, and it
 * logs whenever the time is past the plan. This evaluator writes into the outputs it is given, and its intermediate
 * values into buffers it allocates once, at construction, for the model's dimensions. It evaluates the controllers the
 * MPC link carries (MpcMessageConversions.h), of exactly those types:
 *   - on a flat state (no state manifold): a FeedforwardController or a LinearController, with the planned state
 *     interpolated linearly, as MRT_BASE::evaluatePolicy() does without a manifold;
 *   - on a state manifold: a FeedforwardController or a ManifoldLinearController, with the planned state interpolated
 *     along the manifold, as MRT_BASE::evaluatePolicy() does after MRT_BASE::setStateManifold(), through the manifold's
 *     allocation-free StateManifold::differenceInto() and interpolateInto(). A LinearController is refused there, whose
 *     gains would act on the stored state rather than on its tangent; so is a ManifoldLinearController on a flat state.
 *     Neither is left to MRT_BASE::evaluatePolicy(), which allocates.
 *
 * Construct it off the realtime thread. evaluate() then allocates nothing while the outputs have the model's sizes (the
 * first call with outputs of other sizes resizes them, as an assignment would), checks nothing it would abort on and
 * takes no lock. Not thread-safe: one evaluator per thread that evaluates.
 *
 *   mrt.updatePolicy();
 *   if (mrt.isActivePolicyCurrent()) evaluator.evaluate(mrt.getPolicy(), time, state, mpcState, mpcInput, mode);
 */
class RealtimePolicyEvaluator {
 public:
  enum class Outcome {
    /** mpcState, mpcInput and mode are written. */
    kEvaluated,
    /**
     * The policy has no controller, or one of a type this evaluator does not evaluate on its state (see the class
     * comment): another type than FeedforwardController, LinearController and ManifoldLinearController, a
     * LinearController on a state manifold or a ManifoldLinearController on a flat state.
     */
    kUnsupportedController,
    /**
     * The policy, or `state`, is not of the model's dimensions (or a ManifoldLinearController's gains not of the
     * manifold's tangent dimension), or the policy is empty or inconsistent (trajectories of other lengths than its time
     * trajectory, a mode schedule without one mode more than event times).
     */
    kMismatchedPolicy,
  };

  /**
   * Allocates the buffers for a model of `dimensions` whose state lives on `stateManifold`, or is flat when it is null.
   * The manifold is retained and must outlive the evaluator; its ambient dimension must be dimensions.stateDim, which
   * the constructor checks once. A caller that holds it as the std::shared_ptr OCS2 hands around (MRT_BASE::
   * getStateManifold()) passes .get().
   */
  explicit RealtimePolicyEvaluator(const ModelDimensions& dimensions, const StateManifold* absl_nullable stateManifold = nullptr);

  /**
   * What MRT_BASE::evaluatePolicy(time, state, mpcState, mpcInput, mode) computes on `policy`, the same doubles. Before
   * the first node and past the last one the nearest node is held, as OCS2 holds it, and nothing is logged: the link's
   * watchdog makes the controller hold the robot past the end of the plan (RemoteMpcLink, HEALTH). Unless the outcome
   * is kEvaluated, the outputs are left as they were.
   */
  Outcome evaluate(
      const PrimalSolution& policy, scalar_t time, const vector_t& state, vector_t& mpcState, vector_t& mpcInput, size_t& mode);

 private:
  /**
   * LinearController::computeInput(time, state) into `mpcInput`, its interpolated gain into gain_. False, without
   * writing, when the controller is inconsistent or its nodes at `time` are not of the model's dimensions.
   */
  bool computeLinearInput(const LinearController& controller, scalar_t time, const vector_t& state, vector_t& mpcInput);

  /**
   * ManifoldLinearController::computeInput(time, state) into `mpcInput`, through delta_ and nodeInput_. False, without
   * writing, when the controller is inconsistent or its nodes at `time` are not of the model's dimensions (its gains of
   * the manifold's tangent dimension).
   */
  bool computeManifoldInput(const ManifoldLinearController& controller, scalar_t time, const vector_t& state, vector_t& mpcInput);

  /** The law u*_k + K_k (state (-) xbar_k) of node k of `controller`, into `input`. */
  void computeNodeInput(const ManifoldLinearController& controller, size_t node, const vector_t& state, vector_t& input);

  /** interpolateStateTrajectory(stateManifold_, ...) of the policy's state trajectory at `segment`, into `mpcState`. */
  void interpolatePlanState(const LinearInterpolation::index_alpha_t& segment, const vector_array_t& stateTrajectory, vector_t& mpcState);

  const ModelDimensions dimensions_;
  // Retained, owned by the caller; null for a flat state.
  const StateManifold* absl_nullable stateManifold_;
  // The interpolated gain of a LinearController, inputDim x stateDim, which OCS2 allocates at every call. Empty on a
  // state manifold, where a ManifoldLinearController blends the laws of its nodes, each with its own gain.
  matrix_t gain_;
  // On a state manifold: x (-) xbar_k at a node, of the manifold's tangent dimension; empty on a flat state.
  vector_t delta_;
  // On a state manifold: the law of the later node of a segment, inputDim.
  vector_t nodeInput_;
  // On a state manifold: the planned state interpolated along it, stateDim, before it is copied into the output.
  vector_t planState_;
};

}  // namespace ocs2::humanoid::ipc
