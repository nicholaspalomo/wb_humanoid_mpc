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

#include "ocs2_core/Types.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_mpc_ipc/MpcMessageConversions.h"

namespace ocs2::humanoid::ipc {

/**
 * Evaluates the policy in use on the realtime thread: the nominal state, the input and the mode that
 * MRT_BASE::evaluatePolicy() computes, bit for bit, without a heap allocation and without logging.
 *
 * MRT_BASE::evaluatePolicy() is OCS2's (lib/ocs2), and does both: ControllerBase::computeInput() and
 * LinearInterpolation::interpolate() return by value, a LinearController's interpolated gain matrix included, and it
 * logs whenever the time is past the plan. This evaluator writes into the outputs it is given, and the gain into a
 * matrix it allocates once, at construction, for the model's dimensions. It evaluates the two controllers the MPC link
 * carries (MpcMessageConversions.h): a FeedforwardController or a LinearController, of exactly those types.
 *
 * Construct it off the realtime thread. evaluate() then allocates nothing while the outputs have the model's sizes (the
 * first call with outputs of other sizes resizes them, as an assignment would). Not thread-safe: one evaluator per
 * thread that evaluates.
 *
 *   mrt.updatePolicy();
 *   if (mrt.isActivePolicyCurrent()) evaluator.evaluate(mrt.getPolicy(), time, state, mpcState, mpcInput, mode);
 */
class RealtimePolicyEvaluator {
 public:
  enum class Outcome {
    /** mpcState, mpcInput and mode are written. */
    kEvaluated,
    /** The policy has no controller, or one of another type than FeedforwardController and LinearController. */
    kUnsupportedController,
    /**
     * The policy, or `state`, is not of the model's dimensions, or the policy is empty or inconsistent (trajectories of
     * other lengths than its time trajectory, a mode schedule without one mode more than event times).
     */
    kMismatchedPolicy,
  };

  /** Allocates the gain matrix of a LinearController of `dimensions`. */
  explicit RealtimePolicyEvaluator(const ModelDimensions& dimensions);

  /**
   * What MRT_BASE::evaluatePolicy(time, state, mpcState, mpcInput, mode) computes on `policy`, the same doubles. Before
   * the first node and past the last one the nearest node is held, as OCS2 holds it, and nothing is logged: the link's
   * watchdog makes the controller hold the robot past the end of the plan (RemoteMpcLink, HEALTH). Unless the outcome
   * is kEvaluated, the outputs are left as they were.
   */
  Outcome evaluate(
      const PrimalSolution& policy, scalar_t time, const vector_t& state, vector_t& mpcState, vector_t& mpcInput, size_t& mode);

 private:
  const ModelDimensions dimensions_;
  // The interpolated gain of a LinearController, inputDim x stateDim, which OCS2 allocates at every call.
  matrix_t gain_;
};

}  // namespace ocs2::humanoid::ipc
