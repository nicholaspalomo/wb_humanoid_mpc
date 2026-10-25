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
#include "absl/status/status.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/cost/StateInputCost.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/TuningLayoutDeviation.h"

namespace ocs2::humanoid {

/** The references a TuningQuadraticStateInputCost measures one node's state and
 * input from. */
struct StateInputReference {
  /** x_ref, the robot model's getStateDim() entries. */
  vector_t state;
  /** u_ref, the robot model's getInputDim() entries. */
  vector_t input;
};

/**
 * The quadratic state-input cost of the task file's tuning layout,
 *   l(x, u) = 1/2 r' Q r + 1/2 (u - u_ref)' R (u - u_ref),   r =
 * TuningLayoutDeviation(x, x_ref), with Q of state_weights (n_t square, the
 * robot model's getStateTangentDim()) and R of input_weights, both written by
 * setStateWeights() and setInputWeights() at start-up and on every live reload,
 * so that the deviation is the one map from the Euler-written rows to the
 * residual in both paths. Its derivatives are those of the Gauss-Newton model
 * (cost/TuningGaussNewton.h) with respect to the stored state. On a layout of
 * differences only (no rotation block) they are OCS2's QuadraticStateInputCost
 * of x - x_ref, bit for bit. With the orientation rows its Gauss-Newton Hessian
 * is that cost's at a level state (zero pitch and roll of the state) against
 * any reference, and its value and gradient are that cost's there only against
 * a reference tilted about at most one axis (a mixed tilt shifts the yaw row by
 * -o(theta_r, phi_r) and the tilt rows at third order). The whole body's
 * angular-velocity rows keep that cost's value at a level state against a
 * reference whose Euler rate is a world yaw rate (every shipped one); its
 * Hessian there only where, in addition, the yaw and pitch rate errors vanish
 * (the roll-rate error is free), and its gradient only where the orientation
 * slots' extra term B' Q_w v = (0, -q_x v_yaw v_roll, (q_y - q_z) v_yaw
 * v_pitch) vanishes, v the Euler-rate error and Q_w = diag(q_z, q_y, q_x):
 * without a yaw-rate error, for example (TuningLayoutDeviation.h, design
 * section 4.4).
 *
 * A base class: a derived term says where the references come from
 * (getReference()) and builds itself through a static Create() that sets both
 * weights. The robot model must outlive it. Like every OCS2 term it is cloned
 * for each solver thread; its const functions are safe to call concurrently,
 * the setters are not.
 */
class TuningQuadraticStateInputCost : public StateInputCost {
 public:
  ~TuningQuadraticStateInputCost() override = default;
  TuningQuadraticStateInputCost& operator=(
      const TuningQuadraticStateInputCost&) = delete;
  TuningQuadraticStateInputCost(TuningQuadraticStateInputCost&&) = delete;
  TuningQuadraticStateInputCost& operator=(TuningQuadraticStateInputCost&&) =
      delete;

  /** l(x, u) at `time`, the references from getReference(). */
  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const vector_t& input,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComputation) const final;

  /** The Gauss-Newton model of l at (state, input): value, J' Q r, J' Q J, R (u
   * - u_ref), R and a zero dfdux. */
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const TargetTrajectories& targetTrajectories,
      const PreComputation& preComputation) const final;

  /**
   * Replaces Q. InvalidArgument, with Q left as it was, unless `stateWeights`
   * is getTuningDim() square, finite and symmetric
   * (validateQuadraticWeights()).
   */
  absl::Status setStateWeights(matrix_t stateWeights);

  /** Replaces R. InvalidArgument, with R left as it was, unless `inputWeights`
   * is the input dimension square, finite and symmetric. */
  absl::Status setInputWeights(matrix_t inputWeights);

  /** Q, getTuningDim() square once set. */
  const matrix_t& getStateWeights() const { return stateWeights_; }
  /** R, the input dimension square once set. */
  const matrix_t& getInputWeights() const { return inputWeights_; }

  /** Rows of the residual and of Q: the robot model's getStateTangentDim(). */
  size_t getTuningDim() const { return deviation_.getTuningDim(); }

 protected:
  /**
   * The cost of `mpcRobotModel`'s tuning layout and input; its
   * TuningLayoutDeviation retains the model. The weights are empty until the
   * derived Create() sets them, which it does before the cost is used.
   */
  explicit TuningQuadraticStateInputCost(
      const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel);
  TuningQuadraticStateInputCost(const TuningQuadraticStateInputCost& other) =
      default;

  /** The references of the node at `time`, (state, input) the point the cost is
   * evaluated at. */
  virtual StateInputReference getReference(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const TargetTrajectories& targetTrajectories,
      const PreComputation& preComputation) const = 0;

 private:
  TuningLayoutDeviation deviation_;
  // Entries of the input, the size of R.
  size_t inputDim_ = 0;
  // Q and R; empty only until the derived Create() sets them.
  matrix_t stateWeights_;
  matrix_t inputWeights_;
};

}  // namespace ocs2::humanoid
