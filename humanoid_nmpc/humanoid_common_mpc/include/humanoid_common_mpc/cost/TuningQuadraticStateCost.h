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
#include "ocs2_core/cost/StateCost.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/TuningLayoutDeviation.h"

namespace ocs2::humanoid {

/**
 * The quadratic state-only cost of the task file's tuning layout, l(x) = 1/2 r'
 * Q r with r = TuningLayoutDeviation(x, x_ref): the terminal counterpart of
 * TuningQuadraticStateInputCost, Q of final_state_weights (times
 * terminal_cost_scaling) written by setStateWeights() at start-up and on every
 * live reload. On a layout of differences only OCS2's QuadraticStateCost of x -
 * x_ref, bit for bit; with the rotation rows, that cost's value, gradient and
 * Hessian at a level state only under the conditions
 * TuningQuadraticStateInputCost states (the orientation rows' Hessian against
 * any reference, the angular-velocity rows' only without yaw and pitch rate
 * errors).
 *
 * A base class: a derived term says where the reference comes from
 * (getReference()) and builds itself through a static Create() that sets Q. The
 * robot model must outlive it. Like every OCS2 term it is cloned for each
 * solver thread; its const functions are safe to call concurrently, the setter
 * is not.
 */
class TuningQuadraticStateCost : public StateCost {
 public:
  ~TuningQuadraticStateCost() override = default;
  TuningQuadraticStateCost& operator=(const TuningQuadraticStateCost&) = delete;
  TuningQuadraticStateCost(TuningQuadraticStateCost&&) = delete;
  TuningQuadraticStateCost& operator=(TuningQuadraticStateCost&&) = delete;

  /** l(x) at `time`, the reference from getReference(). */
  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComputation) const final;

  /** The Gauss-Newton model of l at `state`: value, J' Q r and J' Q J. */
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(
      scalar_t time,
      const vector_t& state,
      const TargetTrajectories& targetTrajectories,
      const PreComputation& preComputation) const final;

  /**
   * Replaces Q. InvalidArgument, with Q left as it was, unless `stateWeights`
   * is getTuningDim() square, finite and symmetric
   * (validateQuadraticWeights()).
   */
  absl::Status setStateWeights(matrix_t stateWeights);

  /** Q, getTuningDim() square once set. */
  const matrix_t& getStateWeights() const { return stateWeights_; }

  /** Rows of the residual and of Q: the robot model's getStateTangentDim(). */
  size_t getTuningDim() const { return deviation_.getTuningDim(); }

 protected:
  /**
   * The cost of `mpcRobotModel`'s tuning layout, which is retained; Q is empty
   * until the derived Create() sets it, which it does before the cost is used.
   */
  explicit TuningQuadraticStateCost(
      const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel);
  TuningQuadraticStateCost(const TuningQuadraticStateCost& other) = default;

  /** The robot model the cost is built on. */
  const MpcRobotModelBase<scalar_t>& mpcRobotModel() const {
    return *mpcRobotModelPtr_;
  }

  /** The reference state x_ref of the node at `time`, `state` the point the
   * cost is evaluated at. */
  virtual vector_t getReference(scalar_t time,
                                const vector_t& state,
                                const TargetTrajectories& targetTrajectories,
                                const PreComputation& preComputation) const = 0;

 private:
  const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModelPtr_;
  TuningLayoutDeviation deviation_;
  // Q; empty only until the derived Create() sets it.
  matrix_t stateWeights_;
};

}  // namespace ocs2::humanoid
