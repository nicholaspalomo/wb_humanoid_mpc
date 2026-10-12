/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateInputCost.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * The quadratic state-input regularization of the MPC around the target trajectories (state_input_quadratic_cost), with
 * the input reference that carries the robot's weight on the feet in contact: a TuningQuadraticStateInputCost whose
 * references are the reference manager's getDesiredState() and getDesiredInput().
 *
 * The reference manager and the robot model must outlive it. Like every OCS2 term it is cloned for each solver thread;
 * its const functions are safe to call concurrently, the weight setters are not.
 */
class StateInputQuadraticCost final : public TuningQuadraticStateInputCost {
 public:
  /**
   * The cost of `stateWeights` (Q, the tuning layout's: getStateTangentDim() square) and `inputWeights` (R, getInputDim()
   * square) on `mpcRobotModel`, both retained with `referenceManager`. InvalidArgument when a weight matrix is refused
   * (TuningQuadraticStateInputCost::setStateWeights(), setInputWeights()).
   */
  static absl::StatusOr<std::unique_ptr<StateInputQuadraticCost>> Create(matrix_t stateWeights,
                                                                         matrix_t inputWeights,
                                                                         const SwitchedModelReferenceManager* absl_nonnull referenceManager,
                                                                         const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel);

  ~StateInputQuadraticCost() override = default;
  StateInputQuadraticCost& operator=(const StateInputQuadraticCost&) = delete;
  StateInputQuadraticCost(StateInputQuadraticCost&&) = delete;
  StateInputQuadraticCost& operator=(StateInputQuadraticCost&&) = delete;
  StateInputQuadraticCost* absl_nonnull clone() const override { return new StateInputQuadraticCost(*this); }

 private:
  StateInputQuadraticCost(const SwitchedModelReferenceManager* absl_nonnull referenceManager,
                          const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel);
  StateInputQuadraticCost(const StateInputQuadraticCost& other) = default;

  StateInputReference getReference(scalar_t time,
                                   const vector_t& state,
                                   const vector_t& input,
                                   const TargetTrajectories& targetTrajectories,
                                   const PreComputation& preComputation) const override;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
};

}  // namespace ocs2::humanoid
