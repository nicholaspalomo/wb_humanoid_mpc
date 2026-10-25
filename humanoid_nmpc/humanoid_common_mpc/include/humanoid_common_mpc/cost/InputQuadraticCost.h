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
#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/cost/StateInputCost.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * The quadratic input effort cost (input_quadratic_cost), 1/2 (u - u_ref)' R (u
 * - u_ref), with the reference manager's getDesiredInput() as u_ref.
 * Input-only: it holds no state weight, and its derivatives with respect to the
 * state are zeros of the evaluated state's size, so that no state-sized matrix
 * of it has to follow the state's layout.
 *
 * The reference manager must outlive it. Like every OCS2 term it is cloned for
 * each solver thread; its const functions are safe to call concurrently,
 * setInputWeights() is not.
 */
class InputQuadraticCost final : public StateInputCost {
 public:
  /**
   * The cost of `inputWeights` (R, `mpcRobotModel`'s getInputDim() square)
   * against `referenceManager`'s input reference; the reference manager is
   * retained, the model is not. InvalidArgument when R is refused
   * (setInputWeights()).
   */
  static absl::StatusOr<std::unique_ptr<InputQuadraticCost>> Create(
      matrix_t inputWeights,
      const SwitchedModelReferenceManager* absl_nonnull referenceManager,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel);

  ~InputQuadraticCost() override = default;
  InputQuadraticCost& operator=(const InputQuadraticCost&) = delete;
  InputQuadraticCost(InputQuadraticCost&&) = delete;
  InputQuadraticCost& operator=(InputQuadraticCost&&) = delete;
  InputQuadraticCost* absl_nonnull clone() const override {
    return new InputQuadraticCost(*this);
  }

  /** 1/2 (u - u_ref)' R (u - u_ref). */
  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const vector_t& input,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComputation) const override;

  /** The value, R (u - u_ref) and R; zero dfdx, dfdxx and dfdux of `state`'s
   * size. */
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const TargetTrajectories& targetTrajectories,
      const PreComputation& preComputation) const override;

  /**
   * Replaces R. InvalidArgument, with R left as it was, unless `inputWeights`
   * is the input dimension square, finite and symmetric
   * (validateQuadraticWeights()).
   */
  absl::Status setInputWeights(matrix_t inputWeights);

  /** R, the input dimension square. */
  const matrix_t& getInputWeights() const { return inputWeights_; }

 private:
  InputQuadraticCost(
      const SwitchedModelReferenceManager* absl_nonnull referenceManager,
      size_t inputDim);
  InputQuadraticCost(const InputQuadraticCost& other) = default;

  /** u - u_ref, u_ref the reference manager's desired input at `time`. */
  vector_t inputDeviation(scalar_t time,
                          const vector_t& state,
                          const vector_t& input,
                          const TargetTrajectories& targetTrajectories) const;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  // Entries of the input, the size of R.
  size_t inputDim_ = 0;
  // R; empty only until Create() sets it.
  matrix_t inputWeights_;
};

}  // namespace ocs2::humanoid
