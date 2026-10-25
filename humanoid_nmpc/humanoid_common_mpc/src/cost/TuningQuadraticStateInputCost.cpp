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

#include "humanoid_common_mpc/cost/TuningQuadraticStateInputCost.h"

#include <cstddef>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/cost/TuningGaussNewton.h"
#include "humanoid_common_mpc/orientation/TuningLayoutDeviation.h"

namespace ocs2::humanoid {

// deviation_, initialized first, checks that the model is not null before the
// input dimension is read off it.
TuningQuadraticStateInputCost::TuningQuadraticStateInputCost(
    const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel)
    : deviation_(mpcRobotModel), inputDim_(mpcRobotModel->getInputDim()) {}

scalar_t TuningQuadraticStateInputCost::getValue(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const TargetTrajectories& targetTrajectories,
    const PreComputation& preComputation) const {
  const StateInputReference reference =
      getReference(time, state, input, targetTrajectories, preComputation);
  const vector_t residual = deviation_.evaluate(state, reference.state);
  const vector_t inputDeviation = input - reference.input;
  // OCS2's QuadraticStateInputCost::getValue, term for term.
  return 0.5 * residual.dot(stateWeights_ * residual) +
         0.5 * inputDeviation.dot(inputWeights_ * inputDeviation);
}

ScalarFunctionQuadraticApproximation
TuningQuadraticStateInputCost::getQuadraticApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const TargetTrajectories& targetTrajectories,
    const PreComputation& preComputation) const {
  const StateInputReference reference =
      getReference(time, state, input, targetTrajectories, preComputation);
  const TuningDeviationApproximation deviation =
      deviation_.linearize(state, reference.state);
  const vector_t inputDeviation = input - reference.input;
  const size_t stateDim = static_cast<size_t>(state.size());

  ScalarFunctionQuadraticApproximation approximation;
  approximation.dfdxx = tuningHessian(deviation, stateWeights_, stateDim);
  approximation.dfduu = inputWeights_;
  approximation.dfdx = tuningGradient(deviation, stateWeights_, stateDim);
  approximation.dfdu.noalias() = inputWeights_ * inputDeviation;
  // On the identity Jacobian dfdx is Q r, and the value is taken from it as
  // OCS2's QuadraticStateInputCost takes it.
  const scalar_t stateValue =
      isIdentityTuningJacobian(deviation, stateDim)
          ? 0.5 * deviation.residual.dot(approximation.dfdx)
          : 0.5 * deviation.residual.dot(stateWeights_ * deviation.residual);
  approximation.f = stateValue + 0.5 * inputDeviation.dot(approximation.dfdu);
  approximation.dfdux.setZero(input.size(), state.size());
  return approximation;
}

absl::Status TuningQuadraticStateInputCost::setStateWeights(
    matrix_t stateWeights) {
  RETURN_IF_ERROR(validateQuadraticWeights(
      stateWeights, deviation_.getTuningDim(),
      "[TuningQuadraticStateInputCost] the state weights Q"));
  stateWeights_ = std::move(stateWeights);
  return absl::OkStatus();
}

absl::Status TuningQuadraticStateInputCost::setInputWeights(
    matrix_t inputWeights) {
  RETURN_IF_ERROR(validateQuadraticWeights(
      inputWeights, inputDim_,
      "[TuningQuadraticStateInputCost] the input weights R"));
  inputWeights_ = std::move(inputWeights);
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid
