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

#include "humanoid_common_mpc/cost/TuningQuadraticStateCost.h"

#include <cstddef>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/cost/TuningGaussNewton.h"
#include "humanoid_common_mpc/orientation/TuningLayoutDeviation.h"

namespace ocs2::humanoid {

TuningQuadraticStateCost::TuningQuadraticStateCost(const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel)
    : mpcRobotModelPtr_(ABSL_DIE_IF_NULL(mpcRobotModel)), deviation_(mpcRobotModel) {}

scalar_t TuningQuadraticStateCost::getValue(scalar_t time,
                                            const vector_t& state,
                                            const TargetTrajectories& targetTrajectories,
                                            const PreComputation& preComputation) const {
  const vector_t residual = deviation_.evaluate(state, getReference(time, state, targetTrajectories, preComputation));
  // OCS2's QuadraticStateCost::getValue.
  return 0.5 * residual.dot(stateWeights_ * residual);
}

ScalarFunctionQuadraticApproximation TuningQuadraticStateCost::getQuadraticApproximation(scalar_t time,
                                                                                         const vector_t& state,
                                                                                         const TargetTrajectories& targetTrajectories,
                                                                                         const PreComputation& preComputation) const {
  const TuningDeviationApproximation deviation = deviation_.linearize(state, getReference(time, state, targetTrajectories, preComputation));
  const size_t stateDim = static_cast<size_t>(state.size());

  ScalarFunctionQuadraticApproximation approximation;
  approximation.dfdxx = tuningHessian(deviation, stateWeights_, stateDim);
  approximation.dfdx = tuningGradient(deviation, stateWeights_, stateDim);
  // On the identity Jacobian dfdx is Q r, and the value is taken from it as OCS2's QuadraticStateCost takes it.
  approximation.f = isIdentityTuningJacobian(deviation, stateDim) ? 0.5 * deviation.residual.dot(approximation.dfdx)
                                                                  : 0.5 * deviation.residual.dot(stateWeights_ * deviation.residual);
  return approximation;
}

absl::Status TuningQuadraticStateCost::setStateWeights(matrix_t stateWeights) {
  RETURN_IF_ERROR(validateQuadraticWeights(stateWeights, deviation_.getTuningDim(), "[TuningQuadraticStateCost] the state weights Q"));
  stateWeights_ = std::move(stateWeights);
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid
