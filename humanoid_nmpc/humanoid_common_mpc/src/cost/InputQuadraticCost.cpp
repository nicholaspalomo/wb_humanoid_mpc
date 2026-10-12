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

#include "humanoid_common_mpc/cost/InputQuadraticCost.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/die_if_null.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/cost/TuningGaussNewton.h"

namespace ocs2::humanoid {

absl::StatusOr<std::unique_ptr<InputQuadraticCost>> InputQuadraticCost::Create(
    matrix_t inputWeights,
    const SwitchedModelReferenceManager* absl_nonnull referenceManager,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  // The constructor is private, so std::make_unique cannot reach it.
  std::unique_ptr<InputQuadraticCost> cost = absl::WrapUnique(new InputQuadraticCost(referenceManager, mpcRobotModel.getInputDim()));
  RETURN_IF_ERROR(cost->setInputWeights(std::move(inputWeights)));
  return cost;
}

InputQuadraticCost::InputQuadraticCost(const SwitchedModelReferenceManager* absl_nonnull referenceManager, size_t inputDim)
    : referenceManagerPtr_(ABSL_DIE_IF_NULL(referenceManager)), inputDim_(inputDim) {}

vector_t InputQuadraticCost::inputDeviation(scalar_t time,
                                            const vector_t& state,
                                            const vector_t& input,
                                            const TargetTrajectories& targetTrajectories) const {
  // The contact-force reference, from the reference manager rather than computed here. It is still weight compensation
  // and nothing else on every robot shipped in this repository - getDesiredInput() makes exactly the call this line
  // used to make when no wrench heuristic is listed, so this is a no-op refactor there - but it is now the one place
  // that reference is decided, which is what lets Bledt's H_f heuristics shape it
  // (humanoid_nmpc/docs/locomotion_heuristics/README.md). The reference manager is also the only object that knows
  // the contact flags at an arbitrary node time, which is why the reference cannot live in targetTrajectories: the
  // stance set changes several times inside one horizon and that trajectory's knots are placed by the command (three on
  // a straight walk).
  return input - referenceManagerPtr_->getDesiredInput(targetTrajectories, state, time);
}

scalar_t InputQuadraticCost::getValue(scalar_t time,
                                      const vector_t& state,
                                      const vector_t& input,
                                      const TargetTrajectories& targetTrajectories,
                                      const PreComputation& /*preComputation*/) const {
  const vector_t deviation = inputDeviation(time, state, input, targetTrajectories);
  return 0.5 * deviation.dot(inputWeights_ * deviation);
}

ScalarFunctionQuadraticApproximation InputQuadraticCost::getQuadraticApproximation(scalar_t time,
                                                                                   const vector_t& state,
                                                                                   const vector_t& input,
                                                                                   const TargetTrajectories& targetTrajectories,
                                                                                   const PreComputation& /*preComputation*/) const {
  const vector_t deviation = inputDeviation(time, state, input, targetTrajectories);
  ScalarFunctionQuadraticApproximation approximation;
  approximation.dfdxx.setZero(state.size(), state.size());
  approximation.dfduu = inputWeights_;
  approximation.dfdx.setZero(state.size());
  approximation.dfdu.noalias() = inputWeights_ * deviation;
  approximation.f = 0.5 * deviation.dot(approximation.dfdu);
  approximation.dfdux.setZero(input.size(), state.size());
  return approximation;
}

absl::Status InputQuadraticCost::setInputWeights(matrix_t inputWeights) {
  RETURN_IF_ERROR(validateQuadraticWeights(inputWeights, inputDim_, "[InputQuadraticCost] the input weights R"));
  inputWeights_ = std::move(inputWeights);
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid
