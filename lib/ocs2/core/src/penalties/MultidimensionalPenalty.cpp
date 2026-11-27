/******************************************************************************
Copyright (c) 2017, Farbod Farshidian. All rights reserved.

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

#include <cassert>

#include <ocs2_core/penalties/MultidimensionalPenalty.h>

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
MultidimensionalPenalty::MultidimensionalPenalty(std::vector<std::unique_ptr<PenaltyBase>> penaltyPtrArray)
    : penaltyPtrArray_(std::move(penaltyPtrArray)) {
  if (penaltyPtrArray_.empty()) {
    throw std::runtime_error("[MultidimensionalPenalty::MultidimensionalPenalty] The penalty array cannot be empty!");
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
MultidimensionalPenalty::MultidimensionalPenalty(std::unique_ptr<PenaltyBase> penaltyPtr) {
  penaltyPtrArray_.push_back(std::move(penaltyPtr));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
MultidimensionalPenalty::MultidimensionalPenalty(const MultidimensionalPenalty& other) {
  for (const auto& penalty : other.penaltyPtrArray_) {
    penaltyPtrArray_.emplace_back(penalty->clone());
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t MultidimensionalPenalty::getValue(scalar_t t, const vector_t& h) const {
  const auto numConstraints = h.rows();
  assert(penaltyPtrArray_.size() == 1 || penaltyPtrArray_.size() == numConstraints);

  scalar_t penalty = 0;
  for (size_t i = 0; i < static_cast<size_t>(numConstraints); i++) {
    const auto& penaltyTerm = (penaltyPtrArray_.size() == 1) ? penaltyPtrArray_[0] : penaltyPtrArray_[i];
    penalty += penaltyTerm->getValue(t, h(i));
  }

  return penalty;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionQuadraticApproximation MultidimensionalPenalty::getQuadraticApproximation(
    scalar_t t, const VectorFunctionLinearApproximation& h) const {
  const auto stateDim = h.dfdx.cols();
  const auto inputDim = h.dfdu.cols();

  scalar_t penaltyValue = 0.0;
  vector_t penaltyDerivative, penaltySecondDerivative;
  std::tie(penaltyValue, penaltyDerivative, penaltySecondDerivative) = getPenaltyValue1stDev2ndDev(t, h.f);
  const matrix_t penaltySecondDev_dhdx = penaltySecondDerivative.asDiagonal() * h.dfdx;

  // to make sure that dfdux in the state-only case has a right size
  ScalarFunctionQuadraticApproximation penaltyApproximation(stateDim, inputDim);

  penaltyApproximation.f = penaltyValue;
  penaltyApproximation.dfdx.noalias() = h.dfdx.transpose() * penaltyDerivative;
  penaltyApproximation.dfdxx.noalias() = h.dfdx.transpose() * penaltySecondDev_dhdx;
  if (inputDim > 0) {
    penaltyApproximation.dfdu.noalias() = h.dfdu.transpose() * penaltyDerivative;
    penaltyApproximation.dfdux.noalias() = h.dfdu.transpose() * penaltySecondDev_dhdx;
    penaltyApproximation.dfduu.noalias() = h.dfdu.transpose() * penaltySecondDerivative.asDiagonal() * h.dfdu;
  }

  return penaltyApproximation;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionQuadraticApproximation MultidimensionalPenalty::getQuadraticApproximation(
    scalar_t t, const VectorFunctionQuadraticApproximation& h) const {
  const auto stateDim = h.dfdx.cols();
  const auto inputDim = h.dfdu.cols();
  const auto numConstraints = h.f.rows();

  scalar_t penaltyValue = 0.0;
  vector_t penaltyDerivative, penaltySecondDerivative;
  std::tie(penaltyValue, penaltyDerivative, penaltySecondDerivative) = getPenaltyValue1stDev2ndDev(t, h.f);
  const matrix_t penaltySecondDev_dhdx = penaltySecondDerivative.asDiagonal() * h.dfdx;

  // to make sure that dfdux in the state-only case has a right size
  ScalarFunctionQuadraticApproximation penaltyApproximation(stateDim, inputDim);

  penaltyApproximation.f = penaltyValue;
  penaltyApproximation.dfdx.noalias() = h.dfdx.transpose() * penaltyDerivative;
  penaltyApproximation.dfdxx.noalias() = h.dfdx.transpose() * penaltySecondDev_dhdx;
  for (size_t i = 0; i < static_cast<size_t>(numConstraints); i++) {
    penaltyApproximation.dfdxx.noalias() += penaltyDerivative(i) * h.dfdxx[i];
  }

  if (inputDim > 0) {
    penaltyApproximation.dfdu.noalias() = h.dfdu.transpose() * penaltyDerivative;
    penaltyApproximation.dfdux.noalias() = h.dfdu.transpose() * penaltySecondDev_dhdx;
    penaltyApproximation.dfduu.noalias() = h.dfdu.transpose() * penaltySecondDerivative.asDiagonal() * h.dfdu;
    for (size_t i = 0; i < static_cast<size_t>(numConstraints); i++) {
      penaltyApproximation.dfduu.noalias() += penaltyDerivative(i) * h.dfduu[i];
      penaltyApproximation.dfdux.noalias() += penaltyDerivative(i) * h.dfdux[i];
    }
  }

  return penaltyApproximation;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::tuple<scalar_t, vector_t, vector_t> MultidimensionalPenalty::getPenaltyValue1stDev2ndDev(scalar_t t, const vector_t& h) const {
  const auto numConstraints = h.rows();
  assert(penaltyPtrArray_.size() == 1 || penaltyPtrArray_.size() == numConstraints);

  scalar_t penaltyValue = 0.0;
  vector_t penaltyDerivative(numConstraints);
  vector_t penaltySecondDerivative(numConstraints);
  for (size_t i = 0; i < static_cast<size_t>(numConstraints); i++) {
    const auto& penaltyTerm = (penaltyPtrArray_.size() == 1) ? penaltyPtrArray_[0] : penaltyPtrArray_[i];
    penaltyValue += penaltyTerm->getValue(t, h(i));
    penaltyDerivative(i) = penaltyTerm->getDerivative(t, h(i));
    penaltySecondDerivative(i) = penaltyTerm->getSecondDerivative(t, h(i));
  }  // end of i loop

  return {penaltyValue, penaltyDerivative, penaltySecondDerivative};
}

}  // namespace ocs2
