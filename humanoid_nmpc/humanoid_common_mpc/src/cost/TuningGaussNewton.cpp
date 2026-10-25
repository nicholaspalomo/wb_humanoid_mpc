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

#include "humanoid_common_mpc/cost/TuningGaussNewton.h"

#include <algorithm>
#include <cstddef>

#include "Eigen/Eigenvalues"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/orientation/TuningLayoutDeviation.h"

namespace ocs2::humanoid {

bool isIdentityTuningJacobian(const TuningDeviationApproximation& approximation,
                              size_t stateDim) {
  if (!approximation.denseBlocks.empty() ||
      approximation.identityRuns.size() != 1)
    return false;
  const TuningJacobianIdentityRun& run = approximation.identityRuns.front();
  return run.tuningRow == 0 && run.stateColumn == 0 && run.size == stateDim &&
         static_cast<size_t>(approximation.residual.size()) == stateDim;
}

vector_t tuningGradient(const TuningDeviationApproximation& approximation,
                        const matrix_t& stateWeights,
                        size_t stateDim) {
  if (isIdentityTuningJacobian(approximation, stateDim)) {
    // The expression of OCS2's QuadraticStateInputCost, L.dfdx.noalias() = Q_ *
    // stateDeviation.
    vector_t gradient;
    gradient.noalias() = stateWeights * approximation.residual;
    return gradient;
  }
  const vector_t weightedResidual = stateWeights * approximation.residual;
  vector_t gradient = vector_t::Zero(static_cast<Eigen::Index>(stateDim));
  for (const TuningJacobianIdentityRun& run : approximation.identityRuns) {
    const Eigen::Index size = static_cast<Eigen::Index>(run.size);
    gradient.segment(static_cast<Eigen::Index>(run.stateColumn), size) +=
        weightedResidual.segment(static_cast<Eigen::Index>(run.tuningRow),
                                 size);
  }
  for (const TuningJacobianDenseBlock& block : approximation.denseBlocks) {
    gradient
        .segment(static_cast<Eigen::Index>(block.stateColumn),
                 block.value.cols())
        .noalias() +=
        block.value.transpose() *
        weightedResidual.segment(static_cast<Eigen::Index>(block.tuningRow),
                                 block.value.rows());
  }
  return gradient;
}

matrix_t tuningHessian(const TuningDeviationApproximation& approximation,
                       const matrix_t& stateWeights,
                       size_t stateDim) {
  if (isIdentityTuningJacobian(approximation, stateDim)) {
    // OCS2's QuadraticStateInputCost: L.dfdxx = Q_.
    return stateWeights;
  }
  const Eigen::Index columns = static_cast<Eigen::Index>(stateDim);
  // Q J, column block by column block: Q's columns copied for an identity run,
  // Q(:, rows) times the block for a dense one.
  matrix_t weightedJacobian = matrix_t::Zero(stateWeights.rows(), columns);
  for (const TuningJacobianIdentityRun& run : approximation.identityRuns) {
    const Eigen::Index size = static_cast<Eigen::Index>(run.size);
    weightedJacobian.middleCols(static_cast<Eigen::Index>(run.stateColumn),
                                size) +=
        stateWeights.middleCols(static_cast<Eigen::Index>(run.tuningRow), size);
  }
  for (const TuningJacobianDenseBlock& block : approximation.denseBlocks) {
    weightedJacobian
        .middleCols(static_cast<Eigen::Index>(block.stateColumn),
                    block.value.cols())
        .noalias() +=
        stateWeights.middleCols(static_cast<Eigen::Index>(block.tuningRow),
                                block.value.rows()) *
        block.value;
  }
  // J' (Q J), row block by row block likewise.
  matrix_t hessian = matrix_t::Zero(columns, columns);
  for (const TuningJacobianIdentityRun& run : approximation.identityRuns) {
    const Eigen::Index size = static_cast<Eigen::Index>(run.size);
    hessian.middleRows(static_cast<Eigen::Index>(run.stateColumn), size) +=
        weightedJacobian.middleRows(static_cast<Eigen::Index>(run.tuningRow),
                                    size);
  }
  for (const TuningJacobianDenseBlock& block : approximation.denseBlocks) {
    hessian
        .middleRows(static_cast<Eigen::Index>(block.stateColumn),
                    block.value.cols())
        .noalias() +=
        block.value.transpose() *
        weightedJacobian.middleRows(static_cast<Eigen::Index>(block.tuningRow),
                                    block.value.rows());
  }
  return hessian;
}

absl::Status validateQuadraticWeights(const matrix_t& weights,
                                      size_t dimension,
                                      absl::string_view name) {
  if (static_cast<size_t>(weights.rows()) != dimension ||
      static_cast<size_t>(weights.cols()) != dimension) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " is ", weights.rows(), " x ", weights.cols(),
                     ", but the cost weighs ", dimension, " rows."));
  }
  if (dimension == 0) return absl::OkStatus();
  if (!weights.allFinite()) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " has an entry that is not a finite number."));
  }
  const scalar_t asymmetry =
      (weights - weights.transpose()).cwiseAbs().maxCoeff();
  const scalar_t scale = std::max(1.0, weights.cwiseAbs().maxCoeff());
  if (asymmetry > kQuadraticWeightsSymmetryTolerance * scale) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " is not symmetric: max |W - W'| is ", asymmetry,
                     ", more than ", kQuadraticWeightsSymmetryTolerance,
                     " times max(1, max |W|) = ", scale, "."));
  }
  // A weight below 0 makes the cost, and the Gauss-Newton Hessian J' W J the
  // solver builds from it, indefinite. The eigenvalues of the symmetric part,
  // to the round-off a basis-space R carries.
  const Eigen::SelfAdjointEigenSolver<matrix_t> eigen(
      0.5 * (weights + weights.transpose()), Eigen::EigenvaluesOnly);
  const scalar_t smallest = eigen.eigenvalues().minCoeff();
  if (eigen.info() != Eigen::Success ||
      !(smallest >= -kQuadraticWeightsSymmetryTolerance * scale)) {
    return absl::InvalidArgumentError(absl::StrCat(
        name, " is not positive semidefinite: its smallest eigenvalue is ",
        smallest, " (a weight below 0 makes the cost indefinite)."));
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid
