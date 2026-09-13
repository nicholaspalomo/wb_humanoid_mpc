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

#include <cassert>

#include <ocs2_core/Types.h>

namespace ocs2::humanoid {

/**
 * Transforms a quadratic input-cost weight matrix defined in wrench space into basis-vector space.
 *
 *   R_basis = Mᵀ · R_wrench · M + diag([λ-regularization, ..., 0, ...])
 *
 * where M = blkdiag(B_0, ..., B_{N-1}, I_joints) is the constant map from the basis-vector input to the
 * wrench-space input with every contact wrench expressed in its *local contact frame*
 * (see BasisInputsModelDecorator::getLocalBasisToWrenchMap). Consequently the wrench weights of R are
 * applied to the contact-frame wrench when basis-vector inputs are active.
 *
 * Each basis matrix has more generators (columns) than wrench components (rows), so Mᵀ R M is singular on
 * the null space of M. The optional regularization adds a small quadratic penalty on every λ, which makes the
 * input Hessian positive definite and the optimal λ unique without noticeably changing the wrench cost.
 *
 * @param R_wrench            Wrench-space weight matrix (wrenchInputDim × wrenchInputDim).
 * @param M                   Local basis-to-wrench map (wrenchInputDim × basisInputDim).
 * @param numBasisInputs      Number of leading λ entries in the basis-vector input (numBasisPerFoot · N_CONTACTS).
 * @param lambdaRegularization Non-negative value added to the diagonal of the λ block.
 * @return Basis-space weight matrix (basisInputDim × basisInputDim).
 */
inline matrix_t transformWrenchInputCostToBasisSpace(const matrix_t& R_wrench,
                                                     const matrix_t& M,
                                                     size_t numBasisInputs,
                                                     scalar_t lambdaRegularization) {
  assert(R_wrench.rows() == M.rows());
  assert(R_wrench.cols() == M.rows());
  assert(static_cast<Eigen::Index>(numBasisInputs) <= M.cols());
  assert(lambdaRegularization >= 0.0);
  matrix_t R_basis = M.transpose() * R_wrench * M;
  R_basis.diagonal().head(numBasisInputs).array() += lambdaRegularization;
  return R_basis;
}

/**
 * Bundles everything needed to transform a wrench-space input cost into basis-vector space, so that the
 * OCP factory and the online parameter updater apply exactly the same transformation.
 */
struct BasisInputsCostTransformConfig {
  matrix_t basisToWrenchMap;            ///< M (wrenchInputDim × basisInputDim), see BasisInputsModelDecorator::getLocalBasisToWrenchMap
  size_t wrenchInputDim = 0;            ///< Dimension of the wrench-space input (rows of M).
  size_t numBasisInputs = 0;            ///< Number of leading λ entries in the basis-vector input.
  scalar_t lambdaRegularization = 0.0;  ///< Diagonal regularization added to the λ block.

  size_t basisInputDim() const { return static_cast<size_t>(basisToWrenchMap.cols()); }
};

inline matrix_t transformWrenchInputCostToBasisSpace(const matrix_t& R_wrench, const BasisInputsCostTransformConfig& config) {
  return transformWrenchInputCostToBasisSpace(R_wrench, config.basisToWrenchMap, config.numBasisInputs, config.lambdaRegularization);
}

}  // namespace ocs2::humanoid
