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

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/TuningLayoutDeviation.h"

// The Gauss-Newton model of the quadratic tuning-layout cost l(x) = 1/2 r(x)' Q r(x) that TuningQuadraticStateInputCost
// and TuningQuadraticStateCost share: gradient J' Q r and Hessian J' Q J of the block-sparse Jacobian J of a
// TuningDeviationApproximation, assembled block by block. Q (n_t x n_t) is in the tuning layout and the derivatives are
// with respect to the state (stateDim entries: ambient, the solver pulls them back onto a state manifold). Neither
// function forms J densely: the identity runs together cost O(n_t stateDim), and a dense block of r x c entries
// O((n_t + stateDim) r c). Pure functions, safe to call concurrently.

namespace ocs2::humanoid {

/**
 * Whether J of `approximation` is the identity on a state of `stateDim` entries: one identity run from residual row and
 * state column 0 over the whole state, and no dense block - the Jacobian of a layout of differences only (no rotation
 * block), where the cost is OCS2's QuadraticStateInputCost of x - x_ref.
 */
bool isIdentityTuningJacobian(const TuningDeviationApproximation& approximation, size_t stateDim);

/**
 * J' Q r, stateDim entries. On the identity Jacobian (isIdentityTuningJacobian()) Q r, computed as OCS2's
 * QuadraticStateInputCost computes its gradient, bit for bit. `stateWeights` is getTuningDim() square, the residual's
 * size, and the blocks lie inside the residual and the state (preconditions, unchecked).
 */
vector_t tuningGradient(const TuningDeviationApproximation& approximation, const matrix_t& stateWeights, size_t stateDim);

/** J' Q J, stateDim square, under tuningGradient()'s preconditions. On the identity Jacobian a copy of Q, bit for bit. */
matrix_t tuningHessian(const TuningDeviationApproximation& approximation, const matrix_t& stateWeights, size_t stateDim);

/**
 * OK when `weights` can weigh a quadratic cost of `dimension` rows: `dimension` square, finite, and symmetric to
 * kQuadraticWeightsSymmetryTolerance of its largest entry (a basis-space R, M' R_wrench M, is symmetric only to
 * round-off). Otherwise InvalidArgument naming `name` and what is wrong.
 */
absl::Status validateQuadraticWeights(const matrix_t& weights, size_t dimension, absl::string_view name);

/** The relative asymmetry validateQuadraticWeights() accepts: max |W - W'| <= this times max(1, max |W|). */
inline constexpr scalar_t kQuadraticWeightsSymmetryTolerance = 1.0e-9;

}  // namespace ocs2::humanoid
