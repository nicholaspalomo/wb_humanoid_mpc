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

#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"

#include <Eigen/SVD>

#include <cmath>

namespace ocs2::humanoid {

namespace {

static constexpr size_t kWrenchDim = 6;
static constexpr size_t kExtraGenerators = 7;  // 1 normal + 4 CoP + 2 torsional

static constexpr size_t kFxIdx = 0;
static constexpr size_t kFyIdx = 1;
static constexpr size_t kFzIdx = 2;
static constexpr size_t kTxIdx = 3;
static constexpr size_t kTyIdx = 4;
static constexpr size_t kTzIdx = 5;

static constexpr scalar_t kHalf = 0.5;
static constexpr scalar_t kPatchOffsetZeroThreshold = 1e-9;

}  // namespace

ContactWrenchConeBasisMatrix::ContactWrenchConeBasisMatrix(const ContactWrenchConeConstraint::Config& config,
                                                           const ContactRectangle& contactRectangle) {
  build(config, contactRectangle);
}

void ContactWrenchConeBasisMatrix::build(const ContactWrenchConeConstraint::Config& config, const ContactRectangle& contactRectangle) {
  const size_t N = config.numBasisVectors;
  numBasis_ = N + kExtraGenerators;
  B_local_ = matrix_t::Zero(kWrenchDim, numBasis_);

  size_t col = 0;

  // 1. Friction-pyramid edge rays: b_k = [cos(θ_k), sin(θ_k), μ, 0, 0, 0]^T
  const scalar_t angleStep = 2.0 * M_PI / static_cast<scalar_t>(N);
  for (size_t k = 0; k < N; ++k) {
    const scalar_t theta_k = k * angleStep;
    B_local_(kFxIdx, col) = std::cos(theta_k);
    B_local_(kFyIdx, col) = std::sin(theta_k);
    B_local_(kFzIdx, col) = config.frictionCoefficient;
    col++;
  }

  // 2. Pure normal force ray: [0, 0, 1, 0, 0, 0]^T
  B_local_(kFzIdx, col) = 1.0;
  col++;

  // 3. CoP corner rays — couple Fz with τ_x, τ_y at the footprint corners.
  // These produce wrenches that are exactly on the CoP boundary.
  const PolygonBounds& bounds = contactRectangle.getBounds();

  // Corner (+x_max, +y_max): τ_y = -x_max * Fz,  τ_x = +y_max * Fz
  B_local_(kFzIdx, col) = 1.0;
  B_local_(kTxIdx, col) = bounds.y_max;
  B_local_(kTyIdx, col) = -bounds.x_max;
  col++;

  // Corner (+x_max, +y_min): τ_y = -x_max * Fz,  τ_x = +y_min * Fz
  B_local_(kFzIdx, col) = 1.0;
  B_local_(kTxIdx, col) = bounds.y_min;
  B_local_(kTyIdx, col) = -bounds.x_max;
  col++;

  // Corner (+x_min, +y_max): τ_y = -x_min * Fz,  τ_x = +y_max * Fz
  B_local_(kFzIdx, col) = 1.0;
  B_local_(kTxIdx, col) = bounds.y_max;
  B_local_(kTyIdx, col) = -bounds.x_min;
  col++;

  // Corner (+x_min, +y_min): τ_y = -x_min * Fz,  τ_x = +y_min * Fz
  B_local_(kFzIdx, col) = 1.0;
  B_local_(kTxIdx, col) = bounds.y_min;
  B_local_(kTyIdx, col) = -bounds.x_min;
  col++;

  // 4. Torsional friction rays — couple Fz with ±τ_z.
  // Compute patch center offset for the torsional constraint
  vector3_t offset;
  if (config.patchOffset.isZero(kPatchOffsetZeroThreshold)) {
    offset = vector3_t(kHalf * (bounds.x_min + bounds.x_max), kHalf * (bounds.y_min + bounds.y_max), 0.0);
  } else {
    offset = config.patchOffset;
  }

  // Positive torsion: τ_z = +μ_torsion * Fz + offset effects
  B_local_(kFxIdx, col) = -offset.y();
  B_local_(kFyIdx, col) = offset.x();
  B_local_(kFzIdx, col) = config.torsionalFrictionCoefficient;
  B_local_(kTzIdx, col) = 1.0;
  col++;

  // Negative torsion: τ_z = -μ_torsion * Fz - offset effects
  B_local_(kFxIdx, col) = offset.y();
  B_local_(kFyIdx, col) = -offset.x();
  B_local_(kFzIdx, col) = config.torsionalFrictionCoefficient;
  B_local_(kTzIdx, col) = -1.0;
  col++;

  assert(col == numBasis_);

  // Compute Moore-Penrose pseudoinverse via SVD
  Eigen::JacobiSVD<matrix_t> svd(B_local_, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const vector_t& singularValues = svd.singularValues();
  const scalar_t tolerance = 1e-6 * singularValues(0);
  matrix_t singularValuesInv = matrix_t::Zero(singularValues.size(), singularValues.size());
  for (int i = 0; i < singularValues.size(); ++i) {
    if (singularValues(i) > tolerance) {
      singularValuesInv(i, i) = 1.0 / singularValues(i);
    }
  }
  B_pinv_local_ = svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
}

}  // namespace ocs2::humanoid
