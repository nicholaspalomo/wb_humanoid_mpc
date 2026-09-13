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

#include <Eigen/Core>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"

namespace ocs2::humanoid {

/**
 * Builds the wrench-cone basis matrix B ∈ ℝ^{6 × numBasis} in the *local* contact frame.
 *
 * Each column of B is a "generator ray" of the contact wrench cone.  A wrench
 * W_local = B * λ with λ ≥ 0 (element-wise) is guaranteed to lie inside the
 * linearized contact wrench cone defined by the supplied Config and
 * ContactRectangle.
 *
 * Generator layout (numBasis = N + 7, where N = config.numBasisVectors):
 *
 *   Columns 0..N-1  — friction-pyramid edge rays
 *       b_k = [cos(θ_k), sin(θ_k), μ, 0, 0, 0]^T
 *
 *   Column  N       — pure normal force ray
 *       b   = [0, 0, 1, 0, 0, 0]^T
 *
 *   Columns N+1..N+4 — CoP corner rays  (couple force & moment)
 *       These ensure that any λ ≥ 0 combination produces moments
 *       within the rectangular footprint [x_min,x_max] × [y_min,y_max].
 *
 *   Columns N+5..N+6 — torsional friction rays
 *       Couple Fz with ±τ_z within ±μ_torsion * Fz.
 */
class ContactWrenchConeBasisMatrix {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * Constructs the local-frame basis matrix from the same parameters used by
   * ContactWrenchConeConstraint.
   */
  ContactWrenchConeBasisMatrix(const ContactWrenchConeConstraint::Config& config, const ContactRectangle& contactRectangle);

  /** Number of basis vectors (columns of B).  Equal to N + 7. */
  size_t numBasis() const { return numBasis_; }

  /** The basis matrix B ∈ ℝ^{6 × numBasis}, expressed in the *local* contact frame. */
  const matrix_t& getBasisMatrix() const { return B_local_; }

  /** The Moore-Penrose pseudoinverse B⁺ ∈ ℝ^{numBasis × 6}. */
  const matrix_t& getBasisMatrixPseudoInverse() const { return B_pinv_local_; }

 private:
  void build(const ContactWrenchConeConstraint::Config& config, const ContactRectangle& contactRectangle);

  size_t numBasis_;
  matrix_t B_local_;       // 6 × numBasis
  matrix_t B_pinv_local_;  // numBasis × 6
};

}  // namespace ocs2::humanoid
