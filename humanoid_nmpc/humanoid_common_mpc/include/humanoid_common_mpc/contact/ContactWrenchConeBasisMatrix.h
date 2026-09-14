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
 * Builds the wrench-cone basis matrix B in R^{6 x numBasis} in the *local* contact frame.
 *
 * Each column of B is a generator ray of the contact wrench cone. A wrench W_local = B * lambda with lambda >= 0
 * (element-wise) is guaranteed to lie inside the linearized contact wrench cone defined by the supplied Config and
 * ContactRectangle, because the cone is convex and every generator lies inside it. The constructor verifies exactly
 * that against ContactWrenchConeConstraint's own rows (buildLocalWrenchConeRows) and throws if it does not hold:
 * when basis-vector inputs are active the explicit cone constraint is dropped, so an infeasible generator would
 * silently remove the friction or torsion limit from the MPC.
 *
 * Every generator is the wrench of a unit normal force applied at a point of the footprint, optionally with a
 * tangential force inside the friction cone and a torsion inside the torsional friction limit. Writing them that way
 * keeps the centre of pressure of each generator inside the footprint and the torsion about the patch reference point
 * within its bound, for any footprint and any patch offset.
 *
 * Generator layout (numBasis = N + 7, where N = config.numBasisVectors, p_c the patch reference point):
 *
 *   Columns 0..N-1   - friction-pyramid edge rays at p_c
 *       F = (r cos(theta_k), r sin(theta_k), 1),  theta_k = (k + 1/2) * 2 pi / N,  r = mu / cos(pi / N)
 *       The friction facets bound the tangential force by mu * Fz along N evenly spaced directions, so the feasible
 *       set is a polyhedral cone whose edges sit half a sector away from those directions at radius mu / cos(pi / N).
 *       Using exactly those edges makes the conic hull reproduce the friction facets: neither over- nor
 *       under-approximating the cone the wrench-space formulation enforces.
 *
 *   Column  N        - pure normal force ray at p_c
 *
 *   Columns N+1..N+4 - CoP corner rays: a unit normal force at each corner of the footprint, which places the centre
 *       of pressure on that corner; their conic hull spans the whole CoP rectangle.
 *
 *   Columns N+5..N+6 - torsional friction rays: a unit normal force at p_c with a torsion of +/- mu_torsion about the
 *       contact normal.
 *
 * Not representable: `minNormalForce` and `gripperForce` are affine offsets of the cone (the `b` vector of
 * ContactWrenchConeRows), while a conic combination is homogeneous - lambda = 0 always yields the zero wrench. A
 * basis-vector formulation therefore cannot enforce a positive minimum normal force or a gripper adhesion force;
 * CentroidalMpcInterface warns when either is configured together with basis-vector inputs.
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
