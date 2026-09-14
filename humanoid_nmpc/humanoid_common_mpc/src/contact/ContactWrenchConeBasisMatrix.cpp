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

#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

namespace {

static constexpr size_t kWrenchDim = 6;
static constexpr size_t kExtraGenerators = 7;  // 1 normal + 4 CoP + 2 torsional

static constexpr scalar_t kHalf = 0.5;
static constexpr scalar_t kPatchInsideTolerance = 1e-9;      // [m] slack when checking that the patch point is on the footprint
static constexpr scalar_t kConeFeasibilityTolerance = 1e-9;  // slack of the generator feasibility self-check

}  // namespace

ContactWrenchConeBasisMatrix::ContactWrenchConeBasisMatrix(const ContactWrenchConeConstraint::Config& config,
                                                           const ContactRectangle& contactRectangle) {
  build(config, contactRectangle);
}

void ContactWrenchConeBasisMatrix::build(const ContactWrenchConeConstraint::Config& config, const ContactRectangle& contactRectangle) {
  const size_t N = config.numBasisVectors;
  if (N < 3) {
    throw std::invalid_argument("[ContactWrenchConeBasisMatrix] numBasisVectors must be at least 3");
  }
  numBasis_ = N + kExtraGenerators;
  B_local_ = matrix_t::Zero(kWrenchDim, numBasis_);

  // Every generator is the wrench of a unit normal force applied at a point of the footprint, optionally with a
  // tangential force inside the friction cone and a torsion inside the torsional friction limit. Writing the
  // generators this way makes each of them satisfy every facet of ContactWrenchConeConstraint by construction, which
  // is what lets the basis-vector formulation drop the explicit cone constraint.
  const PolygonBounds& bounds = contactRectangle.getBounds();
  const vector3_t patchPoint = contactPatchReferencePoint(config, contactRectangle);
  if (patchPoint.x() < bounds.x_min - kPatchInsideTolerance || patchPoint.x() > bounds.x_max + kPatchInsideTolerance ||
      patchPoint.y() < bounds.y_min - kPatchInsideTolerance || patchPoint.y() > bounds.y_max + kPatchInsideTolerance) {
    throw std::invalid_argument(
        "[ContactWrenchConeBasisMatrix] the torsional patch offset lies outside the contact rectangle; every wrench "
        "generator is applied at that point and would violate the CoP constraint");
  }

  // Moment about the contact frame origin of a force F applied at the point p (p_z is ignored, the contact patch is
  // planar), plus an optional torsion about the contact normal.
  const auto wrenchAt = [](const vector2_t& p, const vector3_t& force, scalar_t torsion) {
    vector6_t wrench;
    wrench.head<3>() = force;
    wrench.tail<3>() = vector3_t(p.x(), p.y(), 0.0).cross(force) + vector3_t(0.0, 0.0, torsion);
    return wrench;
  };
  const vector2_t patchXy(patchPoint.x(), patchPoint.y());

  size_t col = 0;

  // 1. Friction-pyramid edge rays, applied at the patch point.
  // The friction facets bound the tangential force by mu * Fz along numBasisVectors evenly spaced directions, so the
  // feasible set is a polyhedral cone whose edges sit half a sector away from those directions, at a tangential
  // magnitude of mu / cos(pi / N). Using exactly those edges makes the conic hull of the generators reproduce the
  // friction facets without either over- or under-approximating them.
  const scalar_t angleStep = 2.0 * M_PI / static_cast<scalar_t>(N);
  const scalar_t edgeRadius = config.frictionCoefficient / std::cos(M_PI / static_cast<scalar_t>(N));
  for (size_t k = 0; k < N; ++k) {
    const scalar_t theta_k = (static_cast<scalar_t>(k) + kHalf) * angleStep;
    const vector3_t force(edgeRadius * std::cos(theta_k), edgeRadius * std::sin(theta_k), 1.0);
    B_local_.col(col) = wrenchAt(patchXy, force, 0.0);
    col++;
  }

  // 2. Pure normal force ray at the patch point.
  B_local_.col(col) = wrenchAt(patchXy, vector3_t::UnitZ(), 0.0);
  col++;

  // 3. CoP corner rays: a unit normal force at each corner of the footprint, which places the centre of pressure
  // exactly on that corner. Their conic hull spans the whole CoP rectangle.
  const std::array<vector2_t, 4> corners = {vector2_t(bounds.x_max, bounds.y_max), vector2_t(bounds.x_max, bounds.y_min),
                                            vector2_t(bounds.x_min, bounds.y_max), vector2_t(bounds.x_min, bounds.y_min)};
  for (const vector2_t& corner : corners) {
    B_local_.col(col) = wrenchAt(corner, vector3_t::UnitZ(), 0.0);
    col++;
  }

  // 4. Torsional friction rays: a unit normal force at the patch point with the maximum admissible torsion of either
  // sign about the contact normal.
  B_local_.col(col) = wrenchAt(patchXy, vector3_t::UnitZ(), config.torsionalFrictionCoefficient);
  col++;
  B_local_.col(col) = wrenchAt(patchXy, vector3_t::UnitZ(), -config.torsionalFrictionCoefficient);
  col++;

  assert(col == numBasis_);

  // Guard the invariant the whole formulation rests on: a non-negative combination of the generators is inside the
  // cone if and only if every generator is. A violation here would silently remove the friction or torsion limit from
  // the MPC, because the explicit cone constraint is dropped when basis-vector inputs are active.
  const ContactWrenchConeRows coneRows = buildLocalWrenchConeRows(config, contactRectangle);
  for (size_t j = 0; j < numBasis_; ++j) {
    const vector_t rowValues = coneRows.evaluateCone(vector6_t(B_local_.col(j)));
    if (rowValues.minCoeff() < -kConeFeasibilityTolerance) {
      std::ostringstream message;
      message << "[ContactWrenchConeBasisMatrix] generator " << j << " lies outside the contact wrench cone (worst row "
              << rowValues.minCoeff() << "): " << B_local_.col(j).transpose();
      throw std::logic_error(message.str());
    }
  }

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
