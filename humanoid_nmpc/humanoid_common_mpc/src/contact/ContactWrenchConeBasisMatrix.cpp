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

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/QR"
#include "Eigen/SVD"
#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

namespace {

constexpr Eigen::Index kWrenchDim = 6;
constexpr size_t kNumFootprintCorners = 4;
constexpr size_t kNumTorsionSigns = 2;
constexpr size_t kNumConservativeExtraGenerators = 7;  // 1 normal + 4 CoP corners + 2 torsional

constexpr scalar_t kHalf = 0.5;
constexpr scalar_t kPatchInsideTolerance = 1.0e-9;      // [m] slack when checking that the patch point is on the footprint
constexpr scalar_t kConeFeasibilityTolerance = 1.0e-9;  // slack of the generator feasibility self-check
constexpr scalar_t kPseudoInverseRelativeTolerance = 1.0e-6;
constexpr scalar_t kNonNegativeRelativeTolerance = 1.0e-12;  // how negative B+ W may be and still count as non-negative
constexpr scalar_t kNnlsRelativeTolerance = 1.0e-12;         // Lawson-Hanson optimality tolerance, relative to |A| |b|
constexpr size_t kNnlsIterationsPerColumn = 5;

constexpr absl::string_view kContactRectanglePrefix = "contacts.contact_rectangle.";

// FNV-1a, 64 bit: https://datatracker.ietf.org/doc/html/draft-eastlake-fnv
constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;
constexpr scalar_t kHashZeroThreshold = 1.0e-12;  // entries below this magnitude are hashed as zero

/** The footprint corners in the documented order: (x_max,y_max), (x_max,y_min), (x_min,y_max), (x_min,y_min). */
std::array<vector2_t, kNumFootprintCorners> footprintCorners(const PolygonBounds& bounds) {
  return {vector2_t(bounds.x_max, bounds.y_max), vector2_t(bounds.x_max, bounds.y_min), vector2_t(bounds.x_min, bounds.y_max),
          vector2_t(bounds.x_min, bounds.y_min)};
}

/** The N edges of the friction pyramid at unit normal force: half a sector away from the facets, at mu / cos(pi / N). */
std::vector<vector3_t> frictionPyramidEdges(const ContactWrenchConeConstraint::Config& config) {
  const size_t numDirections = config.numBasisVectors;
  const scalar_t angleStep = 2.0 * M_PI / static_cast<scalar_t>(numDirections);
  const scalar_t edgeRadius = config.frictionCoefficient / std::cos(M_PI / static_cast<scalar_t>(numDirections));
  std::vector<vector3_t> edges;
  edges.reserve(numDirections);
  for (size_t k = 0; k < numDirections; ++k) {
    const scalar_t theta = (static_cast<scalar_t>(k) + kHalf) * angleStep;
    edges.emplace_back(edgeRadius * std::cos(theta), edgeRadius * std::sin(theta), 1.0);
  }
  return edges;
}

/**
 * Wrench of a force applied at the planar point `point` of the contact frame, plus a torsion about the contact normal.
 * The moment is taken about the contact frame origin.
 */
vector6_t wrenchAt(const vector2_t& point, const vector3_t& force, scalar_t torsion) {
  vector6_t wrench;
  wrench.head<3>() = force;
  wrench.tail<3>() = vector3_t(point.x(), point.y(), 0.0).cross(force) + vector3_t(0.0, 0.0, torsion);
  return wrench;
}

/** The N + 7 set, see kConservativeInnerApproximationGeneratorSet. */
absl::StatusOr<matrix_t> buildConservativeInnerApproximation(const ContactWrenchConeConstraint::Config& config,
                                                             const ContactRectangle& contactRectangle) {
  const PolygonBounds& bounds = contactRectangle.getBounds();
  const vector3_t patchPoint = contactPatchReferencePoint(config, contactRectangle);
  if (patchPoint.x() < bounds.x_min - kPatchInsideTolerance || patchPoint.x() > bounds.x_max + kPatchInsideTolerance ||
      patchPoint.y() < bounds.y_min - kPatchInsideTolerance || patchPoint.y() > bounds.y_max + kPatchInsideTolerance) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ContactWrenchConeBasisMatrix] the torsional patch point (", patchPoint.x(), ", ", patchPoint.y(), ") lies outside ",
                     kContactRectanglePrefix, "{x_min, x_max, y_min, y_max} = [", bounds.x_min, ", ", bounds.x_max, "] x [", bounds.y_min,
                     ", ", bounds.y_max, "]. The '", kConservativeInnerApproximationGeneratorSet,
                     "' generator set applies its friction and torsion rays at that point, so they would leave the footprint; enlarge the "
                     "footprint, move the patch offset onto it, or set ",
                     kBasisGeneratorSetKey, ": ", kExactWrenchConeGeneratorSet, "."));
  }
  const vector2_t patchXy(patchPoint.x(), patchPoint.y());

  const size_t numDirections = config.numBasisVectors;
  matrix_t basis = matrix_t::Zero(kWrenchDim, static_cast<Eigen::Index>(numDirections + kNumConservativeExtraGenerators));
  Eigen::Index column = 0;
  // 1. Friction-pyramid edges at the patch point.
  for (const vector3_t& edge : frictionPyramidEdges(config)) {
    basis.col(column++) = wrenchAt(patchXy, edge, /*torsion=*/0.0);
  }
  // 2. A pure normal force at the patch point.
  basis.col(column++) = wrenchAt(patchXy, vector3_t::UnitZ(), /*torsion=*/0.0);
  // 3. A unit normal force at each footprint corner, which puts the center of pressure on that corner.
  for (const vector2_t& corner : footprintCorners(bounds)) {
    basis.col(column++) = wrenchAt(corner, vector3_t::UnitZ(), /*torsion=*/0.0);
  }
  // 4. A unit normal force at the patch point with the torsional limit of either sign.
  basis.col(column++) = wrenchAt(patchXy, vector3_t::UnitZ(), config.torsionalFrictionCoefficient);
  basis.col(column++) = wrenchAt(patchXy, vector3_t::UnitZ(), -config.torsionalFrictionCoefficient);
  ABSL_CHECK_EQ(column, basis.cols()) << "ContactWrenchConeBasisMatrix: a generator set filled another number of columns than it has";
  return basis;
}

/**
 * The 8N set, see kExactWrenchConeGeneratorSet. With Fz = 1 the rows of buildLocalWrenchConeRows bound three things
 * independently: the tangential force F_t (the friction facets, a polygon P), the center of pressure c = (-tau_y, tau_x)
 * (the footprint R) and the torsion about the patch point tau_z + p_y F_x - p_x F_y (the interval T = [-mu_t, mu_t]).
 * (F_t, c, t) -> W is an affine bijection from P x R x T onto that slice, so the slice's vertices are the images of the
 * product's vertices: N friction edges x 4 corners x 2 torsion signs.
 */
absl::StatusOr<matrix_t> buildExactWrenchCone(const ContactWrenchConeConstraint::Config& config, const ContactRectangle& contactRectangle) {
  const vector3_t patchPoint = contactPatchReferencePoint(config, contactRectangle);
  const std::vector<vector3_t> edges = frictionPyramidEdges(config);
  const std::array<vector2_t, kNumFootprintCorners> corners = footprintCorners(contactRectangle.getBounds());
  const std::array<scalar_t, kNumTorsionSigns> torsionSigns = {1.0, -1.0};

  matrix_t basis = matrix_t::Zero(kWrenchDim, static_cast<Eigen::Index>(edges.size() * kNumFootprintCorners * kNumTorsionSigns));
  Eigen::Index column = 0;
  for (const vector3_t& edge : edges) {
    for (const vector2_t& corner : corners) {
      for (const scalar_t sign : torsionSigns) {
        vector6_t generator;
        generator << edge.x(), edge.y(), edge.z(), corner.y(), -corner.x(),
            sign * config.torsionalFrictionCoefficient + patchPoint.x() * edge.y() - patchPoint.y() * edge.x();
        basis.col(column++) = generator;
      }
    }
  }
  ABSL_CHECK_EQ(column, basis.cols()) << "ContactWrenchConeBasisMatrix: a generator set filled another number of columns than it has";
  return basis;
}

struct GeneratorSetEntry {
  // NOLINTNEXTLINE(totw-view-member): every entry is a string literal of a constexpr registry, alive for the whole program.
  absl::string_view name;
  BasisGeneratorSetBuilder builder;
};

// LINT.IfChange(basis_generator_set_registry)
constexpr std::array<GeneratorSetEntry, 2> kGeneratorSetRegistry = {{
    {.name = kConservativeInnerApproximationGeneratorSet, .builder = &buildConservativeInnerApproximation},
    {.name = kExactWrenchConeGeneratorSet, .builder = &buildExactWrenchCone},
}};
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h:basis_generator_set_names)

/**
 * The configuration checks every generator set shares, with messages naming the key to change: the cone's own
 * (ContactWrenchConeConstraint::validateConfig, the one definition of its ranges) and the footprint's ordering.
 */
absl::Status validateConeConfiguration(const ContactWrenchConeConstraint::Config& config, const ContactRectangle& contactRectangle) {
  RETURN_IF_ERROR(ContactWrenchConeConstraint::validateConfig(config));
  const PolygonBounds& bounds = contactRectangle.getBounds();
  if (!(bounds.x_min <= bounds.x_max) || !(bounds.y_min <= bounds.y_max)) {
    return absl::InvalidArgumentError(absl::StrCat("[ContactWrenchConeBasisMatrix] ", kContactRectanglePrefix,
                                                   "x_min/x_max and y_min/y_max must be ordered min <= max, got x in [", bounds.x_min, ", ",
                                                   bounds.x_max, "] and y in [", bounds.y_min, ", ", bounds.y_max, "]."));
  }
  return absl::OkStatus();
}

matrix_t pseudoInverse(const matrix_t& matrix) {
  const Eigen::JacobiSVD<matrix_t> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const vector_t& singularValues = svd.singularValues();
  const scalar_t tolerance = kPseudoInverseRelativeTolerance * singularValues(0);
  vector_t singularValuesInverse = vector_t::Zero(singularValues.size());
  for (Eigen::Index i = 0; i < singularValues.size(); ++i) {
    if (singularValues(i) > tolerance) {
      singularValuesInverse(i) = 1.0 / singularValues(i);
    }
  }
  return svd.matrixV() * singularValuesInverse.asDiagonal() * svd.matrixU().transpose();
}

/** Least-squares solution restricted to the columns flagged in `passive`, scattered back to the full length. */
vector_t solvePassiveLeastSquares(const matrix_t& A, const vector_t& b, const std::vector<bool>& passive) {
  std::vector<Eigen::Index> passiveColumns;
  for (Eigen::Index j = 0; j < A.cols(); ++j) {
    if (passive[static_cast<size_t>(j)]) {
      passiveColumns.push_back(j);
    }
  }
  matrix_t Apassive(A.rows(), static_cast<Eigen::Index>(passiveColumns.size()));
  for (size_t i = 0; i < passiveColumns.size(); ++i) {
    Apassive.col(static_cast<Eigen::Index>(i)) = A.col(passiveColumns[i]);
  }
  const vector_t zPassive = Apassive.completeOrthogonalDecomposition().solve(b);
  vector_t z = vector_t::Zero(A.cols());
  for (size_t i = 0; i < passiveColumns.size(); ++i) {
    z(passiveColumns[i]) = zPassive(static_cast<Eigen::Index>(i));
  }
  return z;
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<std::string> basisGeneratorSetNames() {
  std::vector<std::string> names;
  names.reserve(kGeneratorSetRegistry.size());
  for (const GeneratorSetEntry& entry : kGeneratorSetRegistry) {
    names.emplace_back(entry.name);
  }
  return names;
}

absl::StatusOr<BasisGeneratorSetBuilder> getBasisGeneratorSetBuilder(absl::string_view name) {
  for (const GeneratorSetEntry& entry : kGeneratorSetRegistry) {
    if (entry.name == name) {
      return entry.builder;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat("[ContactWrenchConeBasisMatrix] unknown ", kBasisGeneratorSetKey, " '", name,
                                                 "'; valid names are: ", absl::StrJoin(basisGeneratorSetNames(), ", "), "."));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::StatusOr<ContactWrenchConeBasisMatrix> ContactWrenchConeBasisMatrix::Create(const ContactWrenchConeConstraint::Config& config,
                                                                                  const ContactRectangle& contactRectangle,
                                                                                  absl::string_view generatorSet) {
  const absl::StatusOr<BasisGeneratorSetBuilder> builder = getBasisGeneratorSetBuilder(generatorSet);
  if (!builder.ok()) {
    return builder.status();
  }
  RETURN_IF_ERROR(validateConeConfiguration(config, contactRectangle));
  absl::StatusOr<matrix_t> basis = (*builder)(config, contactRectangle);
  if (!basis.ok()) {
    return basis.status();
  }

  // Guard the invariant the whole formulation rests on: a non-negative combination of the generators is inside the
  // cone if and only if every generator is. A violation here would silently remove the friction or torsion limit from
  // the MPC, because the explicit cone constraint is dropped when basis-vector inputs are active.
  const ContactWrenchConeRows coneRows = buildLocalWrenchConeRows(config, contactRectangle);
  for (Eigen::Index j = 0; j < basis->cols(); ++j) {
    const vector_t rowValues = coneRows.evaluateCone(vector6_t(basis->col(j)));
    if (rowValues.minCoeff() < -kConeFeasibilityTolerance) {
      Eigen::Index worstRow = 0;
      rowValues.minCoeff(&worstRow);
      return absl::InternalError(absl::StrCat("[ContactWrenchConeBasisMatrix] generator ", j, " of the '", generatorSet,
                                              "' set lies outside the contact wrench cone: row ", worstRow, " evaluates to ",
                                              rowValues(worstRow), ". This is a bug in the generator set, not a configuration error."));
    }
  }
  return ContactWrenchConeBasisMatrix(std::string(generatorSet), *std::move(basis));
}

ContactWrenchConeBasisMatrix::ContactWrenchConeBasisMatrix(std::string generatorSet, matrix_t basisMatrix)
    : generatorSet_(std::move(generatorSet)), B_local_(std::move(basisMatrix)), B_pinv_local_(pseudoInverse(B_local_)) {
  nullSpaceProjector_ = matrix_t::Identity(B_local_.cols(), B_local_.cols()) - B_pinv_local_ * B_local_;
  // Symmetrize away the round-off, so that the projector can go straight into a quadratic cost.
  nullSpaceProjector_ = (kHalf * (nullSpaceProjector_ + nullSpaceProjector_.transpose())).eval();
}

vector_t ContactWrenchConeBasisMatrix::solveNonNegativeScalings(const vector6_t& wrenchLocal) const {
  return solveNonNegativeBasisScalings(B_local_, B_pinv_local_, wrenchLocal);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t solveNonNegativeLeastSquares(const matrix_t& A, const vector_t& b) {
  const Eigen::Index numColumns = A.cols();
  vector_t x = vector_t::Zero(numColumns);
  if (numColumns == 0) {
    return x;
  }
  // Stationarity tolerance of the gradient A^T (b - A x), relative to the scale of the problem.
  const scalar_t gradientTolerance =
      kNnlsRelativeTolerance * std::max<scalar_t>(1.0, A.cwiseAbs().maxCoeff()) * std::max<scalar_t>(1.0, b.norm());
  std::vector<bool> passive(static_cast<size_t>(numColumns), false);
  // A column whose least-squares coefficient came out non-positive right after it entered the passive set is left out
  // until the next outer step, which is the textbook guard against cycling on round-off.
  std::vector<bool> blocked(static_cast<size_t>(numColumns), false);

  const size_t maxIterations = kNnlsIterationsPerColumn * static_cast<size_t>(numColumns);
  for (size_t iteration = 0; iteration < maxIterations; ++iteration) {
    const vector_t gradient = A.transpose() * (b - A * x);
    Eigen::Index entering = -1;
    scalar_t largestGradient = gradientTolerance;
    for (Eigen::Index j = 0; j < numColumns; ++j) {
      const size_t jj = static_cast<size_t>(j);
      if (!passive[jj] && !blocked[jj] && gradient(j) > largestGradient) {
        largestGradient = gradient(j);
        entering = j;
      }
    }
    if (entering < 0) {
      break;
    }
    std::fill(blocked.begin(), blocked.end(), false);
    passive[static_cast<size_t>(entering)] = true;

    // Inner loop: step towards the unconstrained least-squares solution on the passive set, dropping every column
    // whose coefficient would turn negative, until that solution is strictly positive.
    for (size_t inner = 0; inner < static_cast<size_t>(numColumns) + 1; ++inner) {
      const vector_t z = solvePassiveLeastSquares(A, b, passive);
      if (z(entering) <= 0.0 && passive[static_cast<size_t>(entering)] && x(entering) <= 0.0) {
        // Round-off: the entering column does not actually reduce the residual. Undo it and try another one.
        passive[static_cast<size_t>(entering)] = false;
        blocked[static_cast<size_t>(entering)] = true;
        break;
      }
      bool allPositive = true;
      scalar_t step = 1.0;
      for (Eigen::Index j = 0; j < numColumns; ++j) {
        if (passive[static_cast<size_t>(j)] && z(j) <= 0.0) {
          allPositive = false;
          step = std::min(step, x(j) / (x(j) - z(j)));
        }
      }
      if (allPositive) {
        x = z;
        break;
      }
      x += step * (z - x);
      for (Eigen::Index j = 0; j < numColumns; ++j) {
        if (passive[static_cast<size_t>(j)] && x(j) <= 0.0) {
          passive[static_cast<size_t>(j)] = false;
          x(j) = 0.0;
        } else if (passive[static_cast<size_t>(j)] && z(j) <= 0.0 && x(j) <= kNnlsRelativeTolerance * std::max<scalar_t>(1.0, x.norm())) {
          passive[static_cast<size_t>(j)] = false;
          x(j) = 0.0;
        }
      }
    }
  }
  return x.cwiseMax(0.0);
}

vector_t solveNonNegativeBasisScalings(const matrix_t& basisMatrix, const matrix_t& basisPseudoInverse, const vector_t& wrench) {
  const vector_t minimumNorm = basisPseudoInverse * wrench;
  const scalar_t tolerance = kNonNegativeRelativeTolerance * std::max<scalar_t>(1.0, minimumNorm.cwiseAbs().maxCoeff());
  if (minimumNorm.size() == 0 || minimumNorm.minCoeff() >= -tolerance) {
    return minimumNorm.cwiseMax(0.0);
  }
  return solveNonNegativeLeastSquares(basisMatrix, wrench);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

uint64_t basisContentHash(const feet_array_t<matrix_t>& localBasisMatrices) {
  std::string serialized;
  for (const matrix_t& basis : localBasisMatrices) {
    absl::StrAppend(&serialized, basis.rows(), "x", basis.cols(), ":");
    for (Eigen::Index row = 0; row < basis.rows(); ++row) {
      for (Eigen::Index column = 0; column < basis.cols(); ++column) {
        // Entries that are zero up to round-off (a cosine of pi / 2, say) print as zero, and adding zero turns -0.0 into
        // 0.0, so that neither the last bits of a libm nor the sign of a zero can change the key.
        const scalar_t entry = std::abs(basis(row, column)) < kHashZeroThreshold ? 0.0 : basis(row, column);
        absl::StrAppend(&serialized, absl::StrFormat("%.12g", entry + 0.0), ",");
      }
    }
    absl::StrAppend(&serialized, ";");
  }
  uint64_t hash = kFnvOffsetBasis;
  for (const char character : serialized) {
    hash ^= static_cast<uint64_t>(static_cast<unsigned char>(character));
    hash *= kFnvPrime;
  }
  return hash;
}

std::string basisInputsLibraryKey(const feet_array_t<matrix_t>& localBasisMatrices) {
  return absl::StrCat("basis", localBasisMatrices[0].cols(), "_", absl::Hex(basisContentHash(localBasisMatrices), absl::kZeroPad16));
}

}  // namespace ocs2::humanoid
