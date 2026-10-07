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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"

namespace ocs2::humanoid {

// LINT.IfChange(basis_generator_set_names)
/**
 * The N + 7 generators the basis-vector formulation has always shipped with (the default): N friction-pyramid edges,
 * one pure normal force and two torsion rays, all applied at the torsional patch point, and four normal forces at the
 * footprint corners. Every generator lies inside the wrench cone, so the set is SOUND, but it is a strongly
 * conservative INNER APPROXIMATION of the cone wrench mode enforces: each generator exercises only one of the three
 * limits, so friction, center-of-pressure excursion and torsion share one normal-force budget. It cannot represent a
 * tangential force together with a center of pressure away from the patch point, nor torsion together with either.
 * See humanoid_nmpc/docs/contact_basis_vectors/README.md.
 */
inline constexpr absl::string_view kConservativeInnerApproximationGeneratorSet = "conservative_inner_approximation";
/**
 * The 8N extreme rays of the wrench cone ContactWrenchConeConstraint enforces in wrench mode (its V-representation):
 * every friction-pyramid edge, at every footprint corner, with the torsional limit of either sign about the patch
 * point. The conic hull of these columns IS that cone, neither larger nor smaller. With a positive torsional
 * coefficient and a footprint of positive area every column is an extreme ray, so 8N is the minimum number of
 * generators that achieves it; with torsionalFrictionCoefficient = 0 (or a footprint of zero width or length) some
 * columns coincide, which is harmless.
 */
inline constexpr absl::string_view kExactWrenchConeGeneratorSet = "exact_wrench_cone";
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/contact/ContactWrenchConeBasisMatrix.cpp:basis_generator_set_registry, //humanoid_nmpc/docs/contact_basis_vectors/README.md:generator_set_names, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto:basis_generator_set_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto:basis_generator_set_config, //humanoid_nmpc/humanoid_mpc_config/contacts_config.proto:basis_generator_set)
// clang-format on

/** The set used when nothing names one. It stays the shipped set until the exact one has been validated in closed loop. */
inline constexpr absl::string_view kDefaultBasisGeneratorSet = kConservativeInnerApproximationGeneratorSet;

/** The task-file field naming the generator set (contactWrenchConeBasesFromConfig); the registry's error messages name it. */
inline constexpr absl::string_view kBasisGeneratorSetKey = "contacts.basis_generator_set";

/**
 * Builds the 6 x numBasis generator matrix of one foot, in its local contact frame. `config` and `contactRectangle`
 * have already passed the checks every set shares (ContactWrenchConeBasisMatrix::Create); a builder reports only the
 * problems specific to its own construction.
 */
using BasisGeneratorSetBuilder = absl::StatusOr<matrix_t> (*absl_nonnull)(const ContactWrenchConeConstraint::Config& config,
                                                                          const ContactRectangle& contactRectangle);

/** Every registered generator-set name, in registration order. */
std::vector<std::string> basisGeneratorSetNames();

/**
 * Resolves a generator-set name to its builder. An unknown name is an InvalidArgumentError that names
 * kBasisGeneratorSetKey and lists every valid name.
 */
absl::StatusOr<BasisGeneratorSetBuilder> getBasisGeneratorSetBuilder(absl::string_view name);

/**
 * The wrench-cone basis B in R^{6 x numBasis} of one foot, in its *local* contact frame: W_local = B * lambda with
 * lambda >= 0 (element-wise) is a wrench inside the linearized contact wrench cone of the supplied Config and
 * ContactRectangle - the homogeneous rows of buildLocalWrenchConeRows, which ContactWrenchConeConstraint enforces in
 * wrench mode. The cone is convex, so that holds for every lambda >= 0 exactly when it holds for every column, and
 * Create() verifies it against those rows: basis-vector inputs drop the explicit cone constraint, so a generator
 * outside the cone would silently remove a limit from the MPC.
 *
 * Which part of the cone the columns SPAN depends on the generator set, chosen by name (basisGeneratorSetNames()):
 *
 *   conservative_inner_approximation (default, N + 7 columns, p_c the patch reference point)
 *     Columns 0..N-1   friction-pyramid edges at p_c: F = (r cos(theta_k), r sin(theta_k), 1),
 *                      theta_k = (k + 1/2) * 2 pi / N, r = mu / cos(pi / N)
 *     Column  N        a pure normal force at p_c
 *     Columns N+1..N+4 a unit normal force at each footprint corner (x_max,y_max), (x_max,y_min), (x_min,y_max),
 *                      (x_min,y_min)
 *     Columns N+5..N+6 a unit normal force at p_c with a torsion of +mu_t and -mu_t
 *     Its projection onto the force alone is exact, but with Fz normalized to one, friction use, center-of-pressure
 *     excursion from p_c and torsion use draw on one shared budget, where wrench mode bounds each of them separately.
 *     At the footprint's toe it admits no tangential force at all, and at the torsional limit none either.
 *
 *   exact_wrench_cone (8N columns)
 *     Column (k * 4 + c) * 2 + s, for friction edge k, corner c (same order as above) and sign s (+mu_t, then -mu_t):
 *       [F_k; c_y; -c_x; s * mu_t + p_x * F_ky - p_y * F_kx]
 *     i.e. the edge force F_k applied at corner c with the torsion about p_c at its limit. The Fz = 1 slice of the
 *     cone is the friction polygon x the footprint x the torsion interval, whose vertices are exactly these products.
 *
 * Every column of either set carries a unit local normal force, so sum(lambda) is the foot's normal force.
 *
 * Not representable by any set: `minNormalForce` and `gripperForce` are affine offsets of the cone (the `b` vector of
 * ContactWrenchConeRows), while a conic combination is homogeneous - lambda = 0 always yields the zero wrench.
 */
class ContactWrenchConeBasisMatrix {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * Builds the basis of the named generator set. Configuration errors are an InvalidArgumentError naming the
   * task-file field to change (contacts.contact_wrench_cone_soft_constraint.*, contacts.contact_rectangle.*,
   * kBasisGeneratorSetKey); a generator outside the cone is an InternalError.
   */
  static absl::StatusOr<ContactWrenchConeBasisMatrix> Create(const ContactWrenchConeConstraint::Config& config,
                                                             const ContactRectangle& contactRectangle,
                                                             absl::string_view generatorSet = kDefaultBasisGeneratorSet);

  /** Name of the generator set this basis was built from. */
  const std::string& generatorSet() const { return generatorSet_; }

  /** Number of basis vectors (columns of B): N + 7 or 8N, depending on the generator set. */
  size_t numBasis() const { return static_cast<size_t>(B_local_.cols()); }

  /** The basis matrix B in R^{6 x numBasis}, expressed in the *local* contact frame. */
  const matrix_t& getBasisMatrix() const { return B_local_; }

  /** The Moore-Penrose pseudoinverse B+ in R^{numBasis x 6}. */
  const matrix_t& getBasisMatrixPseudoInverse() const { return B_pinv_local_; }

  /** The orthogonal projector onto null(B), I - B+ B in R^{numBasis x numBasis}: the lambda that produce no wrench. */
  const matrix_t& getNullSpaceProjector() const { return nullSpaceProjector_; }

  /** Non-negative scalings reproducing a local-frame wrench; see solveNonNegativeBasisScalings(). */
  vector_t solveNonNegativeScalings(const vector6_t& wrenchLocal) const;

 private:
  ContactWrenchConeBasisMatrix(std::string generatorSet, matrix_t basisMatrix);

  std::string generatorSet_;
  matrix_t B_local_;             // 6 x numBasis
  matrix_t B_pinv_local_;        // numBasis x 6
  matrix_t nullSpaceProjector_;  // numBasis x numBasis
};

/**
 * Non-negative least squares, argmin ||A x - b|| subject to x >= 0, by the Lawson-Hanson active-set method. Returns an
 * exact solution (A x = b) whenever b is a non-negative combination of the columns of A.
 */
vector_t solveNonNegativeLeastSquares(const matrix_t& A, const vector_t& b);

/**
 * The non-negative scalings lambda whose wrench B * lambda is closest to `wrench`. When the minimum-norm solution
 * B+ * wrench is already non-negative it is returned, because it spreads the load over every generator; otherwise the
 * scalings come from solveNonNegativeLeastSquares, which reproduces every wrench the cone of B contains exactly and
 * returns the closest representable wrench for one it does not. (Clamping B+ * wrench at zero instead distorts
 * wrenches that are exactly representable, such as the weight seen from a pitched foot.)
 */
vector_t solveNonNegativeBasisScalings(const matrix_t& basisMatrix, const matrix_t& basisPseudoInverse, const vector_t& wrench);

/**
 * A 64-bit content hash of the basis matrices that is stable across processes, builds and platforms: FNV-1a over the
 * dimensions and the entries printed to 12 significant digits, with entries below 1e-12 in magnitude taken as zero and
 * -0 as 0. Round-off in the last bits of an entry therefore leaves it unchanged unless the entry happens to sit on a
 * rounding boundary of the 12th digit, in which case the only consequence is one extra CppAD compilation.
 */
uint64_t basisContentHash(const feet_array_t<matrix_t>& localBasisMatrices);

/**
 * The key of a basis-vector input parameterization for naming CppAD libraries: "basis<numBasisPerFoot>_<16 hex digits
 * of basisContentHash>". Every CppAD tape whose domain or body depends on the contact input parameterization has to be
 * keyed by it, so that a cached library is never loaded for another parameterization or another basis.
 */
std::string basisInputsLibraryKey(const feet_array_t<matrix_t>& localBasisMatrices);

/** The counterpart of basisInputsLibraryKey() for the wrench-space input parameterization. */
inline constexpr absl::string_view kWrenchInputsLibraryKey = "wrench_inputs";

}  // namespace ocs2::humanoid
