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

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "Eigen/Eigenvalues"
#include "Eigen/SVD"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"

namespace ocs2::humanoid {

/**
 * The input cost of the basis-vector formulation.
 *
 * A quadratic input-cost weight R is written in wrench space in the task file. With basis-vector inputs it becomes
 *
 *   R_basis = Mᵀ · R · M + reg · blkdiag(S, 0_joints)
 *
 * where M = blkdiag(B_0, ..., B_{N-1}, I_joints) maps the basis-vector input to the wrench-space input with every
 * contact wrench expressed in its *local contact frame* (BasisInputsModelDecorator::getLocalBasisToWrenchMap), so the
 * wrench weights of R act on the contact-frame wrench. Each B_i has more columns than rows, so Mᵀ R M alone is
 * singular on null(B_i): the λ that produce no wrench. The regularization S, chosen by name, removes that singularity.
 *
 *   full_diagonal (default)  S = I. Simple, but not negligible: minimizing over the λ that produce a given wrench, the
 *                            metric the optimizer sees on the wrench becomes R + reg · (B Bᵀ)⁻¹, and with the DRC
 *                            Atlas weights at the time of writing (reg = 1e-4 against R entries of 1e-5 .. 2e-4) the
 *                            second term dominates, so R's moment and tangential weights barely matter.
 *   null_space               S = blkdiag(I - B_i⁺ B_i): only the λ that produce no wrench are penalized. The metric on
 *                            the wrench is then exactly R, and the λ block is still positive definite whenever R is
 *                            positive definite on the contact wrenches and reg > 0.
 *
 * See humanoid_nmpc/docs/contact_basis_vectors/README.md.
 */

// LINT.IfChange(basis_regularization_names)
/** reg · I on every λ: the regularization the formulation has always shipped with. */
inline constexpr absl::string_view kFullDiagonalBasisRegularization = "full_diagonal";
/** reg · (I - B⁺B) per foot: penalizes only the λ that produce no wrench, so R acts unchanged on every wrench. */
inline constexpr absl::string_view kNullSpaceBasisRegularization = "null_space";
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/BasisInputsCostTransform.h:basis_regularization_registry, //humanoid_nmpc/docs/contact_basis_vectors/README.md:regularization_names, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto:basis_regularization_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto:basis_regularization_config, //humanoid_nmpc/humanoid_mpc_config/contacts_config.proto:basis_regularization)
// clang-format on

/** The regularization used when nothing names one; it stays the shipped one until null_space has been validated. */
inline constexpr absl::string_view kDefaultBasisRegularization = kFullDiagonalBasisRegularization;

/** The task-file field naming the regularization; the registry's error messages name it. */
inline constexpr absl::string_view kBasisRegularizationField = "contacts.basis_regularization";
/** The task-file field of the regularization weight `reg`. */
inline constexpr absl::string_view kBasisScalingRegularizationField = "contacts.basis_scaling_regularization";

/**
 * Builds the shape S (numBasisInputs x numBasisInputs) of the λ-block regularization from the local basis-to-wrench
 * map M (wrenchInputDim x basisInputDim), whose first numBasisInputs columns are the λ.
 */
using BasisRegularizationBuilder = matrix_t (*absl_nonnull)(const matrix_t& basisToWrenchMap, size_t numBasisInputs);

/** S = I. */
inline matrix_t fullDiagonalRegularizationShape(const matrix_t& /*basisToWrenchMap*/, size_t numBasisInputs) {
  const Eigen::Index n = static_cast<Eigen::Index>(numBasisInputs);
  return matrix_t::Identity(n, n);
}

/**
 * S = the orthogonal projector onto the null space of the λ columns of M. Those columns are blkdiag(B_i) over the
 * contact rows and zero over the joint rows, so this is blkdiag(I - B_i⁺ B_i).
 */
inline matrix_t nullSpaceRegularizationShape(const matrix_t& basisToWrenchMap, size_t numBasisInputs) {
  const Eigen::Index n = static_cast<Eigen::Index>(numBasisInputs);
  const matrix_t lambdaColumns = basisToWrenchMap.leftCols(n);
  const Eigen::JacobiSVD<matrix_t> svd(lambdaColumns, Eigen::ComputeFullV);
  const vector_t& singularValues = svd.singularValues();
  constexpr scalar_t kRankRelativeTolerance = 1.0e-9;
  const scalar_t tolerance = singularValues.size() > 0 ? kRankRelativeTolerance * singularValues(0) : 0.0;
  Eigen::Index rank = 0;
  while (rank < singularValues.size() && singularValues(rank) > tolerance) {
    ++rank;
  }
  const matrix_t nullSpaceBasis = svd.matrixV().rightCols(n - rank);
  return nullSpaceBasis * nullSpaceBasis.transpose();
}

namespace basis_inputs_cost_internal {
/** One registered regularization shape: the name the task file selects it by and the function that builds it. */
struct RegularizationEntry {
  // NOLINTNEXTLINE(totw-view-member): every entry is a string literal of a constexpr registry, alive for the whole program.
  absl::string_view name;
  BasisRegularizationBuilder builder;
};
// LINT.IfChange(basis_regularization_registry)
inline constexpr std::array<RegularizationEntry, 2> kRegularizationRegistry = {{
    {.name = kFullDiagonalBasisRegularization, .builder = &fullDiagonalRegularizationShape},
    {.name = kNullSpaceBasisRegularization, .builder = &nullSpaceRegularizationShape},
}};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/BasisInputsCostTransform.h:basis_regularization_names)
// clang-format on
}  // namespace basis_inputs_cost_internal

/** Every registered regularization name, in registration order. */
inline std::vector<std::string> basisRegularizationNames() {
  std::vector<std::string> names;
  for (const basis_inputs_cost_internal::RegularizationEntry& entry : basis_inputs_cost_internal::kRegularizationRegistry) {
    names.emplace_back(entry.name);
  }
  return names;
}

/**
 * Resolves a regularization name to its builder. An unknown name is an InvalidArgumentError naming
 * kBasisRegularizationField and listing every valid name.
 */
inline absl::StatusOr<BasisRegularizationBuilder> getBasisRegularizationBuilder(absl::string_view name) {
  for (const basis_inputs_cost_internal::RegularizationEntry& entry : basis_inputs_cost_internal::kRegularizationRegistry) {
    if (entry.name == name) {
      return entry.builder;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat("[BasisInputsCostTransform] unknown ", kBasisRegularizationField, " '", name,
                                                 "'; valid names are: ", absl::StrJoin(basisRegularizationNames(), ", "), "."));
}

/**
 * R_basis = Mᵀ · R_wrench · M + lambdaRegularization · blkdiag(S, 0), with S built by `builder`.
 *
 * @param R_wrench             Wrench-space weight matrix (wrenchInputDim × wrenchInputDim).
 * @param M                    Local basis-to-wrench map (wrenchInputDim × basisInputDim).
 * @param numBasisInputs       Number of leading λ entries in the basis-vector input (numBasisPerFoot · kNumContacts).
 * @param lambdaRegularization Non-negative weight of the regularization.
 */
inline matrix_t transformWrenchInputCostToBasisSpace(
    const matrix_t& R_wrench, const matrix_t& M, size_t numBasisInputs, scalar_t lambdaRegularization, BasisRegularizationBuilder builder) {
  ABSL_CHECK(R_wrench.rows() == M.rows() && R_wrench.cols() == M.rows())
      << "transformWrenchInputCostToBasisSpace: R_wrench is not square of M's rows";
  ABSL_CHECK_LE(static_cast<Eigen::Index>(numBasisInputs), M.cols())
      << "transformWrenchInputCostToBasisSpace: more basis inputs than M has columns";
  ABSL_CHECK_GE(lambdaRegularization, 0.0) << "transformWrenchInputCostToBasisSpace: lambdaRegularization must be non-negative";
  const Eigen::Index n = static_cast<Eigen::Index>(numBasisInputs);
  matrix_t R_basis = M.transpose() * R_wrench * M;
  if (lambdaRegularization != 0.0) {
    R_basis.topLeftCorner(n, n) += lambdaRegularization * builder(M, numBasisInputs);
  }
  return R_basis;
}

/** The full_diagonal form, R_basis = Mᵀ · R_wrench · M + lambdaRegularization · blkdiag(I, 0). */
inline matrix_t transformWrenchInputCostToBasisSpace(const matrix_t& R_wrench,
                                                     const matrix_t& M,
                                                     size_t numBasisInputs,
                                                     scalar_t lambdaRegularization) {
  return transformWrenchInputCostToBasisSpace(R_wrench, M, numBasisInputs, lambdaRegularization, &fullDiagonalRegularizationShape);
}

/**
 * Bundles everything needed to transform a wrench-space input cost into basis-vector space. CentroidalMpcInterface
 * builds it from the task file (getBasisInputsCostTransformConfig), and the OCP factory
 * (HumanoidCostConstraintFactory::setBasisInputsCostTransform) and the online parameter updater both transform through
 * it, so a hot reload applies exactly the regularization the start-up did.
 */
struct BasisInputsCostTransformConfig {
  matrix_t basisToWrenchMap;            ///< M (wrenchInputDim × basisInputDim), see BasisInputsModelDecorator::getLocalBasisToWrenchMap
  size_t wrenchInputDim = 0;            ///< Dimension of the wrench-space input (rows of M).
  size_t numBasisInputs = 0;            ///< Number of leading λ entries in the basis-vector input.
  scalar_t lambdaRegularization = 0.0;  ///< Weight `reg` of the λ regularization (kBasisScalingRegularizationField).
  /// Name of the λ regularization (kBasisRegularizationField); one of basisRegularizationNames().
  std::string regularization = std::string(kDefaultBasisRegularization);

  size_t basisInputDim() const { return static_cast<size_t>(basisToWrenchMap.cols()); }
};

/** Checks the parts of the config a task file sets; the messages name the key to change. */
inline absl::Status validateBasisInputsCostTransformConfig(const BasisInputsCostTransformConfig& config) {
  const absl::StatusOr<BasisRegularizationBuilder> builder = getBasisRegularizationBuilder(config.regularization);
  if (!builder.ok()) {
    return builder.status();
  }
  if (!(config.lambdaRegularization >= 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat("[BasisInputsCostTransform] ", kBasisScalingRegularizationField,
                                                   " must be non-negative, got ", config.lambdaRegularization, "."));
  }
  return absl::OkStatus();
}

/** The transform of a validated config (validateBasisInputsCostTransformConfig); an unknown regularization name is a CHECK failure. */
inline matrix_t transformWrenchInputCostToBasisSpace(const matrix_t& R_wrench, const BasisInputsCostTransformConfig& config) {
  const absl::StatusOr<BasisRegularizationBuilder> builder = getBasisRegularizationBuilder(config.regularization);
  ABSL_CHECK(builder.ok()) << builder.status();
  return transformWrenchInputCostToBasisSpace(R_wrench, config.basisToWrenchMap, config.numBasisInputs, config.lambdaRegularization,
                                              *builder);
}

/**
 * Whether the λ block of a transformed input cost is positive definite, which the QP needs for a unique input. It
 * fails when the regularization weight is zero, or under null_space when R gives some contact wrench direction no
 * weight; the message names the keys that fix it.
 */
inline absl::Status checkLambdaBlockPositiveDefinite(const matrix_t& R_basis, size_t numBasisInputs) {
  const Eigen::Index n = static_cast<Eigen::Index>(numBasisInputs);
  if (n == 0) {
    return absl::OkStatus();
  }
  const Eigen::SelfAdjointEigenSolver<matrix_t> eigenSolver(matrix_t(R_basis.topLeftCorner(n, n)), Eigen::EigenvaluesOnly);
  if (eigenSolver.info() != Eigen::Success) {
    return absl::InternalError("[BasisInputsCostTransform] the eigenvalue decomposition of the λ block of R did not converge.");
  }
  const scalar_t smallest = eigenSolver.eigenvalues().minCoeff();
  constexpr scalar_t kRelativeTolerance = 1.0e-12;
  const scalar_t tolerance = kRelativeTolerance * std::max<scalar_t>(1.0, eigenSolver.eigenvalues().cwiseAbs().maxCoeff());
  if (smallest <= tolerance) {
    return absl::InvalidArgumentError(
        absl::StrCat("[BasisInputsCostTransform] the λ block of the basis-space input cost is not positive definite (smallest eigenvalue ",
                     smallest, "). Give ", kBasisScalingRegularizationField, " a positive value, and with ", kBasisRegularizationField,
                     ": ", kNullSpaceBasisRegularization, " give every contact force and moment a positive weight in R."));
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid
