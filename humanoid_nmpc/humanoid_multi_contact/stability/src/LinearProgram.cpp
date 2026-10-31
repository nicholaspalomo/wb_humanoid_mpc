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

#include "humanoid_multi_contact/stability/LinearProgram.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "Eigen/QR"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

extern "C" {
#include "hpipm_common.h"  // NOLINT(build/include_subdir): HPIPM installs its headers in no directory
#include "hpipm_d_dense_qp.h"  // NOLINT(build/include_subdir): HPIPM installs its headers in no directory
#include "hpipm_d_dense_qp_dim.h"  // NOLINT(build/include_subdir): HPIPM installs its headers in no directory
#include "hpipm_d_dense_qp_ipm.h"  // NOLINT(build/include_subdir): HPIPM installs its headers in no directory
#include "hpipm_d_dense_qp_sol.h"  // NOLINT(build/include_subdir): HPIPM installs its headers in no directory
}

namespace ocs2::humanoid::multi_contact {
namespace {

absl::Status sizeError(absl::string_view what) {
  return absl::InvalidArgumentError(
      absl::StrCat("[solveLinearProgram] ", what));
}

absl::Status checkSizes(const LinearProgram& program) {
  const Eigen::Index variables = program.cost.size();
  if (program.equalityMatrix.cols() != variables ||
      program.equalityMatrix.rows() != program.equalityVector.size()) {
    return sizeError("the equalities do not match the variables");
  }
  if (program.lowerBound.size() != variables ||
      program.upperBound.size() != variables) {
    return sizeError("the bounds do not match the variables");
  }
  if (program.inequalityMatrix.cols() != variables ||
      program.inequalityMatrix.rows() != program.inequalityLower.size() ||
      program.inequalityMatrix.rows() != program.inequalityUpper.size()) {
    return sizeError("the inequalities do not match the variables");
  }
  if ((program.lowerBound.array() > program.upperBound.array()).any() ||
      (program.inequalityLower.array() > program.inequalityUpper.array())
          .any()) {
    return sizeError("a lower bound is above its upper bound");
  }
  if (!program.cost.allFinite() || !program.equalityMatrix.allFinite() ||
      !program.equalityVector.allFinite() ||
      !program.inequalityMatrix.allFinite()) {
    return sizeError("the data is not finite");
  }
  return absl::OkStatus();
}

/** A finite value, and the mask HPIPM reads (1: the bound is there). */
struct MaskedBounds {
  Eigen::VectorXd values;
  Eigen::VectorXd mask;
};

MaskedBounds masked(const Eigen::VectorXd& bounds) {
  MaskedBounds result{.values = bounds,
                      .mask = Eigen::VectorXd::Ones(bounds.size())};
  for (Eigen::Index i = 0; i < bounds.size(); ++i) {
    if (!std::isfinite(bounds(i))) {
      result.values(i) = 0.0;
      result.mask(i) = 0.0;
    }
  }
  return result;
}

std::string statusName(int status) {
  switch (status) {
    case MAX_ITER:
      return "MAX_ITER";
    case MIN_STEP:
      return "MIN_STEP";
    case NAN_SOL:
      return "NAN_SOL";
    case INCONS_EQ:
      return "INCONS_EQ";
    default:
      return absl::StrCat("status ", status);
  }
}

}  // namespace

LinearProgram LinearProgram::Unconstrained(int variables) {
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  LinearProgram program;
  program.cost = Eigen::VectorXd::Zero(variables);
  program.equalityMatrix = Eigen::MatrixXd::Zero(0, variables);
  program.equalityVector = Eigen::VectorXd::Zero(0);
  program.lowerBound = Eigen::VectorXd::Constant(variables, -kInfinity);
  program.upperBound = Eigen::VectorXd::Constant(variables, kInfinity);
  program.inequalityMatrix = Eigen::MatrixXd::Zero(0, variables);
  program.inequalityLower = Eigen::VectorXd::Zero(0);
  program.inequalityUpper = Eigen::VectorXd::Zero(0);
  return program;
}

namespace {

/**
 * Solves `program` with the Hessian `hessian` (checked by the callers) using
 * HPIPM with `settings`.
 */
absl::StatusOr<LinearProgramSolution> solveDenseQp(
    const Eigen::MatrixXd& hessianIn,
    const LinearProgram& program,
    const QpSolverSettings& settings) {
  const int variables = static_cast<int>(program.cost.size());
  const int equalities = static_cast<int>(program.equalityMatrix.rows());
  const int inequalities = static_cast<int>(program.inequalityMatrix.rows());

  // Every variable carries a box constraint; the masks drop the infinite sides.
  std::vector<int> boxIndices(variables);
  for (int i = 0; i < variables; ++i) boxIndices[i] = i;
  MaskedBounds lower = masked(program.lowerBound);
  MaskedBounds upper = masked(program.upperBound);
  MaskedBounds inequalityLower = masked(program.inequalityLower);
  MaskedBounds inequalityUpper = masked(program.inequalityUpper);

  Eigen::MatrixXd hessian = hessianIn;
  Eigen::VectorXd gradient = program.cost;
  Eigen::MatrixXd equalityMatrix = program.equalityMatrix;
  Eigen::VectorXd equalityVector = program.equalityVector;
  // HPIPM's C is ng x nv, column-major like Eigen's default.
  Eigen::MatrixXd inequalityMatrix = program.inequalityMatrix;

  d_dense_qp_dim dim;
  std::vector<char> dimMemory(d_dense_qp_dim_memsize());
  d_dense_qp_dim_create(&dim, dimMemory.data());
  d_dense_qp_dim_set_all(variables, equalities, variables, inequalities,
                         /*nsb=*/0, /*nsg=*/0, &dim);

  d_dense_qp qp;
  std::vector<char> qpMemory(d_dense_qp_memsize(&dim));
  d_dense_qp_create(&dim, &qp, qpMemory.data());
  d_dense_qp_set_all(hessian.data(), gradient.data(), equalityMatrix.data(),
                     equalityVector.data(), boxIndices.data(),
                     lower.values.data(), upper.values.data(),
                     inequalityMatrix.data(), inequalityLower.values.data(),
                     inequalityUpper.values.data(), /*Zl=*/nullptr,
                     /*Zu=*/nullptr, /*zl=*/nullptr, /*zu=*/nullptr,
                     /*idxs=*/nullptr, /*d_ls=*/nullptr, /*d_us=*/nullptr, &qp);
  d_dense_qp_set_lb_mask(lower.mask.data(), &qp);
  d_dense_qp_set_ub_mask(upper.mask.data(), &qp);
  if (inequalities > 0) {
    d_dense_qp_set_lg_mask(inequalityLower.mask.data(), &qp);
    d_dense_qp_set_ug_mask(inequalityUpper.mask.data(), &qp);
  }

  d_dense_qp_sol solution;
  std::vector<char> solutionMemory(d_dense_qp_sol_memsize(&dim));
  d_dense_qp_sol_create(&dim, &solution, solutionMemory.data());

  d_dense_qp_ipm_arg arg;
  std::vector<char> argMemory(d_dense_qp_ipm_arg_memsize(&dim));
  d_dense_qp_ipm_arg_create(&dim, &arg, argMemory.data());
  d_dense_qp_ipm_arg_set_default(
      ROBUST, &arg);  // NOLINT(claudisms): HPIPM mode enumerator
  int maxIterations = settings.maxIterations;
  double stationarityTolerance = settings.stationarityTolerance;
  double equalityTolerance = settings.equalityTolerance;
  double inequalityTolerance = settings.inequalityTolerance;
  double complementarityTolerance = settings.complementarityTolerance;
  d_dense_qp_ipm_arg_set_iter_max(&maxIterations, &arg);
  d_dense_qp_ipm_arg_set_tol_stat(&stationarityTolerance, &arg);
  d_dense_qp_ipm_arg_set_tol_eq(&equalityTolerance, &arg);
  d_dense_qp_ipm_arg_set_tol_ineq(&inequalityTolerance, &arg);
  d_dense_qp_ipm_arg_set_tol_comp(&complementarityTolerance, &arg);

  d_dense_qp_ipm_ws workspace;
  std::vector<char> workspaceMemory(d_dense_qp_ipm_ws_memsize(&dim, &arg));
  d_dense_qp_ipm_ws_create(&dim, &arg, &workspace, workspaceMemory.data());

  d_dense_qp_ipm_solve(&qp, &solution, &arg, &workspace);
  int status = 0;
  d_dense_qp_ipm_get_status(&workspace, &status);
  LinearProgramSolution result;
  d_dense_qp_ipm_get_iter(&workspace, &result.iterations);
  if (status != SUCCESS) {
    return absl::InternalError(absl::StrCat(
        "[solveLinearProgram] HPIPM did not converge: ", statusName(status),
        " after ", result.iterations, " iterations (", variables,
        " variables, ", equalities, " equalities, ", inequalities,
        " inequalities)"));
  }
  result.x.resize(variables);
  d_dense_qp_sol_get_v(&solution, result.x.data());
  result.objective =
      program.cost.dot(result.x) + 0.5 * result.x.dot(hessianIn * result.x);
  return result;
}

}  // namespace

absl::StatusOr<LinearProgramSolution> solveLinearProgram(
    const LinearProgram& program) {
  RETURN_IF_ERROR(checkSizes(program));
  const Eigen::Index variables = program.cost.size();
  const Eigen::MatrixXd hessian =
      kLinearProgramRegularization *
      Eigen::MatrixXd::Identity(variables, variables);
  absl::StatusOr<LinearProgramSolution> solution =
      solveDenseQp(hessian, program, QpSolverSettings());
  if (!solution.ok()) {
    // A program HPIPM stops on (MAX_ITER on a knot of a tracked climb) gets
    // more iterations, at tolerances still below the callers' 1e-6, once.
    const QpSolverSettings patient{
        .maxIterations = kRetryQpMaxIterations,
        .stationarityTolerance = kRetryQpTolerance,
        .equalityTolerance = kRetryQpTolerance,
        .inequalityTolerance = kRetryQpTolerance,
        .complementarityTolerance = kRetryQpTolerance};
    solution = solveDenseQp(hessian, program, patient);
  }
  if (!solution.ok()) return solution.status();
  solution->objective = program.cost.dot(solution->x);
  return solution;
}

absl::StatusOr<LinearProgramSolution> solveQuadraticProgram(
    const Eigen::MatrixXd& hessian,
    const LinearProgram& program,
    const QpSolverSettings& settings) {
  RETURN_IF_ERROR(checkSizes(program));
  const Eigen::Index variables = program.cost.size();
  if (hessian.rows() != variables || hessian.cols() != variables ||
      !hessian.allFinite()) {
    return sizeError("the Hessian does not match the variables");
  }
  return solveDenseQp(hessian, program, settings);
}

Eigen::VectorXi independentRows(const Eigen::MatrixXd& matrix,
                                double relativeTolerance) {
  if (matrix.rows() == 0) return Eigen::VectorXi(0);
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> decomposition(matrix.cols(),
                                                            matrix.rows());
  decomposition.setThreshold(relativeTolerance);
  decomposition.compute(matrix.transpose());
  const Eigen::Index rank = decomposition.rank();
  std::vector<int> rows(rank);
  for (Eigen::Index i = 0; i < rank; ++i) {
    rows[i] = decomposition.colsPermutation().indices()(i);
  }
  std::sort(rows.begin(), rows.end());
  return Eigen::Map<const Eigen::VectorXi>(rows.data(),
                                           static_cast<Eigen::Index>(rank));
}

}  // namespace ocs2::humanoid::multi_contact
