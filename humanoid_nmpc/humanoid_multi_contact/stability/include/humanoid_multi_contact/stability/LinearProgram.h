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

#include "Eigen/Core"
#include "absl/status/statusor.h"

namespace ocs2::humanoid::multi_contact {

/**
 * The Hessian regularization of every LP: HPIPM's dense QP solves
 * min c'x + 0.5 * kLinearProgramRegularization * |x|^2, which is strictly
 * convex and so has one minimizer, the LP's optimum of least norm up to
 * O(1e-9). It also bounds the directions in which an LP's objective is flat,
 * such as the internal forces between two opposing contacts.
 */
inline constexpr double kLinearProgramRegularization = 1.0e-9;

/**
 * A linear program in the form HPIPM's dense QP takes (design section 6):
 *
 *   minimize    cost' x
 *   subject to  equalityMatrix x = equalityVector
 *               lowerBound <= x <= upperBound            (per variable)
 *               inequalityLower <= inequalityMatrix x <= inequalityUpper
 *
 * A bound of -inf or +inf is absent. The equality rows must be linearly
 * independent; independentRows() removes the dependent ones. Passive data.
 */
struct LinearProgram {
  Eigen::VectorXd cost;
  Eigen::MatrixXd equalityMatrix;
  Eigen::VectorXd equalityVector;
  Eigen::VectorXd lowerBound;
  Eigen::VectorXd upperBound;
  Eigen::MatrixXd inequalityMatrix;
  Eigen::VectorXd inequalityLower;
  Eigen::VectorXd inequalityUpper;

  /** An LP with `variables` unbounded variables and no constraints: zero cost,
   * bounds of -inf and +inf, and empty matrices of the right width. */
  static LinearProgram Unconstrained(int variables);
};

/** What solveLinearProgram() returns. Passive data. */
struct LinearProgramSolution {
  Eigen::VectorXd x;
  double objective = 0.0;  // cost' x, plus 0.5 x' H x for a QP
  int iterations = 0;
};

/**
 * Solves `program` with HPIPM's dense interior-point QP solver, with the
 * Hessian kLinearProgramRegularization * I.
 *
 * InvalidArgument for inconsistent sizes or a lower bound above an upper one;
 * Internal, naming HPIPM's status, when HPIPM does not converge, also on one
 * retry with kRetryQpMaxIterations iterations (an infeasible or unbounded
 * program among them: callers build programs that are neither).
 * Allocates HPIPM's workspace on every call; thread-safe.
 */
absl::StatusOr<LinearProgramSolution> solveLinearProgram(
    const LinearProgram& program);

/** HPIPM's settings for the certifier's LPs by default: tolerances well below
 * the 1e-6 the callers judge feasibility with. */
inline constexpr int kDefaultQpMaxIterations = 200;
inline constexpr double kDefaultQpStationarityTolerance = 1.0e-9;
inline constexpr double kDefaultQpTolerance = 1.0e-10;
/** The iterations and tolerances of solveLinearProgram()'s one retry of a
 * program HPIPM did not converge on. */
inline constexpr int kRetryQpMaxIterations = 2000;
inline constexpr double kRetryQpTolerance = 1.0e-8;

/** The settings of HPIPM's interior-point iterations. */
struct QpSolverSettings {
  int maxIterations = kDefaultQpMaxIterations;
  double stationarityTolerance = kDefaultQpStationarityTolerance;
  double equalityTolerance = kDefaultQpTolerance;
  double inequalityTolerance = kDefaultQpTolerance;
  double complementarityTolerance = kDefaultQpTolerance;
};

/**
 * Solves the quadratic program
 *
 *   minimize 0.5 x' hessian x + program.cost' x
 *
 * under the constraints of `program`, with HPIPM's dense interior-point QP
 * solver. `hessian` must be symmetric and positive definite, of the size of
 * the variables. The errors of solveLinearProgram(), and InvalidArgument for a
 * Hessian of the wrong size or not finite. The objective is the quadratic one.
 * `settings` are HPIPM's tolerances, the LPs' by default. Thread-safe.
 */
absl::StatusOr<LinearProgramSolution> solveQuadraticProgram(
    const Eigen::MatrixXd& hessian,
    const LinearProgram& program,
    const QpSolverSettings& settings = QpSolverSettings());

/**
 * The rows of `matrix` that a rank-revealing QR keeps as linearly independent,
 * ascending: equalities on them hold exactly when all rows do and the system is
 * consistent. A row whose pivot is below `relativeTolerance` times the largest
 * is dependent.
 */
Eigen::VectorXi independentRows(const Eigen::MatrixXd& matrix,
                                double relativeTolerance);

}  // namespace ocs2::humanoid::multi_contact
