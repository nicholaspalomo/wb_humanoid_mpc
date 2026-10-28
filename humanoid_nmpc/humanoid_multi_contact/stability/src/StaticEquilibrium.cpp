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

#include "humanoid_multi_contact/stability/StaticEquilibrium.h"

#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/stability/ConvexPolygon.h"
#include "humanoid_multi_contact/stability/LinearProgram.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// The fewest directions whose support points can span a polygon.
constexpr int kMinMarginDirections = 3;
// The relative pivot below which a row of an equality system is dependent.
constexpr double kRankTolerance = 1.0e-10;
// [m] Support points closer than this are one vertex of the polygon.
constexpr double kHullTolerance = 1.0e-7;
// The rows of a wrench: the moment about x and about y, which a horizontal CoM
// offset (a, b) changes: (c - r) x e_z = (b, -a, 0).
constexpr int kMomentXRow = kForceDim + kXAxisIndex;
constexpr int kMomentYRow = kForceDim + kYAxisIndex;

/** The columns that add a horizontal CoM offset (a, b) to the static wrench
 * equation W lambda + A (a, b) = e_z: A (a, b) = (0, 0, 0, -b, a, 0). */
Eigen::Matrix<double, kWrenchDim, kHorizontalPositionDim> comColumns() {
  Eigen::Matrix<double, kWrenchDim, kHorizontalPositionDim> columns =
      Eigen::Matrix<double, kWrenchDim, kHorizontalPositionDim>::Zero();
  columns(kMomentYRow, kXAxisIndex) = 1.0;
  columns(kMomentXRow, kYAxisIndex) = -1.0;
  return columns;
}

Wrench weightSupport() {
  Wrench wrench = Wrench::Zero();
  wrench(kZAxisIndex) = 1.0;
  return wrench;
}

// The bound on every generator's scaling in the retry of an LP HPIPM stops on
// (solveBoundedOnFailure()): ten body weights along one generator, which no
// stance needs. A generator's force part is a unit normal plus a tangent
// within the friction cone, and the wrenches are per body weight.
constexpr double kRetryGeneratorBound = 10.0;

/** solveLinearProgram() of `program`, whose first `generators` variables are
 * the generators' scalings; when HPIPM stops on it, solved again with those
 * scalings bounded by kRetryGeneratorBound. The LPs are degenerate: internal
 * forces between two contacts (opposed tangents) add scalings that change
 * no wrench, so the optimal set is unbounded, and HPIPM's interior point, on
 * the 1e-9 regularization alone, can run out of iterations on it (MAX_ITER on
 * the wrench cone of a tracked climb). The bound makes the set compact
 * without excluding any solution a stance can use. The first error when the
 * retry fails too. */
absl::StatusOr<LinearProgramSolution> solveBoundedOnFailure(
    LinearProgram program, int generators) {
  absl::StatusOr<LinearProgramSolution> solution = solveLinearProgram(program);
  if (solution.ok()) return solution;
  program.upperBound.head(generators).setConstant(kRetryGeneratorBound);
  absl::StatusOr<LinearProgramSolution> bounded = solveLinearProgram(program);
  if (bounded.ok()) return bounded;
  return solution.status();
}

/** `program` restricted to the independent rows of its equalities. */
void keepIndependentEqualities(LinearProgram& program) {
  const Eigen::VectorXi rows =
      independentRows(program.equalityMatrix, kRankTolerance);
  program.equalityMatrix =
      Eigen::MatrixXd(program.equalityMatrix(rows, Eigen::all));
  program.equalityVector = Eigen::VectorXd(program.equalityVector(rows));
}

}  // namespace

StaticEquilibrium::StaticEquilibrium(Stance stance,
                                     WrenchGenerators generators,
                                     const StaticEquilibriumOptions& options)
    : stance_(std::move(stance)),
      generators_(std::move(generators)),
      options_(options) {}

absl::StatusOr<StaticEquilibrium> StaticEquilibrium::Create(
    Stance stance, const StaticEquilibriumOptions& options) {
  if (options.marginDirections < kMinMarginDirections) {
    return absl::InvalidArgumentError(
        absl::StrCat("the static margin needs at least ", kMinMarginDirections,
                     " directions, not ", options.marginDirections));
  }
  if (!(options.searchRadius > 0.0) || !(options.feasibilityTolerance > 0.0)) {
    return absl::InvalidArgumentError(
        "the search radius and the feasibility tolerance must be positive");
  }
  ASSIGN_OR_RETURN(WrenchGenerators generators,
                   buildWrenchGenerators(stance, options.frictionConeEdges));
  return StaticEquilibrium(std::move(stance), std::move(generators), options);
}

Wrench StaticEquilibrium::staticWrench(const Eigen::Vector2d& comXy) const {
  const Eigen::Vector2d offset =
      comXy - generators_.reference.head<kHorizontalPositionDim>();
  return weightSupport() - comColumns() * offset;
}

absl::StatusOr<double> StaticEquilibrium::wrenchResidual(
    const Wrench& wrench) const {
  // Variables: lambda >= 0, then the slacks s+ >= 0 and s- >= 0 of the six
  // rows; minimize the L1 mismatch 1'(s+ + s-).
  const int generators = generators_.size();
  LinearProgram program =
      LinearProgram::Unconstrained(generators + kWrenchDim + kWrenchDim);
  program.cost.tail(kWrenchDim + kWrenchDim).setOnes();
  program.lowerBound.setZero();
  program.equalityMatrix.resize(kWrenchDim, program.cost.size());
  program.equalityMatrix << generators_.wrenches,
      Eigen::MatrixXd::Identity(kWrenchDim, kWrenchDim),
      -Eigen::MatrixXd::Identity(kWrenchDim, kWrenchDim);
  program.equalityVector = wrench;
  ASSIGN_OR_RETURN(const LinearProgramSolution solution,
                   solveBoundedOnFailure(std::move(program), generators));
  return solution.x.tail(kWrenchDim + kWrenchDim).sum();
}

absl::StatusOr<double> StaticEquilibrium::equilibriumResidual(
    const Eigen::Vector2d& comXy) const {
  return wrenchResidual(staticWrench(comXy));
}

absl::StatusOr<Eigen::Vector2d> StaticEquilibrium::supportPoint(
    const Eigen::Vector2d& direction) const {
  // Variables: lambda >= 0, then the CoM offset (a, b) from the reference
  // point, within the search radius; maximize direction' (a, b).
  const int generators = generators_.size();
  LinearProgram program =
      LinearProgram::Unconstrained(generators + kHorizontalPositionDim);
  program.cost.tail<kHorizontalPositionDim>() = -direction;
  program.lowerBound.head(generators).setZero();
  program.lowerBound.tail<kHorizontalPositionDim>().setConstant(
      -options_.searchRadius);
  program.upperBound.tail<kHorizontalPositionDim>().setConstant(
      options_.searchRadius);
  program.equalityMatrix.resize(kWrenchDim, program.cost.size());
  program.equalityMatrix << generators_.wrenches, comColumns();
  program.equalityVector = weightSupport();
  keepIndependentEqualities(program);
  ASSIGN_OR_RETURN(const LinearProgramSolution solution,
                   solveBoundedOnFailure(std::move(program), generators));
  return Eigen::Vector2d(solution.x.tail<kHorizontalPositionDim>());
}

absl::StatusOr<std::vector<Eigen::Vector2d>> StaticEquilibrium::supportPolygon()
    const {
  // Is any CoM position within the search radius in equilibrium? The L1
  // mismatch of the static wrench with the offset free.
  const int generators = generators_.size();
  LinearProgram existence = LinearProgram::Unconstrained(
      generators + kHorizontalPositionDim + kWrenchDim + kWrenchDim);
  existence.cost.tail(kWrenchDim + kWrenchDim).setOnes();
  existence.lowerBound.setZero();
  existence.lowerBound.segment<kHorizontalPositionDim>(generators)
      .setConstant(-options_.searchRadius);
  existence.upperBound.segment<kHorizontalPositionDim>(generators)
      .setConstant(options_.searchRadius);
  existence.equalityMatrix.resize(kWrenchDim, existence.cost.size());
  existence.equalityMatrix << generators_.wrenches, comColumns(),
      Eigen::MatrixXd::Identity(kWrenchDim, kWrenchDim),
      -Eigen::MatrixXd::Identity(kWrenchDim, kWrenchDim);
  existence.equalityVector = weightSupport();
  ASSIGN_OR_RETURN(const LinearProgramSolution solution,
                   solveBoundedOnFailure(std::move(existence), generators));
  if (!(solution.objective <= options_.feasibilityTolerance)) {
    return std::vector<Eigen::Vector2d>();
  }

  const Eigen::Vector2d reference =
      generators_.reference.head<kHorizontalPositionDim>();
  std::vector<Eigen::Vector2d> points;
  points.reserve(options_.marginDirections);
  for (int k = 0; k < options_.marginDirections; ++k) {
    const double angle = 2.0 * std::numbers::pi * k / options_.marginDirections;
    ASSIGN_OR_RETURN(
        const Eigen::Vector2d offset,
        supportPoint(Eigen::Vector2d(std::cos(angle), std::sin(angle))));
    points.push_back(reference + offset);
  }
  return convexHull(points, kHullTolerance);
}

absl::StatusOr<Eigen::VectorXd> StaticEquilibrium::solveWithExtraVariables(
    const Wrench& wrench,
    const Eigen::VectorXd& extraCost,
    const Eigen::MatrixXd& rows,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper) const {
  const int generators = generators_.size();
  const int extra = static_cast<int>(extraCost.size());
  if (rows.cols() != generators + extra || rows.rows() != lower.size() ||
      rows.rows() != upper.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the extra rows must have ", generators + extra, " columns"));
  }
  LinearProgram program = LinearProgram::Unconstrained(generators + extra);
  program.cost.tail(extra) = extraCost;
  program.lowerBound.head(generators).setZero();
  program.equalityMatrix.resize(kWrenchDim, generators + extra);
  program.equalityMatrix << generators_.wrenches,
      Eigen::MatrixXd::Zero(kWrenchDim, extra);
  program.equalityVector = wrench;
  keepIndependentEqualities(program);
  program.inequalityMatrix = rows;
  program.inequalityLower = lower;
  program.inequalityUpper = upper;
  ASSIGN_OR_RETURN(const LinearProgramSolution solution,
                   solveBoundedOnFailure(std::move(program), generators));
  return solution.x;
}

std::optional<double> staticMargin(absl::Span<const Eigen::Vector2d> polygon,
                                   const Eigen::Vector2d& comXy) {
  if (polygon.empty()) return std::nullopt;
  return signedDistanceToConvexPolygon(polygon, comXy);
}

}  // namespace ocs2::humanoid::multi_contact
