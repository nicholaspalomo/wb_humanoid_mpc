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
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
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
constexpr int kMomentXRowIndex = kForceDim + kXAxisIndex;
constexpr int kMomentYRowIndex = kForceDim + kYAxisIndex;

/** The columns that add a horizontal CoM offset (a, b) to the static wrench
 * equation W lambda + A (a, b) = e_z: A (a, b) = (0, 0, 0, -b, a, 0). */
Eigen::Matrix<double, kWrenchDim, kHorizontalPositionDim> comColumns() {
  Eigen::Matrix<double, kWrenchDim, kHorizontalPositionDim> columns =
      Eigen::Matrix<double, kWrenchDim, kHorizontalPositionDim>::Zero();
  columns(kMomentYRowIndex, kXAxisIndex) = 1.0;
  columns(kMomentXRowIndex, kYAxisIndex) = -1.0;
  return columns;
}

/** The normalized wrench (e_z, 0) that holds the weight with the CoM at the
 * stance point, about the stance point in world axes. */
Wrench computeWeightWnormWrenchStanceEwrtWorld() {
  Wrench wnormWrenchStanceEwrtWorld = Wrench::Zero();
  wnormWrenchStanceEwrtWorld(kZAxisIndex) = 1.0;
  return wnormWrenchStanceEwrtWorld;
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

Wrench StaticEquilibrium::computeStaticWnormWrenchStanceEwrtWorld(
    const Eigen::Vector2d& posComRtWorldEwrtWorldXy) const {
  const Eigen::Vector2d posComRtStanceEwrtWorldXy =
      posComRtWorldEwrtWorldXy -
      generators_.posStanceRtWorldEwrtWorld.head<kHorizontalPositionDim>();
  return computeWeightWnormWrenchStanceEwrtWorld() -
         comColumns() * posComRtStanceEwrtWorldXy;
}

absl::StatusOr<double>
StaticEquilibrium::computeResidualWnormWrenchStanceEwrtWorld(
    const Wrench& wnormWrenchStanceEwrtWorld) const {
  // Variables: lambda >= 0, then the slacks s+ >= 0 and s- >= 0 of the six
  // rows; minimize the L1 mismatch 1'(s+ + s-).
  const int generators = generators_.size();
  LinearProgram program =
      LinearProgram::Unconstrained(generators + kWrenchDim + kWrenchDim);
  program.cost.tail(kWrenchDim + kWrenchDim).setOnes();
  program.lowerBound.setZero();
  program.equalityMatrix.resize(kWrenchDim, program.cost.size());
  program.equalityMatrix << generators_.generatorWrenches,
      Eigen::MatrixXd::Identity(kWrenchDim, kWrenchDim),
      -Eigen::MatrixXd::Identity(kWrenchDim, kWrenchDim);
  program.equalityVector = wnormWrenchStanceEwrtWorld;
  ASSIGN_OR_RETURN(const LinearProgramSolution solution,
                   solveBoundedOnFailure(std::move(program), generators));
  return solution.x.tail(kWrenchDim + kWrenchDim).sum();
}

absl::StatusOr<double> StaticEquilibrium::equilibriumResidual(
    const Eigen::Vector2d& posComRtWorldEwrtWorldXy) const {
  return computeResidualWnormWrenchStanceEwrtWorld(
      computeStaticWnormWrenchStanceEwrtWorld(posComRtWorldEwrtWorldXy));
}

absl::StatusOr<Eigen::Vector2d>
StaticEquilibrium::computeSupportPosComRtStanceEwrtWorldXy(
    const Eigen::Vector2d& searchDirEwrtWorldXy) const {
  // Variables: lambda >= 0, then the CoM offset (a, b) from
  // posStanceRtWorldEwrtWorld (posComRtStanceEwrtWorldXy), within the search
  // radius; maximize searchDirEwrtWorldXy' (a, b).
  const int generators = generators_.size();
  LinearProgram program =
      LinearProgram::Unconstrained(generators + kHorizontalPositionDim);
  program.cost.tail<kHorizontalPositionDim>() = -searchDirEwrtWorldXy;
  program.lowerBound.head(generators).setZero();
  program.lowerBound.tail<kHorizontalPositionDim>().setConstant(
      -options_.searchRadius);
  program.upperBound.tail<kHorizontalPositionDim>().setConstant(
      options_.searchRadius);
  program.equalityMatrix.resize(kWrenchDim, program.cost.size());
  program.equalityMatrix << generators_.generatorWrenches, comColumns();
  program.equalityVector = computeWeightWnormWrenchStanceEwrtWorld();
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
  existence.equalityMatrix << generators_.generatorWrenches, comColumns(),
      Eigen::MatrixXd::Identity(kWrenchDim, kWrenchDim),
      -Eigen::MatrixXd::Identity(kWrenchDim, kWrenchDim);
  existence.equalityVector = computeWeightWnormWrenchStanceEwrtWorld();
  ASSIGN_OR_RETURN(const LinearProgramSolution solution,
                   solveBoundedOnFailure(std::move(existence), generators));
  if (!(solution.objective <= options_.feasibilityTolerance)) {
    return std::vector<Eigen::Vector2d>();
  }

  const Eigen::Vector2d posStanceRtWorldEwrtWorldXy =
      generators_.posStanceRtWorldEwrtWorld.head<kHorizontalPositionDim>();
  std::vector<Eigen::Vector2d> supportPosComRtWorldEwrtWorldXySamples;
  supportPosComRtWorldEwrtWorldXySamples.reserve(options_.marginDirections);
  for (int k = 0; k < options_.marginDirections; ++k) {
    const double angle = 2.0 * std::numbers::pi * k / options_.marginDirections;
    ASSIGN_OR_RETURN(const Eigen::Vector2d posComRtStanceEwrtWorldXy,
                     computeSupportPosComRtStanceEwrtWorldXy(
                         Eigen::Vector2d(std::cos(angle), std::sin(angle))));
    supportPosComRtWorldEwrtWorldXySamples.push_back(
        posStanceRtWorldEwrtWorldXy + posComRtStanceEwrtWorldXy);
  }
  switch (options_.supportPolygon) {
    case SupportPolygon::kSampled:
      return convexHull(supportPosComRtWorldEwrtWorldXySamples, kHullTolerance);
    case SupportPolygon::kRefined:
      return refinedHull(std::move(supportPosComRtWorldEwrtWorldXySamples));
  }
  return absl::InternalError("unknown SupportPolygon");
}

absl::StatusOr<std::vector<Eigen::Vector2d>> StaticEquilibrium::refinedHull(
    std::vector<Eigen::Vector2d> supportPosComRtWorldEwrtWorldXySamples) const {
  const Eigen::Vector2d posStanceRtWorldEwrtWorldXy =
      generators_.posStanceRtWorldEwrtWorld.head<kHorizontalPositionDim>();
  std::vector<Eigen::Vector2d> hullPosComRtWorldEwrtWorldXySamples =
      convexHull(supportPosComRtWorldEwrtWorldXySamples, kHullTolerance);
  int added = 0;
  bool grew = true;
  while (grew && added < kMaxRefinementPoints) {
    grew = false;
    for (size_t i = 0; i < hullPosComRtWorldEwrtWorldXySamples.size() &&
                       added < kMaxRefinementPoints;
         ++i) {
      const Eigen::Vector2d& fromPosComRtWorldEwrtWorldXy =
          hullPosComRtWorldEwrtWorldXySamples[i];
      const Eigen::Vector2d edgeErrPosComRtWorldEwrtWorldXy =
          hullPosComRtWorldEwrtWorldXySamples
              [(i + 1) % hullPosComRtWorldEwrtWorldXySamples.size()] -
          fromPosComRtWorldEwrtWorldXy;
      if (edgeErrPosComRtWorldEwrtWorldXy.norm() <= kHullTolerance) continue;
      // Counterclockwise: the outside is on the edge's right.
      const Eigen::Vector2d outwardDirEwrtWorldXy =
          Eigen::Vector2d(edgeErrPosComRtWorldEwrtWorldXy.y(),
                          -edgeErrPosComRtWorldEwrtWorldXy.x())
              .normalized();
      ASSIGN_OR_RETURN(
          const Eigen::Vector2d posComRtStanceEwrtWorldXy,
          computeSupportPosComRtStanceEwrtWorldXy(outwardDirEwrtWorldXy));
      const Eigen::Vector2d supportPosComRtWorldEwrtWorldXy =
          posStanceRtWorldEwrtWorldXy + posComRtStanceEwrtWorldXy;
      if (outwardDirEwrtWorldXy.dot(supportPosComRtWorldEwrtWorldXy -
                                    fromPosComRtWorldEwrtWorldXy) >
          kRefinementTolerance) {
        supportPosComRtWorldEwrtWorldXySamples.push_back(
            supportPosComRtWorldEwrtWorldXy);
        ++added;
        grew = true;
      }
    }
    if (grew)
      hullPosComRtWorldEwrtWorldXySamples =
          convexHull(supportPosComRtWorldEwrtWorldXySamples, kHullTolerance);
  }
  return hullPosComRtWorldEwrtWorldXySamples;
}

absl::StatusOr<Eigen::VectorXd> StaticEquilibrium::solveWithExtraVariables(
    const Wrench& wnormWrenchStanceEwrtWorld,
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
  program.equalityMatrix << generators_.generatorWrenches,
      Eigen::MatrixXd::Zero(kWrenchDim, extra);
  program.equalityVector = wnormWrenchStanceEwrtWorld;
  keepIndependentEqualities(program);
  program.inequalityMatrix = rows;
  program.inequalityLower = lower;
  program.inequalityUpper = upper;
  ASSIGN_OR_RETURN(const LinearProgramSolution solution,
                   solveBoundedOnFailure(std::move(program), generators));
  return solution.x;
}

absl::StatusOr<SupportPolygon> supportPolygonOf(absl::string_view name) {
  for (size_t i = 0; i < kSupportPolygonNames.size(); ++i) {
    if (kSupportPolygonNames[i] == name) return static_cast<SupportPolygon>(i);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "unknown support_polygon '", name,
      "' (known: ", absl::StrJoin(kSupportPolygonNames, ", "), ")"));
}

std::optional<double> staticMargin(
    absl::Span<const Eigen::Vector2d> supportPosComRtWorldEwrtWorldXySamples,
    const Eigen::Vector2d& posComRtWorldEwrtWorldXy) {
  if (supportPosComRtWorldEwrtWorldXySamples.empty()) return std::nullopt;
  return signedDistanceToConvexPolygon(supportPosComRtWorldEwrtWorldXySamples,
                                       posComRtWorldEwrtWorldXy);
}

}  // namespace ocs2::humanoid::multi_contact
