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

// The LP wrapper, the convex polygon helpers, static equilibrium on stances
// with closed-form answers (design section 9) and the contact wrench cone
// check.

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_multi_contact/stability/ContactWrenchConeCheck.h"
#include "humanoid_multi_contact/stability/ConvexPolygon.h"
#include "humanoid_multi_contact/stability/LinearProgram.h"
#include "humanoid_multi_contact/stability/Stance.h"
#include "humanoid_multi_contact/stability/StaticEquilibrium.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr double kFriction = 0.7;
// The DRC Atlas's soles, 0.24 x 0.11 m, about their centers.
constexpr double kSoleHalfLength = 0.12;
constexpr double kSoleHalfWidth = 0.055;
constexpr double kFootY = 0.1115;
constexpr double kFootX = 0.04;

StanceContact foot(const std::string& name, double y) {
  return StanceContact{.name = name,
                       .kind = ContactKind::kSurface,
                       .position = Eigen::Vector3d(kFootX, y, 0.0),
                       .rotation = Eigen::Matrix3d::Identity(),
                       .friction = kFriction,
                       .rectangle = ContactRectangle{.xMin = -kSoleHalfLength,
                                                     .xMax = kSoleHalfLength,
                                                     .yMin = -kSoleHalfWidth,
                                                     .yMax = kSoleHalfWidth}};
}

/** A point contact whose cone's z axis is `normal`. */
StanceContact point(const std::string& name,
                    const Eigen::Vector3d& position,
                    const Eigen::Vector3d& normal) {
  Eigen::Matrix3d rotation;
  rotation.col(2) = normal.normalized();
  rotation.col(0) = rotation.col(2).unitOrthogonal();
  rotation.col(1) = rotation.col(2).cross(rotation.col(0));
  return StanceContact{.name = name,
                       .kind = ContactKind::kPoint,
                       .position = position,
                       .rotation = rotation,
                       .friction = kFriction,
                       .rectangle = std::nullopt};
}

Stance twoFeet() {
  return {foot("left", kFootY), foot("right", -kFootY)};
}

StaticEquilibrium equilibriumOf(const Stance& stance) {
  absl::StatusOr<StaticEquilibrium> equilibrium =
      StaticEquilibrium::Create(stance, StaticEquilibriumOptions());
  EXPECT_TRUE(equilibrium.ok()) << equilibrium.status();
  return *std::move(equilibrium);
}

std::vector<Eigen::Vector2d> polygonOf(const Stance& stance) {
  const absl::StatusOr<std::vector<Eigen::Vector2d>> polygon =
      equilibriumOf(stance).supportPolygon();
  EXPECT_TRUE(polygon.ok()) << polygon.status();
  return polygon.ok() ? *polygon : std::vector<Eigen::Vector2d>();
}

double polygonArea(const std::vector<Eigen::Vector2d>& polygon) {
  double area = 0.0;
  for (size_t i = 0; i < polygon.size(); ++i) {
    const Eigen::Vector2d& a = polygon[i];
    const Eigen::Vector2d& b = polygon[(i + 1) % polygon.size()];
    area += 0.5 * (a.x() * b.y() - a.y() * b.x());
  }
  return area;
}

TEST(LinearProgramTest, SolvesASmallProgram) {
  // min -x0 - 2 x1 s.t. x0 + x1 <= 1, x0 = 0.3, x >= 0: x = (0.3, 0.7).
  LinearProgram program = LinearProgram::Unconstrained(2);
  program.cost << -1.0, -2.0;
  program.lowerBound.setZero();
  program.equalityMatrix.resize(1, 2);
  program.equalityMatrix << 1.0, 0.0;
  program.equalityVector = Eigen::VectorXd::Constant(1, 0.3);
  program.inequalityMatrix.resize(1, 2);
  program.inequalityMatrix << 1.0, 1.0;
  program.inequalityLower =
      Eigen::VectorXd::Constant(1, -std::numeric_limits<double>::infinity());
  program.inequalityUpper = Eigen::VectorXd::Ones(1);
  const absl::StatusOr<LinearProgramSolution> solution =
      solveLinearProgram(program);
  ASSERT_TRUE(solution.ok()) << solution.status();
  EXPECT_NEAR(solution->x(0), 0.3, 1.0e-7);
  EXPECT_NEAR(solution->x(1), 0.7, 1.0e-7);
  EXPECT_NEAR(solution->objective, -1.7, 1.0e-7);

  program.lowerBound(0) = 1.0;
  program.upperBound(0) = 0.0;
  EXPECT_EQ(solveLinearProgram(program).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(LinearProgramTest, IndependentRows) {
  Eigen::MatrixXd matrix(4, 3);
  matrix << 1.0, 0.0, 0.0,  //
      2.0, 0.0, 0.0,        //
      0.0, 1.0, 1.0,        //
      1.0, 1.0, 1.0;
  const Eigen::VectorXi rows =
      independentRows(matrix, /*relativeTolerance=*/1.0e-10);
  EXPECT_EQ(rows.size(), 2);
}

TEST(ConvexPolygonTest, HullAndSignedDistance) {
  const std::vector<Eigen::Vector2d> points = {
      Eigen::Vector2d(1.0, 1.0),         Eigen::Vector2d(-1.0, 1.0),
      Eigen::Vector2d(0.0, 0.0),         Eigen::Vector2d(-1.0, -1.0),
      Eigen::Vector2d(1.0, -1.0),        Eigen::Vector2d(1.0, 0.0),
      Eigen::Vector2d(1.0, 1.0 + 1.0e-9)};
  const std::vector<Eigen::Vector2d> hull =
      convexHull(points, /*tolerance=*/1.0e-7);
  ASSERT_EQ(hull.size(), 4u);
  EXPECT_NEAR(polygonArea(hull), 4.0, 1.0e-12);
  EXPECT_NEAR(signedDistanceToConvexPolygon(hull, Eigen::Vector2d(0.5, 0.0)),
              0.5, 1.0e-12);
  EXPECT_NEAR(signedDistanceToConvexPolygon(hull, Eigen::Vector2d(2.0, 2.0)),
              -std::sqrt(2.0), 1.0e-12);
  const std::vector<Eigen::Vector2d> segment =
      convexHull({Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(1.0, 0.0),
                  Eigen::Vector2d(2.0, 0.0)},
                 /*tolerance=*/1.0e-7);
  ASSERT_EQ(segment.size(), 2u);
  EXPECT_NEAR(signedDistanceToConvexPolygon(segment, Eigen::Vector2d(1.0, 0.5)),
              -0.5, 1.0e-12);
}

TEST(StaticEquilibriumTest, TwoFlatFeetGiveTheHullOfTheSoles) {
  const std::vector<Eigen::Vector2d> polygon = polygonOf(twoFeet());
  ASSERT_GE(polygon.size(), 4u);
  // The convex hull of the two soles: a rectangle, exactly (on flat ground the
  // cones' linearization does not matter).
  const double xMin = kFootX - kSoleHalfLength;
  const double xMax = kFootX + kSoleHalfLength;
  const double yMax = kFootY + kSoleHalfWidth;
  const std::vector<Eigen::Vector2d> hull = {
      Eigen::Vector2d(xMin, -yMax), Eigen::Vector2d(xMax, -yMax),
      Eigen::Vector2d(xMax, yMax), Eigen::Vector2d(xMin, yMax)};
  for (const Eigen::Vector2d& vertex : polygon) {
    EXPECT_NEAR(signedDistanceToConvexPolygon(hull, vertex), 0.0, 1.0e-6)
        << vertex.transpose();
  }
  EXPECT_NEAR(polygonArea(polygon), polygonArea(hull), 1.0e-6);
  const std::optional<double> margin =
      staticMargin(polygon, Eigen::Vector2d(kFootX, 0.0));
  ASSERT_TRUE(margin.has_value());
  EXPECT_NEAR(*margin, kSoleHalfLength, 1.0e-6);
  // The feasibility LP agrees.
  const StaticEquilibrium equilibrium = equilibriumOf(twoFeet());
  EXPECT_LT(*equilibrium.equilibriumResidual(Eigen::Vector2d(kFootX, 0.1)),
            1.0e-6);
  EXPECT_GT(*equilibrium.equilibriumResidual(Eigen::Vector2d(xMax + 0.02, 0.0)),
            1.0e-3);
}

TEST(StaticEquilibriumTest, MarginSignFlipsAtTheBoundary) {
  const std::vector<Eigen::Vector2d> polygon = polygonOf(twoFeet());
  const double toe = kFootX + kSoleHalfLength;
  EXPECT_NEAR(*staticMargin(polygon, Eigen::Vector2d(toe - 0.005, 0.0)), 0.005,
              1.0e-6);
  EXPECT_NEAR(*staticMargin(polygon, Eigen::Vector2d(toe, 0.0)), 0.0, 1.0e-6);
  EXPECT_NEAR(*staticMargin(polygon, Eigen::Vector2d(toe + 0.005, 0.0)), -0.005,
              1.0e-6);
  const StaticEquilibrium equilibrium = equilibriumOf(twoFeet());
  EXPECT_LT(*equilibrium.equilibriumResidual(Eigen::Vector2d(toe - 0.005, 0.0)),
            1.0e-6);
  EXPECT_GT(*equilibrium.equilibriumResidual(Eigen::Vector2d(toe + 0.005, 0.0)),
            1.0e-4);
}

TEST(StaticEquilibriumTest, AHandOnAWallWidensTheRegionTowardTheWall) {
  const std::vector<Eigen::Vector2d> feetOnly = polygonOf(twoFeet());
  Stance withHand = twoFeet();
  // A wall in front at x = 0.5, its normal toward the robot, the hand at 1.2 m.
  withHand.push_back(
      point("hand", Eigen::Vector3d(0.5, 0.0, 1.2), -Eigen::Vector3d::UnitX()));
  const std::vector<Eigen::Vector2d> widened = polygonOf(withHand);
  ASSERT_GE(widened.size(), 3u);
  double feetMaxX = -1.0;
  double feetMinX = 1.0;
  for (const Eigen::Vector2d& vertex : feetOnly) {
    feetMaxX = std::max(feetMaxX, vertex.x());
    feetMinX = std::min(feetMinX, vertex.x());
  }
  double widenedMaxX = -1.0;
  double widenedMinX = 1.0;
  for (const Eigen::Vector2d& vertex : widened) {
    widenedMaxX = std::max(widenedMaxX, vertex.x());
    widenedMinX = std::min(widenedMinX, vertex.x());
  }
  // Forward, toward the wall, and not backward: pushing on the wall can only
  // move the weight forward.
  EXPECT_GT(widenedMaxX, feetMaxX + 0.05);
  EXPECT_NEAR(widenedMinX, feetMinX, 1.0e-6);
  // A CoM past the toes is in equilibrium with the hand, not without it.
  const Eigen::Vector2d leaning(feetMaxX + 0.04, 0.0);
  EXPECT_LT(*staticMargin(feetOnly, leaning), 0.0);
  EXPECT_GT(*staticMargin(widened, leaning), 0.0);
  EXPECT_LT(*equilibriumOf(withHand).equilibriumResidual(leaning), 1.0e-6);
}

TEST(StaticEquilibriumTest, OnePointHoldsOnlyTheCenterOfMassAboveIt) {
  for (const Eigen::Vector3d& normal :
       {Eigen::Vector3d(Eigen::Vector3d::UnitZ()),
        Eigen::Vector3d(std::sin(0.35), 0.0, std::cos(0.35))}) {
    const Eigen::Vector3d position(0.3, -0.2, 0.5);
    const Stance stance = {point("knee", position, normal)};
    const StaticEquilibrium equilibrium = equilibriumOf(stance);
    const Eigen::Vector2d above = position.head<2>();
    EXPECT_LT(*equilibrium.equilibriumResidual(above), 1.0e-6);
    EXPECT_GT(*equilibrium.equilibriumResidual(
                  Eigen::Vector2d(above + Eigen::Vector2d(0.01, 0.0))),
              1.0e-4);
    EXPECT_GT(*equilibrium.equilibriumResidual(
                  Eigen::Vector2d(above + Eigen::Vector2d(0.0, -0.01))),
              1.0e-4);
    const std::vector<Eigen::Vector2d> polygon = polygonOf(stance);
    ASSERT_FALSE(polygon.empty());
    for (const Eigen::Vector2d& vertex : polygon) {
      EXPECT_LT((vertex - above).norm(), 1.0e-5);
    }
    EXPECT_LE(*staticMargin(polygon, above), 1.0e-5);
    EXPECT_NEAR(
        *staticMargin(polygon, Eigen::Vector2d(above.x() + 0.01, above.y())),
        -0.01, 1.0e-5);
  }
}

TEST(StaticEquilibriumTest, AWallAloneHoldsNothing) {
  const Stance stance = {
      point("hand", Eigen::Vector3d(0.5, 0.0, 1.2), -Eigen::Vector3d::UnitX())};
  EXPECT_TRUE(polygonOf(stance).empty());
  EXPECT_FALSE(
      staticMargin(/*polygon=*/{}, Eigen::Vector2d::Zero()).has_value());
}

TEST(StaticEquilibriumTest, RefusesInvalidStances) {
  EXPECT_FALSE(
      StaticEquilibrium::Create(Stance(), StaticEquilibriumOptions()).ok());
  StanceContact frictionless = foot("left", /*y=*/0.0);
  frictionless.friction = 0.0;
  EXPECT_FALSE(
      StaticEquilibrium::Create({frictionless}, StaticEquilibriumOptions())
          .ok());
  StanceContact noRectangle = foot("left", /*y=*/0.0);
  noRectangle.rectangle.reset();
  EXPECT_FALSE(
      StaticEquilibrium::Create({noRectangle}, StaticEquilibriumOptions())
          .ok());
  StaticEquilibriumOptions twoEdges;
  twoEdges.frictionConeEdges = 2;
  EXPECT_FALSE(StaticEquilibrium::Create(twoFeet(), twoEdges).ok());
}

/** Whether the stance can accelerate a CoM of `mass` at `com` by
 * `acceleration` without turning it. */
bool coneHolds(const StaticEquilibrium& equilibrium,
               double mass,
               const Eigen::Vector3d& com,
               const Eigen::Vector3d& acceleration) {
  const absl::StatusOr<ContactWrenchConeResult> result = checkContactWrenchCone(
      equilibrium, mass, com,
      CentroidalMomentumRate{.linear = mass * acceleration,
                             .angular = Eigen::Vector3d::Zero()});
  EXPECT_TRUE(result.ok()) << result.status();
  return result.ok() && result->feasible;
}

/** A contact of a stance HPIPM stalled on, its numbers as printed with 17
 * digits: `rotation` column-major. */
StanceContact contactOf(const std::string& name,
                        ContactKind kind,
                        const Eigen::Vector3d& position,
                        const std::vector<double>& rotation) {
  StanceContact contact{
      .name = name,
      .kind = kind,
      .position = position,
      .rotation = Eigen::Map<const Eigen::Matrix3d>(rotation.data()),
      .friction = 0.80000000000000004};
  if (kind == ContactKind::kSurface) {
    contact.rectangle = ContactRectangle{.xMin = -kSoleHalfLength,
                                         .xMax = kSoleHalfLength,
                                         .yMin = -kSoleHalfWidth,
                                         .yMax = kSoleHalfWidth};
  }
  return contact;
}

TEST(StaticEquilibriumTest, AnLpHpipmStallsOnIsSolvedWithBoundedScalings) {
  // Atlas on the box with both feet and the left hand, between two knots of a
  // box_climb plan: HPIPM runs out of iterations on the L1 residual of the
  // static wrench, whose optimal set the internal forces between the contacts
  // leave unbounded.
  const Stance stance = {
      contactOf("foot_l_contact", ContactKind::kSurface,
                Eigen::Vector3d(0.59996856838535328, 0.15001729080766199,
                                0.59992317586488852),
                {0.89298812528693605, -0.45008022406735848, 0,
                 0.45008022406735848, 0.89298812528693605, -0, 0, 0, 1}),
      contactOf("foot_r_contact", ContactKind::kSurface,
                Eigen::Vector3d(0.65002010944985522, -0.14994606512730235,
                                0.59987936446924206),
                {0.98018106911343661, 0.19810368939431802, 0,
                 -0.19810368939431802, 0.98018106911343661, 0, 0, 0, 1}),
      contactOf("hand_l_contact", ContactKind::kPoint,
                Eigen::Vector3d(0.84984881475237917, 0.2499602823575684,
                                0.59980264626403712),
                {-0.99849818425927506, -0.054784815696785602, 0,
                 0.054784815696785602, -0.99849818425927506, 0, 0, 0, 1})};
  Wrench wrench;
  wrench << 0, 0, 1, 0.056193997562106593, -0.081064886834193994, 0;
  const StaticEquilibrium equilibrium = equilibriumOf(stance);
  // The residual's LP as it is: HPIPM stalls.
  const int generators = equilibrium.generators().size();
  LinearProgram program =
      LinearProgram::Unconstrained(generators + kWrenchDim + kWrenchDim);
  program.cost.tail(kWrenchDim + kWrenchDim).setOnes();
  program.lowerBound.setZero();
  program.equalityMatrix.resize(kWrenchDim, program.cost.size());
  program.equalityMatrix << equilibrium.generators().wrenches,
      Eigen::MatrixXd::Identity(kWrenchDim, kWrenchDim),
      -Eigen::MatrixXd::Identity(kWrenchDim, kWrenchDim);
  program.equalityVector = wrench;
  EXPECT_FALSE(solveLinearProgram(program).ok());
  // The residual retries with the scalings bounded, and finds the CoM, 5 cm
  // inside the support polygon, in equilibrium.
  const absl::StatusOr<double> residual = equilibrium.wrenchResidual(wrench);
  ASSERT_TRUE(residual.ok()) << residual.status();
  EXPECT_LE(*residual, StaticEquilibriumOptions().feasibilityTolerance);
}

TEST(ContactWrenchConeTest, AFastAccelerationFailsAndASlowOnePasses) {
  constexpr double kMass = 100.0;
  const Eigen::Vector3d com(kFootX, 0.0, 0.9);
  const StaticEquilibrium equilibrium = equilibriumOf(twoFeet());
  EXPECT_TRUE(coneHolds(equilibrium, kMass, com, Eigen::Vector3d::Zero()));
  // 8 m/s^2 forward needs the center of pressure 0.9 * 8 / 9.81 = 0.73 m ahead
  // of the CoM; 0.5 m/s^2 only 0.046 m, within the 0.12 m of the soles.
  EXPECT_FALSE(
      coneHolds(equilibrium, kMass, com, Eigen::Vector3d(8.0, 0.0, 0.0)));
  EXPECT_TRUE(
      coneHolds(equilibrium, kMass, com, Eigen::Vector3d(0.5, 0.0, 0.0)));
  // Falling faster than gravity needs the ground to pull.
  EXPECT_FALSE(
      coneHolds(equilibrium, kMass, com, Eigen::Vector3d(0.0, 0.0, -12.0)));
  // A spin about z within the torsional friction of the soles passes, a
  // violent one fails.
  const absl::StatusOr<ContactWrenchConeResult> spin = checkContactWrenchCone(
      equilibrium, kMass, com,
      CentroidalMomentumRate{.linear = Eigen::Vector3d::Zero(),
                             .angular = Eigen::Vector3d(0.0, 0.0, 2000.0)});
  ASSERT_TRUE(spin.ok());
  EXPECT_FALSE(spin->feasible);
  // With no rate, the required wrench is the static one.
  EXPECT_LT((requiredContactWrench(kMass, com, CentroidalMomentumRate(),
                                   equilibrium.generators().reference) -
             equilibrium.staticWrench(com.head<2>()))
                .norm(),
            1.0e-12);
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
