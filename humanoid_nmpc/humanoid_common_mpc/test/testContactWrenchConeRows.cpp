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

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/LU"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"

/**
 * Tests of the rows of the contact wrench cone (buildLocalWrenchConeRows)
 * against oracles that do not share its code:
 *   - the friction pyramid is INSCRIBED in the Coulomb cone: no direction
 * admits a tangential ratio above mu;
 *   - SOUNDNESS: every wrench that non-negative point forces inside the corner
 * friction pyramids produce satisfies the rows (the forces and their moments
 * are built here from cross products and the polygon's own vertices);
 *   - TIGHTNESS: every wrench that satisfies the rows, interior and boundary
 * alike, is produced by such corner forces (a non-negative least-squares fit
 * with a zero residual);
 *   - for N = 4 the rows are Caron, Pham and Nakamura's closed-form CWC (ICRA
 * 2015), written out here.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kTol = 1.0e-9;

ContactRectangle makeRectangle(scalar_t xMin,
                               scalar_t xMax,
                               scalar_t yMin,
                               scalar_t yMax) {
  return ContactRectangle(
      PolygonBounds(xMin, xMax, yMin, yMax),
      ContactCenterPoint("test_contact", "test_joint", vector3_t::Zero()));
}

ContactWrenchConeConstraint::Config makeConfig(size_t numFacets,
                                               scalar_t friction) {
  ContactWrenchConeConstraint::Config config;
  config.numBasisVectors = numFacets;
  config.frictionCoefficient = friction;
  config.torsionalFrictionCoefficient = 0.05;
  config.minNormalForce = 0.0;
  config.gripperForce = 0.0;
  return config;
}

struct Geometry {
  size_t numFacets;
  scalar_t friction;
  scalar_t xMin, xMax, yMin, yMax;
};

std::vector<Geometry> geometries() {
  return {
      // DRC Atlas.
      {4, 0.5, -0.12, 0.12, -0.055, 0.055},
      // EngineAI SA01: the contact frame is not the footprint center.
      {4, 0.5, -0.125, 0.09, -0.035, 0.035},
      // The contact frame on a footprint edge.
      {4, 0.9, 0.0, 0.2, -0.02, 0.09},
      {3, 0.9, -0.10, 0.10, -0.05, 0.05},
      {5, 0.3, -0.08, 0.08, -0.04, 0.04},
      {6, 0.6, -0.05, 0.20, -0.03, 0.09},
      {8, 0.7, -0.10, 0.10, -0.05, 0.05},
  };
}

std::string describe(const Geometry& g) {
  return absl::StrCat("N=", g.numFacets, " mu=", g.friction, " footprint [",
                      g.xMin, ", ", g.xMax, "] x [", g.yMin, ", ", g.yMax, "]");
}

/** The wrench of the point force `force` at the planar point `point`, moments
 * about the contact frame origin. */
vector6_t wrenchOfPointForce(const vector2_t& point, const vector3_t& force) {
  vector6_t wrench;
  wrench.head<3>() = force;
  wrench.tail<3>() = vector3_t(point.x(), point.y(), 0.0).cross(force);
  return wrench;
}

std::vector<vector2_t> cornersOf(const Geometry& g) {
  return {vector2_t(g.xMax, g.yMax), vector2_t(g.xMax, g.yMin),
          vector2_t(g.xMin, g.yMax), vector2_t(g.xMin, g.yMin)};
}

/** The vertices of the Fz = 1 friction polygon inscribed in the circle of
 * radius mu, counterclockwise. */
std::vector<vector2_t> inscribedPolygon(size_t numFacets, scalar_t friction) {
  std::vector<vector2_t> vertices;
  for (size_t k = 0; k < numFacets; ++k) {
    const scalar_t angle = (2.0 * static_cast<scalar_t>(k) + 1.0) * M_PI /
                           static_cast<scalar_t>(numFacets);
    vertices.emplace_back(friction * std::cos(angle),
                          friction * std::sin(angle));
  }
  return vertices;
}

bool insidePolygon(const std::vector<vector2_t>& vertices,
                   const vector2_t& point) {
  for (size_t k = 0; k < vertices.size(); ++k) {
    const vector2_t edge = vertices[(k + 1) % vertices.size()] - vertices[k];
    const vector2_t toPoint = point - vertices[k];
    if (edge.x() * toPoint.y() - edge.y() * toPoint.x() < 0.0) return false;
  }
  return true;
}

/** The corner generators, built here from the polygon vertices and cross
 * products. */
matrix_t independentCornerGenerators(const Geometry& g) {
  const std::vector<vector2_t> polygon =
      inscribedPolygon(g.numFacets, g.friction);
  const std::vector<vector2_t> corners = cornersOf(g);
  matrix_t generators(
      6, static_cast<Eigen::Index>(polygon.size() * corners.size()));
  Eigen::Index column = 0;
  for (const vector2_t& vertex : polygon) {
    for (const vector2_t& corner : corners) {
      generators.col(column++) =
          wrenchOfPointForce(corner, vector3_t(vertex.x(), vertex.y(), 1.0));
    }
  }
  return generators;
}

/** The largest alpha >= 0 with rows(base + alpha direction) >= 0, for a base
 * inside the homogeneous cone. */
scalar_t distanceToBoundary(const ContactWrenchConeRows& rows,
                            const vector6_t& base,
                            const vector6_t& direction) {
  const vector_t values = rows.evaluateCone(base);
  const vector_t slopes = rows.evaluateCone(direction);
  scalar_t alpha = std::numeric_limits<scalar_t>::infinity();
  for (Eigen::Index row = 0; row < values.size(); ++row) {
    if (slopes(row) < 0.0) alpha = std::min(alpha, values(row) / -slopes(row));
  }
  return alpha;
}

// ==================== The friction pyramid is inscribed in the Coulomb cone
// ====================

TEST(ContactWrenchConeRowsTest, NoDirectionAdmitsATangentialRatioAboveMu) {
  constexpr size_t kNumDirections = 7200;
  for (size_t numFacets =
           ContactWrenchConeConstraint::Config::kMinNumBasisVectors;
       numFacets <= ContactWrenchConeConstraint::Config::kMaxNumBasisVectors;
       ++numFacets) {
    for (const scalar_t friction : {0.3, 0.5, 0.9}) {
      const ContactWrenchConeConstraint::Config config =
          makeConfig(numFacets, friction);
      const ContactRectangle rectangle = makeRectangle(-0.1, 0.15, -0.05, 0.06);
      const ContactWrenchConeRows rows =
          buildLocalWrenchConeRows(config, rectangle);
      const vector2_t center(0.025, 0.005);
      scalar_t largest = 0.0;
      scalar_t smallest = std::numeric_limits<scalar_t>::infinity();
      for (size_t i = 0; i < kNumDirections; ++i) {
        const scalar_t angle = 2.0 * M_PI * static_cast<scalar_t>(i) /
                               static_cast<scalar_t>(kNumDirections);
        // A force through the footprint center, with every row (friction, CoP,
        // torsion) checked.
        const vector6_t normal = wrenchOfPointForce(center, vector3_t::UnitZ());
        const vector6_t tangential = wrenchOfPointForce(
            center, vector3_t(std::cos(angle), std::sin(angle), 0.0));
        const scalar_t ratio = distanceToBoundary(rows, normal, tangential);
        EXPECT_LE(ratio, friction * (1.0 + 1.0e-12))
            << "N=" << numFacets << " mu=" << friction << " direction "
            << angle;
        largest = std::max(largest, ratio);
        smallest = std::min(smallest, ratio);
      }
      // Tight at the pyramid's edges, and never below the inscribed circle.
      EXPECT_NEAR(largest, friction, 1.0e-5 * friction) << "N=" << numFacets;
      EXPECT_NEAR(smallest, inscribedFrictionCoefficient(config),
                  1.0e-5 * friction)
          << "N=" << numFacets;
      EXPECT_NEAR(inscribedFrictionCoefficient(config),
                  friction * std::cos(M_PI / static_cast<scalar_t>(numFacets)),
                  1.0e-15);
    }
  }
}

TEST(ContactWrenchConeRowsTest,
     EveryPyramidEdgeAndGeneratorHasATangentialRatioOfAtMostMu) {
  for (const Geometry& g : geometries()) {
    const ContactWrenchConeConstraint::Config config =
        makeConfig(g.numFacets, g.friction);
    for (const vector3_t& edge : frictionPyramidEdges(config)) {
      EXPECT_LE(edge.head<2>().norm(), g.friction * (1.0 + 1.0e-15))
          << describe(g);
      EXPECT_EQ(edge.z(), 1.0);
    }
    const matrix_t generators = cornerContactWrenchGenerators(
        config, makeRectangle(g.xMin, g.xMax, g.yMin, g.yMax));
    for (Eigen::Index j = 0; j < generators.cols(); ++j) {
      EXPECT_LE(generators.col(j).head<2>().norm(),
                g.friction * (1.0 + 1.0e-15) * generators(/*row=*/2, j))
          << describe(g);
    }
  }
}

// ==================== H-representation and V-representation describe the same
// cone ====================

TEST(ContactWrenchConeRowsTest,
     GeneratorsAreTheIndependentlyBuiltCornerForcesAndExtremeRaysOfTheRows) {
  for (const Geometry& g : geometries()) {
    const ContactWrenchConeConstraint::Config config =
        makeConfig(g.numFacets, g.friction);
    const ContactRectangle rectangle =
        makeRectangle(g.xMin, g.xMax, g.yMin, g.yMax);
    const ContactWrenchConeRows rows =
        buildLocalWrenchConeRows(config, rectangle);
    const matrix_t generators =
        cornerContactWrenchGenerators(config, rectangle);
    EXPECT_TRUE(generators.isApprox(independentCornerGenerators(g), 1.0e-14))
        << describe(g);
    for (Eigen::Index j = 0; j < generators.cols(); ++j) {
      const vector_t values = rows.evaluateCone(vector6_t(generators.col(j)));
      EXPECT_GE(values.minCoeff(), -kTol)
          << describe(g) << " generator " << j << " leaves the cone";
      // An extreme ray of a six-dimensional cone lies on facets whose normals
      // span five dimensions.
      matrix_t activeRows(0, 6);
      for (Eigen::Index row = 0; row < values.size(); ++row) {
        if (std::abs(values(row)) <= kTol) {
          activeRows.conservativeResize(activeRows.rows() + 1, Eigen::NoChange);
          activeRows.bottomRows(1) << rows.A_f.row(row), rows.A_tau.row(row);
        }
      }
      EXPECT_EQ(activeRows.fullPivLu().rank(), 5)
          << describe(g) << " generator " << j
          << " is not an extreme ray of the rows";
    }
  }
}

TEST(ContactWrenchConeRowsTest,
     SoundEveryWrenchOfCornerForcesInsideTheirPyramidsSatisfiesTheRows) {
  std::mt19937 gen(7);
  std::uniform_real_distribution<scalar_t> unit(0.0, 1.0);
  for (const Geometry& g : geometries()) {
    const ContactWrenchConeRows rows =
        buildLocalWrenchConeRows(makeConfig(g.numFacets, g.friction),
                                 makeRectangle(g.xMin, g.xMax, g.yMin, g.yMax));
    const std::vector<vector2_t> polygon =
        inscribedPolygon(g.numFacets, g.friction);
    const std::vector<vector2_t> corners = cornersOf(g);
    for (int trial = 0; trial < 2000; ++trial) {
      vector6_t wrench = vector6_t::Zero();
      for (const vector2_t& corner : corners) {
        if (unit(gen) < 0.25) continue;  // some corners unloaded
        const scalar_t normalForce = 500.0 * unit(gen);
        vector2_t ratio;
        if (unit(gen) < 0.3) {
          ratio = polygon[static_cast<size_t>(unit(gen) * static_cast<scalar_t>(
                                                              polygon.size())) %
                          polygon.size()];  // on an edge
        } else {
          do {
            ratio = vector2_t(g.friction * (2.0 * unit(gen) - 1.0),
                              g.friction * (2.0 * unit(gen) - 1.0));
          } while (!insidePolygon(polygon, ratio));
        }
        wrench += wrenchOfPointForce(
            corner, normalForce * vector3_t(ratio.x(), ratio.y(), 1.0));
      }
      EXPECT_GE(rows.evaluateCone(wrench).minCoeff(),
                -kTol * std::max(1.0, wrench.norm()))
          << describe(g) << " " << wrench.transpose();
    }
  }
}

TEST(ContactWrenchConeRowsTest,
     TightEveryWrenchThatSatisfiesTheRowsIsProducedByCornerForces) {
  std::mt19937 gen(11);
  std::uniform_real_distribution<scalar_t> unit(0.0, 1.0);
  std::normal_distribution<scalar_t> gaussian(0.0, 1.0);
  for (const Geometry& g : geometries()) {
    const ContactWrenchConeRows rows =
        buildLocalWrenchConeRows(makeConfig(g.numFacets, g.friction),
                                 makeRectangle(g.xMin, g.xMax, g.yMin, g.yMax));
    const matrix_t generators = independentCornerGenerators(g);
    const scalar_t halfLength = 0.5 * (g.xMax - g.xMin);
    const scalar_t halfWidth = 0.5 * (g.yMax - g.yMin);
    size_t numInterior = 0;
    size_t numBoundary = 0;
    for (int trial = 0; trial < 4000 && numBoundary < 300; ++trial) {
      // A candidate at Fz = 1 from a box that contains the cone's slice, scaled
      // by a random normal force.
      const scalar_t normalForce = 1.0 + 999.0 * unit(gen);
      const vector2_t cop(g.xMin + (g.xMax - g.xMin) * unit(gen),
                          g.yMin + (g.yMax - g.yMin) * unit(gen));
      vector6_t wrench;
      wrench << g.friction * (2.0 * unit(gen) - 1.0),
          g.friction * (2.0 * unit(gen) - 1.0), 1.0, cop.y(), -cop.x(),
          g.friction * (halfLength + halfWidth + 0.2) * (2.0 * unit(gen) - 1.0);
      wrench *= normalForce;
      if (rows.evaluateCone(wrench).minCoeff() < 0.0) continue;
      ++numInterior;
      std::vector<vector6_t> targets = {wrench};
      // The boundary point along a random direction: where an inner
      // approximation would be caught.
      vector6_t direction;
      for (Eigen::Index i = 0; i < 6; ++i) direction(i) = gaussian(gen);
      const scalar_t alpha = distanceToBoundary(rows, wrench, direction);
      if (std::isfinite(alpha)) {
        targets.push_back(wrench + alpha * direction);
        ++numBoundary;
      }
      for (const vector6_t& target : targets) {
        const vector_t lambda =
            solveNonNegativeLeastSquares(generators, target);
        EXPECT_GE(lambda.minCoeff(), 0.0);
        EXPECT_LE((generators * lambda - target).norm(), 1.0e-9 * target.norm())
            << describe(g) << ": " << target.transpose()
            << " satisfies the rows but no corner forces produce it";
      }
    }
    EXPECT_GT(numInterior, 100u)
        << describe(g) << ": too few samples inside the cone";
    EXPECT_GT(numBoundary, 100u) << describe(g);
  }
}

// ==================== N = 4: Caron, Pham and Nakamura's closed form
// ====================

/**
 * The 16 rows of the CWC of Caron et al. (ICRA 2015) for the axis-aligned
 * pyramid |f_x|, |f_y| <= mu' f_z at each corner of the footprint with half
 * extents X, Y, about its center (F, tau_c), each as a coefficient vector u
 * with u . (F, tau_c) >= 0, then taken to moments about the contact frame
 * origin with tau_c = tau - c x F.
 */
std::vector<vector6_t> caronRows(scalar_t facetFriction, const Geometry& g) {
  const scalar_t X = 0.5 * (g.xMax - g.xMin);
  const scalar_t Y = 0.5 * (g.yMax - g.yMin);
  const vector3_t center(0.5 * (g.xMin + g.xMax), 0.5 * (g.yMin + g.yMax), 0.0);
  const scalar_t mu = facetFriction;
  std::vector<vector6_t> aboutCenter;
  for (const scalar_t s : {1.0, -1.0}) {
    aboutCenter.push_back((vector6_t() << -s, 0.0, mu, 0.0, 0.0, 0.0)
                              .finished());  // |Fx| <= mu Fz
    aboutCenter.push_back((vector6_t() << 0.0, -s, mu, 0.0, 0.0, 0.0)
                              .finished());  // |Fy| <= mu Fz
    aboutCenter.push_back((vector6_t() << 0.0, 0.0, Y, -s, 0.0, 0.0)
                              .finished());  // |tau_x| <= Y Fz
    aboutCenter.push_back((vector6_t() << 0.0, 0.0, X, 0.0, -s, 0.0)
                              .finished());  // |tau_y| <= X Fz
  }
  for (const scalar_t s1 : {1.0, -1.0}) {
    for (const scalar_t s2 : {1.0, -1.0}) {
      // tau_z >= -mu (X + Y) Fz + |Y Fx - mu tau_x| + |X Fy - mu tau_y|
      aboutCenter.push_back(
          (vector6_t() << -s1 * Y, -s2 * X, mu * (X + Y), s1 * mu, s2 * mu, 1.0)
              .finished());
      // tau_z <= mu (X + Y) Fz - |Y Fx + mu tau_x| - |X Fy + mu tau_y|
      aboutCenter.push_back((vector6_t() << -s1 * Y, -s2 * X, mu * (X + Y),
                             -s1 * mu, -s2 * mu, -1.0)
                                .finished());
    }
  }
  std::vector<vector6_t> rows;
  for (const vector6_t& u : aboutCenter) {
    vector6_t row;
    row.tail<3>() = u.tail<3>();
    row.head<3>() = u.head<3>() - u.tail<3>().cross(center);
    rows.push_back(row);
  }
  return rows;
}

/** Whether `row` is a positive multiple of one of `rows`. */
bool containsDirection(const std::vector<vector6_t>& rows,
                       const vector6_t& row) {
  return std::any_of(rows.begin(), rows.end(), [&row](const vector6_t& other) {
    return (other.normalized() - row.normalized()).norm() < 1.0e-9;
  });
}

TEST(ContactWrenchConeRowsTest, FourFacetsGiveCaronsClosedFormRows) {
  for (const Geometry& g : geometries()) {
    if (g.numFacets != 4) continue;
    const ContactWrenchConeConstraint::Config config =
        makeConfig(g.numFacets, g.friction);
    const ContactWrenchConeRows rows = buildLocalWrenchConeRows(
        config, makeRectangle(g.xMin, g.xMax, g.yMin, g.yMax));
    const std::vector<vector6_t> caron =
        caronRows(g.friction * std::cos(M_PI / 4.0), g);
    ASSERT_EQ(rows.numTorsionRows(), 8u) << describe(g);
    ASSERT_EQ(rows.numRows(), caron.size() + 1)
        << describe(g) << ": Caron's 16 rows and the normal-force row";
    std::vector<vector6_t> ours;
    for (size_t row = 0; row < rows.numRows(); ++row) {
      if (row == rows.numFrictionRows)
        continue;  // Fz >= minNormalForce, implied by the friction rows when
                   // homogeneous
      vector6_t u;
      u << rows.A_f.row(static_cast<Eigen::Index>(row)).transpose(),
          rows.A_tau.row(static_cast<Eigen::Index>(row)).transpose();
      ours.push_back(u);
      EXPECT_TRUE(containsDirection(caron, u))
          << describe(g) << " row " << row << " = " << u.transpose()
          << " is not one of Caron's";
    }
    for (const vector6_t& u : caron) {
      EXPECT_TRUE(containsDirection(ours, u))
          << describe(g) << ": Caron's row " << u.transpose() << " is missing";
    }
    // The torsion rows carry a unit yaw coefficient, as Caron's do.
    for (size_t row = rows.torsionRowStart(); row < rows.numRows(); ++row) {
      EXPECT_NEAR(
          std::abs(rows.A_tau(static_cast<Eigen::Index>(row), /*col=*/2)), 1.0,
          1.0e-12);
    }
  }
}

TEST(ContactWrenchConeRowsTest, TorsionLimitsOfANormalForce) {
  const Geometry atlas = geometries()[0];
  const ContactWrenchConeConstraint::Config config =
      makeConfig(atlas.numFacets, atlas.friction);
  const ContactWrenchConeRows rows = buildLocalWrenchConeRows(
      config, makeRectangle(atlas.xMin, atlas.xMax, atlas.yMin, atlas.yMax));
  // At the center: Caron's +-mu' (X + Y), with no tangential force and no CoP
  // excursion.
  const scalar_t expected =
      inscribedFrictionCoefficient(config) *
      (0.5 * (atlas.xMax - atlas.xMin) + 0.5 * (atlas.yMax - atlas.yMin));
  const std::optional<TorsionInterval> atCenter =
      torsionLimitsOfNormalForceAt(rows, vector2_t::Zero());
  if (!atCenter.has_value()) {
    FAIL() << "the center of the footprint admits a normal force";
  }
  EXPECT_NEAR(atCenter->upper, expected, 1.0e-10);
  EXPECT_NEAR(atCenter->lower, -expected, 1.0e-10);
  // At a corner all the load is on one corner, whose tangential force has no
  // lever arm: no torsion.
  const std::optional<TorsionInterval> atCorner =
      torsionLimitsOfNormalForceAt(rows, vector2_t(atlas.xMax, atlas.yMin));
  if (!atCorner.has_value()) {
    FAIL() << "a corner of the footprint admits a normal force";
  }
  EXPECT_NEAR(atCorner->upper, 0.0, 1.0e-10);
  EXPECT_NEAR(atCorner->lower, 0.0, 1.0e-10);
  // Off the footprint no yaw moment makes the wrench admissible.
  EXPECT_FALSE(
      torsionLimitsOfNormalForceAt(rows, vector2_t(atlas.xMax + 0.01, 0.0))
          .has_value());
}

// ==================== The audit's counterexample (common C8 / centroidal C10)
// ====================

TEST(ContactWrenchConeRowsTest,
     ALateralForceAtAnOffCenterCopIsAdmittedWithoutAnyTorsion) {
  // A lateral force applied near the toe produces a yaw moment about the
  // footprint center, but no free torsion: four corner forces produce it
  // without any twisting. The old rows bounded the yaw moment about the center,
  // |tau_z - (p_x F_y - p_y F_x)| <= mu_t Fz, and rejected it.
  const Geometry g{.numFacets = 4,
                   .friction = 0.5,
                   .xMin = -0.2,
                   .xMax = 0.2,
                   .yMin = -0.05,
                   .yMax = 0.05};
  const ContactWrenchConeConstraint::Config config =
      makeConfig(g.numFacets, g.friction);
  const ContactWrenchConeRows rows = buildLocalWrenchConeRows(
      config, makeRectangle(g.xMin, g.xMax, g.yMin, g.yMax));
  const scalar_t normalForce = 400.0;
  const vector6_t wrench = wrenchOfPointForce(
      vector2_t(0.9 * g.xMax, 0.0), normalForce * vector3_t(0.0, 0.3, 1.0));
  ASSERT_GT(std::abs(wrench(5)),
            config.torsionalFrictionCoefficient * normalForce)
      << "the old rows rejected this wrench";
  EXPECT_GT(rows.evaluateCone(wrench).minCoeff(), 0.0);
  const matrix_t generators = independentCornerGenerators(g);
  const vector_t lambda = solveNonNegativeLeastSquares(generators, wrench);
  EXPECT_LE((generators * lambda - wrench).norm(), 1.0e-9 * wrench.norm());
}

// ==================== The affine offsets ====================

TEST(ContactWrenchConeRowsTest, GripperAndMinimumNormalForceOffsets) {
  ContactWrenchConeConstraint::Config config =
      makeConfig(/*numFacets=*/6, /*friction=*/0.6);
  config.minNormalForce = 12.0;
  config.gripperForce = 30.0;
  const Geometry g{.numFacets = 6,
                   .friction = 0.6,
                   .xMin = -0.05,
                   .xMax = 0.20,
                   .yMin = -0.03,
                   .yMax = 0.09};
  const ContactWrenchConeRows rows = buildLocalWrenchConeRows(
      config, makeRectangle(g.xMin, g.xMax, g.yMin, g.yMax));
  const ContactWrenchConeRows homogeneous =
      buildLocalWrenchConeRows(makeConfig(g.numFacets, g.friction),
                               makeRectangle(g.xMin, g.xMax, g.yMin, g.yMax));
  EXPECT_TRUE(rows.A_f.isApprox(homogeneous.A_f) &&
              rows.A_tau.isApprox(homogeneous.A_tau))
      << "the offsets only move b";
  // The gripper force acts like an extra normal force at the footprint center
  // on the friction and torsion rows.
  const vector6_t gripper = wrenchOfPointForce(
      vector2_t(0.075, 0.03), vector3_t(0.0, 0.0, config.gripperForce));
  const vector_t gripperOffsets = homogeneous.evaluateCone(gripper);
  for (size_t row = 0; row < rows.numRows(); ++row) {
    const Eigen::Index r = static_cast<Eigen::Index>(row);
    if (row < rows.numFrictionRows) {
      EXPECT_NEAR(rows.b(r),
                  inscribedFrictionCoefficient(config) * config.gripperForce,
                  1.0e-12);
    } else if (row == rows.numFrictionRows) {
      EXPECT_EQ(rows.b(r), -config.minNormalForce);
    } else if (row < rows.torsionRowStart()) {
      EXPECT_EQ(rows.b(r), 0.0) << "CoP row " << row;
    } else {
      EXPECT_NEAR(rows.b(r), gripperOffsets(r), 1.0e-12)
          << "torsion row " << row;
      EXPECT_GT(rows.b(r), 0.0) << "torsion row " << row;
    }
  }
}

// ==================== Validation ====================

TEST(ContactWrenchConeRowsTest,
     DegenerateFootprintsAndTooManyFacetsAreRefusedNamingTheKey) {
  EXPECT_TRUE(ContactWrenchConeConstraint::validateFootprint(
                  makeRectangle(-0.1, 0.1, -0.05, 0.05))
                  .ok());
  for (const ContactRectangle& rectangle :
       {makeRectangle(0.1, 0.1, -0.05, 0.05),
        makeRectangle(-0.1, 0.1, 0.0, 0.0),
        makeRectangle(0.1, -0.1, -0.05, 0.05),
        makeRectangle(std::nan(""), /*xMax=*/0.1, /*yMin=*/-0.05,
                      /*yMax=*/0.05)}) {
    const absl::Status status =
        ContactWrenchConeConstraint::validateFootprint(rectangle);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("contacts.contact_rectangle."),
              absl::string_view::npos)
        << status;
  }
  EXPECT_TRUE(
      ContactWrenchConeConstraint::validateConfig(
          makeConfig(ContactWrenchConeConstraint::Config::kMaxNumBasisVectors,
                     /*friction=*/0.5))
          .ok());
  const absl::Status tooMany = ContactWrenchConeConstraint::validateConfig(
      makeConfig(ContactWrenchConeConstraint::Config::kMaxNumBasisVectors + 1,
                 /*friction=*/0.5));
  EXPECT_EQ(tooMany.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(
      tooMany.message().find(
          "contacts.contact_wrench_cone_soft_constraint.num_basis_vectors"),
      absl::string_view::npos)
      << tooMany;
}

}  // namespace
}  // namespace ocs2::humanoid
