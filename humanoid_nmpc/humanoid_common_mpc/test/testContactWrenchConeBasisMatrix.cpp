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
#include <random>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"

/**
 * Tests of the wrench-cone generator bases used by the basis-vector contact input formulation.
 *
 * Two properties matter, and both are checked against the rows ContactWrenchConeConstraint enforces in wrench mode
 * (buildLocalWrenchConeRows), never against a hand-written expectation of what each column contains:
 *   - SOUNDNESS: every generator lies inside that cone, for every generator set. The explicit cone constraint is
 *     dropped when basis-vector inputs are active, so this is what keeps the MPC inside the friction, center-of-pressure
 *     and torsion limits.
 *   - COMPLETENESS: every wrench of that cone is a non-negative combination of the generators. Only the
 *     exact_wrench_cone set has it; conservative_inner_approximation is shown here to be a strict inner approximation,
 *     and how much of the cone it covers is reported rather than claimed.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kTol = 1.0e-12;
/// Residual below which a wrench counts as reproduced, relative to its norm.
constexpr scalar_t kRepresentableTolerance = 1.0e-9;
/// Residual above which a wrench counts as not reproduced, relative to its norm (with a verified NNLS optimum).
constexpr scalar_t kNotRepresentableTolerance = 1.0e-4;
constexpr size_t kNumSamples = 3000;

const std::vector<std::string>& allGeneratorSets() {
  static const absl::NoDestructor<std::vector<std::string>> kSets(
      std::vector<std::string>{std::string(kConservativeInnerApproximationGeneratorSet), std::string(kExactWrenchConeGeneratorSet)});
  return *kSets;
}

ContactRectangle makeRectangle(scalar_t xMin, scalar_t xMax, scalar_t yMin, scalar_t yMax) {
  return ContactRectangle(PolygonBounds(xMin, xMax, yMin, yMax), ContactCenterPoint("test_contact", "test_joint", vector3_t::Zero()));
}

ContactRectangle makeTestContactRectangle() {
  return makeRectangle(-0.10, 0.10, -0.05, 0.05);
}

ContactWrenchConeConstraint::Config makeTestConeConfig(size_t numBasisVectors = 4) {
  ContactWrenchConeConstraint::Config config;
  config.numBasisVectors = numBasisVectors;
  config.frictionCoefficient = 0.7;
  config.torsionalFrictionCoefficient = 0.05;
  config.minNormalForce = 5.0;
  config.gripperForce = 0.0;
  return config;
}

/** The DRC Atlas parameters of the shipped task file, which the audit's counterexample was worked out for. */
ContactWrenchConeConstraint::Config makeAtlasConeConfig() {
  ContactWrenchConeConstraint::Config config = makeTestConeConfig(4);
  config.frictionCoefficient = 0.5;
  config.torsionalFrictionCoefficient = 0.05;
  return config;
}
ContactRectangle makeAtlasContactRectangle() {
  return makeRectangle(-0.12, 0.12, -0.055, 0.055);
}

ContactWrenchConeBasisMatrix createOrDie(const ContactWrenchConeConstraint::Config& config,
                                         const ContactRectangle& rectangle,
                                         const std::string& generatorSet) {
  absl::StatusOr<ContactWrenchConeBasisMatrix> basis = ContactWrenchConeBasisMatrix::Create(config, rectangle, generatorSet);
  if (!basis.ok()) {
    throw std::runtime_error(std::string(basis.status().message()));
  }
  return *std::move(basis);
}

/** Worst (most negative) row of the homogeneous cone evaluated on a wrench; non-negative means admissible. */
scalar_t worstConeRow(const ContactWrenchConeRows& rows, const vector6_t& wrench) {
  return rows.evaluateCone(wrench).minCoeff();
}

/** Footprints that are off-center, patch offsets, and different friction and generator counts. */
struct GeometryCase {
  size_t numBasisVectors;
  scalar_t friction;
  scalar_t torsionalFriction;
  scalar_t xMin, xMax, yMin, yMax;
  vector3_t patchOffset;
};

const std::vector<GeometryCase>& geometryCases() {
  static const absl::NoDestructor<std::vector<GeometryCase>> kCases(std::vector<GeometryCase>{
      {4, 0.5, 0.05, -0.12, 0.12, -0.055, 0.055, vector3_t::Zero()},   // DRC Atlas
      {4, 0.5, 0.05, -0.125, 0.09, -0.035, 0.035, vector3_t::Zero()},  // EngineAI SA01: asymmetric footprint
      {8, 0.7, 0.05, -0.10, 0.10, -0.05, 0.05, vector3_t::Zero()},
      {3, 0.9, 0.10, -0.10, 0.10, -0.05, 0.05, vector3_t::Zero()},
      {6, 0.6, 0.08, -0.05, 0.20, -0.03, 0.09, vector3_t::Zero()},             // off-center footprint
      {6, 0.6, 0.08, -0.05, 0.20, -0.03, 0.09, vector3_t(0.04, 0.02, 0.0)},    // explicit patch offset
      {5, 0.3, 0.02, -0.08, 0.08, -0.04, 0.04, vector3_t(-0.08, -0.04, 0.0)},  // offset on the footprint edge
  });
  return *kCases;
}

ContactWrenchConeConstraint::Config configFor(const GeometryCase& c) {
  ContactWrenchConeConstraint::Config config = makeTestConeConfig(c.numBasisVectors);
  config.frictionCoefficient = c.friction;
  config.torsionalFrictionCoefficient = c.torsionalFriction;
  config.patchOffset = c.patchOffset;
  return config;
}

std::string describe(const GeometryCase& c) {
  return absl::StrCat("N=", c.numBasisVectors, " mu=", c.friction, " mu_t=", c.torsionalFriction, " footprint [", c.xMin, ", ", c.xMax,
                      "] x [", c.yMin, ", ", c.yMax, "] patch offset (", c.patchOffset.x(), ", ", c.patchOffset.y(), ")");
}

/**
 * The wrench of a normal force `normalForce` with tangential force normalForce * tangentialRatio applied at the planar
 * point `centerOfPressure`, plus whatever free torsion makes the torsion about `patchPoint` equal normalForce *
 * torsionRatio. Built from cross products, independently of the closed form the exact generator set uses.
 */
vector6_t physicalWrench(const vector2_t& tangentialRatio,
                         const vector2_t& centerOfPressure,
                         scalar_t torsionRatio,
                         scalar_t normalForce,
                         const vector3_t& patchPoint) {
  const vector3_t force = normalForce * vector3_t(tangentialRatio.x(), tangentialRatio.y(), 1.0);
  const vector3_t application(centerOfPressure.x(), centerOfPressure.y(), 0.0);
  const vector3_t patch(patchPoint.x(), patchPoint.y(), 0.0);
  const scalar_t torsionOfTheForceAboutThePatch = (application - patch).cross(force).z();
  const scalar_t freeTorsion = normalForce * torsionRatio - torsionOfTheForceAboutThePatch;
  vector6_t wrench;
  wrench.head<3>() = force;
  wrench.tail<3>() = application.cross(force) + vector3_t(0.0, 0.0, freeTorsion);
  return wrench;
}

/**
 * The vertices of the friction polygon at unit normal force, each computed as the intersection of two adjacent facet
 * lines of the rows (mu - cos(theta_k) f_x - sin(theta_k) f_y = 0), so independently of the generator construction.
 */
std::vector<vector2_t> frictionPolygonVertices(const ContactWrenchConeRows& rows, size_t numFacets) {
  std::vector<vector2_t> vertices;
  for (size_t k = 0; k < numFacets; ++k) {
    const size_t next = (k + 1) % numFacets;
    matrix_t lhs(2, 2);
    lhs << -rows.A_f(k, /*col=*/0), -rows.A_f(k, /*col=*/1), -rows.A_f(next, /*col=*/0), -rows.A_f(next, /*col=*/1);
    const vector_t rhs = (vector_t(2) << rows.A_f(k, /*col=*/2), rows.A_f(next, /*col=*/2)).finished();
    vertices.emplace_back(lhs.fullPivLu().solve(rhs));
  }
  return vertices;
}

/** Whether a tangential force (at unit normal force) satisfies the friction rows, the first `numFacets` rows. */
bool satisfiesFrictionRows(const ContactWrenchConeRows& rows, size_t numFacets, const vector2_t& tangentialRatio) {
  const vector3_t force(tangentialRatio.x(), tangentialRatio.y(), 1.0);
  return (rows.A_f.topRows(static_cast<Eigen::Index>(numFacets)) * force).minCoeff() >= 0.0;
}

/**
 * Samples wrenches of the cone the rows define: a tangential force inside the friction polygon (rejection sampling on
 * the friction rows themselves), a center of pressure on the footprint and a torsion inside its interval, at a random
 * normal force. A third of the samples are pushed onto vertices and faces, where an inner approximation is weakest.
 */
std::vector<vector6_t> sampleConeWrenches(const ContactWrenchConeConstraint::Config& config,
                                          const ContactRectangle& rectangle,
                                          size_t numSamples,
                                          uint32_t seed) {
  const ContactWrenchConeRows rows = buildLocalWrenchConeRows(config, rectangle);
  const std::vector<vector2_t> polygon = frictionPolygonVertices(rows, config.numBasisVectors);
  const PolygonBounds& bounds = rectangle.getBounds();
  const vector3_t patchPoint = contactPatchReferencePoint(config, rectangle);
  const scalar_t radius = config.frictionCoefficient / std::cos(M_PI / static_cast<scalar_t>(config.numBasisVectors));

  std::mt19937 gen(seed);
  std::uniform_real_distribution<scalar_t> unit(0.0, 1.0);
  std::uniform_real_distribution<scalar_t> tangential(-radius, radius);
  std::uniform_real_distribution<scalar_t> normalForce(1.0, 1000.0);
  std::uniform_int_distribution<size_t> pick(0, 1000000);

  std::vector<vector6_t> wrenches;
  wrenches.reserve(numSamples);
  while (wrenches.size() < numSamples) {
    const bool onBoundary = pick(gen) % 3 == 0;
    vector2_t ratio;
    if (onBoundary) {
      ratio = polygon[pick(gen) % polygon.size()];
    } else {
      do {
        ratio = vector2_t(tangential(gen), tangential(gen));
      } while (!satisfiesFrictionRows(rows, config.numBasisVectors, ratio));
    }
    vector2_t cop(bounds.x_min + unit(gen) * (bounds.x_max - bounds.x_min), bounds.y_min + unit(gen) * (bounds.y_max - bounds.y_min));
    scalar_t torsion = (2.0 * unit(gen) - 1.0) * config.torsionalFrictionCoefficient;
    if (onBoundary) {
      cop = vector2_t(pick(gen) % 2 == 0 ? bounds.x_min : bounds.x_max, pick(gen) % 2 == 0 ? bounds.y_min : bounds.y_max);
      torsion = (pick(gen) % 2 == 0 ? 1.0 : -1.0) * config.torsionalFrictionCoefficient;
    }
    wrenches.push_back(physicalWrench(ratio, cop, torsion, normalForce(gen), patchPoint));
  }
  return wrenches;
}

/** Whether a non-negative combination of the columns of B reproduces the wrench. */
bool isRepresentable(const matrix_t& B, const vector6_t& wrench) {
  const vector_t lambda = solveNonNegativeLeastSquares(B, wrench);
  return lambda.minCoeff() >= 0.0 && (B * lambda - wrench).norm() <= kRepresentableTolerance * std::max(1.0, wrench.norm());
}

/**
 * Proves that a wrench is NOT a non-negative combination of the columns of B: the NNLS solution satisfies the KKT
 * conditions (so its residual is the distance from the wrench to the cone of B) and that distance is not zero.
 */
::testing::AssertionResult isProvablyNotRepresentable(const matrix_t& B, const vector6_t& wrench) {
  const vector_t lambda = solveNonNegativeLeastSquares(B, wrench);
  const vector_t residual = wrench - B * lambda;
  const vector_t gradient = B.transpose() * residual;
  const scalar_t kktTolerance = 1.0e-9 * std::max(1.0, wrench.norm());
  if (lambda.minCoeff() < 0.0) {
    return ::testing::AssertionFailure() << "NNLS returned a negative scaling";
  }
  for (Eigen::Index j = 0; j < lambda.size(); ++j) {
    if (gradient(j) > kktTolerance || (lambda(j) > 0.0 && std::abs(gradient(j)) > kktTolerance)) {
      return ::testing::AssertionFailure() << "the NNLS solution is not optimal (KKT violated at column " << j << ")";
    }
  }
  if (residual.norm() <= kNotRepresentableTolerance * wrench.norm()) {
    return ::testing::AssertionFailure() << "the wrench IS representable, residual " << residual.norm();
  }
  return ::testing::AssertionSuccess() << "distance to the cone " << residual.norm();
}

// ==================== The generator-set registry ====================

TEST(ContactWrenchConeBasisRegistryTest, ListsBothGeneratorSetsAndResolvesEach) {
  const std::vector<std::string> names = basisGeneratorSetNames();
  for (const std::string& set : allGeneratorSets()) {
    EXPECT_NE(std::find(names.begin(), names.end(), set), names.end()) << set << " is not registered";
    EXPECT_TRUE(getBasisGeneratorSetBuilder(set).ok()) << set;
    const absl::StatusOr<ContactWrenchConeBasisMatrix> basis =
        ContactWrenchConeBasisMatrix::Create(makeTestConeConfig(), makeTestContactRectangle(), set);
    ASSERT_TRUE(basis.ok()) << basis.status();
    EXPECT_EQ(basis->generatorSet(), set);
  }
  EXPECT_EQ(names.size(), allGeneratorSets().size());
}

TEST(ContactWrenchConeBasisRegistryTest, TheShippedDefaultIsStillTheConservativeSet) {
  // The exact set changes the closed loop (a larger admissible wrench set and 8N instead of N + 7 inputs per foot), so
  // it stays opt-in until it has been validated; Create() without a name keeps the shipped set.
  EXPECT_EQ(kDefaultBasisGeneratorSet, kConservativeInnerApproximationGeneratorSet);
  const absl::StatusOr<ContactWrenchConeBasisMatrix> named =
      ContactWrenchConeBasisMatrix::Create(makeTestConeConfig(), makeTestContactRectangle(), kConservativeInnerApproximationGeneratorSet);
  ASSERT_TRUE(named.ok());
  const absl::StatusOr<ContactWrenchConeBasisMatrix> byDefault =
      ContactWrenchConeBasisMatrix::Create(makeTestConeConfig(), makeTestContactRectangle());
  ASSERT_TRUE(byDefault.ok());
  EXPECT_EQ(byDefault->generatorSet(), kConservativeInnerApproximationGeneratorSet);
  EXPECT_TRUE((byDefault->getBasisMatrix() - named->getBasisMatrix()).isZero(0.0));
}

TEST(ContactWrenchConeBasisRegistryTest, UnknownGeneratorSetIsRejectedWithEveryValidName) {
  for (const absl::Status& status :
       {getBasisGeneratorSetBuilder("exact").status(),
        ContactWrenchConeBasisMatrix::Create(makeTestConeConfig(), makeTestContactRectangle(), "exact").status()}) {
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find(kBasisGeneratorSetKey), absl::string_view::npos) << status;
    EXPECT_NE(status.message().find("'exact'"), absl::string_view::npos) << status;
    for (const std::string& set : allGeneratorSets()) {
      EXPECT_NE(status.message().find(set), absl::string_view::npos) << status << " does not list " << set;
    }
  }
}

// ==================== Configuration errors name the key to change ====================

void expectInvalidArgumentNaming(const ContactWrenchConeConstraint::Config& config,
                                 const ContactRectangle& rectangle,
                                 const std::string& generatorSet,
                                 const std::string& key) {
  const absl::StatusOr<ContactWrenchConeBasisMatrix> basis = ContactWrenchConeBasisMatrix::Create(config, rectangle, generatorSet);
  ASSERT_FALSE(basis.ok()) << generatorSet << " accepted a configuration that should name " << key;
  EXPECT_EQ(basis.status().code(), absl::StatusCode::kInvalidArgument) << basis.status();
  EXPECT_NE(basis.status().message().find(key), absl::string_view::npos) << basis.status() << "\n does not name " << key;
}

TEST(ContactWrenchConeBasisValidationTest, ConfigurationErrorsNameTheirKey) {
  for (const std::string& set : allGeneratorSets()) {
    expectInvalidArgumentNaming(makeTestConeConfig(2), makeTestContactRectangle(), set,
                                "contacts.contact_wrench_cone_soft_constraint.num_basis_vectors");

    ContactWrenchConeConstraint::Config noFriction = makeTestConeConfig();
    noFriction.frictionCoefficient = 0.0;
    expectInvalidArgumentNaming(noFriction, makeTestContactRectangle(), set,
                                "contacts.contact_wrench_cone_soft_constraint.friction_coefficient");

    // A negative torsional coefficient used to surface as "generator 0 lies outside the cone", naming neither key.
    ContactWrenchConeConstraint::Config negativeTorsion = makeTestConeConfig();
    negativeTorsion.torsionalFrictionCoefficient = -0.05;
    expectInvalidArgumentNaming(negativeTorsion, makeTestContactRectangle(), set,
                                "contacts.contact_wrench_cone_soft_constraint.torsional_friction_coefficient");

    expectInvalidArgumentNaming(makeTestConeConfig(), makeRectangle(0.10, -0.10, -0.05, 0.05), set, "contacts.contact_rectangle.");
  }
}

/**
 * validateConfig() is the one definition of the cone's ranges, and each refusal names the task-file key to change. It
 * is what ContactWrenchConeConstraint::Create() checks for a Config built in code, which never passes through
 * loadConfig()'s own checks (//humanoid_nmpc/humanoid_wb_mpc:testWBContactWrenchConeCreate covers that path on a real
 * model).
 */
TEST(ContactWrenchConeConfigValidationTest, EveryOutOfRangeValueIsRefusedNamingItsKey) {
  // Positive control: the test configuration and its boundary values are accepted.
  EXPECT_TRUE(ContactWrenchConeConstraint::validateConfig(makeTestConeConfig()).ok());
  ContactWrenchConeConstraint::Config boundary = makeTestConeConfig(3);
  boundary.torsionalFrictionCoefficient = 0.0;
  boundary.minNormalForce = 0.0;
  boundary.gripperForce = 0.0;
  EXPECT_TRUE(ContactWrenchConeConstraint::validateConfig(boundary).ok());

  const std::string block = absl::StrCat(ContactWrenchConeConstraint::kConfigField, ".");
  std::vector<std::pair<std::string, ContactWrenchConeConstraint::Config>> bad;
  bad.emplace_back("num_basis_vectors", makeTestConeConfig(2));
  bad.emplace_back("num_basis_vectors", makeTestConeConfig(0));
  for (const scalar_t value : {0.0, -0.3, std::nan(""), std::numeric_limits<scalar_t>::infinity()}) {
    ContactWrenchConeConstraint::Config config = makeTestConeConfig();
    config.frictionCoefficient = value;
    bad.emplace_back("friction_coefficient", config);
  }
  for (const scalar_t value : {-0.01, std::nan("")}) {
    ContactWrenchConeConstraint::Config torsion = makeTestConeConfig();
    torsion.torsionalFrictionCoefficient = value;
    bad.emplace_back("torsional_friction_coefficient", torsion);
    ContactWrenchConeConstraint::Config minForce = makeTestConeConfig();
    minForce.minNormalForce = value;
    bad.emplace_back("min_normal_force", minForce);
    ContactWrenchConeConstraint::Config gripper = makeTestConeConfig();
    gripper.gripperForce = value;
    bad.emplace_back("gripper_force", gripper);
  }
  for (const std::pair<std::string, ContactWrenchConeConstraint::Config>& entry : bad) {
    SCOPED_TRACE(entry.first);
    const absl::Status status = ContactWrenchConeConstraint::validateConfig(entry.second);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
    EXPECT_NE(status.message().find(block + entry.first), absl::string_view::npos) << status;
  }

  ContactWrenchConeConstraint::Config offPatch = makeTestConeConfig();
  offPatch.patchOffset = vector3_t(std::nan(""), 0.0, 0.0);
  EXPECT_EQ(ContactWrenchConeConstraint::validateConfig(offPatch).code(), absl::StatusCode::kInvalidArgument);
}

/**
 * The rows themselves are only reachable with a validated Config through the factories. A caller that bypasses them
 * used to get std::invalid_argument("numBasisVectors must be at least 3"), naming neither the block nor the key; it
 * now stops on validateConfig()'s message.
 */
TEST(ContactWrenchConeBasisValidationDeathTest, TheConeRowsRefuseTooFewFrictionFacetsNamingTheKey) {
  // Positive control: three facets build.
  EXPECT_EQ(buildLocalWrenchConeRows(makeTestConeConfig(3), makeTestContactRectangle()).numRows(), 3u + 7u);
  EXPECT_DEATH(buildLocalWrenchConeRows(makeTestConeConfig(2), makeTestContactRectangle()),
               "contacts\\.contact_wrench_cone_soft_constraint\\.num_basis_vectors");
}

TEST(ContactWrenchConeBasisValidationTest, PatchPointOffTheFootprintIsRejectedOnlyWhereItMatters) {
  // The conservative set applies its friction and torsion rays at the patch point, so a patch point off the footprint
  // would put their center of pressure outside the support.
  ContactWrenchConeConstraint::Config config = makeTestConeConfig();
  config.patchOffset = vector3_t(0.5, 0.0, 0.0);
  expectInvalidArgumentNaming(config, makeTestContactRectangle(), std::string(kConservativeInnerApproximationGeneratorSet),
                              "contacts.contact_rectangle.");
  // The exact set applies every generator at a footprint corner and only measures the torsion about the patch point,
  // so the cone the rows define is still well formed and still reproduced.
  const ContactWrenchConeBasisMatrix exact = createOrDie(config, makeTestContactRectangle(), std::string(kExactWrenchConeGeneratorSet));
  for (const vector6_t& wrench : sampleConeWrenches(config, makeTestContactRectangle(), /*numSamples=*/200, /*seed=*/5)) {
    EXPECT_TRUE(isRepresentable(exact.getBasisMatrix(), wrench)) << wrench.transpose();
  }
}

// ==================== Soundness: every generator inside the cone, for every set ====================

TEST(ContactWrenchConeBasisSoundnessTest, EveryGeneratorOfEverySetSatisfiesTheWrenchConeRows) {
  for (const std::string& set : allGeneratorSets()) {
    for (const GeometryCase& c : geometryCases()) {
      const ContactWrenchConeConstraint::Config config = configFor(c);
      const ContactRectangle rectangle = makeRectangle(c.xMin, c.xMax, c.yMin, c.yMax);
      const ContactWrenchConeBasisMatrix basis = createOrDie(config, rectangle, set);
      const ContactWrenchConeRows rows = buildLocalWrenchConeRows(config, rectangle);
      const matrix_t& B = basis.getBasisMatrix();
      EXPECT_EQ(B.fullPivLu().rank(), 6) << set << " " << describe(c) << ": some admissible wrench direction is unreachable";
      for (Eigen::Index j = 0; j < B.cols(); ++j) {
        EXPECT_GE(worstConeRow(rows, vector6_t(B.col(j))), -kTol) << set << " " << describe(c) << " generator " << j << " leaves the cone";
        // The contact-implicit load indicator sums the scalings, which is the normal force only if every column
        // carries a unit normal force (humanoid_nmpc/docs/contact_implicit_mpc/README.md).
        EXPECT_NEAR(B(/*row=*/2, j), 1.0, kTol) << set << " " << describe(c) << " generator " << j;
      }
    }
  }
}

TEST(ContactWrenchConeBasisSoundnessTest, AnyNonNegativeLambdaStaysInsideTheCone) {
  for (const std::string& set : allGeneratorSets()) {
    const ContactWrenchConeBasisMatrix basis = createOrDie(makeTestConeConfig(), makeTestContactRectangle(), set);
    const ContactWrenchConeRows rows = buildLocalWrenchConeRows(makeTestConeConfig(), makeTestContactRectangle());
    const matrix_t& B = basis.getBasisMatrix();
    std::mt19937 gen(42);
    std::uniform_real_distribution<scalar_t> magnitude(0.0, 100.0);
    std::bernoulli_distribution active(0.5);
    for (int trial = 0; trial < 1000; ++trial) {
      vector_t lambda = vector_t::Zero(B.cols());
      for (Eigen::Index i = 0; i < lambda.size(); ++i) {
        if (active(gen)) lambda(i) = magnitude(gen);
      }
      const vector6_t wrench = B * lambda;
      // The cone rows are homogeneous, so the absolute slack grows with the wrench.
      EXPECT_GE(worstConeRow(rows, wrench), -kTol * std::max(1.0, wrench.norm())) << set << " trial " << trial;
    }
  }
}

TEST(ContactWrenchConeBasisSoundnessTest, DimensionsOfEachSet) {
  for (size_t numDirections : {3, 4, 8}) {
    const ContactWrenchConeBasisMatrix conservative = createOrDie(makeTestConeConfig(numDirections), makeTestContactRectangle(),
                                                                  std::string(kConservativeInnerApproximationGeneratorSet));
    EXPECT_EQ(conservative.numBasis(), numDirections + 7);
    const ContactWrenchConeBasisMatrix exact =
        createOrDie(makeTestConeConfig(numDirections), makeTestContactRectangle(), std::string(kExactWrenchConeGeneratorSet));
    EXPECT_EQ(exact.numBasis(), 8 * numDirections);
    for (const ContactWrenchConeBasisMatrix* absl_nonnull basis : {&conservative, &exact}) {
      EXPECT_EQ(basis->getBasisMatrix().rows(), 6);
      EXPECT_EQ(static_cast<size_t>(basis->getBasisMatrix().cols()), basis->numBasis());
      EXPECT_EQ(basis->getBasisMatrixPseudoInverse().rows(), static_cast<Eigen::Index>(basis->numBasis()));
      EXPECT_EQ(basis->getBasisMatrixPseudoInverse().cols(), 6);
    }
  }
}

// ==================== Completeness: the exact set IS the cone wrench mode enforces ====================

TEST(ContactWrenchConeExactSetTest, EveryWrenchOfTheConeIsANonNegativeCombination) {
  for (const GeometryCase& c : geometryCases()) {
    const ContactWrenchConeConstraint::Config config = configFor(c);
    const ContactRectangle rectangle = makeRectangle(c.xMin, c.xMax, c.yMin, c.yMax);
    const ContactWrenchConeBasisMatrix exact = createOrDie(config, rectangle, std::string(kExactWrenchConeGeneratorSet));
    const ContactWrenchConeRows rows = buildLocalWrenchConeRows(config, rectangle);
    size_t numFailures = 0;
    for (const vector6_t& wrench : sampleConeWrenches(config, rectangle, kNumSamples, /*seed=*/17)) {
      // Positive control on the sampler: every sample really is a wrench wrench mode admits.
      ASSERT_GE(worstConeRow(rows, wrench), -1.0e-9 * wrench.norm()) << describe(c) << ": the sampler left the cone";
      if (!isRepresentable(exact.getBasisMatrix(), wrench)) {
        ++numFailures;
        ADD_FAILURE() << describe(c) << ": admissible wrench " << wrench.transpose() << " is not a non-negative combination";
        if (numFailures > 5) break;
      }
    }
  }
}

TEST(ContactWrenchConeExactSetTest, ColumnsAreExactlyTheVerticesOfTheCone) {
  // The Fz = 1 slice of the cone is (friction polygon) x (footprint) x (torsion interval) up to an affine map, so its
  // vertices are the products of the three polytopes' vertices. Build those independently - polygon vertices as
  // intersections of adjacent facet lines of the rows, the wrench from cross products - and match them to the columns.
  for (const GeometryCase& c : geometryCases()) {
    const ContactWrenchConeConstraint::Config config = configFor(c);
    const ContactRectangle rectangle = makeRectangle(c.xMin, c.xMax, c.yMin, c.yMax);
    const ContactWrenchConeBasisMatrix exact = createOrDie(config, rectangle, std::string(kExactWrenchConeGeneratorSet));
    const ContactWrenchConeRows rows = buildLocalWrenchConeRows(config, rectangle);
    const vector3_t patchPoint = contactPatchReferencePoint(config, rectangle);
    const PolygonBounds& bounds = rectangle.getBounds();

    std::vector<bool> matched(exact.numBasis(), false);
    for (const vector2_t& frictionVertex : frictionPolygonVertices(rows, config.numBasisVectors)) {
      for (const vector2_t& corner : {vector2_t(bounds.x_max, bounds.y_max), vector2_t(bounds.x_max, bounds.y_min),
                                      vector2_t(bounds.x_min, bounds.y_max), vector2_t(bounds.x_min, bounds.y_min)}) {
        for (const scalar_t sign : {1.0, -1.0}) {
          const vector6_t vertex =
              physicalWrench(frictionVertex, corner, sign * config.torsionalFrictionCoefficient, /*normalForce=*/1.0, patchPoint);
          bool found = false;
          for (size_t j = 0; j < exact.numBasis(); ++j) {
            if (!matched[j] && (vector6_t(exact.getBasisMatrix().col(static_cast<Eigen::Index>(j))) - vertex).norm() < 1.0e-12) {
              matched[j] = found = true;
              break;
            }
          }
          EXPECT_TRUE(found) << describe(c) << ": cone vertex " << vertex.transpose() << " is not a column";
        }
      }
    }
    // NOLINTNEXTLINE(argument-comment): std::count's value parameter is a reserved name in libstdc++ (__value)
    EXPECT_EQ(std::count(matched.begin(), matched.end(), true), static_cast<ptrdiff_t>(exact.numBasis())) << describe(c);
  }
}

TEST(ContactWrenchConeExactSetTest, NoColumnIsRedundant) {
  // Every column is an extreme ray: none is a non-negative combination of the others, so 8N is the minimum.
  const ContactWrenchConeBasisMatrix exact =
      createOrDie(makeAtlasConeConfig(), makeAtlasContactRectangle(), std::string(kExactWrenchConeGeneratorSet));
  const matrix_t& B = exact.getBasisMatrix();
  for (Eigen::Index j = 0; j < B.cols(); ++j) {
    matrix_t others(6, B.cols() - 1);
    others.leftCols(j) = B.leftCols(j);
    others.rightCols(B.cols() - j - 1) = B.rightCols(B.cols() - j - 1);
    EXPECT_TRUE(isProvablyNotRepresentable(others, vector6_t(B.col(j)))) << "column " << j;
  }
}

// ==================== The conservative set: a strict inner approximation, and how strict ====================

TEST(ContactWrenchConeConservativeSetTest, IsContainedInTheExactSet) {
  for (const GeometryCase& c : geometryCases()) {
    const ContactWrenchConeConstraint::Config config = configFor(c);
    const ContactRectangle rectangle = makeRectangle(c.xMin, c.xMax, c.yMin, c.yMax);
    const ContactWrenchConeBasisMatrix conservative =
        createOrDie(config, rectangle, std::string(kConservativeInnerApproximationGeneratorSet));
    const ContactWrenchConeBasisMatrix exact = createOrDie(config, rectangle, std::string(kExactWrenchConeGeneratorSet));
    for (Eigen::Index j = 0; j < conservative.getBasisMatrix().cols(); ++j) {
      EXPECT_TRUE(isRepresentable(exact.getBasisMatrix(), vector6_t(conservative.getBasisMatrix().col(j))))
          << describe(c) << " column " << j;
    }
  }
}

TEST(ContactWrenchConeConservativeSetTest, CannotCombineFrictionWithAnOffCenterCopOrTorsion) {
  // The audit's counterexamples on the shipped Atlas cone: a braking / push-off wrench with Fx = 0.25 Fz and the center
  // of pressure at 0.8 x_max, half the friction limit with 0.9 of the torsional limit, and a toe-side center of
  // pressure with half the torsional limit. Wrench mode admits all three with margin, the exact set reproduces all
  // three, and the conservative set provably reproduces none.
  const ContactWrenchConeConstraint::Config config = makeAtlasConeConfig();
  const ContactRectangle rectangle = makeAtlasContactRectangle();
  const ContactWrenchConeRows rows = buildLocalWrenchConeRows(config, rectangle);
  const vector3_t patchPoint = contactPatchReferencePoint(config, rectangle);
  const ContactWrenchConeBasisMatrix conservative =
      createOrDie(config, rectangle, std::string(kConservativeInnerApproximationGeneratorSet));
  const ContactWrenchConeBasisMatrix exact = createOrDie(config, rectangle, std::string(kExactWrenchConeGeneratorSet));

  const std::vector<vector6_t> counterexamples = {
      physicalWrench(vector2_t(0.25, 0.0), vector2_t(0.8 * 0.12, 0.0), /*torsionRatio=*/0.0, /*normalForce=*/400.0, patchPoint),
      physicalWrench(vector2_t(0.25, 0.0), vector2_t(0.0, 0.0), 0.9 * config.torsionalFrictionCoefficient, /*normalForce=*/400.0,
                     patchPoint),
      physicalWrench(vector2_t(0.0, 0.0), vector2_t(0.8 * 0.12, 0.0), 0.5 * config.torsionalFrictionCoefficient, /*normalForce=*/400.0,
                     patchPoint),
  };
  for (const vector6_t& wrench : counterexamples) {
    EXPECT_GT(worstConeRow(rows, wrench), 0.0) << wrench.transpose() << " should be strictly inside the wrench-mode cone";
    EXPECT_TRUE(isRepresentable(exact.getBasisMatrix(), wrench)) << wrench.transpose();
    EXPECT_TRUE(isProvablyNotRepresentable(conservative.getBasisMatrix(), wrench)) << wrench.transpose();
  }
}

TEST(ContactWrenchConeConservativeSetTest, ReportsHowMuchOfTheConeItCovers) {
  // Neither pinned nor claimed: the fraction of the sampled cone the conservative set reproduces is written to the test
  // log and the XML report, and all that is asserted is that it is a strict part of the cone.
  for (const GeometryCase& c : geometryCases()) {
    const ContactWrenchConeConstraint::Config config = configFor(c);
    const ContactRectangle rectangle = makeRectangle(c.xMin, c.xMax, c.yMin, c.yMax);
    const ContactWrenchConeBasisMatrix conservative =
        createOrDie(config, rectangle, std::string(kConservativeInnerApproximationGeneratorSet));
    const std::vector<vector6_t> samples = sampleConeWrenches(config, rectangle, kNumSamples, /*seed=*/23);
    const size_t numCovered = static_cast<size_t>(std::count_if(samples.begin(), samples.end(), [&conservative](const vector6_t& wrench) {
      return isRepresentable(conservative.getBasisMatrix(), wrench);
    }));
    const scalar_t coverage = static_cast<scalar_t>(numCovered) / static_cast<scalar_t>(samples.size());
    LOG(INFO) << "[ coverage ] conservative_inner_approximation reproduces " << 100.0 * coverage << " % of the sampled cone, "
              << describe(c);
    ::testing::Test::RecordProperty(absl::StrCat("coverage_", c.numBasisVectors, "_", c.friction, "_", c.xMax), absl::StrCat(coverage));
    EXPECT_LT(coverage, 1.0) << describe(c) << ": the conservative set is documented as a strict inner approximation";
    EXPECT_GT(coverage, 0.0) << describe(c);
  }
}

TEST(ContactWrenchConeConservativeSetTest, ReproducesTheFrictionPyramidAtThePatchPoint) {
  // What the conservative set DOES reproduce exactly: any force inside the friction pyramid applied at the patch point
  // with no torsion, including the pyramid's edges. Its projection onto the force alone is therefore exact.
  for (const GeometryCase& c : geometryCases()) {
    const ContactWrenchConeConstraint::Config config = configFor(c);
    const ContactRectangle rectangle = makeRectangle(c.xMin, c.xMax, c.yMin, c.yMax);
    const ContactWrenchConeBasisMatrix conservative =
        createOrDie(config, rectangle, std::string(kConservativeInnerApproximationGeneratorSet));
    const ContactWrenchConeRows rows = buildLocalWrenchConeRows(config, rectangle);
    const vector3_t patchPoint = contactPatchReferencePoint(config, rectangle);
    for (const vector2_t& vertex : frictionPolygonVertices(rows, config.numBasisVectors)) {
      for (const scalar_t fraction : {1.0, 0.5}) {
        const vector6_t wrench =
            physicalWrench(fraction * vertex, patchPoint.head<2>(), /*torsionRatio=*/0.0, /*normalForce=*/100.0, patchPoint);
        EXPECT_TRUE(isRepresentable(conservative.getBasisMatrix(), wrench)) << describe(c) << " " << wrench.transpose();
      }
    }
  }
}

TEST(ContactWrenchConeConservativeSetTest, LayoutMatchesTheDocumentation) {
  const ContactWrenchConeConstraint::Config config = makeTestConeConfig(4);
  const ContactRectangle rectangle = makeTestContactRectangle();
  const ContactWrenchConeBasisMatrix basis = createOrDie(config, rectangle, std::string(kConservativeInnerApproximationGeneratorSet));
  const matrix_t& B = basis.getBasisMatrix();
  const size_t N = config.numBasisVectors;
  const ContactWrenchConeRows rows = buildLocalWrenchConeRows(config, rectangle);
  // Columns 0..N-1: friction-pyramid edges, each tight on the two friction facets it sits between.
  for (size_t k = 0; k < N; ++k) {
    const vector_t coneValues = rows.evaluateCone(vector6_t(B.col(static_cast<Eigen::Index>(k))));
    EXPECT_NEAR(coneValues(static_cast<Eigen::Index>(k)), 0.0, kTol) << "friction ray " << k;
    EXPECT_NEAR(coneValues(static_cast<Eigen::Index>((k + 1) % N)), 0.0, kTol) << "friction ray " << k;
  }
  // Column N: the pure normal force, strictly inside the cone.
  EXPECT_GT(worstConeRow(rows, vector6_t(B.col(static_cast<Eigen::Index>(N)))), 0.0);
  // Columns N+1..N+4: normal forces whose center of pressure (-tau_y / Fz, tau_x / Fz) is a footprint corner.
  const PolygonBounds& bounds = rectangle.getBounds();
  const std::vector<vector2_t> corners = {vector2_t(bounds.x_max, bounds.y_max), vector2_t(bounds.x_max, bounds.y_min),
                                          vector2_t(bounds.x_min, bounds.y_max), vector2_t(bounds.x_min, bounds.y_min)};
  for (size_t c = 0; c < corners.size(); ++c) {
    const vector6_t ray = B.col(static_cast<Eigen::Index>(N + 1 + c));
    EXPECT_TRUE(ray.head<2>().isZero(kTol));
    EXPECT_TRUE(vector2_t(-ray(4), ray(3)).isApprox(corners[c], 1.0e-12)) << "CoP ray " << c;
  }
  // Columns N+5, N+6: the torsional limit of either sign, tau_z = +/- mu_torsion * Fz (not its reciprocal).
  EXPECT_NEAR(B(/*row=*/5, static_cast<Eigen::Index>(N + 5)), config.torsionalFrictionCoefficient, kTol);
  EXPECT_NEAR(B(/*row=*/5, static_cast<Eigen::Index>(N + 6)), -config.torsionalFrictionCoefficient, kTol);
}

// ==================== Pseudoinverse and null-space projector ====================

TEST(ContactWrenchConeBasisAlgebraTest, PseudoinverseAndNullSpaceProjectorOfEverySet) {
  for (const std::string& set : allGeneratorSets()) {
    const ContactWrenchConeBasisMatrix basis = createOrDie(makeTestConeConfig(), makeTestContactRectangle(), set);
    const matrix_t& B = basis.getBasisMatrix();
    const matrix_t& Bpinv = basis.getBasisMatrixPseudoInverse();
    const matrix_t& P = basis.getNullSpaceProjector();
    const Eigen::Index n = B.cols();
    // B has full row rank, so B B+ is the identity on the whole wrench space.
    EXPECT_TRUE((B * Bpinv).isApprox(matrix_t::Identity(6, 6), 1.0e-10)) << set;
    ASSERT_EQ(P.rows(), n);
    ASSERT_EQ(P.cols(), n);
    EXPECT_LE((P - P.transpose()).cwiseAbs().maxCoeff(), 1.0e-14) << set << ": the projector must be symmetric";
    EXPECT_LE((P * P - P).cwiseAbs().maxCoeff(), 1.0e-10) << set << ": the projector must be idempotent";
    EXPECT_LE((B * P).cwiseAbs().maxCoeff(), 1.0e-10) << set << ": the projector must map into null(B)";
    EXPECT_NEAR(P.trace(), static_cast<scalar_t>(n - 6), 1.0e-9) << set << ": null(B) has dimension numBasis - 6";
    EXPECT_LE((P * B.transpose()).cwiseAbs().maxCoeff(), 1.0e-10) << set << ": the projector must vanish on range(B^T)";
  }
}

// ==================== Non-negative scalings for a wrench (setContactWrench) ====================

TEST(NonNegativeLeastSquaresTest, SolvesSmallProblemsWithKnownAnswers) {
  // Identity: the answer is the clamped right-hand side.
  const vector_t clamped = solveNonNegativeLeastSquares(matrix_t::Identity(3, 3), (vector_t(3) << 1.0, -2.0, 3.0).finished());
  EXPECT_TRUE(clamped.isApprox((vector_t(3) << 1.0, 0.0, 3.0).finished(), 1.0e-14)) << clamped.transpose();
  // A right-hand side generated by a sparse non-negative combination is reproduced exactly.
  std::mt19937 gen(3);
  std::uniform_real_distribution<scalar_t> entry(-1.0, 1.0);
  for (int trial = 0; trial < 50; ++trial) {
    matrix_t A(6, 20);
    for (Eigen::Index i = 0; i < A.size(); ++i) A.data()[i] = entry(gen);
    vector_t xTrue = vector_t::Zero(20);
    xTrue(trial % 20) = 2.0;
    xTrue((trial * 7 + 3) % 20) = 0.5;
    const vector_t b = A * xTrue;
    const vector_t x = solveNonNegativeLeastSquares(A, b);
    EXPECT_GE(x.minCoeff(), 0.0);
    EXPECT_LE((A * x - b).norm(), 1.0e-10 * b.norm()) << "trial " << trial;
  }
  // The zero right-hand side gives the zero solution.
  EXPECT_TRUE(solveNonNegativeLeastSquares(matrix_t::Identity(3, 5), vector_t::Zero(3)).isZero(0.0));
}

TEST(NonNegativeBasisScalingsTest, ReproduceEveryWrenchOfTheConeEvenWhenTheMinimumNormScalingsAreNegative) {
  for (const std::string& set : allGeneratorSets()) {
    const ContactWrenchConeBasisMatrix basis = createOrDie(makeAtlasConeConfig(), makeAtlasContactRectangle(), set);
    const matrix_t& B = basis.getBasisMatrix();
    // An extreme ray of the cone, and the weight seen from a foot pitched by 15 degrees (the tilted-foot weight
    // compensation the audit found distorted by 7 % in magnitude and 6 % sideways).
    const scalar_t pitch = 15.0 * M_PI / 180.0;
    const std::vector<vector6_t> wrenches = {
        vector6_t(500.0 * B.col(0)), (vector6_t() << 600.0 * std::sin(pitch), 0.0, 600.0 * std::cos(pitch), 0.0, 0.0, 0.0).finished()};
    size_t numExercised = 0;
    for (const vector6_t& wrench : wrenches) {
      const vector_t minimumNorm = basis.getBasisMatrixPseudoInverse() * wrench;
      if (minimumNorm.minCoeff() >= 0.0) {
        continue;  // Only the wrenches where the old clamp went wrong are of interest here.
      }
      ++numExercised;
      // Positive control: clamping the minimum-norm scalings, the old behavior, does NOT reproduce this wrench.
      EXPECT_GT((B * minimumNorm.cwiseMax(0.0) - wrench).norm(), 1.0e-3 * wrench.norm()) << set;
      const vector_t lambda = basis.solveNonNegativeScalings(wrench);
      EXPECT_GE(lambda.minCoeff(), 0.0) << set;
      EXPECT_LE((B * lambda - wrench).norm(), kRepresentableTolerance * wrench.norm())
          << set << ": " << wrench.transpose() << " came back as " << (B * lambda).transpose();
    }
    EXPECT_GT(numExercised, 0u) << set << ": no test wrench has negative minimum-norm scalings";
  }
  // The pitched-foot weight does exercise the NNLS path for the conservative set.
  const ContactWrenchConeBasisMatrix conservative =
      createOrDie(makeAtlasConeConfig(), makeAtlasContactRectangle(), std::string(kConservativeInnerApproximationGeneratorSet));
  const vector6_t pitchedWeight =
      (vector6_t() << 600.0 * std::sin(15.0 * M_PI / 180.0), 0.0, 600.0 * std::cos(15.0 * M_PI / 180.0), 0.0, 0.0, 0.0).finished();
  EXPECT_LT((conservative.getBasisMatrixPseudoInverse() * pitchedWeight).minCoeff(), 0.0);
}

TEST(NonNegativeBasisScalingsTest, KeepTheSpreadMinimumNormScalingsWhenTheyAreNonNegative) {
  for (const std::string& set : allGeneratorSets()) {
    const ContactWrenchConeBasisMatrix basis = createOrDie(makeTestConeConfig(), makeTestContactRectangle(), set);
    const vector6_t wrench = (vector6_t() << 5.0, -3.0, 80.0, 0.0, 0.0, 0.0).finished();
    const vector_t minimumNorm = basis.getBasisMatrixPseudoInverse() * wrench;
    ASSERT_GE(minimumNorm.minCoeff(), 0.0) << set;
    EXPECT_TRUE((basis.solveNonNegativeScalings(wrench) - minimumNorm).isZero(0.0)) << set;
  }
}

TEST(NonNegativeBasisScalingsTest, ReplaceAWrenchOutsideTheConeByTheClosestOneInside) {
  for (const std::string& set : allGeneratorSets()) {
    const ContactWrenchConeBasisMatrix basis = createOrDie(makeTestConeConfig(), makeTestContactRectangle(), set);
    const ContactWrenchConeRows rows = buildLocalWrenchConeRows(makeTestConeConfig(), makeTestContactRectangle());
    const vector6_t outside = (vector6_t() << 100.0, 0.0, 50.0, 0.0, 0.0, 0.0).finished();  // twice the friction limit
    ASSERT_LT(worstConeRow(rows, outside), 0.0);
    const vector_t lambda = basis.solveNonNegativeScalings(outside);
    EXPECT_GE(lambda.minCoeff(), 0.0) << set;
    EXPECT_GE(worstConeRow(rows, vector6_t(basis.getBasisMatrix() * lambda)), -1.0e-9 * outside.norm()) << set;
    EXPECT_TRUE(isProvablyNotRepresentable(basis.getBasisMatrix(), outside)) << set;
  }
}

// ==================== The CppAD library key ====================

feet_array_t<matrix_t> basesFor(const ContactWrenchConeConstraint::Config& config,
                                const ContactRectangle& rectangle,
                                const std::string& set) {
  const ContactWrenchConeBasisMatrix basis = createOrDie(config, rectangle, set);
  return {basis.getBasisMatrix(), basis.getBasisMatrix()};
}

TEST(BasisInputsLibraryKeyTest, IsDeterministicAndSeparatesEveryParameterization) {
  const std::string conservative(kConservativeInnerApproximationGeneratorSet);
  const std::string exact(kExactWrenchConeGeneratorSet);
  const feet_array_t<matrix_t> reference = basesFor(makeTestConeConfig(), makeTestContactRectangle(), conservative);
  EXPECT_EQ(basisInputsLibraryKey(reference),
            basisInputsLibraryKey(basesFor(makeTestConeConfig(), makeTestContactRectangle(), conservative)));

  const std::string key = basisInputsLibraryKey(reference);
  EXPECT_TRUE(std::regex_match(key, std::regex("basis11_[0-9a-f]{16}"))) << key;
  EXPECT_TRUE(std::regex_match(basisInputsLibraryKey(basesFor(makeTestConeConfig(), makeTestContactRectangle(), exact)),
                               std::regex("basis32_[0-9a-f]{16}")));
  EXPECT_NE(key, std::string(kWrenchInputsLibraryKey));

  ContactWrenchConeConstraint::Config otherFriction = makeTestConeConfig();
  otherFriction.frictionCoefficient = 0.6;
  ContactWrenchConeConstraint::Config otherTorsion = makeTestConeConfig();
  otherTorsion.torsionalFrictionCoefficient = 0.04;
  const std::vector<feet_array_t<matrix_t>> others = {
      basesFor(makeTestConeConfig(), makeTestContactRectangle(), exact),
      basesFor(otherFriction, makeTestContactRectangle(), conservative),
      basesFor(otherTorsion, makeTestContactRectangle(), conservative),
      basesFor(makeTestConeConfig(8), makeTestContactRectangle(), conservative),
      basesFor(makeTestConeConfig(), makeRectangle(-0.10, 0.12, -0.05, 0.05), conservative),
  };
  for (const feet_array_t<matrix_t>& other : others) {
    EXPECT_NE(basisInputsLibraryKey(other), key);
  }
  // The two feet are part of the key: an asymmetric pair differs from a symmetric one.
  feet_array_t<matrix_t> mixed = reference;
  mixed[1] = others[1][1];
  EXPECT_NE(basisInputsLibraryKey(mixed), key);
}

TEST(BasisInputsLibraryKeyTest, IgnoresRoundOffButNotRealChanges) {
  const feet_array_t<matrix_t> reference =
      basesFor(makeTestConeConfig(), makeTestContactRectangle(), std::string(kConservativeInnerApproximationGeneratorSet));
  feet_array_t<matrix_t> roundOff = reference;
  roundOff[0](0, 0) *= 1.0 + 1.0e-15;
  ASSERT_EQ(roundOff[1](0, 4), 0.0) << "the pure normal ray (column N = 4) has no tangential force";
  roundOff[1](0, 4) = -0.0;     // the sign of a zero entry
  roundOff[0](1, 4) = 6.0e-17;  // a zero that a libm computed as cos(pi / 2)
  EXPECT_EQ(basisContentHash(roundOff), basisContentHash(reference));
  feet_array_t<matrix_t> changed = reference;
  changed[0](0, 0) += 1.0e-6;
  EXPECT_NE(basisContentHash(changed), basisContentHash(reference));
}

}  // namespace
}  // namespace ocs2::humanoid
