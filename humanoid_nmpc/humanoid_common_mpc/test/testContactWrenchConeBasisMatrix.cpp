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

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"

/**
 * Tests of the wrench-cone generator basis used by the basis-vector contact input formulation.
 *
 * The property the whole formulation rests on is that every generator lies inside the wrench cone that
 * ContactWrenchConeConstraint would otherwise enforce, because that constraint is dropped when basis-vector inputs are
 * active. The generators are therefore checked against the constraint's own rows (buildLocalWrenchConeRows) rather
 * than against a hand-written expectation of what each column should contain.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kTol = 1e-12;

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

/** Worst (most negative) row of the homogeneous cone evaluated on a wrench; non-negative means admissible. */
scalar_t worstConeRow(const ContactWrenchConeRows& rows, const vector6_t& wrench) {
  return rows.evaluateCone(wrench).minCoeff();
}

class ContactWrenchConeBasisMatrixTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_ = makeTestConeConfig(4);
    rect_ = std::make_unique<ContactRectangle>(makeTestContactRectangle());
    basis_ = std::make_unique<ContactWrenchConeBasisMatrix>(config_, *rect_);
    rows_ = buildLocalWrenchConeRows(config_, *rect_);
  }

  ContactWrenchConeConstraint::Config config_;
  std::unique_ptr<ContactRectangle> rect_;
  std::unique_ptr<ContactWrenchConeBasisMatrix> basis_;
  ContactWrenchConeRows rows_;
};

// ==================== Dimension tests ====================

TEST_F(ContactWrenchConeBasisMatrixTest, DimensionsAreCorrect) {
  // numBasis = N + 7 (N friction + 1 normal + 4 CoP + 2 torsional)
  const size_t expectedNumBasis = config_.numBasisVectors + 7;
  EXPECT_EQ(basis_->numBasis(), expectedNumBasis);

  const matrix_t& B = basis_->getBasisMatrix();
  EXPECT_EQ(B.rows(), 6);
  EXPECT_EQ(B.cols(), static_cast<int>(expectedNumBasis));

  const matrix_t& B_pinv = basis_->getBasisMatrixPseudoInverse();
  EXPECT_EQ(B_pinv.rows(), static_cast<int>(expectedNumBasis));
  EXPECT_EQ(B_pinv.cols(), 6);

  // The generators have to span the whole wrench space, otherwise some admissible wrench is unreachable.
  EXPECT_EQ(B.fullPivLu().rank(), 6);
}

TEST_F(ContactWrenchConeBasisMatrixTest, DimensionsWithDifferentN) {
  ContactWrenchConeBasisMatrix basis8(makeTestConeConfig(8), *rect_);
  EXPECT_EQ(basis8.numBasis(), 15u);  // 8 + 7
  EXPECT_EQ(basis8.getBasisMatrix().cols(), 15);

  ContactWrenchConeBasisMatrix basis3(makeTestConeConfig(3), *rect_);
  EXPECT_EQ(basis3.numBasis(), 10u);  // 3 + 7
}

TEST_F(ContactWrenchConeBasisMatrixTest, RejectsFewerThanThreeFrictionDirections) {
  EXPECT_THROW(ContactWrenchConeBasisMatrix(makeTestConeConfig(2), *rect_), std::invalid_argument);
  EXPECT_THROW(buildLocalWrenchConeRows(makeTestConeConfig(2), *rect_), std::invalid_argument);
}

// ==================== The invariant: generators inside the cone ====================

TEST_F(ContactWrenchConeBasisMatrixTest, EveryGeneratorSatisfiesTheWrenchConeRows) {
  const matrix_t& B = basis_->getBasisMatrix();
  for (size_t j = 0; j < basis_->numBasis(); ++j) {
    const vector6_t generator = B.col(j);
    EXPECT_GE(worstConeRow(rows_, generator), -kTol)
        << "generator " << j << " = " << generator.transpose() << " violates the wrench cone by " << worstConeRow(rows_, generator);
  }
}

TEST_F(ContactWrenchConeBasisMatrixTest, GeneratorsSitOnTheConeBoundaryExceptThePureNormalRay) {
  // A generator that is strictly inside the cone wastes part of the admissible set. Every generator except the pure
  // normal force (which is interior by construction) has to be tight on at least one row.
  const matrix_t& B = basis_->getBasisMatrix();
  const size_t normalColumn = config_.numBasisVectors;
  for (size_t j = 0; j < basis_->numBasis(); ++j) {
    const scalar_t worst = worstConeRow(rows_, vector6_t(B.col(j)));
    if (j == normalColumn) {
      EXPECT_GT(worst, 0.0) << "the pure normal ray should be strictly inside the cone";
    } else {
      EXPECT_NEAR(worst, 0.0, kTol) << "generator " << j << " is strictly inside the cone and needlessly conservative";
    }
  }
}

TEST_F(ContactWrenchConeBasisMatrixTest, AnyNonNegativeLambdaStaysInsideTheCone) {
  const matrix_t& B = basis_->getBasisMatrix();
  std::mt19937 gen(42);
  std::uniform_real_distribution<scalar_t> magnitude(0.0, 100.0);
  std::bernoulli_distribution active(0.5);

  for (int trial = 0; trial < 2000; ++trial) {
    vector_t lambda = vector_t::Zero(basis_->numBasis());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) {
      if (active(gen)) lambda(i) = magnitude(gen);
    }
    const vector6_t wrench = B * lambda;
    // Scale the tolerance with the wrench: the cone rows are homogeneous, so the absolute slack grows with it.
    EXPECT_GE(worstConeRow(rows_, wrench), -kTol * std::max(1.0, wrench.norm()))
        << "trial " << trial << ": lambda = " << lambda.transpose() << " leaves the cone";
  }
}

TEST_F(ContactWrenchConeBasisMatrixTest, ConeIsReproducedForEveryConfiguration) {
  // Footprints that are off-centre, patch offsets, and different friction and generator counts.
  struct Case {
    size_t numBasisVectors;
    scalar_t friction;
    scalar_t torsionalFriction;
    scalar_t xMin, xMax, yMin, yMax;
    vector3_t patchOffset;
  };
  const std::vector<Case> cases = {
      {4, 0.5, 0.05, -0.13, 0.13, -0.065, 0.065, vector3_t::Zero()},
      {8, 0.7, 0.05, -0.10, 0.10, -0.05, 0.05, vector3_t::Zero()},
      {3, 0.9, 0.10, -0.10, 0.10, -0.05, 0.05, vector3_t::Zero()},
      {6, 0.6, 0.08, -0.05, 0.20, -0.03, 0.09, vector3_t::Zero()},             // off-centre footprint
      {6, 0.6, 0.08, -0.05, 0.20, -0.03, 0.09, vector3_t(0.04, 0.02, 0.0)},    // explicit patch offset
      {5, 0.3, 0.02, -0.08, 0.08, -0.04, 0.04, vector3_t(-0.08, -0.04, 0.0)},  // offset on the footprint edge
  };

  for (const Case& c : cases) {
    ContactWrenchConeConstraint::Config config = makeTestConeConfig(c.numBasisVectors);
    config.frictionCoefficient = c.friction;
    config.torsionalFrictionCoefficient = c.torsionalFriction;
    config.patchOffset = c.patchOffset;
    const ContactRectangle rect = makeRectangle(c.xMin, c.xMax, c.yMin, c.yMax);

    const ContactWrenchConeBasisMatrix basis(config, rect);
    const ContactWrenchConeRows rows = buildLocalWrenchConeRows(config, rect);
    const matrix_t& B = basis.getBasisMatrix();
    EXPECT_EQ(B.fullPivLu().rank(), 6);
    for (size_t j = 0; j < basis.numBasis(); ++j) {
      EXPECT_GE(worstConeRow(rows, vector6_t(B.col(j))), -kTol)
          << "N=" << c.numBasisVectors << " mu=" << c.friction << " generator " << j << " leaves the cone";
    }
  }
}

TEST_F(ContactWrenchConeBasisMatrixTest, RejectsPatchOffsetOutsideTheFootprint) {
  // Every generator is applied at the patch point, so a patch point off the footprint would put the centre of
  // pressure outside the support and silently break the CoP limit.
  ContactWrenchConeConstraint::Config config = makeTestConeConfig(4);
  config.patchOffset = vector3_t(0.5, 0.0, 0.0);
  EXPECT_THROW(ContactWrenchConeBasisMatrix(config, *rect_), std::invalid_argument);
}

// ==================== Tightness against the wrench-space formulation ====================

TEST_F(ContactWrenchConeBasisMatrixTest, FrictionEdgeRaysReproduceTheFrictionFacets) {
  // The friction facets bound the tangential force along numBasisVectors evenly spaced directions, so the feasible
  // cone has its edges half a sector away at a tangential magnitude of mu / cos(pi / N). The generators have to be
  // exactly those edges: a smaller radius would make basis mode more conservative than wrench mode, a larger one
  // would let the MPC exceed the friction limit.
  const matrix_t& B = basis_->getBasisMatrix();
  const size_t N = config_.numBasisVectors;
  const scalar_t expectedRadius = config_.frictionCoefficient / std::cos(M_PI / static_cast<scalar_t>(N));

  for (size_t k = 0; k < N; ++k) {
    const vector6_t ray = B.col(k);
    EXPECT_NEAR(ray(2), 1.0, kTol) << "friction ray " << k << " should carry a unit normal force";
    EXPECT_NEAR(ray.head<2>().norm(), expectedRadius, kTol) << "friction ray " << k << " has the wrong tangential magnitude";

    const scalar_t expectedAngle = (static_cast<scalar_t>(k) + 0.5) * 2.0 * M_PI / static_cast<scalar_t>(N);
    EXPECT_NEAR(std::atan2(ray(1), ray(0)), std::atan2(std::sin(expectedAngle), std::cos(expectedAngle)), 1e-12);

    // Tight on exactly the two friction facets it sits between.
    const vector_t coneValues = rows_.evaluateCone(ray);
    EXPECT_NEAR(coneValues(k), 0.0, kTol) << "friction ray " << k << " should touch facet " << k;
    EXPECT_NEAR(coneValues((k + 1) % N), 0.0, kTol) << "friction ray " << k << " should touch facet " << (k + 1) % N;
  }

  // A wrench on the friction limit is reachable, which is what makes basis mode as permissive as wrench mode.
  vector_t lambda = vector_t::Zero(basis_->numBasis());
  lambda(0) = 10.0;
  const vector6_t wrench = B * lambda;
  EXPECT_NEAR(wrench.head<2>().norm(), expectedRadius * wrench(2), 1e-10);
  EXPECT_GT(wrench.head<2>().norm(), config_.frictionCoefficient * wrench(2))
      << "the polyhedral cone extends past the inscribed circle, as the wrench-space facets do";
}

TEST_F(ContactWrenchConeBasisMatrixTest, CopCornerRaysReachEveryCornerOfTheFootprint) {
  const matrix_t& B = basis_->getBasisMatrix();
  const PolygonBounds& bounds = rect_->getBounds();
  const std::vector<vector2_t> expectedCorners = {vector2_t(bounds.x_max, bounds.y_max), vector2_t(bounds.x_max, bounds.y_min),
                                                  vector2_t(bounds.x_min, bounds.y_max), vector2_t(bounds.x_min, bounds.y_min)};
  for (size_t c = 0; c < expectedCorners.size(); ++c) {
    const vector6_t ray = B.col(config_.numBasisVectors + 1 + c);
    EXPECT_NEAR(ray(2), 1.0, kTol) << "CoP ray " << c << " should carry a unit normal force";
    EXPECT_TRUE(ray.head<2>().isZero(kTol)) << "CoP ray " << c << " should carry no tangential force";
    // Centre of pressure of a wrench: (-tau_y / Fz, tau_x / Fz).
    const vector2_t cop(-ray(4) / ray(2), ray(3) / ray(2));
    EXPECT_TRUE(cop.isApprox(expectedCorners[c], 1e-12)) << "CoP ray " << c << " lands at " << cop.transpose();
  }
}

TEST_F(ContactWrenchConeBasisMatrixTest, TorsionalRaysReachTheTorsionalFrictionLimit) {
  const matrix_t& B = basis_->getBasisMatrix();
  const vector6_t positive = B.col(config_.numBasisVectors + 5);
  const vector6_t negative = B.col(config_.numBasisVectors + 6);

  for (const vector6_t& ray : {positive, negative}) {
    EXPECT_NEAR(ray(2), 1.0, kTol) << "a torsional ray should carry a unit normal force";
    EXPECT_TRUE(ray.head<2>().isZero(kTol));
  }
  // tau_z = +/- mu_torsion * Fz, i.e. exactly the torsional friction limit, not its reciprocal.
  EXPECT_NEAR(positive(5), config_.torsionalFrictionCoefficient * positive(2), kTol);
  EXPECT_NEAR(negative(5), -config_.torsionalFrictionCoefficient * negative(2), kTol);

  vector_t lambda = vector_t::Zero(basis_->numBasis());
  lambda(config_.numBasisVectors + 5) = 1.0;
  lambda(config_.numBasisVectors + 6) = 1.0;
  EXPECT_NEAR((B * lambda)(5), 0.0, kTol) << "equal torsional lambdas should cancel tau_z";
}

TEST_F(ContactWrenchConeBasisMatrixTest, HigherFrictionWidensTheFrictionRays) {
  ContactWrenchConeConstraint::Config configHigh = makeTestConeConfig(4);
  configHigh.frictionCoefficient = 1.5;
  const ContactWrenchConeBasisMatrix basisHigh(configHigh, *rect_);

  const matrix_t& B_high = basisHigh.getBasisMatrix();
  const matrix_t& B_low = basis_->getBasisMatrix();
  for (size_t k = 0; k < 4; ++k) {
    // The normal component stays at one; the tangential reach is what scales with the friction coefficient.
    EXPECT_NEAR(B_high(2, k), 1.0, kTol);
    EXPECT_NEAR(B_low(2, k), 1.0, kTol);
    EXPECT_GT(B_high.col(k).head<2>().norm(), B_low.col(k).head<2>().norm()) << "higher friction should widen friction ray " << k;
  }
}

// ==================== Pseudoinverse ====================

TEST_F(ContactWrenchConeBasisMatrixTest, PseudoinverseRoundTripOnColumnSpace) {
  const matrix_t& B = basis_->getBasisMatrix();
  const matrix_t& B_pinv = basis_->getBasisMatrixPseudoInverse();

  vector_t lambda = vector_t::Random(basis_->numBasis()).cwiseAbs();
  vector_t W = B * lambda;
  vector_t W_recovered = B * (B_pinv * W);
  EXPECT_TRUE(W_recovered.isApprox(W, 1e-10)) << "Round-trip failed:\n  W = " << W.transpose()
                                              << "\n  W_recovered = " << W_recovered.transpose();
}

TEST_F(ContactWrenchConeBasisMatrixTest, PseudoinverseIdentityOnColumnSpace) {
  const matrix_t& B = basis_->getBasisMatrix();
  const matrix_t& B_pinv = basis_->getBasisMatrixPseudoInverse();

  matrix_t P = B * B_pinv;
  EXPECT_EQ(P.rows(), 6);
  EXPECT_EQ(P.cols(), 6);
  // B has full row rank, so B * B_pinv is the identity on the whole wrench space.
  EXPECT_TRUE(P.isApprox(matrix_t::Identity(6, 6), 1e-10)) << "B * B_pinv =\n" << P;
}

TEST_F(ContactWrenchConeBasisMatrixTest, RecoverArbitraryFeasibleWrench) {
  const matrix_t& B = basis_->getBasisMatrix();
  const matrix_t& B_pinv = basis_->getBasisMatrixPseudoInverse();

  for (int trial = 0; trial < 10; ++trial) {
    vector_t lambda_orig = vector_t::Random(basis_->numBasis()).cwiseAbs();
    vector_t W = B * lambda_orig;
    vector_t W_from_recovered = B * (B_pinv * W);
    EXPECT_TRUE(W_from_recovered.isApprox(W, 1e-9)) << "Trial " << trial << ": B * B_pinv * W should recover W\n"
                                                    << "  W = " << W.transpose() << "\n  W_rec = " << W_from_recovered.transpose();
  }
}

TEST_F(ContactWrenchConeBasisMatrixTest, NonNegativeLambdaProducesPositiveNormalForce) {
  const matrix_t& B = basis_->getBasisMatrix();
  vector_t lambda = vector_t::Zero(basis_->numBasis());
  lambda(config_.numBasisVectors) = 10.0;  // pure normal force column
  EXPECT_GT((B * lambda)(2), 0.0) << "Pure normal ray should produce positive Fz";

  lambda.setConstant(1.0);
  EXPECT_GT((B * lambda)(2), 0.0) << "All positive lambda should produce positive Fz";
}

}  // namespace
}  // namespace ocs2::humanoid
