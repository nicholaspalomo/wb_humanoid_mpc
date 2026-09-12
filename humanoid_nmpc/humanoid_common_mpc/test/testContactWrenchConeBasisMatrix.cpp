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

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"

namespace ocs2::humanoid {
namespace {

// Helper to create a typical foot contact rectangle for testing.
ContactRectangle makeTestContactRectangle() {
  // Approximate Atlas foot dimensions: ~20cm long, ~10cm wide
  const PolygonBounds bounds(-0.10, 0.10, -0.05, 0.05);
  const ContactCenterPoint center("test_contact", "test_joint", vector3_t::Zero());
  return ContactRectangle(bounds, center);
}

// Default cone config matching typical Atlas parameters.
ContactWrenchConeConstraint::Config makeTestConeConfig(size_t numBasisVectors = 4) {
  ContactWrenchConeConstraint::Config config;
  config.numBasisVectors = numBasisVectors;
  config.frictionCoefficient = 0.7;
  config.torsionalFrictionCoefficient = 0.05;
  config.minNormalForce = 5.0;
  config.gripperForce = 0.0;
  return config;
}

class ContactWrenchConeBasisMatrixTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_ = makeTestConeConfig(4);
    rect_ = std::make_unique<ContactRectangle>(PolygonBounds(-0.10, 0.10, -0.05, 0.05),
                                               ContactCenterPoint("test_contact", "test_joint", vector3_t::Zero()));
    basis_ = std::make_unique<ContactWrenchConeBasisMatrix>(config_, *rect_);
  }

  ContactWrenchConeConstraint::Config config_;
  std::unique_ptr<ContactRectangle> rect_;
  std::unique_ptr<ContactWrenchConeBasisMatrix> basis_;
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
}

TEST_F(ContactWrenchConeBasisMatrixTest, DimensionsWithDifferentN) {
  // Test with N = 8 basis vectors
  ContactWrenchConeConstraint::Config config8 = makeTestConeConfig(8);
  ContactWrenchConeBasisMatrix basis8(config8, *rect_);
  EXPECT_EQ(basis8.numBasis(), 15u);  // 8 + 7
  EXPECT_EQ(basis8.getBasisMatrix().cols(), 15);

  // Test with minimum N = 3
  ContactWrenchConeConstraint::Config config3 = makeTestConeConfig(3);
  ContactWrenchConeBasisMatrix basis3(config3, *rect_);
  EXPECT_EQ(basis3.numBasis(), 10u);  // 3 + 7
}

// ==================== Pseudoinverse round-trip ====================

TEST_F(ContactWrenchConeBasisMatrixTest, PseudoinverseRoundTripOnColumnSpace) {
  // For any vector W in the column space of B, B * B⁺ * W ≈ W.
  // Generate W from B by picking random non-negative λ.
  const matrix_t& B = basis_->getBasisMatrix();
  const matrix_t& B_pinv = basis_->getBasisMatrixPseudoInverse();
  const size_t numBasis = basis_->numBasis();

  // Generate λ ≥ 0
  vector_t lambda = vector_t::Random(numBasis).cwiseAbs();
  vector_t W = B * lambda;

  // Round trip: B * B⁺ * W should recover W
  vector_t W_recovered = B * (B_pinv * W);
  EXPECT_TRUE(W_recovered.isApprox(W, 1e-10)) << "Round-trip failed:\n  W = " << W.transpose()
                                              << "\n  W_recovered = " << W_recovered.transpose();
}

TEST_F(ContactWrenchConeBasisMatrixTest, PseudoinverseIdentityOnColumnSpace) {
  // B * B⁺ should act as the identity on the column space.
  // Since B is 6 × numBasis with numBasis > 6, B has rank ≤ 6.
  // B * B⁺ should be a 6×6 projection matrix (P² = P) on the column space.
  const matrix_t& B = basis_->getBasisMatrix();
  const matrix_t& B_pinv = basis_->getBasisMatrixPseudoInverse();

  matrix_t P = B * B_pinv;
  EXPECT_EQ(P.rows(), 6);
  EXPECT_EQ(P.cols(), 6);

  // P should be idempotent: P² ≈ P
  matrix_t P2 = P * P;
  EXPECT_TRUE(P2.isApprox(P, 1e-10)) << "B * B_pinv is not idempotent:\n  P =\n" << P << "\n  P² =\n" << P2;
}

// ==================== Wrench cone feasibility ====================

TEST_F(ContactWrenchConeBasisMatrixTest, NonNegativeLambdaProducesPositiveNormalForce) {
  // Any λ ≥ 0 with at least one positive entry for a normal-force generator
  // should yield Fz > 0.
  const matrix_t& B = basis_->getBasisMatrix();
  const size_t numBasis = basis_->numBasis();

  // λ with only the pure normal force ray active (column N)
  vector_t lambda = vector_t::Zero(numBasis);
  lambda(config_.numBasisVectors) = 10.0;  // Pure normal force column
  vector_t W = B * lambda;
  EXPECT_GT(W(2), 0.0) << "Pure normal ray should produce positive Fz";

  // λ with all entries uniformly positive — should still have Fz > 0
  lambda.setConstant(1.0);
  W = B * lambda;
  EXPECT_GT(W(2), 0.0) << "All positive lambda should produce positive Fz";
}

TEST_F(ContactWrenchConeBasisMatrixTest, FrictionPyramidRaysHaveCorrectStructure) {
  const matrix_t& B = basis_->getBasisMatrix();
  const scalar_t mu = config_.frictionCoefficient;

  for (size_t k = 0; k < config_.numBasisVectors; ++k) {
    // Each friction ray: [cos(θ), sin(θ), μ, 0, 0, 0]^T
    const scalar_t fx = B(0, k);
    const scalar_t fy = B(1, k);
    const scalar_t fz = B(2, k);

    // Check: tangential magnitude should be ~1
    const scalar_t tangent_norm = std::sqrt(fx * fx + fy * fy);
    EXPECT_NEAR(tangent_norm, 1.0, 1e-12) << "Friction ray " << k << " tangential norm should be 1";

    // Check: normal component equals friction coefficient
    EXPECT_NEAR(fz, mu, 1e-12) << "Friction ray " << k << " Fz should equal friction coefficient";

    // Moments should be zero for friction rays
    EXPECT_NEAR(B(3, k), 0.0, 1e-12) << "Friction ray " << k << " τx should be 0";
    EXPECT_NEAR(B(4, k), 0.0, 1e-12) << "Friction ray " << k << " τy should be 0";
    EXPECT_NEAR(B(5, k), 0.0, 1e-12) << "Friction ray " << k << " τz should be 0";
  }
}

TEST_F(ContactWrenchConeBasisMatrixTest, PureNormalRayStructure) {
  const matrix_t& B = basis_->getBasisMatrix();
  const size_t col = config_.numBasisVectors;  // Pure normal force column

  EXPECT_NEAR(B(0, col), 0.0, 1e-12) << "Normal ray Fx should be 0";
  EXPECT_NEAR(B(1, col), 0.0, 1e-12) << "Normal ray Fy should be 0";
  EXPECT_NEAR(B(2, col), 1.0, 1e-12) << "Normal ray Fz should be 1";
  EXPECT_NEAR(B(3, col), 0.0, 1e-12) << "Normal ray τx should be 0";
  EXPECT_NEAR(B(4, col), 0.0, 1e-12) << "Normal ray τy should be 0";
  EXPECT_NEAR(B(5, col), 0.0, 1e-12) << "Normal ray τz should be 0";
}

TEST_F(ContactWrenchConeBasisMatrixTest, CoPCornerRaysProduceValidMoments) {
  // Each CoP corner ray should produce Fz > 0 and moments at the corners.
  const matrix_t& B = basis_->getBasisMatrix();
  const PolygonBounds& bounds = rect_->getBounds();

  // CoP corner rays start at column N+1 (columns N+1 through N+4)
  for (size_t c = 0; c < 4; ++c) {
    const size_t col = config_.numBasisVectors + 1 + c;
    EXPECT_NEAR(B(2, col), 1.0, 1e-12) << "CoP ray " << c << " Fz should be 1";
  }

  // Verify that activating only CoP rays with equal weights produces a
  // wrench with zero net moment (since the corners cancel out for symmetric bounds)
  vector_t lambda = vector_t::Zero(basis_->numBasis());
  lambda(config_.numBasisVectors + 1) = 1.0;
  lambda(config_.numBasisVectors + 2) = 1.0;
  lambda(config_.numBasisVectors + 3) = 1.0;
  lambda(config_.numBasisVectors + 4) = 1.0;

  vector_t W = B * lambda;

  // For symmetric bounds (y_min = -y_max, x_min = -x_max), moments should cancel
  if (std::abs(bounds.y_min + bounds.y_max) < 1e-10) {
    EXPECT_NEAR(W(3), 0.0, 1e-10) << "Equal CoP lambdas with symmetric bounds should cancel τx";
  }
  if (std::abs(bounds.x_min + bounds.x_max) < 1e-10) {
    EXPECT_NEAR(W(4), 0.0, 1e-10) << "Equal CoP lambdas with symmetric bounds should cancel τy";
  }
}

TEST_F(ContactWrenchConeBasisMatrixTest, TorsionalRaysAreAntisymmetric) {
  const matrix_t& B = basis_->getBasisMatrix();
  const size_t pos_col = config_.numBasisVectors + 5;
  const size_t neg_col = config_.numBasisVectors + 6;

  // The τz components should have opposite signs
  EXPECT_NEAR(B(5, pos_col), 1.0, 1e-12) << "Positive torsion τz should be +1";
  EXPECT_NEAR(B(5, neg_col), -1.0, 1e-12) << "Negative torsion τz should be -1";

  // Equal activation of both torsional rays should cancel τz
  vector_t lambda = vector_t::Zero(basis_->numBasis());
  lambda(pos_col) = 1.0;
  lambda(neg_col) = 1.0;
  vector_t W = B * lambda;
  EXPECT_NEAR(W(5), 0.0, 1e-12) << "Equal torsional lambdas should cancel τz";
}

// ==================== Random wrench recovery ====================

TEST_F(ContactWrenchConeBasisMatrixTest, RecoverArbitraryFeasibleWrench) {
  // Generate a wrench W from a known non-negative λ, then verify B⁺ * W gives a λ'
  // such that B * λ' ≈ W (not necessarily λ = λ', since B⁺ gives min-norm solution).
  const matrix_t& B = basis_->getBasisMatrix();
  const matrix_t& B_pinv = basis_->getBasisMatrixPseudoInverse();
  const size_t numBasis = basis_->numBasis();

  // Repeat with several random λ
  for (int trial = 0; trial < 10; ++trial) {
    vector_t lambda_orig = vector_t::Random(numBasis).cwiseAbs();
    vector_t W = B * lambda_orig;

    vector_t lambda_recovered = B_pinv * W;
    vector_t W_from_recovered = B * lambda_recovered;

    EXPECT_TRUE(W_from_recovered.isApprox(W, 1e-9)) << "Trial " << trial << ": B * B⁺ * W should recover W\n"
                                                    << "  W = " << W.transpose() << "\n  W_rec = " << W_from_recovered.transpose();
  }
}

// ==================== BasisMatrix with different configs ====================

TEST_F(ContactWrenchConeBasisMatrixTest, HighFrictionIncreasesNormalInFrictionRays) {
  ContactWrenchConeConstraint::Config configHigh = makeTestConeConfig(4);
  configHigh.frictionCoefficient = 1.5;
  ContactWrenchConeBasisMatrix basisHigh(configHigh, *rect_);

  const matrix_t& B_high = basisHigh.getBasisMatrix();
  const matrix_t& B_low = basis_->getBasisMatrix();

  // With higher friction, the normal component (Fz) of friction rays should be higher
  for (size_t k = 0; k < 4; ++k) {
    EXPECT_GT(B_high(2, k), B_low(2, k)) << "Higher friction should increase Fz in friction ray " << k;
  }
}

}  // namespace
}  // namespace ocs2::humanoid
