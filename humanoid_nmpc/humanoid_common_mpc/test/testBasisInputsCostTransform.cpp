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
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"

/**
 * Tests of the basis-space input cost R_basis = Mᵀ R M + reg · blkdiag(S, 0) and of its two regularizations.
 *
 * The property that matters for null_space is that it leaves the wrench-space cost exactly as the task file writes
 * it: the cost of any input is the R cost of the wrench it produces plus a term that vanishes on every λ in range(Mᵀ),
 * and the metric the optimizer sees on the wrench is R itself. full_diagonal does not have it, which is the point.
 */
namespace ocs2::humanoid {
namespace {

constexpr size_t kJointDim = 5;
constexpr scalar_t kRegularization = 1.0e-4;

/** The DRC Atlas contact weights of R (force x, y, z, moment x, y, z), which are of the order of the regularization. */
vector_t atlasLikeContactWeights() {
  return (vector_t(6) << 1.0e-5, 2.0e-5, 5.0e-5, 2.0e-5, 1.0e-5, 2.0e-4).finished();
}

feet_array_t<matrix_t> makeBases(absl::string_view generatorSet) {
  ContactWrenchConeConstraint::Config config;
  config.numBasisVectors = 4;
  config.frictionCoefficient = 0.5;
  config.torsionalFrictionCoefficient = 0.05;
  const ContactRectangle rectangle(PolygonBounds(-0.12, 0.12, -0.055, 0.055), ContactCenterPoint("foot", "ankle", vector3_t::Zero()));
  const absl::StatusOr<ContactWrenchConeBasisMatrix> basis = ContactWrenchConeBasisMatrix::Create(config, rectangle, generatorSet);
  EXPECT_TRUE(basis.ok()) << basis.status();
  return {basis->getBasisMatrix(), basis->getBasisMatrix()};
}

/** M = blkdiag(B_0, B_1, I_joints), as BasisInputsModelDecorator::getLocalBasisToWrenchMap builds it. */
matrix_t makeBasisToWrenchMap(const feet_array_t<matrix_t>& bases) {
  const Eigen::Index numBasisPerFoot = bases[0].cols();
  const Eigen::Index numContacts = static_cast<Eigen::Index>(bases.size());
  const Eigen::Index jointDim = static_cast<Eigen::Index>(kJointDim);
  matrix_t M = matrix_t::Zero(6 * numContacts + jointDim, numBasisPerFoot * numContacts + jointDim);
  for (Eigen::Index i = 0; i < numContacts; ++i) {
    M.block(6 * i, numBasisPerFoot * i, 6, numBasisPerFoot) = bases[static_cast<size_t>(i)];
  }
  M.bottomRightCorner(jointDim, jointDim).setIdentity();
  return M;
}

/** A block-diagonal wrench-space R: the given weights on every contact wrench and 1e-3 on the joint velocities. */
matrix_t makeWrenchSpaceR(const vector_t& contactWeights) {
  vector_t diagonal(6 * kNumContacts + kJointDim);
  for (size_t i = 0; i < kNumContacts; ++i) {
    diagonal.segment(static_cast<Eigen::Index>(6 * i), 6) = contactWeights;
  }
  diagonal.tail(kJointDim).setConstant(1.0e-3);
  return diagonal.asDiagonal();
}

BasisInputsCostTransformConfig makeConfig(const matrix_t& M, absl::string_view regularization, scalar_t weight = kRegularization) {
  BasisInputsCostTransformConfig config;
  config.basisToWrenchMap = M;
  config.wrenchInputDim = static_cast<size_t>(M.rows());
  config.numBasisInputs = static_cast<size_t>(M.cols()) - kJointDim;
  config.lambdaRegularization = weight;
  config.regularization = std::string(regularization);
  return config;
}

std::vector<std::string> allGeneratorSets() {
  return {std::string(kConservativeInnerApproximationGeneratorSet), std::string(kExactWrenchConeGeneratorSet)};
}

// ==================== The regularization registry ====================

TEST(BasisRegularizationRegistryTest, ListsBothRegularizationsAndKeepsTheShippedDefault) {
  const std::vector<std::string> names = basisRegularizationNames();
  for (const absl::string_view name : {kFullDiagonalBasisRegularization, kNullSpaceBasisRegularization}) {
    EXPECT_NE(std::find(names.begin(), names.end(), std::string(name)), names.end()) << name;
    EXPECT_TRUE(getBasisRegularizationBuilder(name).ok()) << name;
  }
  EXPECT_EQ(names.size(), 2u);
  // null_space changes the closed loop, so it stays opt-in: the default, and a default-constructed config, keep the
  // regularization that has always shipped.
  EXPECT_EQ(kDefaultBasisRegularization, kFullDiagonalBasisRegularization);
  EXPECT_EQ(BasisInputsCostTransformConfig().regularization, kFullDiagonalBasisRegularization);
}

TEST(BasisRegularizationRegistryTest, RejectsUnknownNamesAndNegativeWeightsNamingTheKey) {
  const absl::Status unknown = getBasisRegularizationBuilder("nullspace").status();
  EXPECT_EQ(unknown.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(unknown.message().find(kBasisRegularizationField), absl::string_view::npos) << unknown;
  for (const std::string& name : basisRegularizationNames()) {
    EXPECT_NE(unknown.message().find(name), absl::string_view::npos) << unknown << " does not list " << name;
  }

  const matrix_t M = makeBasisToWrenchMap(makeBases(kConservativeInnerApproximationGeneratorSet));
  EXPECT_TRUE(validateBasisInputsCostTransformConfig(makeConfig(M, kNullSpaceBasisRegularization)).ok());
  const absl::Status badName = validateBasisInputsCostTransformConfig(makeConfig(M, "diagonal"));
  EXPECT_NE(badName.message().find(kBasisRegularizationField), absl::string_view::npos) << badName;
  const absl::Status negative = validateBasisInputsCostTransformConfig(makeConfig(M, kFullDiagonalBasisRegularization, /*weight=*/-1.0e-4));
  EXPECT_EQ(negative.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(negative.message().find(kBasisScalingRegularizationField), absl::string_view::npos) << negative;
}

// ==================== full_diagonal: the shipped behavior, unchanged ====================

TEST(BasisInputsCostTransformTest, FullDiagonalConfigIsTheLegacyTransform) {
  for (const std::string& set : allGeneratorSets()) {
    const matrix_t M = makeBasisToWrenchMap(makeBases(set));
    const matrix_t R = makeWrenchSpaceR(atlasLikeContactWeights());
    const BasisInputsCostTransformConfig config = makeConfig(M, kFullDiagonalBasisRegularization);
    const matrix_t viaConfig = transformWrenchInputCostToBasisSpace(R, config);
    const matrix_t legacy = transformWrenchInputCostToBasisSpace(R, M, config.numBasisInputs, kRegularization);
    EXPECT_TRUE((viaConfig - legacy).isZero(0.0)) << set;
    matrix_t expected = M.transpose() * R * M;
    expected.diagonal().head(static_cast<Eigen::Index>(config.numBasisInputs)).array() += kRegularization;
    EXPECT_LE((viaConfig - expected).cwiseAbs().maxCoeff(), 1.0e-18) << set;
  }
}

// The argument checks were assert()s, which every -c opt build compiles out; they are ABSL_CHECKs now.
TEST(BasisInputsCostTransformDeathTest, MismatchedSizesOrANegativeWeightEndTheProcess) {
  const matrix_t M = makeBasisToWrenchMap(makeBases(allGeneratorSets().front()));
  const matrix_t R = makeWrenchSpaceR(atlasLikeContactWeights());
  const size_t numBasisInputs = static_cast<size_t>(M.cols());
  EXPECT_DEATH(transformWrenchInputCostToBasisSpace(matrix_t::Identity(3, 3), M, numBasisInputs, kRegularization),
               "R_wrench is not square");
  EXPECT_DEATH(transformWrenchInputCostToBasisSpace(R, M, numBasisInputs + 1, kRegularization), "more basis inputs than M has columns");
  EXPECT_DEATH(transformWrenchInputCostToBasisSpace(R, M, numBasisInputs, /*lambdaRegularization=*/-1.0), "must be non-negative");
}

TEST(BasisInputsCostTransformTest, ZeroWeightIsAnExactNoOpForEveryRegularization) {
  for (const std::string& set : allGeneratorSets()) {
    const matrix_t M = makeBasisToWrenchMap(makeBases(set));
    const matrix_t R = makeWrenchSpaceR(atlasLikeContactWeights());
    for (const std::string& name : basisRegularizationNames()) {
      const matrix_t R_basis = transformWrenchInputCostToBasisSpace(R, makeConfig(M, name, /*weight=*/0.0));
      EXPECT_TRUE((R_basis - M.transpose() * R * M).isZero(0.0)) << set << " " << name;
    }
  }
}

TEST(BasisInputsCostTransformTest, JointVelocityBlockIsUntouchedByEveryRegularization) {
  const matrix_t M = makeBasisToWrenchMap(makeBases(kExactWrenchConeGeneratorSet));
  const matrix_t R = makeWrenchSpaceR(atlasLikeContactWeights());
  for (const std::string& name : basisRegularizationNames()) {
    const matrix_t R_basis = transformWrenchInputCostToBasisSpace(R, makeConfig(M, name));
    EXPECT_LE((R_basis.bottomRightCorner(kJointDim, kJointDim) - R.bottomRightCorner(kJointDim, kJointDim)).cwiseAbs().maxCoeff(), 1.0e-18)
        << name;
  }
}

// ==================== null_space: R acts unchanged on every wrench ====================

TEST(BasisInputsCostTransformTest, NullSpaceShapeIsTheBlockDiagonalProjectorOfEachFoot) {
  for (const std::string& set : allGeneratorSets()) {
    const feet_array_t<matrix_t> bases = makeBases(set);
    const matrix_t M = makeBasisToWrenchMap(bases);
    const size_t numBasisInputs = static_cast<size_t>(M.cols()) - kJointDim;
    const matrix_t S = nullSpaceRegularizationShape(M, numBasisInputs);
    // Cross-check against the projector ContactWrenchConeBasisMatrix computes from its own pseudoinverse.
    ContactWrenchConeConstraint::Config config;
    config.numBasisVectors = 4;
    config.frictionCoefficient = 0.5;
    config.torsionalFrictionCoefficient = 0.05;
    const ContactRectangle rectangle(PolygonBounds(-0.12, 0.12, -0.055, 0.055), ContactCenterPoint("foot", "ankle", vector3_t::Zero()));
    const absl::StatusOr<ContactWrenchConeBasisMatrix> basis = ContactWrenchConeBasisMatrix::Create(config, rectangle, set);
    ASSERT_TRUE(basis.ok());
    const Eigen::Index n = static_cast<Eigen::Index>(basis->numBasis());
    matrix_t expected = matrix_t::Zero(2 * n, 2 * n);
    expected.topLeftCorner(n, n) = basis->getNullSpaceProjector();
    expected.bottomRightCorner(n, n) = basis->getNullSpaceProjector();
    EXPECT_LE((S - expected).cwiseAbs().maxCoeff(), 1.0e-10) << set;
  }
}

TEST(BasisInputsCostTransformTest, NullSpaceCostIsTheWrenchCostPlusATermThatVanishesOnRangeOfMTranspose) {
  std::mt19937 gen(7);
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  for (const std::string& set : allGeneratorSets()) {
    const matrix_t M = makeBasisToWrenchMap(makeBases(set));
    const matrix_t R = makeWrenchSpaceR(atlasLikeContactWeights());
    const BasisInputsCostTransformConfig config = makeConfig(M, kNullSpaceBasisRegularization);
    const matrix_t R_basis = transformWrenchInputCostToBasisSpace(R, config);
    const Eigen::Index n = static_cast<Eigen::Index>(config.numBasisInputs);
    const matrix_t S = nullSpaceRegularizationShape(M, config.numBasisInputs);

    for (int trial = 0; trial < 50; ++trial) {
      vector_t u(M.cols());
      for (Eigen::Index i = 0; i < u.size(); ++i) u(i) = 100.0 * normal(gen);
      const vector_t wrench = M * u;
      const scalar_t wrenchCost = wrench.dot(R * wrench);
      const vector_t lambda = u.head(n);
      // uᵀ R_basis u = (M u)ᵀ R (M u) + reg ‖P λ‖², for every input.
      EXPECT_NEAR(u.dot(R_basis * u), wrenchCost + kRegularization * lambda.dot(S * lambda), 1.0e-10 * (1.0 + wrenchCost)) << set;

      // On λ = Mᵀ y (the range of Mᵀ, where the minimum-norm λ of every wrench lives) the regularization vanishes and
      // the cost is exactly the wrench-space cost.
      vector_t y(M.rows());
      for (Eigen::Index i = 0; i < y.size(); ++i) y(i) = normal(gen);
      const vector_t inRange = M.transpose() * y;
      const vector_t inRangeWrench = M * inRange;
      const scalar_t inRangeWrenchCost = inRangeWrench.dot(R * inRangeWrench);
      EXPECT_NEAR(inRange.dot(R_basis * inRange), inRangeWrenchCost, 1.0e-10 * inRangeWrenchCost) << set;
    }
  }
}

TEST(BasisInputsCostTransformTest, NullSpaceInducesExactlyRAsTheWrenchMetricAndFullDiagonalDoesNot) {
  // Minimizing uᵀ R_basis u over the inputs that produce a given wrench W leaves Wᵀ G W, with G = (M H⁻¹ Mᵀ)⁻¹ and H
  // the transformed R. null_space must give G = R exactly; full_diagonal gives R + reg (B Bᵀ)⁻¹ on the contact block,
  // which is what made R's moment and tangential weights nearly irrelevant on Atlas.
  for (const std::string& set : allGeneratorSets()) {
    const feet_array_t<matrix_t> bases = makeBases(set);
    const matrix_t M = makeBasisToWrenchMap(bases);
    const matrix_t R = makeWrenchSpaceR(atlasLikeContactWeights());

    const matrix_t H_null = transformWrenchInputCostToBasisSpace(R, makeConfig(M, kNullSpaceBasisRegularization));
    const matrix_t G_null = (M * H_null.inverse() * M.transpose()).inverse();
    EXPECT_LE(((G_null - R).cwiseAbs().array() / R.diagonal().maxCoeff()).maxCoeff(), 1.0e-6) << set << ":\n"
                                                                                              << G_null.diagonal().transpose();

    const matrix_t H_full = transformWrenchInputCostToBasisSpace(R, makeConfig(M, kFullDiagonalBasisRegularization));
    const matrix_t G_full = (M * H_full.inverse() * M.transpose()).inverse();
    const matrix_t& B = bases[0];
    const matrix_t expectedContactMetric = R.topLeftCorner(6, 6) + kRegularization * (B * B.transpose()).inverse();
    EXPECT_LE(
        ((G_full.topLeftCorner(6, 6) - expectedContactMetric).cwiseAbs().array() / expectedContactMetric.cwiseAbs().maxCoeff()).maxCoeff(),
        1.0e-6)
        << set;
    EXPECT_GT((G_full.topLeftCorner(6, 6) - R.topLeftCorner(6, 6)).cwiseAbs().maxCoeff(), R.topLeftCorner(6, 6).maxCoeff())
        << set << ": full_diagonal is expected to distort the wrench metric by more than R's largest weight";
  }
}

TEST(BasisInputsCostTransformTest, LambdaBlockIsPositiveDefiniteExactlyWhenItShouldBe) {
  for (const std::string& set : allGeneratorSets()) {
    const matrix_t M = makeBasisToWrenchMap(makeBases(set));
    const size_t numBasisInputs = static_cast<size_t>(M.cols()) - kJointDim;
    const matrix_t R = makeWrenchSpaceR(atlasLikeContactWeights());
    for (const std::string& name : basisRegularizationNames()) {
      EXPECT_TRUE(checkLambdaBlockPositiveDefinite(transformWrenchInputCostToBasisSpace(R, makeConfig(M, name)), numBasisInputs).ok())
          << set << " " << name;
      // Without a regularization weight the λ block is singular: there are more generators than wrench components.
      const absl::Status unregularized =
          checkLambdaBlockPositiveDefinite(transformWrenchInputCostToBasisSpace(R, makeConfig(M, name, /*weight=*/0.0)), numBasisInputs);
      EXPECT_EQ(unregularized.code(), absl::StatusCode::kInvalidArgument) << set << " " << name;
      EXPECT_NE(unregularized.message().find(kBasisScalingRegularizationField), absl::string_view::npos) << unregularized;
    }
    // null_space penalizes only null(B), so a wrench direction R gives no weight is left unpenalized; full_diagonal
    // still covers it. The message names what to change.
    vector_t noTorsionWeight = atlasLikeContactWeights();
    noTorsionWeight(5) = 0.0;
    const matrix_t R_singular = makeWrenchSpaceR(noTorsionWeight);
    const absl::Status nullSpace = checkLambdaBlockPositiveDefinite(
        transformWrenchInputCostToBasisSpace(R_singular, makeConfig(M, kNullSpaceBasisRegularization)), numBasisInputs);
    EXPECT_FALSE(nullSpace.ok()) << set;
    EXPECT_NE(nullSpace.message().find(kBasisRegularizationField), absl::string_view::npos) << nullSpace;
    EXPECT_TRUE(checkLambdaBlockPositiveDefinite(
                    transformWrenchInputCostToBasisSpace(R_singular, makeConfig(M, kFullDiagonalBasisRegularization)), numBasisInputs)
                    .ok())
        << set;
  }
}

}  // namespace
}  // namespace ocs2::humanoid
