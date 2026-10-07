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

#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/acom/AcomSirenWeightsAtlas.h"
#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"

namespace ocs2::humanoid {

namespace {

/// Seed for Eigen's Random, so a marginal finite-difference failure is
/// reproducible instead of flaky.
constexpr unsigned int kRandomSeed = 0;

}  // namespace

class AngularCenterOfMassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::srand(kRandomSeed);
    const std::vector<std::string> jointNames(std::begin(acom::AcomSirenWeightsAtlas::joint_names),
                                              std::end(acom::AcomSirenWeightsAtlas::joint_names));
    absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom = AngularCenterOfMass::Create("atlas", jointNames);
    ASSERT_TRUE(acom.ok()) << acom.status();
    acomPtr_ = *std::move(acom);
  }

  std::unique_ptr<AngularCenterOfMass> acomPtr_;
};

/**
 * Verifies that the forward pass produces the correct output shape
 * and that calling it twice with the same input gives identical results.
 */
TEST_F(AngularCenterOfMassTest, ForwardPassDeterministic) {
  const size_t inputDim = acomPtr_->getInputDim();
  vector_t qJoints = vector_t::Random(inputDim);

  vector3_t offset1 = acomPtr_->computeJointOrientationOffset(qJoints);
  vector3_t offset2 = acomPtr_->computeJointOrientationOffset(qJoints);

  EXPECT_EQ(offset1.size(), 3);
  EXPECT_TRUE(offset1.isApprox(offset2, 1.0e-15)) << "Forward pass is not deterministic.";
}

/**
 * Verifies that the zero-input configuration produces a finite output
 * (no NaN or Inf from the sin activations).
 */
TEST_F(AngularCenterOfMassTest, ZeroInputProducesFiniteOutput) {
  const size_t inputDim = acomPtr_->getInputDim();
  vector_t qJoints = vector_t::Zero(inputDim);

  vector3_t offset = acomPtr_->computeJointOrientationOffset(qJoints);
  EXPECT_TRUE(offset.allFinite()) << "Output contains NaN or Inf at zero input.";

  matrix_t jacobian = acomPtr_->computeJointOffsetJacobian(qJoints);
  EXPECT_TRUE(jacobian.allFinite()) << "Jacobian contains NaN or Inf at zero input.";
}

/**
 * Core test: verifies the exact analytical Jacobian from the chain rule
 * matches a central finite-difference approximation of the forward pass.
 */
TEST_F(AngularCenterOfMassTest, AnalyticalJacobianMatchesFiniteDifference) {
  const size_t inputDim = acomPtr_->getInputDim();
  vector_t qJoints = vector_t::Random(inputDim);

  // Analytical Jacobian
  matrix_t jacobian_analytical = acomPtr_->computeJointOffsetJacobian(qJoints);

  EXPECT_EQ(jacobian_analytical.rows(), 3);
  EXPECT_EQ(jacobian_analytical.cols(), static_cast<int>(inputDim));

  // Finite difference test
  const scalar_t eps = 1.0e-6;
  matrix_t jacobian_fd = matrix_t::Zero(3, inputDim);

  for (size_t i = 0; i < inputDim; ++i) {
    vector_t q_plus = qJoints;
    q_plus(i) += eps;
    vector3_t offset_plus = acomPtr_->computeJointOrientationOffset(q_plus);

    vector_t q_minus = qJoints;
    q_minus(i) -= eps;
    vector3_t offset_minus = acomPtr_->computeJointOrientationOffset(q_minus);

    jacobian_fd.col(i) = (offset_plus - offset_minus) / (2.0 * eps);
  }

  EXPECT_TRUE(jacobian_analytical.isApprox(jacobian_fd, 1.0e-4))
      << "Analytical Jacobian does not match Finite Difference approximation.\n"
      << "Max abs error: " << (jacobian_analytical - jacobian_fd).cwiseAbs().maxCoeff();
}

/**
 * Verifies the full ACoM orientation has the correct equivariance property in the
 * centroidal state's ZYX Euler convention:
 * theta_aCOM(q) = euler_zyx_base + P * Delta_theta(q_joints).
 */
TEST_F(AngularCenterOfMassTest, FullAcomOrientationEquivariance) {
  const size_t inputDim = acomPtr_->getInputDim();
  // q = [pos_base(3), euler_zyx_base(3), q_joints(n_j)]
  vector_t q = vector_t::Random(6 + inputDim);

  const vector3_t eulerZyxBase = q.segment<3>(3);
  const vector3_t deltaThetaZyx = acomXyzToZyx(acomPtr_->computeJointOrientationOffset(q.tail(inputDim)));

  EXPECT_TRUE(acomPtr_->computeAcomOrientation(q).isApprox(eulerZyxBase + deltaThetaZyx, 1.0e-12))
      << "ACoM orientation does not satisfy theta_aCOM = euler_zyx_base + P * Delta_theta(q_j).";

  // Translating the base must leave the orientation untouched.
  vector_t qTranslated = q;
  qTranslated.head<3>() += vector3_t(1.0, -2.0, 0.5);
  EXPECT_TRUE(acomPtr_->computeAcomOrientation(qTranslated).isApprox(acomPtr_->computeAcomOrientation(q), 1.0e-15));
}

/**
 * Verifies the full ACoM Jacobian structure:
 * J_aCOM = [0_(3x3), I_(3x3), J_Delta_theta_(3xn_j)].
 */
TEST_F(AngularCenterOfMassTest, FullAcomJacobianStructure) {
  const size_t inputDim = acomPtr_->getInputDim();
  vector_t q = vector_t::Random(6 + inputDim);

  matrix_t J_acom = acomPtr_->computeAcomJacobian(q);
  EXPECT_EQ(J_acom.rows(), 3);
  EXPECT_EQ(J_acom.cols(), static_cast<int>(6 + inputDim));

  // Base position block must be zero: translating the base does not rotate it.
  EXPECT_TRUE((J_acom.block<3, 3>(0, 0).isZero(1.0e-15))) << "Base position block is not zero.";

  // Base orientation block must be identity, since theta_aCOM is the base Euler
  // triple plus a joint-only offset.
  EXPECT_TRUE((J_acom.block<3, 3>(0, 3).isApprox(matrix_t::Identity(3, 3), 1.0e-15))) << "Base orientation block is not identity.";

  // Joint block must equal the standalone joint Jacobian, reordered to ZYX.
  const matrix_t J_delta_zyx = acomJacobianXyzToZyx(acomPtr_->computeJointOffsetJacobian(q.tail(inputDim)));
  EXPECT_TRUE(J_acom.block(0, 6, 3, inputDim).isApprox(J_delta_zyx, 1.0e-15))
      << "Joint block of full Jacobian does not match the reordered standalone Jacobian.";
}

/** Verifies that loadWeights rejects an incorrect number of layers and leaves the loaded network as it was. */
TEST_F(AngularCenterOfMassTest, LoadWeightsRejectsWrongLayerCount) {
  const size_t inputDim = acomPtr_->getInputDim();
  constexpr int kArbitraryWidth = 8;
  std::vector<SirenLayerWeights> tooFew;
  tooFew.push_back({matrix_t::Zero(kArbitraryWidth, inputDim), vector_t::Zero(kArbitraryWidth)});
  const vector_t qJoints = vector_t::Constant(inputDim, 0.1);
  const vector3_t before = acomPtr_->computeJointOrientationOffset(qJoints);

  const absl::Status refused = acomPtr_->loadWeights(tooFew);
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(refused.message(), ::testing::HasSubstr("layers"));
  EXPECT_TRUE(acomPtr_->computeJointOrientationOffset(qJoints).isApprox(before, 0.0)) << "a refused load changed the network";
}

// The evaluation guards are ABSL_CHECKs: a broken invariant ends the process with its message.
using AngularCenterOfMassDeathTest = AngularCenterOfMassTest;

/**
 * Verifies that a joint vector of the wrong length is rejected rather than read
 * out of bounds. Eigen's own size assertions are compiled out in opt builds, so
 * this guard is the only thing standing between a stale weights header and
 * silent memory corruption inside the MPC.
 */
TEST_F(AngularCenterOfMassDeathTest, RejectsWrongJointVectorSize) {
  const size_t inputDim = acomPtr_->getInputDim();
  const vector_t tooShort = vector_t::Zero(inputDim - 1);
  const vector_t tooLong = vector_t::Zero(inputDim + 1);

  EXPECT_DEATH(acomPtr_->computeJointOrientationOffset(tooShort), "wrong number of joint positions");
  EXPECT_DEATH(acomPtr_->computeJointOrientationOffset(tooLong), "wrong number of joint positions");
  EXPECT_DEATH(acomPtr_->computeJointOffsetJacobian(tooShort), "wrong number of joint positions");
  EXPECT_DEATH(acomPtr_->computeJointOffsetJacobian(tooLong), "wrong number of joint positions");
}

/**
 * An evaluator built through the raw constructor has no weights until loadWeights
 * is called. It must say so instead of silently returning a zero offset, which
 * in an MPC cost is indistinguishable from a zero tracking weight.
 */
TEST(AngularCenterOfMassUnloadedDeathTest, EvaluatingWithoutWeightsDies) {
  const AngularCenterOfMass unloaded(/*inputDim=*/6, /*numLayers=*/2);
  EXPECT_DEATH(unloaded.computeJointOrientationOffset(vector_t::Zero(6)), "no weights loaded");
  EXPECT_DEATH(unloaded.computeJointOffsetJacobian(vector_t::Zero(6)), "no weights loaded");
}

/**
 * The trained weights must be indexed by the same joints, in the same order, as
 * the Pinocchio model the MPC builds. The generated header records that ordering
 * so it can be checked here rather than discovered as degraded tracking.
 */
TEST_F(AngularCenterOfMassTest, WeightsRecordJointOrdering) {
  const size_t inputDim = acomPtr_->getInputDim();
  ASSERT_EQ(inputDim, acom::AcomSirenWeightsAtlas::input_dim);
  for (size_t i = 0; i < inputDim; ++i) {
    EXPECT_NE(acom::AcomSirenWeightsAtlas::joint_names[i], nullptr);
  }
  // The torso chain must run parent to child, which is Pinocchio's ordering and
  // not the alphabetical order the Atlas URDF happens to list its joints in.
  EXPECT_STREQ(acom::AcomSirenWeightsAtlas::joint_names[0], "back_bkz");
  EXPECT_STREQ(acom::AcomSirenWeightsAtlas::joint_names[1], "back_bky");
  EXPECT_STREQ(acom::AcomSirenWeightsAtlas::joint_names[2], "back_bkx");
}

}  // namespace ocs2::humanoid
