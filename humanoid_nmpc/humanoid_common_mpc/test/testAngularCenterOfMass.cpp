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
#include <array>
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
#include "humanoid_common_mpc/orientation/BaseOrientation.h"

namespace ocs2::humanoid {

namespace {

/// Seed for Eigen's Random, so a marginal finite-difference failure is
/// reproducible instead of flaky.
constexpr unsigned int kRandomSeed = 0;

/** A unit quaternion of the rotation by `angle` about the unit axis `axis`, as
 * coefficients (x, y, z, w). */
vector4_t rotationAbout(const vector3_t& axis, scalar_t angle) {
  return quaternionExp(vector3_t(angle * axis.normalized()));
}

/** The body rotation vector Log(a^-1 (x) b) that takes the unit quaternion a to
 * b. */
vector3_t bodyRotationBetween(const vector4_t& a, const vector4_t& b) {
  return quaternionLog(quaternionProduct(quaternionConjugate(a), b));
}

}  // namespace

class AngularCenterOfMassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::srand(kRandomSeed);
    const std::vector<std::string> jointNames(
        std::begin(acom::AcomSirenWeightsAtlas::joint_names),
        std::end(acom::AcomSirenWeightsAtlas::joint_names));
    absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
        AngularCenterOfMass::Create("atlas", jointNames);
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
  EXPECT_TRUE(offset1.isApprox(offset2, 1.0e-15))
      << "Forward pass is not deterministic.";
}

/**
 * Verifies that the zero-input configuration produces a finite output
 * (no NaN or Inf from the sin activations).
 */
TEST_F(AngularCenterOfMassTest, ZeroInputProducesFiniteOutput) {
  const size_t inputDim = acomPtr_->getInputDim();
  vector_t qJoints = vector_t::Zero(inputDim);

  vector3_t offset = acomPtr_->computeJointOrientationOffset(qJoints);
  EXPECT_TRUE(offset.allFinite())
      << "Output contains NaN or Inf at zero input.";

  matrix_t jacobian = acomPtr_->computeJointOffsetJacobian(qJoints);
  EXPECT_TRUE(jacobian.allFinite())
      << "Jacobian contains NaN or Inf at zero input.";
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
      << "Max abs error: "
      << (jacobian_analytical - jacobian_fd).cwiseAbs().maxCoeff();
}

/**
 * The ACoM attitude is the base orientation composed with the network's
 * rotation, xi_a = xi_hat (x) Exp(Delta_theta): a unit quaternion, read through
 * the safe normalization of xi (any positive scale gives the same attitude), in
 * the hemisphere of xi.
 */
TEST_F(AngularCenterOfMassTest,
       AcomAttitudeIsTheBaseComposedWithTheJointRotation) {
  const vector_t qJoints =
      vector_t::Random(static_cast<Eigen::Index>(acomPtr_->getInputDim()));
  const vector4_t xi = rotationAbout(vector3_t(0.3, -0.5, 0.8), /*angle=*/1.1);
  const vector3_t offset = acomPtr_->computeJointOrientationOffset(qJoints);
  ASSERT_GT(offset.norm(), 1.0e-3)
      << "the joints must rotate the ACoM for this to tell it from the base";

  const vector4_t attitude = acomPtr_->acomAttitude(xi, qJoints);
  EXPECT_LT((attitude - quaternionProduct(xi, quaternionExp(offset))).norm(),
            1.0e-15);
  EXPECT_NEAR(attitude.norm(), 1.0, 1.0e-15);
  EXPECT_LT((acomPtr_->acomAttitude(2.5 * xi, qJoints) - attitude).norm(),
            1.0e-15)
      << "the scale of xi must not matter";
  EXPECT_EQ(acomPtr_->acomAttitude(-xi, qJoints), -attitude)
      << "-xi is the same rotation, in the other hemisphere";
  // At the identity base the ACoM is the network's rotation itself.
  EXPECT_LT((acomPtr_->acomAttitude(vector4_t(0.0, 0.0, 0.0, 1.0), qJoints) -
             quaternionExp(offset))
                .norm(),
            1.0e-15);
}

/**
 * A rotation of the whole robot rotates the ACoM one for one, whatever the
 * joints: xi_a(q (x) xi) = q (x) xi_a(xi). This is the floating-base
 * equivariance the additive Euler rule only had to first order away from a
 * level base.
 */
TEST_F(AngularCenterOfMassTest, AWholeBodyRotationRotatesTheAcomOneForOne) {
  const vector_t qJoints =
      vector_t::Random(static_cast<Eigen::Index>(acomPtr_->getInputDim()));
  const vector4_t xi = rotationAbout(vector3_t(0.1, 0.9, -0.2), /*angle=*/0.7);
  const std::array<vector4_t, 2> rotations = {
      rotationAbout(vector3_t::UnitZ(), /*angle=*/2.9),
      rotationAbout(vector3_t(1.0, -1.0, 0.5), /*angle=*/1.3)};
  for (const vector4_t& rotation : rotations) {
    EXPECT_LT(
        (acomPtr_->acomAttitude(quaternionProduct(rotation, xi), qJoints) -
         quaternionProduct(rotation, acomPtr_->acomAttitude(xi, qJoints)))
            .norm(),
        1.0e-14);
  }
}

/**
 * The two blocks of the tangent Jacobian are the adjoint R(Delta_theta)^T on
 * the base and the right Jacobian of SO(3) times the network's joint Jacobian
 * on the joints.
 */
TEST_F(AngularCenterOfMassTest,
       TangentJacobianBlocksAreTheAdjointAndTheRightJacobian) {
  const vector_t qJoints =
      vector_t::Random(static_cast<Eigen::Index>(acomPtr_->getInputDim()));
  const vector3_t offset = acomPtr_->computeJointOrientationOffset(qJoints);
  const AcomAttitudeTangentJacobian jacobian =
      acomPtr_->acomAttitudeTangentJacobian(qJoints);
  // Bit for bit: the one-pass evaluation reproduces
  // computeJointOrientationOffset()'s operations.
  EXPECT_EQ(
      jacobian.baseOrientation,
      matrix3_t(quaternionRotationMatrix(quaternionExp(offset)).transpose()));
  ASSERT_EQ(jacobian.jointAngles.rows(), 3);
  ASSERT_EQ(jacobian.jointAngles.cols(),
            static_cast<Eigen::Index>(acomPtr_->getInputDim()));
  EXPECT_LT(
      (jacobian.jointAngles -
       so3RightJacobian(offset) * acomPtr_->computeJointOffsetJacobian(qJoints))
          .norm(),
      1.0e-14);
}

/**
 * The tangent Jacobian is the derivative of the attitude: a body rotation dphi
 * of the base moves the ACoM by R(Delta_theta)^T dphi in its own frame,
 * exactly, and a joint step by J_r J_Delta_theta dq to first order. Checked by
 * central differences of the body rotation between the perturbed and the
 * unperturbed attitude.
 */
TEST_F(AngularCenterOfMassTest,
       TangentJacobianMatchesFiniteDifferencesOfTheAttitude) {
  const Eigen::Index inputDim =
      static_cast<Eigen::Index>(acomPtr_->getInputDim());
  const vector_t qJoints = vector_t::Random(inputDim);
  const vector4_t xi = rotationAbout(vector3_t(-0.4, 0.2, 0.9), /*angle=*/2.2);
  const vector4_t attitude = acomPtr_->acomAttitude(xi, qJoints);
  const AcomAttitudeTangentJacobian jacobian =
      acomPtr_->acomAttitudeTangentJacobian(qJoints);
  constexpr scalar_t kStep = 1.0e-6;

  for (Eigen::Index axis = 0; axis < 3; ++axis) {
    const vector3_t step = kStep * vector3_t::Unit(axis);
    const vector3_t plus = bodyRotationBetween(
        attitude, acomPtr_->acomAttitude(
                      quaternionProduct(xi, quaternionExp(step)), qJoints));
    const vector3_t minus = bodyRotationBetween(
        attitude,
        acomPtr_->acomAttitude(
            quaternionProduct(xi, quaternionExp(vector3_t(-step))), qJoints));
    EXPECT_LT(
        ((plus - minus) / (2.0 * kStep) - jacobian.baseOrientation.col(axis))
            .norm(),
        1.0e-8)
        << "base axis " << axis;
  }
  const scalar_t scale =
      std::max(1.0, jacobian.jointAngles.cwiseAbs().maxCoeff());
  for (Eigen::Index joint = 0; joint < inputDim; ++joint) {
    vector_t plusJoints = qJoints;
    vector_t minusJoints = qJoints;
    plusJoints(joint) += kStep;
    minusJoints(joint) -= kStep;
    const vector3_t plus =
        bodyRotationBetween(attitude, acomPtr_->acomAttitude(xi, plusJoints));
    const vector3_t minus =
        bodyRotationBetween(attitude, acomPtr_->acomAttitude(xi, minusJoints));
    EXPECT_LT(((plus - minus) / (2.0 * kStep) - jacobian.jointAngles.col(joint))
                  .norm(),
              1.0e-5 * scale)
        << "joint " << joint;
  }
}

/** Verifies that loadWeights rejects an incorrect number of layers and leaves
 * the loaded network as it was. */
TEST_F(AngularCenterOfMassTest, LoadWeightsRejectsWrongLayerCount) {
  const size_t inputDim = acomPtr_->getInputDim();
  constexpr int kArbitraryWidth = 8;
  std::vector<SirenLayerWeights> tooFew;
  tooFew.push_back({matrix_t::Zero(kArbitraryWidth, inputDim),
                    vector_t::Zero(kArbitraryWidth)});
  const vector_t qJoints = vector_t::Constant(inputDim, 0.1);
  const vector3_t before = acomPtr_->computeJointOrientationOffset(qJoints);

  const absl::Status refused = acomPtr_->loadWeights(tooFew);
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(refused.message(), ::testing::HasSubstr("layers"));
  EXPECT_TRUE(
      acomPtr_->computeJointOrientationOffset(qJoints).isApprox(before, 0.0))
      << "a refused load changed the network";
}

// The evaluation guards are ABSL_CHECKs: a broken invariant ends the process
// with its message.
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

  EXPECT_DEATH(acomPtr_->computeJointOrientationOffset(tooShort),
               "wrong number of joint positions");
  EXPECT_DEATH(acomPtr_->computeJointOrientationOffset(tooLong),
               "wrong number of joint positions");
  EXPECT_DEATH(acomPtr_->computeJointOffsetJacobian(tooShort),
               "wrong number of joint positions");
  EXPECT_DEATH(acomPtr_->computeJointOffsetJacobian(tooLong),
               "wrong number of joint positions");
}

/** The attitude and its tangent Jacobian go through the same joint-vector guard
 * as the network. */
TEST_F(AngularCenterOfMassDeathTest, TheAttitudeRejectsAWrongJointVectorSize) {
  const vector_t tooShort =
      vector_t::Zero(static_cast<Eigen::Index>(acomPtr_->getInputDim() - 1));
  EXPECT_DEATH(acomPtr_->acomAttitude(vector4_t(0.0, 0.0, 0.0, 1.0), tooShort),
               "wrong number of joint positions");
  EXPECT_DEATH(acomPtr_->acomAttitudeTangentJacobian(tooShort),
               "wrong number of joint positions");
}

/**
 * An evaluator built through the raw constructor has no weights until
 * loadWeights is called. It must say so instead of silently returning a zero
 * offset, which in an MPC cost is indistinguishable from a zero tracking
 * weight.
 */
TEST(AngularCenterOfMassUnloadedDeathTest, EvaluatingWithoutWeightsDies) {
  const AngularCenterOfMass unloaded(/*inputDim=*/6, /*numLayers=*/2);
  EXPECT_DEATH(unloaded.computeJointOrientationOffset(vector_t::Zero(6)),
               "no weights loaded");
  EXPECT_DEATH(unloaded.computeJointOffsetJacobian(vector_t::Zero(6)),
               "no weights loaded");
}

/**
 * The trained weights must be indexed by the same joints, in the same order, as
 * the Pinocchio model the MPC builds. The generated header records that
 * ordering so it can be checked here rather than discovered as degraded
 * tracking.
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
