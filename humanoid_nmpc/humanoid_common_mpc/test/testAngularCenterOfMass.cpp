#include <gtest/gtest.h>

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"

using namespace ocs2;
using namespace ocs2::humanoid;

class AngularCenterOfMassTest : public ::testing::Test {
 protected:
  void SetUp() override { acomPtr_ = AngularCenterOfMass::createForRobot("atlas"); }

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
  EXPECT_TRUE(offset1.isApprox(offset2, 1e-15)) << "Forward pass is not deterministic.";
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
  const scalar_t eps = 1e-6;
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

  EXPECT_TRUE(jacobian_analytical.isApprox(jacobian_fd, 1e-4))
      << "Analytical Jacobian does not match Finite Difference approximation.\n"
      << "Max abs error: " << (jacobian_analytical - jacobian_fd).cwiseAbs().maxCoeff();
}

/**
 * Verifies the full ACoM orientation has the correct equivariance property:
 * theta_aCOM(q) = rpy_base + Delta_theta(q_joints).
 */
TEST_F(AngularCenterOfMassTest, FullAcomOrientationEquivariance) {
  const size_t inputDim = acomPtr_->getInputDim();
  // q = [pos_base(3), rpy_base(3), q_joints(n_j)]
  vector_t q = vector_t::Random(6 + inputDim);

  vector3_t rpyBase = q.segment<3>(3);
  vector_t qJoints = q.tail(inputDim);

  vector3_t deltaTheta = acomPtr_->computeJointOrientationOffset(qJoints);
  vector3_t expected = rpyBase + deltaTheta;

  vector3_t actual = acomPtr_->computeAcomOrientation(q);
  EXPECT_TRUE(actual.isApprox(expected, 1e-12)) << "ACoM orientation does not satisfy theta_aCOM = rpy_base + Delta_theta(q_j).";
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

  // Base linear velocity block must be zero
  EXPECT_TRUE((J_acom.block<3, 3>(0, 0).isZero(1e-15))) << "Base linear velocity block is not zero.";

  // Base angular velocity block must be identity
  EXPECT_TRUE((J_acom.block<3, 3>(0, 3).isApprox(matrix_t::Identity(3, 3), 1e-15))) << "Base angular velocity block is not identity.";

  // Joint block must equal the standalone joint offset Jacobian
  vector_t qJoints = q.tail(inputDim);
  matrix_t J_delta = acomPtr_->computeJointOffsetJacobian(qJoints);
  EXPECT_TRUE(J_acom.block(0, 6, 3, inputDim).isApprox(J_delta, 1e-15))
      << "Joint block of full Jacobian does not match standalone Jacobian.";
}

/**
 * Verifies that setWeights rejects an incorrect number of layers.
 */
TEST_F(AngularCenterOfMassTest, SetWeightsRejectsWrongLayerCount) {
  const size_t inputDim = acomPtr_->getInputDim();
  std::vector<SirenLayerWeights> tooFew;
  tooFew.push_back({matrix_t::Zero(16, inputDim), vector_t::Zero(16)});
  EXPECT_THROW(acomPtr_->setWeights(tooFew), std::runtime_error);
}
