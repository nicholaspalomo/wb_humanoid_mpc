#include <gtest/gtest.h>

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"

using namespace ocs2;
using namespace ocs2::humanoid;

class AngularCenterOfMassTest : public ::testing::Test {
 protected:
  void SetUp() override { acomPtr_ = AngularCenterOfMass::createFromStaticWeights(); }

  std::unique_ptr<AngularCenterOfMass> acomPtr_;
};

TEST_F(AngularCenterOfMassTest, testJointOrientationOffsetJacobian) {
  if (!acomPtr_) return;

  const size_t inputDim = acomPtr_->getInputDim();
  vector_t qJoints = vector_t::Random(inputDim);

  // Analytical Jacobian
  matrix_t jacobian_analytical = acomPtr_->computeJointOrientationOffsetJacobian(qJoints);

  EXPECT_EQ(jacobian_analytical.rows(), 3);
  EXPECT_EQ(jacobian_analytical.cols(), inputDim);

  // Finite difference test
  const scalar_t eps = 1e-5;
  matrix_t jacobian_fd = matrix_t::Zero(3, inputDim);

  for (size_t i = 0; i < inputDim; ++i) {
    vector_t q_plus = qJoints;
    q_plus(i) += eps;
    vector3_t offset_plus = acomPtr_->computeAcomOrientationOffset(q_plus);

    vector_t q_minus = qJoints;
    q_minus(i) -= eps;
    vector3_t offset_minus = acomPtr_->computeAcomOrientationOffset(q_minus);

    jacobian_fd.col(i) = (offset_plus - offset_minus) / (2.0 * eps);
  }

  EXPECT_TRUE(jacobian_analytical.isApprox(jacobian_fd, 1e-3)) << "Analytical Jacobian does not match Finite Difference approximation.";
}
