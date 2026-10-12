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

#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"

namespace ocs2::humanoid {

class AngularCenterOfMassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    inputDim_ = 6;
    hiddenDim_ = 16;
    numLayers_ = 2;
    omega0_ = 30.0;

    acom_ = std::make_unique<AngularCenterOfMass>(inputDim_, numLayers_, omega0_);

    // Initialize with deterministic pseudo-random weights
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    std::vector<SirenLayerWeights> layers;
    // Layer 0: hiddenDim x inputDim
    SirenLayerWeights l0;
    l0.weight = matrix_t(hiddenDim_, inputDim_);
    l0.bias = vector_t(hiddenDim_);
    for (int r = 0; r < l0.weight.rows(); ++r) {
      for (int c = 0; c < l0.weight.cols(); ++c) l0.weight(r, c) = dist(gen);
      l0.bias(r) = dist(gen);
    }
    layers.push_back(l0);

    // Layer 1: hiddenDim x hiddenDim
    SirenLayerWeights l1;
    l1.weight = matrix_t(hiddenDim_, hiddenDim_);
    l1.bias = vector_t(hiddenDim_);
    for (int r = 0; r < l1.weight.rows(); ++r) {
      for (int c = 0; c < l1.weight.cols(); ++c) l1.weight(r, c) = dist(gen);
      l1.bias(r) = dist(gen);
    }
    layers.push_back(l1);

    // Output Layer: 3 x hiddenDim
    SirenLayerWeights lOut;
    lOut.weight = matrix_t(3, hiddenDim_);
    lOut.bias = vector_t(3);
    for (int r = 0; r < lOut.weight.rows(); ++r) {
      for (int c = 0; c < lOut.weight.cols(); ++c) lOut.weight(r, c) = dist(gen);
      lOut.bias(r) = dist(gen);
    }
    layers.push_back(lOut);

    ASSERT_TRUE(acom_->loadWeights(layers).ok());
  }

  size_t inputDim_ = 0;
  size_t hiddenDim_ = 0;  // Width of the synthetic hidden layers built in SetUp.
  size_t numLayers_ = 0;
  double omega0_ = 0.0;
  std::unique_ptr<AngularCenterOfMass> acom_;
};

TEST_F(AngularCenterOfMassTest, FloatingBaseEquivariance) {
  const vector_t qJoints = vector_t::Random(static_cast<Eigen::Index>(inputDim_));

  // The ACoM attitude of a base at the identity, and of the same joints on a turned and tilted base: the joint rotation
  // composes on the right of the base, so rotating the whole robot rotates the ACoM one for one.
  const vector4_t identity(0.0, 0.0, 0.0, 1.0);
  const vector4_t baseRotation = quaternionExp(vector3_t(0.2, -0.3, 0.5));
  const vector4_t atIdentity = acom_->acomAttitude(identity, qJoints);
  const vector4_t rotated = acom_->acomAttitude(baseRotation, qJoints);

  EXPECT_LT((atIdentity - quaternionExp(acom_->computeJointOrientationOffset(qJoints))).norm(), 1.0e-15);
  EXPECT_LT((rotated - quaternionProduct(baseRotation, atIdentity)).norm(), 1.0e-15);
  EXPECT_NEAR(rotated.norm(), 1.0, 1.0e-15);
}

TEST_F(AngularCenterOfMassTest, AnalyticalJacobianVsFiniteDifference) {
  vector_t qJoints = vector_t::Random(inputDim_);

  matrix_t J_ana = acom_->computeJointOffsetJacobian(qJoints);

  // Finite-difference numerical Jacobian
  const double eps = 1.0e-7;
  matrix_t J_num = matrix_t::Zero(3, inputDim_);

  for (size_t col = 0; col < inputDim_; ++col) {
    vector_t q_plus = qJoints;
    vector_t q_minus = qJoints;
    q_plus(col) += eps;
    q_minus(col) -= eps;

    vector3_t f_plus = acom_->computeJointOrientationOffset(q_plus);
    vector3_t f_minus = acom_->computeJointOrientationOffset(q_minus);

    J_num.col(col) = (f_plus - f_minus) / (2.0 * eps);
  }

  double error = (J_ana - J_num).norm();
  EXPECT_LT(error, 1.0e-5);
}

TEST_F(AngularCenterOfMassTest, AttitudeTangentJacobianStructure) {
  const vector_t qJoints = vector_t::Random(static_cast<Eigen::Index>(inputDim_));
  const vector3_t offset = acom_->computeJointOrientationOffset(qJoints);
  const AcomAttitudeTangentJacobian jacobian = acom_->acomAttitudeTangentJacobian(qJoints);

  // The base block is the adjoint R(Delta_theta)^T, a rotation: a base perturbation is carried into the ACoM frame.
  EXPECT_LT((jacobian.baseOrientation - quaternionRotationMatrix(quaternionExp(offset)).transpose()).norm(), 1.0e-12);
  EXPECT_LT((jacobian.baseOrientation.transpose() * jacobian.baseOrientation - matrix3_t::Identity()).norm(), 1.0e-12);

  // The joint block is the right Jacobian of SO(3) at Delta_theta times the network's joint Jacobian.
  ASSERT_EQ(jacobian.jointAngles.cols(), static_cast<Eigen::Index>(inputDim_));
  EXPECT_LT((jacobian.jointAngles - so3RightJacobian(offset) * acom_->computeJointOffsetJacobian(qJoints)).norm(), 1.0e-12);

  // And it is the derivative of the attitude: central differences of the body rotation a joint step makes.
  const vector4_t base = quaternionExp(vector3_t(-0.4, 0.1, 1.2));
  const vector4_t attitude = acom_->acomAttitude(base, qJoints);
  const double eps = 1.0e-7;
  for (Eigen::Index joint = 0; joint < static_cast<Eigen::Index>(inputDim_); ++joint) {
    vector_t plus = qJoints;
    vector_t minus = qJoints;
    plus(joint) += eps;
    minus(joint) -= eps;
    const vector3_t forward = quaternionLog(quaternionProduct(quaternionConjugate(attitude), acom_->acomAttitude(base, plus)));
    const vector3_t backward = quaternionLog(quaternionProduct(quaternionConjugate(attitude), acom_->acomAttitude(base, minus)));
    EXPECT_LT(((forward - backward) / (2.0 * eps) - jacobian.jointAngles.col(joint)).norm(), 1.0e-5) << "joint " << joint;
  }
}

}  // namespace ocs2::humanoid
