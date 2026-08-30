/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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
#include <random>

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"

using namespace ocs2;
using namespace ocs2::humanoid;

class AngularCenterOfMassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    inputDim = 6;
    hiddenDim = 16;
    numLayers = 2;
    omega0 = 30.0;

    acom = std::make_unique<AngularCenterOfMass>(inputDim, hiddenDim, numLayers, omega0);

    // Initialize with deterministic pseudo-random weights
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    std::vector<SirenLayerWeights> layers;
    // Layer 0: hiddenDim x inputDim
    SirenLayerWeights l0;
    l0.weight = matrix_t(hiddenDim, inputDim);
    l0.bias = vector_t(hiddenDim);
    for (int r = 0; r < l0.weight.rows(); ++r) {
      for (int c = 0; c < l0.weight.cols(); ++c) l0.weight(r, c) = dist(gen);
      l0.bias(r) = dist(gen);
    }
    layers.push_back(l0);

    // Layer 1: hiddenDim x hiddenDim
    SirenLayerWeights l1;
    l1.weight = matrix_t(hiddenDim, hiddenDim);
    l1.bias = vector_t(hiddenDim);
    for (int r = 0; r < l1.weight.rows(); ++r) {
      for (int c = 0; c < l1.weight.cols(); ++c) l1.weight(r, c) = dist(gen);
      l1.bias(r) = dist(gen);
    }
    layers.push_back(l1);

    // Output Layer: 3 x hiddenDim
    SirenLayerWeights lOut;
    lOut.weight = matrix_t(3, hiddenDim);
    lOut.bias = vector_t(3);
    for (int r = 0; r < lOut.weight.rows(); ++r) {
      for (int c = 0; c < lOut.weight.cols(); ++c) lOut.weight(r, c) = dist(gen);
      lOut.bias(r) = dist(gen);
    }
    layers.push_back(lOut);

    acom->setWeights(layers);
  }

  size_t inputDim;
  size_t hiddenDim;
  size_t numLayers;
  double omega0;
  std::unique_ptr<AngularCenterOfMass> acom;
};

TEST_F(AngularCenterOfMassTest, FloatingBaseEquivariance) {
  vector_t qJoints = vector_t::Random(inputDim);

  // Configuration with base at origin
  vector_t q1 = vector_t::Zero(6 + inputDim);
  q1.tail(inputDim) = qJoints;

  // Configuration with translated and rotated base
  vector_t q2 = vector_t::Zero(6 + inputDim);
  q2.segment<3>(0) << 1.5, -2.0, 0.8;  // Position shift
  q2.segment<3>(3) << 0.2, -0.3, 0.5;  // RPY shift
  q2.tail(inputDim) = qJoints;

  vector3_t theta1 = acom->computeAcomOrientation(q1);
  vector3_t theta2 = acom->computeAcomOrientation(q2);

  // Theta_aCOM(q2) - Theta_aCOM(q1) must equal delta_rpy_base exactly
  vector3_t deltaTheta = theta2 - theta1;
  vector3_t expectedDelta = q2.segment<3>(3) - q1.segment<3>(3);

  EXPECT_LT((deltaTheta - expectedDelta).norm(), 1e-12);
}

TEST_F(AngularCenterOfMassTest, AnalyticalJacobianVsFiniteDifference) {
  vector_t qJoints = vector_t::Random(inputDim);

  matrix_t J_ana = acom->computeJointOffsetJacobian(qJoints);

  // Finite-difference numerical Jacobian
  const double eps = 1e-7;
  matrix_t J_num = matrix_t::Zero(3, inputDim);

  for (size_t col = 0; col < inputDim; ++col) {
    vector_t q_plus = qJoints;
    vector_t q_minus = qJoints;
    q_plus(col) += eps;
    q_minus(col) -= eps;

    vector3_t f_plus = acom->computeJointOrientationOffset(q_plus);
    vector3_t f_minus = acom->computeJointOrientationOffset(q_minus);

    J_num.col(col) = (f_plus - f_minus) / (2.0 * eps);
  }

  double error = (J_ana - J_num).norm();
  EXPECT_LT(error, 1e-5);
}

TEST_F(AngularCenterOfMassTest, FullAcomJacobianStructure) {
  vector_t q = vector_t::Random(6 + inputDim);

  matrix_t J_full = acom->computeAcomJacobian(q);

  // Base linear velocity block must be 0
  matrix_t linearBlock = J_full.block(0, 0, 3, 3);
  EXPECT_LT(linearBlock.norm(), 1e-12);

  // Base angular velocity block must be Identity (3x3)
  matrix_t rotBlock = J_full.block(0, 3, 3, 3);
  EXPECT_LT((rotBlock - matrix_t::Identity(3, 3)).norm(), 1e-12);

  // Joint block must equal computeJointOffsetJacobian
  matrix_t J_joints = acom->computeJointOffsetJacobian(q.tail(inputDim));
  EXPECT_LT((J_full.block(0, 6, 3, inputDim) - J_joints).norm(), 1e-12);
}
