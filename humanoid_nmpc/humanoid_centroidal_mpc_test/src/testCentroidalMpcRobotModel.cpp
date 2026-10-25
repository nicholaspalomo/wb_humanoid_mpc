/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include "Eigen/Core"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

class CentroidalMpcRobotModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pinocchioInterface_ = std::make_shared<PinocchioInterface>(
        testingModelInterface_.getPinocchioInterface());

    // Setup test state and input vectors
    setupTestVectors();
  }

  void setupTestVectors() {
    const CentroidalMpcRobotModel<scalar_t>& model =
        testingModelInterface_.getMpcRobotModel();
    // State vector: [centroidal_momentum(6), base_position(3),
    // base_quaternion(4: x, y, z, w), joint_angles(nj)]
    testState_ = model.getNeutralState();

    // Centroidal momentum [vcom_x, vcom_y, vcom_z, L_x/m, L_y/m, L_z/m]
    testState_.segment(0, 6) << 0.1, 0.2, 0.0, 0.05, -0.03, 0.01;

    // Base pose [px, py, pz] and the ZYX Euler angles [yaw, pitch, roll] of its
    // quaternion
    model.setBasePoseEulerZyx(
        testState_, (vector6_t() << 0.0, 0.0, 0.85, 0.1, 0.0, 0.0).finished());

    // Joint angles
    vector_t jointAngles =
        vector_t::Random(static_cast<Eigen::Index>(model.getJointDim()));
    jointAngles.head<2>() = vector2_t(0.1, -0.2);
    model.setJointAngles(testState_, jointAngles);

    // Input vector: [contact_wrenches(12), joint_velocities(12)]
    testInput_ =
        vector_t::Zero(testingModelInterface_.getMpcRobotModel().getInputDim());

    // Left foot wrench [fx, fy, fz, mx, my, mz]
    testInput_.segment(0, 6) << 0.0, 0.0, 400.0, 5.0, -2.0, 1.0;

    // Right foot wrench [fx, fy, fz, mx, my, mz]
    testInput_.segment(6, 6) << 0.0, 0.0, 400.0, -5.0, 2.0, -1.0;

    // Joint velocities (12 joints)
    testInput_.segment(
        12, testingModelInterface_.getMpcRobotModel().getJointDim()) =
        vector_t::Random(
            testingModelInterface_.getMpcRobotModel().getJointDim());
    testInput_.segment(12, 2) = vector2_t(0.1, 0.2);
  }

  CentroidalTestingModelInterface testingModelInterface_;
  std::shared_ptr<PinocchioInterface> pinocchioInterface_;
  vector_t testState_;
  vector_t testInput_;
};

// Test constructor and basic properties
TEST_F(CentroidalMpcRobotModelTest, ConstructorAndDimensions) {
  const CentroidalMpcRobotModel<scalar_t>& model =
      testingModelInterface_.getMpcRobotModel();
  // The quaternion layout: 13 + nj stored entries on a tangent of 12 + nj;
  // Pinocchio's q has 7 + nj and v 6 + nj.
  EXPECT_EQ(model.getStateDim(), 13 + model.getJointDim());
  EXPECT_EQ(model.getStateTangentDim(), 12 + model.getJointDim());
  EXPECT_EQ(model.getConfigurationDim(), 7 + model.getJointDim());
  EXPECT_EQ(model.getVelocityDim(), 6 + model.getJointDim());
  EXPECT_EQ(model.getInputDim(), 12 + model.getJointDim());
}

// Test clone functionality
TEST_F(CentroidalMpcRobotModelTest, CloneTest) {
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> clonedModel(
      testingModelInterface_.getMpcRobotModel().clone());
  ASSERT_NE(clonedModel, nullptr);
  EXPECT_EQ(clonedModel->getStateDim(),
            testingModelInterface_.getMpcRobotModel().getStateDim());
  EXPECT_EQ(clonedModel->getInputDim(),
            testingModelInterface_.getMpcRobotModel().getInputDim());
}

// Test start indices
TEST_F(CentroidalMpcRobotModelTest, StartIndices) {
  EXPECT_EQ(testingModelInterface_.getMpcRobotModel().getBaseStartindex(), 6);
  EXPECT_EQ(
      testingModelInterface_.getMpcRobotModel().getBaseOrientationStartIndex(),
      9);
  EXPECT_EQ(testingModelInterface_.getMpcRobotModel().getJointStartindex(), 13);
  EXPECT_EQ(
      testingModelInterface_.getMpcRobotModel().getJointVelocitiesStartindex(),
      12);  // 6 * kNumContacts

  // Contact indices
  EXPECT_EQ(
      testingModelInterface_.getMpcRobotModel().getContactWrenchStartIndices(0),
      0);
  EXPECT_EQ(
      testingModelInterface_.getMpcRobotModel().getContactWrenchStartIndices(1),
      6);
  EXPECT_EQ(
      testingModelInterface_.getMpcRobotModel().getContactForceStartIndices(0),
      0);
  EXPECT_EQ(
      testingModelInterface_.getMpcRobotModel().getContactMomentStartIndices(0),
      3);
}

// Test generalized coordinates extraction
TEST_F(CentroidalMpcRobotModelTest, GeneralizedCoordinates) {
  const CentroidalMpcRobotModel<scalar_t>& model =
      testingModelInterface_.getMpcRobotModel();
  vector_t genCoords = model.getGeneralizedCoordinates(testState_);
  // q = [p_W(3), xi(4), q_j]: Pinocchio's configuration of the
  // Composite(Translation, Spherical) root.
  EXPECT_EQ(genCoords.size(), 7 + model.getJointDim());

  // Should contain the base position, the unit quaternion and the joint angles
  EXPECT_DOUBLE_EQ(genCoords[2], 0.85);  // base z position
  EXPECT_EQ(vector4_t(genCoords.segment<4>(3)),
            model.getBaseOrientation(testState_));
  EXPECT_NEAR(genCoords.segment<4>(3).norm(), 1.0, 1.0e-15);
  EXPECT_DOUBLE_EQ(genCoords[7], 0.1);  // first joint angle
}

// Test base pose extraction and setting
TEST_F(CentroidalMpcRobotModelTest, BasePose) {
  vector6_t basePose =
      testingModelInterface_.getMpcRobotModel().getBasePoseEulerZyx(testState_);
  EXPECT_EQ(basePose.size(), 6);
  EXPECT_DOUBLE_EQ(basePose[2], 0.85);     // z position
  EXPECT_NEAR(basePose[3], 0.1, 1.0e-15);  // yaw, read back from the quaternion

  // Test setting base pose
  vector_t modifiedState = testState_;
  vector6_t newBasePose;
  newBasePose << 1.0, 2.0, 1.0, 0.1, 0.2, 0.3;
  testingModelInterface_.getMpcRobotModel().setBasePoseEulerZyx(modifiedState,
                                                                newBasePose);

  vector6_t retrievedPose =
      testingModelInterface_.getMpcRobotModel().getBasePoseEulerZyx(
          modifiedState);
  EXPECT_TRUE(retrievedPose.isApprox(newBasePose, 1.0e-10));
}

// Test base position extraction and setting
TEST_F(CentroidalMpcRobotModelTest, BasePosition) {
  vector3_t basePos =
      testingModelInterface_.getMpcRobotModel().getBasePosition(testState_);
  EXPECT_EQ(basePos.size(), 3);
  EXPECT_DOUBLE_EQ(basePos[2], 0.85);

  // Test setting position
  vector_t modifiedState = testState_;
  vector3_t newPos(1.5, 2.5, 1.2);
  testingModelInterface_.getMpcRobotModel().setBasePosition(modifiedState,
                                                            newPos);

  vector3_t retrievedPos =
      testingModelInterface_.getMpcRobotModel().getBasePosition(modifiedState);
  EXPECT_TRUE(retrievedPos.isApprox(newPos, 1.0e-10));
}

// Test base orientation extraction and setting
TEST_F(CentroidalMpcRobotModelTest, BaseOrientation) {
  vector3_t baseOri =
      testingModelInterface_.getMpcRobotModel().getBaseOrientationEulerZyx(
          testState_);
  EXPECT_EQ(baseOri.size(), 3);
  EXPECT_NEAR(baseOri[0], 0.1, 1.0e-15);  // yaw, read back from the quaternion

  // Test setting orientation
  vector_t modifiedState = testState_;
  vector3_t newOri(0.05, 0.1, 0.15);
  testingModelInterface_.getMpcRobotModel().setBaseOrientationEulerZyx(
      modifiedState, newOri);

  vector3_t retrievedOri =
      testingModelInterface_.getMpcRobotModel().getBaseOrientationEulerZyx(
          modifiedState);
  EXPECT_TRUE(retrievedOri.isApprox(newOri, 1.0e-10));
}

// Test COM velocity extraction
TEST_F(CentroidalMpcRobotModelTest, BaseComVelocity) {
  vector3_t comLinVel =
      testingModelInterface_.getMpcRobotModel().getBaseComLinearVelocity(
          testState_);
  EXPECT_EQ(comLinVel.size(), 3);
  EXPECT_DOUBLE_EQ(comLinVel[0], 0.1);
  EXPECT_DOUBLE_EQ(comLinVel[1], 0.2);
  EXPECT_DOUBLE_EQ(comLinVel[2], 0.0);

  vector6_t comVel =
      testingModelInterface_.getMpcRobotModel().getBaseComVelocity(testState_);
  EXPECT_EQ(comVel.size(), 6);
  EXPECT_DOUBLE_EQ(comVel[3], 0.05);  // Angular momentum component
}

// Test joint angles extraction and setting
TEST_F(CentroidalMpcRobotModelTest, JointAngles) {
  vector_t jointAngles =
      testingModelInterface_.getMpcRobotModel().getJointAngles(testState_);
  EXPECT_DOUBLE_EQ(jointAngles[0], 0.1);
  EXPECT_DOUBLE_EQ(jointAngles[1], -0.2);

  // Test setting joint angles
  vector_t modifiedState = testState_;
  vector_t newJointAngles =
      vector_t::Random(testingModelInterface_.getMpcRobotModel().getJointDim());
  testingModelInterface_.getMpcRobotModel().setJointAngles(modifiedState,
                                                           newJointAngles);

  vector_t retrievedAngles =
      testingModelInterface_.getMpcRobotModel().getJointAngles(modifiedState);
  EXPECT_TRUE(retrievedAngles.isApprox(newJointAngles, 1.0e-10));
}

// Test joint velocities extraction and setting
TEST_F(CentroidalMpcRobotModelTest, JointVelocities) {
  vector_t jointVels =
      testingModelInterface_.getMpcRobotModel().getJointVelocities(testState_,
                                                                   testInput_);
  EXPECT_EQ(jointVels.size(),
            testingModelInterface_.getMpcRobotModel().getJointDim());
  EXPECT_DOUBLE_EQ(jointVels[0], 0.1);
  EXPECT_DOUBLE_EQ(jointVels[1], 0.2);

  // Test setting joint velocities
  vector_t modifiedState = testState_;
  vector_t modifiedInput = testInput_;
  vector_t newJointVels =
      vector_t::Random(testingModelInterface_.getMpcRobotModel().getJointDim());
  testingModelInterface_.getMpcRobotModel().setJointVelocities(
      modifiedState, modifiedInput, newJointVels);

  vector_t retrievedVels =
      testingModelInterface_.getMpcRobotModel().getJointVelocities(
          modifiedState, modifiedInput);
  EXPECT_TRUE(retrievedVels.isApprox(newJointVels, 1.0e-10));
}

// Test contact wrench extraction and setting
TEST_F(CentroidalMpcRobotModelTest, ContactWrench) {
  // Test left foot (contact 0)
  vector6_t leftWrench =
      testingModelInterface_.getMpcRobotModel().getContactWrench(
          testInput_, /*contactIndex=*/0);
  EXPECT_EQ(leftWrench.size(), 6);
  EXPECT_DOUBLE_EQ(leftWrench[2], 400.0);  // Force in z
  EXPECT_DOUBLE_EQ(leftWrench[3], 5.0);    // Moment about x

  // Test right foot (contact 1)
  vector6_t rightWrench =
      testingModelInterface_.getMpcRobotModel().getContactWrench(
          testInput_, /*contactIndex=*/1);
  EXPECT_EQ(rightWrench.size(), 6);
  EXPECT_DOUBLE_EQ(rightWrench[2], 400.0);  // Force in z
  EXPECT_DOUBLE_EQ(rightWrench[3], -5.0);   // Moment about x

  // Test setting contact wrench
  vector_t modifiedInput = testInput_;
  vector6_t newWrench;
  newWrench << 10.0, 20.0, 500.0, 1.0, 2.0, 3.0;
  testingModelInterface_.getMpcRobotModel().setContactWrench(
      modifiedInput, newWrench, /*contactIndex=*/0);

  vector6_t retrievedWrench =
      testingModelInterface_.getMpcRobotModel().getContactWrench(
          modifiedInput, /*contactIndex=*/0);
  EXPECT_TRUE(retrievedWrench.isApprox(newWrench, 1.0e-10));
}

// Test contact force extraction and setting
TEST_F(CentroidalMpcRobotModelTest, ContactForce) {
  vector3_t leftForce =
      testingModelInterface_.getMpcRobotModel().getContactForce(
          testInput_, /*contactIndex=*/0);
  EXPECT_EQ(leftForce.size(), 3);
  EXPECT_DOUBLE_EQ(leftForce[2], 400.0);

  // Test setting contact force
  vector_t modifiedInput = testInput_;
  vector3_t newForce(50.0, 60.0, 700.0);
  testingModelInterface_.getMpcRobotModel().setContactForce(
      modifiedInput, newForce, /*contactIndex=*/1);

  vector3_t retrievedForce =
      testingModelInterface_.getMpcRobotModel().getContactForce(
          modifiedInput, /*contactIndex=*/1);
  EXPECT_TRUE(retrievedForce.isApprox(newForce, 1.0e-10));
}

// Test contact moment extraction and setting
TEST_F(CentroidalMpcRobotModelTest, ContactMoment) {
  vector3_t leftMoment =
      testingModelInterface_.getMpcRobotModel().getContactMoment(
          testInput_, /*contactIndex=*/0);
  EXPECT_EQ(leftMoment.size(), 3);
  EXPECT_DOUBLE_EQ(leftMoment[0], 5.0);
  EXPECT_DOUBLE_EQ(leftMoment[1], -2.0);

  // Test setting contact moment
  vector_t modifiedInput = testInput_;
  vector3_t newMoment(8.0, 9.0, 10.0);
  testingModelInterface_.getMpcRobotModel().setContactMoment(
      modifiedInput, newMoment, /*contactIndex=*/1);

  vector3_t retrievedMoment =
      testingModelInterface_.getMpcRobotModel().getContactMoment(
          modifiedInput, /*contactIndex=*/1);
  EXPECT_TRUE(retrievedMoment.isApprox(newMoment, 1.0e-10));
}

// Test centroidal momentum extraction
TEST_F(CentroidalMpcRobotModelTest, CentroidalMomentum) {
  vector_t centroidalMomentum =
      testingModelInterface_.getMpcRobotModel().getCentroidalMomentum(
          testState_);
  EXPECT_EQ(centroidalMomentum.size(), 6);
  EXPECT_DOUBLE_EQ(centroidalMomentum[0], 0.1);   // vcom_x
  EXPECT_DOUBLE_EQ(centroidalMomentum[1], 0.2);   // vcom_y
  EXPECT_DOUBLE_EQ(centroidalMomentum[3], 0.05);  // L_x/m
}

// Test height adaptation
TEST_F(CentroidalMpcRobotModelTest, AdaptBasePoseHeight) {
  vector_t modifiedState = testState_;
  scalar_t heightChange = 0.1;
  scalar_t originalHeight = modifiedState[6 + 2];  // Base z position

  testingModelInterface_.getMpcRobotModel().adaptBasePoseHeight(modifiedState,
                                                                heightChange);

  EXPECT_DOUBLE_EQ(modifiedState[6 + 2], originalHeight + heightChange);
}

// Test generalized coordinates setting
TEST_F(CentroidalMpcRobotModelTest, SetGeneralizedCoordinates) {
  vector_t modifiedState = testState_;
  // 3 base position + 4 unit quaternion + nj joints: the quaternion is read
  // back safely normalized, so it is written unit.
  vector_t newGenCoords = vector_t::Random(
      testingModelInterface_.getMpcRobotModel().getConfigurationDim());
  newGenCoords.segment<4>(3).normalize();

  testingModelInterface_.getMpcRobotModel().setGeneralizedCoordinates(
      modifiedState, newGenCoords);
  vector_t retrievedGenCoords =
      testingModelInterface_.getMpcRobotModel().getGeneralizedCoordinates(
          modifiedState);

  EXPECT_TRUE(retrievedGenCoords.isApprox(newGenCoords, 1.0e-10));
}

// Test generalized coordinates setting
TEST_F(CentroidalMpcRobotModelTest, testGeneralizedVelocities) {
  vector_t qd =
      testingModelInterface_.getMpcRobotModel().getGeneralizedVelocities(
          testState_, testInput_);
  vector_t qd_j =
      qd.tail(testingModelInterface_.getMpcRobotModel().getJointDim());
  EXPECT_TRUE(qd_j.isApprox(testInput_.tail(qd_j.size()), 1.0e-10));
}

// Test boundary conditions and error handling
TEST_F(CentroidalMpcRobotModelTest, BoundaryConditions) {
  // Test with zero vectors
  vector_t zeroState =
      vector_t::Zero(testingModelInterface_.getMpcRobotModel().getStateDim());
  vector_t zeroInput =
      vector_t::Zero(testingModelInterface_.getMpcRobotModel().getInputDim());

  EXPECT_NO_THROW(
      testingModelInterface_.getMpcRobotModel().getBasePoseEulerZyx(zeroState));
  EXPECT_NO_THROW(
      testingModelInterface_.getMpcRobotModel().getJointAngles(zeroState));
  EXPECT_NO_THROW(testingModelInterface_.getMpcRobotModel().getContactWrench(
      zeroInput, /*contactIndex=*/0));
  EXPECT_NO_THROW(testingModelInterface_.getMpcRobotModel().getContactWrench(
      zeroInput, /*contactIndex=*/1));

  // Test contact indices
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NO_THROW(testingModelInterface_.getMpcRobotModel().getContactWrench(
        testInput_, i));
    EXPECT_NO_THROW(testingModelInterface_.getMpcRobotModel().getContactForce(
        testInput_, i));
    EXPECT_NO_THROW(testingModelInterface_.getMpcRobotModel().getContactMoment(
        testInput_, i));
  }
}

}  // namespace ocs2::humanoid
