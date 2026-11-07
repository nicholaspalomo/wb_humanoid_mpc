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

#include <cstddef>
#include <memory>

#include "Eigen/Core"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
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
    // State vector: [mnorm_cmom_com_rt_world_ewrt_world(6),
    // pos_base_rt_world_ewrt_world(3), quat_base_rt_world_xyzw(4),
    // joint_angles(nj)]
    testState_ = model.getNeutralState();

    // Centroidal momentum [vcom_x, vcom_y, vcom_z, L_x/m, L_y/m, L_z/m]
    testState_.segment(0, 6) << 0.1, 0.2, 0.0, 0.05, -0.03, 0.01;

    // Base pose [px, py, pz] and the ZYX Euler angles [yaw, pitch, roll] of its
    // quaternion
    model.setPoseeulerzyxBaseRtWorld(
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
      12);  // 6 per foot, two feet

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
            model.getQuatBaseRtWorldXyzw(testState_));
  EXPECT_NEAR(genCoords.segment<4>(3).norm(), 1.0, 1.0e-15);
  EXPECT_DOUBLE_EQ(genCoords[7], 0.1);  // first joint angle
}

// Test base pose extraction and setting
TEST_F(CentroidalMpcRobotModelTest, BasePose) {
  vector6_t poseeulerzyxBaseRtWorld =
      testingModelInterface_.getMpcRobotModel().getPoseeulerzyxBaseRtWorld(
          testState_);
  EXPECT_EQ(poseeulerzyxBaseRtWorld.size(), 6);
  EXPECT_DOUBLE_EQ(poseeulerzyxBaseRtWorld[2], 0.85);  // z position
  EXPECT_NEAR(poseeulerzyxBaseRtWorld[3], 0.1,
              1.0e-15);  // yaw, read back from the quaternion

  // Test setting base pose
  vector_t modifiedState = testState_;
  vector6_t newPoseeulerzyxBaseRtWorld;
  newPoseeulerzyxBaseRtWorld << 1.0, 2.0, 1.0, 0.1, 0.2, 0.3;
  testingModelInterface_.getMpcRobotModel().setPoseeulerzyxBaseRtWorld(
      modifiedState, newPoseeulerzyxBaseRtWorld);

  vector6_t retrievedPoseeulerzyxBaseRtWorld =
      testingModelInterface_.getMpcRobotModel().getPoseeulerzyxBaseRtWorld(
          modifiedState);
  EXPECT_TRUE(retrievedPoseeulerzyxBaseRtWorld.isApprox(
      newPoseeulerzyxBaseRtWorld, 1.0e-10));
}

// Test base position extraction and setting
TEST_F(CentroidalMpcRobotModelTest, BasePosition) {
  vector3_t posBaseRtWorldEwrtWorld =
      testingModelInterface_.getMpcRobotModel().getPosBaseRtWorldEwrtWorld(
          testState_);
  EXPECT_EQ(posBaseRtWorldEwrtWorld.size(), 3);
  EXPECT_DOUBLE_EQ(posBaseRtWorldEwrtWorld[2], 0.85);

  // Test setting position
  vector_t modifiedState = testState_;
  vector3_t newPosBaseRtWorldEwrtWorld(1.5, 2.5, 1.2);
  testingModelInterface_.getMpcRobotModel().setPosBaseRtWorldEwrtWorld(
      modifiedState, newPosBaseRtWorldEwrtWorld);

  vector3_t retrievedPosBaseRtWorldEwrtWorld =
      testingModelInterface_.getMpcRobotModel().getPosBaseRtWorldEwrtWorld(
          modifiedState);
  EXPECT_TRUE(retrievedPosBaseRtWorldEwrtWorld.isApprox(
      newPosBaseRtWorldEwrtWorld, 1.0e-10));
}

// Test base orientation extraction and setting
TEST_F(CentroidalMpcRobotModelTest, BaseOrientation) {
  vector3_t eulerzyxBaseRtWorld =
      testingModelInterface_.getMpcRobotModel().getEulerzyxBaseRtWorld(
          testState_);
  EXPECT_EQ(eulerzyxBaseRtWorld.size(), 3);
  EXPECT_NEAR(eulerzyxBaseRtWorld[0], 0.1,
              1.0e-15);  // yaw, read back from the quaternion

  // Test setting orientation
  vector_t modifiedState = testState_;
  vector3_t newEulerzyxBaseRtWorld(0.05, 0.1, 0.15);
  testingModelInterface_.getMpcRobotModel().setEulerzyxBaseRtWorld(
      modifiedState, newEulerzyxBaseRtWorld);

  vector3_t retrievedEulerzyxBaseRtWorld =
      testingModelInterface_.getMpcRobotModel().getEulerzyxBaseRtWorld(
          modifiedState);
  EXPECT_TRUE(
      retrievedEulerzyxBaseRtWorld.isApprox(newEulerzyxBaseRtWorld, 1.0e-10));
}

// Test COM velocity extraction
TEST_F(CentroidalMpcRobotModelTest, BaseComVelocity) {
  vector3_t linvelComRtWorldEwrtWorld =
      testingModelInterface_.getMpcRobotModel().getLinvelComRtWorldEwrtWorld(
          testState_);
  EXPECT_EQ(linvelComRtWorldEwrtWorld.size(), 3);
  EXPECT_DOUBLE_EQ(linvelComRtWorldEwrtWorld[0], 0.1);
  EXPECT_DOUBLE_EQ(linvelComRtWorldEwrtWorld[1], 0.2);
  EXPECT_DOUBLE_EQ(linvelComRtWorldEwrtWorld[2], 0.0);

  vector6_t mnormCmomComRtWorldEwrtWorld =
      testingModelInterface_.getMpcRobotModel().getMnormCmomComRtWorldEwrtWorld(
          testState_);
  EXPECT_EQ(mnormCmomComRtWorldEwrtWorld.size(), 6);
  EXPECT_DOUBLE_EQ(mnormCmomComRtWorldEwrtWorld[3],
                   0.05);  // Angular momentum component
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

  vector_t retrievedJointVelocities =
      testingModelInterface_.getMpcRobotModel().getJointVelocities(
          modifiedState, modifiedInput);
  EXPECT_TRUE(retrievedJointVelocities.isApprox(newJointVels, 1.0e-10));
}

// Test contact wrench extraction and setting
TEST_F(CentroidalMpcRobotModelTest, ContactWrench) {
  // Test left foot (contact 0)
  vector6_t wrenchLfootEwrtParam =
      testingModelInterface_.getMpcRobotModel().getWrenchContactEwrtParam(
          testInput_, /*contactIndex=*/0);
  EXPECT_EQ(wrenchLfootEwrtParam.size(), 6);
  EXPECT_DOUBLE_EQ(wrenchLfootEwrtParam[2], 400.0);  // Force in z
  EXPECT_DOUBLE_EQ(wrenchLfootEwrtParam[3], 5.0);    // Moment about x

  // Test right foot (contact 1)
  vector6_t wrenchRfootEwrtParam =
      testingModelInterface_.getMpcRobotModel().getWrenchContactEwrtParam(
          testInput_, /*contactIndex=*/1);
  EXPECT_EQ(wrenchRfootEwrtParam.size(), 6);
  EXPECT_DOUBLE_EQ(wrenchRfootEwrtParam[2], 400.0);  // Force in z
  EXPECT_DOUBLE_EQ(wrenchRfootEwrtParam[3], -5.0);   // Moment about x

  // Test setting contact wrench
  vector_t modifiedInput = testInput_;
  vector6_t newWrenchContactEwrtParam;
  newWrenchContactEwrtParam << 10.0, 20.0, 500.0, 1.0, 2.0, 3.0;
  testingModelInterface_.getMpcRobotModel().setWrenchContactEwrtParam(
      modifiedInput, newWrenchContactEwrtParam, /*contactIndex=*/0);

  vector6_t retrievedWrenchContactEwrtParam =
      testingModelInterface_.getMpcRobotModel().getWrenchContactEwrtParam(
          modifiedInput, /*contactIndex=*/0);
  EXPECT_TRUE(retrievedWrenchContactEwrtParam.isApprox(
      newWrenchContactEwrtParam, 1.0e-10));
}

// Test contact force extraction and setting
TEST_F(CentroidalMpcRobotModelTest, ContactForce) {
  vector3_t forceLfootEwrtParam =
      testingModelInterface_.getMpcRobotModel().getForceContactEwrtParam(
          testInput_, /*contactIndex=*/0);
  EXPECT_EQ(forceLfootEwrtParam.size(), 3);
  EXPECT_DOUBLE_EQ(forceLfootEwrtParam[2], 400.0);

  // Test setting contact force
  vector_t modifiedInput = testInput_;
  vector3_t newForceContactEwrtParam(50.0, 60.0, 700.0);
  testingModelInterface_.getMpcRobotModel().setForceContactEwrtParam(
      modifiedInput, newForceContactEwrtParam, /*contactIndex=*/1);

  vector3_t retrievedForceContactEwrtParam =
      testingModelInterface_.getMpcRobotModel().getForceContactEwrtParam(
          modifiedInput, /*contactIndex=*/1);
  EXPECT_TRUE(retrievedForceContactEwrtParam.isApprox(newForceContactEwrtParam,
                                                      1.0e-10));
}

// Test contact moment extraction and setting
TEST_F(CentroidalMpcRobotModelTest, ContactMoment) {
  vector3_t torqueLfootEwrtParam =
      testingModelInterface_.getMpcRobotModel().getTorqueContactEwrtParam(
          testInput_, /*contactIndex=*/0);
  EXPECT_EQ(torqueLfootEwrtParam.size(), 3);
  EXPECT_DOUBLE_EQ(torqueLfootEwrtParam[0], 5.0);
  EXPECT_DOUBLE_EQ(torqueLfootEwrtParam[1], -2.0);

  // Test setting contact moment
  vector_t modifiedInput = testInput_;
  vector3_t newTorqueContactEwrtParam(8.0, 9.0, 10.0);
  testingModelInterface_.getMpcRobotModel().setTorqueContactEwrtParam(
      modifiedInput, newTorqueContactEwrtParam, /*contactIndex=*/1);

  vector3_t retrievedTorqueContactEwrtParam =
      testingModelInterface_.getMpcRobotModel().getTorqueContactEwrtParam(
          modifiedInput, /*contactIndex=*/1);
  EXPECT_TRUE(retrievedTorqueContactEwrtParam.isApprox(
      newTorqueContactEwrtParam, 1.0e-10));
}

// Test centroidal momentum extraction
TEST_F(CentroidalMpcRobotModelTest, CentroidalMomentum) {
  vector_t mnormCmomComRtWorldEwrtWorld =
      testingModelInterface_.getMpcRobotModel().getMnormCmomComRtWorldEwrtWorld(
          testState_);
  EXPECT_EQ(mnormCmomComRtWorldEwrtWorld.size(), 6);
  EXPECT_DOUBLE_EQ(mnormCmomComRtWorldEwrtWorld[0], 0.1);   // vcom_x
  EXPECT_DOUBLE_EQ(mnormCmomComRtWorldEwrtWorld[1], 0.2);   // vcom_y
  EXPECT_DOUBLE_EQ(mnormCmomComRtWorldEwrtWorld[3], 0.05);  // L_x/m
}

// Test height adaptation
TEST_F(CentroidalMpcRobotModelTest, AdaptBasePoseHeight) {
  vector_t modifiedState = testState_;
  scalar_t errPosBaseRtWorldEwrtWorldZ = 0.1;
  scalar_t initPosBaseRtWorldEwrtWorldZ =
      modifiedState[6 + 2];  // Base z position

  testingModelInterface_.getMpcRobotModel().adaptPosBaseRtWorldEwrtWorldZ(
      modifiedState, errPosBaseRtWorldEwrtWorldZ);

  EXPECT_DOUBLE_EQ(modifiedState[6 + 2],
                   initPosBaseRtWorldEwrtWorldZ + errPosBaseRtWorldEwrtWorldZ);
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
      testingModelInterface_.getMpcRobotModel().getPoseeulerzyxBaseRtWorld(
          zeroState));
  EXPECT_NO_THROW(
      testingModelInterface_.getMpcRobotModel().getJointAngles(zeroState));
  EXPECT_NO_THROW(
      testingModelInterface_.getMpcRobotModel().getWrenchContactEwrtParam(
          zeroInput, /*contactIndex=*/0));
  EXPECT_NO_THROW(
      testingModelInterface_.getMpcRobotModel().getWrenchContactEwrtParam(
          zeroInput, /*contactIndex=*/1));

  // Test contact indices
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NO_THROW(
        testingModelInterface_.getMpcRobotModel().getWrenchContactEwrtParam(
            testInput_, i));
    EXPECT_NO_THROW(
        testingModelInterface_.getMpcRobotModel().getForceContactEwrtParam(
            testInput_, i));
    EXPECT_NO_THROW(
        testingModelInterface_.getMpcRobotModel().getTorqueContactEwrtParam(
            testInput_, i));
  }
}

}  // namespace ocs2::humanoid
