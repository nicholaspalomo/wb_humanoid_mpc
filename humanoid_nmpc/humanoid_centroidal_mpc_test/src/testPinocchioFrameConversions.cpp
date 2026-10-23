/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include <vector>

#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/PinocchioFrameConversions.h"

namespace ocs2::humanoid {

class TestPinocchioFrameConversions : public ::testing::Test {
 protected:
  CentroidalTestingModelInterface testingModelInterface_ = CentroidalTestingModelInterface();
  PinocchioInterface pinocchioInterface_ = testingModelInterface_.getPinocchioInterface();
  const CentroidalMpcRobotModel<scalar_t>& mpcRobotModel_ = testingModelInterface_.getMpcRobotModel();
  vector_t q_ = levelConfiguration(/*hipPitchAngle=*/0.0);

  /**
   * q = [p_W, xi, q_j] of the level base (the identity quaternion) at standing height, both hip pitches at
   * `hipPitchAngle` and every other joint straight. The testing model is the G1, whose MPC joint order starts
   * left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll, right_hip_pitch, ...: the
   * hip pitches are joints 0 and 6. Both legs are rotated, because both contact frames are checked against the same
   * expected rotation (setting only the left one left the right foot at identity).
   */
  vector_t levelConfiguration(scalar_t hipPitchAngle) const {
    vector_t state = mpcRobotModel_.getNeutralState();
    mpcRobotModel_.setBasePosition(state, vector3_t(0.0, 0.0, 0.8415));
    vector_t joints = vector_t::Zero(static_cast<Eigen::Index>(mpcRobotModel_.getJointDim()));
    joints(0) = hipPitchAngle;
    joints(6) = hipPitchAngle;
    mpcRobotModel_.setJointAngles(state, joints);
    return mpcRobotModel_.getGeneralizedCoordinates(state);
  }

  /** A configuration of the base at 0.88 m height, a random attitude and random joint angles. */
  vector_t randomConfiguration() const {
    vector_t state = mpcRobotModel_.getNeutralState();
    vector3_t position = vector3_t::Random();
    position.z() = 0.88;
    mpcRobotModel_.setBasePosition(state, position);
    mpcRobotModel_.setBaseOrientation(state, vector4_t::Random());
    mpcRobotModel_.setJointAngles(state, vector_t::Random(static_cast<Eigen::Index>(mpcRobotModel_.getJointDim())));
    return mpcRobotModel_.getGeneralizedCoordinates(state);
  }
};

TEST_F(TestPinocchioFrameConversions, rotateVectorLocalToWorld3D) {
  updateFramePlacements<scalar_t>(q_, pinocchioInterface_);

  for (const pinocchio::FrameIndex& contactIndex : getContactFrameIndices(pinocchioInterface_, mpcRobotModel_)) {
    vector3_t testVector = vector3_t::Random();
    EXPECT_TRUE(testVector.isApprox(rotateVectorLocalToWorld<scalar_t>(testVector, pinocchioInterface_.getData(), contactIndex)));
  }

  // Both legs rotated by the hip pitch (levelConfiguration()).
  scalar_t hipPitchAngle = 0.5;
  q_ = levelConfiguration(hipPitchAngle);

  matrix3_t R_l_w = getRotationMatrixFromZyxEulerAngles<scalar_t>(vector3_t(0.0, hipPitchAngle, 0.0));
  updateFramePlacements<scalar_t>(q_, pinocchioInterface_);

  for (const pinocchio::FrameIndex& contactIndex : getContactFrameIndices(pinocchioInterface_, mpcRobotModel_)) {
    vector3_t testVector = vector3_t::Random();
    vector3_t testVectorRotated = R_l_w * testVector;
    EXPECT_TRUE(testVectorRotated.isApprox(rotateVectorLocalToWorld<scalar_t>(testVector, pinocchioInterface_.getData(), contactIndex)));
  }
}

TEST_F(TestPinocchioFrameConversions, rotateVectorWorldToLocal3D) {
  updateFramePlacements<scalar_t>(q_, pinocchioInterface_);

  for (const pinocchio::FrameIndex& contactIndex : getContactFrameIndices(pinocchioInterface_, mpcRobotModel_)) {
    vector3_t testVector = vector3_t::Random();
    EXPECT_TRUE(testVector.isApprox(rotateVectorWorldToLocal<scalar_t>(testVector, pinocchioInterface_.getData(), contactIndex)));
  }

  // Both legs rotated by the hip pitch (levelConfiguration()).
  scalar_t hipPitchAngle = 0.5;
  q_ = levelConfiguration(hipPitchAngle);

  matrix3_t R_l_w = getRotationMatrixFromZyxEulerAngles<scalar_t>(vector3_t(0.0, -hipPitchAngle, 0.0));
  updateFramePlacements<scalar_t>(q_, pinocchioInterface_);

  for (const pinocchio::FrameIndex& contactIndex : getContactFrameIndices(pinocchioInterface_, mpcRobotModel_)) {
    vector3_t testVector = vector3_t::Random();
    vector3_t testVectorRotated = R_l_w * testVector;
    EXPECT_TRUE(testVectorRotated.isApprox(rotateVectorWorldToLocal<scalar_t>(testVector, pinocchioInterface_.getData(), contactIndex)));
  }
}

TEST_F(TestPinocchioFrameConversions, backAndForthVector3D) {
  for (int i = 0; i < 25; ++i) {
    const vector_t q = randomConfiguration();

    updateFramePlacements<scalar_t>(q, pinocchioInterface_);

    for (const pinocchio::FrameIndex& contactIndex : getContactFrameIndices(pinocchioInterface_, mpcRobotModel_)) {
      vector3_t testVector = vector3_t::Random();
      EXPECT_TRUE(testVector.isApprox(
          rotateVectorWorldToLocal<scalar_t>(rotateVectorLocalToWorld<scalar_t>(testVector, pinocchioInterface_.getData(), contactIndex),
                                             pinocchioInterface_.getData(), contactIndex)));
    }
  }
}

TEST_F(TestPinocchioFrameConversions, rotateVectorLocalToWorld6D) {
  updateFramePlacements<scalar_t>(q_, pinocchioInterface_);

  for (const pinocchio::FrameIndex& contactIndex : getContactFrameIndices(pinocchioInterface_, mpcRobotModel_)) {
    vector6_t testVector = vector6_t::Random();
    EXPECT_TRUE(testVector.isApprox(rotateVectorLocalToWorld<scalar_t>(testVector, pinocchioInterface_.getData(), contactIndex)));
  }

  // Both legs rotated by the hip pitch (levelConfiguration()).
  scalar_t hipPitchAngle = 0.5;
  q_ = levelConfiguration(hipPitchAngle);

  matrix3_t R_l_w = getRotationMatrixFromZyxEulerAngles<scalar_t>(vector3_t(0.0, hipPitchAngle, 0.0));
  updateFramePlacements<scalar_t>(q_, pinocchioInterface_);

  for (const pinocchio::FrameIndex& contactIndex : getContactFrameIndices(pinocchioInterface_, mpcRobotModel_)) {
    vector6_t testVector = vector6_t::Random();
    vector6_t testVectorRotated(6);
    testVectorRotated << R_l_w * testVector.head(3), R_l_w * testVector.tail(3);
    EXPECT_TRUE(testVectorRotated.isApprox(rotateVectorLocalToWorld<scalar_t>(testVector, pinocchioInterface_.getData(), contactIndex)));
  }
}

TEST_F(TestPinocchioFrameConversions, rotateVectorWorldToLocal6D) {
  updateFramePlacements<scalar_t>(q_, pinocchioInterface_);

  for (const pinocchio::FrameIndex& contactIndex : getContactFrameIndices(pinocchioInterface_, mpcRobotModel_)) {
    vector6_t testVector = vector6_t::Random();
    EXPECT_TRUE(testVector.isApprox(rotateVectorWorldToLocal<scalar_t>(testVector, pinocchioInterface_.getData(), contactIndex)));
  }

  // Both legs rotated by the hip pitch (levelConfiguration()).
  scalar_t hipPitchAngle = 0.5;
  q_ = levelConfiguration(hipPitchAngle);

  matrix3_t R_l_w = getRotationMatrixFromZyxEulerAngles<scalar_t>(vector3_t(0.0, -hipPitchAngle, 0.0));
  updateFramePlacements<scalar_t>(q_, pinocchioInterface_);

  for (const pinocchio::FrameIndex& contactIndex : getContactFrameIndices(pinocchioInterface_, mpcRobotModel_)) {
    vector6_t testVector = vector6_t::Random();
    vector6_t testVectorRotated(6);
    testVectorRotated << R_l_w * testVector.head(3), R_l_w * testVector.tail(3);
    EXPECT_TRUE(testVectorRotated.isApprox(rotateVectorWorldToLocal<scalar_t>(testVector, pinocchioInterface_.getData(), contactIndex)));
  }
}

TEST_F(TestPinocchioFrameConversions, backAndForthVector6D) {
  for (int i = 0; i < 25; ++i) {
    const vector_t q = randomConfiguration();

    updateFramePlacements<scalar_t>(q, pinocchioInterface_);

    for (const pinocchio::FrameIndex& contactIndex : getContactFrameIndices(pinocchioInterface_, mpcRobotModel_)) {
      vector6_t testVector = vector6_t::Random();
      EXPECT_TRUE(testVector.isApprox(
          rotateVectorWorldToLocal<scalar_t>(rotateVectorLocalToWorld<scalar_t>(testVector, pinocchioInterface_.getData(), contactIndex),
                                             pinocchioInterface_.getData(), contactIndex)));
    }
  }
}

TEST_F(TestPinocchioFrameConversions, transformPointBackAndForth) {
  for (int i = 0; i < 25; ++i) {
    const vector_t q = randomConfiguration();

    updateFramePlacements<scalar_t>(q, pinocchioInterface_);

    for (const pinocchio::FrameIndex& contactIndex : getContactFrameIndices(pinocchioInterface_, mpcRobotModel_)) {
      vector3_t testPoint = vector3_t::Random();
      // transformPoint* carries the frame translation as well as its rotation, so the inverse of
      // transformPointLocalToWorld is transformPointWorldToLocal - not rotateVectorWorldToLocal, which drops the
      // translation and so cannot round-trip a point.
      EXPECT_TRUE(testPoint.isApprox(
          transformPointLocalToWorld<scalar_t>(transformPointWorldToLocal<scalar_t>(testPoint, pinocchioInterface_.getData(), contactIndex),
                                               pinocchioInterface_.getData(), contactIndex)));
    }
  }
}

}  // namespace ocs2::humanoid
