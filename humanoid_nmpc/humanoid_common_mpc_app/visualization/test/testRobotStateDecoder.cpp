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

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include "absl/status/status.h"

#include "VisualizationTestRobot.h"
#include "humanoid_common_mpc_app/visualization/EulerAngles.h"
#include "humanoid_common_mpc_app/visualization/RobotStateDecoder.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.pb.h"

namespace ocs2::humanoid::visualization {
namespace {

constexpr scalar_t kTolerance = 1e-12;

TEST(EulerAnglesTest, TheAnglesOfARotationComeBack) {
  for (const vector3_t& eulerAnglesZyx : {vector3_t(0.3, 0.2, 0.1), vector3_t(-2.5, -1.2, 3.0), vector3_t(1.0, 0.0, -0.7)}) {
    const matrix3_t rotation = getRotationMatrixFromZyxEulerAngles<scalar_t>(eulerAnglesZyx);
    EXPECT_TRUE(eulerAnglesZyxFromRotation(rotation).isApprox(eulerAnglesZyx, kTolerance)) << eulerAnglesZyxFromRotation(rotation);
    // The plots' order is (roll, pitch, yaw); PinocchioTelemetryPublisher's quaternionToEulerZYX returned that order
    // under the other's name.
    EXPECT_TRUE(
        rollPitchYawFromRotation(rotation).isApprox(vector3_t(eulerAnglesZyx(2), eulerAnglesZyx(1), eulerAnglesZyx(0)), kTolerance));
  }
}

TEST(EulerAnglesTest, AYawAloneIsAYaw) {
  const matrix3_t rotation = test::quaternionFromRollPitchYaw(vector3_t(0.0, 0.0, 0.8)).toRotationMatrix();
  EXPECT_TRUE(rollPitchYawFromRotation(rotation).isApprox(vector3_t(0.0, 0.0, 0.8), kTolerance));
}

TEST(EulerAnglesTest, ARotationAtTheGimbalLockIsFinite) {
  matrix3_t rotation = getRotationMatrixFromZyxEulerAngles<scalar_t>(vector3_t(0.4, M_PI / 2.0, 0.1));
  rotation(2, 0) = -1.0 - 1e-12;  // rounding past the domain of asin()
  EXPECT_TRUE(eulerAnglesZyxFromRotation(rotation).allFinite());
  EXPECT_NEAR(eulerAnglesZyxFromRotation(rotation)(1), M_PI / 2.0, 1e-9);
}

class RobotStateDecoderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { robot_ = test::TestRobot::load(test::g1CentroidalFiles(), test::Formulation::kCentroidal).release(); }
  static void TearDownTestSuite() {
    delete robot_;
    robot_ = nullptr;
  }

  static msgs::RobotStateSample toStruct(const humanoid_mpc_msgs::RobotStateSample& proto) {
    msgs::RobotStateSample sample;
    EXPECT_TRUE(msgs::FromProto(proto, &sample).ok());
    return sample;
  }

  /** The index of joint `name` in `names`. */
  static size_t indexOf(const std::vector<std::string>& names, const std::string& name) {
    for (size_t index = 0; index < names.size(); ++index) {
      if (names[index] == name) {
        return index;
      }
    }
    ADD_FAILURE() << "no joint " << name;
    return 0;
  }

  static test::TestRobot* robot_;
};

test::TestRobot* RobotStateDecoderTest::robot_ = nullptr;

TEST_F(RobotStateDecoderTest, TheBaseIsInTheWorldFrame) {
  const vector3_t rollPitchYaw(0.1, 0.2, M_PI / 2.0);
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(/*time=*/1.5, vector3_t(1.0, 2.0, 3.0), rollPitchYaw);
  proto.mutable_base_linear_velocity_local()->set_x(1.0);
  proto.mutable_base_angular_velocity_local()->set_z(0.5);
  RobotStateDecoder decoder(robot_->modelSettings());
  DecodedRobotState state;
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());

  const matrix3_t rotation = test::quaternionFromRollPitchYaw(rollPitchYaw).toRotationMatrix();
  EXPECT_EQ(state.time, 1.5);
  EXPECT_TRUE(state.basePosition.isApprox(vector3_t(1.0, 2.0, 3.0)));
  EXPECT_TRUE(state.baseEulerAnglesZyx.isApprox(vector3_t(M_PI / 2.0, 0.2, 0.1), kTolerance));
  EXPECT_TRUE(state.baseLinearVelocity.isApprox(rotation * vector3_t(1.0, 0.0, 0.0), kTolerance));
  EXPECT_TRUE(state.baseAngularVelocity.isApprox(rotation * vector3_t(0.0, 0.0, 0.5), kTolerance));

  // Generalized coordinates of the MPC's model: position, Euler ZYX, then the MPC joints by name.
  EXPECT_TRUE(state.generalizedCoordinates.head<3>().isApprox(vector3_t(1.0, 2.0, 3.0)));
  EXPECT_TRUE(state.generalizedCoordinates.segment<3>(3).isApprox(vector3_t(M_PI / 2.0, 0.2, 0.1), kTolerance));
  const std::vector<std::string>& fullJoints = robot_->modelSettings().fullJointNames;
  const std::vector<std::string>& mpcJoints = robot_->modelSettings().mpcModelJointNames;
  for (size_t joint = 0; joint < mpcJoints.size(); ++joint) {
    const scalar_t expected = 0.01 * static_cast<scalar_t>(indexOf(fullJoints, mpcJoints[joint]) + 1);
    EXPECT_DOUBLE_EQ(state.generalizedCoordinates[6 + joint], expected) << mpcJoints[joint];
    EXPECT_DOUBLE_EQ(state.generalizedVelocities[6 + joint], 0.1);
  }
  for (size_t joint = 0; joint < fullJoints.size(); ++joint) {
    EXPECT_DOUBLE_EQ(state.jointPositions[joint], 0.01 * static_cast<scalar_t>(joint + 1));
  }
}

TEST_F(RobotStateDecoderTest, TheBaseVelocityIsTheEulerAngleRates) {
  // Level and turning about the vertical: the yaw rate is the angular velocity, the other rates are zero.
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t(0.0, 0.0, 0.4));
  proto.mutable_base_angular_velocity_local()->set_z(0.7);
  RobotStateDecoder decoder(robot_->modelSettings());
  DecodedRobotState state;
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());
  EXPECT_TRUE(state.generalizedVelocities.segment<3>(3).isApprox(vector3_t(0.7, 0.0, 0.0), kTolerance))
      << state.generalizedVelocities.segment<3>(3).transpose();
}

TEST_F(RobotStateDecoderTest, TheEffortAppliedIsThePdLawOfTheAction) {
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t::Zero());
  const int joints = proto.joint_names_size();
  for (int joint = 0; joint < joints; ++joint) {
    proto.add_joint_position_targets(0.5);
    proto.add_joint_velocity_targets(0.2);
    proto.add_joint_kp(100.0);
    proto.add_joint_kd(2.0);
    proto.add_joint_feed_forward_efforts(3.0);
  }
  // The sample's first joint has no action this cycle.
  proto.set_joint_position_targets(/*index=*/0, std::numeric_limits<double>::quiet_NaN());
  RobotStateDecoder decoder(robot_->modelSettings());
  DecodedRobotState state;
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());

  const std::vector<std::string>& fullJoints = robot_->modelSettings().fullJointNames;
  const size_t withoutAction = indexOf(fullJoints, proto.joint_names(0));
  for (size_t joint = 0; joint < fullJoints.size(); ++joint) {
    const scalar_t position = 0.01 * static_cast<scalar_t>(joint + 1);
    if (joint == withoutAction) {
      EXPECT_EQ(state.jointEfforts[joint], 0.0);
      EXPECT_EQ(state.jointPositionTargets[joint], position);
      EXPECT_EQ(state.jointVelocityTargets[joint], 0.0);
      EXPECT_EQ(state.jointFeedForwardEfforts[joint], 0.0);
      continue;
    }
    EXPECT_NEAR(state.jointEfforts[joint], 3.0 + 100.0 * (0.5 - position) + 2.0 * (0.2 - 0.1), 1e-9) << fullJoints[joint];
    EXPECT_EQ(state.jointPositionTargets[joint], 0.5);
    EXPECT_EQ(state.jointVelocityTargets[joint], 0.2);
    EXPECT_EQ(state.jointFeedForwardEfforts[joint], 3.0);
  }
  const std::vector<std::string>& mpcJoints = robot_->modelSettings().mpcModelJointNames;
  EXPECT_TRUE(state.generalizedForces.head<6>().isZero());
  for (size_t joint = 0; joint < mpcJoints.size(); ++joint) {
    EXPECT_EQ(state.generalizedForces[6 + joint], state.jointEfforts[indexOf(fullJoints, mpcJoints[joint])]);
  }
}

TEST_F(RobotStateDecoderTest, WithoutAnActionTheTargetsAreTheMeasuredPositionAndZero) {
  const humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t::Zero());
  RobotStateDecoder decoder(robot_->modelSettings());
  DecodedRobotState state;
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());
  EXPECT_EQ(state.jointPositionTargets, state.jointPositions);
  EXPECT_TRUE(state.jointVelocityTargets.isZero());
  EXPECT_TRUE(state.jointFeedForwardEfforts.isZero());
  EXPECT_TRUE(state.jointEfforts.isZero());
}

TEST_F(RobotStateDecoderTest, JointsAreFoundByNameWhateverTheirOrder) {
  RobotStateDecoder decoder(robot_->modelSettings());
  DecodedRobotState first;
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t::Zero());
  ASSERT_TRUE(decoder.decode(toStruct(proto), &first).ok());
  // The same joints in another order, and one the robot does not have: the mapping follows the names.
  proto.mutable_joint_names()->SwapElements(0, 1);
  proto.mutable_joint_positions()->SwapElements(0, 1);
  proto.add_joint_names("unknown_joint");
  proto.add_joint_positions(9.0);
  proto.add_joint_velocities(9.0);
  proto.add_joint_measured_efforts(9.0);
  DecodedRobotState second;
  ASSERT_TRUE(decoder.decode(toStruct(proto), &second).ok());
  EXPECT_EQ(first.jointPositions, second.jointPositions);
  EXPECT_EQ(first.generalizedCoordinates, second.generalizedCoordinates);

  // A joint the sample does not carry is zero.
  humanoid_mpc_msgs::RobotStateSample partial = robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t::Zero());
  const std::string missing = partial.joint_names(0);
  partial.set_joint_names(/*index=*/0, "renamed_joint");
  DecodedRobotState third;
  ASSERT_TRUE(decoder.decode(toStruct(partial), &third).ok());
  EXPECT_EQ(third.jointPositions[indexOf(robot_->modelSettings().fullJointNames, missing)], 0.0);
}

TEST_F(RobotStateDecoderTest, TheMeasuredContactWrenchesAreCopied) {
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t::Zero());
  proto.mutable_measured_contact_wrenches(1)->mutable_force()->set_z(300.0);
  proto.mutable_measured_contact_wrenches(1)->mutable_torque()->set_x(4.0);
  RobotStateDecoder decoder(robot_->modelSettings());
  DecodedRobotState state;
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());
  EXPECT_TRUE(state.measuredContactWrenches[0].isZero());
  EXPECT_EQ(state.measuredContactWrenches[1][2], 300.0);
  EXPECT_EQ(state.measuredContactWrenches[1][3], 4.0);

  proto.clear_measured_contact_wrenches();
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());
  EXPECT_TRUE(state.measuredContactWrenches[1].isZero());
}

TEST_F(RobotStateDecoderTest, AnInconsistentSampleIsRefused) {
  RobotStateDecoder decoder(robot_->modelSettings());
  DecodedRobotState state;
  const humanoid_mpc_msgs::RobotStateSample valid = robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t::Zero());

  humanoid_mpc_msgs::RobotStateSample sample = valid;
  sample.mutable_joint_positions()->RemoveLast();
  EXPECT_EQ(decoder.decode(toStruct(sample), &state).code(), absl::StatusCode::kInvalidArgument);

  sample = valid;
  sample.clear_joint_velocities();
  EXPECT_EQ(decoder.decode(toStruct(sample), &state).code(), absl::StatusCode::kInvalidArgument);

  sample = valid;
  sample.add_joint_kp(1.0);
  EXPECT_EQ(decoder.decode(toStruct(sample), &state).code(), absl::StatusCode::kInvalidArgument);

  sample = valid;
  sample.mutable_base_orientation_world()->set_w(0.0);
  EXPECT_EQ(decoder.decode(toStruct(sample), &state).code(), absl::StatusCode::kInvalidArgument);

  sample = valid;
  sample.mutable_base_orientation_world()->set_w(2.0);
  EXPECT_EQ(decoder.decode(toStruct(sample), &state).code(), absl::StatusCode::kInvalidArgument);

  EXPECT_TRUE(decoder.decode(toStruct(valid), &state).ok());
}

}  // namespace
}  // namespace ocs2::humanoid::visualization
