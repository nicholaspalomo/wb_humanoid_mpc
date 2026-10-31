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
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "ocs2_robotic_tools/common/RotationDerivativesTransforms.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc_app/visualization/RobotStateDecoder.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/visualization/test/VisualizationTestRobot.h"

namespace ocs2::humanoid::visualization {
namespace {

constexpr scalar_t kTolerance = 1.0e-12;

/** The ZYX Euler angles (yaw, pitch, roll) of (roll, pitch, yaw). */
vector3_t eulerZyxOf(const vector3_t& rollPitchYaw) {
  return vector3_t(rollPitchYaw.z(), rollPitchYaw.y(), rollPitchYaw.x());
}

/** The angle a - b wrapped into (-pi, pi]. */
scalar_t wrappedDifference(scalar_t a, scalar_t b) {
  return std::remainder(a - b, 2.0 * M_PI);
}

/**
 * The time derivative of the ZYX Euler angles of the attitude `orientation`
 * turning at the body angular velocity `angularVelocityLocal`, by a central
 * difference over +-`step` of the exact attitude xi (x) Exp(t w_B): an
 * independent check of T_B(Theta)^-1 w_B.
 */
vector3_t eulerZyxRatesByCentralDifference(
    const quaternion_t& orientation,
    const vector3_t& angularVelocityLocal,
    scalar_t step) {
  const scalar_t speed = angularVelocityLocal.norm();
  const vector3_t axis = angularVelocityLocal / speed;
  const quaternion_t later =
      orientation *
      quaternion_t(Eigen::AngleAxis<scalar_t>(step * speed, axis));
  const quaternion_t earlier =
      orientation *
      quaternion_t(Eigen::AngleAxis<scalar_t>(-step * speed, axis));
  const vector3_t laterAngles = eulerZyxFromQuaternion(later.coeffs());
  const vector3_t earlierAngles = eulerZyxFromQuaternion(earlier.coeffs());
  vector3_t rates;
  for (Eigen::Index angle = 0; angle < 3; ++angle) {
    rates[angle] = wrappedDifference(laterAngles[angle], earlierAngles[angle]) /
                   (2.0 * step);
  }
  return rates;
}

TEST(EulerZyxRatesForDisplayTest,
     TheRatesAreTheInverseOfTheBodyRateMapAndNotANumberAtTheSingularity) {
  const vector3_t angularVelocityLocal(0.3, -0.2, 0.6);
  const vector3_t eulerZyx(2.9, -0.7, 0.4);
  const vector3_t rates =
      eulerZyxRatesForDisplay(eulerZyx, angularVelocityLocal);
  EXPECT_EQ(rates, eulerZyxRatesFromBodyRates(eulerZyx, angularVelocityLocal));
  EXPECT_TRUE(localAngularVelocityFromEulerZyxRates(eulerZyx, rates)
                  .isApprox(angularVelocityLocal, kTolerance));
  // Just above and below the threshold of |cos(pitch)|, and a pitch that is not
  // a number.
  EXPECT_TRUE(
      eulerZyxRatesForDisplay(
          vector3_t(0.1, std::acos(2.0 * kDisplayEulerRateMinCosPitch), 0.2),
          angularVelocityLocal)
          .allFinite());
  for (const scalar_t pitch :
       {std::acos(0.5 * kDisplayEulerRateMinCosPitch),
        -std::acos(0.5 * kDisplayEulerRateMinCosPitch), M_PI / 2.0, -M_PI / 2.0,
        std::numeric_limits<scalar_t>::quiet_NaN()}) {
    const vector3_t singular = eulerZyxRatesForDisplay(
        vector3_t(0.1, pitch, 0.2), angularVelocityLocal);
    EXPECT_TRUE(singular.array().isNaN().all())
        << pitch << ": " << singular.transpose();
  }
}

class RobotStateDecoderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    robot_ = test::TestRobot::load(test::g1CentroidalFiles(),
                                   test::Formulation::kCentroidal)
                 .release();
  }
  static void TearDownTestSuite() {
    delete robot_;
    robot_ = nullptr;
  }

  static msgs::RobotStateSample toStruct(
      const humanoid_mpc_msgs::RobotStateSample& proto) {
    msgs::RobotStateSample sample;
    EXPECT_TRUE(msgs::FromProto(proto, &sample).ok());
    return sample;
  }

  /** The index of joint `name` in `names`. */
  static size_t indexOf(const std::vector<std::string>& names,
                        const std::string& name) {
    for (size_t index = 0; index < names.size(); ++index) {
      if (names[index] == name) {
        return index;
      }
    }
    ADD_FAILURE() << "no joint " << name;
    return 0;
  }

  static test::TestRobot* absl_nullable robot_;
};

test::TestRobot* absl_nullable RobotStateDecoderTest::robot_ = nullptr;

TEST_F(RobotStateDecoderTest, TheBaseIsInTheWorldFrame) {
  const vector3_t rollPitchYaw(0.1, 0.2, M_PI / 2.0);
  humanoid_mpc_msgs::RobotStateSample proto =
      robot_->robotState(/*time=*/1.5, vector3_t(1.0, 2.0, 3.0), rollPitchYaw);
  proto.mutable_linvel_base_rt_world_ewrt_base()->set_x(1.0);
  proto.mutable_angvel_base_rt_world_ewrt_base()->set_z(0.5);
  RobotStateDecoder decoder(robot_->robotModel());
  DecodedRobotState state;
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());

  const quaternion_t orientation =
      test::quaternionFromRollPitchYaw(rollPitchYaw);
  const matrix3_t rotation = orientation.toRotationMatrix();
  EXPECT_EQ(state.time, 1.5);
  EXPECT_TRUE(state.basePosition.isApprox(vector3_t(1.0, 2.0, 3.0)));
  EXPECT_TRUE(state.baseEulerAnglesZyx.isApprox(vector3_t(M_PI / 2.0, 0.2, 0.1),
                                                kTolerance));
  EXPECT_TRUE(state.baseLinearVelocity.isApprox(
      rotation * vector3_t(1.0, 0.0, 0.0), kTolerance));
  EXPECT_TRUE(state.baseAngularVelocity.isApprox(
      rotation * vector3_t(0.0, 0.0, 0.5), kTolerance));
  EXPECT_EQ(state.baseAngularVelocityLocal, vector3_t(0.0, 0.0, 0.5));

  // Generalized coordinates of the MPC's model: position, the quaternion (x, y,
  // z, w), then the MPC joints by name.
  ASSERT_EQ(state.generalizedCoordinates.size(),
            static_cast<Eigen::Index>(kBaseConfigurationDim +
                                      robot_->robotModel().getJointDim()));
  ASSERT_EQ(state.generalizedVelocities.size(),
            static_cast<Eigen::Index>(kBaseVelocityDim +
                                      robot_->robotModel().getJointDim()));
  EXPECT_TRUE(state.generalizedCoordinates.head<3>().isApprox(
      vector3_t(1.0, 2.0, 3.0)));
  EXPECT_TRUE(state.generalizedCoordinates.segment<4>(kBaseTranslationDim)
                  .isApprox(orientation.coeffs(), kTolerance));
  const std::vector<std::string>& fullJoints =
      robot_->modelSettings().fullJointNames;
  const std::vector<std::string>& mpcJoints =
      robot_->modelSettings().mpcModelJointNames;
  for (size_t joint = 0; joint < mpcJoints.size(); ++joint) {
    const scalar_t expected =
        0.01 * static_cast<scalar_t>(indexOf(fullJoints, mpcJoints[joint]) + 1);
    EXPECT_DOUBLE_EQ(
        state.generalizedCoordinates[kJointConfigurationOffset + joint],
        expected)
        << mpcJoints[joint];
    EXPECT_DOUBLE_EQ(state.generalizedVelocities[kJointVelocityOffset + joint],
                     0.1);
  }
  for (size_t joint = 0; joint < fullJoints.size(); ++joint) {
    EXPECT_DOUBLE_EQ(state.jointPositions[joint],
                     0.01 * static_cast<scalar_t>(joint + 1));
  }
}

/** `proto` with the base's body angular velocity `angularVelocityLocal`. */
void setAngularVelocityLocal(
    const vector3_t& angularVelocityLocal,
    humanoid_mpc_msgs::RobotStateSample* absl_nonnull proto) {
  proto->mutable_angvel_base_rt_world_ewrt_base()->set_x(
      angularVelocityLocal.x());
  proto->mutable_angvel_base_rt_world_ewrt_base()->set_y(
      angularVelocityLocal.y());
  proto->mutable_angvel_base_rt_world_ewrt_base()->set_z(
      angularVelocityLocal.z());
}

// Level: v holds the body angular velocity, and the display's Euler rates of a
// turn about the vertical are the yaw rate.
TEST_F(RobotStateDecoderTest,
       TheBaseVelocityIsTheBodyAngularVelocityAndALevelTurnIsAYawRate) {
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(
      /*time=*/0.0, vector3_t::Zero(), vector3_t(0.0, 0.0, 0.4));
  proto.mutable_angvel_base_rt_world_ewrt_base()->set_z(0.7);
  RobotStateDecoder decoder(robot_->robotModel());
  DecodedRobotState state;
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());
  EXPECT_EQ(
      vector3_t(state.generalizedVelocities.segment<3>(kBaseTranslationDim)),
      vector3_t(0.0, 0.0, 0.7));
  EXPECT_TRUE(
      state.baseEulerZyxRates.isApprox(vector3_t(0.7, 0.0, 0.0), kTolerance))
      << state.baseEulerZyxRates.transpose();
}

// Tilted: the display's Euler angles are the sample's, and their rates those
// the Euler layout's decoder plotted, the Euler rates of the world angular
// velocity (getEulerAnglesZyxDerivativesFromGlobalAngularVelocity), to 1e-12: a
// different but exact formula. Both are the time derivative of the Euler angles
// of the turning attitude.
TEST_F(
    RobotStateDecoderTest,
    TheDisplayEulerRatesOfATiltedBaseAreTheEulerLayoutsAndTheAnglesDerivative) {
  RobotStateDecoder decoder(robot_->robotModel());
  const vector3_t angularVelocityLocal(0.3, -0.2, 0.6);
  for (const vector3_t& rollPitchYaw :
       {vector3_t(0.15, -0.25, 2.8), vector3_t(-0.6, 1.2, -3.1),
        vector3_t(2.5, -1.3, 0.9), vector3_t(0.0, 0.7, M_PI - 1.0e-9)}) {
    SCOPED_TRACE(rollPitchYaw.transpose());
    humanoid_mpc_msgs::RobotStateSample proto =
        robot_->robotState(/*time=*/0.0, vector3_t::Zero(), rollPitchYaw);
    setAngularVelocityLocal(angularVelocityLocal, &proto);
    DecodedRobotState state;
    ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());
    const vector3_t eulerZyx = eulerZyxOf(rollPitchYaw);
    for (Eigen::Index angle = 0; angle < 3; ++angle) {
      EXPECT_NEAR(
          wrappedDifference(state.baseEulerAnglesZyx[angle], eulerZyx[angle]),
          0.0, kTolerance)
          << angle;
    }
    const vector3_t eulerLayoutRates =
        // NOLINTNEXTLINE(frame-naming): a generic Euler conversion
        getEulerAnglesZyxDerivativesFromGlobalAngularVelocity<scalar_t>(
            state.baseEulerAnglesZyx, state.baseAngularVelocity);
    EXPECT_TRUE(state.baseEulerZyxRates.isApprox(eulerLayoutRates, kTolerance))
        << state.baseEulerZyxRates.transpose() << " vs "
        << eulerLayoutRates.transpose();
    const vector3_t derivative = eulerZyxRatesByCentralDifference(
        state.baseOrientation, angularVelocityLocal, /*step=*/1.0e-6);
    EXPECT_TRUE(state.baseEulerZyxRates.isApprox(derivative, 1.0e-8))
        << derivative.transpose();
  }
}

// Near the pitch singularity, down to |cos(pitch)| = 1e-5, inside
// eulerZyxFromQuaternion's gimbal-lock branch: the rates grow as 1 / cos(pitch)
// and stay the angles' derivative.
TEST_F(RobotStateDecoderTest,
       TheDisplayEulerRatesNearThePitchSingularityAreTheAnglesDerivative) {
  RobotStateDecoder decoder(robot_->robotModel());
  const vector3_t angularVelocityLocal(0.3, -0.2, 0.6);
  for (const scalar_t cosPitch : {1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5}) {
    for (const scalar_t sign : {1.0, -1.0}) {
      const vector3_t rollPitchYaw(0.3, sign * std::acos(cosPitch), -1.1);
      SCOPED_TRACE(rollPitchYaw.transpose());
      humanoid_mpc_msgs::RobotStateSample proto =
          robot_->robotState(/*time=*/0.0, vector3_t::Zero(), rollPitchYaw);
      setAngularVelocityLocal(angularVelocityLocal, &proto);
      DecodedRobotState state;
      ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());
      ASSERT_TRUE(state.baseEulerZyxRates.allFinite())
          << state.baseEulerZyxRates.transpose();
      EXPECT_GT(std::abs(state.baseEulerZyxRates[kBaseRotYawIndex]),
                0.1 / cosPitch);
      // A step that turns the yaw by about 1e-4 rad.
      const vector3_t derivative = eulerZyxRatesByCentralDifference(
          state.baseOrientation, angularVelocityLocal,
          /*step=*/1.0e-4 * cosPitch);
      EXPECT_TRUE(state.baseEulerZyxRates.isApprox(derivative, 1.0e-6))
          << state.baseEulerZyxRates.transpose() << " vs "
          << derivative.transpose();
      // The rates map back to the body angular velocity through T_B of the
      // displayed angles.
      EXPECT_TRUE(localAngularVelocityFromEulerZyxRates(
                      state.baseEulerAnglesZyx, state.baseEulerZyxRates)
                      .isApprox(angularVelocityLocal, 1.0e-9));
    }
  }
}

// At the pitch singularity (|cos(pitch)| below 1e-6) the display's Euler rates
// are not a number, a gap in the plots; the angles stay finite, and the MPC's q
// and v, which hold the quaternion and the body angular velocity, are regular.
TEST_F(RobotStateDecoderTest,
       TheDisplayEulerRatesAreNotANumberAtThePitchSingularity) {
  RobotStateDecoder decoder(robot_->robotModel());
  const vector3_t angularVelocityLocal(0.3, -0.2, 0.6);
  for (const scalar_t pitch :
       {M_PI / 2.0, -M_PI / 2.0, std::acos(5.0e-7), -std::acos(5.0e-7)}) {
    const vector3_t rollPitchYaw(0.3, pitch, -1.1);
    SCOPED_TRACE(rollPitchYaw.transpose());
    humanoid_mpc_msgs::RobotStateSample proto =
        robot_->robotState(/*time=*/0.0, vector3_t::Zero(), rollPitchYaw);
    setAngularVelocityLocal(angularVelocityLocal, &proto);
    DecodedRobotState state;
    ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());
    EXPECT_TRUE(state.baseEulerZyxRates.array().isNaN().all())
        << state.baseEulerZyxRates.transpose();
    EXPECT_TRUE(state.baseEulerAnglesZyx.allFinite());
    EXPECT_NEAR(state.baseEulerAnglesZyx[kBaseRotPitchIndex], pitch, 1.0e-9);
    // The angles still reproduce the attitude: quaternionFromEulerZyx gives xi
    // or -xi.
    const vector4_t xi = state.baseOrientation.coeffs();
    const vector4_t fromAngles =
        quaternionFromEulerZyx(state.baseEulerAnglesZyx);
    EXPECT_NEAR(std::abs(fromAngles.dot(xi)), 1.0, 1.0e-12);
    EXPECT_TRUE(state.generalizedCoordinates.allFinite());
    EXPECT_EQ(
        vector3_t(state.generalizedVelocities.segment<3>(kBaseTranslationDim)),
        angularVelocityLocal);
  }
  // Just outside the gap, the rates are finite.
  humanoid_mpc_msgs::RobotStateSample proto = robot_->robotState(
      /*time=*/0.0, vector3_t::Zero(), vector3_t(0.3, std::acos(2.0e-6), -1.1));
  setAngularVelocityLocal(angularVelocityLocal, &proto);
  DecodedRobotState state;
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());
  EXPECT_TRUE(state.baseEulerZyxRates.allFinite())
      << state.baseEulerZyxRates.transpose();
}

// The base and the MPC joints become the MPC's q and v through the robot
// model's measurement helpers, the MRT controllers' conversion, in every
// formulation: q = [p_W, xi, q_j] with the sample's quaternion, safely
// normalized, and v = [v_W, w_B, dq_j] with its body angular velocity as sent.
// The display's Euler rates are the Euler layout's.
TEST_F(RobotStateDecoderTest,
       TheBaseIsTheRobotModelsMeasurementConversionInEveryFormulation) {
  const vector3_t rollPitchYaw(0.15, -0.25, 2.8);
  const vector3_t linearVelocityLocal(0.4, -0.1, 0.05);
  const vector3_t angularVelocityLocal(0.3, -0.2, 0.6);
  for (const std::pair<test::RobotFiles, test::Formulation>& configuration :
       test::shippedConfigurations()) {
    SCOPED_TRACE(configuration.first.taskFile);
    const std::unique_ptr<test::TestRobot> robot =
        test::TestRobot::load(configuration.first, configuration.second);
    const MpcRobotModelBase<scalar_t>& model = robot->robotModel();
    humanoid_mpc_msgs::RobotStateSample proto = robot->robotState(
        /*time=*/0.0, vector3_t(0.5, -0.4, 0.9), rollPitchYaw);
    proto.mutable_linvel_base_rt_world_ewrt_base()->set_x(
        linearVelocityLocal.x());
    proto.mutable_linvel_base_rt_world_ewrt_base()->set_y(
        linearVelocityLocal.y());
    proto.mutable_linvel_base_rt_world_ewrt_base()->set_z(
        linearVelocityLocal.z());
    proto.mutable_angvel_base_rt_world_ewrt_base()->set_x(
        angularVelocityLocal.x());
    proto.mutable_angvel_base_rt_world_ewrt_base()->set_y(
        angularVelocityLocal.y());
    proto.mutable_angvel_base_rt_world_ewrt_base()->set_z(
        angularVelocityLocal.z());
    RobotStateDecoder decoder(model);
    DecodedRobotState state;
    ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());

    // The helpers, called with the decoded orientation and the MPC joints in
    // the MPC's order.
    const std::vector<std::string>& fullJoints =
        robot->modelSettings().fullJointNames;
    const std::vector<std::string>& mpcJoints =
        robot->modelSettings().mpcModelJointNames;
    vector_t jointPositions(static_cast<Eigen::Index>(mpcJoints.size()));
    for (size_t joint = 0; joint < mpcJoints.size(); ++joint) {
      jointPositions[static_cast<Eigen::Index>(joint)] =
          0.01 *
          static_cast<scalar_t>(indexOf(fullJoints, mpcJoints[joint]) + 1);
    }
    const vector_t jointVelocities =
        vector_t::Constant(static_cast<Eigen::Index>(mpcJoints.size()), 0.1);
    vector_t q(static_cast<Eigen::Index>(model.getConfigurationDim()));
    vector_t v(static_cast<Eigen::Index>(model.getVelocityDim()));
    model.fillGeneralizedCoordinatesFromMeasurement(
        state.basePosition, state.baseOrientation.coeffs(), jointPositions, q);
    model.fillGeneralizedVelocitiesFromMeasurement(
        state.baseOrientation.coeffs(), linearVelocityLocal,
        angularVelocityLocal, jointVelocities, v);
    EXPECT_EQ(state.generalizedCoordinates, q);
    EXPECT_EQ(state.generalizedVelocities, v);
    ASSERT_EQ(state.generalizedForces.size(), v.size());

    // The quaternion layout: q's base is the position and the normalized
    // quaternion, v's the world linear velocity and the body angular velocity.
    ASSERT_EQ(q.size(), static_cast<Eigen::Index>(kBaseConfigurationDim +
                                                  model.getJointDim()));
    ASSERT_EQ(v.size(), static_cast<Eigen::Index>(kBaseVelocityDim +
                                                  model.getJointDim()));
    EXPECT_EQ(q.head<3>(), state.basePosition);
    EXPECT_EQ(vector4_t(q.segment<4>(kBaseTranslationDim)),
              safelyNormalizedQuaternion(state.baseOrientation.coeffs()));
    EXPECT_TRUE(v.head<3>().isApprox(state.baseRotation * linearVelocityLocal,
                                     kTolerance));
    EXPECT_EQ(vector3_t(v.segment<3>(kBaseTranslationDim)),
              angularVelocityLocal);
    EXPECT_EQ(state.baseEulerAnglesZyx,
              eulerZyxFromQuaternion(state.baseOrientation.coeffs()));
    const vector3_t eulerLayoutRates =
        // NOLINTNEXTLINE(frame-naming): a generic Euler conversion
        getEulerAnglesZyxDerivativesFromGlobalAngularVelocity<scalar_t>(
            state.baseEulerAnglesZyx, state.baseAngularVelocity);
    EXPECT_TRUE(state.baseEulerZyxRates.isApprox(eulerLayoutRates, kTolerance))
        << state.baseEulerZyxRates.transpose();
  }
}

TEST_F(RobotStateDecoderTest, TheEffortAppliedIsThePdLawOfTheAction) {
  humanoid_mpc_msgs::RobotStateSample proto =
      robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t::Zero());
  const int joints = proto.joint_names_size();
  for (int joint = 0; joint < joints; ++joint) {
    proto.add_joint_position_targets(0.5);
    proto.add_joint_velocity_targets(0.2);
    proto.add_joint_kp(100.0);
    proto.add_joint_kd(2.0);
    proto.add_joint_feed_forward_efforts(3.0);
  }
  // The sample's first joint has no action this cycle.
  proto.set_joint_position_targets(/*index=*/0,
                                   std::numeric_limits<double>::quiet_NaN());
  RobotStateDecoder decoder(robot_->robotModel());
  DecodedRobotState state;
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());

  const std::vector<std::string>& fullJoints =
      robot_->modelSettings().fullJointNames;
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
    EXPECT_NEAR(state.jointEfforts[joint],
                3.0 + 100.0 * (0.5 - position) + 2.0 * (0.2 - 0.1), 1.0e-9)
        << fullJoints[joint];
    EXPECT_EQ(state.jointPositionTargets[joint], 0.5);
    EXPECT_EQ(state.jointVelocityTargets[joint], 0.2);
    EXPECT_EQ(state.jointFeedForwardEfforts[joint], 3.0);
  }
  const std::vector<std::string>& mpcJoints =
      robot_->modelSettings().mpcModelJointNames;
  ASSERT_EQ(state.generalizedForces.size(),
            static_cast<Eigen::Index>(robot_->robotModel().getVelocityDim()));
  EXPECT_TRUE(
      state.generalizedForces.head(static_cast<Eigen::Index>(kBaseVelocityDim))
          .isZero());
  for (size_t joint = 0; joint < mpcJoints.size(); ++joint) {
    EXPECT_EQ(state.generalizedForces[kJointVelocityOffset + joint],
              state.jointEfforts[indexOf(fullJoints, mpcJoints[joint])]);
  }
}

TEST_F(RobotStateDecoderTest,
       WithoutAnActionTheTargetsAreTheMeasuredPositionAndZero) {
  const humanoid_mpc_msgs::RobotStateSample proto =
      robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t::Zero());
  RobotStateDecoder decoder(robot_->robotModel());
  DecodedRobotState state;
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());
  EXPECT_EQ(state.jointPositionTargets, state.jointPositions);
  EXPECT_TRUE(state.jointVelocityTargets.isZero());
  EXPECT_TRUE(state.jointFeedForwardEfforts.isZero());
  EXPECT_TRUE(state.jointEfforts.isZero());
}

TEST_F(RobotStateDecoderTest, JointsAreFoundByNameWhateverTheirOrder) {
  RobotStateDecoder decoder(robot_->robotModel());
  DecodedRobotState first;
  humanoid_mpc_msgs::RobotStateSample proto =
      robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t::Zero());
  ASSERT_TRUE(decoder.decode(toStruct(proto), &first).ok());
  // The same joints in another order, and one the robot does not have: the
  // mapping follows the names.
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
  humanoid_mpc_msgs::RobotStateSample partial =
      robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t::Zero());
  const std::string missing = partial.joint_names(0);
  partial.set_joint_names(/*index=*/0, "renamed_joint");
  DecodedRobotState third;
  ASSERT_TRUE(decoder.decode(toStruct(partial), &third).ok());
  EXPECT_EQ(third.jointPositions[indexOf(robot_->modelSettings().fullJointNames,
                                         missing)],
            0.0);
}

TEST_F(RobotStateDecoderTest, TheMeasuredContactWrenchesAreCopied) {
  humanoid_mpc_msgs::RobotStateSample proto =
      robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t::Zero());
  proto.mutable_meas_wrench_contacts_ewrt_world(1)->mutable_force()->set_z(
      300.0);
  proto.mutable_meas_wrench_contacts_ewrt_world(1)->mutable_torque()->set_x(
      4.0);
  RobotStateDecoder decoder(robot_->robotModel());
  DecodedRobotState state;
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());
  EXPECT_TRUE(state.measuredContactWrenches[0].isZero());
  EXPECT_EQ(state.measuredContactWrenches[1][2], 300.0);
  EXPECT_EQ(state.measuredContactWrenches[1][3], 4.0);

  proto.clear_meas_wrench_contacts_ewrt_world();
  ASSERT_TRUE(decoder.decode(toStruct(proto), &state).ok());
  EXPECT_TRUE(state.measuredContactWrenches[1].isZero());
}

TEST_F(RobotStateDecoderTest, AnInconsistentSampleIsRefused) {
  RobotStateDecoder decoder(robot_->robotModel());
  DecodedRobotState state;
  const humanoid_mpc_msgs::RobotStateSample valid =
      robot_->robotState(/*time=*/0.0, vector3_t::Zero(), vector3_t::Zero());

  humanoid_mpc_msgs::RobotStateSample sample = valid;
  sample.mutable_joint_positions()->RemoveLast();
  EXPECT_EQ(decoder.decode(toStruct(sample), &state).code(),
            absl::StatusCode::kInvalidArgument);

  sample = valid;
  sample.clear_joint_velocities();
  EXPECT_EQ(decoder.decode(toStruct(sample), &state).code(),
            absl::StatusCode::kInvalidArgument);

  sample = valid;
  sample.add_joint_kp(1.0);
  EXPECT_EQ(decoder.decode(toStruct(sample), &state).code(),
            absl::StatusCode::kInvalidArgument);

  sample = valid;
  sample.mutable_quat_base_rt_world()->set_w(0.0);
  EXPECT_EQ(decoder.decode(toStruct(sample), &state).code(),
            absl::StatusCode::kInvalidArgument);

  sample = valid;
  sample.mutable_quat_base_rt_world()->set_w(2.0);
  EXPECT_EQ(decoder.decode(toStruct(sample), &state).code(),
            absl::StatusCode::kInvalidArgument);

  EXPECT_TRUE(decoder.decode(toStruct(valid), &state).ok());
}

}  // namespace
}  // namespace ocs2::humanoid::visualization
