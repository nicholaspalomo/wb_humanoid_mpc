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

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

#include "humanoid_common_mpc_app/robot/JointNamesByIndex.h"
#include "humanoid_common_mpc_app/robot/TelemetrySampler.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"

/*
 * The telemetry of the realtime loop: one sample every decimation-th cycle, carrying the robot state, the joint action,
 * the measured contacts and forces and every joint's name, which survives the conversion to robot/state and back; a
 * ring that the communication thread does not drain drops and counts.
 */

namespace ocs2::humanoid {
namespace {

constexpr const char* kAtlasUrdf = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

class TelemetrySamplerTest : public ::testing::Test {
 protected:
  TelemetrySamplerTest() : description_(kAtlasUrdf), state_(description_), action_(description_) {
    state_.setTime(1.25);
    state_.setRootPositionInWorldFrame(vector3_t(0.1, -0.2, 0.9));
    state_.setRootRotationLocalToWorldFrame(quaternion_t(Eigen::AngleAxis<scalar_t>(0.3, vector3_t::UnitZ())));
    state_.setRootLinearVelocityInLocalFrame(vector3_t(0.5, 0.0, -0.1));
    state_.setRootAngularVelocityInLocalFrame(vector3_t(0.0, 0.2, 0.0));
    for (size_t joint = 0; joint < description_.getNumJoints(); ++joint) {
      state_.setJointPosition(joint, 0.01 * static_cast<scalar_t>(joint));
      state_.setJointVelocity(joint, -0.02 * static_cast<scalar_t>(joint));
      robot::model::JointAction& action = *action_.at(joint);
      action.q_des = 0.03 * static_cast<scalar_t>(joint);
      action.qd_des = 0.5;
      action.kp = 100.0 + static_cast<scalar_t>(joint);
      action.kd = 2.0;
      action.feed_forward_effort = -3.0 * static_cast<scalar_t>(joint);
    }
  }

  TelemetrySampler::Config config(size_t decimation, size_t capacity = 16) const {
    TelemetrySampler::Config config;
    config.jointNames = jointNamesByIndex(description_);
    config.decimation = decimation;
    config.capacity = capacity;
    return config;
  }

  robot::model::RobotDescription description_;
  robot::model::RobotState state_;
  robot::model::RobotJointAction action_;
  const contact_flag_t flags_{true, false};
  const std::array<vector3_t, N_CONTACTS> forces_{vector3_t(1.0, 2.0, 300.0), vector3_t(0.0, 0.0, 0.5)};
};

TEST_F(TelemetrySamplerTest, TakesOneSampleEveryDecimationCycles) {
  TelemetrySampler sampler(config(/*decimation=*/5));
  size_t taken = 0;
  for (int cycle = 0; cycle < 23; ++cycle) {
    if (sampler.sample(state_, action_, "WB_MPC", flags_, forces_)) ++taken;
  }
  EXPECT_EQ(taken, 4u);
  EXPECT_EQ(sampler.samplesTaken(), 4u);
  EXPECT_EQ(sampler.drain([](const msgs::RobotStateSample&) {}), 4u);
  EXPECT_EQ(sampler.dropped(), 0u);
}

TEST_F(TelemetrySamplerTest, TheSampleIsTheCycleAndRoundTripsThroughItsMessage) {
  TelemetrySampler sampler(config(/*decimation=*/1));
  ASSERT_TRUE(sampler.sample(state_, action_, "GRAVITY_COMP", flags_, forces_));
  msgs::RobotStateSample sample;
  ASSERT_EQ(sampler.drain([&](const msgs::RobotStateSample& taken) { sample = taken; }), 1u);

  EXPECT_DOUBLE_EQ(sample.time, 1.25);
  EXPECT_EQ(sample.control_mode, "GRAVITY_COMP");
  EXPECT_DOUBLE_EQ(sample.base_position_world.z, 0.9);
  const quaternion_t orientation = state_.getRootRotationLocalToWorldFrame();
  EXPECT_DOUBLE_EQ(sample.base_orientation_world.w, orientation.w());
  EXPECT_DOUBLE_EQ(sample.base_orientation_world.z, orientation.z());
  EXPECT_DOUBLE_EQ(sample.base_linear_velocity_local.x, 0.5);
  EXPECT_DOUBLE_EQ(sample.base_angular_velocity_local.y, 0.2);
  ASSERT_EQ(sample.joint_names, jointNamesByIndex(description_)) << "every joint, by name, in the order of the arrays";
  const Eigen::Index numJoints = static_cast<Eigen::Index>(description_.getNumJoints());
  ASSERT_EQ(sample.joint_positions.size(), numJoints);
  for (Eigen::Index joint = 0; joint < numJoints; ++joint) {
    const size_t index = static_cast<size_t>(joint);
    EXPECT_DOUBLE_EQ(sample.joint_positions[joint], state_.getJointPosition(index));
    EXPECT_DOUBLE_EQ(sample.joint_velocities[joint], state_.getJointVelocity(index));
    EXPECT_DOUBLE_EQ(sample.joint_position_targets[joint], action_.at(index)->q_des);
    EXPECT_DOUBLE_EQ(sample.joint_velocity_targets[joint], action_.at(index)->qd_des);
    EXPECT_DOUBLE_EQ(sample.joint_kp[joint], action_.at(index)->kp);
    EXPECT_DOUBLE_EQ(sample.joint_kd[joint], action_.at(index)->kd);
    EXPECT_DOUBLE_EQ(sample.joint_feed_forward_efforts[joint], action_.at(index)->feed_forward_effort);
  }
  EXPECT_EQ(sample.contact_flags, (std::vector<bool>{true, false}));
  ASSERT_EQ(sample.measured_contact_wrenches.size(), N_CONTACTS);
  EXPECT_DOUBLE_EQ(sample.measured_contact_wrenches[0].force.z, 300.0);
  EXPECT_DOUBLE_EQ(sample.measured_contact_wrenches[1].force.z, 0.5);

  // What the communication thread publishes on robot/state, and what a subscriber reads back.
  humanoid_mpc_msgs::RobotStateSample message;
  ToProto(sample, &message);
  EXPECT_EQ(message.joint_names_size(), static_cast<int>(numJoints));
  EXPECT_EQ(message.joint_names(0), description_.getJointName(0));
  msgs::RobotStateSample decoded;
  ASSERT_TRUE(FromProto(message, &decoded).ok());
  EXPECT_EQ(decoded, sample);
}

TEST_F(TelemetrySamplerTest, AJointWithoutAnActionHasANanPositionTarget) {
  // robot_state_sample.proto's contract, which the visualization reads: NaN, not a target of 0 rad.
  const size_t idle = description_.getNumJoints() / 2;
  action_.at(idle).reset();
  TelemetrySampler sampler(config(/*decimation=*/1));
  ASSERT_TRUE(sampler.sample(state_, action_, "JOINT_PD", flags_, forces_));
  msgs::RobotStateSample sample;
  ASSERT_EQ(sampler.drain([&](const msgs::RobotStateSample& taken) { sample = taken; }), 1u);
  const Eigen::Index row = static_cast<Eigen::Index>(idle);
  EXPECT_TRUE(std::isnan(sample.joint_position_targets[row]));
  EXPECT_DOUBLE_EQ(sample.joint_kp[row], 0.0);
  EXPECT_DOUBLE_EQ(sample.joint_feed_forward_efforts[row], 0.0);
  EXPECT_DOUBLE_EQ(sample.joint_position_targets[row + 1], action_.at(idle + 1)->q_des) << "the other joints keep theirs";
}

TEST_F(TelemetrySamplerTest, AFullRingDropsAndCountsTheSamples) {
  TelemetrySampler sampler(config(/*decimation=*/1, /*capacity=*/3));
  for (int cycle = 0; cycle < 5; ++cycle) sampler.sample(state_, action_, "WB_MPC", flags_, forces_);
  EXPECT_EQ(sampler.dropped(), 2u);
  std::vector<double> times;
  EXPECT_EQ(sampler.drain([&](const msgs::RobotStateSample& sample) { times.push_back(sample.time); }), 3u);
  // Room again, and the counter keeps the drops of before.
  EXPECT_TRUE(sampler.sample(state_, action_, "WB_MPC", flags_, forces_));
  EXPECT_EQ(sampler.drain([](const msgs::RobotStateSample&) {}), 1u);
  EXPECT_EQ(sampler.dropped(), 2u);
}

TEST_F(TelemetrySamplerTest, ADecimationOfZeroSamplesEveryCycle) {
  TelemetrySampler sampler(config(/*decimation=*/0));
  EXPECT_EQ(sampler.decimation(), 1u);
  EXPECT_TRUE(sampler.sample(state_, action_, "WB_MPC", flags_, forces_));
}

}  // namespace
}  // namespace ocs2::humanoid
