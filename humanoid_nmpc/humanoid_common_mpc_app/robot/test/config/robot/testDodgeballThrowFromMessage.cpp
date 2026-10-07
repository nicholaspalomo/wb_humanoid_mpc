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

// The simulator's throw from a DodgeballThrow message: a complete throw converts, every value the simulator uses is
// required and finite, the flight time is not negative and the mass is positive, and what is there for the record is
// not read.

#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc_app/robot/config/robot/DodgeballThrowFromMessage.h"
#include "humanoid_mpc_msgs/dodgeball_throw.nproto.h"
#include "humanoid_mpc_msgs/vector3.nproto.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"

namespace ocs2::humanoid {
namespace {

using ::testing::HasSubstr;
using DodgeballThrow = robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow;

msgs::Vector3 vector3(double x, double y, double z) {
  msgs::Vector3 value;
  value.x = x;
  value.y = y;
  value.z = z;
  return value;
}

msgs::DodgeballThrow completeThrow() {
  msgs::DodgeballThrow message;
  message.azimuth_deg = 30.0;
  message.elevation_deg = 15.0;
  message.distance = 3.0;
  message.speed = 9.0;
  message.mass = 1.2;
  message.spawn_offset = vector3(/*x=*/2.5, /*y=*/1.4, /*z=*/0.8);
  message.launch_velocity = vector3(/*x=*/-7.8, /*y=*/-4.5, /*z=*/-0.8);
  message.flight_time = 0.32;
  message.launch_speed = 9.0;
  message.impact_momentum = 11.8;
  return message;
}

TEST(DodgeballThrowFromMessageTest, ACompleteThrowConverts) {
  const absl::StatusOr<DodgeballThrow> command = dodgeballThrowFromMessage(completeThrow());
  ASSERT_TRUE(command.ok()) << command.status();
  EXPECT_EQ(command->spawnOffset[0], 2.5);
  EXPECT_EQ(command->spawnOffset[1], 1.4);
  EXPECT_EQ(command->spawnOffset[2], 0.8);
  EXPECT_EQ(command->launchVelocity[0], -7.8);
  EXPECT_EQ(command->launchVelocity[1], -4.5);
  EXPECT_EQ(command->launchVelocity[2], -0.8);
  EXPECT_EQ(command->flightTime, 0.32);
  EXPECT_EQ(command->mass, 1.2);
}

TEST(DodgeballThrowFromMessageTest, EveryValueTheSimulatorUsesIsRequired) {
  struct Missing {
    msgs::DodgeballThrow message;
    std::string field;
  };
  msgs::DodgeballThrow noOffset = completeThrow();
  noOffset.spawn_offset.reset();
  msgs::DodgeballThrow noVelocity = completeThrow();
  noVelocity.launch_velocity.reset();
  msgs::DodgeballThrow noFlightTime = completeThrow();
  noFlightTime.flight_time.reset();
  msgs::DodgeballThrow noMass = completeThrow();
  noMass.mass.reset();
  const std::vector<Missing> cases = {
      {.message = noOffset, .field = "spawn_offset"},
      {.message = noVelocity, .field = "launch_velocity"},
      {.message = noFlightTime, .field = "flight_time"},
      {.message = noMass, .field = "mass"},
  };
  for (const Missing& missing : cases) {
    const absl::StatusOr<DodgeballThrow> command = dodgeballThrowFromMessage(missing.message);
    EXPECT_EQ(command.status().code(), absl::StatusCode::kInvalidArgument) << missing.field;
    EXPECT_THAT(command.status().message(), HasSubstr("the throw has no " + missing.field));
  }
}

TEST(DodgeballThrowFromMessageTest, AValueThatIsNotFiniteIsRefusedByItsField) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  msgs::DodgeballThrow velocity = completeThrow();
  velocity.launch_velocity = vector3(/*x=*/-7.8, /*y=*/nan, /*z=*/-0.8);
  EXPECT_THAT(dodgeballThrowFromMessage(velocity).status().message(), HasSubstr("launch_velocity.y is nan"));
  msgs::DodgeballThrow offset = completeThrow();
  offset.spawn_offset = vector3(/*x=*/2.5, /*y=*/1.4, /*z=*/inf);
  EXPECT_THAT(dodgeballThrowFromMessage(offset).status().message(), HasSubstr("spawn_offset.z is inf"));
  msgs::DodgeballThrow mass = completeThrow();
  mass.mass = nan;
  EXPECT_THAT(dodgeballThrowFromMessage(mass).status().message(), HasSubstr("mass is nan"));
}

TEST(DodgeballThrowFromMessageTest, TheFlightTimeIsNotNegativeAndTheMassIsPositive) {
  msgs::DodgeballThrow early = completeThrow();
  early.flight_time = -0.1;
  EXPECT_THAT(dodgeballThrowFromMessage(early).status().message(), HasSubstr("before the throw"));
  msgs::DodgeballThrow now = completeThrow();
  now.flight_time = 0.0;
  EXPECT_TRUE(dodgeballThrowFromMessage(now).ok()) << "a ball that arrives at once is a push";
  for (const double bad : {0.0, -1.0}) {
    msgs::DodgeballThrow weightless = completeThrow();
    weightless.mass = bad;
    EXPECT_THAT(dodgeballThrowFromMessage(weightless).status().message(), HasSubstr("a ball needs a positive mass"));
  }
}

TEST(DodgeballThrowFromMessageTest, WhatIsThereForTheRecordIsNotRead) {
  msgs::DodgeballThrow recorded = completeThrow();
  recorded.azimuth_deg = std::numeric_limits<double>::quiet_NaN();
  recorded.distance = -1.0;
  recorded.launch_speed = std::numeric_limits<double>::infinity();
  const absl::StatusOr<DodgeballThrow> command = dodgeballThrowFromMessage(recorded);
  ASSERT_TRUE(command.ok()) << command.status();
  EXPECT_EQ(command->mass, 1.2);
}

}  // namespace
}  // namespace ocs2::humanoid
