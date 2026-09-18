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

#include <cmath>

#include "humanoid_common_mpc_ros2/fsm/SimFsmBridge.h"

namespace ocs2::humanoid {
namespace {

/** A robot state with the given base rotation; nothing else of the state enters the tilt. */
robot::model::RobotState stateWithRotation(const quaternion_t& rotation) {
  robot::model::RobotState state;
  state.setRootRotationLocalToWorldFrame(rotation);
  return state;
}

quaternion_t aboutAxis(const vector3_t& axis, scalar_t angle) {
  return quaternion_t(Eigen::AngleAxis<scalar_t>(angle, axis.normalized()));
}

}  // namespace

TEST(BaseTiltAngle, uprightIsZeroWhateverTheHeading) {
  EXPECT_NEAR(SimFsmBridge::baseTiltAngle(stateWithRotation(quaternion_t::Identity())), 0.0, 1e-12);
  // Turning on the spot is not a tilt: the measure compares verticals, so it must not see a yaw at all.
  for (const scalar_t yaw : {0.5, 2.0, -3.0}) {
    EXPECT_NEAR(SimFsmBridge::baseTiltAngle(stateWithRotation(aboutAxis(vector3_t::UnitZ(), yaw))), 0.0, 1e-12) << "yaw " << yaw;
  }
}

TEST(BaseTiltAngle, isTheAngleBetweenTheVerticals) {
  // A pitch or a roll tilts the base's vertical by exactly that angle, either way about either axis.
  for (const scalar_t angle : {0.1, 0.5, 1.0, 2.0}) {
    for (const vector3_t& axis : {vector3_t(vector3_t::UnitX()), vector3_t(vector3_t::UnitY())}) {
      EXPECT_NEAR(SimFsmBridge::baseTiltAngle(stateWithRotation(aboutAxis(axis, angle))), angle, 1e-9) << "angle " << angle;
      EXPECT_NEAR(SimFsmBridge::baseTiltAngle(stateWithRotation(aboutAxis(axis, -angle))), angle, 1e-9) << "angle " << -angle;
    }
  }
}

TEST(BaseTiltAngle, onItsSideIsAQuarterTurnAndUpsideDownIsAHalf) {
  EXPECT_NEAR(SimFsmBridge::baseTiltAngle(stateWithRotation(aboutAxis(vector3_t::UnitX(), 0.5 * M_PI))), 0.5 * M_PI, 1e-9);
  EXPECT_NEAR(SimFsmBridge::baseTiltAngle(stateWithRotation(aboutAxis(vector3_t::UnitY(), M_PI))), M_PI, 1e-9);
}

TEST(BaseTiltAngle, staysFiniteAtTheLimits) {
  // acos of a rotation matrix entry that rounds just past one must not produce a NaN: the fall check compares this
  // against a threshold every control cycle, and a NaN would compare false and silently disable the recovery.
  const scalar_t upsideDown = SimFsmBridge::baseTiltAngle(stateWithRotation(aboutAxis(vector3_t::UnitX(), M_PI)));
  EXPECT_TRUE(std::isfinite(upsideDown));
  EXPECT_NEAR(upsideDown, M_PI, 1e-9);
}

}  // namespace ocs2::humanoid
