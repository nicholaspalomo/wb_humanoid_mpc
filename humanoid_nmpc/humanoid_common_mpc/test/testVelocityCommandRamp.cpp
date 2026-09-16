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

#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"

using namespace ocs2;
using namespace ocs2::humanoid;

// [v_x, v_y, pelvis height, yaw rate]
TEST(VelocityCommandRamp, limitsOffPassTheTargetThrough) {
  const vector4_t target(1.6, 0.2, 0.9, 0.8);
  const vector4_t current(0.0, 0.0, 0.8, 0.0);
  EXPECT_TRUE(ProceduralMpcMotionManager::rateLimitVelocityCommand(target, current, 0.01, 0.0, 0.0).isApprox(target));
  EXPECT_TRUE(ProceduralMpcMotionManager::rateLimitVelocityCommand(target, current, 0.01, -1.0, -1.0).isApprox(target));
}

TEST(VelocityCommandRamp, linearChangeIsBoundedAsAVectorAndKeepsItsDirection) {
  const vector4_t target(1.6, 1.2, 0.9, 0.0);  // a 2 m/s jump at 36.87 degrees
  const vector4_t current(0.0, 0.0, 0.8, 0.0);
  const vector4_t limited = ProceduralMpcMotionManager::rateLimitVelocityCommand(target, current, 0.1, 1.0, 0.0);
  EXPECT_NEAR(limited.head<2>().norm(), 0.1, 1e-12);                   // 1 m/s^2 for 0.1 s
  EXPECT_NEAR(limited(0) / limited(1), target(0) / target(1), 1e-12);  // same direction as the requested change
  EXPECT_DOUBLE_EQ(limited(2), target(2));                             // the pelvis height is not ramped
  // A change within the limit is taken whole.
  const vector4_t near(0.05, 0.0, 0.9, 0.0);
  EXPECT_TRUE(ProceduralMpcMotionManager::rateLimitVelocityCommand(near, current, 0.1, 1.0, 0.0).isApprox(near));
  // Ramping down is bounded the same way.
  const vector4_t down = ProceduralMpcMotionManager::rateLimitVelocityCommand(current, target, 0.1, 1.0, 0.0);
  EXPECT_NEAR((down.head<2>() - target.head<2>()).norm(), 0.1, 1e-12);
}

TEST(VelocityCommandRamp, yawRateChangeIsBoundedSeparately) {
  const vector4_t target(0.0, 0.0, 0.9, 1.0);
  const vector4_t current(0.0, 0.0, 0.9, 0.0);
  const vector4_t limited = ProceduralMpcMotionManager::rateLimitVelocityCommand(target, current, 0.05, 0.0, 2.0);
  EXPECT_DOUBLE_EQ(limited(3), 0.1);
  const vector4_t back = ProceduralMpcMotionManager::rateLimitVelocityCommand(current, target, 0.05, 0.0, 2.0);
  EXPECT_DOUBLE_EQ(back(3), 0.9);
}

TEST(VelocityCommandRamp, aNonPositiveTimeStepHoldsTheCurrentReference) {
  const vector4_t target(1.0, 0.0, 0.9, 0.5);
  const vector4_t current(0.2, 0.0, 0.9, 0.1);
  EXPECT_TRUE(ProceduralMpcMotionManager::rateLimitVelocityCommand(target, current, 0.0, 1.0, 1.0).isApprox(current));
  EXPECT_TRUE(ProceduralMpcMotionManager::rateLimitVelocityCommand(target, current, -0.1, 1.0, 1.0).isApprox(current));
}

TEST(VelocityCommandRamp, aStickJumpBecomesARampOfTheConfiguredAcceleration) {
  const vector4_t target(1.6, 0.0, 0.9, 0.0);
  vector4_t reference(0.0, 0.0, 0.9, 0.0);
  int steps = 0;
  while ((reference - target).norm() > 1e-9 && steps < 10000) {
    reference = ProceduralMpcMotionManager::rateLimitVelocityCommand(target, reference, 0.01, 1.0, 0.0);
    ++steps;
  }
  EXPECT_EQ(steps, 160);  // 1.6 m/s at 1 m/s^2 in 1.6 s of 10 ms solves
}
