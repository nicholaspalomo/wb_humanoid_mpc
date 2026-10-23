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

#include "gtest/gtest.h"

#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"

/*
 * The gait switches of ProceduralMpcMotionManager compare the command with the robot's stride-mean motion (its
 * velocity in the heading frame and its heading rate) in both directions (design defect D2): a faster
 * gait needs the robot already moving near the gait's limit, and a slower one needs it to have slowed down and stopped
 * turning. The slower switch used to test the command's own yaw rate where the measurement was meant, so a robot still
 * turning fast was let down into the slower gait, and its linear clauses read the instantaneous velocity, whose sway
 * let a walking robot down. Before the window is measured throughout (null) only a switch that reads no motion is made.
 * Builds no model.
 */

namespace ocs2::humanoid {
namespace {

/** The thresholds of the shipped "slow_walk" gait (ProceduralMpcMotionManager::gaitModeStates_). */
ProceduralMpcMotionManager::GaitModeStateConfig slowWalk() {
  return ProceduralMpcMotionManager::GaitModeStateConfig{.gaitCommand = "slow_walk",
                                                         .minLinVelCmd = 0.05,
                                                         .maxLinVelCmd = 0.3,
                                                         .minAngVelCmd = 0.05,
                                                         .maxAngVelCmd = 0.2,
                                                         .linVelErrorThresh = 0.05,
                                                         .angVelErrorThresh = 0.05};
}

/** The thresholds of the shipped "stance", whose error thresholds let it go on the command alone. */
ProceduralMpcMotionManager::GaitModeStateConfig stance() {
  return ProceduralMpcMotionManager::GaitModeStateConfig{.gaitCommand = "stance",
                                                         .minLinVelCmd = -0.1,
                                                         .maxLinVelCmd = 0.1,
                                                         .minAngVelCmd = -0.1,
                                                         .maxAngVelCmd = 0.1,
                                                         .linVelErrorThresh = 10.0,
                                                         .angVelErrorThresh = 10.0};
}

/** A command [v_x, v_y, pelvis height, yaw rate] that only turns, at `yawRate`. */
vector4_t turnCommand(scalar_t yawRate) {
  return vector4_t(0.0, 0.0, 0.0, yawRate);
}

/** The stride-mean motion of a robot turning on the spot at `yawRate`. */
BasePlanarMotion turningOnTheSpot(scalar_t yawRate) {
  return BasePlanarMotion{.velocityInHeadingFrame = vector2_t::Zero(), .headingRate = yawRate};
}

/** The stride-mean motion of a robot walking at (forward, lateral) in its heading frame without turning. */
BasePlanarMotion walking(scalar_t forward, scalar_t lateral) {
  return BasePlanarMotion{.velocityInHeadingFrame = vector2_t(forward, lateral), .headingRate = 0.0};
}

bool faster(const vector4_t& command, const BasePlanarMotion& motion, const ProceduralMpcMotionManager::GaitModeStateConfig& cfg) {
  return ProceduralMpcMotionManager::transitionToFasterGait(command, &motion, cfg);
}

bool slower(const vector4_t& command, const BasePlanarMotion& motion, const ProceduralMpcMotionManager::GaitModeStateConfig& cfg) {
  return ProceduralMpcMotionManager::transitionToSlowerGait(command, &motion, cfg);
}

TEST(ProceduralGaitTransitions, AFasterGaitWaitsForTheMeasuredTurn) {
  // Turning on the spot faster than slow_walk allows: the next gait once the robot turns faster than 0.2 - 0.05 rad/s.
  const vector4_t command = turnCommand(0.3);
  EXPECT_FALSE(faster(command, turningOnTheSpot(0.1), slowWalk()));
  EXPECT_TRUE(faster(command, turningOnTheSpot(0.18), slowWalk()));
  EXPECT_TRUE(faster(command, turningOnTheSpot(-0.18), slowWalk())) << "either direction";
  // Not without the command, however fast the robot turns.
  EXPECT_FALSE(faster(turnCommand(0.1), turningOnTheSpot(0.5), slowWalk()));
}

TEST(ProceduralGaitTransitions, ASlowerGaitWaitsForTheMeasuredTurnToStop) {
  // Commanded to stand still: the slower gait once the robot turns slower than 0.05 + 0.05 rad/s. The command's own yaw
  // rate stood in this test, which the request had already bounded below 0.05 rad/s, so it never held the switch back.
  const vector4_t command = turnCommand(0.0);
  EXPECT_FALSE(slower(command, turningOnTheSpot(0.3), slowWalk()));
  EXPECT_FALSE(slower(command, turningOnTheSpot(-0.3), slowWalk())) << "either direction";
  EXPECT_TRUE(slower(command, turningOnTheSpot(0.08), slowWalk()));
  // Not while the command still asks for the gait.
  EXPECT_FALSE(slower(turnCommand(0.1), turningOnTheSpot(0.0), slowWalk()));
}

TEST(ProceduralGaitTransitions, TheStrideMeanVelocityGatesBothSwitches) {
  // Walking, not turning: the linear entries of the stride mean, in the heading frame, gate the switches.
  const vector4_t walkFaster(0.4, 0.0, 0.0, 0.0);
  EXPECT_FALSE(faster(walkFaster, walking(/*forward=*/0.2, /*lateral=*/0.0), slowWalk()));
  EXPECT_TRUE(faster(walkFaster, walking(/*forward=*/0.26, /*lateral=*/0.0), slowWalk()));
  EXPECT_TRUE(faster(walkFaster, walking(/*forward=*/0.0, /*lateral=*/-0.26), slowWalk())) << "either axis, either direction";
  const vector4_t stop(0.0, 0.0, 0.0, 0.0);
  EXPECT_FALSE(slower(stop, walking(/*forward=*/0.2, /*lateral=*/0.0), slowWalk()));
  EXPECT_FALSE(slower(stop, walking(/*forward=*/0.0, /*lateral=*/0.11), slowWalk()));
  EXPECT_TRUE(slower(stop, walking(/*forward=*/0.05, /*lateral=*/-0.05), slowWalk()));
}

TEST(ProceduralGaitTransitions, WithoutAMeasuredWindowOnlyASwitchThatReadsNoMotionIsMade) {
  // No window measured yet (after a restart, or a stall): slow_walk neither rises nor falls, whatever the command asks,
  // since its thresholds read the motion; stance, whose error thresholds every motion passes (0.1 - 10 < 0), rises on
  // the command alone, as it does with any measurement.
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToFasterGait(turnCommand(0.5), /*strideMeanMotion=*/nullptr, slowWalk()));
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToSlowerGait(turnCommand(0.0), /*strideMeanMotion=*/nullptr, slowWalk()));
  EXPECT_TRUE(ProceduralMpcMotionManager::transitionToFasterGait(turnCommand(0.5), /*strideMeanMotion=*/nullptr, stance()));
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToFasterGait(turnCommand(0.05), /*strideMeanMotion=*/nullptr, stance()))
      << "not without the command";
  for (const BasePlanarMotion& motion : {turningOnTheSpot(0.0), turningOnTheSpot(3.0), walking(/*forward=*/-2.0, /*lateral=*/1.0)}) {
    EXPECT_TRUE(faster(turnCommand(0.5), motion, stance())) << "stance's switch reads no motion";
  }
}

}  // namespace
}  // namespace ocs2::humanoid
