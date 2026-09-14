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

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

/**
 * Unit tests of the toe-up swing pitch reference of the SwingTrajectoryPlanner. The foot height reference is tracked at
 * the centre of the sole, so the toe -- half a foot length ahead of it -- has no height guarantee of its own. The pitch
 * reference raises it, and these tests pin down the properties the foot cost relies on: it is inert by default, it is
 * zero in stance and at both ends of a swing, and it never lowers the toe.
 */
namespace ocs2::humanoid {

namespace {

constexpr scalar_t kTol = 1e-9;
constexpr scalar_t kLiftOff = 1.0;
constexpr scalar_t kTouchDown = 1.5;

/** STANCE | foot 0 swings [kLiftOff, kTouchDown] | STANCE. */
ModeSchedule singleSwingSchedule() {
  contact_flag_t swinging = makeFeetArray(true);
  swinging[0] = false;
  const size_t stance = stanceLeg2ModeNumber(makeFeetArray(true));
  return ModeSchedule({kLiftOff, kTouchDown}, {stance, stanceLeg2ModeNumber(swinging), stance});
}

SwingTrajectoryPlanner::Config pitchedConfig(scalar_t angle) {
  SwingTrajectoryPlanner::Config config;
  config.swingHeight = 0.12;
  config.touchDownVelocity = -0.1;
  config.swingTimeScale = 0.4;
  config.swingPitchAngle = angle;
  config.swingPitchRiseFraction = 0.25;
  config.swingPitchFallFraction = 0.15;
  return config;
}

scalar_t atSwingFraction(const SwingTrajectoryPlanner& planner, scalar_t tau) {
  return planner.getSwingPitchAngle(0, kLiftOff + tau * (kTouchDown - kLiftOff));
}

}  // namespace

/** The default configuration must reproduce the previous flat-foot reference exactly. */
TEST(SwingPitchReference, disabledByDefault) {
  SwingTrajectoryPlanner planner(SwingTrajectoryPlanner::Config(), N_CONTACTS);
  planner.update(singleSwingSchedule(), 0.0);

  EXPECT_DOUBLE_EQ(SwingTrajectoryPlanner::Config().swingPitchAngle, 0.0);
  for (scalar_t tau = 0.0; tau <= 1.0; tau += 0.05) {
    EXPECT_NEAR(atSwingFraction(planner, tau), 0.0, kTol);
  }
}

/** Zero at both ends of the swing so that lift-off and touch-down are made with a flat sole. */
TEST(SwingPitchReference, flatAtBothEndsOfTheSwing) {
  SwingTrajectoryPlanner planner(pitchedConfig(0.1), N_CONTACTS);
  planner.update(singleSwingSchedule(), 0.0);

  EXPECT_NEAR(atSwingFraction(planner, 0.0), 0.0, kTol);
  EXPECT_NEAR(atSwingFraction(planner, 1.0), 0.0, kTol);
}

/** A foot in contact is never pitched, on either side of the swing. */
TEST(SwingPitchReference, zeroInStance) {
  SwingTrajectoryPlanner planner(pitchedConfig(0.1), N_CONTACTS);
  planner.update(singleSwingSchedule(), 0.0);

  EXPECT_NEAR(planner.getSwingPitchAngle(0, kLiftOff - 0.1), 0.0, kTol);
  EXPECT_NEAR(planner.getSwingPitchAngle(0, kTouchDown + 0.1), 0.0, kTol);
  // The other foot stands through the whole schedule.
  for (scalar_t tau = 0.0; tau <= 1.0; tau += 0.1) {
    EXPECT_NEAR(planner.getSwingPitchAngle(1, kLiftOff + tau * (kTouchDown - kLiftOff)), 0.0, kTol);
  }
}

/** The configured angle is reached over the held section between the two ramps, and is never exceeded. */
TEST(SwingPitchReference, reachesTheConfiguredAngleOverTheHeldSection) {
  const scalar_t angle = 0.1;
  SwingTrajectoryPlanner planner(pitchedConfig(angle), N_CONTACTS);
  planner.update(singleSwingSchedule(), 0.0);

  for (scalar_t tau = 0.30; tau <= 0.80; tau += 0.05) {
    EXPECT_NEAR(atSwingFraction(planner, tau), angle, kTol);
  }
  for (scalar_t tau = 0.0; tau <= 1.0; tau += 0.01) {
    const scalar_t pitch = atSwingFraction(planner, tau);
    EXPECT_GE(pitch, -kTol);  // toe-up only: the reference never pitches the toe down
    EXPECT_LE(pitch, angle + kTol);
  }
}

/** Monotone up through the rise and monotone down through the fall, so the reference is smooth for the solver. */
TEST(SwingPitchReference, monotoneRampsWithNoPlateauOvershoot) {
  SwingTrajectoryPlanner planner(pitchedConfig(0.1), N_CONTACTS);
  planner.update(singleSwingSchedule(), 0.0);

  scalar_t previous = atSwingFraction(planner, 0.0);
  for (scalar_t tau = 0.01; tau <= 0.25; tau += 0.01) {
    const scalar_t pitch = atSwingFraction(planner, tau);
    EXPECT_GE(pitch, previous - kTol);
    previous = pitch;
  }
  previous = atSwingFraction(planner, 0.85);
  for (scalar_t tau = 0.86; tau <= 1.0; tau += 0.01) {
    const scalar_t pitch = atSwingFraction(planner, tau);
    EXPECT_LE(pitch, previous + kTol);
    previous = pitch;
  }
}

/**
 * The point of the whole thing: through the part of the descent where the sole centre is within a centimetre of the
 * ground, the toe must still be clear of it. Atlas' toe contact points are 0.12 m ahead of the tracked sole centre.
 */
TEST(SwingPitchReference, keepsTheToeAboveGroundThroughTheLateDescent) {
  constexpr scalar_t kToeLever = 0.12;
  const scalar_t angle = 0.1;
  SwingTrajectoryPlanner planner(pitchedConfig(angle), N_CONTACTS);
  planner.update(singleSwingSchedule(), 0.0);

  bool sawLowSoleCentre = false;
  for (scalar_t tau = 0.5; tau < 1.0; tau += 0.005) {
    const scalar_t time = kLiftOff + tau * (kTouchDown - kLiftOff);
    const scalar_t soleCentre = planner.getZpositionConstraint(0, time);
    if (soleCentre > 0.01) continue;
    sawLowSoleCentre = true;
    const scalar_t toe = soleCentre + kToeLever * std::sin(planner.getSwingPitchAngle(0, time));
    EXPECT_GE(toe, soleCentre);  // the pitch only ever helps the toe
  }
  EXPECT_TRUE(sawLowSoleCentre);
}

/** A swing shorter than swingTimeScale is scaled down in pitch, exactly as it is scaled down in height. */
TEST(SwingPitchReference, shortSwingsAreScaledDown) {
  const scalar_t angle = 0.1;
  SwingTrajectoryPlanner planner(pitchedConfig(angle), N_CONTACTS);

  const scalar_t shortLiftOff = 1.0;
  const scalar_t shortTouchDown = 1.2;  // 0.2 s, half of swingTimeScale
  contact_flag_t swinging = makeFeetArray(true);
  swinging[0] = false;
  const size_t stance = stanceLeg2ModeNumber(makeFeetArray(true));
  planner.update(ModeSchedule({shortLiftOff, shortTouchDown}, {stance, stanceLeg2ModeNumber(swinging), stance}), 0.0);

  const scalar_t midSwing = 0.5 * (shortLiftOff + shortTouchDown);
  EXPECT_NEAR(planner.getSwingPitchAngle(0, midSwing), 0.5 * angle, kTol);
}

}  // namespace ocs2::humanoid
