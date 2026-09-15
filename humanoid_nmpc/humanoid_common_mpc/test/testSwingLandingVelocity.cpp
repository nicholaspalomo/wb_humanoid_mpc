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
#include <optional>

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

namespace ocs2::humanoid {

namespace {

constexpr scalar_t kTol = 1e-9;
constexpr scalar_t kLiftOff = 1.0;
constexpr scalar_t kTouchDown = 1.5;  // a 0.5 s swing, longer than swingTimeScale so nothing is scaled down
constexpr scalar_t kTerrainHeight = 0.0;

ModeSchedule singleSwingSchedule() {
  contact_flag_t swinging = makeFeetArray(true);
  swinging[0] = false;
  const size_t stance = stanceLeg2ModeNumber(makeFeetArray(true));
  return ModeSchedule({kLiftOff, kTouchDown}, {stance, stanceLeg2ModeNumber(swinging), stance});
}

/** The DRC Atlas swing configuration, with the touch-down velocity as a parameter. */
SwingTrajectoryPlanner::Config atlasConfig(scalar_t touchDownVelocity) {
  SwingTrajectoryPlanner::Config config;
  config.liftOffVelocity = 0.05;
  config.touchDownVelocity = touchDownVelocity;
  config.swingHeight = 0.08;
  config.touchDownHeightOffset = -0.001;
  config.swingTimeScale = 0.4;
  return config;
}

SwingTrajectoryPlanner plannerWith(scalar_t touchDownVelocity) {
  SwingTrajectoryPlanner planner(atlasConfig(touchDownVelocity), N_CONTACTS);
  planner.update(singleSwingSchedule(), kTerrainHeight);
  return planner;
}

/**
 * The swing foot's vertical velocity is imposed by the hard normal-velocity constraint that HumanoidPreComputation
 * builds from the planner: v_z = zdot_ref + gain * (z_ref - z). Evaluates that for a foot at height `z`.
 */
scalar_t constrainedVerticalVelocity(const SwingTrajectoryPlanner& planner, scalar_t time, scalar_t gain, scalar_t z) {
  return planner.getZvelocityConstraint(0, time) + gain * (planner.getZpositionConstraint(0, time) - z);
}

}  // namespace

/**
 * The reference must arrive at the ground at the configured descent rate. With the rate at zero the height spline ends
 * with zero slope and the descent has to come from tracking-error feedback instead, which is what made the foot kick
 * into the ground.
 */
TEST(SwingLandingVelocity, ReferenceArrivesAtTheConfiguredDescentRate) {
  for (const scalar_t touchDownVelocity : {-0.05, -0.1, 0.0}) {
    const SwingTrajectoryPlanner planner = plannerWith(touchDownVelocity);
    EXPECT_NEAR(planner.getZvelocityConstraint(0, kTouchDown), touchDownVelocity, kTol) << "touchDownVelocity " << touchDownVelocity;
    EXPECT_NEAR(planner.getZpositionConstraint(0, kTouchDown), kTerrainHeight + atlasConfig(touchDownVelocity).touchDownHeightOffset, kTol);
  }
}

/**
 * With a real descent rate the foot lands at that rate however well it tracked; with a zero rate the entire landing
 * velocity is the feedback gain times the tracking lag, so a foot that is late by a couple of centimetres is driven
 * into the floor at a speed set by nothing the operator chose.
 */
TEST(SwingLandingVelocity, LandingVelocityIsSetByTheReferenceNotByTheTrackingLag) {
  constexpr scalar_t gain = 2.0;
  const SwingTrajectoryPlanner descending = plannerWith(-0.05);
  const SwingTrajectoryPlanner flat = plannerWith(0.0);

  const scalar_t zRef = descending.getZpositionConstraint(0, kTouchDown);
  // A foot exactly on its reference lands at the configured rate, not at zero.
  EXPECT_NEAR(constrainedVerticalVelocity(descending, kTouchDown, gain, zRef), -0.05, kTol);
  EXPECT_NEAR(constrainedVerticalVelocity(flat, kTouchDown, gain, zRef), 0.0, kTol);

  // A foot lagging 2 cm above its reference: the descending reference contributes most of the landing speed, the flat
  // one contributes nothing and the whole landing speed is feedback.
  const scalar_t lag = 0.02;
  EXPECT_NEAR(constrainedVerticalVelocity(descending, kTouchDown, gain, zRef + lag), -0.05 - gain * lag, kTol);
  EXPECT_NEAR(constrainedVerticalVelocity(flat, kTouchDown, gain, zRef + lag), -gain * lag, kTol);

  // The gain that shipped with the stomping robot, five, turned the same lag into a 0.1 m/s kick with a flat reference.
  EXPECT_NEAR(constrainedVerticalVelocity(flat, kTouchDown, 5.0, zRef + lag), -0.1, kTol);
}

/**
 * A zero terminal slope makes the cubic flatten out asymptotically, so the sole hovers within millimetres of the
 * ground over the last stretch of the swing while still moving forward. A real descent rate keeps the foot higher over
 * that stretch and only brings it down at the end, which is the clearance the toe needs.
 */
TEST(SwingLandingVelocity, ADescentRateKeepsMoreClearanceOverTheFinalApproach) {
  const SwingTrajectoryPlanner descending = plannerWith(-0.05);
  const SwingTrajectoryPlanner flat = plannerWith(0.0);

  // Over the final 20% of the swing (excluding the touch-down instant, where both references meet the ground).
  for (scalar_t tau = 0.80; tau < 0.999; tau += 0.02) {
    const scalar_t t = kLiftOff + tau * (kTouchDown - kLiftOff);
    EXPECT_GT(descending.getZpositionConstraint(0, t), flat.getZpositionConstraint(0, t)) << "swing fraction " << tau;
    // Both descend monotonically over the final approach: neither reference asks the foot to go back up.
    EXPECT_LE(descending.getZvelocityConstraint(0, t), kTol) << "swing fraction " << tau;
    EXPECT_LE(flat.getZvelocityConstraint(0, t), kTol) << "swing fraction " << tau;
  }
  // And the two agree at touch-down itself, the descent rate does not lower the landing height.
  EXPECT_NEAR(descending.getZpositionConstraint(0, kTouchDown), flat.getZpositionConstraint(0, kTouchDown), kTol);
}

/**
 * Lift-off is the mirror image and must be untouched by the descent rate: the foot leaves the ground at the configured
 * lift-off velocity from the terrain height. The planner assigns the event instant itself to the phase that ends there,
 * so the swing is sampled just after lift-off.
 */
TEST(SwingLandingVelocity, LiftOffIsUnaffectedByTheDescentRate) {
  constexpr scalar_t kJustAfter = 1e-9;
  for (const scalar_t touchDownVelocity : {-0.05, 0.0}) {
    const SwingTrajectoryPlanner planner = plannerWith(touchDownVelocity);
    EXPECT_NEAR(planner.getZvelocityConstraint(0, kLiftOff + kJustAfter), 0.05, 1e-7);
    EXPECT_NEAR(planner.getZpositionConstraint(0, kLiftOff + kJustAfter), kTerrainHeight, 1e-7);
    // Still standing at the instant of lift-off.
    EXPECT_NEAR(planner.getZvelocityConstraint(0, kLiftOff), 0.0, kTol);
  }
}

/**
 * A swing extended past its planned touch-down (late touch-down: the foot searches for the ground) keeps the planned
 * swing's reference up to the planned touch-down and continues from there as a straight descent at the search
 * velocity, flat and with the touch-down impact proximity held. Re-fitting the spline over the extended swing, as
 * before, moved its apex: the reference rose at the current time and then descended at the spline's slope, several
 * times the search velocity, which is the opposite of a foot feeling for the ground.
 */
TEST(SwingLandingVelocity, AnExtendedSwingContinuesDownAtTheSearchVelocity) {
  constexpr scalar_t kSearchVelocity = 0.05;
  constexpr scalar_t kExtension = 0.055;
  const SwingTrajectoryPlanner::Config config = atlasConfig(-0.05);
  const SwingTrajectoryPlanner planned = plannerWith(-0.05);

  contact_flag_t swinging = makeFeetArray(true);
  swinging[0] = false;
  const size_t stance = stanceLeg2ModeNumber(makeFeetArray(true));
  const ModeSchedule extendedSchedule({kLiftOff, kTouchDown + kExtension}, {stance, stanceLeg2ModeNumber(swinging), stance});
  const scalar_array_t liftOffHeights(3, kTerrainHeight);
  const scalar_array_t touchDownHeights(3, kTerrainHeight + config.touchDownHeightOffset);
  feet_array_t<std::optional<SwingTrajectoryPlanner::GroundSearch>> searches =
      makeFeetArray(std::optional<SwingTrajectoryPlanner::GroundSearch>{});
  searches[0] = SwingTrajectoryPlanner::GroundSearch{kLiftOff, kTouchDown, kSearchVelocity};
  SwingTrajectoryPlanner extended(config, N_CONTACTS);
  extended.update(extendedSchedule, makeFeetArray(liftOffHeights), makeFeetArray(touchDownHeights), searches);

  // Up to the planned touch-down nothing changes.
  for (scalar_t t = kLiftOff + 1e-6; t <= kTouchDown; t += 0.01) {
    EXPECT_NEAR(extended.getZpositionConstraint(0, t), planned.getZpositionConstraint(0, t), kTol) << "t=" << t;
    EXPECT_NEAR(extended.getZvelocityConstraint(0, t), planned.getZvelocityConstraint(0, t), kTol) << "t=" << t;
  }
  // Past it the reference descends at the search velocity from the planned touch-down height, flat, never rising.
  const scalar_t landingHeight = kTerrainHeight + config.touchDownHeightOffset;
  for (scalar_t t = kTouchDown + 0.001; t < kTouchDown + kExtension; t += 0.005) {
    EXPECT_NEAR(extended.getZpositionConstraint(0, t), landingHeight - kSearchVelocity * (t - kTouchDown), kTol) << "t=" << t;
    EXPECT_NEAR(extended.getZvelocityConstraint(0, t), -kSearchVelocity, kTol) << "t=" << t;
    EXPECT_NEAR(extended.getZaccelerationConstraint(0, t), 0.0, kTol) << "t=" << t;
    EXPECT_NEAR(extended.getSwingPitchAngle(0, t), 0.0, kTol) << "flat while searching, t=" << t;
    EXPECT_NEAR(extended.getImpactProximityFactor(0, t), 1.0, kTol) << "the touch-down proximity is held, t=" << t;
  }
  // The foot is back in contact at the extended touch-down.
  EXPECT_NEAR(extended.getZvelocityConstraint(0, kTouchDown + kExtension + 1e-6), 0.0, kTol);

  // What re-fitting the spline over the extended swing did instead: right after the planned touch-down the reference
  // sits higher and descends much faster than the search velocity.
  SwingTrajectoryPlanner refitted(config, N_CONTACTS);
  refitted.update(extendedSchedule, kTerrainHeight);
  const scalar_t probe = kTouchDown + 0.005;
  EXPECT_GT(refitted.getZpositionConstraint(0, probe), extended.getZpositionConstraint(0, probe) + 2e-3);
  EXPECT_LT(refitted.getZvelocityConstraint(0, probe), -2.0 * kSearchVelocity);

  // A search that does not name this swing (or a zero velocity) leaves the reference at the planned touch-down height.
  searches[0] = SwingTrajectoryPlanner::GroundSearch{kLiftOff, kTouchDown, 0.0};
  extended.update(extendedSchedule, makeFeetArray(liftOffHeights), makeFeetArray(touchDownHeights), searches);
  EXPECT_NEAR(extended.getZpositionConstraint(0, kTouchDown + 0.03), landingHeight, kTol);
  EXPECT_NEAR(extended.getZvelocityConstraint(0, kTouchDown + 0.03), 0.0, kTol);
}

}  // namespace ocs2::humanoid
