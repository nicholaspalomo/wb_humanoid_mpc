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
#include <initializer_list>

#include "humanoid_common_mpc/contact_planning/TargetContactPose.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

/**
 * Unit tests of the target contact poses the MuJoCo viewer draws (TargetContactPose.h). Written in terms of
 * N_CONTACTS: every scenario is run for every foot.
 */
namespace ocs2::humanoid {

namespace {

constexpr scalar_t kTol = 1e-9;

size_t modeWithSwinging(std::initializer_list<size_t> swingingFeet) {
  contact_flag_t flags = makeFeetArray(true);
  for (size_t foot : swingingFeet) flags[foot] = false;
  return stanceLeg2ModeNumber(flags);
}

const size_t kAllInContact = modeWithSwinging({});

/** STANCE | foot swings [liftOff, touchDown] | STANCE. */
ModeSchedule singleSwingSchedule(size_t foot, scalar_t liftOff, scalar_t touchDown) {
  return ModeSchedule({liftOff, touchDown}, {kAllInContact, modeWithSwinging({foot}), kAllInContact});
}

/**
 * A valid plan from 0 s in 0.1 s nodes. The footholds move by 0.1 m per node along x so that a lookup at the wrong node
 * is visible, and foot i sits at y = 0.1 - 0.2 i. With the heading model the foot yaws are 0.05 k + 0.01 i at node k.
 */
ContactPlan makePlan(bool heading) {
  constexpr int numIntervals = 25;
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 0.0;
  plan.dt = 0.1;
  plan.yaw = 0.3;
  plan.contacts.assign(numIntervals, makeFeetArray(true));
  plan.footholds.resize(numIntervals + 1);
  for (int k = 0; k <= numIntervals; ++k) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      plan.footholds[k][foot] = vector2_t(0.1 * k, 0.1 - 0.2 * static_cast<scalar_t>(foot));
    }
  }
  if (heading) {
    plan.heading.assign(numIntervals + 1, 0.3);
    plan.headingRate.assign(numIntervals + 1, 0.0);
    plan.footYaws.resize(numIntervals + 1);
    for (int k = 0; k <= numIntervals; ++k) {
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) plan.footYaws[k][foot] = 0.05 * k + 0.01 * static_cast<scalar_t>(foot);
    }
  }
  return plan;
}

scalar_t plannedYaw(int node, size_t foot) {
  return 0.05 * node + 0.01 * static_cast<scalar_t>(foot);
}

/** Measured feet somewhere else entirely, so that a stance pose is distinguishable from any planned foothold. */
TargetContactPoseInputs makeInputs(scalar_t time) {
  TargetContactPoseInputs inputs;
  inputs.time = time;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    inputs.footPositions[foot] = vector3_t(-1.0, 0.5 - static_cast<scalar_t>(foot), 0.02 * static_cast<scalar_t>(foot + 1));
    inputs.footYaws[foot] = -0.7 + 0.1 * static_cast<scalar_t>(foot);
  }
  inputs.terrainHeight = 0.15;
  return inputs;
}

void expectStancePlacement(const TargetContactPose& pose, size_t foot, const TargetContactPoseInputs& inputs) {
  EXPECT_TRUE(pose.valid);
  EXPECT_EQ(pose.kind, TargetContactPose::Kind::STANCE);
  EXPECT_NEAR(pose.position(0), inputs.footPositions[foot](0), kTol);
  EXPECT_NEAR(pose.position(1), inputs.footPositions[foot](1), kTol);
  EXPECT_NEAR(pose.height, inputs.footPositions[foot](2), kTol) << "a stance foot sits at its measured height";
  EXPECT_NEAR(pose.yaw, inputs.footYaws[foot], kTol);
  EXPECT_FALSE(pose.yawPlanned);
  EXPECT_NEAR(pose.touchDownTime, inputs.time, kTol);
}

}  // namespace

TEST(TargetContactPose, InvalidPlanGivesNoTargets) {
  ContactPlan plan = makePlan(true);
  plan.valid = false;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const auto poses = computeTargetContactPoses(plan, singleSwingSchedule(foot, 1.0, 1.4), makeInputs(1.2));
    for (const TargetContactPose& pose : poses) EXPECT_FALSE(pose.valid);
  }
  const auto poses = computeTargetContactPoses(ContactPlan{}, ModeSchedule(), makeInputs(0.0));
  for (const TargetContactPose& pose : poses) EXPECT_FALSE(pose.valid);
}

TEST(TargetContactPose, SwingInFlightTargetsThePlannedLandingWithTheStepAdjustment) {
  const ContactPlan plan = makePlan(true);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    TargetContactPoseInputs inputs = makeInputs(1.2);
    inputs.dcmStepAdjustment[foot] = vector2_t(0.02, -0.01);
    const auto poses = computeTargetContactPoses(plan, singleSwingSchedule(foot, 1.0, 1.4), inputs);

    const TargetContactPose& pose = poses[foot];
    EXPECT_TRUE(pose.valid);
    EXPECT_EQ(pose.kind, TargetContactPose::Kind::SWING_IN_FLIGHT);
    EXPECT_NEAR(pose.touchDownTime, 1.4, kTol);
    // Foothold at the touch-down node (1.4 s -> node 14) plus the step adjustment.
    EXPECT_NEAR(pose.position(0), 1.4 + 0.02, kTol);
    EXPECT_NEAR(pose.position(1), 0.1 - 0.2 * static_cast<scalar_t>(foot) - 0.01, kTol);
    EXPECT_NEAR(pose.height, inputs.terrainHeight, kTol) << "a landing target sits on the ground";
    EXPECT_TRUE(pose.yawPlanned);
    EXPECT_NEAR(pose.yaw, plannedYaw(14, foot), kTol);

    for (size_t other = 0; other < N_CONTACTS; ++other) {
      if (other != foot) expectStancePlacement(poses[other], other, inputs);
    }
  }
}

TEST(TargetContactPose, StanceFootWithAnUpcomingSwingTargetsItsNextLanding) {
  const ContactPlan plan = makePlan(true);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);
    const TargetContactPoseInputs inputs = makeInputs(0.5);
    const TargetContactPose& pose = computeTargetContactPoses(plan, schedule, inputs)[foot];
    EXPECT_TRUE(pose.valid);
    EXPECT_EQ(pose.kind, TargetContactPose::Kind::NEXT_SWING);
    EXPECT_NEAR(pose.touchDownTime, 1.4, kTol);
    EXPECT_NEAR(pose.position(0), 1.4, kTol);
    EXPECT_NEAR(pose.position(1), 0.1 - 0.2 * static_cast<scalar_t>(foot), kTol);
    EXPECT_NEAR(pose.height, inputs.terrainHeight, kTol);
    EXPECT_TRUE(pose.yawPlanned);
    EXPECT_NEAR(pose.yaw, plannedYaw(14, foot), kTol);

    // At the lift-off instant the event has passed: the same landing, now as the swing in flight.
    const TargetContactPose& atLiftOff = computeTargetContactPoses(plan, schedule, makeInputs(1.0))[foot];
    EXPECT_EQ(atLiftOff.kind, TargetContactPose::Kind::SWING_IN_FLIGHT);
    EXPECT_NEAR(atLiftOff.position(0), 1.4, kTol);

    // After the touch-down there is no further swing: the foot's placement.
    const TargetContactPoseInputs later = makeInputs(2.0);
    expectStancePlacement(computeTargetContactPoses(plan, schedule, later)[foot], foot, later);
  }
}

TEST(TargetContactPose, WithoutTheHeadingModelTheYawIsTheMeasuredFootYaw) {
  const ContactPlan plan = makePlan(false);
  ASSERT_FALSE(plan.hasHeading());
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const TargetContactPoseInputs inputs = makeInputs(1.2);
    const TargetContactPose& pose = computeTargetContactPoses(plan, singleSwingSchedule(foot, 1.0, 1.4), inputs)[foot];
    EXPECT_EQ(pose.kind, TargetContactPose::Kind::SWING_IN_FLIGHT);
    EXPECT_NEAR(pose.position(0), 1.4, kTol);
    EXPECT_FALSE(pose.yawPlanned);
    EXPECT_NEAR(pose.yaw, inputs.footYaws[foot], kTol);
  }
}

TEST(TargetContactPose, SwingWithoutTouchDownEventFallsBackToTheStancePlacement) {
  const ContactPlan plan = makePlan(true);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const ModeSchedule noTouchDown({1.0}, {kAllInContact, modeWithSwinging({foot})});
    for (scalar_t time : {0.5, 1.2}) {
      const TargetContactPoseInputs inputs = makeInputs(time);
      expectStancePlacement(computeTargetContactPoses(plan, noTouchDown, inputs)[foot], foot, inputs);
    }
  }
}

TEST(TargetContactPose, TwoStepSequenceGivesEachFootItsOwnLanding) {
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  const ContactPlan plan = makePlan(true);
  // foot 0 swings [1.0, 1.4], double support, foot 1 swings [1.5, 1.9].
  const ModeSchedule schedule({1.0, 1.4, 1.5, 1.9},
                              {kAllInContact, modeWithSwinging({0}), kAllInContact, modeWithSwinging({1}), kAllInContact});
  const auto poses = computeTargetContactPoses(plan, schedule, makeInputs(1.2));
  EXPECT_EQ(poses[0].kind, TargetContactPose::Kind::SWING_IN_FLIGHT);
  EXPECT_NEAR(poses[0].touchDownTime, 1.4, kTol);
  EXPECT_NEAR(poses[0].position(0), 1.4, kTol);
  EXPECT_NEAR(poses[0].yaw, plannedYaw(14, 0), kTol);
  EXPECT_EQ(poses[1].kind, TargetContactPose::Kind::NEXT_SWING);
  EXPECT_NEAR(poses[1].touchDownTime, 1.9, kTol);
  EXPECT_NEAR(poses[1].position(0), 1.9, kTol);
  EXPECT_NEAR(poses[1].position(1), -0.1, kTol);
  EXPECT_NEAR(poses[1].yaw, plannedYaw(19, 1), kTol);
}

}  // namespace ocs2::humanoid
