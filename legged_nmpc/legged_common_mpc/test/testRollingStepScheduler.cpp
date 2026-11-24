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

// The event-driven steps of RollingStepScheduler on the four parallel wheels of
// a quadruped and on the two coaxial wheels of a biped: a straight drive never
// steps, a sideways command steps where a wheel's lateral debt reaches its
// bound, one group at a time with the one leading the motion first, a turn
// steps the quadruped and not the biped (whose axle may pass off the base),
// committed swings keep their times, and the mode schedule is the swings'.

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/btree_map.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/gait/RollingStepScheduler.h"

namespace ocs2::legged {
namespace {

// [m] The quadruped's contacts from the base center, and the biped's.
constexpr scalar_t kQuadrupedHalfLength = 0.36;
constexpr scalar_t kHalfTrack = 0.25;
// The quadruped's mode bits: LF = 8, RF = 4, LH = 2, RH = 1.
constexpr size_t kQuadrupedStance = 15;
constexpr size_t kBipedStance = 3;
constexpr size_t kNumQuadrupedContacts = 4;
constexpr size_t kNumBipedContacts = 2;
// LF and RH in the air (RF and LH stand), and RF and LH in the air.
constexpr size_t kLfRhSwingMode = 6;
constexpr size_t kRfLhSwingMode = 9;
// [m], [s]
constexpr scalar_t kMaxLateralDebt = 0.05;
constexpr scalar_t kSwingDuration = 0.3;
constexpr scalar_t kMinStanceDuration = 0.15;
constexpr scalar_t kMinLiftOffLead = 0.1;
constexpr scalar_t kCommitWindow = 0.25;
// [s] The solver's period and horizon.
constexpr scalar_t kSolvePeriod = 0.01;
constexpr scalar_t kHorizon = 1.0;
constexpr scalar_t kTimeTolerance = 1.0e-9;

RollingStepSchedulerSettings settingsOf(
    std::vector<std::vector<size_t>> groups) {
  return RollingStepSchedulerSettings{.stepGroups = std::move(groups),
                                      .maxLateralDebt = kMaxLateralDebt,
                                      .swingDuration = kSwingDuration,
                                      .minStanceDuration = kMinStanceDuration,
                                      .minLiftOffLead = kMinLiftOffLead,
                                      .commitWindow = kCommitWindow};
}

/** LF, RF, LH, RH, every wheel rolling along the heading; steps in diagonal
 * pairs LF + RH and RF + LH. */
std::vector<RollingContactGeometry> quadrupedGeometry() {
  const vector2_t forward(1.0, 0.0);
  return {{.contactIndex = 0,
           .nomPosContactRtBaseEwrtHeadingXy =
               vector2_t(kQuadrupedHalfLength, kHalfTrack),
           .dirWheelEwrtHeadingXy = forward},
          {.contactIndex = 1,
           .nomPosContactRtBaseEwrtHeadingXy =
               vector2_t(kQuadrupedHalfLength, -kHalfTrack),
           .dirWheelEwrtHeadingXy = forward},
          {.contactIndex = 2,
           .nomPosContactRtBaseEwrtHeadingXy =
               vector2_t(-kQuadrupedHalfLength, kHalfTrack),
           .dirWheelEwrtHeadingXy = forward},
          {.contactIndex = 3,
           .nomPosContactRtBaseEwrtHeadingXy =
               vector2_t(-kQuadrupedHalfLength, -kHalfTrack),
           .dirWheelEwrtHeadingXy = forward}};
}

RollingStepScheduler quadruped() {
  return RollingStepScheduler(settingsOf({{0, 3}, {1, 2}}),
                              kNumQuadrupedContacts, kQuadrupedStance);
}

/** Left and right wheels on one axle through the base center; the right
 * wheel's joint axis points the other way, so it rolls backward for a positive
 * joint velocity. */
std::vector<RollingContactGeometry> bipedGeometry() {
  return {{.contactIndex = 0,
           .nomPosContactRtBaseEwrtHeadingXy = vector2_t(0.0, kHalfTrack),
           .dirWheelEwrtHeadingXy = vector2_t(1.0, 0.0)},
          {.contactIndex = 1,
           .nomPosContactRtBaseEwrtHeadingXy = vector2_t(0.0, -kHalfTrack),
           .dirWheelEwrtHeadingXy = vector2_t(-1.0, 0.0)}};
}

RollingStepScheduler biped() {
  return RollingStepScheduler(settingsOf({{0}, {1}}), kNumBipedContacts,
                              kBipedStance);
}

/** Every committed swing of updates every kSolvePeriod over [start, end) with
 * the command `cmd`, by lift-off time. */
absl::btree_map<scalar_t, RollingStepSwing> runCommitted(
    RollingStepScheduler& scheduler,
    absl::Span<const RollingContactGeometry> geometry,
    const vector3_t& cmd,
    scalar_t start,
    scalar_t end) {
  absl::btree_map<scalar_t, RollingStepSwing> committed;
  for (scalar_t time = start; time < end - kTimeTolerance;
       time += kSolvePeriod) {
    scheduler.update(time, time + kHorizon, cmd, geometry);
    for (const RollingStepSwing& swing : scheduler.swings()) {
      if (swing.committed) committed.try_emplace(swing.liftOffTime, swing);
    }
  }
  return committed;
}

TEST(RollingStepScheduler, TheDebtRateIsTheNominalVelocityAlongTheAxle) {
  const RollingContactGeometry frontLeft = quadrupedGeometry()[0];
  // Sideways at 0.2 m/s: all of it along the axle.
  const vector2_t nomPosFeetmeanRtBaseEwrtHeadingXy = vector2_t::Zero();
  EXPECT_DOUBLE_EQ(
      RollingStepScheduler::lateralDebtRate(vector3_t(0.0, 0.2, 0.0), frontLeft,
                                            nomPosFeetmeanRtBaseEwrtHeadingXy),
      0.2);
  // Forward and the turn's part along the heading build none; a turn moves
  // the front wheel sideways at omega x.
  EXPECT_DOUBLE_EQ(
      RollingStepScheduler::lateralDebtRate(vector3_t(0.7, 0.0, 0.5), frontLeft,
                                            nomPosFeetmeanRtBaseEwrtHeadingXy),
      0.5 * kQuadrupedHalfLength);
  // About a pivot level with the wheel, the turn moves it along its heading.
  EXPECT_DOUBLE_EQ(RollingStepScheduler::lateralDebtRate(
                       vector3_t(0.7, 0.0, 0.5), frontLeft,
                       vector2_t(kQuadrupedHalfLength, 0.0)),
                   0.0);
  // A wheel rolling the other way measures its debt along the opposite axle.
  const RollingContactGeometry right = bipedGeometry()[1];
  EXPECT_DOUBLE_EQ(
      RollingStepScheduler::lateralDebtRate(vector3_t(0.0, 0.2, 0.0), right,
                                            nomPosFeetmeanRtBaseEwrtHeadingXy),
      -0.2);
}

TEST(RollingStepScheduler, AStraightDriveNeverSteps) {
  RollingStepScheduler scheduler = quadruped();
  const absl::btree_map<scalar_t, RollingStepSwing> committed =
      runCommitted(scheduler, quadrupedGeometry(), vector3_t(0.5, 0.0, 0.0),
                   /*start=*/0.0, /*end=*/3.0);
  EXPECT_TRUE(committed.empty());
  EXPECT_TRUE(scheduler.swings().empty());
  for (const scalar_t debt : scheduler.lateralDebts()) EXPECT_EQ(debt, 0.0);
}

TEST(RollingStepScheduler, ABipedTurnsInPlaceWithoutStepping) {
  RollingStepScheduler scheduler = biped();
  const absl::btree_map<scalar_t, RollingStepSwing> committed =
      runCommitted(scheduler, bipedGeometry(), vector3_t(0.3, 0.0, 1.0),
                   /*start=*/0.0, /*end=*/3.0);
  EXPECT_TRUE(committed.empty());
}

TEST(RollingStepScheduler, ABipedTurnsAboutAnAxleInFrontOfTheBase) {
  // Tron1's axle stands a few centimeters off its base: the turn is about the
  // axle, which builds no debt.
  constexpr scalar_t kAxleOffset = 0.05;
  std::vector<RollingContactGeometry> geometry = bipedGeometry();
  for (RollingContactGeometry& wheel : geometry) {
    wheel.nomPosContactRtBaseEwrtHeadingXy.x() = kAxleOffset;
  }
  RollingStepScheduler scheduler = biped();
  const absl::btree_map<scalar_t, RollingStepSwing> committed =
      runCommitted(scheduler, geometry, vector3_t(0.0, 0.0, 1.0),
                   /*start=*/0.0, /*end=*/3.0);
  EXPECT_TRUE(committed.empty());
  for (const scalar_t debt : scheduler.lateralDebts()) EXPECT_EQ(debt, 0.0);
}

TEST(RollingStepScheduler, ASidewaysCommandStepsWhereTheDebtReachesItsBound) {
  RollingStepScheduler scheduler = quadruped();
  constexpr scalar_t kSideSpeed = 0.1;
  const ModeSchedule first =
      scheduler.update(/*initTime=*/0.0, kHorizon,
                       vector3_t(0.0, kSideSpeed, 0.0), quadrupedGeometry());
  ASSERT_FALSE(scheduler.swings().empty());
  // Both pairs are due at D / v; the one of the lower index goes first, and
  // the other after its swing and the stance that follows it.
  const RollingStepSwing lead = scheduler.swings()[0];
  EXPECT_EQ(lead.group, 0U);
  EXPECT_NEAR(lead.liftOffTime, kMaxLateralDebt / kSideSpeed, kTimeTolerance);
  EXPECT_NEAR(lead.touchDownTime - lead.liftOffTime, kSwingDuration,
              kTimeTolerance);
  EXPECT_FALSE(lead.committed);
  ASSERT_GE(scheduler.swings().size(), 2U);
  EXPECT_EQ(scheduler.swings()[1].group, 1U);
  EXPECT_NEAR(scheduler.swings()[1].liftOffTime,
              lead.touchDownTime + kMinStanceDuration, kTimeTolerance);
  // The schedule: stance, the LF + RH swing, stance, the RF + LH swing, ...
  ASSERT_GE(first.modeSequence.size(), 4U);
  EXPECT_EQ(first.modeSequence[0], kQuadrupedStance);
  EXPECT_EQ(first.modeSequence[1], kLfRhSwingMode);
  EXPECT_EQ(first.modeSequence[2], kQuadrupedStance);
  EXPECT_EQ(first.modeSequence[3], kRfLhSwingMode);
}

TEST(RollingStepScheduler, TheGroupsStepOneAtATimeAndAlternate) {
  RollingStepScheduler scheduler = quadruped();
  const absl::btree_map<scalar_t, RollingStepSwing> committed =
      runCommitted(scheduler, quadrupedGeometry(), vector3_t(0.0, 0.3, 0.0),
                   /*start=*/0.0, /*end=*/4.0);
  ASSERT_GE(committed.size(), 4U);
  const RollingStepSwing* absl_nullable previous = nullptr;
  for (const std::pair<const scalar_t, RollingStepSwing>& entry : committed) {
    const RollingStepSwing& swing = entry.second;
    if (previous != nullptr) {
      EXPECT_NE(swing.group, previous->group);
      EXPECT_GE(swing.liftOffTime,
                previous->touchDownTime + kMinStanceDuration - kTimeTolerance);
    }
    previous = &swing;
  }
}

TEST(RollingStepScheduler, ATurnStepsTheQuadrupedsDiagonals) {
  RollingStepScheduler scheduler = quadruped();
  const absl::btree_map<scalar_t, RollingStepSwing> committed =
      runCommitted(scheduler, quadrupedGeometry(), vector3_t(0.0, 0.0, 0.5),
                   /*start=*/0.0, /*end=*/3.0);
  EXPECT_GE(committed.size(), 2U);
}

TEST(RollingStepScheduler, TheWheelLeadingTheMotionStepsFirst) {
  for (const scalar_t sideSpeed : {0.2, -0.2}) {
    SCOPED_TRACE(sideSpeed);
    RollingStepScheduler scheduler = biped();
    scheduler.update(/*initTime=*/0.0, kHorizon, vector3_t(0.0, sideSpeed, 0.0),
                     bipedGeometry());
    ASSERT_FALSE(scheduler.swings().empty());
    // The left wheel leads a step to the left, the right wheel one to the
    // right.
    EXPECT_EQ(scheduler.swings()[0].group, sideSpeed > 0.0 ? 0U : 1U);
  }
}

TEST(RollingStepScheduler, ACommittedSwingKeepsItsTimesWhenTheCommandStops) {
  RollingStepScheduler scheduler = quadruped();
  const std::vector<RollingContactGeometry> geometry = quadrupedGeometry();
  // The first swing is due at 0.5 s and committed from 0.25 s on.
  const absl::btree_map<scalar_t, RollingStepSwing> committed =
      runCommitted(scheduler, geometry, vector3_t(0.0, 0.1, 0.0), /*start=*/0.0,
                   /*end=*/0.3);
  ASSERT_EQ(committed.size(), 1U);
  const RollingStepSwing swing = committed.begin()->second;
  scheduler.update(/*initTime=*/0.3, 0.3 + kHorizon, vector3_t::Zero(),
                   geometry);
  ASSERT_EQ(scheduler.swings().size(), 1U);
  EXPECT_EQ(scheduler.swings()[0].group, swing.group);
  EXPECT_EQ(scheduler.swings()[0].liftOffTime, swing.liftOffTime);
  EXPECT_EQ(scheduler.swings()[0].touchDownTime, swing.touchDownTime);
}

TEST(RollingStepScheduler, ALandingClearsTheDebtOfItsGroup) {
  RollingStepScheduler scheduler = quadruped();
  const std::vector<RollingContactGeometry> geometry = quadrupedGeometry();
  constexpr scalar_t kSideSpeed = 0.1;
  // LF + RH land at 0.8 s, and RF + LH lift off at 0.95 s.
  runCommitted(scheduler, geometry, vector3_t(0.0, kSideSpeed, 0.0),
               /*start=*/0.0, /*end=*/0.9);
  scheduler.update(/*initTime=*/0.9, 0.9 + kHorizon,
                   vector3_t(0.0, kSideSpeed, 0.0), geometry);
  // LF + RH have built 0.1 s of debt since they landed, RF + LH 0.9 s.
  EXPECT_NEAR(scheduler.lateralDebts()[0], kSideSpeed * 0.1, kTimeTolerance);
  EXPECT_NEAR(scheduler.lateralDebts()[3], kSideSpeed * 0.1, kTimeTolerance);
  EXPECT_NEAR(scheduler.lateralDebts()[1], kSideSpeed * 0.9, kTimeTolerance);
}

TEST(RollingStepScheduler, AClockThatRunsBackwardsStartsAfresh) {
  RollingStepScheduler scheduler = quadruped();
  const std::vector<RollingContactGeometry> geometry = quadrupedGeometry();
  runCommitted(scheduler, geometry, vector3_t(0.0, 0.1, 0.0), /*start=*/0.0,
               /*end=*/2.0);
  scheduler.update(/*initTime=*/0.0, kHorizon, vector3_t::Zero(), geometry);
  EXPECT_TRUE(scheduler.swings().empty());
  for (const scalar_t debt : scheduler.lateralDebts()) EXPECT_EQ(debt, 0.0);
}

TEST(RollingStepScheduler, TheModeScheduleIsTheSwings) {
  RollingStepScheduler scheduler = quadruped();
  const std::vector<RollingContactGeometry> geometry = quadrupedGeometry();
  runCommitted(scheduler, geometry, vector3_t(0.0, 0.4, 0.2), /*start=*/0.0,
               /*end=*/1.5);
  const ModeSchedule schedule = scheduler.update(
      /*initTime=*/1.5, 1.5 + kHorizon, vector3_t(0.0, 0.4, 0.2), geometry);
  const size_t numSwings = scheduler.swings().size();
  ASSERT_GT(numSwings, 0U);
  // Two events per swing and the stance event after the horizon.
  ASSERT_EQ(schedule.eventTimes.size(), 2 * numSwings + 1);
  ASSERT_EQ(schedule.modeSequence.size(), schedule.eventTimes.size() + 1);
  for (size_t i = 0; i + 1 < schedule.eventTimes.size(); ++i) {
    EXPECT_LT(schedule.eventTimes[i], schedule.eventTimes[i + 1]);
  }
  EXPECT_GT(schedule.eventTimes.back(), 1.5 + kHorizon);
  for (size_t swing = 0; swing < numSwings; ++swing) {
    EXPECT_EQ(schedule.eventTimes[2 * swing],
              scheduler.swings()[swing].liftOffTime);
    EXPECT_EQ(
        schedule.modeSequence[2 * swing + 1],
        scheduler.swings()[swing].group == 0 ? kLfRhSwingMode : kRfLhSwingMode);
    EXPECT_EQ(schedule.modeSequence[2 * swing + 2], kQuadrupedStance);
  }
  EXPECT_EQ(schedule.modeSequence.front(), kQuadrupedStance);
  EXPECT_EQ(schedule.modeSequence.back(), kQuadrupedStance);
}

TEST(RollingStepScheduler, AFastCommandStepsBackToBack) {
  RollingStepScheduler scheduler = quadruped();
  // Each pair builds 2 m/s * 0.9 s of debt per cycle, far above the bound:
  // every swing follows the last after exactly the stance between them.
  const absl::btree_map<scalar_t, RollingStepSwing> committed =
      runCommitted(scheduler, quadrupedGeometry(), vector3_t(0.0, 2.0, 0.0),
                   /*start=*/0.0, /*end=*/3.0);
  ASSERT_GE(committed.size(), 3U);
  const RollingStepSwing* absl_nullable previous = nullptr;
  for (const std::pair<const scalar_t, RollingStepSwing>& entry : committed) {
    if (previous != nullptr) {
      EXPECT_NEAR(entry.second.liftOffTime,
                  previous->touchDownTime + kMinStanceDuration, kTimeTolerance);
    }
    previous = &entry.second;
  }
}

TEST(RollingStepScheduler, OverdueGroupsAlternate) {
  // At 1 rad/s a wheel builds 0.36 m/s * 0.15 s = 5.4 cm of debt during the
  // stance between two swings, which is more than the 5 cm bound. A group that
  // has just landed is then overdue again when the next swing is planned. The
  // other group has the larger debt and steps next.
  for (const vector3_t& cmd :
       {vector3_t(0.0, 0.0, 1.0), vector3_t(0.0, 0.0, -1.0),
        vector3_t(1.0, 0.0, 1.0), vector3_t(0.0, 2.0, 0.0)}) {
    SCOPED_TRACE(testing::Message() << "command " << cmd.transpose());
    RollingStepScheduler scheduler = quadruped();
    const absl::btree_map<scalar_t, RollingStepSwing> committed =
        runCommitted(scheduler, quadrupedGeometry(), cmd, /*start=*/0.0,
                     /*end=*/5.0);
    ASSERT_GE(committed.size(), 10U);
    const RollingStepSwing* absl_nullable previous = nullptr;
    for (const std::pair<const scalar_t, RollingStepSwing>& entry : committed) {
      if (previous != nullptr) {
        EXPECT_NE(entry.second.group, previous->group);
      }
      previous = &entry.second;
    }
  }
}

}  // namespace
}  // namespace ocs2::legged
