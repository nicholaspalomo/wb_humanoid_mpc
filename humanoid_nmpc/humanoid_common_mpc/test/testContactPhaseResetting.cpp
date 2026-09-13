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
#include <fstream>
#include <string>
#include <vector>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

/**
 * Unit tests of the adaptive execution of a contact schedule (phase resetting on early / late touch-down, cadence
 * modulation, DCM step adjustment and the LIP helpers). Everything is written in terms of N_CONTACTS: the schedules are
 * built from contact flags and every scenario is run for every foot.
 */
namespace ocs2::humanoid {

namespace {

constexpr scalar_t kTol = 1e-9;

/** Mode number in which exactly the given feet are out of contact. */
size_t modeWithSwinging(std::initializer_list<size_t> swingingFeet) {
  contact_flag_t flags = makeFeetArray(true);
  for (size_t foot : swingingFeet) flags[foot] = false;
  return stanceLeg2ModeNumber(flags);
}

const size_t kAllInContact = modeWithSwinging({});

/** STANCE | foot swings [liftOff, touchDown] | STANCE | (other feet may follow) ... ending in STANCE. */
ModeSchedule singleSwingSchedule(size_t foot, scalar_t liftOff, scalar_t touchDown) {
  return ModeSchedule({liftOff, touchDown}, {kAllInContact, modeWithSwinging({foot}), kAllInContact});
}

/**
 * Two consecutive steps: `foot` swings [1.0, 1.4], double support until 1.5, `other` swings [1.5, 1.9], double support
 * from 1.9 on. With a single foot `other` == `foot` and the second swing is a repeated swing of the same foot.
 */
ModeSchedule twoStepSchedule(size_t foot, size_t other) {
  return ModeSchedule({1.0, 1.4, 1.5, 1.9},
                      {kAllInContact, modeWithSwinging({foot}), kAllInContact, modeWithSwinging({other}), kAllInContact});
}

size_t otherFoot(size_t foot) {
  return (foot + 1) % N_CONTACTS;
}

ContactPlanningConfig makeConfig() {
  ContactPlanningConfig config;
  config.dt = 0.1;
  config.minSwingDuration = 0.3;
  config.maxSwingDuration = 0.6;
  config.enablePhaseResetting = true;
  config.earlyTouchdownMinSwingRatio = 0.25;
  config.maxLateTouchdownExtension = 0.15;
  config.lateTouchdownExtensionStep = 0.05;
  config.enableEnergyCadenceModulation = false;
  config.validate();
  return config;
}

bool strictlyIncreasing(const std::vector<scalar_t>& times) {
  for (size_t i = 1; i < times.size(); ++i) {
    if (times[i] <= times[i - 1]) return false;
  }
  return true;
}

bool hasFlightPhase(const ModeSchedule& schedule) {
  for (size_t mode : schedule.modeSequence) {
    const contact_flag_t flags = modeNumber2StanceLeg(mode);
    bool any = false;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) any = any || flags[foot];
    if (!any) return true;
  }
  return false;
}

bool consistentSchedule(const ModeSchedule& schedule) {
  return schedule.modeSequence.size() == schedule.eventTimes.size() + 1 && strictlyIncreasing(schedule.eventTimes);
}

/** A valid plan with constant ZMP (CoM at rest over the ZMP is a LIP equilibrium) and a foothold per foot. */
ContactPlan makePlan(
    scalar_t startTime, scalar_t dt, int numIntervals, const vector2_t& zmp, const vector2_t& com, const vector2_t& comVelocity) {
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = startTime;
  plan.dt = dt;
  plan.contacts.assign(numIntervals, makeFeetArray(true));
  plan.zmp.assign(numIntervals, zmp);
  plan.comPosition.resize(numIntervals + 1);
  plan.comVelocity.resize(numIntervals + 1);
  plan.footholds.assign(numIntervals + 1, makeFeetArray(vector2_t(vector2_t::Zero())));
  const scalar_t omega = std::sqrt(9.81 / 0.85);
  for (int k = 0; k <= numIntervals; ++k) {
    // Exact LIP propagation with constant ZMP so that the node states are consistent with the closed form.
    const scalar_t tau = dt * static_cast<scalar_t>(k);
    const scalar_t ch = std::cosh(omega * tau), sh = std::sinh(omega * tau);
    plan.comPosition[k] = zmp + (com - zmp) * ch + comVelocity * (sh / omega);
    plan.comVelocity[k] = (com - zmp) * (omega * sh) + comVelocity * ch;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) plan.footholds[k][foot] = vector2_t(0.3, 0.1 - 0.2 * static_cast<scalar_t>(foot));
  }
  return plan;
}

}  // namespace

/*============================================ schedule queries ============================================*/

TEST(ContactScheduleQueries, EventAtQueryTimeCountsAsPassed) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);
    EXPECT_EQ(modeIndexAtTime(schedule, 0.5), 0u);
    EXPECT_EQ(modeIndexAtTime(schedule, 1.0), 1u) << "lift-off at the query time is passed";
    EXPECT_EQ(modeIndexAtTime(schedule, 1.2), 1u);
    EXPECT_EQ(modeIndexAtTime(schedule, 1.4), 2u) << "touch-down at the query time is passed";
    EXPECT_EQ(modeIndexAtTime(schedule, 9.0), 2u);
    EXPECT_FALSE(contactFlagsAtTime(schedule, 1.0)[foot]);
    EXPECT_TRUE(contactFlagsAtTime(schedule, 1.4)[foot]);
    EXPECT_TRUE(contactFlagsAtTime(schedule, 0.9)[foot]);
    // ocs2's ModeSchedule uses the opposite convention; the queries here must not silently inherit it.
    EXPECT_EQ(schedule.modeAtTime(1.4), modeWithSwinging({foot}));
  }
  const ModeSchedule defaultSchedule;  // one mode, no event
  EXPECT_EQ(modeIndexAtTime(defaultSchedule, 3.0), 0u);
  ModeSchedule empty;
  empty.clear();
  EXPECT_EQ(modeIndexAtTime(empty, 3.0), 0u);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) EXPECT_TRUE(contactFlagsAtTime(empty, 0.0)[foot]);
}

TEST(ContactScheduleQueries, SwingPhaseLookup) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);
    const auto phase = swingPhaseAtTime(schedule, foot, 1.2);
    ASSERT_TRUE(phase.has_value());
    EXPECT_NEAR(phase->first, 1.0, kTol);
    EXPECT_NEAR(phase->second, 1.4, kTol);
    EXPECT_FALSE(swingPhaseAtTime(schedule, foot, 0.5).has_value());
    EXPECT_FALSE(swingPhaseAtTime(schedule, foot, 1.4).has_value());
    for (size_t other = 0; other < N_CONTACTS; ++other) {
      if (other != foot) EXPECT_FALSE(swingPhaseAtTime(schedule, other, 1.2).has_value());
    }
    const auto range = swingPhaseIndexRange(schedule, foot, 1.2);
    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->first, 1u);
    EXPECT_EQ(range->second, 1u);
    const auto tdIndex = touchDownEventIndex(schedule, foot, 1.2);
    ASSERT_TRUE(tdIndex.has_value());
    EXPECT_EQ(*tdIndex, 1u);
  }
}

TEST(ContactScheduleQueries, SwingWithoutLiftOffOrTouchDownEvent) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const ModeSchedule noLiftOff({1.4}, {modeWithSwinging({foot}), kAllInContact});
    EXPECT_FALSE(swingPhaseAtTime(noLiftOff, foot, 1.0).has_value());
    EXPECT_TRUE(touchDownEventIndex(noLiftOff, foot, 1.0).has_value());
    const ModeSchedule noTouchDown({1.0}, {kAllInContact, modeWithSwinging({foot})});
    EXPECT_FALSE(swingPhaseAtTime(noTouchDown, foot, 1.2).has_value());
    EXPECT_FALSE(touchDownEventIndex(noTouchDown, foot, 1.2).has_value());
    EXPECT_TRUE(swingPhaseIndexRange(noTouchDown, foot, 1.2).has_value());
  }
}

TEST(ContactScheduleQueries, SwingSpanningSeveralPhases) {
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  // foot 0 in the air over [1.0, 1.6] while foot 1 lifts at 1.2 and lands at 1.4 (a flight phase, only for the query).
  const ModeSchedule schedule({1.0, 1.2, 1.4, 1.6},
                              {kAllInContact, modeWithSwinging({0}), modeWithSwinging({0, 1}), modeWithSwinging({0}), kAllInContact});
  const auto phase = swingPhaseAtTime(schedule, 0, 1.3);
  ASSERT_TRUE(phase.has_value());
  EXPECT_NEAR(phase->first, 1.0, kTol);
  EXPECT_NEAR(phase->second, 1.6, kTol);
  const auto range = swingPhaseIndexRange(schedule, 0, 1.3);
  ASSERT_TRUE(range.has_value());
  EXPECT_EQ(range->first, 1u);
  EXPECT_EQ(range->second, 3u);
  EXPECT_EQ(*touchDownEventIndex(schedule, 0, 1.1), 3u);
  const auto inner = swingPhaseAtTime(schedule, 1, 1.3);
  ASSERT_TRUE(inner.has_value());
  EXPECT_NEAR(inner->first, 1.2, kTol);
  EXPECT_NEAR(inner->second, 1.4, kTol);
}

/*============================================ schedule edits ==============================================*/

TEST(ContactScheduleEdits, RemoveRedundantEvents) {
  ModeSchedule schedule({1.0, 1.4, 1.5, 2.0}, {kAllInContact, kAllInContact, modeWithSwinging({0}), kAllInContact, kAllInContact});
  removeRedundantEvents(schedule);
  ASSERT_EQ(schedule.modeSequence.size(), 3u);
  EXPECT_EQ(schedule.eventTimes, (std::vector<scalar_t>{1.4, 1.5}));
  EXPECT_EQ(schedule.modeSequence, (std::vector<size_t>{kAllInContact, modeWithSwinging({0}), kAllInContact}));
  ModeSchedule allSame({1.0, 2.0}, {kAllInContact, kAllInContact, kAllInContact});
  removeRedundantEvents(allSame);
  EXPECT_TRUE(allSame.eventTimes.empty());
  EXPECT_EQ(allSame.modeSequence, std::vector<size_t>{kAllInContact});
}

TEST(ContactScheduleEdits, TruncateSwingMidPhase) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    ModeSchedule schedule = twoStepSchedule(foot, otherFoot(foot));
    const auto cut = truncateSwingPhase(schedule, foot, 1.25);
    ASSERT_TRUE(cut.has_value());
    EXPECT_NEAR(*cut, 1.4, kTol);
    EXPECT_TRUE(consistentSchedule(schedule));
    // The foot is in contact from 1.25 on; the old touch-down event vanished; the later step is untouched.
    EXPECT_FALSE(contactFlagsAtTime(schedule, 1.2)[foot]);
    EXPECT_TRUE(contactFlagsAtTime(schedule, 1.25)[foot]);
    EXPECT_TRUE(contactFlagsAtTime(schedule, 1.3)[foot]);
    EXPECT_EQ(schedule.eventTimes, (std::vector<scalar_t>{1.0, 1.25, 1.5, 1.9}));
    if (N_CONTACTS > 1) {
      EXPECT_FALSE(contactFlagsAtTime(schedule, 1.6)[otherFoot(foot)]);
      EXPECT_TRUE(contactFlagsAtTime(schedule, 1.95)[otherFoot(foot)]);
    }
    EXPECT_FALSE(hasFlightPhase(schedule));
  }
}

TEST(ContactScheduleEdits, TruncateSwingAtPhaseStartDoesNotDuplicateEvent) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);
    ASSERT_TRUE(truncateSwingPhase(schedule, foot, 1.0).has_value());
    EXPECT_TRUE(consistentSchedule(schedule));
    EXPECT_TRUE(schedule.eventTimes.empty()) << "the whole swing collapsed into stance";
    EXPECT_EQ(schedule.modeSequence, std::vector<size_t>{kAllInContact});
  }
}

TEST(ContactScheduleEdits, TruncateSwingRejectsStanceAndMissingTouchDown) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);
    const ModeSchedule before = schedule;
    EXPECT_FALSE(truncateSwingPhase(schedule, foot, 0.5).has_value());
    EXPECT_FALSE(truncateSwingPhase(schedule, foot, 1.4).has_value());
    EXPECT_EQ(schedule.eventTimes, before.eventTimes);
    EXPECT_EQ(schedule.modeSequence, before.modeSequence);
    ModeSchedule noTouchDown({1.0}, {kAllInContact, modeWithSwinging({foot})});
    EXPECT_FALSE(truncateSwingPhase(noTouchDown, foot, 1.2).has_value());
    EXPECT_EQ(noTouchDown.eventTimes.size(), 1u);
  }
}

TEST(ContactScheduleEdits, TruncateSwingSpanningSeveralPhasesKeepsOtherFeet) {
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  ModeSchedule schedule({1.0, 1.2, 1.4, 1.6},
                        {kAllInContact, modeWithSwinging({0}), modeWithSwinging({0, 1}), modeWithSwinging({0}), kAllInContact});
  ASSERT_TRUE(truncateSwingPhase(schedule, 0, 1.1).has_value());
  EXPECT_TRUE(consistentSchedule(schedule));
  for (scalar_t t : {1.1, 1.3, 1.5, 1.7}) EXPECT_TRUE(contactFlagsAtTime(schedule, t)[0]) << t;
  EXPECT_TRUE(contactFlagsAtTime(schedule, 1.1)[1]);
  EXPECT_FALSE(contactFlagsAtTime(schedule, 1.3)[1]);
  EXPECT_TRUE(contactFlagsAtTime(schedule, 1.5)[1]);
  EXPECT_EQ(schedule.eventTimes, (std::vector<scalar_t>{1.0, 1.1, 1.2, 1.4}));
}

TEST(ContactScheduleEdits, ShiftEventsPreservesLaterDurations) {
  ModeSchedule schedule = twoStepSchedule(0, otherFoot(0));
  EXPECT_TRUE(shiftEventsFrom(schedule, 1, 0.07));
  EXPECT_EQ(schedule.eventTimes.size(), 4u);
  EXPECT_NEAR(schedule.eventTimes[0], 1.0, kTol);
  EXPECT_NEAR(schedule.eventTimes[1], 1.47, kTol);
  EXPECT_NEAR(schedule.eventTimes[2], 1.57, kTol);
  EXPECT_NEAR(schedule.eventTimes[3], 1.97, kTol);
  EXPECT_TRUE(shiftEventsFrom(schedule, 1, -0.07));
  EXPECT_NEAR(schedule.eventTimes[3], 1.9, kTol);
  EXPECT_FALSE(shiftEventsFrom(schedule, 1, -0.4)) << "would move the touch-down before the lift-off";
  EXPECT_NEAR(schedule.eventTimes[1], 1.4, kTol);
  EXPECT_FALSE(shiftEventsFrom(schedule, 4, 0.1)) << "out of range";
  EXPECT_TRUE(shiftEventsFrom(schedule, 0, -5.0)) << "the first event has no predecessor";
  EXPECT_TRUE(consistentSchedule(schedule));
}

/*============================================ contact events ==============================================*/

class ContactEventTest : public ::testing::Test {
 protected:
  ContactPlanningConfig config = makeConfig();
  feet_array_t<SwingTimingLatch> latches = makeFeetArray(SwingTimingLatch{});
  feet_array_t<scalar_t> noCadence = makeFeetArray(0.0);

  contact_flag_t measured(size_t swingingFoot, bool inContact) {
    contact_flag_t flags = makeFeetArray(true);
    flags[swingingFoot] = inContact;
    return flags;
  }

  feet_array_t<ContactEventReport> step(ModeSchedule& schedule, scalar_t time, const contact_flag_t& contact) {
    return adaptScheduleToContactEvents(schedule, time, contact, noCadence, config, latches);
  }

  static void expectNoEventExcept(const feet_array_t<ContactEventReport>& reports, size_t foot) {
    for (size_t i = 0; i < N_CONTACTS; ++i) {
      if (i != foot) EXPECT_EQ(reports[i].type, ContactEventReport::Type::NONE) << "foot " << i;
    }
  }
};

TEST_F(ContactEventTest, NothingHappensWhenContactMatchesSchedule) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = twoStepSchedule(foot, otherFoot(foot));
    const ModeSchedule before = schedule;
    for (scalar_t t : {0.5, 1.0, 1.2, 1.39}) {
      const auto reports = step(schedule, t, measured(foot, false));
      for (size_t i = 0; i < N_CONTACTS; ++i) EXPECT_EQ(reports[i].type, ContactEventReport::Type::NONE);
    }
    EXPECT_TRUE(latches[foot].active);
    EXPECT_NEAR(latches[foot].liftOffTime, 1.0, kTol);
    EXPECT_NEAR(latches[foot].nominalTouchDownTime, 1.4, kTol);
    const auto landed = step(schedule, 1.4, measured(foot, true));
    EXPECT_EQ(landed[foot].type, ContactEventReport::Type::NONE);
    EXPECT_FALSE(latches[foot].active) << "a swing that landed as scheduled releases its latch";
    EXPECT_EQ(schedule.eventTimes, before.eventTimes);
    EXPECT_EQ(schedule.modeSequence, before.modeSequence);
  }
}

TEST_F(ContactEventTest, EarlyTouchDownTruncatesSwingInPlace) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = twoStepSchedule(foot, otherFoot(foot));
    step(schedule, 1.1, measured(foot, false));  // in flight, latch created
    const auto reports = step(schedule, 1.25, measured(foot, true));
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::EARLY_TOUCH_DOWN);
    EXPECT_NEAR(reports[foot].touchDownTime, 1.25, kTol);
    EXPECT_NEAR(reports[foot].timeShift, 0.0, kTol);
    expectNoEventExcept(reports, foot);
    EXPECT_FALSE(latches[foot].active);
    EXPECT_TRUE(consistentSchedule(schedule));
    EXPECT_TRUE(contactFlagsAtTime(schedule, 1.25)[foot]);
    EXPECT_FALSE(contactFlagsAtTime(schedule, 1.24)[foot]);
    EXPECT_EQ(schedule.eventTimes, (std::vector<scalar_t>{1.0, 1.25, 1.5, 1.9})) << "later events keep their timing";
    // The next cycle sees a foot in contact as scheduled: nothing more happens, even if the sensor flickers.
    EXPECT_EQ(step(schedule, 1.27, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_EQ(step(schedule, 1.29, measured(foot, false))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_EQ(schedule.eventTimes, (std::vector<scalar_t>{1.0, 1.25, 1.5, 1.9}));
  }
}

TEST_F(ContactEventTest, EarlyTouchDownIgnoredDuringScuffingWindowButAcceptedWhenContactPersists) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);  // nominal duration 0.4, window 0.1
    EXPECT_EQ(step(schedule, 1.02, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_EQ(step(schedule, 1.09, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_FALSE(contactFlagsAtTime(schedule, 1.09)[foot]);
    // Level-triggered: contact that persists past the window is a landing even though it started inside the window.
    EXPECT_EQ(step(schedule, 1.1, measured(foot, true))[foot].type, ContactEventReport::Type::EARLY_TOUCH_DOWN);
    EXPECT_TRUE(contactFlagsAtTime(schedule, 1.1)[foot]);
  }
}

TEST_F(ContactEventTest, EarlyTouchDownWindowUsesNominalDurationEvenAfterCadenceShift) {
  config.enableEnergyCadenceModulation = true;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);
    feet_array_t<scalar_t> cadence = makeFeetArray(0.0);
    cadence[foot] = 0.2;  // touch-down 1.6 (within maxSwingDuration)
    adaptScheduleToContactEvents(schedule, 1.05, measured(foot, false), cadence, config, latches);
    EXPECT_NEAR(latches[foot].plannedTouchDownTime(), 1.6, kTol);
    EXPECT_NEAR(latches[foot].nominalTouchDownTime, 1.4, kTol);
    // The window is 25 % of the nominal 0.4 s, i.e. contact at 1.1 counts although the swing now lasts until 1.6.
    const auto reports = adaptScheduleToContactEvents(schedule, 1.1, measured(foot, true), cadence, config, latches);
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::EARLY_TOUCH_DOWN);
  }
}

TEST_F(ContactEventTest, PhaseResettingDisabledLeavesScheduleAlone) {
  config.enablePhaseResetting = false;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);
    const ModeSchedule before = schedule;
    EXPECT_EQ(step(schedule, 1.2, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_EQ(step(schedule, 1.41, measured(foot, false))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_EQ(schedule.eventTimes, before.eventTimes);
    EXPECT_EQ(schedule.modeSequence, before.modeSequence);
  }
}

TEST_F(ContactEventTest, LateTouchDownExtendsInStepsUpToBudgetThenGivesUp) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = twoStepSchedule(foot, otherFoot(foot));
    step(schedule, 1.2, measured(foot, false));

    // 1.405: scheduled in contact since 1.4, no contact measured -> touch-down pushed to 1.455, tail shifted by 0.055.
    auto reports = step(schedule, 1.405, measured(foot, false));
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::LATE_TOUCH_DOWN);
    EXPECT_NEAR(reports[foot].touchDownTime, 1.455, kTol);
    EXPECT_NEAR(reports[foot].timeShift, 0.055, kTol);
    expectNoEventExcept(reports, foot);
    EXPECT_EQ(schedule.eventTimes.size(), 4u);
    EXPECT_NEAR(schedule.eventTimes[1], 1.455, kTol);
    EXPECT_NEAR(schedule.eventTimes[2], 1.555, kTol) << "the double support keeps its duration";
    EXPECT_NEAR(schedule.eventTimes[3], 1.955, kTol);
    EXPECT_NEAR(latches[foot].lateExtension, 0.055, kTol);
    EXPECT_FALSE(hasFlightPhase(schedule));

    // While the extended swing is in flight nothing changes.
    EXPECT_EQ(step(schedule, 1.43, measured(foot, false))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_NEAR(schedule.eventTimes[1], 1.455, kTol);

    // 1.46: again past the touch-down -> 1.51. 1.52: -> capped at 1.4 + 0.15 = 1.55.
    reports = step(schedule, 1.46, measured(foot, false));
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::LATE_TOUCH_DOWN);
    EXPECT_NEAR(schedule.eventTimes[1], 1.51, kTol);
    reports = step(schedule, 1.52, measured(foot, false));
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::LATE_TOUCH_DOWN);
    EXPECT_NEAR(schedule.eventTimes[1], 1.55, kTol);
    EXPECT_NEAR(reports[foot].timeShift, 0.04, kTol);
    EXPECT_NEAR(latches[foot].lateExtension, 0.15, kTol);

    // 1.56: the budget is used up, the contact phase proceeds and the latch is released.
    reports = step(schedule, 1.56, measured(foot, false));
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::NONE);
    EXPECT_FALSE(latches[foot].active);
    EXPECT_NEAR(schedule.eventTimes[1], 1.55, kTol);
    EXPECT_TRUE(contactFlagsAtTime(schedule, 1.56)[foot]);
    EXPECT_EQ(step(schedule, 1.58, measured(foot, false))[foot].type, ContactEventReport::Type::NONE) << "no restart without a latch";
    EXPECT_TRUE(consistentSchedule(schedule));
    EXPECT_NEAR(schedule.eventTimes[3] - schedule.eventTimes[2], 0.4, kTol) << "the next swing keeps its duration";
  }
}

TEST_F(ContactEventTest, ContactDuringLateExtensionEndsSwingImmediately) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = twoStepSchedule(foot, otherFoot(foot));
    step(schedule, 1.2, measured(foot, false));
    ASSERT_EQ(step(schedule, 1.405, measured(foot, false))[foot].type, ContactEventReport::Type::LATE_TOUCH_DOWN);
    const auto reports = step(schedule, 1.43, measured(foot, true));
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::EARLY_TOUCH_DOWN);
    EXPECT_TRUE(contactFlagsAtTime(schedule, 1.43)[foot]);
    EXPECT_FALSE(contactFlagsAtTime(schedule, 1.42)[foot]);
    EXPECT_FALSE(latches[foot].active);
    // The later step keeps the timing it got from the extension (the plan is shifted alongside by the caller).
    ASSERT_EQ(schedule.eventTimes.size(), 4u);
    EXPECT_NEAR(schedule.eventTimes[0], 1.0, kTol);
    EXPECT_NEAR(schedule.eventTimes[1], 1.43, kTol);
    EXPECT_NEAR(schedule.eventTimes[2], 1.555, kTol);
    EXPECT_NEAR(schedule.eventTimes[3], 1.955, kTol);
    EXPECT_TRUE(consistentSchedule(schedule));
  }
}

TEST_F(ContactEventTest, LateTouchDownNeedsALatchedSwing) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);
    const ModeSchedule before = schedule;
    // The swing was never observed in flight (e.g. the controller started after it): no extension is attempted.
    EXPECT_EQ(step(schedule, 1.405, measured(foot, false))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_EQ(schedule.eventTimes, before.eventTimes);
    // A stale latch from another swing is dropped.
    latches[foot].active = true;
    latches[foot].liftOffTime = 0.2;
    latches[foot].nominalTouchDownTime = 0.6;
    EXPECT_EQ(step(schedule, 1.405, measured(foot, false))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_FALSE(latches[foot].active);
    EXPECT_EQ(schedule.eventTimes, before.eventTimes);
  }
}

TEST_F(ContactEventTest, CadenceShiftMovesTouchDownRelativeToNominalWithinLimits) {
  config.enableEnergyCadenceModulation = true;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = twoStepSchedule(foot, otherFoot(foot));  // swing [1.0, 1.4], limits [1.3, 1.6]
    feet_array_t<scalar_t> cadence = makeFeetArray(0.0);

    cadence[foot] = 0.1;
    auto reports = adaptScheduleToContactEvents(schedule, 1.1, measured(foot, false), cadence, config, latches);
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::CADENCE_SHIFT);
    EXPECT_NEAR(reports[foot].timeShift, 0.1, kTol);
    EXPECT_NEAR(schedule.eventTimes[1], 1.5, kTol);
    EXPECT_NEAR(schedule.eventTimes[3], 2.0, kTol) << "later events move with the touch-down";

    // Relative to the nominal touch-down, not cumulative: the same request leaves the schedule unchanged.
    reports = adaptScheduleToContactEvents(schedule, 1.12, measured(foot, false), cadence, config, latches);
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::NONE);
    EXPECT_NEAR(schedule.eventTimes[1], 1.5, kTol);

    // A shorter request moves it back, clamped to the minimum swing duration.
    cadence[foot] = -0.5;
    reports = adaptScheduleToContactEvents(schedule, 1.14, measured(foot, false), cadence, config, latches);
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::CADENCE_SHIFT);
    EXPECT_NEAR(schedule.eventTimes[1], 1.3, kTol);
    EXPECT_NEAR(schedule.eventTimes[2], 1.4, kTol);
    EXPECT_NEAR(latches[foot].cadenceShift, -0.1, kTol);

    // Clamped to the maximum swing duration.
    cadence[foot] = 1.0;
    adaptScheduleToContactEvents(schedule, 1.16, measured(foot, false), cadence, config, latches);
    EXPECT_NEAR(schedule.eventTimes[1], 1.6, kTol);

    // Never re-timed into the past: at 1.59 the earliest admissible touch-down is 1.61 > minimum swing.
    cadence[foot] = -0.5;
    reports = adaptScheduleToContactEvents(schedule, 1.59, measured(foot, false), cadence, config, latches);
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::CADENCE_SHIFT);
    EXPECT_NEAR(schedule.eventTimes[1], 1.61, kTol);
    EXPECT_TRUE(consistentSchedule(schedule));
    EXPECT_FALSE(hasFlightPhase(schedule));
  }
}

TEST_F(ContactEventTest, CadenceShiftIgnoredWhenDisabledOrWhileSearchingForGround) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);
    feet_array_t<scalar_t> cadence = makeFeetArray(0.1);
    config.enableEnergyCadenceModulation = false;
    EXPECT_EQ(adaptScheduleToContactEvents(schedule, 1.1, measured(foot, false), cadence, config, latches)[foot].type,
              ContactEventReport::Type::NONE);
    EXPECT_NEAR(schedule.eventTimes[1], 1.4, kTol);

    config.enableEnergyCadenceModulation = true;
    ASSERT_EQ(adaptScheduleToContactEvents(schedule, 1.405, measured(foot, false), cadence, config, latches)[foot].type,
              ContactEventReport::Type::LATE_TOUCH_DOWN);
    const scalar_t extended = schedule.eventTimes[1];
    EXPECT_EQ(adaptScheduleToContactEvents(schedule, 1.42, measured(foot, false), cadence, config, latches)[foot].type,
              ContactEventReport::Type::NONE);
    EXPECT_NEAR(schedule.eventTimes[1], extended, kTol);
  }
}

TEST_F(ContactEventTest, EmptyScheduleIsHarmless) {
  ModeSchedule empty;
  empty.clear();
  const auto reports = step(empty, 1.0, makeFeetArray(true));
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) EXPECT_EQ(reports[foot].type, ContactEventReport::Type::NONE);
}

TEST_F(ContactEventTest, AlternatingGaitSimulationStaysConsistent) {
  // Feet take turns; every second swing lands early, every third misses the ground for a while.
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  std::vector<scalar_t> events;
  std::vector<size_t> modes{kAllInContact};
  scalar_t t = 1.0;
  for (int step = 0; step < 8; ++step) {
    const size_t foot = static_cast<size_t>(step) % N_CONTACTS;
    events.push_back(t);
    modes.push_back(modeWithSwinging({foot}));
    events.push_back(t + 0.4);
    modes.push_back(kAllInContact);
    t += 0.5;
  }
  ModeSchedule schedule(events, modes);
  int early = 0, late = 0;
  for (scalar_t time = 0.5; time < 6.0; time += 0.02) {
    contact_flag_t contact = makeFeetArray(true);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const auto phase = swingPhaseAtTime(schedule, foot, time);
      if (phase.has_value()) {
        const int swingNumber = static_cast<int>(std::floor((phase->first - 1.0) / 0.5 + 0.5));
        const bool landsEarly = swingNumber % 2 == 1 && time - phase->first > 0.3;
        contact[foot] = landsEarly;
      } else {
        const auto& latch = latches[foot];
        const bool searching = latch.active && static_cast<int>(std::floor((latch.liftOffTime - 1.0) / 0.5 + 0.5)) % 3 == 2 &&
                               time < latch.plannedTouchDownTime() + 0.08;
        contact[foot] = !searching;
      }
    }
    const auto reports = step(schedule, time, contact);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (reports[foot].type == ContactEventReport::Type::EARLY_TOUCH_DOWN) ++early;
      if (reports[foot].type == ContactEventReport::Type::LATE_TOUCH_DOWN) ++late;
    }
    ASSERT_TRUE(consistentSchedule(schedule)) << "at t=" << time;
    ASSERT_FALSE(hasFlightPhase(schedule)) << "at t=" << time;
    // Every phase of every foot keeps a positive duration and a swing never lasts longer than max + extension.
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const auto phase = swingPhaseAtTime(schedule, foot, time);
      if (phase.has_value()) ASSERT_LE(phase->second - phase->first, config.maxSwingDuration + config.maxLateTouchdownExtension + kTol);
    }
  }
  EXPECT_GT(early, 0);
  EXPECT_GT(late, 0);
}

/*============================================ LIP helpers =================================================*/

TEST(LipHelpers, ReferenceStateMatchesClosedFormAndNodes) {
  const scalar_t omega = std::sqrt(9.81 / 0.85);
  const ContactPlan plan = makePlan(2.0, 0.1, 6, vector2_t(0.1, 0.0), vector2_t(0.15, 0.02), vector2_t(0.3, -0.1));
  for (int k = 0; k <= 6; ++k) {
    const auto state = lipReferenceState(plan, omega, 2.0 + 0.1 * k);
    ASSERT_TRUE(state.has_value());
    EXPECT_TRUE(state->com.isApprox(plan.comPosition[k], 1e-9)) << k;
    EXPECT_TRUE(state->comVelocity.isApprox(plan.comVelocity[k], 1e-9)) << k;
    EXPECT_TRUE(state->zmp.isApprox(plan.zmp[std::min(k, 5)], 1e-12));
  }
  // Between nodes the DCM follows xi(t) = z + (xi_k - z) exp(omega tau), and the CoM lies between the node values.
  const auto mid = lipReferenceState(plan, omega, 2.05);
  ASSERT_TRUE(mid.has_value());
  const vector2_t dcm0 = computeDcm(plan.comPosition[0], plan.comVelocity[0], omega);
  const vector2_t expected = plan.zmp[0] + (dcm0 - plan.zmp[0]) * std::exp(omega * 0.05);
  EXPECT_TRUE(computeDcm(mid->com, mid->comVelocity, omega).isApprox(expected, 1e-9));
  EXPECT_FALSE(lipReferenceState(plan, omega, 1.9).has_value());
  EXPECT_FALSE(lipReferenceState(plan, omega, 2.7).has_value());
  ContactPlan invalid = plan;
  invalid.valid = false;
  EXPECT_FALSE(lipReferenceState(invalid, omega, 2.1).has_value());
  ContactPlan inconsistent = plan;
  inconsistent.zmp.pop_back();
  EXPECT_FALSE(lipReferenceState(inconsistent, omega, 2.1).has_value());
}

TEST(LipHelpers, OrbitalEnergyIsConservedAlongTheReference) {
  const scalar_t omega = std::sqrt(9.81 / 0.85);
  const ContactPlan plan = makePlan(0.0, 0.1, 8, vector2_t(0.0, 0.0), vector2_t(-0.1, 0.0), vector2_t(0.4, 0.0));
  const scalar_t e0 = lipOrbitalEnergy(-0.1, 0.4, omega, 100.0);
  for (scalar_t t = 0.0; t <= 0.8; t += 0.037) {
    const auto state = lipReferenceState(plan, omega, t);
    ASSERT_TRUE(state.has_value());
    EXPECT_NEAR(lipOrbitalEnergy(state->com(0) - state->zmp(0), state->comVelocity(0), omega, 100.0), e0, 1e-8) << t;
  }
  EXPECT_GT(lipOrbitalEnergy(-0.1, 0.5, omega, 100.0), e0) << "more speed, more energy";
  EXPECT_LT(lipOrbitalEnergy(-0.2, 0.4, omega, 100.0), e0) << "further from the ZMP, less energy";
}

TEST(LipHelpers, StepAdjustmentPropagatesToTouchDownAndIsBounded) {
  const scalar_t omega = 3.4;
  const vector2_t error(0.01, -0.02);
  EXPECT_TRUE(dcmStepAdjustment(vector2_t::Zero(), omega, 0.3, 1.0, 0.15).isZero());
  EXPECT_TRUE(dcmStepAdjustment(error, omega, 0.0, 1.0, 0.15).isApprox(error, 1e-12)) << "at touch-down the error maps 1:1";
  EXPECT_TRUE(dcmStepAdjustment(error, omega, 0.3, 1.0, 0.15).isApprox(error * std::exp(omega * 0.3), 1e-12));
  EXPECT_TRUE(dcmStepAdjustment(error, omega, 0.3, 0.5, 0.15).isApprox(0.5 * error * std::exp(omega * 0.3), 1e-12));
  EXPECT_TRUE(dcmStepAdjustment(error, omega, -0.5, 1.0, 0.15).isApprox(error, 1e-12)) << "a past touch-down is not propagated";
  const vector2_t large = dcmStepAdjustment(vector2_t(0.2, 0.0), omega, 0.5, 1.0, 0.15);
  EXPECT_NEAR(large.norm(), 0.15, 1e-12);
  EXPECT_GT(large(0), 0.0);
  EXPECT_TRUE(dcmStepAdjustment(error, omega, 0.3, 1.0, 0.0).isZero()) << "a zero bound switches the adjustment off";
  EXPECT_TRUE(dcmStepAdjustment(error, omega, 0.3, 0.0, 0.15).isZero()) << "a zero gain switches the adjustment off";
}

TEST(LipHelpers, ReachClippingInYawFrameKeepsSideOfBody) {
  ContactPlanningConfig config = makeConfig();
  config.reachX = 0.4;
  config.reachYInner = 0.05;
  config.reachYOuter = 0.3;
  const vector2_t com(1.0, 2.0);
  for (scalar_t yaw : {0.0, 0.7, -2.0}) {
    const vector2_t ex(std::cos(yaw), std::sin(yaw)), ey(-std::sin(yaw), std::cos(yaw));
    const vector2_t nominalLeft = com + 0.1 * ex + 0.15 * ey;
    const vector2_t nominalRight = com + 0.1 * ex - 0.15 * ey;
    // Inside the region: unchanged.
    EXPECT_TRUE(clipFootholdToReach(nominalLeft, nominalLeft, com, yaw, config).isApprox(nominalLeft, 1e-12));
    // Too far forward: clipped to reachX along the heading, lateral offset kept.
    const vector2_t forward = clipFootholdToReach(com + 0.9 * ex + 0.15 * ey, nominalLeft, com, yaw, config);
    EXPECT_NEAR(ex.dot(forward - com), 0.4, 1e-12);
    EXPECT_NEAR(ey.dot(forward - com), 0.15, 1e-12);
    // Too far back.
    EXPECT_NEAR(ex.dot(clipFootholdToReach(com - 0.9 * ex + 0.15 * ey, nominalLeft, com, yaw, config) - com), -0.4, 1e-12);
    // Across the body: a left foot never crosses to the right of the CoM, a right foot never to the left.
    const vector2_t crossedLeft = clipFootholdToReach(com - 0.2 * ey, nominalLeft, com, yaw, config);
    EXPECT_NEAR(ey.dot(crossedLeft - com), 0.05, 1e-12);
    const vector2_t crossedRight = clipFootholdToReach(com + 0.2 * ey, nominalRight, com, yaw, config);
    EXPECT_NEAR(ey.dot(crossedRight - com), -0.05, 1e-12);
    // Too far outward.
    EXPECT_NEAR(ey.dot(clipFootholdToReach(com + 0.8 * ey, nominalLeft, com, yaw, config) - com), 0.3, 1e-12);
    EXPECT_NEAR(ey.dot(clipFootholdToReach(com - 0.8 * ey, nominalRight, com, yaw, config) - com), -0.3, 1e-12);
  }
}

/*============================================ plan and config ============================================*/

TEST(ContactPlanShift, ShiftInTimeKeepsFootholdLookupRelativeToEvents) {
  ContactPlan plan = makePlan(1.0, 0.1, 5, vector2_t::Zero(), vector2_t::Zero(), vector2_t::Zero());
  plan.committedUntil = 1.3;
  for (int k = 0; k <= 5; ++k) plan.footholds[k][0] = vector2_t(0.1 * k, 0.0);
  const vector2_t before = *plan.footholdAtTime(0, 1.3);
  plan.shiftInTime(0.07);
  EXPECT_NEAR(plan.startTime, 1.07, kTol);
  EXPECT_NEAR(plan.committedUntil, 1.37, kTol);
  EXPECT_NEAR(plan.endTime(), 1.57, kTol);
  EXPECT_TRUE(plan.footholdAtTime(0, 1.37)->isApprox(before, 1e-12));
  EXPECT_EQ(plan.toModeSchedule().eventTimes.size(), 0u);
}

TEST(ContactPlanningConfigAdaptive, DefaultsAreValidAndLoadable) {
  ContactPlanningConfig config;
  EXPECT_NO_THROW(config.validate());
  const std::string file = testing::TempDir() + "/adaptive_contact_planning.yaml";
  {
    std::ofstream out(file);
    out << "contact_planning:\n"
        << "  enablePhaseResetting: false\n"
        << "  earlyTouchdownMinSwingRatio: 0.4\n"
        << "  maxLateTouchdownExtension: 0.2\n"
        << "  lateTouchdownExtensionStep: 0.02\n"
        << "  lateTouchdownSearchVelocity: 0.08\n"
        << "  enableDcmStepAdjustment: false\n"
        << "  dcmAdjustmentGain: 0.7\n"
        << "  dcmAdjustmentMaxOffset: 0.1\n"
        << "  enableEnergyCadenceModulation: true\n"
        << "  energyCadenceGain: 0.02\n";
  }
  const ContactPlanningConfig loaded = loadContactPlanningConfig(file, "contact_planning.", false);
  std::remove(file.c_str());
  EXPECT_FALSE(loaded.enablePhaseResetting);
  EXPECT_NEAR(loaded.earlyTouchdownMinSwingRatio, 0.4, kTol);
  EXPECT_NEAR(loaded.maxLateTouchdownExtension, 0.2, kTol);
  EXPECT_NEAR(loaded.lateTouchdownExtensionStep, 0.02, kTol);
  EXPECT_NEAR(loaded.lateTouchdownSearchVelocity, 0.08, kTol);
  EXPECT_FALSE(loaded.enableDcmStepAdjustment);
  EXPECT_NEAR(loaded.dcmAdjustmentGain, 0.7, kTol);
  EXPECT_NEAR(loaded.dcmAdjustmentMaxOffset, 0.1, kTol);
  EXPECT_TRUE(loaded.enableEnergyCadenceModulation);
  EXPECT_NEAR(loaded.energyCadenceGain, 0.02, kTol);
  EXPECT_NEAR(loaded.dt, config.dt, kTol) << "missing keys keep their defaults";
}

TEST(ContactPlanningConfigAdaptive, ValidationRejectsBadValues) {
  const auto rejects = [](auto mutate) {
    ContactPlanningConfig config;
    mutate(config);
    EXPECT_THROW(config.validate(), std::invalid_argument);
  };
  rejects([](ContactPlanningConfig& c) { c.earlyTouchdownMinSwingRatio = -0.1; });
  rejects([](ContactPlanningConfig& c) { c.earlyTouchdownMinSwingRatio = 1.1; });
  rejects([](ContactPlanningConfig& c) { c.maxLateTouchdownExtension = -0.1; });
  rejects([](ContactPlanningConfig& c) { c.lateTouchdownExtensionStep = 0.0; });
  rejects([](ContactPlanningConfig& c) { c.lateTouchdownSearchVelocity = -1.0; });
  rejects([](ContactPlanningConfig& c) { c.dcmAdjustmentGain = -1.0; });
  rejects([](ContactPlanningConfig& c) { c.dcmAdjustmentMaxOffset = -0.1; });
  rejects([](ContactPlanningConfig& c) { c.energyCadenceGain = -0.1; });
  ContactPlanningConfig zeros;
  zeros.earlyTouchdownMinSwingRatio = 0.0;
  zeros.maxLateTouchdownExtension = 0.0;
  zeros.lateTouchdownSearchVelocity = 0.0;
  zeros.dcmAdjustmentGain = 0.0;
  zeros.dcmAdjustmentMaxOffset = 0.0;
  zeros.energyCadenceGain = 0.0;
  EXPECT_NO_THROW(zeros.validate());
}

}  // namespace ocs2::humanoid
