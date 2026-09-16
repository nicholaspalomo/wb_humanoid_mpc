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
#include <deque>
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
  config.planner.dt = 0.1;
  config.shared.gaitLimits.minSwingDuration = 0.3;
  config.shared.gaitLimits.maxSwingDuration = 0.6;
  config.formulation.setExecutionRule(term::kPhaseResetting, true);
  config.phaseResetting.earlyTouchdownMinSwingRatio = 0.25;
  config.phaseResetting.earlyTouchdownMinContactDuration = 0.0;  // the timing tests below act on a single cycle; the debounce has its own
  config.phaseResetting.maxLateTouchdownExtension = 0.15;
  config.phaseResetting.lateTouchdownExtensionStep = 0.05;
  config.formulation.setExecutionRule(term::kEnergyCadenceModulation, false);
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

TEST_F(ContactEventTest, EarlyTouchDownRequiresContactToPersistForTheDebounceDuration) {
  config.phaseResetting.earlyTouchdownMinContactDuration = 0.04;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);  // scuffing window ends at 1.1
    EXPECT_EQ(step(schedule, 1.20, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_TRUE(latches[foot].contactObserved);
    EXPECT_NEAR(latches[foot].contactObservedSince, 1.20, kTol);
    EXPECT_EQ(step(schedule, 1.22, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_FALSE(contactFlagsAtTime(schedule, 1.22)[foot]) << "the swing must not end before the contact has persisted";
    const auto reports = step(schedule, 1.24, measured(foot, true));
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::EARLY_TOUCH_DOWN);
    EXPECT_NEAR(reports[foot].touchDownTime, 1.24, kTol) << "the schedule switches at the current cycle, not retroactively";
    EXPECT_TRUE(contactFlagsAtTime(schedule, 1.24)[foot]);
    EXPECT_FALSE(latches[foot].active);
  }
}

TEST_F(ContactEventTest, SingleChatteringContactSampleDoesNotEndTheSwing) {
  config.phaseResetting.earlyTouchdownMinContactDuration = 0.04;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);
    EXPECT_EQ(step(schedule, 1.20, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_EQ(step(schedule, 1.22, measured(foot, false))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_FALSE(latches[foot].contactObserved) << "a lost contact restarts the debounce";
    // The timer restarts with the next contact sample: 1.24 -> 1.26 is only 0.02 s, 1.28 completes the 0.04 s.
    EXPECT_EQ(step(schedule, 1.24, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_EQ(step(schedule, 1.26, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_FALSE(contactFlagsAtTime(schedule, 1.26)[foot]);
    EXPECT_EQ(step(schedule, 1.28, measured(foot, true))[foot].type, ContactEventReport::Type::EARLY_TOUCH_DOWN);
    EXPECT_TRUE(contactFlagsAtTime(schedule, 1.28)[foot]);
  }
}

TEST_F(ContactEventTest, DebounceStartsOnlyAfterTheScuffingWindow) {
  config.phaseResetting.earlyTouchdownMinContactDuration = 0.04;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);  // window 0.25 * 0.4 = 0.1
    // Contact from lift-off on: ignored as scuffing inside the window, and the debounce timer does not run there.
    EXPECT_EQ(step(schedule, 1.02, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_EQ(step(schedule, 1.06, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_FALSE(latches[foot].contactObserved);
    EXPECT_EQ(step(schedule, 1.10, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_TRUE(latches[foot].contactObserved);
    EXPECT_NEAR(latches[foot].contactObservedSince, 1.10, kTol);
    EXPECT_EQ(step(schedule, 1.12, measured(foot, true))[foot].type, ContactEventReport::Type::NONE);
    EXPECT_EQ(step(schedule, 1.14, measured(foot, true))[foot].type, ContactEventReport::Type::EARLY_TOUCH_DOWN);
  }
}

TEST_F(ContactEventTest, ZeroDebounceEndsTheSwingOnTheFirstSample) {
  config.phaseResetting.earlyTouchdownMinContactDuration = 0.0;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);
    EXPECT_EQ(step(schedule, 1.2, measured(foot, true))[foot].type, ContactEventReport::Type::EARLY_TOUCH_DOWN);
  }
}

TEST_F(ContactEventTest, EarlyTouchDownWindowUsesNominalDurationEvenAfterCadenceShift) {
  config.formulation.setExecutionRule(term::kEnergyCadenceModulation, true);
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
  config.formulation.setExecutionRule(term::kPhaseResetting, false);
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
  config.formulation.setExecutionRule(term::kEnergyCadenceModulation, true);
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

    // A request earlier than what is still feasible never pushes an imminent touch-down out: at 1.59 the touch-down at
    // 1.6 stands (flooring it at now + margin every cycle would drag it along with the clock, see the dedicated test).
    cadence[foot] = -0.5;
    reports = adaptScheduleToContactEvents(schedule, 1.59, measured(foot, false), cadence, config, latches);
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::NONE);
    EXPECT_NEAR(schedule.eventTimes[1], 1.6, kTol);
    EXPECT_TRUE(consistentSchedule(schedule));
    EXPECT_FALSE(hasFlightPhase(schedule));
  }
}

TEST_F(ContactEventTest, CadenceRequestEarlierThanFeasibleLetsTheFootLand) {
  // A constant request for an earlier touch-down than the remaining swing allows must not keep the touch-down "now +
  // margin" ahead of the clock: the foot would then never land. Swing [1.1, 1.5], request -0.10 -> wanted 1.40.
  // Phase resetting is off: with no measured contact it would (correctly) start the late touch-down search once the
  // re-timed touch-down has passed, which is a different mechanism from the one under test.
  config.formulation.setExecutionRule(term::kEnergyCadenceModulation, true);
  config.formulation.setExecutionRule(term::kPhaseResetting, false);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = ModeSchedule(
        {1.1, 1.5, 1.6, 2.0}, {kAllInContact, modeWithSwinging({foot}), kAllInContact, modeWithSwinging({otherFoot(foot)}), kAllInContact});
    feet_array_t<scalar_t> cadence = makeFeetArray(0.0);
    cadence[foot] = -0.10;
    auto reports = adaptScheduleToContactEvents(schedule, 1.30, measured(foot, false), cadence, config, latches);
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::CADENCE_SHIFT);
    EXPECT_NEAR(schedule.eventTimes[1], 1.40, kTol);
    // 10 ms cycles up to and past the re-timed touch-down: it must stay at 1.40 and the foot must land there.
    for (int i = 1; i <= 12; ++i) {
      const scalar_t time = 1.30 + 0.01 * i;
      reports = adaptScheduleToContactEvents(schedule, time, measured(foot, false), cadence, config, latches);
      EXPECT_NEAR(schedule.eventTimes[1], 1.40, kTol) << "touch-down dragged at t=" << time;
    }
    EXPECT_TRUE(contactFlagsAtTime(schedule, 1.41)[foot]) << "the foot must be scheduled in contact after 1.40";
    EXPECT_TRUE(consistentSchedule(schedule));
    EXPECT_NEAR(schedule.eventTimes[3] - schedule.eventTimes[2], 0.4, kTol) << "later phases keep their duration";
  }
}

TEST_F(ContactEventTest, CadenceRequestIsClippedToWhatRemainsFeasible) {
  // An early request at a time where the wanted touch-down is already in the past but the current one is not yet
  // imminent brings the touch-down forward to now + margin, not to the current touch-down and not into the past.
  config.formulation.setExecutionRule(term::kEnergyCadenceModulation, true);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = singleSwingSchedule(foot, 1.1, 1.5);
    feet_array_t<scalar_t> cadence = makeFeetArray(0.0);
    cadence[foot] = -0.10;  // wanted 1.40
    const auto reports = adaptScheduleToContactEvents(schedule, 1.42, measured(foot, false), cadence, config, latches);
    EXPECT_EQ(reports[foot].type, ContactEventReport::Type::CADENCE_SHIFT);
    EXPECT_NEAR(schedule.eventTimes[1], 1.44, kTol);
    const auto again = adaptScheduleToContactEvents(schedule, 1.43, measured(foot, false), cadence, config, latches);
    EXPECT_EQ(again[foot].type, ContactEventReport::Type::NONE) << "once imminent the touch-down stands";
    EXPECT_NEAR(schedule.eventTimes[1], 1.44, kTol);
  }
}

TEST_F(ContactEventTest, CadenceShiftIgnoredWhenDisabledOrWhileSearchingForGround) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    latches.fill(SwingTimingLatch{});
    ModeSchedule schedule = singleSwingSchedule(foot, 1.0, 1.4);
    feet_array_t<scalar_t> cadence = makeFeetArray(0.1);
    config.formulation.setExecutionRule(term::kEnergyCadenceModulation, false);
    EXPECT_EQ(adaptScheduleToContactEvents(schedule, 1.1, measured(foot, false), cadence, config, latches)[foot].type,
              ContactEventReport::Type::NONE);
    EXPECT_NEAR(schedule.eventTimes[1], 1.4, kTol);

    config.formulation.setExecutionRule(term::kEnergyCadenceModulation, true);
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
      if (phase.has_value())
        ASSERT_LE(phase->second - phase->first,
                  config.shared.gaitLimits.maxSwingDuration + config.phaseResetting.maxLateTouchdownExtension + kTol);
    }
  }
  EXPECT_GT(early, 0);
  EXPECT_GT(late, 0);
}

/*====================================== committed contacts and plan shifts ================================*/

TEST(CommittedContacts, StraddlingNodeTakesTheStateHandedOverAtTheBoundary) {
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  // Right foot swings 0.07 -> 0.47. Planner grid from 0.0 with dt 0.1, boundary at the touch-down 0.47.
  const ModeSchedule schedule({0.07, 0.47}, {kAllInContact, modeWithSwinging({1}), kAllInContact});
  const std::vector<contact_flag_t> committed = committedContactsForPlanner(schedule, 0.0, 0.1, 11, 0.47);
  ASSERT_EQ(committed.size(), 5u) << "nodes starting at 0.0 .. 0.4 start before the boundary";
  EXPECT_TRUE(committed[0][1]) << "node 0 (midpoint 0.05) is before the lift-off";
  for (size_t k = 1; k < 4; ++k) EXPECT_FALSE(committed[k][1]) << "node " << k << " is in the swing";
  // Node 4 straddles the touch-down at 0.47: sampled at its midpoint 0.45 it would read "swinging" and let the plan keep
  // the foot in the air until 0.5; sampled at the boundary it reads the landed state the schedule hands over.
  EXPECT_TRUE(committed[4][1]);
  for (const contact_flag_t& c : committed) EXPECT_TRUE(c[0]);

  // The merge of such a plan with the executed schedule keeps the touch-down where it is.
  std::vector<contact_flag_t> planContacts = committed;
  while (planContacts.size() < 12) planContacts.push_back(makeFeetArray(true));
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 0.0;
  plan.dt = 0.1;
  plan.committedUntil = 0.47;
  plan.contacts = planContacts;
  const ModeSchedule merged = mergeModeSchedules(schedule, plan.toModeSchedule(), 0.47, -1.2, 2.4);
  EXPECT_FALSE(contactFlagsAtTime(merged, 0.469)[1]);
  EXPECT_TRUE(contactFlagsAtTime(merged, 0.471)[1]) << "the in-flight touch-down must not be delayed by the merge";
  for (scalar_t t = 0.47; t < 2.0; t += 0.01) EXPECT_TRUE(contactFlagsAtTime(merged, t)[1]) << "phantom re-lift at " << t;
}

TEST(CommittedContacts, BoundaryInsideAContactPhaseAndNodeLimit) {
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  // Left swings 0.83 -> 1.23; boundary 1.25 (commit window), grid from 1.0.
  const ModeSchedule schedule({0.83, 1.23, 1.33, 1.73},
                              {kAllInContact, modeWithSwinging({0}), kAllInContact, modeWithSwinging({1}), kAllInContact});
  const std::vector<contact_flag_t> committed = committedContactsForPlanner(schedule, 1.0, 0.1, 11, 1.25);
  ASSERT_EQ(committed.size(), 3u) << "nodes starting at 1.0, 1.1, 1.2";
  EXPECT_FALSE(committed[0][0]);
  EXPECT_FALSE(committed[1][0]);
  EXPECT_TRUE(committed[2][0]) << "the straddling node reads the landed state at the boundary, not the swing at its midpoint";
  EXPECT_TRUE(committed[2][1]) << "the right foot's later lift-off at 1.33 is outside the window";
  // At most maxNodes nodes, and a boundary at or before the grid start commits nothing.
  EXPECT_EQ(committedContactsForPlanner(schedule, 1.0, 0.1, 2, 1.25).size(), 2u);
  EXPECT_TRUE(committedContactsForPlanner(schedule, 1.0, 0.1, 11, 1.0).empty());
  EXPECT_TRUE(committedContactsForPlanner(schedule, 1.0, 0.1, 11, 0.9).empty());
}

TEST(CommittedContacts, LastCommittedNodeEndingOnTheBoundaryIsSampledAtTheBoundary) {
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  // Right foot swings -0.13 -> 0.27 and lands 0.03 s before a boundary at 0.3 that lies on the node grid from 0.0 (the
  // live commit time of three whole nodes, with no swing to extend it). Sampled at its midpoint 0.25 the node [0.2, 0.3)
  // read "swinging" although the foot is down by the boundary; a plan that kept it in the air over node 3 then had the
  // merge re-lift the foot 0.03 s after it had landed.
  const ModeSchedule schedule({-0.13, 0.27}, {kAllInContact, modeWithSwinging({1}), kAllInContact});
  const std::vector<scalar_t> samples = committedSampleTimes(0.0, 0.1, 11, 0.3);
  ASSERT_EQ(samples.size(), 3u) << "nodes starting at 0.0, 0.1, 0.2 start before the boundary";
  EXPECT_NEAR(samples[0], 0.05, kTol);
  EXPECT_NEAR(samples[1], 0.15, kTol);
  EXPECT_NEAR(samples[2], 0.3, kTol) << "the node that ends on the boundary is sampled at the boundary, not at 0.25";
  const std::vector<contact_flag_t> committed = committedContactsForPlanner(schedule, 0.0, 0.1, 11, 0.3);
  ASSERT_EQ(committed.size(), 3u);
  EXPECT_FALSE(committed[0][1]);
  EXPECT_FALSE(committed[1][1]);
  EXPECT_TRUE(committed[2][1]) << "landed before the boundary: the plan continues from a foot on the ground";
  const std::vector<feet_array_t<scalar_t>> starts = committedPhaseStartsForPlanner(schedule, 0.0, 0.1, 11, 0.3);
  ASSERT_EQ(starts.size(), 3u);
  EXPECT_NEAR(starts[2][1], 0.27, kTol) << "the contact is counted from the executed touch-down";
  // A touch-down exactly on the boundary has passed as well (the event at the query time counts as passed).
  const ModeSchedule onBoundary({-0.1, 0.3}, {kAllInContact, modeWithSwinging({1}), kAllInContact});
  EXPECT_TRUE(committedContactsForPlanner(onBoundary, 0.0, 0.1, 11, 0.3)[2][1]);
  EXPECT_NEAR(committedPhaseStartsForPlanner(onBoundary, 0.0, 0.1, 11, 0.3)[2][1], 0.3, kTol);

  // The merge of a plan built on these committed contacts keeps the executed touch-down and never lifts the foot again.
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 0.0;
  plan.dt = 0.1;
  plan.committedUntil = 0.3;
  plan.contacts = committed;
  while (plan.contacts.size() < 12) plan.contacts.push_back(makeFeetArray(true));
  const ModeSchedule merged = mergeModeSchedules(schedule, plan.toModeSchedule(), 0.3, -1.2, 2.4);
  EXPECT_FALSE(contactFlagsAtTime(merged, 0.269)[1]);
  for (scalar_t t = 0.27; t < 2.0; t += 0.01) EXPECT_TRUE(contactFlagsAtTime(merged, t)[1]) << "phantom re-lift at " << t;
}

TEST(CommitBoundary, CoversSwingsOfEveryFootThatStartOnAnExtendedBoundary) {
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  // Right swings 1.0 -> 1.4; the left lifts at the very touch-down of the right (no double support) and lands at 1.8.
  // The window from 0.9 (commit time 0.25 -> 1.15) overlaps the right swing, which extends the boundary to 1.4; the
  // left swing starts exactly there and is in the window too, so the boundary is its touch-down. Walking the schedule
  // once per foot in foot order saw the left foot before the right one had extended the boundary and stopped at 1.4.
  const ModeSchedule schedule({1.0, 1.4, 1.8}, {kAllInContact, modeWithSwinging({1}), modeWithSwinging({0}), kAllInContact});
  EXPECT_NEAR(commitBoundaryForSchedule(schedule, 0.9, 0.25), 1.8, kTol);
  // The mirrored schedule (left first) gave the right answer already; both orders agree now.
  const ModeSchedule mirrored({1.0, 1.4, 1.8}, {kAllInContact, modeWithSwinging({0}), modeWithSwinging({1}), kAllInContact});
  EXPECT_NEAR(commitBoundaryForSchedule(mirrored, 0.9, 0.25), 1.8, kTol);
  // A swing that starts after the (extended) boundary is not covered; a window with no swing is not extended.
  const ModeSchedule later({1.0, 1.4, 1.5, 1.9},
                           {kAllInContact, modeWithSwinging({1}), kAllInContact, modeWithSwinging({0}), kAllInContact});
  EXPECT_NEAR(commitBoundaryForSchedule(later, 0.9, 0.25), 1.4, kTol);
  EXPECT_NEAR(commitBoundaryForSchedule(later, 0.2, 0.25), 0.45, kTol);
  EXPECT_NEAR(commitBoundaryForSchedule(ModeSchedule({}, {kAllInContact}), 0.2, 0.25), 0.45, kTol);
}

TEST(CommittedContacts, PhaseStartsAreTheExecutedEventTimes) {
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  // Right foot swings 0.57 -> 0.97. Planner grid from 0.7 with dt 0.1, boundary at the touch-down 0.97.
  const ModeSchedule schedule({0.57, 0.97}, {kAllInContact, modeWithSwinging({1}), kAllInContact});
  const feet_array_t<scalar_t> before = contactPhaseStartTimes(schedule, 0.7);
  EXPECT_TRUE(std::isinf(before[0]) && before[0] < 0.0) << "the left foot's contact began before the schedule";
  EXPECT_NEAR(before[1], 0.57, kTol);
  EXPECT_NEAR(contactPhaseStartTimes(schedule, 0.969)[1], 0.57, kTol);
  EXPECT_NEAR(contactPhaseStartTimes(schedule, 0.97)[1], 0.97, kTol) << "the touch-down at the query time counts as passed";
  EXPECT_NEAR(contactPhaseStartTimes(schedule, 1.5)[1], 0.97, kTol);

  const std::vector<scalar_t> samples = committedSampleTimes(0.7, 0.1, 11, 0.97);
  ASSERT_EQ(samples.size(), 3u) << "nodes starting at 0.7, 0.8, 0.9 start before the boundary";
  EXPECT_NEAR(samples[0], 0.75, kTol);
  EXPECT_NEAR(samples[1], 0.85, kTol);
  EXPECT_NEAR(samples[2], 0.97, kTol) << "the straddling node is sampled at the boundary";

  const std::vector<feet_array_t<scalar_t>> starts = committedPhaseStartsForPlanner(schedule, 0.7, 0.1, 11, 0.97);
  ASSERT_EQ(starts.size(), 3u);
  for (size_t k = 0; k < 2; ++k) EXPECT_NEAR(starts[k][1], 0.57, kTol) << "node " << k << " is in the swing that began at 0.57";
  // The straddling node reports the landed state; the landing happened at 0.97, 0.07 s after the node start.
  EXPECT_NEAR(starts[2][1], 0.97, kTol);
  for (const auto& nodeStarts : starts) EXPECT_TRUE(std::isinf(nodeStarts[0]));
  EXPECT_TRUE(committedPhaseStartsForPlanner(schedule, 0.7, 0.1, 11, 0.7).empty());
}

TEST(PlanShiftLog, ShiftsYoungerThanTheSnapshotAreSummedAgainstTheOriginalStartTime) {
  ContactPlan plan = makePlan(1.005, 0.1, 5, vector2_t::Zero(), vector2_t::Zero(), vector2_t::Zero());
  plan.committedUntil = 1.3;
  // A cadence advance undone one cycle later: net zero. Comparing against the already-shifted start time would apply
  // the first (+0.10 -> start 1.105) and then skip the second (1.020 < 1.105).
  std::deque<std::pair<scalar_t, scalar_t>> log = {{1.010, +0.10}, {1.020, -0.10}};
  EXPECT_NEAR(applyScheduleShiftsToPlan(plan, log), 0.0, kTol);
  EXPECT_NEAR(plan.startTime, 1.005, kTol);
  EXPECT_NEAR(plan.committedUntil, 1.3, kTol);
  // Shifts at or before the snapshot were already seen by it; later ones apply.
  log = {{1.005, +0.5}, {1.0, +0.3}, {1.1, +0.07}, {1.2, -0.02}};
  EXPECT_NEAR(applyScheduleShiftsToPlan(plan, log), 0.05, kTol);
  EXPECT_NEAR(plan.startTime, 1.055, kTol);
  EXPECT_NEAR(plan.committedUntil, 1.35, kTol);
  EXPECT_NEAR(applyScheduleShiftsToPlan(plan, {}), 0.0, kTol);
}

TEST(PlanShiftLog, PlanMadeBeforeASwingWasCommittedDisagreesWithItInFlight) {
  // Executed schedule: foot 0 swings [1.2, 1.7).
  contact_flag_t foot0InAir = makeFeetArray(true);
  foot0InAir[0] = false;
  const ModeSchedule applied({1.2, 1.7}, {ModeNumber::STANCE, stanceLeg2ModeNumber(foot0InAir), ModeNumber::STANCE});

  // A plan that keeps both feet down was made from a snapshot without that swing: merging it while the swing is in
  // flight would land the foot at the merge point and lift it again.
  ContactPlan plan = makePlan(1.0, 0.1, 10, vector2_t::Zero(), vector2_t::Zero(), vector2_t::Zero());
  EXPECT_TRUE(planAgreesWithSwingsInFlight(applied, plan, 1.1)) << "before the swing";
  EXPECT_TRUE(planAgreesWithSwingsInFlight(applied, plan, 1.2)) << "a lift-off at the merge point itself is not executing yet";
  EXPECT_FALSE(planAgreesWithSwingsInFlight(applied, plan, 1.3)) << "in flight";
  EXPECT_FALSE(planAgreesWithSwingsInFlight(applied, plan, 1.65)) << "still in flight";
  EXPECT_TRUE(planAgreesWithSwingsInFlight(applied, plan, 1.7)) << "landed (the touch-down counts as passed)";

  // A plan that has the foot in the air there agrees, whatever it does afterwards.
  for (int k = 2; k < 7; ++k) plan.contacts[k][0] = false;  // nodes [1.2, 1.7)
  EXPECT_TRUE(planAgreesWithSwingsInFlight(applied, plan, 1.3));
  EXPECT_TRUE(planAgreesWithSwingsInFlight(applied, plan, 1.65));
  // The other foot is free to be lifted by the plan at any time.
  for (int k = 0; k < 10; ++k) plan.contacts[k][1] = false;
  EXPECT_TRUE(planAgreesWithSwingsInFlight(applied, plan, 1.3));

  // Nothing to disagree with.
  plan.valid = false;
  EXPECT_TRUE(planAgreesWithSwingsInFlight(applied, plan, 1.3));
}

TEST(PlanHeading, InterpolatesHeadingAndLooksUpFootYaw) {
  ContactPlan plan = makePlan(1.0, 0.1, 4, vector2_t::Zero(), vector2_t::Zero(), vector2_t::Zero());
  EXPECT_FALSE(plan.hasHeading());
  EXPECT_FALSE(plan.headingAtTime(1.0).has_value());
  EXPECT_FALSE(plan.footYawAtTime(0, 1.0).has_value());
  plan.heading = {0.0, 0.1, 0.2, 0.3, 0.4};
  plan.headingRate = {1.0, 1.0, 1.0, 1.0, 1.0};
  plan.footYaws = {makeFeetArray(0.0), makeFeetArray(0.0), makeFeetArray(0.2), makeFeetArray(0.2), makeFeetArray(0.4)};
  ASSERT_TRUE(plan.hasHeading());
  EXPECT_NEAR(*plan.headingAtTime(1.0), 0.0, kTol);
  EXPECT_NEAR(*plan.headingAtTime(1.15), 0.15, kTol) << "linear between nodes";
  EXPECT_NEAR(*plan.headingAtTime(0.5), 0.0, kTol) << "clamped before the plan";
  EXPECT_NEAR(*plan.headingAtTime(9.0), 0.4, kTol) << "clamped after the plan";
  EXPECT_NEAR(*plan.headingRateAtTime(1.23), 1.0, kTol);
  EXPECT_NEAR(*plan.footYawAtTime(0, 1.24), 0.2, kTol) << "nearest node";
  EXPECT_NEAR(*plan.footYawAtTime(0, 1.36), 0.4, kTol);
  plan.shiftInTime(0.5);
  EXPECT_NEAR(*plan.headingAtTime(1.65), 0.15, kTol) << "the heading moves with the plan";
}

TEST(SwingQueries, CurrentOrNextLiftOff) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const ModeSchedule schedule = twoStepSchedule(foot, foot);  // the same foot swings [1.0, 1.4] and [1.5, 1.9]
    ASSERT_TRUE(currentOrNextLiftOffTime(schedule, foot, 0.5).has_value());
    EXPECT_NEAR(*currentOrNextLiftOffTime(schedule, foot, 0.5), 1.0, kTol) << "next swing while standing";
    EXPECT_NEAR(*currentOrNextLiftOffTime(schedule, foot, 1.2), 1.0, kTol) << "the swing in flight";
    EXPECT_NEAR(*currentOrNextLiftOffTime(schedule, foot, 1.45), 1.5, kTol) << "the next swing during double support";
    EXPECT_NEAR(*currentOrNextLiftOffTime(schedule, foot, 1.7), 1.5, kTol);
    EXPECT_FALSE(currentOrNextLiftOffTime(schedule, foot, 1.95).has_value()) << "no further swing";
    const ModeSchedule noLiftOff({1.4}, {modeWithSwinging({foot}), kAllInContact});
    EXPECT_FALSE(currentOrNextLiftOffTime(noLiftOff, foot, 1.0).has_value());
  }
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
  config.reachability.reachX = 0.4;
  config.reachability.reachYInner = 0.05;
  config.reachability.reachYOuter = 0.3;
  const vector2_t com(1.0, 2.0);
  constexpr size_t kLeft = 0, kRight = 1;
  for (scalar_t yaw : {0.0, 0.7, -2.0}) {
    const vector2_t ex(std::cos(yaw), std::sin(yaw)), ey(-std::sin(yaw), std::cos(yaw));
    const vector2_t nominalLeft = com + 0.1 * ex + 0.15 * ey;
    const vector2_t nominalRight = com + 0.1 * ex - 0.15 * ey;
    // Inside the region: unchanged.
    EXPECT_TRUE(clipFootholdToReach(nominalLeft, kLeft, com, yaw, config).isApprox(nominalLeft, 1e-12));
    EXPECT_TRUE(clipFootholdToReach(nominalRight, kRight, com, yaw, config).isApprox(nominalRight, 1e-12));
    // Too far forward: clipped to reachX along the heading, lateral offset kept.
    const vector2_t forward = clipFootholdToReach(com + 0.9 * ex + 0.15 * ey, kLeft, com, yaw, config);
    EXPECT_NEAR(ex.dot(forward - com), 0.4, 1e-12);
    EXPECT_NEAR(ey.dot(forward - com), 0.15, 1e-12);
    // Too far back.
    EXPECT_NEAR(ex.dot(clipFootholdToReach(com - 0.9 * ex + 0.15 * ey, kLeft, com, yaw, config) - com), -0.4, 1e-12);
    // Across the body: a left foot never crosses to the right of the CoM, a right foot never to the left. The side is
    // the foot's own, not read off where the foothold happens to be, so an adjusted foothold that has already crossed
    // the CoM is pulled back to its own side rather than clipped into the other foot's region.
    const vector2_t crossedLeft = clipFootholdToReach(com - 0.2 * ey, kLeft, com, yaw, config);
    EXPECT_NEAR(ey.dot(crossedLeft - com), 0.05, 1e-12);
    const vector2_t crossedRight = clipFootholdToReach(com + 0.2 * ey, kRight, com, yaw, config);
    EXPECT_NEAR(ey.dot(crossedRight - com), -0.05, 1e-12);
    // Too far outward.
    EXPECT_NEAR(ey.dot(clipFootholdToReach(com + 0.8 * ey, kLeft, com, yaw, config) - com), 0.3, 1e-12);
    EXPECT_NEAR(ey.dot(clipFootholdToReach(com - 0.8 * ey, kRight, com, yaw, config) - com), -0.3, 1e-12);
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
        << "  execution:\n"
        << "    - energy_cadence_modulation\n"
        << "  phase_resetting:\n"
        << "    earlyTouchdownMinSwingRatio: 0.4\n"
        << "    earlyTouchdownMinContactDuration: 0.03\n"
        << "    maxLateTouchdownExtension: 0.2\n"
        << "    lateTouchdownExtensionStep: 0.02\n"
        << "    lateTouchdownSearchVelocity: 0.08\n"
        << "  dcm_step_adjustment:\n"
        << "    gain: 0.7\n"
        << "    maxOffset: 0.1\n"
        << "  energy_cadence_modulation:\n"
        << "    gain: 0.02\n"
        << "    deadband: 0.6\n";
  }
  const ContactPlanningConfig loaded = loadContactPlanningConfig(file, "contact_planning.", false);
  std::remove(file.c_str());
  EXPECT_FALSE(loaded.formulation.hasExecutionRule(term::kPhaseResetting));
  EXPECT_NEAR(loaded.phaseResetting.earlyTouchdownMinSwingRatio, 0.4, kTol);
  EXPECT_NEAR(loaded.phaseResetting.earlyTouchdownMinContactDuration, 0.03, kTol);
  EXPECT_NEAR(loaded.phaseResetting.maxLateTouchdownExtension, 0.2, kTol);
  EXPECT_NEAR(loaded.phaseResetting.lateTouchdownExtensionStep, 0.02, kTol);
  EXPECT_NEAR(loaded.phaseResetting.lateTouchdownSearchVelocity, 0.08, kTol);
  EXPECT_FALSE(loaded.formulation.hasExecutionRule(term::kDcmStepAdjustment));
  EXPECT_NEAR(loaded.dcmStepAdjustment.gain, 0.7, kTol);
  EXPECT_NEAR(loaded.dcmStepAdjustment.maxOffset, 0.1, kTol);
  EXPECT_TRUE(loaded.formulation.hasExecutionRule(term::kEnergyCadenceModulation));
  EXPECT_NEAR(loaded.energyCadenceModulation.gain, 0.02, kTol);
  EXPECT_NEAR(loaded.energyCadenceModulation.deadband, 0.6, kTol);
  EXPECT_NEAR(config.energyCadenceModulation.deadband, 0.0, kTol) << "no deadband by default: the shipped behaviour is unchanged";
  EXPECT_NEAR(loaded.planner.dt, config.planner.dt, kTol) << "missing keys keep their defaults";
}

TEST(ContactPlanningConfigAdaptive, ValidationRejectsBadValues) {
  const auto rejects = [](auto mutate) {
    ContactPlanningConfig config;
    mutate(config);
    EXPECT_THROW(config.validate(), std::invalid_argument);
  };
  rejects([](ContactPlanningConfig& c) { c.phaseResetting.earlyTouchdownMinSwingRatio = -0.1; });
  rejects([](ContactPlanningConfig& c) { c.phaseResetting.earlyTouchdownMinSwingRatio = 1.1; });
  rejects([](ContactPlanningConfig& c) { c.phaseResetting.earlyTouchdownMinContactDuration = -0.01; });
  rejects([](ContactPlanningConfig& c) { c.phaseResetting.maxLateTouchdownExtension = -0.1; });
  rejects([](ContactPlanningConfig& c) { c.phaseResetting.lateTouchdownExtensionStep = 0.0; });
  rejects([](ContactPlanningConfig& c) { c.phaseResetting.lateTouchdownSearchVelocity = -1.0; });
  rejects([](ContactPlanningConfig& c) { c.energyCadenceModulation.deadband = -1.0; });
  rejects([](ContactPlanningConfig& c) { c.dcmStepAdjustment.gain = -1.0; });
  rejects([](ContactPlanningConfig& c) { c.dcmStepAdjustment.maxOffset = -0.1; });
  rejects([](ContactPlanningConfig& c) { c.energyCadenceModulation.gain = -0.1; });
  ContactPlanningConfig zeros;
  zeros.phaseResetting.earlyTouchdownMinSwingRatio = 0.0;
  zeros.phaseResetting.earlyTouchdownMinContactDuration = 0.0;
  zeros.phaseResetting.maxLateTouchdownExtension = 0.0;
  zeros.phaseResetting.lateTouchdownSearchVelocity = 0.0;
  zeros.dcmStepAdjustment.gain = 0.0;
  zeros.dcmStepAdjustment.maxOffset = 0.0;
  zeros.energyCadenceModulation.gain = 0.0;
  EXPECT_NO_THROW(zeros.validate());
}

}  // namespace ocs2::humanoid
