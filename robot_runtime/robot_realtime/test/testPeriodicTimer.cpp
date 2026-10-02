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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>

#include "robot_realtime/PeriodicTimer.h"

namespace robot::realtime {
namespace {

using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;

// Holds the thread busy, as a cycle's computation would, without sleeping.
void busyWaitFor(nanoseconds duration) {
  const nanoseconds end = monotonicNow() + duration;
  while (monotonicNow() < end) {
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The deadline arithmetic, exactly
// ---------------------------------------------------------------------------------------------------------------------

TEST(AdvanceDeadlineTest, anOnTimeWakeupMovesOnePeriodUnderEitherPolicy) {
  for (const OverrunPolicy policy : {OverrunPolicy::kSkipMissedPeriods, OverrunPolicy::kCatchUp}) {
    const DeadlineAdvance advance = advanceDeadline(nanoseconds(100), nanoseconds(105), nanoseconds(10), policy);
    EXPECT_EQ(advance.nextDeadline, nanoseconds(110));
    EXPECT_EQ(advance.missedPeriods, 0);
  }
}

TEST(AdvanceDeadlineTest, skippingDropsEveryDeadlineThatHasPassed) {
  const DeadlineAdvance late = advanceDeadline(nanoseconds(100), nanoseconds(123), nanoseconds(10), OverrunPolicy::kSkipMissedPeriods);
  EXPECT_EQ(late.nextDeadline, nanoseconds(130));
  EXPECT_EQ(late.missedPeriods, 2) << "the deadlines at 110 and 120 passed while the cycle served 100";

  // A deadline reached exactly has passed too: the loop is already due for it.
  const DeadlineAdvance onTheNext = advanceDeadline(nanoseconds(100), nanoseconds(110), nanoseconds(10), OverrunPolicy::kSkipMissedPeriods);
  EXPECT_EQ(onTheNext.nextDeadline, nanoseconds(120));
  EXPECT_EQ(onTheNext.missedPeriods, 1);
}

TEST(AdvanceDeadlineTest, catchingUpServesTheNextDeadlineEvenWhenItHasPassed) {
  const DeadlineAdvance late = advanceDeadline(nanoseconds(100), nanoseconds(123), nanoseconds(10), OverrunPolicy::kCatchUp);
  EXPECT_EQ(late.nextDeadline, nanoseconds(110));
  EXPECT_EQ(late.missedPeriods, 0);
}

TEST(AdvanceDeadlineTest, skippingAlwaysLandsOnTheFirstGridDeadlineAfterTheWakeup) {
  const nanoseconds served(1000);
  const nanoseconds period(7);
  for (std::int64_t offset = 0; offset < 1000; ++offset) {
    const nanoseconds wakeup = served + nanoseconds(offset);
    const DeadlineAdvance advance = advanceDeadline(served, wakeup, period, OverrunPolicy::kSkipMissedPeriods);
    EXPECT_EQ((advance.nextDeadline - served) % period, nanoseconds(0)) << "offset " << offset;
    EXPECT_GT(advance.nextDeadline, wakeup) << "offset " << offset;
    EXPECT_LE(advance.nextDeadline - period, std::max(wakeup, served)) << "offset " << offset;
    EXPECT_EQ(advance.missedPeriods, (advance.nextDeadline - served) / period - 1) << "offset " << offset;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The timer against the clock
// ---------------------------------------------------------------------------------------------------------------------

TEST(PeriodicTimerTest, neverWakesBeforeItsDeadlineAndKeepsDeadlinesOnTheGrid) {
  const nanoseconds period = milliseconds(1);
  PeriodicTimer timer(period, OverrunPolicy::kSkipMissedPeriods);
  timer.start();
  const nanoseconds firstDeadline = timer.nextDeadline();

  nanoseconds expectedDeadline = firstDeadline;
  for (int cycle = 0; cycle < 50; ++cycle) {
    const TimerWakeup wakeup = timer.waitForNextPeriod();
    const nanoseconds now = monotonicNow();
    EXPECT_EQ(wakeup.deadline, expectedDeadline);
    EXPECT_GE(wakeup.lateness, nanoseconds(0));
    EXPECT_GE(now, wakeup.deadline);
    EXPECT_EQ((wakeup.deadline - firstDeadline) % period, nanoseconds(0));
    expectedDeadline = wakeup.deadline + (wakeup.missedPeriods + 1) * period;
  }
}

TEST(PeriodicTimerTest, pacesWithAbsoluteDeadlinesSoComputeTimeDoesNotAccumulate) {
  // Half of every period is spent computing. A timer that slept one period after each cycle would take 1.5 times as
  // long; with absolute deadlines (and catching up after any scheduling hiccup) the loop ends on its last deadline.
  const nanoseconds period = milliseconds(3);
  constexpr int kCycles = 100;
  PeriodicTimer timer(period, OverrunPolicy::kCatchUp);
  const nanoseconds begin = monotonicNow();
  timer.start();
  const nanoseconds firstDeadline = timer.nextDeadline();

  TimerWakeup last;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    last = timer.waitForNextPeriod();
    busyWaitFor(period / 2);
  }
  const nanoseconds elapsed = monotonicNow() - begin;

  EXPECT_EQ(last.deadline, firstDeadline + (kCycles - 1) * period);
  EXPECT_GE(elapsed, kCycles * period);
  EXPECT_LT(elapsed, kCycles * period + period / 2 + milliseconds(100)) << "a drifting timer would take about 450 ms";
}

TEST(PeriodicTimerTest, skippingAfterAnOverrunDropsTheMissedDeadlinesAndSleepsAgain) {
  const nanoseconds period = milliseconds(2);
  PeriodicTimer timer(period, OverrunPolicy::kSkipMissedPeriods);
  timer.start();
  const TimerWakeup onTime = timer.waitForNextPeriod();

  // An overrun of three and a half periods.
  std::this_thread::sleep_for(microseconds(7000));
  const TimerWakeup late = timer.waitForNextPeriod();
  const nanoseconds wokeAt = late.deadline + late.lateness;

  EXPECT_EQ(late.deadline, onTime.deadline + period);
  EXPECT_GE(late.lateness, 2 * period + period / 2);
  EXPECT_GE(late.missedPeriods, 2);
  EXPECT_EQ(timer.totalMissedPeriods(), late.missedPeriods);
  // The next deadline is the first of the grid after the late wake-up, so the loop sleeps again instead of running
  // the missed cycles back to back.
  EXPECT_EQ(timer.nextDeadline(), late.deadline + (late.missedPeriods + 1) * period);
  EXPECT_GT(timer.nextDeadline(), wokeAt);
  EXPECT_LE(timer.nextDeadline() - period, wokeAt);

  const TimerWakeup next = timer.waitForNextPeriod();
  EXPECT_EQ(next.deadline, late.deadline + (late.missedPeriods + 1) * period);
  EXPECT_GE(next.lateness, nanoseconds(0));
}

TEST(PeriodicTimerTest, catchingUpAfterAnOverrunRunsTheMissedCyclesBackToBack) {
  const nanoseconds period = milliseconds(2);
  PeriodicTimer timer(period, OverrunPolicy::kCatchUp);
  timer.start();
  const TimerWakeup onTime = timer.waitForNextPeriod();

  std::this_thread::sleep_for(microseconds(7000));
  // The deadlines one, two and three periods after onTime all passed during the overrun: each wake-up returns at once
  // (it is late) and serves the next one of them.
  for (int k = 1; k <= 3; ++k) {
    const TimerWakeup wakeup = timer.waitForNextPeriod();
    EXPECT_EQ(wakeup.deadline, onTime.deadline + k * period);
    EXPECT_GT(wakeup.lateness, nanoseconds(0));
    EXPECT_EQ(wakeup.missedPeriods, 0);
  }
  EXPECT_EQ(timer.totalMissedPeriods(), 0);
}

TEST(PeriodicTimerTest, startAtSetsTheFirstDeadline) {
  PeriodicTimer timer(milliseconds(1), OverrunPolicy::kSkipMissedPeriods);
  const nanoseconds firstDeadline = monotonicNow() + milliseconds(5);
  timer.startAt(firstDeadline);
  const TimerWakeup wakeup = timer.waitForNextPeriod();
  EXPECT_EQ(wakeup.deadline, firstDeadline);
  EXPECT_GE(monotonicNow(), firstDeadline);
  EXPECT_EQ(timer.nextDeadline(), firstDeadline + (wakeup.missedPeriods + 1) * milliseconds(1));
}

TEST(PeriodicTimerTest, waitingWithoutStartStartsTheSchedule) {
  const nanoseconds period = milliseconds(1);
  PeriodicTimer timer(period, OverrunPolicy::kCatchUp);
  const nanoseconds before = monotonicNow();
  const TimerWakeup wakeup = timer.waitForNextPeriod();
  EXPECT_GE(wakeup.deadline, before + period);
  EXPECT_EQ(timer.nextDeadline(), wakeup.deadline + period);
}

TEST(PeriodicTimerTest, restartingClearsTheMissedPeriods) {
  const nanoseconds period = milliseconds(1);
  PeriodicTimer timer(period, OverrunPolicy::kSkipMissedPeriods);
  timer.start();
  std::this_thread::sleep_for(milliseconds(5));
  timer.waitForNextPeriod();
  ASSERT_GT(timer.totalMissedPeriods(), 0);
  timer.start();
  EXPECT_EQ(timer.totalMissedPeriods(), 0);
}

TEST(PeriodicTimerTest, sleepUntilAPastDeadlineReturnsAtOnce) {
  const nanoseconds before = monotonicNow();
  sleepUntil(before - std::chrono::seconds(1));
  EXPECT_LT(monotonicNow() - before, milliseconds(500));
}

TEST(PeriodicTimerDeathTest, refusesANonPositivePeriod) {
  EXPECT_DEATH(PeriodicTimer(nanoseconds(0), OverrunPolicy::kCatchUp), "positive period");
}

}  // namespace
}  // namespace robot::realtime
