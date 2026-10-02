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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#include "robot_core/TripleBuffer.h"
#include "robot_realtime/LoopTimingSnapshot.h"
#include "robot_realtime/LoopTimingStats.h"
#include "robot_realtime/PeriodicTimer.h"

namespace robot::realtime {
namespace {

using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;

// An arbitrary CLOCK_MONOTONIC origin for the synthetic cycles; nothing depends on it being zero.
const nanoseconds kOrigin = std::chrono::seconds(1000);

CycleTiming cycleAt(nanoseconds start, nanoseconds computeTime) {
  return CycleTiming{.start = start, .end = start + computeTime};
}

TEST(LoopTimingStatsTest, reportsNothingBeforeTheFirstWindowCloses) {
  LoopTimingStats stats(milliseconds(2), milliseconds(100));
  EXPECT_EQ(stats.lastSnapshot().window, 0u);
  EXPECT_DOUBLE_EQ(stats.lastSnapshot().targetPeriodS, 0.002);
  for (int k = 0; k < 50; ++k) {
    EXPECT_FALSE(stats.addCycle(cycleAt(kOrigin + k * milliseconds(2), microseconds(500))));
  }
  EXPECT_EQ(stats.lastSnapshot().window, 0u);
}

TEST(LoopTimingStatsTest, closesAWindowAtTheFirstCycleThatStartsAWindowLengthAfterItOpened) {
  LoopTimingStats stats(milliseconds(2), milliseconds(100));
  std::vector<int> closingCycles;
  for (int k = 0; k <= 100; ++k) {
    if (stats.addCycle(cycleAt(kOrigin + k * milliseconds(2), microseconds(500)))) {
      closingCycles.push_back(k);
    }
    if (k == 50) {
      // The first window holds the cycles that started at 0 ms through 100 ms: 51 cycles and 50 periods.
      const LoopTimingSnapshot& first = stats.lastSnapshot();
      EXPECT_EQ(first.window, 1u);
      EXPECT_EQ(first.windowCycles, 51u);
      EXPECT_EQ(first.totalCycles, 51u);
      EXPECT_DOUBLE_EQ(first.windowDurationS, 0.1);
      EXPECT_DOUBLE_EQ(first.meanPeriodS, 0.002);
      EXPECT_DOUBLE_EQ(first.minPeriodS, 0.002);
      EXPECT_DOUBLE_EQ(first.maxPeriodS, 0.002);
      EXPECT_DOUBLE_EQ(first.meanComputeTimeS, 0.0005);
      EXPECT_DOUBLE_EQ(first.maxComputeTimeS, 0.0005);
      EXPECT_EQ(first.windowOverruns, 0u);
    }
  }
  EXPECT_EQ(closingCycles, (std::vector<int>{50, 100}));

  // The second window starts where the first ended: 50 more cycles, each with the period that led to it.
  const LoopTimingSnapshot& second = stats.lastSnapshot();
  EXPECT_EQ(second.window, 2u);
  EXPECT_EQ(second.windowCycles, 50u);
  EXPECT_EQ(second.totalCycles, 101u);
  EXPECT_DOUBLE_EQ(second.windowDurationS, 0.1);
  EXPECT_DOUBLE_EQ(second.meanPeriodS, 0.002);
}

TEST(LoopTimingStatsTest, windowsTileTimeAndCountEveryCycleOnce) {
  const nanoseconds period = milliseconds(2);
  LoopTimingStats stats(period, milliseconds(50));
  std::mt19937 generator(/*seed=*/7);
  std::uniform_int_distribution<std::int64_t> jitter(-500'000, 500'000);
  std::uniform_int_distribution<std::int64_t> compute(100'000, 3'000'000);

  nanoseconds start = kOrigin;
  std::uint64_t cyclesInClosedWindows = 0;
  double closedDurationS = 0.0;
  nanoseconds lastClose = kOrigin;
  int windows = 0;
  for (int k = 0; k < 10'000; ++k) {
    if (k > 0) {
      start += period + nanoseconds(jitter(generator));
    }
    if (!stats.addCycle(cycleAt(start, nanoseconds(compute(generator))))) {
      continue;
    }
    ++windows;
    const LoopTimingSnapshot& snapshot = stats.lastSnapshot();
    cyclesInClosedWindows += snapshot.windowCycles;
    closedDurationS += snapshot.windowDurationS;
    lastClose = start;

    EXPECT_EQ(snapshot.window, static_cast<std::uint64_t>(windows));
    EXPECT_EQ(snapshot.totalCycles, cyclesInClosedWindows) << "every cycle belongs to exactly one window";
    EXPECT_GE(snapshot.windowDurationS, 0.05);
    // The first window's first cycle has no period before it; every later cycle has one.
    const std::uint64_t periods = snapshot.windowCycles - (snapshot.window == 1 ? 1 : 0);
    EXPECT_NEAR(snapshot.meanPeriodS * static_cast<double>(periods), snapshot.windowDurationS, 1e-9);
    EXPECT_LE(snapshot.minPeriodS, snapshot.meanPeriodS);
    EXPECT_GE(snapshot.maxPeriodS, snapshot.meanPeriodS);
    EXPECT_LE(snapshot.meanComputeTimeS, snapshot.maxComputeTimeS);
    EXPECT_LE(snapshot.windowOverruns, snapshot.windowCycles);
    EXPECT_LE(snapshot.windowOverruns, snapshot.totalOverruns);
  }
  ASSERT_GT(windows, 100);
  EXPECT_NEAR(closedDurationS, std::chrono::duration<double>(lastClose - kOrigin).count(), 1e-6)
      << "the windows tile time from the first cycle to the last close without gaps or overlaps";
}

TEST(LoopTimingStatsTest, anOverrunIsAComputeTimeLongerThanThePeriod) {
  const nanoseconds period = milliseconds(2);
  LoopTimingStats stats(period, milliseconds(7));
  const nanoseconds computeTimes[] = {period - nanoseconds(1), period, period + nanoseconds(1), 3 * period, microseconds(10)};
  nanoseconds start = kOrigin;
  for (const nanoseconds computeTime : computeTimes) {
    stats.addCycle(cycleAt(start, computeTime));
    start += period;
  }
  // The cycle at 8 ms closed the window that opened at 0 ms, so the window holds all five.
  const LoopTimingSnapshot& snapshot = stats.lastSnapshot();
  ASSERT_EQ(snapshot.window, 1u);
  EXPECT_EQ(snapshot.windowOverruns, 2u);
  EXPECT_EQ(snapshot.totalOverruns, 2u);
  EXPECT_DOUBLE_EQ(snapshot.maxComputeTimeS, 0.006);
}

TEST(LoopTimingStatsTest, periodExtremesFollowTheJitter) {
  LoopTimingStats stats(milliseconds(2), milliseconds(20));
  nanoseconds start = kOrigin;
  for (int k = 0; k < 20; ++k) {
    stats.addCycle(cycleAt(start, microseconds(100)));
    start += k % 2 == 0 ? microseconds(1500) : microseconds(2500);
  }
  const LoopTimingSnapshot& snapshot = stats.lastSnapshot();
  ASSERT_EQ(snapshot.window, 1u);
  EXPECT_DOUBLE_EQ(snapshot.minPeriodS, 0.0015);
  EXPECT_DOUBLE_EQ(snapshot.maxPeriodS, 0.0025);
  EXPECT_DOUBLE_EQ(snapshot.meanPeriodS, 0.002);
}

TEST(LoopTimingStatsTest, takesLatenessAndMissedPeriodsFromTheTimerWakeup) {
  const nanoseconds period = milliseconds(1);
  LoopTimingStats stats(period, milliseconds(5));
  const TimerWakeup wakeups[] = {
      {.deadline = kOrigin, .lateness = microseconds(20), .missedPeriods = 0},
      {.deadline = kOrigin + period, .lateness = microseconds(80), .missedPeriods = 0},
      {.deadline = kOrigin + 2 * period, .lateness = microseconds(1200), .missedPeriods = 1},
      {.deadline = kOrigin + 4 * period, .lateness = microseconds(30), .missedPeriods = 0},
      {.deadline = kOrigin + 5 * period, .lateness = microseconds(40), .missedPeriods = 0},
  };
  bool closed = false;
  for (const TimerWakeup& wakeup : wakeups) {
    closed = stats.addCycle(wakeup, wakeup.deadline + wakeup.lateness + microseconds(300));
  }
  ASSERT_TRUE(closed) << "the cycle that started 5.04 ms after the first closes the 5 ms window";
  const LoopTimingSnapshot& snapshot = stats.lastSnapshot();
  EXPECT_DOUBLE_EQ(snapshot.maxLatenessS, 0.0012);
  EXPECT_EQ(snapshot.windowMissedPeriods, 1u);
  EXPECT_EQ(snapshot.totalMissedPeriods, 1u);
  EXPECT_DOUBLE_EQ(snapshot.windowDurationS, 0.00502);
  EXPECT_DOUBLE_EQ(snapshot.meanComputeTimeS, 0.0003);
}

// The communication thread reads the snapshots while the realtime thread writes them. Each window's compute time is
// derived from the window's number, so a snapshot torn between two writes would not add up.
TEST(LoopTimingStatsTest, snapshotsReachAnotherThreadWholeThroughATripleBuffer) {
  constexpr std::uint64_t kWindows = 2000;
  const nanoseconds period = milliseconds(2);
  TripleBuffer<LoopTimingSnapshot> buffer;
  std::atomic<bool> producerDone{false};

  std::thread realtime([&buffer, &producerDone, period]() {
    LoopTimingStats stats(period, milliseconds(100));
    nanoseconds start = kOrigin;
    while (stats.lastSnapshot().window < kWindows) {
      const nanoseconds computeTime = microseconds(static_cast<std::int64_t>(stats.lastSnapshot().window) + 1);
      if (stats.addCycle(cycleAt(start, computeTime))) {
        buffer.writeSlot() = stats.lastSnapshot();
        buffer.publishWrite();
      }
      start += period;
    }
    producerDone.store(/*desired=*/true, std::memory_order_release);
  });

  std::uint64_t lastWindow = 0;
  std::uint64_t inconsistent = 0;
  std::uint64_t reordered = 0;
  for (;;) {
    const bool done = producerDone.load(std::memory_order_acquire);
    if (buffer.acquireRead()) {
      const LoopTimingSnapshot& snapshot = buffer.readSlot();
      reordered += snapshot.window > lastWindow ? 0 : 1;
      lastWindow = snapshot.window;
      const bool consistent =
          snapshot.totalCycles == 1 + 50 * snapshot.window && snapshot.windowCycles == (snapshot.window == 1 ? 51u : 50u) &&
          snapshot.maxComputeTimeS ==
              std::chrono::duration<double>(nanoseconds(microseconds(static_cast<std::int64_t>(snapshot.window)))).count();
      inconsistent += consistent ? 0 : 1;
    }
    if (done && !buffer.hasNewData()) {
      break;
    }
  }
  realtime.join();

  EXPECT_EQ(lastWindow, kWindows) << "the last snapshot written is the one the reader ends with";
  EXPECT_EQ(inconsistent, 0u);
  EXPECT_EQ(reordered, 0u);
}

TEST(LoopTimingStatsDeathTest, refusesNonPositiveDurations) {
  EXPECT_DEATH(LoopTimingStats(nanoseconds(0), milliseconds(100)), "positive target period");
  EXPECT_DEATH(LoopTimingStats(milliseconds(2), nanoseconds(0)), "positive reporting window");
}

}  // namespace
}  // namespace robot::realtime
