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
#include <stdexcept>
#include <string>
#include <thread>

#include "absl/status/status.h"

#include "humanoid_common_mpc_app/robot/RealtimeLoopRunner.h"
#include "robot_realtime/PeriodicTimer.h"

/*
 * The realtime loop of the robot process: it paces its cycles on absolute deadlines, counts the cycles that overran
 * their period, hands the timing of every reporting window to the communication thread, stops cleanly, and ends - not
 * the process - when a cycle throws.
 */

namespace ocs2::humanoid {
namespace {

using std::chrono::milliseconds;
using std::chrono::nanoseconds;

RealtimeLoopConfig loopConfig(nanoseconds period, nanoseconds window) {
  RealtimeLoopConfig config;
  config.period = period;
  config.reportingWindow = window;
  return config;
}

/** Busy-waits for `duration` (a cycle that computes rather than sleeps). */
void spinFor(nanoseconds duration) {
  const nanoseconds end = robot::realtime::monotonicNow() + duration;
  while (robot::realtime::monotonicNow() < end) {
  }
}

/** Waits until `condition` holds or 5 s pass; true if it held. */
template <typename Condition>
bool waitFor(Condition condition) {
  const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!condition()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(milliseconds(1));
  }
  return true;
}

TEST(RealtimeLoopRunner, PacesTheCyclesOnAbsoluteDeadlines) {
  constexpr nanoseconds kPeriod = milliseconds(2);
  constexpr int kCycles = 100;
  RealtimeLoopRunner runner(loopConfig(kPeriod, milliseconds(1000)));
  std::atomic<int> cycles{0};
  std::atomic<std::int64_t> firstCycleNs{0};
  std::atomic<std::int64_t> lastCycleNs{0};
  // Half of every period spent computing: a loop that slept a whole period after its work would drift by that much.
  ASSERT_TRUE(runner
                  .start([&]() {
                    const int cycle = cycles.fetch_add(1);
                    const std::int64_t now = robot::realtime::monotonicNow().count();
                    if (cycle == 0) firstCycleNs.store(now);
                    if (cycle == kCycles) lastCycleNs.store(now);
                    spinFor(kPeriod / 2);
                  })
                  .ok());
  ASSERT_TRUE(waitFor([&]() { return cycles.load() > kCycles; }));
  runner.stop();
  const double elapsed = static_cast<double>(lastCycleNs.load() - firstCycleNs.load()) * 1e-9;
  const double expected = kCycles * 1e-3 * 2.0;
  // On the grid: no drift of half a period per cycle (that would be 1.5x), and a little slack for a loaded machine.
  EXPECT_GE(elapsed, expected * 0.99);
  EXPECT_LT(elapsed, expected * 1.25);
  EXPECT_GE(runner.cycles(), static_cast<std::uint64_t>(kCycles));
}

TEST(RealtimeLoopRunner, ReportsEveryWindowAndCountsTheOverruns) {
  constexpr nanoseconds kPeriod = milliseconds(2);
  RealtimeLoopRunner runner(loopConfig(kPeriod, milliseconds(50)));
  std::atomic<int> cycles{0};
  std::atomic<bool> overrun{false};
  robot::realtime::LoopTimingSnapshot snapshot;
  EXPECT_FALSE(runner.takeTimingSnapshot(snapshot)) << "no window has closed before the loop runs";

  ASSERT_TRUE(runner
                  .start([&]() {
                    cycles.fetch_add(1);
                    if (overrun.load()) spinFor(milliseconds(3));
                  })
                  .ok());
  // A window of cycles that keep their period: no overrun.
  ASSERT_TRUE(waitFor([&]() { return runner.takeTimingSnapshot(snapshot) && snapshot.window >= 2; }));
  EXPECT_EQ(snapshot.totalOverruns, 0u);
  EXPECT_NEAR(snapshot.targetPeriodS, 2e-3, 1e-12);
  EXPECT_GT(snapshot.windowCycles, 0u);

  // Cycles that compute for longer than the period: every one of them is an overrun.
  overrun.store(true);
  const int overrunStart = cycles.load();
  ASSERT_TRUE(waitFor([&]() { return cycles.load() >= overrunStart + 10; }));
  overrun.store(false);
  ASSERT_TRUE(waitFor([&]() { return runner.takeTimingSnapshot(snapshot) && snapshot.totalOverruns >= 9; }));
  EXPECT_GE(snapshot.maxComputeTimeS, 3e-3);
  runner.stop();
}

TEST(RealtimeLoopRunner, StopsAfterTheCurrentCycleAndRunsNoMore) {
  RealtimeLoopRunner runner(loopConfig(milliseconds(1), milliseconds(1000)));
  std::atomic<int> cycles{0};
  ASSERT_TRUE(runner.start([&]() { cycles.fetch_add(1); }).ok());
  ASSERT_TRUE(waitFor([&]() { return cycles.load() >= 5; }));
  EXPECT_TRUE(runner.isRunning());
  runner.stop();
  EXPECT_FALSE(runner.isRunning());
  const int stoppedAt = cycles.load();
  std::this_thread::sleep_for(milliseconds(20));
  EXPECT_EQ(cycles.load(), stoppedAt);
  runner.stop();  // idempotent
}

TEST(RealtimeLoopRunner, RunsOnceAndRefusesWhatItCannotRun) {
  RealtimeLoopRunner runner(loopConfig(milliseconds(1), milliseconds(1000)));
  EXPECT_EQ(runner.start(nullptr).code(), absl::StatusCode::kInvalidArgument);
  ASSERT_TRUE(runner.start([]() {}).ok());
  EXPECT_EQ(runner.start([]() {}).code(), absl::StatusCode::kFailedPrecondition);
  runner.stop();

  RealtimeLoopRunner zeroPeriod(loopConfig(nanoseconds(0), milliseconds(1000)));
  EXPECT_EQ(zeroPeriod.start([]() {}).code(), absl::StatusCode::kInvalidArgument);
}

TEST(RealtimeLoopRunner, AThreadSetUpTheProcessMayNotTakeIsReportedAndTheLoopRunsAnyway) {
  RealtimeLoopConfig config = loopConfig(milliseconds(1), milliseconds(1000));
  // A core that no machine has: the pinning step fails, and the loop runs regardless.
  config.thread.cores = {100000};
  RealtimeLoopRunner runner(config);
  std::atomic<int> cycles{0};
  const absl::Status configured = runner.start([&]() { cycles.fetch_add(1); });
  EXPECT_FALSE(configured.ok());
  EXPECT_TRUE(waitFor([&]() { return cycles.load() >= 3; }));
  runner.stop();
}

TEST(RealtimeLoopRunner, ACycleThatThrowsEndsTheLoopAndRunsTheFaultFunctionOnce) {
  RealtimeLoopRunner runner(loopConfig(milliseconds(1), milliseconds(1000)));
  std::atomic<int> cycles{0};
  std::atomic<int> faults{0};
  std::atomic<std::thread::id> faultThread{};
  std::atomic<std::thread::id> cycleThread{};
  ASSERT_TRUE(runner
                  .start(
                      [&]() {
                        cycleThread.store(std::this_thread::get_id());
                        if (cycles.fetch_add(1) == 10) throw std::runtime_error("contact estimator 'magic' reported 3 contact flags");
                      },
                      [&]() {
                        faults.fetch_add(1);
                        faultThread.store(std::this_thread::get_id());
                      })
                  .ok());
  ASSERT_TRUE(waitFor([&]() { return runner.faulted(); }));
  EXPECT_FALSE(runner.isRunning());
  const int ran = cycles.load();
  std::this_thread::sleep_for(milliseconds(20));
  EXPECT_EQ(cycles.load(), ran) << "no cycle after the one that threw";
  EXPECT_EQ(ran, 11);
  EXPECT_EQ(runner.cycles(), 10u) << "the cycle that threw is not counted";
  EXPECT_EQ(faults.load(), 1);
  EXPECT_EQ(faultThread.load(), cycleThread.load()) << "on the realtime thread";
  EXPECT_EQ(runner.faultMessage(), "contact estimator 'magic' reported 3 contact flags");
  runner.stop();
}

TEST(RealtimeLoopRunner, AnythingThrownIsCaughtAndAThrowingFaultFunctionStillEndsTheLoop) {
  RealtimeLoopRunner runner(loopConfig(milliseconds(1), milliseconds(1000)));
  ASSERT_TRUE(runner.start([]() { throw 42; }, []() { throw std::logic_error("the fault function must not throw"); }).ok());
  ASSERT_TRUE(waitFor([&]() { return runner.faulted(); }));
  EXPECT_NE(runner.faultMessage().find("not a std::exception"), std::string::npos) << runner.faultMessage();
  runner.stop();
}

TEST(RealtimeLoopRunner, ALoopThatDoesNotThrowHasNoFault) {
  RealtimeLoopRunner runner(loopConfig(milliseconds(1), milliseconds(1000)));
  ASSERT_TRUE(runner.start([]() {}).ok());
  ASSERT_TRUE(waitFor([&]() { return runner.cycles() > 5; }));
  runner.stop();
  EXPECT_FALSE(runner.faulted());
  EXPECT_TRUE(runner.faultMessage().empty());
}

}  // namespace
}  // namespace ocs2::humanoid
