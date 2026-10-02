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

#pragma once

#include <chrono>
#include <cstdint>

namespace robot::realtime {

// Times here are std::chrono::nanoseconds on CLOCK_MONOTONIC: the clock clock_nanosleep() paces against, which neither
// jumps with the wall clock nor stops while the thread runs. (absl::Time is a wall-clock instant, and has no monotonic
// counterpart.)

/// The current CLOCK_MONOTONIC time.
std::chrono::nanoseconds monotonicNow();

/// Sleeps the calling thread until CLOCK_MONOTONIC reaches `deadline`; returns at once if it already has.
void sleepUntil(std::chrono::nanoseconds deadline);

/// What PeriodicTimer does with the deadlines that pass while a cycle overruns.
enum class OverrunPolicy {
  /**
   * Drop them. The cycle after an overrun starts at once and the next deadline is the first one of the original grid
   * that is still ahead, so the loop never runs cycles back to back and stays in phase. For a controller, which should
   * act on the latest state rather than replay the cycles it missed.
   */
  kSkipMissedPeriods,
  /**
   * Serve every one. After an overrun the loop runs cycles back to back, without sleeping, until it is back on
   * schedule, so that the number of cycles keeps pace with time. For a fixed-step simulation or anything that counts
   * cycles as time.
   */
  kCatchUp,
};

/// What one PeriodicTimer::waitForNextPeriod() call did.
struct TimerWakeup {
  /// The deadline the call waited for (CLOCK_MONOTONIC).
  std::chrono::nanoseconds deadline{0};
  /// How long after that deadline the thread woke: the scheduling latency, or the overrun when it was already past.
  std::chrono::nanoseconds lateness{0};
  /// Deadlines after `deadline` that had passed as well and that kSkipMissedPeriods dropped; 0 with kCatchUp.
  std::int64_t missedPeriods = 0;
};

/// Where the schedule goes after a wake-up; see advanceDeadline().
struct DeadlineAdvance {
  std::chrono::nanoseconds nextDeadline{0};
  std::int64_t missedPeriods = 0;
};

/**
 * The deadline arithmetic of PeriodicTimer, apart from any clock so that the policies can be tested exactly: after a
 * wake-up at `wakeupTime` that served `servedDeadline`, the next deadline and the deadlines the policy dropped. The
 * deadlines always stay on the grid `servedDeadline + k * period`.
 */
DeadlineAdvance advanceDeadline(std::chrono::nanoseconds servedDeadline,
                                std::chrono::nanoseconds wakeupTime,
                                std::chrono::nanoseconds period,
                                OverrunPolicy overrunPolicy);

/**
 * Paces a loop with absolute deadlines on CLOCK_MONOTONIC (clock_nanosleep with TIMER_ABSTIME), so that neither the
 * cycle's compute time nor the wake-up latency accumulates into drift:
 *
 *   PeriodicTimer timer(std::chrono::milliseconds(2), OverrunPolicy::kSkipMissedPeriods);
 *   timer.start();
 *   while (running) {
 *     const TimerWakeup wakeup = timer.waitForNextPeriod();
 *     ...
 *   }
 *
 * Allocation-free and lock-free; for the one thread that runs the loop.
 */
class PeriodicTimer {
 public:
  /// `period` must be positive.
  PeriodicTimer(std::chrono::nanoseconds period, OverrunPolicy overrunPolicy);

  /// (Re)starts the schedule with the first deadline one period from now.
  void start();

  /// (Re)starts the schedule with the first deadline at `firstDeadline` (CLOCK_MONOTONIC), e.g. to phase two loops.
  void startAt(std::chrono::nanoseconds firstDeadline);

  /**
   * Sleeps until the next deadline, or returns at once when it has already passed, and moves the schedule on as the
   * overrun policy says. Starts the schedule first if start() has not been called.
   */
  TimerWakeup waitForNextPeriod();

  std::chrono::nanoseconds period() const { return period_; }
  OverrunPolicy overrunPolicy() const { return overrunPolicy_; }
  /// The deadline the next waitForNextPeriod() waits for.
  std::chrono::nanoseconds nextDeadline() const { return nextDeadline_; }
  /// The sum of TimerWakeup::missedPeriods since the last start.
  std::int64_t totalMissedPeriods() const { return totalMissedPeriods_; }

 private:
  std::chrono::nanoseconds period_;
  OverrunPolicy overrunPolicy_;
  std::chrono::nanoseconds nextDeadline_{0};
  std::int64_t totalMissedPeriods_ = 0;
  bool started_ = false;
};

}  // namespace robot::realtime
