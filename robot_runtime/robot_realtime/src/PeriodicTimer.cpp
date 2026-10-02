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

#include "robot_realtime/PeriodicTimer.h"

#include <time.h>

#include <cerrno>
#include <chrono>
#include <cstdint>

#include "absl/log/check.h"

namespace robot::realtime {
namespace {

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000;

timespec toTimespec(std::chrono::nanoseconds time) {
  timespec converted{};
  converted.tv_sec = static_cast<time_t>(time.count() / kNanosecondsPerSecond);
  converted.tv_nsec = static_cast<long>(time.count() % kNanosecondsPerSecond);
  return converted;
}

}  // namespace

std::chrono::nanoseconds monotonicNow() {
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return std::chrono::nanoseconds(static_cast<std::int64_t>(now.tv_sec) * kNanosecondsPerSecond + now.tv_nsec);
}

void sleepUntil(std::chrono::nanoseconds deadline) {
  const timespec target = toTimespec(deadline);
  // clock_nanosleep returns the error rather than setting errno. A signal ends the sleep early with EINTR; sleeping
  // again towards the same absolute deadline loses nothing.
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target, /*remain=*/nullptr) == EINTR) {
  }
}

DeadlineAdvance advanceDeadline(std::chrono::nanoseconds servedDeadline,
                                std::chrono::nanoseconds wakeupTime,
                                std::chrono::nanoseconds period,
                                OverrunPolicy overrunPolicy) {
  const std::chrono::nanoseconds following = servedDeadline + period;
  if (overrunPolicy == OverrunPolicy::kCatchUp || wakeupTime < following) {
    return DeadlineAdvance{.nextDeadline = following, .missedPeriods = 0};
  }
  // The deadlines servedDeadline + k * period with 1 <= k <= missed have passed as well; the next is the one after.
  const std::int64_t missed = (wakeupTime - servedDeadline) / period;
  return DeadlineAdvance{.nextDeadline = servedDeadline + (missed + 1) * period, .missedPeriods = missed};
}

PeriodicTimer::PeriodicTimer(std::chrono::nanoseconds period, OverrunPolicy overrunPolicy)
    : period_(period), overrunPolicy_(overrunPolicy) {
  CHECK_GT(period_.count(), 0) << "a PeriodicTimer needs a positive period";
}

void PeriodicTimer::start() {
  startAt(monotonicNow() + period_);
}

void PeriodicTimer::startAt(std::chrono::nanoseconds firstDeadline) {
  nextDeadline_ = firstDeadline;
  totalMissedPeriods_ = 0;
  started_ = true;
}

TimerWakeup PeriodicTimer::waitForNextPeriod() {
  if (!started_) {
    start();
  }
  const std::chrono::nanoseconds deadline = nextDeadline_;
  sleepUntil(deadline);
  const std::chrono::nanoseconds now = monotonicNow();
  const DeadlineAdvance advance = advanceDeadline(deadline, now, period_, overrunPolicy_);
  nextDeadline_ = advance.nextDeadline;
  totalMissedPeriods_ += advance.missedPeriods;
  return TimerWakeup{.deadline = deadline, .lateness = now - deadline, .missedPeriods = advance.missedPeriods};
}

}  // namespace robot::realtime
