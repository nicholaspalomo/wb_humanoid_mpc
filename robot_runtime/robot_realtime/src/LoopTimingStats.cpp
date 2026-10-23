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

#include "robot_realtime/LoopTimingStats.h"

#include <algorithm>
#include <chrono>
#include <cstdint>

#include "absl/log/check.h"

#include "robot_realtime/LoopTimingSnapshot.h"
#include "robot_realtime/PeriodicTimer.h"

namespace robot::realtime {
namespace {

double toSeconds(std::chrono::nanoseconds duration) {
  return std::chrono::duration<double>(duration).count();
}

}  // namespace

LoopTimingStats::LoopTimingStats(std::chrono::nanoseconds targetPeriod, std::chrono::nanoseconds reportingWindow)
    : targetPeriod_(targetPeriod), reportingWindow_(reportingWindow) {
  CHECK_GT(targetPeriod_.count(), 0) << "LoopTimingStats needs a positive target period";
  CHECK_GT(reportingWindow_.count(), 0) << "LoopTimingStats needs a positive reporting window";
  snapshot_.targetPeriodS = toSeconds(targetPeriod_);
}

bool LoopTimingStats::addCycle(const TimerWakeup& wakeup, std::chrono::nanoseconds end) {
  return addCycle(CycleTiming{
      .start = wakeup.deadline + wakeup.lateness, .end = end, .lateness = wakeup.lateness, .missedPeriods = wakeup.missedPeriods});
}

bool LoopTimingStats::addCycle(const CycleTiming& cycle) {
  if (!hasPreviousCycle_) {
    window_.start = cycle.start;
  } else {
    const std::chrono::nanoseconds period = cycle.start - previousStart_;
    window_.minPeriod = window_.periods == 0 ? period : std::min(window_.minPeriod, period);
    window_.maxPeriod = window_.periods == 0 ? period : std::max(window_.maxPeriod, period);
    window_.periodSum += period;
    ++window_.periods;
  }
  hasPreviousCycle_ = true;
  previousStart_ = cycle.start;

  const std::chrono::nanoseconds computeTime = cycle.end - cycle.start;
  const bool overran = computeTime > targetPeriod_;
  const uint64_t missedPeriods = cycle.missedPeriods > 0 ? static_cast<uint64_t>(cycle.missedPeriods) : 0;
  window_.computeTimeSum += computeTime;
  window_.maxComputeTime = window_.cycles == 0 ? computeTime : std::max(window_.maxComputeTime, computeTime);
  window_.maxLateness = window_.cycles == 0 ? cycle.lateness : std::max(window_.maxLateness, cycle.lateness);
  window_.overruns += overran ? 1 : 0;
  window_.missedPeriods += missedPeriods;
  ++window_.cycles;

  ++totalCycles_;
  totalOverruns_ += overran ? 1 : 0;
  totalMissedPeriods_ += missedPeriods;

  if (cycle.start - window_.start < reportingWindow_) {
    return false;
  }
  closeWindow(cycle.start);
  return true;
}

void LoopTimingStats::closeWindow(std::chrono::nanoseconds end) {
  ++windowsClosed_;
  snapshot_.window = windowsClosed_;
  snapshot_.totalCycles = totalCycles_;
  snapshot_.totalOverruns = totalOverruns_;
  snapshot_.totalMissedPeriods = totalMissedPeriods_;
  snapshot_.windowDurationS = toSeconds(end - window_.start);
  snapshot_.windowCycles = window_.cycles;
  snapshot_.windowOverruns = window_.overruns;
  snapshot_.windowMissedPeriods = window_.missedPeriods;
  snapshot_.meanPeriodS = window_.periods == 0 ? 0.0 : toSeconds(window_.periodSum) / static_cast<double>(window_.periods);
  snapshot_.minPeriodS = toSeconds(window_.minPeriod);
  snapshot_.maxPeriodS = toSeconds(window_.maxPeriod);
  snapshot_.meanComputeTimeS = toSeconds(window_.computeTimeSum) / static_cast<double>(window_.cycles);
  snapshot_.maxComputeTimeS = toSeconds(window_.maxComputeTime);
  snapshot_.maxLatenessS = toSeconds(window_.maxLateness);

  // The closing cycle's start opens the next window, so the next period measured belongs to it.
  window_ = Window{};
  window_.start = end;
}

}  // namespace robot::realtime
