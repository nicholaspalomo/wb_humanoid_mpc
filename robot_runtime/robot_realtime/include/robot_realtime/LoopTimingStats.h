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

#include "robot_realtime/LoopTimingSnapshot.h"
#include "robot_realtime/PeriodicTimer.h"

namespace robot::realtime {

/// The timing of one cycle of the loop, on CLOCK_MONOTONIC (see monotonicNow()).
struct CycleTiming {
  /// When the cycle's work began, normally the wake-up from PeriodicTimer::waitForNextPeriod().
  std::chrono::nanoseconds start{0};
  /// When its work finished.
  std::chrono::nanoseconds end{0};
  /// TimerWakeup::lateness of the wake-up that started it.
  std::chrono::nanoseconds lateness{0};
  /// TimerWakeup::missedPeriods of that wake-up.
  std::int64_t missedPeriods = 0;
};

/**
 * Accumulates the realtime loop's period, compute time, overruns and wake-up latency over a reporting window, and
 * condenses them into a LoopTimingSnapshot when the window closes. Allocation-free and lock-free; owned by the realtime
 * thread, which hands the snapshot on:
 *
 *   if (stats.addCycle(wakeup, monotonicNow())) {
 *     timingBuffer.writeSlot() = stats.lastSnapshot();
 *     timingBuffer.publishWrite();
 *   }
 *
 * A window opens at the start of a cycle and closes at the start of the first cycle that begins `reportingWindow` or
 * more after it; that cycle belongs to the closing window, and its start opens the next one. So the periods of a window
 * (the intervals between consecutive cycle starts) add up to its duration, and the windows tile time without gaps.
 */
class LoopTimingStats {
 public:
  /// `targetPeriod` and `reportingWindow` must be positive.
  LoopTimingStats(std::chrono::nanoseconds targetPeriod, std::chrono::nanoseconds reportingWindow);

  /// Adds one cycle. True when it closed a reporting window, i.e. lastSnapshot() has just been replaced.
  bool addCycle(const CycleTiming& cycle);

  /// addCycle() for a cycle that PeriodicTimer woke with `wakeup` and whose work finished at `end`.
  bool addCycle(const TimerWakeup& wakeup, std::chrono::nanoseconds end);

  /// The snapshot of the last closed window; LoopTimingSnapshot::window is 0 until the first has closed.
  const LoopTimingSnapshot& lastSnapshot() const { return snapshot_; }

  std::chrono::nanoseconds targetPeriod() const { return targetPeriod_; }
  std::chrono::nanoseconds reportingWindow() const { return reportingWindow_; }

 private:
  // Accumulated over the open window, in integer nanoseconds so that a long window loses no precision.
  struct Window {
    std::chrono::nanoseconds start{0};
    std::uint64_t cycles = 0;
    std::uint64_t overruns = 0;
    std::uint64_t missedPeriods = 0;
    std::uint64_t periods = 0;
    std::chrono::nanoseconds periodSum{0};
    std::chrono::nanoseconds minPeriod{0};
    std::chrono::nanoseconds maxPeriod{0};
    std::chrono::nanoseconds computeTimeSum{0};
    std::chrono::nanoseconds maxComputeTime{0};
    std::chrono::nanoseconds maxLateness{0};
  };

  void closeWindow(std::chrono::nanoseconds end);

  std::chrono::nanoseconds targetPeriod_;
  std::chrono::nanoseconds reportingWindow_;

  bool hasPreviousCycle_ = false;
  std::chrono::nanoseconds previousStart_{0};
  Window window_;

  std::uint64_t totalCycles_ = 0;
  std::uint64_t totalOverruns_ = 0;
  std::uint64_t totalMissedPeriods_ = 0;
  std::uint64_t windowsClosed_ = 0;

  LoopTimingSnapshot snapshot_;
};

}  // namespace robot::realtime
