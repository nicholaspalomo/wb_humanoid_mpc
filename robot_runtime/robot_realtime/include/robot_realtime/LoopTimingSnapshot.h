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

#include <cstdint>
#include <type_traits>

namespace robot::realtime {

/**
 * How well the realtime loop held its period: what LoopTimingStats reports at the end of each reporting window.
 *
 * Plain data and trivially copyable, so that the realtime thread can hand it to the communication thread through a
 * robot::TripleBuffer<LoopTimingSnapshot> without allocating. Times are in seconds, as in the message.
 *
 * Onto humanoid_mpc_msgs.LoopTiming (loop_timing.proto), which the communication thread fills:
 *   target_period_s    <- targetPeriodS
 *   cycles             <- totalCycles
 *   mean_period_s      <- meanPeriodS
 *   max_period_s       <- maxPeriodS
 *   max_compute_time_s <- maxComputeTimeS
 *   overruns           <- totalOverruns
 *   missed_periods     <- totalMissedPeriods
 *   max_lateness_s     <- maxLatenessS
 * The remaining LoopTiming fields (telemetry_samples_dropped, stale_policies_dropped, policy_age_s) come from the
 * telemetry ring (SpscQueue::droppedCount()) and the MPC link, not from the loop's timing.
 */
struct LoopTimingSnapshot {
  /// The number of the reporting window this snapshot closes, from 1; 0 until the first window has closed.
  std::uint64_t window = 0;
  double targetPeriodS = 0.0;

  // Since the statistics were constructed.
  std::uint64_t totalCycles = 0;
  /// Cycles whose compute time exceeded the target period.
  std::uint64_t totalOverruns = 0;
  /// Deadlines the timer dropped (TimerWakeup::missedPeriods), i.e. cycles that never ran.
  std::uint64_t totalMissedPeriods = 0;

  // Over the reporting window.
  /// From the start of the window's first period to the start of its last cycle.
  double windowDurationS = 0.0;
  std::uint64_t windowCycles = 0;
  std::uint64_t windowOverruns = 0;
  std::uint64_t windowMissedPeriods = 0;
  /// The time between the starts of consecutive cycles. All zero when the window measured no period.
  double meanPeriodS = 0.0;
  double minPeriodS = 0.0;
  double maxPeriodS = 0.0;
  /// The time from a cycle's start to the end of its work.
  double meanComputeTimeS = 0.0;
  double maxComputeTimeS = 0.0;
  /// The longest wake-up latency past a deadline (TimerWakeup::lateness).
  double maxLatenessS = 0.0;
};

static_assert(std::is_trivially_copyable_v<LoopTimingSnapshot>, "the snapshot must be copyable through a TripleBuffer without allocating");

}  // namespace robot::realtime
