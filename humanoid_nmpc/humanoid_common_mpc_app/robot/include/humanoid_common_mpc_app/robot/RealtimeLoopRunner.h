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

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"

#include "robot_core/TripleBuffer.h"
#include "robot_realtime/LoopTimingSnapshot.h"
#include "robot_realtime/PeriodicTimer.h"
#include "robot_realtime/RealtimeThread.h"

namespace ocs2::humanoid {

/** The realtime thread "robot_rt": not realtime (priority 0), not pinned, memory locked when it is given a priority. */
robot::realtime::RealtimeThreadConfig defaultRealtimeThreadConfig();

/** How the realtime loop of the robot process runs. */
struct RealtimeLoopConfig {
  /** The control period, 1 / mrtDesiredFrequency. */
  std::chrono::nanoseconds period{std::chrono::milliseconds(2)};
  /**
   * The thread's name, SCHED_FIFO priority (0: not realtime, --realtime_priority) and cores (empty: not pinned). With a
   * priority the process's memory is locked and the stack prefaulted (robot_realtime/RealtimeThread.h).
   */
  robot::realtime::RealtimeThreadConfig thread = defaultRealtimeThreadConfig();
  /** The window LoopTimingStats condenses into one snapshot: robot/loop_timing goes out once per window. */
  std::chrono::nanoseconds reportingWindow{std::chrono::seconds(1)};
  /** A controller acts on the newest state rather than replaying the cycles it missed. */
  robot::realtime::OverrunPolicy overrunPolicy = robot::realtime::OverrunPolicy::kSkipMissedPeriods;
};

/**
 * Runs the realtime loop of the robot process on a thread of its own: the cycle function once per period, paced on
 * absolute deadlines (robot::realtime::PeriodicTimer: clock_nanosleep with TIMER_ABSTIME on CLOCK_MONOTONIC, so neither
 * the compute time nor the wake-up latency accumulates into drift), with the period, compute time and overruns of every
 * cycle accumulated by robot::realtime::LoopTimingStats. When a reporting window closes, its snapshot goes through a
 * robot::TripleBuffer to the communication thread, which takes it with takeTimingSnapshot() and publishes
 * robot/loop_timing. Nothing the runner itself does in a cycle allocates, locks or logs.
 *
 * The thread blocks SIGINT and SIGTERM, which the main thread waits for, and configures itself
 * (robot::realtime::configureCurrentThread()): name, cores and, with a priority,
 * mlockall, a prefaulted stack and SCHED_FIFO. A step the process may not take (no CAP_SYS_NICE, an rtprio limit of
 * 0, a core outside the container's cpuset) is reported by start() and skipped; the loop runs regardless, as the old
 * sims ran without any of it.
 *
 * stop() asks the loop to finish its current cycle and joins the thread; a stopped runner cannot be restarted.
 *
 * A CYCLE THAT THROWS ends the loop instead of the process: the exception is caught on the thread, its message kept
 * (faultMessage()), the fault function of start() run once on the thread - the robot process puts its backend in the
 * safe state there - and faulted() turns true, for the main thread to act on. Without this boundary the exception would
 * leave std::thread, std::terminate would abort the process, and nothing would take the actuators off the last action.
 */
class RealtimeLoopRunner {
 public:
  using CycleFunction = std::function<void()>;
  /** Run on the realtime thread after a cycle threw, before the loop ends. Must not throw. */
  using FaultFunction = std::function<void()>;

  explicit RealtimeLoopRunner(RealtimeLoopConfig config);
  ~RealtimeLoopRunner();

  RealtimeLoopRunner(const RealtimeLoopRunner&) = delete;
  RealtimeLoopRunner& operator=(const RealtimeLoopRunner&) = delete;

  /**
   * Starts the thread, which runs `cycle` once per period until stop() or until it throws (then `fault`, if set, see A
   * CYCLE THAT THROWS). Waits until the thread has configured itself and returns the outcome of that (OK, or the steps
   * that did not take effect); the loop runs either way. A second start() is FailedPrecondition.
   */
  absl::Status start(CycleFunction cycle, FaultFunction fault = nullptr);

  /** Ends the loop after its current cycle and joins the thread. Idempotent; called by the destructor. */
  void stop();

  bool isRunning() const { return running_.load(std::memory_order_acquire); }

  /**
   * The snapshot of the newest reporting window, if one closed since the last call; for one consumer thread (the
   * communication thread). Lock-free.
   */
  bool takeTimingSnapshot(robot::realtime::LoopTimingSnapshot& snapshot);

  /** Cycles run so far. Any thread. */
  std::uint64_t cycles() const { return cycles_.load(std::memory_order_relaxed); }

  /** True once a cycle has thrown and the loop has ended. Any thread. */
  bool faulted() const { return faulted_.load(std::memory_order_acquire); }
  /** What the cycle threw (its what(), cut to 255 characters); empty until faulted(). Any thread. */
  std::string faultMessage() const;

  const RealtimeLoopConfig& config() const { return config_; }

 private:
  void run();
  /** The realtime thread's handling of a cycle that threw `message`. */
  void recordFault(const char* message);

  const RealtimeLoopConfig config_;
  CycleFunction cycle_;
  FaultFunction fault_;
  std::atomic<bool> faulted_{false};
  std::array<char, 256> faultMessage_{};  // written by the realtime thread before faulted_ is set
  std::atomic<bool> stopRequested_{false};
  std::atomic<bool> running_{false};
  std::atomic<std::uint64_t> cycles_{0};
  robot::TripleBuffer<robot::realtime::LoopTimingSnapshot> timing_;

  absl::Mutex lifecycleMutex_;
  std::thread thread_ ABSL_GUARDED_BY(lifecycleMutex_);
  bool started_ ABSL_GUARDED_BY(lifecycleMutex_) = false;
  absl::Notification configured_;
  absl::Status configureStatus_;  // written by the thread before configured_ is notified
};

}  // namespace ocs2::humanoid
