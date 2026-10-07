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

#include "humanoid_common_mpc_app/robot/RealtimeLoopRunner.h"

#include <pthread.h>
#include <signal.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"

#include "robot_realtime/LoopTimingStats.h"

namespace ocs2::humanoid {

robot::realtime::RealtimeThreadConfig defaultRealtimeThreadConfig() {
  robot::realtime::RealtimeThreadConfig config;
  config.name = "robot_rt";
  return config;
}

RealtimeLoopRunner::RealtimeLoopRunner(RealtimeLoopConfig config) : config_(std::move(config)) {}

RealtimeLoopRunner::~RealtimeLoopRunner() {
  stop();
}

absl::Status RealtimeLoopRunner::start(CycleFunction cycle, FaultFunction fault) {
  if (cycle == nullptr) {
    return absl::InvalidArgumentError("RealtimeLoopRunner::start(): the cycle function is empty");
  }
  if (config_.period <= std::chrono::nanoseconds::zero()) {
    return absl::InvalidArgumentError(absl::StrCat("RealtimeLoopRunner: the period must be positive, got ", config_.period.count(), " ns"));
  }
  {
    absl::MutexLock lock(lifecycleMutex_);
    if (started_) {
      return absl::FailedPreconditionError("RealtimeLoopRunner::start(): the loop was started before; a runner runs once");
    }
    started_ = true;
    cycle_ = std::move(cycle);
    fault_ = std::move(fault);
    running_.store(true, std::memory_order_release);
    thread_ = std::thread([this]() { run(); });
  }
  configured_.WaitForNotification();
  return configureStatus_;
}

void RealtimeLoopRunner::stop() {
  stopRequested_.store(true, std::memory_order_release);
  absl::MutexLock lock(lifecycleMutex_);
  if (thread_.joinable()) {
    thread_.join();
  }
}

bool RealtimeLoopRunner::takeTimingSnapshot(robot::realtime::LoopTimingSnapshot& snapshot) {
  if (!timing_.acquireRead()) {
    return false;
  }
  snapshot = timing_.readSlot();
  return true;
}

void RealtimeLoopRunner::run() {
  // The process's shutdown signals go to the other threads: a handler running on this thread would steal its time.
  sigset_t shutdownSignals;
  sigemptyset(&shutdownSignals);
  sigaddset(&shutdownSignals, SIGINT);
  sigaddset(&shutdownSignals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &shutdownSignals, /*oldset=*/nullptr);
  configureStatus_ = robot::realtime::configureCurrentThread(config_.thread);
  // Everything the loop needs is built before the first cycle, on this thread, so that the cycles allocate nothing.
  robot::realtime::LoopTimingStats statistics(config_.period, config_.reportingWindow);
  robot::realtime::PeriodicTimer timer(config_.period, config_.overrunPolicy);
  configured_.Notify();

  timer.start();
  while (!stopRequested_.load(std::memory_order_acquire)) {
    const robot::realtime::TimerWakeup wakeup = timer.waitForNextPeriod();
    if (stopRequested_.load(std::memory_order_acquire)) break;
    // The cycle runs the controller, which calls OCS2 and Pinocchio code that throws: the class comment's A CYCLE THAT
    // THROWS boundary, which puts the robot in its safe state instead of letting the exception end the process.
    try {  // NOLINT(exceptions): see above
      cycle_();
    } catch (const std::exception& error) {  // NOLINT(exceptions): see above
      recordFault(error.what());
      break;
    } catch (...) {  // NOLINT(exceptions): see above
      recordFault("an exception that is not a std::exception");
      break;
    }
    cycles_.fetch_add(1, std::memory_order_relaxed);
    if (statistics.addCycle(wakeup, robot::realtime::monotonicNow())) {
      timing_.writeSlot() = statistics.lastSnapshot();
      timing_.publishWrite();
    }
  }
  running_.store(false, std::memory_order_release);
}

void RealtimeLoopRunner::recordFault(const char* absl_nonnull message) {
  const size_t length = std::min(std::strlen(message), faultMessage_.size() - 1);
  std::memcpy(faultMessage_.data(), message, length);
  faultMessage_[length] = '\0';
  if (fault_ != nullptr) {
    try {  // NOLINT(exceptions): the fault function must not throw; if it does, the loop ends all the same
      fault_();
    } catch (...) {  // NOLINT(exceptions, bugprone-empty-catch): see above; the cycle's fault is what faultMessage() reports
    }
  }
  faulted_.store(true, std::memory_order_release);
}

std::string RealtimeLoopRunner::faultMessage() const {
  if (!faulted()) return std::string();
  return std::string(faultMessage_.data(), ::strnlen(faultMessage_.data(), faultMessage_.size()));
}

}  // namespace ocs2::humanoid
