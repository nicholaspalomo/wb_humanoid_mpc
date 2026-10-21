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

#include "web_viewer/ViewerFrameMailbox.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "web_viewer/EncodedFrame.h"
#include "web_viewer/ViewerStatus.h"

namespace robot::web_viewer {
namespace {

std::chrono::steady_clock::time_point steadyNow() {
  return std::chrono::steady_clock::now();
}

// The monotonic deadline `timeout` from now; a timeout too long for the clock is the clock's maximum.
std::chrono::steady_clock::time_point steadyDeadline(absl::Duration timeout) {
  const std::chrono::steady_clock::time_point now = steadyNow();
  if (timeout >= absl::FromChrono(std::chrono::steady_clock::time_point::max() - now)) {
    return std::chrono::steady_clock::time_point::max();
  }
  return now + absl::ToChronoNanoseconds(timeout);
}

// The time left until `deadline`, for the relative waits of absl::CondVar: these wait on the monotonic clock, while
// an absolute deadline waits on the wall clock.
absl::Duration remainingUntil(std::chrono::steady_clock::time_point deadline) {
  return absl::FromChrono(deadline - steadyNow());
}

}  // namespace

ViewerFrameMailbox::ViewerFrameMailbox(int maxClients) : maxClients_(maxClients) {
  ABSL_CHECK_GE(maxClients, 1) << "a ViewerFrameMailbox admits at least one client";
}

void ViewerFrameMailbox::publish(EncodedFrame frame) {
  {
    absl::MutexLock lock(&frameMutex_);
    frame.sequence = frame_.sequence + 1;
    frame.publishTime = absl::Now();
    frame.publishSteadyTime = steadyNow();
    std::swap(frame_, frame);
    frameCondition_.SignalAll();
  }
  // `frame` now holds the previous frame, freed here, outside the mutex.
}

std::optional<EncodedFrame> ViewerFrameMailbox::copyNewer(int64_t afterSequence, absl::Duration timeout) const {
  const std::chrono::steady_clock::time_point deadline = steadyDeadline(timeout);
  absl::MutexLock lock(&frameMutex_);
  while (!closed_.load() && frame_.sequence <= afterSequence) {
    const absl::Duration remaining = remainingUntil(deadline);
    if (remaining <= absl::ZeroDuration() || frameCondition_.WaitWithTimeout(&frameMutex_, remaining)) break;
  }
  if (closed_.load() || frame_.sequence <= afterSequence) return std::nullopt;
  return frame_;
}

std::optional<EncodedFrame> ViewerFrameMailbox::latestFrameMetadata() const {
  absl::MutexLock lock(&frameMutex_);
  if (frame_.sequence == 0) return std::nullopt;
  return EncodedFrame{.sequence = frame_.sequence,
                      .width = frame_.width,
                      .height = frame_.height,
                      .simTime = frame_.simTime,
                      .publishTime = frame_.publishTime,
                      .publishSteadyTime = frame_.publishSteadyTime,
                      .jpeg = {}};
}

void ViewerFrameMailbox::publishStatus(ViewerStatus status) {
  {
    absl::MutexLock lock(&statusMutex_);
    std::swap(status_, status);
  }
  // `status` now holds the previous status, freed here, outside the mutex.
}

ViewerStatus ViewerFrameMailbox::status() const {
  absl::MutexLock lock(&statusMutex_);
  return status_;
}

bool ViewerFrameMailbox::leaseHolds(std::chrono::steady_clock::time_point now) const {
  return lastFrameRequest_.has_value() && absl::FromChrono(now - *lastFrameRequest_) < kDemandLease;
}

bool ViewerFrameMailbox::waitForDemand(absl::Duration timeout) const {
  const std::chrono::steady_clock::time_point deadline = steadyDeadline(timeout);
  absl::MutexLock lock(&demandMutex_);
  while (true) {
    if (closed_.load()) return false;
    if (activeClients_.load() > 0) return true;
    if (leaseHolds(steadyNow())) return true;
    const absl::Duration remaining = remainingUntil(deadline);
    if (remaining <= absl::ZeroDuration()) return false;
    demandCondition_.WaitWithTimeout(&demandMutex_, remaining);
  }
}

void ViewerFrameMailbox::noteFrameRequest() {
  absl::MutexLock lock(&demandMutex_);
  lastFrameRequest_ = steadyNow();
  demandCondition_.SignalAll();
}

void ViewerFrameMailbox::noteViewer(absl::string_view viewerId) {
  const std::chrono::steady_clock::time_point now = steadyNow();
  absl::MutexLock lock(&demandMutex_);
  absl::erase_if(viewers_, [now](const std::pair<const std::string, std::chrono::steady_clock::time_point>& viewer) {
    return absl::FromChrono(now - viewer.second) >= kDemandLease;
  });
  if (!viewers_.contains(viewerId) && static_cast<int>(viewers_.size()) >= kMaxTrackedViewers) {
    absl::flat_hash_map<std::string, std::chrono::steady_clock::time_point>::iterator oldest = viewers_.begin();
    for (absl::flat_hash_map<std::string, std::chrono::steady_clock::time_point>::iterator viewer = viewers_.begin();
         viewer != viewers_.end(); ++viewer) {
      if (viewer->second < oldest->second) oldest = viewer;
    }
    viewers_.erase(oldest);
  }
  viewers_.insert_or_assign(std::string(viewerId), now);
}

int ViewerFrameMailbox::viewerCount() const {
  const std::chrono::steady_clock::time_point now = steadyNow();
  absl::MutexLock lock(&demandMutex_);
  int count = 0;
  for (const std::pair<const std::string, std::chrono::steady_clock::time_point>& viewer : viewers_) {
    if (absl::FromChrono(now - viewer.second) < kDemandLease) ++count;
  }
  return count;
}

std::optional<std::chrono::steady_clock::time_point> ViewerFrameMailbox::demandResumedAt() const {
  absl::MutexLock lock(&demandMutex_);
  return demandResumedAt_;
}

bool ViewerFrameMailbox::tryAcquireClient() {
  int admitted = activeClients_.load();
  while (admitted < maxClients_) {
    if (activeClients_.compare_exchange_weak(admitted, admitted + 1)) {
      // waitForDemand() reads the count under demandMutex_: taking it here means a waiter either saw the new count or
      // is already waiting and receives the signal.
      absl::MutexLock lock(&demandMutex_);
      const std::chrono::steady_clock::time_point now = steadyNow();
      if (admitted == 0 && !leaseHolds(now)) demandResumedAt_ = now;
      demandCondition_.SignalAll();
      return true;
    }
  }
  return false;
}

void ViewerFrameMailbox::releaseClient() {
  int admitted = activeClients_.load();
  while (admitted > 0) {
    if (activeClients_.compare_exchange_weak(admitted, admitted - 1)) return;
  }
  LOG(ERROR) << "ViewerFrameMailbox::releaseClient() without an admitted client; the count stays 0";
}

int ViewerFrameMailbox::activeClients() const {
  return activeClients_.load();
}

void ViewerFrameMailbox::close() {
  closed_.store(true);
  {
    absl::MutexLock lock(&frameMutex_);
    frameCondition_.SignalAll();
  }
  absl::MutexLock lock(&demandMutex_);
  demandCondition_.SignalAll();
}

bool ViewerFrameMailbox::closed() const {
  return closed_.load();
}

}  // namespace robot::web_viewer
