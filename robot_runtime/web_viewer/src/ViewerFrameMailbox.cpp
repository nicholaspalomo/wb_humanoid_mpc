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
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "web_viewer/EncodedFrame.h"
#include "web_viewer/ViewerStatus.h"

namespace robot::web_viewer {

ViewerFrameMailbox::ViewerFrameMailbox(int maxClients) : maxClients_(maxClients) {
  ABSL_CHECK_GE(maxClients, 1) << "a ViewerFrameMailbox admits at least one client";
}

void ViewerFrameMailbox::publish(EncodedFrame frame) {
  {
    absl::MutexLock lock(&frameMutex_);
    frame.sequence = frame_.sequence + 1;
    frame.publishTime = absl::Now();
    std::swap(frame_, frame);
    frameCondition_.SignalAll();
  }
  // `frame` now holds the previous frame, freed here, outside the mutex.
}

std::optional<EncodedFrame> ViewerFrameMailbox::copyNewer(int64_t afterSequence, absl::Duration timeout) const {
  const absl::Time deadline = absl::Now() + timeout;
  absl::MutexLock lock(&frameMutex_);
  while (!closed_.load() && frame_.sequence <= afterSequence) {
    if (frameCondition_.WaitWithDeadline(&frameMutex_, deadline)) break;
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

bool ViewerFrameMailbox::waitForDemand(absl::Duration timeout) const {
  const absl::Time deadline = absl::Now() + timeout;
  absl::MutexLock lock(&demandMutex_);
  while (true) {
    if (closed_.load()) return false;
    if (activeClients_.load() > 0) return true;
    const absl::Time now = absl::Now();
    if (now - lastFrameRequest_ < kDemandLease) return true;
    if (now >= deadline) return false;
    demandCondition_.WaitWithDeadline(&demandMutex_, deadline);
  }
}

void ViewerFrameMailbox::noteFrameRequest() {
  absl::MutexLock lock(&demandMutex_);
  lastFrameRequest_ = absl::Now();
  demandCondition_.SignalAll();
}

bool ViewerFrameMailbox::tryAcquireClient() {
  int admitted = activeClients_.load();
  while (admitted < maxClients_) {
    if (activeClients_.compare_exchange_weak(admitted, admitted + 1)) {
      // waitForDemand() reads the count under demandMutex_: taking it here means a waiter either saw the new count or
      // is already waiting and receives the signal.
      absl::MutexLock lock(&demandMutex_);
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
