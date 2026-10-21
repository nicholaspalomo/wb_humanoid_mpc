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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "web_viewer/EncodedFrame.h"
#include "web_viewer/ViewerStatus.h"

namespace robot::web_viewer {

/**
 * The hand-over between the viewer's render thread and the web server's request threads: the newest frame, the newest
 * status, the render demand and the count of frame clients.
 *
 * The render thread publishes each frame and status; it never waits on a socket, only on the mutex that a request
 * thread holds while it copies the frame into its response (about 10 us for 100 KB). Older frames are dropped: a
 * client that is slower than the renderer skips frames. The mailbox holds the frame by value, so a reader always copies
 * a whole frame and never sees one being replaced.
 *
 * Demand: waitForDemand() lets the render thread idle while nobody watches. A frame client counts as demand while it
 * is admitted (tryAcquireClient() to releaseClient()), and each frame request (noteFrameRequest()) for kDemandLease
 * after it. demandResumedAt() is when the last admission found no demand, so the renderer was idle or about to be: a
 * frame published before it is stale.
 *
 * Viewers: each page names itself on its frame requests (noteViewer()); viewerCount() is the pages seen within
 * kDemandLease, at most kMaxTrackedViewers.
 *
 * Every time is measured on the monotonic clock, so that a step of the wall clock (NTP, a VM resync) neither extends
 * the lease nor stretches a wait.
 *
 * Thread-safe: every method may be called from any thread. close() wakes every waiter for good.
 */
class ViewerFrameMailbox {
 public:
  // How long a frame request keeps the renderer busy after it was made, and a page counts as a viewer.
  static constexpr absl::Duration kDemandLease = absl::Seconds(1);
  // The most page ids viewerCount() keeps; a request with a new id beyond them replaces the oldest, so that a client
  // inventing ids cannot grow the map.
  static constexpr int kMaxTrackedViewers = 256;

  /** Creates an empty mailbox that admits at most `maxClients` (>= 1) frame clients at a time. */
  explicit ViewerFrameMailbox(int maxClients);
  ~ViewerFrameMailbox() = default;

  ViewerFrameMailbox(const ViewerFrameMailbox&) = delete;
  ViewerFrameMailbox& operator=(const ViewerFrameMailbox&) = delete;

  /**
   * Replaces the newest frame with `frame` and wakes the requests waiting for it. Stamps `frame.sequence` (one more than
   * the previous frame's, 1 for the first), `frame.publishTime` and `frame.publishSteadyTime` (now); the caller's values
   * of these are ignored. The previous frame is freed after the mutex is released.
   */
  void publish(EncodedFrame frame);

  /**
   * Returns a copy of the newest frame once its sequence exceeds `afterSequence`: at once when it already does,
   * otherwise when one is published within `timeout`. Returns nullopt on the timeout and once the mailbox is closed.
   */
  std::optional<EncodedFrame> copyNewer(int64_t afterSequence, absl::Duration timeout) const;

  /** Returns the newest frame without its JPEG bytes (`jpeg` empty), or nullopt before the first one. Never waits. */
  std::optional<EncodedFrame> latestFrameMetadata() const;

  /** Replaces the newest status with `status`. */
  void publishStatus(ViewerStatus status);

  /** Returns a copy of the newest status; a default-constructed one before the first publishStatus(). */
  ViewerStatus status() const;

  /**
   * Returns true as soon as somebody wants frames: a frame client is admitted, or a frame was requested less than
   * kDemandLease ago. Returns false when `timeout` passes without demand, and at once once the mailbox is closed.
   */
  bool waitForDemand(absl::Duration timeout) const;

  /** Records a frame request, which counts as demand for kDemandLease, and wakes waitForDemand(). */
  void noteFrameRequest();

  /** Records a frame request of the page `viewerId`, which counts as a viewer for kDemandLease. */
  void noteViewer(absl::string_view viewerId);

  /** Returns the number of distinct page ids noted within kDemandLease. */
  int viewerCount() const;

  /**
   * Returns when tryAcquireClient() last admitted a client with no demand before it (none admitted, the lease lapsed),
   * on the monotonic clock; nullopt before the first admission. Frames published before it were drawn before the
   * renderer idled.
   */
  std::optional<std::chrono::steady_clock::time_point> demandResumedAt() const;

  /**
   * Admits one more frame client if fewer than maxClients() are admitted and returns true; returns false otherwise. A
   * client that was admitted calls releaseClient() exactly once when it leaves. An admission with no demand before it
   * moves demandResumedAt().
   */
  bool tryAcquireClient();

  /** Releases a client that tryAcquireClient() admitted. */
  void releaseClient();

  /** Returns the number of frame clients admitted now. */
  int activeClients() const;

  /** Returns the most frame clients admitted at a time, as given to the constructor. */
  int maxClients() const { return maxClients_; }

  /** Closes the mailbox: every waiting copyNewer() and waitForDemand() returns at once, and every later one too. */
  void close();

  /** Returns true once close() was called. */
  bool closed() const;

 private:
  const int maxClients_;
  std::atomic<int> activeClients_ = 0;
  std::atomic<bool> closed_ = false;

  mutable absl::Mutex frameMutex_;
  mutable absl::CondVar frameCondition_;
  // sequence 0 until the first publish().
  EncodedFrame frame_ ABSL_GUARDED_BY(frameMutex_);

  // Whether a frame was requested less than kDemandLease before `now`.
  bool leaseHolds(std::chrono::steady_clock::time_point now) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(demandMutex_);

  mutable absl::Mutex demandMutex_;
  mutable absl::CondVar demandCondition_;
  // Nullopt until the first frame request.
  std::optional<std::chrono::steady_clock::time_point> lastFrameRequest_ ABSL_GUARDED_BY(demandMutex_) = std::nullopt;
  std::optional<std::chrono::steady_clock::time_point> demandResumedAt_ ABSL_GUARDED_BY(demandMutex_) = std::nullopt;
  // The last request of each page id, pruned to kDemandLease on every noteViewer().
  absl::flat_hash_map<std::string, std::chrono::steady_clock::time_point> viewers_ ABSL_GUARDED_BY(demandMutex_);

  mutable absl::Mutex statusMutex_;
  ViewerStatus status_ ABSL_GUARDED_BY(statusMutex_);
};

}  // namespace robot::web_viewer
