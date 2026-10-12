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
#include <deque>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

#include "web_viewer/ViewerInput.h"

namespace robot::web_viewer {

/**
 * The bounded queue of input events from the web server's request threads to the viewer's render thread, which drains
 * it once per frame.
 *
 * It never grows past kCapacity: a push into a full queue drops the oldest event (counted by droppedCount()). A drag
 * that follows a drag with the same button and shift is merged into it (the motions add up, clamped to [-2, 2] like
 * one drag), so a burst of mouse motion between two frames costs one entry.
 *
 * Thread-safe.
 */
class ViewerInputQueue {
 public:
  // The most events the queue holds.
  static constexpr int kCapacity = 256;

  ViewerInputQueue() = default;
  ~ViewerInputQueue() = default;
  ViewerInputQueue(const ViewerInputQueue&) = delete;
  ViewerInputQueue& operator=(const ViewerInputQueue&) = delete;

  /** Appends `input`, or merges it into the last event when both are drags with the same button and shift. */
  void push(ViewerInput input);

  /** Returns every queued event, oldest first, and empties the queue. */
  std::vector<ViewerInput> drain();

  /** Returns the number of events queued now. */
  int size() const;

  /** Returns the number of events dropped because the queue was full, since construction. */
  int64_t droppedCount() const;

 private:
  mutable absl::Mutex mutex_;
  std::deque<ViewerInput> events_ ABSL_GUARDED_BY(mutex_);
  int64_t droppedCount_ ABSL_GUARDED_BY(mutex_) = 0;
};

}  // namespace robot::web_viewer
