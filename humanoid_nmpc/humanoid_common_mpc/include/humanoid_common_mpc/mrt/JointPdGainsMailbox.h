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
#include <cstddef>
#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "robot_core/TripleBuffer.h"

namespace ocs2::humanoid {

/**
 * Hands JointPdGains from the threads that parse them (the robot process's communication thread, with the GUI's
 * operator/pd_gains and the gains file watcher) to the realtime control thread of an MRT joint controller.
 *
 * post() may be called from any number of non-realtime threads; they take turns through a mutex the control thread never
 * takes. receive() is for the control thread alone, and it neither locks nor waits nor allocates: the gains move through
 * a robot::TripleBuffer of preallocated slots, all sized like the initial gains, so a producer that is slow or holds the
 * mutex never delays a control cycle. The control thread sees the newest gains posted since its last receive(), or none;
 * gains posted in between are superseded.
 *
 * "Newest" is the order in which the documents were handed in, not the order in which their parses finished: a producer
 * takes a ticket (takeTicket()) when a document arrives, before it reads or parses it, and posts the gains with it. Gains
 * whose ticket is older than that of gains already posted are dropped, so a document handed in earlier but slower to
 * parse never overwrites a newer one.
 */
class JointPdGainsMailbox {
 public:
  /** Every slot starts as `initial`, whose dimensions every posted set must have. */
  explicit JointPdGainsMailbox(const JointPdGains& initial);

  JointPdGainsMailbox(const JointPdGainsMailbox&) = delete;
  JointPdGainsMailbox& operator=(const JointPdGainsMailbox&) = delete;
  ~JointPdGainsMailbox() = default;

  /** Any thread: the place in line of a document handed in now, for post(). Take it before the document is parsed. */
  uint64_t takeTicket() { return nextTicket_.fetch_add(1) + 1; }

  /**
   * Any non-realtime thread. InvalidArgument, and nothing posted, when `gains` are not sized like the initial gains.
   * Gains whose `ticket` is older than the ticket of gains already posted are dropped and OK is returned: a document
   * handed in later has superseded them, as a later post supersedes one the control thread has not received yet.
   */
  absl::Status post(const JointPdGains& gains, uint64_t ticket) ABSL_LOCKS_EXCLUDED(producerMutex_);

  /** post() with a ticket taken now, for a producer that has nothing to parse. */
  absl::Status post(const JointPdGains& gains) ABSL_LOCKS_EXCLUDED(producerMutex_) { return post(gains, takeTicket()); }

  /**
   * Control thread. Copies the newest gains posted since the last call into `active` and returns true, or returns false
   * and leaves `active` alone when nothing was posted. `active` must be sized like the initial gains, which makes the
   * copy allocation-free.
   */
  bool receive(JointPdGains& active);

 private:
  friend class JointPdGainsMailboxTestPeer;

  const size_t numMpcJoints_;
  const size_t numOtherJoints_;
  std::atomic<uint64_t> nextTicket_{0};  // the last ticket taken; tickets start at 1
  absl::Mutex producerMutex_;
  uint64_t lastPostedTicket_ ABSL_GUARDED_BY(producerMutex_) = 0;
  // The write slot is the producers', under producerMutex_; the read slot is the control thread's.
  robot::TripleBuffer<JointPdGains> buffer_;
};

}  // namespace ocs2::humanoid
