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

namespace robot::ipc {

/**
 * What a bus did with one topic since it was created. Once a drain of the socket has dispatched,
 * received == delivered + superseded + rejected.
 */
struct TopicStatistics {
  /**
   * Messages written to the PUB socket. ZeroMQ may still drop one at the high-water mark of a slow or absent
   * subscriber, which no counter of the publisher can see (a PUB socket never reports it).
   */
  uint64_t sent = 0;
  /** Bus::publish() calls refused because the queue to the IO thread was full (BusOptions::publishQueueCapacity). */
  uint64_t sendDropped = 0;
  /** Messages read from the SUB socket with exactly this topic. */
  uint64_t received = 0;
  /** Messages handed to the subscription's handler (including those whose handler then threw). */
  uint64_t delivered = 0;
  /** Delivery::kLatest only: messages replaced by a newer one of the same drain before they were parsed. */
  uint64_t superseded = 0;
  /** Messages that were not three frames, carried another type than the subscription's, or did not parse. */
  uint64_t rejected = 0;
  /** Handler calls that threw; the bus logs and counts them and carries on. */
  uint64_t handlerErrors = 0;
};

}  // namespace robot::ipc
