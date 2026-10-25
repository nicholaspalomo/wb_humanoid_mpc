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
 * What a bus did with one topic since it was created. Once a drain has
 * dispatched, received == delivered + superseded + rejected. The last four
 * fields are DDS's own account, refreshed by the IO thread at most every
 * BusOptions::ioPollPeriod.
 */
struct TopicStatistics {
  // The Python bus counts the same (robot_ipc/bus.py TopicStatistics).
  // LINT.IfChange(topic_statistics)
  /**
   * Samples handed to the topic's DDS writer - not samples delivered: a
   * BEST_EFFORT sample can still be lost on the link, and a newer sample
   * replaces one a reliable KEEP_LAST writer is still repairing. The
   * subscriber's `missed` and `sampleLost` show what a link lost.
   */
  uint64_t written = 0;
  /**
   * Samples not written: Bus::publish() refused because the queue to the IO
   * thread was full (BusOptions::publishQueueCapacity), or DDS refused the
   * write (a reliable writer at its unacknowledged-bytes limit,
   * TransportOptions::writerUnackedBytesLimit, which the bus never waits for).
   */
  uint64_t sendDropped = 0;
  /** Samples taken from the topic's DDS reader. */
  uint64_t received = 0;
  /** Samples handed to the subscription's handler (including those whose
   * handler then threw). */
  uint64_t delivered = 0;
  /** Delivery::kLatest only: samples replaced by a newer one of the same topic
   * in the same drain before they were parsed. */
  uint64_t superseded = 0;
  /**
   * Samples refused: another wire protocol version (WireProtocol.h), another
   * type than the subscription's, or a payload that does not parse.
   */
  uint64_t rejected = 0;
  /** Handler calls that threw; the bus logs and counts them and carries on. */
  uint64_t handlerErrors = 0;
  /**
   * Samples a matched writer wrote that the reader never took, from the gaps in
   * the writers' sequence numbers: replaced in a KEEP_LAST history (the
   * writer's, or the reader's while the IO thread was busy) or, under
   * BEST_EFFORT, lost on the link. Counted from the second sample of each
   * writer on: what a writer wrote before this reader matched it is not a gap.
   */
  uint64_t missed = 0;
  /** DDS SAMPLE_LOST of the reader: samples DDS knows were lost on the way (a
   * gap it saw in a BEST_EFFORT stream). */
  uint64_t sampleLost = 0;
  /** DDS SAMPLE_REJECTED of the reader: samples DDS dropped for the reader's
   * resource limits. */
  uint64_t sampleRejected = 0;
  /**
   * DDS REQUESTED_ and OFFERED_INCOMPATIBLE_QOS of the topic's reader and
   * writer: remote endpoints that will not match them, e.g. a RELIABLE reader
   * and a BEST_EFFORT writer from two different network files.
   */
  uint64_t incompatibleQos = 0;
  /** The DDS readers this bus's writer of the topic currently matches: the
   * subscribers that are up. */
  uint64_t matchedReaders = 0;
  /** The DDS writers this bus's reader of the topic currently matches: the
   * publishers that are up. */
  uint64_t matchedWriters = 0;
  // LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/bus.py:topic_statistics)
};

}  // namespace robot::ipc
