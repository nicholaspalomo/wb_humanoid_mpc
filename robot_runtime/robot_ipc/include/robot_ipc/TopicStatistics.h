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
 * dispatched, received == delivered + superseded + rejected. The last three
 * fields count the liveliness tokens of the other buses (KeyExprs.h), as the
 * IO thread last saw them.
 */
struct TopicStatistics {
  // The Python bus counts the same (robot_ipc/bus.py TopicStatistics).
  // LINT.IfChange(topic_statistics)
  /**
   * Samples handed to the topic's Zenoh publisher - not samples delivered: a
   * congested link drops a sample of a dropping profile (QosProfile.h). The
   * subscriber's `missed` shows what a link lost.
   */
  uint64_t written = 0;
  /**
   * Samples not written: Bus::publish() refused because the queue to the IO
   * thread was full (BusOptions::publishQueueCapacity), or Zenoh refused the
   * put.
   */
  uint64_t sendDropped = 0;
  /** Samples taken from the subscription's queue. */
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
   * Samples a writer wrote that the subscription never took, from the gaps in
   * the writers' sequence numbers: replaced in the subscription's queue while
   * the IO thread was busy (BusOptions' delivery history depths), or dropped by
   * a congested link. Counted from the second sample of each writer on: what a
   * writer wrote before this subscriber was declared is not a gap.
   */
  uint64_t missed = 0;
  /**
   * The publishers and subscribers of the topic, on any bus, whose profile is
   * another than this bus's: two different network files. They exchange no
   * samples (the profile is a chunk of the key expression).
   */
  uint64_t incompatibleQos = 0;
  /** The subscribers of the topic with this bus's profile, on any bus this one
   * included, while this bus has a publisher of it: those that are up. */
  uint64_t matchedReaders = 0;
  /** The publishers of the topic with this bus's profile, on any bus this one
   * included, while this bus subscribes to it: those that are up. */
  uint64_t matchedWriters = 0;
  // LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/bus.py:topic_statistics)
};

}  // namespace robot::ipc
