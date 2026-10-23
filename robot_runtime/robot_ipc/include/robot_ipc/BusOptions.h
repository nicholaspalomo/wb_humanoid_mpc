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

#include <cstddef>
#include <string>

#include "absl/time/time.h"

#include "robot_ipc/NetworkConfig.h"

namespace robot::ipc {

/** How Bus::Create() sets up a bus. The defaults suit every process of the humanoid runtime. */
struct BusOptions {
  /**
   * The node this process publishes as: the bus uses its node's network interface and creates the DDS writers of the
   * topics the network file lists it as a publisher of. Empty: the bus only subscribes (a monitoring tool).
   */
  std::string nodeName;
  /** The DDS network: domain, discovery peers, topic QoS and transport (NetworkConfig.h). */
  NetworkConfig network;

  // The Python bus has the same defaults.
  // LINT.IfChange(bus_defaults)
  /** The defaults of the fields below. */
  static constexpr absl::Duration kDefaultIoPollPeriod = absl::Milliseconds(100);
  static constexpr size_t kDefaultPublishQueueCapacity = 1024;
  static constexpr size_t kDefaultMaxMessagesPerDrain = 4096;
  static constexpr int kDefaultLatestDeliveryHistoryDepth = 4;
  static constexpr int kDefaultAllDeliveryHistoryDepth = 256;

  /**
   * The longest the IO thread waits for something to happen. publish(), stop(), arriving samples and due periodic
   * callbacks all wake it at once, so this only bounds how stale a wait (and the DDS status in TopicStatistics) gets.
   */
  absl::Duration ioPollPeriod = kDefaultIoPollPeriod;
  /** Messages publish() queues for the IO thread before it refuses (counted as sendDropped). */
  size_t publishQueueCapacity = kDefaultPublishQueueCapacity;
  /** The most samples one drain takes, across topics, before the IO thread dispatches and serves its other work. */
  size_t maxMessagesPerDrain = kDefaultMaxMessagesPerDrain;
  /**
   * The KEEP_LAST depth of the DDS reader of a Delivery::kLatest subscription. More than one, so that a sample of the
   * wrong type does not displace a valid one before the drain sees both; the drain then hands over only the newest
   * (the others are superseded, unparsed), and what the reader replaced before the drain is counted as missed.
   */
  int latestDeliveryHistoryDepth = kDefaultLatestDeliveryHistoryDepth;
  /**
   * The KEEP_LAST depth of the DDS reader of a Delivery::kAll subscription: the samples it holds while the IO thread
   * is busy before the oldest are replaced (counted as missed).
   */
  int allDeliveryHistoryDepth = kDefaultAllDeliveryHistoryDepth;
  // LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/bus.py:bus_defaults)
};

}  // namespace robot::ipc
