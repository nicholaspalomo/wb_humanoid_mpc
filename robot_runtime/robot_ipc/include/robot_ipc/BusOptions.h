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
   * The node this process publishes as: the bus binds its endpoint of the network. Empty: the bus only subscribes and
   * binds nothing (a monitoring tool).
   */
  std::string nodeName;
  /** The nodes; the SUB socket connects to every one of them, this process's own included. */
  NetworkConfig network;

  // The Python bus has the same defaults.
  // LINT.IfChange(bus_defaults)
  /**
   * The longest the IO thread waits for something to happen. publish(), stop(), arriving messages and due periodic
   * callbacks all wake it at once, so this only bounds how stale a wait can get.
   */
  absl::Duration ioPollPeriod = absl::Milliseconds(100);
  /** ZMQ_SNDHWM: messages queued per subscriber before ZeroMQ drops for it (0: no limit). */
  int sendHighWaterMark = 1000;
  /** ZMQ_RCVHWM: messages queued per publisher before the connection pushes back (0: no limit). */
  int receiveHighWaterMark = 1000;
  /** Messages publish() queues for the IO thread before it refuses (counted as sendDropped). */
  std::size_t publishQueueCapacity = 1024;
  /** The most messages one drain reads before the IO thread dispatches and serves its other work. */
  std::size_t maxMessagesPerDrain = 4096;

  /** ZMTP heartbeats (ZMQ_HEARTBEAT_IVL / _TIMEOUT / _TTL): a peer silent for heartbeatTimeout is disconnected. */
  absl::Duration heartbeatInterval = absl::Seconds(1);
  absl::Duration heartbeatTimeout = absl::Seconds(3);
  /** ZMQ_RECONNECT_IVL / _MAX: how often the SUB socket retries a node that is down. */
  absl::Duration reconnectInterval = absl::Milliseconds(100);
  absl::Duration reconnectIntervalMax = absl::Seconds(1);
  // LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/bus.py:bus_defaults)
};

}  // namespace robot::ipc
