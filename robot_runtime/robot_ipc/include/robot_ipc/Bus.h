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
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "google/protobuf/message.h"

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_ipc/TopicStatistics.h"

namespace robot::ipc {

/**
 * The ZeroMQ bus of humanoid_nmpc/docs/distributed_runtime/README.md: a publish/subscribe network without a broker.
 *
 * A bus binds one PUB socket at the endpoint of its node (unless it only subscribes) and connects one SUB socket to
 * every endpoint of the network, its own included, so any process may publish any topic and processes start in any
 * order. Every message is three frames: the topic, the full protobuf type name and the serialized message.
 *
 * Threads. Both sockets belong to the bus's IO thread, which start() launches. Handlers and periodic callbacks run on
 * it, one at a time, so they must return quickly and need no locking among themselves. publish() may be called from
 * any thread except a realtime one: it serializes on the caller and hands the frames to the IO thread through a short
 * mutex-protected queue. A realtime thread never calls the bus; it exchanges data with a periodic callback through
 * lock-free mailboxes instead.
 *
 * Errors are absl::Status values. Nothing throws out of the IO thread: a handler or callback that throws is logged
 * and counted, and the bus carries on.
 *
 * Lifecycle. subscribe(), subscribeRaw(), addPeriodicCallback() and connect() configure a bus that is not running
 * (before start(), or after stop()); start() and stop() are idempotent, and the destructor stops.
 */
class Bus {
 public:
  using RawHandler = std::function<void(absl::string_view typeName, absl::string_view payload)>;
  using MessageHandler = std::function<void(const google::protobuf::Message& message)>;

  /**
   * Binds and connects the sockets; the IO thread starts with start(). A node name that is not in the network or an
   * invalid network is InvalidArgument; an endpoint that cannot be bound (taken, or an address of another machine) is
   * Unavailable.
   */
  static absl::StatusOr<std::unique_ptr<Bus>> Create(BusOptions options);

  ~Bus();
  Bus(const Bus&) = delete;
  Bus& operator=(const Bus&) = delete;

  /**
   * Hands every message of exactly `topic` (ZeroMQ filters by prefix; the bus also checks the whole topic) that
   * carries the type Msg to `handler`, on the IO thread. A message of another type, or one that does not parse, is
   * rejected and counted. The reference is valid during the call only, and the message object is reused for the next
   * one, so copy what you keep. One subscription per topic.
   */
  template <typename Msg>
  absl::Status subscribe(absl::string_view topic, Delivery delivery, std::function<void(const Msg&)> handler) {
    static_assert(std::is_base_of_v<google::protobuf::Message, Msg>, "subscribe<Msg>() needs a generated protobuf message");
    if (handler == nullptr) {
      return absl::InvalidArgumentError("subscribe(): the handler is empty");
    }
    return subscribeMessage(
        topic, delivery, std::make_unique<Msg>(),
        [handler = std::move(handler)](const google::protobuf::Message& message) { handler(static_cast<const Msg&>(message)); });
  }

  /**
   * Hands every message of exactly `topic` to `handler` as its type name and serialized bytes, whatever its type, on
   * the IO thread. The views are valid during the call only.
   */
  absl::Status subscribeRaw(absl::string_view topic, Delivery delivery, RawHandler handler);

  /**
   * Calls `callback` on the IO thread every `period`, on absolute deadlines from start(). A callback that is late by a
   * whole period skips the missed calls rather than bursting.
   */
  absl::Status addPeriodicCallback(absl::Duration period, std::function<void()> callback);

  /**
   * Connects the SUB socket to one more endpoint ("tcp://host:port"), such as a node bound to kEphemeralPort or a
   * publisher outside the network file.
   */
  absl::Status connect(absl::string_view endpoint);

  /** Launches the IO thread. Idempotent. */
  absl::Status start();

  /**
   * Stops and joins the IO thread, sending what is still queued first. Idempotent. Called from a handler or callback,
   * it only asks the IO thread to stop after the current call; a later stop() or the destructor joins it.
   */
  void stop();

  /** True between start() and stop(). */
  bool isRunning() const;

  /**
   * Publishes `message` on `topic` from any non-realtime thread. Serializes on the caller and queues the frames for
   * the IO thread, which sends them promptly; never waits for the network. A message published while the bus is not
   * running stays queued until start(). A full queue refuses with ResourceExhausted (counted as sendDropped); a bus
   * without a node is FailedPrecondition.
   */
  absl::Status publish(absl::string_view topic, const google::protobuf::Message& message);

  /**
   * Publishes from a handler or a periodic callback, straight onto the socket. Called from any other thread it is
   * FailedPrecondition.
   */
  absl::Status publishFromIoThread(absl::string_view topic, const google::protobuf::Message& message);

  /** True on the bus's IO thread, i.e. inside a handler or a periodic callback. */
  bool isIoThread() const;

  /** BusOptions::nodeName; empty for a bus that only subscribes. */
  const std::string& nodeName() const;
  /** The endpoint the PUB socket is bound to, e.g. "tcp://127.0.0.1:5600"; empty for a bus that only subscribes. */
  const std::string& boundEndpoint() const;
  /** The port of boundEndpoint() (the kernel's choice for kEphemeralPort), or kEphemeralPort when there is none. */
  int boundPort() const;
  /** The endpoints the SUB socket is connected to, in the order they were connected. */
  std::vector<std::string> subscriberEndpoints() const;

  /** The counters of one topic (all zero for a topic the bus has not seen). Thread-safe. */
  TopicStatistics topicStatistics(absl::string_view topic) const;
  /** The counters of every topic the bus has published or subscribed. Thread-safe. */
  absl::flat_hash_map<std::string, TopicStatistics> statistics() const;
  /** Periodic callback calls that threw. Thread-safe. */
  uint64_t periodicCallbackErrors() const;

 private:
  class Impl;

  explicit Bus(std::unique_ptr<Impl> impl);

  absl::Status subscribeMessage(absl::string_view topic,
                                Delivery delivery,
                                std::unique_ptr<google::protobuf::Message> message,
                                MessageHandler handler);

  std::unique_ptr<Impl> impl_;
};

}  // namespace robot::ipc
