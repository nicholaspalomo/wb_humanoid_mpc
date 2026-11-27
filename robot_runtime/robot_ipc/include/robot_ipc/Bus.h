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
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "google/protobuf/message.h"

#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_ipc/TopicStatistics.h"

namespace robot::ipc {

/**
 * The bus of legged_nmpc/docs/distributed_runtime/README.md: publish/subscribe
 * over Eclipse Zenoh, peer to peer without a router and without ROS. Every bus
 * topic is a Zenoh key expression (KeyExprs.h) whose samples are one envelope
 * (WireProtocol.h): the wire protocol's version, the writer's id and sequence
 * number, the source timestamp, the full protobuf type name and the serialized
 * message. A sample of another protocol version is refused and counted as
 * rejected.
 *
 * The buses of a process share one Zenoh peer session of
 * NetworkConfig::domainId, configured from the network file (SessionConfig.h):
 * a TCP listener on the node's host, connections to the host of every node,
 * the quality of service of each topic by its named profile (QosProfile.h). Any
 * process may publish any topic and processes start in any order; every bus
 * declares a liveliness token per publisher and subscriber, and
 * TopicStatistics::matchedReaders / matchedWriters count those that are up.
 *
 * Order. One drain takes up to BusOptions::maxMessagesPerDrain samples across
 * the topics and dispatches them in the order they were written (their source
 * timestamps, which every bus makes strictly increasing across its topics):
 * kAll's every sample, kLatest's newest one at the place it was written (the
 * older ones of the drain are superseded). Within a drain, a handler never sees
 * a sample of one topic before one of another topic that the same publisher
 * wrote earlier. Across drains Zenoh gives no such guarantee: the profiles have
 * different priorities, and a batched sample may arrive after an express one
 * written later.
 *
 * Threads. The Zenoh publishers belong to the bus's IO thread, which start()
 * launches; Zenoh's own threads copy each arriving sample into its
 * subscription's queue (BusOptions' delivery history depths) and wake it.
 * Handlers and periodic callbacks run on it, one at a time, so they must return
 * quickly and need no locking among themselves. publish() may be called from
 * any thread except a realtime one: it serializes on the caller and passes the
 * sample to the IO thread through a short mutex-protected queue; only the IO
 * thread calls z_publisher_put(). A realtime thread never calls the bus; it
 * exchanges data with a periodic callback through lock-free mailboxes instead.
 *
 * Errors are absl::Status values. Nothing throws out of the IO thread: a
 * handler or callback that throws is logged and counted, and the bus carries
 * on.
 *
 * Lifecycle. subscribe(), subscribeRaw() and addPeriodicCallback() configure a
 * bus that is not running (before start(), or after stop()); start() and stop()
 * are idempotent, and the destructor stops.
 */
class Bus {
 public:
  // The bus keeps each handler and calls it on the IO thread only.
  using RawHandler = absl::AnyInvocable<void(absl::string_view typeName,
                                             absl::string_view payload)>;
  using MessageHandler =
      absl::AnyInvocable<void(const google::protobuf::Message& message)>;

  /**
   * Opens (or shares) the Zenoh session of the domain and declares the
   * publishers of the topics the network file lists this node as a publisher
   * of; the IO thread starts with start(). A node name that is not in the
   * network or an invalid network is InvalidArgument; a node whose host is an
   * address of another machine, or a domain whose every port is taken, is
   * Unavailable; a process whose buses ask for two different configurations of
   * one domain is FailedPrecondition; a Zenoh failure is Internal.
   */
  static absl::StatusOr<std::unique_ptr<Bus>> Create(BusOptions options);

  ~Bus();
  Bus(const Bus&) = delete;
  Bus& operator=(const Bus&) = delete;

  /**
   * Passes every message of `topic` that has the type Msg to `handler`, on
   * the IO thread. A message of another type, or one that does not parse, is
   * rejected and counted. The reference is valid during the call only, and the
   * message object is reused for the next one, so copy what you keep. One
   * subscription per topic.
   */
  template <typename Msg>
  absl::Status subscribe(absl::string_view topic,
                         Delivery delivery,
                         absl::AnyInvocable<void(const Msg&)> handler) {
    static_assert(std::is_base_of_v<google::protobuf::Message, Msg>,
                  "subscribe<Msg>() needs a generated protobuf message");
    if (handler == nullptr) {
      return absl::InvalidArgumentError("subscribe(): the handler is empty");
    }
    return subscribeMessage(
        topic, delivery, std::make_unique<Msg>(),
        [handler = std::move(handler)](
            const google::protobuf::Message& message) mutable {
          handler(static_cast<const Msg&>(message));
        });
  }

  /**
   * Passes every message of exactly `topic` to `handler` as its type name and
   * serialized bytes, whatever its type, on the IO thread. The views are valid
   * during the call only.
   */
  absl::Status subscribeRaw(absl::string_view topic,
                            Delivery delivery,
                            RawHandler handler);

  /**
   * Calls `callback` on the IO thread every `period`, on absolute deadlines
   * from start(). A deadline that has passed when the call before it returns (a
   * callback that overran its period, or a late IO thread) is skipped, to the
   * next deadline still ahead, rather than called at once or in a burst.
   */
  absl::Status addPeriodicCallback(absl::Duration period,
                                   absl::AnyInvocable<void()> callback);

  /** Launches the IO thread. Idempotent. */
  absl::Status start();

  /**
   * Stops and joins the IO thread, sending what is still queued first.
   * Idempotent. Called from a handler or callback, it only asks the IO thread
   * to stop after the current call; a later stop() or the destructor joins it.
   */
  void stop();

  /** True between start() and stop(). */
  bool isRunning() const;

  /**
   * Publishes `message` on `topic` from any non-realtime thread. Serializes on
   * the caller and queues the sample for the IO thread, which writes it
   * promptly; never waits for the network. A message published while the bus is
   * not running stays queued until start(). A full queue refuses with
   * ResourceExhausted (counted as sendDropped); a bus without a node is
   * FailedPrecondition.
   */
  absl::Status publish(absl::string_view topic,
                       const google::protobuf::Message& message);

  /**
   * publish() of a message serialized already, of the full protobuf type name
   * `typeName` (a relay, or a tool that forwards what subscribeRaw() passed
   * it). The bus does not check that `payload` parses as that type.
   */
  absl::Status publishRaw(absl::string_view topic,
                          absl::string_view typeName,
                          absl::string_view payload);

  /**
   * Publishes from a handler or a periodic callback, straight to the Zenoh
   * publisher (after what publish() queued). Called from any other thread it is
   * FailedPrecondition.
   */
  absl::Status publishFromIoThread(absl::string_view topic,
                                   const google::protobuf::Message& message);

  /** True on the bus's IO thread, i.e. inside a handler or a periodic callback.
   */
  bool isIoThread() const;

  /** BusOptions::nodeName; empty for a bus that only subscribes. */
  const std::string& nodeName() const;
  /** NetworkConfig::domainId. */
  int domainId() const;
  /** The hosts the bus connects to: the distinct hosts of the network's nodes.
   */
  const std::vector<std::string>& discoveryPeers() const;
  /** The Zenoh configuration the bus's session was opened with
   * (SessionConfig.h). */
  const std::string& domainConfig() const;

  /** The counters of one topic (all zero for a topic the bus has not seen).
   * Thread-safe. */
  TopicStatistics topicStatistics(absl::string_view topic) const;
  /** The counters of every topic the bus has published or subscribed.
   * Thread-safe. */
  absl::flat_hash_map<std::string, TopicStatistics> statistics() const;
  /** Periodic callback calls that threw. Thread-safe. */
  uint64_t periodicCallbackErrors() const;

 private:
  class Impl;

  explicit Bus(std::unique_ptr<Impl> impl);

  absl::Status subscribeMessage(
      absl::string_view topic,
      Delivery delivery,
      std::unique_ptr<google::protobuf::Message> message,
      MessageHandler handler);

  std::unique_ptr<Impl> impl_;
};

}  // namespace robot::ipc
