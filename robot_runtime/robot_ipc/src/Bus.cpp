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

#include "robot_ipc/Bus.h"

#include <sys/eventfd.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "zmq.h"    // NOLINT(build/include_subdir): libzmq installs its header in no directory
#include "zmq.hpp"  // NOLINT(build/include_subdir): cppzmq installs its header in no directory

#include "robot_ipc/NodeEndpoint.h"
#include "robot_runtime/robot_ipc/src/StatusMacros.h"

namespace robot::ipc {
namespace {

using Clock = std::chrono::steady_clock;

// The Python bus sets the same socket options.
// LINT.IfChange(socket_options)
// LINGER 0: closing a socket discards what it has not sent, so a process exits at once.
constexpr int kLingerMs = 0;
// TCP keepalive notices a peer whose machine vanished (cable pulled, power lost) at the transport level, next to the
// ZMTP heartbeats of BusOptions.
constexpr int kTcpKeepaliveOn = 1;
constexpr int kTcpKeepaliveIdleSeconds = 2;
constexpr int kTcpKeepaliveIntervalSeconds = 1;
constexpr int kTcpKeepaliveProbes = 3;
// ZMQ_HEARTBEAT_TTL is carried in deciseconds in a 16-bit field.
constexpr int kMaxHeartbeatTtlMs = 6553599;
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/bus.py:socket_options)
constexpr unsigned int kWakeCounterStart = 0;
// The IO thread logs a recurring problem at most this often.
constexpr int kLogPeriodSeconds = 5;
constexpr std::chrono::milliseconds kPollFailureBackoff(10);
// What one counted event adds to its counter.
constexpr uint64_t kOneEvent = 1;

int toMilliseconds(absl::Duration duration) {
  return static_cast<int>(std::clamp<int64_t>(absl::ToInt64Milliseconds(duration), 0, INT_MAX));
}

absl::string_view frameView(const zmq::message_t& frame) {
  return absl::string_view(static_cast<const char*>(frame.data()), frame.size());
}

// The counters of one topic. Written by the IO thread (and by publish() for sendDropped), read by any thread.
struct TopicCounters {
  std::atomic<uint64_t> sent{0};
  std::atomic<uint64_t> sendDropped{0};
  std::atomic<uint64_t> received{0};
  std::atomic<uint64_t> delivered{0};
  std::atomic<uint64_t> superseded{0};
  std::atomic<uint64_t> rejected{0};
  std::atomic<uint64_t> handlerErrors{0};

  TopicStatistics snapshot() const {
    TopicStatistics statistics;
    statistics.sent = sent.load(std::memory_order_relaxed);
    statistics.sendDropped = sendDropped.load(std::memory_order_relaxed);
    statistics.received = received.load(std::memory_order_relaxed);
    statistics.delivered = delivered.load(std::memory_order_relaxed);
    statistics.superseded = superseded.load(std::memory_order_relaxed);
    statistics.rejected = rejected.load(std::memory_order_relaxed);
    statistics.handlerErrors = handlerErrors.load(std::memory_order_relaxed);
    return statistics;
  }
};

void increment(std::atomic<uint64_t>& counter) {
  counter.fetch_add(kOneEvent, std::memory_order_relaxed);
}

// The three frames of one message, serialized on the publishing thread.
struct OutgoingMessage {
  zmq::message_t topic;
  zmq::message_t typeName;
  zmq::message_t payload;
  TopicCounters* absl_nullable counters = nullptr;
};

struct Subscription {
  std::string topic;
  Delivery delivery = Delivery::kLatest;
  // The full name of the subscribed message type; empty for subscribeRaw(), which takes any type.
  std::string expectedTypeName;
  // Reused for every message, so that its repeated fields keep their capacity.
  std::unique_ptr<google::protobuf::Message> message;
  Bus::MessageHandler messageHandler;
  Bus::RawHandler rawHandler;
  TopicCounters* absl_nullable counters = nullptr;
  // Delivery::kLatest: the newest message of the current drain, not yet parsed.
  bool hasLatest = false;
  zmq::message_t latestTypeName;
  zmq::message_t latestPayload;
};

struct PeriodicCallback {
  Clock::duration period = Clock::duration::zero();
  Clock::time_point nextDeadline;
  std::function<void()> callback;
};

absl::Status validateTopic(absl::string_view topic) {
  if (topic.empty()) {
    return absl::InvalidArgumentError("the topic is empty (ZeroMQ would match every topic with it)");
  }
  return absl::OkStatus();
}

absl::Status validateOptions(const BusOptions& options) {
  const absl::Status network = validateNetworkConfig(options.network);
  if (!network.ok()) {
    return absl::InvalidArgumentError(absl::StrCat("BusOptions.network: ", network.message()));
  }
  if (!options.nodeName.empty() && options.network.find(options.nodeName) == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat("BusOptions.nodeName: node '", options.nodeName, "' is not in the network (nodes: ",
                                                   absl::StrJoin(options.network.nodeNames(), ", "), ")"));
  }
  if (options.ioPollPeriod <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError("BusOptions.ioPollPeriod must be positive");
  }
  if (options.sendHighWaterMark < 0 || options.receiveHighWaterMark < 0) {
    return absl::InvalidArgumentError("BusOptions.sendHighWaterMark and receiveHighWaterMark must be >= 0 (0: no limit)");
  }
  if (options.publishQueueCapacity == 0) {
    return absl::InvalidArgumentError("BusOptions.publishQueueCapacity must be positive");
  }
  if (options.maxMessagesPerDrain == 0) {
    return absl::InvalidArgumentError("BusOptions.maxMessagesPerDrain must be positive");
  }
  if (options.heartbeatInterval < absl::ZeroDuration() || options.heartbeatTimeout < absl::ZeroDuration()) {
    return absl::InvalidArgumentError("BusOptions.heartbeatInterval and heartbeatTimeout must be >= 0 (0: no heartbeats)");
  }
  if (options.reconnectInterval <= absl::ZeroDuration() || options.reconnectIntervalMax < absl::ZeroDuration()) {
    return absl::InvalidArgumentError("BusOptions.reconnectInterval must be positive and reconnectIntervalMax >= 0");
  }
  return absl::OkStatus();
}

int portOf(absl::string_view endpoint) {
  const size_t colon = endpoint.rfind(':');
  int port = kEphemeralPort;
  if (colon == absl::string_view::npos || !absl::SimpleAtoi(endpoint.substr(colon + 1), &port)) {
    return kEphemeralPort;
  }
  return port;
}

// The endpoint this process's own SUB socket connects to: the bound one, over loopback if it is bound to every
// interface.
std::string selfConnectEndpoint(absl::string_view boundEndpoint) {
  constexpr absl::string_view kWildcardPrefix = "tcp://0.0.0.0:";
  if (absl::StartsWith(boundEndpoint, kWildcardPrefix)) {
    return absl::StrCat("tcp://127.0.0.1:", boundEndpoint.substr(kWildcardPrefix.size()));
  }
  return std::string(boundEndpoint);
}

}  // namespace

class Bus::Impl {
 public:
  explicit Impl(BusOptions options) : options_(std::move(options)) {}

  ~Impl() {
    if (isIoThread()) {
      LOG(FATAL) << "robot_ipc: a Bus was destroyed from its own IO thread (a handler or a periodic callback)";
    }
    stop();
    if (wakeFd_ >= 0) {
      ::close(wakeFd_);
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  absl::Status initialize();
  absl::Status addSubscription(std::unique_ptr<Subscription> subscription);
  absl::Status addPeriodicCallback(absl::Duration period, std::function<void()> callback);
  absl::Status connect(absl::string_view endpoint);
  absl::Status start();
  void stop();
  bool isRunning() const { return running_.load() && !stopRequested_.load(); }
  absl::Status publish(absl::string_view topic, const google::protobuf::Message& message);
  absl::Status publishFromIoThread(absl::string_view topic, const google::protobuf::Message& message);
  bool isIoThread() const { return ioThreadId_.load() == std::this_thread::get_id(); }

  const std::string& nodeName() const { return options_.nodeName; }
  const std::string& boundEndpoint() const { return boundEndpoint_; }
  int boundPort() const { return boundPort_; }
  std::vector<std::string> subscriberEndpoints() const;
  TopicStatistics topicStatistics(absl::string_view topic) const;
  absl::flat_hash_map<std::string, TopicStatistics> statistics() const;
  uint64_t periodicCallbackErrors() const { return periodicCallbackErrors_.load(); }

 private:
  void configureSocket(zmq::socket_t& socket) const;
  absl::Status checkCanPublish(absl::string_view topic) const;
  absl::StatusOr<OutgoingMessage> serialize(absl::string_view topic, const google::protobuf::Message& message);
  TopicCounters& countersFor(absl::string_view topic);
  void wake() const;
  void clearWake() const;

  // The IO thread.
  void runIoLoop();
  int pollTimeoutMs(Clock::time_point now) const;
  void sendQueued();
  void send(OutgoingMessage& message);
  void drainSubscriber();
  void dispatch(Subscription& subscription, const zmq::message_t& typeName, const zmq::message_t& payload);
  void runDueCallbacks(Clock::time_point now);

  const BusOptions options_;
  // Declared before the sockets, so that the sockets close before the context terminates.
  zmq::context_t context_;
  // No socket on a bus that only subscribes.
  zmq::socket_t publisher_;
  zmq::socket_t subscriber_;
  int wakeFd_ = -1;
  std::string boundEndpoint_;
  int boundPort_ = kEphemeralPort;

  // Serializes the configuration calls, start() and stop(). The IO thread never takes it.
  mutable absl::Mutex lifecycleMutex_;
  std::thread ioThread_ ABSL_GUARDED_BY(lifecycleMutex_);
  std::vector<std::string> subscriberEndpoints_ ABSL_GUARDED_BY(lifecycleMutex_);
  std::atomic<bool> running_{false};
  std::atomic<bool> stopRequested_{false};
  std::atomic<std::thread::id> ioThreadId_{};

  // Changed only while no IO thread runs (under lifecycleMutex_), and used by the IO thread only while it runs; the
  // thread's start and join order the two.
  absl::flat_hash_map<std::string, std::unique_ptr<Subscription>> subscriptions_;
  std::vector<Subscription* absl_nonnull> latestSubscriptions_;
  std::vector<PeriodicCallback> periodicCallbacks_;

  // publish() -> IO thread. The IO thread holds the mutex only to swap the vector with sending_.
  absl::Mutex queueMutex_;
  std::vector<OutgoingMessage> queue_ ABSL_GUARDED_BY(queueMutex_);
  std::vector<OutgoingMessage> sending_;

  mutable absl::Mutex countersMutex_;
  absl::flat_hash_map<std::string, std::unique_ptr<TopicCounters>> counters_ ABSL_GUARDED_BY(countersMutex_);
  std::atomic<uint64_t> periodicCallbackErrors_{0};
};

void Bus::Impl::configureSocket(zmq::socket_t& socket) const {
  socket.set(zmq::sockopt::linger, kLingerMs);
  socket.set(zmq::sockopt::heartbeat_ivl, toMilliseconds(options_.heartbeatInterval));
  socket.set(zmq::sockopt::heartbeat_timeout, toMilliseconds(options_.heartbeatTimeout));
  socket.set(zmq::sockopt::heartbeat_ttl, std::min(toMilliseconds(options_.heartbeatTimeout), kMaxHeartbeatTtlMs));
  socket.set(zmq::sockopt::tcp_keepalive, kTcpKeepaliveOn);
  socket.set(zmq::sockopt::tcp_keepalive_idle, kTcpKeepaliveIdleSeconds);
  socket.set(zmq::sockopt::tcp_keepalive_intvl, kTcpKeepaliveIntervalSeconds);
  socket.set(zmq::sockopt::tcp_keepalive_cnt, kTcpKeepaliveProbes);
  socket.set(zmq::sockopt::reconnect_ivl, toMilliseconds(options_.reconnectInterval));
  socket.set(zmq::sockopt::reconnect_ivl_max, toMilliseconds(options_.reconnectIntervalMax));
}

absl::Status Bus::Impl::initialize() {
  wakeFd_ = ::eventfd(kWakeCounterStart, EFD_NONBLOCK | EFD_CLOEXEC);
  if (wakeFd_ < 0) {
    return absl::InternalError(absl::StrCat("robot_ipc: eventfd() failed: ", std::strerror(errno)));
  }
  queue_.reserve(options_.publishQueueCapacity);
  sending_.reserve(options_.publishQueueCapacity);

  const NodeEndpoint* absl_nullable node = options_.nodeName.empty() ? nullptr : options_.network.find(options_.nodeName);
  try {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
    if (node != nullptr) {
      publisher_ = zmq::socket_t(context_, zmq::socket_type::pub);
      configureSocket(publisher_);
      publisher_.set(zmq::sockopt::sndhwm, options_.sendHighWaterMark);
      const std::string endpoint = node->bindEndpoint();
      try {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
        publisher_.bind(endpoint);
      } catch (const zmq::error_t& error) {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
        return absl::UnavailableError(absl::StrCat("robot_ipc: cannot bind node '", node->name, "' at ", endpoint, ": ", error.what()));
      }
      boundEndpoint_ = publisher_.get(zmq::sockopt::last_endpoint);
      boundPort_ = portOf(boundEndpoint_);
    }

    subscriber_ = zmq::socket_t(context_, zmq::socket_type::sub);
    configureSocket(subscriber_);
    subscriber_.set(zmq::sockopt::rcvhwm, options_.receiveHighWaterMark);
    // No pipe for a node that is not up: the subscriptions go out on the connection, when it completes.
    subscriber_.set(zmq::sockopt::immediate, /*val=*/true);

    absl::MutexLock lock(lifecycleMutex_);
    for (const NodeEndpoint& peer : options_.network.nodes) {
      std::string endpoint;
      if (node != nullptr && peer.name == node->name) {
        endpoint = selfConnectEndpoint(boundEndpoint_);
      } else if (peer.port == kEphemeralPort) {
        // Bound wherever its kernel chose; reachable through connect() only.
        continue;
      } else {
        endpoint = peer.connectEndpoint();
      }
      try {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
        subscriber_.connect(endpoint);
      } catch (const zmq::error_t& error) {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
        return absl::InvalidArgumentError(
            absl::StrCat("robot_ipc: cannot connect to node '", peer.name, "' at ", endpoint, ": ", error.what()));
      }
      subscriberEndpoints_.push_back(endpoint);
    }
  } catch (const zmq::error_t& error) {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
    return absl::InternalError(absl::StrCat("robot_ipc: cannot set up the ZeroMQ sockets: ", error.what()));
  }
  return absl::OkStatus();
}

absl::Status Bus::Impl::addSubscription(std::unique_ptr<Subscription> subscription) {
  ROBOT_IPC_RETURN_IF_ERROR(validateTopic(subscription->topic));
  if (isIoThread()) {
    return absl::FailedPreconditionError("robot_ipc: subscribe() from a handler; subscribe before start()");
  }
  absl::MutexLock lock(lifecycleMutex_);
  if (ioThread_.joinable()) {
    return absl::FailedPreconditionError(absl::StrCat("robot_ipc: cannot subscribe to '", subscription->topic,
                                                      "' while the bus runs; subscribe before start() or after stop()"));
  }
  if (subscriptions_.contains(subscription->topic)) {
    return absl::AlreadyExistsError(absl::StrCat("robot_ipc: the topic '", subscription->topic, "' already has a subscription"));
  }
  try {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
    subscriber_.set(zmq::sockopt::subscribe, subscription->topic);
  } catch (const zmq::error_t& error) {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
    return absl::InternalError(absl::StrCat("robot_ipc: cannot subscribe to '", subscription->topic, "': ", error.what()));
  }
  subscription->counters = &countersFor(subscription->topic);
  if (subscription->delivery == Delivery::kLatest) {
    latestSubscriptions_.push_back(subscription.get());
  }
  const std::string topic = subscription->topic;
  subscriptions_.emplace(topic, std::move(subscription));
  return absl::OkStatus();
}

absl::Status Bus::Impl::addPeriodicCallback(absl::Duration period, std::function<void()> callback) {
  if (period <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError("robot_ipc: a periodic callback needs a positive period");
  }
  if (callback == nullptr) {
    return absl::InvalidArgumentError("robot_ipc: the periodic callback is empty");
  }
  if (isIoThread()) {
    return absl::FailedPreconditionError("robot_ipc: addPeriodicCallback() from a handler; add it before start()");
  }
  absl::MutexLock lock(lifecycleMutex_);
  if (ioThread_.joinable()) {
    return absl::FailedPreconditionError(
        "robot_ipc: cannot add a periodic callback while the bus runs; add it before start() or after stop()");
  }
  PeriodicCallback periodic;
  periodic.period = std::chrono::duration_cast<Clock::duration>(absl::ToChronoNanoseconds(period));
  periodic.callback = std::move(callback);
  periodicCallbacks_.push_back(std::move(periodic));
  return absl::OkStatus();
}

absl::Status Bus::Impl::connect(absl::string_view endpoint) {
  if (isIoThread()) {
    return absl::FailedPreconditionError("robot_ipc: connect() from a handler; connect before start()");
  }
  absl::MutexLock lock(lifecycleMutex_);
  if (ioThread_.joinable()) {
    return absl::FailedPreconditionError("robot_ipc: cannot connect while the bus runs; connect before start() or after stop()");
  }
  const std::string address(endpoint);
  try {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
    subscriber_.connect(address);
  } catch (const zmq::error_t& error) {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
    return absl::InvalidArgumentError(absl::StrCat("robot_ipc: cannot connect to '", address, "': ", error.what()));
  }
  subscriberEndpoints_.push_back(address);
  return absl::OkStatus();
}

absl::Status Bus::Impl::start() {
  if (isIoThread()) {
    return absl::OkStatus();
  }
  absl::MutexLock lock(lifecycleMutex_);
  if (isRunning()) {
    return absl::OkStatus();
  }
  if (ioThread_.joinable()) {
    // Asked to stop from one of its own handlers and not joined yet.
    ioThread_.join();
  }
  stopRequested_.store(false);
  const Clock::time_point now = Clock::now();
  for (PeriodicCallback& periodic : periodicCallbacks_) {
    periodic.nextDeadline = now + periodic.period;
  }
  running_.store(true);
  try {  // NOLINT(exceptions): std::thread throws std::system_error; converted to a Status at once
    ioThread_ = std::thread([this] { runIoLoop(); });
  } catch (const std::system_error& error) {  // NOLINT(exceptions): std::thread throws std::system_error; converted to a Status at once
    running_.store(false);
    return absl::InternalError(absl::StrCat("robot_ipc: cannot start the IO thread: ", error.what()));
  }
  return absl::OkStatus();
}

void Bus::Impl::stop() {
  if (isIoThread()) {
    // Joining itself would deadlock: the loop ends after this call, and the next stop() or the destructor joins.
    stopRequested_.store(true);
    wake();
    return;
  }
  absl::MutexLock lock(lifecycleMutex_);
  if (ioThread_.joinable()) {
    stopRequested_.store(true);
    wake();
    ioThread_.join();
  }
  running_.store(false);
}

absl::Status Bus::Impl::checkCanPublish(absl::string_view topic) const {
  if (options_.nodeName.empty()) {
    return absl::FailedPreconditionError("robot_ipc: this bus has no node name, so it only subscribes and cannot publish");
  }
  return validateTopic(topic);
}

absl::StatusOr<OutgoingMessage> Bus::Impl::serialize(absl::string_view topic, const google::protobuf::Message& message) {
  const absl::string_view typeName = message.GetDescriptor()->full_name();
  const size_t size = message.ByteSizeLong();
  if (size > static_cast<size_t>(INT_MAX)) {
    return absl::InvalidArgumentError(
        absl::StrCat("robot_ipc: a ", typeName, " of ", size, " bytes is larger than protobuf can serialize"));
  }
  OutgoingMessage outgoing;
  outgoing.topic.rebuild(topic.data(), topic.size());
  outgoing.typeName.rebuild(typeName.data(), typeName.size());
  outgoing.payload.rebuild(size);
  if (!message.SerializeToArray(outgoing.payload.data(), static_cast<int>(size))) {
    return absl::InternalError(absl::StrCat("robot_ipc: cannot serialize a ", typeName, " for '", topic, "'"));
  }
  outgoing.counters = &countersFor(topic);
  return outgoing;
}

absl::Status Bus::Impl::publish(absl::string_view topic, const google::protobuf::Message& message) {
  ROBOT_IPC_RETURN_IF_ERROR(checkCanPublish(topic));
  ROBOT_IPC_ASSIGN_OR_RETURN(OutgoingMessage outgoing, serialize(topic, message));
  bool wasEmpty = false;
  {
    absl::MutexLock lock(queueMutex_);
    if (queue_.size() >= options_.publishQueueCapacity) {
      increment(outgoing.counters->sendDropped);
      return absl::ResourceExhaustedError(absl::StrCat("robot_ipc: the publish queue is full (", options_.publishQueueCapacity,
                                                       " messages); dropped one on '", topic, "'"));
    }
    wasEmpty = queue_.empty();
    queue_.push_back(std::move(outgoing));
  }
  // The IO thread empties the whole queue on one wake-up, so only the first message needs to wake it.
  if (wasEmpty) {
    wake();
  }
  return absl::OkStatus();
}

absl::Status Bus::Impl::publishFromIoThread(absl::string_view topic, const google::protobuf::Message& message) {
  if (!isIoThread()) {
    return absl::FailedPreconditionError("robot_ipc: publishFromIoThread() outside the bus's IO thread; call publish()");
  }
  ROBOT_IPC_RETURN_IF_ERROR(checkCanPublish(topic));
  ROBOT_IPC_ASSIGN_OR_RETURN(OutgoingMessage outgoing, serialize(topic, message));
  // What publish() queued before goes first, so that the messages of one thread keep their order.
  sendQueued();
  send(outgoing);
  return absl::OkStatus();
}

TopicCounters& Bus::Impl::countersFor(absl::string_view topic) {
  absl::MutexLock lock(countersMutex_);
  absl::flat_hash_map<std::string, std::unique_ptr<TopicCounters>>::iterator it = counters_.find(topic);
  if (it == counters_.end()) {
    it = counters_.emplace(std::string(topic), std::make_unique<TopicCounters>()).first;
  }
  return *it->second;
}

std::vector<std::string> Bus::Impl::subscriberEndpoints() const {
  absl::MutexLock lock(lifecycleMutex_);
  return subscriberEndpoints_;
}

TopicStatistics Bus::Impl::topicStatistics(absl::string_view topic) const {
  absl::MutexLock lock(countersMutex_);
  const absl::flat_hash_map<std::string, std::unique_ptr<TopicCounters>>::const_iterator it = counters_.find(topic);
  return it == counters_.end() ? TopicStatistics() : it->second->snapshot();
}

absl::flat_hash_map<std::string, TopicStatistics> Bus::Impl::statistics() const {
  absl::MutexLock lock(countersMutex_);
  absl::flat_hash_map<std::string, TopicStatistics> statistics;
  statistics.reserve(counters_.size());
  for (const std::pair<const std::string, std::unique_ptr<TopicCounters>>& entry : counters_) {
    statistics.emplace(entry.first, entry.second->snapshot());
  }
  return statistics;
}

void Bus::Impl::wake() const {
  const uint64_t one = 1;
  // Fails only when the counter would overflow, and then the IO thread is due to wake anyway.
  const ssize_t written = ::write(wakeFd_, &one, sizeof(one));
  static_cast<void>(written);
}

void Bus::Impl::clearWake() const {
  uint64_t count = 0;
  const ssize_t consumed = ::read(wakeFd_, &count, sizeof(count));
  static_cast<void>(consumed);
}

void Bus::Impl::runIoLoop() {
  ioThreadId_.store(std::this_thread::get_id());
  std::array<zmq_pollitem_t, 2> items = {
      zmq_pollitem_t{subscriber_.handle(), 0, ZMQ_POLLIN, 0},
      zmq_pollitem_t{nullptr, wakeFd_, ZMQ_POLLIN, 0},
  };
  while (!stopRequested_.load(std::memory_order_acquire)) {
    try {  // NOLINT(exceptions): the IO thread's outermost loop: what escapes cppzmq or a handler is logged
      items[0].revents = 0;
      items[1].revents = 0;
      const int ready = zmq_poll(items.data(), static_cast<int>(items.size()), pollTimeoutMs(Clock::now()));
      if (ready < 0) {
        const int error = zmq_errno();
        if (error == ETERM) {
          break;
        }
        if (error != EINTR) {
          LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: zmq_poll() failed: " << zmq_strerror(error);
          std::this_thread::sleep_for(kPollFailureBackoff);
        }
        continue;
      }
      if ((items[1].revents & ZMQ_POLLIN) != 0) {
        clearWake();
      }
      sendQueued();
      if ((items[0].revents & ZMQ_POLLIN) != 0) {
        drainSubscriber();
      }
      runDueCallbacks(Clock::now());
      // NOLINTNEXTLINE(exceptions): the IO thread's outermost loop: what escapes cppzmq or a handler is logged
    } catch (const std::exception& exception) {
      LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: the IO thread caught an exception: " << exception.what();
    } catch (...) {  // NOLINT(exceptions): the IO thread's outermost loop: what escapes cppzmq or a handler is logged
      LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: the IO thread caught an exception of unknown type";
    }
  }
  try {  // NOLINT(exceptions): cppzmq throws zmq::error_t; the last sends at stop() are logged
    sendQueued();
  } catch (const std::exception& exception) {  // NOLINT(exceptions): cppzmq throws zmq::error_t; the last sends at stop() are logged
    LOG(ERROR) << "robot_ipc: sending the queued messages at stop() failed: " << exception.what();
  }
  ioThreadId_.store(std::thread::id());
}

int Bus::Impl::pollTimeoutMs(Clock::time_point now) const {
  Clock::duration wait = std::chrono::duration_cast<Clock::duration>(absl::ToChronoNanoseconds(options_.ioPollPeriod));
  for (const PeriodicCallback& periodic : periodicCallbacks_) {
    wait = std::min(wait, periodic.nextDeadline - now);
  }
  if (wait <= Clock::duration::zero()) {
    return 0;
  }
  // Rounded up, so that the thread does not wake just before a deadline and spin until it.
  return static_cast<int>(std::min<int64_t>(std::chrono::ceil<std::chrono::milliseconds>(wait).count(), INT_MAX));
}

void Bus::Impl::sendQueued() {
  {
    absl::MutexLock lock(queueMutex_);
    if (queue_.empty()) {
      return;
    }
    // Both vectors keep their capacity, so the swap allocates nothing.
    queue_.swap(sending_);
  }
  for (OutgoingMessage& message : sending_) {
    send(message);
  }
  sending_.clear();
}

void Bus::Impl::send(OutgoingMessage& message) {
  try {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
    // A PUB socket never blocks: at a subscriber's high-water mark ZeroMQ drops the message for that subscriber.
    const bool sent = publisher_.send(message.topic, zmq::send_flags::sndmore | zmq::send_flags::dontwait).has_value() &&
                      publisher_.send(message.typeName, zmq::send_flags::sndmore | zmq::send_flags::dontwait).has_value() &&
                      publisher_.send(message.payload, zmq::send_flags::dontwait).has_value();
    increment(sent ? message.counters->sent : message.counters->sendDropped);
  } catch (const zmq::error_t& error) {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
    increment(message.counters->sendDropped);
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: sending on '" << frameView(message.topic) << "' failed: " << error.what();
  }
}

void Bus::Impl::drainSubscriber() {
  for (size_t count = 0; count < options_.maxMessagesPerDrain; ++count) {
    zmq::message_t topic;
    if (!subscriber_.recv(topic, zmq::recv_flags::dontwait).has_value()) {
      break;
    }
    // A message arrives whole, so its other frames are there already.
    zmq::message_t typeName;
    zmq::message_t payload;
    size_t frames = 1;
    bool more = topic.more();
    if (more && subscriber_.recv(typeName, zmq::recv_flags::dontwait).has_value()) {
      ++frames;
      more = typeName.more();
    }
    if (more && subscriber_.recv(payload, zmq::recv_flags::dontwait).has_value()) {
      ++frames;
      more = payload.more();
    }
    while (more) {
      zmq::message_t extra;
      if (!subscriber_.recv(extra, zmq::recv_flags::dontwait).has_value()) {
        break;
      }
      ++frames;
      more = extra.more();
    }

    // ZeroMQ matched a prefix; the topic must match whole.
    const absl::flat_hash_map<std::string, std::unique_ptr<Subscription>>::iterator it = subscriptions_.find(frameView(topic));
    if (it == subscriptions_.end()) {
      continue;
    }
    Subscription& subscription = *it->second;
    increment(subscription.counters->received);
    if (frames != 3) {
      increment(subscription.counters->rejected);
      LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds) << "robot_ipc: a message on '" << subscription.topic << "' has " << frames
                                                  << " frames instead of three (topic, type name, payload)";
      continue;
    }
    if (subscription.delivery == Delivery::kAll) {
      dispatch(subscription, typeName, payload);
      continue;
    }
    if (subscription.hasLatest) {
      increment(subscription.counters->superseded);
    }
    subscription.latestTypeName = std::move(typeName);
    subscription.latestPayload = std::move(payload);
    subscription.hasLatest = true;
  }

  for (Subscription* absl_nonnull subscription : latestSubscriptions_) {
    if (subscription->hasLatest) {
      subscription->hasLatest = false;
      dispatch(*subscription, subscription->latestTypeName, subscription->latestPayload);
    }
  }
}

void Bus::Impl::dispatch(Subscription& subscription, const zmq::message_t& typeName, const zmq::message_t& payload) {
  TopicCounters& counters = *subscription.counters;
  const absl::string_view type = frameView(typeName);
  if (!subscription.expectedTypeName.empty() && type != subscription.expectedTypeName) {
    increment(counters.rejected);
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds) << "robot_ipc: '" << subscription.topic << "' carries a " << type
                                                << ", but its subscription takes a " << subscription.expectedTypeName;
    return;
  }
  if (subscription.message != nullptr && (payload.size() > static_cast<size_t>(INT_MAX) ||
                                          !subscription.message->ParseFromArray(payload.data(), static_cast<int>(payload.size())))) {
    increment(counters.rejected);
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds) << "robot_ipc: a message on '" << subscription.topic << "' does not parse as a " << type;
    return;
  }
  increment(counters.delivered);
  try {  // NOLINT(exceptions): a handler's exception is counted and logged, not let end the IO thread
    if (subscription.message != nullptr) {
      subscription.messageHandler(*subscription.message);
    } else {
      subscription.rawHandler(type, frameView(payload));
    }
  } catch (const std::exception& exception) {  // NOLINT(exceptions): a handler's exception is counted and logged, not let end the IO thread
    increment(counters.handlerErrors);
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: the handler of '" << subscription.topic << "' threw: " << exception.what();
  } catch (...) {  // NOLINT(exceptions): a handler's exception is counted and logged, not let end the IO thread
    increment(counters.handlerErrors);
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: the handler of '" << subscription.topic
                                              << "' threw an exception of unknown type";
  }
}

void Bus::Impl::runDueCallbacks(Clock::time_point now) {
  for (PeriodicCallback& periodic : periodicCallbacks_) {
    if (now < periodic.nextDeadline) {
      continue;
    }
    try {  // NOLINT(exceptions): a periodic callback's exception is counted and logged, not let end the IO thread
      periodic.callback();
      // NOLINTNEXTLINE(exceptions): a periodic callback's exception is counted and logged, not let end the IO thread
    } catch (const std::exception& exception) {
      periodicCallbackErrors_.fetch_add(kOneEvent);
      LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: a periodic callback threw: " << exception.what();
    } catch (...) {  // NOLINT(exceptions): a periodic callback's exception is counted and logged, not let end the IO thread
      periodicCallbackErrors_.fetch_add(kOneEvent);
      LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: a periodic callback threw an exception of unknown type";
    }
    periodic.nextDeadline += periodic.period;
    if (periodic.nextDeadline <= now) {
      // A whole period late: skip the missed calls instead of bursting through them.
      periodic.nextDeadline = now + periodic.period;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Bus
// ---------------------------------------------------------------------------------------------------------------------

absl::StatusOr<std::unique_ptr<Bus>> Bus::Create(BusOptions options) {
  ROBOT_IPC_RETURN_IF_ERROR(validateOptions(options));
  std::unique_ptr<Impl> impl;
  try {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
    impl = std::make_unique<Impl>(std::move(options));
  } catch (const zmq::error_t& error) {  // NOLINT(exceptions): cppzmq throws zmq::error_t; converted at once
    return absl::InternalError(absl::StrCat("robot_ipc: cannot create a ZeroMQ context: ", error.what()));
  }
  ROBOT_IPC_RETURN_IF_ERROR(impl->initialize());
  return absl::WrapUnique(new Bus(std::move(impl)));
}

Bus::Bus(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Bus::~Bus() = default;

absl::Status Bus::subscribeMessage(absl::string_view topic,
                                   Delivery delivery,
                                   std::unique_ptr<google::protobuf::Message> message,
                                   MessageHandler handler) {
  std::unique_ptr<Subscription> subscription = std::make_unique<Subscription>();
  subscription->topic = std::string(topic);
  subscription->delivery = delivery;
  subscription->expectedTypeName = std::string(message->GetDescriptor()->full_name());
  subscription->message = std::move(message);
  subscription->messageHandler = std::move(handler);
  return impl_->addSubscription(std::move(subscription));
}

absl::Status Bus::subscribeRaw(absl::string_view topic, Delivery delivery, RawHandler handler) {
  if (handler == nullptr) {
    return absl::InvalidArgumentError("subscribeRaw(): the handler is empty");
  }
  std::unique_ptr<Subscription> subscription = std::make_unique<Subscription>();
  subscription->topic = std::string(topic);
  subscription->delivery = delivery;
  subscription->rawHandler = std::move(handler);
  return impl_->addSubscription(std::move(subscription));
}

absl::Status Bus::addPeriodicCallback(absl::Duration period, std::function<void()> callback) {
  return impl_->addPeriodicCallback(period, std::move(callback));
}

absl::Status Bus::connect(absl::string_view endpoint) {
  return impl_->connect(endpoint);
}

absl::Status Bus::start() {
  return impl_->start();
}

void Bus::stop() {
  impl_->stop();
}

bool Bus::isRunning() const {
  return impl_->isRunning();
}

absl::Status Bus::publish(absl::string_view topic, const google::protobuf::Message& message) {
  return impl_->publish(topic, message);
}

absl::Status Bus::publishFromIoThread(absl::string_view topic, const google::protobuf::Message& message) {
  return impl_->publishFromIoThread(topic, message);
}

bool Bus::isIoThread() const {
  return impl_->isIoThread();
}

const std::string& Bus::nodeName() const {
  return impl_->nodeName();
}

const std::string& Bus::boundEndpoint() const {
  return impl_->boundEndpoint();
}

int Bus::boundPort() const {
  return impl_->boundPort();
}

std::vector<std::string> Bus::subscriberEndpoints() const {
  return impl_->subscriberEndpoints();
}

TopicStatistics Bus::topicStatistics(absl::string_view topic) const {
  return impl_->topicStatistics(topic);
}

absl::flat_hash_map<std::string, TopicStatistics> Bus::statistics() const {
  return impl_->statistics();
}

uint64_t Bus::periodicCallbackErrors() const {
  return impl_->periodicCallbackErrors();
}

}  // namespace robot::ipc
