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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/protobuf/message.h"
#include "zenoh.h"  // NOLINT(build/include_subdir): zenoh-c installs its headers in no directory

#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_ipc/KeyExprs.h"
#include "robot_ipc/NetworkConfig.h"
#include "robot_ipc/QosProfile.h"
#include "robot_ipc/TopicStatistics.h"
#include "robot_ipc/WireProtocol.h"
#include "robot_runtime/robot_ipc/src/SessionRegistry.h"
#include "robot_runtime/robot_ipc/src/StatusMacros.h"

namespace robot::ipc {
namespace {

using Clock = std::chrono::steady_clock;

// The IO thread logs a recurring problem at most this often.
constexpr int kLogPeriodSeconds = 5;
// What one counted event adds to its counter.
constexpr uint64_t kOneEvent = 1;
// The samples a drain may hold before it dispatches, before the vector that
// holds them grows (it keeps its capacity).
constexpr size_t kInitialPendingCapacity = 256;

// The counters of one topic. Written by the IO thread (and by publish() for
// sendDropped), read by any thread.
struct TopicCounters {
  std::atomic<uint64_t> written = 0;
  std::atomic<uint64_t> sendDropped = 0;
  std::atomic<uint64_t> received = 0;
  std::atomic<uint64_t> delivered = 0;
  std::atomic<uint64_t> superseded = 0;
  std::atomic<uint64_t> rejected = 0;
  std::atomic<uint64_t> handlerErrors = 0;
  std::atomic<uint64_t> missed = 0;
  // The liveliness tokens of the topic, as the IO thread last counted them.
  std::atomic<uint64_t> incompatibleQos = 0;
  std::atomic<uint64_t> matchedReaders = 0;
  std::atomic<uint64_t> matchedWriters = 0;

  TopicStatistics snapshot() const {
    TopicStatistics statistics;
    statistics.written = written.load(std::memory_order_relaxed);
    statistics.sendDropped = sendDropped.load(std::memory_order_relaxed);
    statistics.received = received.load(std::memory_order_relaxed);
    statistics.delivered = delivered.load(std::memory_order_relaxed);
    statistics.superseded = superseded.load(std::memory_order_relaxed);
    statistics.rejected = rejected.load(std::memory_order_relaxed);
    statistics.handlerErrors = handlerErrors.load(std::memory_order_relaxed);
    statistics.missed = missed.load(std::memory_order_relaxed);
    statistics.incompatibleQos =
        incompatibleQos.load(std::memory_order_relaxed);
    statistics.matchedReaders = matchedReaders.load(std::memory_order_relaxed);
    statistics.matchedWriters = matchedWriters.load(std::memory_order_relaxed);
    return statistics;
  }
};

void increment(std::atomic<uint64_t>& counter, uint64_t count = kOneEvent) {
  counter.fetch_add(count, std::memory_order_relaxed);
}

/**
 * Wakes the IO thread: publish(), stop() and the Zenoh threads that deliver a
 * sample or a liveliness change call wake(), and the IO thread waits for it.
 * Thread-safe.
 */
class Waker {
 public:
  Waker() = default;
  Waker(const Waker&) = delete;
  Waker& operator=(const Waker&) = delete;

  void wake() {
    absl::MutexLock lock(mutex_);
    woken_ = true;
  }

  /** Waits until wake() was called (since the last wait) or for `timeout`. */
  void wait(absl::Duration timeout) {
    absl::MutexLock lock(mutex_);
    mutex_.AwaitWithTimeout(absl::Condition(&woken_), timeout);
    woken_ = false;
  }

 private:
  absl::Mutex mutex_;
  bool woken_ ABSL_GUARDED_BY(mutex_) = false;
};

// One message, serialized on the publishing thread.
struct OutgoingMessage {
  std::string topic;
  std::string typeName;
  std::string payload;
  TopicCounters* absl_nullable counters = nullptr;
};

// The Zenoh publisher of one topic and its liveliness token. The IO thread's
// own, once it runs.
struct TopicWriter {
  z_owned_publisher_t publisher;
  z_owned_liveliness_token_t token;
  QosProfile profile = QosProfile::kReliableEvents;
  uint64_t writerId = 0;
  // The sequence number of the last sample written; the next one is one more.
  uint64_t lastSequence = 0;
  TopicCounters* absl_nullable counters = nullptr;
};

struct Subscription {
  std::string topic;
  Delivery delivery = Delivery::kLatest;
  QosProfile profile = QosProfile::kReliableEvents;
  // The full name of the subscribed message type; empty for subscribeRaw(),
  // which takes any type.
  std::string expectedTypeName;
  // Reused for every message, so that its repeated fields keep their capacity.
  std::unique_ptr<google::protobuf::Message> message;
  Bus::MessageHandler messageHandler;
  Bus::RawHandler rawHandler;
  TopicCounters* absl_nullable counters = nullptr;
  z_owned_subscriber_t subscriber;
  z_owned_liveliness_token_t token;
  // True once the subscriber and the token are declared; the bus drops them.
  bool declared = false;

  // Zenoh's threads -> the IO thread: the envelopes that arrived, the newest
  // `depth` of them (an older one is replaced, and counted as missed from the
  // gap in its writer's sequence numbers).
  Waker* absl_nullable waker = nullptr;
  size_t depth = 1;
  absl::Mutex inboxMutex;
  std::deque<std::string> inbox ABSL_GUARDED_BY(inboxMutex);

  // The envelopes of the current drain (IO thread; keeps its capacity).
  std::vector<std::string> taken;
  // The last sequence number of each writer this subscription took a sample
  // of.
  absl::flat_hash_map<uint64_t, uint64_t> lastSequenceByWriter;
  // Delivery::kLatest: the index, in the drain's pending samples, of the newest
  // sample of this subscription so far.
  std::optional<size_t> latestPending;
};

// A sample a drain has taken and will dispatch once it has taken the rest, in
// the order the samples were written.
struct PendingMessage {
  Subscription* absl_nonnull subscription;
  EnvelopeView envelope;
  // Delivery::kLatest: a newer sample of the same topic is in the same drain;
  // this one is not dispatched.
  bool superseded = false;
};

// A liveliness token that appeared (alive) or went away.
struct TokenEvent {
  std::string keyExpr;
  bool alive = false;
};

// Zenoh's threads -> the IO thread: the liveliness changes of the domain.
struct TokenInbox {
  Waker* absl_nullable waker = nullptr;
  absl::Mutex mutex;
  std::vector<TokenEvent> events ABSL_GUARDED_BY(mutex);
};

struct PeriodicCallback {
  Clock::duration period = Clock::duration::zero();
  Clock::time_point nextDeadline;
  absl::AnyInvocable<void()> callback;
};

absl::Status validateOptions(const BusOptions& options) {
  const absl::Status network = validateNetworkConfig(options.network);
  if (!network.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("BusOptions.network: ", network.message()));
  }
  if (!options.nodeName.empty() &&
      options.network.find(options.nodeName) == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("BusOptions.nodeName: node '", options.nodeName,
                     "' is not in the network (nodes: ",
                     absl::StrJoin(options.network.nodeNames(), ", "), ")"));
  }
  if (options.ioPollPeriod <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError(
        "BusOptions.ioPollPeriod must be positive");
  }
  if (options.publishQueueCapacity == 0) {
    return absl::InvalidArgumentError(
        "BusOptions.publishQueueCapacity must be positive");
  }
  if (options.maxMessagesPerDrain == 0) {
    return absl::InvalidArgumentError(
        "BusOptions.maxMessagesPerDrain must be positive");
  }
  if (options.latestDeliveryHistoryDepth <= 0 ||
      options.allDeliveryHistoryDepth <= 0) {
    return absl::InvalidArgumentError(
        "BusOptions.latestDeliveryHistoryDepth and allDeliveryHistoryDepth "
        "must be positive");
  }
  return absl::OkStatus();
}

z_congestion_control_t zenohCongestionControl(CongestionControl control) {
  switch (control) {
    case CongestionControl::kDrop:
      return Z_CONGESTION_CONTROL_DROP;
    case CongestionControl::kBlock:
      return Z_CONGESTION_CONTROL_BLOCK;
  }
  return Z_CONGESTION_CONTROL_BLOCK;
}

z_priority_t zenohPriority(Priority priority) {
  switch (priority) {
    case Priority::kInteractiveHigh:
      return Z_PRIORITY_INTERACTIVE_HIGH;
    case Priority::kData:
      return Z_PRIORITY_DATA;
  }
  return Z_PRIORITY_DATA;
}

// A random id, the same width in every bus: a writer id, or a bus id in hex.
uint64_t randomId() {
  static absl::NoDestructor<absl::Mutex> mutex;
  static absl::NoDestructor<std::mt19937_64> generator(std::random_device{}());
  absl::MutexLock lock(*mutex);
  return (*generator)();
}

std::string keyExprText(const z_loaned_keyexpr_t* absl_nonnull keyExpr) {
  z_view_string_t text;
  z_keyexpr_as_view_string(keyExpr, &text);
  const z_loaned_string_t* absl_nonnull loaned = z_view_string_loan(&text);
  return std::string(z_string_data(loaned), z_string_len(loaned));
}

// The sample callback of a subscription, on a Zenoh thread: copies the
// envelope into the subscription's inbox and wakes the IO thread.
void onSample(z_loaned_sample_t* absl_nonnull sample,
              void* absl_nullable context) {
  Subscription& subscription = *static_cast<Subscription*>(context);
  const z_loaned_bytes_t* absl_nonnull payload = z_sample_payload(sample);
  std::string bytes(z_bytes_len(payload), '\0');
  z_bytes_reader_t reader = z_bytes_get_reader(payload);
  bytes.resize(z_bytes_reader_read(
      &reader, reinterpret_cast<uint8_t*>(bytes.data()), bytes.size()));
  {
    absl::MutexLock lock(subscription.inboxMutex);
    if (subscription.inbox.size() >= subscription.depth) {
      subscription.inbox.pop_front();
    }
    subscription.inbox.push_back(std::move(bytes));
  }
  subscription.waker->wake();
}

// The liveliness callback of a bus, on a Zenoh thread.
void onToken(z_loaned_sample_t* absl_nonnull sample,
             void* absl_nullable context) {
  TokenInbox& inbox = *static_cast<TokenInbox*>(context);
  TokenEvent event{.keyExpr = keyExprText(z_sample_keyexpr(sample)),
                   .alive = z_sample_kind(sample) == Z_SAMPLE_KIND_PUT};
  {
    absl::MutexLock lock(inbox.mutex);
    inbox.events.push_back(std::move(event));
  }
  inbox.waker->wake();
}

// A closure that does not own `context`.
void dropNothing(void* absl_nullable context) {
  static_cast<void>(context);
}

absl::Status declareToken(const z_loaned_session_t* absl_nonnull session,
                          absl::string_view keyExpr,
                          z_owned_liveliness_token_t& token) {
  // z_view_keyexpr_from_str() needs a NUL-terminated key.
  const std::string keyText(keyExpr);
  z_view_keyexpr_t key;
  if (z_view_keyexpr_from_str(&key, keyText.c_str()) != Z_OK ||
      z_liveliness_declare_token(session, &token, z_view_keyexpr_loan(&key),
                                 /*_options=*/nullptr) != Z_OK) {
    return absl::InternalError(absl::StrCat(
        "robot_ipc: cannot declare the liveliness token '", keyExpr, "'"));
  }
  return absl::OkStatus();
}

}  // namespace

class Bus::Impl {
 public:
  explicit Impl(BusOptions options)
      : options_(std::move(options)),
        busId_(absl::StrCat(absl::Hex(randomId(), absl::kZeroPad16))) {}

  ~Impl() {
    if (isIoThread()) {
      LOG(FATAL) << "robot_ipc: a Bus was destroyed from its own IO thread (a "
                    "handler or a periodic callback)";
    }
    stop();
    // The subscribers first, so that no Zenoh thread calls into a subscription
    // or the token inbox once they go.
    if (tokenSubscriberDeclared_) {
      z_subscriber_drop(z_subscriber_move(&tokenSubscriber_));
    }
    for (std::pair<const std::string, std::unique_ptr<Subscription>>& entry :
         subscriptions_) {
      if (entry.second->declared) {
        z_subscriber_drop(z_subscriber_move(&entry.second->subscriber));
        z_liveliness_token_drop(z_liveliness_token_move(&entry.second->token));
      }
    }
    for (std::pair<const std::string, std::unique_ptr<TopicWriter>>& entry :
         writers_) {
      z_publisher_drop(z_publisher_move(&entry.second->publisher));
      z_liveliness_token_drop(z_liveliness_token_move(&entry.second->token));
    }
    if (session_ != nullptr) {
      sessionRegistry().release(options_.network.domainId);
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  absl::Status initialize();
  absl::Status addSubscription(std::unique_ptr<Subscription> subscription);
  absl::Status addPeriodicCallback(absl::Duration period,
                                   absl::AnyInvocable<void()> callback);
  absl::Status start();
  void stop();
  bool isRunning() const { return running_.load() && !stopRequested_.load(); }
  absl::Status publish(absl::string_view topic,
                       const google::protobuf::Message& message);
  absl::Status publishRaw(absl::string_view topic,
                          absl::string_view typeName,
                          absl::string_view payload);
  absl::Status publishFromIoThread(absl::string_view topic,
                                   const google::protobuf::Message& message);
  bool isIoThread() const {
    return ioThreadId_.load() == std::this_thread::get_id();
  }

  const std::string& nodeName() const { return options_.nodeName; }
  int domainId() const { return options_.network.domainId; }
  const std::vector<std::string>& discoveryPeers() const {
    return discoveryPeers_;
  }
  const std::string& domainConfig() const { return sessionConfig_; }
  TopicStatistics topicStatistics(absl::string_view topic) const;
  absl::flat_hash_map<std::string, TopicStatistics> statistics() const;
  uint64_t periodicCallbackErrors() const {
    return periodicCallbackErrors_.load();
  }

 private:
  absl::Status checkCanPublish(absl::string_view topic) const;
  absl::StatusOr<OutgoingMessage> serialize(
      absl::string_view topic, const google::protobuf::Message& message);
  TopicCounters& countersFor(absl::string_view topic);
  absl::StatusOr<TopicWriter* absl_nonnull> writerFor(absl::string_view topic);
  absl::Status declareSubscriber(Subscription& subscription);
  absl::Status declareTokenSubscriber();
  absl::Status enqueue(OutgoingMessage outgoing);

  // The IO thread.
  void runIoLoop();
  absl::Duration waitTimeout(Clock::time_point now) const;
  void sendQueued();
  void send(OutgoingMessage& message);
  int64_t nextSourceTimestamp();
  void drain();
  void take(Subscription& subscription, size_t& budget);
  std::optional<EnvelopeView> decode(const Subscription& subscription,
                                     absl::string_view bytes) const;
  bool acceptsType(const Subscription& subscription,
                   const EnvelopeView& envelope) const;
  void trackSequence(Subscription& subscription, const EnvelopeHeader& header);
  void coalesceLatest();
  void dispatch(Subscription& subscription, const EnvelopeView& envelope);
  void applyTokenEvents();
  void countPeers();
  void runDueCallbacks(Clock::time_point now);

  const BusOptions options_;
  // This bus's id in its liveliness tokens (KeyExprs.h).
  const std::string busId_;
  std::string sessionConfig_;
  std::vector<std::string> discoveryPeers_;
  // Null until initialize() acquired the session of the domain.
  const z_loaned_session_t* absl_nullable session_ = nullptr;
  Waker waker_;
  TokenInbox tokenInbox_;
  z_owned_subscriber_t tokenSubscriber_;
  bool tokenSubscriberDeclared_ = false;

  // Serializes the configuration calls, start() and stop(), which hold it
  // across the join of the IO thread. The IO thread never takes it: every call
  // that does refuses on the IO thread before it locks.
  mutable absl::Mutex lifecycleMutex_;
  std::thread ioThread_ ABSL_GUARDED_BY(lifecycleMutex_);
  std::atomic<bool> running_ = false;
  std::atomic<bool> stopRequested_ = false;
  std::atomic<std::thread::id> ioThreadId_;

  // Changed only while no IO thread runs (under lifecycleMutex_, or in
  // initialize()), and used by the IO thread only while it runs; the thread's
  // start and join order the two. writers_ also grows on the IO thread (a topic
  // published for the first time).
  absl::flat_hash_map<std::string, std::unique_ptr<TopicWriter>> writers_;
  absl::flat_hash_map<std::string, std::unique_ptr<Subscription>>
      subscriptions_;
  // The live liveliness tokens of the domain by key expression (IO thread).
  absl::flat_hash_map<std::string, EndpointToken> liveTokens_;
  // The samples of the current drain, in the order they were written (the IO
  // thread's own; it keeps its capacity).
  std::vector<PendingMessage> pending_;
  std::vector<PeriodicCallback> periodicCallbacks_;
  // The source timestamp of the last sample written: every next one is later,
  // across topics (IO thread).
  int64_t lastSourceTimestamp_ = 0;

  // publish() -> IO thread. The IO thread holds the mutex only to swap the
  // vector with sending_.
  absl::Mutex queueMutex_;
  std::vector<OutgoingMessage> queue_ ABSL_GUARDED_BY(queueMutex_);
  std::vector<OutgoingMessage> sending_;

  mutable absl::Mutex countersMutex_;
  absl::flat_hash_map<std::string, std::unique_ptr<TopicCounters>> counters_
      ABSL_GUARDED_BY(countersMutex_);
  std::atomic<uint64_t> periodicCallbackErrors_ = 0;
};

absl::Status Bus::Impl::initialize() {
  queue_.reserve(options_.publishQueueCapacity);
  sending_.reserve(options_.publishQueueCapacity);
  pending_.reserve(
      std::min(options_.maxMessagesPerDrain, kInitialPendingCapacity));
  discoveryPeers_ = options_.network.peerHosts();
  tokenInbox_.waker = &waker_;
  ROBOT_IPC_ASSIGN_OR_RETURN(
      SessionLease lease,
      sessionRegistry().acquire(options_.network, options_.nodeName));
  session_ = lease.session;
  sessionConfig_ = std::move(lease.config);
  ROBOT_IPC_RETURN_IF_ERROR(declareTokenSubscriber());

  // The publishers of the topics this node publishes exist from the start, so
  // that the subscribers know of them before the first sample (a subscriber
  // gets only what is put after the declarations met).
  if (!options_.nodeName.empty()) {
    for (const std::string& topic :
         options_.network.topicsPublishedBy(options_.nodeName)) {
      ROBOT_IPC_RETURN_IF_ERROR(writerFor(topic).status());
    }
  }
  return absl::OkStatus();
}

absl::Status Bus::Impl::declareTokenSubscriber() {
  const std::string keyExpr = allEndpointTokensKeyExpr(options_.network);
  z_view_keyexpr_t key;
  if (z_view_keyexpr_from_str(&key, keyExpr.c_str()) != Z_OK) {
    return absl::InternalError(
        absl::StrCat("robot_ipc: '", keyExpr, "' is no key expression"));
  }
  z_owned_closure_sample_t callback;
  z_closure_sample(&callback, &onToken, &dropNothing, &tokenInbox_);
  z_liveliness_subscriber_options_t subscriberOptions;
  z_liveliness_subscriber_options_default(&subscriberOptions);
  // The tokens declared before this bus came up, too.
  subscriberOptions.history = true;
  if (z_liveliness_declare_subscriber(
          session_, &tokenSubscriber_, z_view_keyexpr_loan(&key),
          z_closure_sample_move(&callback), &subscriberOptions) != Z_OK) {
    return absl::InternalError(
        absl::StrCat("robot_ipc: cannot subscribe to the liveliness tokens '",
                     keyExpr, "'"));
  }
  tokenSubscriberDeclared_ = true;
  return absl::OkStatus();
}

absl::StatusOr<TopicWriter* absl_nonnull> Bus::Impl::writerFor(
    absl::string_view topic) {
  const absl::flat_hash_map<std::string, std::unique_ptr<TopicWriter>>::iterator
      it = writers_.find(topic);
  if (it != writers_.end()) {
    return it->second.get();
  }
  ROBOT_IPC_RETURN_IF_ERROR(validateTopicName(topic));
  std::unique_ptr<TopicWriter> writer = std::make_unique<TopicWriter>();
  writer->profile = options_.network.qosProfileOf(topic);
  writer->writerId = randomId();
  writer->counters = &countersFor(topic);
  const std::string keyExpr = dataKeyExpr(options_.network, topic);
  z_view_keyexpr_t key;
  if (z_view_keyexpr_from_str(&key, keyExpr.c_str()) != Z_OK) {
    return absl::InternalError(
        absl::StrCat("robot_ipc: '", keyExpr, "' is no key expression"));
  }
  const QosPolicy policy = qosPolicy(writer->profile);
  z_publisher_options_t publisherOptions;
  z_publisher_options_default(&publisherOptions);
  publisherOptions.congestion_control =
      zenohCongestionControl(policy.congestionControl);
  publisherOptions.priority = zenohPriority(policy.priority);
  publisherOptions.is_express = policy.express;
  if (z_declare_publisher(session_, &writer->publisher,
                          z_view_keyexpr_loan(&key),
                          &publisherOptions) != Z_OK) {
    return absl::InternalError(
        absl::StrCat("robot_ipc: cannot declare the Zenoh publisher of '",
                     topic, "' (", keyExpr, ")"));
  }
  const absl::Status token = declareToken(
      session_,
      endpointTokenKeyExpr(options_.network,
                           EndpointToken{.role = EndpointRole::kPublisher,
                                         .profile = writer->profile,
                                         .busId = busId_,
                                         .topic = std::string(topic)}),
      writer->token);
  if (!token.ok()) {
    z_publisher_drop(z_publisher_move(&writer->publisher));
    return token;
  }
  TopicWriter* absl_nonnull entry = writer.get();
  writers_.emplace(std::string(topic), std::move(writer));
  countPeers();
  return entry;
}

absl::Status Bus::Impl::declareSubscriber(Subscription& subscription) {
  subscription.profile = options_.network.qosProfileOf(subscription.topic);
  subscription.depth =
      static_cast<size_t>(subscription.delivery == Delivery::kAll
                              ? options_.allDeliveryHistoryDepth
                              : options_.latestDeliveryHistoryDepth);
  subscription.waker = &waker_;
  subscription.taken.reserve(
      std::min(subscription.depth, options_.maxMessagesPerDrain));
  const std::string keyExpr = dataKeyExpr(options_.network, subscription.topic);
  z_view_keyexpr_t key;
  if (z_view_keyexpr_from_str(&key, keyExpr.c_str()) != Z_OK) {
    return absl::InternalError(
        absl::StrCat("robot_ipc: '", keyExpr, "' is no key expression"));
  }
  z_owned_closure_sample_t callback;
  z_closure_sample(&callback, &onSample, &dropNothing, &subscription);
  if (z_declare_subscriber(session_, &subscription.subscriber,
                           z_view_keyexpr_loan(&key),
                           z_closure_sample_move(&callback),
                           /*options=*/nullptr) != Z_OK) {
    return absl::InternalError(
        absl::StrCat("robot_ipc: cannot declare the Zenoh subscriber of '",
                     subscription.topic, "' (", keyExpr, ")"));
  }
  const absl::Status token = declareToken(
      session_,
      endpointTokenKeyExpr(options_.network,
                           EndpointToken{.role = EndpointRole::kSubscriber,
                                         .profile = subscription.profile,
                                         .busId = busId_,
                                         .topic = subscription.topic}),
      subscription.token);
  if (!token.ok()) {
    z_subscriber_drop(z_subscriber_move(&subscription.subscriber));
    return token;
  }
  subscription.declared = true;
  return absl::OkStatus();
}

absl::Status Bus::Impl::addSubscription(
    std::unique_ptr<Subscription> subscription) {
  ROBOT_IPC_RETURN_IF_ERROR(validateTopicName(subscription->topic));
  if (isIoThread()) {
    return absl::FailedPreconditionError(
        "robot_ipc: subscribe() from a handler; subscribe before start()");
  }
  absl::MutexLock lock(lifecycleMutex_);
  if (ioThread_.joinable()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "robot_ipc: cannot subscribe to '", subscription->topic,
        "' while the bus runs; subscribe before start() or after stop()"));
  }
  if (subscriptions_.contains(subscription->topic)) {
    return absl::AlreadyExistsError(
        absl::StrCat("robot_ipc: the topic '", subscription->topic,
                     "' already has a subscription"));
  }
  subscription->counters = &countersFor(subscription->topic);
  // The subscription's address is the Zenoh callback's context: it is in the
  // map before the first sample can arrive.
  Subscription& entry = *subscription;
  const std::string topic = subscription->topic;
  subscriptions_.emplace(topic, std::move(subscription));
  const absl::Status declared = declareSubscriber(entry);
  if (!declared.ok()) {
    subscriptions_.erase(topic);
    return declared;
  }
  countPeers();
  return absl::OkStatus();
}

absl::Status Bus::Impl::addPeriodicCallback(
    absl::Duration period, absl::AnyInvocable<void()> callback) {
  if (period <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError(
        "robot_ipc: a periodic callback needs a positive period");
  }
  if (callback == nullptr) {
    return absl::InvalidArgumentError(
        "robot_ipc: the periodic callback is empty");
  }
  if (isIoThread()) {
    return absl::FailedPreconditionError(
        "robot_ipc: addPeriodicCallback() from a handler; add it before "
        "start()");
  }
  absl::MutexLock lock(lifecycleMutex_);
  if (ioThread_.joinable()) {
    return absl::FailedPreconditionError(
        "robot_ipc: cannot add a periodic callback while the bus runs; add it "
        "before start() or after stop()");
  }
  PeriodicCallback periodic;
  periodic.period = std::chrono::duration_cast<Clock::duration>(
      absl::ToChronoNanoseconds(period));
  periodic.callback = std::move(callback);
  periodicCallbacks_.push_back(std::move(periodic));
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
    // NOLINTNEXTLINE(exceptions): std::thread throws std::system_error; converted to a Status at once
  } catch (const std::system_error& error) {
    running_.store(false);
    return absl::InternalError(
        absl::StrCat("robot_ipc: cannot start the IO thread: ", error.what()));
  }
  return absl::OkStatus();
}

void Bus::Impl::stop() {
  if (isIoThread()) {
    // Joining itself would deadlock: the loop ends after this call, and the
    // next stop() or the destructor joins.
    stopRequested_.store(true);
    waker_.wake();
    return;
  }
  absl::MutexLock lock(lifecycleMutex_);
  if (ioThread_.joinable()) {
    stopRequested_.store(true);
    waker_.wake();
    ioThread_.join();
  }
  running_.store(false);
}

absl::Status Bus::Impl::checkCanPublish(absl::string_view topic) const {
  if (options_.nodeName.empty()) {
    return absl::FailedPreconditionError(
        "robot_ipc: this bus has no node name, so it only subscribes and "
        "cannot publish");
  }
  return validateTopicName(topic);
}

absl::StatusOr<OutgoingMessage> Bus::Impl::serialize(
    absl::string_view topic, const google::protobuf::Message& message) {
  const absl::string_view typeName = message.GetDescriptor()->full_name();
  OutgoingMessage outgoing;
  outgoing.topic = std::string(topic);
  outgoing.typeName = std::string(typeName);
  if (!message.SerializeToString(&outgoing.payload)) {
    return absl::InternalError(absl::StrCat("robot_ipc: cannot serialize a ",
                                            typeName, " for '", topic, "'"));
  }
  outgoing.counters = &countersFor(topic);
  return outgoing;
}

absl::Status Bus::Impl::publish(absl::string_view topic,
                                const google::protobuf::Message& message) {
  ROBOT_IPC_RETURN_IF_ERROR(checkCanPublish(topic));
  ROBOT_IPC_ASSIGN_OR_RETURN(OutgoingMessage outgoing,
                             serialize(topic, message));
  return enqueue(std::move(outgoing));
}

absl::Status Bus::Impl::publishRaw(absl::string_view topic,
                                   absl::string_view typeName,
                                   absl::string_view payload) {
  ROBOT_IPC_RETURN_IF_ERROR(checkCanPublish(topic));
  return enqueue(OutgoingMessage{.topic = std::string(topic),
                                 .typeName = std::string(typeName),
                                 .payload = std::string(payload),
                                 .counters = &countersFor(topic)});
}

absl::Status Bus::Impl::enqueue(OutgoingMessage outgoing) {
  bool wasEmpty = false;
  {
    absl::MutexLock lock(queueMutex_);
    if (queue_.size() >= options_.publishQueueCapacity) {
      increment(outgoing.counters->sendDropped);
      return absl::ResourceExhaustedError(
          absl::StrCat("robot_ipc: the publish queue is full (",
                       options_.publishQueueCapacity,
                       " messages); dropped one on '", outgoing.topic, "'"));
    }
    wasEmpty = queue_.empty();
    queue_.push_back(std::move(outgoing));
  }
  // The IO thread empties the whole queue on one wake-up, so only the first
  // message needs to wake it.
  if (wasEmpty) {
    waker_.wake();
  }
  return absl::OkStatus();
}

absl::Status Bus::Impl::publishFromIoThread(
    absl::string_view topic, const google::protobuf::Message& message) {
  if (!isIoThread()) {
    return absl::FailedPreconditionError(
        "robot_ipc: publishFromIoThread() outside the bus's IO thread; call "
        "publish()");
  }
  ROBOT_IPC_RETURN_IF_ERROR(checkCanPublish(topic));
  ROBOT_IPC_ASSIGN_OR_RETURN(OutgoingMessage outgoing,
                             serialize(topic, message));
  // What publish() queued before goes first, so that the messages of one thread
  // keep their order.
  sendQueued();
  send(outgoing);
  return absl::OkStatus();
}

TopicCounters& Bus::Impl::countersFor(absl::string_view topic) {
  absl::MutexLock lock(countersMutex_);
  absl::flat_hash_map<std::string, std::unique_ptr<TopicCounters>>::iterator
      it = counters_.find(topic);
  if (it == counters_.end()) {
    it =
        counters_.emplace(std::string(topic), std::make_unique<TopicCounters>())
            .first;
  }
  return *it->second;
}

TopicStatistics Bus::Impl::topicStatistics(absl::string_view topic) const {
  absl::MutexLock lock(countersMutex_);
  const absl::flat_hash_map<std::string,
                            std::unique_ptr<TopicCounters>>::const_iterator it =
      counters_.find(topic);
  return it == counters_.end() ? TopicStatistics() : it->second->snapshot();
}

absl::flat_hash_map<std::string, TopicStatistics> Bus::Impl::statistics()
    const {
  absl::MutexLock lock(countersMutex_);
  absl::flat_hash_map<std::string, TopicStatistics> statistics;
  statistics.reserve(counters_.size());
  for (const std::pair<const std::string, std::unique_ptr<TopicCounters>>&
           entry : counters_) {
    statistics.emplace(entry.first, entry.second->snapshot());
  }
  return statistics;
}

void Bus::Impl::runIoLoop() {
  ioThreadId_.store(std::this_thread::get_id());
  while (!stopRequested_.load(std::memory_order_acquire)) {
    try {  // NOLINT(exceptions): the IO thread's outermost loop: what escapes a handler is logged
      waker_.wait(waitTimeout(Clock::now()));
      sendQueued();
      applyTokenEvents();
      drain();
      runDueCallbacks(Clock::now());
      // NOLINTNEXTLINE(exceptions): the IO thread's outermost loop: what escapes a handler is logged
    } catch (const std::exception& exception) {
      LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds)
          << "robot_ipc: the IO thread caught an exception: "
          << exception.what();
      // NOLINTNEXTLINE(exceptions): the IO thread's outermost loop: what escapes a handler is logged
    } catch (...) {
      LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds)
          << "robot_ipc: the IO thread caught an exception of unknown type";
    }
  }
  sendQueued();
  applyTokenEvents();
  ioThreadId_.store(std::thread::id());
}

absl::Duration Bus::Impl::waitTimeout(Clock::time_point now) const {
  Clock::duration wait = std::chrono::duration_cast<Clock::duration>(
      absl::ToChronoNanoseconds(options_.ioPollPeriod));
  for (const PeriodicCallback& periodic : periodicCallbacks_) {
    wait = std::min(wait, periodic.nextDeadline - now);
  }
  if (wait <= Clock::duration::zero()) {
    return absl::ZeroDuration();
  }
  return absl::FromChrono(wait);
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

int64_t Bus::Impl::nextSourceTimestamp() {
  lastSourceTimestamp_ =
      std::max(absl::GetCurrentTimeNanos(), lastSourceTimestamp_ + 1);
  return lastSourceTimestamp_;
}

void Bus::Impl::send(OutgoingMessage& message) {
  const absl::StatusOr<TopicWriter* absl_nonnull> writer =
      writerFor(message.topic);
  if (!writer.ok()) {
    increment(message.counters->sendDropped);
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << writer.status();
    return;
  }
  TopicWriter& entry = **writer;
  if (message.typeName.size() > static_cast<size_t>(UINT32_MAX)) {
    increment(message.counters->sendDropped);
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds)
        << "robot_ipc: a type name of " << message.typeName.size()
        << " bytes on '" << message.topic
        << "' is longer than an envelope holds";
    return;
  }
  const EnvelopeHeader header{.protocolVersion = kWireProtocolVersion,
                              .writerId = entry.writerId,
                              .sequenceNumber = entry.lastSequence + 1,
                              .sourceTimestampNs = nextSourceTimestamp()};
  const std::string envelope =
      encodeEnvelope(header, message.typeName, message.payload);
  z_owned_bytes_t payload;
  if (z_bytes_copy_from_buf(&payload,
                            reinterpret_cast<const uint8_t*>(envelope.data()),
                            envelope.size()) != Z_OK ||
      z_publisher_put(z_publisher_loan(&entry.publisher),
                      z_bytes_move(&payload), /*options=*/nullptr) != Z_OK) {
    increment(message.counters->sendDropped);
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
        << "robot_ipc: putting a sample on '" << message.topic << "' failed";
    return;
  }
  entry.lastSequence = header.sequenceNumber;
  increment(message.counters->written);
}

void Bus::Impl::drain() {
  pending_.clear();
  size_t budget = options_.maxMessagesPerDrain;
  // Every subscription's envelopes are taken before any is decoded, so that
  // the views of pending_ point into vectors that no longer grow.
  for (std::pair<const std::string, std::unique_ptr<Subscription>>& entry :
       subscriptions_) {
    take(*entry.second, budget);
  }
  for (std::pair<const std::string, std::unique_ptr<Subscription>>& entry :
       subscriptions_) {
    Subscription& subscription = *entry.second;
    for (const std::string& bytes : subscription.taken) {
      increment(subscription.counters->received);
      const std::optional<EnvelopeView> envelope = decode(subscription, bytes);
      if (!envelope.has_value()) {
        increment(subscription.counters->rejected);
        continue;
      }
      trackSequence(subscription, envelope->header);
      if (!acceptsType(subscription, *envelope)) {
        increment(subscription.counters->rejected);
        continue;
      }
      pending_.push_back(PendingMessage{.subscription = &subscription,
                                        .envelope = *envelope,
                                        .superseded = false});
    }
  }
  // In the order the samples were written: each bus's source timestamps
  // increase strictly across its topics.
  std::stable_sort(
      pending_.begin(), pending_.end(),
      [](const PendingMessage& first, const PendingMessage& second) {
        return first.envelope.header.sourceTimestampNs <
               second.envelope.header.sourceTimestampNs;
      });
  coalesceLatest();
  for (PendingMessage& message : pending_) {
    if (!message.superseded) {
      dispatch(*message.subscription, message.envelope);
    }
  }
  pending_.clear();
  for (std::pair<const std::string, std::unique_ptr<Subscription>>& entry :
       subscriptions_) {
    entry.second->taken.clear();
  }
}

void Bus::Impl::take(Subscription& subscription, size_t& budget) {
  absl::MutexLock lock(subscription.inboxMutex);
  while (budget > 0 && !subscription.inbox.empty()) {
    subscription.taken.push_back(std::move(subscription.inbox.front()));
    subscription.inbox.pop_front();
    --budget;
  }
  // What the budget left behind waits for the next drain, which comes at once.
  if (!subscription.inbox.empty()) {
    waker_.wake();
  }
}

void Bus::Impl::trackSequence(Subscription& subscription,
                              const EnvelopeHeader& header) {
  const std::pair<absl::flat_hash_map<uint64_t, uint64_t>::iterator, bool>
      inserted = subscription.lastSequenceByWriter.try_emplace(
          header.writerId, header.sequenceNumber);
  if (inserted.second) {
    return;
  }
  uint64_t& last = inserted.first->second;
  if (header.sequenceNumber > last + 1) {
    increment(subscription.counters->missed, header.sequenceNumber - last - 1);
  }
  last = std::max(last, header.sequenceNumber);
}

std::optional<EnvelopeView> Bus::Impl::decode(const Subscription& subscription,
                                              absl::string_view bytes) const {
  const std::optional<uint32_t> version = envelopeProtocolVersion(bytes);
  if (version != kWireProtocolVersion) {
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
        << "robot_ipc: a sample on '" << subscription.topic
        << "' speaks wire protocol "
        << (version.has_value() ? absl::StrCat(*version) : "none")
        << ", but this bus speaks " << kWireProtocolVersion
        << "; rebuild both from the same checkout";
    return std::nullopt;
  }
  std::optional<EnvelopeView> envelope = decodeEnvelope(bytes);
  if (!envelope.has_value()) {
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
        << "robot_ipc: a sample on '" << subscription.topic
        << "' is shorter than its envelope says";
  }
  return envelope;
}

bool Bus::Impl::acceptsType(const Subscription& subscription,
                            const EnvelopeView& envelope) const {
  if (!subscription.expectedTypeName.empty() &&
      envelope.typeName != subscription.expectedTypeName) {
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
        << "robot_ipc: '" << subscription.topic << "' carries a "
        << envelope.typeName << ", but its subscription takes a "
        << subscription.expectedTypeName;
    return false;
  }
  return true;
}

void Bus::Impl::coalesceLatest() {
  for (std::pair<const std::string, std::unique_ptr<Subscription>>& entry :
       subscriptions_) {
    entry.second->latestPending.reset();
  }
  for (size_t index = 0; index < pending_.size(); ++index) {
    Subscription& subscription = *pending_[index].subscription;
    if (subscription.delivery != Delivery::kLatest) {
      continue;
    }
    if (subscription.latestPending.has_value()) {
      pending_[*subscription.latestPending].superseded = true;
      increment(subscription.counters->superseded);
    }
    subscription.latestPending = index;
  }
}

void Bus::Impl::dispatch(Subscription& subscription,
                         const EnvelopeView& envelope) {
  TopicCounters& counters = *subscription.counters;
  const absl::string_view payload = envelope.payload;
  if (subscription.message != nullptr &&
      (payload.size() > static_cast<size_t>(INT_MAX) ||
       !subscription.message->ParseFromArray(
           payload.data(), static_cast<int>(payload.size())))) {
    increment(counters.rejected);
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
        << "robot_ipc: a sample on '" << subscription.topic
        << "' does not parse as a " << envelope.typeName;
    return;
  }
  increment(counters.delivered);
  try {  // NOLINT(exceptions): a handler's exception is counted and logged, not let end the IO thread
    if (subscription.message != nullptr) {
      subscription.messageHandler(*subscription.message);
    } else {
      subscription.rawHandler(envelope.typeName, payload);
    }
    // NOLINTNEXTLINE(exceptions): a handler's exception is counted and logged, not let end the IO thread
  } catch (const std::exception& exception) {
    increment(counters.handlerErrors);
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds)
        << "robot_ipc: the handler of '" << subscription.topic
        << "' threw: " << exception.what();
    // NOLINTNEXTLINE(exceptions): a handler's exception is counted and logged, not let end the IO thread
  } catch (...) {
    increment(counters.handlerErrors);
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds)
        << "robot_ipc: the handler of '" << subscription.topic
        << "' threw an exception of unknown type";
  }
}

void Bus::Impl::applyTokenEvents() {
  std::vector<TokenEvent> events;
  {
    absl::MutexLock lock(tokenInbox_.mutex);
    if (tokenInbox_.events.empty()) {
      return;
    }
    events.swap(tokenInbox_.events);
  }
  for (TokenEvent& event : events) {
    if (!event.alive) {
      liveTokens_.erase(event.keyExpr);
      continue;
    }
    std::optional<EndpointToken> token =
        parseEndpointTokenKeyExpr(options_.network, event.keyExpr);
    if (token.has_value()) {
      liveTokens_.insert_or_assign(std::move(event.keyExpr), *std::move(token));
    }
  }
  countPeers();
}

void Bus::Impl::countPeers() {
  for (std::pair<const std::string, std::unique_ptr<TopicWriter>>& entry :
       writers_) {
    const TopicWriter& writer = *entry.second;
    uint64_t matched = 0;
    uint64_t incompatible = 0;
    for (const std::pair<const std::string, EndpointToken>& live :
         liveTokens_) {
      const EndpointToken& token = live.second;
      if (token.role != EndpointRole::kSubscriber ||
          token.topic != entry.first) {
        continue;
      }
      ++(token.profile == writer.profile ? matched : incompatible);
    }
    writer.counters->matchedReaders.store(matched, std::memory_order_relaxed);
    writer.counters->incompatibleQos.store(incompatible,
                                           std::memory_order_relaxed);
  }
  for (std::pair<const std::string, std::unique_ptr<Subscription>>& entry :
       subscriptions_) {
    const Subscription& subscription = *entry.second;
    uint64_t matched = 0;
    uint64_t incompatible = 0;
    for (const std::pair<const std::string, EndpointToken>& live :
         liveTokens_) {
      const EndpointToken& token = live.second;
      if (token.role != EndpointRole::kPublisher ||
          token.topic != subscription.topic) {
        continue;
      }
      ++(token.profile == subscription.profile ? matched : incompatible);
    }
    subscription.counters->matchedWriters.store(matched,
                                                std::memory_order_relaxed);
    // A topic this bus both publishes and subscribes counts the incompatible
    // endpoints of both roles.
    const absl::flat_hash_map<
        std::string, std::unique_ptr<TopicWriter>>::const_iterator writer =
        writers_.find(subscription.topic);
    const uint64_t writerIncompatible =
        writer == writers_.end()
            ? 0
            : writer->second->counters->incompatibleQos.load(
                  std::memory_order_relaxed);
    subscription.counters->incompatibleQos.store(
        writerIncompatible + incompatible, std::memory_order_relaxed);
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
      LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds)
          << "robot_ipc: a periodic callback threw: " << exception.what();
      // NOLINTNEXTLINE(exceptions): a periodic callback's exception is counted and logged, not let end the IO thread
    } catch (...) {
      periodicCallbackErrors_.fetch_add(kOneEvent);
      LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds)
          << "robot_ipc: a periodic callback threw an exception of unknown "
             "type";
    }
    // Against the time the callback returned, not `now`: a callback that
    // overran its period must not be called again at once for the deadline it
    // overran. A missed deadline is skipped, to the first one still ahead on
    // the grid of start(), instead of bursting through the missed calls.
    const Clock::time_point after = Clock::now();
    periodic.nextDeadline += periodic.period;
    if (periodic.nextDeadline <= after) {
      periodic.nextDeadline +=
          ((after - periodic.nextDeadline) / periodic.period + 1) *
          periodic.period;
    }
  }
}
// ---------------------------------------------------------------------------------------------------------------------
// Bus
// ---------------------------------------------------------------------------------------------------------------------

absl::StatusOr<std::unique_ptr<Bus>> Bus::Create(BusOptions options) {
  ROBOT_IPC_RETURN_IF_ERROR(validateOptions(options));
  std::unique_ptr<Impl> impl = std::make_unique<Impl>(std::move(options));
  ROBOT_IPC_RETURN_IF_ERROR(impl->initialize());
  return absl::WrapUnique(new Bus(std::move(impl)));
}

Bus::Bus(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Bus::~Bus() = default;

absl::Status Bus::subscribeMessage(
    absl::string_view topic,
    Delivery delivery,
    std::unique_ptr<google::protobuf::Message> message,
    MessageHandler handler) {
  std::unique_ptr<Subscription> subscription = std::make_unique<Subscription>();
  subscription->topic = std::string(topic);
  subscription->delivery = delivery;
  subscription->expectedTypeName =
      std::string(message->GetDescriptor()->full_name());
  subscription->message = std::move(message);
  subscription->messageHandler = std::move(handler);
  return impl_->addSubscription(std::move(subscription));
}

absl::Status Bus::subscribeRaw(absl::string_view topic,
                               Delivery delivery,
                               RawHandler handler) {
  if (handler == nullptr) {
    return absl::InvalidArgumentError("subscribeRaw(): the handler is empty");
  }
  std::unique_ptr<Subscription> subscription = std::make_unique<Subscription>();
  subscription->topic = std::string(topic);
  subscription->delivery = delivery;
  subscription->rawHandler = std::move(handler);
  return impl_->addSubscription(std::move(subscription));
}

absl::Status Bus::addPeriodicCallback(absl::Duration period,
                                      absl::AnyInvocable<void()> callback) {
  return impl_->addPeriodicCallback(period, std::move(callback));
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

absl::Status Bus::publish(absl::string_view topic,
                          const google::protobuf::Message& message) {
  return impl_->publish(topic, message);
}

absl::Status Bus::publishRaw(absl::string_view topic,
                             absl::string_view typeName,
                             absl::string_view payload) {
  return impl_->publishRaw(topic, typeName, payload);
}

absl::Status Bus::publishFromIoThread(
    absl::string_view topic, const google::protobuf::Message& message) {
  return impl_->publishFromIoThread(topic, message);
}

bool Bus::isIoThread() const {
  return impl_->isIoThread();
}

const std::string& Bus::nodeName() const {
  return impl_->nodeName();
}

int Bus::domainId() const {
  return impl_->domainId();
}

const std::vector<std::string>& Bus::discoveryPeers() const {
  return impl_->discoveryPeers();
}

const std::string& Bus::domainConfig() const {
  return impl_->domainConfig();
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
