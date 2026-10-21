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
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "dds/dds.h"

#include "robot_ipc/DomainConfig.h"
#include "robot_ipc/QosProfile.h"
#include "robot_ipc/WireProtocol.h"
#include "robot_runtime/robot_ipc/idl/Envelope.h"
#include "robot_runtime/robot_ipc/src/DomainRegistry.h"
#include "robot_runtime/robot_ipc/src/StatusMacros.h"

namespace robot::ipc {
namespace {

using Clock = std::chrono::steady_clock;

// The IO thread logs a recurring problem at most this often.
constexpr int kLogPeriodSeconds = 5;
constexpr std::chrono::milliseconds kWaitFailureBackoff(10);
// What one counted event adds to its counter.
constexpr uint64_t kOneEvent = 1;
// The samples a drain may hold before it dispatches, before the vector that holds them grows (it keeps its capacity).
constexpr size_t kInitialPendingCapacity = 256;
// The name of the DDS participant of a bus that only subscribes.
constexpr absl::string_view kSubscriberParticipantName = "robot_ipc_subscriber";
// A reliable write never waits for acknowledgments: at the writer's unacknowledged-bytes limit it fails at once and
// is counted as sendDropped.
constexpr dds_duration_t kReliableWriteBlockingTime = 0;
// The data representation of every writer, and the ones a reader accepts (a Python writer may pick either).
constexpr dds_data_representation_id_t kWriterRepresentation = DDS_DATA_REPRESENTATION_XCDR2;
constexpr std::array<dds_data_representation_id_t, 2> kReaderRepresentations = {DDS_DATA_REPRESENTATION_XCDR1,
                                                                                DDS_DATA_REPRESENTATION_XCDR2};

// The counters of one topic. Written by the IO thread (and by publish() for sendDropped), read by any thread.
struct TopicCounters {
  std::atomic<uint64_t> written = 0;
  std::atomic<uint64_t> sendDropped = 0;
  std::atomic<uint64_t> received = 0;
  std::atomic<uint64_t> delivered = 0;
  std::atomic<uint64_t> superseded = 0;
  std::atomic<uint64_t> rejected = 0;
  std::atomic<uint64_t> handlerErrors = 0;
  std::atomic<uint64_t> missed = 0;
  // The DDS status of the topic's reader and writer, as the IO thread last read it.
  std::atomic<uint64_t> sampleLost = 0;
  std::atomic<uint64_t> sampleRejected = 0;
  std::atomic<uint64_t> requestedIncompatibleQos = 0;
  std::atomic<uint64_t> offeredIncompatibleQos = 0;
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
    statistics.sampleLost = sampleLost.load(std::memory_order_relaxed);
    statistics.sampleRejected = sampleRejected.load(std::memory_order_relaxed);
    statistics.incompatibleQos =
        requestedIncompatibleQos.load(std::memory_order_relaxed) + offeredIncompatibleQos.load(std::memory_order_relaxed);
    statistics.matchedReaders = matchedReaders.load(std::memory_order_relaxed);
    statistics.matchedWriters = matchedWriters.load(std::memory_order_relaxed);
    return statistics;
  }
};

void increment(std::atomic<uint64_t>& counter, uint64_t count = kOneEvent) {
  counter.fetch_add(count, std::memory_order_relaxed);
}

// One message, serialized on the publishing thread.
struct OutgoingMessage {
  std::string topic;
  std::string typeName;
  std::string payload;
  TopicCounters* absl_nullable counters = nullptr;
};

// The DDS writer of one topic. The IO thread's own, once it runs.
struct TopicWriter {
  dds_entity_t writer = 0;
  // The sequence number of the last sample written; the next one is one more.
  uint64_t lastSequence = 0;
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
  dds_entity_t reader = 0;
  dds_entity_t readCondition = 0;
  // The loans of the current drain: dds_take() fills `samples` with pointers into the reader's cache, and the drain
  // returns them (dds_return_loan) once it has dispatched. Sized to the reader's history depth.
  std::vector<void* absl_nullable> samples;
  std::vector<dds_sample_info_t> infos;
  size_t taken = 0;
  // The last sequence number of each writer (its publication handle) this reader took a sample of.
  absl::flat_hash_map<dds_instance_handle_t, uint64_t> lastSequenceByWriter;
  // Delivery::kLatest: the index, in the drain's pending samples, of the newest sample of this subscription so far.
  std::optional<size_t> latestPending;
};

// A sample a drain has taken and will dispatch once it has taken the rest, in the order the samples were written.
struct PendingMessage {
  Subscription* absl_nonnull subscription;
  dds_time_t sourceTimestamp = 0;
  const robot_ipc_Envelope* absl_nonnull envelope;
  // Delivery::kLatest: a newer sample of the same topic is in the same drain; this one is not dispatched.
  bool superseded = false;
};

struct PeriodicCallback {
  Clock::duration period = Clock::duration::zero();
  Clock::time_point nextDeadline;
  std::function<void()> callback;
};

absl::Status ddsError(absl::string_view what, dds_return_t code) {
  return absl::InternalError(absl::StrCat("robot_ipc: ", what, ": ", dds_strretcode(code)));
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
  if (options.publishQueueCapacity == 0) {
    return absl::InvalidArgumentError("BusOptions.publishQueueCapacity must be positive");
  }
  if (options.maxMessagesPerDrain == 0) {
    return absl::InvalidArgumentError("BusOptions.maxMessagesPerDrain must be positive");
  }
  if (options.latestDeliveryHistoryDepth <= 0 || options.allDeliveryHistoryDepth <= 0) {
    return absl::InvalidArgumentError("BusOptions.latestDeliveryHistoryDepth and allDeliveryHistoryDepth must be positive");
  }
  return absl::OkStatus();
}

// The QoS of a reader or writer of `profile` with a KEEP_LAST history of `depth`. The caller deletes it.
dds_qos_t* absl_nonnull createQos(QosProfile profile, int depth) {
  dds_qos_t* absl_nonnull qos = dds_create_qos();
  const QosPolicy policy = qosPolicy(profile);
  if (policy.reliability == Reliability::kReliable) {
    dds_qset_reliability(qos, DDS_RELIABILITY_RELIABLE, kReliableWriteBlockingTime);
  } else {
    dds_qset_reliability(qos, DDS_RELIABILITY_BEST_EFFORT, /*max_blocking_time=*/0);
  }
  dds_qset_history(qos, DDS_HISTORY_KEEP_LAST, depth);
  dds_qset_durability(qos, DDS_DURABILITY_VOLATILE);
  return qos;
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
    if (participant_ > 0) {
      // Deletes the readers, writers, topics, the waitset and the guard condition with it.
      const dds_return_t deleted = dds_delete(participant_);
      if (deleted < 0) {
        LOG(ERROR) << "robot_ipc: deleting the DDS participant failed: " << dds_strretcode(deleted);
      }
    }
    if (domainAcquired_) {
      domainRegistry().release(options_.network.domainId);
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  absl::Status initialize();
  absl::Status addSubscription(std::unique_ptr<Subscription> subscription);
  absl::Status addPeriodicCallback(absl::Duration period, std::function<void()> callback);
  absl::Status start();
  void stop();
  bool isRunning() const { return running_.load() && !stopRequested_.load(); }
  absl::Status publish(absl::string_view topic, const google::protobuf::Message& message);
  absl::Status publishRaw(absl::string_view topic, absl::string_view typeName, absl::string_view payload);
  absl::Status publishFromIoThread(absl::string_view topic, const google::protobuf::Message& message);
  bool isIoThread() const { return ioThreadId_.load() == std::this_thread::get_id(); }

  const std::string& nodeName() const { return options_.nodeName; }
  int domainId() const { return options_.network.domainId; }
  const std::vector<std::string>& discoveryPeers() const { return discoveryPeers_; }
  const std::string& domainConfig() const { return domainConfig_; }
  TopicStatistics topicStatistics(absl::string_view topic) const;
  absl::flat_hash_map<std::string, TopicStatistics> statistics() const;
  uint64_t periodicCallbackErrors() const { return periodicCallbackErrors_.load(); }

 private:
  absl::Status checkCanPublish(absl::string_view topic) const;
  absl::StatusOr<OutgoingMessage> serialize(absl::string_view topic, const google::protobuf::Message& message);
  TopicCounters& countersFor(absl::string_view topic);
  absl::StatusOr<dds_entity_t> topicFor(const std::string& name);
  absl::StatusOr<TopicWriter* absl_nonnull> writerFor(const std::string& topic);
  absl::Status createReader(Subscription& subscription);
  void wake() const;
  absl::Status enqueue(OutgoingMessage outgoing);

  // The IO thread.
  void runIoLoop();
  dds_duration_t waitTimeout(Clock::time_point now) const;
  void sendQueued();
  void send(OutgoingMessage& message);
  dds_time_t nextSourceTimestamp();
  void drain();
  size_t take(Subscription& subscription, size_t budget);
  bool accepts(Subscription& subscription, const robot_ipc_Envelope& envelope) const;
  void trackSequence(Subscription& subscription, dds_instance_handle_t writer, uint64_t sequence);
  void coalesceLatest();
  void dispatch(Subscription& subscription, const robot_ipc_Envelope& envelope);
  void refreshStatusIfDue(Clock::time_point now);
  void runDueCallbacks(Clock::time_point now);

  const BusOptions options_;
  std::string domainConfig_;
  std::vector<std::string> discoveryPeers_;
  bool domainAcquired_ = false;
  // 0 until initialize() created them; the participant owns the others.
  dds_entity_t participant_ = 0;
  dds_entity_t waitset_ = 0;
  dds_entity_t wakeCondition_ = 0;

  // Serializes the configuration calls, start() and stop(), which hold it across the join of the IO thread. The IO
  // thread never takes it: every call that does refuses on the IO thread before it locks.
  mutable absl::Mutex lifecycleMutex_;
  std::thread ioThread_ ABSL_GUARDED_BY(lifecycleMutex_);
  std::atomic<bool> running_ = false;
  std::atomic<bool> stopRequested_ = false;
  std::atomic<std::thread::id> ioThreadId_;

  // Changed only while no IO thread runs (under lifecycleMutex_, or in initialize()), and used by the IO thread only
  // while it runs; the thread's start and join order the two. writers_ also grows on the IO thread (a topic published
  // for the first time).
  absl::flat_hash_map<std::string, dds_entity_t> topics_;
  absl::flat_hash_map<std::string, TopicWriter> writers_;
  absl::flat_hash_map<std::string, std::unique_ptr<Subscription>> subscriptions_;
  // The samples of the current drain, in the order they were written (the IO thread's own; it keeps its capacity).
  std::vector<PendingMessage> pending_;
  std::vector<PeriodicCallback> periodicCallbacks_;
  // The source timestamp of the last sample written: every next one is later, across topics (IO thread).
  dds_time_t lastSourceTimestamp_ = 0;
  Clock::time_point nextStatusRefresh_;

  // publish() -> IO thread. The IO thread holds the mutex only to swap the vector with sending_.
  absl::Mutex queueMutex_;
  std::vector<OutgoingMessage> queue_ ABSL_GUARDED_BY(queueMutex_);
  std::vector<OutgoingMessage> sending_;

  mutable absl::Mutex countersMutex_;
  absl::flat_hash_map<std::string, std::unique_ptr<TopicCounters>> counters_ ABSL_GUARDED_BY(countersMutex_);
  std::atomic<uint64_t> periodicCallbackErrors_ = 0;
};

absl::Status Bus::Impl::initialize() {
  queue_.reserve(options_.publishQueueCapacity);
  sending_.reserve(options_.publishQueueCapacity);
  pending_.reserve(std::min(options_.maxMessagesPerDrain, kInitialPendingCapacity));
  discoveryPeers_ = options_.network.peerHosts();
  ROBOT_IPC_ASSIGN_OR_RETURN(domainConfig_, cycloneDomainConfig(options_.network, options_.nodeName));
  ROBOT_IPC_RETURN_IF_ERROR(domainRegistry().acquire(options_.network.domainId, domainConfig_));
  domainAcquired_ = true;

  dds_qos_t* absl_nonnull qos = dds_create_qos();
  const std::string participantName = options_.nodeName.empty() ? std::string(kSubscriberParticipantName) : options_.nodeName;
  dds_qset_entity_name(qos, participantName.c_str());
  participant_ = dds_create_participant(options_.network.domainId, qos, /*listener=*/nullptr);
  dds_delete_qos(qos);
  if (participant_ < 0) {
    const dds_return_t code = participant_;
    participant_ = 0;
    return ddsError(absl::StrCat("cannot join DDS domain ", options_.network.domainId), code);
  }
  waitset_ = dds_create_waitset(participant_);
  if (waitset_ < 0) return ddsError("cannot create the waitset", waitset_);
  wakeCondition_ = dds_create_guardcondition(participant_);
  if (wakeCondition_ < 0) return ddsError("cannot create the wake-up condition", wakeCondition_);
  const dds_return_t attached = dds_waitset_attach(waitset_, wakeCondition_, /*x=*/0);
  if (attached < 0) return ddsError("cannot attach the wake-up condition", attached);

  // The writers of the topics this node publishes exist from the start, so that the subscribers have discovered them
  // before the first sample (a VOLATILE reader gets only what is written after it matched).
  if (!options_.nodeName.empty()) {
    for (const std::string& topic : options_.network.topicsPublishedBy(options_.nodeName)) {
      ROBOT_IPC_RETURN_IF_ERROR(writerFor(topic).status());
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<dds_entity_t> Bus::Impl::topicFor(const std::string& name) {
  const absl::flat_hash_map<std::string, dds_entity_t>::const_iterator it = topics_.find(name);
  if (it != topics_.end()) {
    return it->second;
  }
  ROBOT_IPC_RETURN_IF_ERROR(validateTopicName(name));
  const dds_entity_t topic = dds_create_topic(participant_, &robot_ipc_Envelope_desc, name.c_str(), /*qos=*/nullptr, /*listener=*/nullptr);
  if (topic < 0) {
    return ddsError(absl::StrCat("cannot create the DDS topic '", name, "'"), topic);
  }
  topics_.emplace(name, topic);
  return topic;
}

absl::StatusOr<TopicWriter* absl_nonnull> Bus::Impl::writerFor(const std::string& topic) {
  const absl::flat_hash_map<std::string, TopicWriter>::iterator it = writers_.find(topic);
  if (it != writers_.end()) {
    return &it->second;
  }
  ROBOT_IPC_ASSIGN_OR_RETURN(const dds_entity_t ddsTopic, topicFor(topic));
  const QosProfile profile = options_.network.qosProfileOf(topic);
  dds_qos_t* absl_nonnull qos = createQos(profile, qosPolicy(profile).writerHistoryDepth);
  dds_qset_data_representation(qos, /*n=*/1, &kWriterRepresentation);
  dds_qset_entity_name(qos, options_.nodeName.c_str());
  const dds_entity_t writer = dds_create_writer(participant_, ddsTopic, qos, /*listener=*/nullptr);
  dds_delete_qos(qos);
  if (writer < 0) {
    return ddsError(absl::StrCat("cannot create the DDS writer of '", topic, "'"), writer);
  }
  TopicWriter& entry = writers_[topic];
  entry.writer = writer;
  entry.counters = &countersFor(topic);
  return &entry;
}

absl::Status Bus::Impl::createReader(Subscription& subscription) {
  ROBOT_IPC_ASSIGN_OR_RETURN(const dds_entity_t ddsTopic, topicFor(subscription.topic));
  const int depth = subscription.delivery == Delivery::kAll ? options_.allDeliveryHistoryDepth : options_.latestDeliveryHistoryDepth;
  dds_qos_t* absl_nonnull qos = createQos(options_.network.qosProfileOf(subscription.topic), depth);
  dds_qset_data_representation(qos, static_cast<uint32_t>(kReaderRepresentations.size()), kReaderRepresentations.data());
  const dds_entity_t reader = dds_create_reader(participant_, ddsTopic, qos, /*listener=*/nullptr);
  dds_delete_qos(qos);
  if (reader < 0) {
    return ddsError(absl::StrCat("cannot create the DDS reader of '", subscription.topic, "'"), reader);
  }
  const dds_entity_t condition = dds_create_readcondition(reader, DDS_ANY_STATE);
  if (condition < 0) {
    dds_delete(reader);
    return ddsError(absl::StrCat("cannot create the read condition of '", subscription.topic, "'"), condition);
  }
  const dds_return_t attached = dds_waitset_attach(waitset_, condition, /*x=*/reader);
  if (attached < 0) {
    dds_delete(reader);
    return ddsError(absl::StrCat("cannot attach the reader of '", subscription.topic, "'"), attached);
  }
  subscription.reader = reader;
  subscription.readCondition = condition;
  const size_t capacity = std::min(static_cast<size_t>(depth), options_.maxMessagesPerDrain);
  subscription.samples.assign(capacity, nullptr);
  subscription.infos.resize(capacity);
  return absl::OkStatus();
}

absl::Status Bus::Impl::addSubscription(std::unique_ptr<Subscription> subscription) {
  ROBOT_IPC_RETURN_IF_ERROR(validateTopicName(subscription->topic));
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
  ROBOT_IPC_RETURN_IF_ERROR(createReader(*subscription));
  subscription->counters = &countersFor(subscription->topic);
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
  nextStatusRefresh_ = now;
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
  return validateTopicName(topic);
}

absl::StatusOr<OutgoingMessage> Bus::Impl::serialize(absl::string_view topic, const google::protobuf::Message& message) {
  const absl::string_view typeName = message.GetDescriptor()->full_name();
  OutgoingMessage outgoing;
  outgoing.topic = std::string(topic);
  outgoing.typeName = std::string(typeName);
  if (!message.SerializeToString(&outgoing.payload)) {
    return absl::InternalError(absl::StrCat("robot_ipc: cannot serialize a ", typeName, " for '", topic, "'"));
  }
  outgoing.counters = &countersFor(topic);
  return outgoing;
}

absl::Status Bus::Impl::publish(absl::string_view topic, const google::protobuf::Message& message) {
  ROBOT_IPC_RETURN_IF_ERROR(checkCanPublish(topic));
  ROBOT_IPC_ASSIGN_OR_RETURN(OutgoingMessage outgoing, serialize(topic, message));
  return enqueue(std::move(outgoing));
}

absl::Status Bus::Impl::publishRaw(absl::string_view topic, absl::string_view typeName, absl::string_view payload) {
  ROBOT_IPC_RETURN_IF_ERROR(checkCanPublish(topic));
  return enqueue(OutgoingMessage{
      .topic = std::string(topic), .typeName = std::string(typeName), .payload = std::string(payload), .counters = &countersFor(topic)});
}

absl::Status Bus::Impl::enqueue(OutgoingMessage outgoing) {
  bool wasEmpty = false;
  {
    absl::MutexLock lock(queueMutex_);
    if (queue_.size() >= options_.publishQueueCapacity) {
      increment(outgoing.counters->sendDropped);
      return absl::ResourceExhaustedError(absl::StrCat("robot_ipc: the publish queue is full (", options_.publishQueueCapacity,
                                                       " messages); dropped one on '", outgoing.topic, "'"));
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
  // Fails only for a deleted condition, i.e. a bus being destroyed, whose IO thread has stopped already.
  static_cast<void>(dds_set_guardcondition(wakeCondition_, /*triggered=*/true));
}

void Bus::Impl::runIoLoop() {
  ioThreadId_.store(std::this_thread::get_id());
  while (!stopRequested_.load(std::memory_order_acquire)) {
    try {  // NOLINT(exceptions): the IO thread's outermost loop: what escapes a handler is logged
      const dds_return_t waited = dds_waitset_wait(waitset_, /*xs=*/nullptr, /*nxs=*/0, waitTimeout(Clock::now()));
      if (waited < 0) {
        LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: dds_waitset_wait() failed: " << dds_strretcode(waited);
        std::this_thread::sleep_for(kWaitFailureBackoff);
        continue;
      }
      bool woken = false;
      static_cast<void>(dds_take_guardcondition(wakeCondition_, &woken));
      sendQueued();
      drain();
      const Clock::time_point now = Clock::now();
      refreshStatusIfDue(now);
      runDueCallbacks(now);
      // NOLINTNEXTLINE(exceptions): the IO thread's outermost loop: what escapes a handler is logged
    } catch (const std::exception& exception) {
      LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: the IO thread caught an exception: " << exception.what();
    } catch (...) {  // NOLINT(exceptions): the IO thread's outermost loop: what escapes a handler is logged
      LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: the IO thread caught an exception of unknown type";
    }
  }
  sendQueued();
  refreshStatusIfDue(Clock::now());
  ioThreadId_.store(std::thread::id());
}

dds_duration_t Bus::Impl::waitTimeout(Clock::time_point now) const {
  Clock::duration wait = std::chrono::duration_cast<Clock::duration>(absl::ToChronoNanoseconds(options_.ioPollPeriod));
  for (const PeriodicCallback& periodic : periodicCallbacks_) {
    wait = std::min(wait, periodic.nextDeadline - now);
  }
  if (wait <= Clock::duration::zero()) {
    return 0;
  }
  return static_cast<dds_duration_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(wait).count());
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

dds_time_t Bus::Impl::nextSourceTimestamp() {
  lastSourceTimestamp_ = std::max(dds_time(), lastSourceTimestamp_ + 1);
  return lastSourceTimestamp_;
}

void Bus::Impl::send(OutgoingMessage& message) {
  const absl::StatusOr<TopicWriter* absl_nonnull> writer = writerFor(message.topic);
  if (!writer.ok()) {
    increment(message.counters->sendDropped);
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << writer.status();
    return;
  }
  TopicWriter& entry = **writer;
  if (message.payload.size() > static_cast<size_t>(UINT32_MAX)) {
    increment(message.counters->sendDropped);
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: a message of " << message.payload.size() << " bytes on '" << message.topic
                                              << "' is larger than a DDS sequence holds";
    return;
  }
  robot_ipc_Envelope envelope{};
  envelope.protocol_version = kWireProtocolVersion;
  envelope.sequence_number = entry.lastSequence + 1;
  // dds_write() serializes the sample and changes nothing in it.
  envelope.type_name = message.typeName.data();
  envelope.payload._buffer = reinterpret_cast<uint8_t*>(message.payload.data());
  envelope.payload._length = static_cast<uint32_t>(message.payload.size());
  envelope.payload._maximum = envelope.payload._length;
  envelope.payload._release = false;
  const dds_return_t written = dds_write_ts(entry.writer, &envelope, nextSourceTimestamp());
  if (written < 0) {
    increment(message.counters->sendDropped);
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds) << "robot_ipc: writing on '" << message.topic << "' failed: " << dds_strretcode(written);
    return;
  }
  entry.lastSequence = envelope.sequence_number;
  increment(message.counters->written);
}

void Bus::Impl::drain() {
  // Empty unless the previous drain was cut short by an exception; its loans are returned below as usual.
  pending_.clear();
  size_t budget = options_.maxMessagesPerDrain;
  for (std::pair<const std::string, std::unique_ptr<Subscription>>& entry : subscriptions_) {
    if (budget == 0) {
      break;
    }
    budget -= take(*entry.second, budget);
  }
  // In the order the samples were written: each bus's source timestamps increase strictly across its topics.
  std::stable_sort(pending_.begin(), pending_.end(), [](const PendingMessage& first, const PendingMessage& second) {
    return first.sourceTimestamp < second.sourceTimestamp;
  });
  coalesceLatest();
  for (PendingMessage& message : pending_) {
    if (!message.superseded) {
      dispatch(*message.subscription, *message.envelope);
    }
  }
  pending_.clear();
  for (std::pair<const std::string, std::unique_ptr<Subscription>>& entry : subscriptions_) {
    Subscription& subscription = *entry.second;
    if (subscription.taken > 0) {
      const dds_return_t returned =
          dds_return_loan(subscription.reader, subscription.samples.data(), static_cast<int32_t>(subscription.taken));
      if (returned < 0) {
        LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds)
            << "robot_ipc: returning the samples of '" << subscription.topic << "' failed: " << dds_strretcode(returned);
      }
      subscription.taken = 0;
    }
  }
}

size_t Bus::Impl::take(Subscription& subscription, size_t budget) {
  const size_t capacity = std::min(subscription.samples.size(), budget);
  // Null pointers ask dds_take() to lend the samples of the reader's cache instead of copying them.
  std::fill(subscription.samples.begin(), subscription.samples.end(), nullptr);
  const dds_return_t taken =
      dds_take(subscription.reader, subscription.samples.data(), subscription.infos.data(), capacity, static_cast<uint32_t>(capacity));
  if (taken < 0) {
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "robot_ipc: taking the samples of '" << subscription.topic
                                              << "' failed: " << dds_strretcode(taken);
    return 0;
  }
  subscription.taken = static_cast<size_t>(taken);
  for (size_t index = 0; index < subscription.taken; ++index) {
    const dds_sample_info_t& info = subscription.infos[index];
    // A writer's disposal or unregistration carries no sample.
    if (!info.valid_data || subscription.samples[index] == nullptr) {
      continue;
    }
    const robot_ipc_Envelope& envelope = *static_cast<const robot_ipc_Envelope*>(subscription.samples[index]);
    increment(subscription.counters->received);
    trackSequence(subscription, info.publication_handle, envelope.sequence_number);
    if (!accepts(subscription, envelope)) {
      increment(subscription.counters->rejected);
      continue;
    }
    pending_.push_back(PendingMessage{
        .subscription = &subscription, .sourceTimestamp = info.source_timestamp, .envelope = &envelope, .superseded = false});
  }
  return subscription.taken;
}

void Bus::Impl::trackSequence(Subscription& subscription, dds_instance_handle_t writer, uint64_t sequence) {
  const std::pair<absl::flat_hash_map<dds_instance_handle_t, uint64_t>::iterator, bool> inserted =
      subscription.lastSequenceByWriter.try_emplace(writer, sequence);
  if (inserted.second) {
    return;
  }
  uint64_t& last = inserted.first->second;
  if (sequence > last + 1) {
    increment(subscription.counters->missed, sequence - last - 1);
  }
  last = std::max(last, sequence);
}

bool Bus::Impl::accepts(Subscription& subscription, const robot_ipc_Envelope& envelope) const {
  if (envelope.protocol_version != kWireProtocolVersion) {
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
        << "robot_ipc: a sample on '" << subscription.topic << "' speaks wire protocol " << envelope.protocol_version
        << ", but this bus speaks " << kWireProtocolVersion << "; rebuild both from the same checkout";
    return false;
  }
  const absl::string_view type = envelope.type_name == nullptr ? absl::string_view() : absl::string_view(envelope.type_name);
  if (!subscription.expectedTypeName.empty() && type != subscription.expectedTypeName) {
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds) << "robot_ipc: '" << subscription.topic << "' carries a " << type
                                                << ", but its subscription takes a " << subscription.expectedTypeName;
    return false;
  }
  return true;
}

void Bus::Impl::coalesceLatest() {
  for (std::pair<const std::string, std::unique_ptr<Subscription>>& entry : subscriptions_) {
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

void Bus::Impl::dispatch(Subscription& subscription, const robot_ipc_Envelope& envelope) {
  TopicCounters& counters = *subscription.counters;
  const absl::string_view type = envelope.type_name == nullptr ? absl::string_view() : absl::string_view(envelope.type_name);
  const absl::string_view payload(reinterpret_cast<const char*>(envelope.payload._buffer), envelope.payload._length);
  if (subscription.message != nullptr && (payload.size() > static_cast<size_t>(INT_MAX) ||
                                          !subscription.message->ParseFromArray(payload.data(), static_cast<int>(payload.size())))) {
    increment(counters.rejected);
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds) << "robot_ipc: a sample on '" << subscription.topic << "' does not parse as a " << type;
    return;
  }
  increment(counters.delivered);
  try {  // NOLINT(exceptions): a handler's exception is counted and logged, not let end the IO thread
    if (subscription.message != nullptr) {
      subscription.messageHandler(*subscription.message);
    } else {
      subscription.rawHandler(type, payload);
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

void Bus::Impl::refreshStatusIfDue(Clock::time_point now) {
  if (now < nextStatusRefresh_) {
    return;
  }
  nextStatusRefresh_ = now + std::chrono::duration_cast<Clock::duration>(absl::ToChronoNanoseconds(options_.ioPollPeriod));
  for (std::pair<const std::string, std::unique_ptr<Subscription>>& entry : subscriptions_) {
    Subscription& subscription = *entry.second;
    TopicCounters& counters = *subscription.counters;
    dds_sample_lost_status_t lost{};
    if (dds_get_sample_lost_status(subscription.reader, &lost) == DDS_RETCODE_OK) {
      counters.sampleLost.store(lost.total_count, std::memory_order_relaxed);
    }
    dds_sample_rejected_status_t rejected{};
    if (dds_get_sample_rejected_status(subscription.reader, &rejected) == DDS_RETCODE_OK) {
      counters.sampleRejected.store(rejected.total_count, std::memory_order_relaxed);
    }
    dds_requested_incompatible_qos_status_t incompatible{};
    if (dds_get_requested_incompatible_qos_status(subscription.reader, &incompatible) == DDS_RETCODE_OK) {
      counters.requestedIncompatibleQos.store(incompatible.total_count, std::memory_order_relaxed);
    }
    dds_subscription_matched_status_t matched{};
    if (dds_get_subscription_matched_status(subscription.reader, &matched) == DDS_RETCODE_OK) {
      counters.matchedWriters.store(matched.current_count, std::memory_order_relaxed);
    }
  }
  for (std::pair<const std::string, TopicWriter>& entry : writers_) {
    TopicWriter& writer = entry.second;
    dds_offered_incompatible_qos_status_t incompatible{};
    if (dds_get_offered_incompatible_qos_status(writer.writer, &incompatible) == DDS_RETCODE_OK) {
      writer.counters->offeredIncompatibleQos.store(incompatible.total_count, std::memory_order_relaxed);
    }
    dds_publication_matched_status_t matched{};
    if (dds_get_publication_matched_status(writer.writer, &matched) == DDS_RETCODE_OK) {
      writer.counters->matchedReaders.store(matched.current_count, std::memory_order_relaxed);
    }
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
    // Against the time the callback returned, not `now`: a callback that overran its period must not be called again
    // at once for the deadline it overran. A missed deadline is skipped, to the first one still ahead on the grid of
    // start(), instead of bursting through the missed calls.
    const Clock::time_point after = Clock::now();
    periodic.nextDeadline += periodic.period;
    if (periodic.nextDeadline <= after) {
      periodic.nextDeadline += ((after - periodic.nextDeadline) / periodic.period + 1) * periodic.period;
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

absl::Status Bus::publishRaw(absl::string_view topic, absl::string_view typeName, absl::string_view payload) {
  return impl_->publishRaw(topic, typeName, payload);
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
