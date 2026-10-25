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

// The bus over Cyclone DDS: topics kept apart, kLatest against kAll, the order
// of a drain across topics, publishing from several threads, processes that
// start in either order or restart, a prompt stop(), the publish queue's drop
// counting, the counters' invariants, rejection of malformed samples and of
// another wire protocol version, handlers that throw, and a policy-sized
// sample. Every test process has a DDS domain and tag of its own
// (isolatedTestNetwork()), so tests that run at once never see each other's
// samples.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "dds/dds.h"
#include "gtest/gtest.h"

#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_ipc/NetworkConfig.h"
#include "robot_ipc/NodeEndpoint.h"
#include "robot_ipc/QosProfile.h"
#include "robot_ipc/TopicStatistics.h"
#include "robot_ipc/WireProtocol.h"
#include "robot_ipc_test/test_event.pb.h"
#include "robot_ipc_test/test_sample.pb.h"
#include "robot_runtime/robot_ipc/idl/Envelope.h"
#include "robot_runtime/robot_ipc/test/TestNetwork.h"

namespace robot::ipc {
namespace {

using robot_ipc_test::TestEvent;
using robot_ipc_test::TestSample;

constexpr absl::Duration kTimeout = absl::Seconds(20);
constexpr char kPublisherNode[] = "publisher";
constexpr char kSubscriberNode[] = "subscriber";
constexpr char kProbeTopic[] = "test/probe";

// Polls `condition` until it holds or `timeout` passes.
bool waitFor(const std::function<bool()>& condition,
             absl::Duration timeout = kTimeout) {
  const absl::Time deadline = absl::Now() + timeout;
  while (absl::Now() < deadline) {
    if (condition()) {
      return true;
    }
    absl::SleepFor(absl::Milliseconds(1));
  }
  return condition();
}

NetworkConfig testNetwork() {
  return isolatedTestNetwork({kPublisherNode, kSubscriberNode});
}

std::unique_ptr<Bus> createBus(BusOptions options) {
  absl::StatusOr<std::unique_ptr<Bus>> bus = Bus::Create(std::move(options));
  EXPECT_TRUE(bus.ok()) << bus.status();
  return bus.ok() ? *std::move(bus) : nullptr;
}

// A bus publishing as kPublisherNode of the test network.
std::unique_ptr<Bus> createPublisher(BusOptions options = BusOptions()) {
  options.nodeName = kPublisherNode;
  if (options.network.nodes.empty()) {
    options.network = testNetwork();
  }
  return createBus(std::move(options));
}

// A bus of the test network that only subscribes.
std::unique_ptr<Bus> createSubscriber(BusOptions options = BusOptions()) {
  options.nodeName.clear();
  if (options.network.nodes.empty()) {
    options.network = testNetwork();
  }
  return createBus(std::move(options));
}

absl::Status subscribeProbe(Bus& bus) {
  return bus.subscribeRaw(
      kProbeTopic, Delivery::kLatest,
      [](absl::string_view /*typeName*/, absl::string_view /*payload*/) {});
}

// Publishes on kProbeTopic until the subscriber has received more probes than
// before: DDS discovery has matched the publisher's writers with the
// subscriber's readers.
bool waitUntilConnected(Bus& publisher, const Bus& subscriber) {
  const uint64_t before = subscriber.topicStatistics(kProbeTopic).received;
  TestEvent probe;
  probe.set_name("probe");
  return waitFor([&]() {
    EXPECT_TRUE(publisher.publish(kProbeTopic, probe).ok());
    absl::SleepFor(absl::Milliseconds(5));
    return subscriber.topicStatistics(kProbeTopic).received > before;
  });
}

TestSample makeSample(uint64_t sequence, uint32_t publisherIndex = 0) {
  TestSample sample;
  sample.set_sequence(sequence);
  sample.set_publisher(publisherIndex);
  sample.set_text("sample");
  sample.add_values(static_cast<double>(sequence));
  return sample;
}

// Collects what a handler receives, on the IO thread, for the test thread to
// read.
class Collector {
 public:
  std::function<void(const TestSample&)> handler() {
    return [this](const TestSample& sample) {
      absl::MutexLock lock(mutex_);
      samples_.push_back(sample);
    };
  }

  std::vector<TestSample> samples() const {
    absl::MutexLock lock(mutex_);
    return samples_;
  }

  size_t size() const {
    absl::MutexLock lock(mutex_);
    return samples_.size();
  }

  bool contains(uint64_t sequence) const {
    absl::MutexLock lock(mutex_);
    for (const TestSample& sample : samples_) {
      if (sample.sequence() == sequence) {
        return true;
      }
    }
    return false;
  }

 private:
  mutable absl::Mutex mutex_;
  std::vector<TestSample> samples_ ABSL_GUARDED_BY(mutex_);
};

// The counters' invariant once a drain has dispatched.
void expectReceivedIsAccountedFor(const TopicStatistics& statistics) {
  EXPECT_EQ(statistics.received,
            statistics.delivered + statistics.superseded + statistics.rejected);
}

// A raw DDS participant of the test network's domain (which the process's buses
// created), to write and read what a Bus would not. Owns its entities.
class RawParticipant {
 public:
  RawParticipant()
      : participant_(dds_create_participant(
            testNetwork().domainId, /*qos=*/nullptr, /*listener=*/nullptr)) {}
  ~RawParticipant() {
    if (participant_ > 0) dds_delete(participant_);
  }
  RawParticipant(const RawParticipant&) = delete;
  RawParticipant& operator=(const RawParticipant&) = delete;

  bool ok() const { return participant_ > 0; }

  dds_entity_t writer(const std::string& topic) {
    dds_qos_t* absl_nonnull qos = dds_create_qos();
    dds_qset_reliability(qos, DDS_RELIABILITY_RELIABLE, DDS_SECS(1));
    dds_qset_history(qos, DDS_HISTORY_KEEP_LAST, kReliableEventsHistoryDepth);
    const dds_entity_t writer = dds_create_writer(
        participant_, topicEntity(topic), qos, /*listener=*/nullptr);
    dds_delete_qos(qos);
    return writer;
  }

  dds_entity_t reader(const std::string& topic) {
    dds_qos_t* absl_nonnull qos = dds_create_qos();
    dds_qset_reliability(qos, DDS_RELIABILITY_RELIABLE, DDS_SECS(1));
    dds_qset_history(qos, DDS_HISTORY_KEEP_LAST, kReliableEventsHistoryDepth);
    const dds_entity_t reader = dds_create_reader(
        participant_, topicEntity(topic), qos, /*listener=*/nullptr);
    dds_delete_qos(qos);
    return reader;
  }

 private:
  dds_entity_t topicEntity(const std::string& topic) const {
    return dds_create_topic(participant_, &robot_ipc_Envelope_desc,
                            topic.c_str(), /*qos=*/nullptr,
                            /*listener=*/nullptr);
  }

  const dds_entity_t participant_;
};

// Writes one raw envelope.
dds_return_t writeEnvelope(dds_entity_t writer,
                           uint32_t protocolVersion,
                           uint64_t sequence,
                           std::string typeName,
                           std::string payload) {
  robot_ipc_Envelope envelope{};
  envelope.protocol_version = protocolVersion;
  envelope.sequence_number = sequence;
  envelope.type_name = typeName.data();
  envelope.payload._buffer = reinterpret_cast<uint8_t*>(payload.data());
  envelope.payload._length = static_cast<uint32_t>(payload.size());
  envelope.payload._maximum = envelope.payload._length;
  envelope.payload._release = false;
  return dds_write(writer, &envelope);
}

// ---------------------------------------------------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------------------------------------------------

TEST(BusCreateTest, RejectsANodeOutsideTheNetworkAndListsTheNodes) {
  BusOptions options;
  options.nodeName = "ghost";
  options.network = localhostNetworkConfig();
  const absl::StatusOr<std::unique_ptr<Bus>> bus =
      Bus::Create(std::move(options));
  EXPECT_EQ(bus.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(bus.status().message(), "ghost"))
      << bus.status();
  EXPECT_TRUE(absl::StrContains(bus.status().message(), "robot"))
      << bus.status();
}

TEST(BusCreateTest, RejectsAnInvalidNetworkAndOptions) {
  EXPECT_EQ(Bus::Create(BusOptions()).status().code(),
            absl::StatusCode::kInvalidArgument);

  BusOptions noQueue;
  noQueue.network = testNetwork();
  noQueue.publishQueueCapacity = 0;
  EXPECT_EQ(Bus::Create(std::move(noQueue)).status().code(),
            absl::StatusCode::kInvalidArgument);

  BusOptions noHistory;
  noHistory.network = testNetwork();
  noHistory.latestDeliveryHistoryDepth = 0;
  EXPECT_EQ(Bus::Create(std::move(noHistory)).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(BusCreateTest, ANodeOnAnotherMachineIsUnavailable) {
  // 192.0.2.1 (TEST-NET-1, RFC 5737) is no address of this machine: the node's
  // process runs elsewhere.
  BusOptions options;
  options.nodeName = kPublisherNode;
  options.network = testNetwork();
  options.network.nodes[0].host = "192.0.2.1";
  const absl::StatusOr<std::unique_ptr<Bus>> bus =
      Bus::Create(std::move(options));
  EXPECT_EQ(bus.status().code(), absl::StatusCode::kUnavailable)
      << bus.status();
  EXPECT_TRUE(absl::StrContains(bus.status().message(), "192.0.2.1"))
      << bus.status();
}

TEST(BusCreateTest,
     TheBusesOfAProcessShareADomainAndRefuseAnotherConfigurationOfIt) {
  std::unique_ptr<Bus> first = createPublisher();
  ASSERT_NE(first, nullptr);
  std::unique_ptr<Bus> second = createSubscriber();
  ASSERT_NE(second, nullptr);

  BusOptions otherTag;
  otherTag.network = testNetwork();
  otherTag.network.domainTag = "another_tag";
  EXPECT_EQ(Bus::Create(otherTag).status().code(),
            absl::StatusCode::kFailedPrecondition);
  first.reset();
  EXPECT_EQ(Bus::Create(otherTag).status().code(),
            absl::StatusCode::kFailedPrecondition)
      << "the subscriber still uses the domain";

  // Once its last bus is gone, the domain may be configured anew.
  second.reset();
  EXPECT_NE(createBus(otherTag), nullptr);
}

TEST(BusCreateTest, ASubscribeOnlyBusReportsItsDomainAndCannotPublish) {
  const std::unique_ptr<Bus> bus = createSubscriber();
  ASSERT_NE(bus, nullptr);
  EXPECT_TRUE(bus->nodeName().empty());
  EXPECT_EQ(bus->domainId(), testNetwork().domainId);
  EXPECT_EQ(bus->discoveryPeers(),
            std::vector<std::string>{std::string(kLoopbackHost)});
  EXPECT_TRUE(absl::StrContains(
      bus->domainConfig(),
      absl::StrCat("<Tag>", testNetwork().domainTag, "</Tag>")))
      << bus->domainConfig();
  EXPECT_EQ(bus->publish("test/topic", makeSample(/*sequence=*/0)).code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(BusCreateTest, TheEnvironmentDoesNotConfigureTheBus) {
  // BUILD.bazel sets CYCLONEDDS_URI to a configuration whose interface does not
  // exist; every bus of this test passes its own configuration to
  // dds_create_domain() and never reads it.
  const char* absl_nullable uri = std::getenv("CYCLONEDDS_URI");
  ASSERT_NE(uri, nullptr);
  EXPECT_TRUE(absl::StrContains(uri, "no_such_interface"));
  const std::unique_ptr<Bus> bus = createPublisher();
  ASSERT_NE(bus, nullptr);
  ASSERT_TRUE(subscribeProbe(*bus).ok());
  ASSERT_TRUE(bus->start().ok());
  EXPECT_TRUE(waitUntilConnected(*bus, *bus));
}

TEST(BusCreateTest, ANodeCreatesTheWritersOfItsTopicsUpFront) {
  // Before the publisher writes anything, the subscriber's reader has matched
  // its writer: a VOLATILE reader gets only what is written after it matched,
  // so the first sample of a one-off event must not race discovery.
  NetworkConfig network = testNetwork();
  network.topics = {TopicQos{.topic = "test/declared",
                             .profile = QosProfile::kReliableEvents,
                             .publishers = {kPublisherNode}}};
  BusOptions subscriberOptions;
  subscriberOptions.network = network;
  const std::unique_ptr<Bus> subscriber = createSubscriber(subscriberOptions);
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscriber
                  ->subscribe<TestSample>("test/declared", Delivery::kAll,
                                          collector.handler())
                  .ok());
  ASSERT_TRUE(subscriber->start().ok());
  BusOptions publisherOptions;
  publisherOptions.network = network;
  const std::unique_ptr<Bus> publisher = createPublisher(publisherOptions);
  ASSERT_NE(publisher, nullptr);
  ASSERT_TRUE(publisher->start().ok());

  EXPECT_TRUE(waitFor([&]() {
    return subscriber->topicStatistics("test/declared").matchedWriters == 1u;
  }));
  EXPECT_TRUE(waitFor([&]() {
    return publisher->topicStatistics("test/declared").matchedReaders == 1u;
  }));
  EXPECT_EQ(publisher->topicStatistics("test/declared").written, 0u);
  ASSERT_TRUE(
      publisher->publish("test/declared", makeSample(/*sequence=*/1)).ok());
  EXPECT_TRUE(waitFor([&]() { return collector.contains(1); }));
}

// ---------------------------------------------------------------------------------------------------------------------
// Configuration and lifecycle
// ---------------------------------------------------------------------------------------------------------------------

TEST(BusLifecycleTest, ConfigurationIsRefusedWhileRunningAndAcceptedAfterStop) {
  const std::unique_ptr<Bus> bus = createPublisher();
  ASSERT_NE(bus, nullptr);
  Collector collector;
  EXPECT_EQ(bus->subscribe<TestSample>(/*topic=*/"", Delivery::kAll,
                                       collector.handler())
                .code(),
            absl::StatusCode::kInvalidArgument);
  // DDS topic names are letters, digits, '_' and '/', not starting with a
  // digit.
  EXPECT_EQ(bus->subscribe<TestSample>("test/a-b", Delivery::kAll,
                                       collector.handler())
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      bus->subscribe<TestSample>("1test", Delivery::kAll, collector.handler())
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(bus->publish("test.a", makeSample(/*sequence=*/0)).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      bus->subscribe<TestSample>("test/a", Delivery::kAll, /*handler=*/nullptr)
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(bus->addPeriodicCallback(absl::ZeroDuration(), []() {}).code(),
            absl::StatusCode::kInvalidArgument);
  ASSERT_TRUE(
      bus->subscribe<TestSample>("test/a", Delivery::kAll, collector.handler())
          .ok());
  EXPECT_EQ(bus->subscribe<TestSample>("test/a", Delivery::kLatest,
                                       collector.handler())
                .code(),
            absl::StatusCode::kAlreadyExists);

  EXPECT_FALSE(bus->isRunning());
  ASSERT_TRUE(bus->start().ok());
  ASSERT_TRUE(bus->start().ok());
  EXPECT_TRUE(bus->isRunning());
  EXPECT_EQ(
      bus->subscribe<TestSample>("test/b", Delivery::kAll, collector.handler())
          .code(),
      absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(bus->addPeriodicCallback(absl::Milliseconds(1), []() {}).code(),
            absl::StatusCode::kFailedPrecondition);

  bus->stop();
  bus->stop();
  EXPECT_FALSE(bus->isRunning());
  EXPECT_TRUE(
      bus->subscribe<TestSample>("test/b", Delivery::kAll, collector.handler())
          .ok());
}

TEST(BusLifecycleTest, StopIsPromptWhateverThePollPeriod) {
  BusOptions options;
  options.ioPollPeriod = absl::Seconds(60);
  std::unique_ptr<Bus> bus = createPublisher(std::move(options));
  ASSERT_NE(bus, nullptr);
  ASSERT_TRUE(bus->start().ok());
  absl::SleepFor(absl::Milliseconds(20));

  const absl::Time stopStart = absl::Now();
  bus->stop();
  EXPECT_LT(absl::Now() - stopStart, absl::Seconds(1));

  ASSERT_TRUE(bus->start().ok());
  absl::SleepFor(absl::Milliseconds(20));
  const absl::Time destroyStart = absl::Now();
  bus.reset();
  EXPECT_LT(absl::Now() - destroyStart, absl::Seconds(1));
}

TEST(BusLifecycleTest, StopFromAHandlerEndsTheLoopAndALaterStopJoins) {
  const std::unique_ptr<Bus> bus = createPublisher();
  ASSERT_NE(bus, nullptr);
  ASSERT_TRUE(subscribeProbe(*bus).ok());
  Bus* absl_nonnull const self = bus.get();
  std::atomic<int> calls = 0;
  ASSERT_TRUE(
      bus->subscribe<TestSample>("test/stop", Delivery::kAll,
                                 [self, &calls](const TestSample& /*sample*/) {
                                   ++calls;
                                   self->stop();
                                 })
          .ok());
  ASSERT_TRUE(bus->start().ok());
  ASSERT_TRUE(waitUntilConnected(*bus, *bus));
  ASSERT_TRUE(bus->publish("test/stop", makeSample(/*sequence=*/1)).ok());
  ASSERT_TRUE(waitFor([&]() { return calls.load() > 0; }));
  EXPECT_TRUE(waitFor([&]() { return !bus->isRunning(); }));
  bus->stop();
  ASSERT_TRUE(bus->start().ok());
  EXPECT_TRUE(bus->isRunning());
}

TEST(BusLifecycleTest,
     AHandlerReadingOrConfiguringTheBusDuringAStopDoesNotDeadlockIt) {
  // stop() holds the lifecycle lock while it joins the IO thread: a handler
  // that waited for that lock would never return, and stop() never with it.
  const std::unique_ptr<Bus> bus = createPublisher();
  ASSERT_NE(bus, nullptr);
  Bus* absl_nonnull const self = bus.get();
  absl::Notification entered;
  absl::Notification go;
  absl::Mutex mutex;
  std::vector<std::string> peers;
  absl::Status subscribed;
  bool ran = false;
  ASSERT_TRUE(
      bus->addPeriodicCallback(
             absl::Milliseconds(10),
             [&ran, &entered, &go, &mutex, &peers, &subscribed, self]() {
               if (ran) return;
               ran = true;
               entered.Notify();
               go.WaitForNotification();
               absl::MutexLock lock(mutex);
               peers = self->discoveryPeers();
               subscribed =
                   self->subscribeRaw("test/late", Delivery::kAll,
                                      [](absl::string_view /*typeName*/,
                                         absl::string_view /*payload*/) {});
             })
          .ok());
  ASSERT_TRUE(bus->start().ok());
  entered.WaitForNotification();
  absl::Notification stopped;
  std::thread stopper([&]() {
    bus->stop();
    stopped.Notify();
  });
  // stop() asks the loop to end only once it holds the lifecycle lock.
  ASSERT_TRUE(waitFor([&]() { return !bus->isRunning(); }));
  go.Notify();
  EXPECT_TRUE(stopped.WaitForNotificationWithTimeout(absl::Seconds(2)));
  if (!stopped.HasBeenNotified()) {
    // Leaves the deadlocked thread behind rather than hang the test binary on
    // its join.
    stopper.detach();
    FAIL() << "stop() deadlocked with a handler";
  }
  stopper.join();
  absl::MutexLock lock(mutex);
  EXPECT_EQ(peers.size(), 1u);
  EXPECT_EQ(subscribed.code(), absl::StatusCode::kFailedPrecondition);
}

TEST(BusLifecycleTest,
     APeriodicCallbackThatOverranItsPeriodIsNotCalledAgainAtOnce) {
  const std::unique_ptr<Bus> bus = createPublisher();
  ASSERT_NE(bus, nullptr);
  constexpr absl::Duration kPeriod = absl::Milliseconds(20);
  constexpr absl::Duration kOverrun = absl::Milliseconds(70);
  absl::Mutex mutex;
  std::vector<absl::Time> starts;
  std::vector<absl::Time> ends;
  ASSERT_TRUE(bus->addPeriodicCallback(kPeriod,
                                       [&mutex, &starts, &ends, kOverrun]() {
                                         const absl::Time start = absl::Now();
                                         bool first = false;
                                         {
                                           absl::MutexLock lock(mutex);
                                           first = starts.empty();
                                           starts.push_back(start);
                                         }
                                         if (first) absl::SleepFor(kOverrun);
                                         absl::MutexLock lock(mutex);
                                         ends.push_back(absl::Now());
                                       })
                  .ok());
  ASSERT_TRUE(bus->start().ok());
  ASSERT_TRUE(waitFor([&]() {
    absl::MutexLock lock(mutex);
    return starts.size() >= 3;
  }));
  bus->stop();
  absl::MutexLock lock(mutex);
  // The deadline it overran is skipped: the next call waits for the next
  // deadline on the grid, about 10 ms later.
  EXPECT_GE(starts[1] - ends[0], absl::Milliseconds(5));
  EXPECT_LT(starts[1] - ends[0], kPeriod + absl::Milliseconds(15));
}

// ---------------------------------------------------------------------------------------------------------------------
// Delivery
// ---------------------------------------------------------------------------------------------------------------------

TEST(BusDeliveryTest,
     AProcessReceivesItsOwnTopicsAndHandlersPublishFromTheIoThread) {
  const std::unique_ptr<Bus> bus = createPublisher();
  ASSERT_NE(bus, nullptr);
  Bus* absl_nonnull const self = bus.get();
  std::atomic<bool> handlersOnIoThread = true;
  Collector pongs;
  ASSERT_TRUE(subscribeProbe(*bus).ok());
  ASSERT_TRUE(
      bus->subscribe<TestSample>(
             "test/ping", Delivery::kAll,
             [self, &handlersOnIoThread](const TestSample& ping) {
               handlersOnIoThread = handlersOnIoThread && self->isIoThread();
               TestSample pong = ping;
               pong.set_origin("pong");
               EXPECT_TRUE(self->publishFromIoThread("test/pong", pong).ok());
             })
          .ok());
  ASSERT_TRUE(
      bus->subscribe<TestSample>("test/pong", Delivery::kAll, pongs.handler())
          .ok());
  ASSERT_TRUE(bus->start().ok());
  ASSERT_TRUE(waitUntilConnected(*bus, *bus));

  EXPECT_EQ(
      bus->publishFromIoThread("test/pong", makeSample(/*sequence=*/0)).code(),
      absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(bus->publish("test/ping", makeSample(/*sequence=*/7)).ok());
  ASSERT_TRUE(waitFor([&]() { return pongs.size() == 1; }));
  EXPECT_EQ(pongs.samples()[0].sequence(), 7u);
  EXPECT_EQ(pongs.samples()[0].origin(), "pong");
  EXPECT_TRUE(handlersOnIoThread.load());
  EXPECT_FALSE(bus->isIoThread());
}

TEST(BusDeliveryTest, EveryTopicIsADdsTopicOfItsOwn) {
  const std::unique_ptr<Bus> publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber();
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(
      subscriber
          ->subscribe<TestSample>("test/a", Delivery::kAll, collector.handler())
          .ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));

  ASSERT_TRUE(publisher->publish("test/ab", makeSample(/*sequence=*/100)).ok());
  ASSERT_TRUE(publisher->publish("test/a", makeSample(/*sequence=*/1)).ok());
  ASSERT_TRUE(
      publisher->publish("test/a/b", makeSample(/*sequence=*/101)).ok());
  ASSERT_TRUE(publisher->publish("test/a", makeSample(/*sequence=*/2)).ok());
  ASSERT_TRUE(waitFor([&]() { return collector.contains(2); }));

  const std::vector<TestSample> samples = collector.samples();
  ASSERT_EQ(samples.size(), 2u);
  EXPECT_EQ(samples[0].sequence(), 1u);
  EXPECT_EQ(samples[1].sequence(), 2u);
  EXPECT_EQ(subscriber->topicStatistics("test/a").received, 2u);
  EXPECT_EQ(subscriber->topicStatistics("test/ab").received, 0u);
  EXPECT_EQ(publisher->topicStatistics("test/ab").written, 1u);
}

TEST(BusDeliveryTest,
     LatestHandsOverOnlyTheNewestOfABurstAndAllHandsOverEveryMessage) {
  constexpr uint64_t kBurst = 500;
  BusOptions options;
  options.publishQueueCapacity = 4 * kBurst;
  options.allDeliveryHistoryDepth = 2 * kBurst;
  const std::unique_ptr<Bus> publisher = createPublisher(options);
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber(options);
  ASSERT_NE(subscriber, nullptr);
  Collector latest;
  Collector all;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber
                  ->subscribe<TestSample>("test/latest", Delivery::kLatest,
                                          latest.handler())
                  .ok());
  ASSERT_TRUE(
      subscriber
          ->subscribe<TestSample>("test/all", Delivery::kAll, all.handler())
          .ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));
  // A first sample of the latest topic, so that the gaps after it count as
  // missed.
  ASSERT_TRUE(
      publisher->publish("test/latest", makeSample(/*sequence=*/0)).ok());
  ASSERT_TRUE(waitFor([&]() { return latest.contains(0); }));

  // With the subscriber's IO thread stopped, the burst piles up in its DDS
  // readers, so that the next drain holds several samples of each topic.
  subscriber->stop();
  for (uint64_t sequence = 1; sequence <= kBurst; ++sequence) {
    ASSERT_TRUE(publisher->publish("test/latest", makeSample(sequence)).ok());
    ASSERT_TRUE(publisher->publish("test/all", makeSample(sequence)).ok());
  }
  ASSERT_TRUE(waitFor([&]() {
    return publisher->topicStatistics("test/all").written == kBurst;
  }));
  absl::SleepFor(absl::Milliseconds(300));
  ASSERT_TRUE(subscriber->start().ok());

  ASSERT_TRUE(waitFor([&]() { return all.size() == kBurst; }));
  ASSERT_TRUE(waitFor([&]() { return latest.contains(kBurst); }));

  // kAll: every message, in order.
  const std::vector<TestSample> allSamples = all.samples();
  for (uint64_t index = 0; index < kBurst; ++index) {
    EXPECT_EQ(allSamples[index].sequence(), index + 1);
  }
  const TopicStatistics allStatistics = subscriber->topicStatistics("test/all");
  EXPECT_EQ(allStatistics.received, kBurst);
  EXPECT_EQ(allStatistics.delivered, kBurst);
  EXPECT_EQ(allStatistics.superseded, 0u);
  EXPECT_EQ(allStatistics.missed, 0u);

  // kLatest: fewer messages than were sent, always newer than the one before,
  // ending with the newest. Every sample written is accounted for: taken, or a
  // gap of the writer's sequence numbers.
  const std::vector<TestSample> latestSamples = latest.samples();
  EXPECT_LT(latestSamples.size(), kBurst);
  for (size_t i = 1; i < latestSamples.size(); ++i) {
    EXPECT_GT(latestSamples[i].sequence(), latestSamples[i - 1].sequence());
  }
  EXPECT_EQ(latestSamples.back().sequence(), kBurst);
  const TopicStatistics latestStatistics =
      subscriber->topicStatistics("test/latest");
  EXPECT_EQ(latestStatistics.delivered, latestSamples.size());
  expectReceivedIsAccountedFor(latestStatistics);
  EXPECT_EQ(latestStatistics.rejected, 0u);
  EXPECT_EQ(latestStatistics.received + latestStatistics.missed,
            publisher->topicStatistics("test/latest").written);
  EXPECT_GT(latestStatistics.missed, 0u);
  EXPECT_LE(latestStatistics.received,
            1 + static_cast<uint64_t>(options.latestDeliveryHistoryDepth) +
                kBurst / 2)
      << "the reader keeps latestDeliveryHistoryDepth samples while the IO "
         "thread is stopped";
}

TEST(BusDeliveryTest,
     WrongTypedMessageDoesNotDisplaceValidMessageInLatestDelivery) {
  const std::unique_ptr<Bus> publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber();
  ASSERT_NE(subscriber, nullptr);
  Collector latest;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber
                  ->subscribe<TestSample>("test/latest", Delivery::kLatest,
                                          latest.handler())
                  .ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));

  subscriber->stop();
  ASSERT_TRUE(
      publisher->publish("test/latest", makeSample(/*sequence=*/42)).ok());
  TestEvent event;
  event.set_name("event");
  ASSERT_TRUE(publisher->publish("test/latest", event).ok());
  ASSERT_TRUE(waitFor([&]() {
    return publisher->topicStatistics("test/latest").written == 2u;
  }));
  absl::SleepFor(absl::Milliseconds(50));
  ASSERT_TRUE(subscriber->start().ok());

  ASSERT_TRUE(waitFor([&]() { return latest.size() == 1u; }));
  EXPECT_EQ(latest.samples()[0].sequence(), 42u);
  const TopicStatistics stats = subscriber->topicStatistics("test/latest");
  EXPECT_EQ(stats.received, 2u);
  EXPECT_EQ(stats.delivered, 1u);
  EXPECT_EQ(stats.rejected, 1u);
  EXPECT_EQ(stats.superseded, 0u);
}

TEST(BusDeliveryTest, PublishingFromSeveralThreadsKeepsTheOrderOfEachThread) {
  constexpr uint32_t kThreads = 4;
  constexpr uint64_t kPerThread = 250;
  BusOptions options;
  options.publishQueueCapacity = kThreads * kPerThread;
  options.allDeliveryHistoryDepth = 2 * kThreads * kPerThread;
  const std::unique_ptr<Bus> publisher = createPublisher(options);
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber(options);
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber
                  ->subscribe<TestSample>("test/threads", Delivery::kAll,
                                          collector.handler())
                  .ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));

  std::vector<std::thread> threads;
  for (uint32_t index = 0; index < kThreads; ++index) {
    threads.emplace_back([&publisher, index]() {
      for (uint64_t sequence = 0; sequence < kPerThread; ++sequence) {
        EXPECT_TRUE(
            publisher->publish("test/threads", makeSample(sequence, index))
                .ok());
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  ASSERT_TRUE(
      waitFor([&]() { return collector.size() == kThreads * kPerThread; }));
  std::vector<uint64_t> nextSequence(kThreads, 0);
  for (const TestSample& sample : collector.samples()) {
    ASSERT_LT(sample.publisher(), kThreads);
    EXPECT_EQ(sample.sequence(), nextSequence[sample.publisher()]);
    nextSequence[sample.publisher()] = sample.sequence() + 1;
  }
  const TopicStatistics statistics = publisher->topicStatistics("test/threads");
  EXPECT_EQ(statistics.written, kThreads * kPerThread);
  EXPECT_EQ(statistics.sendDropped, 0u);
}

TEST(BusDeliveryTest,
     RejectsMalformedSamplesAndAnotherProtocolVersionAndCarriesOn) {
  const std::unique_ptr<Bus> subscriber = createSubscriber();
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscriber
                  ->subscribe<TestSample>("test/typed", Delivery::kAll,
                                          collector.handler())
                  .ok());
  ASSERT_TRUE(subscriber->start().ok());
  // A raw DDS writer, to write what a Bus never would.
  RawParticipant raw;
  ASSERT_TRUE(raw.ok());
  const dds_entity_t writer = raw.writer("test/typed");
  ASSERT_GT(writer, 0);
  ASSERT_TRUE(waitFor([&]() {
    return subscriber->topicStatistics("test/typed").matchedWriters == 1u;
  }));

  const std::string sampleType(TestSample::descriptor()->full_name());
  const std::string eventType(TestEvent::descriptor()->full_name());
  TestEvent event;
  event.set_name("not a sample");
  uint64_t sequence = 0;
  ASSERT_EQ(writeEnvelope(writer, kWireProtocolVersion, ++sequence, eventType,
                          event.SerializeAsString()),
            DDS_RETCODE_OK);
  // A length-delimited field (text) that claims five bytes and carries two.
  ASSERT_EQ(writeEnvelope(writer, kWireProtocolVersion, ++sequence, sampleType,
                          std::string("\x12\x05"
                                      "ab")),
            DDS_RETCODE_OK);
  // The ZeroMQ framing's version, and a version from the future.
  ASSERT_EQ(
      writeEnvelope(writer, kWireProtocolVersion - 1, ++sequence, sampleType,
                    makeSample(/*sequence=*/1).SerializeAsString()),
      DDS_RETCODE_OK);
  ASSERT_EQ(
      writeEnvelope(writer, kWireProtocolVersion + 1, ++sequence, sampleType,
                    makeSample(/*sequence=*/2).SerializeAsString()),
      DDS_RETCODE_OK);
  ASSERT_EQ(writeEnvelope(writer, kWireProtocolVersion, ++sequence, sampleType,
                          makeSample(/*sequence=*/5).SerializeAsString()),
            DDS_RETCODE_OK);
  ASSERT_TRUE(waitFor([&]() { return collector.size() == 1; }));

  EXPECT_EQ(collector.samples()[0].sequence(), 5u);
  const TopicStatistics statistics = subscriber->topicStatistics("test/typed");
  EXPECT_EQ(statistics.received, 5u);
  EXPECT_EQ(statistics.rejected, 4u);
  EXPECT_EQ(statistics.delivered, 1u);
  EXPECT_EQ(statistics.missed, 0u);
  expectReceivedIsAccountedFor(statistics);
  EXPECT_TRUE(subscriber->isRunning());
}

TEST(BusDeliveryTest, TheBusWritesTheEnvelopeOfItsWireProtocol) {
  const std::unique_ptr<Bus> publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  RawParticipant raw;
  ASSERT_TRUE(raw.ok());
  const dds_entity_t reader = raw.reader("test/version");
  ASSERT_GT(reader, 0);
  ASSERT_TRUE(publisher->start().ok());

  std::vector<std::pair<uint64_t, TestSample>> received;
  ASSERT_TRUE(waitFor([&]() {
    EXPECT_TRUE(
        publisher->publish("test/version", makeSample(/*sequence=*/3)).ok());
    absl::SleepFor(absl::Milliseconds(5));
    void* absl_nullable sample = nullptr;
    dds_sample_info_t info{};
    while (dds_take(reader, &sample, &info, /*bufsz=*/1, /*maxs=*/1) == 1) {
      const robot_ipc_Envelope& envelope =
          *static_cast<const robot_ipc_Envelope*>(sample);
      EXPECT_EQ(envelope.protocol_version, kWireProtocolVersion);
      EXPECT_EQ(absl::string_view(envelope.type_name),
                TestSample::descriptor()->full_name());
      TestSample parsed;
      EXPECT_TRUE(
          parsed.ParseFromArray(envelope.payload._buffer,
                                static_cast<int>(envelope.payload._length)));
      received.emplace_back(envelope.sequence_number, parsed);
      dds_return_loan(reader, &sample, /*bufsz=*/1);
      sample = nullptr;
    }
    return received.size() >= 2;
  }));
  // The writer's sequence numbers count every sample it wrote, from 1; the
  // reader may have matched after the first.
  for (size_t index = 1; index < received.size(); ++index) {
    EXPECT_EQ(received[index].first, received[index - 1].first + 1);
  }
  EXPECT_EQ(received[0].second.sequence(), 3u);
}

// Records what the handlers of several topics receive, in the order they are
// called, as "topic:sequence".
class OrderLog {
 public:
  std::function<void(const TestSample&)> handler(const std::string& topic) {
    return [this, topic](const TestSample& sample) {
      absl::MutexLock lock(mutex_);
      entries_.push_back(absl::StrCat(topic, ":", sample.sequence()));
    };
  }

  std::vector<std::string> entries() const {
    absl::MutexLock lock(mutex_);
    return entries_;
  }

 private:
  mutable absl::Mutex mutex_;
  std::vector<std::string> entries_ ABSL_GUARDED_BY(mutex_);
};

TEST(BusDeliveryTest, OneDrainDispatchesInWriteOrderAcrossTopics) {
  const std::unique_ptr<Bus> publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber();
  ASSERT_NE(subscriber, nullptr);
  OrderLog log;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(
      subscriber
          ->subscribe<TestSample>("test/a", Delivery::kAll, log.handler("a"))
          .ok());
  ASSERT_TRUE(
      subscriber
          ->subscribe<TestSample>("test/b", Delivery::kLatest, log.handler("b"))
          .ok());
  ASSERT_TRUE(
      subscriber
          ->subscribe<TestSample>("test/c", Delivery::kAll, log.handler("c"))
          .ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));

  // With the subscriber's IO thread stopped, the samples pile up in its readers
  // and are taken in one drain.
  subscriber->stop();
  ASSERT_TRUE(publisher->publish("test/a", makeSample(/*sequence=*/0)).ok());
  ASSERT_TRUE(publisher->publish("test/b", makeSample(/*sequence=*/0)).ok());
  ASSERT_TRUE(publisher->publish("test/c", makeSample(/*sequence=*/0)).ok());
  ASSERT_TRUE(publisher->publish("test/b", makeSample(/*sequence=*/1)).ok());
  ASSERT_TRUE(publisher->publish("test/a", makeSample(/*sequence=*/1)).ok());
  ASSERT_TRUE(publisher->publish("test/c", makeSample(/*sequence=*/1)).ok());
  ASSERT_TRUE(waitFor(
      [&]() { return publisher->topicStatistics("test/c").written == 2u; }));
  absl::SleepFor(absl::Milliseconds(300));
  ASSERT_TRUE(subscriber->start().ok());

  ASSERT_TRUE(waitFor([&]() { return log.entries().size() == 5u; }));
  // b:0 is superseded by b:1, which is dispatched where it was written: after
  // c:0, before a:1.
  EXPECT_EQ(log.entries(),
            (std::vector<std::string>{"a:0", "c:0", "b:1", "a:1", "c:1"}));
  EXPECT_EQ(subscriber->topicStatistics("test/b").superseded, 1u);
  expectReceivedIsAccountedFor(subscriber->topicStatistics("test/b"));
}

TEST(BusDeliveryTest, AStalledSubscriberGetsTheNewestLatestValueNotABacklog) {
  // The DDS reader of a kLatest subscription keeps latestDeliveryHistoryDepth
  // samples, the newest: after a stall, the first samples the handler sees are
  // at most that many, whatever the stall's length. This replaces the ZeroMQ
  // high-water marks that bounded the same backlog per node.
  constexpr uint64_t kStallMessages = 2000;
  BusOptions options;
  options.publishQueueCapacity = kStallMessages + 1;
  const std::unique_ptr<Bus> publisher = createPublisher(options);
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber(options);
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber
                  ->subscribe<TestSample>("test/stall", Delivery::kLatest,
                                          collector.handler())
                  .ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));

  subscriber->stop();
  for (uint64_t sequence = 1; sequence <= kStallMessages; ++sequence) {
    ASSERT_TRUE(publisher->publish("test/stall", makeSample(sequence)).ok());
  }
  ASSERT_TRUE(waitFor([&]() {
    return publisher->topicStatistics("test/stall").written == kStallMessages;
  }));
  absl::SleepFor(absl::Milliseconds(100));
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitFor([&]() { return collector.contains(kStallMessages); }));
  EXPECT_EQ(collector.size(), 1u)
      << "one drain takes what the reader kept and hands over its newest";
  EXPECT_LE(subscriber->topicStatistics("test/stall").received,
            static_cast<uint64_t>(options.latestDeliveryHistoryDepth));
}

TEST(BusDeliveryTest, APolicySizedSampleArrivesWhole) {
  // An MPC policy is 0.5-0.75 MB; reliable_latest carries it whole.
  constexpr size_t kPolicyBytes = 768 * 1024;
  NetworkConfig network = testNetwork();
  network.topics = {TopicQos{.topic = "test/policy",
                             .profile = QosProfile::kReliableLatest,
                             .publishers = {kPublisherNode}}};
  BusOptions options;
  options.network = network;
  const std::unique_ptr<Bus> publisher = createPublisher(options);
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber(options);
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscriber
                  ->subscribe<TestSample>("test/policy", Delivery::kLatest,
                                          collector.handler())
                  .ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitFor([&]() {
    return subscriber->topicStatistics("test/policy").matchedWriters == 1u;
  }));

  TestSample policy = makeSample(/*sequence=*/9);
  std::string text(kPolicyBytes, 'x');
  for (size_t index = 0; index < text.size(); ++index) {
    text[index] = static_cast<char>('a' + index % 26);
  }
  policy.set_text(text);
  ASSERT_TRUE(publisher->publish("test/policy", policy).ok());
  ASSERT_TRUE(waitFor([&]() { return collector.size() == 1u; }));
  EXPECT_EQ(collector.samples()[0].text(), text);
}

TEST(BusDeliveryTest, AThrowingHandlerIsCountedAndTheBusCarriesOn) {
  const std::unique_ptr<Bus> publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber();
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  const std::function<void(const TestSample&)> collect = collector.handler();
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(
      subscriber
          ->subscribe<TestSample>("test/throws", Delivery::kAll,
                                  [&collect](const TestSample& sample) {
                                    if (sample.sequence() == 0) {
                                      throw std::runtime_error(
                                          "the first sample is unwelcome");
                                    }
                                    collect(sample);
                                  })
          .ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));

  ASSERT_TRUE(
      publisher->publish("test/throws", makeSample(/*sequence=*/0)).ok());
  ASSERT_TRUE(
      publisher->publish("test/throws", makeSample(/*sequence=*/1)).ok());
  ASSERT_TRUE(waitFor([&]() { return collector.size() == 1; }));
  const TopicStatistics statistics = subscriber->topicStatistics("test/throws");
  EXPECT_EQ(statistics.handlerErrors, 1u);
  EXPECT_EQ(statistics.delivered, 2u);
  EXPECT_TRUE(subscriber->isRunning());
}

TEST(BusDeliveryTest, PeriodicCallbacksRunOnTheIoThreadAndMayPublish) {
  const std::unique_ptr<Bus> publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber();
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber
                  ->subscribe<TestSample>("test/periodic", Delivery::kAll,
                                          collector.handler())
                  .ok());

  Bus* absl_nonnull const self = publisher.get();
  std::atomic<uint64_t> calls = 0;
  std::atomic<bool> onIoThread = true;
  ASSERT_TRUE(publisher
                  ->addPeriodicCallback(
                      absl::Milliseconds(2),
                      [self, &calls, &onIoThread]() {
                        onIoThread = onIoThread && self->isIoThread();
                        EXPECT_TRUE(self->publishFromIoThread(
                                            "test/periodic",
                                            makeSample(calls.fetch_add(1)))
                                        .ok());
                      })
                  .ok());
  ASSERT_TRUE(publisher
                  ->addPeriodicCallback(
                      absl::Milliseconds(5),
                      []() { throw std::runtime_error("a failing callback"); })
                  .ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));

  const uint64_t callsBefore = calls.load();
  ASSERT_TRUE(waitFor([&]() { return calls.load() >= callsBefore + 20; }));
  ASSERT_TRUE(waitFor([&]() { return collector.size() >= 10; }));
  EXPECT_TRUE(onIoThread.load());
  EXPECT_GT(publisher->periodicCallbackErrors(), 0u);
  EXPECT_TRUE(publisher->isRunning());
}

// ---------------------------------------------------------------------------------------------------------------------
// Start order, restarts and the publish queue
// ---------------------------------------------------------------------------------------------------------------------

TEST(BusConnectionTest,
     SubscriberStartedBeforeThePublisherReceivesOnceItAppears) {
  const std::unique_ptr<Bus> subscriber = createSubscriber();
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber
                  ->subscribe<TestSample>("test/late", Delivery::kLatest,
                                          collector.handler())
                  .ok());
  ASSERT_TRUE(subscriber->start().ok());
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_EQ(subscriber->topicStatistics("test/late").matchedWriters, 0u);

  const std::unique_ptr<Bus> publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));
  ASSERT_TRUE(publisher->publish("test/late", makeSample(/*sequence=*/3)).ok());
  EXPECT_TRUE(waitFor([&]() { return collector.contains(3); }));
  EXPECT_TRUE(waitFor([&]() {
    return subscriber->topicStatistics("test/late").matchedWriters == 1u;
  }));
}

TEST(BusConnectionTest,
     ARestartedPublisherIsMatchedAgainAndItsGoneWriterUnmatched) {
  std::unique_ptr<Bus> publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber();
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber
                  ->subscribe<TestSample>("test/restart", Delivery::kAll,
                                          collector.handler())
                  .ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));
  ASSERT_TRUE(
      publisher->publish("test/restart", makeSample(/*sequence=*/1)).ok());
  ASSERT_TRUE(waitFor([&]() { return collector.contains(1); }));
  EXPECT_TRUE(waitFor([&]() {
    return subscriber->topicStatistics("test/restart").matchedWriters == 1u;
  }));

  // The process dies (its writers go with it) and comes back.
  publisher.reset();
  EXPECT_TRUE(waitFor([&]() {
    return subscriber->topicStatistics("test/restart").matchedWriters == 0u;
  }));
  publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));
  ASSERT_TRUE(
      publisher->publish("test/restart", makeSample(/*sequence=*/2)).ok());
  EXPECT_TRUE(waitFor([&]() { return collector.contains(2); }));
  // A new writer starts its sequence numbers anew: no gap is counted between
  // the two.
  EXPECT_EQ(subscriber->topicStatistics("test/restart").missed, 0u);
}

TEST(BusConnectionTest,
     ThePublishQueueDropsAtCapacityCountsAndSendsTheRestOnStart) {
  constexpr size_t kCapacity = 8;
  constexpr size_t kPublished = 20;
  BusOptions options;
  options.publishQueueCapacity = kCapacity;
  const std::unique_ptr<Bus> publisher = createPublisher(std::move(options));
  ASSERT_NE(publisher, nullptr);

  // Not started: nothing drains the queue.
  size_t refused = 0;
  for (size_t sequence = 0; sequence < kPublished; ++sequence) {
    const absl::Status status =
        publisher->publish("test/queue", makeSample(sequence));
    if (!status.ok()) {
      EXPECT_EQ(status.code(), absl::StatusCode::kResourceExhausted) << status;
      ++refused;
    }
  }
  EXPECT_EQ(refused, kPublished - kCapacity);
  EXPECT_EQ(publisher->topicStatistics("test/queue").sendDropped,
            kPublished - kCapacity);
  EXPECT_EQ(publisher->topicStatistics("test/queue").written, 0u);

  ASSERT_TRUE(publisher->start().ok());
  EXPECT_TRUE(waitFor([&]() {
    return publisher->topicStatistics("test/queue").written == kCapacity;
  }));
  EXPECT_EQ(publisher->statistics().count("test/queue"), 1u);
}

}  // namespace
}  // namespace robot::ipc
