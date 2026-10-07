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

// The bus over loopback TCP: exact topic matching, kLatest against kAll, publishing from several threads, processes
// that start in either order or restart, a prompt stop(), the publish queue's drop counting, rejection of malformed
// messages and handlers that throw. Every publisher binds an ephemeral port (kEphemeralPort) that the subscriber's
// network then names, so concurrent tests never collide on a port.

#include <atomic>
#include <cstdint>
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
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "zmq.hpp"  // NOLINT(build/include_subdir): cppzmq installs its header in no directory

#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_ipc/NetworkConfig.h"
#include "robot_ipc/NodeEndpoint.h"
#include "robot_ipc/TopicStatistics.h"
#include "robot_ipc_test/test_event.pb.h"
#include "robot_ipc_test/test_sample.pb.h"

namespace robot::ipc {
namespace {

using robot_ipc_test::TestEvent;
using robot_ipc_test::TestSample;

constexpr absl::Duration kTimeout = absl::Seconds(20);
constexpr char kPublisherNode[] = "publisher";
constexpr char kProbeTopic[] = "test/probe";

// Polls `condition` until it holds or `timeout` passes.
bool waitFor(const std::function<bool()>& condition, absl::Duration timeout = kTimeout) {
  const absl::Time deadline = absl::Now() + timeout;
  while (absl::Now() < deadline) {
    if (condition()) {
      return true;
    }
    absl::SleepFor(absl::Milliseconds(1));
  }
  return condition();
}

NodeEndpoint loopbackNode(const std::string& name, int port) {
  return NodeEndpoint{.name = name, .host = "127.0.0.1", .port = port};
}

std::unique_ptr<Bus> createBus(BusOptions options) {
  absl::StatusOr<std::unique_ptr<Bus>> bus = Bus::Create(std::move(options));
  EXPECT_TRUE(bus.ok()) << bus.status();
  return bus.ok() ? std::move(*bus) : nullptr;
}

// A publishing bus on an ephemeral loopback port, alone in its network.
std::unique_ptr<Bus> createPublisher(BusOptions options = BusOptions(), int port = kEphemeralPort) {
  options.nodeName = kPublisherNode;
  options.network.nodes = {loopbackNode(kPublisherNode, port)};
  return createBus(std::move(options));
}

// A bus that only subscribes, to a network that holds the publisher at the port it bound.
std::unique_ptr<Bus> createSubscriber(int publisherPort, BusOptions options = BusOptions()) {
  options.nodeName.clear();
  options.network.nodes = {loopbackNode(kPublisherNode, publisherPort)};
  return createBus(std::move(options));
}

absl::Status subscribeProbe(Bus& bus) {
  return bus.subscribeRaw(kProbeTopic, Delivery::kLatest, [](absl::string_view /*typeName*/, absl::string_view /*payload*/) {});
}

// Publishes on kProbeTopic until the subscriber has received more probes than before: its connection is up and its
// subscriptions have reached the publisher (ZeroMQ's "slow joiner").
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

// Collects what a handler receives, on the IO thread, for the test thread to read.
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

// ---------------------------------------------------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------------------------------------------------

TEST(BusCreateTest, RejectsANodeOutsideTheNetworkAndListsTheNodes) {
  BusOptions options;
  options.nodeName = "ghost";
  options.network = localhostNetworkConfig();
  const absl::StatusOr<std::unique_ptr<Bus>> bus = Bus::Create(std::move(options));
  EXPECT_EQ(bus.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(bus.status().message(), "ghost")) << bus.status();
  EXPECT_TRUE(absl::StrContains(bus.status().message(), "robot")) << bus.status();
}

TEST(BusCreateTest, RejectsAnInvalidNetworkAndOptions) {
  EXPECT_EQ(Bus::Create(BusOptions()).status().code(), absl::StatusCode::kInvalidArgument);

  BusOptions noQueue;
  noQueue.network.nodes = {loopbackNode(kPublisherNode, kEphemeralPort)};
  noQueue.publishQueueCapacity = 0;
  EXPECT_EQ(Bus::Create(std::move(noQueue)).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(BusCreateTest, ReportsATakenEndpointAsUnavailable) {
  const std::unique_ptr<Bus> first = createPublisher();
  ASSERT_NE(first, nullptr);
  BusOptions options;
  options.nodeName = "second";
  options.network.nodes = {loopbackNode("second", first->boundPort())};
  const absl::StatusOr<std::unique_ptr<Bus>> second = Bus::Create(std::move(options));
  EXPECT_EQ(second.status().code(), absl::StatusCode::kUnavailable);
  EXPECT_TRUE(absl::StrContains(second.status().message(), "second")) << second.status();
  EXPECT_TRUE(absl::StrContains(second.status().message(), first->boundEndpoint())) << second.status();
}

TEST(BusCreateTest, EphemeralPortIsReportedAndItsOwnSubscriberConnectsToIt) {
  const std::unique_ptr<Bus> bus = createPublisher();
  ASSERT_NE(bus, nullptr);
  EXPECT_EQ(bus->nodeName(), kPublisherNode);
  EXPECT_GT(bus->boundPort(), kEphemeralPort);
  EXPECT_TRUE(absl::StartsWith(bus->boundEndpoint(), "tcp://127.0.0.1:")) << bus->boundEndpoint();
  EXPECT_EQ(bus->subscriberEndpoints(), std::vector<std::string>{bus->boundEndpoint()});
}

TEST(BusCreateTest, SubscribeOnlyBusBindsNothingAndCannotPublish) {
  const std::unique_ptr<Bus> bus = createSubscriber(/*publisherPort=*/5999);
  ASSERT_NE(bus, nullptr);
  EXPECT_TRUE(bus->nodeName().empty());
  EXPECT_TRUE(bus->boundEndpoint().empty());
  EXPECT_EQ(bus->boundPort(), kEphemeralPort);
  EXPECT_EQ(bus->subscriberEndpoints(), std::vector<std::string>{"tcp://127.0.0.1:5999"});
  EXPECT_EQ(bus->publish("test/topic", makeSample(/*sequence=*/0)).code(), absl::StatusCode::kFailedPrecondition);
}

// ---------------------------------------------------------------------------------------------------------------------
// Configuration and lifecycle
// ---------------------------------------------------------------------------------------------------------------------

TEST(BusLifecycleTest, ConfigurationIsRefusedWhileRunningAndAcceptedAfterStop) {
  const std::unique_ptr<Bus> bus = createPublisher();
  ASSERT_NE(bus, nullptr);
  Collector collector;
  EXPECT_EQ(bus->subscribe<TestSample>(/*topic=*/"", Delivery::kAll, collector.handler()).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(bus->subscribe<TestSample>("test/a", Delivery::kAll, /*handler=*/nullptr).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(bus->addPeriodicCallback(absl::ZeroDuration(), []() {}).code(), absl::StatusCode::kInvalidArgument);
  ASSERT_TRUE(bus->subscribe<TestSample>("test/a", Delivery::kAll, collector.handler()).ok());
  EXPECT_EQ(bus->subscribe<TestSample>("test/a", Delivery::kLatest, collector.handler()).code(), absl::StatusCode::kAlreadyExists);

  EXPECT_FALSE(bus->isRunning());
  ASSERT_TRUE(bus->start().ok());
  ASSERT_TRUE(bus->start().ok());
  EXPECT_TRUE(bus->isRunning());
  EXPECT_EQ(bus->subscribe<TestSample>("test/b", Delivery::kAll, collector.handler()).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(bus->addPeriodicCallback(absl::Milliseconds(1), []() {}).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(bus->connect("tcp://127.0.0.1:5999").code(), absl::StatusCode::kFailedPrecondition);

  bus->stop();
  bus->stop();
  EXPECT_FALSE(bus->isRunning());
  EXPECT_TRUE(bus->subscribe<TestSample>("test/b", Delivery::kAll, collector.handler()).ok());
  EXPECT_EQ(bus->connect("not an endpoint").code(), absl::StatusCode::kInvalidArgument);
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
  std::atomic<int> calls{0};
  ASSERT_TRUE(bus->subscribe<TestSample>("test/stop", Delivery::kAll,
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

// ---------------------------------------------------------------------------------------------------------------------
// Delivery
// ---------------------------------------------------------------------------------------------------------------------

TEST(BusDeliveryTest, AProcessReceivesItsOwnTopicsAndHandlersPublishFromTheIoThread) {
  const std::unique_ptr<Bus> bus = createPublisher();
  ASSERT_NE(bus, nullptr);
  Bus* absl_nonnull const self = bus.get();
  std::atomic<bool> handlersOnIoThread{true};
  Collector pongs;
  ASSERT_TRUE(subscribeProbe(*bus).ok());
  ASSERT_TRUE(bus->subscribe<TestSample>("test/ping", Delivery::kAll,
                                         [self, &handlersOnIoThread](const TestSample& ping) {
                                           handlersOnIoThread = handlersOnIoThread && self->isIoThread();
                                           TestSample pong = ping;
                                           pong.set_origin("pong");
                                           EXPECT_TRUE(self->publishFromIoThread("test/pong", pong).ok());
                                         })
                  .ok());
  ASSERT_TRUE(bus->subscribe<TestSample>("test/pong", Delivery::kAll, pongs.handler()).ok());
  ASSERT_TRUE(bus->start().ok());
  ASSERT_TRUE(waitUntilConnected(*bus, *bus));

  EXPECT_EQ(bus->publishFromIoThread("test/pong", makeSample(/*sequence=*/0)).code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(bus->publish("test/ping", makeSample(/*sequence=*/7)).ok());
  ASSERT_TRUE(waitFor([&]() { return pongs.size() == 1; }));
  EXPECT_EQ(pongs.samples()[0].sequence(), 7u);
  EXPECT_EQ(pongs.samples()[0].origin(), "pong");
  EXPECT_TRUE(handlersOnIoThread.load());
  EXPECT_FALSE(bus->isIoThread());
}

TEST(BusDeliveryTest, TopicsMatchExactlyNotByPrefix) {
  const std::unique_ptr<Bus> publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber(publisher->boundPort());
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber->subscribe<TestSample>("test/a", Delivery::kAll, collector.handler()).ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));

  // ZeroMQ passes every topic that starts with "test/a"; only "test/a" itself may be delivered.
  ASSERT_TRUE(publisher->publish("test/ab", makeSample(/*sequence=*/100)).ok());
  ASSERT_TRUE(publisher->publish("test/a", makeSample(/*sequence=*/1)).ok());
  ASSERT_TRUE(publisher->publish("test/a/b", makeSample(/*sequence=*/101)).ok());
  ASSERT_TRUE(publisher->publish("test/", makeSample(/*sequence=*/102)).ok());
  // One connection keeps the order, so once this one is in, all of the above have arrived.
  ASSERT_TRUE(publisher->publish("test/a", makeSample(/*sequence=*/2)).ok());
  ASSERT_TRUE(waitFor([&]() { return collector.contains(2); }));

  const std::vector<TestSample> samples = collector.samples();
  ASSERT_EQ(samples.size(), 2u);
  EXPECT_EQ(samples[0].sequence(), 1u);
  EXPECT_EQ(samples[1].sequence(), 2u);
  EXPECT_EQ(subscriber->topicStatistics("test/a").received, 2u);
  EXPECT_EQ(subscriber->topicStatistics("test/ab").received, 0u);
  EXPECT_EQ(publisher->topicStatistics("test/ab").sent, 1u);
}

TEST(BusDeliveryTest, LatestHandsOverOnlyTheNewestOfABurstAndAllHandsOverEveryMessage) {
  constexpr uint64_t kBurst = 500;
  BusOptions unlimited;
  unlimited.sendHighWaterMark = 0;
  unlimited.receiveHighWaterMark = 0;
  unlimited.publishQueueCapacity = 4 * kBurst;
  const std::unique_ptr<Bus> publisher = createPublisher(unlimited);
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber(publisher->boundPort(), unlimited);
  ASSERT_NE(subscriber, nullptr);
  Collector latest;
  Collector all;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber->subscribe<TestSample>("test/latest", Delivery::kLatest, latest.handler()).ok());
  ASSERT_TRUE(subscriber->subscribe<TestSample>("test/all", Delivery::kAll, all.handler()).ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));

  // With the subscriber's IO thread stopped, the burst piles up in its socket, so that the next drain holds many
  // messages of each topic.
  subscriber->stop();
  for (uint64_t sequence = 0; sequence < kBurst; ++sequence) {
    ASSERT_TRUE(publisher->publish("test/latest", makeSample(sequence)).ok());
    ASSERT_TRUE(publisher->publish("test/all", makeSample(sequence)).ok());
  }
  ASSERT_TRUE(waitFor([&]() { return publisher->topicStatistics("test/all").sent == kBurst; }));
  absl::SleepFor(absl::Milliseconds(300));
  ASSERT_TRUE(subscriber->start().ok());

  ASSERT_TRUE(waitFor([&]() { return all.size() == kBurst; }));
  ASSERT_TRUE(waitFor([&]() { return latest.contains(kBurst - 1); }));

  // kAll: every message, in order.
  const std::vector<TestSample> allSamples = all.samples();
  for (uint64_t sequence = 0; sequence < kBurst; ++sequence) {
    EXPECT_EQ(allSamples[sequence].sequence(), sequence);
  }
  const TopicStatistics allStatistics = subscriber->topicStatistics("test/all");
  EXPECT_EQ(allStatistics.received, kBurst);
  EXPECT_EQ(allStatistics.delivered, kBurst);
  EXPECT_EQ(allStatistics.superseded, 0u);

  // kLatest: fewer messages than were sent, always newer than the one before, ending with the newest.
  const std::vector<TestSample> latestSamples = latest.samples();
  EXPECT_LT(latestSamples.size(), kBurst);
  for (size_t i = 1; i < latestSamples.size(); ++i) {
    EXPECT_GT(latestSamples[i].sequence(), latestSamples[i - 1].sequence());
  }
  EXPECT_EQ(latestSamples.back().sequence(), kBurst - 1);
  const TopicStatistics latestStatistics = subscriber->topicStatistics("test/latest");
  EXPECT_EQ(latestStatistics.received, kBurst);
  EXPECT_EQ(latestStatistics.delivered, latestSamples.size());
  EXPECT_EQ(latestStatistics.received, latestStatistics.delivered + latestStatistics.superseded + latestStatistics.rejected);
  EXPECT_EQ(latestStatistics.rejected, 0u);
}

TEST(BusDeliveryTest, PublishingFromSeveralThreadsKeepsTheOrderOfEachThread) {
  constexpr uint32_t kThreads = 4;
  constexpr uint64_t kPerThread = 250;
  BusOptions unlimited;
  unlimited.sendHighWaterMark = 0;
  unlimited.receiveHighWaterMark = 0;
  unlimited.publishQueueCapacity = kThreads * kPerThread;
  const std::unique_ptr<Bus> publisher = createPublisher(unlimited);
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber(publisher->boundPort(), unlimited);
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber->subscribe<TestSample>("test/threads", Delivery::kAll, collector.handler()).ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));

  std::vector<std::thread> threads;
  for (uint32_t index = 0; index < kThreads; ++index) {
    threads.emplace_back([&publisher, index]() {
      for (uint64_t sequence = 0; sequence < kPerThread; ++sequence) {
        EXPECT_TRUE(publisher->publish("test/threads", makeSample(sequence, index)).ok());
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  ASSERT_TRUE(waitFor([&]() { return collector.size() == kThreads * kPerThread; }));
  std::vector<uint64_t> nextSequence(kThreads, 0);
  for (const TestSample& sample : collector.samples()) {
    ASSERT_LT(sample.publisher(), kThreads);
    EXPECT_EQ(sample.sequence(), nextSequence[sample.publisher()]);
    nextSequence[sample.publisher()] = sample.sequence() + 1;
  }
  const TopicStatistics statistics = publisher->topicStatistics("test/threads");
  EXPECT_EQ(statistics.sent, kThreads * kPerThread);
  EXPECT_EQ(statistics.sendDropped, 0u);
}

TEST(BusDeliveryTest, RejectsMalformedMessagesAndCarriesOn) {
  // A bare ZeroMQ publisher, to send what a Bus never would.
  zmq::context_t context;
  zmq::socket_t raw(context, zmq::socket_type::pub);
  raw.set(zmq::sockopt::linger, /*val=*/0);
  raw.bind("tcp://127.0.0.1:*");
  const std::string rawEndpoint = raw.get(zmq::sockopt::last_endpoint);

  const std::unique_ptr<Bus> subscriber = createSubscriber(/*publisherPort=*/5999);
  ASSERT_NE(subscriber, nullptr);
  ASSERT_TRUE(subscriber->connect(rawEndpoint).ok());
  Collector collector;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber->subscribe<TestSample>("test/typed", Delivery::kAll, collector.handler()).ok());
  ASSERT_TRUE(subscriber->start().ok());

  const std::function<void(const std::vector<std::string>&)> sendFrames = [&raw](const std::vector<std::string>& frames) {
    for (size_t i = 0; i < frames.size(); ++i) {
      const zmq::send_flags flags = i + 1 < frames.size() ? zmq::send_flags::sndmore : zmq::send_flags::none;
      ASSERT_TRUE(raw.send(zmq::buffer(frames[i]), flags).has_value());
    }
  };
  ASSERT_TRUE(waitFor([&]() {
    sendFrames({kProbeTopic, "probe", ""});
    absl::SleepFor(absl::Milliseconds(5));
    return subscriber->topicStatistics(kProbeTopic).received > 0;
  }));

  const std::string sampleType(TestSample::descriptor()->full_name());
  const std::string eventType(TestEvent::descriptor()->full_name());
  TestEvent event;
  event.set_name("not a sample");
  sendFrames({"test/typed", eventType, event.SerializeAsString()});
  // A length-delimited field (text) that claims five bytes and carries two.
  sendFrames({"test/typed", sampleType,
              std::string("\x12\x05"
                          "ab")});
  sendFrames({"test/typed", sampleType});
  sendFrames({"test/typed", sampleType, makeSample(/*sequence=*/0).SerializeAsString(), "extra"});
  sendFrames({"test/typed", sampleType, makeSample(/*sequence=*/5).SerializeAsString()});
  ASSERT_TRUE(waitFor([&]() { return collector.size() == 1; }));

  EXPECT_EQ(collector.samples()[0].sequence(), 5u);
  const TopicStatistics statistics = subscriber->topicStatistics("test/typed");
  EXPECT_EQ(statistics.received, 5u);
  EXPECT_EQ(statistics.rejected, 4u);
  EXPECT_EQ(statistics.delivered, 1u);
  subscriber->stop();
}

TEST(BusDeliveryTest, AThrowingHandlerIsCountedAndTheBusCarriesOn) {
  const std::unique_ptr<Bus> publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber(publisher->boundPort());
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  const std::function<void(const TestSample&)> collect = collector.handler();
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber
                  ->subscribe<TestSample>("test/throws", Delivery::kAll,
                                          [&collect](const TestSample& sample) {
                                            if (sample.sequence() == 0) {
                                              throw std::runtime_error("the first sample is unwelcome");
                                            }
                                            collect(sample);
                                          })
                  .ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));

  ASSERT_TRUE(publisher->publish("test/throws", makeSample(/*sequence=*/0)).ok());
  ASSERT_TRUE(publisher->publish("test/throws", makeSample(/*sequence=*/1)).ok());
  ASSERT_TRUE(waitFor([&]() { return collector.size() == 1; }));
  const TopicStatistics statistics = subscriber->topicStatistics("test/throws");
  EXPECT_EQ(statistics.handlerErrors, 1u);
  EXPECT_EQ(statistics.delivered, 2u);
  EXPECT_TRUE(subscriber->isRunning());
}

TEST(BusDeliveryTest, PeriodicCallbacksRunOnTheIoThreadAndMayPublish) {
  const std::unique_ptr<Bus> publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  const std::unique_ptr<Bus> subscriber = createSubscriber(publisher->boundPort());
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber->subscribe<TestSample>("test/periodic", Delivery::kAll, collector.handler()).ok());

  Bus* absl_nonnull const self = publisher.get();
  std::atomic<uint64_t> calls{0};
  std::atomic<bool> onIoThread{true};
  ASSERT_TRUE(publisher
                  ->addPeriodicCallback(absl::Milliseconds(2),
                                        [self, &calls, &onIoThread]() {
                                          onIoThread = onIoThread && self->isIoThread();
                                          EXPECT_TRUE(self->publishFromIoThread("test/periodic", makeSample(calls.fetch_add(1))).ok());
                                        })
                  .ok());
  ASSERT_TRUE(publisher->addPeriodicCallback(absl::Milliseconds(5), []() { throw std::runtime_error("a failing callback"); }).ok());
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

// Starts a subscriber before any publisher is bound at `port`, then the publisher. False if the port was taken in
// between by another process, which the caller retries with a fresh port.
bool subscriberBeforePublisherReceives(int port) {
  const std::unique_ptr<Bus> subscriber = createSubscriber(port);
  if (subscriber == nullptr) {
    return false;
  }
  Collector collector;
  EXPECT_TRUE(subscribeProbe(*subscriber).ok());
  EXPECT_TRUE(subscriber->subscribe<TestSample>("test/late", Delivery::kLatest, collector.handler()).ok());
  EXPECT_TRUE(subscriber->start().ok());
  absl::SleepFor(absl::Milliseconds(50));

  BusOptions options;
  options.nodeName = kPublisherNode;
  options.network.nodes = {loopbackNode(kPublisherNode, port)};
  absl::StatusOr<std::unique_ptr<Bus>> publisher = Bus::Create(std::move(options));
  if (!publisher.ok()) {
    EXPECT_EQ(publisher.status().code(), absl::StatusCode::kUnavailable) << publisher.status();
    return false;
  }
  EXPECT_TRUE((*publisher)->start().ok());
  EXPECT_TRUE(waitUntilConnected(**publisher, *subscriber));
  EXPECT_TRUE((*publisher)->publish("test/late", makeSample(/*sequence=*/3)).ok());
  EXPECT_TRUE(waitFor([&]() { return collector.contains(3); }));
  return true;
}

TEST(BusConnectionTest, SubscriberStartedBeforeThePublisherReceivesOnceItAppears) {
  constexpr int kAttempts = 3;
  bool done = false;
  for (int attempt = 0; attempt < kAttempts && !done; ++attempt) {
    // A port the kernel just handed out and that is free again.
    int port = kEphemeralPort;
    {
      const std::unique_ptr<Bus> probe = createPublisher();
      ASSERT_NE(probe, nullptr);
      port = probe->boundPort();
    }
    done = subscriberBeforePublisherReceives(port);
  }
  EXPECT_TRUE(done);
}

TEST(BusConnectionTest, ARestartedPublisherIsReconnected) {
  std::unique_ptr<Bus> publisher = createPublisher();
  ASSERT_NE(publisher, nullptr);
  const int port = publisher->boundPort();
  const std::unique_ptr<Bus> subscriber = createSubscriber(port);
  ASSERT_NE(subscriber, nullptr);
  Collector collector;
  ASSERT_TRUE(subscribeProbe(*subscriber).ok());
  ASSERT_TRUE(subscriber->subscribe<TestSample>("test/restart", Delivery::kAll, collector.handler()).ok());
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(subscriber->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));
  ASSERT_TRUE(publisher->publish("test/restart", makeSample(/*sequence=*/1)).ok());
  ASSERT_TRUE(waitFor([&]() { return collector.contains(1); }));

  // The process dies and comes back on its port.
  publisher.reset();
  publisher = createPublisher(BusOptions(), port);
  ASSERT_NE(publisher, nullptr);
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(waitUntilConnected(*publisher, *subscriber));
  ASSERT_TRUE(publisher->publish("test/restart", makeSample(/*sequence=*/2)).ok());
  EXPECT_TRUE(waitFor([&]() { return collector.contains(2); }));
}

TEST(BusConnectionTest, ThePublishQueueDropsAtCapacityCountsAndSendsTheRestOnStart) {
  constexpr size_t kCapacity = 8;
  constexpr size_t kPublished = 20;
  BusOptions options;
  options.publishQueueCapacity = kCapacity;
  const std::unique_ptr<Bus> publisher = createPublisher(std::move(options));
  ASSERT_NE(publisher, nullptr);

  // Not started: nothing drains the queue.
  size_t refused = 0;
  for (size_t sequence = 0; sequence < kPublished; ++sequence) {
    const absl::Status status = publisher->publish("test/queue", makeSample(sequence));
    if (!status.ok()) {
      EXPECT_EQ(status.code(), absl::StatusCode::kResourceExhausted) << status;
      ++refused;
    }
  }
  EXPECT_EQ(refused, kPublished - kCapacity);
  EXPECT_EQ(publisher->topicStatistics("test/queue").sendDropped, kPublished - kCapacity);
  EXPECT_EQ(publisher->topicStatistics("test/queue").sent, 0u);

  ASSERT_TRUE(publisher->start().ok());
  EXPECT_TRUE(waitFor([&]() { return publisher->topicStatistics("test/queue").sent == kCapacity; }));
  EXPECT_EQ(publisher->statistics().count("test/queue"), 1u);
}

}  // namespace
}  // namespace robot::ipc
