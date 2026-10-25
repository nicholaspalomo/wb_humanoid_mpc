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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "robot_runtime/web_viewer/test/WebViewerTestSupport.h"
#include "web_viewer/EncodedFrame.h"
#include "web_viewer/ViewerFrameMailbox.h"
#include "web_viewer/WebViewerServer.h"

namespace robot::web_viewer {
namespace {

using test_support::HttpReply;
using test_support::HttpRequest;
using test_support::HttpResponseReader;
using test_support::MjpegPart;
using test_support::MjpegPartParser;
using test_support::TestConnection;
using test_support::TestServer;

// The tunnel the relay models: an SSH or VS Code port forward that drains the
// server's socket at once into a channel buffer of up to 8 MiB and delivers 1
// MB/s to the browser.
constexpr int64_t kRelayBytesPerSecond = 1'000'000;
constexpr int64_t kRelayBufferBytes = 8 * 1024 * 1024;
constexpr absl::Duration kRelayTick = absl::Milliseconds(10);
// The frames: 100 KB at 16 Hz, which is more than the tunnel carries.
constexpr int64_t kFrameBytes = 100'000;
constexpr absl::Duration kFramePeriod = absl::Microseconds(62'500);

/**
 * A one-connection TCP relay between a test client and the server that reads
 * from the server as fast as it can into a buffer of up to kRelayBufferBytes
 * and forwards it to the client at kRelayBytesPerSecond. The client's bytes
 * pass to the server at once. Not thread-safe; its threads stop and join on
 * destruction.
 */
class ThrottledRelay {
 public:
  static absl::StatusOr<std::unique_ptr<ThrottledRelay>> Create(
      int serverPort) {
    const int listener =
        socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, /*protocol=*/0);
    if (listener < 0) return absl::ErrnoToStatus(errno, "socket");
    std::unique_ptr<ThrottledRelay> relay =
        absl::WrapUnique(new ThrottledRelay(listener, serverPort));
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(listener, reinterpret_cast<const sockaddr*>(&address),
             sizeof(address)) != 0)
      return absl::ErrnoToStatus(errno, "bind");
    if (listen(listener, /*n=*/1) != 0)
      return absl::ErrnoToStatus(errno, "listen");
    socklen_t length = sizeof(address);
    if (getsockname(listener, reinterpret_cast<sockaddr*>(&address), &length) !=
        0)
      return absl::ErrnoToStatus(errno, "getsockname");
    relay->port_ = ntohs(address.sin_port);
    relay->acceptThread_ =
        std::thread([relay = relay.get()] { relay->acceptAndRelay(); });
    return relay;
  }

  ~ThrottledRelay() {
    stopping_.store(true);
    shutdown(listener_, SHUT_RDWR);
    acceptThread_.join();
    // The accept thread has stored both sockets, if it got them: shutting them
    // down ends the pumps' blocking calls.
    for (const int socket : {client_.load(), server_.load()}) {
      if (socket >= 0) shutdown(socket, SHUT_RDWR);
    }
    for (std::thread& pump : pumps_) pump.join();
    for (const int socket : {listener_, client_.load(), server_.load()}) {
      if (socket >= 0) close(socket);
    }
  }
  ThrottledRelay(const ThrottledRelay&) = delete;
  ThrottledRelay& operator=(const ThrottledRelay&) = delete;

  int port() const { return port_; }

 private:
  ThrottledRelay(int listener, int serverPort)
      : listener_(listener), serverPort_(serverPort) {}

  void acceptAndRelay() {
    const int client = accept4(listener_, /*addr=*/nullptr,
                               /*addr_len=*/nullptr, SOCK_CLOEXEC);
    if (client < 0) return;
    client_.store(client);
    const int server =
        socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, /*protocol=*/0);
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<uint16_t>(serverPort_));
    if (server < 0 ||
        connect(server, reinterpret_cast<const sockaddr*>(&address),
                sizeof(address)) != 0)
      return;
    server_.store(server);
    pumps_.emplace_back(
        [this, client, server] { forwardAtOnce(client, server); });
    pumps_.emplace_back([this, server] { drainServer(server); });
    pumps_.emplace_back([this, client] { deliverThrottled(client); });
  }

  void forwardAtOnce(int from, int to) {
    std::array<char, 16384> buffer = {};
    while (!stopping_.load()) {
      const ssize_t received =
          recv(from, buffer.data(), buffer.size(), /*flags=*/0);
      if (received <= 0) return;
      if (send(to, buffer.data(), static_cast<size_t>(received),
               MSG_NOSIGNAL) != received)
        return;
    }
  }

  void drainServer(int server) {
    std::array<char, 65536> buffer = {};
    while (!stopping_.load()) {
      bool full = false;
      {
        absl::MutexLock lock(&mutex_);
        full = static_cast<int64_t>(pending_.size() - pendingStart_) >=
               kRelayBufferBytes;
      }
      if (full) {
        absl::SleepFor(absl::Milliseconds(1));
        continue;
      }
      const ssize_t received =
          recv(server, buffer.data(), buffer.size(), /*flags=*/0);
      if (received <= 0) return;
      absl::MutexLock lock(&mutex_);
      pending_.append(buffer.data(), static_cast<size_t>(received));
    }
  }

  void deliverThrottled(int client) {
    const size_t perTick =
        static_cast<size_t>(kRelayBytesPerSecond *
                            absl::ToInt64Microseconds(kRelayTick) / 1'000'000);
    absl::Time next = absl::Now();
    while (!stopping_.load()) {
      next += kRelayTick;
      absl::SleepFor(next - absl::Now());
      std::string chunk;
      {
        absl::MutexLock lock(&mutex_);
        const size_t taken = std::min(perTick, pending_.size() - pendingStart_);
        chunk = pending_.substr(pendingStart_, taken);
        pendingStart_ += taken;
        if (pendingStart_ > 1'000'000) {
          pending_.erase(0, pendingStart_);
          pendingStart_ = 0;
        }
      }
      if (!chunk.empty() &&
          send(client, chunk.data(), chunk.size(), MSG_NOSIGNAL) !=
              static_cast<ssize_t>(chunk.size()))
        return;
    }
  }

  const int listener_;
  const int serverPort_;
  int port_ = 0;
  std::atomic<int> client_ = -1;
  std::atomic<int> server_ = -1;
  std::atomic<bool> stopping_ = false;
  absl::Mutex mutex_;
  // The bytes from the server not yet delivered: pending_[pendingStart_:].
  std::string pending_ ABSL_GUARDED_BY(mutex_);
  size_t pendingStart_ ABSL_GUARDED_BY(mutex_) = 0;
  std::thread acceptThread_;
  std::vector<std::thread> pumps_;
};

/**
 * Publishes kFrameBytes frames every kFramePeriod and records when each was
 * published, until destroyed. Thread-safe.
 */
class TimedPublisher {
 public:
  explicit TimedPublisher(ViewerFrameMailbox* absl_nonnull mailbox)
      : mailbox_(mailbox), thread_([this] {
          absl::Time next = absl::Now();
          while (!done_.load()) {
            mailbox_->publish(test_support::opaqueFrame(kFrameBytes));
            const std::optional<EncodedFrame> published =
                mailbox_->latestFrameMetadata();
            if (published.has_value()) {
              absl::MutexLock lock(&mutex_);
              publishTimes_.resize(static_cast<size_t>(published->sequence) + 1,
                                   absl::InfinitePast());
              publishTimes_[static_cast<size_t>(published->sequence)] =
                  published->publishTime;
            }
            next += kFramePeriod;
            absl::SleepFor(next - absl::Now());
          }
        }) {}
  ~TimedPublisher() {
    done_.store(true);
    thread_.join();
  }
  TimedPublisher(const TimedPublisher&) = delete;
  TimedPublisher& operator=(const TimedPublisher&) = delete;

  absl::Time publishTime(int64_t sequence) const {
    absl::MutexLock lock(&mutex_);
    return sequence > 0 && static_cast<size_t>(sequence) < publishTimes_.size()
               ? publishTimes_[static_cast<size_t>(sequence)]
               : absl::InfinitePast();
  }

 private:
  ViewerFrameMailbox* absl_nonnull mailbox_;
  std::atomic<bool> done_ = false;
  mutable absl::Mutex mutex_;
  std::vector<absl::Time> publishTimes_ ABSL_GUARDED_BY(mutex_);
  std::thread thread_;
};

// How late a frame arrived beyond the time its own bytes need through the
// tunnel.
absl::Duration lagBeyondTransfer(absl::Time received,
                                 absl::Time published,
                                 int64_t bytes) {
  return received - published -
         absl::Microseconds(bytes * 1'000'000 / kRelayBytesPerSecond);
}

TEST(WebViewerLatencyTest,
     APullingPageStaysFreshThroughATunnelWhereTheStreamFallsBehind) {
  absl::StatusOr<TestServer> test =
      test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  const TimedPublisher publisher(test->mailbox.get());

  // The page's protocol: one GET /frame.jpg?after=<sequence> in flight, on one
  // keep-alive connection.
  {
    absl::StatusOr<std::unique_ptr<ThrottledRelay>> relay =
        ThrottledRelay::Create(test->server->port());
    ASSERT_TRUE(relay.ok()) << relay.status();
    absl::StatusOr<std::unique_ptr<TestConnection>> connection =
        TestConnection::Connect((*relay)->port(),
                                /*receiveBufferBytes=*/std::nullopt);
    ASSERT_TRUE(connection.ok()) << connection.status();
    HttpResponseReader reader(connection->get());
    int64_t after = 0;
    int frames = 0;
    absl::Duration worstLag = absl::ZeroDuration();
    const absl::Time end = absl::Now() + absl::Seconds(3);
    while (absl::Now() < end) {
      const std::string request = test_support::formatRequest(
          (*relay)->port(),
          HttpRequest{.method = "GET",
                      .target = absl::StrCat("/frame.jpg?after=", after),
                      .keepAlive = true});
      ASSERT_TRUE((*connection)->send(request).ok());
      absl::StatusOr<HttpReply> reply = reader.readResponse(absl::Seconds(5));
      const absl::Time received = absl::Now();
      ASSERT_TRUE(reply.ok()) << reply.status();
      if (reply->status == 204) continue;
      ASSERT_EQ(reply->status, 200) << reply->body;
      ASSERT_EQ(static_cast<int64_t>(reply->body.size()), kFrameBytes);
      ASSERT_TRUE(absl::SimpleAtoi(
          reply->header("X-Frame-Sequence").value_or(""), &after));
      ++frames;
      // The first answer also carries the connection's start.
      if (frames > 1)
        worstLag = std::max(
            worstLag, lagBeyondTransfer(received, publisher.publishTime(after),
                                        kFrameBytes));
    }
    LOG(INFO) << "pull: " << frames
              << " frames, the worst lag beyond the transfer " << worstLag;
    EXPECT_GE(frames, 10);
    EXPECT_LE(worstLag, 2 * kFramePeriod);
  }

  // The control: GET /stream.mjpg through the same tunnel queues frames in it
  // and falls behind.
  {
    absl::StatusOr<std::unique_ptr<ThrottledRelay>> relay =
        ThrottledRelay::Create(test->server->port());
    ASSERT_TRUE(relay.ok()) << relay.status();
    absl::StatusOr<std::unique_ptr<TestConnection>> connection =
        TestConnection::Connect((*relay)->port(),
                                /*receiveBufferBytes=*/std::nullopt);
    ASSERT_TRUE(connection.ok()) << connection.status();
    ASSERT_TRUE((*connection)
                    ->send(test_support::formatRequest(
                        (*relay)->port(),
                        HttpRequest{.method = "GET", .target = "/stream.mjpg"}))
                    .ok());
    HttpResponseReader reader(connection->get());
    absl::StatusOr<HttpReply> head =
        reader.readHead(absl::Seconds(5), /*headRequest=*/false);
    ASSERT_TRUE(head.ok()) << head.status();
    MjpegPartParser parser;
    absl::Duration lastLag = absl::ZeroDuration();
    int parts = 0;
    const absl::Time end = absl::Now() + absl::Seconds(3);
    while (absl::Now() < end) {
      absl::StatusOr<std::string> bytes = reader.readBody(end - absl::Now());
      if (!bytes.ok()) break;
      parser.append(*bytes);
      const absl::Time received = absl::Now();
      while (true) {
        absl::StatusOr<std::optional<MjpegPart>> part = parser.nextPart();
        ASSERT_TRUE(part.ok()) << part.status();
        if (!part->has_value()) break;
        ++parts;
        lastLag = lagBeyondTransfer(
            received, publisher.publishTime((*part)->sequence), kFrameBytes);
      }
    }
    LOG(INFO) << "stream: " << parts << " parts, the last one late by "
              << lastLag << " beyond its transfer";
    EXPECT_GT(lastLag, 4 * kFramePeriod);
  }
}

TEST(WebViewerLatencyTest, AClientThatStopsReadingNeverBlocksThePublisher) {
  absl::StatusOr<TestServer> test =
      test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  ViewerFrameMailbox* absl_nonnull mailbox = test->mailbox.get();
  // 1 MB frames at 30 Hz fill the stuck connection's buffers within a fraction
  // of a second.
  std::atomic<bool> done = false;
  std::vector<absl::Duration> publishDurations;
  std::thread publisher([mailbox, &done, &publishDurations] {
    while (!done.load()) {
      EncodedFrame frame = test_support::opaqueFrame(/*bytes=*/1'000'000);
      const absl::Time start = absl::Now();
      mailbox->publish(std::move(frame));
      publishDurations.push_back(absl::Now() - start);
      absl::SleepFor(absl::Milliseconds(33));
    }
  });
  absl::StatusOr<std::unique_ptr<TestConnection>> stuck =
      TestConnection::Connect(test->server->port(),
                              /*receiveBufferBytes=*/4096);
  ASSERT_TRUE(stuck.ok()) << stuck.status();
  ASSERT_TRUE((*stuck)
                  ->send(test_support::formatRequest(
                      test->server->port(),
                      HttpRequest{.method = "GET", .target = "/stream.mjpg"}))
                  .ok());
  ASSERT_TRUE(test_support::waitUntil(
      [mailbox] { return mailbox->activeClients() == 1; }, absl::Seconds(2)));
  // Stuck: its socket buffers are full and the server's write waits. Destroying
  // the server must not wait longer than the 2 s write timeout and a margin.
  absl::SleepFor(absl::Seconds(1));
  const absl::Time stopping = absl::Now();
  test->server.reset();
  const absl::Duration stopTook = absl::Now() - stopping;
  done.store(true);
  publisher.join();
  EXPECT_LT(stopTook, absl::Seconds(3));
  EXPECT_EQ(mailbox->activeClients(), 0);
  ASSERT_GT(publishDurations.size(), 20U);
  std::sort(publishDurations.begin(), publishDurations.end());
  const absl::Duration p99 =
      publishDurations[publishDurations.size() * 99 / 100];
  LOG(INFO) << publishDurations.size() << " publishes, p99 " << p99 << ", max "
            << publishDurations.back();
  // A publish waits only for a request thread's copy of the frame, never for a
  // socket (2 s when it does).
  EXPECT_LT(p99, absl::Milliseconds(1));
  EXPECT_LT(publishDurations.back(), absl::Milliseconds(100));
}

TEST(WebViewerLatencyTest, TheServerDropsAStuckClientAfterTheWriteTimeout) {
  absl::StatusOr<TestServer> test =
      test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  ViewerFrameMailbox* absl_nonnull mailbox = test->mailbox.get();
  std::atomic<bool> done = false;
  std::thread publisher([mailbox, &done] {
    while (!done.load()) {
      mailbox->publish(test_support::opaqueFrame(/*bytes=*/1'000'000));
      absl::SleepFor(absl::Milliseconds(33));
    }
  });
  absl::StatusOr<std::unique_ptr<TestConnection>> stuck =
      TestConnection::Connect(test->server->port(),
                              /*receiveBufferBytes=*/4096);
  ASSERT_TRUE(stuck.ok()) << stuck.status();
  ASSERT_TRUE((*stuck)
                  ->send(test_support::formatRequest(
                      test->server->port(),
                      HttpRequest{.method = "GET", .target = "/stream.mjpg"}))
                  .ok());
  ASSERT_TRUE(test_support::waitUntil(
      [mailbox] { return mailbox->activeClients() == 1; }, absl::Seconds(2)));
  // The stream's request thread and client slot come back once a write has
  // waited 2 s.
  EXPECT_TRUE(test_support::waitUntil(
      [mailbox] { return mailbox->activeClients() == 0; }, absl::Seconds(6)));
  // Meanwhile the server serves everyone else.
  absl::StatusOr<HttpReply> state = test_support::fetch(
      test->server->port(), HttpRequest{.method = "GET", .target = "/state"});
  ASSERT_TRUE(state.ok()) << state.status();
  EXPECT_EQ(state->status, 200);
  done.store(true);
  publisher.join();
}

}  // namespace
}  // namespace robot::web_viewer
