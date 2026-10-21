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

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "robot_runtime/web_viewer/test/WebViewerTestSupport.h"
#include "web_viewer/EncodedFrame.h"
#include "web_viewer/JpegDecoder.h"
#include "web_viewer/ViewerFrameMailbox.h"
#include "web_viewer/ViewerInput.h"
#include "web_viewer/ViewerInputQueue.h"
#include "web_viewer/ViewerStatus.h"
#include "web_viewer/WebViewerServer.h"
#include "web_viewer/WebViewerServerOptions.h"

namespace robot::web_viewer {
namespace {

using test_support::fetch;
using test_support::HttpReply;
using test_support::HttpRequest;
using test_support::HttpResponseReader;
using test_support::MjpegPart;
using test_support::MjpegPartParser;
using test_support::TestConnection;
using test_support::TestServer;

HttpReply fetchOrFail(int port, const HttpRequest& request) {
  absl::StatusOr<HttpReply> reply = fetch(port, request);
  EXPECT_TRUE(reply.ok()) << request.method << " " << request.target << ": " << reply.status();
  return reply.ok() ? *std::move(reply) : HttpReply();
}

HttpReply get(int port, const std::string& target) {
  return fetchOrFail(port, HttpRequest{.method = "GET", .target = target});
}

HttpReply postFromPage(int port, const std::string& target) {
  return fetchOrFail(port, HttpRequest{.method = "POST", .target = target, .headers = {{"X-MuJoCo-Viewer", "1"}}});
}

void expectSecurityHeaders(const HttpReply& reply) {
  EXPECT_EQ(reply.header("Cross-Origin-Resource-Policy"), "same-origin");
  EXPECT_EQ(reply.header("X-Frame-Options"), "DENY");
  EXPECT_EQ(reply.header("Content-Security-Policy"), "frame-ancestors 'none'");
  EXPECT_EQ(reply.header("X-Content-Type-Options"), "nosniff");
  EXPECT_FALSE(reply.header("Access-Control-Allow-Origin").has_value());
}

// Opens GET /stream.mjpg and reads the head of the answer.
struct OpenStream {
  std::unique_ptr<TestConnection> connection;
  std::unique_ptr<HttpResponseReader> reader;
  HttpReply head;
};

absl::StatusOr<OpenStream> openStream(int port) {
  absl::StatusOr<std::unique_ptr<TestConnection>> connection = TestConnection::Connect(port, /*receiveBufferBytes=*/std::nullopt);
  if (!connection.ok()) return connection.status();
  absl::Status sent = (*connection)->send(test_support::formatRequest(port, HttpRequest{.method = "GET", .target = "/stream.mjpg"}));
  if (!sent.ok()) return sent;
  OpenStream stream = {.connection = *std::move(connection), .reader = nullptr, .head = {}};
  stream.reader = std::make_unique<HttpResponseReader>(stream.connection.get());
  absl::StatusOr<HttpReply> head = stream.reader->readHead(absl::Seconds(5), /*headRequest=*/false);
  if (!head.ok()) return head.status();
  stream.head = *std::move(head);
  return stream;
}

// Reads parts from `stream` until `count` have arrived or `timeout` passes.
std::vector<MjpegPart> readParts(OpenStream& stream, int count, absl::Duration timeout) {
  MjpegPartParser parser;
  std::vector<MjpegPart> parts;
  const absl::Time deadline = absl::Now() + timeout;
  while (static_cast<int>(parts.size()) < count && absl::Now() < deadline) {
    absl::StatusOr<std::string> bytes = stream.reader->readBody(deadline - absl::Now());
    if (!bytes.ok()) break;
    parser.append(*bytes);
    while (true) {
      absl::StatusOr<std::optional<MjpegPart>> part = parser.nextPart();
      EXPECT_TRUE(part.ok()) << part.status();
      if (!part.ok() || !part->has_value()) break;
      parts.push_back(**std::move(part));
    }
  }
  return parts;
}

// Publishes a gradient frame every `period` until destroyed.
class Publisher {
 public:
  Publisher(ViewerFrameMailbox* absl_nonnull mailbox, absl::Duration period, int width, int height)
      : thread_([this, mailbox, period, width, height] {
          for (int phase = 0; !done_.load(); ++phase) {
            mailbox->publish(test_support::gradientFrame(width, height, phase, /*simTime=*/0.01 * phase));
            absl::SleepFor(period);
          }
        }) {}
  ~Publisher() {
    done_.store(true);
    thread_.join();
  }
  Publisher(const Publisher&) = delete;
  Publisher& operator=(const Publisher&) = delete;

 private:
  std::atomic<bool> done_ = false;
  std::thread thread_;
};

TEST(WebViewerServerTest, ServesThePage) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  const HttpReply page = get(test->server->port(), "/");
  EXPECT_EQ(page.status, 200);
  EXPECT_TRUE(absl::StartsWith(page.header("Content-Type").value_or(""), "text/html")) << page.header("Content-Type").value_or("");
  EXPECT_EQ(page.header("Cache-Control"), "no-store");
  EXPECT_TRUE(absl::StrContains(page.body, "/frame.jpg"));
  EXPECT_TRUE(absl::StrContains(page.body, "X-MuJoCo-Viewer"));
  expectSecurityHeaders(page);
  EXPECT_EQ(test->server->url(), absl::StrCat("http://127.0.0.1:", test->server->port(), "/"));
}

TEST(WebViewerServerTest, ServesTheNewestFrameAtItsSize) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  test->mailbox->publish(test_support::gradientFrame(/*width=*/320, /*height=*/240, /*phase=*/3, /*simTime=*/1.5));
  const HttpReply frame = get(test->server->port(), "/frame.jpg");
  ASSERT_EQ(frame.status, 200) << frame.body;
  EXPECT_EQ(frame.header("Content-Type"), "image/jpeg");
  EXPECT_EQ(frame.header("Cache-Control"), "no-store");
  EXPECT_EQ(frame.header("X-Frame-Sequence"), "1");
  EXPECT_EQ(frame.header("X-Sim-Time"), "1.5");
  EXPECT_TRUE(frame.header("X-Frame-Age-Ms").has_value());
  expectSecurityHeaders(frame);
  absl::StatusOr<std::unique_ptr<JpegDecoder>> decoder = JpegDecoder::Create();
  ASSERT_TRUE(decoder.ok()) << decoder.status();
  absl::StatusOr<JpegDecoder::Image> image = (*decoder)->decode(frame.body);
  ASSERT_TRUE(image.ok()) << image.status();
  EXPECT_EQ(image->width, 320);
  EXPECT_EQ(image->height, 240);
}

TEST(WebViewerServerTest, AfterWaitsForTheNextFrame) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  ViewerFrameMailbox* absl_nonnull mailbox = test->mailbox.get();
  mailbox->publish(test_support::opaqueFrame(/*bytes=*/100));
  std::thread publisher([mailbox] {
    absl::SleepFor(absl::Milliseconds(300));
    mailbox->publish(test_support::opaqueFrame(/*bytes=*/200));
  });
  const absl::Time start = absl::Now();
  const HttpReply next = get(test->server->port(), "/frame.jpg?after=1");
  publisher.join();
  EXPECT_EQ(next.status, 200);
  EXPECT_EQ(next.header("X-Frame-Sequence"), "2");
  EXPECT_EQ(next.body.size(), 200U);
  EXPECT_GE(absl::Now() - start, absl::Milliseconds(250));
}

TEST(WebViewerServerTest, AfterAnswersNoContentWhenNothingNewArrives) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  test->mailbox->publish(test_support::opaqueFrame(/*bytes=*/100));
  const absl::Time start = absl::Now();
  const HttpReply reply = get(test->server->port(), "/frame.jpg?after=1");
  const absl::Duration waited = absl::Now() - start;
  EXPECT_EQ(reply.status, 204);
  EXPECT_TRUE(reply.body.empty());
  EXPECT_GE(waited, absl::Milliseconds(1900));
  EXPECT_LT(waited, absl::Seconds(4));
  EXPECT_EQ(test->mailbox->activeClients(), 0);
}

TEST(WebViewerServerTest, ASequenceFromAnEarlierViewerGetsTheNewestFrame) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  test->mailbox->publish(test_support::opaqueFrame(/*bytes=*/100));
  const absl::Time start = absl::Now();
  const HttpReply reply = get(test->server->port(), "/frame.jpg?after=5000");
  EXPECT_EQ(reply.status, 200);
  EXPECT_EQ(reply.header("X-Frame-Sequence"), "1");
  EXPECT_LT(absl::Now() - start, absl::Seconds(1));
}

TEST(WebViewerServerTest, APageComingBackToAnIdleRendererGetsAFreshFrame) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  ViewerFrameMailbox* absl_nonnull mailbox = test->mailbox.get();
  ViewerStatus status;
  status.render.targetFps = 30;
  mailbox->publishStatus(status);
  mailbox->publish(test_support::opaqueFrame(/*bytes=*/100));
  // Older than two frame periods: the renderer went idle after it.
  absl::SleepFor(absl::Milliseconds(200));
  std::thread renderer([mailbox] {
    absl::SleepFor(absl::Milliseconds(100));
    mailbox->publish(test_support::opaqueFrame(/*bytes=*/200));
  });
  const HttpReply fresh = get(test->server->port(), "/frame.jpg?after=0");
  renderer.join();
  EXPECT_EQ(fresh.status, 200);
  EXPECT_EQ(fresh.header("X-Frame-Sequence"), "2");
  // The page comes back after the lease lapsed (the renderer idled): with no fresh frame within 2 s, the stale one is
  // still newer than what the page shows.
  absl::SleepFor(ViewerFrameMailbox::kDemandLease + absl::Milliseconds(200));
  const absl::Time start = absl::Now();
  const HttpReply stale = get(test->server->port(), "/frame.jpg?after=1");
  EXPECT_EQ(stale.status, 200);
  EXPECT_EQ(stale.header("X-Frame-Sequence"), "2");
  EXPECT_GE(absl::Now() - start, absl::Milliseconds(1900));
}

// A frame published while the page's lease keeps the renderer busy is fresh, however old in frame periods: a renderer
// below half its target rate (llvmpipe at 60 fps) never idled. This used to wait 2 s for a next frame.
TEST(WebViewerServerTest, AFrameOfABusyRendererIsFreshWhateverItsAge) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  ViewerFrameMailbox* absl_nonnull mailbox = test->mailbox.get();
  ViewerStatus status;
  status.render.targetFps = 60;
  mailbox->publishStatus(status);
  std::thread renderer([mailbox] {
    absl::SleepFor(absl::Milliseconds(100));
    mailbox->publish(test_support::opaqueFrame(/*bytes=*/100));
  });
  EXPECT_EQ(get(test->server->port(), "/frame.jpg?after=0").header("X-Frame-Sequence"), "1");
  renderer.join();
  mailbox->publish(test_support::opaqueFrame(/*bytes=*/200));
  absl::SleepFor(absl::Milliseconds(100));  // six frame periods at 60 fps, inside the lease
  const absl::Time start = absl::Now();
  const HttpReply next = get(test->server->port(), "/frame.jpg?after=1");
  EXPECT_EQ(next.status, 200);
  EXPECT_EQ(next.header("X-Frame-Sequence"), "2");
  EXPECT_LT(absl::Now() - start, absl::Milliseconds(500)) << "a frame of a busy renderer was taken for an idle one's";
}

// GET /frame.jpg without after= on an idle renderer's frame: the next frame within 2 s, else 503.
TEST(WebViewerServerTest, ASnapshotOfAStaleFrameWaitsForTheNext) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  ViewerFrameMailbox* absl_nonnull mailbox = test->mailbox.get();
  ViewerStatus status;
  status.render.targetFps = 30;
  mailbox->publishStatus(status);
  mailbox->publish(test_support::opaqueFrame(/*bytes=*/100));
  absl::SleepFor(absl::Milliseconds(200));
  std::thread renderer([mailbox] {
    absl::SleepFor(absl::Milliseconds(100));
    mailbox->publish(test_support::opaqueFrame(/*bytes=*/200));
  });
  const HttpReply fresh = get(test->server->port(), "/frame.jpg");
  renderer.join();
  EXPECT_EQ(fresh.status, 200);
  EXPECT_EQ(fresh.header("X-Frame-Sequence"), "2");

  absl::SleepFor(ViewerFrameMailbox::kDemandLease + absl::Milliseconds(200));
  const absl::Time start = absl::Now();
  const HttpReply none = get(test->server->port(), "/frame.jpg");
  EXPECT_EQ(none.status, 503);
  EXPECT_TRUE(absl::StrContains(none.body, "no frame")) << none.body;
  EXPECT_GE(absl::Now() - start, absl::Milliseconds(1900));
  EXPECT_EQ(mailbox->activeClients(), 0);
}

// X-Frame-Age-Ms and /state's ageMs are the time since the frame was published.
TEST(WebViewerServerTest, TheFrameAgeIsTheTimeSinceItWasPublished) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  ViewerFrameMailbox* absl_nonnull mailbox = test->mailbox.get();
  // A page's request keeps the renderer busy, so the frame published after it is fresh.
  ASSERT_TRUE(mailbox->tryAcquireClient());
  mailbox->noteFrameRequest();
  mailbox->releaseClient();
  mailbox->publish(test_support::opaqueFrame(/*bytes=*/100));
  absl::SleepFor(absl::Milliseconds(300));
  const HttpReply frame = get(test->server->port(), "/frame.jpg?after=0");
  ASSERT_EQ(frame.status, 200);
  int64_t ageMs = 0;
  ASSERT_TRUE(absl::SimpleAtoi(frame.header("X-Frame-Age-Ms").value_or(""), &ageMs)) << frame.header("X-Frame-Age-Ms").value_or("");
  EXPECT_GE(ageMs, 250);
  EXPECT_LT(ageMs, 2000);
  const std::string state = get(test->server->port(), "/state").body;
  constexpr absl::string_view kAgeField = R"("ageMs":)";
  const size_t field = state.find(kAgeField);
  ASSERT_NE(field, std::string::npos) << state;
  int64_t stateAgeMs = 0;
  const absl::string_view digits = absl::string_view(state).substr(field + kAgeField.size());
  ASSERT_TRUE(absl::SimpleAtoi(digits.substr(0, digits.find_first_not_of("0123456789")), &stateAgeMs)) << state;
  EXPECT_GE(stateAgeMs, ageMs);
}

// /state's `viewers`: the pages (by the id each sends with its frame requests) that pulled a frame within the demand
// lease, and the open streams. `clients` counts only the requests waiting at that instant, 0 for a page between two
// pulls.
TEST(WebViewerServerTest, CountsTheViewersByTheirPagesAndStreams) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  const int port = test->server->port();
  test->mailbox->publish(test_support::opaqueFrame(/*bytes=*/10));
  EXPECT_EQ(get(port, "/frame.jpg?after=0&viewer=0a1b").status, 200);
  EXPECT_TRUE(absl::StrContains(get(port, "/state").body, R"("viewers":1,)")) << get(port, "/state").body;
  EXPECT_EQ(get(port, "/frame.jpg?after=0&viewer=0a1b").status, 200);
  EXPECT_TRUE(absl::StrContains(get(port, "/state").body, R"("viewers":1,)"));
  EXPECT_EQ(get(port, "/frame.jpg?after=0&viewer=ff").status, 200);
  EXPECT_TRUE(absl::StrContains(get(port, "/state").body, R"("viewers":2,)")) << get(port, "/state").body;
  // curl or an old page: demand, but no viewer.
  EXPECT_EQ(get(port, "/frame.jpg?after=0").status, 200);
  EXPECT_TRUE(absl::StrContains(get(port, "/state").body, R"("viewers":2,)"));
  for (const char* absl_nonnull const id : {"", "XYZ", "0123456789abcdef0123456789abcdef0"}) {
    EXPECT_EQ(get(port, absl::StrCat("/frame.jpg?after=0&viewer=", id)).status, 400) << id;
  }
  absl::SleepFor(ViewerFrameMailbox::kDemandLease + absl::Milliseconds(200));
  EXPECT_TRUE(absl::StrContains(get(port, "/state").body, R"("viewers":0,)")) << get(port, "/state").body;
  {
    absl::StatusOr<OpenStream> stream = openStream(port);
    ASSERT_TRUE(stream.ok()) << stream.status();
    ASSERT_EQ(stream->head.status, 200);
    EXPECT_TRUE(absl::StrContains(get(port, "/state").body, R"("viewers":1,)")) << get(port, "/state").body;
  }
  EXPECT_TRUE(test_support::waitUntil([&test] { return test->mailbox->activeClients() == 0; }, absl::Seconds(2)));
  absl::SleepFor(ViewerFrameMailbox::kDemandLease + absl::Milliseconds(200));
  EXPECT_TRUE(absl::StrContains(get(port, "/state").body, R"("viewers":0,)")) << get(port, "/state").body;
}

TEST(WebViewerServerTest, ASnapshotWithoutFramesFails) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  const HttpReply reply = get(test->server->port(), "/frame.jpg");
  EXPECT_EQ(reply.status, 503);
  EXPECT_TRUE(absl::StrContains(reply.body, "no frame")) << reply.body;
  EXPECT_EQ(get(test->server->port(), "/frame.jpg?after=-1").status, 400);
  EXPECT_EQ(get(test->server->port(), "/frame.jpg?after=x").status, 400);
}

TEST(WebViewerServerTest, StreamsPartsWithIncreasingSequences) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  {
    const Publisher publisher(test->mailbox.get(), absl::Milliseconds(40), /*width=*/160, /*height=*/120);
    absl::StatusOr<OpenStream> stream = openStream(test->server->port());
    ASSERT_TRUE(stream.ok()) << stream.status();
    EXPECT_EQ(stream->head.status, 200);
    EXPECT_EQ(stream->head.header("Content-Type"), "multipart/x-mixed-replace; boundary=mujocoframe");
    expectSecurityHeaders(stream->head);
    EXPECT_EQ(test->mailbox->activeClients(), 1);
    const std::vector<MjpegPart> parts = readParts(*stream, /*count=*/3, absl::Seconds(5));
    ASSERT_EQ(parts.size(), 3U);
    absl::StatusOr<std::unique_ptr<JpegDecoder>> decoder = JpegDecoder::Create();
    ASSERT_TRUE(decoder.ok()) << decoder.status();
    for (size_t index = 0; index < parts.size(); ++index) {
      if (index > 0) {
        EXPECT_GT(parts[index].sequence, parts[index - 1].sequence);
      }
      absl::StatusOr<JpegDecoder::Image> image = (*decoder)->decode(parts[index].jpeg);
      ASSERT_TRUE(image.ok()) << image.status();
      EXPECT_EQ(image->width, 160);
    }
  }
  EXPECT_TRUE(test_support::waitUntil([&test] { return test->mailbox->activeClients() == 0; }, absl::Seconds(2)));
}

TEST(WebViewerServerTest, AFifthClientIsRefused) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  const int port = test->server->port();
  std::vector<OpenStream> streams;
  for (int client = 0; client < 4; ++client) {
    absl::StatusOr<OpenStream> stream = openStream(port);
    ASSERT_TRUE(stream.ok()) << stream.status();
    ASSERT_EQ(stream->head.status, 200);
    streams.push_back(*std::move(stream));
  }
  EXPECT_EQ(test->mailbox->activeClients(), 4);
  absl::StatusOr<OpenStream> fifth = openStream(port);
  ASSERT_TRUE(fifth.ok()) << fifth.status();
  EXPECT_EQ(fifth->head.status, 503);
  const HttpReply longPoll = get(port, "/frame.jpg?after=0");
  EXPECT_EQ(longPoll.status, 503);
  EXPECT_TRUE(absl::StrContains(longPoll.body, "too many viewers (4)")) << longPoll.body;
  streams.pop_back();
  ASSERT_TRUE(test_support::waitUntil([&test] { return test->mailbox->activeClients() == 3; }, absl::Seconds(3)));
  absl::StatusOr<OpenStream> again = openStream(port);
  ASSERT_TRUE(again.ok()) << again.status();
  EXPECT_EQ(again->head.status, 200);
}

TEST(WebViewerServerTest, AClientLeavingMidPartLeavesTheOthersServed) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  ViewerFrameMailbox* absl_nonnull mailbox = test->mailbox.get();
  std::atomic<bool> done = false;
  std::thread publisher([mailbox, &done] {
    while (!done.load()) {
      mailbox->publish(test_support::opaqueFrame(/*bytes=*/400'000));
      absl::SleepFor(absl::Milliseconds(30));
    }
  });
  absl::StatusOr<OpenStream> leaving = openStream(test->server->port());
  absl::StatusOr<OpenStream> staying = openStream(test->server->port());
  ASSERT_TRUE(leaving.ok()) << leaving.status();
  ASSERT_TRUE(staying.ok()) << staying.status();
  // The leaving client reads part of a part, then goes away in the middle of it.
  absl::StatusOr<std::string> partial = leaving->reader->readBody(absl::Seconds(5));
  EXPECT_TRUE(partial.ok()) << partial.status();
  leaving->reader.reset();
  leaving->connection.reset();
  const std::vector<MjpegPart> parts = readParts(*staying, /*count=*/5, absl::Seconds(10));
  EXPECT_EQ(parts.size(), 5U);
  EXPECT_TRUE(test_support::waitUntil([mailbox] { return mailbox->activeClients() == 1; }, absl::Seconds(1)));
  staying->reader.reset();
  staying->connection.reset();
  EXPECT_TRUE(test_support::waitUntil([mailbox] { return mailbox->activeClients() == 0; }, absl::Seconds(1)));
  done.store(true);
  publisher.join();
}

// The trust model of README.md, on a loopback bind and on a wildcard bind.
void checkTrust(const std::string& address) {
  SCOPED_TRACE(address);
  WebViewerServerOptions options = test_support::testServerOptions();
  options.address = address;
  options.allowedHosts = {"Robot.LAN"};
  absl::StatusOr<TestServer> test = test_support::startTestServer(std::move(options));
  ASSERT_TRUE(test.ok()) << test.status();
  const int port = test->server->port();
  test->mailbox->publish(test_support::opaqueFrame(/*bytes=*/100));

  // Host: the name decides, never the port (a tunnel may use another local port).
  // An IPv6 Host in any spelling: "[0:0:0:0:0:0:0:1]" is "[::1]".
  for (const char* absl_nonnull const host :
       {"localhost:8080", "LOCALHOST", "127.0.0.1:1", "[::1]:7000", "robot.lan:6090", "[0:0:0:0:0:0:0:1]:1"}) {
    const HttpReply reply = fetchOrFail(port, HttpRequest{.method = "GET", .target = "/state", .host = host});
    EXPECT_EQ(reply.status, 200) << host;
  }
  // Only the loopback names of README.md: not the rest of 127.0.0.0/8, nor ::1 mapped to IPv4.
  for (const char* absl_nonnull const host : {"evil.example", "evil.example:6090", "0.0.0.0:6090", "localhost.evil.example", "::1",
                                              "127.0.0.2:6090", "[::ffff:127.0.0.1]:6090"}) {
    const HttpReply reply = fetchOrFail(port, HttpRequest{.method = "GET", .target = "/", .host = host});
    EXPECT_EQ(reply.status, 403) << host;
    expectSecurityHeaders(reply);
  }

  // Fetch metadata: another site's <img> or fetch neither streams nor counts as a client.
  for (const char* absl_nonnull const site : {"cross-site", "same-site"}) {
    const HttpReply reply =
        fetchOrFail(port, HttpRequest{.method = "GET", .target = "/stream.mjpg", .headers = {{"Sec-Fetch-Site", site}}});
    EXPECT_EQ(reply.status, 403) << site;
    EXPECT_EQ(test->mailbox->activeClients(), 0);
  }
  for (const char* absl_nonnull const site : {"same-origin", "none"}) {
    EXPECT_EQ(fetchOrFail(port, HttpRequest{.method = "GET", .target = "/state", .headers = {{"Sec-Fetch-Site", site}}}).status, 200)
        << site;
  }

  // POST: the custom header, and an Origin equal to http:// or https:// and the Host when there is one (https: behind an
  // HTTPS-terminating forward that keeps the Host).
  EXPECT_EQ(fetchOrFail(port, HttpRequest{.method = "POST", .target = "/input?type=reset_camera"}).status, 403);
  EXPECT_EQ(postFromPage(port, "/input?type=reset_camera").status, 204);
  EXPECT_EQ(fetchOrFail(port, HttpRequest{.method = "POST",
                                          .target = "/input?type=reset_camera",
                                          .headers = {{"X-MuJoCo-Viewer", "1"}, {"Origin", "http://evil.example"}}})
                .status,
            403);
  EXPECT_EQ(fetchOrFail(port, HttpRequest{.method = "POST",
                                          .target = "/input?type=reset_camera",
                                          .host = "localhost:9000",
                                          .headers = {{"X-MuJoCo-Viewer", "1"}, {"Origin", "http://localhost:9000"}}})
                .status,
            204);
  EXPECT_EQ(fetchOrFail(port, HttpRequest{.method = "POST",
                                          .target = "/input?type=reset_camera",
                                          .headers = {{"X-MuJoCo-Viewer", "1"}, {"Origin", "null"}}})
                .status,
            403);
  struct HostAndOrigin {
    std::string host;
    std::string origin;
    int status = 0;
  };
  for (const HostAndOrigin& post : {HostAndOrigin{.host = "localhost:9000", .origin = "https://localhost:9000", .status = 204},
                                    HostAndOrigin{.host = "robot.lan", .origin = "https://Robot.LAN", .status = 204},
                                    HostAndOrigin{.host = "localhost:9000", .origin = "https://localhost:9001", .status = 403},
                                    HostAndOrigin{.host = "localhost:9000", .origin = "https://evil.example", .status = 403},
                                    HostAndOrigin{.host = "localhost:9000", .origin = "ftp://localhost:9000", .status = 403}}) {
    EXPECT_EQ(fetchOrFail(port, HttpRequest{.method = "POST",
                                            .target = "/input?type=reset_camera",
                                            .host = post.host,
                                            .headers = {{"X-MuJoCo-Viewer", "1"}, {"Origin", post.origin}}})
                  .status,
              post.status)
        << post.host << " " << post.origin;
  }

  // Bodies are capped at 1 KiB.
  const HttpReply large = fetchOrFail(
      port,
      HttpRequest{
          .method = "POST", .target = "/input?type=reset_camera", .headers = {{"X-MuJoCo-Viewer", "1"}}, .body = std::string(2048, 'b')});
  EXPECT_EQ(large.status, 413);
  expectSecurityHeaders(large);
  EXPECT_EQ(test->mailbox->activeClients(), 0);
}

TEST(WebViewerServerTest, EnforcesTheTrustModelOnALoopbackBind) {
  checkTrust("127.0.0.1");
}

TEST(WebViewerServerTest, EnforcesTheTrustModelOnAWildcardBind) {
  checkTrust("0.0.0.0");
}

TEST(WebViewerServerTest, ARequestWithoutHostIsRefused) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  absl::StatusOr<std::unique_ptr<TestConnection>> connection =
      TestConnection::Connect(test->server->port(), /*receiveBufferBytes=*/std::nullopt);
  ASSERT_TRUE(connection.ok()) << connection.status();
  ASSERT_TRUE((*connection)->send("GET / HTTP/1.0\r\n\r\n").ok());
  HttpResponseReader reader(connection->get());
  absl::StatusOr<HttpReply> reply = reader.readResponse(absl::Seconds(5));
  ASSERT_TRUE(reply.ok()) << reply.status();
  EXPECT_EQ(reply->status, 403);
}

TEST(WebViewerServerTest, AnswersUnknownRoutesAndMethods) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  const int port = test->server->port();
  const HttpReply missing = get(port, "/index.html");
  EXPECT_EQ(missing.status, 404);
  expectSecurityHeaders(missing);
  EXPECT_EQ(get(port, "/frameXjpg").status, 404);
  const HttpReply postPage = postFromPage(port, "/");
  EXPECT_EQ(postPage.status, 405);
  EXPECT_EQ(postPage.header("Allow"), "GET, HEAD");
  const HttpReply getInput = get(port, "/input?type=reset_camera");
  EXPECT_EQ(getInput.status, 405);
  EXPECT_EQ(getInput.header("Allow"), "POST");
  EXPECT_EQ(fetchOrFail(port, HttpRequest{.method = "OPTIONS", .target = "/frame.jpg"}).status, 405);
  EXPECT_EQ(fetchOrFail(port, HttpRequest{.method = "HEAD", .target = "/"}).status, 200);
  EXPECT_TRUE(test->inputs->drain().empty());
}

TEST(WebViewerServerTest, QueuesInputAndSettings) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  const int port = test->server->port();
  EXPECT_EQ(postFromPage(port, "/input?type=drag&button=left&shift=0&dx=0.1&dy=-0.2").status, 204);
  EXPECT_EQ(postFromPage(port, "/input?type=key&key=k").status, 204);
  EXPECT_EQ(postFromPage(port, "/input?type=visualization&name=zmp").status, 204);
  EXPECT_EQ(postFromPage(port, "/settings?width=99999&fps=0").status, 204);
  const HttpReply badKey = postFromPage(port, "/input?type=key&key=9");
  EXPECT_EQ(badKey.status, 400);
  EXPECT_TRUE(absl::StrContains(badKey.body, "'key'")) << badKey.body;
  EXPECT_EQ(postFromPage(port, "/input?type=drag&button=left&shift=0&dx=nan&dy=0").status, 400);
  EXPECT_EQ(postFromPage(port, "/input?type=key&key=a&key=b").status, 400);
  // A repeat with the same value, or the same name spelled otherwise, is a repeat too, and the routes take no body.
  for (const char* absl_nonnull const target :
       {"/input?type=key&key=a&key=a", "/input?type=reset_camera&type=reset_camera", "/settings?fps=20&fps=20",
        "/input?type=key&key=a&key=%61", "/input?type=key&%6Bey=a&key=a"}) {
    const HttpReply repeated = postFromPage(port, target);
    EXPECT_EQ(repeated.status, 400) << target;
    EXPECT_TRUE(absl::StrContains(repeated.body, "more than once")) << target << ": " << repeated.body;
  }
  const HttpReply form =
      fetchOrFail(port, HttpRequest{.method = "POST",
                                    .target = "/input",
                                    .headers = {{"X-MuJoCo-Viewer", "1"}, {"Content-Type", "application/x-www-form-urlencoded"}},
                                    .body = "type=key&key=z"});
  EXPECT_EQ(form.status, 400);
  EXPECT_TRUE(absl::StrContains(form.body, "query parameters only")) << form.body;
  EXPECT_EQ(fetchOrFail(port, HttpRequest{.method = "POST",
                                          .target = "/settings?fps=7",
                                          .headers = {{"X-MuJoCo-Viewer", "1"}, {"Content-Type", "text/plain"}},
                                          .body = "x"})
                .status,
            400);
  const HttpReply badSetting = postFromPage(port, "/settings?quality=high");
  EXPECT_EQ(badSetting.status, 400);
  EXPECT_TRUE(absl::StrContains(badSetting.body, "'quality'")) << badSetting.body;

  const std::vector<ViewerInput> inputs = test->inputs->drain();
  ASSERT_EQ(inputs.size(), 4U);
  EXPECT_EQ(inputs[0].type, ViewerInputType::kDrag);
  EXPECT_DOUBLE_EQ(inputs[0].dx, 0.1);
  EXPECT_DOUBLE_EQ(inputs[0].dy, -0.2);
  EXPECT_EQ(inputs[1].key, 'k');
  EXPECT_EQ(inputs[2].visualizationName, "zmp");
  EXPECT_EQ(inputs[3].type, ViewerInputType::kSettings);
  EXPECT_EQ(inputs[3].width, 3840);
  EXPECT_EQ(inputs[3].fps, 1);
  EXPECT_FALSE(inputs[3].height.has_value());
}

TEST(WebViewerServerTest, AnswersTheState) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  const int port = test->server->port();
  const HttpReply before = get(port, "/state");
  EXPECT_EQ(before.status, 200);
  EXPECT_TRUE(absl::StrContains(before.body, R"("frame":null)")) << before.body;
  ViewerStatus status;
  status.render.width = 640;
  status.visualizations.push_back({.name = "zmp", .hotkey = 'z', .enabled = true, .description = "the ZMP"});
  test->mailbox->publishStatus(status);
  test->mailbox->publish(test_support::opaqueFrame(/*bytes=*/10));
  const HttpReply after = get(port, "/state");
  EXPECT_EQ(after.header("Content-Type"), "application/json");
  EXPECT_TRUE(absl::StrContains(after.body, R"({"clients":0,"viewers":0,"maxClients":4,"droppedInputs":0,"frame":{"sequence":1,)"))
      << after.body;
  EXPECT_TRUE(absl::StrContains(after.body, R"("viewer":{"camera":)")) << after.body;
  EXPECT_TRUE(absl::StrContains(after.body, R"("name":"zmp","hotkey":"z","enabled":true)")) << after.body;
}

TEST(WebViewerServerTest, FrameRequestsAreTheRenderDemand) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  EXPECT_FALSE(test->mailbox->waitForDemand(absl::Milliseconds(200)));
  EXPECT_EQ(get(test->server->port(), "/state").status, 200);
  EXPECT_FALSE(test->mailbox->waitForDemand(absl::ZeroDuration()));
  test->mailbox->publish(test_support::opaqueFrame(/*bytes=*/10));
  EXPECT_EQ(get(test->server->port(), "/frame.jpg").status, 200);
  EXPECT_TRUE(test->mailbox->waitForDemand(absl::ZeroDuration()));
  absl::SleepFor(ViewerFrameMailbox::kDemandLease + absl::Milliseconds(200));
  EXPECT_FALSE(test->mailbox->waitForDemand(absl::ZeroDuration()));
}

TEST(WebViewerServerTest, ClosingTheMailboxEndsWaitingRequests) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  ViewerFrameMailbox* absl_nonnull mailbox = test->mailbox.get();
  mailbox->publish(test_support::opaqueFrame(/*bytes=*/10));
  std::thread closer([mailbox] {
    absl::SleepFor(absl::Milliseconds(200));
    mailbox->close();
  });
  const absl::Time start = absl::Now();
  const HttpReply reply = get(test->server->port(), "/frame.jpg?after=1");
  closer.join();
  EXPECT_EQ(reply.status, 503);
  EXPECT_LT(absl::Now() - start, absl::Seconds(1));
  EXPECT_EQ(get(test->server->port(), "/stream.mjpg").status, 503);
}

// Whether Create() refuses `options` as InvalidArgument, with a mailbox for 4 clients.
bool refused(WebViewerServerOptions options) {
  ViewerFrameMailbox mailbox(/*maxClients=*/4);
  ViewerInputQueue inputs;
  return WebViewerServer::Create(std::move(options), &mailbox, &inputs).status().code() == absl::StatusCode::kInvalidArgument;
}

TEST(WebViewerServerTest, CreateRefusesBadOptions) {
  for (const char* absl_nonnull const address : {"localhost", "300.1.1.1", "", "127.0.0.1:6090", "[::1]"}) {
    WebViewerServerOptions options = test_support::testServerOptions();
    options.address = address;
    EXPECT_TRUE(refused(options)) << address;
  }
  WebViewerServerOptions port = test_support::testServerOptions();
  port.port = 70000;
  EXPECT_TRUE(refused(port));
  WebViewerServerOptions clients = test_support::testServerOptions();
  clients.maxClients = 3;  // the mailbox admits 4
  EXPECT_TRUE(refused(clients));
  for (const char* absl_nonnull const host : {"robot lan", "", "robot/lan", "[robot.lan]", "[fe80::1", "fe80::1::2", ":6090"}) {
    WebViewerServerOptions hosts = test_support::testServerOptions();
    hosts.allowedHosts = {host};
    EXPECT_TRUE(refused(hosts)) << host;
  }
}

// An allowed-host entry with a port, and the name the refusal says to give instead.
struct HostWithPort {
  std::string entry;
  std::string name;
};

// An allowed host with a port would never match a request (the Host check compares names alone), so Create() refuses
// it, naming the flag and the name to give instead.
TEST(WebViewerServerTest, CreateRefusesAnAllowedHostWithAPort) {
  for (const HostWithPort& host :
       {HostWithPort{.entry = "robot.lan:6090", .name = "robot.lan"}, HostWithPort{.entry = "Robot.LAN:", .name = "robot.lan"},
        HostWithPort{.entry = "[fe80::1]:6090", .name = "fe80::1"}}) {
    WebViewerServerOptions options = test_support::testServerOptions();
    options.allowedHosts = {host.entry};
    ViewerFrameMailbox mailbox(/*maxClients=*/4);
    ViewerInputQueue inputs;
    const absl::Status status = WebViewerServer::Create(std::move(options), &mailbox, &inputs).status();
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << host.entry;
    EXPECT_TRUE(absl::StrContains(status.message(), "--web_viewer_allowed_hosts")) << status;
    EXPECT_TRUE(absl::StrContains(status.message(), absl::StrCat("give '", host.name, "', not '", host.entry, "'"))) << status;
  }
}

// An IPv6 allowed host matches with or without the brackets a Host header puts around it, in any case.
TEST(WebViewerServerTest, AnIpv6AllowedHostMatchesWithOrWithoutBrackets) {
  WebViewerServerOptions options = test_support::testServerOptions();
  options.allowedHosts = {"[FE80::1]", "fd00::2", "2001:0DB8:0:0::1", "[FD00:0:0:0:0:0:0:3]", "::ffff:192.168.1.5"};
  absl::StatusOr<TestServer> test = test_support::startTestServer(std::move(options));
  ASSERT_TRUE(test.ok()) << test.status();
  const int port = test->server->port();
  // A browser sends the RFC 5952 form ("[2001:db8::1]", an IPv4-mapped address in hex); either side may spell it
  // otherwise.
  for (const char* absl_nonnull const host : {"[fe80::1]:6090", "[FE80::1]", "[fd00::2]:80", "[fd00::2]", "[2001:db8::1]:6090", "[fd00::3]",
                                              "[::ffff:c0a8:105]", "[::ffff:192.168.1.5]"}) {
    EXPECT_EQ(fetchOrFail(port, HttpRequest{.method = "GET", .target = "/state", .host = host}).status, 200) << host;
  }
  for (const char* absl_nonnull const host : {"[fe80::2]:6090", "fe80::1", "[fd00::2]x", "[2001:db8::2]"}) {
    EXPECT_EQ(fetchOrFail(port, HttpRequest{.method = "GET", .target = "/state", .host = host}).status, 403) << host;
  }
}

TEST(WebViewerServerTest, BindsTheIpv6Loopback) {
  WebViewerServerOptions options = test_support::testServerOptions();
  options.address = "::1";
  absl::StatusOr<TestServer> test = test_support::startTestServer(std::move(options));
  if (absl::IsUnavailable(test.status())) GTEST_SKIP() << "no IPv6 loopback here: " << test.status();
  ASSERT_TRUE(test.ok()) << test.status();
  EXPECT_EQ(test->server->url(), absl::StrCat("http://[::1]:", test->server->port(), "/"));
}

}  // namespace
}  // namespace robot::web_viewer
