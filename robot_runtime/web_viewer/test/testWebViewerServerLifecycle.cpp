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

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "robot_runtime/web_viewer/test/WebViewerTestSupport.h"
#include "web_viewer/ViewerFrameMailbox.h"
#include "web_viewer/ViewerInputQueue.h"
#include "web_viewer/WebViewerServer.h"
#include "web_viewer/WebViewerServerOptions.h"

namespace robot::web_viewer {
namespace {

using test_support::HttpReply;
using test_support::HttpRequest;
using test_support::HttpResponseReader;
using test_support::TestConnection;
using test_support::TestServer;

// A port no server holds: an ephemeral one, bound and released.
int freePort() {
  absl::StatusOr<TestServer> probe = test_support::startTestServer(test_support::testServerOptions());
  EXPECT_TRUE(probe.ok()) << probe.status();
  return probe.ok() ? probe->server->port() : 0;
}

TEST(WebViewerServerLifecycleTest, ThousandServersStartAndStopPromptly) {
  ViewerFrameMailbox mailbox(/*maxClients=*/4);
  ViewerInputQueue inputs;
  absl::Duration slowestStop = absl::ZeroDuration();
  const absl::Time start = absl::Now();
  for (int cycle = 0; cycle < 1000; ++cycle) {
    absl::StatusOr<std::unique_ptr<WebViewerServer>> server = WebViewerServer::Create(test_support::testServerOptions(), &mailbox, &inputs);
    ASSERT_TRUE(server.ok()) << "cycle " << cycle << ": " << server.status();
    const absl::Time stopping = absl::Now();
    server->reset();
    slowestStop = std::max(slowestStop, absl::Now() - stopping);
  }
  EXPECT_LT(slowestStop, absl::Seconds(1));
  LOG(INFO) << "1000 servers in " << absl::Now() - start << ", the slowest stop " << slowestStop;
}

TEST(WebViewerServerLifecycleTest, APortIsFreeAgainAtOnce) {
  const int port = freePort();
  ASSERT_GT(port, 0);
  WebViewerServerOptions options = test_support::testServerOptions();
  options.port = port;
  for (int round = 0; round < 3; ++round) {
    absl::StatusOr<TestServer> test = test_support::startTestServer(options);
    ASSERT_TRUE(test.ok()) << "round " << round << ": " << test.status();
    EXPECT_EQ(test->server->port(), port);
    // A served request leaves the server's side of the connection in TIME_WAIT, which SO_REUSEADDR binds over.
    absl::StatusOr<HttpReply> reply = test_support::fetch(port, HttpRequest{.method = "GET", .target = "/state"});
    ASSERT_TRUE(reply.ok()) << reply.status();
    EXPECT_EQ(reply->status, 200);
  }
}

TEST(WebViewerServerLifecycleTest, ASecondServerOnThePortIsRefused) {
  absl::StatusOr<TestServer> first = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(first.ok()) << first.status();
  const int port = first->server->port();
  for (const std::string& address : {std::string("127.0.0.1"), std::string("0.0.0.0")}) {
    WebViewerServerOptions options = test_support::testServerOptions();
    options.address = address;
    options.port = port;
    absl::StatusOr<TestServer> second = test_support::startTestServer(options);
    ASSERT_FALSE(second.ok()) << address;
    EXPECT_EQ(second.status().code(), absl::StatusCode::kUnavailable);
    EXPECT_TRUE(absl::StrContains(second.status().message(), absl::StrCat(":", port))) << second.status();
    EXPECT_TRUE(absl::StrContains(second.status().message(), "--web_viewer_port")) << second.status();
    EXPECT_TRUE(absl::StrContains(second.status().message(), "make kill-sims")) << second.status();
  }
  // The first server still serves.
  absl::StatusOr<HttpReply> reply = test_support::fetch(port, HttpRequest{.method = "GET", .target = "/state"});
  ASSERT_TRUE(reply.ok()) << reply.status();
  EXPECT_EQ(reply->status, 200);
}

TEST(WebViewerServerLifecycleTest, InputIsAnsweredWhileStreamsAndIdleConnectionsHoldThePool) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  const int port = test->server->port();
  const int maxClients = test->mailbox->maxClients();
  std::vector<std::unique_ptr<TestConnection>> connections;
  std::vector<std::unique_ptr<HttpResponseReader>> readers;
  // maxClients streams, each holding its request thread for as long as it is open.
  for (int stream = 0; stream < maxClients; ++stream) {
    absl::StatusOr<std::unique_ptr<TestConnection>> connection = TestConnection::Connect(port, /*receiveBufferBytes=*/std::nullopt);
    ASSERT_TRUE(connection.ok()) << connection.status();
    ASSERT_TRUE((*connection)->send(test_support::formatRequest(port, HttpRequest{.method = "GET", .target = "/stream.mjpg"})).ok());
    readers.push_back(std::make_unique<HttpResponseReader>(connection->get()));
    absl::StatusOr<HttpReply> head = readers.back()->readHead(absl::Seconds(5), /*headRequest=*/false);
    ASSERT_TRUE(head.ok()) << head.status();
    ASSERT_EQ(head->status, 200);
    connections.push_back(*std::move(connection));
  }
  // 2 x maxClients keep-alive connections that made one request each and now idle, as a browser's do.
  const absl::Time idleSince = absl::Now();
  for (int idle = 0; idle < 2 * maxClients; ++idle) {
    absl::StatusOr<std::unique_ptr<TestConnection>> connection = TestConnection::Connect(port, /*receiveBufferBytes=*/std::nullopt);
    ASSERT_TRUE(connection.ok()) << connection.status();
    ASSERT_TRUE(
        (*connection)->send(test_support::formatRequest(port, HttpRequest{.method = "GET", .target = "/state", .keepAlive = true})).ok());
    HttpResponseReader reader(connection->get());
    absl::StatusOr<HttpReply> reply = reader.readResponse(absl::Seconds(5));
    ASSERT_TRUE(reply.ok()) << reply.status();
    EXPECT_EQ(reply->status, 200);
    connections.push_back(*std::move(connection));
  }
  const absl::Time posting = absl::Now();
  absl::StatusOr<HttpReply> input =
      test_support::fetch(port, HttpRequest{.method = "POST", .target = "/input?type=reset_camera", .headers = {{"X-MuJoCo-Viewer", "1"}}});
  const absl::Duration answeredIn = absl::Now() - posting;
  ASSERT_TRUE(input.ok()) << input.status();
  EXPECT_EQ(input->status, 204);
  EXPECT_LT(answeredIn, absl::Milliseconds(200));
  EXPECT_LT(posting - idleSince, absl::Seconds(1)) << "the idle connections timed out before the POST: the test proves nothing";

  // The server closes an idle keep-alive connection after 1 s.
  absl::StatusOr<std::string> idle = connections.back()->receive(absl::Seconds(3));
  EXPECT_TRUE(absl::IsOutOfRange(idle.status())) << idle.status();
}

TEST(WebViewerServerLifecycleTest, DestroyingTheServerEndsOpenStreams) {
  absl::StatusOr<TestServer> test = test_support::startTestServer(test_support::testServerOptions());
  ASSERT_TRUE(test.ok()) << test.status();
  const int port = test->server->port();
  test->mailbox->publish(test_support::opaqueFrame(/*bytes=*/1000));
  absl::StatusOr<std::unique_ptr<TestConnection>> stream = TestConnection::Connect(port, /*receiveBufferBytes=*/std::nullopt);
  absl::StatusOr<std::unique_ptr<TestConnection>> longPoll = TestConnection::Connect(port, /*receiveBufferBytes=*/std::nullopt);
  ASSERT_TRUE(stream.ok()) << stream.status();
  ASSERT_TRUE(longPoll.ok()) << longPoll.status();
  ASSERT_TRUE((*stream)->send(test_support::formatRequest(port, HttpRequest{.method = "GET", .target = "/stream.mjpg"})).ok());
  ASSERT_TRUE((*longPoll)->send(test_support::formatRequest(port, HttpRequest{.method = "GET", .target = "/frame.jpg?after=1"})).ok());
  ASSERT_TRUE(test_support::waitUntil([&test] { return test->mailbox->activeClients() == 2; }, absl::Seconds(2)));
  const absl::Time stopping = absl::Now();
  test->server.reset();
  EXPECT_LT(absl::Now() - stopping, absl::Seconds(1));
  EXPECT_EQ(test->mailbox->activeClients(), 0);
}

}  // namespace
}  // namespace robot::web_viewer
