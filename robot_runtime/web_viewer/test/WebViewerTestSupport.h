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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

#include "web_viewer/EncodedFrame.h"
#include "web_viewer/ViewerFrameMailbox.h"
#include "web_viewer/ViewerInputQueue.h"
#include "web_viewer/WebViewerServer.h"
#include "web_viewer/WebViewerServerOptions.h"

namespace robot::web_viewer::test_support {

/** Returns a width x height RGB gradient, top row first, whose colors shift with `phase`. */
std::vector<uint8_t> gradientImage(int width, int height, int phase);

/** Returns gradientImage(width, height, phase) as a JPEG frame (quality 80) with `simTime`, ready to publish. */
EncodedFrame gradientFrame(int width, int height, int phase, double simTime);

/** Returns a frame of `bytes` arbitrary bytes, for the tests that only move frames (the server never decodes them). */
EncodedFrame opaqueFrame(int64_t bytes);

/** Returns true as soon as `condition` holds, polling every 5 ms; false when `timeout` passes first. */
bool waitUntil(absl::FunctionRef<bool()> condition, absl::Duration timeout);

// A server on its own mailbox and input queue. The members are destroyed in reverse order: the server first.
struct TestServer {
  std::unique_ptr<ViewerFrameMailbox> mailbox;
  std::unique_ptr<ViewerInputQueue> inputs;
  std::unique_ptr<WebViewerServer> server;
};

/** Returns the options of a test server: an ephemeral port of 127.0.0.1, 4 clients, two visualizations. */
WebViewerServerOptions testServerOptions();

/** Starts a server with `options` and a mailbox for options.maxClients clients. */
absl::StatusOr<TestServer> startTestServer(WebViewerServerOptions options);

/** A blocking TCP connection to 127.0.0.1 for the tests, closed on destruction. Not thread-safe. */
class TestConnection {
 public:
  /** Connects to 127.0.0.1:`port`, with SO_RCVBUF set to `receiveBufferBytes` first when given. */
  static absl::StatusOr<std::unique_ptr<TestConnection>> Connect(int port, std::optional<int> receiveBufferBytes);

  ~TestConnection();
  TestConnection(const TestConnection&) = delete;
  TestConnection& operator=(const TestConnection&) = delete;

  /** Sends all of `bytes`. */
  absl::Status send(absl::string_view bytes) const;

  /**
   * Returns the bytes that arrive within `timeout` (at least one): DeadlineExceeded when none do, OutOfRange when the
   * peer has closed the connection.
   */
  absl::StatusOr<std::string> receive(absl::Duration timeout);

 private:
  explicit TestConnection(int socket) : socket_(socket) {}

  int socket_;
};

// One HTTP/1.1 request of a test. The Host header is `host`, or 127.0.0.1:<port> when it is empty.
struct HttpRequest {
  std::string method = "GET";
  std::string target = "/";
  std::string host = "";
  std::vector<std::pair<std::string, std::string>> headers = {};
  std::string body = "";
  // Whether to ask for the connection to stay open (no "Connection: close").
  bool keepAlive = false;
};

// An HTTP response; header names are compared without case.
struct HttpReply {
  int status = 0;
  std::vector<std::pair<std::string, std::string>> headers = {};
  std::string body = "";

  /** Returns the first value of header `name`, or nullopt. */
  std::optional<std::string> header(absl::string_view name) const;
};

/** Returns `request` as the bytes of an HTTP/1.1 request to 127.0.0.1:`port`. */
std::string formatRequest(int port, const HttpRequest& request);

/**
 * Reads HTTP responses from one connection: the head, then the body, with Content-Length, chunked or up to the close.
 * Reusable for the next response of a keep-alive connection. Retains `connection`, which must outlive the reader.
 */
class HttpResponseReader {
 public:
  explicit HttpResponseReader(TestConnection* absl_nonnull connection) : connection_(connection) {}

  /** Reads the status line and the headers of the next response (`body` empty). `headRequest` means no body follows. */
  absl::StatusOr<HttpReply> readHead(absl::Duration timeout, bool headRequest);

  /** Returns the next decoded body bytes, waiting up to `timeout` for some; OutOfRange once the body has ended. */
  absl::StatusOr<std::string> readBody(absl::Duration timeout);

  /** Reads the next whole response. */
  absl::StatusOr<HttpReply> readResponse(absl::Duration timeout);

 private:
  enum class BodyMode {
    kNone,
    kLength,
    kChunked,
    kUntilClose,
  };

  // Decodes what buffered_ holds of the body.
  std::string decodeBuffered();

  TestConnection* absl_nonnull connection_;
  std::string buffered_;
  BodyMode mode_ = BodyMode::kNone;
  int64_t remaining_ = 0;
  bool chunkDataEnded_ = false;
  bool ended_ = true;
};

/** Sends `request` on a new connection to 127.0.0.1:`port` and reads the whole response. */
absl::StatusOr<HttpReply> fetch(int port, const HttpRequest& request);

// One part of a GET /stream.mjpg.
struct MjpegPart {
  int64_t sequence = 0;
  std::string jpeg = "";
};

/** Splits the parts out of a multipart/x-mixed-replace body (boundary "mujocoframe") as its bytes arrive. */
class MjpegPartParser {
 public:
  void append(absl::string_view bytes) { buffered_.append(bytes.data(), bytes.size()); }

  /** Returns the next complete part, or nullopt while it has not fully arrived; Internal for a malformed part. */
  absl::StatusOr<std::optional<MjpegPart>> nextPart();

 private:
  std::string buffered_;
};

}  // namespace robot::web_viewer::test_support
