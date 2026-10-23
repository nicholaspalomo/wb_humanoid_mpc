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

#include "robot_runtime/web_viewer/test/WebViewerTestSupport.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "web_viewer/EncodedFrame.h"
#include "web_viewer/JpegEncoder.h"
#include "web_viewer/ViewerFrameMailbox.h"
#include "web_viewer/ViewerInputQueue.h"
#include "web_viewer/ViewerThreadPriority.h"
#include "web_viewer/WebViewerServer.h"
#include "web_viewer/WebViewerServerOptions.h"

namespace robot::web_viewer::test_support {
namespace {

constexpr absl::string_view kHeadEnd = "\r\n\r\n";
constexpr absl::string_view kLineEnd = "\r\n";
constexpr absl::string_view kPartStart = "--mujocoframe\r\n";

}  // namespace

std::vector<uint8_t> gradientImage(int width, int height, int phase) {
  std::vector<uint8_t> rgb(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
  for (int row = 0; row < height; ++row) {
    for (int column = 0; column < width; ++column) {
      const size_t pixel = (static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(column)) * 3;
      rgb[pixel] = static_cast<uint8_t>((column * 255 / std::max(1, width - 1) + phase) % 256);
      rgb[pixel + 1] = static_cast<uint8_t>((row * 255 / std::max(1, height - 1) + 2 * phase) % 256);
      rgb[pixel + 2] = static_cast<uint8_t>((128 + 3 * phase) % 256);
    }
  }
  return rgb;
}

EncodedFrame gradientFrame(int width, int height, int phase, double simTime) {
  absl::StatusOr<std::unique_ptr<JpegEncoder>> encoder = JpegEncoder::Create();
  EncodedFrame frame = {.sequence = 0, .width = width, .height = height, .simTime = simTime, .publishTime = absl::UnixEpoch(), .jpeg = {}};
  if (!encoder.ok()) return frame;
  absl::StatusOr<std::string> jpeg =
      (*encoder)->encode(gradientImage(width, height, phase), width, height, /*quality=*/80, RowOrder::kTopDown);
  if (jpeg.ok()) frame.jpeg = *std::move(jpeg);
  return frame;
}

EncodedFrame opaqueFrame(int64_t bytes) {
  return EncodedFrame{.sequence = 0,
                      .width = 1280,
                      .height = 720,
                      .simTime = 0.0,
                      .publishTime = absl::UnixEpoch(),
                      .jpeg = std::string(static_cast<size_t>(bytes), 'j')};
}

bool waitUntil(absl::FunctionRef<bool()> condition, absl::Duration timeout) {
  const absl::Time deadline = absl::Now() + timeout;
  while (!condition()) {
    if (absl::Now() >= deadline) return false;
    absl::SleepFor(absl::Milliseconds(5));
  }
  return true;
}

WebViewerServerOptions testServerOptions() {
  return WebViewerServerOptions{.address = "127.0.0.1",
                                .port = 0,
                                .allowedHosts = {},
                                .maxClients = 4,
                                .visualizationNames = {"contact_timeline", "zmp"},
                                .cores = {},
                                .threadPriority = ViewerThreadPriority::kInherited};
}

absl::StatusOr<TestServer> startTestServer(WebViewerServerOptions options) {
  TestServer test = {.mailbox = std::make_unique<ViewerFrameMailbox>(options.maxClients),
                     .inputs = std::make_unique<ViewerInputQueue>(),
                     .server = nullptr};
  absl::StatusOr<std::unique_ptr<WebViewerServer>> server =
      WebViewerServer::Create(std::move(options), test.mailbox.get(), test.inputs.get());
  if (!server.ok()) return server.status();
  test.server = *std::move(server);
  return test;
}

absl::StatusOr<std::unique_ptr<TestConnection>> TestConnection::Connect(int port, std::optional<int> receiveBufferBytes) {
  const int descriptor = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, /*protocol=*/0);
  if (descriptor < 0) return absl::ErrnoToStatus(errno, "socket");
  std::unique_ptr<TestConnection> connection = absl::WrapUnique(new TestConnection(descriptor));
  if (receiveBufferBytes.has_value()) {
    const int bytes = *receiveBufferBytes;
    if (setsockopt(descriptor, SOL_SOCKET, SO_RCVBUF, &bytes, sizeof(bytes)) != 0) return absl::ErrnoToStatus(errno, "SO_RCVBUF");
  }
  const int noDelay = 1;
  setsockopt(descriptor, IPPROTO_TCP, TCP_NODELAY, &noDelay, sizeof(noDelay));
  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(descriptor, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("connect to 127.0.0.1:", port));
  }
  return connection;
}

TestConnection::~TestConnection() {
  close(socket_);
}

absl::Status TestConnection::send(absl::string_view bytes) const {
  while (!bytes.empty()) {
    const ssize_t sent = ::send(socket_, bytes.data(), bytes.size(), MSG_NOSIGNAL);
    if (sent < 0) {
      if (errno == EINTR) continue;
      return absl::ErrnoToStatus(errno, "send");
    }
    bytes.remove_prefix(static_cast<size_t>(sent));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> TestConnection::receive(absl::Duration timeout) {
  pollfd descriptor = {.fd = socket_, .events = POLLIN, .revents = 0};
  const int ready = poll(&descriptor, /*nfds=*/1, static_cast<int>(absl::ToInt64Milliseconds(timeout)));
  if (ready < 0) return absl::ErrnoToStatus(errno, "poll");
  if (ready == 0) return absl::DeadlineExceededError(absl::StrCat("nothing arrived within ", absl::FormatDuration(timeout)));
  std::array<char, 65536> buffer = {};
  const ssize_t received = recv(socket_, buffer.data(), buffer.size(), /*flags=*/0);
  if (received < 0) return absl::ErrnoToStatus(errno, "recv");
  if (received == 0) return absl::OutOfRangeError("the peer closed the connection");
  return std::string(buffer.data(), static_cast<size_t>(received));
}

std::optional<std::string> HttpReply::header(absl::string_view name) const {
  for (const std::pair<std::string, std::string>& entry : headers) {
    if (absl::EqualsIgnoreCase(entry.first, name)) return entry.second;
  }
  return std::nullopt;
}

std::string formatRequest(int port, const HttpRequest& request) {
  std::string text = absl::StrCat(request.method, " ", request.target,
                                  " HTTP/1.1\r\nHost: ", request.host.empty() ? absl::StrCat("127.0.0.1:", port) : request.host, "\r\n");
  for (const std::pair<std::string, std::string>& header : request.headers) {
    absl::StrAppend(&text, header.first, ": ", header.second, "\r\n");
  }
  if (!request.keepAlive) text += "Connection: close\r\n";
  if (!request.body.empty() || request.method == "POST") absl::StrAppend(&text, "Content-Length: ", request.body.size(), "\r\n");
  absl::StrAppend(&text, "\r\n", request.body);
  return text;
}

absl::StatusOr<HttpReply> HttpResponseReader::readHead(absl::Duration timeout, bool headRequest) {
  const absl::Time deadline = absl::Now() + timeout;
  while (!absl::StrContains(buffered_, kHeadEnd)) {
    absl::StatusOr<std::string> received = connection_->receive(std::max(deadline - absl::Now(), absl::ZeroDuration()));
    if (!received.ok()) return received.status();
    buffered_ += *received;
  }
  const size_t headEnd = buffered_.find(kHeadEnd);
  const std::string head = buffered_.substr(0, headEnd);
  buffered_.erase(0, headEnd + kHeadEnd.size());
  std::vector<std::string> lines = absl::StrSplit(head, kLineEnd);
  HttpReply reply;
  const std::vector<std::string> statusLine = absl::StrSplit(lines[0], absl::MaxSplits(' ', 2));
  if (statusLine.size() < 2 || !absl::SimpleAtoi(statusLine[1], &reply.status)) {
    return absl::InternalError(absl::StrCat("not an HTTP status line: '", lines[0], "'"));
  }
  for (size_t index = 1; index < lines.size(); ++index) {
    const size_t colon = lines[index].find(':');
    if (colon == std::string::npos) continue;
    reply.headers.emplace_back(lines[index].substr(0, colon), std::string(absl::StripAsciiWhitespace(lines[index].substr(colon + 1))));
  }
  const std::optional<std::string> encoding = reply.header("Transfer-Encoding");
  const std::optional<std::string> length = reply.header("Content-Length");
  ended_ = false;
  chunkDataEnded_ = false;
  remaining_ = 0;
  if (headRequest || reply.status == 204 || reply.status == 304) {
    mode_ = BodyMode::kNone;
    ended_ = true;
  } else if (encoding.has_value() && absl::EqualsIgnoreCase(*encoding, "chunked")) {
    mode_ = BodyMode::kChunked;
  } else if (length.has_value()) {
    mode_ = BodyMode::kLength;
    if (!absl::SimpleAtoi(*length, &remaining_)) return absl::InternalError(absl::StrCat("bad Content-Length '", *length, "'"));
    ended_ = remaining_ == 0;
  } else {
    mode_ = BodyMode::kUntilClose;
  }
  return reply;
}

std::string HttpResponseReader::decodeBuffered() {
  std::string decoded;
  switch (mode_) {
    case BodyMode::kNone:
      return decoded;
    case BodyMode::kUntilClose:
      decoded.swap(buffered_);
      return decoded;
    case BodyMode::kLength: {
      const size_t taken = std::min(buffered_.size(), static_cast<size_t>(remaining_));
      decoded = buffered_.substr(0, taken);
      buffered_.erase(0, taken);
      remaining_ -= static_cast<int64_t>(taken);
      if (remaining_ == 0) ended_ = true;
      return decoded;
    }
    case BodyMode::kChunked:
      while (!ended_) {
        if (chunkDataEnded_) {
          if (buffered_.size() < kLineEnd.size()) break;
          buffered_.erase(0, kLineEnd.size());
          chunkDataEnded_ = false;
        }
        if (remaining_ == 0) {
          const size_t lineEnd = buffered_.find(kLineEnd);
          if (lineEnd == std::string::npos) break;
          int64_t size = 0;
          const bool parsed = absl::SimpleHexAtoi(buffered_.substr(0, lineEnd), &size);
          buffered_.erase(0, lineEnd + kLineEnd.size());
          // A malformed chunk size ends the body; the test then sees it short.
          if (!parsed || size <= 0) {
            ended_ = true;  // the final CRLF after the last chunk is left unread
            break;
          }
          remaining_ = size;
        }
        const size_t taken = std::min(buffered_.size(), static_cast<size_t>(remaining_));
        if (taken == 0) break;
        decoded.append(buffered_, 0, taken);
        buffered_.erase(0, taken);
        remaining_ -= static_cast<int64_t>(taken);
        if (remaining_ == 0) chunkDataEnded_ = true;
      }
      return decoded;
  }
  return decoded;
}

absl::StatusOr<std::string> HttpResponseReader::readBody(absl::Duration timeout) {
  std::string decoded = decodeBuffered();
  if (!decoded.empty()) return decoded;
  if (ended_) return absl::OutOfRangeError("the body has ended");
  const absl::Time deadline = absl::Now() + timeout;
  while (decoded.empty() && !ended_) {
    absl::StatusOr<std::string> received = connection_->receive(std::max(deadline - absl::Now(), absl::ZeroDuration()));
    if (!received.ok()) {
      if (absl::IsOutOfRange(received.status()) && mode_ == BodyMode::kUntilClose) ended_ = true;
      return received.status();
    }
    buffered_ += *received;
    decoded = decodeBuffered();
  }
  return decoded;
}

absl::StatusOr<HttpReply> HttpResponseReader::readResponse(absl::Duration timeout) {
  const absl::Time deadline = absl::Now() + timeout;
  absl::StatusOr<HttpReply> reply = readHead(timeout, /*headRequest=*/false);
  if (!reply.ok()) return reply.status();
  while (true) {
    absl::StatusOr<std::string> body = readBody(std::max(deadline - absl::Now(), absl::ZeroDuration()));
    if (!body.ok()) {
      if (absl::IsOutOfRange(body.status())) return reply;
      return body.status();
    }
    reply->body += *body;
  }
}

absl::StatusOr<HttpReply> fetch(int port, const HttpRequest& request) {
  absl::StatusOr<std::unique_ptr<TestConnection>> connection = TestConnection::Connect(port, /*receiveBufferBytes=*/std::nullopt);
  if (!connection.ok()) return connection.status();
  absl::Status sent = (*connection)->send(formatRequest(port, request));
  if (!sent.ok()) return sent;
  HttpResponseReader reader(connection->get());
  if (request.method == "HEAD") return reader.readHead(absl::Seconds(10), /*headRequest=*/true);
  return reader.readResponse(absl::Seconds(10));
}

absl::StatusOr<std::optional<MjpegPart>> MjpegPartParser::nextPart() {
  if (buffered_.size() < kPartStart.size()) return std::optional<MjpegPart>();
  if (!absl::StartsWith(buffered_, kPartStart)) return absl::InternalError("a part does not start with --mujocoframe");
  const size_t headEnd = buffered_.find(kHeadEnd);
  if (headEnd == std::string::npos) return std::optional<MjpegPart>();
  const std::vector<std::string> lines = absl::StrSplit(buffered_.substr(kPartStart.size(), headEnd - kPartStart.size()), kLineEnd);
  int64_t length = -1;
  MjpegPart part;
  for (const std::string& line : lines) {
    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    const absl::string_view name = absl::string_view(line).substr(0, colon);
    const absl::string_view value = absl::StripAsciiWhitespace(absl::string_view(line).substr(colon + 1));
    if (absl::EqualsIgnoreCase(name, "Content-Length") && !absl::SimpleAtoi(value, &length)) return absl::InternalError("bad part length");
    if (absl::EqualsIgnoreCase(name, "X-Frame-Sequence") && !absl::SimpleAtoi(value, &part.sequence)) {
      return absl::InternalError("bad part sequence");
    }
  }
  if (length < 0) return absl::InternalError("a part without Content-Length");
  const size_t bodyStart = headEnd + kHeadEnd.size();
  const size_t partEnd = bodyStart + static_cast<size_t>(length) + kLineEnd.size();
  if (buffered_.size() < partEnd) return std::optional<MjpegPart>();
  part.jpeg = buffered_.substr(bodyStart, static_cast<size_t>(length));
  buffered_.erase(0, partEnd);
  return std::optional<MjpegPart>(std::move(part));
}

}  // namespace robot::web_viewer::test_support
