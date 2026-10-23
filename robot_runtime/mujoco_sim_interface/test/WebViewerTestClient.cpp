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

#include "robot_runtime/mujoco_sim_interface/test/WebViewerTestClient.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "httplib.h"  // NOLINT(build/include_subdir): cpp-httplib is a single header without a directory

namespace robot::mujoco_sim_interface {
namespace {

HttpReply replyOf(const httplib::Result& result) {
  HttpReply reply;
  if (!result) return reply;
  reply.status = result->status;
  reply.body = result->body;
  if (!absl::SimpleAtoi(result->get_header_value("X-Frame-Sequence"), &reply.frameSequence)) reply.frameSequence = 0;
  return reply;
}

httplib::Client client(int port) {
  httplib::Client client("127.0.0.1", port);
  client.set_connection_timeout(/*sec=*/2, /*usec=*/0);
  client.set_read_timeout(/*sec=*/5, /*usec=*/0);
  client.set_write_timeout(/*sec=*/5, /*usec=*/0);
  return client;
}

/** The text after `"key":` in `json`; none without it. */
std::optional<absl::string_view> fieldValue(absl::string_view json, absl::string_view key) {
  const std::string pattern = absl::StrCat("\"", key, "\":");
  const size_t at = json.find(pattern);
  if (at == absl::string_view::npos) return std::nullopt;
  return json.substr(at + pattern.size());
}

}  // namespace

HttpReply httpGet(int port, absl::string_view target) {
  httplib::Client server = client(port);
  return replyOf(server.Get(std::string(target)));
}

HttpReply httpPost(int port, absl::string_view target) {
  httplib::Client server = client(port);
  return replyOf(server.Post(std::string(target), httplib::Headers{{"X-MuJoCo-Viewer", "1"}}, /*body=*/"", /*content_type=*/"text/plain"));
}

std::optional<double> jsonNumberField(absl::string_view json, absl::string_view key) {
  const std::optional<absl::string_view> value = fieldValue(json, key);
  if (!value.has_value()) return std::nullopt;
  const absl::string_view number = value->substr(0, value->find_first_of(",}]"));
  double parsed = 0.0;
  if (!absl::SimpleAtod(absl::StripAsciiWhitespace(number), &parsed)) return std::nullopt;
  return parsed;
}

std::optional<std::string> jsonStringField(absl::string_view json, absl::string_view key) {
  const std::optional<absl::string_view> value = fieldValue(json, key);
  if (!value.has_value() || value->empty() || value->front() != '"') return std::nullopt;
  const size_t end = value->find('"', 1);
  if (end == absl::string_view::npos) return std::nullopt;
  return std::string(value->substr(1, end - 1));
}

std::optional<bool> visualizationEnabled(absl::string_view json, absl::string_view name) {
  const std::string entry = absl::StrCat(R"({"name":")", name, "\"");
  const size_t at = json.find(entry);
  if (at == absl::string_view::npos) return std::nullopt;
  const std::optional<absl::string_view> enabled = fieldValue(json.substr(at), "enabled");
  if (!enabled.has_value()) return std::nullopt;
  return absl::StartsWith(*enabled, "true");
}

MjpegStreamTestClient::MjpegStreamTestClient(int port)
    : thread_([this, port] {
        httplib::Client server = client(port);
        server.Get("/stream.mjpg", [this](const char* absl_nonnull /*data*/, size_t length) {
          bytes_.fetch_add(static_cast<int64_t>(length));
          return !stop_.load();
        });
        finished_.store(true);
      }) {}

MjpegStreamTestClient::~MjpegStreamTestClient() {
  stop_.store(true);
  if (thread_.joinable()) thread_.join();
}

}  // namespace robot::mujoco_sim_interface
