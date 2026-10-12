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

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>

#include "absl/strings/string_view.h"

/*
 * A browser's side of the web viewer for the tests: requests to its server on 127.0.0.1, field reads of its /state
 * JSON, and a client that keeps a /stream.mjpg open.
 */

namespace robot::mujoco_sim_interface {

/** One answer of the viewer's server; status 0 when the request failed (no connection, a timeout). */
struct HttpReply {
  int status = 0;
  std::string body;
  int64_t frameSequence = 0;  // X-Frame-Sequence, 0 without one
};

/** GET `target` (such as "/frame.jpg?after=3") from the server on 127.0.0.1:`port`. */
HttpReply httpGet(int port, absl::string_view target);

/** POST `target` (such as "/input?type=key&key=k") as the page does it, with the X-MuJoCo-Viewer header. */
HttpReply httpPost(int port, absl::string_view target);

/** The number `"key":<number>` of the /state JSON `json`, the first one in it; none without one. */
std::optional<double> jsonNumberField(absl::string_view json, absl::string_view key);

/** The string `"key":"<text>"` of the /state JSON `json`, the first one in it; none without one. */
std::optional<std::string> jsonStringField(absl::string_view json, absl::string_view key);

/** Whether the visualization `name` of the /state JSON `json` is enabled; none when it lists no such visualization. */
std::optional<bool> visualizationEnabled(absl::string_view json, absl::string_view name);

/**
 * A client that opens GET /stream.mjpg on 127.0.0.1:`port` on a thread of its own and reads it until the server ends
 * it (or stop()). Thread-safe.
 */
class MjpegStreamTestClient {
 public:
  explicit MjpegStreamTestClient(int port);
  MjpegStreamTestClient(const MjpegStreamTestClient&) = delete;
  MjpegStreamTestClient& operator=(const MjpegStreamTestClient&) = delete;
  MjpegStreamTestClient(MjpegStreamTestClient&&) = delete;
  MjpegStreamTestClient& operator=(MjpegStreamTestClient&&) = delete;
  /** Stops reading and joins the thread. */
  ~MjpegStreamTestClient();

  /** The bytes of the stream received so far. */
  int64_t bytesReceived() const { return bytes_.load(); }
  /** True once the stream has ended. */
  bool finished() const { return finished_.load(); }

 private:
  std::atomic<bool> stop_ = false;
  std::atomic<bool> finished_ = false;
  std::atomic<int64_t> bytes_ = 0;
  std::thread thread_;
};

}  // namespace robot::mujoco_sim_interface
