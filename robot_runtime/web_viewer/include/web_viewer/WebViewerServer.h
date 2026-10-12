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

#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"

#include "web_viewer/ViewerFrameMailbox.h"
#include "web_viewer/ViewerInputQueue.h"
#include "web_viewer/WebViewerServerOptions.h"

namespace robot::web_viewer {

/**
 * The HTTP server of the MuJoCo web viewer (README.md): it serves the page, hands the frames of a ViewerFrameMailbox
 * to browsers that pull them (GET /frame.jpg?after=<sequence>) or stream them (GET /stream.mjpg), answers the
 * viewer's status (GET /state), and queues the browsers' camera input (POST /input, POST /settings) for the render
 * thread. It never touches the simulation: everything it can change is in the ViewerInputQueue.
 *
 * Threads: a listen thread and a pool of 3 x maxClients + 2 request threads, all placed on `cores` at `threadPriority`
 * (placeCurrentThread()). A request thread blocks only on its own socket (2 s write timeout, 5 s read timeout) and on
 * the mailbox, never the render thread.
 *
 * Every request passes the checks of README.md's trust model before it is routed or counted: the Host header, the
 * Fetch-Metadata header and, for a POST, the X-MuJoCo-Viewer and Origin headers.
 *
 * Lifecycle: Create() binds, starts listening and returns once the server accepts connections; there is no separate
 * start, so a server that is bound but does not listen never exists. The destructor stops accepting, ends the open
 * requests (a waiting one within 250 ms, a blocked write within the write timeout) and joins every thread; the port
 * can be bound again at once. Thread-safe: port() and url() may be called from any thread.
 */
class WebViewerServer {
 public:
  /**
   * Checks `options`, binds `options.address`:`options.port` (SO_REUSEADDR only, so a second process on the same port
   * fails here) and starts serving. `mailbox` and `inputs` are retained and must outlive the server. InvalidArgument
   * for bad options; Unavailable when the address cannot be bound (the port is in use, or the address is not this
   * machine's), naming the address, the port and the fix; Internal when a thread cannot start.
   */
  static absl::StatusOr<std::unique_ptr<WebViewerServer>> Create(WebViewerServerOptions options,
                                                                 ViewerFrameMailbox* absl_nonnull mailbox,
                                                                 ViewerInputQueue* absl_nonnull inputs);

  ~WebViewerServer();
  WebViewerServer(const WebViewerServer&) = delete;
  WebViewerServer& operator=(const WebViewerServer&) = delete;

  /** Returns the bound port: options.port, or the ephemeral port chosen for options.port = 0. */
  int port() const;

  /** Returns the address to open the viewer at, such as "http://127.0.0.1:6090/" ("[::1]" for an IPv6 address). */
  std::string url() const;

 private:
  // The server's state and handlers (WebViewerServer.cpp), at a fixed address that the handlers capture.
  class Core;

  explicit WebViewerServer(std::unique_ptr<Core> core);

  std::unique_ptr<Core> core_;
};

}  // namespace robot::web_viewer
