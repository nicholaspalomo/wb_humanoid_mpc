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

#include <optional>
#include <string>
#include <vector>

#include "absl/time/time.h"

#include "web_viewer/ViewerThreadPriority.h"
#include "web_viewer/WebViewerServerOptions.h"

namespace robot::mujoco_sim_interface {

// The ranges validateViewerSettings() (ViewerSettings.h) accepts for the fields below.
inline constexpr int kMinWebViewerWidth = 160;    // [px]
inline constexpr int kMinWebViewerHeight = 120;   // [px]
inline constexpr int kMaxWebViewerWidth = 3840;   // [px] hard cap of maxWidth
inline constexpr int kMaxWebViewerHeight = 2160;  // [px] hard cap of maxHeight
inline constexpr int kMinWebViewerFps = 1;
inline constexpr int kMaxWebViewerFps = 60;
inline constexpr int kMinWebViewerQuality = 10;
inline constexpr int kMaxWebViewerQuality = 95;
inline constexpr int kMaxWebViewerClients = 16;
inline constexpr int kMaxWebViewerMsaaSamples = 16;
inline constexpr int kMaxWebViewerShadowSize = 8192;  // [px]

/**
 * How the web viewer (WebMujocoViewer, `--viewer=web`) renders and serves its frames. Passive data, read only when the
 * viewer list names `web`; validateViewerSettings() checks every field against the ranges above.
 *
 * The server binds `address` (loopback by default: see the trust model in robot_runtime/web_viewer/README.md) and
 * answers only requests whose Host is loopback, the bound address or one of `allowedHosts`.
 */
struct WebViewerOptions {
  // The largest port, and the defaults of the fields below; the port and the client count are the server's.
  static constexpr int kMaxPort = web_viewer::WebViewerServerOptions::kMaxPort;
  static constexpr int kDefaultPort = web_viewer::WebViewerServerOptions::kDefaultPort;
  static constexpr int kDefaultMaxClients = web_viewer::WebViewerServerOptions::kDefaultMaxClients;
  static constexpr int kDefaultWidth = 1280;      // [px]
  static constexpr int kDefaultHeight = 720;      // [px]
  static constexpr int kDefaultMaxWidth = 1920;   // [px]
  static constexpr int kDefaultMaxHeight = 1080;  // [px]
  static constexpr int kDefaultFps = 30;
  static constexpr int kDefaultQuality = 70;
  static constexpr int kDefaultShadowSize = 2048;  // [px]
  static constexpr absl::Duration kDefaultStartTimeout = absl::Seconds(10);

  std::string address = "127.0.0.1";      // interface the server binds; 0.0.0.0 or :: opts in to every interface
  int port = kDefaultPort;                // [0, kMaxPort]; 0 binds an ephemeral port (tests), see webViewerPort()
  std::vector<std::string> allowedHosts;  // extra host names a request's Host may carry (a LAN name, say)
  int maxClients = kDefaultMaxClients;    // concurrent frame requests and streams; more get 503
  int width = kDefaultWidth;              // [px] initial frame size; a page may change it up to maxWidth x maxHeight
  int height = kDefaultHeight;            // [px]
  int maxWidth = kDefaultMaxWidth;        // [px] size of the offscreen buffer, checked against the GL limits at start-up
  int maxHeight = kDefaultMaxHeight;      // [px]
  int fps = kDefaultFps;                  // target frame rate while a client watches; nothing is rendered without one
  int quality = kDefaultQuality;          // JPEG quality
  int msaaSamples = 0;                    // offscreen multisampling; 0 is fastest on llvmpipe
  int shadowSize = kDefaultShadowSize;    // [px] shadow map size; 0 turns shadows off
  std::vector<int> cores;                 // CPU cores of the viewer's threads; empty leaves them unpinned
  // The scheduling of the viewer's threads: SCHED_IDLE, so that a physics or MPC thread on their cores preempts them.
  web_viewer::ViewerThreadPriority threadPriority = web_viewer::ViewerThreadPriority::kIdle;
  std::optional<int> eglDeviceId;          // EGL device index (MUJOCO_EGL_DEVICE_ID); unset picks a GPU, else llvmpipe
  std::string eglLibrary = "libEGL.so.1";  // the EGL library dlopen()ed; another name only in tests
  // How long start() waits for the render thread's EGL and GL set-up to succeed; on the timeout it returns once the
  // set-up step running then has ended (WebMujocoViewer::start()).
  absl::Duration startTimeout = kDefaultStartTimeout;
};

}  // namespace robot::mujoco_sim_interface
