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
  std::string address = "127.0.0.1";  // interface the server binds; 0.0.0.0 or :: opts in to every interface
  // LINT.IfChange(web_viewer_port)
  int port = 6090;  // [0, 65535]; 0 binds an ephemeral port (tests), see webViewerPort()
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/RobotAppFlags.cpp:web_viewer_port, //robot_runtime/web_viewer/include/web_viewer/WebViewerServerOptions.h:web_viewer_port)
  // clang-format on
  std::vector<std::string> allowedHosts;  // extra host names a request's Host may carry (a LAN name, say)
  int maxClients = 4;                     // concurrent frame requests and streams; more get 503
  int width = 1280;                       // [px] initial frame size; a page may change it up to maxWidth x maxHeight
  int height = 720;                       // [px]
  int maxWidth = 1920;                    // [px] size of the offscreen buffer, checked against the GL limits at start-up
  int maxHeight = 1080;                   // [px]
  int fps = 30;                           // target frame rate while a client watches; nothing is rendered without one
  int quality = 70;                       // JPEG quality
  int msaaSamples = 0;                    // offscreen multisampling; 0 is fastest on llvmpipe
  int shadowSize = 2048;                  // [px] shadow map size; 0 turns shadows off
  std::vector<int> cores;                 // CPU cores of the viewer's threads; empty leaves them unpinned
  // The scheduling of the viewer's threads: SCHED_IDLE, so that a physics or MPC thread on their cores preempts them.
  web_viewer::ViewerThreadPriority threadPriority = web_viewer::ViewerThreadPriority::kIdle;
  std::optional<int> eglDeviceId;                   // EGL device index (MUJOCO_EGL_DEVICE_ID); unset picks a GPU, else llvmpipe
  std::string eglLibrary = "libEGL.so.1";           // the EGL library dlopen()ed; another name only in tests
  absl::Duration startTimeout = absl::Seconds(10);  // how long start() waits for the render thread's EGL and GL set-up
};

}  // namespace robot::mujoco_sim_interface
