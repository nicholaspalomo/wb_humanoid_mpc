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

#include <string>
#include <vector>

#include "web_viewer/ViewerThreadPriority.h"

namespace robot::web_viewer {

/**
 * How WebViewerServer::Create() binds and whom it serves (README.md, "Trust
 * model").
 *
 * Passive data; Create() checks it.
 */
struct WebViewerServerOptions {
  // The numeric IPv4 or IPv6 address to bind. The loopback default serves this
  // machine only; another address (an interface's, or the wildcard 0.0.0.0 or
  // ::) lets other machines watch and steer the camera, with no authentication.
  std::string address = "127.0.0.1";
  // The TCP port; 0 binds an ephemeral one (tests), which
  // WebViewerServer::port() reports. Create() refuses a port above kMaxPort,
  // the largest TCP port. The default is also the MuJoCo viewer's
  // (WebViewerOptions) and the robot's
  // --web_viewer_port's, which take it from here, and the one the deployment
  // and the containers publish.
  static constexpr int kMaxPort = 65'535;
  // LINT.IfChange(web_viewer_port)
  static constexpr int kDefaultPort = 6090;
  // clang-format off
  // LINT.ThenChange(//tools/deploy/session.sh:web_viewer_port, //humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:robot_flags, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/launch/robot.textproto:robot_variables, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/launch/robot.textproto:robot_variables, //robot_models/unitree_g1/g1_centroidal_mpc/launch/robot.textproto:robot_variables, //robot_models/unitree_g1/g1_wb_mpc/launch/robot.textproto:robot_variables, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/launch/robot.textproto:robot_variables, //Makefile:viewer_variable, //.devcontainer/devcontainer.json:web_viewer_port, //docker-compose.bridge.yaml:web_viewer_port, //.devcontainer/README.md:web_viewer_port)
  // clang-format on
  int port = kDefaultPort;
  // Host names besides localhost, 127.0.0.1, ::1 and a non-wildcard `address`
  // that a request's Host header may carry, such as the machine's LAN name or
  // address for a 0.0.0.0 bind. Compared without case. Names only: Create()
  // refuses an entry with a port ("robot.lan:6090", "[fe80::1]:6090"); an IPv6
  // address may be given with or without brackets.
  std::vector<std::string> allowedHosts = {};
  // The most frame clients (a waiting GET /frame.jpg or an open GET
  // /stream.mjpg) at a time, 1 to 64. Must equal the mailbox's
  // ViewerFrameMailbox::maxClients(). The request pool has 3 x maxClients + 2
  // threads.
  static constexpr int kDefaultMaxClients = 4;
  int maxClients = kDefaultMaxClients;
  // The registry names of the visualizations, which POST
  // /input?type=visualization&name= accepts.
  std::vector<std::string> visualizationNames = {};
  // The CPUs of the listen thread and the request pool (placeCurrentThread());
  // empty leaves them unpinned.
  std::vector<int> cores = {};
  // The scheduling of the listen thread and the request pool: SCHED_IDLE by
  // default.
  ViewerThreadPriority threadPriority = ViewerThreadPriority::kIdle;
};

}  // namespace robot::web_viewer
