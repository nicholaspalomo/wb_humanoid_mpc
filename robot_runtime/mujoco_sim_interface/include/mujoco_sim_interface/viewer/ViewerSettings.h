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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"

namespace robot::mujoco_sim_interface {

/**
 * Which viewer the simulator starts, as MujocoSimConfig::viewer and the robot
 * app's options carry it. Passive data; validateViewerSettings() says whether
 * it is usable.
 *
 * `names` is a preference order of registry names (ViewerNames.h): initSim()
 * starts the first one that starts, and fails
 * (MujocoSimInterface::viewerStatus()) when none does. `web,none` therefore
 * falls back to no viewer when the web viewer cannot start, and `web` alone
 * refuses to run without it.
 */
struct ViewerSettings {
  std::vector<std::string> names = {std::string(kGlfwViewerName)};
  WebViewerOptions web;  // used only when `names` lists kWebViewerName
};

/**
 * Splits a comma-separated viewer list, such as `web,none`, and checks it with
 * checkViewerNames(). Spaces around a name are ignored. InvalidArgument says
 * what is wrong and lists the valid names.
 */
absl::StatusOr<std::vector<std::string>> parseViewerNames(
    absl::string_view list);

/**
 * Checks a viewer preference list: it is not empty, every name is a known
 * viewer (InvalidArgument lists them), none repeats, and nothing follows `none`
 * (which always starts) or `glfw` (which cannot fail at start-up: a window it
 * cannot open is logged and the simulation runs on), since a later entry could
 * never be used.
 */
absl::Status checkViewerNames(absl::Span<const std::string> names);

/** Checks `settings.names` with checkViewerNames() and every WebViewerOptions
 * field against its range. */
absl::Status validateViewerSettings(const ViewerSettings& settings);

/**
 * Returns the number of llvmpipe rasterizer threads (Mesa's LP_NUM_THREADS)
 * that fits the web viewer's cores: one per core, and at least 2. Without
 * LP_NUM_THREADS, llvmpipe starts one per CPU of the machine, all of them on
 * those cores.
 */
int softwareRasterizerThreads(const WebViewerOptions& web);

}  // namespace robot::mujoco_sim_interface
