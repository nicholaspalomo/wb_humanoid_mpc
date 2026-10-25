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

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc_app/node/MpcFiles.h"
#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/ViewerSettings.h"

namespace ocs2::humanoid {

/**
 * The cores a --realtime_cores or --backend_cores value names: `default` is
 * `defaultCores` (ThreadAffinity.h's allocation for this machine), `none` (or
 * empty) pins nothing, otherwise a comma-separated list of CPU numbers ("4,5").
 * InvalidArgument for anything else.
 */
absl::StatusOr<std::vector<int>> parseCoreList(
    absl::string_view value, const std::vector<int>& defaultCores);

/** The values of the robot binaries' viewer flags (RobotAppFlags.h) as given,
 * before viewerSettingsFromFlags() checks them. */
struct ViewerFlagValues {
  /**
   * --viewer: a viewer's registry name, or a comma-separated preference list
   * such as `web,none`. Empty (or blank): not given, so the plot
   * configuration's mujoco_viewer chooses (viewerNamesFromFlags()).
   */
  std::string viewer;
  /** --plot_config: the plot configuration whose mujoco_viewer names the viewer
   * when `viewer` is empty; empty: none. */
  std::string plotConfig;
  /** --web_viewer_address: the interface the web viewer binds. */
  std::string webViewerAddress;
  /** --web_viewer_port: its TCP port; 0 is any free one. */
  int webViewerPort = 0;
  /** --web_viewer_allowed_hosts: comma-separated host names the web viewer
   * answers besides loopback's. */
  std::string webViewerAllowedHosts;
  /** --viewer_cores: `default`, `none` or a list of CPU numbers
   * (parseCoreList()). */
  std::string viewerCores;
  /** MUJOCO_EGL_DEVICE_ID when the environment sets it: the index of the EGL
   * device the web viewer renders on. */
  std::optional<std::string> eglDeviceId;
};

// LINT.IfChange(default_viewer)
/** The MuJoCo viewer of the robot binaries when neither --viewer nor a plot
 * configuration's mujoco_viewer names one. */
inline constexpr absl::string_view kDefaultViewerName =
    robot::mujoco_sim_interface::kWebViewerName;
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/proto/plot_config.proto:default_mujoco_viewer, //humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto:default_mujoco_viewer, //tools/deploy/session.sh:default_viewer_hint, //humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:robot_flags)
// clang-format on

/**
 * Returns the viewer preference list of the robot binaries, by precedence:
 *   1. --viewer when it is given (not blank), checked with parseViewerNames();
 *   2. otherwise the mujoco_viewer of the plot configuration --plot_config
 * names, when one is named: the file is read strictly
 * (nproto::LoadTextprotoFile(), a humanoid_rerun_viewer_proto.PlotConfig) and
 * its list checked with checkViewerNames(); a file that lists none gives
 * kDefaultViewerName;
 *   3. otherwise kDefaultViewerName.
 * The plot configuration is read only in case 2. Errors name the flag:
 * InvalidArgument for a list that is refused or a file that does not parse (at
 * its <file>:<line>:<column>), and the error of a file that cannot be read
 * (NotFound, ...). Logs where the list came from in cases 2 and 3. Reads a
 * file: call it at start-up only.
 */
absl::StatusOr<std::vector<std::string>> viewerNamesFromFlags(
    const ViewerFlagValues& flags);

/**
 * The MuJoCo backend's viewer settings from the viewer flags, checked: the
 * viewer list (viewerNamesFromFlags(): --viewer, else the plot configuration's
 * mujoco_viewer, else the web viewer; known names, no repeat, nothing after
 * `none` or `glfw`), the port (0-65535), the address (not empty), the allowed
 * hosts (split at commas, blanks dropped), the cores
 * (`default` is `defaultViewerCores`) and MUJOCO_EGL_DEVICE_ID (a device
 * index), and then every web viewer option (validateViewerSettings()). An error
 * names the flag or the variable.
 */
absl::StatusOr<robot::mujoco_sim_interface::ViewerSettings>
viewerSettingsFromFlags(const ViewerFlagValues& flags,
                        const std::vector<int>& defaultViewerCores);

/** The environment variable that bounds llvmpipe's (Mesa's software OpenGL's)
 * rasterizer threads. */
inline constexpr char kSoftwareRasterizerThreadsVariable[] = "LP_NUM_THREADS";

/**
 * Returns the LP_NUM_THREADS value the robot process sets, given the one the
 * environment has (`current`, null when it has none): when `viewer` lists the
 * web viewer and the environment sets no value (null or empty), one rasterizer
 * thread per viewer core (softwareRasterizerThreads()) instead of llvmpipe's
 * one per CPU, all of them on the viewer's cores; otherwise nullopt, which
 * leaves the environment's value, or Mesa's default for the GLFW window.
 */
std::optional<std::string> softwareRasterizerThreadsToSet(
    const robot::mujoco_sim_interface::ViewerSettings& viewer,
    const char* absl_nullable current);

/**
 * Returns the log line of the LP_NUM_THREADS value `threads` the robot process
 * set (softwareRasterizerThreadsToSet()): "LP_NUM_THREADS=4: llvmpipe's
 * rasterizer threads, for 4 viewer core(s)." for a web viewer pinned to cores,
 * and "LP_NUM_THREADS=2: llvmpipe's rasterizer threads (viewer cores
 * unrestricted)." when `viewer` pins it to none.
 */
std::string describeSoftwareRasterizerThreads(
    const robot::mujoco_sim_interface::ViewerSettings& viewer,
    absl::string_view threads);

/** Everything the robot binaries take from their command line
 * (RobotAppFlags.h). */
struct RobotAppOptions {
  std::string robotName;
  /** --task_file, --reference_file, --urdf_file (--gait_file is the MPC
   * node's). */
  node::MpcFiles files;
  std::string mjcfFile;
  std::string networkConfig;
  std::string ipcNode;
  std::string backend;
  int realtimePriority = 0;
  std::vector<int> realtimeCores;
  std::vector<int> backendCores;
  /** The MuJoCo backend's viewer by name and the web viewer's options
   * (--viewer, --web_viewer_*, --viewer_cores). */
  robot::mujoco_sim_interface::ViewerSettings viewer;
  /** The persistent directory of this robot configuration
   * (RobotConfigDirectory); empty: read the files in place. */
  std::string configStoreDirectory;
  /** When the stored copies are replaced by the bundled files (--config_seed).
   */
  ConfigSeedPolicy configSeedPolicy = ConfigSeedPolicy::kWhenBundleChanges;
};

}  // namespace ocs2::humanoid
