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

#include "humanoid_common_mpc_app/robot/RobotAppFlags.h"

#include <cstdlib>
#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/ThreadAffinity.h"
#include "humanoid_common_mpc_app/robot/RobotAppOptions.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"
#include "robot_realtime/RealtimeThread.h"

// The environment variable that picks the EGL device the web viewer renders on (WebViewerOptions::eglDeviceId).
constexpr char kEglDeviceVariable[] = "MUJOCO_EGL_DEVICE_ID";

// LINT.IfChange(robot_flags)
ABSL_FLAG(std::string, mjcf_file, "", "The robot's MuJoCo scene (urdf/*.xml); required by --backend=mujoco.");
ABSL_FLAG(std::string, ipc_node, "robot", "The bus node this process publishes as (a node of the network file).");
ABSL_FLAG(std::string, backend, "mujoco", "The robot backend, by name (RobotBackendRegistry): mujoco is the MuJoCo simulator.");
ABSL_FLAG(int,
          realtime_priority,
          ::robot::realtime::kTimeSharingPriority,
          "SCHED_FIFO priority of the realtime thread, 1-99, with the process's memory locked; 0 keeps it on the time-sharing "
          "scheduler. Needs CAP_SYS_NICE and CAP_IPC_LOCK (or rtprio and memlock limits); a step it may not take is reported "
          "and skipped.");
ABSL_FLAG(std::string,
          realtime_cores,
          "default",
          "Cores of the realtime thread: default (ThreadAffinity.h's MRT cores), none, or a list such as 4,5.");
ABSL_FLAG(std::string,
          backend_cores,
          "default",
          "Cores of the backend's threads (the MuJoCo physics): default (ThreadAffinity.h's simulation cores), none, or a list.");
ABSL_FLAG(std::string,
          viewer,
          "",
          "The MuJoCo backend's viewer by name, or a comma-separated list tried in order: glfw (a window on DISPLAY), web (a page "
          "at http://<--web_viewer_address>:<--web_viewer_port>, rendered offscreen with EGL), none. web,none falls back to none "
          "when the web viewer cannot start; web alone refuses to start then. Empty: the mujoco_viewer of --plot_config, else web.");
ABSL_FLAG(std::string,
          plot_config,
          "",
          "A plot configuration (a humanoid_rerun_viewer_proto.PlotConfig textproto, the Rerun bridge's --plot_config) whose "
          "mujoco_viewer names the MuJoCo viewer when --viewer is empty; one that lists none gives web. Read strictly at start-up, "
          "and only then: a file that does not parse or names an unknown viewer is refused. Empty: none.");
ABSL_FLAG(std::string,
          web_viewer_address,
          "127.0.0.1",
          "The address the web viewer binds: 127.0.0.1 serves this machine only (another reaches it through an SSH tunnel); "
          "0.0.0.0 or an interface's address serves anyone who can reach it, without authentication.");
ABSL_FLAG(int,
          web_viewer_port,
          ::robot::mujoco_sim_interface::WebViewerOptions::kDefaultPort,
          "The web viewer's TCP port; 0 binds any free one, which the log names.");
ABSL_FLAG(std::string,
          web_viewer_allowed_hosts,
          "",
          "Comma-separated host names the web viewer answers besides localhost, 127.0.0.1, ::1 and its bound address (a "
          "machine's LAN name with --web_viewer_address=0.0.0.0), without ports; other Host headers are refused.");
ABSL_FLAG(std::string,
          viewer_cores,
          "default",
          "Cores of the web viewer's threads: default (ThreadAffinity.h's viewer cores, never the MPC's), none, or a list.");
ABSL_FLAG(std::string,
          config_store_dir,
          "",
          "The persistent directory of this robot configuration, where the GUI's saves are stored and the robot reads its "
          "files from (the bundled --task_file, --reference_file and the PD gains beside them seed it); empty: read the "
          "given files in place.");
ABSL_FLAG(std::string,
          config_seed,
          std::string(ocs2::humanoid::kWhenBundleChangesSeedPolicyName),
          "When a stored copy is replaced by the bundled file, by name: when_bundle_changes (a deploy changed the file), "
          "every_start, or never.");
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:robot_flags, //humanoid_nmpc/docs/distributed_runtime/README.md:command_line_flags)
// clang-format on

namespace ocs2::humanoid {

namespace {

/** The highest SCHED_FIFO priority of Linux, the largest --realtime_priority. */
constexpr int kMaxRealtimePriority = 99;

/** The viewer flags as given, with MUJOCO_EGL_DEVICE_ID. */
ViewerFlagValues viewerFlagValues() {
  ViewerFlagValues values;
  values.viewer = absl::GetFlag(FLAGS_viewer);
  values.plotConfig = absl::GetFlag(FLAGS_plot_config);
  values.webViewerAddress = absl::GetFlag(FLAGS_web_viewer_address);
  values.webViewerPort = absl::GetFlag(FLAGS_web_viewer_port);
  values.webViewerAllowedHosts = absl::GetFlag(FLAGS_web_viewer_allowed_hosts);
  values.viewerCores = absl::GetFlag(FLAGS_viewer_cores);
  // Read once, on the main thread before the process starts its threads (getenv races with a setenv elsewhere).
  const char* absl_nullable eglDevice = std::getenv(kEglDeviceVariable);
  if (eglDevice != nullptr) values.eglDeviceId = std::string(eglDevice);
  return values;
}

}  // namespace

void exportSoftwareRasterizerThreads(const robot::mujoco_sim_interface::ViewerSettings& viewer) {
  const std::optional<std::string> threads = softwareRasterizerThreadsToSet(viewer, std::getenv(kSoftwareRasterizerThreadsVariable));
  if (!threads.has_value()) return;
  if (setenv(kSoftwareRasterizerThreadsVariable, threads->c_str(), /*replace=*/1) != 0) {
    LOG(WARNING) << "[robot] " << kSoftwareRasterizerThreadsVariable << " could not be set; llvmpipe starts a rasterizer thread per CPU.";
    return;
  }
  LOG(INFO) << "[robot] " << describeSoftwareRasterizerThreads(viewer, *threads);
}

absl::StatusOr<RobotAppOptions> robotAppOptionsFromFlags(const SystemCoreAllocation& cores) {
  RobotAppOptions options;
  options.robotName = absl::GetFlag(FLAGS_robot_name);
  options.files = node::mpcFilesFromFlags();
  options.mjcfFile = absl::GetFlag(FLAGS_mjcf_file);
  options.networkConfig = absl::GetFlag(FLAGS_network_config);
  options.ipcNode = absl::GetFlag(FLAGS_ipc_node);
  options.backend = absl::GetFlag(FLAGS_backend);
  options.realtimePriority = absl::GetFlag(FLAGS_realtime_priority);
  options.configStoreDirectory = absl::GetFlag(FLAGS_config_store_dir);
  ASSIGN_OR_RETURN(options.configSeedPolicy, configSeedPolicyFromName(absl::GetFlag(FLAGS_config_seed)));
  ASSIGN_OR_RETURN(options.realtimeCores, parseCoreList(absl::GetFlag(FLAGS_realtime_cores), cores.mrtCores));
  ASSIGN_OR_RETURN(options.backendCores, parseCoreList(absl::GetFlag(FLAGS_backend_cores), cores.simCores));
  ASSIGN_OR_RETURN(options.viewer, viewerSettingsFromFlags(viewerFlagValues(), cores.viewerCores));

  RETURN_IF_ERROR(node::validateFileFlag("--task_file", options.files.taskFile));
  RETURN_IF_ERROR(node::validateFileFlag("--reference_file", options.files.referenceFile));
  RETURN_IF_ERROR(node::validateFileFlag("--urdf_file", options.files.urdfFile));
  if (options.backend == kMujocoBackendName) {
    RETURN_IF_ERROR(node::validateFileFlag("--mjcf_file", options.mjcfFile));
  }
  if (options.realtimePriority < 0 || options.realtimePriority > kMaxRealtimePriority) {
    return absl::InvalidArgumentError(
        absl::StrCat("--realtime_priority=", options.realtimePriority, ": give 0 (off) or a priority of 1-", kMaxRealtimePriority));
  }
  if (options.ipcNode.empty()) {
    return absl::InvalidArgumentError("--ipc_node is empty; the robot publishes as a node of the network file (robot)");
  }
  return options;
}

}  // namespace ocs2::humanoid
