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

#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"

// The defaults of the flags below that are numbers or booleans.
constexpr int kNotRealtime = 0;
constexpr bool kHeadlessByDefault = false;

// LINT.IfChange(robot_flags)
ABSL_FLAG(std::string, mjcf_file, "", "The robot's MuJoCo scene (urdf/*.xml); required by --backend=mujoco.");
ABSL_FLAG(std::string, ipc_node, "robot", "The bus node this process publishes as (a node of the network file).");
ABSL_FLAG(std::string, backend, "mujoco", "The robot backend, by name (RobotBackendRegistry): mujoco is the MuJoCo simulator.");
ABSL_FLAG(int,
          realtime_priority,
          kNotRealtime,
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
          mpc_link,
          "",
          "Retired: the robot reaches its MPC over the bus only (the MPC node). Any value is refused at start-up.");
ABSL_FLAG(bool, headless, kHeadlessByDefault, "Run the MuJoCo backend without its viewer window (no display, the robot-runtime image).");
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
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:robot_flags)

namespace ocs2::humanoid {

absl::StatusOr<RobotAppOptions> robotAppOptionsFromFlags(const std::vector<int>& defaultRealtimeCores,
                                                         const std::vector<int>& defaultBackendCores) {
  RobotAppOptions options;
  options.robotName = absl::GetFlag(FLAGS_robot_name);
  options.files = node::mpcFilesFromFlags();
  options.mjcfFile = absl::GetFlag(FLAGS_mjcf_file);
  options.networkConfig = absl::GetFlag(FLAGS_network_config);
  options.ipcNode = absl::GetFlag(FLAGS_ipc_node);
  options.backend = absl::GetFlag(FLAGS_backend);
  options.realtimePriority = absl::GetFlag(FLAGS_realtime_priority);
  options.headless = absl::GetFlag(FLAGS_headless);
  options.configStoreDirectory = absl::GetFlag(FLAGS_config_store_dir);
  ASSIGN_OR_RETURN(options.configSeedPolicy, configSeedPolicyFromName(absl::GetFlag(FLAGS_config_seed)));
  RETURN_IF_ERROR(checkRetiredMpcLinkFlag(absl::GetFlag(FLAGS_mpc_link)));
  ASSIGN_OR_RETURN(options.realtimeCores, parseCoreList(absl::GetFlag(FLAGS_realtime_cores), defaultRealtimeCores));
  ASSIGN_OR_RETURN(options.backendCores, parseCoreList(absl::GetFlag(FLAGS_backend_cores), defaultBackendCores));

  RETURN_IF_ERROR(node::validateFileFlag("--task_file", options.files.taskFile));
  RETURN_IF_ERROR(node::validateFileFlag("--reference_file", options.files.referenceFile));
  RETURN_IF_ERROR(node::validateFileFlag("--urdf_file", options.files.urdfFile));
  if (options.backend == kMujocoBackendName) {
    RETURN_IF_ERROR(node::validateFileFlag("--mjcf_file", options.mjcfFile));
  }
  if (options.realtimePriority < 0 || options.realtimePriority > 99) {
    return absl::InvalidArgumentError(
        absl::StrCat("--realtime_priority=", options.realtimePriority, ": give 0 (off) or a priority of 1-99"));
  }
  if (options.ipcNode.empty()) {
    return absl::InvalidArgumentError("--ipc_node is empty; the robot publishes as a node of the network file (robot)");
  }
  return options;
}

}  // namespace ocs2::humanoid
