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

#include "humanoid_common_mpc_app/robot/RobotStartup.h"

#include <functional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"
#include "humanoid_common_mpc_app/robot/RobotStack.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {
namespace {

/** Confirms the boot of `directory`; a failure is logged, the process carries on. */
void confirmBoot(RobotConfigDirectory& directory) {
  if (const absl::Status confirmed = directory.confirmBoot(); !confirmed.ok()) {
    LOG(WARNING) << "[RobotStartup] The boot could not be confirmed (" << confirmed.message()
                 << "); the next start falls back to the bundled files.";
  }
}

}  // namespace

absl::StatusOr<RobotConfigDirectory> openRobotConfigDirectory(const RobotAppOptions& options) {
  RobotConfigDirectory::Options directoryOptions;
  directoryOptions.storeDirectory = options.configStoreDirectory;
  directoryOptions.seedPolicy = options.configSeedPolicy;
  directoryOptions.seeds.taskFile = options.files.taskFile;
  directoryOptions.seeds.referenceFile = options.files.referenceFile;
  ASSIGN_OR_RETURN(directoryOptions.seeds.pdGainsFile, existingJointPdGainsFileBeside(options.files.taskFile));
  return RobotConfigDirectory::Open(std::move(directoryOptions));
}

absl::StatusOr<std::string> bundledRobotName(const RobotConfigDirectory& directory) {
  ASSIGN_OR_RETURN(const mpc_config::TaskFile task, loadTaskFile(directory.seeds().taskFile));
  return task.model_settings.robot_name;
}

absl::StatusOr<RobotStack> bringUpRobot(RobotConfigDirectory& directory, const RobotBringUp& bringUp) {
  absl::StatusOr<RobotStack> stack = bringUp(directory);
  if (stack.ok() || !directory.usesStoredCopies()) return stack;
  LOG(ERROR) << "[RobotStartup] The robot did not start on its stored configuration (" << stack.status()
             << "); trying the bundled files, without touching the stored copies.";
  {
    // A trial on the bundle, torn down before the start below binds what it bound (the bus, the backend) again.
    const absl::StatusOr<RobotStack> onBundle = bringUp(directory.bundleInPlace());
    if (!onBundle.ok()) {
      LOG(ERROR) << "[RobotStartup] The robot does not start on the bundled files either (" << onBundle.status()
                 << "): the failure is not the stored copies', which are kept.";
      if (const absl::Status withdrawn = directory.withdrawBootMarker(); !withdrawn.ok()) {
        LOG(WARNING) << "[RobotStartup] The boot marker could not be withdrawn (" << withdrawn.message()
                     << "); the next start falls back to the bundled files.";
      }
      return stack.status();
    }
  }
  LOG(ERROR) << "[RobotStartup] The robot starts on the bundled files: rejecting the stored copies and starting on them.";
  RETURN_IF_ERROR(directory.rejectStoredCopies());
  return bringUp(directory);
}

absl::Status runRobot(const RobotAppOptions& options, const RobotSetUp& setUpRobot, const std::function<bool()>& shutdownRequested) {
  ASSIGN_OR_RETURN(RobotConfigDirectory directory, openRobotConfigDirectory(options));
  const RobotBringUp bringUp = [&setUpRobot](const RobotConfigDirectory& files) -> absl::StatusOr<RobotStack> {
    // A stack that fails to start is destroyed on the way out, which stops what it started.
    ASSIGN_OR_RETURN(RobotStack stack, setUpRobot(files));
    RETURN_IF_ERROR(stack.bus->start());
    RETURN_IF_ERROR(stack.process->start());
    return stack;
  };
  ASSIGN_OR_RETURN(RobotStack stack, bringUpRobot(directory, bringUp));
  // Until SIGINT or SIGTERM, or until a cycle of the realtime loop throws (the backend is then in its safe state and the
  // binary exits with a failure, which the container's restart policy answers).
  bool confirmed = false;
  const absl::Status ended = stack.process->runUntilShutdown(shutdownRequested, kBootConfirmationTime, [&directory, &confirmed]() {
    confirmBoot(directory);
    confirmed = true;
  });
  // A clean stop is no failed boot.
  if (ended.ok() && !confirmed) confirmBoot(directory);
  LOG(INFO) << "The robot process has stopped.";
  return ended;
}

}  // namespace ocs2::humanoid
