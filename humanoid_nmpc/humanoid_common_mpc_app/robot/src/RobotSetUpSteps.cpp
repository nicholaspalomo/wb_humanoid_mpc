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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_common_mpc_app/robot/RobotSetUpSteps.h"

#include <filesystem>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc_app/robot/ConfigFileStore.h"
#include "humanoid_common_mpc_app/robot/RobotStartup.h"
#include "humanoid_common_mpc_app/robot/TaskFileWatcher.h"
#include "humanoid_common_mpc_app/robot/config/robot/RobotProcessSettingsFromConfig.h"
#include "humanoid_mpc_ipc/RemoteMpcLink.h"

namespace ocs2::humanoid {

absl::StatusOr<RobotConfiguration> loadRobotConfiguration(const RobotConfigDirectory& directory) {
  const RobotConfigDirectory::Files& files = directory.files();
  RobotConfiguration configuration;
  // The task file's write time first: a save while the process starts is applied by the task file watcher's first poll.
  configuration.taskFileReadAt = TaskFileWatcher::writeTimeOf(files.taskFile);
  ASSIGN_OR_RETURN(configuration.files, loadRobotConfigFiles(files));
  ASSIGN_OR_RETURN(configuration.bundledRobotName, bundledRobotName(directory));
  absl::StatusOr<RobotProcessSettings> settings = robotProcessSettingsFromConfig(configuration.files.task);
  if (!settings.ok()) return withConfigFile(settings.status(), files.taskFile);
  configuration.settings = *std::move(settings);
  return configuration;
}

RobotConfigurationCheckContext robotConfigurationCheckContext(
    const RobotAppOptions& options,
    std::string robotName,
    const ModelSettings& modelSettings,
    const JointPdGainsDefaults& pdGainsDefaults,
    const RobotBackendOptions& backendOptions,
    const robot::model::ContactEstimatorRegistry* absl_nonnull contactEstimators) {
  RobotConfigurationCheckContext context;
  context.robotName = std::move(robotName);
  context.mpcJointNames = modelSettings.mpcModelJointNames;
  context.otherJointNames = modelSettings.fixedJointNames;
  context.pdGainsDefaults = pdGainsDefaults;
  context.contactEstimators = contactEstimators;
  context.backendName = options.backend;
  context.backendOptions = backendOptions;
  return context;
}

RobotProcess::Hooks robotProcessHooks(RemoteMpcLinkAdapter* absl_nonnull remoteLink,
                                      const RobotConfigDirectory::Files& storedFiles,
                                      RobotConfigurationCheckContext context) {
  RobotProcess::Hooks hooks;
  hooks.takeViewerAnnotations = [remoteLink](msgs::ViewerAnnotations& annotations) {
    return remoteLink->remote().takeAnnotations(annotations);
  };
  hooks.fillLinkStatistics = [remoteLink](humanoid_mpc_msgs::LoopTiming& loopTiming) {
    const ipc::RemoteMpcLink::Statistics statistics = remoteLink->remote().statistics();
    loopTiming.set_stale_policies_dropped(statistics.stalePoliciesDropped);
    loopTiming.set_policy_age_s(statistics.policyAge);
  };
  hooks.validateConfigFile = [storedFiles, context = std::move(context)](const ConfigFileCandidate& candidate) {
    return checkConfigFileCandidate(storedFiles, candidate, context);
  };
  return hooks;
}

RobotProcess::Config robotProcessConfig(const RobotAppOptions& options,
                                        const RobotConfigDirectory& directory,
                                        const RobotConfiguration& configuration,
                                        const ModelSettings& modelSettings) {
  RobotProcess::Config config;
  config.realtimePriority = options.realtimePriority;
  config.realtimeCores = options.realtimeCores;
  config.backendCores = options.backendCores;
  config.settings = configuration.settings;
  config.taskFile = directory.files().taskFile;
  config.taskFileReadAt = configuration.taskFileReadAt;
  config.robotName = modelSettings.robotName;
  config.taskFileIdentity = directory.taskFileIdentity();
  config.configStore = configFileStoreConfig(directory, modelSettings.robotName);
  return config;
}

}  // namespace ocs2::humanoid
