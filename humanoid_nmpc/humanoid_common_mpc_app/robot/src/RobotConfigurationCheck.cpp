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

#include "humanoid_common_mpc_app/robot/RobotConfigurationCheck.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/robot/JointPdGainsFromConfig.h"
#include "humanoid_common_mpc/mrt/ContactEstimateIntake.h"
#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/robot/TelemetrySinkRegistry.h"
#include "humanoid_common_mpc_app/robot/config/robot/RobotProcessSettingsFromConfig.h"

namespace ocs2::humanoid {
namespace {

/** The contact estimator `name` of the task file at `taskSource`, against the registry and its probe. */
absl::Status checkContactEstimatorName(const std::string& name,
                                       const std::string& taskSource,
                                       const RobotConfigurationCheckContext& context) {
  if (context.contactEstimators == nullptr) return absl::OkStatus();
  const robot::model::ContactEstimatorRegistry& registry = *context.contactEstimators;
  const std::string canonical = robot::model::ContactEstimatorRegistry::canonicalName(name);
  if (!registry.has(canonical)) {
    return absl::InvalidArgumentError(absl::StrCat(taskSource, ": contact_estimator: there is no contact estimator '", name,
                                                   "' on this robot. Available: ", registry.availableNames(), "."));
  }
  if (!context.backendOptions.initialState.has_value()) return absl::OkStatus();
  const std::shared_ptr<robot::model::ContactEstimator> estimator = registry.create(canonical);
  return withConfigFile(checkContactEstimator(*estimator, *context.backendOptions.initialState, canonical),
                        absl::StrCat(taskSource, ": contact_estimator"));
}

/** Every name of the task file's telemetry_sinks against the sinks there are. */
absl::Status checkTelemetrySinks(const RobotProcessSettings& settings, const std::string& taskSource) {
  const TelemetrySinkRegistry sinks;
  for (const std::string& name : settings.telemetrySinks) {
    if (!sinks.has(name)) {
      return absl::InvalidArgumentError(
          absl::StrCat(taskSource, ": telemetry_sinks: there is no telemetry sink '", name, "'. Available: ", sinks.availableNames(), "."));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<RobotConfigFiles> loadRobotConfigFiles(const RobotConfigDirectory::Files& paths) {
  RobotConfigFiles files;
  ASSIGN_OR_RETURN(files.task, loadTaskFile(paths.taskFile));
  ASSIGN_OR_RETURN(files.reference, loadReferenceFile(paths.referenceFile));
  ASSIGN_OR_RETURN(files.pdGains, loadJointPdGainsFile(paths.pdGainsFile));
  ASSIGN_OR_RETURN(files.contactPlanning, loadContactPlanningFileBeside(paths.taskFile));
  files.taskSource = paths.taskFile;
  files.referenceSource = paths.referenceFile;
  files.pdGainsSource = paths.pdGainsFile;
  return files;
}

absl::Status checkRobotConfiguration(const RobotConfigFiles& files, const RobotConfigurationCheckContext& context) {
  if (!context.robotName.empty() && files.task.model_settings.robot_name != context.robotName) {
    return absl::FailedPreconditionError(absl::StrCat(files.taskSource, ": model_settings.robot_name is '",
                                                      files.task.model_settings.robot_name, "', and this robot is '", context.robotName,
                                                      "'"));
  }
  absl::StatusOr<RobotProcessSettings> settings = robotProcessSettingsFromConfig(files.task);
  if (!settings.ok()) return withConfigFile(settings.status(), files.taskSource);
  if (context.checkFormulation) RETURN_IF_ERROR(context.checkFormulation(files));
  if (!context.mpcJointNames.empty() || !context.otherJointNames.empty()) {
    // As the controller's loader reads it.
    const absl::StatusOr<JointPdGains> gains =
        jointPdGainsFromConfig(files.pdGains, context.pdGainsDefaults, context.mpcJointNames, context.otherJointNames);
    if (!gains.ok()) return withConfigFile(gains.status(), files.pdGainsSource);
  }
  RETURN_IF_ERROR(checkContactEstimatorName(settings->contactEstimator, files.taskSource, context));
  RETURN_IF_ERROR(checkTelemetrySinks(*settings, files.taskSource));
  if (!context.backendName.empty()) {
    RobotBackendOptions options = context.backendOptions;
    options.simulator = settings->simulator;
    RETURN_IF_ERROR(withConfigFile(RobotBackendRegistry().checkOptions(context.backendName, options), files.taskSource));
  }
  return absl::OkStatus();
}

absl::Status checkConfigFileCandidate(const RobotConfigDirectory::Files& paths,
                                      const ConfigFileCandidate& candidate,
                                      const RobotConfigurationCheckContext& context) {
  ASSIGN_OR_RETURN(RobotConfigFiles files, loadRobotConfigFiles(paths));
  if (candidate.task.has_value()) {
    files.task = *candidate.task;
    files.taskSource = candidate.source;
  }
  if (candidate.reference.has_value()) {
    files.reference = *candidate.reference;
    files.referenceSource = candidate.source;
  }
  if (candidate.pdGains.has_value()) {
    files.pdGains = *candidate.pdGains;
    files.pdGainsSource = candidate.source;
  }
  return checkRobotConfiguration(files, context);
}

}  // namespace ocs2::humanoid
