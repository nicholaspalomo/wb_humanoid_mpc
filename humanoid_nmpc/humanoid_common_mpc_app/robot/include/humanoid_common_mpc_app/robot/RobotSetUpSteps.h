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

#include <filesystem>
#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_common_mpc_app/robot/RemoteMpcLinkAdapter.h"
#include "humanoid_common_mpc_app/robot/RobotAppOptions.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"
#include "humanoid_common_mpc_app/robot/RobotConfigurationCheck.h"
#include "humanoid_common_mpc_app/robot/RobotProcess.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "robot_model/ContactEstimatorRegistry.h"

/**
 * The steps of a robot binary's set-up that do not depend on its formulation, shared by the setUpRobot() of
 * humanoid_centroidal_mpc_robot and humanoid_wb_mpc_robot, which keep their formulation's models, controller and checks.
 * For the main thread, before the start.
 */
namespace ocs2::humanoid {

/** What a set-up reads of a robot configuration's files, once. */
struct RobotConfiguration {
  // The task file's write time, taken before it was read (RobotProcess::Config::taskFileReadAt).
  std::optional<std::filesystem::file_time_type> taskFileReadAt;
  RobotConfigFiles files;
  // model_settings.robot_name of the bundled task file (bundledRobotName()): what every stored and saved one must be.
  std::string bundledRobotName;
  // The robot process's settings of the task file.
  RobotProcessSettings settings;
};

/** Returns the configuration of the files of `directory` (RobotConfigDirectory::files()); the errors name the file. */
absl::StatusOr<RobotConfiguration> loadRobotConfiguration(const RobotConfigDirectory& directory);

/**
 * Returns the checks a save is held to and the set-up runs (checkRobotConfiguration()) on the robot being set up: the
 * robot `robotName`, the joints of `modelSettings`, the PD gains `pdGainsDefaults` of the formulation's controller (its
 * pdGainsDefaults(): what its loader reads the file with), the backend --backend names with `backendOptions`, and the
 * backend's contact estimators `contactEstimators`, which the context keeps: they must outlive it. The formulation adds
 * its own check (RobotConfigurationCheckContext::checkFormulation).
 */
RobotConfigurationCheckContext robotConfigurationCheckContext(const RobotAppOptions& options,
                                                              std::string robotName,
                                                              const ModelSettings& modelSettings,
                                                              const JointPdGainsDefaults& pdGainsDefaults,
                                                              const RobotBackendOptions& backendOptions,
                                                              const robot::model::ContactEstimatorRegistry* absl_nonnull contactEstimators);

/**
 * Returns the hooks of the robot process: the viewer annotations and the statistics of the MPC link `remoteLink` (kept:
 * the controller that owns it outlives the process), and the check of a save against the stored files `storedFiles`
 * with `context` (checkConfigFileCandidate()).
 */
RobotProcess::Hooks robotProcessHooks(RemoteMpcLinkAdapter* absl_nonnull remoteLink,
                                      const RobotConfigDirectory::Files& storedFiles,
                                      RobotConfigurationCheckContext context);

/**
 * Returns the robot process's configuration as far as the options, the directory and the configuration give it: the
 * realtime thread's priority and cores, the backend's cores, the settings, the task file watched (with the time it was
 * read at) and its identity, the robot of `modelSettings` and the configuration store. The formulation fills in the
 * control rate, the initial state, the rest joints and the PD gains file's check interval.
 */
RobotProcess::Config robotProcessConfig(const RobotAppOptions& options,
                                        const RobotConfigDirectory& directory,
                                        const RobotConfiguration& configuration,
                                        const ModelSettings& modelSettings);

}  // namespace ocs2::humanoid
