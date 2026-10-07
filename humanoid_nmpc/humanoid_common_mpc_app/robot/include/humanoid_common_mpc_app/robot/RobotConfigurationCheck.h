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

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_common_mpc_app/robot/ConfigFileStore.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "robot_model/ContactEstimatorRegistry.h"

namespace ocs2::humanoid {

/** The configuration files of a robot process, parsed: what it is set up from and what checkRobotConfiguration() checks. */
struct RobotConfigFiles {
  mpc_config::TaskFile task;
  mpc_config::ReferenceFile reference;
  mpc_config::JointPdGainsFile pdGains;
  /** The contact planner's file beside the task file; absent for a robot that ships none. */
  std::optional<mpc_config::ContactPlanningFile> contactPlanning;
  /** What each file is, for errors: its path, or the ConfigFileCandidate's source for a saved one. */
  std::string taskSource;
  std::string referenceSource;
  std::string pdGainsSource;
};

/**
 * The files at `paths` read strictly (loadTaskFile() and its siblings), with the contact planner's file beside the task
 * file when there is one (loadContactPlanningFileBeside()). The loaders' errors, which name the file.
 */
absl::StatusOr<RobotConfigFiles> loadRobotConfigFiles(const RobotConfigDirectory::Files& paths);

/** What checkRobotConfiguration() checks the files against: the running robot, or the one being set up. */
struct RobotConfigurationCheckContext {
  /** The model_settings.robot_name the task file must have (the bundled task file's); empty: not checked. */
  std::string robotName;
  /** The joints the PD gains file may name (ModelSettings::mpcModelJointNames, fixedJointNames); both empty: not checked. */
  std::vector<std::string> mpcJointNames;
  std::vector<std::string> otherJointNames;
  /**
   * The defaults the MRT joint controller reads its PD gains file with (its pdGainsDefaults()), so that the file is read
   * here as the controller reads it: whether the controller commands a torque limit (JointPdGainsDefaults::torqueLimit)
   * decides whether the file's torque_limit fields are read, and so may refuse it.
   */
  JointPdGainsDefaults pdGainsDefaults;
  /**
   * The contact estimators the backend filled in (the robot process's registry); nullptr: contact_estimator is not
   * checked. Must outlive every call with the context. backendOptions.initialState, when set, is the state each
   * estimator is asked once on (checkContactEstimator()), as the operator mailbox asks it.
   */
  const robot::model::ContactEstimatorRegistry* absl_nullable contactEstimators = nullptr;
  /** The backend (--backend); empty: its options are not checked. */
  std::string backendName;
  /** The backend's options as the binary builds them; the simulator's come from the task file being checked. */
  RobotBackendOptions backendOptions;
  /**
   * The formulation's own check of the files: its controller models (and, for the whole-body MPC, its feedforward).
   * Empty: none - the set-up builds the models itself, through the same function.
   */
  std::function<absl::Status(const RobotConfigFiles& files)> checkFormulation;
};

/**
 * OK when a robot process would start with `files`: the one check the start-up set-up and the configuration store's
 * writer share (ConfigFileStore::Hooks::validateConfigFile), so that a saved file is held to exactly what start-up
 * holds it to. In order: the task file's model_settings.robot_name, robotProcessSettingsFromConfig(), the formulation's
 * check, jointPdGainsFromConfig() on the controller's joints, contact_estimator against the registry (and its probe),
 * every telemetry_sinks name against TelemetrySinkRegistry, and the backend's options (RobotBackendRegistry::checkOptions()).
 * A start-up check added later belongs here, once.
 *
 * @return The first refusal, naming the file and the field.
 */
absl::Status checkRobotConfiguration(const RobotConfigFiles& files, const RobotConfigurationCheckContext& context);

/**
 * checkRobotConfiguration() of the files at `paths` (the robot's stored copies) with `candidate` in place of its kind's
 * file: what the configuration store's writer checks a save with. Reads the other files from disk.
 */
absl::Status checkConfigFileCandidate(const RobotConfigDirectory::Files& paths,
                                      const ConfigFileCandidate& candidate,
                                      const RobotConfigurationCheckContext& context);

}  // namespace ocs2::humanoid
