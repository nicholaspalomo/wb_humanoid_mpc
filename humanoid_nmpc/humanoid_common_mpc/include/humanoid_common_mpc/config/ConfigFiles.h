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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_mpc_config/contact_planning_file.nproto.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

/**
 * The configuration files of a robot as the typed structs of humanoid_nmpc/humanoid_mpc_config, read strictly
 * (nproto::LoadTextprotoFile(): an unknown or retired field, a value of the wrong type or a file of another message is
 * an InvalidArgument naming the file, the line and the column). For the roots that take their files by path; everything
 * below them takes the structs.
 */
namespace ocs2::humanoid {

/** The name of the contact planner's own file, in the directory of the robot's task file. */
// LINT.IfChange(contact_planning_file_name)
inline constexpr absl::string_view kContactPlanningFileName = "contact_planning.textproto";
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/RobotConfiguration.cpp:contact_planning_file_name, //humanoid_nmpc/remote_control/remote_control/config_files.py:contact_planning_file_name, //tools/locomotion_heuristics/derive_parameters.py:contact_planning_file_name)
// clang-format on

/**
 * The joint PD gains file of a robot, relative to its config/ directory, the parent of the task file's config/mpc/
 * (config/mpc/task.textproto, config/command/reference.textproto, config/controller/joint_pd_gains.textproto).
 */
// LINT.IfChange(config_layout)
inline constexpr absl::string_view kJointPdGainsFileInConfigDirectory = "controller/joint_pd_gains.textproto";
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/config_files.py:config_layout, //humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/RobotConfiguration.cpp:config_layout, //tools/deploy/test_robot_bundle.py:config_layout)
// clang-format on

/** A task file (config/mpc/task.textproto, humanoid_mpc_config.TaskFile). */
absl::StatusOr<mpc_config::TaskFile> loadTaskFile(absl::string_view path);

/** A reference file (config/command/reference.textproto, humanoid_mpc_config.ReferenceFile). */
absl::StatusOr<mpc_config::ReferenceFile> loadReferenceFile(absl::string_view path);

/** A gait file (humanoid_common_mpc/config/command/gait.textproto, humanoid_mpc_config.GaitFile). */
absl::StatusOr<mpc_config::GaitFile> loadGaitFile(absl::string_view path);

/** A contact planner's file (config/mpc/contact_planning.textproto, humanoid_mpc_config.ContactPlanningFile). */
absl::StatusOr<mpc_config::ContactPlanningFile> loadContactPlanningFile(absl::string_view path);

/** A file of PD gains (config/controller/joint_pd_gains.textproto, humanoid_mpc_config.JointPdGainsFile). */
absl::StatusOr<mpc_config::JointPdGainsFile> loadJointPdGainsFile(absl::string_view path);

/**
 * The text of a task file, read as strictly as loadTaskFile() reads the file (the same leading-comment check and
 * parser); `source` names the text in errors, as a path would ("<source>:<line>:<column>: ..."). For a file that
 * arrives as text, such as the tuning GUI's Save on the bus (ConfigFileSave), which must be checked before it is
 * written anywhere.
 */
absl::StatusOr<mpc_config::TaskFile> parseTaskFile(absl::string_view text, absl::string_view source);

/** The text of a reference file, read as loadReferenceFile() reads the file; see parseTaskFile(). */
absl::StatusOr<mpc_config::ReferenceFile> parseReferenceFile(absl::string_view text, absl::string_view source);

/** The text of a PD gains file, read as loadJointPdGainsFile() reads the file; see parseTaskFile(). */
absl::StatusOr<mpc_config::JointPdGainsFile> parseJointPdGainsFile(absl::string_view text, absl::string_view source);

/**
 * The identity of the configuration file at `path`: the path from its last robot_models/ component on, with `/`
 * separators and `.` and `..` resolved lexically ("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto"),
 * whichever directory the robot_models/ tree lies in (a checkout, a runfiles tree, the robot bundle). The path
 * normalized as it is when it has no robot_models/ component. The tuning GUI stamps it on what it sends
 * (MpcParameterUpdate.config_path, ConfigFileSave.config_path), and a receiver refuses a file of another
 * configuration: two configurations of one robot (unitree_g1's centroidal and whole-body MPCs) share their robot_name.
 * It names, and is never joined into a path: nothing reads a file by it.
 */
std::string configFileIdentity(absl::string_view path);

/**
 * The path of the contact planner's file of the robot whose task file is `taskFile`: kContactPlanningFileName in the
 * directory of the task file.
 */
std::string contactPlanningFileBeside(absl::string_view taskFile);

/**
 * The path of the joint PD gains file of the robot whose task file is `taskFile`: kJointPdGainsFileInConfigDirectory in
 * the config/ directory above the task file's.
 */
std::string jointPdGainsFileBeside(absl::string_view taskFile);

/**
 * jointPdGainsFileBeside(), for a robot process, which must not start without its gains: NotFound naming the file
 * (and the task file it belongs to) when it is not a regular file, such as a robot directory that still holds
 * joint_pd_gains.yaml, which would leave the controller on its built-in gains, chosen for no robot.
 */
absl::StatusOr<std::string> existingJointPdGainsFileBeside(absl::string_view taskFile);

/**
 * The contact planner's file beside the task file `taskFile` (contactPlanningFileBeside()), or nullopt when the robot has
 * none: the robots that do not plan their contacts ship no such file, and run the planner's library defaults
 * (contactPlanningConfigFromOptionalFile()) where one is built anyway. The task file itself holds no contact planner
 * configuration any more (TaskFile's retired field contact_planning).
 *
 * @return The file; nullopt when there is no file of that name; Unavailable naming the path when it cannot be looked at
 *         (a permission or I/O error), FailedPrecondition when it is not a regular file, and loadContactPlanningFile()'s
 *         error for a file that does not parse.
 */
absl::StatusOr<std::optional<mpc_config::ContactPlanningFile>> loadContactPlanningFileBeside(absl::string_view taskFile);

/**
 * `status` with its message prefixed by `path` ("<path>: <message>"), for an error of a conversion of the file at
 * `path`, whose messages name the field but not the file; OK stays OK.
 */
absl::Status withConfigFile(const absl::Status& status, absl::string_view path);

}  // namespace ocs2::humanoid
