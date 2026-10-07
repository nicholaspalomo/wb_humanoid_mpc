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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

/** The files of a shipped robot configuration of the centroidal MPC in the test's runfiles: textprotos and the URDF. */
struct CentroidalRobotFiles {
  std::string taskFile;
  std::string referenceFile;
  std::string urdfFile;
};

/** The DRC Atlas files (robot_models/drc_atlas); a missing one fails the calling test and is empty. */
CentroidalRobotFiles atlasFiles();

/** The EngineAI SA01 files (robot_models/engineai_sa01); a missing one fails the calling test and is empty. */
CentroidalRobotFiles sa01Files();

/** The Unitree G1 files (robot_models/unitree_g1, the 29-DoF URDF); a missing one fails the calling test and is empty. */
CentroidalRobotFiles g1Files();

/** The Unitree R1 files (robot_models/unitree_r1); a missing one fails the calling test and is empty. */
CentroidalRobotFiles r1Files();

/** The typed configuration of `files` (loadCentroidalMpcConfig()). */
absl::StatusOr<CentroidalMpcConfig> loadConfigOf(const CentroidalRobotFiles& files);

/**
 * The joint state of the reference file's default_joint_state on the MPC joints of `modelSettings`
 * (defaultJointStateFromConfig()), the one CentroidalMpcInterface builds its centroidal model with.
 */
absl::StatusOr<vector_t> defaultJointStateOf(const mpc_config::ReferenceFile& reference, const ModelSettings& modelSettings);

/** The task file's initial_state on the centroidal state of `modelSettings` (stateValuesFromConfig()). */
absl::StatusOr<vector_t> initialStateOf(const mpc_config::TaskFile& task, const ModelSettings& modelSettings);

/**
 * The centroidal model info of `config` on `pinocchioInterface` and `modelSettings`, as CentroidalMpcInterface builds it:
 * the task file's centroidal_model and the reference file's default_joint_state. Keeps no reference to its
 * arguments.
 */
absl::StatusOr<CentroidalModelInfo> centroidalModelInfoOf(const CentroidalMpcConfig& config,
                                                          const PinocchioInterface& pinocchioInterface,
                                                          const ModelSettings& modelSettings);

/** The textprotos of the typed files, as the strict parser reads them back (nproto::WriteTextproto()). */
std::string taskFileText(const mpc_config::TaskFile& task);
std::string referenceFileText(const mpc_config::ReferenceFile& reference);
std::string contactPlanningFileText(const mpc_config::ContactPlanningFile& contactPlanning);

/** Writes `text` to `path`, replacing the file; Internal naming the path when it cannot be written. */
absl::Status writeTextFile(const std::string& path, absl::string_view text);

/**
 * Writes the configuration `config` into the directory `directory` (created if needed) as a robot's files are laid out
 * beside each other: task.textproto, reference.textproto and, when `config` has one, contact_planning.textproto. Returns
 * the paths of the task and reference files with `urdfFile`.
 */
absl::StatusOr<CentroidalRobotFiles> writeConfig(const std::string& directory,
                                                 const CentroidalMpcConfig& config,
                                                 const std::string& urdfFile);

}  // namespace ocs2::humanoid
