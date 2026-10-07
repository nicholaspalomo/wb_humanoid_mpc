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

#include "support/TypedConfigFiles.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/FactoryFunctions.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.pb.h"
#include "humanoid_mpc_config/contact_planning_file.pb.h"
#include "humanoid_mpc_config/reference_file.nproto.pb.h"
#include "humanoid_mpc_config/reference_file.pb.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/Textproto.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid {

namespace {

/** The runfiles path of `relativePath`; a missing file fails the calling test and is empty. */
std::string runfile(absl::string_view relativePath) {
  absl::StatusOr<std::string> path = robot::resolveResourcePath(relativePath);
  EXPECT_TRUE(path.ok()) << path.status();
  return path.ok() ? *std::move(path) : std::string();
}

}  // namespace

CentroidalRobotFiles atlasFiles() {
  return CentroidalRobotFiles{
      .taskFile = runfile("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto"),
      .referenceFile = runfile("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.textproto"),
      .urdfFile = runfile("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf"),
  };
}

CentroidalRobotFiles sa01Files() {
  return CentroidalRobotFiles{
      .taskFile = runfile("robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto"),
      .referenceFile = runfile("robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/command/reference.textproto"),
      .urdfFile = runfile("robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf"),
  };
}

CentroidalRobotFiles g1Files() {
  return CentroidalRobotFiles{
      .taskFile = runfile("robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto"),
      .referenceFile = runfile("robot_models/unitree_g1/g1_centroidal_mpc/config/command/reference.textproto"),
      .urdfFile = runfile("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"),
  };
}

CentroidalRobotFiles r1Files() {
  return CentroidalRobotFiles{
      .taskFile = runfile("robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto"),
      .referenceFile = runfile("robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/command/reference.textproto"),
      .urdfFile = runfile("robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf"),
  };
}

absl::StatusOr<CentroidalMpcConfig> loadConfigOf(const CentroidalRobotFiles& files) {
  return loadCentroidalMpcConfig(files.taskFile, files.referenceFile);
}

absl::StatusOr<vector_t> defaultJointStateOf(const mpc_config::ReferenceFile& reference, const ModelSettings& modelSettings) {
  return defaultJointStateFromConfig(reference, modelSettings.mpcModelJointNames, modelSettings.fixedJointNames);
}

absl::StatusOr<vector_t> initialStateOf(const mpc_config::TaskFile& task, const ModelSettings& modelSettings) {
  return stateValuesFromConfig(task.initial_state, stateInputLayout(modelSettings, StateInputLayout::Mpc::kCentroidal), "initial_state");
}

absl::StatusOr<CentroidalModelInfo> centroidalModelInfoOf(const CentroidalMpcConfig& config,
                                                          const PinocchioInterface& pinocchioInterface,
                                                          const ModelSettings& modelSettings) {
  ASSIGN_OR_RETURN(const CentroidalModelType type, centroidalModelTypeFromConfig(config.task));
  ASSIGN_OR_RETURN(const vector_t defaultJointState, defaultJointStateOf(config.reference, modelSettings));
  return centroidal_model::createCentroidalModelInfo(pinocchioInterface, type, defaultJointState, modelSettings.contactNames3DoF,
                                                     modelSettings.contactNames6DoF);
}

std::string taskFileText(const mpc_config::TaskFile& task) {
  humanoid_mpc_config::TaskFile message;
  mpc_config::ToProto(task, &message);
  return nproto::WriteTextproto(message);
}

std::string referenceFileText(const mpc_config::ReferenceFile& reference) {
  humanoid_mpc_config::ReferenceFile message;
  mpc_config::ToProto(reference, &message);
  return nproto::WriteTextproto(message);
}

std::string contactPlanningFileText(const mpc_config::ContactPlanningFile& contactPlanning) {
  humanoid_mpc_config::ContactPlanningFile message;
  mpc_config::ToProto(contactPlanning, &message);
  return nproto::WriteTextproto(message);
}

absl::Status writeTextFile(const std::string& path, absl::string_view text) {
  std::ofstream out(path, std::ios::trunc);
  out << text;
  out.close();
  if (!out) {
    return absl::InternalError(absl::StrCat("could not write ", path));
  }
  return absl::OkStatus();
}

absl::StatusOr<CentroidalRobotFiles> writeConfig(const std::string& directory,
                                                 const CentroidalMpcConfig& config,
                                                 const std::string& urdfFile) {
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    return absl::InternalError(absl::StrCat("could not create ", directory, ": ", error.message()));
  }
  const std::filesystem::path root(directory);
  CentroidalRobotFiles files{
      .taskFile = (root / "task.textproto").string(),
      .referenceFile = (root / "reference.textproto").string(),
      .urdfFile = urdfFile,
  };
  RETURN_IF_ERROR(writeTextFile(files.taskFile, taskFileText(config.task)));
  RETURN_IF_ERROR(writeTextFile(files.referenceFile, referenceFileText(config.reference)));
  const std::string contactPlanningFile = (root / std::string(kContactPlanningFileName)).string();
  if (config.contactPlanning.has_value()) {
    RETURN_IF_ERROR(writeTextFile(contactPlanningFile, contactPlanningFileText(*config.contactPlanning)));
  } else {
    std::filesystem::remove(contactPlanningFile, error);
  }
  return files;
}

}  // namespace ocs2::humanoid
