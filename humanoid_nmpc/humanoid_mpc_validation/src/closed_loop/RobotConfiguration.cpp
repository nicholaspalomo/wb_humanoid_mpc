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

#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"

#include <filesystem>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid::validation {
namespace {

constexpr const char* kGaitFile = "humanoid_nmpc/humanoid_common_mpc/config/command/gait.yaml";

/** A configuration whose MPC package keeps its files in `configDir` (config/mpc, config/command, config/controller). */
RobotConfiguration makeConfiguration(const std::string& name,
                                     MpcFormulation formulation,
                                     const std::string& configDir,
                                     const std::string& urdfFile,
                                     const std::string& sceneFile,
                                     const std::string& walkingScenario) {
  RobotConfiguration configuration;
  configuration.name = name;
  configuration.formulation = formulation;
  configuration.taskFile = absl::StrCat(configDir, "/mpc/task.yaml");
  configuration.referenceFile = absl::StrCat(configDir, "/command/reference.yaml");
  configuration.gaitFile = kGaitFile;
  configuration.urdfFile = urdfFile;
  configuration.sceneFile = sceneFile;
  configuration.pdGainsFile = absl::StrCat(configDir, "/controller/joint_pd_gains.yaml");
  configuration.walkingScenario = walkingScenario;
  return configuration;
}

std::vector<RobotConfiguration> makeConfigurations() {
  // The files of robot_models/<robot>/<package>/launch/robot.textproto and mpc.textproto.
  return {
      makeConfiguration("drc_atlas", MpcFormulation::kCentroidal, "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config",
                        "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
                        "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml", /*walkingScenario=*/"walk_0p5"),
      makeConfiguration("engineai_sa01", MpcFormulation::kCentroidal, "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config",
                        "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf",
                        "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.xml", /*walkingScenario=*/"walk_0p3"),
      makeConfiguration("unitree_g1", MpcFormulation::kCentroidal, "robot_models/unitree_g1/g1_centroidal_mpc/config",
                        "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
                        "robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml", /*walkingScenario=*/"walk_0p5"),
      makeConfiguration("unitree_r1", MpcFormulation::kCentroidal, "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config",
                        "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf",
                        "robot_models/unitree_r1/unitree_r1_description/urdf/R1.xml", /*walkingScenario=*/"walk_0p3"),
      makeConfiguration("unitree_g1_wb", MpcFormulation::kWholeBody, "robot_models/unitree_g1/g1_wb_mpc/config",
                        "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
                        "robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml", /*walkingScenario=*/"walk_0p5"),
  };
}

}  // namespace

std::string formulationName(MpcFormulation formulation) {
  return formulation == MpcFormulation::kCentroidal ? "centroidal" : "whole_body";
}

std::vector<std::string> RobotConfiguration::configurationFiles() const {
  std::vector<std::string> files = {taskFile, referenceFile, gaitFile, pdGainsFile, urdfFile, sceneFile};
  const std::string contactPlanningFile = (std::filesystem::path(taskFile).parent_path() / "contact_planning.yaml").string();
  if (std::filesystem::exists(contactPlanningFile)) files.push_back(contactPlanningFile);
  return files;
}

const std::vector<RobotConfiguration>& robotConfigurations() {
  static const std::vector<RobotConfiguration> configurations = makeConfigurations();
  return configurations;
}

absl::StatusOr<RobotConfiguration> findRobotConfiguration(absl::string_view name) {
  std::string names;
  for (const RobotConfiguration& configuration : robotConfigurations()) {
    if (configuration.name == name) return configuration;
    absl::StrAppend(&names, names.empty() ? "" : ", ", configuration.name);
  }
  return absl::NotFoundError(absl::StrCat("[findRobotConfiguration] no configuration '", name, "'; the configurations are ", names));
}

}  // namespace ocs2::humanoid::validation
