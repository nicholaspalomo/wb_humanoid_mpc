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

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::validation {
namespace {

// The contact planner's own file, beside the task file:
// kContactPlanningFileName of humanoid_common_mpc, spelled here so that the
// metrics library does not link the MPC's.
// LINT.IfChange(contact_planning_file_name)
constexpr char kContactPlanningFile[] = "contact_planning.textproto";
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/config/ConfigFiles.h:contact_planning_file_name)

constexpr char kGaitFile[] =
    "humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto";
// The quadrupeds' gait file, the gait_file of their launch files.
// LINT.IfChange(quadruped_gait_file)
constexpr char kQuadrupedGaitFile[] =
    "humanoid_nmpc/humanoid_common_mpc/config/command/quadruped_gait.textproto";
// LINT.ThenChange(//robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/launch/mpc.textproto:gait_file, //robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/launch/dummy_sim.textproto:gait_file)

/**
 * Returns `configuration` on the quadruped gait file (kQuadrupedGaitFile),
 * running standing, walk_0p5 and turn_in_place_720 only: the other scenarios
 * (stepping, side steps, pushes and the walk_0p3 the slower bipeds use) are
 * tuned for the bipeds' gaits.
 */
RobotConfiguration onQuadrupedGait(RobotConfiguration configuration) {
  configuration.gaitFile = kQuadrupedGaitFile;
  configuration.scenarios = {"standing", "walk_0p5", "turn_in_place_720"};
  return configuration;
}

// The wheeled bipeds' gait file, the gait_file of their launch files.
// LINT.IfChange(wheeled_gait_file)
constexpr char kWheeledGaitFile[] =
    "humanoid_nmpc/humanoid_common_mpc/config/command/wheeled_gait.textproto";
// LINT.ThenChange(//robot_models/limx_tron1/tron1_centroidal_mpc/launch/mpc.textproto:gait_file, //robot_models/limx_tron1/tron1_centroidal_mpc/launch/dummy_sim.textproto:gait_file)

/**
 * Returns `configuration` on the wheeled bipeds' gait file (kWheeledGaitFile),
 * running standing, walk_0p5 (a drive on the wheels at 0.5 m/s, no steps) and
 * turn_in_place_720 only: the other scenarios step, side-step or push the
 * robot, and the drive gait takes no step.
 */
RobotConfiguration onWheeledGait(RobotConfiguration configuration) {
  configuration.gaitFile = kWheeledGaitFile;
  configuration.scenarios = {"standing", "walk_0p5", "turn_in_place_720"};
  return configuration;
}

// The wheeled quadrupeds' gait file, the gait_file of their launch files.
// LINT.IfChange(wheeled_quadruped_gait_file)
constexpr char kWheeledQuadrupedGaitFile[] =
    "humanoid_nmpc/humanoid_common_mpc/config/command/"
    "wheeled_quadruped_gait.textproto";
// LINT.ThenChange(//robot_models/anybotics_anymal_c_wheels/anymal_c_wheels_centroidal_mpc/launch/mpc.textproto:gait_file, //robot_models/anybotics_anymal_c_wheels/anymal_c_wheels_centroidal_mpc/launch/dummy_sim.textproto:gait_file)

/**
 * Returns `configuration` on the wheeled quadrupeds' gait file
 * (kWheeledQuadrupedGaitFile), running standing, walk_0p5 (a drive on the
 * wheels at 0.5 m/s) and turn_in_place_720 (a trot) only: the other scenarios
 * step sideways, push the robot or turn while they walk, which the drive gait
 * cannot.
 */
RobotConfiguration onWheeledQuadrupedGait(RobotConfiguration configuration) {
  configuration.gaitFile = kWheeledQuadrupedGaitFile;
  configuration.scenarios = {"standing", "walk_0p5", "turn_in_place_720"};
  return configuration;
}

/** A configuration whose MPC package keeps its files in `configDir`
 * (config/mpc, config/command, config/controller). */
RobotConfiguration makeConfiguration(std::string name,
                                     MpcFormulation formulation,
                                     absl::string_view configDir,
                                     std::string urdfFile,
                                     std::string sceneFile,
                                     std::string walkingScenario) {
  RobotConfiguration configuration;
  configuration.name = std::move(name);
  configuration.formulation = formulation;
  // LINT.IfChange(config_layout)
  configuration.taskFile = absl::StrCat(configDir, "/mpc/task.textproto");
  configuration.referenceFile =
      absl::StrCat(configDir, "/command/reference.textproto");
  configuration.pdGainsFile =
      absl::StrCat(configDir, "/controller/joint_pd_gains.textproto");
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/config/ConfigFiles.h:config_layout, //humanoid_nmpc/remote_control/remote_control/config_files.py:config_layout)
  // clang-format on
  configuration.gaitFile = kGaitFile;
  configuration.urdfFile = std::move(urdfFile);
  configuration.sceneFile = std::move(sceneFile);
  configuration.walkingScenario = std::move(walkingScenario);
  return configuration;
}

// The state-estimator variant's estimator and scenarios. kalman is
// kKalmanStateEstimatorName of humanoid_state_estimator, spelled here so that
// the metrics library does not link the estimator.
// LINT.IfChange(kalman_variant)
constexpr char kKalmanStateEstimator[] = "kalman";
// clang-format off
// LINT.ThenChange(//humanoid_state_estimation/humanoid_state_estimator/include/humanoid_state_estimator/StateEstimatorNames.h:kalman_name)
// clang-format on

/**
 * unitree_g1 with the kalman state estimator in the robot process, held to
 * unitree_g1's documents: standing (the filter at rest), walk_0p5 (stance
 * switches) and turn_in_place_720 (the yaw the filter takes from the IMU while
 * it estimates the position), which keeps the run to three scenarios.
 */
RobotConfiguration makeKalmanVariant(const RobotConfiguration& base) {
  RobotConfiguration variant = base;
  variant.name = absl::StrCat(base.name, "_kalman");
  variant.stateEstimator = kKalmanStateEstimator;
  variant.scenarios = {"standing", "walk_0p5", "turn_in_place_720"};
  variant.baselineConfiguration = base.name;
  return variant;
}

std::vector<RobotConfiguration> makeConfigurations() {
  // The files of robot_models/<robot>/<package>/launch/robot.textproto and
  // mpc.textproto.
  std::vector<RobotConfiguration> configurations = {
      makeConfiguration(
          "drc_atlas", MpcFormulation::kCentroidal,
          "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config",
          "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
          "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml",
          /*walkingScenario=*/"walk_0p5"),
      makeConfiguration(
          "engineai_sa01", MpcFormulation::kCentroidal,
          "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config",
          "robot_models/engineai_sa01/engineai_sa01_description/urdf/"
          "zq_sa01.urdf",
          "robot_models/engineai_sa01/engineai_sa01_description/urdf/"
          "zq_sa01.xml",
          /*walkingScenario=*/"walk_0p3"),
      makeConfiguration(
          "unitree_g1", MpcFormulation::kCentroidal,
          "robot_models/unitree_g1/g1_centroidal_mpc/config",
          "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
          "robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml",
          /*walkingScenario=*/"walk_0p5"),
      makeConfiguration(
          "unitree_r1", MpcFormulation::kCentroidal,
          "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config",
          "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf",
          "robot_models/unitree_r1/unitree_r1_description/urdf/R1.xml",
          /*walkingScenario=*/"walk_0p3"),
      onQuadrupedGait(makeConfiguration(
          "anymal_b", MpcFormulation::kCentroidal,
          "robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/config",
          "robot_models/anybotics_anymal_b/anymal_b_description/urdf/"
          "anymal_b.urdf",
          "robot_models/anybotics_anymal_b/anymal_b_description/urdf/"
          "anymal_b.xml",
          /*walkingScenario=*/"walk_0p5")),
      onWheeledGait(makeConfiguration(
          "tron1", MpcFormulation::kCentroidal,
          "robot_models/limx_tron1/tron1_centroidal_mpc/config",
          "robot_models/limx_tron1/tron1_description/urdf/tron1.urdf",
          "robot_models/limx_tron1/tron1_description/urdf/tron1.xml",
          /*walkingScenario=*/"walk_0p5")),
      onWheeledQuadrupedGait(makeConfiguration(
          "anymal_c_wheels", MpcFormulation::kCentroidal,
          "robot_models/anybotics_anymal_c_wheels/"
          "anymal_c_wheels_centroidal_mpc/config",
          "robot_models/anybotics_anymal_c_wheels/anymal_c_wheels_description/"
          "urdf/anymal_c_wheels.urdf",
          "robot_models/anybotics_anymal_c_wheels/anymal_c_wheels_description/"
          "urdf/anymal_c_wheels.xml",
          /*walkingScenario=*/"walk_0p5")),
      makeConfiguration(
          "unitree_g1_wb", MpcFormulation::kWholeBody,
          "robot_models/unitree_g1/g1_wb_mpc/config",
          "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
          "robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml",
          /*walkingScenario=*/"walk_0p5"),
  };
  std::vector<RobotConfiguration> variants;
  for (const RobotConfiguration& configuration : configurations) {
    if (configuration.name == "unitree_g1") {
      variants.push_back(makeKalmanVariant(configuration));
    }
  }
  configurations.insert(configurations.end(), variants.begin(), variants.end());
  return configurations;
}

}  // namespace

std::string formulationName(MpcFormulation formulation) {
  return formulation == MpcFormulation::kCentroidal ? "centroidal"
                                                    : "whole_body";
}

std::vector<std::string> RobotConfiguration::configurationFiles() const {
  std::vector<std::string> files = {taskFile,    referenceFile, gaitFile,
                                    pdGainsFile, urdfFile,      sceneFile};
  const std::string contactPlanningFile =
      (std::filesystem::path(taskFile).parent_path() / kContactPlanningFile)
          .string();
  std::error_code
      error;  // a file that cannot be examined is not listed, as a missing one
  if (std::filesystem::exists(contactPlanningFile, error)) {
    files.push_back(contactPlanningFile);
  }
  return files;
}

bool RobotConfiguration::runsScenario(absl::string_view scenario) const {
  return scenarios.empty() || std::find(scenarios.begin(), scenarios.end(),
                                        scenario) != scenarios.end();
}

const std::vector<RobotConfiguration>& robotConfigurations() {
  static const absl::NoDestructor<std::vector<RobotConfiguration>>
      kConfigurations(makeConfigurations());
  return *kConfigurations;
}

absl::StatusOr<RobotConfiguration> findRobotConfiguration(
    absl::string_view name) {
  std::string names;
  for (const RobotConfiguration& configuration : robotConfigurations()) {
    if (configuration.name == name) return configuration;
    absl::StrAppend(&names, names.empty() ? "" : ", ", configuration.name);
  }
  return absl::NotFoundError(
      absl::StrCat("[findRobotConfiguration] no configuration '", name,
                   "'; the configurations are ", names));
}

}  // namespace ocs2::humanoid::validation
