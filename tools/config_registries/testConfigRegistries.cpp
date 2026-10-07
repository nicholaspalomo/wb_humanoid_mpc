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

// config_registries.textproto against the registries it is written from: the committed file is what
// print_config_registries writes and parses strictly into its schema, every registry has names, once each, and every
// name it offers is one its registry accepts.

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/config/contact_planning/ContactPlanningFromConfig.h"
#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"
#include "humanoid_common_mpc/config/model/ModelSettingsFromConfig.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"
#include "humanoid_common_mpc_app/robot/TelemetrySinkRegistry.h"
#include "humanoid_common_mpc_app/robot/config/robot/RobotProcessSettingsFromConfig.h"
#include "humanoid_mpc_config/config_registries.nproto.h"
#include "humanoid_mpc_config/config_registries.nproto.pb.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_mpc_config/rollout_settings_config.nproto.h"
#include "humanoid_mpc_config/sqp_settings_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "mujoco_sim_interface/CheaterSimContactEstimator.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/Projectile.h"
#include "mujoco_sim_interface/visualization/VisualizationRegistry.h"
#include "nproto/Textproto.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "tools/config_registries/ConfigRegistries.h"

namespace ocs2::humanoid::config_registries {
namespace {

using Registry = mpc_config::ConfigRegistries::Registry;
using Accepts = std::function<bool(const std::string& name)>;

constexpr char kWriteIt[] = "write it with `bazel run //tools/config_registries:print_config_registries`";

/** Whether some list of the contact planner holds the term `name`, spelled so. */
bool isContactPlanningTerm(const std::string& name) {
  const std::vector<TermKind>& kinds = allTermKinds();
  return std::any_of(kinds.begin(), kinds.end(), [&name](TermKind kind) { return canonicalTermName(kind, name) == name; });
}

/** Whether some list of the locomotion heuristics holds `name`, spelled so. */
bool isLocomotionHeuristic(const std::string& name) {
  for (const HeuristicKind kind : allHeuristicKinds()) {
    if (canonicalHeuristicName(kind, name) == name) return true;
  }
  return false;
}

/** Whether the robot process reads a task file naming `name` as its whole-body feedforward. */
bool isWbMpcFeedforward(const std::string& name) {
  mpc_config::TaskFile task;
  task.wb_mpc_feedforward = name;
  return robotProcessSettingsFromConfig(task).ok();
}

/** Whether the contact planner's conversion reads a planning file whose planner.threading is `name`. */
bool isPlannerThreading(const std::string& name) {
  mpc_config::ContactPlanningFile file;
  file.planner.threading = name;
  return contactPlanningConfigFromConfig(file, ContactPlanningValidation::kDeferUntilModelParametersApplied).ok();
}

/** Whether the contact planner's conversion reads a planning file whose terminal_dcm.target is `name`. */
bool isTerminalDcmTarget(const std::string& name) {
  mpc_config::ContactPlanningFile file;
  file.terminal_dcm.target = name;
  return contactPlanningConfigFromConfig(file, ContactPlanningValidation::kDeferUntilModelParametersApplied).ok();
}

/** Whether the model settings read a foot_constraint block whose stance_constraint is `name`. */
bool isStanceConstraint(const std::string& name) {
  mpc_config::ModelSettingsConfig::FootConstraintConfig footConstraint;
  footConstraint.stance_constraint = name;
  return footConstraintFromConfig(footConstraint).ok();
}

/** Whether the SQP settings read a multiple_shooting block whose integrator_type is `name`. */
bool isSensitivityIntegrator(const std::string& name) {
  mpc_config::SqpSettingsConfig config;
  config.integrator_type = name;
  return toSqpSettings(config).ok();
}

/** Whether the rollout settings read a rollout block whose integrator_type is `name`. */
bool isRolloutIntegrator(const std::string& name) {
  mpc_config::RolloutSettingsConfig config;
  config.integrator_type = name;
  return toRolloutSettings(config).ok();
}

/** Whether the rollout settings read a rollout block whose root_finder_type is `name`. */
bool isRootFinder(const std::string& name) {
  mpc_config::RolloutSettingsConfig config;
  config.root_finder_type = name;
  return toRolloutSettings(config).ok();
}

/** What each registry accepts, by its own resolver: a name it offers must be a name it reads. */
absl::flat_hash_map<std::string, Accepts> acceptors() {
  return {
      {"basis_generator_set", [](const std::string& name) { return getBasisGeneratorSetBuilder(name).ok(); }},
      {"basis_regularization", [](const std::string& name) { return getBasisRegularizationBuilder(name).ok(); }},
      {"centroidal_model", [](const std::string& name) { return centroidalModelTypeFromName(name).ok(); }},
      {"contact_estimator",
       [](const std::string& name) {
         return robot::model::ContactEstimatorRegistry().has(name) || name == robot::mujoco_sim_interface::kCheaterSimContactEstimatorName;
       }},
      {"contact_input_parameterization", [](const std::string& name) { return contactInputParameterizationFromName(name).ok(); }},
      {"contact_planner", [](const std::string& name) { return canonicalPlannerName(name) == name; }},
      {"contact_planner_threading", &isPlannerThreading},
      {"contact_planning_term", &isContactPlanningTerm},
      {"contact_schedule_source", [](const std::string& name) { return contactScheduleSourceFromName(name).ok(); }},
      {"foot_cost_phases",
       [](const std::string& name) { return footCostActiveInStanceFromName(name, "task_space_foot_cost.active_phases").ok(); }},
      {"gantry_hold", [](const std::string& name) { return robot::mujoco_sim_interface::gantryHoldFromName(name).ok(); }},
      {"hard_constraint", [](const std::string& name) { return stringToMpcHardConstraintType(name).ok(); }},
      {"locomotion_heuristic", &isLocomotionHeuristic},
      {"mpc_cost", [](const std::string& name) { return stringToMpcCostType(name).ok(); }},
      {"projectile", [](const std::string& name) { return robot::mujoco_sim_interface::projectileFromName(name).ok(); }},
      {"rollout_integrator", &isRolloutIntegrator},
      {"root_finder", &isRootFinder},
      {"sensitivity_integrator", &isSensitivityIntegrator},
      {"soft_constraint", [](const std::string& name) { return stringToMpcSoftConstraintType(name).ok(); }},
      {"stance_constraint", &isStanceConstraint},
      {"telemetry_sink", [](const std::string& name) { return TelemetrySinkRegistry().has(name); }},
      {"terminal_dcm_target", &isTerminalDcmTarget},
      {"visualization", [](const std::string& name) { return robot::mujoco_sim_interface::createVisualization(name) != nullptr; }},
      {"wb_mpc_feedforward", &isWbMpcFeedforward},
  };
}

TEST(ConfigRegistriesFileTest, IsWhatPrintConfigRegistriesWrites) {
  const absl::StatusOr<std::string> text = nproto::ReadTextFile(kConfigRegistriesFile);
  ASSERT_TRUE(text.ok()) << text.status() << "; " << kWriteIt;
  EXPECT_EQ(*text, configRegistriesText(collectConfigRegistries()))
      << kConfigRegistriesFile << " differs from the registries; " << kWriteIt;
}

TEST(ConfigRegistriesFileTest, ParsesStrictlyIntoTheRegistries) {
  const absl::StatusOr<mpc_config::ConfigRegistries> file =
      nproto::LoadTextprotoFile<mpc_config::ConfigRegistries, humanoid_mpc_config::ConfigRegistries>(kConfigRegistriesFile);
  ASSERT_TRUE(file.ok()) << file.status();
  EXPECT_TRUE(*file == collectConfigRegistries()) << kWriteIt;
}

TEST(ConfigRegistriesTest, EveryRegistryIsListedOnceInOrderAndHasEachOfItsNamesOnce) {
  const mpc_config::ConfigRegistries registries = collectConfigRegistries();
  ASSERT_FALSE(registries.registries.empty());
  for (size_t i = 0; i < registries.registries.size(); ++i) {
    const Registry& registry = registries.registries[i];
    EXPECT_FALSE(registry.name.empty());
    if (i > 0) {
      EXPECT_LT(registries.registries[i - 1].name, registry.name) << "sorted, each registry once";
    }
    EXPECT_FALSE(registry.names.empty()) << registry.name << " offers no name";
    for (const std::string& name : registry.names) {
      EXPECT_FALSE(name.empty()) << registry.name;
      EXPECT_EQ(std::count(registry.names.begin(), registry.names.end(), name), 1) << registry.name << ": " << name;
    }
  }
}

TEST(ConfigRegistriesTest, EveryNameARegistryOffersIsOneItAccepts) {
  const absl::flat_hash_map<std::string, Accepts> accepts = acceptors();
  const mpc_config::ConfigRegistries registries = collectConfigRegistries();
  EXPECT_EQ(registries.registries.size(), accepts.size()) << "a registry without its resolver in acceptors(), or one gone";
  for (const Registry& registry : registries.registries) {
    const absl::flat_hash_map<std::string, Accepts>::const_iterator accept = accepts.find(registry.name);
    ASSERT_NE(accept, accepts.end()) << registry.name << " has no resolver in acceptors()";
    for (const std::string& name : registry.names) EXPECT_TRUE(accept->second(name)) << registry.name << " offers '" << name << "'";
    EXPECT_FALSE(accept->second("no_such_name")) << registry.name << " accepts anything, so it offers nothing";
  }
}

}  // namespace
}  // namespace ocs2::humanoid::config_registries
