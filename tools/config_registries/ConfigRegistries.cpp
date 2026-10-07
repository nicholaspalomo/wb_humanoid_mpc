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

#include "tools/config_registries/ConfigRegistries.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"

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
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/robot/TelemetrySinkRegistry.h"
#include "humanoid_mpc_config/config_registries.nproto.h"
#include "mujoco_sim_interface/CheaterSimContactEstimator.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/Projectile.h"
#include "mujoco_sim_interface/visualization/VisualizationRegistry.h"
#include "robot_model/ContactEstimatorRegistry.h"

namespace ocs2::humanoid::config_registries {
namespace {

using Registry = mpc_config::ConfigRegistries::Registry;

/** `names` with `name` appended unless it holds it already. */
void appendOnce(const std::string& name, std::vector<std::string>& names) {
  if (std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name);
}

/** Every term of the contact planner, over its lists in the order of the file (no_flight is in two of them), once each. */
std::vector<std::string> contactPlanningTermNames() {
  std::vector<std::string> names;
  for (const TermKind kind : allTermKinds()) {
    for (const std::string& name : knownTermNames(kind)) appendOnce(name, names);
  }
  return names;
}

/** Every locomotion heuristic, over its three lists. */
std::vector<std::string> locomotionHeuristicNames() {
  std::vector<std::string> names;
  for (const HeuristicKind kind : allHeuristicKinds()) {
    for (const std::string& name : knownHeuristicNames(kind)) appendOnce(name, names);
  }
  return names;
}

/** The estimators of robot_model and the simulator's, which a simulated robot process adds. */
std::vector<std::string> contactEstimatorNames() {
  std::vector<std::string> names;
  for (const robot::model::ContactEstimatorRegistry::Entry& entry : robot::model::ContactEstimatorRegistry().available()) {
    names.push_back(entry.name);
  }
  appendOnce(robot::mujoco_sim_interface::kCheaterSimContactEstimatorName, names);
  return names;
}

std::vector<std::string> visualizationNames() {
  std::vector<std::string> names;
  for (const robot::mujoco_sim_interface::VisualizationInfo& info : robot::mujoco_sim_interface::availableVisualizations()) {
    names.push_back(info.name);
  }
  return names;
}

}  // namespace

mpc_config::ConfigRegistries collectConfigRegistries() {
  // The registry names of the schemas' (humanoid_mpc_config.tuning) options, each with its registry's names.
  // LINT.IfChange(registry_names)
  std::vector<Registry> registries = {
      {.name = "basis_generator_set", .names = basisGeneratorSetNames()},
      {.name = "basis_regularization", .names = basisRegularizationNames()},
      {.name = "centroidal_model", .names = centroidalModelNames()},
      {.name = "contact_estimator", .names = contactEstimatorNames()},
      {.name = "contact_input_parameterization", .names = contactInputParameterizationNames()},
      {.name = "contact_planner", .names = knownPlannerNames()},
      {.name = "contact_planner_threading", .names = plannerThreadingNames()},
      {.name = "contact_planning_term", .names = contactPlanningTermNames()},
      {.name = "contact_schedule_source", .names = contactScheduleSourceNames()},
      {.name = "foot_cost_phases", .names = footCostPhasesNames()},
      {.name = "gantry_hold", .names = robot::mujoco_sim_interface::gantryHoldNames()},
      {.name = "hard_constraint", .names = mpcHardConstraintNames()},
      {.name = "locomotion_heuristic", .names = locomotionHeuristicNames()},
      {.name = "mpc_cost", .names = mpcCostNames()},
      {.name = "projectile", .names = robot::mujoco_sim_interface::availableProjectiles()},
      {.name = "rollout_integrator", .names = rolloutIntegratorNames()},
      {.name = "root_finder", .names = rootFinderNames()},
      {.name = "sensitivity_integrator", .names = sensitivityIntegratorNames()},
      {.name = "soft_constraint", .names = mpcSoftConstraintNames()},
      {.name = "stance_constraint", .names = stanceConstraintNames()},
      {.name = "telemetry_sink", .names = TelemetrySinkRegistry().names()},
      {.name = "terminal_dcm_target", .names = terminalDcmTargetNames()},
      {.name = "visualization", .names = visualizationNames()},
      {.name = "wb_mpc_feedforward", .names = wbMpcFeedforwardNames()},
  };
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/tuning_options.proto:registry)
  std::sort(registries.begin(), registries.end(), [](const Registry& lhs, const Registry& rhs) { return lhs.name < rhs.name; });
  mpc_config::ConfigRegistries collected;
  collected.registries = std::move(registries);
  return collected;
}

std::string configRegistriesText(const mpc_config::ConfigRegistries& registries) {
  std::string text =
      "# proto-file: humanoid_nmpc/humanoid_mpc_config/config_registries.proto\n"
      "# proto-message: humanoid_mpc_config.ConfigRegistries\n"
      "#\n"
      "# The names each registry of the stack accepts: the choices the tuning GUI offers for a string field whose\n"
      "# (humanoid_mpc_config.tuning) option names a registry. Written from the registries by\n"
      "# `bazel run //tools/config_registries:print_config_registries`; do not edit it by hand:\n"
      "# //tools/config_registries:config_registries_test fails when it and the registries differ.\n";
  for (const Registry& registry : registries.registries) {
    absl::StrAppend(&text, "\nregistries {\n  name: \"", absl::CEscape(registry.name), "\"\n");
    for (const std::string& name : registry.names) absl::StrAppend(&text, "  names: \"", absl::CEscape(name), "\"\n");
    absl::StrAppend(&text, "}\n");
  }
  return text;
}

}  // namespace ocs2::humanoid::config_registries
