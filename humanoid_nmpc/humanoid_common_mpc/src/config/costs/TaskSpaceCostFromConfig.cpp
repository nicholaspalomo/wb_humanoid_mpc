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

#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_mpc_config/task_space_cost_config.nproto.h"
#include "humanoid_mpc_config/task_space_weights.nproto.h"

namespace ocs2::humanoid {
namespace {

using Weights = mpc_config::TaskSpaceWeights;

/** A weight of the task_space_weights block and the entry of EndEffectorKinematicsWeights it becomes. */
struct KinematicsWeightField {
  const char* absl_nonnull name;
  double Weights::*absl_nonnull field;
  vector3_t EndEffectorKinematicsWeights::*absl_nonnull weight;
  Eigen::Index axis;
};

// The twelve weights of a task-space kinematics cost, in the order of EndEffectorKinematicsWeights::toVector().
constexpr std::array<KinematicsWeightField, 12> kKinematicsWeightFields = {{
    {.name = "pos_x", .field = &Weights::pos_x, .weight = &EndEffectorKinematicsWeights::contactPositionErrorWeight, .axis = 0},
    {.name = "pos_y", .field = &Weights::pos_y, .weight = &EndEffectorKinematicsWeights::contactPositionErrorWeight, .axis = 1},
    {.name = "pos_z", .field = &Weights::pos_z, .weight = &EndEffectorKinematicsWeights::contactPositionErrorWeight, .axis = 2},
    {.name = "orientation_x",
     .field = &Weights::orientation_x,
     .weight = &EndEffectorKinematicsWeights::contactOrientationErrorWeight,
     .axis = 0},
    {.name = "orientation_y",
     .field = &Weights::orientation_y,
     .weight = &EndEffectorKinematicsWeights::contactOrientationErrorWeight,
     .axis = 1},
    {.name = "orientation_z",
     .field = &Weights::orientation_z,
     .weight = &EndEffectorKinematicsWeights::contactOrientationErrorWeight,
     .axis = 2},
    {.name = "lin_velocity_x",
     .field = &Weights::lin_velocity_x,
     .weight = &EndEffectorKinematicsWeights::contactLinearVelocityErrorWeight,
     .axis = 0},
    {.name = "lin_velocity_y",
     .field = &Weights::lin_velocity_y,
     .weight = &EndEffectorKinematicsWeights::contactLinearVelocityErrorWeight,
     .axis = 1},
    {.name = "lin_velocity_z",
     .field = &Weights::lin_velocity_z,
     .weight = &EndEffectorKinematicsWeights::contactLinearVelocityErrorWeight,
     .axis = 2},
    {.name = "ang_velocity_x",
     .field = &Weights::ang_velocity_x,
     .weight = &EndEffectorKinematicsWeights::contactAngularVelocityErrorWeight,
     .axis = 0},
    {.name = "ang_velocity_y",
     .field = &Weights::ang_velocity_y,
     .weight = &EndEffectorKinematicsWeights::contactAngularVelocityErrorWeight,
     .axis = 1},
    {.name = "ang_velocity_z",
     .field = &Weights::ang_velocity_z,
     .weight = &EndEffectorKinematicsWeights::contactAngularVelocityErrorWeight,
     .axis = 2},
}};

/** A weight of the block that the kinematics costs have no term for. */
struct UnusedWeightField {
  const char* absl_nonnull name;
  double Weights::*absl_nonnull field;
};

// The acceleration weights, which only the whole-body MPC's foot cost weighs (EndEffectorDynamicsWeights).
constexpr std::array<UnusedWeightField, 6> kAccelerationWeightFields = {{
    {.name = "lin_acceleration_x", .field = &Weights::lin_acceleration_x},
    {.name = "lin_acceleration_y", .field = &Weights::lin_acceleration_y},
    {.name = "lin_acceleration_z", .field = &Weights::lin_acceleration_z},
    {.name = "ang_acceleration_x", .field = &Weights::ang_acceleration_x},
    {.name = "ang_acceleration_y", .field = &Weights::ang_acceleration_y},
    {.name = "ang_acceleration_z", .field = &Weights::ang_acceleration_z},
}};

}  // namespace

absl::StatusOr<EndEffectorKinematicsWeights> endEffectorKinematicsWeightsFromConfig(const mpc_config::TaskSpaceWeights& weights,
                                                                                    absl::string_view path) {
  EndEffectorKinematicsWeights result;
  for (const KinematicsWeightField& entry : kKinematicsWeightFields) {
    const double value = weights.*entry.field;
    if (!std::isfinite(value)) {
      return absl::InvalidArgumentError(absl::StrCat(path, ".", entry.name, " is ", value, ", but a weight must be finite."));
    }
    (result.*entry.weight)(entry.axis) = value;
  }
  for (const UnusedWeightField& entry : kAccelerationWeightFields) {
    const double value = weights.*entry.field;
    if (value != 0.0) {
      return absl::InvalidArgumentError(absl::StrCat(path, ".", entry.name, " is ", value,
                                                     ", but the task-space kinematics costs of the centroidal MPC weigh no acceleration: "
                                                     "set it to 0 or leave it out."));
    }
  }
  return result;
}

std::vector<std::string> footCostPhasesNames() {
  return {std::string(kSwingFootCostPhases), std::string(kSwingAndStanceFootCostPhases)};
}

absl::StatusOr<bool> footCostActiveInStanceFromName(absl::string_view name, absl::string_view path) {
  if (name == kSwingFootCostPhases) return false;
  if (name == kSwingAndStanceFootCostPhases) return true;
  return absl::InvalidArgumentError(absl::StrCat(
      path, " is '", name,
      "', which names no contact phases of the foot cost; valid names are: ", absl::StrJoin(footCostPhasesNames(), ", "), "."));
}

absl::StatusOr<TaskSpaceFootCostSettings> taskSpaceFootCostFromConfig(const mpc_config::TaskSpaceCostConfig& footCost) {
  if (!footCost.name.empty() || !footCost.link_name.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("task_space_foot_cost sets name \"", footCost.name, "\" and link_name \"",
                                                   footCost.link_name,
                                                   "\", which only the costs of task_space_costs have: the foot cost tracks the "
                                                   "contact frames of model_settings."));
  }
  TaskSpaceFootCostSettings settings;
  ASSIGN_OR_RETURN(settings.weights, endEffectorKinematicsWeightsFromConfig(footCost.weights, "task_space_foot_cost.weights"));
  ASSIGN_OR_RETURN(settings.activeInStance, footCostActiveInStanceFromName(footCost.active_phases, "task_space_foot_cost.active_phases"));
  return settings;
}

absl::StatusOr<std::vector<TaskSpaceLinkCostSettings>> taskSpaceLinkCostsFromConfig(
    const std::vector<mpc_config::TaskSpaceCostConfig>& costs) {
  std::vector<TaskSpaceLinkCostSettings> settings;
  settings.reserve(costs.size());
  absl::flat_hash_set<std::string> names;
  for (size_t index = 0; index < costs.size(); ++index) {
    const mpc_config::TaskSpaceCostConfig& cost = costs[index];
    if (cost.name.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("task_space_costs[", index, "] has no name; the name names the cost term <name>_TaskSpaceKinematicsCost."));
    }
    const std::string where = absl::StrCat("task_space_costs[name=", cost.name, "]");
    if (!names.insert(cost.name).second) {
      return absl::InvalidArgumentError(absl::StrCat(where, " is the second cost of that name; give each cost its own name."));
    }
    if (cost.link_name.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(where, " has no link_name, the frame the cost tracks."));
    }
    if (cost.active_phases != kSwingFootCostPhases) {
      return absl::InvalidArgumentError(absl::StrCat(where, " sets active_phases \"", cost.active_phases,
                                                     "\", which only task_space_foot_cost has: a link cost does not depend on contact."));
    }
    TaskSpaceLinkCostSettings& entry = settings.emplace_back();
    entry.name = cost.name;
    entry.linkName = cost.link_name;
    ASSIGN_OR_RETURN(entry.weights, endEffectorKinematicsWeightsFromConfig(cost.weights, absl::StrCat(where, ".weights")));
  }
  return settings;
}

}  // namespace ocs2::humanoid
