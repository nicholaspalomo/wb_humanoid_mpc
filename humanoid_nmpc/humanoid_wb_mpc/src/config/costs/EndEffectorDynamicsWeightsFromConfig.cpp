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

#include "humanoid_wb_mpc/config/costs/EndEffectorDynamicsWeightsFromConfig.h"

#include <array>
#include <cmath>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"
#include "humanoid_mpc_config/task_space_cost_config.nproto.h"
#include "humanoid_mpc_config/task_space_weights.nproto.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsCostHelpers.h"

namespace ocs2::humanoid {
namespace {

using Weights = mpc_config::TaskSpaceWeights;
using DynamicsWeights = EndEffectorDynamicsWeights;

/** A weight of the task_space_weights block and the entry of EndEffectorDynamicsWeights it becomes. */
struct DynamicsWeightField {
  const char* absl_nonnull name;
  double Weights::*absl_nonnull field;
  vector3_t DynamicsWeights::*absl_nonnull weight;
  Eigen::Index axis;
};

// The eighteen weights of a whole-body task-space cost, in the order of EndEffectorDynamicsWeights::toVector().
constexpr std::array<DynamicsWeightField, 18> kDynamicsWeightFields = {{
    {.name = "pos_x", .field = &Weights::pos_x, .weight = &DynamicsWeights::contactPositionErrorWeight, .axis = 0},
    {.name = "pos_y", .field = &Weights::pos_y, .weight = &DynamicsWeights::contactPositionErrorWeight, .axis = 1},
    {.name = "pos_z", .field = &Weights::pos_z, .weight = &DynamicsWeights::contactPositionErrorWeight, .axis = 2},
    {.name = "orientation_x", .field = &Weights::orientation_x, .weight = &DynamicsWeights::contactOrientationErrorWeight, .axis = 0},
    {.name = "orientation_y", .field = &Weights::orientation_y, .weight = &DynamicsWeights::contactOrientationErrorWeight, .axis = 1},
    {.name = "orientation_z", .field = &Weights::orientation_z, .weight = &DynamicsWeights::contactOrientationErrorWeight, .axis = 2},
    {.name = "lin_velocity_x", .field = &Weights::lin_velocity_x, .weight = &DynamicsWeights::contactLinearVelocityErrorWeight, .axis = 0},
    {.name = "lin_velocity_y", .field = &Weights::lin_velocity_y, .weight = &DynamicsWeights::contactLinearVelocityErrorWeight, .axis = 1},
    {.name = "lin_velocity_z", .field = &Weights::lin_velocity_z, .weight = &DynamicsWeights::contactLinearVelocityErrorWeight, .axis = 2},
    {.name = "ang_velocity_x", .field = &Weights::ang_velocity_x, .weight = &DynamicsWeights::contactAngularVelocityErrorWeight, .axis = 0},
    {.name = "ang_velocity_y", .field = &Weights::ang_velocity_y, .weight = &DynamicsWeights::contactAngularVelocityErrorWeight, .axis = 1},
    {.name = "ang_velocity_z", .field = &Weights::ang_velocity_z, .weight = &DynamicsWeights::contactAngularVelocityErrorWeight, .axis = 2},
    {.name = "lin_acceleration_x",
     .field = &Weights::lin_acceleration_x,
     .weight = &DynamicsWeights::contactLinearAccelerationErrorWeight,
     .axis = 0},
    {.name = "lin_acceleration_y",
     .field = &Weights::lin_acceleration_y,
     .weight = &DynamicsWeights::contactLinearAccelerationErrorWeight,
     .axis = 1},
    {.name = "lin_acceleration_z",
     .field = &Weights::lin_acceleration_z,
     .weight = &DynamicsWeights::contactLinearAccelerationErrorWeight,
     .axis = 2},
    {.name = "ang_acceleration_x",
     .field = &Weights::ang_acceleration_x,
     .weight = &DynamicsWeights::contactAngularAccelerationErrorWeight,
     .axis = 0},
    {.name = "ang_acceleration_y",
     .field = &Weights::ang_acceleration_y,
     .weight = &DynamicsWeights::contactAngularAccelerationErrorWeight,
     .axis = 1},
    {.name = "ang_acceleration_z",
     .field = &Weights::ang_acceleration_z,
     .weight = &DynamicsWeights::contactAngularAccelerationErrorWeight,
     .axis = 2},
}};

// LINT.IfChange(inert_foot_cost_weights)
// The weights of task_space_weights that EndEffectorDynamicsFootCost multiplies by an error that is zero by construction:
// its position error is zero (the swing height is the swing constraint's). Its orientation error is the shortest arc from
// the foot's z axis to the plane's normal, which has a component about the world z axis once the normal is tilted (a
// swing pitch), so orientation_z is not among them.
constexpr std::array<DynamicsWeightField, 3> kInertFootCostWeightFields = {{
    kDynamicsWeightFields[0],
    kDynamicsWeightFields[1],
    kDynamicsWeightFields[2],
}};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/src/cost/EndEffectorDynamicsFootCost.cpp:foot_cost_errors, //humanoid_nmpc/humanoid_wb_mpc/src/cost/EndEffectorDynamicsFootCost.cpp:foot_cost_ground_normal)
// clang-format on

}  // namespace

absl::StatusOr<EndEffectorDynamicsWeights> endEffectorDynamicsWeightsFromConfig(const mpc_config::TaskSpaceWeights& weights,
                                                                                absl::string_view path) {
  EndEffectorDynamicsWeights result;
  for (const DynamicsWeightField& entry : kDynamicsWeightFields) {
    const double value = weights.*entry.field;
    if (!std::isfinite(value)) {
      return absl::InvalidArgumentError(absl::StrCat(path, ".", entry.name, " is ", value, ", but a weight must be finite."));
    }
    (result.*entry.weight)(entry.axis) = value;
  }
  return result;
}

absl::StatusOr<EndEffectorDynamicsWeights> wholeBodyFootCostWeightsFromConfig(const mpc_config::TaskSpaceCostConfig& footCost) {
  if (!footCost.name.empty() || !footCost.link_name.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("task_space_foot_cost sets name \"", footCost.name, "\" and link_name \"",
                                                   footCost.link_name,
                                                   "\", which only the costs of task_space_costs have: the foot cost tracks the "
                                                   "contact frames of model_settings."));
  }
  ASSIGN_OR_RETURN(const bool activeInStance, footCostActiveInStanceFromName(footCost.active_phases, "task_space_foot_cost.active_phases"));
  if (activeInStance) {
    return absl::InvalidArgumentError(absl::StrCat("task_space_foot_cost.active_phases is \"", footCost.active_phases,
                                                   "\", but the whole-body MPC's foot cost weighs a swing foot only: set it to \"",
                                                   kSwingFootCostPhases, "\" or leave it out."));
  }
  for (const DynamicsWeightField& entry : kInertFootCostWeightFields) {
    if (const double value = footCost.weights.*entry.field; value != 0.0) {
      return absl::InvalidArgumentError(absl::StrCat("task_space_foot_cost.weights.", entry.name, " is ", value,
                                                     ", but the whole-body foot cost multiplies it by an error that is zero by "
                                                     "construction, so it would weigh nothing: set it to 0 or leave it out."));
    }
  }
  return endEffectorDynamicsWeightsFromConfig(footCost.weights, "task_space_foot_cost.weights");
}

}  // namespace ocs2::humanoid
