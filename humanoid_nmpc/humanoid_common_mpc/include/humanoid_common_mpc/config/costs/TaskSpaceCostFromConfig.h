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

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_mpc_config/task_space_cost_config.nproto.h"
#include "humanoid_mpc_config/task_space_weights.nproto.h"

namespace ocs2::humanoid {

// The names of task_space_foot_cost.active_phases: the contact phases in which the foot cost weighs a foot.
// LINT.IfChange(foot_cost_phases_names)
/** A swing foot only (the default, and the only one the whole-body MPC's foot cost has). */
inline constexpr absl::string_view kSwingFootCostPhases = "swing";
/** Every foot, in swing and in stance (CentroidalMpcEndEffectorFootCost's activeInStance). */
inline constexpr absl::string_view kSwingAndStanceFootCostPhases = "swing_and_stance";
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_space_cost_config.proto:active_phases)

/** Every name active_phases accepts. */
std::vector<std::string> footCostPhasesNames();

/**
 * Whether the active_phases `name` weighs a foot in stance as well: false for "swing", true for "swing_and_stance".
 *
 * @param name The name.
 * @param path The field's path in the task file, for the error (e.g. "task_space_foot_cost.active_phases").
 * @return InvalidArgument naming `path` and listing the names, for any other name.
 */
absl::StatusOr<bool> footCostActiveInStanceFromName(absl::string_view name, absl::string_view path);

/** The swing-foot cost of the centroidal MPC (the task file's task_space_foot_cost; the cost task_space_foot_cost). */
struct TaskSpaceFootCostSettings {
  EndEffectorKinematicsWeights weights;
  // Whether the cost also weighs a foot in stance (active_phases "swing_and_stance"; CentroidalMpcEndEffectorFootCost's
  // activeInStance).
  bool activeInStance = false;
};

/** A task-space cost on a named link (an entry of the task file's task_space_costs; the cost task_space_torso_cost). */
struct TaskSpaceLinkCostSettings {
  // The cost's name: its term is "<name>_TaskSpaceKinematicsCost".
  std::string name;
  // The Pinocchio frame it tracks.
  std::string linkName;
  EndEffectorKinematicsWeights weights;
};

/**
 * The weights of a task-space kinematics cost: the position, orientation, linear and angular velocity weights, x, y, z
 * each. An absent field is 0.
 *
 * @param weights The weights block.
 * @param path The block's path in the task file, for the errors (e.g. "task_space_foot_cost.weights").
 * @return The weights; InvalidArgument naming the field when a weight is not finite, or when an acceleration weight is
 *         not 0: the kinematics costs have no acceleration term, so the weight would do nothing.
 */
absl::StatusOr<EndEffectorKinematicsWeights> endEffectorKinematicsWeightsFromConfig(const mpc_config::TaskSpaceWeights& weights,
                                                                                    absl::string_view path);

/**
 * The swing-foot cost from the task file's task_space_foot_cost (formerly task_space_foot_cost_weights): its weights
 * (endEffectorKinematicsWeightsFromConfig()) and active_phases (footCostActiveInStanceFromName()).
 *
 * @return The settings; InvalidArgument for the weights' errors, for an active_phases that is no name of
 *         footCostPhasesNames(), and when the block sets a name or a link_name, which only the named costs of
 *         task_space_costs have.
 */
absl::StatusOr<TaskSpaceFootCostSettings> taskSpaceFootCostFromConfig(const mpc_config::TaskSpaceCostConfig& footCost);

/**
 * The named link costs from the task file's task_space_costs, in file order, as CentroidalMpcInterface read the map
 * of the same name.
 *
 * @return The costs; InvalidArgument naming the entry when its name or link_name is empty, when a name is given twice,
 *         when it sets an active_phases other than "swing" (the foot cost's alone), and for the weights' errors.
 */
absl::StatusOr<std::vector<TaskSpaceLinkCostSettings>> taskSpaceLinkCostsFromConfig(
    const std::vector<mpc_config::TaskSpaceCostConfig>& costs);

}  // namespace ocs2::humanoid
