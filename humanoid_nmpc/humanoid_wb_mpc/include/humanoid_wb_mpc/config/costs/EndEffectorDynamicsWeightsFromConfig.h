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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_mpc_config/task_space_cost_config.nproto.h"
#include "humanoid_mpc_config/task_space_weights.nproto.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsCostHelpers.h"

namespace ocs2::humanoid {

/**
 * The eighteen weights of a whole-body task-space cost: position, orientation, linear and angular velocity and
 * acceleration, x, y, z each. An absent field is 0.
 *
 * @param weights The weights block.
 * @param path The block's path in the task file, for the errors (e.g. "task_space_foot_cost.weights").
 * @return The weights; InvalidArgument naming the field when a weight is not finite (inf or nan).
 */
absl::StatusOr<EndEffectorDynamicsWeights> endEffectorDynamicsWeightsFromConfig(const mpc_config::TaskSpaceWeights& weights,
                                                                                absl::string_view path);

/**
 * The weights of the whole-body MPC's swing-foot cost (EndEffectorDynamicsFootCost) from the task file's
 * task_space_foot_cost.
 *
 * @return The weights; InvalidArgument for the weights' errors, and when the block sets a name or a link_name (the named
 *         costs' fields) or an active_phases other than "swing", which the whole-body foot cost does not have: it weighs
 *         a swing foot only. InvalidArgument naming the field, too, for a weight the foot cost multiplies by an error
 *         that is zero by construction, so that it would weigh nothing: pos_x, pos_y and pos_z (the cost has no
 *         position error; the swing height is the swing constraint's) must be 0 or absent.
 */
absl::StatusOr<EndEffectorDynamicsWeights> wholeBodyFootCostWeightsFromConfig(const mpc_config::TaskSpaceCostConfig& footCost);

}  // namespace ocs2::humanoid
