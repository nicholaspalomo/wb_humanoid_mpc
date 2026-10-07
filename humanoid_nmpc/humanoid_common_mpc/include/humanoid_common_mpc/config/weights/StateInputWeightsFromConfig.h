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

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_mpc_config/acom_weights.nproto.h"
#include "humanoid_mpc_config/com_weights.nproto.h"
#include "humanoid_mpc_config/input_weights.nproto.h"
#include "humanoid_mpc_config/joint_weights.nproto.h"
#include "humanoid_mpc_config/state_values.nproto.h"
#include "humanoid_mpc_config/state_weights.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

// The task file's weights and states by name as the matrices and vectors the MPC is set up with, on the coordinates of a
// StateInputLayout: a matrix element is `scaling` times its entry, an element off the diagonal or left out `scaling`
// times 0. What the file must give, and what it
// may not, is refused with an InvalidArgument that names the field path (`fieldPath`, "state_weights" in a task
// file) and lists every problem at once: a missing block, joint or contact; a block, joint list, joint or contact the
// MPC does not have (a fixed joint said so); a name given twice; a value that is not finite.

/**
 * The diagonal state weight matrix (stateDimension(layout) square) of `weights`: Q of state_weights, Q_final of
 * final_state_weights, before the terminal scaling and before any zeroing of the base pose block.
 */
absl::StatusOr<matrix_t> stateWeightsFromConfig(const mpc_config::StateWeights& weights,
                                                const StateInputLayout& layout,
                                                absl::string_view fieldPath);

/** The state vector (stateDimension(layout)) of `values`: the initial state of initial_state. */
absl::StatusOr<vector_t> stateValuesFromConfig(const mpc_config::StateValues& values,
                                               const StateInputLayout& layout,
                                               absl::string_view fieldPath);

/**
 * The diagonal input weight matrix (inputDimension(layout) square) of `weights`: R of input_weights, in wrench space,
 * which the MPC transforms into basis space for basis-vector contact inputs.
 */
absl::StatusOr<matrix_t> inputWeightsFromConfig(const mpc_config::InputWeights& weights,
                                                const StateInputLayout& layout,
                                                absl::string_view fieldPath);

/**
 * The diagonal 3x3 center-of-mass weight of com_and_acom_tracking_cost (formerly Q_com), rows x, y, z. A block that
 * names none of them is refused: the cost would weigh nothing.
 */
absl::StatusOr<matrix_t> comWeightsFromConfig(const mpc_config::ComWeights& weights, absl::string_view fieldPath);

/**
 * The diagonal 3x3 angular-center-of-mass weight of com_and_acom_tracking_cost (formerly Q_acom), rows yaw, pitch, roll
 * (Euler ZYX). A block that names none of them is refused.
 */
absl::StatusOr<matrix_t> acomWeightsFromConfig(const mpc_config::AcomWeights& weights, absl::string_view fieldPath);

/**
 * The task file's terminal_cost_scaling, the factor of final_state_weights and of the terminal instance of
 * com_and_acom_tracking_cost; InvalidArgument when the file does not give it or it is not finite.
 */
absl::StatusOr<scalar_t> terminalCostScalingFromConfig(const mpc_config::TaskFile& task);

/**
 * The torque cost of one leg (left_leg_torque_cost, right_leg_torque_cost): its joints in the order the block lists
 * them, each weighing `scaling` times its value. Refuses a block without a joint, a joint given twice, a fixed joint and
 * a name that is not a joint of the MPC model.
 */
absl::StatusOr<ExternalTorqueQuadraticCostAD::Config> legTorqueCostFromConfig(const mpc_config::JointWeights& weights,
                                                                              const StateInputLayout& layout,
                                                                              absl::string_view fieldPath);

/**
 * The joint torque weights of the whole-body MPC's joint_torque_cost (joint_torque_weights): one per joint of the MPC
 * model, in its order, `scaling` times the value the block names it with; every joint exactly once.
 */
absl::StatusOr<vector_t> jointTorqueWeightsFromConfig(const mpc_config::JointWeights& weights,
                                                      const StateInputLayout& layout,
                                                      absl::string_view fieldPath);

}  // namespace ocs2::humanoid
