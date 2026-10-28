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

#include "absl/status/status.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

/**
 * Checks that what `task` selects runs on the contact set of `modelSettings`
 * (humanoid_nmpc/docs/multi_contact_planning/README.md, section 7.2). On the
 * biped's two 6-DoF feet everything does. On any other set the centroidal MPC
 * runs the terms written per contact for any contact - the dynamics, the cones,
 * zero_wrench and zero_velocity - and the costs of the whole state and input,
 * and refuses those written for exactly two feet: the costs
 * task_space_foot_cost, external_torque_cost and dcm_terminal_cost, the soft
 * constraints foot_collision, foot_collision_cbf, normal_velocity and the
 * contact-implicit terms, the hard constraints normal_velocity and
 * knee_joint_mimic, contact_schedule_source "contact_planner" (its H-LIP and
 * MIQP planners and their contact-planning rules and constraints),
 * contact_input_parameterization "basis_vectors" and any locomotion
 * heuristic.
 *
 * @return OkStatus, or one InvalidArgument naming the contact set and every
 * refused selection by its key; the errors of the conversions it reads the
 * selections with.
 */
absl::Status checkCentroidalContactSet(const ModelSettings& modelSettings,
                                       const mpc_config::TaskFile& task);

}  // namespace ocs2::humanoid
