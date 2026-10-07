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

#include <cstddef>
#include <string>
#include <vector>

#include "humanoid_common_mpc/common/ModelSettings.h"

namespace ocs2::humanoid {

/**
 * The coordinates of an MPC's state and input that the task file's weights and values address by name (state_weights,
 * final_state_weights, input_weights, initial_state, joint_torque_weights; humanoid_nmpc/humanoid_mpc_config/README.md, "Conventions"):
 * which MPC it is, its joints in state order, the joints the task file fixes, and its wrench contacts in input order. The index of each
 * coordinate is the one the MPC's robot model gives it, for n joints and c contacts:
 *
 *   centroidal   state  h_com / m (x, y, z), L / m (x, y, z), base position (x, y, z), base orientation (yaw, pitch,
 *                       roll), joint positions                                                      12 + n
 *                input  per contact: force (x, y, z), moment (x, y, z); then the joint velocities  6 c + n
 *   whole body   state  base position, base orientation, joint positions, base linear velocity (x, y, z), base Euler
 *                       ZYX rates (yaw, pitch, roll), joint velocities                              2 (6 + n)
 *                input  per contact: force, moment; then the joint accelerations                    6 c + n
 *
 * The orientation is the Euler ZYX layout the tuning is written in, also where the MPC's orientation is a quaternion:
 * that MPC maps the vectors built on this layout (humanoid_nmpc/docs/quaternion_base_orientation/README.md).
 */
struct StateInputLayout {
  /** The MPC whose state and input the layout describes. */
  enum class Mpc {
    kCentroidal,
    kWholeBody,
  };

  Mpc mpc = Mpc::kCentroidal;
  // The joints of the MPC model, in state order (ModelSettings::mpcModelJointNames).
  std::vector<std::string> jointNames;
  // The joints the task file fixes (ModelSettings::fixedJointNames): no weight or value may name one.
  std::vector<std::string> fixedJointNames;
  // The wrench contacts, in input order (ModelSettings::contactNames).
  std::vector<std::string> contactNames;
};

/** The layout of the MPC `mpc` of the robot model `settings`: its MPC joints, fixed joints and contacts. */
StateInputLayout stateInputLayout(const ModelSettings& settings, StateInputLayout::Mpc mpc);

/** The size of `layout`'s state: 12 + n (centroidal) or 2 (6 + n) (whole body) for n joints. */
size_t stateDimension(const StateInputLayout& layout);

/** The size of `layout`'s input: six wrench coordinates per contact, then one per joint. */
size_t inputDimension(const StateInputLayout& layout);

}  // namespace ocs2::humanoid
