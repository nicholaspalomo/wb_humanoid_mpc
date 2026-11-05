/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The operator's walking command in physical units: the base velocities in m/s
 * and rad/s and the pelvis height in meters.
 * ProceduralMpcMotionManager::setAndScaleVelocityCommand() makes it from the
 * NormalizedWalkingVelocityCommand the operator bus carries. A plain value,
 * free of any transport.
 */
struct WalkingVelocityCommand {
 public:
  /** The entries of the command's vector form, toVector(): [v_x, v_y, base
   * height, yaw rate]. */
  static constexpr int kLinearVelocityXIndex = 0;
  static constexpr int kLinearVelocityYIndex = 1;
  static constexpr int kBaseHeightIndex = 2;
  static constexpr int kAngularVelocityZIndex = 3;

  /** [m] The pelvis height above the ground of a command that sets none. */
  static constexpr scalar_t kNomCmdPosBaseRtSurfaceEwrtWorldZ = 0.8;

  WalkingVelocityCommand() = default;
  WalkingVelocityCommand(scalar_t cmd_linvel_base_rt_world_ewrt_heading_x,
                         scalar_t cmd_linvel_base_rt_world_ewrt_heading_y,
                         scalar_t cmd_pos_base_rt_surface_ewrt_world_z,
                         scalar_t cmd_angvel_base_rt_world_ewrt_world_z)
      : cmd_linvel_base_rt_world_ewrt_heading_x(
            cmd_linvel_base_rt_world_ewrt_heading_x),
        cmd_linvel_base_rt_world_ewrt_heading_y(
            cmd_linvel_base_rt_world_ewrt_heading_y),
        cmd_pos_base_rt_surface_ewrt_world_z(
            cmd_pos_base_rt_surface_ewrt_world_z),
        cmd_angvel_base_rt_world_ewrt_world_z(
            cmd_angvel_base_rt_world_ewrt_world_z) {}
  explicit WalkingVelocityCommand(const vector4_t& velCommand)
      : cmd_linvel_base_rt_world_ewrt_heading_x(
            velCommand(kLinearVelocityXIndex)),
        cmd_linvel_base_rt_world_ewrt_heading_y(
            velCommand(kLinearVelocityYIndex)),
        cmd_pos_base_rt_surface_ewrt_world_z(velCommand(kBaseHeightIndex)),
        cmd_angvel_base_rt_world_ewrt_world_z(
            velCommand(kAngularVelocityZIndex)) {}
  scalar_t cmd_linvel_base_rt_world_ewrt_heading_x = 0.0;
  scalar_t cmd_linvel_base_rt_world_ewrt_heading_y = 0.0;
  scalar_t cmd_pos_base_rt_surface_ewrt_world_z =
      kNomCmdPosBaseRtSurfaceEwrtWorldZ;  // Above ground
  scalar_t cmd_angvel_base_rt_world_ewrt_world_z = 0.0;

  vector4_t toVector() const {
    return vector4_t(cmd_linvel_base_rt_world_ewrt_heading_x,
                     cmd_linvel_base_rt_world_ewrt_heading_y,
                     cmd_pos_base_rt_surface_ewrt_world_z,
                     cmd_angvel_base_rt_world_ewrt_world_z);
  }
};
}  // namespace ocs2::humanoid
