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

#include "ocs2_core/Types.h"

#include "legged_common_mpc/command/WalkingVelocityCommand.h"
#include "legged_common_mpc/common/Types.h"

namespace ocs2::legged {

/**
 * The operator's walking command as the operator bus carries it: the stick
 * deflections of the base velocities, each a dimensionless fraction of the
 * reference file's cmd_max_... limit (the `lnorm` qualifier), and the pelvis
 * height in meters. ProceduralMpcMotionManager::setAndScaleVelocityCommand()
 * multiplies the fractions by the limits into a WalkingVelocityCommand. A
 * plain value, free of any transport; the conversion from the bus message
 * lives in legged_common_mpc_app/node (WalkingVelocityCommandConversions.h).
 * Its vector form, toVector(), has WalkingVelocityCommand's layout.
 */
struct NormalizedWalkingVelocityCommand {
 public:
  NormalizedWalkingVelocityCommand() = default;
  NormalizedWalkingVelocityCommand(
      scalar_t cmd_lnorm_linvel_base_rt_world_ewrt_heading_x,
      scalar_t cmd_lnorm_linvel_base_rt_world_ewrt_heading_y,
      scalar_t cmd_pos_base_rt_surface_ewrt_world_z,
      scalar_t cmd_lnorm_angvel_base_rt_world_ewrt_world_z)
      : cmd_lnorm_linvel_base_rt_world_ewrt_heading_x(
            cmd_lnorm_linvel_base_rt_world_ewrt_heading_x),
        cmd_lnorm_linvel_base_rt_world_ewrt_heading_y(
            cmd_lnorm_linvel_base_rt_world_ewrt_heading_y),
        cmd_pos_base_rt_surface_ewrt_world_z(
            cmd_pos_base_rt_surface_ewrt_world_z),
        cmd_lnorm_angvel_base_rt_world_ewrt_world_z(
            cmd_lnorm_angvel_base_rt_world_ewrt_world_z) {}
  /** From the vector form [v_x, v_y, pelvis height, yaw rate], the velocities
   * normalized. */
  explicit NormalizedWalkingVelocityCommand(const vector4_t& lnormVelCommand)
      : cmd_lnorm_linvel_base_rt_world_ewrt_heading_x(
            lnormVelCommand(WalkingVelocityCommand::kLinearVelocityXIndex)),
        cmd_lnorm_linvel_base_rt_world_ewrt_heading_y(
            lnormVelCommand(WalkingVelocityCommand::kLinearVelocityYIndex)),
        cmd_pos_base_rt_surface_ewrt_world_z(
            lnormVelCommand(WalkingVelocityCommand::kBaseHeightIndex)),
        cmd_lnorm_angvel_base_rt_world_ewrt_world_z(
            lnormVelCommand(WalkingVelocityCommand::kAngularVelocityZIndex)) {}

  /** In [-1, 1] once node::clampWalkingVelocityCommand() has run. */
  scalar_t cmd_lnorm_linvel_base_rt_world_ewrt_heading_x = 0.0;
  /** In [-1, 1] once node::clampWalkingVelocityCommand() has run. */
  scalar_t cmd_lnorm_linvel_base_rt_world_ewrt_heading_y = 0.0;
  /** [m] Above the ground. */
  scalar_t cmd_pos_base_rt_surface_ewrt_world_z =
      WalkingVelocityCommand::kNomCmdPosBaseRtSurfaceEwrtWorldZ;
  /** In [-1, 1] once node::clampWalkingVelocityCommand() has run. */
  scalar_t cmd_lnorm_angvel_base_rt_world_ewrt_world_z = 0.0;

  /** The vector form, [v_x, v_y, pelvis height, yaw rate]. */
  vector4_t toVector() const {
    return vector4_t(cmd_lnorm_linvel_base_rt_world_ewrt_heading_x,
                     cmd_lnorm_linvel_base_rt_world_ewrt_heading_y,
                     cmd_pos_base_rt_surface_ewrt_world_z,
                     cmd_lnorm_angvel_base_rt_world_ewrt_world_z);
  }
};

}  // namespace ocs2::legged
