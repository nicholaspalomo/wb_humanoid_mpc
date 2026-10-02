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

#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"

namespace ocs2::humanoid::node {

/**
 * The ranges of operator/walking_velocity_command (humanoid_mpc_msgs/walking_velocity_command.proto): the velocities are
 * normalized, and ProceduralMpcMotionManager scales them by the command limits of reference.yaml.
 */
inline constexpr scalar_t kMaxNormalizedVelocity = 1.0;
/** [m] The pelvis height of the command is absolute, above the ground. */
inline constexpr scalar_t kMinPelvisHeight = 0.2;
inline constexpr scalar_t kMaxPelvisHeight = 1.0;

/**
 * `command` within the ranges above: the velocities clamped to [-kMaxNormalizedVelocity, kMaxNormalizedVelocity], the
 * pelvis height to [kMinPelvisHeight, kMaxPelvisHeight]. A value that is not finite passes std::clamp() unchanged;
 * walkingVelocityCommandFromProto() refuses it before it gets here. The lockstep closed loop
 * (humanoid_nmpc/humanoid_mpc_validation) clamps the GUI's messages with this, so that the clamp has one home.
 */
WalkingVelocityCommand clampWalkingVelocityCommand(const WalkingVelocityCommand& command);

/**
 * The operator's command of an operator/walking_velocity_command message, for
 * ProceduralMpcMotionManager::setAndScaleVelocityCommand(): clampWalkingVelocityCommand() of its values, as the ROS
 * conversion did.
 *
 * InvalidArgument naming the field when a value is not finite: a NaN would pass std::clamp() unchanged and reach every
 * target trajectory of the MPC.
 */
absl::StatusOr<WalkingVelocityCommand> walkingVelocityCommandFromProto(const humanoid_mpc_msgs::WalkingVelocityCommand& message);

}  // namespace ocs2::humanoid::node
