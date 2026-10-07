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
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"

#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"
#include "humanoid_mpc_config/collision_constraint_config.nproto.h"

namespace ocs2::humanoid {

/**
 * The spheres of the foot collision constraint from the task file's collision_constraint block: the ankle and knee
 * frames and the sphere radii, with FootCollisionConstraint::Config's own contact and collision-point frames. A frame
 * the block leaves out is empty and a radius 0.
 *
 * @return The configuration; InvalidArgument naming a radius that is not finite.
 */
absl::StatusOr<FootCollisionConstraint::Config> footCollisionConstraintConfigFromConfig(
    const mpc_config::CollisionConstraintConfig& collisionConstraint);

/**
 * The barrier of the soft constraint foot_collision, the block's mu and delta, with
 * PieceWisePolynomialBarrierPenalty::Config's own for the ones the block leaves out.
 *
 * @return The barrier; InvalidArgument naming mu or delta when it is not finite.
 */
absl::StatusOr<PieceWisePolynomialBarrierPenalty::Config> footCollisionBarrierFromConfig(
    const mpc_config::CollisionConstraintConfig& collisionConstraint);

}  // namespace ocs2::humanoid
