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

#include "humanoid_common_mpc/config/costs/CollisionConstraintFromConfig.h"

#include <cmath>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"
#include "humanoid_mpc_config/collision_constraint_config.nproto.h"

namespace ocs2::humanoid {
namespace {

/** InvalidArgument naming `collision_constraint.<field>` unless `value` is finite. */
absl::Status checkFinite(absl::string_view field, double value) {
  if (std::isfinite(value)) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat("collision_constraint.", field, " is ", value, ", but it must be finite."));
}

}  // namespace

absl::StatusOr<FootCollisionConstraint::Config> footCollisionConstraintConfigFromConfig(
    const mpc_config::CollisionConstraintConfig& collisionConstraint) {
  const mpc_config::CollisionConstraintConfig::Foot& foot = collisionConstraint.foot;
  const mpc_config::CollisionConstraintConfig::Knee& knee = collisionConstraint.knee;
  RETURN_IF_ERROR(checkFinite("foot.foot_collision_sphere_radius", foot.foot_collision_sphere_radius));
  RETURN_IF_ERROR(checkFinite("knee.knee_collision_sphere_radius", knee.knee_collision_sphere_radius));
  FootCollisionConstraint::Config config;
  config.leftAnkleFrame = foot.left_ankle_frame;
  config.rightAnkleFrame = foot.right_ankle_frame;
  config.footCollisionSphereRadius = foot.foot_collision_sphere_radius;
  config.leftKneeFrame = knee.left_knee_frame;
  config.rightKneeFrame = knee.right_knee_frame;
  config.kneeCollisionSphereRadius = knee.knee_collision_sphere_radius;
  return config;
}

absl::StatusOr<PieceWisePolynomialBarrierPenalty::Config> footCollisionBarrierFromConfig(
    const mpc_config::CollisionConstraintConfig& collisionConstraint) {
  RETURN_IF_ERROR(checkFinite("mu", collisionConstraint.mu));
  RETURN_IF_ERROR(checkFinite("delta", collisionConstraint.delta));
  return PieceWisePolynomialBarrierPenalty::Config(collisionConstraint.mu, collisionConstraint.delta);
}

}  // namespace ocs2::humanoid
