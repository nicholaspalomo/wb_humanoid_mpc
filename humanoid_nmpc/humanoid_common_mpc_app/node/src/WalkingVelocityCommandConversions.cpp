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

#include "humanoid_common_mpc_app/node/WalkingVelocityCommandConversions.h"

#include <algorithm>
#include <cmath>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::node {
namespace {

absl::Status checkFinite(absl::string_view field, double value) {
  if (std::isfinite(value)) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat("WalkingVelocityCommand.", field, " is ", value, "; every value must be finite"));
}

}  // namespace

WalkingVelocityCommand clampWalkingVelocityCommand(const WalkingVelocityCommand& command) {
  WalkingVelocityCommand clamped;
  clamped.linear_velocity_x = std::clamp(command.linear_velocity_x, -kMaxNormalizedVelocity, kMaxNormalizedVelocity);
  clamped.linear_velocity_y = std::clamp(command.linear_velocity_y, -kMaxNormalizedVelocity, kMaxNormalizedVelocity);
  clamped.desired_pelvis_height = std::clamp(command.desired_pelvis_height, kMinPelvisHeight, kMaxPelvisHeight);
  clamped.angular_velocity_z = std::clamp(command.angular_velocity_z, -kMaxNormalizedVelocity, kMaxNormalizedVelocity);
  return clamped;
}

absl::StatusOr<WalkingVelocityCommand> walkingVelocityCommandFromProto(const humanoid_mpc_msgs::WalkingVelocityCommand& message) {
  for (const absl::Status& status :
       {checkFinite("linear_velocity_x", message.linear_velocity_x()), checkFinite("linear_velocity_y", message.linear_velocity_y()),
        checkFinite("desired_pelvis_height", message.desired_pelvis_height()),
        checkFinite("angular_velocity_z", message.angular_velocity_z())}) {
    if (!status.ok()) return status;
  }
  return clampWalkingVelocityCommand(WalkingVelocityCommand(message.linear_velocity_x(), message.linear_velocity_y(),
                                                            message.desired_pelvis_height(), message.angular_velocity_z()));
}

}  // namespace ocs2::humanoid::node
