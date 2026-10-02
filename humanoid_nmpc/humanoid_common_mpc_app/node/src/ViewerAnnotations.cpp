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

#include "humanoid_common_mpc_app/node/ViewerAnnotations.h"

#include <cmath>

namespace ocs2::humanoid::node {

void targetContactPatchToProto(const TargetContactPose& pose, humanoid_mpc_msgs::TargetContactPatch* patch) {
  patch->set_valid(pose.valid && pose.position.allFinite() && std::isfinite(pose.height) && std::isfinite(pose.yaw));
  switch (pose.kind) {
    case TargetContactPose::Kind::SWING_IN_FLIGHT:
      patch->set_kind(humanoid_mpc_msgs::TargetContactPatch::KIND_SWING_IN_FLIGHT);
      break;
    case TargetContactPose::Kind::NEXT_SWING:
      patch->set_kind(humanoid_mpc_msgs::TargetContactPatch::KIND_NEXT_SWING);
      break;
    case TargetContactPose::Kind::STANCE:
    default:
      patch->set_kind(humanoid_mpc_msgs::TargetContactPatch::KIND_STANCE);
      break;
  }
  patch->set_x(pose.position(0));
  patch->set_y(pose.position(1));
  patch->set_z(pose.height);
  patch->set_yaw(pose.yaw);
  patch->set_yaw_planned(pose.yawPlanned);
}

void fillViewerAnnotations(const feet_array_t<TargetContactPose>* targetContactPoses,
                           const WalkingVelocityCommand& scaledVelocityCommand,
                           humanoid_mpc_msgs::ViewerAnnotations* annotations) {
  annotations->Clear();
  if (targetContactPoses != nullptr) {
    for (const TargetContactPose& pose : *targetContactPoses) {
      targetContactPatchToProto(pose, annotations->add_target_contact_patches());
    }
  }
  annotations->set_scaled_velocity_x(scaledVelocityCommand.linear_velocity_x);
  annotations->set_scaled_velocity_y(scaledVelocityCommand.linear_velocity_y);
  annotations->set_scaled_yaw_rate(scaledVelocityCommand.angular_velocity_z);
}

}  // namespace ocs2::humanoid::node
