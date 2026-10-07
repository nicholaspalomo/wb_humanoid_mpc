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

#include "humanoid_common_mpc_app/robot/MujocoViewerAnnotator.h"

namespace ocs2::humanoid {
namespace {

robot::mujoco_sim_interface::TargetContactPatch::Kind viewerKind(msgs::TargetContactPatch::Kind kind) {
  switch (kind) {
    case msgs::TargetContactPatch::Kind::kSwingInFlight:
      return robot::mujoco_sim_interface::TargetContactPatch::Kind::kSwingInFlight;
    case msgs::TargetContactPatch::Kind::kNextSwing:
      return robot::mujoco_sim_interface::TargetContactPatch::Kind::kNextSwing;
    case msgs::TargetContactPatch::Kind::kStance:
      return robot::mujoco_sim_interface::TargetContactPatch::Kind::kStance;
  }
  // A value from the wire that is no enumerator (the message is open, as a proto enum is) is drawn as a stance patch.
  return robot::mujoco_sim_interface::TargetContactPatch::Kind::kStance;
}

}  // namespace

MujocoViewerAnnotator::MujocoViewerAnnotator(robot::mujoco_sim_interface::MujocoSimInterface& simulator) : simulator_(simulator) {
  patches_.reserve(kNumContacts);
}

void MujocoViewerAnnotator::apply(const msgs::ViewerAnnotations& annotations) {
  if (!annotations.target_contact_patches.empty()) {
    patches_.resize(annotations.target_contact_patches.size());
    for (size_t contact = 0; contact < patches_.size(); ++contact) {
      const msgs::TargetContactPatch& source = annotations.target_contact_patches[contact];
      robot::mujoco_sim_interface::TargetContactPatch& patch = patches_[contact];
      patch.valid = source.valid;
      patch.kind = viewerKind(source.kind);
      patch.x = source.x;
      patch.y = source.y;
      patch.z = source.z;
      patch.yaw = source.yaw;
      patch.yawPlanned = source.yaw_planned;
    }
    simulator_.setTargetContactPatches(patches_);
  }
  simulator_.setTargetVelocities(annotations.scaled_velocity_x, annotations.scaled_velocity_y, annotations.scaled_yaw_rate);
}

}  // namespace ocs2::humanoid
