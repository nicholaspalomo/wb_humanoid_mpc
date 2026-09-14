/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include <vector>

#include "mujoco_sim_interface/MujocoContactPatch.h"
#include "mujoco_sim_interface/visualization/MujocoVisualization.h"

namespace robot::mujoco_sim_interface {

/**
 * The contact patch of every foot (MujocoSimConfig::contactPatchCorners) at the pose the controller wants it on the
 * ground (MujocoSimInterface::copyTargetContactPatches): bright and filled for the landing pose of the swing in flight,
 * translucent for the foot's next swing, a faint outline for a foot held in place. The arrow is the patch's x axis.
 */
class TargetContactPatchVisualization final : public MujocoVisualization {
 public:
  std::string name() const override { return "target_contact_patches"; }
  std::string description() const override { return "contact patch of every foot at the planner's target position and yaw"; }
  char hotkey() const override { return 'g'; }
  void addSceneGeoms(const VisualizationFrame& frame) override;

  /** Colour and fill of a patch: one hue per contact point, the kind sets the intensity and what is drawn. */
  static ContactPatchStyle styleFor(size_t contact, const TargetContactPatch& patch);

 private:
  std::vector<TargetContactPatch> scratch_;
};

}  // namespace robot::mujoco_sim_interface
