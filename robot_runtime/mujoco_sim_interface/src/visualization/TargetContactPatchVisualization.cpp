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

#include "mujoco_sim_interface/visualization/TargetContactPatchVisualization.h"

#include <array>
#include <vector>

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/MujocoUtils.h"

namespace robot::mujoco_sim_interface {

namespace {
constexpr int kRgbChannels = 3;  // a hue: red, green, blue
constexpr std::array<std::array<float, kRgbChannels>, 4> kPatchHues = {{
    {0.25f, 0.85f, 1.00f},  // contact point 0 (left foot): cyan
    {1.00f, 0.55f, 0.15f},  // contact point 1 (right foot): orange
    {0.75f, 0.35f, 1.00f},  // further contact points: violet, green
    {0.45f, 1.00f, 0.45f},
}};
// The alpha and the emission of each kind of patch.
constexpr float kSwingInFlightAlpha = 0.85f;
constexpr float kSwingInFlightEmission = 0.8f;
constexpr float kNextSwingAlpha = 0.45f;
constexpr float kNextSwingEmission = 0.4f;
constexpr float kStanceAlpha = 0.35f;
constexpr float kStanceEmission = 0.2f;
// A polygon: a contact point with fewer configured corners draws the default
// patch.
constexpr int kMinConfiguredCorners = 3;
}  // namespace

ContactPatchStyle TargetContactPatchVisualization::styleFor(
    size_t contact, const TargetContactPatch& patch) {
  const std::array<float, kRgbChannels>& hue =
      kPatchHues[contact % kPatchHues.size()];
  ContactPatchStyle style;
  switch (patch.kind) {
    case TargetContactPatch::Kind::kSwingInFlight:  // the step being executed:
                                                    // bright and filled
      style.rgba = {hue[kRedChannel], hue[kGreenChannel], hue[kBlueChannel],
                    kSwingInFlightAlpha};
      style.fill = true;
      style.arrow = true;
      style.emission = kSwingInFlightEmission;
      return style;
    case TargetContactPatch::Kind::kNextSwing:  // the step after: translucent
      style.rgba = {hue[kRedChannel], hue[kGreenChannel], hue[kBlueChannel],
                    kNextSwingAlpha};
      style.fill = true;
      style.arrow = true;
      style.emission = kNextSwingEmission;
      return style;
    case TargetContactPatch::Kind::kStance:
      break;
  }
  // Where the foot is held: a faint outline. Also a kind outside the
  // enumerators, which a patch decoded from a newer controller can carry.
  style.rgba = {hue[kRedChannel], hue[kGreenChannel], hue[kBlueChannel],
                kStanceAlpha};
  style.fill = false;
  style.arrow = false;
  style.emission = kStanceEmission;
  return style;
}

void TargetContactPatchVisualization::addSceneGeoms(
    const VisualizationFrame& frame) {
  if (frame.sim == nullptr || frame.scene == nullptr) return;
  frame.sim->copyTargetContactPatches(scratch_);
  const std::vector<ContactPatchCorners>& configured =
      frame.sim->getConfig().contactPatchCorners;
  const ContactPatchCorners fallback = defaultContactPatchCorners();
  for (size_t contact = 0; contact < scratch_.size(); ++contact) {
    const TargetContactPatch& patch = scratch_[contact];
    if (!patch.valid) continue;
    const bool hasCorners = contact < configured.size() &&
                            configured[contact].size() >= kMinConfiguredCorners;
    addContactPatchGeoms(frame.scene, patch,
                         hasCorners ? configured[contact] : fallback,
                         styleFor(contact, patch));
  }
}

}  // namespace robot::mujoco_sim_interface
