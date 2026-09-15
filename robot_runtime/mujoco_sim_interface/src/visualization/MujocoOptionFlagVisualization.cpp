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

#include "mujoco_sim_interface/visualization/MujocoOptionFlagVisualization.h"

#include <utility>

#include "mujoco_sim_interface/MujocoSimInterface.h"

namespace robot::mujoco_sim_interface {

MujocoOptionFlagVisualization::MujocoOptionFlagVisualization(std::string name, std::string description, char hotkey, int flag)
    : name_(std::move(name)), description_(std::move(description)), hotkey_(hotkey), flag_(flag) {
  setEnabled(false);
}

std::unique_ptr<MujocoOptionFlagVisualization> MujocoOptionFlagVisualization::contactPoints() {
  return std::make_unique<MujocoOptionFlagVisualization>("mj_contact_points", "MuJoCo's markers at the physical contact points", 'c',
                                                         mjVIS_CONTACTPOINT);
}
std::unique_ptr<MujocoOptionFlagVisualization> MujocoOptionFlagVisualization::contactForces() {
  return std::make_unique<MujocoOptionFlagVisualization>("mj_contact_forces", "MuJoCo's normal and friction force arrows at the contacts",
                                                         'f', mjVIS_CONTACTFORCE);
}
std::unique_ptr<MujocoOptionFlagVisualization> MujocoOptionFlagVisualization::centerOfMass() {
  return std::make_unique<MujocoOptionFlagVisualization>("mj_com", "MuJoCo's centre of mass markers of the bodies", 'm', mjVIS_COM);
}
std::unique_ptr<MujocoOptionFlagVisualization> MujocoOptionFlagVisualization::inertia() {
  return std::make_unique<MujocoOptionFlagVisualization>("mj_inertia", "MuJoCo's equivalent inertia ellipsoids", 'i', mjVIS_INERTIA);
}
std::unique_ptr<MujocoOptionFlagVisualization> MujocoOptionFlagVisualization::convexHull() {
  return std::make_unique<MujocoOptionFlagVisualization>("mj_convex_hull", "MuJoCo's convex hulls of the meshes", 'h', mjVIS_CONVEXHULL);
}
std::unique_ptr<MujocoOptionFlagVisualization> MujocoOptionFlagVisualization::transparency() {
  return std::make_unique<MujocoOptionFlagVisualization>("mj_transparent", "model transparency (30% alpha, x-ray view)", 't',
                                                         kTransparency);
}

void MujocoOptionFlagVisualization::beforeSceneUpdate(const VisualizationFrame& frame) {
  if (flag_ == kTransparency) {
    // The alpha lives in the model shared with the physics, so it is written only when the state changes.
    if (frame.sim == nullptr || frame.sim->getModel() == nullptr || transparencyApplied_ == enabled()) return;
    const mjModel* model = frame.sim->getModel();
    const float alpha = enabled() ? kTransparentAlpha : 1.0f;
    for (int i = 0; i < model->ngeom; ++i) model->geom_rgba[4 * i + 3] = alpha;
    transparencyApplied_ = enabled();
    return;
  }
  if (frame.options == nullptr || flag_ < 0 || flag_ >= mjNVISFLAG) return;
  frame.options->flags[flag_] = enabled() ? 1 : 0;
}

}  // namespace robot::mujoco_sim_interface
