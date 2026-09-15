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

#include "mujoco_sim_interface/visualization/ExternalForceVisualization.h"

#include <cmath>
#include <cstring>

#include "mujoco_sim_interface/MujocoSimInterface.h"

namespace robot::mujoco_sim_interface {

void ExternalForceVisualization::addSceneGeoms(const VisualizationFrame& frame) {
  if (frame.sim == nullptr || frame.state == nullptr || frame.scene == nullptr) return;
  const mjModel* model = frame.sim->getModel();
  const mjData* data = frame.state->data;
  mjvScene& scene = *frame.scene;
  if (model == nullptr || data == nullptr) return;

  for (int body_id = 0; body_id < model->nbody; ++body_id) {
    const double* force = &data->xfrc_applied[6 * body_id];
    const double fx = force[0];
    const double fy = force[1];
    const double fz = force[2];

    const double magnitude = std::sqrt(fx * fx + fy * fy + fz * fz);
    if (magnitude < 1e-6) continue;
    if (scene.ngeom >= scene.maxgeom) return;  // out of geom space

    const double* xpos = &data->xpos[3 * body_id];
    mjvGeom* arrow = &scene.geoms[scene.ngeom++];
    std::memset(arrow, 0, sizeof(mjvGeom));

    // Scale factor that grows with force magnitude
    const double base_scale = 0.1;    // Minimum arrow length
    const double force_scale = 0.05;  // How much to scale with force
    const double scale = base_scale + force_scale * magnitude;

    const mjtNum from[3] = {xpos[0], xpos[1], xpos[2]};
    const mjtNum to[3] = {xpos[0] + scale * (fx / magnitude), xpos[1] + scale * (fy / magnitude), xpos[2] + scale * (fz / magnitude)};
    mjv_connector(arrow, mjGEOM_ARROW, 0.005, from, to);

    arrow->rgba[0] = 0.5f;
    arrow->rgba[1] = 1.0f;
    arrow->rgba[2] = 0.0f;
    arrow->rgba[3] = 1.0f;
    arrow->category = mjCAT_DECOR;
    arrow->emission = 1.0f;  // Makes it glow/bright
  }
}

}  // namespace robot::mujoco_sim_interface
