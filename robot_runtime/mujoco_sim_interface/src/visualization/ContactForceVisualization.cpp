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

#include "mujoco_sim_interface/visualization/ContactForceVisualization.h"

#include <cmath>
#include <cstring>

#include "mujoco_sim_interface/MujocoSimInterface.h"

namespace robot::mujoco_sim_interface {

void ContactForceVisualization::addSceneGeoms(const VisualizationFrame& frame) {
  if (frame.sim == nullptr || frame.state == nullptr || frame.scene == nullptr) return;
  const mjModel* model = frame.sim->getModel();
  const mjData* data = frame.state->data;
  mjvScene& scene = *frame.scene;
  if (model == nullptr || data == nullptr) return;

  for (int i = 0; i < data->ncon; ++i) {
    const mjContact* contact = &data->contact[i];
    if (contact->exclude != 0 || contact->efc_address < 0) continue;  // only active contacts

    mjtNum force[6];
    mj_contactForce(model, data, i, force);

    // Force in the world frame
    double fx = force[0] * contact->frame[0] + force[1] * contact->frame[3] + force[2] * contact->frame[6];
    double fy = force[0] * contact->frame[1] + force[1] * contact->frame[4] + force[2] * contact->frame[7];
    double fz = force[0] * contact->frame[2] + force[1] * contact->frame[5] + force[2] * contact->frame[8];

    // mj_contactForce returns the force ON geom[0] BY geom[1]. If geom[0] is the world, the force on the robot is the
    // negative of the computed force; the drawn vector is negated once more (historical convention of this arrow).
    if (model->geom_bodyid[contact->geom[0]] == 0) {
      fx = -fx;
      fy = -fy;
      fz = -fz;
    }
    fx = -fx;
    fy = -fy;
    fz = -fz;

    const double magnitude = std::sqrt(fx * fx + fy * fy + fz * fz);
    if (magnitude < 1.0) continue;  // only significant forces
    if (scene.ngeom >= scene.maxgeom) return;

    mjvGeom* arrow = &scene.geoms[scene.ngeom++];
    std::memset(arrow, 0, sizeof(mjvGeom));

    const double base_scale = 0.02;
    const double force_scale = 0.002;  // 500N -> 1m
    const double scale = base_scale + force_scale * magnitude;

    const mjtNum from[3] = {contact->pos[0], contact->pos[1], contact->pos[2]};
    const mjtNum to[3] = {contact->pos[0] + scale * (fx / magnitude), contact->pos[1] + scale * (fy / magnitude),
                          contact->pos[2] + scale * (fz / magnitude)};
    mjv_connector(arrow, mjGEOM_ARROW, 0.015, from, to);

    arrow->rgba[0] = 1.0f;
    arrow->rgba[1] = 0.0f;
    arrow->rgba[2] = 0.0f;
    arrow->rgba[3] = 0.8f;
    arrow->category = mjCAT_DECOR;
    arrow->emission = 0.8f;
  }
}

}  // namespace robot::mujoco_sim_interface
