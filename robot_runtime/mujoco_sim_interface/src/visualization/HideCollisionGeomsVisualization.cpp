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

#include "mujoco_sim_interface/visualization/HideCollisionGeomsVisualization.h"

#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/visualization/MujocoVisualization.h"

namespace robot::mujoco_sim_interface {

namespace {
// The body of the world, whose geoms (the floor, the terrain) always show.
constexpr int kWorldBody = 0;

bool collides(const mjModel& model, int geom) {
  return model.geom_contype[geom] != 0 || model.geom_conaffinity[geom] != 0;
}

bool hasVisualGeom(const mjModel& model, int body) {
  const int first = model.body_geomadr[body];
  if (first < 0) return false;
  for (int geom = first; geom < first + model.body_geomnum[body]; ++geom) {
    if (!collides(model, geom)) return true;
  }
  return false;
}
}  // namespace

bool isHiddenCollisionGeom(const mjModel& model, int geom) {
  const int body = model.geom_bodyid[geom];
  return body != kWorldBody && collides(model, geom) &&
         hasVisualGeom(model, body);
}

void HideCollisionGeomsVisualization::addSceneGeoms(
    const VisualizationFrame& frame) {
  if (frame.model == nullptr || frame.scene == nullptr) return;
  const mjModel& model = *frame.model;
  mjvScene& scene = *frame.scene;
  int kept = 0;
  for (int index = 0; index < scene.ngeom; ++index) {
    const mjvGeom& geom = scene.geoms[index];
    const bool hidden = geom.objtype == mjOBJ_GEOM && geom.objid >= 0 &&
                        geom.objid < model.ngeom &&
                        isHiddenCollisionGeom(model, geom.objid);
    if (hidden) continue;
    if (kept != index) scene.geoms[kept] = geom;
    ++kept;
  }
  scene.ngeom = kept;
}

}  // namespace robot::mujoco_sim_interface
