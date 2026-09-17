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

#include "mujoco_sim_interface/visualization/GroundMarkerGeoms.h"

#include <cstring>

namespace robot::mujoco_sim_interface {

namespace {

mjvGeom* nextGeom(mjvScene* scene) {
  if (scene == nullptr || scene->ngeom >= scene->maxgeom) return nullptr;
  mjvGeom* geom = &scene->geoms[scene->ngeom++];
  std::memset(geom, 0, sizeof(mjvGeom));
  return geom;
}

void paint(mjvGeom* geom, const MarkerColor& color) {
  geom->rgba[0] = color.r;
  geom->rgba[1] = color.g;
  geom->rgba[2] = color.b;
  geom->rgba[3] = color.a;
  geom->category = mjCAT_DECOR;
  geom->emission = 0.6f;
}

}  // namespace

bool addGroundDiscGeom(mjvScene* scene, double x, double y, double radius, const MarkerColor& color) {
  mjvGeom* geom = nextGeom(scene);
  if (geom == nullptr) return false;
  constexpr double kHalfHeight = 0.002;  // a thin cylinder just above the ground plane, so it is not z-fighting the floor
  const mjtNum size[3] = {radius, kHalfHeight, 0.0};
  const mjtNum position[3] = {x, y, kHalfHeight};
  const mjtNum identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  mjv_initGeom(geom, mjGEOM_CYLINDER, size, position, identity, nullptr);
  paint(geom, color);
  return true;
}

bool addSphereGeom(mjvScene* scene, const double position[3], double radius, const MarkerColor& color) {
  mjvGeom* geom = nextGeom(scene);
  if (geom == nullptr) return false;
  const mjtNum size[3] = {radius, 0.0, 0.0};
  const mjtNum at[3] = {position[0], position[1], position[2]};
  const mjtNum identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  mjv_initGeom(geom, mjGEOM_SPHERE, size, at, identity, nullptr);
  paint(geom, color);
  return true;
}

bool addVerticalLineGeom(mjvScene* scene, const double position[3], double groundHeight, double width, const MarkerColor& color) {
  mjvGeom* geom = nextGeom(scene);
  if (geom == nullptr) return false;
  const mjtNum from[3] = {position[0], position[1], position[2]};
  const mjtNum to[3] = {position[0], position[1], groundHeight};
  mjv_connector(geom, mjGEOM_CAPSULE, width, from, to);
  paint(geom, color);
  return true;
}

}  // namespace robot::mujoco_sim_interface
