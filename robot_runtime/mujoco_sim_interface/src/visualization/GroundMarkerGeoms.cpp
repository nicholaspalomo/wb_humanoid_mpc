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

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"

#include "mujoco_sim_interface/MujocoUtils.h"

namespace robot::mujoco_sim_interface {

using ::ocs2::kPositionDim;
using ::ocs2::kXAxisIndex;
using ::ocs2::kYAxisIndex;
using ::ocs2::kZAxisIndex;

namespace {

constexpr float kMarkerEmission = 0.6f;

mjvGeom* absl_nullable nextGeom(mjvScene* absl_nullable scene) {
  if (scene == nullptr || scene->ngeom >= scene->maxgeom) return nullptr;
  mjvGeom* absl_nonnull geom = &scene->geoms[scene->ngeom++];
  std::memset(geom, 0, sizeof(mjvGeom));
  return geom;
}

void paint(mjvGeom* absl_nonnull geom, const MarkerColor& color) {
  geom->rgba[kRedChannel] = color.r;
  geom->rgba[kGreenChannel] = color.g;
  geom->rgba[kBlueChannel] = color.b;
  geom->rgba[kAlphaChannel] = color.a;
  geom->category = mjCAT_DECOR;
  geom->emission = kMarkerEmission;
}

}  // namespace

bool addGroundDiscGeom(mjvScene* absl_nullable scene,
                       double x,
                       double y,
                       double radius,
                       const MarkerColor& color) {
  mjvGeom* absl_nullable geom = nextGeom(scene);
  if (geom == nullptr) return false;
  constexpr double kHalfHeight =
      0.002;  // a thin cylinder just above the ground plane, so it is not
              // z-fighting the floor
  const mjtNum size[kMjGeomSizeParameters] = {radius, kHalfHeight, 0.0};
  const mjtNum posLocalRtWorldEwrtWorld[kPositionDim] = {x, y, kHalfHeight};
  const mjtNum identity[kMjMatrixSize] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  mjv_initGeom(geom, mjGEOM_CYLINDER, size, posLocalRtWorldEwrtWorld, identity,
               /*rgba=*/nullptr);
  paint(geom, color);
  return true;
}

bool addSphereGeom(
    mjvScene* absl_nullable scene,
    const double posLocalRtWorldEwrtWorld[absl_nonnull kPositionDim],
    double radius,
    const MarkerColor& color) {
  mjvGeom* absl_nullable geom = nextGeom(scene);
  if (geom == nullptr) return false;
  const mjtNum size[kMjGeomSizeParameters] = {radius, 0.0, 0.0};
  const mjtNum at[kPositionDim] = {posLocalRtWorldEwrtWorld[kXAxisIndex],
                                   posLocalRtWorldEwrtWorld[kYAxisIndex],
                                   posLocalRtWorldEwrtWorld[kZAxisIndex]};
  const mjtNum identity[kMjMatrixSize] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  mjv_initGeom(geom, mjGEOM_SPHERE, size, at, identity, /*rgba=*/nullptr);
  paint(geom, color);
  return true;
}

bool addVerticalLineGeom(
    mjvScene* absl_nullable scene,
    const double posLocalRtWorldEwrtWorld[absl_nonnull kPositionDim],
    double groundHeight,
    double width,
    const MarkerColor& color) {
  mjvGeom* absl_nullable geom = nextGeom(scene);
  if (geom == nullptr) return false;
  const mjtNum from[kPositionDim] = {posLocalRtWorldEwrtWorld[kXAxisIndex],
                                     posLocalRtWorldEwrtWorld[kYAxisIndex],
                                     posLocalRtWorldEwrtWorld[kZAxisIndex]};
  const mjtNum to[kPositionDim] = {posLocalRtWorldEwrtWorld[0],
                                   posLocalRtWorldEwrtWorld[1], groundHeight};
  mjv_connector(geom, mjGEOM_CAPSULE, width, from, to);
  paint(geom, color);
  return true;
}

}  // namespace robot::mujoco_sim_interface
