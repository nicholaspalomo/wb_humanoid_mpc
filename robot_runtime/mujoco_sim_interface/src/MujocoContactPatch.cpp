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

#include "mujoco_sim_interface/MujocoContactPatch.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace robot::mujoco_sim_interface {

namespace {
constexpr double kSlabHalfThickness = 0.002;  // [m] the filled patch is a slab of twice this height on the ground
constexpr double kOutlineRadius = 0.006;      // [m] capsule radius of the outline
constexpr double kArrowWidth = 0.012;         // [m]
constexpr double kMinArrowLength = 0.05;      // [m] arrow length for a patch that does not extend forward
constexpr double kArrowLengthRatio = 0.75;    // arrow length relative to the patch's forward extent

/** Bounds of the corners in the contact frame. */
struct Bounds {
  double xMin{std::numeric_limits<double>::max()}, xMax{std::numeric_limits<double>::lowest()};
  double yMin{std::numeric_limits<double>::max()}, yMax{std::numeric_limits<double>::lowest()};
};

Bounds boundsOf(const ContactPatchCorners& corners) {
  Bounds b;
  for (const auto& corner : corners) {
    b.xMin = std::min(b.xMin, corner[0]);
    b.xMax = std::max(b.xMax, corner[0]);
    b.yMin = std::min(b.yMin, corner[1]);
    b.yMax = std::max(b.yMax, corner[1]);
  }
  return b;
}
}  // namespace

ContactPatchCorners defaultContactPatchCorners() {
  return {{-0.10, -0.05}, {0.10, -0.05}, {0.10, 0.05}, {-0.10, 0.05}};
}

std::vector<std::array<double, 3>> contactPatchWorldCorners(const TargetContactPatch& patch,
                                                            const ContactPatchCorners& corners,
                                                            double heightOffset) {
  const double c = std::cos(patch.yaw), s = std::sin(patch.yaw);
  std::vector<std::array<double, 3>> world;
  world.reserve(corners.size());
  for (const auto& corner : corners) {
    world.push_back({patch.x + c * corner[0] - s * corner[1], patch.y + s * corner[0] + c * corner[1], patch.z + heightOffset});
  }
  return world;
}

int addContactPatchGeoms(mjvScene* scene,
                         const TargetContactPatch& patch,
                         const ContactPatchCorners& corners,
                         const ContactPatchStyle& style) {
  if (scene == nullptr || !patch.valid || corners.size() < 3) return 0;
  const int needed = static_cast<int>(corners.size()) + (style.fill ? 1 : 0) + (style.arrow ? 1 : 0);
  if (scene->ngeom < 0 || scene->ngeom + needed > scene->maxgeom) return 0;

  const float rgba[4] = {style.rgba[0], style.rgba[1], style.rgba[2], style.rgba[3]};
  const double c = std::cos(patch.yaw), s = std::sin(patch.yaw);
  const mjtNum mat[9] = {c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0};  // row-major rotation about z by the yaw
  // mjv_initGeom leaves the object fields alone, so every geom is cleared and marked as a free decoration first.
  const auto nextGeom = [scene]() {
    mjvGeom* geom = &scene->geoms[scene->ngeom++];
    std::memset(geom, 0, sizeof(mjvGeom));
    geom->objtype = mjOBJ_UNKNOWN;
    geom->objid = -1;
    geom->dataid = -1;
    geom->matid = -1;
    geom->segid = -1;
    return geom;
  };
  const auto finish = [&style](mjvGeom* geom) {
    geom->category = mjCAT_DECOR;
    geom->emission = style.emission;
  };
  const Bounds bounds = boundsOf(corners);

  if (style.fill) {
    // A thin box centred on the bounds of the corners, rotated by the yaw: exactly the patch for a rectangle.
    const double cx = 0.5 * (bounds.xMin + bounds.xMax), cy = 0.5 * (bounds.yMin + bounds.yMax);
    const mjtNum size[3] = {0.5 * (bounds.xMax - bounds.xMin), 0.5 * (bounds.yMax - bounds.yMin), kSlabHalfThickness};
    const mjtNum pos[3] = {patch.x + c * cx - s * cy, patch.y + s * cx + c * cy, patch.z + kSlabHalfThickness};
    mjvGeom* slab = nextGeom();
    mjv_initGeom(slab, mjGEOM_BOX, size, pos, mat, rgba);
    finish(slab);
  }

  // The outline and the arrow sit just above the slab so that neither is hidden by it.
  const double lineHeight = 2.0 * kSlabHalfThickness + kOutlineRadius;
  const std::vector<std::array<double, 3>> world = contactPatchWorldCorners(patch, corners, lineHeight);
  for (size_t i = 0; i < world.size(); ++i) {
    const std::array<double, 3>& a = world[i];
    const std::array<double, 3>& b = world[(i + 1) % world.size()];
    const mjtNum from[3] = {a[0], a[1], a[2]};
    const mjtNum to[3] = {b[0], b[1], b[2]};
    mjvGeom* edge = nextGeom();
    mjv_initGeom(edge, mjGEOM_CAPSULE, nullptr, nullptr, nullptr, rgba);
    mjv_connector(edge, mjGEOM_CAPSULE, kOutlineRadius, from, to);
    finish(edge);
  }

  if (style.arrow) {
    const double length = std::max(kMinArrowLength, kArrowLengthRatio * bounds.xMax);
    const mjtNum from[3] = {patch.x, patch.y, patch.z + lineHeight};
    const mjtNum to[3] = {patch.x + length * c, patch.y + length * s, patch.z + lineHeight};
    mjvGeom* arrow = nextGeom();
    mjv_initGeom(arrow, mjGEOM_ARROW, nullptr, nullptr, nullptr, rgba);
    mjv_connector(arrow, mjGEOM_ARROW, kArrowWidth, from, to);
    finish(arrow);
  }
  return needed;
}

}  // namespace robot::mujoco_sim_interface
