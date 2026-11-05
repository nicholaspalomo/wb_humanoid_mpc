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
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"

#include "mujoco_sim_interface/MujocoUtils.h"

namespace robot::mujoco_sim_interface {

using ::ocs2::kHorizontalPositionDim;
using ::ocs2::kPositionDim;
using ::ocs2::kXAxisIndex;
using ::ocs2::kYAxisIndex;
using ::ocs2::kZAxisIndex;

namespace {
constexpr double kSlabHalfThickness =
    0.002;  // [m] the filled patch is a slab of twice this height on the ground
constexpr double kOutlineRadius = 0.006;  // [m] capsule radius of the outline
constexpr double kArrowWidth = 0.012;     // [m]
constexpr double kMinArrowLength =
    0.05;  // [m] arrow length for a patch that does not extend forward
constexpr double kArrowLengthRatio =
    0.75;  // arrow length relative to the patch's forward extent
/// [m] Half the length (along x) and half the width (along y) of the generic
/// foot outline.
constexpr double kDefaultPatchHalfLength = 0.10;
constexpr double kDefaultPatchHalfWidth = 0.05;
/// The fewest corners that outline a polygon.
constexpr int kMinPatchCorners = 3;

/** Bounds of the corners in the contact frame. */
struct Bounds {
  double minPosCornersRtFootEwrtFootX{std::numeric_limits<double>::max()},
      maxPosCornersRtFootEwrtFootX{std::numeric_limits<double>::lowest()};
  double minPosCornersRtFootEwrtFootY{std::numeric_limits<double>::max()},
      maxPosCornersRtFootEwrtFootY{std::numeric_limits<double>::lowest()};
};

Bounds boundsOf(const ContactPatchCorners& posCornersRtFootEwrtFootXy) {
  Bounds bounds;
  for (const std::array<double, kHorizontalPositionDim>&
           posCornerRtFootEwrtFootXy : posCornersRtFootEwrtFootXy) {
    bounds.minPosCornersRtFootEwrtFootX = std::min(
        bounds.minPosCornersRtFootEwrtFootX, posCornerRtFootEwrtFootXy[0]);
    bounds.maxPosCornersRtFootEwrtFootX = std::max(
        bounds.maxPosCornersRtFootEwrtFootX, posCornerRtFootEwrtFootXy[0]);
    bounds.minPosCornersRtFootEwrtFootY = std::min(
        bounds.minPosCornersRtFootEwrtFootY, posCornerRtFootEwrtFootXy[1]);
    bounds.maxPosCornersRtFootEwrtFootY = std::max(
        bounds.maxPosCornersRtFootEwrtFootY, posCornerRtFootEwrtFootXy[1]);
  }
  return bounds;
}
}  // namespace

ContactPatchCorners defaultContactPatchCorners() {
  return {{-kDefaultPatchHalfLength, -kDefaultPatchHalfWidth},
          {kDefaultPatchHalfLength, -kDefaultPatchHalfWidth},
          {kDefaultPatchHalfLength, kDefaultPatchHalfWidth},
          {-kDefaultPatchHalfLength, kDefaultPatchHalfWidth}};
}

std::vector<std::array<double, kPositionDim>> computePosCornersRtWorldEwrtWorld(
    const TargetContactPatch& patch,
    const ContactPatchCorners& posCornersRtFootEwrtFootXy,
    double offsetPosCornersRtFootEwrtWorldZ) {
  const double c = std::cos(patch.desEulerzyxFootRtWorldYaw);
  const double s = std::sin(patch.desEulerzyxFootRtWorldYaw);
  std::vector<std::array<double, kPositionDim>> posCornersRtWorldEwrtWorld;
  posCornersRtWorldEwrtWorld.reserve(posCornersRtFootEwrtFootXy.size());
  for (const std::array<double, kHorizontalPositionDim>&
           posCornerRtFootEwrtFootXy : posCornersRtFootEwrtFootXy) {
    posCornersRtWorldEwrtWorld.push_back(
        {patch.desPosFootRtWorldEwrtWorldX + c * posCornerRtFootEwrtFootXy[0] -
             s * posCornerRtFootEwrtFootXy[1],
         patch.desPosFootRtWorldEwrtWorldY + s * posCornerRtFootEwrtFootXy[0] +
             c * posCornerRtFootEwrtFootXy[1],
         patch.desPosFootRtWorldEwrtWorldZ + offsetPosCornersRtFootEwrtWorldZ});
  }
  return posCornersRtWorldEwrtWorld;
}

int addContactPatchGeoms(mjvScene* absl_nullable scene,
                         const TargetContactPatch& patch,
                         const ContactPatchCorners& posCornersRtFootEwrtFootXy,
                         const ContactPatchStyle& style) {
  if (scene == nullptr || !patch.valid ||
      posCornersRtFootEwrtFootXy.size() < kMinPatchCorners) {
    return 0;
  }
  const int needed = static_cast<int>(posCornersRtFootEwrtFootXy.size()) +
                     (style.fill ? 1 : 0) + (style.arrow ? 1 : 0);
  if (scene->ngeom < 0 || scene->ngeom + needed > scene->maxgeom) return 0;

  const float rgba[kMjRgbaChannels] = {
      style.rgba[kRedChannel], style.rgba[kGreenChannel],
      style.rgba[kBlueChannel], style.rgba[kAlphaChannel]};
  const double c = std::cos(patch.desEulerzyxFootRtWorldYaw);
  const double s = std::sin(patch.desEulerzyxFootRtWorldYaw);
  const mjtNum desRotmatFootRtWorld[kMjMatrixSize] = {
      c,   -s,  0.0, s,  c,
      0.0, 0.0, 0.0, 1.0};  // row-major rotation about z by the yaw
  // mjv_initGeom leaves the object fields alone, so every geom is cleared and
  // marked as a free decoration first.
  const std::function<mjvGeom* absl_nonnull()> nextGeom = [scene]() {
    mjvGeom* absl_nonnull geom = &scene->geoms[scene->ngeom++];
    std::memset(geom, 0, sizeof(mjvGeom));
    geom->objtype = mjOBJ_UNKNOWN;
    geom->objid = -1;
    geom->dataid = -1;
    geom->matid = -1;
    geom->segid = -1;
    return geom;
  };
  const std::function<void(mjvGeom* absl_nonnull)> finish =
      [&style](mjvGeom* absl_nonnull geom) {
        geom->category = mjCAT_DECOR;
        geom->emission = style.emission;
      };
  const Bounds bounds = boundsOf(posCornersRtFootEwrtFootXy);

  if (style.fill) {
    // A thin box centered on the bounds of the corners, rotated by the yaw:
    // exactly the patch for a rectangle.
    const double posFootprintRtFootEwrtFootX =
        0.5 * (bounds.minPosCornersRtFootEwrtFootX +
               bounds.maxPosCornersRtFootEwrtFootX);
    const double posFootprintRtFootEwrtFootY =
        0.5 * (bounds.minPosCornersRtFootEwrtFootY +
               bounds.maxPosCornersRtFootEwrtFootY);
    const mjtNum size[kPositionDim] = {
        0.5 * (bounds.maxPosCornersRtFootEwrtFootX -
               bounds.minPosCornersRtFootEwrtFootX),
        0.5 * (bounds.maxPosCornersRtFootEwrtFootY -
               bounds.minPosCornersRtFootEwrtFootY),
        kSlabHalfThickness};
    // The center of the slab's box geom, which is not the foot's origin: the
    // bounds' center turned by the yaw, half a thickness up.
    const mjtNum desPosFootprintRtWorldEwrtWorld[kPositionDim] = {
        patch.desPosFootRtWorldEwrtWorldX + c * posFootprintRtFootEwrtFootX -
            s * posFootprintRtFootEwrtFootY,
        patch.desPosFootRtWorldEwrtWorldY + s * posFootprintRtFootEwrtFootX +
            c * posFootprintRtFootEwrtFootY,
        patch.desPosFootRtWorldEwrtWorldZ + kSlabHalfThickness};
    mjvGeom* absl_nonnull slab = nextGeom();
    mjv_initGeom(slab, mjGEOM_BOX, size, desPosFootprintRtWorldEwrtWorld,
                 desRotmatFootRtWorld, rgba);
    finish(slab);
  }

  // The outline and the arrow sit just above the slab so that neither is hidden
  // by it.
  const double offsetPosCornersRtFootEwrtWorldZ =
      2.0 * kSlabHalfThickness + kOutlineRadius;
  const std::vector<std::array<double, kPositionDim>>
      posCornersRtWorldEwrtWorld = computePosCornersRtWorldEwrtWorld(
          patch, posCornersRtFootEwrtFootXy, offsetPosCornersRtFootEwrtWorldZ);
  for (size_t i = 0; i < posCornersRtWorldEwrtWorld.size(); ++i) {
    const std::array<double, kPositionDim>& posCornerRtWorldEwrtWorld =
        posCornersRtWorldEwrtWorld[i];
    const std::array<double, kPositionDim>& nextPosCornerRtWorldEwrtWorld =
        posCornersRtWorldEwrtWorld[(i + 1) % posCornersRtWorldEwrtWorld.size()];
    const mjtNum posStrippointRtWorldEwrtWorld[kPositionDim] = {
        posCornerRtWorldEwrtWorld[kXAxisIndex],
        posCornerRtWorldEwrtWorld[kYAxisIndex],
        posCornerRtWorldEwrtWorld[kZAxisIndex]};
    const mjtNum nextPosStrippointRtWorldEwrtWorld[kPositionDim] = {
        nextPosCornerRtWorldEwrtWorld[kXAxisIndex],
        nextPosCornerRtWorldEwrtWorld[kYAxisIndex],
        nextPosCornerRtWorldEwrtWorld[kZAxisIndex]};
    mjvGeom* absl_nonnull edge = nextGeom();
    mjv_initGeom(edge, mjGEOM_CAPSULE, /*size=*/nullptr, /*pos=*/nullptr,
                 /*mat=*/nullptr, rgba);
    mjv_connector(edge, mjGEOM_CAPSULE, kOutlineRadius,
                  posStrippointRtWorldEwrtWorld,
                  nextPosStrippointRtWorldEwrtWorld);
    finish(edge);
  }

  if (style.arrow) {
    const double length =
        std::max(kMinArrowLength,
                 kArrowLengthRatio * bounds.maxPosCornersRtFootEwrtFootX);
    const mjtNum posArrowRtWorldEwrtWorld[kPositionDim] = {
        patch.desPosFootRtWorldEwrtWorldX, patch.desPosFootRtWorldEwrtWorldY,
        patch.desPosFootRtWorldEwrtWorldZ + offsetPosCornersRtFootEwrtWorldZ};
    const mjtNum posArrowheadRtWorldEwrtWorld[kPositionDim] = {
        patch.desPosFootRtWorldEwrtWorldX + length * c,
        patch.desPosFootRtWorldEwrtWorldY + length * s,
        patch.desPosFootRtWorldEwrtWorldZ + offsetPosCornersRtFootEwrtWorldZ};
    mjvGeom* absl_nonnull arrow = nextGeom();
    mjv_initGeom(arrow, mjGEOM_ARROW, /*size=*/nullptr, /*pos=*/nullptr,
                 /*mat=*/nullptr, rgba);
    mjv_connector(arrow, mjGEOM_ARROW, kArrowWidth, posArrowRtWorldEwrtWorld,
                  posArrowheadRtWorldEwrtWorld);
    finish(arrow);
  }
  return needed;
}

}  // namespace robot::mujoco_sim_interface
