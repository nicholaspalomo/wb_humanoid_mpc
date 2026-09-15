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

#include <gtest/gtest.h>
#include <mujoco/mujoco.h>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

#include "mujoco_sim_interface/MujocoContactPatch.h"

using namespace robot::mujoco_sim_interface;

namespace {

constexpr double kTol = 1e-6;
constexpr const char* kScene = R"(
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
  </worldbody>
</mujoco>
)";

/** An abstract scene with room for `maxgeom` geoms. No OpenGL context is needed to fill a scene. */
struct SceneFixture {
  explicit SceneFixture(int maxgeom) {
    const std::string path = testing::TempDir() + "/patch_scene.xml";
    std::ofstream(path) << kScene;
    char error[1000] = "";
    model = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
    if (model == nullptr) throw std::runtime_error(std::string("mj_loadXML: ") + error);
    mjv_defaultScene(&scene);
    mjv_makeScene(model, &scene, maxgeom);
  }
  ~SceneFixture() {
    mjv_freeScene(&scene);
    mj_deleteModel(model);
  }
  mjModel* model{nullptr};
  mjvScene scene;
};

/** A patch at (1, 2, 0.1) turned by 90 degrees: the contact frame's x axis points along world y. */
TargetContactPatch makePatch() {
  TargetContactPatch patch;
  patch.valid = true;
  patch.kind = TargetContactPatch::Kind::SWING_IN_FLIGHT;
  patch.x = 1.0;
  patch.y = 2.0;
  patch.z = 0.1;
  patch.yaw = M_PI / 2.0;
  return patch;
}

// The DRC Atlas sole: an asymmetric order is used on purpose so that the outline follows the given order.
const ContactPatchCorners kRectangle = {{-0.12, -0.05}, {0.12, -0.05}, {0.12, 0.05}, {-0.12, 0.05}};

}  // namespace

TEST(MujocoContactPatch, WorldCornersFollowTheYawAndTheOrigin) {
  const auto world = contactPatchWorldCorners(makePatch(), kRectangle, 0.01);
  ASSERT_EQ(world.size(), kRectangle.size());
  for (size_t i = 0; i < world.size(); ++i) {
    // A rotation by +90 degrees maps a local (x, y) to (-y, x).
    EXPECT_NEAR(world[i][0], 1.0 - kRectangle[i][1], kTol) << "corner " << i;
    EXPECT_NEAR(world[i][1], 2.0 + kRectangle[i][0], kTol) << "corner " << i;
    EXPECT_NEAR(world[i][2], 0.11, kTol) << "corner " << i;
  }
  // No yaw, no offset: a pure translation.
  TargetContactPatch flat = makePatch();
  flat.yaw = 0.0;
  const auto translated = contactPatchWorldCorners(flat, kRectangle);
  EXPECT_NEAR(translated[1][0], 1.12, kTol);
  EXPECT_NEAR(translated[1][1], 1.95, kTol);
  EXPECT_NEAR(translated[1][2], 0.1, kTol);
  EXPECT_TRUE(contactPatchWorldCorners(flat, {}).empty());
}

TEST(MujocoContactPatch, DrawsSlabOutlineAndArrowAsDecorGeoms) {
  SceneFixture fixture(50);
  const TargetContactPatch patch = makePatch();
  ContactPatchStyle style;
  style.rgba = {0.1f, 0.2f, 0.3f, 0.4f};
  style.emission = 0.7f;

  ASSERT_EQ(addContactPatchGeoms(&fixture.scene, patch, kRectangle, style), 6);
  ASSERT_EQ(fixture.scene.ngeom, 6);

  // The slab: a thin box centred on the patch, its x axis (first column) turned to world y.
  const mjvGeom& slab = fixture.scene.geoms[0];
  EXPECT_EQ(slab.type, mjGEOM_BOX);
  EXPECT_NEAR(slab.size[0], 0.12, kTol);
  EXPECT_NEAR(slab.size[1], 0.05, kTol);
  EXPECT_GT(slab.size[2], 0.0);
  EXPECT_LT(slab.size[2], 0.01);
  EXPECT_NEAR(slab.pos[0], 1.0, kTol);
  EXPECT_NEAR(slab.pos[1], 2.0, kTol);
  EXPECT_NEAR(slab.pos[2], 0.1 + slab.size[2], kTol) << "the slab rests on the contact surface";
  EXPECT_NEAR(slab.mat[0], 0.0, kTol);
  EXPECT_NEAR(slab.mat[3], 1.0, kTol);
  EXPECT_NEAR(slab.mat[1], -1.0, kTol);
  EXPECT_NEAR(slab.mat[4], 0.0, kTol);
  EXPECT_NEAR(slab.mat[8], 1.0, kTol);

  // The outline: one capsule per edge, centred on the edge midpoint, above the slab.
  const auto world = contactPatchWorldCorners(patch, kRectangle);
  for (size_t i = 0; i < kRectangle.size(); ++i) {
    const mjvGeom& edge = fixture.scene.geoms[1 + i];
    EXPECT_EQ(edge.type, mjGEOM_CAPSULE) << "edge " << i;
    const auto& a = world[i];
    const auto& b = world[(i + 1) % world.size()];
    EXPECT_NEAR(edge.pos[0], 0.5 * (a[0] + b[0]), kTol) << "edge " << i;
    EXPECT_NEAR(edge.pos[1], 0.5 * (a[1] + b[1]), kTol) << "edge " << i;
    EXPECT_GT(edge.pos[2], 2.0 * slab.size[2] + 0.1) << "edge " << i;
    const double length = std::hypot(b[0] - a[0], b[1] - a[1]);
    EXPECT_NEAR(edge.size[2], 0.5 * length, kTol) << "edge " << i << " spans its two corners";
  }

  // The arrow starts at the origin and points along the patch's x axis (world y here); its z axis is the direction.
  const mjvGeom& arrow = fixture.scene.geoms[5];
  EXPECT_EQ(arrow.type, mjGEOM_ARROW);
  EXPECT_NEAR(arrow.mat[2], 0.0, kTol);
  EXPECT_NEAR(arrow.mat[5], 1.0, kTol);
  EXPECT_NEAR(arrow.mat[8], 0.0, kTol);

  for (int i = 0; i < fixture.scene.ngeom; ++i) {
    const mjvGeom& geom = fixture.scene.geoms[i];
    EXPECT_EQ(geom.category, mjCAT_DECOR) << "geom " << i;
    EXPECT_EQ(geom.objid, -1) << "geom " << i << " is not tied to a model object";
    EXPECT_NEAR(geom.rgba[0], 0.1f, kTol) << "geom " << i;
    EXPECT_NEAR(geom.rgba[3], 0.4f, kTol) << "geom " << i;
    EXPECT_NEAR(geom.emission, 0.7f, kTol) << "geom " << i;
  }
}

TEST(MujocoContactPatch, StyleSelectsSlabAndArrow) {
  SceneFixture fixture(50);
  ContactPatchStyle outlineOnly;
  outlineOnly.fill = false;
  outlineOnly.arrow = false;
  EXPECT_EQ(addContactPatchGeoms(&fixture.scene, makePatch(), kRectangle, outlineOnly), 4);
  EXPECT_EQ(fixture.scene.ngeom, 4);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(fixture.scene.geoms[i].type, mjGEOM_CAPSULE);

  // A triangle is a polygon too; its slab covers the bounds.
  const ContactPatchCorners triangle = {{0.0, 0.0}, {0.2, 0.0}, {0.0, 0.1}};
  ContactPatchStyle fillOnly;
  fillOnly.arrow = false;
  EXPECT_EQ(addContactPatchGeoms(&fixture.scene, makePatch(), triangle, fillOnly), 4);
  const mjvGeom& slab = fixture.scene.geoms[4];
  EXPECT_EQ(slab.type, mjGEOM_BOX);
  EXPECT_NEAR(slab.size[0], 0.1, kTol);
  EXPECT_NEAR(slab.size[1], 0.05, kTol);
}

TEST(MujocoContactPatch, NothingIsDrawnWithoutAValidPatchOrRoom) {
  SceneFixture fixture(5);  // room for five geoms, a full patch needs six
  const ContactPatchStyle style;
  TargetContactPatch invalid = makePatch();
  invalid.valid = false;
  EXPECT_EQ(addContactPatchGeoms(&fixture.scene, invalid, kRectangle, style), 0);
  EXPECT_EQ(addContactPatchGeoms(&fixture.scene, makePatch(), {{0.0, 0.0}, {0.1, 0.0}}, style), 0) << "two corners are no polygon";
  EXPECT_EQ(addContactPatchGeoms(&fixture.scene, makePatch(), kRectangle, style), 0) << "no room for the whole patch";
  EXPECT_EQ(fixture.scene.ngeom, 0) << "a patch is drawn whole or not at all";
  EXPECT_EQ(addContactPatchGeoms(nullptr, makePatch(), kRectangle, style), 0);

  ContactPatchStyle outlineOnly;
  outlineOnly.fill = false;
  outlineOnly.arrow = false;
  EXPECT_EQ(addContactPatchGeoms(&fixture.scene, makePatch(), kRectangle, outlineOnly), 4);
  EXPECT_EQ(addContactPatchGeoms(&fixture.scene, makePatch(), kRectangle, outlineOnly), 0) << "one geom of room left";
  EXPECT_EQ(fixture.scene.ngeom, 4);
  EXPECT_EQ(defaultContactPatchCorners().size(), 4u);
}
