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

// The viewer shows the visual meshes only: the collision geoms of the bodies
// that have a visual geom leave the scene, and the floor, a body whose only
// geom collides and the other visualizations' decor geoms stay.

#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/visualization/HideCollisionGeomsVisualization.h"
#include "mujoco_sim_interface/visualization/MujocoVisualization.h"
#include "mujoco_sim_interface/visualization/VisualizationRegistry.h"

namespace robot::mujoco_sim_interface {
namespace {

// A link with a visual mesh stand-in (group 1, no collision) and its collision
// box (group 2), a ball whose one geom collides, and a link whose collision
// sphere is in group 0 next to its visual box, as the G1's and R1's are.
constexpr char kScene[] = R"(
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="link" pos="0 0 1">
      <freejoint/>
      <geom name="link_visual" type="box" size="0.1 0.1 0.1" group="1"
            contype="0" conaffinity="0"/>
      <geom name="link_collision" type="box" size="0.1 0.1 0.1" group="2"/>
    </body>
    <body name="ball" pos="1 0 1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.1"/>
    </body>
    <body name="foot" pos="2 0 1">
      <freejoint/>
      <geom name="foot_visual" type="box" size="0.1 0.1 0.1" group="1"
            contype="0" conaffinity="0"/>
      <geom name="foot_sphere" type="sphere" size="0.02" group="0"/>
    </body>
  </worldbody>
</mujoco>
)";

MjModelPtr compileScene() {
  char error[1000] = "";
  const MjSpecPtr spec(
      mj_parseXMLString(kScene, /*vfs=*/nullptr, error, sizeof(error)));
  EXPECT_NE(spec, nullptr) << "mj_parseXMLString: " << error;
  if (spec == nullptr) return nullptr;
  return MjModelPtr(mj_compile(spec.get(), /*vfs=*/nullptr));
}

int geomId(const mjModel& model, const char* absl_nonnull name) {
  return mj_name2id(&model, mjOBJ_GEOM, name);
}

/** Whether `scene` holds a geom drawn for model geom `geom`. */
bool drawsGeom(const mjvScene& scene, int geom) {
  for (int index = 0; index < scene.ngeom; ++index) {
    if (scene.geoms[index].objtype == mjOBJ_GEOM &&
        scene.geoms[index].objid == geom) {
      return true;
    }
  }
  return false;
}

TEST(HideCollisionGeoms, HidesTheCollisionGeomsOfTheBodiesWithAVisualGeom) {
  const MjModelPtr model = compileScene();
  ASSERT_NE(model, nullptr);
  EXPECT_TRUE(isHiddenCollisionGeom(*model, geomId(*model, "link_collision")));
  EXPECT_TRUE(isHiddenCollisionGeom(*model, geomId(*model, "foot_sphere")))
      << "a collision geom in group 0 is hidden too";
  EXPECT_FALSE(isHiddenCollisionGeom(*model, geomId(*model, "link_visual")));
  EXPECT_FALSE(isHiddenCollisionGeom(*model, geomId(*model, "foot_visual")));
  EXPECT_FALSE(isHiddenCollisionGeom(*model, geomId(*model, "floor")))
      << "the world's geoms always show";
  EXPECT_FALSE(isHiddenCollisionGeom(*model, geomId(*model, "ball_geom")))
      << "a body without a visual geom would disappear";
}

TEST(HideCollisionGeoms, RemovesThemFromTheSceneAndKeepsTheRest) {
  const MjModelPtr model = compileScene();
  ASSERT_NE(model, nullptr);
  MjState state(model.get());
  ASSERT_NE(state.data, nullptr);
  mj_forward(model.get(), state.data);
  mjvOption options;
  mjv_defaultOption(&options);
  mjvCamera camera;
  mjv_defaultCamera(&camera);
  mjvScene scene;
  mjv_defaultScene(&scene);
  mjv_makeScene(model.get(), &scene, /*maxgeom=*/100);
  mjv_updateScene(model.get(), state.data, &options, /*pert=*/nullptr, &camera,
                  mjCAT_ALL, &scene);
  for (int geom = 0; geom < model->ngeom; ++geom) {
    ASSERT_TRUE(drawsGeom(scene, geom)) << "MuJoCo draws geom " << geom;
  }
  // A decor geom of another visualization, which is no geom of the model.
  ASSERT_LT(scene.ngeom, scene.maxgeom);
  mjvGeom& decor = scene.geoms[scene.ngeom++];
  mjv_initGeom(&decor, mjGEOM_SPHERE, /*size=*/nullptr, /*pos=*/nullptr,
               /*mat=*/nullptr, /*rgba=*/nullptr);
  const int geomsBefore = scene.ngeom;

  VisualizationFrame frame;
  frame.model = model.get();
  frame.state = &state;
  frame.scene = &scene;
  HideCollisionGeomsVisualization visualization;
  visualization.addSceneGeoms(frame);

  EXPECT_EQ(scene.ngeom, geomsBefore - 2);
  EXPECT_FALSE(drawsGeom(scene, geomId(*model, "link_collision")));
  EXPECT_FALSE(drawsGeom(scene, geomId(*model, "foot_sphere")));
  for (const char* absl_nonnull kept :
       {"floor", "link_visual", "foot_visual", "ball_geom"}) {
    EXPECT_TRUE(drawsGeom(scene, geomId(*model, kept))) << kept;
  }
  EXPECT_EQ(scene.geoms[scene.ngeom - 1].objtype, mjOBJ_UNKNOWN)
      << "the decor geom stays, at the end";
  mjv_freeScene(&scene);
}

TEST(HideCollisionGeoms, IsOnByDefaultAndTheVKeyTogglesIt) {
  const std::unique_ptr<MujocoVisualization> visualization =
      createVisualization("hide_collision_geoms");
  ASSERT_NE(visualization, nullptr);
  EXPECT_EQ(visualization->hotkey(), 'v');
  bool onByDefault = false;
  for (const std::string& name : defaultVisualizationNames()) {
    onByDefault = onByDefault || name == "hide_collision_geoms";
  }
  EXPECT_TRUE(onByDefault)
      << "a task file without sim_visualizations shows the visual meshes only";
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
