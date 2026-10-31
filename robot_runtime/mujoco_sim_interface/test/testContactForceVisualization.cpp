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

#include <string>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/visualization/ContactForceVisualization.h"
#include "mujoco_sim_interface/visualization/MujocoVisualization.h"

namespace robot::mujoco_sim_interface {
namespace {

// A 10 kg box resting on the floor plane: MuJoCo orders the pair (floor, box),
// so geom[0] of its contacts is the world.
constexpr char kBoxOnAPlane[] = R"(
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="10"/>
    </body>
  </worldbody>
</mujoco>
)";

// A 10 kg sphere resting on a box of the world: MuJoCo orders the pair (sphere,
// box), so geom[0] is the free body.
constexpr char kSphereOnAWorldBox[] = R"(
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="box" size="1 1 0.1" pos="0 0 -0.1"/>
    <body name="ball" pos="0 0 0.1">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="10"/>
    </body>
  </worldbody>
</mujoco>
)";

/** An mjvScene of up to `maxGeoms` geoms, freed with the object. Not copyable.
 */
class OwnedScene {
 public:
  OwnedScene(const mjModel* absl_nonnull model, int maxGeoms) {
    mjv_defaultScene(&scene_);
    mjv_makeScene(model, &scene_, maxGeoms);
  }
  OwnedScene(const OwnedScene&) = delete;
  OwnedScene& operator=(const OwnedScene&) = delete;
  ~OwnedScene() { mjv_freeScene(&scene_); }
  mjvScene& get() { return scene_; }

 private:
  mjvScene scene_ = {};
};

/** Settles the free body of `xml` on its support and expects every contact
 * arrow to point up, the support's push. */
void expectArrowsPointUp(absl::string_view xml, int expectedFirstGeomBody) {
  char error[1000] = "";
  const MjSpecPtr spec(mj_parseXMLString(
      std::string(xml).c_str(), /*vfs=*/nullptr, error, sizeof(error)));
  ASSERT_NE(spec, nullptr) << "mj_parseXMLString: " << error;
  const MjModelPtr model(mj_compile(spec.get(), /*vfs=*/nullptr));
  ASSERT_NE(model, nullptr);
  MjState state(model.get());
  ASSERT_NE(state.data, nullptr);
  for (int step = 0; step < 500; ++step) mj_step(model.get(), state.data);
  ASSERT_GT(state.data->ncon, 0) << "the body does not rest on its support";
  for (int contact = 0; contact < state.data->ncon; ++contact) {
    ASSERT_EQ(model->geom_bodyid[state.data->contact[contact].geom[0]],
              expectedFirstGeomBody)
        << "the scene orders its pair otherwise";
  }

  OwnedScene scene(model.get(), /*maxGeoms=*/64);
  VisualizationFrame frame;
  frame.model = model.get();
  frame.state = &state;
  frame.scene = &scene.get();
  ContactForceVisualization visualization;
  visualization.addSceneGeoms(frame);

  ASSERT_GT(scene.get().ngeom, 0) << "no contact force arrow";
  for (int geom = 0; geom < scene.get().ngeom; ++geom) {
    // mjvGeom::mat is row-major, so its third column is the arrow's axis.
    EXPECT_GT(scene.get().geoms[geom].mat[8], 0.9)
        << "arrow " << geom << " does not point up";
  }
}

// The arrows are the forces on the body that stands on its support, whichever
// geom of the pair MuJoCo lists first. The behavior was right, but its comment
// said the opposite; this keeps both orders of the pair pinned.
TEST(ContactForceVisualization,
     TheArrowOfABodyOnTheFloorPointsUpWhenTheWorldIsTheFirstGeomOfThePair) {
  expectArrowsPointUp(kBoxOnAPlane, /*expectedFirstGeomBody=*/0);
}

TEST(ContactForceVisualization,
     TheArrowOfABodyOnTheFloorPointsUpWhenTheBodyIsTheFirstGeomOfThePair) {
  expectArrowsPointUp(kSphereOnAWorldBox, /*expectedFirstGeomBody=*/1);
}

// The robot (the first body on a free joint, a box resting on the floor), a
// ball pressed into its side along +x, and a second ball resting on the floor 2
// m away. MuJoCo orders a sphere before a box, so geom[0] of the impact is the
// ball.
constexpr char kBallAgainstARobot[] = R"(
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="robot" pos="0 0 0.099">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="10"/>
    </body>
    <body name="ball" pos="-0.19 0 0.1">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="0.45"/>
    </body>
    <body name="floor_ball" pos="2 0 0.099">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="0.45"/>
    </body>
  </worldbody>
</mujoco>
)";

// The arrows are the forces on the robot: the ball's push on its side points
// into the robot (+x), which the marker used to draw as the force on the ball
// (-x), and the ball resting on the floor, which does not touch the robot,
// draws no arrow (it used to draw its weight).
TEST(ContactForceVisualization,
     TheArrowsAreTheForcesOnTheRobotWhicheverGeomComesFirst) {
  char error[1000] = "";
  const MjSpecPtr spec(mj_parseXMLString(kBallAgainstARobot, /*vfs=*/nullptr,
                                         error, sizeof(error)));
  ASSERT_NE(spec, nullptr) << "mj_parseXMLString: " << error;
  const MjModelPtr model(mj_compile(spec.get(), /*vfs=*/nullptr));
  ASSERT_NE(model, nullptr);
  MjState state(model.get());
  ASSERT_NE(state.data, nullptr);
  mj_forward(model.get(), state.data);
  const int robot = mj_name2id(model.get(), mjOBJ_BODY, "robot");
  const int ball = mj_name2id(model.get(), mjOBJ_BODY, "ball");
  bool impactFound = false;
  for (int contact = 0; contact < state.data->ncon; ++contact) {
    const mjContact& pair = state.data->contact[contact];
    if (model->geom_bodyid[pair.geom[0]] == ball &&
        model->geom_bodyid[pair.geom[1]] == robot)
      impactFound = true;
  }
  ASSERT_TRUE(impactFound)
      << "the scene does not order the impact (ball, robot)";

  OwnedScene scene(model.get(), /*maxGeoms=*/64);
  VisualizationFrame frame;
  frame.model = model.get();
  frame.state = &state;
  frame.scene = &scene.get();
  ContactForceVisualization visualization;
  visualization.addSceneGeoms(frame);

  int impactArrows = 0;
  for (int geom = 0; geom < scene.get().ngeom; ++geom) {
    const mjvGeom& arrow = scene.get().geoms[geom];
    // NOLINTBEGIN(frame-naming): MuJoCo's mjvGeom::pos
    EXPECT_LT(arrow.pos[0], 1.0)
        // NOLINTEND(frame-naming)
        << "an arrow at the floor ball, which does not touch the robot";
    // The impact is on the robot's side at its mid-height; the floor's contacts
    // are at its bottom.
    // NOLINTBEGIN(frame-naming): MuJoCo's mjvGeom::pos
    if (arrow.pos[0] < -0.05 && arrow.pos[2] > 0.05) {
      // NOLINTEND(frame-naming)
      ++impactArrows;
      // mjvGeom::mat is row-major, so its third column is the arrow's axis.
      EXPECT_GT(arrow.mat[2], 0.9)
          << "the impact arrow is not the ball's push on the robot, along +x";
    } else {
      EXPECT_GT(arrow.mat[8], 0.9)
          << "the floor's push on the robot does not point up";
    }
  }
  EXPECT_GT(impactArrows, 0) << "no arrow of the ball's impact";
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
