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

#include <gtest/gtest.h>

#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "mujoco_sim_interface/MujocoUtils.h"

using namespace robot::mujoco_sim_interface;

namespace {

// A floating "robot" (pelvis with two hinged feet, the left foot with a hinged toe) over a plane, plus a crate that is
// not part of the robot. A strut on the left foot overlaps the right sole, so the robot is always in self-contact
// between two sibling bodies (which MuJoCo does not filter out). Heights are chosen so that only the left toe
// penetrates the floor at the initial pose:
//   left sole bottom  z = 0.5 - 0.432 - 0.05 = +0.018 (in the air)
//   left toe bottom   z = 0.5 - 0.432 - 0.02 - 0.05 = -0.002 (touching)
//   right sole bottom z = 0.5 - 0.200 - 0.05 = +0.250 (in the air)
constexpr const char* kScene = R"(
<mujoco>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="pelvis" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="10"/>
      <body name="l_foot" pos="0 0.2 -0.432">
        <joint name="l_ankle" type="hinge" axis="1 0 0"/>
        <geom name="l_sole" type="box" size="0.1 0.05 0.05" mass="1"/>
        <!-- A strut that overlaps the right sole: a permanent self-contact between two sibling bodies. -->
        <geom name="l_strut" type="box" pos="0 -0.4 0.232" size="0.02 0.02 0.02" mass="0.01"/>
        <body name="l_toe" pos="0.14 0 -0.02">
          <joint name="l_toe_joint" type="hinge" axis="0 1 0"/>
          <geom name="l_toe_geom" type="box" size="0.03 0.05 0.05" mass="0.1"/>
        </body>
      </body>
      <body name="r_foot" pos="0 -0.2 -0.2">
        <joint name="r_ankle" type="hinge" axis="1 0 0"/>
        <geom name="r_sole" type="box" size="0.1 0.05 0.05" mass="1"/>
      </body>
    </body>
    <body name="crate" pos="1 0 0.048">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

constexpr const char* kUrdf = R"(
<robot name="test_robot">
  <link name="pelvis"/>
  <link name="l_foot"/>
  <link name="foot_l_contact"/>
  <link name="l_toe"/>
  <link name="r_foot"/>
  <link name="foot_r_contact"/>
  <link name="unmodelled_link"/>
  <joint name="l_ankle" type="revolute">
    <parent link="pelvis"/><child link="l_foot"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="foot_l_contact_joint" type="fixed"><parent link="l_foot"/><child link="foot_l_contact"/></joint>
  <joint name="l_toe_joint" type="revolute">
    <parent link="l_foot"/><child link="l_toe"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="r_ankle" type="revolute">
    <parent link="pelvis"/><child link="r_foot"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="foot_r_contact_joint" type="fixed"><parent link="r_foot"/><child link="foot_r_contact"/></joint>
  <joint name="unmodelled_joint" type="revolute">
    <parent link="pelvis"/><child link="unmodelled_link"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
)";

std::string writeTempFile(const std::string& name, const char* content) {
  const std::string path = testing::TempDir() + "/" + name;
  std::ofstream out(path);
  out << content;
  return path;
}

struct Scene {
  Scene() {
    char error[1000] = "";
    model = mj_loadXML(writeTempFile("contact_scene.xml", kScene).c_str(), nullptr, error, sizeof(error));
    if (model == nullptr) throw std::runtime_error(std::string("mj_loadXML: ") + error);
    data = mj_makeData(model);
    mj_forward(model, data);
  }
  ~Scene() {
    mj_deleteData(data);
    mj_deleteModel(model);
  }
  int body(const char* name) const { return mj_name2id(model, mjOBJ_BODY, name); }
  mjModel* model{nullptr};
  mjData* data{nullptr};
};

}  // namespace

TEST(MujocoContactUtils, GroundTruthMaskFollowsTheContactForcesOfTheSubtree) {
  Scene scene;
  const std::vector<int> feet = {scene.body("l_foot"), scene.body("r_foot")};
  ASSERT_GE(feet[0], 0);
  ASSERT_GE(feet[1], 0);

  // Only the left toe touches the floor: it belongs to the left foot's subtree, so the left foot counts as touching.
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, feet, 1.0), 0b01u);
  // The threshold is a normal-force threshold.
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, feet, 1e9), 0u);

  // Bring the right sole down to the same 2 mm penetration as the left toe (the body offset is changed in the model).
  // Both contacts then ask for the same separation and share the load; a much deeper contact on one side would drive
  // the common upward acceleration alone and leave the shallower one force-free, which is physics, not detection.
  scene.model->body_pos[3 * feet[1] + 2] = -0.452;  // right sole bottom at z = 0.5 - 0.452 - 0.05 = -0.002
  mj_forward(scene.model, scene.data);
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, feet, 1.0), 0b11u);

  // An unresolved contact point (-1) never reports contact and does not disturb the others.
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, {-1, feet[1]}, 1.0), 0b10u);
  // The crate rests on the floor too, but it is nobody's contact point.
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, {scene.body("pelvis")}, 1.0), 0b1u) << "the pelvis subtree contains both feet";
}

TEST(MujocoContactUtils, ContactWithAForeignBodyCountsAndSelfContactDoesNot) {
  Scene scene;
  const std::vector<int> feet = {scene.body("l_foot"), scene.body("r_foot")};

  // The strut on the left foot presses into the right sole: a self-contact with real force, between the two feet.
  bool selfContactSeen = false;
  for (int c = 0; c < scene.data->ncon; ++c) {
    const int b1 = scene.model->geom_bodyid[scene.data->contact[c].geom[0]];
    const int b2 = scene.model->geom_bodyid[scene.data->contact[c].geom[1]];
    mjtNum force[6];
    mj_contactForce(scene.model, scene.data, c, force);
    selfContactSeen = selfContactSeen || (((b1 == feet[0] && b2 == feet[1]) || (b1 == feet[1] && b2 == feet[0])) && force[0] > 1.0);
  }
  ASSERT_TRUE(selfContactSeen) << "the scene must contain a self-contact between the feet for this test to mean anything";
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, feet, 1.0), 0b01u) << "self-contact is not ground contact";

  // Slide the crate under the right foot so that it penetrates the sole from below by 2 mm: a foreign free body counts
  // as ground truth just like the floor.
  const int crate = scene.body("crate");
  const int crateQpos = scene.model->jnt_qposadr[scene.model->body_jntadr[crate]];
  scene.data->qpos[crateQpos + 0] = 0.0;
  scene.data->qpos[crateQpos + 1] = -0.2;
  scene.data->qpos[crateQpos + 2] = 0.25 - 0.05 + 0.002;
  mj_forward(scene.model, scene.data);
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, feet, 1.0), 0b11u);
}

TEST(MujocoContactUtils, ResolvesContactFramesThroughFixedJointsOnly) {
  Scene scene;
  const std::string urdf = writeTempFile("contact_robot.urdf", kUrdf);
  std::vector<std::string> errors;
  const std::vector<int> ids = resolveContactBodies(
      scene.model, urdf, {"foot_l_contact", "foot_r_contact", "l_foot", "unmodelled_link", "no_such_frame"}, {}, &errors);
  ASSERT_EQ(ids.size(), 5u);
  EXPECT_EQ(ids[0], scene.body("l_foot")) << "a fixed-joint child frame resolves to its parent body";
  EXPECT_EQ(ids[1], scene.body("r_foot"));
  EXPECT_EQ(ids[2], scene.body("l_foot")) << "a body name resolves directly";
  EXPECT_EQ(ids[3], -1) << "a link on a movable joint that is not a MuJoCo body is an error, not a guess";
  EXPECT_EQ(ids[4], -1);
  ASSERT_EQ(errors.size(), 2u);
  EXPECT_NE(errors[0].find("unmodelled_link"), std::string::npos);
  EXPECT_NE(errors[0].find("unmodelled_joint"), std::string::npos);
  EXPECT_NE(errors[1].find("no_such_frame"), std::string::npos);

  // Without a readable URDF only direct body names resolve.
  errors.clear();
  const std::vector<int> noUrdf = resolveContactBodies(scene.model, "/nonexistent.urdf", {"l_foot", "foot_l_contact"}, {}, &errors);
  EXPECT_EQ(noUrdf[0], scene.body("l_foot"));
  EXPECT_EQ(noUrdf[1], -1);
  ASSERT_EQ(errors.size(), 1u);
  EXPECT_NE(errors[0].find("could not be parsed"), std::string::npos);
}

TEST(MujocoContactUtils, ParentJointNamesResolveFramesThatExistNowhere) {
  // The controller adds its contact frames to its own kinematic model, on a joint; they exist in neither the URDF nor
  // the scene. The body driven by that joint is the contact body.
  Scene scene;
  std::vector<std::string> errors;
  const std::vector<int> ids =
      resolveContactBodies(scene.model, "/nonexistent.urdf", {"virtual_l_contact", "virtual_r_contact", "l_toe_contact"},
                           {"l_ankle", "r_ankle", "l_toe_joint"}, &errors);
  EXPECT_EQ(ids[0], scene.body("l_foot"));
  EXPECT_EQ(ids[1], scene.body("r_foot"));
  EXPECT_EQ(ids[2], scene.body("l_toe"));
  EXPECT_TRUE(errors.empty());

  // A joint name the scene does not know falls back to the frame name (and reports it); shorter or empty entries are
  // simply not used.
  const std::string urdf = writeTempFile("contact_robot.urdf", kUrdf);
  errors.clear();
  const std::vector<int> fallback = resolveContactBodies(scene.model, urdf, {"foot_l_contact", "r_foot"}, {"no_such_joint", ""}, &errors);
  EXPECT_EQ(fallback[0], scene.body("l_foot")) << "resolved through the URDF after the joint lookup failed";
  EXPECT_EQ(fallback[1], scene.body("r_foot"));
  ASSERT_EQ(errors.size(), 1u);
  EXPECT_NE(errors[0].find("no_such_joint"), std::string::npos);
}

TEST(MujocoContactUtils, SubtreeMembership) {
  Scene scene;
  EXPECT_TRUE(isInBodySubtree(scene.model, scene.body("l_toe"), scene.body("l_foot")));
  EXPECT_TRUE(isInBodySubtree(scene.model, scene.body("l_foot"), scene.body("l_foot")));
  EXPECT_TRUE(isInBodySubtree(scene.model, scene.body("l_toe"), scene.body("pelvis")));
  EXPECT_FALSE(isInBodySubtree(scene.model, scene.body("l_foot"), scene.body("l_toe")));
  EXPECT_FALSE(isInBodySubtree(scene.model, scene.body("r_foot"), scene.body("l_foot")));
  EXPECT_FALSE(isInBodySubtree(scene.model, scene.body("crate"), scene.body("pelvis")));
  EXPECT_TRUE(isInBodySubtree(scene.model, scene.body("crate"), 0)) << "everything hangs under the world body";
  EXPECT_FALSE(isInBodySubtree(scene.model, -1, scene.body("pelvis")));
}

TEST(MujocoContactUtils, TimelineKeepsAWindowAndClearsOnReset) {
  ContactTimeline timeline(2.0);
  EXPECT_DOUBLE_EQ(timeline.window(), 2.0);
  for (int i = 0; i <= 100; ++i) {
    ContactTimelineSample sample;
    sample.time = 0.1 * i;
    sample.actual = (i % 2 == 0) ? 0b01u : 0b10u;
    sample.target = 0b01u;
    sample.targetKnown = i > 10;
    timeline.append(sample);
  }
  ASSERT_FALSE(timeline.samples().empty());
  EXPECT_DOUBLE_EQ(timeline.samples().back().time, 10.0);
  EXPECT_GE(timeline.samples().front().time, 8.0 - 1e-12) << "samples older than the window are dropped";
  EXPECT_LE(timeline.samples().front().time, 8.1 + 1e-12);
  EXPECT_TRUE(timeline.samples().back().targetKnown);

  // The simulation was reset: time runs backwards, the history is cleared.
  ContactTimelineSample afterReset;
  afterReset.time = 1.0;
  timeline.append(afterReset);
  ASSERT_EQ(timeline.samples().size(), 1u);
  EXPECT_DOUBLE_EQ(timeline.samples().front().time, 1.0);

  EXPECT_DOUBLE_EQ(ContactTimeline(-3.0).window(), 5.0) << "a non-positive window falls back to the default";
}

/*============================================ centroidal markers ==========================================*/

TEST(MujocoContactUtils, CentroidalStateIsTheRootSubtreesCentreOfMassAndVelocity) {
  Scene scene;
  const int pelvis = scene.body("pelvis");
  ASSERT_GE(pelvis, 0);
  const RobotCentroidalState resting = robotCentroidalState(scene.model, scene.data);
  ASSERT_TRUE(resting.valid);
  EXPECT_EQ(resting.rootBodyId, pelvis) << "the first body on a free joint, not the crate";
  EXPECT_NEAR(resting.mass, scene.model->body_subtreemass[pelvis], 1e-12);
  EXPECT_NEAR(resting.mass, 10.0 + 1.0 + 0.01 + 0.1 + 1.0, 1e-9);
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_NEAR(resting.com[axis], scene.data->subtree_com[3 * pelvis + axis], 1e-12);
    EXPECT_NEAR(resting.comVelocity[axis], 0.0, 1e-12);
  }
  // The whole robot translating at 1 m/s along x: so does its centre of mass.
  scene.data->qvel[0] = 1.0;
  mj_forward(scene.model, scene.data);
  const RobotCentroidalState moving = robotCentroidalState(scene.model, scene.data);
  ASSERT_TRUE(moving.valid);
  EXPECT_NEAR(moving.comVelocity[0], 1.0, 1e-9);
  EXPECT_NEAR(moving.comVelocity[1], 0.0, 1e-9);
  EXPECT_NEAR(moving.comVelocity[2], 0.0, 1e-9);
  EXPECT_FALSE(robotCentroidalState(nullptr, scene.data).valid);
}

TEST(MujocoContactUtils, GroundReactionZmpSitsUnderTheOnlyGroundContact) {
  Scene scene;
  const int pelvis = scene.body("pelvis");
  // Only the left toe touches the floor; the strut-sole self-contact carries force but is not a ground reaction.
  const GroundReaction reaction = groundReaction(scene.model, scene.data, pelvis, 0.0);
  ASSERT_TRUE(reaction.valid);
  EXPECT_GT(reaction.force[2], 0.0) << "the floor pushes the robot up";
  // The centre of pressure lies in the convex hull of the ground contact points (the corners of the toe box).
  double lo[2] = {1e9, 1e9};
  double hi[2] = {-1e9, -1e9};
  int groundContacts = 0;
  for (int c = 0; c < scene.data->ncon; ++c) {
    const mjContact& contact = scene.data->contact[c];
    const int b1 = scene.model->geom_bodyid[contact.geom[0]];
    const int b2 = scene.model->geom_bodyid[contact.geom[1]];
    if (b1 != 0 && b2 != 0) continue;
    ++groundContacts;
    for (int axis = 0; axis < 2; ++axis) {
      lo[axis] = std::min(lo[axis], contact.pos[axis]);
      hi[axis] = std::max(hi[axis], contact.pos[axis]);
    }
  }
  ASSERT_GT(groundContacts, 0);
  for (int axis = 0; axis < 2; ++axis) {
    EXPECT_GE(reaction.zmp[axis], lo[axis] - 1e-6);
    EXPECT_LE(reaction.zmp[axis], hi[axis] + 1e-6);
  }
  EXPECT_NEAR(reaction.zmp[0], 0.14, 0.05) << "under the toe";
  EXPECT_NEAR(reaction.zmp[1], 0.2, 0.06) << "under the toe";
  // A threshold above the reaction hides the marker (the robot on the gantry).
  EXPECT_FALSE(groundReaction(scene.model, scene.data, pelvis, reaction.force[2] + 1.0).valid);
  EXPECT_FALSE(groundReaction(scene.model, scene.data, -1, 0.0).valid);
}

TEST(MujocoContactUtils, DivergentComponentOfMotionIsTheComPlusVelocityOverOmega) {
  const double com[3] = {1.0, 2.0, 0.85};
  const double velocity[3] = {0.34, -0.17, 0.5};
  double dcm[2];
  divergentComponentOfMotion(com, velocity, com[2], 9.81, dcm);
  const double omega = std::sqrt(9.81 / 0.85);
  EXPECT_NEAR(dcm[0], 1.0 + 0.34 / omega, 1e-12);
  EXPECT_NEAR(dcm[1], 2.0 - 0.17 / omega, 1e-12);
  // The height is clamped so that a CoM on the ground does not blow the DCM up.
  divergentComponentOfMotion(com, velocity, 0.0, 9.81, dcm);
  EXPECT_NEAR(dcm[0], 1.0 + 0.34 / std::sqrt(9.81 / 0.05), 1e-12);
}
