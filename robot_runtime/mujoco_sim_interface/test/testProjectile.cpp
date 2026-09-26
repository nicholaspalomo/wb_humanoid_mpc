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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include <mujoco/mujoco.h>

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/Projectile.h"

namespace robot::mujoco_sim_interface {
namespace {

/**
 * A scene shaped like the robot ones: a floor, a free-jointed "robot" with a foot, and nothing else. Small enough to
 * reason about, and enough to catch the two things that break when a projectile is appended - the robot stopping
 * being the first free-joint body, and a ball being mistaken for the ground.
 */
constexpr const char* kScene = R"(
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 1">
      <freejoint name="base_free"/>
      <geom name="torso" type="box" size="0.1 0.1 0.2"/>
      <body name="foot" pos="0 0 -0.3">
        <joint name="ankle" type="hinge" axis="0 1 0"/>
        <geom name="sole" type="box" size="0.1 0.05 0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

std::string writeTempFile(const std::string& name, const std::string& content) {
  const std::string path = testing::TempDir() + "/" + name;
  std::ofstream out(path);
  out << content;
  return path;
}

/** RAII around a compiled model, with or without a projectile appended. */
struct Scene {
  explicit Scene(const std::string& path, const Projectile* projectile = nullptr) {
    char error[1000] = {0};
    if (projectile == nullptr) {
      model = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
    } else {
      mjSpec* spec = mj_parseXML(path.c_str(), nullptr, error, sizeof(error));
      if (spec != nullptr) {
        addStatus = addProjectileToSpec(spec, *projectile, "sim_projectile");
        if (addStatus.ok()) model = mj_compile(spec, nullptr);
        mj_deleteSpec(spec);
      }
    }
    if (model != nullptr) data = mj_makeData(model);
  }
  ~Scene() {
    if (data != nullptr) mj_deleteData(data);
    if (model != nullptr) mj_deleteModel(model);
  }
  int body(const char* name) const { return mj_name2id(model, mjOBJ_BODY, name); }

  mjModel* model{nullptr};
  mjData* data{nullptr};
  absl::Status addStatus;
};

Projectile dodgeball() {
  return *projectileFromName("dodgeball");
}

/**
 * Puts the ball against the top of the foot's sole and still travelling downwards, which is what a ball in the middle
 * of an impact looks like and what makes the normal force large rather than marginal, then runs the solver.
 */
void restBallOnTheFoot(const Scene& scene, const Projectile& ball) {
  const int ballJoint = scene.model->body_jntadr[scene.body("sim_projectile")];
  const int ballQpos = scene.model->jnt_qposadr[ballJoint];
  const int ballDof = scene.model->jnt_dofadr[ballJoint];
  const int baseQpos = scene.model->jnt_qposadr[scene.model->body_jntadr[scene.body("base")]];
  scene.data->qpos[baseQpos + 2] = 1.0;
  // The sole's top face sits at base z - 0.3 + 0.02.
  scene.data->qpos[ballQpos + 2] = 1.0 - 0.3 + 0.02 + ball.radius - 0.002;
  scene.data->qpos[ballQpos + 3] = 1.0;
  scene.data->qvel[ballDof + 2] = -12.0;
  mj_forward(scene.model, scene.data);
}

}  // namespace

/*========================================== the registry ==============================================*/

TEST(ProjectileRegistry, DodgeballIsARegulationBall) {
  const absl::StatusOr<Projectile> projectile = projectileFromName("dodgeball");
  ASSERT_TRUE(projectile.ok()) << projectile.status().message();
  // A regulation dodgeball is 8.5 inches across.
  EXPECT_NEAR(2.0 * projectile->radius, 0.216, 1e-3);
  EXPECT_GT(projectile->mass, 0.2);
  EXPECT_LT(projectile->mass, 1.0) << "a dodgeball, not a medicine ball";
  EXPECT_GT(projectile->restitution, 0.0);
  EXPECT_LT(projectile->restitution, 1.0) << "a restitution of 1 is a contact that never settles";
}

TEST(ProjectileRegistry, AnUnknownNameListsTheValidOnes) {
  const absl::StatusOr<Projectile> projectile = projectileFromName("cannonball");
  ASSERT_FALSE(projectile.ok());
  EXPECT_EQ(projectile.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(std::string(projectile.status().message()).find("dodgeball"), std::string::npos) << projectile.status().message();
}

TEST(ProjectileRegistry, EveryAdvertisedNameResolves) {
  for (const std::string& name : availableProjectiles()) {
    EXPECT_TRUE(projectileFromName(name).ok()) << name;
  }
}

/*======================================= restitution -> solref =========================================*/

TEST(ContactDampRatio, InvertsTheLogarithmicDecrement) {
  // The relation this has to satisfy is e = exp(-pi zeta / sqrt(1 - zeta^2)); check the round trip rather than the
  // formula, so that a rearrangement that is algebraically wrong cannot pass.
  for (const double restitution : {0.1, 0.3, 0.5, 0.8, 0.95}) {
    const double zeta = contactDampRatioForRestitution(restitution);
    const double recovered = std::exp(-M_PI * zeta / std::sqrt(1.0 - zeta * zeta));
    EXPECT_NEAR(recovered, restitution, 1e-9) << "at e = " << restitution;
  }
}

TEST(ContactDampRatio, MoreBounceMeansLessDamping) {
  EXPECT_LT(contactDampRatioForRestitution(0.9), contactDampRatioForRestitution(0.5));
  EXPECT_LT(contactDampRatioForRestitution(0.5), contactDampRatioForRestitution(0.1));
}

TEST(ContactDampRatio, TheDegenerateEndsAreClampedRatherThanInfinite) {
  // e = 1 would be a contact that never loses energy; e = 0 would take the log of zero.
  EXPECT_TRUE(std::isfinite(contactDampRatioForRestitution(1.0)));
  EXPECT_TRUE(std::isfinite(contactDampRatioForRestitution(0.0)));
  EXPECT_TRUE(std::isfinite(contactDampRatioForRestitution(-1.0)));
  EXPECT_GT(contactDampRatioForRestitution(1.0), 0.0);
}

/*========================================= injecting the ball ==========================================*/

TEST(AddProjectileToSpec, AppendsAFreeSphereWithTheRightMassAndSize) {
  const std::string path = writeTempFile("projectile_scene.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene plain(path);
  const Scene withBall(path, &ball);
  ASSERT_TRUE(withBall.addStatus.ok()) << withBall.addStatus.message();
  ASSERT_NE(withBall.model, nullptr);

  // One more body, one more free joint: seven more qpos and six more qvel.
  EXPECT_EQ(withBall.model->nbody, plain.model->nbody + 1);
  EXPECT_EQ(withBall.model->nq, plain.model->nq + 7);
  EXPECT_EQ(withBall.model->nv, plain.model->nv + 6);

  const int ballBody = withBall.body("sim_projectile");
  ASSERT_GE(ballBody, 0);
  EXPECT_NEAR(withBall.model->body_mass[ballBody], ball.mass, 1e-9)
      << "the geom's mass must be set explicitly, or MuJoCo's 1000 kg/m^3 default makes this a 5 kg medicine ball";

  const int ballGeom = mj_name2id(withBall.model, mjOBJ_GEOM, "sim_projectile_geom");
  ASSERT_GE(ballGeom, 0);
  EXPECT_EQ(withBall.model->geom_type[ballGeom], mjGEOM_SPHERE);
  EXPECT_NEAR(withBall.model->geom_size[3 * ballGeom], ball.radius, 1e-9);
  EXPECT_NEAR(withBall.model->geom_solref[mjNREF * ballGeom + 1], contactDampRatioForRestitution(ball.restitution), 1e-9);
}

TEST(AddProjectileToSpec, TheBallIsLastAndTheRobotIsStillTheFirstFreeBody) {
  // The hazard that would be silent: robotCentroidalState, the viewer's tracking camera and every read of
  // qpos[3..6] as the base quaternion all take the FIRST free-joint body to be the robot. A ball inserted ahead of
  // it would quietly become the robot.
  const std::string path = writeTempFile("projectile_order.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);

  int firstFreeBody = -1;
  for (int body = 1; body < scene.model->nbody && firstFreeBody < 0; ++body) {
    if (scene.model->body_jntnum[body] > 0 && scene.model->jnt_type[scene.model->body_jntadr[body]] == mjJNT_FREE) {
      firstFreeBody = body;
    }
  }
  EXPECT_EQ(firstFreeBody, scene.body("base"));
  EXPECT_EQ(scene.body("sim_projectile"), scene.model->nbody - 1) << "the ball must be the LAST body";
  // And the robot's own addresses are unchanged, so nothing that indexes qpos by offset moves under it.
  const Scene plain(path);
  EXPECT_EQ(scene.model->jnt_qposadr[scene.model->body_jntadr[scene.body("base")]],
            plain.model->jnt_qposadr[plain.model->body_jntadr[plain.body("base")]]);
}

TEST(AddProjectileToSpec, TheBallIsCompiledCollidingAndParkedOutOfPlay) {
  const std::string path = writeTempFile("projectile_parked.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  const int ballGeom = mj_name2id(scene.model, mjOBJ_GEOM, "sim_projectile_geom");
  ASSERT_GE(ballGeom, 0);
  // Compiled COLLIDING - see setProjectileCollisionEnabled - and taken out of play at runtime instead.
  EXPECT_EQ(scene.model->geom_contype[ballGeom], 1);
  EXPECT_EQ(scene.model->geom_conaffinity[ballGeom], 1);
  EXPECT_NE(scene.model->body_contype[ballBody], 0) << "the compiler folds the geom flags into the body aggregates here, and only here";
  // And it starts out of sight rather than in the middle of the scene, which is also where every reset puts it.
  EXPECT_LT(scene.model->body_pos[3 * ballBody + 2], -1.0);
}

TEST(AddProjectileToSpec, RejectsANonsensicalBallAndANullSpec) {
  Projectile broken = dodgeball();
  broken.radius = 0.0;
  const std::string path = writeTempFile("projectile_broken.xml", kScene);
  char error[1000] = {0};
  mjSpec* spec = mj_parseXML(path.c_str(), nullptr, error, sizeof(error));
  ASSERT_NE(spec, nullptr);
  EXPECT_FALSE(addProjectileToSpec(spec, broken, "sim_projectile").ok());
  mj_deleteSpec(spec);
  EXPECT_FALSE(addProjectileToSpec(nullptr, dodgeball(), "sim_projectile").ok());
}

/*======================================= parking and arming ============================================*/

TEST(SetProjectileCollisionEnabled, ClearsAndRestoresTheGeomFlagsAndTheBodyAggregates) {
  const std::string path = writeTempFile("projectile_arming.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  const int ballGeom = mj_name2id(scene.model, mjOBJ_GEOM, "sim_projectile_geom");

  setProjectileCollisionEnabled(scene.model, ballBody, false);
  EXPECT_EQ(scene.model->geom_contype[ballGeom], 0);
  EXPECT_EQ(scene.model->geom_conaffinity[ballGeom], 0);
  EXPECT_EQ(scene.model->body_contype[ballBody], 0);
  EXPECT_EQ(scene.model->body_conaffinity[ballBody], 0);

  setProjectileCollisionEnabled(scene.model, ballBody, true);
  EXPECT_EQ(scene.model->geom_contype[ballGeom], 1);
  EXPECT_EQ(scene.model->geom_conaffinity[ballGeom], 1);
  EXPECT_EQ(scene.model->body_contype[ballBody], 1);
  EXPECT_EQ(scene.model->body_conaffinity[ballBody], 1);

  // The robot is untouched by either call.
  const int soleGeom = mj_name2id(scene.model, mjOBJ_GEOM, "sole");
  EXPECT_EQ(scene.model->geom_contype[soleGeom], 1);
}

TEST(SetProjectileCollisionEnabled, ToleratesAModelWithNoProjectileInIt) {
  const std::string path = writeTempFile("projectile_absent.xml", kScene);
  const Scene scene(path);
  ASSERT_NE(scene.model, nullptr);
  setProjectileCollisionEnabled(scene.model, -1, true);      // no projectile in this scene
  setProjectileCollisionEnabled(nullptr, 0, true);           // no model at all
  setProjectileCollisionEnabled(scene.model, 10'000, true);  // out of range
  SUCCEED();
}

TEST(SetProjectileCollisionEnabled, ABallParkedAndThenThrownStillCollides) {
  // The regression this file exists for. MuJoCo's broadphase prunes whole bodies by body_contype / body_conaffinity,
  // so clearing ONLY the geom flags while parked and restoring them on the throw leaves the body pruned for the rest
  // of the session: the ball flies through the robot and nothing anywhere reports an error.
  const std::string path = writeTempFile("projectile_broadphase.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");

  setProjectileCollisionEnabled(scene.model, ballBody, false);  // parked, as at startup and after every reset
  setProjectileCollisionEnabled(scene.model, ballBody, true);   // and thrown
  if (scene.model->body_gravcomp != nullptr) scene.model->body_gravcomp[ballBody] = 0.0;

  restBallOnTheFoot(scene, ball);
  EXPECT_GT(scene.data->ncon, 0) << "the ball generated no contact at all after a park-then-throw cycle";
}

/*======================================== retuning the mass ============================================*/

TEST(SetProjectileMass, SetsTheMassAndTheInertiaOfASolidSphere) {
  const std::string path = writeTempFile("projectile_mass.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");

  ASSERT_TRUE(setProjectileMass(scene.model, ballBody, 2.5, ball.radius).ok());
  EXPECT_NEAR(scene.model->body_mass[ballBody], 2.5, 1e-9);
  // I = 2/5 m r^2, isotropic, because a solid sphere has no preferred axis.
  const double expected = 0.4 * 2.5 * ball.radius * ball.radius;
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_NEAR(scene.model->body_inertia[3 * ballBody + axis], expected, 1e-12) << "axis " << axis;
  }
}

TEST(SetProjectileMass, RefreshesTheMassDerivedConstants) {
  // The omission this function exists to prevent. body_invweight0 is an inverse mass the COMPILER works out once, and
  // MuJoCo normalises a contact's reference acceleration by it. Setting body_mass and stopping there leaves it stale.
  const std::string path = writeTempFile("projectile_invweight.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  const int robotBody = scene.body("base");
  const double robotBefore = scene.model->body_invweight0[2 * robotBody];

  EXPECT_NEAR(scene.model->body_invweight0[2 * ballBody], 1.0 / ball.mass, 1e-6) << "the compiled value";
  ASSERT_TRUE(setProjectileMass(scene.model, ballBody, 5.0, ball.radius).ok());
  EXPECT_NEAR(scene.model->body_invweight0[2 * ballBody], 1.0 / 5.0, 1e-6)
      << "body_invweight0 was not refreshed, so the ball's restitution now depends on its mass";
  // And the robot's own constants are left where they were: the ball is a separate kinematic tree.
  EXPECT_NEAR(scene.model->body_invweight0[2 * robotBody], robotBefore, 1e-12);
}

TEST(SetProjectileMass, RestitutionDoesNotChangeWithTheMass) {
  // The physical invariant, and the reason the constants have to be refreshed: the coefficient of restitution is a
  // property of the CONTACT, so a heavy ball and a light one dropped from the same height must rebound to the same
  // height. This is the end-to-end form of the test above - it fails if body_invweight0 is left stale, whatever the
  // mechanism.
  const std::string path = writeTempFile("projectile_restitution.xml", kScene);
  const Projectile ball = dodgeball();
  std::vector<double> apexes;
  for (const double mass : {0.45, 2.0, 5.0}) {
    const Scene scene(path, &ball);
    ASSERT_NE(scene.model, nullptr);
    const int ballBody = scene.body("sim_projectile");
    ASSERT_TRUE(setProjectileMass(scene.model, ballBody, mass, ball.radius).ok());
    if (scene.model->body_gravcomp != nullptr) scene.model->body_gravcomp[ballBody] = 0.0;

    // Dropped on the floor well away from the robot, which is left to fall wherever it likes.
    const int ballJoint = scene.model->body_jntadr[ballBody];
    const int ballQpos = scene.model->jnt_qposadr[ballJoint];
    const int ballDof = scene.model->jnt_dofadr[ballJoint];
    scene.data->qpos[ballQpos + 0] = 3.0;
    scene.data->qpos[ballQpos + 1] = 3.0;
    scene.data->qpos[ballQpos + 2] = 0.5;
    scene.data->qpos[ballQpos + 3] = 1.0;
    scene.data->qvel[ballDof + 2] = -5.0;

    double apex = -1.0;
    bool bounced = false;
    for (int step = 0; step < 4000; ++step) {
      mj_step(scene.model, scene.data);
      if (scene.data->ncon > 0) bounced = true;
      if (bounced && scene.data->qvel[ballDof + 2] > 0.0) {
        apex = std::max(apex, scene.data->qpos[ballQpos + 2]);
      }
    }
    ASSERT_GT(apex, 0.0) << "the " << mass << " kg ball never bounced, so this proves nothing";
    apexes.push_back(apex);
  }
  const double spread = *std::max_element(apexes.begin(), apexes.end()) - *std::min_element(apexes.begin(), apexes.end());
  EXPECT_LT(spread, 1e-3) << "the rebound height moved by " << spread << " m across 0.45 -> 5.0 kg, so the mass is "
                          << "changing the contact rather than only the ball";
}

TEST(SetProjectileMass, ClampsToTheSliderRange) {
  const std::string path = writeTempFile("projectile_clamp.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");

  // A topic can be published by hand, and a zero or negative mass is a body MuJoCo cannot integrate.
  ASSERT_TRUE(setProjectileMass(scene.model, ballBody, 0.0, ball.radius).ok());
  EXPECT_NEAR(scene.model->body_mass[ballBody], kMinProjectileMass, 1e-12);
  ASSERT_TRUE(setProjectileMass(scene.model, ballBody, -3.0, ball.radius).ok());
  EXPECT_NEAR(scene.model->body_mass[ballBody], kMinProjectileMass, 1e-12);
  ASSERT_TRUE(setProjectileMass(scene.model, ballBody, 1.0e6, ball.radius).ok());
  EXPECT_NEAR(scene.model->body_mass[ballBody], kMaxProjectileMass, 1e-12);
  // The registry's nominal ball has to be reachable with the slider, or its default would be clamped away.
  EXPECT_GE(ball.mass, kMinProjectileMass);
  EXPECT_LE(ball.mass, kMaxProjectileMass);
}

TEST(SetProjectileMass, RejectsABadModelBodyOrRadiusWithoutTouchingAnything) {
  const std::string path = writeTempFile("projectile_mass_bad.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  const double before = scene.model->body_mass[ballBody];
  const double plainBaseMass = scene.model->body_mass[scene.body("base")];

  EXPECT_FALSE(setProjectileMass(nullptr, ballBody, 2.0, ball.radius).ok());
  EXPECT_FALSE(setProjectileMass(scene.model, -1, 2.0, ball.radius).ok());
  EXPECT_FALSE(setProjectileMass(scene.model, scene.model->nbody, 2.0, ball.radius).ok());
  // And the guard that matters: a body that is not a projectile. Retuning the wrong one would quietly change the
  // ROBOT's mass, which nothing downstream would report.
  EXPECT_FALSE(setProjectileMass(scene.model, 0, 2.0, ball.radius).ok()) << "the world body, which holds the floor";
  EXPECT_FALSE(setProjectileMass(scene.model, scene.body("base"), 2.0, ball.radius).ok())
      << "the robot's own free-jointed base, whose geom is a box";
  EXPECT_FALSE(setProjectileMass(scene.model, scene.body("foot"), 2.0, ball.radius).ok()) << "a hinge-jointed link";
  EXPECT_FALSE(setProjectileMass(scene.model, ballBody, 2.0, 0.0).ok()) << "a zero radius is a zero inertia";
  EXPECT_NEAR(scene.model->body_mass[scene.body("base")], plainBaseMass, 1e-12) << "a rejected call must not have touched the robot";
  EXPECT_NEAR(scene.model->body_mass[ballBody], before, 1e-12) << "a rejected call must change nothing";
}

TEST(SetProjectileMass, TheBallStillFliesAndStillIsNotTheGroundAfterwards) {
  // Retuning the mass must not undo any of the other invariants: the ball still collides, and a contact with it is
  // still excluded from the contact mask.
  const std::string path = writeTempFile("projectile_mass_contact.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  ASSERT_TRUE(setProjectileMass(scene.model, ballBody, kMaxProjectileMass, ball.radius).ok());
  if (scene.model->body_gravcomp != nullptr) scene.model->body_gravcomp[ballBody] = 0.0;
  restBallOnTheFoot(scene, ball);

  const std::vector<int> feet{scene.body("foot")};
  ASSERT_EQ(groundTruthContactMask(scene.model, scene.data, feet, 1.0, -1), 1u) << "the heavy ball is not touching";
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, feet, 1.0, ballBody), 0u);
}

/*==================================== the ball is not the ground =======================================*/

TEST(GroundTruthContactMask, ABallStrikingAFootIsNotThatFootBeingPlanted) {
  // The worst of the hazards: the mask feeds the RobotState's contact flags and the cheater_sim estimator, so a ball
  // against a SWING foot would tell the MPC that foot is planted.
  const std::string path = writeTempFile("projectile_contact.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  if (scene.model->body_gravcomp != nullptr) scene.model->body_gravcomp[ballBody] = 0.0;
  restBallOnTheFoot(scene, ball);

  const std::vector<int> feet{scene.body("foot")};
  // The positive control first: without the exclusion the ball DOES read as ground contact. Without this assertion a
  // mask of zero would prove nothing, because a test in which the ball never touched the foot would also pass.
  ASSERT_EQ(groundTruthContactMask(scene.model, scene.data, feet, 1.0, -1), 1u)
      << "the ball is not actually striking the foot, so this test proves nothing";
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, feet, 1.0, ballBody), 0u)
      << "a contact with the thrown ball must not set a contact bit";
}

TEST(GroundTruthContactMask, ExcludingTheBallLeavesRealGroundContactAlone) {
  // The other half of the contract: the exclusion must be narrow. A foot on the floor still reads as planted while a
  // ball is armed somewhere else in the scene.
  const std::string path = writeTempFile("projectile_ground.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);

  const int baseQpos = scene.model->jnt_qposadr[scene.model->body_jntadr[scene.body("base")]];
  // The sole's underside sits at base z - 0.3 - 0.02; drop it just through the floor so the solver pushes back.
  scene.data->qpos[baseQpos + 2] = 0.3 + 0.02 - 0.002;
  mj_forward(scene.model, scene.data);

  const std::vector<int> feet{scene.body("foot")};
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, feet, 1.0, scene.body("sim_projectile")), 1u)
      << "the foot is on the floor; excluding the ball must not suppress that";
}

}  // namespace robot::mujoco_sim_interface
