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
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/Projectile.h"

namespace robot::mujoco_sim_interface {
namespace {

/**
 * A scene shaped like the robot ones: a floor, a free-jointed "robot" with a foot, and nothing else. Small enough to
 * reason about, and enough to catch what breaks when a projectile is appended - the robot stopping being the first
 * free-joint body, and a ball being mistaken for the ground.
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

/**
 * The same, but with the contact parameters and <statistic> override the shipped robot scenes all use. The floor's
 * solref is what MuJoCo would average the ball's with if the ball did not take priority, and the statistic override
 * is what a runtime mj_setConst would silently throw away.
 */
constexpr const char* kShippedScene = R"(
<mujoco>
  <option timestep="0.0005"/>
  <statistic center="0 0 1.0" extent="1.5"/>
  <default>
    <geom solref="0.004 1" solimp="0.95 0.99 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1"/>
    <body name="base" pos="0 0 1">
      <freejoint name="base_free"/>
      <geom name="torso" type="box" size="0.1 0.1 0.2"/>
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
      model = mj_loadXML(path.c_str(), /*vfs=*/nullptr, error, sizeof(error));
    } else {
      mjSpec* spec = mj_parseXML(path.c_str(), /*vfs=*/nullptr, error, sizeof(error));
      if (spec != nullptr) {
        addStatus = addProjectileToSpec(spec, *projectile, "sim_projectile");
        if (addStatus.ok()) model = mj_compile(spec, /*vfs=*/nullptr);
        mj_deleteSpec(spec);
      }
    }
    if (model != nullptr) data = mj_makeData(model);
  }
  ~Scene() {
    if (data != nullptr) mj_deleteData(data);
    if (model != nullptr) mj_deleteModel(model);
  }
  Scene(const Scene&) = delete;
  Scene& operator=(const Scene&) = delete;

  int body(const char* name) const { return mj_name2id(model, mjOBJ_BODY, name); }
  int geom(const char* name) const { return mj_name2id(model, mjOBJ_GEOM, name); }
  int ballQpos() const { return model->jnt_qposadr[model->body_jntadr[body("sim_projectile")]]; }
  int ballDof() const { return model->jnt_dofadr[model->body_jntadr[body("sim_projectile")]]; }

  /** Arms the ball the way MujocoSimInterface::setProjectileArmed does: collisions on, gravity compensation off. */
  void arm() const {
    setProjectileCollisionEnabled(model, body("sim_projectile"), /*enabled=*/true);
    model->body_gravcomp[body("sim_projectile")] = 0.0;
  }

  /** Puts the ball at `position` with linear velocity `velocity`, unrotated and not spinning. */
  void place(const std::array<double, 3>& position, const std::array<double, 3>& velocity) const {
    for (int axis = 0; axis < 3; ++axis) {
      data->qpos[ballQpos() + axis] = position[axis];
      data->qvel[ballDof() + axis] = velocity[axis];
      data->qvel[ballDof() + 3 + axis] = 0.0;
    }
    data->qpos[ballQpos() + 3] = 1.0;
    data->qpos[ballQpos() + 4] = 0.0;
    data->qpos[ballQpos() + 5] = 0.0;
    data->qpos[ballQpos() + 6] = 0.0;
  }

  bool ballTouchesAnything() const {
    const int ballGeom = geom("sim_projectile_geom");
    for (int c = 0; c < data->ncon; ++c) {
      if (data->contact[c].geom[0] == ballGeom || data->contact[c].geom[1] == ballGeom) return true;
    }
    return false;
  }

  mjModel* model{nullptr};
  mjData* data{nullptr};
  absl::Status addStatus;
};

Projectile dodgeball() {
  return *projectileFromName("dodgeball");
}

/**
 * Puts the ball against the top of the foot's sole and still traveling downwards, which is what a ball in the middle
 * of an impact looks like and what makes the normal force large rather than marginal, then runs the solver.
 */
void strikeTheFoot(const Scene& scene, const Projectile& ball) {
  const int baseQpos = scene.model->jnt_qposadr[scene.model->body_jntadr[scene.body("base")]];
  scene.data->qpos[baseQpos + 2] = 1.0;
  // The sole's top face sits at base z - 0.3 + 0.02.
  scene.place({{0.0, 0.0, 1.0 - 0.3 + 0.02 + ball.radius - 0.002}}, {{0.0, 0.0, -12.0}});
  mj_forward(scene.model, scene.data);
}

/**
 * Coefficient of restitution of one bounce on the floor, measured as rebound speed over impact speed: drops the ball
 * from `height` well away from the robot and returns v_out / v_in for its first floor contact.
 */
double measuredRestitution(const Scene& scene, double height) {
  scene.arm();
  scene.place({{3.0, 3.0, height}}, {{0.0, 0.0, 0.0}});
  const int dof = scene.ballDof() + 2;
  double impactSpeed = -1.0;
  double previous = 0.0;
  const int steps = static_cast<int>(3.0 / scene.model->opt.timestep);
  for (int step = 0; step < steps; ++step) {
    mj_step(scene.model, scene.data);
    const double verticalSpeed = scene.data->qvel[dof];
    const bool touching = scene.ballTouchesAnything();
    if (impactSpeed < 0.0 && touching && previous < 0.0) impactSpeed = -previous;
    if (impactSpeed > 0.0 && !touching && verticalSpeed > 0.0) return verticalSpeed / impactSpeed;
    previous = verticalSpeed;
  }
  return std::numeric_limits<double>::quiet_NaN();
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
  EXPECT_GT(projectile->rollingFriction, 0.0) << "without rolling resistance a ball on the floor rolls forever";
  // The slider's default has to be reachable, or it would be clamped away on the first throw.
  EXPECT_GE(projectile->mass, kMinProjectileMass);
  EXPECT_LE(projectile->mass, kMaxProjectileMass);
}

TEST(ProjectileRegistry, AnUnknownNameListsTheValidOnes) {
  const absl::StatusOr<Projectile> projectile = projectileFromName("cannonball");
  ASSERT_FALSE(projectile.ok());
  EXPECT_EQ(projectile.status().code(), absl::StatusCode::kInvalidArgument);
  // Every valid name must appear, and "dodgeball" is not a substring of "cannonball", so this cannot pass by echo.
  for (const std::string& name : availableProjectiles()) {
    EXPECT_NE(std::string(projectile.status().message()).find(name), std::string::npos) << projectile.status().message();
  }
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

TEST(AddProjectileToSpec, AppendsAFreeSphereWithTheRightMassSizeAndContact) {
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

  const int ballGeom = withBall.geom("sim_projectile_geom");
  ASSERT_GE(ballGeom, 0);
  EXPECT_EQ(withBall.model->geom_type[ballGeom], mjGEOM_SPHERE);
  EXPECT_NEAR(withBall.model->geom_size[3 * ballGeom], ball.radius, 1e-9);
  EXPECT_NEAR(withBall.model->geom_solref[mjNREF * ballGeom + 0], kProjectileContactTimeConstant, 1e-12);
  EXPECT_NEAR(withBall.model->geom_solref[mjNREF * ballGeom + 1], contactDampRatioForRestitution(ball.restitution), 1e-9);
  // Priority, so that it is the BALL's solref that every contact uses rather than an average with the floor's.
  EXPECT_GT(withBall.model->geom_priority[ballGeom], withBall.model->geom_priority[withBall.geom("floor")]);
  // condim 6 and the (slide, spin, roll) friction that makes a rolling ball stop.
  EXPECT_EQ(withBall.model->geom_condim[ballGeom], 6);
  EXPECT_NEAR(withBall.model->geom_friction[3 * ballGeom + 0], ball.friction, 1e-12);
  EXPECT_NEAR(withBall.model->geom_friction[3 * ballGeom + 1], ball.rollingFriction, 1e-12);
  EXPECT_NEAR(withBall.model->geom_friction[3 * ballGeom + 2], ball.rollingFriction, 1e-12);
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

TEST(AddProjectileToSpec, TheBallIsCompiledCollidingParkedAboveTheSceneAndGravityCompensated) {
  const std::string path = writeTempFile("projectile_parked.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  const int ballGeom = scene.geom("sim_projectile_geom");
  // Compiled COLLIDING - see setProjectileCollisionEnabled - and taken out of play at runtime instead.
  EXPECT_EQ(scene.model->geom_contype[ballGeom], 1);
  EXPECT_EQ(scene.model->geom_conaffinity[ballGeom], 1);
  EXPECT_NE(scene.model->body_contype[ballBody], 0) << "the compiler folds the geom flags into the body aggregates here, and only here";
  // Gravity compensation compiled in, so that the compiler COUNTS the body: MuJoCo skips the whole gravity
  // compensation pass when ngravcomp is zero, and a runtime body_gravcomp write then does nothing.
  EXPECT_GE(scene.model->ngravcomp, 1);
  EXPECT_NEAR(scene.model->body_gravcomp[ballBody], 1.0, 1e-12);
  // Parked ABOVE the scene: every floor is a plane, and a plane collides as a half-space.
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_NEAR(scene.model->body_pos[3 * ballBody + axis], kProjectileParkPosition[axis], 1e-12);
  }
  EXPECT_GT(kProjectileParkPosition[2], 10.0);
}

TEST(AddProjectileToSpec, AParkedBallStaysWhereItWasParked) {
  // The regression: a ball compiled without gravcomp free-falls from its parking spot for the whole session, and after
  // about twelve hours its position overflows and MuJoCo resets the entire simulation.
  const std::string path = writeTempFile("projectile_stays.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  setProjectileCollisionEnabled(scene.model, ballBody, /*enabled=*/false);
  for (int step = 0; step < 2000; ++step) mj_step(scene.model, scene.data);
  EXPECT_NEAR(scene.data->qpos[scene.ballQpos() + 2], kProjectileParkPosition[2], 1e-6) << "the parked ball is falling";

  // And switching the compensation off at runtime - which is what a throw does - does let it fall.
  scene.model->body_gravcomp[ballBody] = 0.0;
  const int steps = 1000;
  const double dt = scene.model->opt.timestep;
  for (int step = 0; step < steps; ++step) mj_step(scene.model, scene.data);
  // MuJoCo's semi-implicit Euler drops g dt^2 N (N + 1) / 2 in N steps, a little more than the continuous g t^2 / 2.
  EXPECT_NEAR(scene.data->qpos[scene.ballQpos() + 2], kProjectileParkPosition[2] - 9.81 * dt * dt * steps * (steps + 1) / 2.0, 1e-6);
}

TEST(AddProjectileToSpec, RejectsANonsensicalBallANeverLookedUpBallAndANullSpec) {
  const std::string path = writeTempFile("projectile_broken.xml", kScene);
  char error[1000] = {0};
  mjSpec* spec = mj_parseXML(path.c_str(), /*vfs=*/nullptr, error, sizeof(error));
  ASSERT_NE(spec, nullptr);
  Projectile broken = dodgeball();
  broken.radius = 0.0;
  EXPECT_FALSE(addProjectileToSpec(spec, broken, "sim_projectile").ok());
  broken = dodgeball();
  broken.restitution = 1.0;
  EXPECT_FALSE(addProjectileToSpec(spec, broken, "sim_projectile").ok()) << "a restitution of 1 never settles";
  broken = dodgeball();
  broken.rollingFriction = -0.1;
  EXPECT_FALSE(addProjectileToSpec(spec, broken, "sim_projectile").ok());
  EXPECT_FALSE(addProjectileToSpec(spec, Projectile{}, "sim_projectile").ok()) << "a default Projectile is not a ball";
  mj_deleteSpec(spec);
  EXPECT_FALSE(addProjectileToSpec(/*spec=*/nullptr, dodgeball(), "sim_projectile").ok());
}

/*=================================== the bounce the ball actually gets =================================*/

TEST(ProjectileContact, RestitutionIsTheConfiguredOneAgainstAFloorWithItsOwnContactParameters) {
  // The regression: the ball's solref used to be averaged with the floor's (0.004, 1.0), giving a bounce of about 0.25
  // whatever the ball was configured with. The shipped scenes' floor parameters are in kShippedScene.
  const std::string path = writeTempFile("projectile_restitution.xml", kShippedScene);
  const Projectile ball = dodgeball();
  for (const double timestep : {0.0005, 0.001}) {
    const Scene scene(path, &ball);
    ASSERT_NE(scene.model, nullptr);
    scene.model->opt.timestep = timestep;
    const double restitution = measuredRestitution(scene, /*height=*/1.0);
    ASSERT_TRUE(std::isfinite(restitution)) << "the ball never bounced at dt = " << timestep;
    // The discrete contact is not exactly the continuous one, so a band: 0.797 and 0.842 measured.
    EXPECT_NEAR(restitution, ball.restitution, 0.06) << "at dt = " << timestep;
  }
}

TEST(ProjectileContact, TheContactDoesNotAddEnergyAtTheSimulatorsTimesteps) {
  // A contact time constant the integrator cannot resolve rings and ADDS energy - rebounds of 4 to 50 times the
  // arrival speed were measured at 2 ms. Whatever else changes, a bounce must never come back faster than it went in.
  const std::string path = writeTempFile("projectile_energy.xml", kShippedScene);
  const Projectile ball = dodgeball();
  for (const double timestep : {0.00025, 0.0005, 0.001}) {
    const Scene scene(path, &ball);
    ASSERT_NE(scene.model, nullptr);
    scene.model->opt.timestep = timestep;
    EXPECT_LT(measuredRestitution(scene, /*height=*/2.0), 1.0) << "at dt = " << timestep;
  }
}

TEST(ProjectileContact, ARollingBallComesToRestAndIsParkedBeforeItsLifetimeRunsOut) {
  // With condim 3 a sphere rolling on a plane keeps 5/7 of its speed forever, and a ball that is never slow is never
  // parked. The rolling friction has to bring it to rest, so that it is the at-rest rule that parks it.
  const std::string path = writeTempFile("projectile_roll.xml", kShippedScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  scene.arm();
  scene.place({{3.0, 3.0, ball.radius}}, {{3.0, 0.0, 0.0}});
  ProjectileRestMonitor monitor;
  monitor.start();
  const double timestep = scene.model->opt.timestep;
  double parkedAt = -1.0;
  for (int step = 0; step * timestep < ProjectileRestMonitor::kLifetime && parkedAt < 0.0; ++step) {
    mj_step(scene.model, scene.data);
    const double speed =
        std::hypot(scene.data->qvel[scene.ballDof()], scene.data->qvel[scene.ballDof() + 1], scene.data->qvel[scene.ballDof() + 2]);
    if (monitor.update(speed, timestep)) parkedAt = scene.data->time;
  }
  ASSERT_GT(parkedAt, 0.0) << "the ball was still rolling when its lifetime ran out";
  EXPECT_LT(parkedAt, ProjectileRestMonitor::kLifetime - 1.0) << "it was the lifetime, not the ball coming to rest, that parked it";
}

/*======================================= parking and arming ============================================*/

TEST(SetProjectileCollisionEnabled, ClearsAndRestoresTheGeomFlagsAndTheBodyAggregates) {
  const std::string path = writeTempFile("projectile_arming.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  const int ballGeom = scene.geom("sim_projectile_geom");

  setProjectileCollisionEnabled(scene.model, ballBody, /*enabled=*/false);
  EXPECT_EQ(scene.model->geom_contype[ballGeom], 0);
  EXPECT_EQ(scene.model->geom_conaffinity[ballGeom], 0);
  EXPECT_EQ(scene.model->body_contype[ballBody], 0);
  EXPECT_EQ(scene.model->body_conaffinity[ballBody], 0);

  setProjectileCollisionEnabled(scene.model, ballBody, /*enabled=*/true);
  EXPECT_EQ(scene.model->geom_contype[ballGeom], 1);
  EXPECT_EQ(scene.model->geom_conaffinity[ballGeom], 1);
  EXPECT_EQ(scene.model->body_contype[ballBody], 1);
  EXPECT_EQ(scene.model->body_conaffinity[ballBody], 1);

  // The robot is untouched by either call.
  EXPECT_EQ(scene.model->geom_contype[scene.geom("sole")], 1);
}

TEST(SetProjectileCollisionEnabled, ToleratesAModelWithNoProjectileInIt) {
  const std::string path = writeTempFile("projectile_absent.xml", kScene);
  const Scene scene(path);
  ASSERT_NE(scene.model, nullptr);
  setProjectileCollisionEnabled(scene.model, /*bodyId=*/-1, /*enabled=*/true);       // no projectile in this scene
  setProjectileCollisionEnabled(/*model=*/nullptr, /*bodyId=*/0, /*enabled=*/true);  // no model at all
  setProjectileCollisionEnabled(scene.model, /*bodyId=*/10'000, /*enabled=*/true);   // out of range
  SUCCEED();
}

TEST(SetProjectileCollisionEnabled, ABallParkedAndThenThrownStillCollides) {
  // MuJoCo's broadphase prunes whole bodies by body_contype / body_conaffinity, so clearing ONLY the geom flags while
  // parked and restoring them on the throw leaves the body pruned for the rest of the session: the ball flies through
  // the robot and nothing anywhere reports an error.
  const std::string path = writeTempFile("projectile_broadphase.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");

  setProjectileCollisionEnabled(scene.model, ballBody, /*enabled=*/false);  // parked, as at startup and after every reset
  scene.arm();                                                              // and thrown
  strikeTheFoot(scene, ball);
  EXPECT_TRUE(scene.ballTouchesAnything()) << "the ball generated no contact at all after a park-then-throw cycle";
}

/*======================================== retuning the mass ============================================*/

TEST(ClampProjectileMass, HoldsTheSliderRangeAndMapsNanToTheMinimum) {
  EXPECT_DOUBLE_EQ(clampProjectileMass(2.0), 2.0);
  EXPECT_DOUBLE_EQ(clampProjectileMass(0.0), kMinProjectileMass);
  EXPECT_DOUBLE_EQ(clampProjectileMass(-3.0), kMinProjectileMass);
  EXPECT_DOUBLE_EQ(clampProjectileMass(1.0e6), kMaxProjectileMass);
  EXPECT_DOUBLE_EQ(clampProjectileMass(std::numeric_limits<double>::infinity()), kMaxProjectileMass);
  EXPECT_DOUBLE_EQ(clampProjectileMass(std::numeric_limits<double>::quiet_NaN()), kMinProjectileMass)
      << "NaN fails every comparison, so std::clamp alone would pass it through";
}

TEST(SetProjectileMass, SetsTheMassAndTheInertiaOfASolidSphere) {
  const std::string path = writeTempFile("projectile_mass.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");

  ASSERT_TRUE(setProjectileMass(scene.model, ballBody, /*mass=*/2.5).ok());
  EXPECT_NEAR(scene.model->body_mass[ballBody], 2.5, 1e-9);
  // I = 2/5 m r^2, isotropic, because a solid sphere has no preferred axis.
  const double expected = 0.4 * 2.5 * ball.radius * ball.radius;
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_NEAR(scene.model->body_inertia[3 * ballBody + axis], expected, 1e-12) << "axis " << axis;
  }
}

TEST(SetProjectileMass, LeavesTheModelExactlyAsAFreshCompileAtThatMassWould) {
  // The closed-form constants are only a valid replacement for mj_setConst if they are what the compiler produces.
  // Compare every mass-derived field against a scene compiled from scratch with the ball at the new mass.
  const std::string path = writeTempFile("projectile_fresh.xml", kScene);
  const Projectile ball = dodgeball();
  Projectile heavy = dodgeball();
  heavy.mass = 3.7;
  const Scene retuned(path, &ball);
  const Scene fresh(path, &heavy);
  ASSERT_NE(retuned.model, nullptr);
  ASSERT_NE(fresh.model, nullptr);
  ASSERT_TRUE(setProjectileMass(retuned.model, retuned.body("sim_projectile"), heavy.mass).ok());

  const mjModel* a = retuned.model;
  const mjModel* b = fresh.model;
  for (int body = 0; body < a->nbody; ++body) {
    EXPECT_NEAR(a->body_mass[body], b->body_mass[body], 1e-9) << "body " << body;
    EXPECT_NEAR(a->body_subtreemass[body], b->body_subtreemass[body], 1e-9) << "body " << body;
    for (int k = 0; k < 2; ++k) {
      EXPECT_NEAR(a->body_invweight0[2 * body + k], b->body_invweight0[2 * body + k],
                  1e-6 * std::max(1.0, b->body_invweight0[2 * body + k]))
          << "body " << body << " component " << k;
    }
    for (int axis = 0; axis < 3; ++axis) {
      EXPECT_NEAR(a->body_inertia[3 * body + axis], b->body_inertia[3 * body + axis], 1e-9) << "body " << body;
    }
  }
  for (int dof = 0; dof < a->nv; ++dof) {
    EXPECT_NEAR(a->dof_invweight0[dof], b->dof_invweight0[dof], 1e-6 * std::max(1.0, b->dof_invweight0[dof])) << "dof " << dof;
    EXPECT_NEAR(a->dof_M0[dof], b->dof_M0[dof], 1e-9) << "dof " << dof;
  }
}

TEST(SetProjectileMass, LeavesTheScenesStatisticOverrideAlone) {
  // mj_setConst would recompute stat from the qpos0 bounding box - which includes the parked ball - and discard the
  // scene's own <statistic extent center>, moving the viewer's near clip plane from 1.5 cm to half a meter.
  const std::string path = writeTempFile("projectile_statistic.xml", kShippedScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  ASSERT_NEAR(scene.model->stat.extent, 1.5, 1e-12) << "the fixture's override did not take";
  ASSERT_TRUE(setProjectileMass(scene.model, scene.body("sim_projectile"), /*mass=*/4.0).ok());
  EXPECT_NEAR(scene.model->stat.extent, 1.5, 1e-12);
  EXPECT_NEAR(scene.model->stat.center[2], 1.0, 1e-12);
}

TEST(SetProjectileMass, RestitutionDoesNotChangeWithTheMass) {
  // The physical invariant the derived constants exist for: restitution belongs to the CONTACT, so a heavy ball and a
  // light one dropped from the same height rebound alike. This fails if body_invweight0 is left stale.
  const std::string path = writeTempFile("projectile_restitution_mass.xml", kShippedScene);
  const Projectile ball = dodgeball();
  std::vector<double> restitutions;
  for (const double mass : {kMinProjectileMass, ball.mass, kMaxProjectileMass}) {
    const Scene scene(path, &ball);
    ASSERT_NE(scene.model, nullptr);
    ASSERT_TRUE(setProjectileMass(scene.model, scene.body("sim_projectile"), mass).ok());
    const double restitution = measuredRestitution(scene, /*height=*/1.0);
    ASSERT_TRUE(std::isfinite(restitution)) << "the " << mass << " kg ball never bounced";
    restitutions.push_back(restitution);
  }
  const double spread =
      *std::max_element(restitutions.begin(), restitutions.end()) - *std::min_element(restitutions.begin(), restitutions.end());
  EXPECT_LT(spread, 1e-3) << "the restitution moved by " << spread << " across the mass range";
}

TEST(SetProjectileMass, ClampsToTheSliderRangeInTheMassAndTheInertia) {
  const std::string path = writeTempFile("projectile_clamp.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  // A topic can be published by hand, and a zero or negative mass is a body MuJoCo cannot integrate. The inertia and
  // the derived constants have to come from the CLAMPED mass too, not only body_mass.
  for (const double requested : {0.0, -3.0, std::numeric_limits<double>::quiet_NaN()}) {
    ASSERT_TRUE(setProjectileMass(scene.model, ballBody, requested).ok());
    EXPECT_NEAR(scene.model->body_mass[ballBody], kMinProjectileMass, 1e-12) << requested;
    EXPECT_NEAR(scene.model->body_inertia[3 * ballBody], 0.4 * kMinProjectileMass * ball.radius * ball.radius, 1e-12) << requested;
    EXPECT_NEAR(scene.model->body_invweight0[2 * ballBody], 1.0 / kMinProjectileMass, 1e-9) << requested;
  }
  ASSERT_TRUE(setProjectileMass(scene.model, ballBody, /*mass=*/1.0e6).ok());
  EXPECT_NEAR(scene.model->body_mass[ballBody], kMaxProjectileMass, 1e-12);
  EXPECT_NEAR(scene.model->body_inertia[3 * ballBody], 0.4 * kMaxProjectileMass * ball.radius * ball.radius, 1e-12);
}

TEST(SetProjectileMass, RejectsAnythingThatIsNotAProjectileWithoutTouchingIt) {
  const std::string path = writeTempFile("projectile_mass_bad.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  const double ballMass = scene.model->body_mass[ballBody];
  const double baseMass = scene.model->body_mass[scene.body("base")];

  EXPECT_FALSE(setProjectileMass(/*model=*/nullptr, ballBody, /*mass=*/2.0).ok());
  EXPECT_FALSE(setProjectileMass(scene.model, /*bodyId=*/-1, /*mass=*/2.0).ok());
  EXPECT_FALSE(setProjectileMass(scene.model, scene.model->nbody, /*mass=*/2.0).ok());
  // Retuning the wrong body would quietly change the ROBOT's mass, which nothing downstream would report.
  EXPECT_FALSE(setProjectileMass(scene.model, /*bodyId=*/0, /*mass=*/2.0).ok()) << "the world body, which holds the floor";
  EXPECT_FALSE(setProjectileMass(scene.model, scene.body("base"), /*mass=*/2.0).ok())
      << "the robot's own free-jointed base, whose geom is a box";
  EXPECT_FALSE(setProjectileMass(scene.model, scene.body("foot"), /*mass=*/2.0).ok()) << "a hinge-jointed link";
  EXPECT_NEAR(scene.model->body_mass[ballBody], ballMass, 1e-12);
  EXPECT_NEAR(scene.model->body_mass[scene.body("base")], baseMass, 1e-12) << "a rejected call must not have touched the robot";
}

TEST(SetProjectileMass, RejectsASphereThatIsNotCenteredOrNotItsOwnTree) {
  // The closed-form constants hold only for a centered sphere on its own free joint under the world; anything else
  // has cross terms. Build both counterexamples and require a refusal.
  const std::string path = writeTempFile("projectile_not_centered.xml", R"(
<mujoco>
  <worldbody>
    <body name="offcentre" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" pos="0.05 0 0" mass="1"/>
    </body>
    <body name="carrier" pos="2 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="nested" pos="0 0 0.5">
        <freejoint/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)");
  char error[1000] = {0};
  mjModel* model = mj_loadXML(path.c_str(), /*vfs=*/nullptr, error, sizeof(error));
  if (model == nullptr) GTEST_SKIP() << "MuJoCo rejected the nested free joint fixture: " << error;
  EXPECT_FALSE(setProjectileMass(model, mj_name2id(model, mjOBJ_BODY, "offcentre"), /*mass=*/2.0).ok());
  EXPECT_FALSE(setProjectileMass(model, mj_name2id(model, mjOBJ_BODY, "nested"), /*mass=*/2.0).ok());
  mj_deleteModel(model);
}

TEST(SetProjectileMass, TheBallStillCollidesAndStillIsNotTheGroundAfterwards) {
  const std::string path = writeTempFile("projectile_mass_contact.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  ASSERT_TRUE(setProjectileMass(scene.model, ballBody, kMaxProjectileMass).ok());
  scene.arm();
  strikeTheFoot(scene, ball);

  const std::vector<int> feet{scene.body("foot")};
  ASSERT_EQ(groundTruthContactMask(scene.model, scene.data, feet, /*forceThreshold=*/1.0, /*ignoreBodyId=*/-1), 1u)
      << "the heavy ball is not touching";
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, feet, /*forceThreshold=*/1.0, ballBody), 0u);
}

/*===================================== joint damping skips the ball ====================================*/

TEST(RobotJointDamping, SetsTheRobotsJointsAndLeavesTheRootAndTheBallAlone) {
  // 20 N s/m on the ball's free joint brings a throw to a 0.2 m/s terminal speed within centimeters of its start.
  const std::string path = writeTempFile("projectile_damping.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  setRobotJointDamping(scene.model, ballBody, /*damping=*/20.0);
  for (int dof = 0; dof < scene.model->nv; ++dof) {
    const bool root = dof < 6;
    const bool isBall = isProjectileDof(scene.model, ballBody, dof);
    EXPECT_NEAR(scene.model->dof_damping[dof], (root || isBall) ? 0.0 : 20.0, 1e-12) << "dof " << dof;
  }
  // The ankle is the robot's only joint, and the ball's six dofs follow it.
  EXPECT_NEAR(scene.model->dof_damping[6], 20.0, 1e-12);
  for (int dof = 7; dof < 13; ++dof) EXPECT_TRUE(isProjectileDof(scene.model, ballBody, dof)) << dof;
  EXPECT_FALSE(isProjectileDof(scene.model, ballBody, /*dof=*/6));
  EXPECT_FALSE(isProjectileDof(scene.model, /*projectileBodyId=*/-1, /*dof=*/7)) << "no projectile, no projectile dofs";
}

/*===================================== a spawn point that is not clear =================================*/

namespace {

ProjectileLaunch launch(const std::array<double, 3>& position, const std::array<double, 3>& velocity, double flightTime) {
  ProjectileLaunch requested;
  requested.position = position;
  requested.velocity = velocity;
  requested.flightTime = flightTime;
  return requested;
}

/** The moved launch must lie on the requested one's ballistic path, `tau` seconds along it. */
void expectOnThePath(const ProjectileLaunch& requested, const ProjectileLaunch& moved, double gravity) {
  const double tau = requested.flightTime - moved.flightTime;
  for (int axis = 0; axis < 3; ++axis) {
    const double drop = axis == 2 ? 0.5 * gravity * tau * tau : 0.0;
    EXPECT_NEAR(moved.position[axis], requested.position[axis] + requested.velocity[axis] * tau - drop, 1e-9) << "axis " << axis;
    EXPECT_NEAR(moved.velocity[axis], requested.velocity[axis] - (axis == 2 ? gravity * tau : 0.0), 1e-9) << "axis " << axis;
  }
}

}  // namespace

TEST(ClearProjectileLaunch, AClearSpawnIsLeftExactlyAsRequested) {
  const std::string path = writeTempFile("projectile_clear.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  mj_forward(scene.model, scene.data);
  const ProjectileLaunch requested = launch({{3.0, 0.0, 1.0}}, {{-8.0, 0.0, 1.8}}, /*flightTime=*/0.375);
  const absl::StatusOr<ProjectileLaunch> result =
      clearProjectileLaunch(scene.model, scene.data, scene.body("sim_projectile"), requested, /*gravity=*/9.81);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result->position, requested.position);
  EXPECT_EQ(result->velocity, requested.velocity);
  EXPECT_EQ(result->flightTime, requested.flightTime);
}

TEST(ClearProjectileLaunch, ASpawnInsideTheRobotIsMovedBackAlongItsApproach) {
  // A short throw from the side that starts inside the torso: MuJoCo would fire it out at tens of m/s.
  const std::string path = writeTempFile("projectile_inside.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  mj_forward(scene.model, scene.data);
  const ProjectileLaunch requested = launch({{0.15, 0.0, 1.0}}, {{-4.0, 0.0, 0.5}}, /*flightTime=*/0.04);
  const absl::StatusOr<ProjectileLaunch> result =
      clearProjectileLaunch(scene.model, scene.data, scene.body("sim_projectile"), requested, /*gravity=*/9.81);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_GT(result->flightTime, requested.flightTime) << "it should start EARLIER on its approach, so fly for longer";
  expectOnThePath(requested, *result, /*gravity=*/9.81);
  // And it is now clear of the torso's +x face at x = 0.1, by the clearance.
  EXPECT_GE(result->position[0] - ball.radius - 0.1, kProjectileSpawnClearance - 1e-9);
}

TEST(ClearProjectileLaunch, ASpawnUnderTheFloorIsMovedForwardAlongItsClimb) {
  // A negative elevation from far away starts below the floor plane, which collides as a half-space.
  const std::string path = writeTempFile("projectile_under.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  mj_forward(scene.model, scene.data);
  const ProjectileLaunch requested = launch({{2.0, 0.0, -0.5}}, {{-3.0, 0.0, 6.0}}, /*flightTime=*/0.6);
  const absl::StatusOr<ProjectileLaunch> result =
      clearProjectileLaunch(scene.model, scene.data, scene.body("sim_projectile"), requested, /*gravity=*/9.81);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_LT(result->flightTime, requested.flightTime) << "it should start LATER on its climb";
  expectOnThePath(requested, *result, /*gravity=*/9.81);
  EXPECT_GE(result->position[2], ball.radius + kProjectileSpawnClearance - 1e-9);
}

TEST(ClearProjectileLaunch, ReportsAThrowThatCannotBeClearedAndRejectsNonsense) {
  const std::string path = writeTempFile("projectile_hopeless.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  mj_forward(scene.model, scene.data);
  const int ballBody = scene.body("sim_projectile");
  // Five meters under the floor and heading further down: no point on its path is above ground.
  const absl::StatusOr<ProjectileLaunch> hopeless = clearProjectileLaunch(
      scene.model, scene.data, ballBody, launch({{2.0, 0.0, -5.0}}, {{0.0, 0.0, -1.0}}, /*flightTime=*/0.5), /*gravity=*/9.81);
  ASSERT_FALSE(hopeless.ok());
  EXPECT_EQ(hopeless.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_NE(std::string(hopeless.status().message()).find("spawn distance"), std::string::npos) << "it should name the slider to change";

  const ProjectileLaunch fine = launch({{3.0, 0.0, 1.0}}, {{-8.0, 0.0, 1.8}}, /*flightTime=*/0.375);
  EXPECT_EQ(clearProjectileLaunch(scene.model, scene.data, scene.body("base"), fine, /*gravity=*/9.81).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(clearProjectileLaunch(scene.model, /*data=*/nullptr, ballBody, fine, /*gravity=*/9.81).status().code(),
            absl::StatusCode::kInvalidArgument);
  const ProjectileLaunch notFinite =
      launch({{std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0}}, {{-8.0, 0.0, 1.8}}, /*flightTime=*/0.375);
  EXPECT_EQ(clearProjectileLaunch(scene.model, scene.data, ballBody, notFinite, /*gravity=*/9.81).status().code(),
            absl::StatusCode::kInvalidArgument);
}

/*=========================================== the fallback impulse ======================================*/

TEST(ProjectileArrivalMomentum, IsTheClampedMassTimesTheArrivalVelocity) {
  const std::array<double, 3> launchVelocity{{-8.0, 1.0, 2.0}};
  const std::array<double, 3> momentum = projectileArrivalMomentum(launchVelocity, /*flightTime=*/0.4, /*mass=*/2.0, /*gravity=*/9.81);
  EXPECT_NEAR(momentum[0], 2.0 * -8.0, 1e-12);
  EXPECT_NEAR(momentum[1], 2.0 * 1.0, 1e-12);
  EXPECT_NEAR(momentum[2], 2.0 * (2.0 - 9.81 * 0.4), 1e-12) << "gravity takes g t off the vertical on the way";
  // The same clamp as the real ball, so a hand-published mass cannot schedule an arbitrary impulse.
  const std::array<double, 3> heavy = projectileArrivalMomentum(launchVelocity, /*flightTime=*/0.4, /*mass=*/500.0, /*gravity=*/9.81);
  EXPECT_NEAR(heavy[0], kMaxProjectileMass * -8.0, 1e-12);
}

/*=========================================== when to park it ===========================================*/

TEST(ProjectileRestMonitor, ParksABallThatHasBeenSlowForTheRestDuration) {
  ProjectileRestMonitor monitor;
  monitor.start();
  const double dt = 0.001;
  int steps = 0;
  while (!monitor.update(0.5 * ProjectileRestMonitor::kRestSpeed, dt)) ++steps;
  EXPECT_NEAR((steps + 1) * dt, ProjectileRestMonitor::kRestDuration, 2 * dt);
}

TEST(ProjectileRestMonitor, ABurstOfSpeedRestartsTheRestClock) {
  // A ball balanced on a foot for an instant on its way past must not be parked out from under the robot.
  ProjectileRestMonitor monitor;
  monitor.start();
  const double dt = 0.001;
  for (int step = 0; step < 200; ++step) EXPECT_FALSE(monitor.update(/*speed=*/0.0, dt));
  EXPECT_FALSE(monitor.update(/*speed=*/1.0, dt));
  for (int step = 0; step < 200; ++step) EXPECT_FALSE(monitor.update(/*speed=*/0.0, dt)) << step;
}

TEST(ProjectileRestMonitor, ParksABallThatNeverSlowsDownWhenItsLifetimeRunsOut) {
  ProjectileRestMonitor monitor;
  monitor.start();
  const double dt = 0.01;
  int steps = 0;
  while (!monitor.update(/*speed=*/3.0, dt)) ++steps;
  EXPECT_NEAR((steps + 1) * dt, ProjectileRestMonitor::kLifetime, 2 * dt);
  // And start() is a fresh ball.
  monitor.start();
  EXPECT_FALSE(monitor.update(/*speed=*/3.0, dt));
}

/*==================================== the ball is not the ground =======================================*/

TEST(GroundTruthContactMask, ABallStrikingAFootIsNotThatFootBeingPlanted) {
  // The mask feeds the RobotState's contact flags and the cheater_sim estimator, so a ball against a SWING foot would
  // tell the MPC that foot is planted.
  const std::string path = writeTempFile("projectile_contact.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  scene.arm();
  strikeTheFoot(scene, ball);

  const std::vector<int> feet{scene.body("foot")};
  // The positive control first: without the exclusion the ball DOES read as ground contact. Without this assertion a
  // mask of zero would prove nothing, because a test in which the ball never touched the foot would also pass.
  ASSERT_EQ(groundTruthContactMask(scene.model, scene.data, feet, /*forceThreshold=*/1.0, /*ignoreBodyId=*/-1), 1u)
      << "the ball is not actually striking the foot";
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, feet, /*forceThreshold=*/1.0, ballBody), 0u)
      << "a contact with the thrown ball must not set a contact bit";
}

TEST(GroundTruthContactMask, ExcludingTheBallLeavesRealGroundContactAlone) {
  // The exclusion must be narrow: a foot on the floor still reads as planted while a ball is in the scene.
  const std::string path = writeTempFile("projectile_ground.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int baseQpos = scene.model->jnt_qposadr[scene.model->body_jntadr[scene.body("base")]];
  // The sole's underside sits at base z - 0.3 - 0.02; drop it just through the floor so the solver pushes back.
  scene.data->qpos[baseQpos + 2] = 0.3 + 0.02 - 0.002;
  mj_forward(scene.model, scene.data);
  const std::vector<int> feet{scene.body("foot")};
  EXPECT_EQ(groundTruthContactMask(scene.model, scene.data, feet, /*forceThreshold=*/1.0, scene.body("sim_projectile")), 1u);
}

TEST(GroundReaction, ABallStrikingTheRobotIsNotAGroundReaction) {
  // The ZMP marker is "the zero moment point of the physical ground reaction"; at 25 m/s a ball's contact force alone
  // exceeds the robot's weight and would throw the marker meters from the feet.
  const std::string path = writeTempFile("projectile_zmp.xml", kScene);
  const Projectile ball = dodgeball();
  const Scene scene(path, &ball);
  ASSERT_NE(scene.model, nullptr);
  const int ballBody = scene.body("sim_projectile");
  scene.arm();
  strikeTheFoot(scene, ball);  // the robot is held in the air, so the ball is the only contact

  const GroundReaction counted = groundReaction(scene.model, scene.data, scene.body("base"), /*minNormalForce=*/0.0, /*ignoreBodyId=*/-1);
  ASSERT_GT(std::abs(counted.force[2]), 1.0) << "the ball is not actually pressing on the foot";
  const GroundReaction excluded = groundReaction(scene.model, scene.data, scene.body("base"), /*minNormalForce=*/0.0, ballBody);
  EXPECT_NEAR(excluded.force[0], 0.0, 1e-12);
  EXPECT_NEAR(excluded.force[1], 0.0, 1e-12);
  EXPECT_NEAR(excluded.force[2], 0.0, 1e-12);
}

}  // namespace robot::mujoco_sim_interface
