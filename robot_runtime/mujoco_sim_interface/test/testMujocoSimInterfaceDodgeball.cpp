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

#include <array>
#include <cmath>
#include <memory>
#include <string>

#include <mujoco/mujoco.h>

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/Projectile.h"

/*
 * The dodgeball through the real simulator, headless, on the shipped Atlas scene: the glue between Projectile's pure
 * functions and the simulation loop, which is where the bugs that no unit test could see lived - a ball id that was
 * never resolved, a damping loop that still damped the ball, a mass that reached one throw path and not the other.
 */

namespace robot::mujoco_sim_interface {
namespace {

constexpr const char* kAtlasScene = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml";
constexpr const char* kAtlasUrdf = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
constexpr double kGravity = 9.81;

std::unique_ptr<MujocoSimInterface> makeSim(bool gantryLocked) {
  MujocoSimConfig config;
  config.scenePath = kAtlasScene;
  config.headless = true;
  config.projectile = "dodgeball";
  config.isGantryLocked = gantryLocked;
  config.gantryHold = "weld_constraint";
  // A render snapshot after every step, so readLatestMjState shows the state of the step just taken.
  config.renderFrequencyHz = 1.0e6;
  return std::make_unique<MujocoSimInterface>(config, kAtlasUrdf);
}

int firstFreeBody(const mjModel* model) {
  for (int body = 1; body < model->nbody; ++body) {
    if (model->body_jntnum[body] > 0 && model->jnt_type[model->body_jntadr[body]] == mjJNT_FREE) return body;
  }
  return -1;
}

/** The simulator's state after its last step, and where the ball and the base are in it. */
struct Snapshot {
  explicit Snapshot(const MujocoSimInterface& sim) : state(sim.getModel()), model(sim.getModel()) { sim.readLatestMjState(state); }

  std::array<double, 3> ball() const {
    const int qpos = model->jnt_qposadr[model->body_jntadr[ballBody()]];
    return {{state.data->qpos[qpos], state.data->qpos[qpos + 1], state.data->qpos[qpos + 2]}};
  }
  std::array<double, 3> base() const {
    const int body = firstFreeBody(model);
    return {{state.data->xpos[3 * body], state.data->xpos[3 * body + 1], state.data->xpos[3 * body + 2]}};
  }
  int ballBody() const { return mj_name2id(model, mjOBJ_BODY, "sim_projectile"); }

  MjState state;
  const mjModel* model;
};

bool isParked(const MujocoSimInterface& sim) {
  const Snapshot snapshot(sim);
  const std::array<double, 3> ball = snapshot.ball();
  const int ballGeom = sim.getModel()->body_geomadr[sim.projectileBodyId()];
  return std::abs(ball[0] - kProjectileParkPosition[0]) < 1e-6 && std::abs(ball[1] - kProjectileParkPosition[1]) < 1e-6 &&
         std::abs(ball[2] - kProjectileParkPosition[2]) < 1e-6 && sim.getModel()->geom_contype[ballGeom] == 0;
}

bool isArmed(const MujocoSimInterface& sim) {
  return sim.getModel()->geom_contype[sim.getModel()->body_geomadr[sim.projectileBodyId()]] == 1;
}

/** A throw from `offset` (base yaw frame) that arrives at the base after `flightTime`, as dodgeball.py aims one. */
MujocoSimInterface::DodgeballThrow aimedThrow(const std::array<double, 3>& offset, double flightTime, double mass) {
  MujocoSimInterface::DodgeballThrow command;
  for (int axis = 0; axis < 3; ++axis) {
    command.spawnOffset[axis] = offset[axis];
    command.launchVelocity[axis] = -offset[axis] / flightTime;
  }
  command.launchVelocity[2] += 0.5 * kGravity * flightTime;
  command.flightTime = flightTime;
  command.mass = mass;
  return command;
}

void step(MujocoSimInterface& sim, int steps) {
  for (int i = 0; i < steps; ++i) sim.simulationStep();
}

int stepsFor(const MujocoSimInterface& sim, double seconds) {
  return static_cast<int>(std::ceil(seconds / sim.getModel()->opt.timestep));
}

}  // namespace

TEST(SimDodgeball, TheBallIsFoundParkedAndUndampedAtStartUp) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(true);
  const mjModel* model = sim->getModel();
  ASSERT_GE(sim->projectileBodyId(), 0) << "simProjectile named a ball, and the simulator did not find it";
  // The simulator starts in the zero-torque ragdoll mode, whose damping must reach the robot and not the ball.
  ASSERT_TRUE(sim->isZeroTorqueMode());
  for (int dof = 6; dof < model->nv; ++dof) {
    if (isProjectileDof(model, sim->projectileBodyId(), dof)) {
      EXPECT_EQ(model->dof_damping[dof], 0.0) << "the ball's dof " << dof << " is damped at start-up";
    } else {
      EXPECT_GT(model->dof_damping[dof], 0.0) << "robot dof " << dof << " lost its ragdoll damping";
    }
  }
  step(*sim, /*steps=*/1000);
  EXPECT_TRUE(isParked(*sim)) << "the parked ball moved";
}

TEST(SimDodgeball, TorqueTogglesNeverDampTheBall) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(true);
  const mjModel* model = sim->getModel();
  const int ball = sim->projectileBodyId();
  for (const bool torques : {true, false, true}) {
    if (torques) {
      sim->enableTorques();
    } else {
      sim->disableTorques();
    }
    for (int dof = 6; dof < model->nv; ++dof) {
      if (isProjectileDof(model, ball, dof)) EXPECT_EQ(model->dof_damping[dof], 0.0) << "torques " << torques << " dof " << dof;
    }
  }
}

TEST(SimDodgeball, AThrowFliesTheRealBallAtTheSlidersMassToTheBase) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(true);
  step(*sim, /*steps=*/10);
  const std::array<double, 3> base = Snapshot(*sim).base();
  const std::array<double, 3> offset{{3.0, 0.0, 0.5}};
  const double flight = 0.4;
  sim->throwDodgeball(aimedThrow(offset, flight, /*mass=*/2.0));
  step(*sim, /*steps=*/1);

  EXPECT_TRUE(isArmed(*sim));
  EXPECT_NEAR(sim->getModel()->body_mass[sim->projectileBodyId()], 2.0, 1e-12) << "the slider's mass did not reach the ball";
  const std::array<double, 3> start = Snapshot(*sim).ball();
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_NEAR(start[axis], base[axis] + offset[axis], 0.05) << "axis " << axis << ": the ball did not start at the spawn point";
  }

  // It is aimed at the base, so its center should pass close to it - unless an arm is in the way, which on Atlas held
  // on the gantry it is not from straight ahead.
  double closest = 1e9;
  for (int i = 0; i < stepsFor(*sim, flight + 0.1); ++i) {
    sim->simulationStep();
    const Snapshot snapshot(*sim);
    const std::array<double, 3> ball = snapshot.ball();
    const std::array<double, 3> now = snapshot.base();
    closest = std::min(closest, std::hypot(ball[0] - now[0], ball[1] - now[1], ball[2] - now[2]));
  }
  EXPECT_LT(closest, 0.5) << "the ball never came near the base it was aimed at";
}

TEST(SimDodgeball, AHandPublishedMassOutsideTheSliderRangeIsClamped) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(true);
  sim->throwDodgeball(aimedThrow({{3.0, 0.0, 0.5}}, /*flightTime=*/0.4, /*mass=*/50.0));
  step(*sim, /*steps=*/1);
  EXPECT_NEAR(sim->getModel()->body_mass[sim->projectileBodyId()], kMaxProjectileMass, 1e-12);
  sim->throwDodgeball(aimedThrow({{3.0, 0.0, 0.5}}, /*flightTime=*/0.4, /*mass=*/0.001));
  step(*sim, /*steps=*/1);
  EXPECT_NEAR(sim->getModel()->body_mass[sim->projectileBodyId()], kMinProjectileMass, 1e-12);
}

TEST(SimDodgeball, ASpawnPointUnderTheFloorIsLiftedAlongTheBallsOwnPath) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(true);
  step(*sim, /*steps=*/10);
  const double baseHeight = Snapshot(*sim).base()[2];
  // Far enough below the base to start under the floor, as a negative elevation from a distance does.
  const std::array<double, 3> offset{{2.5, 0.0, -(baseHeight + 0.5)}};
  sim->throwDodgeball(aimedThrow(offset, /*flightTime=*/0.6, /*mass=*/0.45));
  step(*sim, /*steps=*/1);
  ASSERT_TRUE(isArmed(*sim)) << "the throw was refused rather than lifted";
  EXPECT_GT(Snapshot(*sim).ball()[2], 0.108) << "the ball was left inside the floor, which would fire it out";
}

TEST(SimDodgeball, CatchingTheRobotOnTheGantryCancelsTheThrow) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(false);
  sim->throwDodgeball(aimedThrow({{3.0, 0.0, 0.5}}, /*flightTime=*/0.4, /*mass=*/0.45));
  step(*sim, /*steps=*/20);
  ASSERT_TRUE(isArmed(*sim));
  sim->lockGantry();  // what the fall catch and the operator's LOCK_GANTRY both do
  step(*sim, /*steps=*/1);
  EXPECT_TRUE(isParked(*sim)) << "a ball in play kept flying at the robot that had just been caught";
}

TEST(SimDodgeball, LockingAnAlreadyLockedGantryDoesNotEatAThrow) {
  // Only the unlocked-to-locked transition is a catch; a throw at a robot already on the gantry is deliberate.
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(true);
  sim->throwDodgeball(aimedThrow({{3.0, 0.0, 0.5}}, /*flightTime=*/0.4, /*mass=*/0.45));
  step(*sim, /*steps=*/5);
  sim->lockGantry();
  step(*sim, /*steps=*/1);
  EXPECT_TRUE(isArmed(*sim));
}

TEST(SimDodgeball, AResetParksTheBall) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(true);
  sim->throwDodgeball(aimedThrow({{3.0, 0.0, 0.5}}, /*flightTime=*/0.4, /*mass=*/0.45));
  step(*sim, /*steps=*/20);
  ASSERT_TRUE(isArmed(*sim));
  sim->reset();
  step(*sim, /*steps=*/1);
  EXPECT_TRUE(isParked(*sim));
}

TEST(SimDodgeball, AFinishedBallIsParkedAgain) {
  // Whether it comes to rest or runs out its lifetime, a thrown ball must not be left in play for the robot to trip
  // over minutes later.
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(true);
  sim->throwDodgeball(aimedThrow({{3.0, 0.0, 0.5}}, /*flightTime=*/0.4, /*mass=*/0.45));
  step(*sim, /*steps=*/1);
  ASSERT_TRUE(isArmed(*sim));
  step(*sim, stepsFor(*sim, ProjectileRestMonitor::kLifetime + 0.5));
  EXPECT_TRUE(isParked(*sim));
}

}  // namespace robot::mujoco_sim_interface
