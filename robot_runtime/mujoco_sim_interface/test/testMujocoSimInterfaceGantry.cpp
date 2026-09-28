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
#include <cstdint>
#include <memory>

#include <mujoco/mujoco.h>

#include "mujoco_sim_interface/MujocoSimInterface.h"

/*
 * The virtual gantry and the simulator's own resets, headless, on the shipped Atlas scene. A robot caught on the gantry
 * after walking away from the origin used to be welded back to the scene's anchor at the origin; the pull made the step
 * numerically unstable, MuJoCo's automatic reset then rewound the clock, and every later MPC solve failed because the
 * controller's plans were timestamped hundreds of seconds in the future.
 */

namespace robot::mujoco_sim_interface {
namespace {

constexpr const char* kAtlasScene = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml";
constexpr const char* kAtlasUrdf = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

std::unique_ptr<MujocoSimInterface> makeSim(bool gantryLocked) {
  MujocoSimConfig config;
  config.scenePath = kAtlasScene;
  config.headless = true;
  config.isGantryLocked = gantryLocked;
  config.gantryHold = "weld_constraint";
  // A render snapshot after every step, so readLatestMjState shows the state of the step just taken.
  config.renderFrequencyHz = 1.0e6;
  return std::make_unique<MujocoSimInterface>(config, kAtlasUrdf);
}

int baseBody(const mjModel* model) {
  for (int body = 1; body < model->nbody; ++body) {
    if (model->body_jntnum[body] > 0 && model->jnt_type[model->body_jntadr[body]] == mjJNT_FREE) return body;
  }
  return -1;
}

/** The base's position, heading and the clock after the simulator's last step. */
struct BaseSnapshot {
  double x;
  double y;
  double z;
  double yaw;
  double time;
  bool finite;
};

BaseSnapshot snapshot(const MujocoSimInterface& sim) {
  MjState state(sim.getModel());
  sim.readLatestMjState(state);
  const int body = baseBody(sim.getModel());
  const mjtNum* position = state.data->xpos + 3 * body;
  const mjtNum* quaternion = state.data->xquat + 4 * body;
  BaseSnapshot result;
  result.x = position[0];
  result.y = position[1];
  result.z = position[2];
  result.yaw = std::atan2(2.0 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                          1.0 - 2.0 * (quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]));
  result.time = state.data->time;
  result.finite = true;
  for (int i = 0; i < sim.getModel()->nq; ++i) result.finite = result.finite && std::isfinite(state.data->qpos[i]);
  return result;
}

/** A base pose at (x, y, z) with heading `yaw` and no roll or pitch, as setBaseStateForTesting takes it. */
std::array<double, 7> basePose(double x, double y, double z, double yaw) {
  return {x, y, z, std::cos(0.5 * yaw), 0.0, 0.0, std::sin(0.5 * yaw)};
}

constexpr std::array<double, 6> kAtRest{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

void step(MujocoSimInterface& sim, int steps) {
  for (int i = 0; i < steps; ++i) sim.simulationStep();
}

int stepsFor(const MujocoSimInterface& sim, double seconds) {
  return static_cast<int>(std::ceil(seconds / sim.getModel()->opt.timestep));
}

TEST(SimGantry, MuJoCosOwnAutomaticResetIsDisabled) {
  // The simulator recovers from a bad step itself, without rewinding the clock (see the header comment above).
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(true);
  EXPECT_NE(sim->getModel()->opt.disableflags & mjDSBL_AUTORESET, 0);
}

TEST(SimGantry, TheGantryHoldsTheRobotWhereItWasCaught) {
  // A robot caught 5 m and 3 m from the origin, turned by 1 rad: the gantry holds it there, at the gantry height and
  // with its heading, instead of welding it back to the scene's anchor above the origin facing +x.
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(false);
  const double height = sim->getGantryHeight();
  ASSERT_GT(height, 0.5);
  sim->setBaseStateForTesting(basePose(/*x=*/5.0, /*y=*/-3.0, height, /*yaw=*/1.0), kAtRest);
  step(*sim, /*steps=*/5);
  sim->lockGantry();
  step(*sim, stepsFor(*sim, /*seconds=*/2.0));

  const BaseSnapshot held = snapshot(*sim);
  ASSERT_TRUE(held.finite);
  EXPECT_NEAR(held.x, 5.0, 0.05);
  EXPECT_NEAR(held.y, -3.0, 0.05);
  EXPECT_NEAR(held.z, height, 0.05);
  EXPECT_NEAR(held.yaw, 1.0, 0.05);
  EXPECT_EQ(sim->resetEpoch(), 0u) << "holding the robot must not have needed a reset";
}

TEST(SimGantry, EachLockAnchorsAtWhereTheRobotIsThen) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(false);
  const double height = sim->getGantryHeight();
  sim->setBaseStateForTesting(basePose(/*x=*/1.0, /*y=*/1.0, height, /*yaw=*/0.0), kAtRest);
  sim->lockGantry();
  step(*sim, stepsFor(*sim, /*seconds=*/1.0));
  EXPECT_NEAR(snapshot(*sim).x, 1.0, 0.05);

  // Released, carried elsewhere, caught again: the second catch holds it at the second place, not the first.
  sim->unlockGantry();
  step(*sim, /*steps=*/1);
  sim->setBaseStateForTesting(basePose(/*x=*/-4.0, /*y=*/2.0, height, /*yaw=*/-0.5), kAtRest);
  sim->lockGantry();
  step(*sim, stepsFor(*sim, /*seconds=*/1.0));
  const BaseSnapshot held = snapshot(*sim);
  EXPECT_NEAR(held.x, -4.0, 0.05);
  EXPECT_NEAR(held.y, 2.0, 0.05);
  EXPECT_NEAR(held.yaw, -0.5, 0.05);
}

TEST(SimGantry, AnUnstableStepIsRecoveredWithoutRewindingTheClock) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(false);
  step(*sim, /*steps=*/200);
  const BaseSnapshot before = snapshot(*sim);
  ASSERT_GT(before.time, 0.0);
  ASSERT_EQ(sim->resetEpoch(), 0u);

  // A base velocity no integrator survives.
  sim->setBaseStateForTesting(basePose(before.x, before.y, before.z, /*yaw=*/0.0), {1.0e12, 0.0, 0.0, 0.0, 1.0e12, 0.0});
  step(*sim, /*steps=*/1);

  const BaseSnapshot after = snapshot(*sim);
  EXPECT_EQ(sim->resetEpoch(), 1u) << "the recovery must be reported to the control loop";
  EXPECT_TRUE(sim->isGantryLocked()) << "the recovered robot must be caught, as the tilt catch does";
  EXPECT_TRUE(after.finite);
  EXPECT_GE(after.time, before.time) << "the clock went backwards, from " << before.time << " to " << after.time;
  // Back in its initial state: over the origin at the spawn height.
  EXPECT_NEAR(after.x, 0.0, 0.05);
  EXPECT_NEAR(after.y, 0.0, 0.05);

  // And it keeps simulating normally from there.
  step(*sim, /*steps=*/200);
  EXPECT_EQ(sim->resetEpoch(), 1u);
  EXPECT_TRUE(snapshot(*sim).finite);
  EXPECT_GT(snapshot(*sim).time, after.time);
}

TEST(SimGantry, ABaseBelowTheFloorLimitIsResetCaughtAndCounted) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(false);
  step(*sim, /*steps=*/100);
  const double time = snapshot(*sim).time;
  sim->setBaseStateForTesting(basePose(2.0, 0.0, 0.1, 0.0), kAtRest);
  step(*sim, /*steps=*/1);
  EXPECT_EQ(sim->resetEpoch(), 1u);
  EXPECT_TRUE(sim->isGantryLocked());
  EXPECT_GE(snapshot(*sim).time, time);
  EXPECT_NEAR(snapshot(*sim).x, 0.0, 0.05);
}

TEST(SimGantry, AnExplicitResetKeepsTheClockAndCounts) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(true);
  step(*sim, /*steps=*/300);
  const double time = snapshot(*sim).time;
  sim->reset();
  step(*sim, /*steps=*/1);
  EXPECT_EQ(sim->resetEpoch(), 1u);
  EXPECT_GT(snapshot(*sim).time, time);
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
