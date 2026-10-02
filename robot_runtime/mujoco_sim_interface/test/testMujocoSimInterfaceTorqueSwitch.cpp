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

#include <cmath>
#include <memory>
#include <vector>

#include <mujoco/mujoco.h>

#include <robot_model/RobotJointAction.h>

#include "mujoco_sim_interface/MujocoSimInterface.h"

/*
 * The simulator's torque switch and its hand-overs with the control thread, headless, on the shipped Atlas scene,
 * stepped by the test itself. Switching the torques back on used to leave the action latched before they went off - in
 * WB_MPC, with high gains and the inverse dynamics' feedforward - in force until the next control cycle applied one;
 * and the switch wrote MuJoCo's dof_damping from the control thread while the physics thread stepped with it.
 */

namespace robot::mujoco_sim_interface {
namespace {

constexpr const char* kAtlasScene = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml";
constexpr const char* kAtlasUrdf = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

std::unique_ptr<MujocoSimInterface> makeSim() {
  MujocoSimConfig config;
  config.scenePath = kAtlasScene;
  config.headless = true;
  config.isGantryLocked = true;
  config.gantryHold = "weld_constraint";
  return std::make_unique<MujocoSimInterface>(config, kAtlasUrdf);
}

/** Applies a pure feedforward torque of `torque` on every joint, as the control thread would. */
void applyFeedforward(MujocoSimInterface& sim, double torque) {
  model::RobotJointAction& action = sim.getRobotJointAction();
  for (size_t joint = 0; joint < sim.getRobotDescription().getNumJoints(); ++joint) {
    if (!action.at(joint).has_value()) continue;
    model::JointAction& jointAction = *action.at(joint);
    jointAction.kp = 0.0;
    jointAction.kd = 0.0;
    jointAction.feed_forward_effort = torque;
  }
  sim.applyJointAction();
}

double largestControl(const MujocoSimInterface& sim) {
  double largest = 0.0;
  for (double control : sim.actuatorControlsForTesting()) largest = std::max(largest, std::abs(control));
  return largest;
}

std::vector<double> dofDamping(const MujocoSimInterface& sim) {
  const mjModel* model = sim.getModel();
  return std::vector<double>(model->dof_damping, model->dof_damping + model->nv);
}

TEST(SimTorqueSwitch, AnActionAppliedBeforeTheTorquesCameBackOnIsNeverExecuted) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim();
  ASSERT_GT(sim->getModel()->nu, 0);
  sim->enableTorques();
  applyFeedforward(*sim, /*torque=*/5.0);
  sim->simulationStep();
  EXPECT_NEAR(largestControl(*sim), 5.0, 1e-9) << "an action applied while the torques are on is executed";

  // The torques go off and the controller, with them, applies nothing more.
  sim->disableTorques();
  sim->simulationStep();
  EXPECT_EQ(largestControl(*sim), 0.0);

  // Back on: the latched action is that of before, and the actuators stay at zero until a new one is applied.
  sim->enableTorques();
  for (int step = 0; step < 20; ++step) {
    sim->simulationStep();
    EXPECT_EQ(largestControl(*sim), 0.0) << "step " << step << " executed the stale action";
  }
  applyFeedforward(*sim, /*torque=*/3.0);
  sim->simulationStep();
  EXPECT_NEAR(largestControl(*sim), 3.0, 1e-9) << "the action applied after the switch is executed";
}

TEST(SimTorqueSwitch, TheDampingFollowsTheSwitchOnThePhysicsThreadOnly) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim();
  sim->simulationStep();
  const std::vector<double> ragdoll = dofDamping(*sim);

  // The switch of another thread is an atomic: the model is written by the next step, on the thread that steps it.
  sim->enableTorques();
  EXPECT_EQ(dofDamping(*sim), ragdoll) << "enableTorques() wrote the model outside a step";
  sim->simulationStep();
  const std::vector<double> active = dofDamping(*sim);
  EXPECT_NE(active, ragdoll) << "the active damping is restored at the next step";

  sim->disableTorques();
  EXPECT_EQ(dofDamping(*sim), active) << "disableTorques() wrote the model outside a step";
  sim->simulationStep();
  EXPECT_EQ(dofDamping(*sim), ragdoll) << "the ragdoll damping is back at the next step";
}

TEST(SimTorqueSwitch, TheFeetsForcesComeFromThePublishedStep) {
  // The Atlas scene has no foot force sensor: zero, read from the buffer the physics step fills, not from mjData.
  const std::unique_ptr<MujocoSimInterface> sim = makeSim();
  vector3_t left = vector3_t::Constant(1.0);
  vector3_t right = vector3_t::Constant(1.0);
  sim->simulationStep();
  sim->takeMeasuredFootForces(left, right);
  EXPECT_TRUE(left.isZero());
  EXPECT_TRUE(right.isZero());
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
