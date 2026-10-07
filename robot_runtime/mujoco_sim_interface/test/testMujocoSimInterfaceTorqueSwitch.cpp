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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "robot_model/RobotJointAction.h"

/*
 * The simulator's torque switch and its hand-overs with the control thread, headless, on the shipped Atlas scene,
 * stepped by the test itself. Switching the torques back on used to leave the action latched before they went off - in
 * WB_MPC, with high gains and the inverse dynamics' feedforward - in force until the next control cycle applied one;
 * and the switch wrote MuJoCo's dof_damping from the control thread while the physics thread stepped with it.
 *
 * Also the mapping of the scene's actuators to the robot's joints, on a minimal scene of its own: an actuator without a
 * name used to stop the simulator's construction, because mj_id2name returns null for it.
 */

namespace robot::mujoco_sim_interface {
namespace {

constexpr char kAtlasScene[] = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml";
constexpr char kAtlasUrdf[] = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

/** A floating base with an arm on two hinges (the robot description needs at least two joints). */
constexpr char kHingeUrdf[] = R"(
<robot name="hinge_robot">
  <link name="base"/>
  <link name="arm"/>
  <link name="hand"/>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="wrist" type="revolute">
    <parent link="arm"/><child link="hand"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
)";

/** The same robot in MuJoCo, with an unnamed hinge motor and an unnamed arm body: mj_id2name returns null for both. */
constexpr char kUnnamedActuatorScene[] = R"(
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <body pos="0 0 -0.2">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.1"/>
        <body name="hand" pos="0.3 0 0">
          <joint name="wrist" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.03" mass="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge"/>
    <motor name="wrist_motor" joint="wrist"/>
  </actuator>
</mujoco>
)";

std::string writeTempFile(const std::string& name, const char* absl_nonnull content) {
  const std::string path = testing::TempDir() + "/" + name;
  std::ofstream(path) << content;
  return path;
}

std::unique_ptr<MujocoSimInterface> makeSim() {
  MujocoSimConfig config;
  config.scenePath = kAtlasScene;
  config.headless = true;
  config.isGantryLocked = true;
  config.gantryHold = "weld_constraint";
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim = MujocoSimInterface::Create(config, kAtlasUrdf);
  ABSL_CHECK_OK(sim);
  return *std::move(sim);
}

/** Applies a pure feedforward torque of `torque` on every joint, as the control thread would. */
void applyFeedforward(MujocoSimInterface& sim, double torque) {
  model::RobotJointAction& action = sim.getRobotJointAction();
  for (size_t joint = 0; joint < sim.getRobotDescription().getNumJoints(); ++joint) {
    std::optional<model::JointAction>& slot = action[joint];
    if (!slot.has_value()) continue;
    model::JointAction& jointAction = *slot;
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
  const mjModel* absl_nonnull model = sim.getModel();
  return std::vector<double>(model->dof_damping, model->dof_damping + model->nv);
}

TEST(SimTorqueSwitch, AnActionAppliedBeforeTheTorquesCameBackOnIsNeverExecuted) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim();
  ASSERT_GT(sim->getModel()->nu, 0);
  sim->enableTorques();
  applyFeedforward(*sim, /*torque=*/5.0);
  sim->simulationStep();
  EXPECT_NEAR(largestControl(*sim), 5.0, 1.0e-9) << "an action applied while the torques are on is executed";

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
  EXPECT_NEAR(largestControl(*sim), 3.0, 1.0e-9) << "the action applied after the switch is executed";
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

/** The hinge robot on kUnnamedActuatorScene, with one contact point on the unnamed arm body. */
absl::StatusOr<std::unique_ptr<MujocoSimInterface>> makeHingeSim() {
  MujocoSimConfig config;
  config.scenePath = writeTempFile("unnamed_actuator_scene.xml", kUnnamedActuatorScene);
  config.headless = true;
  config.verbose = true;  // logs the MuJoCo body of every resolved contact point, which here has no name
  config.enableGantry = false;
  config.gantryHold = "kinematic_teleport";  // the scene has no gantry weld
  config.contactFrameNames = {"arm_contact"};
  config.contactParentJointNames = {"hinge"};  // resolved to the unnamed body the hinge drives
  return MujocoSimInterface::Create(config, writeTempFile("unnamed_actuator_robot.urdf", kHingeUrdf));
}

TEST(SimActuators, AnUnnamedActuatorDrivesItsJointAndAnUnnamedContactBodyIsLogged) {
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> created = makeHingeSim();
  ASSERT_TRUE(created.ok()) << created.status();
  const std::unique_ptr<MujocoSimInterface> sim = *std::move(created);
  ASSERT_EQ(sim->getModel()->nu, 2);
  EXPECT_EQ(sim->getUnresolvedContactMask(), 0u) << "the contact point is resolved through its parent joint";

  sim->enableTorques();
  applyFeedforward(*sim, /*torque=*/2.0);
  sim->simulationStep();
  const std::vector<double> controls = sim->actuatorControlsForTesting();
  ASSERT_EQ(controls.size(), 2u);
  EXPECT_NEAR(controls[0], 2.0, 1.0e-9) << "the unnamed motor is commanded through the joint it drives";
  EXPECT_NEAR(controls[1], 2.0, 1.0e-9) << "the named one alike";
}

TEST(SimActuators, AJointWhoseActionWasResetIsCommandedNothing) {
  // RobotJointAction holds an action for every joint, but a controller can reset one. Its actuator then gets no torque,
  // and the physics thread carries on with the others.
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> created = makeHingeSim();
  ASSERT_TRUE(created.ok()) << created.status();
  const std::unique_ptr<MujocoSimInterface> sim = *std::move(created);
  sim->enableTorques();
  applyFeedforward(*sim, /*torque=*/2.0);
  sim->getRobotJointAction()[sim->getRobotDescription().getJointIndex("hinge")].reset();
  sim->applyJointAction();
  sim->simulationStep();
  const std::vector<double> controls = sim->actuatorControlsForTesting();
  ASSERT_EQ(controls.size(), 2u);
  EXPECT_EQ(controls[0], 0.0) << "the hinge motor has no action to execute";
  EXPECT_NEAR(controls[1], 2.0, 1.0e-9) << "the wrist motor executes its own";
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
