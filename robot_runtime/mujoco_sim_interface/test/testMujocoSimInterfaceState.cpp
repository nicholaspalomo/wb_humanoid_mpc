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

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/Projectile.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"

/*
 * What the simulator hands the controller and takes from it, against MuJoCo's own physics: the free joint's state in the
 * robot state's frames, the scene's joints and actuators at their own addresses whatever their order, and the impact of
 * the dodgeball when no ball is compiled into the scene. Headless, on minimal scenes in zero gravity, stepped by the test.
 */

namespace robot::mujoco_sim_interface {
namespace {

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

/**
 * The robot in zero gravity, with nothing to touch. Between its two joints the scene has one the robot description does
 * not know, with the first actuator of the scene on it, so neither the description's joints nor its actuators are at
 * the addresses of a compacted list; and the base's center of mass is off its frame's origin.
 */
constexpr char kScene[] = R"(
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.1" pos="0.04 -0.02 0.03" mass="4"/>
      <body name="arm" pos="0 0 -0.2">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.5"/>
        <body name="elbow" pos="0.3 0 0">
          <joint name="passive" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.03" mass="0.2"/>
          <body name="hand" pos="0.1 0 0">
            <joint name="wrist" type="hinge" axis="0 1 0"/>
            <geom type="sphere" size="0.03" mass="0.3"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="passive_motor" joint="passive"/>
    <motor name="hinge_motor" joint="hinge"/>
    <motor name="wrist_motor" joint="wrist"/>
  </actuator>
</mujoco>
)";

constexpr double kTimestep = 0.0005;

std::string writeTempFile(const std::string& name, const char* absl_nonnull content) {
  const std::string path = testing::TempDir() + "/" + name;
  std::ofstream(path) << content;
  return path;
}

/** The simulator of kScene, without a gantry, the ragdoll damping or a projectile, publishing every step to the viewer. */
std::unique_ptr<MujocoSimInterface> makeSim(std::optional<model::RobotState> initialState = std::nullopt,
                                            const char* absl_nonnull scene = kScene) {
  MujocoSimConfig config;
  config.scenePath = writeTempFile("state_scene.xml", scene);
  config.viewer.names = {kNoViewerName};
  config.enableGantry = false;
  config.gantryHold = "kinematic_teleport";
  config.dt = kTimestep;
  config.renderFrequencyHz = 1.0 / kTimestep;
  config.ragdollJointDamping = 0.0;
  config.initialState = std::move(initialState);
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim =
      MujocoSimInterface::Create(config, writeTempFile("state_robot.urdf", kHingeUrdf));
  EXPECT_TRUE(sim.ok()) << sim.status();
  return sim.ok() ? *std::move(sim) : nullptr;
}

model::RobotState publishedState(MujocoSimInterface& sim) {
  sim.updateInterfaceStateFromRobot();
  return sim.getRobotState();
}

/** The latest snapshot's state with every derived quantity of it, the subtree velocities included, computed. */
struct ForwardState {
  MjDataPtr data;
};

ForwardState forwardSnapshot(const MujocoSimInterface& sim) {
  MjState snapshot(sim.getModel());
  sim.readLatestMjState(snapshot);
  absl::StatusOr<MjDataPtr> data = makeMjData(sim.getModel());
  EXPECT_TRUE(data.ok()) << data.status();
  mj_copyData(data->get(), sim.getModel(), snapshot.data);
  mj_forward(sim.getModel(), data->get());
  mj_subtreeVel(sim.getModel(), data->get());
  return ForwardState{.data = *std::move(data)};
}

// ---------------------------------------- the free joint ----------------------------------------------

// MuJoCo's free joint holds the quaternion w first, the linear velocity of the body frame's origin in the WORLD frame and
// the angular velocity in the BODY frame; the robot state holds the orientation, and both velocities in the base frame.
// Pinned against MuJoCo's integration of them, at an attitude far from the identity: a convention swapped anywhere
// (x-first quaternion, a world-frame angular velocity, an unrotated linear one) is an error of the order of the speeds.
TEST(SimFreeJoint, TheReportedBaseVelocitiesAreTheOnesMujocoIntegrates) {
  std::mt19937 generator(/*seed=*/3);
  std::normal_distribution<double> normal(/*mean=*/0.0, /*stddev=*/1.0);
  for (int sample = 0; sample < 5; ++sample) {
    const std::unique_ptr<MujocoSimInterface> probe = makeSim();
    ASSERT_NE(probe, nullptr);
    model::RobotState initial = publishedState(*probe);
    const Eigen::Quaterniond attitude =
        Eigen::Quaterniond(normal(generator), normal(generator), normal(generator), normal(generator)).normalized();
    initial.setRootRotationLocalToWorldFrame(attitude);
    initial.setRootLinearVelocityInLocalFrame(vector3_t(normal(generator), normal(generator), normal(generator)));
    initial.setRootAngularVelocityInLocalFrame(vector3_t(normal(generator), normal(generator), normal(generator)));
    const std::unique_ptr<MujocoSimInterface> sim = makeSim(initial);
    ASSERT_NE(sim, nullptr);

    // The state the simulator was given is the state it reports.
    const model::RobotState start = publishedState(*sim);
    EXPECT_TRUE(start.getRootRotationLocalToWorldFrame().isApprox(attitude, 1.0e-12));
    EXPECT_TRUE(start.getRootLinearVelocityInLocalFrame().isApprox(initial.getRootLinearVelocityInLocalFrame(), 1.0e-12));
    EXPECT_TRUE(start.getRootAngularVelocityInLocalFrame().isApprox(initial.getRootAngularVelocityInLocalFrame(), 1.0e-12));

    // And MuJoCo moves the base by them: the position by the world-frame linear velocity, the attitude by the body-frame
    // angular velocity, q1 = q0 exp(omega dt / 2), to first order in the one step.
    sim->simulationStep();
    const model::RobotState stepped = publishedState(*sim);
    const vector3_t positionRate = (stepped.getRootPositionInWorldFrame() - start.getRootPositionInWorldFrame()) / kTimestep;
    const vector3_t linearWorld = stepped.getRootRotationLocalToWorldFrame() * stepped.getRootLinearVelocityInLocalFrame();
    EXPECT_LT((positionRate - linearWorld).norm(), 1.0e-9) << "sample " << sample << ": MuJoCo integrates qpos with the new qvel";
    const Eigen::Quaterniond relative = start.getRootRotationLocalToWorldFrame().conjugate() * stepped.getRootRotationLocalToWorldFrame();
    const Eigen::AngleAxisd rotation(relative);
    const vector3_t bodyRate = rotation.angle() * rotation.axis() / kTimestep;
    EXPECT_LT((bodyRate - stepped.getRootAngularVelocityInLocalFrame()).norm(), 1.0e-6 * (1.0 + bodyRate.norm())) << "sample " << sample;
  }
}

// The base's accelerometer (RobotState::getImuLinearAccelerationInLocalFrame()): the specific force R^T (a - g) of the
// base frame's origin, in the base frame. In free fall, with no joint torque, every body falls with g and the
// accelerometer reads zero whatever the attitude; under a joint torque the base accelerates, and the reading is the
// change of MuJoCo's world-frame base velocity over the step, minus gravity, rotated into the base frame.
TEST(SimFreeJoint, TheAccelerometerReadsTheSpecificForceOfTheBase) {
  std::string fallingScene(kScene);
  const std::string zeroGravity = R"(<option gravity="0 0 0"/>)";
  fallingScene.replace(fallingScene.find(zeroGravity), zeroGravity.size(), R"(<option gravity="0 0 -9.81"/>)");
  const vector3_t gravity(0.0, 0.0, -9.81);
  std::mt19937 generator(/*seed=*/5);
  std::normal_distribution<double> normal(/*mean=*/0.0, /*stddev=*/1.0);
  for (int sample = 0; sample < 3; ++sample) {
    const std::unique_ptr<MujocoSimInterface> probe = makeSim(/*initialState=*/std::nullopt, fallingScene.c_str());
    ASSERT_NE(probe, nullptr);
    model::RobotState initial = publishedState(*probe);
    initial.setRootRotationLocalToWorldFrame(
        Eigen::Quaterniond(normal(generator), normal(generator), normal(generator), normal(generator)).normalized());
    initial.setRootLinearVelocityInLocalFrame(vector3_t(normal(generator), normal(generator), normal(generator)));
    const std::unique_ptr<MujocoSimInterface> sim = makeSim(initial, fallingScene.c_str());
    ASSERT_NE(sim, nullptr);
    for (int step = 0; step < 5; ++step) {
      sim->simulationStep();
      EXPECT_LT(publishedState(*sim).getImuLinearAccelerationInLocalFrame().norm(), 1.0e-9) << "sample " << sample << ", free fall";
    }

    // A joint torque accelerates the base against the arm.
    sim->enableTorques();
    model::RobotJointAction& action = sim->getRobotJointAction();
    const robot::model::RobotDescription& description = sim->getRobotDescription();
    for (const char* absl_nonnull name : {"hinge", "wrist"}) {
      std::optional<model::JointAction>& slot = action[description.getJointIndex(name)];
      if (!slot.has_value()) {
        FAIL() << "no action slot for " << name;
      }
      model::JointAction& jointAction = *slot;
      jointAction.kp = 0.0;
      jointAction.kd = 0.0;
      jointAction.feed_forward_effort = 5.0 * normal(generator);
    }
    sim->applyJointAction();
    const model::RobotState before = publishedState(*sim);
    sim->simulationStep();
    const model::RobotState after = publishedState(*sim);
    const vector3_t velocityBefore = before.getRootRotationLocalToWorldFrame() * before.getRootLinearVelocityInLocalFrame();
    const vector3_t velocityAfter = after.getRootRotationLocalToWorldFrame() * after.getRootLinearVelocityInLocalFrame();
    const vector3_t expected =
        after.getRootRotationLocalToWorldFrame().inverse() * ((velocityAfter - velocityBefore) / kTimestep - gravity);
    const vector3_t reading = after.getImuLinearAccelerationInLocalFrame();
    EXPECT_GT((reading).norm(), 1.0e-3) << "sample " << sample << ": the torque moves the base";
    EXPECT_LT((reading - expected).norm(), 1.0e-6 * (1.0 + expected.norm())) << "sample " << sample;
  }
}

// ------------------------------------- joints and actuators ---------------------------------------------

TEST(SimJointMap, EveryJointAndActuatorIsAtItsOwnAddressWhateverTheOrderOfTheScene) {
  const std::unique_ptr<MujocoSimInterface> probe = makeSim();
  ASSERT_NE(probe, nullptr);
  const robot::model::RobotDescription& description = probe->getRobotDescription();
  model::RobotState initial = publishedState(*probe);
  initial.setJointPosition(description.getJointIndex("hinge"), /*jointPosition=*/0.3);
  initial.setJointPosition(description.getJointIndex("wrist"), /*jointPosition=*/-0.2);
  initial.setJointVelocity(description.getJointIndex("wrist"), /*jointVelocity=*/0.7);
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(initial);
  ASSERT_NE(sim, nullptr);
  const mjModel* absl_nonnull model = sim->getModel();
  const int hinge = mj_name2id(model, mjOBJ_JOINT, "hinge");
  const int passive = mj_name2id(model, mjOBJ_JOINT, "passive");
  const int wrist = mj_name2id(model, mjOBJ_JOINT, "wrist");

  // Written at the joints' own qpos and dof addresses, which skip the passive joint, and nothing written to it.
  const ForwardState state = forwardSnapshot(*sim);
  EXPECT_EQ(state.data->qpos[model->jnt_qposadr[hinge]], 0.3);
  EXPECT_EQ(state.data->qpos[model->jnt_qposadr[wrist]], -0.2);
  EXPECT_EQ(state.data->qvel[model->jnt_dofadr[wrist]], 0.7);
  EXPECT_EQ(state.data->qpos[model->jnt_qposadr[passive]], 0.0);

  // Each joint's torque goes to its own actuator, and the passive joint's actuator is commanded nothing.
  sim->enableTorques();
  model::RobotJointAction& action = sim->getRobotJointAction();
  for (const char* absl_nonnull name : {"hinge", "wrist"}) {
    std::optional<model::JointAction>& slot = action[description.getJointIndex(name)];
    if (!slot.has_value()) {
      FAIL() << "no action slot for " << name;
    }
    model::JointAction& jointAction = *slot;
    jointAction.kp = 0.0;
    jointAction.kd = 0.0;
  }
  std::optional<model::JointAction>& hingeAction = action[description.getJointIndex("hinge")];
  std::optional<model::JointAction>& wristAction = action[description.getJointIndex("wrist")];
  if (!hingeAction.has_value() || !wristAction.has_value()) {
    FAIL() << "no action slot for the hinge or the wrist";
  }
  hingeAction->feed_forward_effort = 2.0;
  wristAction->feed_forward_effort = 3.0;
  sim->applyJointAction();
  sim->simulationStep();
  const std::vector<double> controls = sim->actuatorControlsForTesting();
  ASSERT_EQ(controls.size(), 3u);
  EXPECT_EQ(controls[mj_name2id(model, mjOBJ_ACTUATOR, "passive_motor")], 0.0);
  EXPECT_NEAR(controls[mj_name2id(model, mjOBJ_ACTUATOR, "hinge_motor")], 2.0, 1.0e-12);
  EXPECT_NEAR(controls[mj_name2id(model, mjOBJ_ACTUATOR, "wrist_motor")], 3.0, 1.0e-12);
}

// ------------------------------- the dodgeball without a ball ------------------------------------------

/** The robot's linear momentum and its angular momentum about the world origin. */
struct Momentum {
  Eigen::Vector3d linear = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular = Eigen::Vector3d::Zero();
};

Momentum robotMomentum(const mjModel* absl_nonnull model, const mjData& data) {
  constexpr int kRoot = 1;
  const double mass = model->body_subtreemass[kRoot];
  const Eigen::Vector3d com(data.subtree_com[3 * kRoot], data.subtree_com[3 * kRoot + 1], data.subtree_com[3 * kRoot + 2]);
  const Eigen::Vector3d comVelocity(data.subtree_linvel[3 * kRoot], data.subtree_linvel[3 * kRoot + 1], data.subtree_linvel[3 * kRoot + 2]);
  const Eigen::Vector3d spin(data.subtree_angmom[3 * kRoot], data.subtree_angmom[3 * kRoot + 1], data.subtree_angmom[3 * kRoot + 2]);
  Momentum momentum;
  momentum.linear = mass * comVelocity;
  momentum.angular = spin + com.cross(momentum.linear);
  return momentum;
}

// The ball hits the base's origin head on. Its impulse J must (1) conserve momentum and reverse the relative normal
// velocity at the restitution, measured on the robot's actual response, which holds only for the right effective mass;
// (2) be the whole change of the robot's linear momentum; and (3) change its angular momentum by the moment of J at the
// base's origin, p x J, which the force at the center of mass alone would miss.
TEST(SimFallbackImpulse, IsTheImpactOfADodgeballOnTheRobotsEffectiveMassAtTheBase) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim();
  ASSERT_NE(sim, nullptr);
  ASSERT_EQ(sim->projectileBodyId(), -1);
  const mjModel* absl_nonnull model = sim->getModel();
  const Projectile dodgeball = *projectileFromName("dodgeball");
  sim->simulationStep();  // the robot at rest, its derived quantities computed

  const ForwardState before = forwardSnapshot(*sim);
  const Eigen::Vector3d point(before.data->xpos[3], before.data->xpos[4], before.data->xpos[5]);
  const Momentum initial = robotMomentum(model, *before.data);
  ASSERT_LT(initial.linear.norm(), 1.0e-12);

  MujocoSimInterface::DodgeballThrow throwCommand;
  const Eigen::Vector3d velocity(-3.0, 1.5, 1.0);  // in the yaw frame of an unyawed base: the world frame
  for (int axis = 0; axis < 3; ++axis) throwCommand.launchVelocity[axis] = velocity[axis];
  throwCommand.flightTime = 0.0;  // arrives on this step, so gravity takes nothing off on the way
  throwCommand.mass = 1.2;
  sim->throwDodgeball(throwCommand);
  sim->simulationStep();

  const ForwardState after = forwardSnapshot(*sim);
  MjState snapshot(model);
  sim->readLatestMjState(snapshot);
  const Eigen::Vector3d impulse(snapshot.push.impulseWorld[0], snapshot.push.impulseWorld[1], snapshot.push.impulseWorld[2]);
  const Eigen::Vector3d normal = velocity.normalized();
  ASSERT_GT(impulse.norm(), 0.0);
  EXPECT_LT((impulse.normalized() - normal).norm(), 1.0e-12) << "a head-on impact pushes along the ball's velocity";

  // (1) The restitution law on the measured velocity of the struck point.
  const model::RobotState state = publishedState(*sim);
  const double pointSpeed = (state.getRootRotationLocalToWorldFrame() * state.getRootLinearVelocityInLocalFrame()).dot(normal);
  const double ballSpeed = velocity.norm() - impulse.norm() / throwCommand.mass;
  EXPECT_NEAR(pointSpeed - ballSpeed, dodgeball.restitution * velocity.norm(), 1.0e-3 * velocity.norm());
  // (2) and (3) The robot's momenta.
  const Momentum final = robotMomentum(model, *after.data);
  // To first order in the step: the momenta are read at the configuration after it, the impulse acted at the one before.
  EXPECT_LT((final.linear - impulse).norm(), 1.0e-3 * impulse.norm());
  const Eigen::Vector3d moment = point.cross(impulse);
  EXPECT_LT((final.angular - moment).norm(), 1.0e-3 * moment.norm()) << "the moment of the push at the base's origin";
  // The base's center of mass is off its origin, so the moment is not that of a push at the center of mass.
  const Eigen::Vector3d baseCom(before.data->xipos[3], before.data->xipos[4], before.data->xipos[5]);
  EXPECT_GT((moment - baseCom.cross(impulse)).norm(), 0.01 * moment.norm());
}

TEST(ProjectileImpactImpulse, ConservesMomentumAndReversesTheRelativeVelocityAtTheRestitution) {
  // Two bodies on a line: a ball of mass m at speed v against a body of mass M at rest.
  for (const double ballMass : {0.1, 0.45, 5.0}) {
    for (const double bodyMass : {0.2, 30.0, 1.0e6}) {
      for (const double restitution : {0.0, 0.5, 0.8, 1.0}) {
        const double speed = 4.0;
        const double impulse = projectileImpactImpulse(ballMass, bodyMass, restitution, speed);
        const double ballAfter = speed - impulse / ballMass;
        const double bodyAfter = impulse / bodyMass;
        EXPECT_NEAR(ballMass * ballAfter + bodyMass * bodyAfter, ballMass * speed, 1.0e-9 * ballMass * speed);
        EXPECT_NEAR(bodyAfter - ballAfter, restitution * speed, 1.0e-9 * speed);
      }
    }
  }
  const double infinite = std::numeric_limits<double>::infinity();
  EXPECT_DOUBLE_EQ(projectileImpactImpulse(/*ballMass=*/0.45, infinite, /*restitution=*/0.8, /*approachSpeed=*/4.0), 1.8 * 0.45 * 4.0)
      << "a rigid wall takes (1 + e) m v";
  EXPECT_EQ(projectileImpactImpulse(/*ballMass=*/0.45, /*effectiveMass=*/30.0, /*restitution=*/0.8, /*approachSpeed=*/-1.0), 0.0)
      << "nothing when the two separate";
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
