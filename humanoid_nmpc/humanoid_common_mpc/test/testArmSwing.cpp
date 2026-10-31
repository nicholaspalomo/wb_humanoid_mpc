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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/ArmSwing.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/arm_swing_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/*
 * The procedural arm swing (SwitchedModelReferenceManager::getDesiredState(),
 * ArmSwing.h) on every shipped robot that names its arm joints, built on the
 * robot's own URDF and task file without CppAD:
 *
 *  - the direction each arm joint turns in is the one that swings its arm
 * forward (a shoulder) or flexes it (an elbow), against a central difference of
 * the forward kinematics at the initial posture, independent of the
 *    derivation's joint axes; on DRC Atlas, whose v5 arm has no shoulder pitch,
 * that means the shoulder yaw and the elbow flexion joints, with opposite
 * senses on the two sides;
 *  - in left stance under a forward command the left arm's center of mass moves
 * forward and its elbow closes, and the right arm does the opposite - the
 * motion the swing is for, end to end through the reference;
 *  - on G1 and R1 the swing is bit for bit the one the reference manager
 * generated before its directions were derived;
 *  - the reference does not depend on the node's state;
 *  - the arm_swing block converts, and refuses an unknown generator, a negative
 * gain and a NaN.
 */

namespace ocs2::humanoid {
namespace {

using test::QuaternionLayoutRobotModel;

/** One robot's files and the MPC whose state layout it uses. Passive data. */
struct ShippedRobot {
  const char* absl_nonnull name;
  const char* absl_nonnull taskFile;
  const char* absl_nonnull referenceFile;
  const char* absl_nonnull urdfFile;
  StateInputLayout::Mpc mpc;
};

const std::vector<ShippedRobot>& shippedRobotsWithArms() {
  static const absl::NoDestructor<std::vector<ShippedRobot>> kRobots(
      std::vector<ShippedRobot>{
          {.name = "drc_atlas",
           .taskFile = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/"
                       "mpc/task.textproto",
           .referenceFile = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/"
                            "config/command/reference.textproto",
           .urdfFile =
               "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
           .mpc = StateInputLayout::Mpc::kCentroidal},
          {.name = "unitree_g1",
           .taskFile = "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/"
                       "task.textproto",
           .referenceFile = "robot_models/unitree_g1/g1_centroidal_mpc/config/"
                            "command/reference.textproto",
           .urdfFile =
               "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
           .mpc = StateInputLayout::Mpc::kCentroidal},
          {.name = "unitree_g1_wb",
           .taskFile =
               "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
           .referenceFile = "robot_models/unitree_g1/g1_wb_mpc/config/command/"
                            "reference.textproto",
           .urdfFile =
               "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
           .mpc = StateInputLayout::Mpc::kWholeBody},
          {.name = "unitree_r1",
           .taskFile = "robot_models/unitree_r1/unitree_r1_centroidal_mpc/"
                       "config/mpc/task.textproto",
           .referenceFile = "robot_models/unitree_r1/unitree_r1_centroidal_mpc/"
                            "config/command/reference.textproto",
           .urdfFile =
               "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf",
           .mpc = StateInputLayout::Mpc::kCentroidal},
      });
  return *kRobots;
}

/** A robot loaded for the arm swing: its settings, Pinocchio model, a robot
 * model of its layout, its initial state. */
struct LoadedRobot {
  mpc_config::TaskFile task;
  std::string referenceFile;
  std::unique_ptr<ModelSettings> settings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;
  std::unique_ptr<QuaternionLayoutRobotModel> model;
  vector_t initialState;
};

absl::StatusOr<LoadedRobot> load(const ShippedRobot& robot) {
  LoadedRobot loaded;
  ASSIGN_OR_RETURN(const std::string taskFile,
                   robot::resolveResourcePath(robot.taskFile));
  ASSIGN_OR_RETURN(loaded.referenceFile,
                   robot::resolveResourcePath(robot.referenceFile));
  ASSIGN_OR_RETURN(const std::string urdfFile,
                   robot::resolveResourcePath(robot.urdfFile));
  ASSIGN_OR_RETURN(loaded.task, loadTaskFile(taskFile));
  ASSIGN_OR_RETURN(ModelSettings settings,
                   ModelSettings::Create(loaded.task, urdfFile, "testArmSwing",
                                         /*verbose=*/false));
  loaded.settings = std::make_unique<ModelSettings>(std::move(settings));
  ASSIGN_OR_RETURN(
      PinocchioInterface pinocchioInterface,
      loadCustomPinocchioInterface(loaded.task, urdfFile, *loaded.settings));
  loaded.pinocchioInterface =
      std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));
  loaded.model =
      std::make_unique<QuaternionLayoutRobotModel>(*loaded.settings, robot.mpc);
  ASSIGN_OR_RETURN(
      const vector_t tuning,
      stateValuesFromConfig(loaded.task.initial_state,
                            stateInputLayout(*loaded.settings, robot.mpc),
                            "initial_state"));
  ASSIGN_OR_RETURN(loaded.initialState, loaded.model->stateFromTuning(tuning));
  return loaded;
}

/** [m] The center of mass of everything joint `jointName` carries, in the
 * world, at the configuration `q`. */
vector3_t carriedCenterOfMass(const PinocchioInterface& pinocchioInterface,
                              const vector_t& q,
                              const std::string& jointName) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data data(model);
  pinocchio::forwardKinematics(model, data, q);
  const pinocchio::JointIndex joint = model.getJointId(jointName);
  vector3_t weighted = vector3_t::Zero();
  scalar_t mass = 0.0;
  for (pinocchio::JointIndex body = joint;
       body < static_cast<pinocchio::JointIndex>(model.njoints); ++body) {
    // Every joint whose support passes through `joint`, found from the parents
    // rather than from model.subtrees.
    pinocchio::JointIndex ancestor = body;
    while (ancestor > joint) ancestor = model.parents[ancestor];
    if (ancestor != joint) continue;
    mass += model.inertias[body].mass();
    weighted += model.inertias[body].mass() *
                data.oMi[body].act(model.inertias[body].lever());
  }
  return vector3_t(weighted / mass);
}

/** [m] The origin of joint `jointName` in the world at `q`. */
vector3_t jointOrigin(const PinocchioInterface& pinocchioInterface,
                      const vector_t& q,
                      const std::string& jointName) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data data(model);
  pinocchio::forwardKinematics(model, data, q);
  return data.oMi[model.getJointId(jointName)].translation();
}

/** [rad] The elbow's interior angle at `q`: between the upper arm (elbow to
 * shoulder) and what the elbow carries. */
scalar_t elbowAngle(const PinocchioInterface& pinocchioInterface,
                    const vector_t& q,
                    const std::string& shoulder,
                    const std::string& elbow) {
  const vector3_t origin = jointOrigin(pinocchioInterface, q, elbow);
  const vector3_t upperArm =
      jointOrigin(pinocchioInterface, q, shoulder) - origin;
  const vector3_t forearm =
      carriedCenterOfMass(pinocchioInterface, q, elbow) - origin;
  return std::atan2(upperArm.cross(forearm).norm(), upperArm.dot(forearm));
}

/** `q` with the MPC joint `mpcJointIndex` moved by `delta`. */
vector_t withJointMoved(const LoadedRobot& robot,
                        const vector_t& state,
                        size_t mpcJointIndex,
                        scalar_t delta) {
  vector_t moved = state;
  vector_t joints = robot.model->getJointAngles(moved);
  joints(static_cast<Eigen::Index>(mpcJointIndex)) += delta;
  robot.model->setJointAngles(moved, joints);
  return robot.model->getGeneralizedCoordinates(moved);
}

TEST(ArmSwing, EachJointTurnsTheWayThatSwingsItsArmForwardOrFlexesIt) {
  const scalar_t step = 1.0e-6;
  for (const ShippedRobot& shipped : shippedRobotsWithArms()) {
    SCOPED_TRACE(shipped.name);
    absl::StatusOr<LoadedRobot> robot = load(shipped);
    ASSERT_TRUE(robot.ok()) << robot.status();
    const ModelSettings& settings = *robot->settings;
    ASSERT_TRUE(settings.hasArmSwingJoints);
    const absl::StatusOr<ArmSwingDirections> directions =
        deriveArmSwingDirections(*robot->pinocchioInterface, *robot->model,
                                 robot->initialState);
    ASSERT_TRUE(directions.ok()) << directions.status();

    const vector3_t forward =
        quaternionRotationMatrix<scalar_t>(
            robot->model->getQuatBaseRtWorldXyzw(robot->initialState))
            .col(0);
    /** One arm: its joints by MPC index and name, and the derived senses. */
    struct Arm {
      size_t shoulder;
      size_t elbow;
      scalar_t shoulderDirection;
      scalar_t elbowDirection;
    };
    for (const Arm& arm : {Arm{.shoulder = settings.j_l_shoulder_y_index,
                               .elbow = settings.j_l_elbow_y_index,
                               .shoulderDirection = directions->leftShoulder,
                               .elbowDirection = directions->leftElbow},
                           Arm{.shoulder = settings.j_r_shoulder_y_index,
                               .elbow = settings.j_r_elbow_y_index,
                               .shoulderDirection = directions->rightShoulder,
                               .elbowDirection = directions->rightElbow}}) {
      const std::string& shoulderName =
          settings.mpcModelJointNames[arm.shoulder];
      const std::string& elbowName = settings.mpcModelJointNames[arm.elbow];
      // Turning the shoulder by +direction * step moves what it carries
      // FORWARD.
      const vector_t ahead =
          withJointMoved(*robot, robot->initialState, arm.shoulder,
                         arm.shoulderDirection * step);
      const vector_t behind =
          withJointMoved(*robot, robot->initialState, arm.shoulder,
                         -arm.shoulderDirection * step);
      const scalar_t forwardRate =
          forward.dot(carriedCenterOfMass(*robot->pinocchioInterface, ahead,
                                          shoulderName) -
                      carriedCenterOfMass(*robot->pinocchioInterface, behind,
                                          shoulderName)) /
          (2.0 * step);
      EXPECT_GT(forwardRate, 0.01)
          << shoulderName << " must swing its arm forward";
      // Turning the elbow by +direction * step CLOSES it.
      const vector_t flexed = withJointMoved(
          *robot, robot->initialState, arm.elbow, arm.elbowDirection * step);
      const vector_t extended = withJointMoved(
          *robot, robot->initialState, arm.elbow, -arm.elbowDirection * step);
      const scalar_t closingRate =
          (elbowAngle(*robot->pinocchioInterface, extended, shoulderName,
                      elbowName) -
           elbowAngle(*robot->pinocchioInterface, flexed, shoulderName,
                      elbowName)) /
          (2.0 * step);
      EXPECT_GT(closingRate, 0.1) << elbowName << " must flex its arm";
    }
  }
}

TEST(ArmSwing, AtlasSwingsItsArmsWithTheShoulderYawAndTheElbowFlexion) {
  // Atlas v5 has no shoulder pitch: with the arm lowered by shx at the nominal
  // posture, shz swings it fore and aft and elx flexes the elbow, while shx
  // mostly abducts the arm and ely twists the upper arm (the joints the task
  // file used to name). The two sides' axes are mirrored, so the senses are
  // opposite.
  absl::StatusOr<LoadedRobot> robot = load(shippedRobotsWithArms().front());
  ASSERT_TRUE(robot.ok()) << robot.status();
  const ModelSettings& settings = *robot->settings;
  EXPECT_EQ(settings.mpcModelJointNames[settings.j_l_shoulder_y_index],
            "l_arm_shz");
  EXPECT_EQ(settings.mpcModelJointNames[settings.j_r_shoulder_y_index],
            "r_arm_shz");
  EXPECT_EQ(settings.mpcModelJointNames[settings.j_l_elbow_y_index],
            "l_arm_elx");
  EXPECT_EQ(settings.mpcModelJointNames[settings.j_r_elbow_y_index],
            "r_arm_elx");
  const absl::StatusOr<ArmSwingDirections> directions =
      deriveArmSwingDirections(*robot->pinocchioInterface, *robot->model,
                               robot->initialState);
  ASSERT_TRUE(directions.ok()) << directions.status();
  EXPECT_EQ(directions->leftShoulder, -1.0);
  EXPECT_EQ(directions->rightShoulder, 1.0);
  EXPECT_EQ(directions->leftElbow, 1.0);
  EXPECT_EQ(directions->rightElbow, -1.0);
}

/** A walk from every whole second k: [k, k + 0.1) double support, [k + 0.1, k +
 * 0.5) LF, then the mirror of it. */
ModeSchedule walkSchedule() {
  std::vector<scalar_t> eventTimes;
  std::vector<size_t> modeSequence{static_cast<size_t>(ModeNumber::kRf)};
  for (int k = -3; k <= 8; ++k) {
    const scalar_t start = static_cast<scalar_t>(k);
    for (const scalar_t offset : {0.0, 0.1, 0.5, 0.6})
      eventTimes.push_back(start + offset);
    for (const ModeNumber mode : {ModeNumber::kStance, ModeNumber::kLf,
                                  ModeNumber::kStance, ModeNumber::kRf}) {
      modeSequence.push_back(static_cast<size_t>(mode));
    }
  }
  return ModeSchedule(eventTimes, modeSequence);
}

/** The reference manager of `robot` on the walk schedule, its arm swing
 * configured from the task file and active. */
std::unique_ptr<SwitchedModelReferenceManager> walkingReferenceManager(
    const LoadedRobot& robot) {
  absl::StatusOr<std::shared_ptr<GaitSchedule>> gaitSchedule =
      GaitSchedule::Create(robot.referenceFile, *robot.settings);
  EXPECT_TRUE(gaitSchedule.ok()) << gaitSchedule.status();
  if (!gaitSchedule.ok()) return nullptr;
  (*gaitSchedule)->updateModeSchedule(walkSchedule());
  auto referenceManager = std::make_unique<SwitchedModelReferenceManager>(
      *gaitSchedule,
      std::make_unique<SwingTrajectoryPlanner>(SwingTrajectoryPlanner::Config(),
                                               /*numFeet=*/2),
      *robot.pinocchioInterface, *robot.model);
  const absl::Status configured = referenceManager->configureArmSwing(
      robot.task.arm_swing, robot.initialState);
  EXPECT_TRUE(configured.ok()) << configured;
  referenceManager->setArmSwingReferenceActive(true);
  return referenceManager;
}

/** `robot`'s initial state turned to `heading` and moving forward along it at
 * `speed`. */
vector_t walkingState(const LoadedRobot& robot,
                      scalar_t heading,
                      scalar_t speed) {
  vector_t state = robot.initialState;
  vector3_t euler = robot.model->getEulerzyxBaseRtWorld(state);
  euler(0) = heading;
  robot.model->setEulerzyxBaseRtWorld(state, euler);
  robot.model->setBaseComLinearVelocity(
      state,
      vector3_t(speed * std::cos(heading), speed * std::sin(heading), 0.0));
  return state;
}

TEST(ArmSwing, InLeftStanceTheLeftArmSwingsForwardAndFlexesAndTheRightOneBack) {
  // phase 0.4 at t = 1.42 (LF from 1.1 to 1.5, phase 0.5 (t - 1.1) / 0.4): the
  // peak of sin(2 pi (phase - 0.15)) and the middle of the right leg's swing,
  // when the left arm - opposite the swinging leg - is farthest forward.
  const scalar_t time = 1.42;
  for (const ShippedRobot& shipped : shippedRobotsWithArms()) {
    SCOPED_TRACE(shipped.name);
    absl::StatusOr<LoadedRobot> robot = load(shipped);
    ASSERT_TRUE(robot.ok()) << robot.status();
    const std::unique_ptr<SwitchedModelReferenceManager> referenceManager =
        walkingReferenceManager(*robot);
    ASSERT_NE(referenceManager, nullptr);
    const ModelSettings& settings = *robot->settings;
    for (const scalar_t heading : {0.0, 1.1}) {
      const vector_t knot = walkingState(*robot, heading, /*speed=*/0.6);
      const TargetTrajectories target(
          {-100.0, 100.0}, {knot, knot},
          {vector_t::Zero(robot->model->getInputDim()),
           vector_t::Zero(robot->model->getInputDim())});
      referenceManager->setTargetTrajectories(target);
      referenceManager->preSolverRun(/*initTime=*/1.02, /*finalTime=*/2.02,
                                     knot, ModeNumber::kStance);
      ASSERT_NEAR(referenceManager->getPhaseVariable(time), 0.4, 1.0e-12);
      const vector_t desired =
          referenceManager->getDesiredState(target, knot, time);
      const vector_t qKnot = robot->model->getGeneralizedCoordinates(knot);
      const vector_t qDesired =
          robot->model->getGeneralizedCoordinates(desired);
      const vector3_t forward(std::cos(heading), std::sin(heading), 0.0);
      const std::string& leftShoulder =
          settings.mpcModelJointNames[settings.j_l_shoulder_y_index];
      const std::string& rightShoulder =
          settings.mpcModelJointNames[settings.j_r_shoulder_y_index];
      const std::string& leftElbow =
          settings.mpcModelJointNames[settings.j_l_elbow_y_index];
      const std::string& rightElbow =
          settings.mpcModelJointNames[settings.j_r_elbow_y_index];
      const PinocchioInterface& pinocchio = *robot->pinocchioInterface;
      EXPECT_GT(
          forward.dot(carriedCenterOfMass(pinocchio, qDesired, leftShoulder) -
                      carriedCenterOfMass(pinocchio, qKnot, leftShoulder)),
          0.0)
          << "heading " << heading << ": the left arm swings forward";
      EXPECT_LT(
          forward.dot(carriedCenterOfMass(pinocchio, qDesired, rightShoulder) -
                      carriedCenterOfMass(pinocchio, qKnot, rightShoulder)),
          0.0)
          << "heading " << heading << ": the right arm swings back";
      EXPECT_LT(elbowAngle(pinocchio, qDesired, leftShoulder, leftElbow),
                elbowAngle(pinocchio, qKnot, leftShoulder, leftElbow))
          << "heading " << heading << ": the left elbow flexes";
      EXPECT_GT(elbowAngle(pinocchio, qDesired, rightShoulder, rightElbow),
                elbowAngle(pinocchio, qKnot, rightShoulder, rightElbow))
          << "heading " << heading << ": the right elbow extends";

      // The reference is the target's and the gait's alone: at a node state
      // turned elsewhere and moving otherwise it is the same, bit for bit, so
      // the Gauss-Newton step that holds it constant has the exact gradient.
      const vector_t elsewhere =
          walkingState(*robot, heading + 2.3, /*speed=*/-0.4);
      EXPECT_TRUE(referenceManager->getDesiredState(target, elsewhere, time) ==
                  desired)
          << "heading " << heading;
    }
  }
}

TEST(ArmSwing,
     OnTheShoulderPitchRobotsTheSwingIsTheOneTheReferenceAlwaysGenerated) {
  // The swing before its directions were derived, written out: -0.15 f on the
  // left shoulder and elbow and +0.15 f on the right ones, f = sin(2 pi (phase
  // - 0.15)) v_x. G1 and R1 derive exactly those senses, so with the default
  // arm_swing block their reference has not moved by a bit.
  const scalar_t time = 1.3;
  for (const ShippedRobot& shipped : shippedRobotsWithArms()) {
    if (std::string(shipped.name) == "drc_atlas") continue;
    SCOPED_TRACE(shipped.name);
    absl::StatusOr<LoadedRobot> robot = load(shipped);
    ASSERT_TRUE(robot.ok()) << robot.status();
    const std::unique_ptr<SwitchedModelReferenceManager> referenceManager =
        walkingReferenceManager(*robot);
    ASSERT_NE(referenceManager, nullptr);
    const ModelSettings& settings = *robot->settings;
    const scalar_t heading = 0.7;
    const scalar_t speed = 0.5;
    const vector_t knot = walkingState(*robot, heading, speed);
    const TargetTrajectories target(
        {-100.0, 100.0}, {knot, knot},
        {vector_t::Zero(robot->model->getInputDim()),
         vector_t::Zero(robot->model->getInputDim())});
    referenceManager->setTargetTrajectories(target);
    referenceManager->preSolverRun(/*initTime=*/1.02, /*finalTime=*/2.02, knot,
                                   ModeNumber::kStance);
    const vector_t desired =
        referenceManager->getDesiredState(target, knot, time);
    // Read off the interpolated target exactly as the reference manager reads
    // it, so that the comparison can be exact.
    const vector_t nominal = robot->model->getDesiredState(target, time);
    const scalar_t nominalHeading = robot->model->getYawBaseRtWorld(nominal);
    const vector3_t velocityCommand =
        robot->model->getBaseComLinearVelocity(nominal);
    const scalar_t forwardSpeed =
        std::cos(nominalHeading) * velocityCommand[0] +
        std::sin(nominalHeading) * velocityCommand[1];
    const scalar_t swing =
        std::sin(2 * M_PI * (referenceManager->getPhaseVariable(time) - 0.15)) *
        forwardSpeed;
    vector_t expected = robot->model->getJointAngles(nominal);
    expected[settings.j_l_shoulder_y_index] += -0.15 * swing;
    expected[settings.j_r_shoulder_y_index] += 0.15 * swing;
    expected[settings.j_l_elbow_y_index] += -0.15 * swing;
    expected[settings.j_r_elbow_y_index] += 0.15 * swing;
    ASSERT_GT(std::abs(swing), 0.1)
        << "the arms must be swinging for this to test anything";
    EXPECT_TRUE(robot->model->getJointAngles(desired) == expected);
  }
}

TEST(ArmSwing, NoneHoldsTheTargetsPosture) {
  absl::StatusOr<LoadedRobot> robot = load(shippedRobotsWithArms()[1]);
  ASSERT_TRUE(robot.ok()) << robot.status();
  const std::unique_ptr<SwitchedModelReferenceManager> referenceManager =
      walkingReferenceManager(*robot);
  ASSERT_NE(referenceManager, nullptr);
  ArmSwingSettings none;
  none.generator = ArmSwingGenerator::kNone;
  referenceManager->setArmSwingSettings(none);
  const vector_t knot = walkingState(*robot, /*heading=*/0.3, /*speed=*/0.6);
  const TargetTrajectories target(
      {-100.0, 100.0}, {knot, knot},
      {vector_t::Zero(robot->model->getInputDim()),
       vector_t::Zero(robot->model->getInputDim())});
  referenceManager->setTargetTrajectories(target);
  referenceManager->preSolverRun(/*initTime=*/1.02, /*finalTime=*/2.02, knot,
                                 ModeNumber::kStance);
  EXPECT_TRUE(robot->model->getJointAngles(referenceManager->getDesiredState(
                  target, knot, /*time=*/1.42)) ==
              robot->model->getJointAngles(knot));
}

TEST(ArmSwing, TheBlockConvertsAndRefusesWhatHasNoMeaning) {
  // The defaults are the swing the reference manager always generated.
  const absl::StatusOr<ArmSwingSettings> defaults =
      armSwingSettingsFromConfig(mpc_config::ArmSwingConfig());
  ASSERT_TRUE(defaults.ok()) << defaults.status();
  EXPECT_EQ(defaults->generator, ArmSwingGenerator::kProcedural);
  EXPECT_EQ(defaults->shoulderGain, 0.15);
  EXPECT_EQ(defaults->elbowGain, 0.15);
  EXPECT_EQ(defaults->phaseOffset, 0.15);

  mpc_config::ArmSwingConfig tuned;
  tuned.generator = "none";
  tuned.shoulder_gain = 0.3;
  tuned.elbow_gain = 0.0;
  tuned.phase_offset = -0.2;
  const absl::StatusOr<ArmSwingSettings> converted =
      armSwingSettingsFromConfig(tuned);
  ASSERT_TRUE(converted.ok()) << converted.status();
  EXPECT_EQ(converted->generator, ArmSwingGenerator::kNone);
  EXPECT_EQ(converted->shoulderGain, 0.3);
  EXPECT_EQ(converted->elbowGain, 0.0);
  EXPECT_EQ(converted->phaseOffset, -0.2);

  mpc_config::ArmSwingConfig unknown;
  unknown.generator = "swing";
  const absl::StatusOr<ArmSwingSettings> refused =
      armSwingSettingsFromConfig(unknown);
  ASSERT_FALSE(refused.ok());
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "procedural, none"))
      << refused.status();

  const scalar_t nan = std::numeric_limits<scalar_t>::quiet_NaN();
  for (const scalar_t bad :
       {-0.1, nan, std::numeric_limits<scalar_t>::infinity()}) {
    mpc_config::ArmSwingConfig shoulder;
    shoulder.shoulder_gain = bad;
    EXPECT_FALSE(armSwingSettingsFromConfig(shoulder).ok())
        << "shoulder_gain " << bad;
    mpc_config::ArmSwingConfig elbow;
    elbow.elbow_gain = bad;
    EXPECT_FALSE(armSwingSettingsFromConfig(elbow).ok())
        << "elbow_gain " << bad;
  }
  mpc_config::ArmSwingConfig phase;
  phase.phase_offset = nan;
  EXPECT_FALSE(armSwingSettingsFromConfig(phase).ok());
}

}  // namespace
}  // namespace ocs2::humanoid
