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

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "robot_core/ResourcePaths.h"

/*
 * The operator's command as the whole-body MPC's reference manager reads it back from the target (design defect D3):
 * getCommandedVelocity() is the planar base velocity of the target's knots and getCommandedYawRate() their turn about
 * the world z axis, read through the model (MpcRobotModelBase::getTargetPlanarCommand()). Both used to read the
 * centroidal slots of the target state on every model: the first two entries, which on the whole-body state are the
 * base POSITION, and the momentum channel's entry 5, a ZYX roll rate here, through the composite yaw inertia. On the
 * shipped G1 whole-body files; builds no CppAD model.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kInitTime = 1.5;
constexpr scalar_t kHorizon = 1.0;

/** The G1 whole-body reference manager and target calculator, wired as the MPC node wires them (no solver). */
struct WholeBodyReferences {
  std::unique_ptr<ModelSettings> modelSettings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model;
  std::unique_ptr<SwitchedModelReferenceManager> referenceManager;
  std::unique_ptr<WBMpcTargetTrajectoriesCalculator> calculator;
  vector_t initialState;
};

absl::StatusOr<WholeBodyReferences> loadG1() {
  // LINT.IfChange(robot_files)
  ASSIGN_OR_RETURN(const std::string taskFile, robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto"));
  ASSIGN_OR_RETURN(const std::string urdfFile, robot::resolveResourcePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"));
  ASSIGN_OR_RETURN(const std::string referenceFile,
                   robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto"));
  // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:commanded_twist_test_data)
  WholeBodyReferences loaded;
  ASSIGN_OR_RETURN(const mpc_config::TaskFile task, loadTaskFile(taskFile));
  ASSIGN_OR_RETURN(ModelSettings modelSettings, ModelSettings::Create(task, urdfFile, "wb_mpc_", /*verbose=*/false));
  loaded.modelSettings = std::make_unique<ModelSettings>(std::move(modelSettings));
  ASSIGN_OR_RETURN(PinocchioInterface pinocchioInterface, loadCustomPinocchioInterface(task, urdfFile, *loaded.modelSettings));
  loaded.pinocchioInterface = std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));
  loaded.model = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*loaded.modelSettings);
  ASSIGN_OR_RETURN(const SwingTrajectoryPlanner::Config swingConfig, swingTrajectorySettingsFromConfig(task.swing_trajectory_config));
  ASSIGN_OR_RETURN(std::shared_ptr<GaitSchedule> gaitSchedule,
                   GaitSchedule::Create(referenceFile, *loaded.modelSettings, /*verbose=*/false));
  loaded.referenceManager = std::make_unique<SwitchedModelReferenceManager>(
      std::move(gaitSchedule), std::make_unique<SwingTrajectoryPlanner>(swingConfig, kNumContacts), *loaded.pinocchioInterface,
      *loaded.model);
  ASSIGN_OR_RETURN(loaded.calculator, WBMpcTargetTrajectoriesCalculator::Create(referenceFile, *loaded.model, kHorizon));
  ASSIGN_OR_RETURN(const vector_t tuningState,
                   stateValuesFromConfig(task.initial_state, stateInputLayout(*loaded.modelSettings, StateInputLayout::Mpc::kWholeBody),
                                         "initial_state"));
  ASSIGN_OR_RETURN(loaded.initialState, loaded.model->stateFromTuning(tuningState));
  return loaded;
}

TEST(WBReferenceManagerCommandedTwist, TheCommandIsTheTargetsBaseVelocityAndTurn) {
  absl::StatusOr<WholeBodyReferences> g1 = loadG1();
  ASSERT_TRUE(g1.ok()) << g1.status();
  const WBAccelMpcRobotModel<scalar_t>& model = *g1->model;
  // Away from the origin and turned, so that no base position or heading can pass for the command.
  vector_t state = g1->initialState;
  model.setBasePoseEulerZyx(state, (vector6_t() << 1.0, 2.0, model.getBasePosition(state)(2), 0.7, 0.0, 0.0).finished());
  const TargetTrajectories target =
      g1->calculator->commandedVelocityToTargetTrajectories(vector4_t(0.08, -0.04, 0.0, 0.1), kInitTime, state,
                                                            /*measuredMotion=*/BasePlanarMotion{});
  ASSERT_EQ(target.stateTrajectory.size(), 3);
  g1->referenceManager->setTargetTrajectories(target);
  g1->referenceManager->preSolverRun(kInitTime, kInitTime + kHorizon, state, ModeNumber::kStance);

  // Every knot moves at the command in its own heading frame and turns at it (design README section 2.9); the reference
  // manager reads the interpolated target, so between knots it answers the linear interpolation of the knots' commands.
  const vector2_t commandVelocity(0.08, -0.04);
  const scalar_t commandYawRate = 0.1;
  std::vector<vector2_t> knotVelocities;
  for (const vector_t& knot : target.stateTrajectory) {
    const scalar_t yaw = model.getBasePoseEulerZyx(knot)(3);
    knotVelocities.emplace_back(std::cos(yaw) * commandVelocity(0) - std::sin(yaw) * commandVelocity(1),
                                std::sin(yaw) * commandVelocity(0) + std::cos(yaw) * commandVelocity(1));
  }
  ASSERT_GT((knotVelocities[2] - knotVelocities[0]).norm(), 1.0e-3) << "the command must turn the base";
  for (const scalar_t time : {kInitTime, kInitTime + 0.3, kInitTime + 0.7, kInitTime + 0.9}) {
    SCOPED_TRACE(time);
    const size_t segment = time < target.timeTrajectory[1] ? 0 : 1;
    const scalar_t fraction =
        (time - target.timeTrajectory[segment]) / (target.timeTrajectory[segment + 1] - target.timeTrajectory[segment]);
    const vector2_t expected = (1.0 - fraction) * knotVelocities[segment] + fraction * knotVelocities[segment + 1];
    EXPECT_LT((g1->referenceManager->getCommandedVelocity(time) - expected).cwiseAbs().maxCoeff(), 1.0e-12);
    EXPECT_NEAR(g1->referenceManager->getCommandedYawRate(time), commandYawRate, 1.0e-12);
  }
}

TEST(WBReferenceManagerCommandedTwist, NoTargetCommandsNothing) {
  absl::StatusOr<WholeBodyReferences> g1 = loadG1();
  ASSERT_TRUE(g1.ok()) << g1.status();
  EXPECT_EQ(g1->referenceManager->getCommandedVelocity(kInitTime), vector2_t::Zero());
  EXPECT_EQ(g1->referenceManager->getCommandedYawRate(kInitTime), 0.0);
}

/** R_z(angle). */
matrix3_t yawRotation(scalar_t angle) {
  return quaternionRotationMatrix<scalar_t>(quaternionFromEulerZyx(vector3_t(angle, 0.0, 0.0)));
}

/**
 * `state` turned by `angle` about the world z axis through the origin: the base position and the world-frame linear
 * velocity turn, the orientation is left-multiplied by the turn, the body angular velocity and the joints stay.
 */
vector_t turnedAboutWorldZ(const WBAccelMpcRobotModel<scalar_t>& model, const vector_t& state, scalar_t angle) {
  vector_t turned = state;
  model.setBasePosition(turned, yawRotation(angle) * model.getBasePosition(state));
  model.setBaseOrientation(turned, quaternionProduct(quaternionFromEulerZyx(vector3_t(angle, 0.0, 0.0)), model.getBaseOrientation(state)));
  // The base linear velocity follows the configuration [p_W, xi, q_j].
  const Eigen::Index velocity = static_cast<Eigen::Index>(model.getConfigurationDim());
  turned.segment<3>(velocity) = yawRotation(angle) * state.segment<3>(velocity);
  return turned;
}

/** Expects `turned` to be `state` turned by `angle` about the world z axis (turnedAboutWorldZ()), the orientation as a rotation. */
void expectTurned(const WBAccelMpcRobotModel<scalar_t>& model, const vector_t& state, const vector_t& turned, scalar_t angle) {
  constexpr scalar_t kTolerance = 1.0e-12;
  const vector_t expected = turnedAboutWorldZ(model, state, angle);
  const Eigen::Index orientation = static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  const scalar_t scale = std::max(1.0, state.cwiseAbs().maxCoeff());
  for (Eigen::Index i = 0; i < state.size(); ++i) {
    if (i >= orientation && i < orientation + 4) continue;
    EXPECT_NEAR(turned(i), expected(i), kTolerance * scale) << "state entry " << i;
  }
  const matrix3_t rotation = quaternionRotationMatrix<scalar_t>(model.getBaseOrientation(turned));
  const matrix3_t expectedRotation = yawRotation(angle) * quaternionRotationMatrix<scalar_t>(model.getBaseOrientation(state));
  EXPECT_LE((rotation - expectedRotation).cwiseAbs().maxCoeff(), kTolerance) << "the orientation, as a rotation";
}

TEST(WBReferenceManagerCommandedTwist, TheCommandReferencesAreEquivariantUnderAWorldTurnAcrossPi) {
  // A world turn of the robot turns its command references with it: the knots of a velocity command and of a pose
  // command, and the reference the costs read between them (getDesiredState(), the quaternion slerped), when the turn
  // takes the heading across +-pi (3.1 + 0.2), where the Euler yaw of the knots jumps by 2 pi.
  absl::StatusOr<WholeBodyReferences> g1 = loadG1();
  ASSERT_TRUE(g1.ok()) << g1.status();
  const WBAccelMpcRobotModel<scalar_t>& model = *g1->model;
  vector_t state = g1->initialState;
  model.setBasePoseEulerZyx(state, (vector6_t() << 1.0, 2.0, model.getBasePosition(state)(2), 3.1, 0.0, 0.0).finished());
  constexpr scalar_t kTurn = 0.2;
  const vector_t turnedState = turnedAboutWorldZ(model, state, kTurn);
  ASSERT_LT(model.getBaseOrientationEulerZyx(turnedState)(0), -2.9) << "the turned heading is across the cut";
  const vector4_t command(0.4, -0.2, 0.0, 0.5);
  for (const bool poseCommand : {false, true}) {
    SCOPED_TRACE(poseCommand ? "pose command" : "velocity command");
    g1->calculator->reset();
    const TargetTrajectories target = poseCommand ? g1->calculator->commandedPositionToTargetTrajectories(command, kInitTime, state)
                                                  : g1->calculator->commandedVelocityToTargetTrajectories(
                                                        command, kInitTime, state, /*measuredMotion=*/BasePlanarMotion{});
    g1->calculator->reset();
    const TargetTrajectories turnedTarget =
        poseCommand ? g1->calculator->commandedPositionToTargetTrajectories(command, kInitTime, turnedState)
                    : g1->calculator->commandedVelocityToTargetTrajectories(command, kInitTime, turnedState,
                                                                            /*measuredMotion=*/BasePlanarMotion{});
    // The knot times are the command's, up to the round-off of the turned arithmetic (a pose command's subdivision).
    ASSERT_EQ(turnedTarget.timeTrajectory.size(), target.timeTrajectory.size());
    for (size_t k = 0; k < target.timeTrajectory.size(); ++k) {
      EXPECT_NEAR(turnedTarget.timeTrajectory[k], target.timeTrajectory[k], 1.0e-12);
    }
    ASSERT_EQ(turnedTarget.stateTrajectory.size(), target.stateTrajectory.size());
    for (size_t k = 0; k < target.stateTrajectory.size(); ++k) {
      SCOPED_TRACE(absl::StrCat("knot ", k));
      expectTurned(model, target.stateTrajectory[k], turnedTarget.stateTrajectory[k], kTurn);
    }
    g1->referenceManager->setTargetTrajectories(target);
    g1->referenceManager->preSolverRun(kInitTime, kInitTime + kHorizon, state, ModeNumber::kStance);
    std::vector<vector_t> desired;
    for (scalar_t time = kInitTime; time <= kInitTime + kHorizon; time += 0.05) {
      desired.push_back(g1->referenceManager->getDesiredState(target, state, time));
    }
    g1->referenceManager->setTargetTrajectories(turnedTarget);
    g1->referenceManager->preSolverRun(kInitTime, kInitTime + kHorizon, turnedState, ModeNumber::kStance);
    size_t node = 0;
    for (scalar_t time = kInitTime; time <= kInitTime + kHorizon; time += 0.05, ++node) {
      SCOPED_TRACE(absl::StrCat("reference at ", time));
      ASSERT_LT(node, desired.size());
      expectTurned(model, desired[node], g1->referenceManager->getDesiredState(turnedTarget, turnedState, time), kTurn);
    }
  }
}

TEST(WBReferenceManagerCommandedTwist, TheBasePoseSeamKeepsTheReferencesTurnInTheWorld) {
  // The whole body ships no heuristic layer (basePoseEmpty()), so shapeBasePose() is the identity there; with one, the
  // seam tilts the reference's attitude, and its stored body angular velocity is carried along so that the reference
  // still turns about the world z axis at the command's rate.
  absl::StatusOr<WholeBodyReferences> g1 = loadG1();
  ASSERT_TRUE(g1.ok()) << g1.status();
  const WBAccelMpcRobotModel<scalar_t>& model = *g1->model;
  ASSERT_TRUE(g1->referenceManager->getLocomotionHeuristicLayer()->basePoseEmpty()) << "the shipped whole body lists no heuristic";
  vector_t state = g1->initialState;
  model.setBasePoseEulerZyx(state, (vector6_t() << 1.0, 2.0, model.getBasePosition(state)(2), 0.7, 0.0, 0.0).finished());
  const TargetTrajectories target =
      g1->calculator->commandedVelocityToTargetTrajectories(vector4_t(0.08, -0.04, 0.0, 0.1), kInitTime, state,
                                                            /*measuredMotion=*/BasePlanarMotion{});
  g1->referenceManager->setTargetTrajectories(target);
  g1->referenceManager->preSolverRun(kInitTime, kInitTime + kHorizon, state, ModeNumber::kStance);
  const vector_t& knot = target.stateTrajectory.back();
  EXPECT_TRUE(g1->referenceManager->shapeBasePose(kInitTime + 0.5, knot) == knot) << "no layer: the identity, bit for bit";

  LocomotionHeuristicConfig config;
  config.formulation.basePose = {"orientation_compensation"};
  config.orientationCompensation.pitchOffset = 0.05;
  config.orientationCompensation.rollOffset = 0.03;
  LocomotionHeuristicModelParameters parameters;
  parameters.totalMass = 35.0;
  parameters.totalWeight = parameters.totalMass * parameters.gravity;
  parameters.nominalComHeight = 0.7;
  absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> layer =
      LocomotionHeuristicLayer::Create(config, parameters, LocomotionHeuristicEnvironment{});
  ASSERT_TRUE(layer.ok()) << layer.status();
  g1->referenceManager->setLocomotionHeuristicLayer(std::shared_ptr<LocomotionHeuristicLayer>(*std::move(layer)));
  const vector_t shaped = g1->referenceManager->shapeBasePose(kInitTime + 0.5, knot);
  const vector3_t knotAngles = model.getBaseOrientationEulerZyx(knot);
  const vector3_t shapedAngles = model.getBaseOrientationEulerZyx(shaped);
  ASSERT_GT((shapedAngles - knotAngles).cwiseAbs().maxCoeff(), 0.01) << "the layer tilts the reference";
  // The world angular velocity R(xi) w_B: the knot's turn about the world z axis, before and after.
  const vector3_t knotTurn = quaternionRotationMatrix<scalar_t>(model.getBaseOrientation(knot)) * model.getBaseAngularVelocityLocal(knot);
  const vector3_t shapedTurn =
      quaternionRotationMatrix<scalar_t>(model.getBaseOrientation(shaped)) * model.getBaseAngularVelocityLocal(shaped);
  ASSERT_GT(std::abs(knotTurn(2)), 0.05) << "the knot turns";
  EXPECT_LE((knotTurn - vector3_t(0.0, 0.0, knotTurn(2))).cwiseAbs().maxCoeff(), 1.0e-12) << "a world yaw rate";
  EXPECT_LE((shapedTurn - knotTurn).cwiseAbs().maxCoeff(), 1.0e-12) << "the shaped reference turns as the knot does";
  // Not the knot's body rate kept as it was, which would turn about the tilted body z axis.
  EXPECT_GT((model.getBaseAngularVelocityLocal(shaped) - model.getBaseAngularVelocityLocal(knot)).cwiseAbs().maxCoeff(), 1.0e-4);
}

}  // namespace
}  // namespace ocs2::humanoid
