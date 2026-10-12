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

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/misc/LinearInterpolation.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "robot_core/ResourcePaths.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/*
 * The layout-agnostic API of the whole-body robot model (MpcRobotModelBase) on its quaternion layout, on the G1: the
 * state [p_W, xi, q_j, dp_W, w_B, dq_j] holds Pinocchio's configuration and velocity of the Composite(Translation,
 * Spherical) root, its tangent has one row less, and the state manifold is the product with the unit quaternion; the
 * tuning layout agrees with the index accessors and maps onto the state through the Euler boundary; the Euler accessors
 * round-trip the pose modulo 2 pi and the rotation of any stored quaternion; interpolation slerps the orientation the
 * shorter way; the heading is the twist of the attitude in (-pi, pi]; the base angular velocity is the stored body
 * rate; the base twist is the world rate of the stored velocities and a knot's command its base twist; the tangent maps
 * are the derivatives they name, Pinocchio's tangent included; and the measurement helpers fill Pinocchio's q and v
 * without allocating.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
constexpr scalar_t kFiniteDifferenceTolerance = 1.0e-8;
/** Yaws beyond +-pi, which the quaternion stores modulo 2 pi. */
constexpr std::array<scalar_t, 2> kLargeYaws = {3.0 * M_PI / 2.0, -5.0 * M_PI / 2.0};

/** The G1 whole-body model of the shipped task file, with the settings it refers to and its Pinocchio model. */
struct LoadedRobot {
  mpc_config::TaskFile task;
  std::unique_ptr<ModelSettings> modelSettings;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;
};

absl::StatusOr<LoadedRobot> loadG1() {
  LoadedRobot robot;
  ASSIGN_OR_RETURN(const std::string taskFile, robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto"));
  ASSIGN_OR_RETURN(const std::string urdfFile, robot::resolveResourcePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"));
  ASSIGN_OR_RETURN(robot.task, loadTaskFile(taskFile));
  ASSIGN_OR_RETURN(ModelSettings modelSettings, ModelSettings::Create(robot.task, urdfFile, "boundary_test", /*verbose=*/false));
  robot.modelSettings = std::make_unique<ModelSettings>(std::move(modelSettings));
  robot.model = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*robot.modelSettings);
  ASSIGN_OR_RETURN(PinocchioInterface pinocchioInterface, loadCustomPinocchioInterface(robot.task, urdfFile, *robot.modelSettings));
  robot.pinocchioInterface = std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));
  return robot;
}

/**
 * The twist heading of the ZYX Euler angles `eulerZyx` = (psi, theta, phi) in closed form, psi - 2 atan(tan(theta / 2)
 * tan(phi / 2)) wrapped into [-pi, pi] (orientation/BaseOrientation.h twistHeading()).
 */
scalar_t twistHeadingOfEulerZyx(const vector3_t& eulerZyx) {
  return std::remainder(eulerZyx(0) - 2.0 * std::atan(std::tan(eulerZyx(1) / 2.0) * std::tan(eulerZyx(2) / 2.0)), 2.0 * M_PI);
}

/** `angle` wrapped into [-pi, pi]. */
scalar_t wrapped(scalar_t angle) {
  return std::remainder(angle, 2.0 * M_PI);
}

/** The first state slot of the base linear velocity dp_W of the whole-body state [p_W, xi, q_j, dp_W, w_B, dq_j]. */
Eigen::Index linearVelocityStart(const MpcRobotModelBase<scalar_t>& model) {
  return static_cast<Eigen::Index>(model.getConfigurationDim());
}

/** The first state slot of the body angular velocity w_B of the whole-body state [p_W, xi, q_j, dp_W, w_B, dq_j]. */
Eigen::Index angularVelocityStart(const MpcRobotModelBase<scalar_t>& model) {
  return static_cast<Eigen::Index>(model.getConfigurationDim() + 3);
}

/**
 * A state of `model` with distinct entries, the base orientation the quaternion `xi` as stored (not normalized) and the
 * body angular velocity `angularVelocity`.
 */
vector_t stateWith(const MpcRobotModelBase<scalar_t>& model, const vector4_t& xi, const vector3_t& angularVelocity) {
  vector_t state = vector_t::LinSpaced(static_cast<Eigen::Index>(model.getStateDim()), -0.7, 0.9);
  state.segment<4>(static_cast<Eigen::Index>(model.getBaseOrientationStartIndex())) = xi;
  state.segment<3>(angularVelocityStart(model)) = angularVelocity;
  return state;
}

/** A uniformly random unit quaternion (x, y, z, w) of `generator`. */
vector4_t randomUnitQuaternion(std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  const vector4_t xi(normal(generator), normal(generator), normal(generator), normal(generator));
  return xi / xi.norm();
}

/** The segment named `name` of `layout`, or null. */
const TuningSegment* absl_nullable findSegment(const TuningLayout& layout, absl::string_view name) {
  for (const TuningSegment& segment : layout.segments) {
    if (segment.name == name) return &segment;
  }
  return nullptr;
}

/** vee(log(R0^T R1)): the body rotation from R0 to R1. */
vector3_t bodyRotation(const matrix3_t& rotation0, const matrix3_t& rotation1) {
  const Eigen::AngleAxis<scalar_t> angleAxis(rotation0.transpose() * rotation1);
  return angleAxis.angle() * angleAxis.axis();
}

matrix3_t rotationOf(const vector4_t& xi) {
  return quaternion_t(xi).normalized().toRotationMatrix();
}

/** The angle of the rotation between the unit quaternions `xi0` and `xi1`, which is 0 for xi1 = -xi0. */
scalar_t rotationDistance(const vector4_t& xi0, const vector4_t& xi1) {
  return quaternion_t(xi0).angularDistance(quaternion_t(xi1));
}

TEST(WBRobotModelBoundary, SizesAndIndicesAreThoseOfTheQuaternionLayout) {
  absl::StatusOr<LoadedRobot> robot = loadG1();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  const size_t joints = model.getJointDim();
  EXPECT_EQ(model.getConfigurationDim(), 7 + joints);
  EXPECT_EQ(model.getVelocityDim(), 6 + joints);
  EXPECT_EQ(model.getStateDim(), model.getConfigurationDim() + model.getVelocityDim());
  EXPECT_EQ(model.getStateTangentDim(), 2 * model.getVelocityDim());
  EXPECT_EQ(model.getStateTangentDim() + 1, model.getStateDim());
  EXPECT_EQ(model.getBaseStartindex(), 0U);
  EXPECT_EQ(model.getBaseOrientationStartIndex(), 3U);
  EXPECT_EQ(model.getJointStartindex(), 7U);
  EXPECT_EQ(model.getJointVelocitiesStartindex(), model.getStateDim() - joints);
  // Pinocchio's q and v of the same model.
  EXPECT_EQ(static_cast<size_t>(robot->pinocchioInterface->getModel().nq), model.getConfigurationDim());
  EXPECT_EQ(static_cast<size_t>(robot->pinocchioInterface->getModel().nv), model.getVelocityDim());

  // The CppAD model has the same layout.
  const WBAccelMpcRobotModel<ad_scalar_t> modelCppAd(*robot->modelSettings);
  EXPECT_EQ(modelCppAd.getStateDim(), model.getStateDim());
  EXPECT_EQ(modelCppAd.getStateTangentDim(), model.getStateTangentDim());
  EXPECT_EQ(modelCppAd.getVelocityDim(), model.getVelocityDim());
  EXPECT_EQ(describeTuningLayout(modelCppAd.getTuningLayout()), describeTuningLayout(model.getTuningLayout()));

  // The state lives on a manifold: one unit quaternion, one row more than its tangent.
  const absl::StatusOr<std::shared_ptr<const StateManifold>> manifold = stateManifoldOf(model);
  ASSERT_TRUE(manifold.ok()) << manifold.status();
  ASSERT_NE(*manifold, nullptr);
  EXPECT_EQ((*manifold)->getAmbientDim(), model.getStateDim());
  EXPECT_EQ((*manifold)->getTangentDim(), model.getStateTangentDim());
  EXPECT_EQ((*manifold)->getAmbientDim(), (*manifold)->getTangentDim() + 1);
}

TEST(WBRobotModelBoundary, TheTuningLayoutAgreesWithTheIndexAccessors) {
  absl::StatusOr<LoadedRobot> robot = loadG1();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  const size_t joints = model.getJointDim();
  const TuningLayout& layout = model.getTuningLayout();
  EXPECT_EQ(layout.name, "whole-body");
  EXPECT_EQ(getTuningLayoutDim(layout), model.getStateTangentDim());
  EXPECT_EQ(getTuningLayoutStateDim(layout), model.getStateDim());
  const std::vector<std::string> names = {"base_position",        "base_orientation",      "joint_positions",
                                          "base_linear_velocity", "base_angular_velocity", "joint_velocities"};
  // The tuning rows hold three Euler angles for the orientation; the state rows are [q, v].
  const std::vector<size_t> tuningStarts = {0, 3, 6, 6 + joints, 9 + joints, 12 + joints};
  const std::vector<size_t> stateStarts = {0, 3, 7, 7 + joints, 10 + joints, 13 + joints};
  ASSERT_EQ(layout.segments.size(), names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    const TuningSegment& segment = layout.segments[i];
    SCOPED_TRACE(segment.name);
    EXPECT_EQ(segment.name, names[i]);
    EXPECT_EQ(segment.stateStart, stateStarts[i]);
    const absl::StatusOr<size_t> tuningStart = tuningSegmentStart(layout, segment.name);
    ASSERT_TRUE(tuningStart.ok()) << tuningStart.status();
    EXPECT_EQ(*tuningStart, tuningStarts[i]);
    if (segment.kind == TuningSegmentKind::kEulerZyxOrientation) {
      EXPECT_EQ(segment.stateDim, kQuaternionOrientationStateDim);
      EXPECT_EQ(segment.tuningDim, 3U);
    } else {
      EXPECT_EQ(segment.stateDim, segment.tuningDim);
    }
  }
  const TuningSegment* absl_nullable position = findSegment(layout, "base_position");
  const TuningSegment* absl_nullable orientation = findSegment(layout, "base_orientation");
  const TuningSegment* absl_nullable jointPositions = findSegment(layout, "joint_positions");
  const TuningSegment* absl_nullable linearVelocity = findSegment(layout, "base_linear_velocity");
  const TuningSegment* absl_nullable angularVelocity = findSegment(layout, "base_angular_velocity");
  const TuningSegment* absl_nullable jointVelocities = findSegment(layout, "joint_velocities");
  ASSERT_NE(position, nullptr);
  ASSERT_NE(orientation, nullptr);
  ASSERT_NE(jointPositions, nullptr);
  ASSERT_NE(linearVelocity, nullptr);
  ASSERT_NE(angularVelocity, nullptr);
  ASSERT_NE(jointVelocities, nullptr);
  EXPECT_EQ(position->stateStart, model.getBaseStartindex());
  EXPECT_EQ(orientation->stateStart, model.getBaseOrientationStartIndex());
  EXPECT_EQ(orientation->kind, TuningSegmentKind::kEulerZyxOrientation);
  EXPECT_EQ(jointPositions->stateStart, model.getJointStartindex());
  EXPECT_EQ(jointPositions->tuningDim, joints);
  EXPECT_EQ(static_cast<Eigen::Index>(linearVelocity->stateStart), linearVelocityStart(model));
  EXPECT_EQ(angularVelocity->kind, TuningSegmentKind::kZyxOrderedAngularVelocity);
  EXPECT_EQ(static_cast<Eigen::Index>(angularVelocity->stateStart), angularVelocityStart(model));
  EXPECT_EQ(jointVelocities->stateStart, model.getJointVelocitiesStartindex());
  EXPECT_EQ(jointVelocities->tuningDim, joints);

  // The accessors write where the layout says; a level knot's turn is the body rate (0, 0, yaw rate), and
  // getBaseComVelocity() is the base velocity in tuning order [dp_W, P_3 w_B] (design decision D18).
  vector_t state = model.getNeutralState();
  model.setBaseComLinearVelocity(state, vector3_t(1.0, 2.0, 3.0));
  model.setBaseAngularVelocityFromWorldYawRate(state, /*yawRate=*/4.0);
  EXPECT_EQ(vector3_t(state.segment<3>(static_cast<Eigen::Index>(linearVelocity->stateStart))), vector3_t(1.0, 2.0, 3.0));
  EXPECT_EQ(vector3_t(state.segment<3>(static_cast<Eigen::Index>(angularVelocity->stateStart))), vector3_t(0.0, 0.0, 4.0));
  EXPECT_EQ(model.getBaseComVelocity(state), (vector6_t() << 1.0, 2.0, 3.0, 4.0, 0.0, 0.0).finished());
}

TEST(WBRobotModelBoundary, TheEulerBoundaryAccessorsRoundTripThePoseModulo2PiAndTheRotationOfAnyQuaternion) {
  absl::StatusOr<LoadedRobot> robot = loadG1();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  std::mt19937 generator(40);
  for (const scalar_t yaw : {kLargeYaws[0], kLargeYaws[1], 0.4, -2.9}) {
    SCOPED_TRACE(yaw);
    const vector_t state = stateWith(model, randomUnitQuaternion(generator), vector3_t(0.4, -0.2, 0.1));

    // get(set(pose)) is the pose, its yaw modulo 2 pi, and nothing else of the state moves.
    const vector6_t pose = (vector6_t() << 0.3, -0.2, 0.71, yaw, -0.05, 0.07).finished();
    vector_t written = state;
    model.setBasePoseEulerZyx(written, pose);
    const vector6_t read = model.getBasePoseEulerZyx(written);
    EXPECT_EQ(vector3_t(read.head<3>()), vector3_t(pose.head<3>()));
    EXPECT_NEAR(read(3), wrapped(yaw), 1.0e-12);
    EXPECT_NEAR(read(4), pose(4), 1.0e-12);
    EXPECT_NEAR(read(5), pose(5), 1.0e-12);
    EXPECT_GT(read(3), -M_PI);
    EXPECT_LE(read(3), M_PI);
    EXPECT_EQ(model.getBaseOrientationEulerZyx(written), vector3_t(read.tail<3>()));
    EXPECT_EQ(vector4_t(written.segment<4>(3)), quaternionFromEulerZyx(vector3_t(pose.tail<3>())));
    // The heading is not the yaw: it is the twist of the attitude (design decision D9), wrapped.
    EXPECT_NEAR(model.getBaseHeading(written), twistHeadingOfEulerZyx(pose.tail<3>()), 1.0e-12);
    const Eigen::Index rest = static_cast<Eigen::Index>(model.getStateDim() - model.getJointStartindex());
    EXPECT_EQ(written.tail(rest), state.tail(rest));

    // set(get(x)) reproduces the rotation of the stored quaternion, whatever its sign, to 1e-15 / cos(pitch): the angles
    // of eulerZyxFromQuaternion() lose about 2e-16 / cos(pitch) of the rotation (orientation/EulerBoundary.h).
    const scalar_t tolerance = 1.0e-15 / std::cos(model.getBaseOrientationEulerZyx(state)(1));
    for (const scalar_t sign : {1.0, -1.0}) {
      vector_t signedState = state;
      signedState.segment<4>(3) *= sign;
      vector_t copy = signedState;
      model.setBaseOrientationEulerZyx(copy, model.getBaseOrientationEulerZyx(signedState));
      EXPECT_LT(rotationDistance(model.getBaseOrientation(copy), model.getBaseOrientation(signedState)), tolerance) << "sign " << sign;
      model.setBasePoseEulerZyx(copy, model.getBasePoseEulerZyx(signedState));
      EXPECT_LT(rotationDistance(model.getBaseOrientation(copy), model.getBaseOrientation(signedState)), tolerance) << "sign " << sign;
      EXPECT_EQ(copy.head<3>(), signedState.head<3>());
    }
  }
}

TEST(WBRobotModelBoundary, InterpolationSlerpsTheOrientationTheShorterWay) {
  absl::StatusOr<LoadedRobot> robot = loadG1();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  vector_t first = model.getNeutralState();
  vector_t second = model.getNeutralState();
  model.setBasePoseEulerZyx(first, (vector6_t() << 0.0, 0.0, 0.7, 3.0, 0.0, 0.0).finished());
  model.setBasePoseEulerZyx(second, (vector6_t() << 0.4, 0.1, 0.7, 3.5, 0.0, 0.0).finished());
  model.setBaseAngularVelocityFromWorldYawRate(first, /*yawRate=*/0.5);
  model.setBaseAngularVelocityFromWorldYawRate(second, /*yawRate=*/0.5);
  const vector_t input = vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));
  const TargetTrajectories targets({0.0, 1.0}, {first, second}, {input, input});
  const absl::StatusOr<std::shared_ptr<const StateManifold>> manifold = stateManifoldOf(model);
  ASSERT_TRUE(manifold.ok()) << manifold.status();
  ASSERT_NE(*manifold, nullptr);

  // Across +-pi the shorter way: the midpoint faces 3.25, read back in (-pi, pi].
  const vector_t midpoint = model.getDesiredState(targets, /*time=*/0.5);
  EXPECT_NEAR(model.getBaseHeading(midpoint), wrapped(3.25), 1.0e-12);
  EXPECT_NEAR(model.getBaseOrientationEulerZyx(midpoint)(0), wrapped(3.25), 1.0e-12);
  for (const scalar_t time : {0.1, 0.25, 0.5, 0.9}) {
    SCOPED_TRACE(time);
    const vector_t interpolated = model.getDesiredState(targets, time);
    // NOLINTNEXTLINE(state-interpolation): every row but the orientation is held to the linear interpolation, bit for bit
    const vector_t linear = targets.getDesiredState(time);
    // The orientation is the slerp of the knots' (the state manifold's interpolate(), bit for bit), every other row the
    // linear interpolation of TargetTrajectories, bit for bit.
    // LinearInterpolation's weight of the first knot: the slerp moves by its complement.
    const scalar_t alpha = 1.0 - LinearInterpolation::timeSegment(time, targets.timeTrajectory).second;
    const vector4_t slerped = quaternionSlerp<scalar_t>(vector4_t(first.segment<4>(3)), vector4_t(second.segment<4>(3)), alpha);
    EXPECT_EQ(vector4_t(interpolated.segment<4>(3)), slerped);
    EXPECT_EQ(vector4_t(interpolated.segment<4>(3)), vector4_t((*manifold)->interpolate(first, second, alpha).segment<4>(3)));
    EXPECT_NEAR(interpolated.segment<4>(3).norm(), 1.0, 1.0e-15);
    vector_t others = interpolated;
    others.segment<4>(3) = linear.segment<4>(3);
    EXPECT_EQ(others, linear);
    EXPECT_NEAR(wrapped(model.getBaseHeading(interpolated) - 3.0), 0.5 * time, 1.0e-12);
  }
  // At the knots and outside the times, the knots themselves.
  for (const scalar_t time : {-1.0, 0.0}) EXPECT_EQ(model.getDesiredState(targets, time), first) << "time " << time;
  for (const scalar_t time : {1.0, 2.0}) EXPECT_EQ(model.getDesiredState(targets, time), second) << "time " << time;

  // Knots of the same rotation and opposite signs: no turn at all in between.
  vector_t antipodal = first;
  antipodal.segment<4>(3) *= -1.0;
  const TargetTrajectories doubleCover({0.0, 1.0}, {first, antipodal}, {input, input});
  for (const scalar_t time : {0.25, 0.5, 0.75}) {
    EXPECT_LT(rotationDistance(model.getBaseOrientation(model.getDesiredState(doubleCover, time)), model.getBaseOrientation(first)),
              1.0e-15)
        << "time " << time;
  }
}

TEST(WBRobotModelBoundary, TuningValuesMapOntoTheStateThroughTheEulerBoundary) {
  absl::StatusOr<LoadedRobot> robot = loadG1();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  const Eigen::Index joints = static_cast<Eigen::Index>(model.getJointDim());
  // A tuning vector with distinct entries: (yaw, pitch, roll) at rows 3 to 5 and (w_z, w_y, w_x) at 9 + nj.
  vector_t tuning = vector_t::LinSpaced(static_cast<Eigen::Index>(model.getStateTangentDim()), -0.7, 0.9);
  tuning.segment<3>(3) = vector3_t(kLargeYaws[1], 0.2, -0.1);
  tuning.segment<3>(9 + joints) = vector3_t(0.3, 0.2, 0.1);
  const absl::StatusOr<vector_t> state = model.stateFromTuning(tuning);
  ASSERT_TRUE(state.ok()) << state.status();
  ASSERT_EQ(static_cast<size_t>(state->size()), model.getStateDim());
  EXPECT_EQ(vector3_t(state->head<3>()), vector3_t(tuning.head<3>()));
  EXPECT_EQ(vector4_t(state->segment<4>(3)), quaternionFromEulerZyx(vector3_t(tuning.segment<3>(3))));
  EXPECT_EQ(vector_t(state->segment(7, joints)), vector_t(tuning.segment(6, joints)));
  EXPECT_EQ(vector3_t(state->segment<3>(linearVelocityStart(model))), vector3_t(tuning.segment<3>(6 + joints)));
  EXPECT_EQ(model.getBaseAngularVelocityLocal(*state), vector3_t(0.1, 0.2, 0.3)) << "the body rate (w_x, w_y, w_z)";
  EXPECT_EQ(vector_t(state->tail(joints)), vector_t(tuning.tail(joints)));

  // Back: the same tuning vector, its yaw modulo 2 pi.
  const absl::StatusOr<vector_t> back = model.tuningFromState(*state);
  ASSERT_TRUE(back.ok()) << back.status();
  vector_t expected = tuning;
  expected(3) = wrapped(tuning(3));
  EXPECT_LT((*back - expected).cwiseAbs().maxCoeff(), 1.0e-12);
  EXPECT_EQ(vector_t(back->tail(back->size() - 6)), vector_t(expected.tail(expected.size() - 6)));

  // The task file's initial state: level at yaw 0, the identity quaternion.
  const absl::StatusOr<vector_t> initialState = stateValuesFromConfig(
      robot->task.initial_state, stateInputLayout(*robot->modelSettings, StateInputLayout::Mpc::kWholeBody), "initial_state");
  ASSERT_TRUE(initialState.ok()) << initialState.status();
  ASSERT_EQ(vector3_t(initialState->segment<3>(3)), vector3_t::Zero());
  const absl::StatusOr<vector_t> mapped = model.stateFromTuning(*initialState);
  ASSERT_TRUE(mapped.ok()) << mapped.status();
  EXPECT_EQ(model.getBaseOrientation(*mapped), vector4_t(0.0, 0.0, 0.0, 1.0));
  EXPECT_EQ(model.getBasePosition(*mapped), vector3_t(initialState->head<3>()));
  EXPECT_EQ(model.getJointAngles(*mapped), vector_t(initialState->segment(6, joints)));

  const absl::StatusOr<vector_t> tooLong = model.stateFromTuning(vector_t::Zero(static_cast<Eigen::Index>(model.getStateTangentDim() + 1)));
  ASSERT_FALSE(tooLong.ok());
  EXPECT_EQ(tooLong.status().code(), absl::StatusCode::kInvalidArgument);
  for (const std::string& part : {std::string("stateFromTuning"), std::string("whole-body tuning layout"),
                                  std::string("base_angular_velocity 3 (body w_z, w_y, w_x)")}) {
    EXPECT_TRUE(absl::StrContains(tooLong.status().message(), part)) << "'" << part << "' missing from: " << tooLong.status();
  }
  // The state has one row more than the tuning layout: a tuning-sized vector is not a state.
  const absl::StatusOr<vector_t> tuningSized = model.tuningFromState(vector_t::Zero(static_cast<Eigen::Index>(model.getStateTangentDim())));
  ASSERT_FALSE(tuningSized.ok());
  EXPECT_EQ(tuningSized.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(tuningSized.status().message(), "base_orientation 4 (quaternion x, y, z, w)")) << tuningSized.status();
}

TEST(WBRobotModelBoundary, TheBaseOrientationIsTheStoredQuaternionSafelyNormalizedItsSignKept) {
  absl::StatusOr<LoadedRobot> robot = loadG1();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  vector_t neutral = model.getNeutralState();
  EXPECT_EQ(model.getBaseOrientation(neutral), vector4_t(0.0, 0.0, 0.0, 1.0));
  neutral.segment<4>(3).setZero();
  EXPECT_TRUE(neutral.isZero(0.0)) << "the neutral state is at rest at the origin, every joint at 0";

  std::mt19937 generator(41);
  for (int sample = 0; sample < 16; ++sample) {
    const vector4_t xi = randomUnitQuaternion(generator);
    for (const scalar_t scale : {1.0, 2.5, 0.4, -1.0, -3.0}) {
      SCOPED_TRACE(scale);
      const vector_t state = stateWith(model, scale * xi, vector3_t::Zero());
      const vector4_t read = model.getBaseOrientation(state);
      EXPECT_EQ(read, safelyNormalizedQuaternion(vector4_t(scale * xi)));
      EXPECT_NEAR(read.norm(), 1.0, 1.0e-15);
      EXPECT_LT((read - (scale > 0.0 ? xi : vector4_t(-xi))).cwiseAbs().maxCoeff(), 1.0e-15) << "the sign is kept";
      // Pinocchio's configuration carries the same normalized quaternion.
      EXPECT_EQ(vector4_t(model.getGeneralizedCoordinates(state).segment<4>(3)), read);
      // The heading and the Euler angles are of the rotation, whichever sign and scale it is stored with.
      EXPECT_EQ(model.getBaseHeading(state), twistHeading(read));
      EXPECT_NEAR(model.getBaseHeading(state), model.getBaseHeading(stateWith(model, xi, vector3_t::Zero())), 1.0e-12);
      EXPECT_GT(model.getBaseHeading(state), -M_PI);
      EXPECT_LE(model.getBaseHeading(state), M_PI);
      EXPECT_LT(rotationDistance(quaternionFromEulerZyx(model.getBaseOrientationEulerZyx(state)), xi), 1.0e-14);

      // Written: normalized, its sign kept, nothing else moved.
      vector_t written = model.getNeutralState();
      written.setConstant(0.25);
      model.setBaseOrientation(written, scale * xi);
      EXPECT_EQ(vector4_t(written.segment<4>(3)), safelyNormalizedQuaternion(vector4_t(scale * xi)));
      EXPECT_EQ(written.head<3>(), vector3_t::Constant(0.25));
      const Eigen::Index rest = static_cast<Eigen::Index>(model.getStateDim() - model.getJointStartindex());
      EXPECT_EQ(written.tail(rest), vector_t::Constant(rest, 0.25));
    }
  }
}

TEST(WBRobotModelBoundary, TheBaseAngularVelocityIsTheStoredBodyRate) {
  absl::StatusOr<LoadedRobot> robot = loadG1();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  std::mt19937 generator(42);
  std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
  for (int sample = 0; sample < 16; ++sample) {
    const vector3_t angularVelocity(uniform(generator), uniform(generator), uniform(generator));
    const vector_t state = stateWith(model, 1.3 * randomUnitQuaternion(generator), angularVelocity);

    // Read and written as stored, in the base frame.
    EXPECT_EQ(model.getBaseAngularVelocityLocal(state), angularVelocity);
    vector_t written = state;
    written.segment<3>(angularVelocityStart(model)).setZero();
    model.setBaseAngularVelocityLocal(written, angularVelocity);
    EXPECT_EQ(written, state);

    // A turn at a world yaw rate: the body rate R(xi)^T (0, 0, yawRate), at the orientation the state holds.
    vector_t turning = state;
    model.setBaseAngularVelocityFromWorldYawRate(turning, /*yawRate=*/0.8);
    const matrix3_t rotation = rotationOf(model.getBaseOrientation(turning));
    EXPECT_LT((model.getBaseAngularVelocityLocal(turning) - rotation.transpose() * vector3_t(0.0, 0.0, 0.8)).cwiseAbs().maxCoeff(),
              1.0e-15);
    EXPECT_LT((rotation * model.getBaseAngularVelocityLocal(turning) - vector3_t(0.0, 0.0, 0.8)).cwiseAbs().maxCoeff(), 1.0e-15);
    turning.segment<3>(angularVelocityStart(model)) = angularVelocity;
    EXPECT_EQ(turning, state) << "nothing but the angular velocity moved";

    // Its Jacobian: the angular velocity slots alone, against finite differences in every slot of the state.
    const BaseAngularVelocityJacobian jacobian = model.getBaseAngularVelocityLocalJacobian(state);
    ASSERT_EQ(jacobian.orientationSlots.rows(), 3);
    ASSERT_EQ(jacobian.orientationSlots.cols(), 4);
    EXPECT_TRUE(jacobian.orientationSlots.isZero(0.0));
    EXPECT_TRUE(jacobian.angularVelocitySlots == matrix3_t::Identity());
    for (Eigen::Index j = 0; j < state.size(); ++j) {
      vector_t plus = state;
      vector_t minus = state;
      plus(j) += kFiniteDifferenceStep;
      minus(j) -= kFiniteDifferenceStep;
      const vector3_t column =
          (model.getBaseAngularVelocityLocal(plus) - model.getBaseAngularVelocityLocal(minus)) / (2.0 * kFiniteDifferenceStep);
      const Eigen::Index row = j - angularVelocityStart(model);
      vector3_t expected = vector3_t::Zero();
      if (row >= 0 && row < 3) expected = jacobian.angularVelocitySlots.col(row);
      EXPECT_LT((column - expected).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance) << "slot " << j;
    }
  }
}

TEST(WBRobotModelBoundaryDeathTest, TheBaseTwistDiesOnJointVelocitiesOfAnotherSize) {
  // Not read here, but held to the interface's size as the centroidal model, which reads them, holds them.
  absl::StatusOr<LoadedRobot> robot = loadG1();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  const vector_t state = model.getNeutralState();
  EXPECT_DEATH(model.getBaseTwistInWorldFrame(state, vector_t::Zero(static_cast<Eigen::Index>(model.getJointDim()) + 1)),
               "one joint velocity per MPC joint");
}

TEST(WBRobotModelBoundary, TheBaseTwistIsTheWorldRateOfTheStoredVelocities) {
  absl::StatusOr<LoadedRobot> robot = loadG1();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  std::mt19937 generator(43);
  std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
  for (int sample = 0; sample < 16; ++sample) {
    const vector3_t angularVelocity(uniform(generator), uniform(generator), uniform(generator));
    const vector_t state = stateWith(model, 0.8 * randomUnitQuaternion(generator), angularVelocity);

    // [dp_W, R(xi) w_B]: the stored linear velocity and the body rate turned into the world. The state holds the joint
    // velocities, so the ones passed are not read.
    const vector_t jointVelocities = vector_t::Constant(static_cast<Eigen::Index>(model.getJointDim()), uniform(generator));
    const vector6_t twist = model.getBaseTwistInWorldFrame(state, jointVelocities);
    EXPECT_EQ(model.getBaseTwistInWorldFrame(state, vector_t::Zero(static_cast<Eigen::Index>(model.getJointDim()))), twist);
    EXPECT_EQ(vector3_t(twist.head<3>()), vector3_t(state.segment<3>(linearVelocityStart(model))));
    EXPECT_LT((twist.tail<3>() - rotationOf(model.getBaseOrientation(state)) * angularVelocity).cwiseAbs().maxCoeff(), 1.0e-15);

    // A knot's command is its twist: the planar velocity and the turn about the world z axis; no inertia is read.
    const vector3_t command = model.getTargetPlanarCommand(state, /*compositeYawInertia=*/0.0);
    EXPECT_EQ(command, vector3_t(twist(0), twist(1), twist(5)));
    EXPECT_EQ(model.getTargetPlanarCommand(state, /*compositeYawInertia=*/7.0), command);
  }

  // A velocity command knot, level at any yaw: its command is what was written.
  for (const scalar_t yaw : {0.0, 2.5, kLargeYaws[0], kLargeYaws[1]}) {
    SCOPED_TRACE(yaw);
    vector_t knot = model.getNeutralState();
    model.setBasePoseEulerZyx(knot, (vector6_t() << 1.0, 2.0, 0.7, yaw, 0.0, 0.0).finished());
    model.setBaseComLinearVelocity(knot, vector3_t(0.3, -0.1, 0.0));
    model.setBaseAngularVelocityFromWorldYawRate(knot, /*yawRate=*/0.6);
    EXPECT_LT((model.getTargetPlanarCommand(knot, /*compositeYawInertia=*/0.0) - vector3_t(0.3, -0.1, 0.6)).cwiseAbs().maxCoeff(), 1.0e-15);
  }
}

TEST(WBRobotModelBoundary, TheHeadingIsRegularAtAPitchOf90Degrees) {
  absl::StatusOr<LoadedRobot> robot = loadG1();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  // At gimbal lock only yaw - roll is a rotation about the world z axis, and the twist reads it: 0.3 - 2 atan(tan(pi / 4)
  // tan(0.1)) = 0.1, where the ZYX yaw is not defined; the same attitude with the yaw and the roll moved together.
  for (const vector3_t& eulerZyx : {vector3_t(0.3, M_PI / 2.0, 0.2), vector3_t(0.5, M_PI / 2.0, 0.4)}) {
    SCOPED_TRACE(eulerZyx.transpose());
    vector_t state = model.getNeutralState();
    model.setBaseOrientationEulerZyx(state, eulerZyx);
    EXPECT_NEAR(model.getBaseHeading(state), 0.1, 1.0e-12);
    // The stored attitude is regular there: its Euler angles read back the same rotation.
    EXPECT_TRUE(model.getBasePoseEulerZyx(state).allFinite());
    EXPECT_LT(rotationDistance(quaternionFromEulerZyx(model.getBaseOrientationEulerZyx(state)), model.getBaseOrientation(state)), 1.0e-12);
  }
}

TEST(WBRobotModelBoundary, TheTangentMapsAreTheirDerivatives) {
  absl::StatusOr<LoadedRobot> robot = loadG1();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  const pinocchio::Model& pinocchioModel = robot->pinocchioInterface->getModel();
  const Eigen::Index orientationStart = static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  std::mt19937 generator(44);
  for (int sample = 0; sample < 8; ++sample) {
    // A stored quaternion off the unit sphere, as the solver's iterates are between projections.
    const scalar_t scale = 0.8 + 0.05 * static_cast<scalar_t>(sample);
    const vector_t state = stateWith(model, scale * randomUnitQuaternion(generator), vector3_t(0.1, 0.2, 0.3));
    const vector4_t xi = state.segment<4>(orientationStart);

    // The body rotation of a step of the four orientation slots, E^+(xi) = 2 G(xi)^T / |xi|^2.
    const matrix_t rotationJacobian = model.getBaseRotationTangentJacobian(state);
    ASSERT_EQ(rotationJacobian.rows(), 3);
    ASSERT_EQ(rotationJacobian.cols(), 4);
    const matrix3_t rotation = rotationOf(model.getBaseOrientation(state));
    for (Eigen::Index j = 0; j < 4; ++j) {
      vector_t plus = state;
      vector_t minus = state;
      plus(orientationStart + j) += kFiniteDifferenceStep;
      minus(orientationStart + j) -= kFiniteDifferenceStep;
      const vector3_t column = (bodyRotation(rotation, rotationOf(model.getBaseOrientation(plus))) -
                                bodyRotation(rotation, rotationOf(model.getBaseOrientation(minus)))) /
                               (2.0 * kFiniteDifferenceStep);
      EXPECT_LT((column - rotationJacobian.col(j)).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance) << "column " << j;
    }
    // A radial step does not rotate, and the quaternion rate 1/2 G(xi) w of the flow map is the body rate w.
    EXPECT_LT((rotationJacobian * xi).cwiseAbs().maxCoeff(), 1.0e-15);
    const vector3_t angularVelocity(0.3, -0.7, 1.1);
    EXPECT_LT((rotationJacobian * (0.5 * quaternionRateMatrix<scalar_t>(xi) * angularVelocity) - angularVelocity).cwiseAbs().maxCoeff(),
              1.0e-14);

    // Pinocchio's tangent of q: pinocchio::difference() of the configurations of two states, column by column.
    const matrix_t pinocchioJacobian = model.getPinocchioTangentJacobian(state);
    ASSERT_EQ(pinocchioJacobian.rows(), static_cast<Eigen::Index>(model.getVelocityDim()));
    ASSERT_EQ(pinocchioJacobian.cols(), static_cast<Eigen::Index>(model.getStateDim()));
    for (Eigen::Index j = 0; j < pinocchioJacobian.cols(); ++j) {
      vector_t plus = state;
      vector_t minus = state;
      plus(j) += kFiniteDifferenceStep;
      minus(j) -= kFiniteDifferenceStep;
      const vector_t column =
          pinocchio::difference(pinocchioModel, model.getGeneralizedCoordinates(minus), model.getGeneralizedCoordinates(plus)) /
          (2.0 * kFiniteDifferenceStep);
      EXPECT_LT((column - pinocchioJacobian.col(j)).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance) << "column " << j;
    }
  }
}

TEST(WBRobotModelBoundary, TheMeasurementHelpersFillPinocchiosCoordinatesWithoutAllocating) {
  // The counter counts this thread's allocations: a vector on the heap is one.
  const size_t probe = robot::realtime::heapAllocationCountOnThisThread();
  const vector_t onTheHeap = vector_t::Constant(64, 1.0);
  EXPECT_EQ(onTheHeap.sum(), 64.0);
  EXPECT_GT(robot::realtime::heapAllocationCountOnThisThread(), probe);

  absl::StatusOr<LoadedRobot> robot = loadG1();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const WBAccelMpcRobotModel<scalar_t>& model = *robot->model;
  const Eigen::Index joints = static_cast<Eigen::Index>(model.getJointDim());
  const Eigen::Index configurationDim = static_cast<Eigen::Index>(model.getConfigurationDim());
  const Eigen::Index velocityDim = static_cast<Eigen::Index>(model.getVelocityDim());
  vector_t state = model.getNeutralState();
  vector_t expected = state;
  vector_t zeroInput = vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));
  std::mt19937 generator(45);
  std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
  for (int sample = 0; sample < 16; ++sample) {
    // A measured attitude of any sign, unit to round-off (the estimator's), at any pitch.
    const vector4_t measured = (1.0 + 1.0e-9 * uniform(generator)) * randomUnitQuaternion(generator);
    const vector3_t position(uniform(generator), uniform(generator), 1.0 + uniform(generator));
    const vector3_t linearVelocityLocal(uniform(generator), uniform(generator), uniform(generator));
    const vector3_t angularVelocityLocal(uniform(generator), uniform(generator), uniform(generator));
    const vector_t jointPositions = vector_t::Random(joints);
    const vector_t jointVelocities = vector_t::Random(joints);

    // Into the state [q, v] directly.
    const size_t before = robot::realtime::heapAllocationCountOnThisThread();
    model.fillGeneralizedCoordinatesFromMeasurement(position, measured, jointPositions, state.head(configurationDim));
    model.fillGeneralizedVelocitiesFromMeasurement(measured, linearVelocityLocal, angularVelocityLocal, jointVelocities,
                                                   state.tail(velocityDim));
    EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0U);

    // q = [p_W, xi_hat, q_j] and v = [R(xi_hat) v_B, w_B, dq_j] of the Composite(Translation, Spherical) root.
    const vector4_t normalized = safelyNormalizedQuaternion(measured);
    model.setBasePosition(expected, position);
    expected.segment<4>(3) = normalized;
    model.setJointAngles(expected, jointPositions);
    model.setBaseLinearVelocity(expected, quaternion_t(normalized) * linearVelocityLocal);
    model.setBaseAngularVelocityLocal(expected, angularVelocityLocal);
    model.setJointVelocities(expected, zeroInput, jointVelocities);
    EXPECT_EQ(state, expected);
    EXPECT_EQ(vector4_t(state.segment<4>(3)), normalized);
    EXPECT_NEAR(state.segment<4>(3).norm(), 1.0, 1.0e-15);
    EXPECT_TRUE(pinocchio::isNormalized(robot->pinocchioInterface->getModel(), vector_t(state.head(configurationDim)), /*prec=*/1.0e-14));
  }
}

}  // namespace
}  // namespace ocs2::humanoid
