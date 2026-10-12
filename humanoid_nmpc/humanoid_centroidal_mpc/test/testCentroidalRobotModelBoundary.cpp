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
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"
#include "support/TypedConfigFiles.h"

/*
 * The layout-agnostic API of the centroidal robot model (MpcRobotModelBase) on its quaternion layout [h, p_W, xi, q_j],
 * for the wrench-input model of the G1 and the Atlas and for the Atlas's basis-vector decorator: the sizes and indices
 * are the quaternion layout's and the state lives on a manifold of one more ambient than tangent coordinate; the Euler
 * boundary accessors round-trip a pose modulo 2 pi and an attitude to round-off; interpolation slerps the attitude along
 * the shorter arc, across +-pi; the heading is the twist of the attitude in (-pi, pi]; the tuning layout's Euler rows map
 * onto the quaternion; the tangent maps are the derivatives they name; the base twist is the base velocity of the
 * momentum at the measured joint velocities; a knot's command reads its momentum back; and the measurement helpers write
 * Pinocchio's q = [p_W, xi_hat, q_j] and v = [v_W, w_B, dq_j] without allocating.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
constexpr scalar_t kFiniteDifferenceTolerance = 1.0e-8;
/** [rad] The rotation an Euler round trip of an attitude may differ by: 18 machine epsilons (the round-trip test). */
constexpr scalar_t kRoundTripRotationTolerance = 4.0e-15;
/** The yaws of the round trips: beyond +-pi, where the quaternion's Euler angles read them back wrapped. */
constexpr std::array<scalar_t, 2> kLargeYaws = {3.0 * M_PI / 2.0, -5.0 * M_PI / 2.0};

/** The centroidal model's input parameterization under test. */
enum class Formulation {
  /** CentroidalMpcRobotModel, world-frame wrenches. */
  kWrenches,
  /** CentroidalMpcRobotModel behind BasisInputsModelDecorator. */
  kBasisVectors,
};

/** A centroidal robot model of a shipped configuration with what it refers to, which outlives it here. */
struct LoadedRobot {
  std::string name;
  CentroidalMpcConfig config;
  std::unique_ptr<ModelSettings> modelSettings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;
  std::unique_ptr<CentroidalModelInfo> info;
  std::unique_ptr<MpcRobotModelBase<scalar_t>> model;
};

/**
 * The robot model of `files` as CentroidalMpcInterface builds it (without its solver), in `formulation`; with the
 * centroidal model `centroidalModel` (task.textproto centroidal_model) instead of the file's when it is not empty.
 */
absl::StatusOr<LoadedRobot> loadRobot(absl::string_view name,
                                      const CentroidalRobotFiles& files,
                                      Formulation formulation,
                                      absl::string_view centroidalModel = "") {
  LoadedRobot robot;
  robot.name = std::string(name);
  ASSIGN_OR_RETURN(robot.config, loadConfigOf(files));
  if (!centroidalModel.empty()) robot.config.task.centroidal_model = std::string(centroidalModel);
  ASSIGN_OR_RETURN(ModelSettings modelSettings,
                   ModelSettings::Create(robot.config.task, files.urdfFile, "boundary_test", /*verbose=*/false));
  robot.modelSettings = std::make_unique<ModelSettings>(std::move(modelSettings));
  ASSIGN_OR_RETURN(PinocchioInterface pinocchioInterface,
                   loadCustomPinocchioInterface(robot.config.task, files.urdfFile, *robot.modelSettings));
  robot.pinocchioInterface = std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));
  ASSIGN_OR_RETURN(CentroidalModelInfo info, centroidalModelInfoOf(robot.config, *robot.pinocchioInterface, *robot.modelSettings));
  robot.info = std::make_unique<CentroidalModelInfo>(std::move(info));
  std::unique_ptr<MpcRobotModelBase<scalar_t>> wrenchModel =
      std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*robot.modelSettings, *robot.pinocchioInterface, *robot.info);
  if (formulation == Formulation::kWrenches) {
    robot.model = std::move(wrenchModel);
    return robot;
  }
  ASSIGN_OR_RETURN(const feet_array_t<ContactWrenchConeBasisMatrix> bases,
                   contactWrenchConeBasesFromConfig(robot.config.task.contacts, *robot.modelSettings));
  robot.model = std::make_unique<BasisInputsModelDecorator<scalar_t>>(std::move(wrenchModel), bases, *robot.pinocchioInterface);
  return robot;
}

/** The G1 and the Atlas with wrench inputs, and the Atlas with basis-vector inputs (its shipped formulation). */
std::vector<LoadedRobot> loadRobots() {
  std::vector<LoadedRobot> robots;
  for (const std::pair<std::string, Formulation>& robot :
       {std::pair<std::string, Formulation>("g1", Formulation::kWrenches),
        std::pair<std::string, Formulation>("atlas", Formulation::kWrenches),
        std::pair<std::string, Formulation>("atlas_basis_vectors", Formulation::kBasisVectors)}) {
    absl::StatusOr<LoadedRobot> loaded = loadRobot(robot.first, robot.first == "g1" ? g1Files() : atlasFiles(), robot.second);
    if (!loaded.ok()) {
      ADD_FAILURE() << robot.first << ": " << loaded.status();
      continue;
    }
    robots.push_back(*std::move(loaded));
  }
  return robots;
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

/** A state of `model` with distinct entries and the base orientation of the ZYX Euler angles `eulerZyx` (yaw, pitch, roll). */
vector_t stateWithEulerAngles(const MpcRobotModelBase<scalar_t>& model, const vector3_t& eulerZyx) {
  vector_t state = vector_t::LinSpaced(static_cast<Eigen::Index>(model.getStateDim()), -0.7, 0.9);
  state.segment<4>(static_cast<Eigen::Index>(model.getBaseOrientationStartIndex())) = quaternionFromEulerZyx(eulerZyx);
  return state;
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
  return quaternion_t(xi).toRotationMatrix();
}

/** The angle of the rotation from the attitude `xi0` to the attitude `xi1`: |Log(conj(xi0) xi1)|, on the shorter arc. */
scalar_t rotationAngleBetween(const vector4_t& xi0, const vector4_t& xi1) {
  return quaternionLog<scalar_t>(quaternionProduct<scalar_t>(quaternionConjugate<scalar_t>(xi0), xi1)).norm();
}

TEST(CentroidalRobotModelBoundary, SizesAndIndicesAreThoseOfTheQuaternionLayout) {
  for (const LoadedRobot& robot : loadRobots()) {
    SCOPED_TRACE(robot.name);
    const MpcRobotModelBase<scalar_t>& model = *robot.model;
    const size_t joints = model.getJointDim();
    EXPECT_EQ(model.getStateDim(), 13 + joints);
    EXPECT_EQ(model.getStateTangentDim(), 12 + joints);
    EXPECT_EQ(model.getConfigurationDim(), 7 + joints);
    EXPECT_EQ(model.getVelocityDim(), 6 + joints);
    EXPECT_EQ(model.getBaseStartindex(), 6U);
    EXPECT_EQ(model.getBaseOrientationStartIndex(), 9U);
    EXPECT_EQ(model.getJointStartindex(), 13U);
    EXPECT_EQ(model.getConfigurationDim(), static_cast<size_t>(robot.pinocchioInterface->getModel().nq));
    EXPECT_EQ(model.getVelocityDim(), static_cast<size_t>(robot.pinocchioInterface->getModel().nv));

    // The centroidal model info under the same names, carried to CppAD.
    EXPECT_EQ(robot.info->getConfigurationDim(), model.getConfigurationDim());
    EXPECT_EQ(robot.info->getVelocityDim(), model.getVelocityDim());
    EXPECT_EQ(robot.info->getStateTangentDim(), model.getStateTangentDim());
    EXPECT_EQ(robot.info->stateDim, model.getStateDim());
    const CentroidalModelInfoCppAd infoCppAd = robot.info->toCppAd();
    EXPECT_EQ(infoCppAd.getConfigurationDim(), model.getConfigurationDim());
    EXPECT_EQ(infoCppAd.getVelocityDim(), model.getVelocityDim());
    EXPECT_EQ(infoCppAd.getStateTangentDim(), model.getStateTangentDim());

    // The CppAD model has the same layout.
    const PinocchioInterfaceCppAd pinocchioInterfaceCppAd = robot.pinocchioInterface->toCppAd();
    const CentroidalMpcRobotModel<ad_scalar_t> modelCppAd(*robot.modelSettings, pinocchioInterfaceCppAd, infoCppAd);
    EXPECT_EQ(modelCppAd.getStateDim(), model.getStateDim());
    EXPECT_EQ(modelCppAd.getStateTangentDim(), model.getStateTangentDim());
    EXPECT_EQ(modelCppAd.getConfigurationDim(), model.getConfigurationDim());
    EXPECT_EQ(describeTuningLayout(modelCppAd.getTuningLayout()), describeTuningLayout(model.getTuningLayout()));

    // The state lives on [E(9), unit quaternion, E(nj)]: one ambient coordinate more than its tangent.
    const absl::StatusOr<std::shared_ptr<const StateManifold>> manifold = stateManifoldOf(model);
    ASSERT_TRUE(manifold.ok()) << manifold.status();
    ASSERT_NE(*manifold, nullptr);
    EXPECT_EQ((*manifold)->getAmbientDim(), model.getStateDim());
    EXPECT_EQ((*manifold)->getTangentDim(), model.getStateTangentDim());
  }
}

TEST(CentroidalRobotModelBoundary, TheTuningLayoutAgreesWithTheIndexAccessors) {
  for (const LoadedRobot& robot : loadRobots()) {
    SCOPED_TRACE(robot.name);
    const MpcRobotModelBase<scalar_t>& model = *robot.model;
    const TuningLayout& layout = model.getTuningLayout();
    EXPECT_EQ(layout.name, "centroidal");
    EXPECT_EQ(getTuningLayoutDim(layout), model.getStateTangentDim());
    EXPECT_EQ(getTuningLayoutStateDim(layout), model.getStateDim());
    const std::vector<std::string> names = {"normalized_linear_momentum", "normalized_angular_momentum", "base_position",
                                            "base_orientation", "joint_positions"};
    ASSERT_EQ(layout.segments.size(), names.size());
    for (size_t i = 0; i < names.size(); ++i) {
      const TuningSegment& segment = layout.segments[i];
      EXPECT_EQ(segment.name, names[i]);
      const absl::StatusOr<size_t> tuningStart = tuningSegmentStart(layout, segment.name);
      ASSERT_TRUE(tuningStart.ok()) << tuningStart.status();
      // The blocks before the quaternion sit at the same offset in the state and the tuning vector, the ones after it
      // one row further in the state: the quaternion has four state rows for its three tuning rows.
      const bool afterTheQuaternion = i > 3;
      EXPECT_EQ(segment.stateStart, *tuningStart + (afterTheQuaternion ? 1 : 0)) << segment.name;
      EXPECT_EQ(segment.stateDim, segment.name == "base_orientation" ? kQuaternionOrientationStateDim : segment.tuningDim) << segment.name;
    }
    const TuningSegment* absl_nullable position = findSegment(layout, "base_position");
    const TuningSegment* absl_nullable orientation = findSegment(layout, "base_orientation");
    const TuningSegment* absl_nullable joints = findSegment(layout, "joint_positions");
    ASSERT_NE(position, nullptr);
    ASSERT_NE(orientation, nullptr);
    ASSERT_NE(joints, nullptr);
    EXPECT_EQ(position->stateStart, model.getBaseStartindex());
    EXPECT_EQ(orientation->stateStart, model.getBaseOrientationStartIndex());
    EXPECT_EQ(orientation->kind, TuningSegmentKind::kEulerZyxOrientation);
    EXPECT_EQ(orientation->tuningDim, 3U);
    EXPECT_EQ(joints->stateStart, model.getJointStartindex());
    EXPECT_EQ(joints->tuningDim, model.getJointDim());

    // The accessors write where the layout says.
    vector_t state = model.getNeutralState();
    model.setBasePosition(state, vector3_t(1.0, 2.0, 3.0));
    model.setBaseOrientationEulerZyx(state, vector3_t(0.4, 0.5, 0.6));
    model.setJointAngles(state, vector_t::Constant(static_cast<Eigen::Index>(model.getJointDim()), 7.0));
    EXPECT_EQ(vector3_t(state.segment<3>(static_cast<Eigen::Index>(position->stateStart))), vector3_t(1.0, 2.0, 3.0));
    EXPECT_EQ(vector4_t(state.segment<4>(static_cast<Eigen::Index>(orientation->stateStart))),
              quaternionFromEulerZyx(vector3_t(0.4, 0.5, 0.6)));
    EXPECT_EQ(state.tail(static_cast<Eigen::Index>(joints->tuningDim)),
              vector_t::Constant(static_cast<Eigen::Index>(joints->tuningDim), 7.0));
  }
}

TEST(CentroidalRobotModelBoundary, TheEulerBoundaryAccessorsRoundTripThePoseModuloTwoPiAndTheAttitudeToRoundOff) {
  std::mt19937 generator(30);
  std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
  for (const LoadedRobot& robot : loadRobots()) {
    SCOPED_TRACE(robot.name);
    const MpcRobotModelBase<scalar_t>& model = *robot.model;
    const Eigen::Index joints = static_cast<Eigen::Index>(model.getJointDim());
    for (const scalar_t yaw : kLargeYaws) {
      SCOPED_TRACE(yaw);
      const vector_t state = stateWithEulerAngles(model, vector3_t(0.2, 0.12, -0.31));

      // get(set(pose)) returns the pose, the yaw modulo 2 pi, to 1e-12; nothing else moves.
      const vector6_t pose = (vector6_t() << 0.3, -0.2, 0.91, yaw, -0.05, 0.07).finished();
      vector_t written = state;
      model.setBasePoseEulerZyx(written, pose);
      const vector6_t read = model.getBasePoseEulerZyx(written);
      EXPECT_EQ(vector3_t(read.head<3>()), vector3_t(pose.head<3>()));
      EXPECT_NEAR(wrapped(read(3) - yaw), 0.0, 1.0e-12);
      EXPECT_GT(read(3), -M_PI);
      EXPECT_LE(read(3), M_PI);
      EXPECT_NEAR(read(4), pose(4), 1.0e-12);
      EXPECT_NEAR(read(5), pose(5), 1.0e-12);
      EXPECT_EQ(model.getBaseOrientationEulerZyx(written), vector3_t(read.tail<3>()));
      // The heading is not the yaw: it is the twist of the attitude (design decision D9), wrapped.
      EXPECT_NEAR(model.getBaseHeading(written), twistHeadingOfEulerZyx(pose.tail<3>()), 1.0e-12);
      EXPECT_EQ(written.head<6>(), state.head<6>());
      EXPECT_EQ(written.tail(joints), state.tail(joints));
      vector_t orientationOnly = state;
      model.setBaseOrientationEulerZyx(orientationOnly, pose.tail<3>());
      EXPECT_EQ(model.getBaseOrientationEulerZyx(orientationOnly), vector3_t(read.tail<3>()));
      EXPECT_EQ(model.getBasePosition(orientationOnly), model.getBasePosition(state));
    }

    // set(get(x)) reproduces the rotation to round-off (the quaternion itself perhaps negated: the same rotation), and
    // moves nothing else. The round trip goes through four transcendental functions (atan2, asin, sin, cos), each
    // within an ulp, and the products of the quaternion's composition: measured up to 1.6e-15 rad on these samples, held
    // to 4e-15 rad (18 machine epsilons). Away from gimbal lock (|pitch| <= 80 degrees), where the Euler angles amplify
    // the round-off by 1 / cos(pitch).
    for (int sample = 0; sample < 32; ++sample) {
      vector_t state = stateWithEulerAngles(model, vector3_t::Zero());
      const vector3_t eulerZyx(M_PI * uniform(generator), 1.4 * uniform(generator), M_PI * uniform(generator));
      const vector4_t attitude = (sample % 2 == 0 ? 1.0 : -1.0) * quaternionFromEulerZyx(eulerZyx);
      model.setBaseOrientation(state, attitude);
      vector_t copy = state;
      model.setBasePoseEulerZyx(copy, model.getBasePoseEulerZyx(state));
      EXPECT_LE(rotationAngleBetween(model.getBaseOrientation(copy), model.getBaseOrientation(state)), kRoundTripRotationTolerance)
          << attitude.transpose();
      vector_t orientationCopy = state;
      model.setBaseOrientationEulerZyx(orientationCopy, model.getBaseOrientationEulerZyx(state));
      EXPECT_LE(rotationAngleBetween(model.getBaseOrientation(orientationCopy), model.getBaseOrientation(state)),
                kRoundTripRotationTolerance)
          << attitude.transpose();
      const Eigen::Index orientationStart = static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
      EXPECT_EQ(copy.head(orientationStart), state.head(orientationStart));
      EXPECT_EQ(copy.tail(joints), state.tail(joints));
    }
  }
}

TEST(CentroidalRobotModelBoundary, InterpolationSlerpsTheAttitudeAcrossPi) {
  for (const LoadedRobot& robot : loadRobots()) {
    SCOPED_TRACE(robot.name);
    const MpcRobotModelBase<scalar_t>& model = *robot.model;
    vector_t first = model.getNeutralState();
    vector_t second = model.getNeutralState();
    model.setBasePoseEulerZyx(first, (vector6_t() << 0.0, 0.0, 0.9, 3.0, 0.0, 0.0).finished());
    model.setBasePoseEulerZyx(second, (vector6_t() << 0.4, 0.1, 0.9, 3.5, 0.0, 0.0).finished());
    const vector_t input = vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));
    const TargetTrajectories targets({0.0, 1.0}, {first, second}, {input, input});

    // The knots' headings straddle +-pi; the slerp turns the short way, by 0.5 rad, not the long way round.
    const vector_t midpoint = model.getDesiredState(targets, /*time=*/0.5);
    EXPECT_NEAR(model.getBaseHeading(midpoint), wrapped(3.25), 1.0e-12);
    EXPECT_NEAR(model.getBaseOrientationEulerZyx(midpoint)(0), wrapped(3.25), 1.0e-12);
    EXPECT_NEAR(model.getBaseOrientation(midpoint).norm(), 1.0, 1.0e-15);
    EXPECT_EQ(vector3_t(model.getBasePosition(midpoint)), vector3_t(0.2, 0.05, 0.9));
    // A quarter of the way: the slerp's heading is linear in time.
    EXPECT_NEAR(model.getBaseHeading(model.getDesiredState(targets, /*time=*/0.25)), wrapped(3.125), 1.0e-12);
    // The manifold's interpolation bit for bit, held outside the knots.
    const absl::StatusOr<std::shared_ptr<const StateManifold>> manifold = stateManifoldOf(model);
    ASSERT_TRUE(manifold.ok()) << manifold.status();
    ASSERT_NE(*manifold, nullptr);
    for (const scalar_t time : {0.0, 0.25, 0.5, 0.9, 1.0}) {
      EXPECT_EQ(model.getDesiredState(targets, time), (*manifold)->interpolate(first, second, time)) << "time " << time;
    }
    EXPECT_EQ(model.getDesiredState(targets, /*time=*/-1.0), first);
    EXPECT_EQ(model.getDesiredState(targets, /*time=*/2.0), second);
  }
}

TEST(CentroidalRobotModelBoundary, TuningValuesMapOntoTheQuaternionState) {
  for (const LoadedRobot& robot : loadRobots()) {
    SCOPED_TRACE(robot.name);
    const MpcRobotModelBase<scalar_t>& model = *robot.model;
    const Eigen::Index orientationStart = static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
    const Eigen::Index joints = static_cast<Eigen::Index>(model.getJointDim());
    // A tuning vector with distinct entries and (yaw, pitch, roll) = (3 pi / 2, 0.2, -0.1) on the orientation rows.
    vector_t tuning = vector_t::LinSpaced(static_cast<Eigen::Index>(model.getStateTangentDim()), -0.7, 0.9);
    const vector3_t eulerZyx(kLargeYaws[0], 0.2, -0.1);
    tuning.segment<3>(orientationStart) = eulerZyx;
    const absl::StatusOr<vector_t> state = model.stateFromTuning(tuning);
    ASSERT_TRUE(state.ok()) << state.status();
    ASSERT_EQ(state->size(), static_cast<Eigen::Index>(model.getStateDim()));
    // The Euclidean rows are copied bit for bit; the Euler rows become their quaternion.
    EXPECT_EQ(state->head(orientationStart), tuning.head(orientationStart));
    EXPECT_EQ(vector4_t(state->segment<4>(orientationStart)), quaternionFromEulerZyx(eulerZyx));
    EXPECT_EQ(state->tail(joints), tuning.tail(joints));
    const absl::StatusOr<vector_t> back = model.tuningFromState(*state);
    ASSERT_TRUE(back.ok()) << back.status();
    ASSERT_EQ(back->size(), tuning.size());
    EXPECT_EQ(back->head(orientationStart), tuning.head(orientationStart));
    EXPECT_EQ(back->tail(joints), tuning.tail(joints));
    const scalar_t yawBack = (*back)(orientationStart);
    EXPECT_NEAR(wrapped(yawBack - eulerZyx(0)), 0.0, 1.0e-12);
    EXPECT_NEAR((*back)(orientationStart + 1), eulerZyx(1), 1.0e-12);
    EXPECT_NEAR((*back)(orientationStart + 2), eulerZyx(2), 1.0e-12);

    // The task file's initial_state, as the interface builds its initial state: a unit quaternion of its angles.
    const absl::StatusOr<vector_t> initialTuning = initialTuningVectorOf(robot.config.task, *robot.modelSettings);
    ASSERT_TRUE(initialTuning.ok()) << initialTuning.status();
    ASSERT_EQ(initialTuning->size(), static_cast<Eigen::Index>(model.getStateTangentDim()));
    const absl::StatusOr<vector_t> initialState = model.stateFromTuning(*initialTuning);
    ASSERT_TRUE(initialState.ok()) << initialState.status();
    EXPECT_EQ(vector4_t(initialState->segment<4>(orientationStart)),
              quaternionFromEulerZyx(vector3_t(initialTuning->segment<3>(orientationStart))));
    EXPECT_NEAR(initialState->segment<4>(orientationStart).norm(), 1.0, 1.0e-15);

    // A vector of another size is refused, naming the layout.
    const absl::StatusOr<vector_t> tooLong = model.stateFromTuning(vector_t::Zero(static_cast<Eigen::Index>(model.getStateDim())));
    ASSERT_FALSE(tooLong.ok());
    EXPECT_EQ(tooLong.status().code(), absl::StatusCode::kInvalidArgument);
    for (const std::string& part :
         {std::string("stateFromTuning"), std::string("centroidal tuning layout"), std::string("base_orientation 3 (yaw, pitch, roll)")}) {
      EXPECT_TRUE(absl::StrContains(tooLong.status().message(), part)) << "'" << part << "' missing from: " << tooLong.status();
    }
    const absl::StatusOr<vector_t> tooShort = model.tuningFromState(vector_t::Zero(static_cast<Eigen::Index>(model.getStateTangentDim())));
    ASSERT_FALSE(tooShort.ok());
    EXPECT_EQ(tooShort.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(tooShort.status().message(), "tuningLayoutFromState")) << tooShort.status();
  }
}

TEST(CentroidalRobotModelBoundary, TheBaseOrientationIsTheStoredQuaternionSafelyNormalized) {
  for (const LoadedRobot& robot : loadRobots()) {
    SCOPED_TRACE(robot.name);
    const MpcRobotModelBase<scalar_t>& model = *robot.model;
    const Eigen::Index orientationStart = static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
    const vector_t neutral = model.getNeutralState();
    vector_t expectedNeutral = vector_t::Zero(static_cast<Eigen::Index>(model.getStateDim()));
    expectedNeutral(orientationStart + 3) = 1.0;
    EXPECT_EQ(neutral, expectedNeutral) << "at rest at the origin, the identity quaternion (x, y, z, w) = (0, 0, 0, 1)";
    EXPECT_EQ(model.getBaseOrientation(neutral), vector4_t(0.0, 0.0, 0.0, 1.0));

    for (const vector3_t& euler : {vector3_t(kLargeYaws[0], 0.12, -0.31), vector3_t(kLargeYaws[1], -0.4, 0.2), vector3_t(1.3, 0.05, 0.0)}) {
      SCOPED_TRACE(euler.transpose());
      const vector_t state = stateWithEulerAngles(model, euler);
      const vector4_t xi = model.getBaseOrientation(state);
      EXPECT_LT((xi - quaternionFromEulerZyx(euler)).cwiseAbs().maxCoeff(), 1.0e-16);
      EXPECT_NEAR(xi.norm(), 1.0, 1.0e-15);
      // The heading is the twist of the state's quaternion, in (-pi, pi]: the yaw, wrapped, less the coupling of the
      // pitch and the roll.
      EXPECT_EQ(model.getBaseHeading(state), twistHeading(xi));
      EXPECT_NEAR(model.getBaseHeading(state), twistHeadingOfEulerZyx(euler), 1.0e-12);
      EXPECT_GT(model.getBaseHeading(state), -M_PI);
      EXPECT_LE(model.getBaseHeading(state), M_PI);

      // Written as a measured attitude: the same rotation, nothing else moved.
      vector_t written = model.getNeutralState();
      written.setConstant(0.25);
      model.setBaseOrientation(written, xi);
      EXPECT_EQ(vector4_t(written.segment<4>(orientationStart)), safelyNormalizedQuaternion(xi));
      EXPECT_EQ(written.head(orientationStart), vector_t::Constant(orientationStart, 0.25));
      EXPECT_EQ(written.tail(static_cast<Eigen::Index>(model.getJointDim())),
                vector_t::Constant(static_cast<Eigen::Index>(model.getJointDim()), 0.25));
      const vector3_t read = model.getBaseOrientationEulerZyx(written);
      EXPECT_NEAR(read(0), wrapped(euler(0)), 1.0e-12);
      EXPECT_NEAR(read(1), euler(1), 1.0e-12);
      EXPECT_NEAR(read(2), euler(2), 1.0e-12);

      // The solver's quaternion between projections: read through its safe normalization, a scaled quaternion is the same
      // attitude, and so is the negated one (the double cover).
      for (const scalar_t scale : {0.5, 2.0, -1.0, -3.0}) {
        vector_t scaled = written;
        scaled.segment<4>(orientationStart) *= scale;
        EXPECT_LT((model.getBaseOrientation(scaled) - std::copysign(1.0, scale) * model.getBaseOrientation(written)).cwiseAbs().maxCoeff(),
                  1.0e-15)
            << "scale " << scale;
        EXPECT_LT((model.getBaseOrientationEulerZyx(scaled) - read).cwiseAbs().maxCoeff(), 1.0e-14) << "scale " << scale;
        EXPECT_NEAR(model.getBaseHeading(scaled), model.getBaseHeading(written), 1.0e-14) << "scale " << scale;
      }
    }
  }
}

TEST(CentroidalRobotModelBoundary, TheHeadingIsRegularAtAPitchOf90Degrees) {
  for (const LoadedRobot& robot : loadRobots()) {
    SCOPED_TRACE(robot.name);
    const MpcRobotModelBase<scalar_t>& model = *robot.model;
    // At gimbal lock only yaw - roll is a rotation about the world z axis, and the twist reads it: 0.3 - 2 atan(tan(pi / 4)
    // tan(0.1)) = 0.1, where the ZYX yaw is not defined.
    const vector_t state = stateWithEulerAngles(model, vector3_t(0.3, M_PI / 2.0, 0.2));
    EXPECT_NEAR(model.getBaseHeading(state), 0.1, 1.0e-12);
    // The same attitude, with the yaw and the roll moved together, has the same heading.
    const vector_t sameAttitude = stateWithEulerAngles(model, vector3_t(0.5, M_PI / 2.0, 0.4));
    EXPECT_NEAR(model.getBaseHeading(sameAttitude), 0.1, 1.0e-12);
  }
}

TEST(CentroidalRobotModelBoundary, TheTangentMapsAreTheirDerivatives) {
  std::mt19937 generator(31);
  std::uniform_real_distribution<scalar_t> angle(-0.5, 0.5);
  for (const LoadedRobot& robot : loadRobots()) {
    SCOPED_TRACE(robot.name);
    const MpcRobotModelBase<scalar_t>& model = *robot.model;
    const pinocchio::Model& pinocchioModel = robot.pinocchioInterface->getModel();
    for (int sample = 0; sample < 8; ++sample) {
      const vector_t state = stateWithEulerAngles(model, vector3_t(4.0 * angle(generator), angle(generator), angle(generator)));
      const Eigen::Index orientationStart = static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
      const vector4_t xi = model.getBaseOrientation(state);

      // The body rotation of a move dxi of the stored quaternion: R(x + dxi) = R(x) Exp(J dxi), J = E^+(xi) (3 x 4). The
      // radial direction xi does not rotate the body, and E^+ E = I on the tangent, E = G(xi) / 2.
      const matrix_t rotationJacobian = model.getBaseRotationTangentJacobian(state);
      ASSERT_EQ(rotationJacobian.rows(), 3);
      ASSERT_EQ(rotationJacobian.cols(), 4);
      const matrix3_t rotation = rotationOf(xi);
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
      EXPECT_LT((rotationJacobian * xi).cwiseAbs().maxCoeff(), 1.0e-15);
      const Eigen::Matrix<scalar_t, 4, 3> tangentMap = quaternionTangentMap<scalar_t>(xi);
      EXPECT_LT((rotationJacobian * tangentMap - matrix3_t::Identity()).cwiseAbs().maxCoeff(), 1.0e-15);

      // Pinocchio's tangent of q moves with the state as the map says: difference(q(x), q(x + dx)) = J dx.
      const matrix_t pinocchioJacobian = model.getPinocchioTangentJacobian(state);
      ASSERT_EQ(pinocchioJacobian.rows(), static_cast<Eigen::Index>(model.getVelocityDim()));
      ASSERT_EQ(pinocchioJacobian.cols(), static_cast<Eigen::Index>(model.getStateDim()));
      const vector_t q = model.getGeneralizedCoordinates(state);
      for (Eigen::Index j = 0; j < pinocchioJacobian.cols(); ++j) {
        vector_t plus = state;
        vector_t minus = state;
        plus(j) += kFiniteDifferenceStep;
        minus(j) -= kFiniteDifferenceStep;
        const vector_t column = (pinocchio::difference(pinocchioModel, q, model.getGeneralizedCoordinates(plus)) -
                                 pinocchio::difference(pinocchioModel, q, model.getGeneralizedCoordinates(minus))) /
                                (2.0 * kFiniteDifferenceStep);
        EXPECT_LT((column - pinocchioJacobian.col(j)).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance) << "column " << j;
      }
    }
  }
}

TEST(CentroidalRobotModelBoundary, TheMeasurementHelpersWritePinocchiosVectorsWithoutAllocating) {
  // The counter counts this thread's allocations: a vector on the heap is one.
  const size_t probe = robot::realtime::heapAllocationCountOnThisThread();
  const vector_t onTheHeap = vector_t::Constant(64, 1.0);
  EXPECT_EQ(onTheHeap.sum(), 64.0);
  EXPECT_GT(robot::realtime::heapAllocationCountOnThisThread(), probe);

  std::mt19937 generator(32);
  std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
  for (const LoadedRobot& robot : loadRobots()) {
    SCOPED_TRACE(robot.name);
    const MpcRobotModelBase<scalar_t>& model = *robot.model;
    const Eigen::Index joints = static_cast<Eigen::Index>(model.getJointDim());
    vector_t q = vector_t::Zero(static_cast<Eigen::Index>(model.getConfigurationDim()));
    vector_t v = vector_t::Zero(static_cast<Eigen::Index>(model.getVelocityDim()));
    for (int sample = 0; sample < 16; ++sample) {
      // Any measured attitude, gimbal lock included: the quaternion carries no singularity.
      const quaternion_t orientation(
          vector4_t(uniform(generator), uniform(generator), uniform(generator), uniform(generator)).normalized());
      const vector3_t position(uniform(generator), uniform(generator), 1.0 + uniform(generator));
      const vector3_t linearVelocityLocal(uniform(generator), uniform(generator), uniform(generator));
      const vector3_t angularVelocityLocal(uniform(generator), uniform(generator), uniform(generator));
      const vector_t jointPositions = vector_t::Random(joints);
      const vector_t jointVelocities = vector_t::Random(joints);

      const size_t before = robot::realtime::heapAllocationCountOnThisThread();
      model.fillGeneralizedCoordinatesFromMeasurement(position, orientation.coeffs(), jointPositions, q);
      model.fillGeneralizedVelocitiesFromMeasurement(orientation.coeffs(), linearVelocityLocal, angularVelocityLocal, jointVelocities, v);
      EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0U);

      // q = [p_W, xi_hat, q_j] and v = [R v_B, w_B, dq_j]: Pinocchio's vectors of the measurement, the angular velocity in
      // the body frame as measured.
      EXPECT_EQ(vector3_t(q.head<3>()), position);
      EXPECT_EQ(vector4_t(q.segment<4>(3)), safelyNormalizedQuaternion(vector4_t(orientation.coeffs())));
      EXPECT_EQ(q.tail(joints), jointPositions);
      EXPECT_EQ(vector3_t(v.head<3>()),
                vector3_t(quaternion_t(safelyNormalizedQuaternion(vector4_t(orientation.coeffs()))) * linearVelocityLocal));
      EXPECT_EQ(vector3_t(v.segment<3>(3)), angularVelocityLocal);
      EXPECT_EQ(v.tail(joints), jointVelocities);

      // A measured quaternion off the unit sphere or negated is read as the same attitude.
      vector_t scaledQ = q;
      vector_t scaledV = v;
      model.fillGeneralizedCoordinatesFromMeasurement(position, -2.0 * orientation.coeffs(), jointPositions, scaledQ);
      model.fillGeneralizedVelocitiesFromMeasurement(-2.0 * orientation.coeffs(), linearVelocityLocal, angularVelocityLocal,
                                                     jointVelocities, scaledV);
      EXPECT_LT((scaledQ.segment<4>(3) + q.segment<4>(3)).cwiseAbs().maxCoeff(), 1.0e-15);
      EXPECT_LT((scaledV - v).cwiseAbs().maxCoeff(), 1.0e-14);

      // Into a segment of a larger buffer, as into the state.
      vector_t buffer = vector_t::Constant(q.size() + 4, -1.0);
      model.fillGeneralizedCoordinatesFromMeasurement(position, orientation.coeffs(), jointPositions, buffer.segment(2, q.size()));
      EXPECT_EQ(buffer.segment(2, q.size()), q);
      EXPECT_EQ(buffer.head<2>(), vector2_t(-1.0, -1.0));
    }
  }
}

TEST(CentroidalRobotModelBoundary, TheBaseTwistIsTheBaseVelocityOfTheMomentumAtTheMeasuredJointVelocities) {
  // The base twist of a centroidal state needs the joint velocities, m h = A_b v_b + A_j dq_j. Checked
  // against the base velocity of Pinocchio's forward kinematics, which knows nothing of the centroidal map; v = [dp_W,
  // w_B, dq_j] of the quaternion root.
  std::mt19937 generator(33);
  std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
  for (const LoadedRobot& robot : loadRobots()) {
    SCOPED_TRACE(robot.name);
    const MpcRobotModelBase<scalar_t>& model = *robot.model;
    const PinocchioInterface::Model& pinocchioModel = robot.pinocchioInterface->getModel();
    PinocchioInterface::Data data(pinocchioModel);
    const Eigen::Index joints = static_cast<Eigen::Index>(model.getJointDim());
    for (int sample = 0; sample < 6; ++sample) {
      SCOPED_TRACE(sample);
      // Turned and tilted, the joints bent (stateWithEulerAngles()) and moving at up to 3 rad/s, as a leg swings.
      const vector3_t eulerZyx(4.0 * uniform(generator), 0.4 * uniform(generator), 0.4 * uniform(generator));
      vector_t state = stateWithEulerAngles(model, eulerZyx);
      const vector_t q = model.getGeneralizedCoordinates(state);
      vector_t velocity(pinocchioModel.nv);
      velocity.head<3>() = vector3_t(uniform(generator), uniform(generator), 0.3 * uniform(generator));
      velocity.segment<3>(3) = vector3_t(uniform(generator), uniform(generator), uniform(generator));
      for (Eigen::Index joint = 0; joint < joints; ++joint) velocity(6 + joint) = 3.0 * uniform(generator);
      const vector_t jointVelocities = velocity.tail(joints);
      // The normalized momentum a measurement of that motion carries, as the MRT computes it: h = A(q) v / m.
      pinocchio::computeCentroidalMap(pinocchioModel, data, q);
      const matrix_t momentumMatrix = data.Ag;
      state.head<6>() = momentumMatrix * velocity / robot.info->robotMass;

      // The base's own velocity, of Pinocchio's forward kinematics: the root joint's, in world axes at its origin.
      pinocchio::forwardKinematics(pinocchioModel, data, q, velocity);
      const pinocchio::Motion baseVelocity = pinocchio::getVelocity(pinocchioModel, data, /*jointId=*/1, pinocchio::LOCAL_WORLD_ALIGNED);
      const vector6_t twist = model.getBaseTwistInWorldFrame(state, jointVelocities);
      EXPECT_LT((twist.head<3>() - baseVelocity.linear()).cwiseAbs().maxCoeff(), 1.0e-9) << twist.transpose();
      EXPECT_LT((twist.tail<3>() - baseVelocity.angular()).cwiseAbs().maxCoeff(), 1.0e-9) << twist.transpose();
      // The body angular velocity v holds, turned into the world: w_W = R(xi) w_B.
      EXPECT_LT((twist.tail<3>() - rotationOf(model.getBaseOrientation(state)) * vector3_t(velocity.segment<3>(3))).cwiseAbs().maxCoeff(),
                1.0e-9);

      // Without the joint velocities it is the robot's twist with its joints locked, off by the joints' angular momentum
      // over the locked inertia: w_locked - w = I_G^-1 (A_j dq_j)_ang, with I_G the composite inertia of ccrba().
      pinocchio::ccrba(pinocchioModel, data, q, vector_t::Zero(pinocchioModel.nv));
      const matrix3_t lockedInertia = data.Ig.inertia().matrix();
      const vector3_t jointAngularMomentum = momentumMatrix.bottomRows<3>().rightCols(joints) * jointVelocities;
      const vector6_t lockedTwist = model.getBaseTwistInWorldFrame(state, vector_t::Zero(joints));
      const vector3_t lockedError = lockedTwist.tail<3>() - twist.tail<3>();
      EXPECT_LT((lockedError - lockedInertia.inverse() * jointAngularMomentum).cwiseAbs().maxCoeff(), 1.0e-9) << lockedError.transpose();
      EXPECT_GT(lockedError.norm(), 1.0e-2) << "the joints must carry angular momentum, or the test cannot tell the twists apart";
    }
  }
}

TEST(CentroidalRobotModelBoundary, UnderSingleRigidBodyDynamicsTheBaseTwistIsTheRigidBodysAtAnyJointVelocity) {
  // Single rigid body dynamics: the momentum map has no joint columns (A_j = 0), m h = A_b v_b with A_b of the nominal
  // inertia I_nom about the CoM and the nominal CoM-to-base offset r_nom, in the base frame. So the twist is the rigid
  // body's, w_W = (R I_nom R')^-1 L and dp_W = h_lin + w_W x R r_nom (L = m h_ang), whatever the joints do. In closed
  // form from the model's info and Eigen's rotation of the state's attitude.
  for (const absl::string_view files : {"g1", "atlas"}) {
    SCOPED_TRACE(files);
    absl::StatusOr<LoadedRobot> robot =
        loadRobot(files, files == "g1" ? g1Files() : atlasFiles(), Formulation::kWrenches, "single_rigid_body_dynamics");
    ASSERT_TRUE(robot.ok()) << robot.status();
    ASSERT_EQ(robot->info->centroidalModelType, CentroidalModelType::SingleRigidBodyDynamics);
    const MpcRobotModelBase<scalar_t>& model = *robot->model;
    const Eigen::Index joints = static_cast<Eigen::Index>(model.getJointDim());
    std::mt19937 generator(41);
    std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
    for (int sample = 0; sample < 6; ++sample) {
      SCOPED_TRACE(sample);
      vector_t state = stateWithEulerAngles(model, vector3_t(4.0 * uniform(generator), 0.4 * uniform(generator), 0.4 * uniform(generator)));
      const vector3_t linearMomentum(uniform(generator), uniform(generator), 0.3 * uniform(generator));
      const vector3_t angularMomentum(2.0 * uniform(generator), 2.0 * uniform(generator), 2.0 * uniform(generator));
      state.head<3>() = linearMomentum;
      state.segment<3>(3) = angularMomentum;
      vector_t jointVelocities(joints);
      for (Eigen::Index joint = 0; joint < joints; ++joint) jointVelocities(joint) = 3.0 * uniform(generator);

      const matrix3_t rotation = rotationOf(model.getBaseOrientation(state));
      const matrix3_t inertiaWorld = rotation * robot->info->centroidalInertiaNominal * rotation.transpose();
      const vector3_t angularVelocity = inertiaWorld.inverse() * (robot->info->robotMass * angularMomentum);
      const vector3_t linearVelocity = linearMomentum + angularVelocity.cross(rotation * robot->info->comToBasePositionNominal);
      const vector6_t twist = model.getBaseTwistInWorldFrame(state, jointVelocities);
      EXPECT_LT((twist.head<3>() - linearVelocity).cwiseAbs().maxCoeff(), 1.0e-11) << twist.transpose();
      EXPECT_LT((twist.tail<3>() - angularVelocity).cwiseAbs().maxCoeff(), 1.0e-11) << twist.transpose();
      EXPECT_EQ(model.getBaseTwistInWorldFrame(state, vector_t::Zero(joints)), twist) << "the joints do not enter";
      EXPECT_GT(angularVelocity.norm(), 1.0e-2) << "the momentum must turn the body";
    }
  }
}

TEST(CentroidalRobotModelBoundaryDeathTest, TheBaseTwistDiesOnJointVelocitiesOfAnotherSize) {
  // A_j dq_j of fewer joint velocities would read only the columns of the joints they hold: the twist with the others
  // locked, silently.
  absl::StatusOr<LoadedRobot> robot = loadRobot("g1", g1Files(), Formulation::kWrenches);
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  const vector_t state = model.getNeutralState();
  const Eigen::Index joints = static_cast<Eigen::Index>(model.getJointDim());
  EXPECT_DEATH(model.getBaseTwistInWorldFrame(state, vector_t::Zero(joints - 1)), "one joint velocity per MPC joint");
  EXPECT_DEATH(model.getBaseTwistInWorldFrame(state, vector_t::Zero(0)), "one joint velocity per MPC joint");
}

TEST(CentroidalRobotModelBoundary, AKnotsCommandIsReadBackFromItsMomentum) {
  for (const LoadedRobot& robot : loadRobots()) {
    SCOPED_TRACE(robot.name);
    const MpcRobotModelBase<scalar_t>& model = *robot.model;
    const PinocchioInterface::Model& pinocchioModel = robot.pinocchioInterface->getModel();
    PinocchioInterface::Data data(pinocchioModel);
    // A knot as CentroidalMpcTargetTrajectoriesCalculator writes one: the CoM velocity, and the momentum of the body
    // locked in the knot's configuration turning at the yaw rate, the yaw column of its composite inertia.
    constexpr scalar_t kYawRate = 0.45;
    vector_t knot = stateWithEulerAngles(model, vector3_t(2.5, 0.0, 0.0));
    pinocchio::ccrba(pinocchioModel, data, model.getGeneralizedCoordinates(knot), vector_t::Zero(pinocchioModel.nv));
    const matrix3_t lockedInertia = data.Ig.inertia().matrix();
    knot.head<3>() = vector3_t(0.4, -0.2, 0.0);
    knot.segment<3>(3) = lockedInertia.col(2) * (kYawRate / robot.info->robotMass);

    const vector3_t command = model.getTargetPlanarCommand(knot, lockedInertia(2, 2));
    EXPECT_EQ(command.head<2>(), vector2_t(0.4, -0.2));
    EXPECT_NEAR(command(2), kYawRate, 1.0e-14);
    // The inertia the reference manager latched before its first measurement: no yaw rate, rather than a wrong one.
    EXPECT_EQ(model.getTargetPlanarCommand(knot, /*compositeYawInertia=*/0.0), vector3_t(0.4, -0.2, 0.0));
    EXPECT_EQ(model.getTargetPlanarCommand(knot, /*compositeYawInertia=*/-1.0)(2), 0.0);
  }
}

TEST(CentroidalRobotModelBoundary, TheDecoratorForwardsTheLayoutAgnosticApi) {
  absl::StatusOr<LoadedRobot> wrench = loadRobot("atlas", atlasFiles(), Formulation::kWrenches);
  absl::StatusOr<LoadedRobot> decorated = loadRobot("atlas_basis_vectors", atlasFiles(), Formulation::kBasisVectors);
  ASSERT_TRUE(wrench.ok()) << wrench.status();
  ASSERT_TRUE(decorated.ok()) << decorated.status();
  const MpcRobotModelBase<scalar_t>& model = *wrench->model;
  const MpcRobotModelBase<scalar_t>& decorator = *decorated->model;
  EXPECT_NE(decorator.getInputDim(), model.getInputDim());
  EXPECT_EQ(decorator.getStateDim(), model.getStateDim());
  EXPECT_EQ(decorator.getStateTangentDim(), model.getStateTangentDim());
  EXPECT_EQ(decorator.getConfigurationDim(), model.getConfigurationDim());
  EXPECT_EQ(decorator.getVelocityDim(), model.getVelocityDim());
  EXPECT_EQ(decorator.getBaseOrientationStartIndex(), model.getBaseOrientationStartIndex());
  EXPECT_EQ(describeTuningLayout(decorator.getTuningLayout()), describeTuningLayout(model.getTuningLayout()));

  const vector_t state = stateWithEulerAngles(model, vector3_t(kLargeYaws[1], 0.3, -0.2));
  EXPECT_EQ(decorator.getBaseOrientation(state), model.getBaseOrientation(state));
  EXPECT_EQ(decorator.getBaseHeading(state), model.getBaseHeading(state));
  EXPECT_EQ(decorator.getBasePoseEulerZyx(state), model.getBasePoseEulerZyx(state));
  EXPECT_EQ(decorator.getBaseOrientationEulerZyx(state), model.getBaseOrientationEulerZyx(state));
  EXPECT_EQ(decorator.getNeutralState(), model.getNeutralState());
  EXPECT_EQ(decorator.getBaseRotationTangentJacobian(state), model.getBaseRotationTangentJacobian(state));
  EXPECT_EQ(decorator.getPinocchioTangentJacobian(state), model.getPinocchioTangentJacobian(state));
  const vector_t jointVelocities = vector_t::LinSpaced(static_cast<Eigen::Index>(model.getJointDim()), -1.0, 1.0);
  EXPECT_EQ(decorator.getBaseTwistInWorldFrame(state, jointVelocities), model.getBaseTwistInWorldFrame(state, jointVelocities));
  EXPECT_EQ(decorator.getTargetPlanarCommand(state, /*compositeYawInertia=*/2.5),
            model.getTargetPlanarCommand(state, /*compositeYawInertia=*/2.5));
  vector_t decoratorWritten = state;
  vector_t modelWritten = state;
  decorator.setBasePoseEulerZyx(decoratorWritten, vector6_t::Constant(0.5));
  model.setBasePoseEulerZyx(modelWritten, vector6_t::Constant(0.5));
  EXPECT_EQ(decoratorWritten, modelWritten);
  decorator.setBaseOrientation(decoratorWritten, vector4_t(0.1, 0.2, 0.3, 0.9));
  model.setBaseOrientation(modelWritten, vector4_t(0.1, 0.2, 0.3, 0.9));
  EXPECT_EQ(decoratorWritten, modelWritten);
  const absl::StatusOr<vector_t> decoratorTuning = decorator.tuningFromState(state);
  const absl::StatusOr<vector_t> modelTuning = model.tuningFromState(state);
  ASSERT_TRUE(decoratorTuning.ok()) << decoratorTuning.status();
  ASSERT_TRUE(modelTuning.ok()) << modelTuning.status();
  EXPECT_EQ(*decoratorTuning, *modelTuning);
  const absl::StatusOr<vector_t> decoratorState = decorator.stateFromTuning(*modelTuning);
  const absl::StatusOr<vector_t> modelState = model.stateFromTuning(*modelTuning);
  ASSERT_TRUE(decoratorState.ok()) << decoratorState.status();
  ASSERT_TRUE(modelState.ok()) << modelState.status();
  EXPECT_EQ(*decoratorState, *modelState);
  EXPECT_EQ(decorator.interpolateStateTrajectory(/*time=*/0.3, {0.0, 1.0}, {state, modelWritten}),
            model.interpolateStateTrajectory(/*time=*/0.3, {0.0, 1.0}, {state, modelWritten}));
}

TEST(CentroidalRobotModelBoundaryDeathTest, TheBaseAngularVelocityAccessorsAbort) {
  absl::StatusOr<LoadedRobot> robot = loadRobot("g1", g1Files(), Formulation::kWrenches);
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  vector_t state = model.getNeutralState();
  EXPECT_DEATH(model.getBaseAngularVelocityLocal(state), "holds no base angular velocity");
  EXPECT_DEATH(model.setBaseAngularVelocityLocal(state, vector3_t::Zero()), "holds no base angular velocity");
  EXPECT_DEATH(model.setBaseAngularVelocityFromWorldYawRate(state, /*yawRate=*/0.3), "holds no base angular velocity");
  EXPECT_DEATH(model.getBaseAngularVelocityLocalJacobian(state), "holds no base angular velocity");
}

TEST(CentroidalRobotModelBoundaryDeathTest, TheCppAdModelHasNoBaseTwistOrKnotCommand) {
  absl::StatusOr<LoadedRobot> robot = loadRobot("g1", g1Files(), Formulation::kWrenches);
  ASSERT_TRUE(robot.ok()) << robot.status();
  const PinocchioInterfaceCppAd pinocchioInterfaceCppAd = robot->pinocchioInterface->toCppAd();
  const CentroidalModelInfoCppAd infoCppAd = robot->info->toCppAd();
  const CentroidalMpcRobotModel<ad_scalar_t> modelCppAd(*robot->modelSettings, pinocchioInterfaceCppAd, infoCppAd);
  const vector_t state = robot->model->getNeutralState();
  EXPECT_DEATH(modelCppAd.getBaseTwistInWorldFrame(state, vector_t::Zero(static_cast<Eigen::Index>(robot->model->getJointDim()))),
               "no double-precision Pinocchio model");
  EXPECT_DEATH(modelCppAd.getTargetPlanarCommand(state, /*compositeYawInertia=*/1.0), "no double-precision Pinocchio model");
}

}  // namespace
}  // namespace ocs2::humanoid
