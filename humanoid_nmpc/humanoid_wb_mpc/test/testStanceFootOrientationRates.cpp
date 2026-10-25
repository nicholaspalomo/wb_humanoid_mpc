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
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/OrientationErrorRateOracle.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/constraint/ZeroAccelerationConstraintCppAd.h"
#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"
#include "humanoid_wb_mpc/end_effector/PinocchioEndEffectorDynamicsCppAd.h"

/*
 * The whole-body MPC's stance-foot constraint, ZeroAccelerationConstraintCppAd
 * on the G1 with the shipped gains and the left foot's CppAD end-effector
 * dynamics generated for this run, at states whose foot is level or tilted by 5
 * to 80 degrees, at rest or turning, with random inputs: (a) Pinocchio's frame
 * velocity and classical acceleration (LOCAL_WORLD_ALIGNED), which the
 * libraries tape, are the time derivatives of the foot's world position and
 * attitude along the robot's motion q(t) = q (+) (v t + a t^2 / 2), a the
 * whole-body dynamics' generalized acceleration: dp/dt and d2p/dt2, and w,
 * alpha with [w]x = skew(dR/dt R') and [alpha]x = skew(d2R/dt2 R') (the rest of
 * d2R/dt2 R' is the symmetric [w]x^2); (b) the constraint's value is the law
 * assembled from those double-precision kinematics and the gains: the linear
 * rows Kv v + Ka a (+ Kp_z (z - z_ground)), the orientation rows Kp e + Kd
 * de/dt + Ka d2e/dt2, e the production orientation error and its rates the
 * numerical oracle's (OrientationErrorRateOracle.h), which shares no formula
 * with the tapes; (c) the gains were rescaled so that the stance is held as
 * before: at a level foot the constraint is the former law on the raw angular
 * velocity and acceleration with half the angular gains - to round-off at rest,
 * and but for the closed form Ka / 2 (n.w)(n x w) when the foot turns - while
 * at a tilted foot it differs from it as the rates do; (d) its linearization is
 * its derivative: the state Jacobian along the quaternion state manifold and
 * the input Jacobian, entry by entry against Richardson-extrapolated
 * differences within their error bound, which rejects the largest entry moved
 * by 1e-6 of max(1, itself), and the largest entry of the orientation rows so
 * moved, and the state Jacobian of the frozen-map linearization the centroidal
 * twist constraint reports (the map from w to de/dt held at the point and
 * applied to alpha too), whose input Jacobian is the exact one; the Jacobians
 * of w and alpha it applies the map to are the libraries', read off the pose
 * rates' anti-parallel fallback, w / 2 and alpha / 2. Large: it tapes and
 * compiles the six end-effector dynamics libraries of one foot.
 */
namespace ocs2::humanoid {
namespace {

using test_support::AttitudeMotion;
using test_support::DifferencedOrientationErrorRates;
using test_support::kDifferencedRateTolerance;

// LINT.IfChange(robot_files)
constexpr char kTaskFile[] =
    "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
constexpr char kUrdfFile[] =
    "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr char kReferenceFile[] =
    "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto";
// LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:stance_foot_orientation_rates_test_data)
constexpr size_t kFoot = 0;
constexpr scalar_t kTime = 0.0;
/**
 * The round-off allowed between the CppAD libraries (Pinocchio on AD scalars,
 * generated with -ffast-math) and Pinocchio in double, relative to the
 * magnitude of the summands: a few units of 1e-16 per operation, over a few
 * thousand operations.
 */
constexpr scalar_t kLibraryRoundOff = 1.0e-11;
/**
 * [s] The step of the differences along the robot's motion (a). Richardson's
 * truncation, h^4 / 480 |f^(5)| for the rates and h^4 / 1440 |f^(6)| for the
 * accelerations, is below 1e-11 of the scale at the motion's rates of a few
 * rad/s
 * (|f^(k)| of the order of their k-th power). The round-off of the extrapolated
 * second difference, 23 delta / h^2 for a sample error delta, is 2.5e-9 for a
 * foot pose of order 1 that carries the unit round-off (5e-9 for machine
 * epsilon), more as the forward kinematics accumulates its error over the leg's
 * joints: kMotionTolerance leaves a margin of 20 to 40 to the scale max(1, |the
 * derivative|) before that accumulation, and the test prints the margin it
 * measures.
 */
constexpr scalar_t kMotionStep = 1.0e-3;
constexpr scalar_t kMotionTolerance = 1.0e-7;
/** [rad], [state units] The step of the state Jacobian's bounded differences
 * (d), as testQuaternionRootTermDerivatives'. */
constexpr scalar_t kStateJacobianStep = 1.0e-3;
/**
 * [N], [N m], [rad/s^2] The step of the input Jacobian's differences: the
 * constraint is affine in the input (the accelerations are, and the rates do
 * not read it), so the truncation vanishes and a larger step only shrinks the
 * round-off part of the bound, 6 delta / h.
 */
constexpr scalar_t kInputJacobianStep = 1.0e-1;
/** The relative error of one evaluation of the constraint, per row of its
 * magnitude (floor 1), in the bound of (d). */
constexpr scalar_t kEvaluationRelativeError = 1.0e-12;
/** The round-off of a reported Jacobian entry, relative to max(1, its row's
 * largest magnitude), in the bound of (d). */
constexpr scalar_t kJacobianRoundOff = 1.0e-12;
/** The change of the largest Jacobian entry, relative to max(1, the entry),
 * that (d) must reject. */
constexpr scalar_t kJacobianMutation = 1.0e-6;

/** The G1 whole-body models and the left foot's end-effector dynamics. Passive
 * data. */
struct StanceFootModels {
  std::unique_ptr<WBMpcInterface> interface;
  /** The robot model the dynamics keeps a pointer to: declared before it, so
   * that it outlives it. */
  std::unique_ptr<WBAccelMpcRobotModel<ad_scalar_t>> adRobotModel;
  std::unique_ptr<PinocchioEndEffectorDynamicsCppAd> dynamics;
};

std::unique_ptr<StanceFootModels> makeStanceFootModels() {
  absl::StatusOr<std::unique_ptr<WBMpcInterface>> interface =
      WBMpcInterface::CreateControllerModels(kTaskFile, kUrdfFile,
                                             kReferenceFile);
  EXPECT_TRUE(interface.ok()) << interface.status();
  if (!interface.ok()) return nullptr;
  std::unique_ptr<StanceFootModels> models =
      std::make_unique<StanceFootModels>();
  models->interface = *std::move(interface);
  // The swing planner's trajectories, whose stance height the z row servoes to,
  // around kTime.
  models->interface->getSwitchedModelReferenceManagerPtr()->preSolverRun(
      kTime - 0.5, kTime + 1.0, models->interface->getInitialState(),
      ModeNumber::kStance);
  const ModelSettings& settings = models->interface->modelSettings();
  models->adRobotModel =
      std::make_unique<WBAccelMpcRobotModel<ad_scalar_t>>(settings);
  // A folder of this run: every library is taped and compiled from this code.
  const std::string folder = (std::filesystem::path(testing::TempDir()) /
                              "stance_foot_orientation_rates")
                                 .string();
  models->dynamics = std::make_unique<PinocchioEndEffectorDynamicsCppAd>(
      models->interface->getPinocchioInterface(), *models->adRobotModel,
      std::vector<std::string>{settings.contactNames[kFoot]},
      settings.contactNames[kFoot], folder, /*recompileLibraries=*/true,
      /*verbose=*/false);
  return models;
}

/** The models of makeStanceFootModels(), built once for the test program; never
 * destroyed. Null after a failure. */
const StanceFootModels* absl_nullable stanceFootModels() {
  static const StanceFootModels* absl_nullable const kModels =
      makeStanceFootModels().release();
  return kModels;
}

/** The constraint the interface builds for the left foot, with the
 * foot-constraint `gains`. */
std::unique_ptr<ZeroAccelerationConstraintCppAd> stanceConstraint(
    const StanceFootModels& models,
    const ModelSettings::FootConstraintConfig& gains) {
  absl::StatusOr<std::unique_ptr<ZeroAccelerationConstraintCppAd>> constraint =
      ZeroAccelerationConstraintCppAd::Create(
          *models.interface->getSwitchedModelReferenceManagerPtr(),
          *models.dynamics, kFoot,
          stanceFootAccelerationConstraintConfig(gains));
  EXPECT_TRUE(constraint.ok()) << constraint.status();
  return constraint.ok() ? *std::move(constraint) : nullptr;
}

/** The left foot's kinematics in double precision, LOCAL_WORLD_ALIGNED. Passive
 * data. */
struct FootKinematics {
  vector3_t position = vector3_t::Zero();
  matrix3_t rotation = matrix3_t::Identity();
  vector3_t linearVelocity = vector3_t::Zero();
  vector3_t angularVelocity = vector3_t::Zero();
  vector3_t linearAcceleration = vector3_t::Zero();
  vector3_t angularAcceleration = vector3_t::Zero();
};

/** Pinocchio's configuration q, velocity v and the whole-body dynamics'
 * generalized acceleration a at (state, input). */
struct GeneralizedMotion {
  vector_t q;
  vector_t v;
  vector_t a;
};

GeneralizedMotion generalizedMotionAt(const StanceFootModels& models,
                                      const vector_t& state,
                                      const vector_t& input) {
  WBAccelMpcRobotModel<scalar_t> robotModel(models.interface->modelSettings());
  const PinocchioInterface& pinocchio =
      models.interface->getPinocchioInterface();
  return GeneralizedMotion{
      .q = robotModel.getGeneralizedCoordinates(state),
      .v = robotModel.getGeneralizedVelocities(state, input),
      .a = computeGeneralizedAccelerations<scalar_t>(state, input, pinocchio,
                                                     robotModel)};
}

pinocchio::FrameIndex footFrameOf(const StanceFootModels& models) {
  return models.interface->getPinocchioInterface().getModel().getFrameId(
      models.interface->modelSettings().contactNames[kFoot]);
}

FootKinematics footKinematicsAt(const StanceFootModels& models,
                                const vector_t& state,
                                const vector_t& input) {
  const GeneralizedMotion motion = generalizedMotionAt(models, state, input);
  PinocchioInterface pinocchio = models.interface->getPinocchioInterface();
  const PinocchioInterface::Model& model = pinocchio.getModel();
  PinocchioInterface::Data& data = pinocchio.getData();
  pinocchio::forwardKinematics(model, data, motion.q, motion.v, motion.a);
  pinocchio::updateFramePlacements(model, data);
  const pinocchio::FrameIndex frame = footFrameOf(models);
  const pinocchio::ReferenceFrame rf =
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
  const pinocchio::Motion velocity =
      pinocchio::getFrameVelocity(model, data, frame, rf);
  const pinocchio::Motion acceleration =
      pinocchio::getFrameClassicalAcceleration(model, data, frame, rf);
  return FootKinematics{.position = data.oMf[frame].translation(),
                        .rotation = data.oMf[frame].rotation(),
                        .linearVelocity = velocity.linear(),
                        .angularVelocity = velocity.angular(),
                        .linearAcceleration = acceleration.linear(),
                        .angularAcceleration = acceleration.angular()};
}

/** The left foot's world placement at the configuration `q`. */
pinocchio::SE3 footPlacementAt(const StanceFootModels& models,
                               const vector_t& q) {
  PinocchioInterface pinocchio = models.interface->getPinocchioInterface();
  pinocchio::forwardKinematics(pinocchio.getModel(), pinocchio.getData(), q);
  pinocchio::updateFramePlacements(pinocchio.getModel(), pinocchio.getData());
  return pinocchio.getData().oMf[footFrameOf(models)];
}

/** One state and input of the constraint. Passive data. */
struct StancePoint {
  std::string name;
  vector_t state;
  vector_t input;
  /** Whether the foot is level, its normal the ground normal e_z. */
  bool level = false;
  /** Whether the robot is at rest: every velocity of the state is zero. */
  bool atRest = false;
};

/**
 * A state of the shipped initial state, its joint angles moved within 0.1 rad,
 * with the base turned so that the left foot's attitude is `footAttitude`; its
 * velocities random within `speed` (rad/s for the base's angular velocity and
 * the joints, a third of it in m/s for the base) and a random input: wrenches
 * of 100 to 300 N up and up to 30 N and 5 N m otherwise, joint accelerations
 * within 5 rad/s^2.
 */
StancePoint stancePoint(const StanceFootModels& models,
                        const std::string& name,
                        const matrix3_t& footAttitude,
                        scalar_t speed,
                        std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> unit(-1.0, 1.0);
  const WBAccelMpcRobotModel<scalar_t>& robotModel =
      models.interface->getMpcRobotModel();
  const Eigen::Index nq =
      static_cast<Eigen::Index>(robotModel.getConfigurationDim());
  const Eigen::Index nv =
      static_cast<Eigen::Index>(robotModel.getVelocityDim());
  const Eigen::Index joints =
      static_cast<Eigen::Index>(robotModel.getJointDim());
  StancePoint point{.name = name,
                    .state = models.interface->getInitialState(),
                    .input = vector_t::Zero(robotModel.getInputDim())};
  for (Eigen::Index j = 0; j < joints; ++j)
    point.state(7 + j) += 0.1 * unit(generator);
  // The foot's attitude relative to the base, at a base of identity attitude.
  vector_t q = point.state.head(nq);
  q.segment<4>(3) = Eigen::Quaterniond::Identity().coeffs();
  const matrix3_t footInBase = footPlacementAt(models, q).rotation();
  const Eigen::Quaterniond baseAttitude(
      matrix3_t(footAttitude * footInBase.transpose()));
  point.state.segment<4>(3) = baseAttitude.normalized().coeffs();
  for (Eigen::Index i = 0; i < nv; ++i)
    point.state(nq + i) = speed * unit(generator) * (i < 3 ? 1.0 / 3.0 : 1.0);
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    const Eigen::Index start = static_cast<Eigen::Index>(6 * foot);
    point.input.segment<3>(start) =
        vector3_t(30.0 * unit(generator), 30.0 * unit(generator),
                  200.0 + 100.0 * unit(generator));
    point.input.segment<3>(start + 3) =
        5.0 * vector3_t(unit(generator), unit(generator), unit(generator));
  }
  for (Eigen::Index j = 0; j < joints; ++j)
    point.input(6 * kNumContacts + j) = 5.0 * unit(generator);
  return point;
}

/** The points of the file comment, from a fixed seed. */
std::vector<StancePoint> stancePoints(const StanceFootModels& models) {
  std::mt19937 generator(20261012);
  std::uniform_real_distribution<scalar_t> angle(-std::numbers::pi,
                                                 std::numbers::pi);
  std::vector<StancePoint> points;
  for (size_t i = 0; i < 2; ++i) {
    const matrix3_t level =
        Eigen::AngleAxisd(angle(generator), vector3_t::UnitZ())
            .toRotationMatrix();
    StancePoint point = stancePoint(models, absl::StrCat("level_at_rest_", i),
                                    level, /*speed=*/0.0, generator);
    point.level = true;
    point.atRest = true;
    points.push_back(std::move(point));
  }
  for (size_t i = 0; i < 4; ++i) {
    const matrix3_t level =
        Eigen::AngleAxisd(angle(generator), vector3_t::UnitZ())
            .toRotationMatrix();
    StancePoint point = stancePoint(models, absl::StrCat("level_turning_", i),
                                    level, /*speed=*/1.5, generator);
    point.level = true;
    points.push_back(std::move(point));
  }
  for (scalar_t degrees : {5.0, 20.0, 45.0, 80.0}) {
    for (size_t i = 0; i < 2; ++i) {
      const scalar_t direction = angle(generator);
      const vector3_t axis(std::cos(direction), std::sin(direction), 0.0);
      const matrix3_t tilted =
          Eigen::AngleAxisd(degrees * std::numbers::pi / 180.0, axis)
              .toRotationMatrix() *
          Eigen::AngleAxisd(angle(generator), vector3_t::UnitZ())
              .toRotationMatrix();
      points.push_back(
          stancePoint(models, absl::StrFormat("tilted_%g_deg_%d", degrees, i),
                      tilted, /*speed=*/1.5, generator));
    }
  }
  return points;
}

/** [w]x -> w of the skew-symmetric part of `matrix`. */
vector3_t skewPart(const matrix3_t& matrix) {
  const matrix3_t skew = 0.5 * (matrix - matrix.transpose());
  return vector3_t(skew(2, 1), skew(0, 2), skew(1, 0));
}

/** The oracle's rates of the foot's orientation error at `kinematics`, against
 * the ground normal. */
DifferencedOrientationErrorRates oracleRatesOf(
    const FootKinematics& kinematics) {
  const AttitudeMotion motion{
      .initialRotation = kinematics.rotation,
      .angularVelocity = kinematics.angularVelocity,
      .angularAcceleration = kinematics.angularAcceleration};
  return test_support::differencedOrientationErrorRates(motion,
                                                        vector3_t::UnitZ());
}

/** The law of (b) and its scale per row: the magnitudes of its summands.
 * Passive data. */
struct AssembledLaw {
  vector6_t value = vector6_t::Zero();
  /** What the value may differ by: kLibraryRoundOff of the summands, plus the
   * oracle's error on the orientation rows. */
  vector6_t tolerance = vector6_t::Zero();
};

/**
 * The stance constraint of `gains` at `kinematics`, `rates` its orientation
 * error's rate and acceleration and `groundHeight` the height the z row servoes
 * to, each row assembled from its gains.
 */
AssembledLaw assembledLaw(const ModelSettings::FootConstraintConfig& gains,
                          const FootKinematics& kinematics,
                          const vector3_t& rate,
                          const vector3_t& acceleration,
                          scalar_t rateScale,
                          scalar_t groundHeight) {
  const vector3_t error = rotationMatrixDistanceToPlane<scalar_t>(
      kinematics.rotation, vector3_t::UnitZ());
  AssembledLaw law;
  for (Eigen::Index i = 0; i < 2; ++i) {
    law.value(i) =
        gains.linearVelocityErrorGain_xy * kinematics.linearVelocity(i) +
        gains.linearAccelerationErrorGain_xy * kinematics.linearAcceleration(i);
    law.tolerance(i) =
        kLibraryRoundOff * (std::abs(gains.linearVelocityErrorGain_xy *
                                     kinematics.linearVelocity(i)) +
                            std::abs(gains.linearAccelerationErrorGain_xy *
                                     kinematics.linearAcceleration(i)) +
                            1.0);
  }
  const scalar_t height = kinematics.position.z() - groundHeight;
  law.value(2) =
      gains.positionErrorGain_z * height +
      gains.linearVelocityErrorGain_z * kinematics.linearVelocity.z() +
      gains.linearAccelerationErrorGain_z * kinematics.linearAcceleration.z();
  law.tolerance(2) =
      kLibraryRoundOff *
      (std::abs(gains.positionErrorGain_z) *
           (std::abs(kinematics.position.z()) + std::abs(groundHeight)) +
       std::abs(gains.linearVelocityErrorGain_z *
                kinematics.linearVelocity.z()) +
       std::abs(gains.linearAccelerationErrorGain_z *
                kinematics.linearAcceleration.z()) +
       1.0);
  const scalar_t rateUnit = std::max(1.0, rateScale);
  for (Eigen::Index i = 0; i < 3; ++i) {
    law.value(3 + i) = gains.orientationErrorGain * error(i) +
                       gains.angularVelocityErrorGain * rate(i) +
                       gains.angularAccelerationErrorGain * acceleration(i);
    law.tolerance(3 + i) =
        kDifferencedRateTolerance *
            (std::abs(gains.angularVelocityErrorGain) * rateUnit +
             std::abs(gains.angularAccelerationErrorGain) * rateUnit *
                 rateUnit) +
        kLibraryRoundOff * (std::abs(gains.orientationErrorGain) + 1.0);
  }
  return law;
}

scalar_t groundHeightOf(const StanceFootModels& models) {
  return models.interface->getSwitchedModelReferenceManagerPtr()
      ->getSwingTrajectoryPlanner()
      ->getZpositionConstraint(kFoot, kTime);
}

TEST(StanceFootOrientationRates,
     PinocchiosFrameRatesAreTheDerivativesAlongTheRobotsMotion) {
  const StanceFootModels* absl_nullable models = stanceFootModels();
  ASSERT_NE(models, nullptr);
  const PinocchioInterface::Model& model =
      models->interface->getPinocchioInterface().getModel();
  scalar_t largestDeviation = 0.0;  // relative to the tolerance
  for (const StancePoint& point : stancePoints(*models)) {
    SCOPED_TRACE(point.name);
    const GeneralizedMotion motion =
        generalizedMotionAt(*models, point.state, point.input);
    const FootKinematics kinematics =
        footKinematicsAt(*models, point.state, point.input);
    // The foot's placement along q(t) = q (+) (v t + a t^2 / 2), at t = 0, +-h
    // / 2 and +-h.
    std::array<pinocchio::SE3, 5> placements;
    const std::array<scalar_t, 5> times = {
        0.0, 0.5 * kMotionStep, -0.5 * kMotionStep, kMotionStep, -kMotionStep};
    for (size_t s = 0; s < times.size(); ++s) {
      const vector_t displacement =
          motion.v * times[s] + 0.5 * times[s] * times[s] * motion.a;
      placements[s] = footPlacementAt(
          *models, pinocchio::integrate(model, motion.q, displacement));
    }
    const scalar_t h = kMotionStep;
    const vector3_t positionRate =
        (4.0 * (placements[1].translation() - placements[2].translation()) / h -
         (placements[3].translation() - placements[4].translation()) /
             (2.0 * h)) /
        3.0;
    const vector3_t positionAcceleration =
        (4.0 *
             (placements[1].translation() - 2.0 * placements[0].translation() +
              placements[2].translation()) /
             (0.25 * h * h) -
         (placements[3].translation() - 2.0 * placements[0].translation() +
          placements[4].translation()) /
             (h * h)) /
        3.0;
    const matrix3_t rotationRate =
        (4.0 * (placements[1].rotation() - placements[2].rotation()) / h -
         (placements[3].rotation() - placements[4].rotation()) / (2.0 * h)) /
        3.0;
    const matrix3_t rotationAcceleration =
        (4.0 *
             (placements[1].rotation() - 2.0 * placements[0].rotation() +
              placements[2].rotation()) /
             (0.25 * h * h) -
         (placements[3].rotation() - 2.0 * placements[0].rotation() +
          placements[4].rotation()) /
             (h * h)) /
        3.0;
    const matrix3_t rotationTransposed = placements[0].rotation().transpose();
    const vector3_t angularVelocity =
        skewPart(rotationRate * rotationTransposed);
    const vector3_t angularAcceleration =
        skewPart(rotationAcceleration * rotationTransposed);
    const scalar_t velocityScale =
        std::max({1.0, kinematics.linearVelocity.norm(),
                  kinematics.angularVelocity.norm()});
    const scalar_t accelerationScale =
        std::max({1.0, kinematics.linearAcceleration.norm(),
                  kinematics.angularAcceleration.norm()});
    // Each comparison's deviation relative to its tolerance.
    const std::array<scalar_t, 4> deviations = {
        (positionRate - kinematics.linearVelocity).cwiseAbs().maxCoeff() /
            (kMotionTolerance * velocityScale),
        (angularVelocity - kinematics.angularVelocity).cwiseAbs().maxCoeff() /
            (kMotionTolerance * velocityScale),
        (positionAcceleration - kinematics.linearAcceleration)
                .cwiseAbs()
                .maxCoeff() /
            (kMotionTolerance * accelerationScale),
        (angularAcceleration - kinematics.angularAcceleration)
                .cwiseAbs()
                .maxCoeff() /
            (kMotionTolerance * accelerationScale)};
    EXPECT_LE(deviations[0], 1.0) << "linear velocity";
    EXPECT_LE(deviations[1], 1.0) << "angular velocity";
    EXPECT_LE(deviations[2], 1.0) << "linear acceleration";
    EXPECT_LE(deviations[3], 1.0)
        << "differenced " << angularAcceleration.transpose() << "\nPinocchio  "
        << kinematics.angularAcceleration.transpose();
    largestDeviation =
        std::max(largestDeviation,
                 *std::max_element(deviations.begin(), deviations.end()));
    if (point.atRest) {
      EXPECT_TRUE(kinematics.angularVelocity.isZero(/*prec=*/0.0));
    } else {
      EXPECT_GT(kinematics.angularVelocity.norm(), 0.1)
          << "the case under test: a turning foot";
    }
    EXPECT_GT(kinematics.angularAcceleration.norm(), 0.1)
        << "the case under test: an accelerating foot";
  }
  absl::PrintF(
      "[stance] Pinocchio's frame rates against the differences along the "
      "motion: largest deviation relative to its "
      "tolerance %.3g\n",
      largestDeviation);
}

TEST(StanceFootOrientationRates,
     TheConstraintIsThePdLawOnTheOrientationErrorAndItsRates) {
  const StanceFootModels* absl_nullable models = stanceFootModels();
  ASSERT_NE(models, nullptr);
  const ModelSettings::FootConstraintConfig& gains =
      models->interface->modelSettings().footConstraintConfig;
  const std::unique_ptr<ZeroAccelerationConstraintCppAd> constraint =
      stanceConstraint(*models, gains);
  ASSERT_NE(constraint, nullptr);
  scalar_t largestDeviation = 0.0;  // relative to the tolerance
  for (const StancePoint& point : stancePoints(*models)) {
    SCOPED_TRACE(point.name);
    const FootKinematics kinematics =
        footKinematicsAt(*models, point.state, point.input);
    const DifferencedOrientationErrorRates rates = oracleRatesOf(kinematics);
    const AssembledLaw law =
        assembledLaw(gains, kinematics, rates.rate, rates.acceleration,
                     rates.rateScale, groundHeightOf(*models));
    const vector_t value =
        constraint->getValue(kTime, point.state, point.input, PreComputation());
    ASSERT_EQ(value.size(), 6);
    for (Eigen::Index row = 0; row < 6; ++row) {
      EXPECT_LE(std::abs(value(row) - law.value(row)), law.tolerance(row))
          << "row " << row << ": constraint " << value(row) << ", law "
          << law.value(row);
      largestDeviation =
          std::max(largestDeviation,
                   std::abs(value(row) - law.value(row)) / law.tolerance(row));
    }
    // The linearization's value is the value, from the same libraries.
    EXPECT_TRUE(constraint
                    ->getLinearApproximation(kTime, point.state, point.input,
                                             PreComputation())
                    .f.isApprox(value, /*prec=*/1.0e-14));
  }
  absl::PrintF(
      "[stance] largest deviation of the constraint from the assembled law, "
      "relative to its tolerance: %.3g\n",
      largestDeviation);
}

TEST(StanceFootOrientationRates,
     TheRescaledGainsHoldALevelStanceFootAsTheFormerLawDid) {
  const StanceFootModels* absl_nullable models = stanceFootModels();
  ASSERT_NE(models, nullptr);
  const ModelSettings::FootConstraintConfig& gains =
      models->interface->modelSettings().footConstraintConfig;
  // The shipped task file doubled the angular gains of the former law on the
  // raw angular velocity and acceleration (20 and 1), which the rates of a
  // level foot halve (design README section 2.8.6).
  EXPECT_EQ(gains.angularVelocityErrorGain, 40.0);
  EXPECT_EQ(gains.angularAccelerationErrorGain, 2.0);
  const scalar_t formerVelocityGain = 0.5 * gains.angularVelocityErrorGain;
  const scalar_t formerAccelerationGain =
      0.5 * gains.angularAccelerationErrorGain;
  const std::unique_ptr<ZeroAccelerationConstraintCppAd> constraint =
      stanceConstraint(*models, gains);
  ASSERT_NE(constraint, nullptr);
  const vector3_t normal = vector3_t::UnitZ();
  scalar_t largestGyroscopicTerm = 0.0;
  scalar_t smallestTiltedDifference = 1.0e300;
  for (const StancePoint& point : stancePoints(*models)) {
    SCOPED_TRACE(point.name);
    const FootKinematics kinematics =
        footKinematicsAt(*models, point.state, point.input);
    const vector_t value =
        constraint->getValue(kTime, point.state, point.input, PreComputation());
    // The former law: the angular velocity and acceleration in place of the
    // error's rates, with the former gains.
    ModelSettings::FootConstraintConfig formerGains = gains;
    formerGains.angularVelocityErrorGain = formerVelocityGain;
    formerGains.angularAccelerationErrorGain = formerAccelerationGain;
    const AssembledLaw former =
        assembledLaw(formerGains, kinematics, kinematics.angularVelocity,
                     kinematics.angularAcceleration,
                     /*rateScale=*/0.0, groundHeightOf(*models));
    const vector6_t difference = value - former.value;
    const scalar_t angularScale =
        gains.angularVelocityErrorGain * kinematics.angularVelocity.norm() +
        gains.angularAccelerationErrorGain *
            (kinematics.angularAcceleration.norm() +
             kinematics.angularVelocity.squaredNorm()) +
        gains.orientationErrorGain + 1.0;
    // The linear rows are unchanged.
    EXPECT_LE(difference.head<3>().cwiseAbs().maxCoeff(),
              former.tolerance.head<3>().maxCoeff());
    if (point.level) {
      // de/dt = w / 2 and d2e/dt2 = alpha / 2 + 1/2 (n.w)(n x w) at a level
      // foot: the difference is Ka_former (n.w)(n x w).
      const vector3_t gyroscopicTerm = formerAccelerationGain *
                                       normal.dot(kinematics.angularVelocity) *
                                       normal.cross(kinematics.angularVelocity);
      EXPECT_LE((difference.tail<3>() - gyroscopicTerm).cwiseAbs().maxCoeff(),
                kLibraryRoundOff * angularScale)
          << difference.tail<3>().transpose() << "\nvs\n"
          << gyroscopicTerm.transpose();
      if (point.atRest) {
        EXPECT_LE(difference.tail<3>().cwiseAbs().maxCoeff(),
                  kLibraryRoundOff * angularScale)
            << "at rest the laws are one";
      }
      largestGyroscopicTerm =
          std::max(largestGyroscopicTerm, gyroscopicTerm.norm());
    } else {
      // A tilted foot: the difference is the rates' departure from half the
      // angular velocity and acceleration.
      const DifferencedOrientationErrorRates rates = oracleRatesOf(kinematics);
      const vector3_t expected =
          gains.angularVelocityErrorGain *
              (rates.rate - 0.5 * kinematics.angularVelocity) +
          gains.angularAccelerationErrorGain *
              (rates.acceleration - 0.5 * kinematics.angularAcceleration);
      const AssembledLaw law =
          assembledLaw(gains, kinematics, rates.rate, rates.acceleration,
                       rates.rateScale, groundHeightOf(*models));
      EXPECT_LE((difference.tail<3>() - expected).cwiseAbs().maxCoeff(),
                law.tolerance.tail<3>().maxCoeff());
      smallestTiltedDifference =
          std::min(smallestTiltedDifference,
                   difference.tail<3>().cwiseAbs().maxCoeff() / angularScale);
    }
  }
  EXPECT_GT(largestGyroscopicTerm, 0.01)
      << "the case under test: level feet that turn about the normal and tilt "
         "at once";
  EXPECT_GT(smallestTiltedDifference, 1.0e-3)
      << "a tilted foot is held by another law than the former one";
  absl::PrintF(
      "[stance] level feet: largest |Ka_former (n.w)(n x w)| = %.3g; tilted "
      "feet: smallest difference from the former law, "
      "relative to the angular rows' scale: %.3g\n",
      largestGyroscopicTerm, smallestTiltedDifference);
}

/** The map M = de/dw from the angular velocity to the rate of the tilt error,
 * plus 1/2 n v' (the centroidal map). */
matrix3_t frozenRateMap(const matrix3_t& rotation) {
  const vector3_t n = vector3_t::UnitZ();
  const vector3_t v = rotation.col(2);
  const vector3_t axis = v.cross(n);
  const scalar_t root = std::sqrt(2.0 * (1.0 + v.dot(n)));
  return (v.dot(n) * matrix3_t::Identity() - v * n.transpose()) / root +
         axis * axis.transpose() / (root * root * root) +
         0.5 * n * v.transpose();
}

/** A Jacobian by differences and its entrywise error bound. Passive data. */
struct BoundedDifferences {
  matrix_t value;
  matrix_t bound;
};

/**
 * Returns the differences of the value `valueAt(i, s)` at the step s along the
 * curve of the column i, for every column (Richardson's extrapolation at h, h /
 * 2 and h / 4, h = `step`), with the bound |fine - coarse| + 6 delta / h, delta
 * = kEvaluationRelativeError of each row's largest magnitude, floor 1.
 */
BoundedDifferences boundedDifferences(
    Eigen::Index rows,
    Eigen::Index columns,
    scalar_t step,
    absl::FunctionRef<vector_t(Eigen::Index, scalar_t)> valueAt) {
  BoundedDifferences differences{.value = matrix_t(rows, columns),
                                 .bound = matrix_t(rows, columns)};
  const std::array<scalar_t, 6> fractions = {1.0, -1.0, 0.5, -0.5, 0.25, -0.25};
  for (Eigen::Index i = 0; i < columns; ++i) {
    std::array<vector_t, 6> samples;
    vector_t magnitude = vector_t::Zero(rows);
    for (size_t s = 0; s < fractions.size(); ++s) {
      samples[s] = valueAt(i, fractions[s] * step);
      magnitude = magnitude.cwiseMax(samples[s].cwiseAbs());
    }
    const scalar_t h = step;
    const vector_t coarse = (4.0 * (samples[2] - samples[3]) / h -
                             (samples[0] - samples[1]) / (2.0 * h)) /
                            3.0;
    const vector_t fine = (4.0 * (samples[4] - samples[5]) / (0.5 * h) -
                           (samples[2] - samples[3]) / h) /
                          3.0;
    differences.value.col(i) = fine;
    differences.bound.col(i) =
        (fine - coarse).cwiseAbs() +
        (6.0 * kEvaluationRelativeError / h) * magnitude.cwiseMax(1.0);
  }
  return differences;
}

/** The largest |reported - differences| relative to its entry's bound plus the
 * reported entry's round-off. */
scalar_t largestResidualToBound(const matrix_t& reported,
                                const BoundedDifferences& differences) {
  scalar_t largest = 0.0;
  for (Eigen::Index row = 0; row < reported.rows(); ++row) {
    const scalar_t roundOff =
        kJacobianRoundOff *
        std::max(1.0, reported.row(row).cwiseAbs().maxCoeff());
    for (Eigen::Index col = 0; col < reported.cols(); ++col) {
      const scalar_t residual =
          std::abs(reported(row, col) - differences.value(row, col));
      largest = std::max(largest,
                         residual / (differences.bound(row, col) + roundOff));
    }
  }
  return largest;
}

/**
 * `jacobian` with the largest entry of its `rows` rows from `firstRow` on moved
 * by kJacobianMutation of max(1, itself): of all rows, or of the orientation
 * rows (3 to 5), which the pose rates' libraries changed.
 */
matrix_t withLargestEntryMoved(const matrix_t& jacobian,
                               Eigen::Index firstRow,
                               Eigen::Index rows) {
  Eigen::Index row = 0;
  Eigen::Index col = 0;
  const scalar_t largest =
      jacobian.middleRows(firstRow, rows).cwiseAbs().maxCoeff(&row, &col);
  matrix_t moved = jacobian;
  moved(firstRow + row, col) += kJacobianMutation * std::max(1.0, largest);
  return moved;
}

/** The Jacobians of the foot's world-frame angular velocity w and acceleration
 * alpha in the state and the input. Passive data. */
struct AngularRateJacobians {
  matrix_t velocityDx;
  matrix_t velocityDu;
  matrix_t accelerationDx;
  matrix_t accelerationDu;
};

/**
 * The AngularRateJacobians at `point`, `kinematics` its foot's: those the pose
 * rates' libraries tape, read off their orientation rows against the plane
 * normal -v, v the foot normal, where both rates fall back to w / 2 and alpha /
 * 2 (end_effector/OrientationErrorWrtPlaneRates.h), the libraries' Jacobians
 * the derivatives of that branch. Expects the fallback's values to be
 * Pinocchio's w / 2 and alpha / 2 in double.
 */
AngularRateJacobians angularRateJacobiansAt(const StanceFootModels& models,
                                            const FootKinematics& kinematics,
                                            const StancePoint& point) {
  const std::vector<vector3_t> awayFromTheFoot = {-kinematics.rotation.col(2)};
  const VectorFunctionLinearApproximation halfRate =
      models.dynamics
          ->getPoseWrtPlaneRateLinearApproximation(point.state, point.input,
                                                   awayFromTheFoot)
          .front();
  const VectorFunctionLinearApproximation halfAcceleration =
      models.dynamics
          ->getPoseWrtPlaneAccelerationLinearApproximation(
              point.state, point.input, awayFromTheFoot)
          .front();
  const scalar_t velocityScale =
      std::max(1.0, kinematics.angularVelocity.norm());
  const scalar_t accelerationScale =
      std::max(1.0, kinematics.angularAcceleration.norm());
  EXPECT_LE((2.0 * halfRate.f.tail<3>() - kinematics.angularVelocity)
                .cwiseAbs()
                .maxCoeff(),
            kLibraryRoundOff * velocityScale)
      << "the fallback is not w / 2";
  EXPECT_LE(
      (2.0 * halfAcceleration.f.tail<3>() - kinematics.angularAcceleration)
          .cwiseAbs()
          .maxCoeff(),
      kLibraryRoundOff * accelerationScale)
      << "the fallback is not alpha / 2";
  return AngularRateJacobians{
      .velocityDx = 2.0 * halfRate.dfdx.bottomRows(3),
      .velocityDu = 2.0 * halfRate.dfdu.bottomRows(3),
      .accelerationDx = 2.0 * halfAcceleration.dfdx.bottomRows(3),
      .accelerationDu = 2.0 * halfAcceleration.dfdu.bottomRows(3)};
}

TEST(StanceFootOrientationRates,
     TheLinearizationIsTheDerivativeAlongTheStateManifold) {
  const StanceFootModels* absl_nullable models = stanceFootModels();
  ASSERT_NE(models, nullptr);
  const ModelSettings::FootConstraintConfig& gains =
      models->interface->modelSettings().footConstraintConfig;
  const std::unique_ptr<ZeroAccelerationConstraintCppAd> created =
      stanceConstraint(*models, gains);
  ASSERT_NE(created, nullptr);
  const ZeroAccelerationConstraintCppAd& constraint = *created;
  const StateManifold& manifold = *models->interface->getStateManifold();
  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(manifold.getTangentDim());
  const std::vector<vector3_t> ground = {vector3_t::UnitZ()};
  scalar_t largestExact = 0.0;
  scalar_t smallestMutation = 1.0e300;
  scalar_t smallestOrientationMutation = 1.0e300;
  scalar_t smallestFrozen = 1.0e300;
  for (const StancePoint& point : stancePoints(*models)) {
    SCOPED_TRACE(point.name);
    const VectorFunctionLinearApproximation linearization =
        constraint.getLinearApproximation(kTime, point.state, point.input,
                                          PreComputation());
    matrix_t stateJacobian = linearization.dfdx;
    manifold.pullBackStateColumns(point.state, stateJacobian);
    ASSERT_EQ(stateJacobian.cols(), tangentDim);
    const BoundedDifferences stateDifferences = boundedDifferences(
        /*rows=*/6, tangentDim, kStateJacobianStep,
        [&](Eigen::Index i, scalar_t s) {
          vector_t moved;
          manifold.retract(point.state, vector_t::Unit(tangentDim, i), s,
                           moved);
          return constraint.getValue(kTime, moved, point.input,
                                     PreComputation());
        });
    const BoundedDifferences inputDifferences =
        boundedDifferences(/*rows=*/6, point.input.size(), kInputJacobianStep,
                           [&](Eigen::Index i, scalar_t s) {
                             vector_t moved = point.input;
                             moved(i) += s;
                             return constraint.getValue(
                                 kTime, point.state, moved, PreComputation());
                           });
    const scalar_t stateResidual =
        largestResidualToBound(stateJacobian, stateDifferences);
    const scalar_t inputResidual =
        largestResidualToBound(linearization.dfdu, inputDifferences);
    EXPECT_LE(stateResidual, 1.0);
    EXPECT_LE(inputResidual, 1.0);
    largestExact = std::max({largestExact, stateResidual, inputResidual});

    // The comparison resolves a change of 1e-6 of the largest entry, of all
    // rows (from row 0) and of the orientation rows (from row 3).
    const std::array<Eigen::Index, 2> firstRows = {0, 3};
    for (const Eigen::Index firstRow : firstRows) {
      const Eigen::Index rows = 6 - firstRow;
      const scalar_t stateMutation = largestResidualToBound(
          withLargestEntryMoved(stateJacobian, firstRow, rows),
          stateDifferences);
      const scalar_t inputMutation = largestResidualToBound(
          withLargestEntryMoved(linearization.dfdu, firstRow, rows),
          inputDifferences);
      EXPECT_GT(stateMutation, 1.0) << "rows from " << firstRow;
      EXPECT_GT(inputMutation, 1.0) << "rows from " << firstRow;
      scalar_t& smallest =
          firstRow == 0 ? smallestMutation : smallestOrientationMutation;
      smallest = std::min({smallest, stateMutation, inputMutation});
    }

    // The frozen-map linearization: the orientation rows' rate maps held at the
    // point, M (and 1/2 n v') applied to the Jacobians of w and of alpha, as
    // the centroidal twist constraint holds its map. Its state Jacobian lacks
    // the map's derivative. Its input Jacobian is the exact one: the state
    // carries the velocities, so w reads no input and the rates read it through
    // alpha alone, d(d2e/dt2 + 1/2 n v.alpha)/du = (M + 1/2 n v') d(alpha)/du.
    const FootKinematics kinematics =
        footKinematicsAt(*models, point.state, point.input);
    const matrix3_t map = frozenRateMap(kinematics.rotation);
    const VectorFunctionLinearApproximation error =
        models->dynamics
            ->getOrientationErrorWrtPlaneLinearApproximation(point.state,
                                                             ground)
            .front();
    const AngularRateJacobians angular =
        angularRateJacobiansAt(*models, kinematics, point);
    matrix_t frozenState = linearization.dfdx;
    frozenState.bottomRows(3) =
        gains.orientationErrorGain * error.dfdx +
        gains.angularVelocityErrorGain * map * angular.velocityDx +
        gains.angularAccelerationErrorGain * map * angular.accelerationDx;
    manifold.pullBackStateColumns(point.state, frozenState);
    matrix_t frozenInput = linearization.dfdu;
    frozenInput.bottomRows(3) =
        gains.angularVelocityErrorGain * map * angular.velocityDu +
        gains.angularAccelerationErrorGain * map * angular.accelerationDu;
    const scalar_t frozen =
        largestResidualToBound(frozenState, stateDifferences);
    EXPECT_GT(frozen, 1.0)
        << "the frozen-map linearization is not the derivative";
    EXPECT_LE(largestResidualToBound(frozenInput, inputDifferences), 1.0)
        << "the frozen map's input Jacobian is the exact one";
    smallestFrozen = std::min(smallestFrozen, frozen);
  }
  absl::PrintF(
      "[stance] linearization: largest residual relative to its bound %.3g; "
      "the comparison rejects a 1e-6 change of the "
      "largest entry by at least %.3g, of the orientation rows' by %.3g, and "
      "the frozen-map linearization's state Jacobian "
      "by at least %.3g times its bound\n",
      largestExact, smallestMutation, smallestOrientationMutation,
      smallestFrozen);
}

}  // namespace
}  // namespace ocs2::humanoid
