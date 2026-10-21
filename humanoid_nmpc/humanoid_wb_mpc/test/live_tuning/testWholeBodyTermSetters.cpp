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

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodyLiveTuningFixture.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/WBMpcPreComputation.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/constraint/EndEffectorDynamicsAccelerationsConstraint.h"
#include "humanoid_wb_mpc/constraint/ZeroAccelerationConstraintCppAd.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsCostHelpers.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsFootCost.h"
#include "humanoid_wb_mpc/cost/JointTorqueCostCppAd.h"
#include "humanoid_wb_mpc/end_effector/PinocchioEndEffectorDynamicsCppAd.h"

/*
 * The setters the whole-body parameter updater writes the running problem with: each leaves a term exactly as a term
 * built with the new values, in value and derivatives, at states in swing and in stance. The CppAD terms load the
 * libraries the MPCs of the test program taped, by the MPC's own model names.
 */
namespace ocs2::humanoid::live_tuning_test {
namespace {

constexpr size_t kNumPoints = 10;

void expectSameQuadraticApproximation(const ScalarFunctionQuadraticApproximation& expected,
                                      const ScalarFunctionQuadraticApproximation& actual,
                                      const std::string& where) {
  EXPECT_EQ(expected.f, actual.f) << where;
  EXPECT_TRUE(expected.dfdx == actual.dfdx) << where;
  EXPECT_TRUE(expected.dfdu == actual.dfdu) << where;
  EXPECT_TRUE(expected.dfdxx == actual.dfdxx) << where;
  EXPECT_TRUE(expected.dfdux == actual.dfdux) << where;
  EXPECT_TRUE(expected.dfduu == actual.dfduu) << where;
}

void expectSameLinearApproximation(const VectorFunctionLinearApproximation& expected,
                                   const VectorFunctionLinearApproximation& actual,
                                   const std::string& where) {
  EXPECT_TRUE(expected.f == actual.f) << where;
  EXPECT_TRUE(expected.dfdx == actual.dfdx) << where;
  EXPECT_TRUE(expected.dfdu == actual.dfdu) << where;
}

/** Expects `retuned` and `fresh` to agree bitwise in value and quadratic approximation at every point. */
void expectSameCost(const StateInputCost& fresh, const StateInputCost& retuned, const std::vector<EvaluationPoint>& points) {
  const TargetTrajectories target;
  const PreComputation preComputation;
  for (size_t i = 0; i < points.size(); ++i) {
    const EvaluationPoint& point = points[i];
    const std::string where = absl::StrCat("point ", i, " at t = ", point.time);
    EXPECT_EQ(fresh.getValue(point.time, point.state, point.input, target, preComputation),
              retuned.getValue(point.time, point.state, point.input, target, preComputation))
        << where;
    expectSameQuadraticApproximation(fresh.getQuadraticApproximation(point.time, point.state, point.input, target, preComputation),
                                     retuned.getQuadraticApproximation(point.time, point.state, point.input, target, preComputation),
                                     where);
  }
}

EndEffectorDynamicsWeights retunedFootWeights() {
  EndEffectorDynamicsWeights weights;
  weights.contactPositionErrorWeight = vector3_t(1.0, 2.0, 3.0);
  weights.contactOrientationErrorWeight = vector3_t(5000.0, 7000.0, 11.0);
  weights.contactLinearVelocityErrorWeight = vector3_t(4.0, 6.0, 0.5);
  weights.contactAngularVelocityErrorWeight = vector3_t(1.5, 2.5, 3.5);
  weights.contactLinearAccelerationErrorWeight = vector3_t(0.02, 0.03, 0.04);
  weights.contactAngularAccelerationErrorWeight = vector3_t(0.05, 0.06, 0.07);
  return weights;
}

/** The foot-constraint gains of `interface` with every gain moved to a value of its own. */
ModelSettings::FootConstraintConfig retunedFootGains(const WBMpcInterface& interface) {
  ModelSettings::FootConstraintConfig gains = interface.modelSettings().footConstraintConfig;
  gains.positionErrorGain_z *= 1.3;
  gains.orientationErrorGain *= 0.7;
  gains.linearVelocityErrorGain_z *= 1.1;
  gains.linearVelocityErrorGain_xy *= 0.9;
  gains.angularVelocityErrorGain *= 1.2;
  gains.linearAccelerationErrorGain_z *= 1.4;
  gains.linearAccelerationErrorGain_xy *= 0.6;
  gains.angularAccelerationErrorGain *= 1.5;
  return gains;
}

/** The foot cost of contact `foot` of `interface` with `weights`, or null (a test failure) when Create() refuses them. */
std::unique_ptr<EndEffectorDynamicsFootCost> footCost(WBMpcInterface& interface, const EndEffectorDynamicsWeights& weights, size_t foot) {
  absl::StatusOr<std::unique_ptr<EndEffectorDynamicsFootCost>> created = EndEffectorDynamicsFootCost::Create(
      *interface.getSwitchedModelReferenceManagerPtr(), weights, interface.getPinocchioInterface(), interface.getMpcRobotModelAD(), foot,
      EndEffectorDynamicsFootCost::libraryModelName(interface.modelSettings().contactNames[foot]), interface.modelSettings());
  EXPECT_TRUE(created.ok()) << created.status();
  return created.ok() ? *std::move(created) : nullptr;
}

/** The joint torque cost of `interface` with `weights`, or null (a test failure) when Create() refuses them. */
std::unique_ptr<JointTorqueCostCppAd> jointTorqueCost(WBMpcInterface& interface, const vector_t& weights) {
  absl::StatusOr<std::unique_ptr<JointTorqueCostCppAd>> created =
      JointTorqueCostCppAd::Create(weights, interface.getPinocchioInterface(), interface.getMpcRobotModelAD(),
                                   JointTorqueCostCppAd::kLibraryCostName, interface.modelSettings());
  EXPECT_TRUE(created.ok()) << created.status();
  return created.ok() ? *std::move(created) : nullptr;
}

TEST(WholeBodyTermSetters, TheFootCostsSetWeightsEqualsAFootCostBuiltWithThem) {
  WBMpcInterface* absl_nullable const interface = shippedWholeBodyMpc();
  ASSERT_NE(interface, nullptr);
  expectQuaternionLayout(*interface);
  const std::vector<EvaluationPoint> points = evaluationPoints(*interface, kNumPoints);
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    scheduleOneSwing(*interface, foot);
    const std::string& footName = interface->modelSettings().contactNames[foot];
    const std::unique_ptr<EndEffectorDynamicsFootCost> retuned = footCost(*interface, EndEffectorDynamicsWeights(), foot);
    const std::unique_ptr<EndEffectorDynamicsFootCost> fresh = footCost(*interface, retunedFootWeights(), foot);
    const std::unique_ptr<EndEffectorDynamicsFootCost> unchanged = footCost(*interface, EndEffectorDynamicsWeights(), foot);
    ASSERT_TRUE(retuned != nullptr && fresh != nullptr && unchanged != nullptr);
    ASSERT_EQ(retuned->setWeights(retunedFootWeights()), absl::OkStatus());
    SCOPED_TRACE(footName);
    expectSameCost(*fresh, *retuned, points);
    // Positive control: the weights reach the cost.
    const PreComputation preComputation;
    const EvaluationPoint& swing = points.front();
    EXPECT_NE(unchanged->getValue(swing.time, swing.state, swing.input, TargetTrajectories(), preComputation),
              fresh->getValue(swing.time, swing.state, swing.input, TargetTrajectories(), preComputation));
    // A copy, the solver's per-worker clone, carries them.
    const std::unique_ptr<EndEffectorDynamicsFootCost> workerCopy(retuned->clone());
    expectSameCost(*fresh, *workerCopy, points);
  }
}

TEST(WholeBodyTermSetters, ANegativeOrNanFootCostWeightIsRefusedAndChangesNothing) {
  // The cost weighs each error by the square root of its weight: a negative weight would be a NaN in the running cost.
  WBMpcInterface* absl_nullable const interface = shippedWholeBodyMpc();
  ASSERT_NE(interface, nullptr);
  expectQuaternionLayout(*interface);
  const std::vector<EvaluationPoint> points = evaluationPoints(*interface, /*count=*/2);
  const size_t foot = 0;
  scheduleOneSwing(*interface, foot);
  EndEffectorDynamicsWeights negative = retunedFootWeights();
  negative.contactLinearVelocityErrorWeight.y() = -1.0;
  EndEffectorDynamicsWeights nan = retunedFootWeights();
  nan.contactAngularAccelerationErrorWeight.z() = std::numeric_limits<scalar_t>::quiet_NaN();
  const std::unique_ptr<EndEffectorDynamicsFootCost> cost = footCost(*interface, retunedFootWeights(), foot);
  const std::unique_ptr<EndEffectorDynamicsFootCost> original = footCost(*interface, retunedFootWeights(), foot);
  ASSERT_TRUE(cost != nullptr && original != nullptr);
  for (const EndEffectorDynamicsWeights& refusedWeights : {negative, nan}) {
    EXPECT_EQ(cost->setWeights(refusedWeights).code(), absl::StatusCode::kInvalidArgument);
    const absl::StatusOr<std::unique_ptr<EndEffectorDynamicsFootCost>> created = EndEffectorDynamicsFootCost::Create(
        *interface->getSwitchedModelReferenceManagerPtr(), refusedWeights, interface->getPinocchioInterface(),
        interface->getMpcRobotModelAD(), foot, EndEffectorDynamicsFootCost::libraryModelName(interface->modelSettings().contactNames[foot]),
        interface->modelSettings());
    EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument);
  }
  // The refused updates kept the running weights.
  expectSameCost(*original, *cost, points);
}

TEST(WholeBodyTermSetters, TheJointTorqueCostsSetWeightsEqualsAJointTorqueCostBuiltWithThem) {
  // The variant's MPC tapes the cost's library; the terms below load it.
  WBMpcInterface* absl_nullable const interface = variantWholeBodyMpc();
  ASSERT_NE(interface, nullptr);
  expectQuaternionLayout(*interface);
  const std::vector<EvaluationPoint> points = evaluationPoints(*interface, kNumPoints);
  const Eigen::Index numJoints = static_cast<Eigen::Index>(interface->getMpcRobotModel().getJointDim());
  vector_t weights(numJoints);
  for (Eigen::Index j = 0; j < numJoints; ++j) weights(j) = 0.5 + 0.25 * static_cast<scalar_t>(j % 7);
  const std::unique_ptr<JointTorqueCostCppAd> retuned = jointTorqueCost(*interface, vector_t::Ones(numJoints));
  const std::unique_ptr<JointTorqueCostCppAd> fresh = jointTorqueCost(*interface, weights);
  ASSERT_TRUE(retuned != nullptr && fresh != nullptr);
  ASSERT_EQ(retuned->setWeights(weights), absl::OkStatus());
  expectSameCost(*fresh, *retuned, points);
  const std::unique_ptr<JointTorqueCostCppAd> workerCopy(retuned->clone());
  expectSameCost(*fresh, *workerCopy, points);
}

TEST(WholeBodyTermSetters, AJointTorqueWeightCountThatIsNotOnePerJointOrANegativeOrNanWeightIsRefusedAndChangesNothing) {
  WBMpcInterface* absl_nullable const interface = variantWholeBodyMpc();
  ASSERT_NE(interface, nullptr);
  expectQuaternionLayout(*interface);
  const std::vector<EvaluationPoint> points = evaluationPoints(*interface, /*count=*/2);
  const Eigen::Index numJoints = static_cast<Eigen::Index>(interface->getMpcRobotModel().getJointDim());
  const std::unique_ptr<JointTorqueCostCppAd> cost = jointTorqueCost(*interface, vector_t::Ones(numJoints));
  const std::unique_ptr<JointTorqueCostCppAd> original = jointTorqueCost(*interface, vector_t::Ones(numJoints));
  ASSERT_TRUE(cost != nullptr && original != nullptr);
  vector_t negative = vector_t::Constant(numJoints, /*value=*/2.0);
  negative(numJoints - 1) = -0.5;
  vector_t nan = vector_t::Constant(numJoints, /*value=*/2.0);
  nan(0) = std::numeric_limits<scalar_t>::quiet_NaN();
  for (const vector_t& refusedWeights : {vector_t(vector_t::Constant(numJoints + 1, /*value=*/2.0)), negative, nan}) {
    const absl::Status refused = cost->setWeights(refusedWeights);
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
    const absl::StatusOr<std::unique_ptr<JointTorqueCostCppAd>> created =
        JointTorqueCostCppAd::Create(refusedWeights, interface->getPinocchioInterface(), interface->getMpcRobotModelAD(),
                                     JointTorqueCostCppAd::kLibraryCostName, interface->modelSettings());
    EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument);
  }
  expectSameCost(*original, *cost, points);
}

TEST(WholeBodyTermSetters, TheStanceConfigHasEachGainOnItsDiagonal) {
  ModelSettings::FootConstraintConfig gains;
  gains.positionErrorGain_z = 2.0;
  gains.orientationErrorGain = 3.0;
  gains.linearVelocityErrorGain_xy = 5.0;
  gains.linearVelocityErrorGain_z = 7.0;
  gains.angularVelocityErrorGain = 11.0;
  gains.linearAccelerationErrorGain_xy = 13.0;
  gains.linearAccelerationErrorGain_z = 17.0;
  gains.angularAccelerationErrorGain = 19.0;
  const EndEffectorDynamicsAccelerationsConstraint::Config config = stanceFootAccelerationConstraintConfig(gains);
  EXPECT_TRUE(config.b == vector_t::Zero(6));
  const vector_t ax = (vector_t(6) << 0.0, 0.0, 2.0, 3.0, 3.0, 3.0).finished();
  const vector_t av = (vector_t(6) << 5.0, 5.0, 7.0, 11.0, 11.0, 11.0).finished();
  const vector_t aa = (vector_t(6) << 13.0, 13.0, 17.0, 19.0, 19.0, 19.0).finished();
  EXPECT_TRUE(config.Ax == matrix_t(ax.asDiagonal())) << config.Ax;
  EXPECT_TRUE(config.Av == matrix_t(av.asDiagonal())) << config.Av;
  EXPECT_TRUE(config.Aa == matrix_t(aa.asDiagonal())) << config.Aa;
  // A position or orientation gain of 0 leaves its rows of Ax empty, as the interface always built them.
  gains.positionErrorGain_z = 0.0;
  gains.orientationErrorGain = 0.0;
  EXPECT_TRUE(stanceFootAccelerationConstraintConfig(gains).Ax == matrix_t::Zero(6, 6));
}

/** CppAD end-effector dynamics and the robot model it keeps a pointer to, declared first so that it outlives them. */
struct FootDynamics {
  std::unique_ptr<WBAccelMpcRobotModel<ad_scalar_t>> model;
  std::unique_ptr<PinocchioEndEffectorDynamicsCppAd> dynamics;
};

/** Returns the end-effector dynamics of the contact `foot`, loaded from the library the MPC taped under its name. */
FootDynamics footDynamicsOf(WBMpcInterface& interface, size_t foot) {
  const ModelSettings& settings = interface.modelSettings();
  FootDynamics footDynamics;
  footDynamics.model = std::make_unique<WBAccelMpcRobotModel<ad_scalar_t>>(settings);
  const std::string& footName = settings.contactNames[foot];
  footDynamics.dynamics = std::make_unique<PinocchioEndEffectorDynamicsCppAd>(
      interface.getPinocchioInterface(), *footDynamics.model, std::vector<std::string>{footName}, footName, settings.modelFolderCppAd,
      /*recompileLibraries=*/false, /*verbose=*/false);
  return footDynamics;
}

/** Expects the constraints to agree bitwise in value and linear approximation at every point. */
void expectSameConstraint(const StateInputConstraint& expected,
                          const StateInputConstraint& actual,
                          const std::vector<EvaluationPoint>& points) {
  const PreComputation preComputation;
  for (size_t i = 0; i < points.size(); ++i) {
    const EvaluationPoint& point = points[i];
    const std::string where = absl::StrCat("point ", i, " at t = ", point.time);
    EXPECT_TRUE(expected.getValue(point.time, point.state, point.input, preComputation) ==
                actual.getValue(point.time, point.state, point.input, preComputation))
        << where;
    expectSameLinearApproximation(expected.getLinearApproximation(point.time, point.state, point.input, preComputation),
                                  actual.getLinearApproximation(point.time, point.state, point.input, preComputation), where);
  }
}

TEST(WholeBodyTermSetters, TheStanceConstraintsConfigureEqualsAStanceConstraintCreatedWithIt) {
  WBMpcInterface* absl_nullable const interface = shippedWholeBodyMpc();
  ASSERT_NE(interface, nullptr);
  expectQuaternionLayout(*interface);
  const std::vector<EvaluationPoint> points = evaluationPoints(*interface, kNumPoints);
  const SwitchedModelReferenceManager& referenceManager = *interface->getSwitchedModelReferenceManagerPtr();
  const OptimalControlProblem& problem = interface->getOptimalControlProblem();
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    const std::string& footName = interface->modelSettings().contactNames[foot];
    SCOPED_TRACE(footName);
    const FootDynamics footDynamics = footDynamicsOf(*interface, foot);
    const PinocchioEndEffectorDynamicsCppAd& dynamics = *footDynamics.dynamics;
    const EndEffectorDynamicsAccelerationsConstraint::Config shipped =
        stanceFootAccelerationConstraintConfig(interface->modelSettings().footConstraintConfig);
    const EndEffectorDynamicsAccelerationsConstraint::Config retunedConfig =
        stanceFootAccelerationConstraintConfig(retunedFootGains(*interface));

    // The interface builds its stance constraint from the shipped gains through the same function.
    absl::StatusOr<std::unique_ptr<ZeroAccelerationConstraintCppAd>> asShipped =
        ZeroAccelerationConstraintCppAd::Create(referenceManager, dynamics, foot, shipped);
    ASSERT_TRUE(asShipped.ok()) << asShipped.status();
    const StateInputConstraint& built = problem.equalityConstraintPtr->get(absl::StrCat(footName, "_zeroVelocity"));
    expectSameConstraint(**asShipped, built, points);

    absl::StatusOr<std::unique_ptr<ZeroAccelerationConstraintCppAd>> fresh =
        ZeroAccelerationConstraintCppAd::Create(referenceManager, dynamics, foot, retunedConfig);
    ASSERT_TRUE(fresh.ok()) << fresh.status();
    (*asShipped)->configure(retunedConfig);
    expectSameConstraint(**fresh, **asShipped, points);
    const std::unique_ptr<ZeroAccelerationConstraintCppAd> workerCopy((*asShipped)->clone());
    expectSameConstraint(**fresh, *workerCopy, points);
  }
}

void expectSameSwingConfigs(const WBMpcPreComputation& expected, const WBMpcPreComputation& actual, const std::string& where) {
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    const EndEffectorDynamicsLinearAccConstraint::Config& e = expected.getEeNormalAccelerationConstraintConfigs()[foot];
    const EndEffectorDynamicsLinearAccConstraint::Config& a = actual.getEeNormalAccelerationConstraintConfigs()[foot];
    EXPECT_TRUE(e.b == a.b) << where << ", foot " << foot;
    EXPECT_TRUE(e.Ax == a.Ax) << where << ", foot " << foot;
    EXPECT_TRUE(e.Av == a.Av) << where << ", foot " << foot;
    EXPECT_TRUE(e.Aa == a.Aa) << where << ", foot " << foot;
  }
}

TEST(WholeBodyTermSetters, ThePrecomputationsSwingGainsEqualAPrecomputationSeededWithThem) {
  WBMpcInterface* absl_nullable const interface = shippedWholeBodyMpc();
  ASSERT_NE(interface, nullptr);
  expectQuaternionLayout(*interface);
  scheduleOneSwing(*interface, /*swingFoot=*/0);
  const std::vector<EvaluationPoint> points = evaluationPoints(*interface, kNumPoints);
  const SwingTrajectoryPlanner& planner = *interface->getSwitchedModelReferenceManagerPtr()->getSwingTrajectoryPlanner();
  const ModelSettings::FootConstraintConfig gains = retunedFootGains(*interface);

  WBMpcPreComputation retuned(interface->getPinocchioInterface(), planner, interface->getMpcRobotModel());
  retuned.setSwingFootGains(
      {.linearVelocityErrorGainZ = gains.linearVelocityErrorGain_z, .linearAccelerationErrorGainZ = gains.linearAccelerationErrorGain_z});
  retuned.setNormalVelocityPositionErrorGain(gains.positionErrorGain_z);
  // The model settings of a task file with those gains, which a pre-computation is seeded from.
  mpc_config::TaskFile seededTask = shippedTaskFile();
  seededTask.model_settings.foot_constraint.position_error_gain_z = gains.positionErrorGain_z;
  seededTask.model_settings.foot_constraint.linear_velocity_error_gain_z = gains.linearVelocityErrorGain_z;
  seededTask.model_settings.foot_constraint.linear_acceleration_error_gain_z = gains.linearAccelerationErrorGain_z;
  const absl::StatusOr<ModelSettings> seededSettings =
      ModelSettings::Create(seededTask, g1WholeBodyFiles().urdfFile, "wb_mpc_", /*verbose=*/false);
  ASSERT_TRUE(seededSettings.ok()) << seededSettings.status();
  const WBAccelMpcRobotModel<scalar_t> seededModel(*seededSettings);
  WBMpcPreComputation seeded(interface->getPinocchioInterface(), planner, seededModel);
  WBMpcPreComputation unchanged(interface->getPinocchioInterface(), planner, interface->getMpcRobotModel());
  const std::unique_ptr<WBMpcPreComputation> workerCopy(retuned.clone());
  EXPECT_EQ(workerCopy->getSwingFootGains().linearVelocityErrorGainZ, gains.linearVelocityErrorGain_z);
  EXPECT_EQ(workerCopy->getSwingFootGains().linearAccelerationErrorGainZ, gains.linearAccelerationErrorGain_z);
  EXPECT_EQ(workerCopy->getNormalVelocityPositionErrorGain(), gains.positionErrorGain_z);

  for (size_t i = 0; i < points.size(); ++i) {
    const EvaluationPoint& point = points[i];
    const std::string where = absl::StrCat("point ", i, " at t = ", point.time);
    retuned.request(Request::Constraint, point.time, point.state, point.input);
    seeded.request(Request::Constraint, point.time, point.state, point.input);
    workerCopy->request(Request::Constraint, point.time, point.state, point.input);
    unchanged.request(Request::Constraint, point.time, point.state, point.input);
    expectSameSwingConfigs(seeded, retuned, where);
    expectSameSwingConfigs(seeded, *workerCopy, where);
    // Positive control: in the swing, the gains reach the coefficients.
    if (point.time > kSwingStart && point.time < kSwingEnd) {
      EXPECT_FALSE(unchanged.getEeNormalAccelerationConstraintConfigs()[0].b == retuned.getEeNormalAccelerationConstraintConfigs()[0].b)
          << where;
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid::live_tuning_test
