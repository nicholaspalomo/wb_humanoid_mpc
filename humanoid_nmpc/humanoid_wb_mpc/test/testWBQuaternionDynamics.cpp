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
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/crba.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"
#include "humanoid_wb_mpc/dynamics/WBAccelDynamicsAD.h"
#include "robot_core/ResourcePaths.h"

/*
 * The whole-body flow map on the quaternion layout (design section 2.5, README
 * 5(c)), on the G1:
 *
 *   x = [p_W, xi, q_j, dp_W, w_B, dq_j],  f(x, u) = [dp_W, 1/2 G(xi) w_B, dq_j,
 * a_b(q_hat, v, u), qdd_j],
 *
 * with q_hat the Pinocchio configuration of the safely normalized xi and a_b
 * the base acceleration of the Composite(Translation, Spherical) root under the
 * contact wrenches. So f is homogeneous of degree 0 in xi but for the
 * quaternion rate, which is linear in it: f(lambda xi) = [f_other, lambda
 * f_xi]; and f(-xi) flips the quaternion rate alone, -xi being the same
 * rotation. The tape the MPC compiles (WBAccelDynamicsAD) is the double flow
 * map, and its Jacobians are the flow map's derivatives in every ambient slot
 * and input. At a pitch of 90 degrees nothing is singular: the dynamics is
 * finite and the mass matrix has the eigenvalues it has upright. The quaternion
 * rate keeps the norm to first order (xi' G(xi) = 0), and RK4 at a step of the
 * rollout's time_step drifts by less than 1e-8 over the MPC's horizon at |w| =
 * 5 rad/s; over one node of the SQP, a single step of its dt, by its truncation
 * alone, which does not add up. The CppAD library of the flow map is generated
 * and compiled once, into a directory made for the run.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
constexpr scalar_t kFiniteDifferenceTolerance =
    1.0e-5;  // relative to max(1, the largest entry of the Jacobian)

/** The G1 whole-body flow map, in double precision and as the MPC's compiled
 * CppAD library. Built once; never destroyed. */
struct WholeBodyDynamics {
  mpc_config::TaskFile task;
  std::unique_ptr<ModelSettings> modelSettings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model;
  // Retained by `dynamics` (WBAccelDynamicsAD keeps a pointer to it), so
  // declared before it.
  std::unique_ptr<WBAccelMpcRobotModel<ad_scalar_t>> modelCppAd;
  std::unique_ptr<WBAccelDynamicsAD> dynamics;
};

absl::StatusOr<std::unique_ptr<WholeBodyDynamics>> loadDynamics() {
  auto loaded = std::make_unique<WholeBodyDynamics>();
  // LINT.IfChange(robot_files)
  ASSIGN_OR_RETURN(
      const std::string taskFile,
      robot::resolveResourcePath(
          "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto"));
  ASSIGN_OR_RETURN(
      const std::string urdfFile,
      robot::resolveResourcePath(
          "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"));
  // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:quaternion_dynamics_test_data)
  ASSIGN_OR_RETURN(loaded->task, loadTaskFile(taskFile));
  ASSIGN_OR_RETURN(ModelSettings modelSettings,
                   ModelSettings::Create(loaded->task, urdfFile, "wb_mpc_",
                                         /*verbose=*/false));
  // A directory of this run's: the library is taped and compiled here, never
  // loaded from an earlier run.
  modelSettings.modelFolderCppAd = (std::filesystem::path(testing::TempDir()) /
                                    "testWBQuaternionDynamics_cppad")
                                       .string();
  modelSettings.recompileLibrariesCppAd = true;
  modelSettings.verboseCppAd = false;
  loaded->modelSettings =
      std::make_unique<ModelSettings>(std::move(modelSettings));
  ASSIGN_OR_RETURN(PinocchioInterface pinocchioInterface,
                   loadCustomPinocchioInterface(loaded->task, urdfFile,
                                                *loaded->modelSettings));
  loaded->pinocchioInterface =
      std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));
  loaded->model =
      std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*loaded->modelSettings);
  loaded->modelCppAd = std::make_unique<WBAccelMpcRobotModel<ad_scalar_t>>(
      *loaded->modelSettings);
  loaded->dynamics = std::make_unique<WBAccelDynamicsAD>(
      *loaded->pinocchioInterface, loaded->modelCppAd.get(),
      "dynamics_flow_map", *loaded->modelSettings);
  return loaded;
}

/** loadDynamics(), released: nullptr (and a test failure) when it fails. */
WholeBodyDynamics* absl_nullable releasedDynamics() {
  absl::StatusOr<std::unique_ptr<WholeBodyDynamics>> loaded = loadDynamics();
  if (!loaded.ok()) {
    ADD_FAILURE() << loaded.status();
    return nullptr;
  }
  return loaded->release();
}

/** The dynamics of loadDynamics(), built once for the test program, since
 * building it compiles the library. */
WholeBodyDynamics* absl_nullable g1Dynamics() {
  static WholeBodyDynamics* absl_nullable const kDynamics = releasedDynamics();
  return kDynamics;
}

/** The double-precision flow map f(x, u) (computeStateDerivative()). */
vector_t flowMap(WholeBodyDynamics& g1,
                 const vector_t& state,
                 const vector_t& input) {
  return computeStateDerivative<scalar_t>(state, input, *g1.pinocchioInterface,
                                          *g1.model);
}

/** A uniformly random unit quaternion (x, y, z, w) of `generator`. */
vector4_t randomUnitQuaternion(std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  const vector4_t xi(normal(generator), normal(generator), normal(generator),
                     normal(generator));
  return xi / xi.norm();
}

/** A state and an input to evaluate the flow map at. Passive data. */
struct StateInput {
  vector_t state;
  vector_t input;
};

/**
 * A walking-like point of `model`: the base near standing height at the
 * attitude `xi`, the joints within half a radian, the base moving and turning
 * at up to 1 m/s and 1 rad/s, a few hundred newtons and some moment under each
 * sole and every joint accelerating.
 */
StateInput randomPoint(const WBAccelMpcRobotModel<scalar_t>& model,
                       const vector4_t& xi,
                       std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> unit(-1.0, 1.0);
  const Eigen::Index joints = static_cast<Eigen::Index>(model.getJointDim());
  StateInput point{
      .state = model.getNeutralState(),
      .input = vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()))};
  model.setBasePosition(point.state,
                        vector3_t(0.2 * unit(generator), 0.2 * unit(generator),
                                  0.75 + 0.05 * unit(generator)));
  point.state.segment<4>(
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex())) = xi;
  vector_t jointAngles(joints);
  vector_t jointVelocities(joints);
  for (Eigen::Index j = 0; j < joints; ++j) {
    jointAngles(j) = 0.5 * unit(generator);
    jointVelocities(j) = unit(generator);
  }
  model.setJointAngles(point.state, jointAngles);
  model.setBaseLinearVelocity(
      point.state,
      vector3_t(unit(generator), unit(generator), 0.3 * unit(generator)));
  model.setBaseAngularVelocityLocal(
      point.state,
      vector3_t(unit(generator), unit(generator), unit(generator)));
  vector_t input = point.input;
  model.setJointVelocities(point.state, input, jointVelocities);
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    const vector6_t wrench =
        (vector6_t() << 40.0 * unit(generator), 40.0 * unit(generator),
         200.0 + 100.0 * unit(generator), 5.0 * unit(generator),
         5.0 * unit(generator), 2.0 * unit(generator))
            .finished();
    model.setContactWrench(point.input, wrench, contact);
  }
  for (Eigen::Index j = 0; j < joints; ++j)
    point.input(point.input.size() - joints + j) = 3.0 * unit(generator);
  return point;
}

/** max_i |a_i - b_i| / max(1, |a|_inf): the difference of two vectors relative
 * to the first's size. */
scalar_t relativeDifference(const vector_t& a, const vector_t& b) {
  return (a - b).cwiseAbs().maxCoeff() / std::max(1.0, a.cwiseAbs().maxCoeff());
}

/** The rows of a state derivative that are not the quaternion's rate. */
vector_t withoutQuaternionRate(const WBAccelMpcRobotModel<scalar_t>& model,
                               const vector_t& derivative) {
  vector_t rest = derivative;
  rest.segment<4>(
          static_cast<Eigen::Index>(model.getBaseOrientationStartIndex()))
      .setZero();
  return rest;
}

/** The quaternion rate of a state derivative. */
vector4_t quaternionRate(const WBAccelMpcRobotModel<scalar_t>& model,
                         const vector_t& derivative) {
  return derivative.segment<4>(
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex()));
}

TEST(WBQuaternionDynamics,
     TheTapedFlowMapIsTheDoubleFlowMapAndItsJacobiansAreItsDerivatives) {
  WholeBodyDynamics* absl_nullable const g1 = g1Dynamics();
  ASSERT_NE(g1, nullptr);
  const WBAccelMpcRobotModel<scalar_t>& model = *g1->model;
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  std::mt19937 generator(51);
  const PreComputation preComputation;
  scalar_t largestStateError = 0.0;
  scalar_t largestInputError = 0.0;
  for (int sample = 0; sample < 4; ++sample) {
    SCOPED_TRACE(sample);
    const StateInput point =
        randomPoint(model, randomUnitQuaternion(generator), generator);
    const vector_t f = flowMap(*g1, point.state, point.input);
    ASSERT_EQ(static_cast<size_t>(f.size()), model.getStateDim());
    EXPECT_LE(relativeDifference(
                  f, g1->dynamics->computeFlowMap(/*t=*/0.0, point.state,
                                                  point.input, preComputation)),
              1.0e-12);

    const VectorFunctionLinearApproximation linear =
        g1->dynamics->linearApproximation(/*t=*/0.0, point.state, point.input,
                                          preComputation);
    ASSERT_EQ(linear.dfdx.rows(), f.size());
    ASSERT_EQ(linear.dfdx.cols(), point.state.size());
    ASSERT_EQ(linear.dfdu.cols(), point.input.size());
    const scalar_t stateScale =
        std::max(1.0, linear.dfdx.cwiseAbs().maxCoeff());
    for (Eigen::Index j = 0; j < point.state.size(); ++j) {
      vector_t plus = point.state;
      vector_t minus = point.state;
      plus(j) += kFiniteDifferenceStep;
      minus(j) -= kFiniteDifferenceStep;
      const vector_t column =
          (flowMap(*g1, plus, point.input) - flowMap(*g1, minus, point.input)) /
          (2.0 * kFiniteDifferenceStep);
      const scalar_t error =
          (column - linear.dfdx.col(j)).cwiseAbs().maxCoeff() / stateScale;
      largestStateError = std::max(largestStateError, error);
      EXPECT_LE(error, kFiniteDifferenceTolerance) << "state slot " << j;
    }
    const scalar_t inputScale =
        std::max(1.0, linear.dfdu.cwiseAbs().maxCoeff());
    for (Eigen::Index j = 0; j < point.input.size(); ++j) {
      vector_t plus = point.input;
      vector_t minus = point.input;
      plus(j) += kFiniteDifferenceStep;
      minus(j) -= kFiniteDifferenceStep;
      const vector_t column =
          (flowMap(*g1, point.state, plus) - flowMap(*g1, point.state, minus)) /
          (2.0 * kFiniteDifferenceStep);
      const scalar_t error =
          (column - linear.dfdu.col(j)).cwiseAbs().maxCoeff() / inputScale;
      largestInputError = std::max(largestInputError, error);
      EXPECT_LE(error, kFiniteDifferenceTolerance) << "input " << j;
    }

    // Euler's theorem on the quaternion columns: the radial direction xi moves
    // the quaternion rate by itself (degree 1) and nothing else (degree 0).
    const vector4_t xi = point.state.segment<4>(orientationStart);
    const vector_t radial = linear.dfdx.middleCols<4>(orientationStart) * xi;
    EXPECT_LE(withoutQuaternionRate(model, radial).cwiseAbs().maxCoeff(),
              1.0e-9 * stateScale);
    EXPECT_LE((quaternionRate(model, radial) - quaternionRate(model, f))
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-12);
  }
  RecordProperty("largest_state_jacobian_error",
                 absl::StrCat(largestStateError));
  RecordProperty("largest_input_jacobian_error",
                 absl::StrCat(largestInputError));
}

TEST(WBQuaternionDynamics, TheFlowMapIsHomogeneousInTheQuaternion) {
  WholeBodyDynamics* absl_nullable const g1 = g1Dynamics();
  ASSERT_NE(g1, nullptr);
  const WBAccelMpcRobotModel<scalar_t>& model = *g1->model;
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  std::mt19937 generator(52);
  const PreComputation preComputation;
  for (int sample = 0; sample < 8; ++sample) {
    const StateInput point =
        randomPoint(model, randomUnitQuaternion(generator), generator);
    const vector_t f = flowMap(*g1, point.state, point.input);
    for (const scalar_t lambda : {0.9, 1.1}) {
      SCOPED_TRACE(lambda);
      vector_t scaled = point.state;
      scaled.segment<4>(orientationStart) *= lambda;
      for (const vector_t& scaledFlow :
           {flowMap(*g1, scaled, point.input),
            g1->dynamics->computeFlowMap(/*t=*/0.0, scaled, point.input,
                                         preComputation)}) {
        EXPECT_LE(relativeDifference(withoutQuaternionRate(model, f),
                                     withoutQuaternionRate(model, scaledFlow)),
                  1.0e-12);
        EXPECT_LE((quaternionRate(model, scaledFlow) -
                   lambda * quaternionRate(model, f))
                      .cwiseAbs()
                      .maxCoeff(),
                  1.0e-14);
      }
    }
  }
}

TEST(WBQuaternionDynamics, ANegatedQuaternionFlipsTheQuaternionRateAlone) {
  WholeBodyDynamics* absl_nullable const g1 = g1Dynamics();
  ASSERT_NE(g1, nullptr);
  const WBAccelMpcRobotModel<scalar_t>& model = *g1->model;
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  std::mt19937 generator(53);
  const PreComputation preComputation;
  for (int sample = 0; sample < 8; ++sample) {
    SCOPED_TRACE(sample);
    const StateInput point =
        randomPoint(model, randomUnitQuaternion(generator), generator);
    vector_t negated = point.state;
    negated.segment<4>(orientationStart) *= -1.0;
    // -xi is the same rotation: the same accelerations, bit for bit, and the
    // opposite quaternion rate.
    const vector_t f = flowMap(*g1, point.state, point.input);
    const vector_t fNegated = flowMap(*g1, negated, point.input);
    EXPECT_EQ(withoutQuaternionRate(model, fNegated),
              withoutQuaternionRate(model, f));
    EXPECT_EQ(quaternionRate(model, fNegated),
              vector4_t(-quaternionRate(model, f)));
    const vector_t taped = g1->dynamics->computeFlowMap(
        /*t=*/0.0, point.state, point.input, preComputation);
    const vector_t tapedNegated = g1->dynamics->computeFlowMap(
        /*t=*/0.0, negated, point.input, preComputation);
    EXPECT_LE(relativeDifference(withoutQuaternionRate(model, taped),
                                 withoutQuaternionRate(model, tapedNegated)),
              1.0e-12);
    EXPECT_LE(
        (quaternionRate(model, tapedNegated) + quaternionRate(model, taped))
            .cwiseAbs()
            .maxCoeff(),
        1.0e-15);
  }
}

TEST(WBQuaternionDynamics, AtAPitchOf90DegreesTheDynamicsIsRegular) {
  WholeBodyDynamics* absl_nullable const g1 = g1Dynamics();
  ASSERT_NE(g1, nullptr);
  const WBAccelMpcRobotModel<scalar_t>& model = *g1->model;
  const pinocchio::Model& pinocchioModel = g1->pinocchioInterface->getModel();
  std::mt19937 generator(54);
  const PreComputation preComputation;
  int attitude = 0;
  for (const vector3_t& eulerZyx :
       {vector3_t(0.0, M_PI / 2.0, 0.0), vector3_t(1.2, M_PI / 2.0, -0.4),
        vector3_t(-2.5, -M_PI / 2.0, 0.3)}) {
    SCOPED_TRACE(eulerZyx.transpose());
    const vector4_t xi = quaternionFromEulerZyx(eulerZyx);
    const StateInput point = randomPoint(model, xi, generator);
    const vector_t f = flowMap(*g1, point.state, point.input);
    EXPECT_TRUE(f.allFinite()) << f.transpose();
    EXPECT_TRUE(g1->dynamics
                    ->computeFlowMap(/*t=*/0.0, point.state, point.input,
                                     preComputation)
                    .allFinite());
    const VectorFunctionLinearApproximation linear =
        g1->dynamics->linearApproximation(/*t=*/0.0, point.state, point.input,
                                          preComputation);
    EXPECT_TRUE(linear.dfdx.allFinite());
    EXPECT_TRUE(linear.dfdu.allFinite());

    // The mass matrix of the attitude has the eigenvalues of the same posture
    // upright: the quaternion root has no singular attitude (the SphericalZYX
    // root's determinant vanishes there).
    pinocchio::Data data(pinocchioModel);
    vector_t upright = point.state;
    upright.segment<4>(static_cast<Eigen::Index>(
        model.getBaseOrientationStartIndex())) = vector4_t(0.0, 0.0, 0.0, 1.0);
    std::vector<vector_t> eigenvalues;
    const std::array<const vector_t* absl_nonnull, 2> states = {&point.state,
                                                                &upright};
    for (const vector_t* absl_nonnull const state : states) {
      pinocchio::crba(pinocchioModel, data,
                      model.getGeneralizedCoordinates(*state));
      matrix_t massMatrix = data.M;
      massMatrix.triangularView<Eigen::StrictlyLower>() =
          massMatrix.transpose().triangularView<Eigen::StrictlyLower>();
      eigenvalues.push_back(Eigen::SelfAdjointEigenSolver<matrix_t>(
                                massMatrix, Eigen::EigenvaluesOnly)
                                .eigenvalues());
    }
    EXPECT_LE(relativeDifference(eigenvalues[1], eigenvalues[0]), 1.0e-9);
    const scalar_t condition =
        eigenvalues[0].maxCoeff() / eigenvalues[0].minCoeff();
    const scalar_t uprightCondition =
        eigenvalues[1].maxCoeff() / eigenvalues[1].minCoeff();
    EXPECT_NEAR(condition / uprightCondition, 1.0, 1.0e-9);
    RecordProperty(absl::StrCat("mass_matrix_condition_", attitude),
                   absl::StrCat(condition));
    ++attitude;
  }
}

/** The state after one classic RK4 step of `step` [s] of the flow map at the
 * constant input `input`. */
vector_t rk4Step(WholeBodyDynamics& g1,
                 const vector_t& state,
                 const vector_t& input,
                 scalar_t step) {
  const vector_t k1 = flowMap(g1, state, input);
  const vector_t k2 = flowMap(g1, state + 0.5 * step * k1, input);
  const vector_t k3 = flowMap(g1, state + 0.5 * step * k2, input);
  const vector_t k4 = flowMap(g1, state + step * k3, input);
  return state + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

TEST(WBQuaternionDynamics,
     TheQuaternionRateKeepsTheNormAndRk4DriftsByLessThan1e8OverTheHorizon) {
  WholeBodyDynamics* absl_nullable const g1 = g1Dynamics();
  ASSERT_NE(g1, nullptr);
  const WBAccelMpcRobotModel<scalar_t>& model = *g1->model;
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  std::mt19937 generator(55);

  // xi' f_xi = 1/2 xi' G(xi) w = 0: the continuous flow keeps |xi|, at any xi.
  for (int sample = 0; sample < 16; ++sample) {
    const vector4_t xi = (0.5 + static_cast<scalar_t>(sample) / 8.0) *
                         randomUnitQuaternion(generator);
    const StateInput point = randomPoint(model, xi, generator);
    const vector4_t rate =
        quaternionRate(model, flowMap(*g1, point.state, point.input));
    EXPECT_LE(std::abs(xi.dot(rate)), 1.0e-15 * xi.norm() * rate.norm())
        << "sample " << sample;
  }

  // A free flight, the joints still and the base spinning at |w_B| = 5 rad/s
  // about a tilted axis, its inertia turning the rate as it goes, by RK4 at a
  // step of the rollout's time_step over the MPC's horizon: the norm drifts by
  // less than 1e-8. (The rollout itself integrates as its task file says and
  // projects every state it outputs onto the manifold.)
  StateInput point =
      randomPoint(model, randomUnitQuaternion(generator), generator);
  model.setBaseAngularVelocityLocal(point.state,
                                    5.0 * vector3_t(0.6, -0.48, 0.64));
  const vector_t input = vector_t::Zero(point.input.size());
  vector_t still = input;
  model.setJointVelocities(
      point.state, still,
      vector_t::Zero(static_cast<Eigen::Index>(model.getJointDim())));
  ASSERT_NEAR(model.getBaseAngularVelocityLocal(point.state).norm(), 5.0,
              1.0e-12);
  const scalar_t rolloutStep = g1->task.rollout.time_step;
  ASSERT_GT(rolloutStep, 0.0);
  const int numSteps =
      static_cast<int>(std::ceil(g1->task.mpc.time_horizon / rolloutStep));
  vector_t state = point.state;
  scalar_t largestDrift = 0.0;
  for (int k = 0; k < numSteps; ++k) {
    state = rk4Step(*g1, state, input, rolloutStep);
    largestDrift =
        std::max(largestDrift,
                 std::abs(state.segment<4>(orientationStart).norm() - 1.0));
  }
  EXPECT_TRUE(state.allFinite());
  EXPECT_LT(largestDrift, 1.0e-8)
      << numSteps << " steps of " << rolloutStep << " s";
  RecordProperty("rk4_norm_drift_over_the_horizon", absl::StrCat(largestDrift));

  // One node of the SQP, a single RK4 step of its dt, drifts further, by the
  // method's truncation alone: halving the step shrinks it at least 16-fold
  // (RK4's local error is of order 5). The drift does not add up over the nodes
  // - each node's state is retracted onto the manifold - and the shooting gap
  // ignores it (StateManifold::difference() reads the flow's quaternion through
  // its direction only).
  const scalar_t sqpStep = g1->task.multiple_shooting.dt;
  ASSERT_GT(sqpStep, 0.0);
  const scalar_t nodeDrift = std::abs(rk4Step(*g1, point.state, input, sqpStep)
                                          .segment<4>(orientationStart)
                                          .norm() -
                                      1.0);
  const scalar_t halfStepDrift =
      std::abs(rk4Step(*g1, point.state, input, 0.5 * sqpStep)
                   .segment<4>(orientationStart)
                   .norm() -
               1.0);
  EXPECT_LT(nodeDrift, 1.0e-7);
  EXPECT_GT(nodeDrift, 16.0 * halfStepDrift)
      << "the drift of one node is not RK4's truncation";
  RecordProperty("rk4_norm_drift_of_one_sqp_node", absl::StrCat(nodeDrift));
}

}  // namespace
}  // namespace ocs2::humanoid
