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
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "Eigen/Geometry"
#include "Eigen/SVD"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/dynamics/SystemDynamicsBase.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/crba.hpp"

#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsAD.h"
#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsBasisInputsAD.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "support/ShippedCentroidalRobot.h"

/*
 * The centroidal flow map on the quaternion state x = [h, p_W, xi, q_j] as the solver tapes it (design README sections
 * 2.5, 2.6 and 5(c)): f_a = [dh; dp_W; G(xi) w_B / 2; dq_j], w_B from the momentum through A_b(xi_hat)^-1, the quaternion
 * row on the raw xi. Its CppAD Jacobians match central finite differences in the ambient coordinates; it is homogeneous
 * in xi, f(c xi) = [f_other, c f_xi] for c in {0.9, 1.1}; f(-xi) flips only dxi; its quaternion rate is the base's body
 * angular velocity, E^+ dxi = w_B; at a pitch of 90 degrees it is finite and the angular block of the momentum map, I_G R,
 * has the condition number of I_G (the Euler root's I_G T_W is singular there); and RK4 at the solver's step keeps
 * |xi| within 1e-8 of 1 over a horizon of a spin at |w| = 5 rad/s about a principal axis, as its stability function
 * says, and loses the norm of a tumble at its fourth order. On the G1 and the DRC Atlas with wrench inputs and the DRC Atlas
 * with its shipped basis-vector inputs; generates one dynamics library each.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;

/** The dynamics under test. */
enum class DynamicsCase { kG1Wrenches, kAtlasWrenches, kAtlasBasisVectors };

std::string dynamicsCaseName(const ::testing::TestParamInfo<DynamicsCase>& info) {
  switch (info.param) {
    case DynamicsCase::kG1Wrenches:
      return "unitree_g1_wrenches";
    case DynamicsCase::kAtlasWrenches:
      return "drc_atlas_wrenches";
    case DynamicsCase::kAtlasBasisVectors:
      return "drc_atlas_basis_vectors";
  }
  return "unknown";
}

/** A robot and its centroidal dynamics, with the input dimension of the dynamics and the per-foot basis size (0: wrenches). */
struct LoadedDynamics {
  LoadedCentroidalRobot robot;
  std::unique_ptr<SystemDynamicsBase> dynamics;
  Eigen::Index inputDim = 0;
  Eigen::Index basisPerFoot = 0;
};

/** The dynamics of `dynamicsCase`, built (and on a cold cache generated) as CentroidalMpcInterface builds them. */
absl::StatusOr<LoadedDynamics> loadDynamics(DynamicsCase dynamicsCase) {
  LoadedDynamics loaded;
  const ShippedCentroidalRobot robot =
      dynamicsCase == DynamicsCase::kG1Wrenches ? ShippedCentroidalRobot::kUnitreeG1 : ShippedCentroidalRobot::kDrcAtlas;
  absl::StatusOr<LoadedCentroidalRobot> loadedRobot = loadShippedCentroidalRobot(robot, /*centroidalModel=*/"");
  if (!loadedRobot.ok()) return loadedRobot.status();
  loaded.robot = *std::move(loadedRobot);
  const LoadedCentroidalRobot& r = loaded.robot;
  if (dynamicsCase != DynamicsCase::kAtlasBasisVectors) {
    loaded.dynamics =
        std::make_unique<CentroidalDynamicsAD>(*r.pinocchioInterface, *r.info, "quaternion_dynamics_test_wrenches", *r.modelSettings);
    loaded.inputDim = static_cast<Eigen::Index>(r.info->inputDim);
    return loaded;
  }
  absl::StatusOr<feet_array_t<ContactWrenchConeBasisMatrix>> bases =
      contactWrenchConeBasesFromConfig(r.config.task.contacts, *r.modelSettings);
  if (!bases.ok()) return bases.status();
  const std::array<matrix_t, kNumContacts> basisMatrices = {(*bases)[0].getBasisMatrix(), (*bases)[1].getBasisMatrix()};
  absl::StatusOr<std::unique_ptr<CentroidalDynamicsBasisInputsAD>> dynamics = CentroidalDynamicsBasisInputsAD::Create(
      *r.pinocchioInterface, *r.info, "quaternion_dynamics_test_basis_vectors", *r.modelSettings, basisMatrices);
  if (!dynamics.ok()) return dynamics.status();
  loaded.dynamics = *std::move(dynamics);
  loaded.basisPerFoot = static_cast<Eigen::Index>((*bases)[0].numBasis());
  loaded.inputDim = static_cast<Eigen::Index>(kNumContacts) * loaded.basisPerFoot + static_cast<Eigen::Index>(r.info->actuatedDofNum);
  return loaded;
}

/** A standing robot's state with momentum, a turned and tilted base of attitude `xi` (stored as given) and bent joints. */
vector_t stateAt(const LoadedCentroidalRobot& robot, const vector4_t& xi, std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
  const CentroidalMpcRobotModel<scalar_t>& model = *robot.model;
  vector_t state = model.getNeutralState();
  for (Eigen::Index i = 0; i < 6; ++i) state(i) = 0.3 * uniform(generator);
  model.setBasePosition(state, vector3_t(0.2 * uniform(generator), 0.2 * uniform(generator), 0.85));
  state.segment<4>(static_cast<Eigen::Index>(model.getBaseOrientationStartIndex())) = xi;
  vector_t joints = model.getJointAngles(robot.info->qPinocchioNominal);
  for (Eigen::Index i = 0; i < joints.size(); ++i) joints(i) += 0.2 * uniform(generator);
  model.setJointAngles(state, joints);
  return state;
}

/** An input: about the weight on both feet (wrenches) or positive scalings (basis vectors), and every joint moving. */
vector_t inputOf(const LoadedDynamics& loaded, std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
  vector_t input(loaded.inputDim);
  const Eigen::Index joints = static_cast<Eigen::Index>(loaded.robot.info->actuatedDofNum);
  const scalar_t halfWeight = 0.5 * 9.81 * loaded.robot.info->robotMass;
  if (loaded.basisPerFoot == 0) {
    for (Eigen::Index foot = 0; foot < 2; ++foot) {
      input.segment<6>(6 * foot) << 20.0 * uniform(generator), 20.0 * uniform(generator), halfWeight * (1.0 + 0.2 * uniform(generator)),
          5.0 * uniform(generator), 5.0 * uniform(generator), 2.0 * uniform(generator);
    }
  } else {
    for (Eigen::Index i = 0; i < 2 * loaded.basisPerFoot; ++i) {
      input(i) = halfWeight / static_cast<scalar_t>(loaded.basisPerFoot) * (1.0 + 0.5 * uniform(generator));
    }
  }
  for (Eigen::Index i = 0; i < joints; ++i) input(input.size() - joints + i) = 0.5 * uniform(generator);
  return input;
}

/** The attitudes of the samples: level, turned past pi, tilted both ways, and two of no special kind. */
std::vector<vector4_t> sampleAttitudes() {
  return {vector4_t(0.0, 0.0, 0.0, 1.0), quaternionFromEulerZyx(vector3_t(3.4, 0.0, 0.0)),
          quaternionFromEulerZyx(vector3_t(0.7, 0.35, -0.2)), quaternionFromEulerZyx(vector3_t(-2.2, -0.5, 0.4)),
          vector4_t(0.3, -0.5, 0.6, 0.55).normalized()};
}

/** max |actual - expected| <= tolerance max(1, max |expected|). */
::testing::AssertionResult isCloseRelative(const matrix_t& actual, const matrix_t& expected, scalar_t tolerance) {
  if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
    return ::testing::AssertionFailure() << actual.rows() << "x" << actual.cols() << " against " << expected.rows() << "x"
                                         << expected.cols();
  }
  const scalar_t scale = std::max(1.0, expected.cwiseAbs().maxCoeff());
  const scalar_t error = (actual - expected).cwiseAbs().maxCoeff();
  if (error <= tolerance * scale) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << "off by " << error << " against " << tolerance << " x " << scale;
}

class CentroidalQuaternionDynamics : public ::testing::TestWithParam<DynamicsCase> {
 protected:
  void SetUp() override {
    absl::StatusOr<LoadedDynamics> loaded = loadDynamics(GetParam());
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    loaded_ = std::make_unique<LoadedDynamics>(*std::move(loaded));
  }

  const CentroidalMpcRobotModel<scalar_t>& model() const { return *loaded_->robot.model; }
  Eigen::Index orientationStart() const { return static_cast<Eigen::Index>(model().getBaseOrientationStartIndex()); }
  vector_t flow(const vector_t& state, const vector_t& input) const {
    return loaded_->dynamics->computeFlowMap(/*time=*/0.0, state, input, PreComputation());
  }

  std::unique_ptr<LoadedDynamics> loaded_;
};

TEST_P(CentroidalQuaternionDynamics, TheCppAdJacobiansAreTheFiniteDifferencesOfTheFlowMap) {
  std::mt19937 generator(11);
  for (const vector4_t& xi : sampleAttitudes()) {
    SCOPED_TRACE(xi.transpose());
    const vector_t state = stateAt(loaded_->robot, xi, generator);
    const vector_t input = inputOf(*loaded_, generator);
    const VectorFunctionLinearApproximation linear = loaded_->dynamics->linearApproximation(/*time=*/0.0, state, input, PreComputation());
    EXPECT_TRUE(isCloseRelative(linear.f, flow(state, input), /*tolerance=*/1.0e-12));
    matrix_t dfdx(state.size(), state.size());
    for (Eigen::Index j = 0; j < state.size(); ++j) {
      vector_t plus = state;
      vector_t minus = state;
      plus(j) += kFiniteDifferenceStep;
      minus(j) -= kFiniteDifferenceStep;
      dfdx.col(j) = (flow(plus, input) - flow(minus, input)) / (2.0 * kFiniteDifferenceStep);
    }
    EXPECT_TRUE(isCloseRelative(linear.dfdx, dfdx, /*tolerance=*/1.0e-6)) << "dfdx";
    matrix_t dfdu(state.size(), input.size());
    for (Eigen::Index j = 0; j < input.size(); ++j) {
      vector_t plus = input;
      vector_t minus = input;
      plus(j) += kFiniteDifferenceStep;
      minus(j) -= kFiniteDifferenceStep;
      dfdu.col(j) = (flow(state, plus) - flow(state, minus)) / (2.0 * kFiniteDifferenceStep);
    }
    EXPECT_TRUE(isCloseRelative(linear.dfdu, dfdu, /*tolerance=*/1.0e-6)) << "dfdu";
  }
}

TEST_P(CentroidalQuaternionDynamics, TheFlowMapIsHomogeneousInTheQuaternionAndFlipsOnlyItsRateUnderItsNegation) {
  std::mt19937 generator(12);
  for (const vector4_t& xi : sampleAttitudes()) {
    SCOPED_TRACE(xi.transpose());
    const vector_t state = stateAt(loaded_->robot, xi, generator);
    const vector_t input = inputOf(*loaded_, generator);
    const vector_t reference = flow(state, input);
    const scalar_t scale = std::max(1.0, reference.cwiseAbs().maxCoeff());
    for (const scalar_t c : {0.9, 1.1, -1.0}) {
      SCOPED_TRACE(c);
      vector_t scaled = state;
      scaled.segment<4>(orientationStart()) *= c;
      const vector_t moved = flow(scaled, input);
      for (Eigen::Index i = 0; i < moved.size(); ++i) {
        const bool isQuaternionRate = i >= orientationStart() && i < orientationStart() + 4;
        // Every row but the quaternion's reads xi_hat; the quaternion rate G(xi) w_B / 2 is linear in the raw xi.
        EXPECT_NEAR(moved(i), isQuaternionRate ? c * reference(i) : reference(i), 1.0e-12 * scale) << "row " << i;
      }
    }
  }
}

TEST_P(CentroidalQuaternionDynamics, TheQuaternionRateIsTheBaseBodyAngularVelocity) {
  if (GetParam() == DynamicsCase::kAtlasBasisVectors) GTEST_SKIP() << "the wrench and basis inputs move the joints alike";
  std::mt19937 generator(13);
  for (const vector4_t& xi : sampleAttitudes()) {
    SCOPED_TRACE(xi.transpose());
    const vector_t state = stateAt(loaded_->robot, xi, generator);
    const vector_t input = inputOf(*loaded_, generator);
    const Eigen::Index joints = static_cast<Eigen::Index>(model().getJointDim());
    const vector_t rate = flow(state, input);
    // E^+(xi) dxi = 2 G(xi)' G(xi) w_B / 2 = w_B on the unit sphere; the base twist of the momentum in the world, R w_B.
    const vector3_t bodyRate = quaternionTangentMapPseudoInverse<scalar_t>(xi) * rate.segment<4>(orientationStart());
    const vector6_t twist = model().getBaseTwistInWorldFrame(state, input.tail(joints));
    const matrix3_t rotation = quaternion_t(xi).toRotationMatrix();
    EXPECT_LT((rotation * bodyRate - twist.tail<3>()).cwiseAbs().maxCoeff(), 1.0e-9 * std::max(1.0, twist.norm()));
    EXPECT_LT((vector3_t(rate.segment<3>(static_cast<Eigen::Index>(model().getBaseStartindex()))) - twist.head<3>()).cwiseAbs().maxCoeff(),
              1.0e-9 * std::max(1.0, twist.norm()));
    EXPECT_EQ(rate.tail(joints), input.tail(joints)) << "the joints move at their commanded velocities";
  }
}

TEST_P(CentroidalQuaternionDynamics, AtAPitchOf90DegreesTheFlowMapIsFiniteAndTheMomentumMapRegular) {
  std::mt19937 generator(14);
  const pinocchio::Model& pinocchioModel = loaded_->robot.pinocchioInterface->getModel();
  pinocchio::Data data(pinocchioModel);
  for (const vector4_t& xi :
       {quaternionFromEulerZyx(vector3_t(0.3, M_PI / 2.0, 0.2)), quaternionFromEulerZyx(vector3_t(-1.0, -M_PI / 2.0, 0.0)),
        quaternionFromEulerZyx(vector3_t(0.7, 0.35, -0.2))}) {
    SCOPED_TRACE(xi.transpose());
    const vector_t state = stateAt(loaded_->robot, xi, generator);
    const vector_t input = inputOf(*loaded_, generator);
    const VectorFunctionLinearApproximation linear = loaded_->dynamics->linearApproximation(/*time=*/0.0, state, input, PreComputation());
    EXPECT_TRUE(linear.f.allFinite());
    EXPECT_TRUE(linear.dfdx.allFinite());
    EXPECT_TRUE(linear.dfdu.allFinite());
    // The angular block of A_b in the body angular velocity: I_G R, of I_G's condition number at every attitude.
    const vector_t q = model().getGeneralizedCoordinates(state);
    pinocchio::computeCentroidalMap(pinocchioModel, data, q);
    const matrix3_t angularBlock = data.Ag.block<3, 3>(3, 3);
    pinocchio::ccrba(pinocchioModel, data, q, vector_t::Zero(pinocchioModel.nv));
    const matrix3_t compositeInertia = data.Ig.inertia().matrix();
    EXPECT_LT((angularBlock - compositeInertia * quaternion_t(xi).toRotationMatrix()).cwiseAbs().maxCoeff(),
              1.0e-12 * compositeInertia.cwiseAbs().maxCoeff());
    const Eigen::JacobiSVD<matrix3_t> block(angularBlock);
    const Eigen::JacobiSVD<matrix3_t> inertia(compositeInertia);
    const scalar_t blockCondition = block.singularValues()(0) / block.singularValues()(2);
    const scalar_t inertiaCondition = inertia.singularValues()(0) / inertia.singularValues()(2);
    EXPECT_NEAR(blockCondition, inertiaCondition, 1.0e-9 * inertiaCondition);
  }
}

/** The composite inertia I_G of `robot` at `state`. */
matrix3_t compositeInertiaAt(const LoadedCentroidalRobot& robot, const vector_t& state) {
  const pinocchio::Model& pinocchioModel = robot.pinocchioInterface->getModel();
  pinocchio::Data data(pinocchioModel);
  pinocchio::ccrba(pinocchioModel, data, robot.model->getGeneralizedCoordinates(state), vector_t::Zero(pinocchioModel.nv));
  return data.Ig.inertia().matrix();
}

/**
 * The free spin of the tests below: `state` with its CoM still and its base turning at `angularVelocityWorld` with the
 * joints still, L = I_G w; no contact force acts on it.
 */
vector_t spinningState(const LoadedCentroidalRobot& robot, vector_t state, const vector3_t& angularVelocityWorld) {
  state.head<3>().setZero();
  state.segment<3>(3) = compositeInertiaAt(robot, state) * angularVelocityWorld / robot.info->robotMass;
  return state;
}

/** What an RK4 integration of the flow map over `steps` steps of `step` did to the quaternion. Passive data. */
struct RungeKuttaRun {
  /** max | |xi| - 1 | over the steps. */
  scalar_t largestNormError = 0.0;
  /** [rad] The rotation from the first attitude to the last. */
  scalar_t turned = 0.0;
};

class CentroidalQuaternionDynamicsIntegration : public CentroidalQuaternionDynamics {
 protected:
  /** RK4 of the flow map from `state` with no contact force and the joints still, as the solver discretizes a node. */
  RungeKuttaRun integrate(vector_t state, scalar_t step, int steps) const {
    const vector_t input = vector_t::Zero(loaded_->inputDim);
    const vector4_t start = model().getBaseOrientation(state);
    RungeKuttaRun run;
    for (int k = 0; k < steps; ++k) {
      const vector_t k1 = flow(state, input);
      const vector_t k2 = flow(state + 0.5 * step * k1, input);
      const vector_t k3 = flow(state + 0.5 * step * k2, input);
      const vector_t k4 = flow(state + step * k3, input);
      state += step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
      run.largestNormError = std::max(run.largestNormError, std::abs(state.segment<4>(orientationStart()).norm() - 1.0));
    }
    const vector4_t end = model().getBaseOrientation(state);
    run.turned = quaternionLog<scalar_t>(quaternionProduct<scalar_t>(quaternionConjugate<scalar_t>(start), end)).norm();
    return run;
  }
};

TEST_P(CentroidalQuaternionDynamicsIntegration, RungeKuttaLosesTheNormOfALinearSpinAsItsStabilityFunctionSays) {
  // A spin at |w| = 5 rad/s about the principal axis of the largest composite inertia: L = I_G w stays, and so do w_W and
  // w_B (the axis turns into itself), so the quaternion's equation is linear, dxi/dt = Omega xi with Omega skew of
  // eigenvalues +-i |w| / 2. Its exact flow keeps |xi| = 1; RK4's stability function R(z) = 1 + z + z^2 / 2 + z^3 / 6 +
  // z^4 / 24 at z = i y, y = h |w| / 2, has |R|^2 = 1 - y^6 / 72 + y^8 / 576, so |xi| shrinks by the factor |R(i y)| per
  // step: by 1.1e-10 per step at h = 0.02 s, 6.5e-9 over the horizon of 60 steps, below the 1e-8 the design states.
  std::mt19937 generator(15);
  const vector_t posture = stateAt(loaded_->robot, quaternionFromEulerZyx(vector3_t(0.4, 0.2, -0.1)), generator);
  const Eigen::SelfAdjointEigenSolver<matrix3_t> principal(compositeInertiaAt(loaded_->robot, posture));
  const vector_t state = spinningState(loaded_->robot, posture, 5.0 * vector3_t(principal.eigenvectors().col(2)));
  constexpr scalar_t kStep = 0.02;
  constexpr int kSteps = 60;
  const RungeKuttaRun run = integrate(state, kStep, kSteps);
  const scalar_t y = 0.5 * kStep * 5.0;
  const scalar_t perStep = 1.0 - std::sqrt(1.0 - std::pow(y, 6) / 72.0 + std::pow(y, 8) / 576.0);
  const scalar_t expected = 1.0 - std::pow(1.0 - perStep, kSteps);
  EXPECT_LT(run.largestNormError, 1.0e-8);
  EXPECT_NEAR(run.largestNormError, expected, 0.02 * expected) << "the stability function's loss, " << expected;
  EXPECT_NEAR(run.turned, 2.0 * M_PI - 5.0 * kStep * kSteps, 1.0e-6) << "turned at 5 rad/s for 1.2 s, 6 rad: read on the shorter arc";
  ::testing::Test::RecordProperty("largest_quaternion_norm_error", absl::StrCat(run.largestNormError));
}

TEST_P(CentroidalQuaternionDynamicsIntegration, RungeKuttaLosesTheNormOfATumbleAtItsFourthOrder) {
  // A spin at |w| = 5 rad/s about an axis of no symmetry: the body tumbles, w changes (I_G turns with it while L stays),
  // and the linear analysis above no longer holds: RK4's local error has a radial part of order h^5, so the loss over a
  // horizon falls by about 2^4 when the step halves. At the solver's 0.02 s it is a few 1e-7, which the solver's
  // projection onto the manifold removes at every iterate (the shooting gaps are differences on the manifold, which do
  // not read |xi|).
  std::mt19937 generator(15);
  const vector_t posture = stateAt(loaded_->robot, quaternionFromEulerZyx(vector3_t(0.4, 0.2, -0.1)), generator);
  const vector_t state = spinningState(loaded_->robot, posture, 5.0 * vector3_t(0.3, -0.5, 0.8).normalized());
  const RungeKuttaRun coarse = integrate(state, /*step=*/0.02, /*steps=*/60);
  const RungeKuttaRun fine = integrate(state, /*step=*/0.01, /*steps=*/120);
  EXPECT_LT(coarse.largestNormError, 1.0e-6);
  EXPECT_GT(coarse.largestNormError, 12.0 * fine.largestNormError) << "fourth order: halving the step divides the loss by about 16";
  EXPECT_GT(coarse.turned, 1.0) << "the base did turn";
  EXPECT_NEAR(coarse.turned, fine.turned, 1.0e-5);
  ::testing::Test::RecordProperty("largest_quaternion_norm_error_at_0_02_s", absl::StrCat(coarse.largestNormError));
  ::testing::Test::RecordProperty("largest_quaternion_norm_error_at_0_01_s", absl::StrCat(fine.largestNormError));
}

INSTANTIATE_TEST_SUITE_P(ShippedDynamics,
                         CentroidalQuaternionDynamics,
                         ::testing::Values(DynamicsCase::kG1Wrenches, DynamicsCase::kAtlasWrenches, DynamicsCase::kAtlasBasisVectors),
                         dynamicsCaseName);

INSTANTIATE_TEST_SUITE_P(ShippedDynamics,
                         CentroidalQuaternionDynamicsIntegration,
                         ::testing::Values(DynamicsCase::kG1Wrenches, DynamicsCase::kAtlasWrenches, DynamicsCase::kAtlasBasisVectors),
                         dynamicsCaseName);

}  // namespace
}  // namespace ocs2::humanoid
