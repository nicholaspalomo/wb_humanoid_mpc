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

#include "pinocchio/fwd.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "ocs2_core/dynamics/SystemDynamicsBase.h"
#include "ocs2_core/integration/SensitivityIntegrator.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/multibody/data.hpp"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/EulerMpcOracle.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/MpcEquivalenceFixture.h"

/*
 * RK4 rollouts of the MPC on the quaternion layout (the SQP's shooting map, projected onto the manifold after every
 * interval) against the Euler oracle's RK4 rollouts from the same physical state with the same inputs, over the MPC's
 * horizon at its node spacing, for both formulations: through ordinary attitudes, across a yaw of +-pi (where the Euler
 * yaw goes on past pi and the quaternion goes round), and through a pitch of 90 degrees (where the Euler coordinates are
 * singular, so the Euler side runs in a world turned to keep the robot near level, under the turned gravity).
 *
 * The two are different discretizations of one ODE: RK4 depends on the coordinates, so the two rollouts differ by their
 * integration errors, not by round-off. The bound is derived from the integration error itself: RK4's global error at
 * step dt is C dt^4 + O(dt^5), so a rollout at dt/2 estimates each rollout's error at node k as
 * (16/15) |x(dt)_k (-) x(dt/2)_2k| (Richardson), and the two rollouts may differ by the sum of the two estimates,
 * kStepHalvingSafety times (the estimate is the leading term only), node by node. And they must converge to the same
 * trajectory: halving dt shrinks their difference by RK4's order, 2^4 = 16 (asserted at least kMinimumConvergenceRatio).
 */

namespace ocs2::humanoid {
namespace {

/** The safety factor on the Richardson estimates of the two rollouts' integration errors (leading term only). */
constexpr scalar_t kStepHalvingSafety = 2.0;

/**
 * The least factor by which halving the step must shrink the two rollouts' difference: RK4 is of order 4 (16); 8 leaves
 * room for the O(dt^5) term at the MPCs' node spacing.
 */
constexpr scalar_t kMinimumConvergenceRatio = 8.0;

/** Round-off floor of a node's difference, relative to the state's largest entry: kFlowMapTolerance of the dynamics test. */
constexpr scalar_t kRoundOffFloor = 1.0e-9;

/** [rad] The largest Euler pitch and roll of the Euler side of a rollout: 1.2 rad keeps T_B's condition number below 3. */
constexpr scalar_t kMaxEulerTilt = 1.2;

class MpcRolloutEquivalence : public MpcEquivalenceTest {};

/** `steps` intervals of `dt` of the MPC's shooting map with the input held, each end projected: the SQP's trajectory model. */
std::vector<vector_t> productionRollout(
    MpcEquivalenceFixture& mpc, const vector_t& initialState, const vector_t& input, scalar_t dt, int steps) {
  SystemDynamicsBase& dynamics = *mpc.problem().dynamicsPtr;
  const DynamicsDiscretizer discretizer = selectDynamicsDiscretization(mpc.sqpSettings().integratorType);
  std::vector<vector_t> states = {initialState};
  for (int step = 0; step < steps; ++step) {
    vector_t next = discretizer(dynamics, step * dt, states.back(), input, dt);
    mpc.manifold().project(next);
    states.push_back(std::move(next));
  }
  return states;
}

/** The same of the Euler oracle, in Euler coordinates. */
std::vector<vector_t> eulerRollout(
    const EulerMpcOracle& oracle, const vector_t& initialEulerState, const vector_t& input, scalar_t dt, int steps) {
  std::vector<vector_t> states = {initialEulerState};
  for (int step = 0; step < steps; ++step) states.push_back(oracle.rk4Step(states.back(), input, dt));
  return states;
}

/** One rollout comparison: the MPC's world and the oracle's (turned by `worldRotation`), the inputs of each. Passive data. */
struct RolloutCase {
  vector_t state;
  vector_t input;
  /** The same physical state in the oracle's world, in Euler coordinates. */
  vector_t eulerState;
  vector_t eulerInput;
  /** g: the oracle's world is the MPC's turned by g (MpcEquivalenceFixture::ambientWorldRotation()). */
  vector4_t worldRotation = vector4_t(0.0, 0.0, 0.0, 1.0);
};

/** max |d|. */
scalar_t maxAbs(const vector_t& d) {
  return d.size() == 0 ? 0.0 : d.cwiseAbs().maxCoeff();
}

/** The two rollouts of a case at the node spacing dt and at dt / 2 and dt / 4. Passive data. */
struct Rollouts {
  /** production[r]: the MPC's rollout at dt / 2^r, in its world. */
  std::vector<std::vector<vector_t>> production;
  /** euler[r]: the oracle's at dt / 2^r, in its world's Euler coordinates. */
  std::vector<std::vector<vector_t>> euler;
  /** physical[r]: euler[r] through the chart, turned back into the MPC's world. */
  std::vector<std::vector<vector_t>> physical;
  /** Intervals at the node spacing. */
  int steps = 0;
};

/** The rollouts of `rolloutCase` over the MPC's horizon, the oracle under `oracle`'s gravity. */
Rollouts rolloutsOf(MpcEquivalenceFixture& mpc, const EulerMpcOracle& oracle, const RolloutCase& rolloutCase) {
  const scalar_t dt = mpc.nodeTimeStep();
  Rollouts rollouts;
  rollouts.steps = static_cast<int>(std::lround(mpc.timeHorizon() / dt));
  const matrix_t back = mpc.ambientWorldRotation(quaternionConjugate(rolloutCase.worldRotation));
  for (int refinement = 0; refinement < 3; ++refinement) {
    const int factor = 1 << refinement;
    rollouts.production.push_back(productionRollout(mpc, rolloutCase.state, rolloutCase.input, dt / factor, rollouts.steps * factor));
    rollouts.euler.push_back(eulerRollout(oracle, rolloutCase.eulerState, rolloutCase.eulerInput, dt / factor, rollouts.steps * factor));
    std::vector<vector_t> physical;
    for (const vector_t& eulerState : rollouts.euler.back()) physical.emplace_back(back * mpc.quaternionStateOf(eulerState));
    rollouts.physical.push_back(std::move(physical));
  }
  return rollouts;
}

/**
 * Expects the two rollouts to be the same physical trajectory within their integration errors (the file comment), and
 * their difference to shrink by kMinimumConvergenceRatio when the step is halved.
 */
void expectSamePhysicalTrajectory(const MpcEquivalenceFixture& mpc, const Rollouts& rollouts) {
  for (const vector_t& eulerState : rollouts.euler[0]) {
    const vector3_t angles = eulerState.segment<3>(static_cast<Eigen::Index>(mpc.model().getBaseOrientationStartIndex()));
    ASSERT_LE(std::max(std::abs(angles(1)), std::abs(angles(2))), kMaxEulerTilt) << "the Euler side left its well-conditioned region";
  }
  const StateManifold& manifold = mpc.manifold();
  std::array<scalar_t, 2> largestDifference = {0.0, 0.0};
  for (size_t refinement = 0; refinement < 2; ++refinement) {
    const size_t factor = size_t{1} << refinement;
    for (size_t node = 0; node <= static_cast<size_t>(rollouts.steps); ++node) {
      const size_t coarse = node * factor;
      const size_t fine = 2 * coarse;
      const vector_t& state = rollouts.production[refinement][coarse];
      const vector_t& physical = rollouts.physical[refinement][coarse];
      const scalar_t difference = maxAbs(manifold.difference(state, physical));
      const scalar_t productionError = (16.0 / 15.0) * maxAbs(manifold.difference(rollouts.production[refinement + 1][fine], state));
      const scalar_t eulerError = (16.0 / 15.0) * maxAbs(manifold.difference(rollouts.physical[refinement + 1][fine], physical));
      const scalar_t bound = kStepHalvingSafety * (productionError + eulerError) + kRoundOffFloor * std::max(1.0, maxAbs(state));
      EXPECT_LE(difference, bound) << "node " << node << " at dt / " << factor;
      largestDifference[refinement] = std::max(largestDifference[refinement], difference);
    }
  }
  ::testing::Test::RecordProperty("largest_node_difference", absl::StrFormat("%.3e", largestDifference[0]));
  ::testing::Test::RecordProperty("largest_node_difference_at_half_step", absl::StrFormat("%.3e", largestDifference[1]));
  // Convergence to one trajectory, unless the rollouts already agree to round-off.
  if (largestDifference[0] > 1.0e3 * kRoundOffFloor) {
    EXPECT_GE(largestDifference[0] / largestDifference[1], kMinimumConvergenceRatio)
        << "halving the step did not shrink the difference by RK4's order: " << largestDifference[0] << " and " << largestDifference[1];
  }
}

/**
 * A small input: contact forces within +-1 % of half the weight per axis and moments within +-1 N m (local basis:
 * scalings within [0, 0.02] of the scaling that carries half the weight on a level foot), joint entries within
 * +-`maxAbsJointEntry`. The robot is then all but in free fall, its attitude driven by its own momentum and joint motion
 * rather than by a net contact moment about its center of mass, so that the Euler side stays in its well-conditioned
 * region over the horizon; the flow-map tests cover arbitrary wrenches.
 */
vector_t smallInput(const MpcEquivalenceFixture& mpc, scalar_t maxAbsJointEntry, std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> unit(-1.0, 1.0);
  const EulerMpcOracle& oracle = mpc.oracle();
  const scalar_t halfWeight = 0.5 * oracle.totalMass() * 9.81;
  vector_t input = vector_t::Zero(oracle.inputDim());
  Eigen::Index row = 0;
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    if (oracle.options().contactInput == OracleContactInput::kWorldWrench) {
      for (Eigen::Index i = 0; i < 3; ++i) input(row + i) = 0.01 * halfWeight * unit(generator);
      for (Eigen::Index i = 3; i < 6; ++i) input(row + i) = unit(generator);
      row += 6;
      continue;
    }
    const matrix_t& basis = oracle.options().localBasisMatrices[contact];
    const scalar_t nominal = halfWeight / basis.row(2).sum();
    for (Eigen::Index i = 0; i < basis.cols(); ++i) input(row + i) = 0.01 * (1.0 + unit(generator)) * nominal;
    row += basis.cols();
  }
  for (Eigen::Index joint = 0; joint < oracle.jointDim(); ++joint) input(row + joint) = maxAbsJointEntry * unit(generator);
  return input;
}

/** Where the rollouts' initial states are drawn: moderate momenta and rates. */
EulerSamplingBounds rolloutBounds(scalar_t maxAbsTilt) {
  EulerSamplingBounds bounds;
  bounds.maxAbsPitch = maxAbsTilt;
  bounds.maxAbsRoll = maxAbsTilt;
  bounds.maxAbsLinearVelocity = 0.3;
  bounds.maxAbsNormalizedAngularMomentum = 0.005;
  bounds.maxAbsEulerRate = 0.3;
  bounds.maxAbsJointVelocity = 0.3;
  return bounds;
}

/**
 * Sets the base velocity of the Euler state `eulerState` of `oracle` to `baseVelocity` = [pd_W; dTheta/dt] with the joint
 * rates `jointRates`: the whole body's state holds it; the centroidal state holds the momentum hbar = A_G,e [v_b; qd_j] / m
 * whose base velocity it is (O2).
 */
void setBaseVelocity(
    const EulerMpcOracle& oracle, bool wholeBody, const vector6_t& baseVelocity, const vector_t& jointRates, vector_t& eulerState) {
  const Eigen::Index nj = oracle.jointDim();
  if (wholeBody) {
    eulerState.segment<6>(6 + nj) = baseVelocity;
    return;
  }
  pinocchio::Data data(oracle.eulerModel());
  const matrix_t centroidalMap = pinocchio::computeCentroidalMap(oracle.eulerModel(), data, vector_t(eulerState.segment(6, 6 + nj)));
  vector_t velocity(6 + nj);
  velocity << baseVelocity, jointRates;
  eulerState.head<6>() = centroidalMap * velocity / oracle.totalMass();
}

/** The joint entries of the input that are the joint rates of the centroidal state's momentum: the input's last n_j. */
vector_t jointRatesOf(const MpcEquivalenceFixture& mpc, const vector_t& input) {
  return mpc.isWholeBody() ? vector_t::Zero(mpc.oracle().jointDim()) : vector_t(input.tail(mpc.oracle().jointDim()));
}

/** The joint entry of a small input: joint velocities [rad/s] (centroidal), joint accelerations [rad/s^2] (whole body). */
scalar_t maxAbsJointInput(const MpcEquivalenceFixture& mpc) {
  return mpc.isWholeBody() ? 1.0 : 0.3;
}

TEST_P(MpcRolloutEquivalence, RolloutsFromOneStateAreOnePhysicalTrajectory) {
  MpcEquivalenceFixture& mpc = this->mpc();
  std::mt19937 generator(/*sd=*/20261020);
  for (int sample = 0; sample < 3; ++sample) {
    SCOPED_TRACE(sample);
    RolloutCase rolloutCase;
    rolloutCase.eulerState = mpc.randomEulerState(rolloutBounds(/*maxAbsTilt=*/0.3), generator);
    rolloutCase.state = mpc.quaternionStateOf(rolloutCase.eulerState);
    rolloutCase.input = smallInput(mpc, maxAbsJointInput(mpc), generator);
    rolloutCase.eulerInput = rolloutCase.input;
    expectSamePhysicalTrajectory(mpc, rolloutsOf(mpc, mpc.oracle(), rolloutCase));
  }
}

TEST_P(MpcRolloutEquivalence, AcrossAYawOfPiTheRolloutTurnsOnThroughTheCut) {
  MpcEquivalenceFixture& mpc = this->mpc();
  std::mt19937 generator(/*sd=*/20261021);
  const Eigen::Index orientation = static_cast<Eigen::Index>(mpc.model().getBaseOrientationStartIndex());
  RolloutCase rolloutCase;
  rolloutCase.eulerState = mpc.randomEulerState(rolloutBounds(/*maxAbsTilt=*/0.1), generator);
  rolloutCase.eulerState(orientation) = M_PI - 0.2;
  // The joints still: the base turns at its own rate.
  rolloutCase.input = smallInput(mpc, /*maxAbsJointEntry=*/0.0, generator);
  // Turning at 0.5 rad/s, the base otherwise still.
  const vector6_t turning = (vector6_t() << 0.0, 0.0, 0.0, 0.5, 0.0, 0.0).finished();
  setBaseVelocity(mpc.oracle(), mpc.isWholeBody(), turning, jointRatesOf(mpc, rolloutCase.input), rolloutCase.eulerState);
  rolloutCase.state = mpc.quaternionStateOf(rolloutCase.eulerState);
  rolloutCase.eulerInput = rolloutCase.input;
  const Rollouts rollouts = rolloutsOf(mpc, mpc.oracle(), rolloutCase);
  // The Euler yaw goes on past pi; the MPC's heading reading wraps from +pi to -pi between two nodes while the quaternion
  // moves on by one interval's turn.
  ASSERT_GT(rollouts.euler[0].back()(orientation), M_PI + 0.05) << "the case does not cross the cut";
  bool wrapped = false;
  for (size_t node = 0; node + 1 < rollouts.production[0].size(); ++node) {
    const vector_t& state = rollouts.production[0][node];
    const vector_t& next = rollouts.production[0][node + 1];
    EXPECT_LE(mpc.manifold().getMaximumRotationAngle(mpc.manifold().difference(state, next)), 0.05) << "node " << node;
    wrapped = wrapped || (mpc.model().getBaseHeading(state) > 2.5 && mpc.model().getBaseHeading(next) < -2.5);
  }
  EXPECT_TRUE(wrapped) << "the heading reading never wrapped";
  expectSamePhysicalTrajectory(mpc, rollouts);
}

TEST_P(MpcRolloutEquivalence, ThroughAPitchOfNinetyDegreesTheRolloutIsTheTurnedOracles) {
  MpcEquivalenceFixture& mpc = this->mpc();
  std::mt19937 generator(/*sd=*/20261022);
  const Eigen::Index orientation = static_cast<Eigen::Index>(mpc.model().getBaseOrientationStartIndex());
  // In the turned world the robot is level but for a pitch of -0.15 rad, pitching up at 0.6 rad/s about its own y axis;
  // in the MPC's world, turned by g^-1 = q_z(yaw) q_y(pi / 2), it starts at a pitch of pi / 2 - 0.15 and pitches through
  // 90 degrees.
  RolloutCase rolloutCase;
  rolloutCase.eulerState = mpc.randomEulerState(rolloutBounds(/*maxAbsTilt=*/0.0), generator);
  rolloutCase.eulerState.segment<3>(orientation) << 0.0, -0.15, 0.0;
  const scalar_t yaw = std::uniform_real_distribution<scalar_t>(-M_PI, M_PI)(generator);
  rolloutCase.worldRotation = quaternionConjugate(quaternionFromEulerZyx(vector3_t(yaw, 0.5 * M_PI, 0.0)));
  const matrix_t back = mpc.ambientWorldRotation(quaternionConjugate(rolloutCase.worldRotation));
  // The joints still: the base pitches at its own rate.
  rolloutCase.input = smallInput(mpc, /*maxAbsJointEntry=*/0.0, generator);
  rolloutCase.eulerInput = mpc.inputWorldRotation(rolloutCase.worldRotation) * rolloutCase.input;
  const vector6_t pitching = (vector6_t() << 0.0, 0.0, 0.0, 0.0, 0.6, 0.0).finished();
  setBaseVelocity(mpc.oracle(), mpc.isWholeBody(), pitching, jointRatesOf(mpc, rolloutCase.eulerInput), rolloutCase.eulerState);
  rolloutCase.state = back * mpc.quaternionStateOf(rolloutCase.eulerState);
  const vector3_t turnedGravity = quaternionRotationMatrix(rolloutCase.worldRotation) * vector3_t(0.0, 0.0, -9.81);
  const absl::StatusOr<std::unique_ptr<EulerMpcOracle>> turnedOracle = mpc.oracleUnderGravity(turnedGravity);
  ASSERT_TRUE(turnedOracle.ok()) << turnedOracle.status();
  const Rollouts rollouts = rolloutsOf(mpc, **turnedOracle, rolloutCase);

  // The MPC's rollout passes through gimbal lock: its body x axis points straight down and turns back.
  scalar_t lowest = 1.0;
  size_t lowestNode = 0;
  for (size_t node = 0; node < rollouts.production[0].size(); ++node) {
    const vector_t& state = rollouts.production[0][node];
    const vector4_t xi = mpc.model().getBaseOrientation(state);
    EXPECT_NEAR(xi.norm(), 1.0, 1.0e-14) << "node " << node;
    const scalar_t bodyXDown = quaternionRotationMatrix(xi)(2, 0);
    if (bodyXDown < lowest) {
      lowest = bodyXDown;
      lowestNode = node;
    }
  }
  ASSERT_LE(lowest, -0.995) << "the rollout does not reach a pitch of 90 degrees";
  ASSERT_LT(lowestNode + 1, rollouts.production[0].size()) << "the rollout does not pass through it";
  expectSamePhysicalTrajectory(mpc, rollouts);
}

INSTANTIATE_TEST_SUITE_P(Shipped, MpcRolloutEquivalence, ::testing::ValuesIn(mpcEquivalenceConfigurationNames()), mpcEquivalenceTestName);

}  // namespace
}  // namespace ocs2::humanoid
