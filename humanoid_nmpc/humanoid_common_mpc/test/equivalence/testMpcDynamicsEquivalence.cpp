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
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/dynamics/SystemDynamicsBase.h"
#include "ocs2_core/integration/SensitivityIntegrator.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_mpc_validation/chart/EulerChart.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/EulerMpcOracle.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/MpcEquivalenceFixture.h"

/*
 * The MPC's dynamics on the quaternion layout against the Euler-coordinate
 * oracle (EulerMpcOracle.h) through the chart (MpcEquivalenceFixture.h
 * (E1)-(E4)), for both formulations:
 *
 *   - the flow map f_a(Phi(x_e), u) = dPhi f_e (E1), its input Jacobian (dPhi
 * df_e/du) and its state Jacobian, the chain rule d(dPhi f_e)/dx_e = (df_a/dx)
 * dPhi (E2), at random states and inputs;
 *   - the SQP's shooting interval: its tangent sensitivities A_t, B_t (the
 * projected RK4 step pushed forward on the manifold,
 * StateManifold::pushForwardDynamics) against the Euler RK4 map's through P_e
 * (E4);
 *   - what only the quaternion layout can do: at a pitch of exactly +-90
 * degrees, where the Euler coordinates are singular, the flow, its
 * linearization and the shooting interval are the oracle's at a level attitude
 * under a gravity turned with the world (the dynamics are equivariant under
 * world rotations, MpcEquivalenceFixture:: ambientWorldRotation()); across a
 * yaw of +-pi, where the Euler yaw jumps by 2 pi, the tangent flow and the
 * shooting interval are the same on both sides of the cut.
 *
 * Tolerances, derived where they are declared: round-off for values and exact
 * relations, the Richardson error estimate for finite differences
 * (richardsonJacobian()), and the step-halving estimate of each RK4 map's local
 * error for the shooting interval, whose coordinate dependence is O(dt^5).
 */

namespace ocs2::humanoid {
namespace {

/**
 * The flow maps' agreement, relative to the largest entry of each block of rows
 * (floor 1; flowRowBlocks()): both sides evaluate the same rigid-body
 * quantities in double precision through different root parameterizations (the
 * root joint suite holds them equal to round-off), then solve a 6 x 6 system,
 * the centroidal momentum's base block (condition number of order m /
 * lambda_min(I_G), at most 1e3 on the shipped robots) or the whole body's
 * floating-base mass matrix (similar); kFunctionEvaluationRelativeError times
 * that condition number is 1e-9. The production side's generated code (CppADCG,
 * reassociated by -ffast-math) adds the same order. The class of every row but
 * the momentum rates' (design section 4.4: the configuration rates, and the
 * whole body's accelerations); also the bound of the states and sensitivities
 * of a shooting interval, relative to their largest entry.
 */
constexpr scalar_t kFlowMapTolerance = 1.0e-9;

/**
 * The centroidal momentum rates' agreement (O1), relative to the block's
 * largest entry (floor 1): Newton's and Euler's sums of the contact wrenches
 * and gravity, about a center of mass and at contact positions that the two
 * roots place within (T1) = 1e-12 relative
 * (testRootJointKinematicsEquivalence.cpp), with no linear solve, so a few
 * units of round-off of the largest moment arm times force; 1e-10 is 100 times
 * (T1) (design section 4.4's momentum class).
 */
constexpr scalar_t kMomentumRateTolerance = 1.0e-10;

/**
 * A Jacobian's agreement beyond the finite-difference error bound, relative to
 * the largest entry (floor 1): the AD Jacobian's own round-off, which the
 * design's E1 class bounds by 1e-8 (section 4.4).
 */
constexpr scalar_t kJacobianTolerance = 1.0e-8;

/** [rad] of the step of the finite differences of the oracle (per unit of the
 * coordinate's magnitude, floor 1). */
constexpr scalar_t kFiniteDifferenceStep = 1.0e-4;

/**
 * The substeps of the reference interval an RK4 map's local error is estimated
 * against: kSubsteps steps of dt / kSubsteps have kSubsteps^-4 of its local
 * error C dt^5 (each C (dt / kSubsteps)^5), so |F(dt) - F_ref(dt)| times
 * kSubstepsErrorFactor = 1 / (1 - kSubsteps^-4) is F(dt)'s error, without
 * relying on the asymptotic constant: 4 steps leave 1/256 of it.
 */
constexpr int kSubsteps = 4;
constexpr scalar_t kSubstepsErrorFactor = 256.0 / 255.0;

/** The safety factor on that estimate, which is of the leading term only: 2. */
constexpr scalar_t kStepHalvingSafety = 2.0;

constexpr int kRandomStates = 12;

class MpcDynamicsEquivalence : public MpcEquivalenceTest {};

/** Which state a flow is the rate of. */
enum class FlowCoordinates {
  /** The Euler state of the oracle (EulerMpcOracle.h): three rows per rotation
     block. */
  kEuler,
  /** The quaternion state of the MPC, ambient: four rows for the base
     orientation's quaternion. */
  kQuaternion,
};

/** One block of rows of a flow, with the relative tolerance of its class.
 * Passive data. */
struct FlowRowBlock {
  Eigen::Index start = 0;
  Eigen::Index size = 0;
  scalar_t relativeTolerance = 0.0;
};

/**
 * The blocks of rows of a flow, one per block of the task file's Euler layout
 * (the momenta, the base position and orientation, the joints, the velocities),
 * so that each block is held to its own scale: on the whole body the joint
 * accelerations reach 64 against 0.6 for the quaternion's rates, which one
 * scale over the whole vector would hold to 1e-7 relative. The momentum blocks
 * (normalized_linear_momentum, normalized_angular_momentum) take
 * kMomentumRateTolerance, the others kFlowMapTolerance.
 */
std::vector<FlowRowBlock> flowRowBlocks(const MpcEquivalenceFixture& mpc,
                                        FlowCoordinates coordinates) {
  std::vector<FlowRowBlock> blocks;
  Eigen::Index start = 0;
  Eigen::Index momentumRows = 0;
  for (const validation::EulerStateSegment& segment :
       mpc.eulerLayout().segments) {
    const bool quaternionRows =
        coordinates == FlowCoordinates::kQuaternion &&
        segment.kind == validation::EulerStateSegmentKind::kEulerZyxAngles;
    const Eigen::Index size =
        quaternionRows ? 4 : static_cast<Eigen::Index>(segment.dim);
    const bool momentum = absl::StartsWith(segment.name, "normalized_") &&
                          absl::EndsWith(segment.name, "_momentum");
    if (momentum) momentumRows += size;
    blocks.push_back(
        FlowRowBlock{.start = start,
                     .size = size,
                     .relativeTolerance = momentum ? kMomentumRateTolerance
                                                   : kFlowMapTolerance});
    start += size;
  }
  EXPECT_EQ(momentumRows, mpc.isWholeBody() ? 0 : 6)
      << "the centroidal state opens with the normalized momentum";
  return blocks;
}

/**
 * The entrywise bound of `expected` (rows by the blocks, any number of
 * columns): each block's tolerance times max(1, its largest entry). NaN on rows
 * no block covers, which fails the comparison.
 */
matrix_t rowClassBound(const matrix_t& expected,
                       const std::vector<FlowRowBlock>& blocks) {
  matrix_t bound =
      matrix_t::Constant(expected.rows(), expected.cols(),
                         std::numeric_limits<scalar_t>::quiet_NaN());
  for (const FlowRowBlock& block : blocks) {
    if (block.start + block.size > expected.rows()) continue;
    const matrix_t rows = expected.middleRows(block.start, block.size);
    const scalar_t scale =
        rows.size() == 0 ? 1.0 : std::max(1.0, rows.cwiseAbs().maxCoeff());
    bound.middleRows(block.start, block.size)
        .setConstant(block.relativeTolerance * scale);
  }
  return bound;
}

/** max(floor, max |m|). */
scalar_t scaleOf(const matrix_t& m, scalar_t floor) {
  return m.size() == 0 ? floor : std::max(floor, m.cwiseAbs().maxCoeff());
}

/** A matrix of `like`'s shape, every entry `value`. */
matrix_t constantLike(const matrix_t& like, scalar_t value) {
  return matrix_t::Constant(like.rows(), like.cols(), value);
}

/** The SQP's linearization of one shooting interval of the MPC at (x, u): what
 * Transcription.cpp builds on the manifold. */
struct ProductionInterval {
  /** Pi(F(x, u)): the projected end of the interval. */
  vector_t next;
  /** difference(next, F): zero, the shooting gap of a node at the interval's
   * own end. */
  vector_t gap;
  matrix_t stateSensitivity;  // A_t, ndx x ndx
  matrix_t inputSensitivity;  // B_t, ndx x nu
};

/**
 * The interval of `halves` RK4 steps of dt / halves (1 or kSubsteps), the
 * ambient sensitivities chained without a projection in between, then pushed
 * forward onto the manifold as the SQP does.
 */
ProductionInterval productionInterval(MpcEquivalenceFixture& mpc,
                                      const vector_t& state,
                                      const vector_t& input,
                                      int halves) {
  SystemDynamicsBase& dynamics = *mpc.problem().dynamicsPtr;
  const DynamicsSensitivityDiscretizer sensitivity =
      selectDynamicsSensitivityDiscretization(mpc.sqpSettings().integratorType);
  const scalar_t dt = mpc.nodeTimeStep() / static_cast<scalar_t>(halves);
  VectorFunctionLinearApproximation interval =
      sensitivity(dynamics, /*t=*/0.0, state, input, dt);
  for (int half = 1; half < halves; ++half) {
    const VectorFunctionLinearApproximation next =
        sensitivity(dynamics, half * dt, interval.f, input, dt);
    interval.dfdu = next.dfdx * interval.dfdu + next.dfdu;
    interval.dfdx = next.dfdx * interval.dfdx;
    interval.f = next.f;
  }
  vector_t end = interval.f;
  mpc.manifold().project(end);
  mpc.manifold().pushForwardDynamics(state, end, interval);
  return ProductionInterval{.next = std::move(end),
                            .gap = interval.f,
                            .stateSensitivity = interval.dfdx,
                            .inputSensitivity = interval.dfdu};
}

/** The Euler RK4 map's interval at (x_e, u) in the tangent of the quaternion
 * layout (E4), with its finite-difference error. */
struct EulerInterval {
  vector_t next;              // F_e(x_e, u)
  matrix_t stateSensitivity;  // P_e(F_e) A_e P_e(x_e)^-1
  matrix_t stateSensitivityError;
  matrix_t inputSensitivity;  // P_e(F_e) B_e
  matrix_t inputSensitivityError;
};

/** The same of `halves` Euler RK4 steps of dt / halves. */
EulerInterval eulerInterval(const MpcEquivalenceFixture& mpc,
                            const EulerMpcOracle& oracle,
                            const vector_t& eulerState,
                            const vector_t& input,
                            int halves) {
  const scalar_t dt = mpc.nodeTimeStep() / static_cast<scalar_t>(halves);
  const std::function<vector_t(const vector_t&, const vector_t&)> flow =
      [&oracle, dt, halves](const vector_t& x, const vector_t& u) {
        vector_t end = x;
        for (int half = 0; half < halves; ++half)
          end = oracle.rk4Step(end, u, dt);
        return end;
      };
  EulerInterval interval;
  interval.next = flow(eulerState, input);
  const FiniteDifferenceJacobian stateJacobian = richardsonJacobian(
      [&flow, &input](const vector_t& x) { return flow(x, input); }, eulerState,
      kFiniteDifferenceStep);
  const FiniteDifferenceJacobian inputJacobian = richardsonJacobian(
      [&flow, &eulerState](const vector_t& u) { return flow(eulerState, u); },
      input, kFiniteDifferenceStep);
  const matrix_t endMap = mpc.physicalTangentJacobian(interval.next);
  const matrix_t startMapInverse =
      mpc.physicalTangentJacobian(eulerState).inverse();
  interval.stateSensitivity = endMap * stateJacobian.value * startMapInverse;
  interval.stateSensitivityError =
      endMap.cwiseAbs() * stateJacobian.errorBound * startMapInverse.cwiseAbs();
  interval.inputSensitivity = endMap * inputJacobian.value;
  interval.inputSensitivityError = endMap.cwiseAbs() * inputJacobian.errorBound;
  return interval;
}

/**
 * Expects the production interval at Phi(x_e) to be the Euler one through the
 * chart (E4), both seen in `frame` (the tangent world rotation that takes the
 * production's world to the oracle's; the identity for the same world) with the
 * input mapped by `inputFrame`:
 *
 *   |A_t - A_e,t| <= 2 (256/255) (|A_t - A_t,4| + |A_e,t - A_e,t,4|) + FD error
 * + kJacobianTolerance scale,
 *
 * A_t,4 being the interval of kSubsteps = 4 quarter steps: each RK4 map's local
 * error is C dt^5 + O(dt^6), the quarter steps' C dt^5 / 256, so (256/255)
 * |F(dt) - F4(dt)| estimates each map's distance from the exact flow's
 * linearization, which is the same through the chart (the exact flow is
 * coordinate-free); RK4's dependence on the coordinates is the sum of the two.
 * The ends are compared the same way, in the tangent at the production's end.
 */
void expectEquivalentIntervals(MpcEquivalenceFixture& mpc,
                               const EulerMpcOracle& oracle,
                               const vector_t& state,
                               const vector_t& input,
                               const vector_t& eulerState,
                               const vector_t& eulerInput,
                               const matrix_t& frame,
                               const matrix_t& inputFrame,
                               const vector4_t& worldRotation) {
  const ProductionInterval production =
      productionInterval(mpc, state, input, /*halves=*/1);
  const ProductionInterval productionHalves =
      productionInterval(mpc, state, input, /*halves=*/kSubsteps);
  const EulerInterval euler =
      eulerInterval(mpc, oracle, eulerState, eulerInput, /*halves=*/1);
  const EulerInterval eulerHalves =
      eulerInterval(mpc, oracle, eulerState, eulerInput, /*halves=*/kSubsteps);
  const matrix_t frameInverse = frame.inverse();
  const matrix_t eulerA = frameInverse * euler.stateSensitivity * frame;
  const matrix_t eulerAHalves =
      frameInverse * eulerHalves.stateSensitivity * frame;
  const matrix_t eulerB = frameInverse * euler.inputSensitivity * inputFrame;
  const matrix_t eulerBHalves =
      frameInverse * eulerHalves.inputSensitivity * inputFrame;
  constexpr scalar_t kHalving = kStepHalvingSafety * kSubstepsErrorFactor;

  EXPECT_LE(production.gap.cwiseAbs().maxCoeff(), 1.0e-12)
      << "the gap of a node at its own end";
  const matrix_t boundA =
      kHalving *
          ((production.stateSensitivity - productionHalves.stateSensitivity)
               .cwiseAbs() +
           (eulerA - eulerAHalves).cwiseAbs()) +
      frameInverse.cwiseAbs() * euler.stateSensitivityError * frame.cwiseAbs() +
      constantLike(eulerA, kJacobianTolerance * scaleOf(eulerA, /*floor=*/1.0));
  EXPECT_TRUE(
      isWithinEntrywiseBound(production.stateSensitivity, eulerA, boundA))
      << "A_t against P_e(F_e) A_e P_e^-1";
  const matrix_t boundB =
      kHalving *
          ((production.inputSensitivity - productionHalves.inputSensitivity)
               .cwiseAbs() +
           (eulerB - eulerBHalves).cwiseAbs()) +
      frameInverse.cwiseAbs() * euler.inputSensitivityError *
          inputFrame.cwiseAbs() +
      constantLike(eulerB, kJacobianTolerance * scaleOf(eulerB, /*floor=*/1.0));
  EXPECT_TRUE(
      isWithinEntrywiseBound(production.inputSensitivity, eulerB, boundB))
      << "B_t against P_e(F_e) B_e";

  // The ends: Pi(F_q(Phi(x_e))) against Phi(F_e(x_e)), turned back into the
  // production's world.
  const matrix_t ambientBack =
      mpc.ambientWorldRotation(quaternionConjugate(worldRotation));
  const vector_t eulerEnd = ambientBack * mpc.quaternionStateOf(euler.next);
  const vector_t eulerEndHalves =
      ambientBack * mpc.quaternionStateOf(eulerHalves.next);
  const vector_t endDifference =
      mpc.manifold().difference(production.next, eulerEnd);
  const scalar_t endBound =
      kHalving * (mpc.manifold()
                      .difference(productionHalves.next, production.next)
                      .cwiseAbs()
                      .maxCoeff() +
                  mpc.manifold()
                      .difference(eulerEndHalves, eulerEnd)
                      .cwiseAbs()
                      .maxCoeff()) +
      kFlowMapTolerance * scaleOf(production.next, /*floor=*/1.0);
  EXPECT_LE(endDifference.cwiseAbs().maxCoeff(), endBound)
      << "the interval's end";
}

/** The largest joint entry of a random input: joint velocities [rad/s]
 * (centroidal), joint accelerations [rad/s^2] (whole body). */
scalar_t maxAbsJointInput(const MpcEquivalenceFixture& mpc) {
  return mpc.isWholeBody() ? 10.0 : 1.0;
}

TEST_P(MpcDynamicsEquivalence, TheFlowMapIsTheOraclesThroughTheChart) {
  MpcEquivalenceFixture& mpc = this->mpc();
  SystemDynamicsBase& dynamics = *mpc.problem().dynamicsPtr;
  const std::vector<FlowRowBlock> blocks =
      flowRowBlocks(mpc, FlowCoordinates::kQuaternion);
  std::mt19937 generator(/*sd=*/20261009);
  for (int sample = 0; sample < kRandomStates; ++sample) {
    SCOPED_TRACE(sample);
    const vector_t eulerState =
        mpc.randomEulerState(EulerSamplingBounds{}, generator);
    const vector_t input = mpc.randomInput(maxAbsJointInput(mpc), generator);
    const vector_t flow = dynamics.computeFlowMap(
        /*t=*/0.0, mpc.quaternionStateOf(eulerState), input);
    // (E1): the ambient flow of the quaternion state is dPhi times the Euler
    // flow.
    const vector_t expected =
        mpc.chartJacobian(eulerState) * mpc.oracle().flowMap(eulerState, input);
    EXPECT_TRUE(isWithinEntrywiseBound(flow, expected,
                                       rowClassBound(expected, blocks)));
  }
}

TEST_P(MpcDynamicsEquivalence,
       TheFlowMapLinearizationIsTheChainRuleThroughTheChart) {
  MpcEquivalenceFixture& mpc = this->mpc();
  SystemDynamicsBase& dynamics = *mpc.problem().dynamicsPtr;
  const EulerMpcOracle& oracle = mpc.oracle();
  const std::vector<FlowRowBlock> blocks =
      flowRowBlocks(mpc, FlowCoordinates::kQuaternion);
  std::mt19937 generator(/*sd=*/20261010);
  for (int sample = 0; sample < kRandomStates / 2; ++sample) {
    SCOPED_TRACE(sample);
    const vector_t eulerState =
        mpc.randomEulerState(EulerSamplingBounds{}, generator);
    const vector_t input = mpc.randomInput(maxAbsJointInput(mpc), generator);
    const vector_t state = mpc.quaternionStateOf(eulerState);
    const VectorFunctionLinearApproximation linearization =
        dynamics.linearApproximation(/*t=*/0.0, state, input);
    const matrix_t chart = mpc.chartJacobian(eulerState);

    // (E2) on the state: d (dPhi(x_e) f_e(x_e, u)) / dx_e, the derivative of
    // the chart's image of the Euler flow, is (df_a / dx) dPhi; the left side
    // includes the chart's curvature d dPhi / dx_e [f_e].
    const FiniteDifferenceJacobian expectedState = richardsonJacobian(
        [&mpc, &oracle, &input](const vector_t& x) {
          return vector_t(mpc.chartJacobian(x) * oracle.flowMap(x, input));
        },
        eulerState, kFiniteDifferenceStep);
    const matrix_t actualState = linearization.dfdx * chart;
    EXPECT_TRUE(isWithinEntrywiseBound(
        actualState, expectedState.value,
        expectedState.errorBound +
            constantLike(actualState,
                         kJacobianTolerance *
                             scaleOf(expectedState.value, /*floor=*/1.0))))
        << "df/dx dPhi";
    // dPhi = E P_e: the same derivative in the manifold's tangent columns,
    // times P_e.
    matrix_t tangentColumns = linearization.dfdx;
    mpc.manifold().pullBackStateColumns(state, tangentColumns);
    EXPECT_TRUE(
        isCloseToScale(tangentColumns * mpc.physicalTangentJacobian(eulerState),
                       actualState, /*relativeTolerance=*/1.0e-12,
                       /*floor=*/1.0))
        << "df/dx E P_e";

    // On the input: f_e is affine in u (the wrenches enter the momentum rate
    // and the base rows linearly, the joint entries the base velocity or
    // acceleration), so central differences of unit step are exact but for
    // round-off.
    matrix_t eulerInputJacobian(oracle.stateDim(), oracle.inputDim());
    for (Eigen::Index column = 0; column < oracle.inputDim(); ++column) {
      eulerInputJacobian.col(column) =
          0.5 *
          (oracle.flowMap(eulerState,
                          input + vector_t::Unit(oracle.inputDim(), column)) -
           oracle.flowMap(eulerState,
                          input - vector_t::Unit(oracle.inputDim(), column)));
    }
    // The input Jacobian's rows are held to the flow's classes (design
    // section 4.4): the same products of the same affine map, without a
    // finite-difference error.
    const matrix_t expectedInput = chart * eulerInputJacobian;
    EXPECT_TRUE(isWithinEntrywiseBound(linearization.dfdu, expectedInput,
                                       rowClassBound(expectedInput, blocks)))
        << "df/du";
  }
}

TEST_P(MpcDynamicsEquivalence,
       TheShootingIntervalIsTheEulerRk4MapThroughTheChart) {
  MpcEquivalenceFixture& mpc = this->mpc();
  ASSERT_EQ(mpc.sqpSettings().integratorType, SensitivityIntegratorType::RK4)
      << "the oracle integrates with RK4";
  std::mt19937 generator(/*sd=*/20261011);
  const matrix_t tangentIdentity = matrix_t::Identity(
      mpc.manifold().getTangentDim(), mpc.manifold().getTangentDim());
  const matrix_t inputIdentity =
      matrix_t::Identity(mpc.oracle().inputDim(), mpc.oracle().inputDim());
  for (int sample = 0; sample < kRandomStates / 3; ++sample) {
    SCOPED_TRACE(sample);
    const vector_t eulerState =
        mpc.randomEulerState(EulerSamplingBounds{}, generator);
    const vector_t input = mpc.randomInput(maxAbsJointInput(mpc), generator);
    expectEquivalentIntervals(mpc, mpc.oracle(),
                              mpc.quaternionStateOf(eulerState), input,
                              eulerState, input, tangentIdentity, inputIdentity,
                              /*worldRotation=*/vector4_t(0.0, 0.0, 0.0, 1.0));
  }
}

/**
 * A state of the MPC's world at a pitch of exactly `pitch` (+-90 degrees:
 * gimbal lock of the Euler coordinates), built as a random Euler state of a
 * world turned by `worldRotation` (g), where the same robot is near level,
 * turned back. Passive data.
 */
struct TurnedWorldCase {
  /** The MPC's state, in its world. */
  vector_t state;
  /** The same physical state in the turned world, in Euler coordinates: pitch
   * and roll within 0.3 rad. */
  vector_t turnedEulerState;
  /** g: the turned state is g . state
   * (MpcEquivalenceFixture::ambientWorldRotation()). */
  vector4_t worldRotation;
};

TurnedWorldCase pitchedCase(const MpcEquivalenceFixture& mpc,
                            scalar_t pitch,
                            std::mt19937& generator) {
  EulerSamplingBounds bounds;
  bounds.maxAbsPitch = 0.3;
  bounds.maxAbsRoll = 0.3;
  TurnedWorldCase pitched;
  pitched.turnedEulerState = mpc.randomEulerState(bounds, generator);
  const vector_t turned = mpc.quaternionStateOf(pitched.turnedEulerState);
  const vector4_t turnedOrientation = mpc.model().getBaseOrientation(turned);
  std::uniform_real_distribution<scalar_t> angle(-M_PI, M_PI);
  // The orientation in the MPC's world: R_z(yaw) R_y(pitch) R_x(roll), exactly
  // at the requested pitch.
  const vector4_t orientation = quaternionFromEulerZyx(
      vector3_t(angle(generator), pitch, angle(generator)));
  pitched.worldRotation =
      quaternionProduct(turnedOrientation, quaternionConjugate(orientation));
  pitched.state =
      mpc.ambientWorldRotation(quaternionConjugate(pitched.worldRotation)) *
      turned;
  return pitched;
}

TEST_P(MpcDynamicsEquivalence,
       AtGimbalLockTheDynamicsAreTheLevelOraclesUnderATurnedGravity) {
  MpcEquivalenceFixture& mpc = this->mpc();
  SystemDynamicsBase& dynamics = *mpc.problem().dynamicsPtr;
  const std::vector<FlowRowBlock> blocks =
      flowRowBlocks(mpc, FlowCoordinates::kQuaternion);
  std::mt19937 generator(/*sd=*/20261012);
  // +-90 degrees exactly, and 1e-4 rad short of it, where the Euler
  // coordinates' T_B is within 1e-4 of singular.
  for (const scalar_t pitch : {0.5 * M_PI, -0.5 * M_PI, 0.5 * M_PI - 1.0e-4}) {
    SCOPED_TRACE(pitch);
    const TurnedWorldCase pitched = pitchedCase(mpc, pitch, generator);
    ASSERT_NEAR(std::abs(std::asin(std::clamp(
                    -quaternionRotationMatrix(
                        mpc.model().getBaseOrientation(pitched.state))(2, 0),
                    -1.0, 1.0))),
                std::abs(pitch), 1.0e-6)
        << "the case is not at the requested pitch";
    const vector3_t turnedGravity =
        quaternionRotationMatrix(pitched.worldRotation) *
        vector3_t(0.0, 0.0, -9.81);
    const absl::StatusOr<std::unique_ptr<EulerMpcOracle>> turnedOracle =
        mpc.oracleUnderGravity(turnedGravity);
    ASSERT_TRUE(turnedOracle.ok()) << turnedOracle.status();
    const EulerMpcOracle& oracle = **turnedOracle;
    const matrix_t ambient = mpc.ambientWorldRotation(pitched.worldRotation);
    const matrix_t ambientInverse =
        mpc.ambientWorldRotation(quaternionConjugate(pitched.worldRotation));
    const matrix_t inputTurn = mpc.inputWorldRotation(pitched.worldRotation);
    const vector_t input = mpc.randomInput(maxAbsJointInput(mpc), generator);
    const vector_t turnedInput = inputTurn * input;
    const vector_t& turnedEulerState = pitched.turnedEulerState;

    // The flow: f(x, u) = M^-1 f(M x, N u; R(g) gravity), the right side the
    // oracle's through the chart (E1).
    const vector_t flow =
        dynamics.computeFlowMap(/*t=*/0.0, pitched.state, input);
    ASSERT_TRUE(flow.allFinite());
    const vector_t expectedFlow = ambientInverse *
                                  mpc.chartJacobian(turnedEulerState) *
                                  oracle.flowMap(turnedEulerState, turnedInput);
    EXPECT_TRUE(isWithinEntrywiseBound(flow, expectedFlow,
                                       rowClassBound(expectedFlow, blocks)))
        << "f";

    // The linearization: df/dx = M^-1 df'/dx' M and df/du = M^-1 df'/du' N, the
    // turned ones the chain rule (E2).
    const VectorFunctionLinearApproximation linearization =
        dynamics.linearApproximation(/*t=*/0.0, pitched.state, input);
    const FiniteDifferenceJacobian expectedState = richardsonJacobian(
        [&mpc, &oracle, &turnedInput](const vector_t& x) {
          return vector_t(mpc.chartJacobian(x) *
                          oracle.flowMap(x, turnedInput));
        },
        turnedEulerState, kFiniteDifferenceStep);
    const matrix_t actualState = ambient * linearization.dfdx * ambientInverse *
                                 mpc.chartJacobian(turnedEulerState);
    EXPECT_TRUE(isWithinEntrywiseBound(
        actualState, expectedState.value,
        expectedState.errorBound +
            constantLike(actualState,
                         kJacobianTolerance *
                             scaleOf(expectedState.value, /*floor=*/1.0))))
        << "df/dx";
    matrix_t turnedInputJacobian(oracle.stateDim(), oracle.inputDim());
    for (Eigen::Index column = 0; column < oracle.inputDim(); ++column) {
      turnedInputJacobian.col(column) =
          0.5 * (oracle.flowMap(
                     turnedEulerState,
                     turnedInput + vector_t::Unit(oracle.inputDim(), column)) -
                 oracle.flowMap(
                     turnedEulerState,
                     turnedInput - vector_t::Unit(oracle.inputDim(), column)));
    }
    // In the turned world's rows (the row classes are the same blocks, turned
    // within each block).
    const matrix_t expectedInput =
        mpc.chartJacobian(turnedEulerState) * turnedInputJacobian;
    EXPECT_TRUE(isWithinEntrywiseBound(
        ambient * linearization.dfdu * inputTurn.inverse(), expectedInput,
        rowClassBound(expectedInput, blocks)))
        << "df/du";

    // The SQP's shooting interval at gimbal lock (E4), in the turned world's
    // tangent.
    expectEquivalentIntervals(mpc, oracle, pitched.state, input,
                              turnedEulerState, turnedInput,
                              mpc.tangentWorldRotation(pitched.worldRotation),
                              inputTurn, pitched.worldRotation);
  }
}

TEST_P(MpcDynamicsEquivalence, AcrossAYawOfPiTheShootingIntervalIsTheSame) {
  MpcEquivalenceFixture& mpc = this->mpc();
  std::mt19937 generator(/*sd=*/20261013);
  const Eigen::Index yaw =
      static_cast<Eigen::Index>(mpc.model().getBaseOrientationStartIndex());
  for (int sample = 0; sample < 3; ++sample) {
    SCOPED_TRACE(sample);
    vector_t eulerState =
        mpc.randomEulerState(EulerSamplingBounds{}, generator);
    const vector_t input = mpc.randomInput(maxAbsJointInput(mpc), generator);
    // The Euler yaw of the state is the row of the base orientation's block
    // (yaw, pitch, roll) at the same start index.
    eulerState(yaw) = M_PI;
    const vector_t above = mpc.quaternionStateOf(eulerState);
    eulerState(yaw) = -M_PI;
    const vector_t below = mpc.quaternionStateOf(eulerState);
    // The Euler yaw jumps by 2 pi; the two quaternions are each other's
    // negative, the same rotation.
    ASSERT_LE((mpc.model().getBaseOrientation(above) +
               mpc.model().getBaseOrientation(below))
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-15);
    const ProductionInterval fromAbove =
        productionInterval(mpc, above, input, /*halves=*/1);
    const ProductionInterval fromBelow =
        productionInterval(mpc, below, input, /*halves=*/1);
    // Round-off of the same computation on the two representations of one
    // physical state.
    EXPECT_TRUE(isCloseToScale(fromBelow.stateSensitivity,
                               fromAbove.stateSensitivity, kFlowMapTolerance,
                               /*floor=*/1.0))
        << "A_t";
    EXPECT_TRUE(isCloseToScale(fromBelow.inputSensitivity,
                               fromAbove.inputSensitivity, kFlowMapTolerance,
                               /*floor=*/1.0))
        << "B_t";
    EXPECT_LE(mpc.manifold()
                  .difference(fromAbove.next, fromBelow.next)
                  .cwiseAbs()
                  .maxCoeff(),
              kFlowMapTolerance * scaleOf(fromAbove.next, /*floor=*/1.0))
        << "the end of the interval";
  }
}

TEST_P(MpcDynamicsEquivalence,
       WithARotationInTheGapTheShootingIntervalIsTheDerivativeOfTheGap) {
  // The SQP's interval when the next node is off the flow's end by a rotation b
  // (a nonzero shooting gap, which every other interval test of the suite keeps
  // at round-off): pushForwardDynamics() applies Jr^-1(b) (design section 2.7).
  // On the MPC's own dynamics, A_t and B_t must be the derivatives of the gap's
  // tangent linearization,
  //   g(d, du) = difference(x_next, Pi(F(x (+) d, u + du))),  A_t = dg/dd(0,
  //   0),  B_t = dg/ddu(0, 0),
  // against Richardson's central differences, with |b| up to 1.5 rad about
  // random axes, and the gap's value g(0, 0).
  MpcEquivalenceFixture& mpc = this->mpc();
  SystemDynamicsBase& dynamics = *mpc.problem().dynamicsPtr;
  const StateManifold& manifold = mpc.manifold();
  const DynamicsSensitivityDiscretizer sensitivity =
      selectDynamicsSensitivityDiscretization(mpc.sqpSettings().integratorType);
  const DynamicsDiscretizer discretizer =
      selectDynamicsDiscretization(mpc.sqpSettings().integratorType);
  const scalar_t dt = mpc.nodeTimeStep();
  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(manifold.getTangentDim());
  const Eigen::Index rotation =
      static_cast<Eigen::Index>(mpc.model().getBaseOrientationStartIndex());
  std::mt19937 generator(/*sd=*/20261014);
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  for (int sample = 0; sample < 3; ++sample) {
    SCOPED_TRACE(sample);
    const vector_t state = mpc.quaternionStateOf(
        mpc.randomEulerState(EulerSamplingBounds{}, generator));
    const vector_t input = mpc.randomInput(maxAbsJointInput(mpc), generator);
    const VectorFunctionLinearApproximation interval =
        sensitivity(dynamics, /*t=*/0.0, state, input, dt);
    vector_t end = interval.f;
    manifold.project(end);
    for (const scalar_t angle : {0.3, 1.5}) {
      SCOPED_TRACE(absl::StrCat("|b| = ", angle));
      // The next node: the end turned by a rotation of `angle` about a random
      // axis (and nothing else moved).
      const vector3_t axis(normal(generator), normal(generator),
                           normal(generator));
      vector_t offset = vector_t::Zero(tangentDim);
      offset.segment<3>(rotation) = angle * axis.normalized();
      vector_t next;
      manifold.retract(end, offset, /*alpha=*/1.0, next);
      VectorFunctionLinearApproximation production = interval;
      manifold.pushForwardDynamics(state, next, production);
      ASSERT_NEAR(manifold.getMaximumRotationAngle(production.f), angle,
                  1.0e-12)
          << "the gap's rotation is |b|";

      const std::function<vector_t(const vector_t&, const vector_t&)> gap =
          [&](const vector_t& stateStep, const vector_t& inputValue) {
            vector_t start;
            manifold.retract(state, stateStep, /*alpha=*/1.0, start);
            vector_t flowEnd =
                discretizer(dynamics, /*t=*/0.0, start, inputValue, dt);
            manifold.project(flowEnd);
            return manifold.difference(next, flowEnd);
          };
      const vector_t zero = vector_t::Zero(tangentDim);
      EXPECT_TRUE(isCloseToScale(production.f, gap(zero, input),
                                 /*relativeTolerance=*/1.0e-12, /*floor=*/1.0))
          << "the gap";
      const FiniteDifferenceJacobian stateJacobian = richardsonJacobian(
          [&gap, &input](const vector_t& d) { return gap(d, input); }, zero,
          kFiniteDifferenceStep);
      const FiniteDifferenceJacobian inputJacobian = richardsonJacobian(
          [&gap, &zero](const vector_t& u) { return gap(zero, u); }, input,
          kFiniteDifferenceStep);
      EXPECT_TRUE(isWithinEntrywiseBound(
          production.dfdx, stateJacobian.value,
          stateJacobian.errorBound +
              constantLike(stateJacobian.value,
                           kJacobianTolerance *
                               scaleOf(stateJacobian.value, /*floor=*/1.0))))
          << "A_t";
      EXPECT_TRUE(isWithinEntrywiseBound(
          production.dfdu, inputJacobian.value,
          inputJacobian.errorBound +
              constantLike(inputJacobian.value,
                           kJacobianTolerance *
                               scaleOf(inputJacobian.value, /*floor=*/1.0))))
          << "B_t";
      // The positive control: without Jr^-1(b) (the quaternion manifold's
      // sensitivity at a zero gap) the rotation rows are off at first order in
      // |b|.
      VectorFunctionLinearApproximation withoutTheGap = interval;
      manifold.pushForwardDynamics(state, end, withoutTheGap);
      EXPECT_GT(
          (withoutTheGap.dfdx - stateJacobian.value).cwiseAbs().maxCoeff(),
          0.05 * angle);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Shipped,
    MpcDynamicsEquivalence,
    ::testing::ValuesIn(mpcEquivalenceConfigurationNames()),
    mpcEquivalenceTestName);

}  // namespace
}  // namespace ocs2::humanoid
