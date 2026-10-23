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
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/dynamics/SystemDynamicsBase.h"
#include "ocs2_core/integration/SensitivityIntegrator.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_data/TimeDiscretization.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_mpc_validation/chart/EulerChart.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/EulerChartStateManifold.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/MpcEquivalenceFixture.h"

/*
 * One real-time iteration of the MPC's SQP (sqp_iteration 1, from a warm start held over the horizon) on the quaternion
 * layout against the Euler formulation's, through the chart, for both formulations.
 *
 * The Euler formulation's iteration with today's terms is the SQP on EulerChartStateManifold: its tangent is
 * d = P_e dx_e, its E is the quaternion manifold's (so every term pulls back as on the quaternion layout), and its gaps
 * and their sensitivities are the Euler coordinates' defects. Every iteration runs under the walking reference of
 * MpcEquivalenceFixture::walkingReference() from warm starts built on its resting warm start (restingWarmStart(): the
 * task file's initial state held with the input that carries the weight on both feet), so the suite reads nothing but
 * the configuration's shipped files. The claims, each a test:
 *
 *   (S1) Where no shooting gap carries a rotation and, on the whole body, no gap carries an angular velocity, the
 *        quaternion manifold's QP is the chart's: E is the same, such a gap has the same value (on the whole body also in
 *        the angular-velocity rows: T_B(Theta) dTheta_F is w_F when Theta_F = Theta_next) and the same sensitivity
 *        (Jr^-1(0) = I = P_e(x_e,next) P_e(x_e,F)^-1, which on the whole body needs equal Euler rates as well as equal
 *        angles: (S1w)). The warm start is then a resting robot carrying its weight with no moment on the base (the
 *        resting warm-start input corrected by the least-norm change that cancels the moment,
 *        MpcEquivalenceFixture::torqueFreeInput()), level or tilted, whose gaps the test shows to carry neither a rotation
 *        nor an angular velocity. From the warm start itself the two QPs are one up to round-off (kSameQpTolerance;
 *        design section 4.4): both integrate the same RK4 flow, and without a rotation or an angular velocity in the
 *        gaps the chart maps the sensitivities by the identity. (RK4's O(dt^5) coordinate dependence separates the chart
 *        SQP from a true Euler-RK4 formulation, which testMpcDynamicsEquivalence and testMpcRolloutEquivalence bound;
 *        it is not in this comparison.) From an initial state off it by an attitude eps they differ only through the
 *        initial gap, Log(R_ws' R_init) against T_B(Theta_ws) (Theta_init - Theta_ws), equal to first order: O(eps^2),
 *        tier 2 (halving eps divides the difference by a ratio in [kTierTwoRatioLow, kTierTwoRatioHigh]). (S1) and (S2)
 *        compare the QPs' solutions (the steps over the line search's step size, qpSolutionOf()) and gains.
 *   (S2) Where the gaps do carry a rotation b (the resting warm start: the weight-carrying input has a moment about the
 *        center of mass, so the base turns within each interval), the two QPs differ in the gaps' sensitivity, Jr^-1(b)
 *        against T_B(Theta_next) T_B(Theta_F)^-1, both I + O(|b|) with different first-order terms (at a level next
 *        node, I + 1/2 [b]x against I + [[0, 0, b_y], [0, 0, -b_x], [0, b_x, 0]]), and in the gaps' value at O(|b|^2);
 *        on the whole body the gaps also carry the angular velocity the moment builds up within the interval, which adds
 *        the coupling block of (S1w), linear in the moment too: the iterations differ by a quantity that vanishes with
 *        the moment, at least linearly. Scaling the warm start's moment (and with it b, linearly) by 1/2 divides their
 *        difference by at least kFirstOrderRatio; more where the QP's weakly weighed directions (a yaw weighed by no
 *        cost on EngineAI SA01) make its response to the perturbation nonlinear at these sizes. Both are Gauss-Newton
 *        SQP iterations of the same discrete problem, in two charts of its defects, with the same fixed points (S3);
 *        this is the coordinate effect of the transcription: their difference there is of order |b| times the QP's
 *        sensitivity, not RK4's O(dt^5).
 *   (S1w) On the whole body, a gap of the angular velocity alone, dw = w_F - w_next with Theta_F = Theta_next, has the
 *        same value in both charts, and the chart's sensitivity P_e(x_e,next) P_e(x_e,F)^-1 is the identity but for one
 *        block, the angular velocity's rows against the rotation's columns, C(Theta, dTheta_next/dt - dTheta_F/dt)
 *        T_B(Theta)^-1 with C(Theta, r) = d(T_B(Theta) r) / dTheta (EulerChartStateManifold.h): linear in the gap, zero for
 *        a roll-rate gap. So the two QPs differ at first order in an angular-velocity gap without any rotation in it. Shown
 *        on one shooting interval (the block, by finite differences of T_B, and its linearity) and on a whole iteration
 *        from a warm start whose every gap is the same angular-velocity kick (the QP solutions' difference halves with
 *        the kick, and vanishes with it).
 *   (S3) The two SQPs solve one discrete problem - the same terms of the same states, the same RK4 node map and
 *        projection, a gap that vanishes exactly where the quaternion manifold's does - in two charts of its defects, so
 *        they have the same fixed points, the trajectories without gaps, where (S1) their QPs are one. At the resting warm
 *        start (whose gaps carry rotations, (S2)) and at the quaternion SQP's iterate from it the difference of the two
 *        charts' gap linearizations is at most proportional to the gaps, and as the iterate's gaps are closed it falls
 *        with them, to none without a gap. The gaps are closed by construction, not by the SQP, which need not converge:
 *        its QP's Hessian is the costs' Gauss-Newton Hessian, without the hard constraints' curvature lambda' grad^2 g
 *        (whether the constraints are projected out or not), its merit has no penalty on their violation, and its line
 *        search accepts any decrease of the merit while the violation stays below g_max. So each full step can lower the
 *        merit and raise the violation until a full step would pass g_max and a reduced one resets it: a cycle whose
 *        iterates, within a period, none of those before dominates. From this warm start it does so on DRC Atlas, the
 *        same cycle in both charts (MpcSqpConvergence.OnAtlasTheEulerChartCyclesAsTheQuaternionManifoldDoes), and on
 *        the whole-body G1 (testMpcSqpConvergence.cpp; design README section 6,
 *        risk 12).
 */

namespace ocs2::humanoid {
namespace {

/**
 * (S1) at a gap-free warm start: the two QPs are one up to round-off (the same RK4 flow, an identity chart map of the
 * gap-free sensitivities), relative to the step's largest entry; 27 times the largest recorded difference, 3.7e-12 (G1-WB).
 */
constexpr scalar_t kSameQpTolerance = 1.0e-10;
/**
 * Tier 1 (design section 4.4): the floor below which a difference is too small to measure an O(eps^2) or first-order
 * ratio (tier 2, S2, S1w), relative to the step's largest entry.
 */
constexpr scalar_t kTierOneTolerance = 1.0e-6;
/** Tier 2: an O(eps^2) difference shrinks by 4 when eps halves; [3, 5] allows the O(eps^3) term. */
constexpr scalar_t kTierTwoRatioLow = 3.0;
constexpr scalar_t kTierTwoRatioHigh = 5.0;
/**
 * (S2): a difference of at least first order in the gaps' rotation shrinks by at least 2 when the rotation halves; 1.5
 * allows the higher-order terms' change of sign.
 */
constexpr scalar_t kFirstOrderRatio = 1.5;
/** [rad] A rotation of a shooting gap at round-off: the free fall's. */
constexpr scalar_t kNoRotation = 1.0e-12;
/** [rad/s] An angular velocity of a shooting gap at round-off: the torque-free warm start's. */
constexpr scalar_t kNoAngularVelocity = 1.0e-10;

/**
 * [rad/s] The angular-velocity gap of (S1w), per shooting interval: a turn about the body's z axis (a yaw rate, which C
 * reads; a pitch-rate kick would pitch the rolled-out warm start towards the chart's singularity), small enough that the
 * kicked warm start turns by well under a radian over the horizon.
 */
constexpr scalar_t kAngularVelocityKick = 0.01;

/**
 * (S1w) on one interval: the chart's sensitivity block C(Theta, r) T_B^-1 against central differences of T_B(Theta) r
 * of step kRateMapStep: truncation h^2 / 6 times the third derivative of T_B (entries at most 1) times |r| <= 0.1 rad/s,
 * 2e-13, round-off eps / h = 2e-10; 1e-8 is 50 times their sum.
 */
constexpr scalar_t kRateMapStep = 1.0e-6;
constexpr scalar_t kCouplingBlockTolerance = 1.0e-8;
/** The chart's and the quaternion manifold's arithmetic on the same interval: the same products, 1e-12 relative. */
constexpr scalar_t kSameArithmeticTolerance = 1.0e-12;

/**
 * (S3): the quaternion SQP's iterations from the resting warm start and its convergence tolerance (deltaTol and costTol); the
 * solver stops earlier when it converges.
 */
constexpr size_t kConvergenceIterations = 20;
constexpr scalar_t kConvergenceTolerance = 1.0e-9;
/**
 * (S3): the difference of the charts' gap linearizations against the largest gap, first order (S2, S1w): their
 * first-order terms are 1/2 [b]x and blocks of T_B, C and T_B^-1 (entries of a few at the shipped attitudes) times the
 * gap, times the base sensitivities (of order one): at most this factor of the gap.
 */
constexpr scalar_t kGapProportionality = 100.0;
/**
 * (S3): the factors by which the iterate's gaps are closed (withScaledGaps()): 1 is the iterate itself, 0 a trajectory
 * without gaps. At 1e-4 the bound kGapProportionality times the gap is a hundredth of the iterate's gap, which a
 * difference that did not fall with the gaps would exceed.
 */
constexpr std::array<scalar_t, 4> kGapScales = {1.0, 1.0e-2, 1.0e-4, 0.0};

class MpcSqpStepEquivalence : public MpcEquivalenceTest {};

/** max |m| over the finite entries (an input step is NaN at the last node). */
scalar_t largestFinite(const matrix_t& m) {
  scalar_t largest = 0.0;
  for (Eigen::Index i = 0; i < m.size(); ++i) {
    if (std::isfinite(m.data()[i])) largest = std::max(largest, std::abs(m.data()[i]));
  }
  return largest;
}

/** max |actual - expected| over the finite entries relative to max |expected|; NaN for other shapes or a NaN on one side only. */
scalar_t relativeDifference(const matrix_t& actual, const matrix_t& expected) {
  if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) return std::numeric_limits<scalar_t>::quiet_NaN();
  scalar_t largest = 0.0;
  for (Eigen::Index i = 0; i < actual.size(); ++i) {
    const scalar_t actualEntry = actual.data()[i];
    const scalar_t expectedEntry = expected.data()[i];
    if (std::isnan(actualEntry) != std::isnan(expectedEntry)) return std::numeric_limits<scalar_t>::quiet_NaN();
    if (std::isfinite(expectedEntry)) largest = std::max(largest, std::abs(actualEntry - expectedEntry));
  }
  return largest / std::max(1.0e-12, largestFinite(expected));
}

/** How far one iteration is from another: the largest relative differences of the steps and the gain. Passive data. */
struct IterationDifference {
  scalar_t stateStep = 0.0;
  scalar_t inputStep = 0.0;
  scalar_t firstGain = 0.0;

  scalar_t largest() const { return std::max({stateStep, inputStep, firstGain}); }
};

IterationDifference differenceOf(const RealTimeIteration& actual, const RealTimeIteration& expected) {
  return IterationDifference{.stateStep = relativeDifference(actual.stateStep, expected.stateStep),
                             .inputStep = relativeDifference(actual.inputStep, expected.inputStep),
                             .firstGain = relativeDifference(actual.firstGain, expected.firstGain)};
}

/**
 * The QP's solution of an iteration: its applied steps divided by the line search's step size. The step size is not an
 * equivalence quantity of the QP: it depends on the merit at the post-step iterate, which two retractions place
 * differently at second order in the step (a free fall's step is large), so two iterations of one QP may accept 1 and 1/2.
 */
RealTimeIteration qpSolutionOf(RealTimeIteration iteration) {
  EXPECT_GT(iteration.stepSize, 0.0) << "the line search took no step";
  if (iteration.stepSize > 0.0) {
    iteration.stateStep /= iteration.stepSize;
    iteration.inputStep /= iteration.stepSize;
  }
  return iteration;
}

/** How far the quaternion manifold's QP solution of `sqpCase` is from the Euler chart's. */
IterationDifference quaternionAgainstChart(const MpcEquivalenceFixture& mpc,
                                           std::shared_ptr<const StateManifold> chart,
                                           const EquivalenceReference& reference,
                                           const SqpCase& sqpCase) {
  return differenceOf(qpSolutionOf(mpc.realTimeIteration(mpc.manifoldPtr(), reference, sqpCase)),
                      qpSolutionOf(mpc.realTimeIteration(std::move(chart), reference, sqpCase)));
}

std::string describe(const IterationDifference& difference) {
  return absl::StrCat("dx ", difference.stateStep, ", du ", difference.inputStep, ", K0 ", difference.firstGain);
}

/** The largest rotation [rad] of the shooting gap of a node of the warm start `state` held with `input`. */
scalar_t warmStartRotationGap(const MpcEquivalenceFixture& mpc, const vector_t& state, const vector_t& input) {
  SystemDynamicsBase& dynamics = *mpc.problem().dynamicsPtr;
  const DynamicsDiscretizer discretizer = selectDynamicsDiscretization(mpc.sqpSettings().integratorType);
  vector_t end = discretizer(dynamics, /*t=*/0.0, state, input, mpc.nodeTimeStep());
  mpc.manifold().project(end);
  return mpc.manifold().getMaximumRotationAngle(mpc.manifold().difference(state, end));
}

/** `state` with its base orientation's ZYX angles offset by `offset`. */
vector_t withAttitudeOffset(const MpcEquivalenceFixture& mpc, const vector_t& state, const vector3_t& offset) {
  vector_t eulerState = mpc.eulerStateOf(state);
  eulerState.segment<3>(static_cast<Eigen::Index>(mpc.model().getBaseOrientationStartIndex())) += offset;
  return mpc.quaternionStateOf(eulerState);
}

/** Where the base orientation's quaternion and, on the whole body, the body angular velocity sit in the ambient state. */
struct OrientationRows {
  Eigen::Index quaternion = 0;
  /** -1 on the centroidal state. */
  Eigen::Index angularVelocity = -1;
  /** The Euler (and tangent) rows of the angles and the rates; the rates -1 on the centroidal state. */
  Eigen::Index eulerAngles = 0;
  Eigen::Index eulerRates = -1;
};

OrientationRows orientationRowsOf(const MpcEquivalenceFixture& mpc) {
  OrientationRows rows;
  Eigen::Index physical = 0;
  Eigen::Index euler = 0;
  for (const validation::EulerStateSegment& segment : mpc.eulerLayout().segments) {
    const Eigen::Index dim = static_cast<Eigen::Index>(segment.dim);
    switch (segment.kind) {
      case validation::EulerStateSegmentKind::kEuclidean:
        physical += dim;
        break;
      case validation::EulerStateSegmentKind::kEulerZyxAngles:
        rows.quaternion = physical;
        rows.eulerAngles = euler;
        physical += 4;
        break;
      case validation::EulerStateSegmentKind::kEulerZyxRates:
        rows.angularVelocity = physical;
        rows.eulerRates = euler;
        physical += 3;
        break;
    }
    euler += dim;
  }
  return rows;
}

/** The largest angular velocity [rad/s] of the shooting gap of a node of the warm start `state` held with `input`. */
scalar_t warmStartAngularVelocityGap(const MpcEquivalenceFixture& mpc, const vector_t& state, const vector_t& input) {
  const OrientationRows rows = orientationRowsOf(mpc);
  if (rows.angularVelocity < 0) return 0.0;
  SystemDynamicsBase& dynamics = *mpc.problem().dynamicsPtr;
  const DynamicsDiscretizer discretizer = selectDynamicsDiscretization(mpc.sqpSettings().integratorType);
  vector_t end = discretizer(dynamics, /*t=*/0.0, state, input, mpc.nodeTimeStep());
  mpc.manifold().project(end);
  return mpc.manifold().difference(state, end).segment<3>(rows.eulerRates).cwiseAbs().maxCoeff();
}

/** C(Theta, r) = d(T_B(Theta) r) / dTheta by central differences of eulerZyxRateToLocalAngularVelocityMatrix (kRateMapStep). */
matrix3_t rateMapDerivative(const vector3_t& angles, const vector3_t& rates) {
  matrix3_t derivative;
  for (Eigen::Index column = 0; column < 3; ++column) {
    vector3_t up = angles;
    vector3_t down = angles;
    up(column) += kRateMapStep;
    down(column) -= kRateMapStep;
    derivative.col(column) =
        (eulerZyxRateToLocalAngularVelocityMatrix(up) * rates - eulerZyxRateToLocalAngularVelocityMatrix(down) * rates) /
        (2.0 * kRateMapStep);
  }
  return derivative;
}

/** A warm start of (S1w): every flow interval of the SQP's grid rolled out with the MPC's node map, its end kicked. Passive data. */
struct KickedWarmStart {
  /** What the solver reads its nodes from. */
  PrimalSolution warmStart;
  /** The state at each node of the grid. */
  vector_array_t nodes;
  /** The largest rotation [rad] of a shooting gap, and the largest error of a gap's angular velocity against -kick [rad/s]. */
  scalar_t largestRotationGap = 0.0;
  scalar_t largestKickError = 0.0;
};

/**
 * The warm start that rolls `input` out from `start` over the SQP's node grid of `reference` with the MPC's own node map
 * (the RK4 step and the projection), adding `kick` to the body angular velocity at the end of every flow interval (an
 * event node's state is the one before it, as the jump map leaves it). Every shooting gap of the warm start is then -kick
 * in the angular velocity's rows and nothing else: no rotation, no other row. The primal solution has a sample at every
 * time the solver reads a node's state (the end of each interval; SqpSolver's initialization interpolates there), and two
 * past the horizon.
 */
KickedWarmStart kickedWarmStart(const MpcEquivalenceFixture& mpc,
                                const EquivalenceReference& reference,
                                const vector_t& start,
                                const vector_t& input,
                                const vector3_t& kick) {
  const OrientationRows rows = orientationRowsOf(mpc);
  const std::vector<AnnotatedTime> grid = mpc.sqpNodeGrid(reference);
  SystemDynamicsBase& dynamics = *mpc.problem().dynamicsPtr;
  const DynamicsDiscretizer discretizer = selectDynamicsDiscretization(mpc.sqpSettings().integratorType);
  KickedWarmStart kicked;
  PrimalSolution& warmStart = kicked.warmStart;
  vector_t state = start;
  warmStart.timeTrajectory_.push_back(getIntervalStart(grid.front()));
  warmStart.stateTrajectory_.push_back(state);
  warmStart.inputTrajectory_.push_back(input);
  kicked.nodes.push_back(state);
  for (size_t i = 0; i + 1 < grid.size(); ++i) {
    if (grid[i].event == AnnotatedTime::Event::PreEvent) {
      kicked.nodes.push_back(state);
      continue;
    }
    const scalar_t dt = getIntervalDuration(grid[i], grid[i + 1]);
    vector_t end = discretizer(dynamics, getIntervalStart(grid[i]), state, input, dt);
    mpc.manifold().project(end);
    vector_t next = end;
    next.segment<3>(rows.angularVelocity) += kick;
    const vector_t gap = mpc.manifold().difference(next, end);
    kicked.largestRotationGap = std::max(kicked.largestRotationGap, mpc.manifold().getMaximumRotationAngle(gap));
    kicked.largestKickError = std::max(kicked.largestKickError, (gap.segment<3>(rows.eulerRates) + kick).cwiseAbs().maxCoeff());
    state = next;
    warmStart.timeTrajectory_.push_back(getIntervalEnd(grid[i + 1]));
    warmStart.stateTrajectory_.push_back(state);
    warmStart.inputTrajectory_.push_back(input);
    kicked.nodes.push_back(state);
  }
  for (int past = 1; past <= 2; ++past) {
    warmStart.timeTrajectory_.push_back(warmStart.timeTrajectory_.back() + mpc.nodeTimeStep());
    warmStart.stateTrajectory_.push_back(state);
    warmStart.inputTrajectory_.push_back(input);
  }
  return kicked;
}

TEST_P(MpcSqpStepEquivalence, WithoutARotationInTheShootingGapsTheQuaternionIterationIsTheEulerOne) {
  MpcEquivalenceFixture& mpc = this->mpc();
  const EquivalenceReference reference = mpc.walkingReference();
  const std::shared_ptr<const StateManifold> chart = mpc.eulerChartPtr();
  const SqpCase resting = mpc.restingWarmStart();
  // The resting warm start, and the same tilted to a small attitude, held with its input corrected to exert no moment on
  // the base: the weight carried, nothing turning the base.
  for (const vector3_t& attitude : {vector3_t(0.0, 0.0, 0.0), vector3_t(0.4, -0.03, 0.05)}) {
    SCOPED_TRACE(absl::StrCat("warm start attitude ", attitude(0), " ", attitude(1), " ", attitude(2)));
    const std::string attitudeName = attitude.isZero() ? "level" : "tilted";
    const vector_t warmStart = withAttitudeOffset(mpc, resting.warmStartState, attitude);
    SqpCase sqpCase{.initState = warmStart,
                    .warmStartState = warmStart,
                    .warmStartInput = mpc.torqueFreeInput(mpc.eulerStateOf(warmStart), resting.warmStartInput)};
    ASSERT_LE(warmStartRotationGap(mpc, sqpCase.warmStartState, sqpCase.warmStartInput), kNoRotation);
    // On the whole body (S1w): nor an angular velocity, which would change the chart's gap sensitivity.
    ASSERT_LE(warmStartAngularVelocityGap(mpc, sqpCase.warmStartState, sqpCase.warmStartInput), kNoAngularVelocity);
    const IterationDifference atTheWarmStart = quaternionAgainstChart(mpc, chart, reference, sqpCase);
    EXPECT_LE(atTheWarmStart.largest(), kSameQpTolerance) << describe(atTheWarmStart);
    RecordProperty(absl::StrCat("torque_free_relative_difference_", attitudeName), absl::StrCat(atTheWarmStart.largest()));

    // Tier 2: initial states off the warm start by eps in roll and pitch.
    std::vector<scalar_t> differences;
    for (const scalar_t eps : {0.02, 0.01, 0.005}) {
      sqpCase.initState = withAttitudeOffset(mpc, sqpCase.warmStartState, vector3_t(0.0, 0.6 * eps, 0.8 * eps));
      differences.push_back(quaternionAgainstChart(mpc, chart, reference, sqpCase).largest());
    }
    for (size_t i = 0; i + 1 < differences.size(); ++i) {
      // Only a difference above tier 1 measures the O(eps^2) term.
      if (differences[i] <= kTierOneTolerance) continue;
      const scalar_t ratio = differences[i] / differences[i + 1];
      EXPECT_GE(ratio, kTierTwoRatioLow) << differences[i] << " at eps, " << differences[i + 1] << " at eps / 2";
      EXPECT_LE(ratio, kTierTwoRatioHigh) << differences[i] << " at eps, " << differences[i + 1] << " at eps / 2";
    }
    RecordProperty(absl::StrCat("tier_two_difference_at_eps_0.02_", attitudeName), absl::StrCat(differences[0]));
  }
}

TEST_P(MpcSqpStepEquivalence, TheIterationsDifferAtFirstOrderInTheRotationOfTheShootingGaps) {
  MpcEquivalenceFixture& mpc = this->mpc();
  const EquivalenceReference reference = mpc.walkingReference();
  const std::shared_ptr<const StateManifold> chart = mpc.eulerChartPtr();
  const SqpCase resting = mpc.restingWarmStart();
  // The resting warm start, the moment its input exerts on the base scaled by s (the torque-free input plus s times the
  // difference): the base's turn per interval scales with s, the rest of the warm start stays.
  const vector_t torqueFree = mpc.torqueFreeInput(mpc.eulerStateOf(resting.warmStartState), resting.warmStartInput);
  std::vector<scalar_t> gaps;
  std::vector<scalar_t> differences;
  for (const scalar_t scale : {1.0, 0.5, 0.25}) {
    SqpCase sqpCase = resting;
    sqpCase.warmStartInput = torqueFree + scale * (resting.warmStartInput - torqueFree);
    gaps.push_back(warmStartRotationGap(mpc, sqpCase.warmStartState, sqpCase.warmStartInput));
    differences.push_back(quaternionAgainstChart(mpc, chart, reference, sqpCase).largest());
    RecordProperty(absl::StrCat("scale_", scale, "_rotation_gap"), absl::StrCat(gaps.back()));
    RecordProperty(absl::StrCat("scale_", scale, "_relative_difference_from_the_chart"), absl::StrCat(differences.back()));
  }
  ASSERT_GT(gaps[0], 1.0e3 * kNoRotation) << "the resting warm start turns the base: the case this test is about";
  for (size_t i = 0; i + 1 < gaps.size(); ++i) {
    EXPECT_NEAR(gaps[i] / gaps[i + 1], 2.0, 0.1) << "the gap's rotation is linear in the moment";
    // Only a difference above tier 1 measures the gaps' effect.
    if (differences[i] <= kTierOneTolerance) continue;
    EXPECT_GE(differences[i] / differences[i + 1], kFirstOrderRatio) << differences[i] << " and " << differences[i + 1];
  }
}

TEST_P(MpcSqpStepEquivalence, OnTheWholeBodyAnAngularVelocityGapChangesTheGapSensitivityByTheCouplingBlock) {
  MpcEquivalenceFixture& mpc = this->mpc();
  if (!mpc.isWholeBody()) GTEST_SKIP() << "the centroidal state has no angular velocity";
  const OrientationRows rows = orientationRowsOf(mpc);
  const std::shared_ptr<const StateManifold> chart = mpc.eulerChartPtr();
  SystemDynamicsBase& dynamics = *mpc.problem().dynamicsPtr;
  const DynamicsSensitivityDiscretizer sensitivity = selectDynamicsSensitivityDiscretization(mpc.sqpSettings().integratorType);
  const Eigen::Index tangentDim = static_cast<Eigen::Index>(mpc.manifold().getTangentDim());
  std::mt19937 generator(/*sd=*/20261011);
  for (int sample = 0; sample < 3; ++sample) {
    SCOPED_TRACE(sample);
    const vector_t eulerState = mpc.randomEulerState(EulerSamplingBounds{}, generator);
    const vector_t input = mpc.randomInput(/*maxAbsJointEntry=*/10.0, generator);
    const vector_t state = mpc.quaternionStateOf(eulerState);
    const VectorFunctionLinearApproximation interval = sensitivity(dynamics, /*t=*/0.0, state, input, mpc.nodeTimeStep());
    vector_t end = interval.f;
    mpc.manifold().project(end);
    std::vector<scalar_t> blockSizes;
    // A body-y-and-z gap (yaw and pitch rates), halved twice; then a body-x gap (the roll rate alone), which C ignores.
    for (const vector3_t& gap :
         {vector3_t(0.0, 0.06, 0.08), vector3_t(0.0, 0.03, 0.04), vector3_t(0.0, 0.015, 0.02), vector3_t(0.1, 0.0, 0.0)}) {
      SCOPED_TRACE(absl::StrCat("gap ", gap(0), " ", gap(1), " ", gap(2)));
      vector_t next = end;
      next.segment<3>(rows.angularVelocity) -= gap;  // w_F - w_next = gap, the same quaternion
      VectorFunctionLinearApproximation onQuaternions = interval;
      mpc.manifold().pushForwardDynamics(state, next, onQuaternions);
      VectorFunctionLinearApproximation onTheChart = interval;
      chart->pushForwardDynamics(state, next, onTheChart);

      // The gap's value: the same in both charts, the angular velocity's rows alone.
      ASSERT_LE(mpc.manifold().getMaximumRotationAngle(onQuaternions.f), kNoRotation);
      EXPECT_TRUE(isCloseToScale(onTheChart.f, onQuaternions.f, kSameArithmeticTolerance, /*floor=*/1.0)) << "the gap's value";
      EXPECT_LE((onQuaternions.f.segment<3>(rows.eulerRates) - gap).cwiseAbs().maxCoeff(), kSameArithmeticTolerance);

      // The chart's sensitivity: the identity but for C(Theta, dTheta_next/dt - dTheta_F/dt) T_B^-1.
      const vector_t endEuler = mpc.eulerStateOf(end);
      const vector_t nextEuler = mpc.eulerStateOf(next);
      const matrix_t endToNext = mpc.physicalTangentJacobian(nextEuler) * mpc.physicalTangentJacobian(endEuler).inverse();
      const vector3_t angles = endEuler.segment<3>(rows.eulerAngles);
      const matrix3_t rateMap = eulerZyxRateToLocalAngularVelocityMatrix(angles);
      const vector3_t rateDifference = nextEuler.segment<3>(rows.eulerRates) - endEuler.segment<3>(rows.eulerRates);
      const matrix3_t expectedBlock = rateMapDerivative(angles, rateDifference) * rateMap.inverse();
      matrix_t identityElsewhere = endToNext - matrix_t::Identity(tangentDim, tangentDim);
      const matrix3_t block = identityElsewhere.block<3, 3>(rows.eulerRates, rows.eulerAngles);
      identityElsewhere.block<3, 3>(rows.eulerRates, rows.eulerAngles).setZero();
      EXPECT_LE(identityElsewhere.cwiseAbs().maxCoeff(), kSameArithmeticTolerance) << "the identity outside the coupling block";
      EXPECT_LE((block - expectedBlock).cwiseAbs().maxCoeff(), kCouplingBlockTolerance) << "C(Theta, dTheta_next - dTheta_F) T_B^-1";
      blockSizes.push_back(block.cwiseAbs().maxCoeff());

      // The QP data: the chart's sensitivities are that times the quaternion manifold's (Jr^-1(0) = I there).
      EXPECT_TRUE(isCloseToScale(onTheChart.dfdx, endToNext * onQuaternions.dfdx, kSameArithmeticTolerance, /*floor=*/1.0)) << "A";
      EXPECT_TRUE(isCloseToScale(onTheChart.dfdu, endToNext * onQuaternions.dfdu, kSameArithmeticTolerance, /*floor=*/1.0)) << "B";
    }
    // Linear in the gap; nothing for a roll-rate gap.
    ASSERT_EQ(blockSizes.size(), 4);
    EXPECT_GT(blockSizes[0], 1.0e-3) << "a yaw- and pitch-rate gap couples the rotation's columns into the rates' rows";
    EXPECT_NEAR(blockSizes[0] / blockSizes[1], 2.0, 1.0e-6);
    EXPECT_NEAR(blockSizes[1] / blockSizes[2], 2.0, 1.0e-6);
    EXPECT_LE(blockSizes[3], kCouplingBlockTolerance) << "a roll-rate gap: C does not read the roll rate";
  }
}

TEST_P(MpcSqpStepEquivalence, OnTheWholeBodyTheIterationsDifferAtFirstOrderInAnAngularVelocityGapWithoutRotation) {
  MpcEquivalenceFixture& mpc = this->mpc();
  if (!mpc.isWholeBody()) GTEST_SKIP() << "the centroidal state has no angular velocity";
  const EquivalenceReference reference = mpc.walkingReference();
  const std::shared_ptr<const StateManifold> chart = mpc.eulerChartPtr();
  const SqpCase resting = mpc.restingWarmStart();
  const vector_t input = mpc.torqueFreeInput(mpc.eulerStateOf(resting.warmStartState), resting.warmStartInput);
  const OrientationRows rows = orientationRowsOf(mpc);
  // The warm start is built on sqpNodeGrid(): it must be the grid the solver discretizes into.
  const std::vector<AnnotatedTime> grid = mpc.sqpNodeGrid(reference);
  {
    const KickedWarmStart unkicked = kickedWarmStart(mpc, reference, resting.warmStartState, input, vector3_t::Zero());
    const PrimalSolution solution = mpc.sqpSolution(mpc.manifoldPtr(), reference, resting.warmStartState, unkicked.warmStart,
                                                    SqpRunOptions{.iterations = 1, .convergenceTolerance = std::nullopt});
    ASSERT_EQ(solution.timeTrajectory_.size(), grid.size());
    for (size_t k = 0; k < grid.size(); ++k) ASSERT_EQ(solution.timeTrajectory_[k], grid[k].time) << k;
  }
  std::vector<scalar_t> differences;
  for (const scalar_t scale : {1.0, 0.5, 0.25, 0.0}) {
    SCOPED_TRACE(absl::StrCat("kick scale ", scale));
    const KickedWarmStart kicked =
        kickedWarmStart(mpc, reference, resting.warmStartState, input, scale * kAngularVelocityKick * vector3_t(0.0, 0.0, 1.0));
    ASSERT_LE(kicked.largestRotationGap, kNoRotation) << "a kicked gap carries no rotation";
    ASSERT_LE(kicked.largestKickError, kSameArithmeticTolerance) << "every gap is the kick";
    for (const vector_t& node : kicked.nodes) {
      // Within the Euler chart's domain: away from gimbal lock and from a yaw of +-pi.
      const vector3_t angles = mpc.eulerStateOf(node).segment<3>(rows.eulerAngles);
      ASSERT_LT(std::abs(angles(1)), 1.2);
      ASSERT_LT(std::abs(angles(0)), M_PI - 0.1);
    }
    const RealTimeIteration quaternion =
        qpSolutionOf(mpc.realTimeIteration(mpc.manifoldPtr(), reference, resting.warmStartState, kicked.warmStart, kicked.nodes, input));
    const RealTimeIteration onTheChart =
        qpSolutionOf(mpc.realTimeIteration(chart, reference, resting.warmStartState, kicked.warmStart, kicked.nodes, input));
    const IterationDifference difference = differenceOf(quaternion, onTheChart);
    RecordProperty(absl::StrCat("kick_", scale, "_relative_difference"), absl::StrCat(difference.largest()));
    differences.push_back(difference.largest());
  }
  ASSERT_EQ(differences.size(), 4);
  // Without the kick the warm start is a rollout: no gap at all, and the two iterations are one (S1).
  EXPECT_LE(differences[3], kTierOneTolerance);
  EXPECT_GT(differences[0], kTierOneTolerance) << "the kick's coupling block shows in the iteration";
  for (size_t i = 0; i + 2 < differences.size(); ++i) {
    if (differences[i] <= kTierOneTolerance) continue;
    EXPECT_GE(differences[i] / differences[i + 1], kFirstOrderRatio) << differences[i] << " and " << differences[i + 1];
  }
}

/**
 * Along a trajectory on the SQP's grid: the largest rotation [rad] and angular velocity [rad/s] of its shooting gaps, the
 * interval that carries the largest of them, and the largest difference of the two charts' linearizations of a gap,
 * max |A_chart - A_quaternion| and the same of B, the QPs' dynamics data. Passive data.
 */
struct GapComparison {
  scalar_t rotation = 0.0;
  scalar_t angularVelocity = 0.0;
  /** The node that starts the interval with the largest gap (rotation or angular velocity); 0 when no interval has a gap. */
  size_t largestGapNode = 0;
  scalar_t sensitivityDifference = 0.0;

  scalar_t largestGap() const { return std::max(rotation, angularVelocity); }
};

/**
 * The gaps of the intervals of `states` and `inputs` (one per node of `grid`) and the two charts' linearizations of each:
 * a flow interval's from the MPC's own RK4 sensitivities, an event node's from its identity jump map, Phi(x) = x with
 * dPhi/dx = I, whose gap's linearization still depends on the chart where the jump gap is not zero (Jr^-1(b) against
 * P_e(x_e,next) P_e(x_e)^-1, both I + O(|b|)).
 */
GapComparison compareGaps(const MpcEquivalenceFixture& mpc,
                          const StateManifold& chart,
                          const std::vector<AnnotatedTime>& grid,
                          const vector_array_t& states,
                          const vector_array_t& inputs) {
  const OrientationRows rows = orientationRowsOf(mpc);
  SystemDynamicsBase& dynamics = *mpc.problem().dynamicsPtr;
  const DynamicsSensitivityDiscretizer sensitivity = selectDynamicsSensitivityDiscretization(mpc.sqpSettings().integratorType);
  GapComparison comparison;
  EXPECT_EQ(states.size(), grid.size());
  for (size_t i = 0; i + 1 < std::min(grid.size(), states.size()); ++i) {
    const bool jump = grid[i].event == AnnotatedTime::Event::PreEvent;
    VectorFunctionLinearApproximation onQuaternions;
    VectorFunctionLinearApproximation onTheChart;
    if (jump) {
      VectorFunctionLinearApproximation identityJump;
      identityJump.f = states[i];
      identityJump.dfdx = matrix_t::Identity(states[i].size(), states[i].size());
      onQuaternions = identityJump;
      mpc.manifold().pushForwardJump(states[i], states[i + 1], onQuaternions);
      onTheChart = identityJump;
      chart.pushForwardJump(states[i], states[i + 1], onTheChart);
    } else {
      const VectorFunctionLinearApproximation interval =
          sensitivity(dynamics, getIntervalStart(grid[i]), states[i], inputs[i], getIntervalDuration(grid[i], grid[i + 1]));
      onQuaternions = interval;
      mpc.manifold().pushForwardDynamics(states[i], states[i + 1], onQuaternions);
      onTheChart = interval;
      chart.pushForwardDynamics(states[i], states[i + 1], onTheChart);
    }
    const scalar_t rotation = mpc.manifold().getMaximumRotationAngle(onQuaternions.f);
    const scalar_t angularVelocity = rows.angularVelocity >= 0 ? onQuaternions.f.segment<3>(rows.eulerRates).cwiseAbs().maxCoeff() : 0.0;
    if (std::max(rotation, angularVelocity) > comparison.largestGap()) comparison.largestGapNode = i;
    comparison.rotation = std::max(comparison.rotation, rotation);
    comparison.angularVelocity = std::max(comparison.angularVelocity, angularVelocity);
    comparison.sensitivityDifference =
        std::max(comparison.sensitivityDifference, (onTheChart.dfdx - onQuaternions.dfdx).cwiseAbs().maxCoeff());
    // A jump's dfdu is empty (tangent x 0) in both charts.
    if (!jump) {
      comparison.sensitivityDifference =
          std::max(comparison.sensitivityDifference, (onTheChart.dfdu - onQuaternions.dfdu).cwiseAbs().maxCoeff());
    }
  }
  return comparison;
}

/** max |a - b| of the QP solutions' state steps of two iterations: an absolute difference. */
scalar_t qpStateStepDistance(const RealTimeIteration& a, const RealTimeIteration& b) {
  const matrix_t difference = qpSolutionOf(a).stateStep - qpSolutionOf(b).stateStep;
  return largestFinite(difference);
}

/**
 * The trajectory on the SQP's `grid` through the flows of `states` under `inputs` (one per node) with every gap scaled by
 * `scale`: from the first node of `states`, each interval's flow (the MPC's RK4 step and the projection) retracted by
 * -scale times the gap of the same interval of `states`, so that its gap is that gap times `scale`; an event node's
 * interval, whose jump map is the identity, retracts the event node itself by -scale times its jump gap (which
 * compareGaps() compares as the identity jump's). A scale of 1 gives back `states` to round-off, 0 a trajectory without gaps: the
 * rollout of `inputs` through the events.
 */
vector_array_t withScaledGaps(const MpcEquivalenceFixture& mpc,
                              const std::vector<AnnotatedTime>& grid,
                              const vector_array_t& states,
                              const vector_array_t& inputs,
                              scalar_t scale) {
  SystemDynamicsBase& dynamics = *mpc.problem().dynamicsPtr;
  const DynamicsDiscretizer discretizer = selectDynamicsDiscretization(mpc.sqpSettings().integratorType);
  vector_array_t scaled = {states.front()};
  for (size_t i = 0; i + 1 < std::min(grid.size(), states.size()); ++i) {
    if (grid[i].event == AnnotatedTime::Event::PreEvent) {
      // The identity jump map's gap, the event node against the post-event node, scaled as a flow's gap is.
      const vector_t jumpGap = mpc.manifold().difference(states[i + 1], states[i]);
      vector_t next;
      mpc.manifold().retract(scaled[i], jumpGap, -scale, next);  // difference(next, scaled[i]) = scale * jumpGap
      scaled.push_back(std::move(next));
      continue;
    }
    const scalar_t start = getIntervalStart(grid[i]);
    const scalar_t dt = getIntervalDuration(grid[i], grid[i + 1]);
    const vector_t gap = mpc.manifold().difference(states[i + 1], discretizer(dynamics, start, states[i], inputs[i], dt));
    vector_t flow = discretizer(dynamics, start, scaled[i], inputs[i], dt);
    mpc.manifold().project(flow);
    vector_t next;
    mpc.manifold().retract(flow, gap, -scale, next);  // difference(next, flow) = scale * gap
    scaled.push_back(std::move(next));
  }
  return scaled;
}

TEST_P(MpcSqpStepEquivalence, AsTheGapsCloseTheTwoChartsLinearizationsCoincide) {
  // (S3): the two QPs differ only through the gaps' chart (S2, S1w): the same costs and constraints pulled back through
  // the same E, and gap linearizations that differ in proportion to the gaps. That difference of the QPs' dynamics data,
  // the largest difference of the charts' A and B, is at most kGapProportionality times the largest gap (rotation, and on
  // the whole body angular velocity) at the resting warm start, whose gaps carry rotations, and at the quaternion SQP's
  // iterate after kConvergenceIterations iterations from it, wherever the SQP has got to. Closing the iterate's gaps
  // (withScaledGaps(), by kGapScales) it falls with them, and it vanishes with them, at the trajectory without gaps where
  // (S1) the two QPs are one: the two SQPs have the same fixed points, the trajectories with no gap. The gaps are closed
  // by construction because the SQP need not close them (the file comment); the iterate's gaps are recorded, with the
  // node of the largest. The QP solutions' difference at the iterate is recorded only: the QPs weigh the yaw by no cost
  // (no shipped robot does), so round-off of the chart's arithmetic moves their solutions along it by far more than the
  // charts' first-order difference.
  MpcEquivalenceFixture& mpc = this->mpc();
  const EquivalenceReference reference = mpc.walkingReference();
  const std::shared_ptr<const StateManifold> chart = mpc.eulerChartPtr();
  const SqpCase resting = mpc.restingWarmStart();
  const std::vector<AnnotatedTime> grid = mpc.sqpNodeGrid(reference);
  PrimalSolution warmStart;
  const scalar_t finalTime = reference.initTime + mpc.timeHorizon();
  for (scalar_t time = reference.initTime; time <= finalTime + 2.0 * mpc.nodeTimeStep(); time += mpc.nodeTimeStep()) {
    warmStart.timeTrajectory_.push_back(time);
    warmStart.stateTrajectory_.push_back(resting.warmStartState);
    warmStart.inputTrajectory_.push_back(resting.warmStartInput);
  }
  const GapComparison first = compareGaps(mpc, *chart, grid, vector_array_t(grid.size(), resting.warmStartState),
                                          vector_array_t(grid.size(), resting.warmStartInput));
  ASSERT_GT(first.largestGap(), 1.0e3 * kNoRotation) << "the resting warm start has gaps: the case this test is about";

  const SqpRunOptions iterated{.iterations = kConvergenceIterations, .convergenceTolerance = kConvergenceTolerance};
  const PrimalSolution iterate = mpc.sqpSolution(mpc.manifoldPtr(), reference, resting.initState, warmStart, iterated);
  ASSERT_EQ(iterate.timeTrajectory_.size(), grid.size());
  const GapComparison last = compareGaps(mpc, *chart, grid, iterate.stateTrajectory_, iterate.inputTrajectory_);
  const scalar_t qpDistance = qpStateStepDistance(
      mpc.realTimeIteration(mpc.manifoldPtr(), reference, resting.initState, iterate, iterate.stateTrajectory_, resting.warmStartInput),
      mpc.realTimeIteration(chart, reference, resting.initState, iterate, iterate.stateTrajectory_, resting.warmStartInput));
  RecordProperty("first_gap", absl::StrCat(first.largestGap()));
  RecordProperty("first_sensitivity_difference", absl::StrCat(first.sensitivityDifference));
  RecordProperty("iterate_rotation_gap", absl::StrCat(last.rotation));
  RecordProperty("iterate_angular_velocity_gap", absl::StrCat(last.angularVelocity));
  RecordProperty("iterate_largest_gap_node", absl::StrCat(last.largestGapNode));
  RecordProperty("iterate_largest_gap_time", absl::StrCat(grid[last.largestGapNode].time));
  RecordProperty("iterate_sensitivity_difference", absl::StrCat(last.sensitivityDifference));
  RecordProperty("iterate_qp_distance", absl::StrCat(qpDistance));
  EXPECT_GT(first.sensitivityDifference, kSameArithmeticTolerance) << "the warm start's gaps change the chart's linearization";
  EXPECT_LE(first.sensitivityDifference, kGapProportionality * first.largestGap());
  EXPECT_LE(last.sensitivityDifference, kGapProportionality * last.largestGap() + kSameArithmeticTolerance)
      << "the charts' linearizations differ beyond the gaps' first order";

  for (const scalar_t scale : kGapScales) {
    SCOPED_TRACE(absl::StrCat("the iterate's gaps scaled by ", scale));
    const GapComparison closing = compareGaps(
        mpc, *chart, grid, withScaledGaps(mpc, grid, iterate.stateTrajectory_, iterate.inputTrajectory_, scale), iterate.inputTrajectory_);
    RecordProperty(absl::StrCat("gap_scale_", scale, "_sensitivity_difference"), absl::StrCat(closing.sensitivityDifference));
    ASSERT_LE(std::abs(closing.largestGap() - scale * last.largestGap()), kSameArithmeticTolerance) << "the iterate's gaps, scaled";
    EXPECT_LE(closing.sensitivityDifference, kGapProportionality * closing.largestGap() + kSameArithmeticTolerance)
        << "the charts' linearizations differ beyond the gaps' first order";
  }
}

INSTANTIATE_TEST_SUITE_P(Shipped, MpcSqpStepEquivalence, ::testing::ValuesIn(mpcEquivalenceConfigurationNames()), mpcEquivalenceTestName);

}  // namespace
}  // namespace ocs2::humanoid
