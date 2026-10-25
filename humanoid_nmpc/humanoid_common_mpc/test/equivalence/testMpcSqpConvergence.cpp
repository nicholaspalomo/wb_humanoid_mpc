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
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_data/ProblemMetrics.h"
#include "ocs2_oc/search_strategy/FilterLinesearch.h"
#include "ocs2_sqp/SqpSolverStatus.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetrics.h"
#include "humanoid_mpc_validation/closed_loop/SqpStepDiagnostics.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/MpcEquivalenceFixture.h"

/*
 * The SQP's convergence on each shipped configuration's own tuning (the task
 * file's multiple_shooting settings: its delta_tol, g_min and g_max, and OCS2's
 * costTol), from the resting warm start
 * (MpcEquivalenceFixture::restingWarmStart(): the initial state held with the
 * input that carries the weight on both feet) under the walking reference
 * (walkingReference(): the first lift-off, of the right foot, at 0.2 s), read
 * from the solver's own records of its iterations
 * (SqpSolver::getIterationRecords()) and the values of its solution's nodes.
 * The claims, each a test:
 *
 *   (C1) Iterated, the SQP converges: within kIterations iterations its own
 * convergence test ends the solve with METRICS (the merit changes by less than
 * costTol with the constraint violation below g_min) or PRIMAL (the step below
 * deltaTol). Measured: Unitree G1 at iteration 17, Unitree R1 at 24, EngineAI
 * SA01 at 40 (it drifts along the yaw no cost weighs before it settles). (C2)
 * In real time, as the MPC runs it - one iteration per solve, each solve one
 * node after the last, from the last plan's state there and warm-started from
 * that plan by the solver itself (realTimeSolves()), from the iterate of (C1) -
 * every solve of the second half (after the first kRealTimeSolves / 2 have left
 * the iterate of (C1)) takes the full step (EngineAI SA01 refuses one full step
 * in the first half, at t = 0.12 s, above g_max), the violation it leaves stays
 * below g_max in most solves, and the hard swing-foot normal-velocity rows stay
 * held (the median of their largest residual below kHeldSwingResidual, in the
 * rows' units: heldSwingResidualOf()). The warm start is the last plan, node
 * for node (trajectorySpread() keeps an unchanged schedule's plan,
 * lib/ocs2/README.md; a post-event state starts as its pre-event state, the
 *        identity jump), but for the newly initialized end of the horizon,
 * whose violation each shift adds (baselines of 0.004 to 0.13 in the second
 * half, a median of 0.08, on the converging configurations) and the step
 * removes (a median of 3.6e-3 to 3.9e-3 left, at most 2.3e-2; swing residuals
 * with a median of 0.0092 to 0.029 m/s, at most 0.13).
 *
 * KNOWN DEFECT (design README section 6, risk 12) on DRC Atlas and on the
 * Unitree G1 whole-body MPC (kCyclingConfigurations): the Gauss-Newton SQP does
 * not converge but cycles. Its QP's Hessian is the costs' alone, without the
 * hard constraints' curvature lambda' grad^2 g; its merit has no penalty on
 * their violation; and its filter line search accepts any decrease of the merit
 * while the violation stays between g_min and g_max. So each full step lowers
 * the merit and multiplies the violation of the hard rows (about 2.5 times per
 * iteration on Atlas, 2.32 on the G1 whole body), until a full step would pass
 * g_max, is refused, and a reduced step and the full step after it reset the
 * violation by an order of magnitude. For these configurations the tests assert
 * the defect's signature instead: (C1) no convergence within kIterations; in
 * their second half at least kMinimumResets resets, each a full step refused
 * above g_max and a reduced step taken below it; between two resets, a full
 * step that lowers the violation while it raises the merit and then full steps
 * that lower the merit while they raise the violation, each accepted by the
 * rule between g_min and g_max; and at the last iterate the largest equality
 * residual a hard swing-foot normal-velocity row's. (C2) in real time the cycle
 * refuses full steps. On Atlas (kCyclingConfigurationsHeldInRealTime) the
 * real-time solves hold the swing rows all the same and leave a violation below
 * g_max in most solves (medians of 0.056 m/s and 8.0e-3, at most 0.27 m/s
 * and 3.8e-2), but the violation climbs above g_max for runs of solves (the
 * shift leaves 18 of the 20 baselines above it), and each full step refused
 * there, for raising the violation above g_max, is a reset: the reduced step
 * lands below g_max (2 in the second half). On the G1 whole body every baseline
 * (0.43 to 4.0) and every full step stay above g_max, where the line search
 * accepts only a step that lowers the violation: every step leaves a violation
 * above g_max (a median of 0.55, at least 0.12), the full steps that would
 * raise it are refused (6 of 20), and the swing rows are not held (a median
 * residual of 2.3 m/s^2, at most 12.6). Before the warm start was fixed
 * (trajectorySpread() handed each post-event interval the next node's input)
 * neither held its swing rows in real time: every full step was taken above
 * g_max and the violation climbed to a plateau, with a median of 0.12 on Atlas
 * (a swing residual of 0.59 m/s) and 2.9 on the G1 whole body (8.3 m/s^2). When
 * a fix lands (the remedies of the design README, each behind a default-off
 * switch), the configuration leaves kCyclingConfigurations and its branches
 * flip to the convergence assertions above.
 *
 * The tests use the shipped settings unchanged, but for one solver thread (the
 * fixture's: the performance sums' bits do not follow the thread schedule) and
 * the iteration limit; they read no recorded data.
 */

namespace ocs2::humanoid {
namespace {

/** The configurations whose SQP cycles: the known defect of the file comment.
 */
constexpr std::array<absl::string_view, 2> kCyclingConfigurations = {
    "drc_atlas", "unitree_g1_wb"};
/** Of those, the ones whose real-time solves hold the swing rows all the same
 * ((C2) of the defect in the file comment). */
constexpr std::array<absl::string_view, 1>
    kCyclingConfigurationsHeldInRealTime = {"drc_atlas"};

/** (C1): the iteration limit; the slowest configuration that converges,
 * EngineAI SA01, needs 40. */
constexpr size_t kIterations = 100;
/** (C1): the resets a cycling configuration repeats in the second half of the
 * iterations (Atlas and the G1 whole body have 13). */
constexpr size_t kMinimumResets = 3;
/** (C2): the real-time solves; the second half of them is held to the claims.
 */
constexpr size_t kRealTimeSolves = 40;
/**
 * [m/s] (C2): the median over the solves of the largest residual of the hard
 * swing rows of a plan that holds them, in the centroidal MPCs' units: their
 * row is a velocity. The converging configurations' median is 0.0092 to 0.029
 * m/s, their largest 0.13, which the median passes over; Atlas's is 0.056 m/s,
 * at most 0.27. The whole-body row is an acceleration, so its bound is this one
 * times its velocity gain (heldSwingResidualOf(): 1 m/s^2 on the G1), against
 * which the G1 whole body's median is 2.3 m/s^2.
 */
constexpr scalar_t kHeldSwingResidual = 0.1;

class MpcSqpConvergence : public MpcEquivalenceTest {};

bool cycles(absl::string_view configuration) {
  return std::find(kCyclingConfigurations.begin(), kCyclingConfigurations.end(),
                   configuration) != kCyclingConfigurations.end();
}

bool holdsTheSwingRowsInRealTime(absl::string_view configuration) {
  return std::find(kCyclingConfigurationsHeldInRealTime.begin(),
                   kCyclingConfigurationsHeldInRealTime.end(),
                   configuration) != kCyclingConfigurationsHeldInRealTime.end();
}

/**
 * (C2): kHeldSwingResidual in the units of the configuration's swing rows. The
 * centroidal row is a velocity, v_z - zdot_ref + k_p (z - z_ref) [m/s]; the
 * whole body's is k_v (v_z - zdot_ref) + k_a (a_z - zddot_ref) + k_p (z -
 * z_ref) [m/s^2] (WBMpcPreComputation::normalAccelerationConstraintConfig()),
 * where a velocity error weighs k_v times as much.
 */
scalar_t heldSwingResidualOf(const MpcEquivalenceFixture& mpc) {
  return mpc.isWholeBody()
             ? mpc.modelSettings()
                       .footConstraintConfig.linearVelocityErrorGain_z *
                   kHeldSwingResidual
             : kHeldSwingResidual;
}

/** The resting warm start held over the horizon, as the S3 test of
 * testMpcSqpStepEquivalence.cpp builds it. */
PrimalSolution heldWarmStart(const MpcEquivalenceFixture& mpc,
                             const EquivalenceReference& reference,
                             const SqpCase& resting) {
  PrimalSolution warmStart;
  const scalar_t finalTime = reference.initTime + mpc.timeHorizon();
  for (scalar_t time = reference.initTime;
       time <= finalTime + 2.0 * mpc.nodeTimeStep();
       time += mpc.nodeTimeStep()) {
    warmStart.timeTrajectory_.push_back(time);
    warmStart.stateTrajectory_.push_back(resting.warmStartState);
    warmStart.inputTrajectory_.push_back(resting.warmStartInput);
  }
  return warmStart;
}

/** The SQP of (C1): up to kIterations iterations from the resting warm start
 * under the walking reference. */
SqpRun iteratedRun(const MpcEquivalenceFixture& mpc,
                   const EquivalenceReference& reference) {
  const SqpCase resting = mpc.restingWarmStart();
  return mpc.sqpRun(mpc.manifoldPtr(), reference, resting.initState,
                    heldWarmStart(mpc, reference, resting),
                    SqpRunOptions{.iterations = kIterations,
                                  .convergenceTolerance = std::nullopt});
}

/** The problem's hard swing-foot normal-velocity terms, which every shipped
 * configuration lists (a test checks it). */
std::vector<size_t> normalVelocityTerms(const MpcEquivalenceFixture& mpc) {
  return validation::normalVelocityTermIndices(
      *mpc.problem().equalityConstraintPtr, mpc.modelSettings().contactNames);
}

/** Whether `iteration` is a reset: its full step refused above `gMax`, and a
 * reduced step taken below it. */
bool isReset(const sqp::IterationRecord& iteration, scalar_t gMax) {
  const sqp::StepInfo& step = iteration.stepInfo;
  return step.stepSize < 1.0 &&
         step.fullStepType == FilterLinesearch::StepType::CONSTRAINT &&
         step.fullStepConstraintViolation > gMax &&
         step.totalConstraintViolationAfterStep <= gMax;
}

/** The largest |value| of any state-input equality term over the intermediate
 * nodes of `metrics`, and the term's index. */
struct LargestResidual {
  scalar_t value = 0.0;
  size_t term = 0;
};

LargestResidual largestEqualityResidual(const ProblemMetrics& metrics) {
  LargestResidual largest;
  for (const Metrics& node : metrics.intermediates) {
    for (size_t term = 0; term < node.stateInputEqConstraint.size(); ++term) {
      const vector_t& value = node.stateInputEqConstraint[term];
      if (value.size() == 0) continue;
      const scalar_t residual = value.cwiseAbs().maxCoeff();
      if (residual > largest.value)
        largest = LargestResidual{.value = residual, .term = term};
    }
  }
  return largest;
}

/** Checks the cycle of a cycling configuration's (C1) iterations (the file
 * comment); records its resets and growth. */
void expectCycle(const std::vector<sqp::IterationRecord>& iterations,
                 scalar_t gMax) {
  std::vector<size_t> resets;
  for (size_t k = iterations.size() / 2; k < iterations.size(); ++k) {
    if (isReset(iterations[k], gMax)) resets.push_back(k);
  }
  ::testing::Test::RecordProperty("resets_in_the_second_half",
                                  absl::StrCat(resets.size()));
  ASSERT_GE(resets.size(), kMinimumResets)
      << "the cycle's resets: a full step refused above g_max, a reduced step "
         "taken";
  scalar_t logGrowth = 0.0;
  size_t growthSteps = 0;
  for (size_t r = 0; r + 1 < resets.size(); ++r) {
    SCOPED_TRACE(absl::StrCat("the period from the reset at iteration ",
                              resets[r], " to the one at ", resets[r + 1]));
    ASSERT_GE(resets[r + 1] - resets[r], size_t{3})
        << "a reset, the step that drops the violation, and a growth";
    for (size_t k = resets[r] + 1; k < resets[r + 1]; ++k) {
      SCOPED_TRACE(absl::StrCat("iteration ", k));
      const sqp::IterationRecord& iteration = iterations[k];
      const sqp::StepInfo& step = iteration.stepInfo;
      EXPECT_EQ(step.stepSize, 1.0) << "a full step";
      EXPECT_EQ(step.stepType, FilterLinesearch::StepType::DUAL)
          << "accepted between g_min and g_max";
      const scalar_t meritChange =
          step.performanceAfterStep.merit - iteration.baselinePerformance.merit;
      const scalar_t violationRatio = step.totalConstraintViolationAfterStep /
                                      iteration.baselineConstraintViolation;
      if (k == resets[r] + 1) {
        // Accepted for the violation it removes, against the merit it adds.
        EXPECT_LT(violationRatio, 1.0)
            << "the step after a reset drops the violation";
        EXPECT_GT(meritChange, 0.0) << "and raises the merit";
      } else {
        // Accepted for the merit it removes, against the violation it adds.
        EXPECT_LT(meritChange, 0.0) << "a growth step lowers the merit";
        EXPECT_GT(violationRatio, 1.0) << "and raises the violation";
        logGrowth += std::log(violationRatio);
        ++growthSteps;
      }
    }
  }
  ASSERT_GT(growthSteps, size_t{0});
  ::testing::Test::RecordProperty(
      "violation_growth_per_iteration",
      absl::StrCat(std::exp(logGrowth / static_cast<scalar_t>(growthSteps))));
  ::testing::Test::RecordProperty("cycle_period_iterations",
                                  absl::StrCat(resets[1] - resets[0]));
}

/**
 * Returns every solve of `solves` as `<initial time>:<the baseline's
 * violation>:<step size>:<the full step's verdict, SqpStepKind>:<the full
 * step's violation>:<the violation after the step>:<the largest residual of the
 * swing rows `swingRows`, 0 without one>`, the solves separated by spaces: the
 * record (C2)'s numbers are re-derived from.
 */
std::string solveRecord(const std::vector<RealTimeSolve>& solves,
                        const std::vector<size_t>& swingRows) {
  std::vector<std::string> record;
  for (const RealTimeSolve& solve : solves) {
    const validation::SqpStepSample sample =
        validation::sqpStepSampleOf(solve.iteration, solve.metrics, swingRows);
    record.push_back(absl::StrFormat(
        "%.3f:%.3g:%.3g:%d:%.3g:%.3g:%.3g", solve.initTime,
        solve.iteration.baselineConstraintViolation, sample.stepSize,
        static_cast<int>(sample.fullStepKind),
        sample.fullStepConstraintViolation, sample.constraintViolation,
        sample.swingNormalVelocityResidual.value_or(0.0)));
  }
  return absl::StrJoin(record, " ");
}

TEST_P(MpcSqpConvergence,
       IteratedFromTheRestingWarmStartTheSqpConvergesOrCyclesAsKnown) {
  MpcEquivalenceFixture& mpc = this->mpc();
  const scalar_t gMax = mpc.sqpSettings().g_max;
  const SqpRun run = iteratedRun(mpc, mpc.walkingReference());
  ASSERT_FALSE(run.iterations.empty());
  const sqp::Convergence convergence = run.iterations.back().convergence;
  RecordProperty("iterations", absl::StrCat(run.iterations.size()));
  RecordProperty("convergence", sqp::toString(convergence));
  RecordProperty(
      "final_constraint_violation",
      absl::StrCat(
          run.iterations.back().stepInfo.totalConstraintViolationAfterStep));
  if (!cycles(GetParam())) {
    EXPECT_TRUE(convergence == sqp::Convergence::METRICS ||
                convergence == sqp::Convergence::PRIMAL)
        << "the SQP's own convergence test: " << sqp::toString(convergence)
        << " after " << run.iterations.size() << " iterations";
    return;
  }

  // KNOWN DEFECT (file comment): when a fix lands, this configuration leaves
  // kCyclingConfigurations and is held to the convergence above.
  EXPECT_EQ(convergence, sqp::Convergence::ITERATIONS)
      << "the cycle does not converge";
  for (const sqp::IterationRecord& iteration : run.iterations) {
    ASSERT_EQ(iteration.convergence == sqp::Convergence::FALSE,
              &iteration != &run.iterations.back());
  }
  expectCycle(run.iterations, gMax);
  // The violation that cycles is the hard swing rows'.
  const sqp::IterationRecord& last = run.iterations.back();
  EXPECT_GT(last.stepInfo.performanceAfterStep.equalityConstraintsSSE,
            last.stepInfo.performanceAfterStep.dynamicsViolationSSE)
      << "the equality constraints carry the violation, not the shooting gaps";
  const std::vector<size_t> swingRows = normalVelocityTerms(mpc);
  const LargestResidual largest = largestEqualityResidual(run.metrics);
  RecordProperty("largest_equality_residual", absl::StrCat(largest.value));
  EXPECT_NE(std::find(swingRows.begin(), swingRows.end(), largest.term),
            swingRows.end())
      << "the largest equality residual at the last iterate is a swing "
         "normal-velocity row's";
}

/**
 * The largest rotation [rad] of a base orientation in the first step of the
 * quaternion SQP of (C1): of the step from the resting warm start to the
 * iterate after one iteration, at any node.
 */
scalar_t largestFirstStepRotation(const MpcEquivalenceFixture& mpc,
                                  const EquivalenceReference& reference) {
  const SqpCase resting = mpc.restingWarmStart();
  const SqpRun first = mpc.sqpRun(
      mpc.manifoldPtr(), reference, resting.initState,
      heldWarmStart(mpc, reference, resting),
      SqpRunOptions{.iterations = 1, .convergenceTolerance = std::nullopt});
  scalar_t largest = 0.0;
  for (const vector_t& state : first.solution.stateTrajectory_) {
    largest = std::max(
        largest, mpc.manifold().getMaximumRotationAngle(
                     mpc.manifold().difference(resting.warmStartState, state)));
  }
  return largest;
}

/**
 * (C1) on DRC Atlas, in the Euler chart: the chart's SQP (the same problem on
 * EulerChartStateManifold, the Euler formulation's iteration with today's
 * terms) cycles as the quaternion manifold's does, and the two runs' merits
 * agree per iteration over the first kChartMeritIterations iterations to second
 * order in the steps: relative to the merit, to
 * kChartMeritAgreementPerSquaredRotation times phi_1^2, phi_1 the largest base
 * rotation of the first step (largestFirstStepRotation()). Their violations
 * agree only to about 1 %, so they are not compared. Run on Atlas only, the
 * configuration whose cycle the design README states in both charts (section 6,
 * risk 12); the others are not iterated twice.
 *
 * Why second order. Both runs solve one problem: the fixture copies the MPC's
 * problem and replaces only its state manifold, and no term reads the manifold,
 * so every residual, the foot tilt and orientation residuals included, is the
 * same function of the state on both. Where the shooting gaps carry no rotation
 * the two QPs are one (testMpcSqpStepEquivalence (S1), to 1e-10); the resting
 * warm start's gaps carry b = 3.9e-5 rad, whose first-order effect (S2) is not
 * measurable here: from the torque-free warm start (b = 5e-19) the merits
 * differ by the same amounts to three digits (4.35e-5 against 4.39e-5 at
 * iteration 0, 1.25e-4 at iteration 1). What separates the runs from the first
 * step on is the retraction: the quaternion manifold takes the step phi of a
 * base orientation as R Exp(phi), the chart as R(Theta + T_B(Theta)^-1 phi) = R
 * Exp(phi) Exp(eps), and for ZYX angles eps = (1/2) (-phi_y phi_z, phi_x phi_z,
 * -phi_x phi_y) + O(|phi|^3) at a level base (the Baker-Campbell-Hausdorff
 * terms of Exp(psi e_z) Exp(theta e_y) Exp(phi e_x)), |eps| <= 0.48 |phi|^2 for
 * any yaw and |roll|, |pitch| <= 0.5 (a scratch check over 2000 random
 * attitudes and directions, the ratio 4.00 when phi halves). The merits then
 * differ by the merit's change along eps, second order in the step, and every
 * later iteration starts from iterates that differ by it. Measured, as the
 * largest difference (at iteration 1) and K, that difference over
 * phi_1^2: 1.25e-4 for phi_1 = 0.111 rad, K = 0.010; with the swing apex above
 * the lower end of the swing instead of the higher one
 * (SwingTrajectoryPlanner::makeSwingSplines(), 0.04 m lower with Atlas's
 * touch_down_height_offset), phi_1 = 0.079 rad and 8.1e-5, K = 0.013. The
 * higher apex doubled the violation after the first step (0.11 to 0.22) and
 * with it the first step; the inscribed friction pyramid of the basis
 * generators added the rest (1.06e-4 with the former, circumscribed one), so a
 * fixed relative bound (1e-4, against 6.7e-5 measured before both) stopped
 * holding while K stayed. A disagreement of first order in the step, such as a
 * residual or a sensitivity that differed between the charts, would be of order
 * 1 / phi_1 = 9 times larger against phi_1^2 and fail the bound.
 */
TEST_P(MpcSqpConvergence,
       OnAtlasTheEulerChartCyclesAsTheQuaternionManifoldDoes) {
  constexpr size_t kChartMeritIterations = 20;
  // Four times the largest K measured (0.013, the comment above).
  constexpr scalar_t kChartMeritAgreementPerSquaredRotation = 0.05;
  if (GetParam() != "drc_atlas") return;
  MpcEquivalenceFixture& mpc = this->mpc();
  const scalar_t gMax = mpc.sqpSettings().g_max;
  const EquivalenceReference reference = mpc.walkingReference();
  const SqpRun quaternion = iteratedRun(mpc, reference);
  const SqpCase resting = mpc.restingWarmStart();
  const SqpRun chart =
      mpc.sqpRun(mpc.eulerChartPtr(), reference, resting.initState,
                 heldWarmStart(mpc, reference, resting),
                 SqpRunOptions{.iterations = kIterations,
                               .convergenceTolerance = std::nullopt});
  ASSERT_FALSE(chart.iterations.empty());
  EXPECT_EQ(chart.iterations.back().convergence, sqp::Convergence::ITERATIONS)
      << "the chart's SQP does not converge either";
  expectCycle(chart.iterations, gMax);
  ASSERT_GE(quaternion.iterations.size(), kChartMeritIterations);
  ASSERT_GE(chart.iterations.size(), kChartMeritIterations);
  const scalar_t firstStepRotation = largestFirstStepRotation(mpc, reference);
  RecordProperty("largest_first_step_rotation",
                 absl::StrCat(firstStepRotation));
  const scalar_t meritAgreement = kChartMeritAgreementPerSquaredRotation *
                                  firstStepRotation * firstStepRotation;
  scalar_t largestMeritDifference = 0.0;
  std::vector<std::string> meritDifferences;
  for (size_t k = 0; k < kChartMeritIterations; ++k) {
    SCOPED_TRACE(absl::StrCat("iteration ", k));
    const scalar_t onQuaternions =
        quaternion.iterations[k].stepInfo.performanceAfterStep.merit;
    const scalar_t onTheChart =
        chart.iterations[k].stepInfo.performanceAfterStep.merit;
    const scalar_t relative = std::abs(onTheChart - onQuaternions) /
                              std::max(std::abs(onQuaternions), 1.0);
    largestMeritDifference = std::max(largestMeritDifference, relative);
    meritDifferences.push_back(absl::StrFormat("%.3g", relative));
    EXPECT_LE(relative, meritAgreement)
        << onQuaternions << " on the quaternions, " << onTheChart
        << " on the chart, a first step of " << firstStepRotation << " rad";
  }
  RecordProperty("largest_relative_merit_difference",
                 absl::StrCat(largestMeritDifference));
  RecordProperty(
      "largest_relative_merit_difference_per_squared_first_step_rotation",
      absl::StrCat(largestMeritDifference /
                   (firstStepRotation * firstStepRotation)));
  RecordProperty("relative_merit_differences",
                 absl::StrJoin(meritDifferences, " "));
}

TEST_P(
    MpcSqpConvergence,
    InRealTimeEverySolveTakesTheFullStepAndHoldsTheSwingRowsOrCyclesAsKnown) {
  MpcEquivalenceFixture& mpc = this->mpc();
  const scalar_t gMax = mpc.sqpSettings().g_max;
  const EquivalenceReference reference = mpc.walkingReference();
  const std::vector<size_t> swingRows = normalVelocityTerms(mpc);
  ASSERT_FALSE(swingRows.empty()) << "every shipped configuration lists the "
                                     "hard normal_velocity constraint";
  const std::vector<RealTimeSolve> solves = mpc.realTimeSolves(
      reference, iteratedRun(mpc, reference).solution, kRealTimeSolves);
  ASSERT_EQ(solves.size(), kRealTimeSolves);
  RecordProperty("solves", solveRecord(solves, swingRows));

  // The second half, after the first solves have left the iterate of (C1).
  std::vector<scalar_t> violations;
  std::vector<scalar_t> swingResiduals;
  size_t fullSteps = 0;
  size_t constraintVerdicts = 0;
  size_t resets = 0;
  for (size_t n = kRealTimeSolves / 2; n < kRealTimeSolves; ++n) {
    const validation::SqpStepSample sample = validation::sqpStepSampleOf(
        solves[n].iteration, solves[n].metrics, swingRows);
    violations.push_back(sample.constraintViolation);
    if (sample.stepSize == 1.0) ++fullSteps;
    if (sample.fullStepKind == validation::SqpStepKind::kConstraint)
      ++constraintVerdicts;
    if (isReset(solves[n].iteration, gMax)) ++resets;
    // No residual where no swing row is active in the horizon.
    swingResiduals.push_back(sample.swingNormalVelocityResidual.value_or(0.0));
  }
  std::sort(violations.begin(), violations.end());
  std::sort(swingResiduals.begin(), swingResiduals.end());
  const scalar_t medianViolation = violations[violations.size() / 2];
  const scalar_t medianSwingResidual =
      swingResiduals[swingResiduals.size() / 2];
  RecordProperty("median_constraint_violation", absl::StrCat(medianViolation));
  RecordProperty("largest_constraint_violation",
                 absl::StrCat(violations.back()));
  RecordProperty("median_swing_normal_velocity_residual",
                 absl::StrCat(medianSwingResidual));
  RecordProperty("largest_swing_normal_velocity_residual",
                 absl::StrCat(swingResiduals.back()));
  RecordProperty("full_steps", absl::StrCat(fullSteps));
  RecordProperty("constraint_verdicts", absl::StrCat(constraintVerdicts));
  RecordProperty("resets", absl::StrCat(resets));
  const scalar_t heldSwingResidual = heldSwingResidualOf(mpc);
  if (!cycles(GetParam())) {
    EXPECT_EQ(fullSteps, violations.size())
        << "every solve takes the full step";
    EXPECT_LT(medianViolation, gMax)
        << "the step removes the shift's violation in most solves";
    EXPECT_LT(medianSwingResidual, heldSwingResidual)
        << "the swing rows are held";
    return;
  }

  // KNOWN DEFECT (file comment): when a fix lands, this configuration leaves
  // kCyclingConfigurations and is held to the claims above.
  EXPECT_LT(fullSteps, violations.size())
      << "the cycle refuses full steps in real time too";
  if (holdsTheSwingRowsInRealTime(GetParam())) {
    EXPECT_EQ(resets, violations.size() - fullSteps)
        << "each a reset: refused above g_max, a reduced step taken below it";
    EXPECT_LT(medianViolation, gMax)
        << "the step removes the shift's violation in most solves";
    EXPECT_LT(medianSwingResidual, heldSwingResidual)
        << "the swing rows are held";
    return;
  }
  EXPECT_GT(violations.front(), gMax)
      << "every step leaves a violation above g_max";
  EXPECT_EQ(constraintVerdicts, violations.size())
      << "every full step judged above g_max, where only a decrease of the "
         "violation is accepted";
  EXPECT_GT(medianSwingResidual, heldSwingResidual)
      << "the swing rows are not held";
}

INSTANTIATE_TEST_SUITE_P(
    Shipped,
    MpcSqpConvergence,
    ::testing::ValuesIn(mpcEquivalenceConfigurationNames()),
    mpcEquivalenceTestName);

}  // namespace
}  // namespace ocs2::humanoid
