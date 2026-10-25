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

/*
 * SqpSolver::setLiveSettings(), the narrow live retuning of the SQP solver (the parameter updater of humanoid_nmpc):
 * the four live settings reach both the settings and the filter line search, which copies g_max and g_min at
 * construction, so that a solve after setLiveSettings() is the solve of a solver constructed with those values, bit for
 * bit; and no other setting changes. And SqpSolver::getIterationRecords(), what the line search those settings steer
 * did at each iteration of the last solve: the performance it started from, the full step's verdict, the step it took
 * and the convergence test after it.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"

#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/dynamics/SystemDynamicsBase.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_oc/oc_data/PerformanceIndex.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>
#include <ocs2_oc/search_strategy/FilterLinesearch.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include "ocs2_sqp/SqpSolver.h"

namespace ocs2 {
namespace {

constexpr size_t kStateDim = 2;
constexpr size_t kInputDim = 1;
constexpr scalar_t kInitTime = 0.0;
constexpr scalar_t kFinalTime = 1.0;

/** A torque-driven pendulum, x = [theta, omega], u = [tau]: theta' = omega, omega' = -9.81 sin(theta) - 0.1 omega + tau. */
class PendulumDynamics final : public SystemDynamicsBase {
 public:
  PendulumDynamics() = default;
  ~PendulumDynamics() override = default;
  PendulumDynamics* absl_nonnull clone() const override { return new PendulumDynamics(*this); }

  vector_t computeFlowMap(scalar_t /*t*/, const vector_t& x, const vector_t& u, const PreComputation& /*preComp*/) override {
    vector_t dxdt(kStateDim);
    dxdt(0) = x(1);
    dxdt(1) = -9.81 * std::sin(x(0)) - 0.1 * x(1) + u(0);
    return dxdt;
  }

  VectorFunctionLinearApproximation linearApproximation(scalar_t t,
                                                        const vector_t& x,
                                                        const vector_t& u,
                                                        const PreComputation& preComp) override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = computeFlowMap(t, x, u, preComp);
    approximation.dfdx.setZero(kStateDim, kStateDim);
    approximation.dfdx(0, 1) = 1.0;
    approximation.dfdx(1, 0) = -9.81 * std::cos(x(0));
    approximation.dfdx(1, 1) = -0.1;
    approximation.dfdu.setZero(kStateDim, kInputDim);
    approximation.dfdu(1, 0) = 1.0;
    return approximation;
  }

  vector_t computeJumpMap(scalar_t /*t*/, const vector_t& x, const PreComputation& /*preComp*/) override { return x; }

  VectorFunctionLinearApproximation jumpMapLinearApproximation(scalar_t t, const vector_t& x, const PreComputation& preComp) override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = computeJumpMap(t, x, preComp);
    approximation.dfdx.setIdentity(kStateDim, kStateDim);
    approximation.dfdu.setZero(kStateDim, 0);
    return approximation;
  }

 private:
  PendulumDynamics(const PendulumDynamics& other) = default;
};

/** The pendulum swung up from hanging: far from the cold start's trajectory, so the first iterates violate the dynamics. */
OptimalControlProblem makeProblem() {
  OptimalControlProblem problem;
  problem.dynamicsPtr = std::make_unique<PendulumDynamics>();
  matrix_t Q = matrix_t::Identity(kStateDim, kStateDim);
  Q(0, 0) = 10.0;
  const matrix_t R = 0.01 * matrix_t::Identity(kInputDim, kInputDim);
  problem.costPtr->add("trackingCost", std::make_unique<QuadraticStateInputCost>(Q, R));
  problem.finalCostPtr->add("finalCost", std::make_unique<QuadraticStateCost>(100.0 * Q));
  return problem;
}

/** The settings both solvers start from: few iterations, so that the line search's choices show in the result. */
sqp::Settings baseSettings() {
  sqp::Settings settings;
  settings.dt = 0.05;
  settings.sqpIteration = 2;
  settings.enableLogging = false;
  settings.printSolverStatus = false;
  settings.printSolverStatistics = false;
  settings.printLinesearch = false;
  settings.nThreads = 1;
  return settings;
}

/** The live settings the tests switch to: a tight filter, so that steps the defaults accept are refused. */
SqpSolver::LiveSettings liveSettings() {
  return SqpSolver::LiveSettings{.sqpIteration = 4, .deltaTol = 1e-9, .gMax = 1e-3, .gMin = 1e-9};
}

/** `settings` with `live` written in. */
sqp::Settings withLiveSettings(sqp::Settings settings, const SqpSolver::LiveSettings& live) {
  settings.sqpIteration = live.sqpIteration;
  settings.deltaTol = live.deltaTol;
  settings.g_max = live.gMax;
  settings.g_min = live.gMin;
  return settings;
}

/** One cold solve of `solver` from the hanging pendulum towards the upright target; returns its primal solution. */
PrimalSolution solveOnce(SqpSolver& solver) {
  vector_t initState(kStateDim);
  initState << 3.0, 0.0;
  const TargetTrajectories target({kInitTime}, {vector_t::Zero(kStateDim)}, {vector_t::Zero(kInputDim)});
  solver.setReferenceManager(std::make_shared<ReferenceManager>(target));
  solver.run(kInitTime, initState, /*initMode=*/0, kFinalTime);
  PrimalSolution solution;
  solver.getPrimalSolution(kFinalTime, &solution);
  return solution;
}

/** Whether `a` and `b` are the same trajectories, bit for bit. */
bool sameTrajectories(const PrimalSolution& a, const PrimalSolution& b) {
  if (a.timeTrajectory_ != b.timeTrajectory_ || a.stateTrajectory_.size() != b.stateTrajectory_.size() ||
      a.inputTrajectory_.size() != b.inputTrajectory_.size()) {
    return false;
  }
  for (size_t k = 0; k < a.stateTrajectory_.size(); ++k) {
    if (a.stateTrajectory_[k] != b.stateTrajectory_[k]) return false;
  }
  for (size_t k = 0; k < a.inputTrajectory_.size(); ++k) {
    if (a.inputTrajectory_[k] != b.inputTrajectory_[k]) return false;
  }
  return true;
}

TEST(SqpLiveSettings, ASolveAfterSetLiveSettingsIsTheSolveOfASolverConstructedWithThem) {
  const OptimalControlProblem problem = makeProblem();
  const DefaultInitializer initializer(kInputDim);

  SqpSolver retuned(baseSettings(), problem, initializer);
  retuned.setLiveSettings(liveSettings());
  SqpSolver constructed(withLiveSettings(baseSettings(), liveSettings()), problem, initializer);
  const PrimalSolution retunedSolution = solveOnce(retuned);
  EXPECT_TRUE(sameTrajectories(retunedSolution, solveOnce(constructed))) << "the retuned solver solves differently";
  EXPECT_EQ(retuned.getNumIterations(), constructed.getNumIterations());

  // Positive control: writing the settings alone, as the parameter updater used to, leaves the line search on the
  // constructed g_max and g_min, and that solve differs - the filter is what setLiveSettings() also reaches.
  SqpSolver settingsOnly(baseSettings(), problem, initializer);
  settingsOnly.getSettings() = withLiveSettings(settingsOnly.getSettings(), liveSettings());
  EXPECT_FALSE(sameTrajectories(retunedSolution, solveOnce(settingsOnly)))
      << "the filter's g_max and g_min change nothing here, so the test cannot tell whether they reach the line search";
}

TEST(SqpLiveSettings, TheLiveSettingsAreWrittenAndNoOtherSettingChanges) {
  const OptimalControlProblem problem = makeProblem();
  const DefaultInitializer initializer(kInputDim);
  sqp::Settings settings = baseSettings();
  // Every other field off its default, so that a write of a default would show.
  settings.costTol = 3e-3;
  settings.alpha_decay = 0.7;
  settings.alpha_min = 2e-3;
  settings.armijoFactor = 3e-4;
  settings.gamma_c = 4e-6;
  settings.useFeedbackPolicy = false;
  settings.integratorType = SensitivityIntegratorType::RK4;
  settings.logSize = 17;
  settings.logFilePath = "/nonexistent/sqp_live_settings/";
  settings.threadPriority = 7;
  SqpSolver solver(settings, problem, initializer);
  const sqp::Settings before = solver.getSettings();
  solver.setLiveSettings(liveSettings());
  const sqp::Settings& after = solver.getSettings();

  EXPECT_EQ(after.sqpIteration, liveSettings().sqpIteration);
  EXPECT_EQ(after.deltaTol, liveSettings().deltaTol);
  EXPECT_EQ(after.g_max, liveSettings().gMax);
  EXPECT_EQ(after.g_min, liveSettings().gMin);

  EXPECT_EQ(after.costTol, before.costTol);
  EXPECT_EQ(after.alpha_decay, before.alpha_decay);
  EXPECT_EQ(after.alpha_min, before.alpha_min);
  EXPECT_EQ(after.armijoFactor, before.armijoFactor);
  EXPECT_EQ(after.gamma_c, before.gamma_c);
  EXPECT_EQ(after.useFeedbackPolicy, before.useFeedbackPolicy);
  EXPECT_EQ(after.dt, before.dt);
  EXPECT_EQ(after.integratorType, before.integratorType);
  EXPECT_EQ(after.projectStateInputEqualityConstraints, before.projectStateInputEqualityConstraints);
  EXPECT_EQ(after.printSolverStatus, before.printSolverStatus);
  EXPECT_EQ(after.printSolverStatistics, before.printSolverStatistics);
  EXPECT_EQ(after.printLinesearch, before.printLinesearch);
  EXPECT_EQ(after.enableLogging, before.enableLogging);
  EXPECT_EQ(after.logSize, before.logSize);
  EXPECT_EQ(after.logFilePath, before.logFilePath);
  EXPECT_EQ(after.nThreads, before.nThreads);
  EXPECT_EQ(after.threadPriority, before.threadPriority);
  EXPECT_EQ(after.hpipmSettings.iter_max, before.hpipmSettings.iter_max);
  EXPECT_EQ(after.hpipmSettings.warm_start, before.hpipmSettings.warm_start);
}

TEST(SqpLiveSettings, TheConstructedValuesAgainLeaveTheSolveAsItWas) {
  // setLiveSettings() with the values the solver was constructed with is no change at all.
  const OptimalControlProblem problem = makeProblem();
  const DefaultInitializer initializer(kInputDim);
  const sqp::Settings settings = baseSettings();
  SqpSolver rewritten(settings, problem, initializer);
  rewritten.setLiveSettings(SqpSolver::LiveSettings{
      .sqpIteration = settings.sqpIteration, .deltaTol = settings.deltaTol, .gMax = settings.g_max, .gMin = settings.g_min});
  SqpSolver untouched(settings, problem, initializer);
  EXPECT_TRUE(sameTrajectories(solveOnce(rewritten), solveOnce(untouched)));
}

TEST(SqpIterationRecords, EveryIterationOfTheLastSolveIsRecordedAsTheLineSearchEndedIt) {
  // The tight filter of the live settings, so that the line search refuses full steps: both of its branches recorded.
  const OptimalControlProblem problem = makeProblem();
  const DefaultInitializer initializer(kInputDim);
  const sqp::Settings settings = withLiveSettings(baseSettings(), liveSettings());
  SqpSolver solver(settings, problem, initializer);
  EXPECT_TRUE(solver.getIterationRecords().empty()) << "no solve yet";
  solveOnce(solver);

  const std::vector<sqp::IterationRecord>& records = solver.getIterationRecords();
  const std::vector<PerformanceIndex>& log = solver.getIterationsLog();
  ASSERT_EQ(records.size(), log.size());
  ASSERT_EQ(records.size(), solver.getNumIterations()) << "one record per iteration of the solve";
  size_t refusedFullSteps = 0;
  for (size_t k = 0; k < records.size(); ++k) {
    SCOPED_TRACE(k);
    const sqp::IterationRecord& record = records[k];
    const sqp::StepInfo& step = record.stepInfo;
    // The performance after the step is the iterations log's, bit for bit.
    EXPECT_EQ(step.performanceAfterStep.merit, log[k].merit);
    EXPECT_EQ(step.performanceAfterStep.dynamicsViolationSSE, log[k].dynamicsViolationSSE);
    EXPECT_EQ(step.totalConstraintViolationAfterStep, FilterLinesearch::totalConstraintViolation(log[k]));
    EXPECT_EQ(record.baselineConstraintViolation, FilterLinesearch::totalConstraintViolation(record.baselinePerformance));
    // The convergence test ends the solve at its last iteration only.
    EXPECT_EQ(record.convergence == sqp::Convergence::FALSE, k + 1 < records.size()) << sqp::toString(record.convergence);
    // The full step is always tried. Taken, it is the step and its verdict; refused, a smaller step or none was taken.
    EXPECT_NE(step.fullStepType, FilterLinesearch::StepType::UNKNOWN);
    if (step.stepSize == 1.0) {
      EXPECT_EQ(step.fullStepType, step.stepType);
      EXPECT_EQ(step.fullStepConstraintViolation, step.totalConstraintViolationAfterStep);
    } else {
      ++refusedFullSteps;
      EXPECT_LT(step.stepSize, 1.0);
      if (step.fullStepType == FilterLinesearch::StepType::CONSTRAINT) {
        EXPECT_GT(step.fullStepConstraintViolation, settings.g_max) << "a CONSTRAINT verdict is the rule above g_max";
      }
    }
  }
  EXPECT_GT(refusedFullSteps, size_t{0}) << "no full step was refused, so the test cannot tell the refused branch is recorded";

  // The next solve's iterations replace the last solve's; a reset forgets them.
  solveOnce(solver);
  EXPECT_EQ(solver.getIterationRecords().size(), solver.getIterationsLog().size());
  solver.reset();
  EXPECT_TRUE(solver.getIterationRecords().empty());
}

TEST(SqpIterationRecords, RecordingChangesNoSolve) {
  // The records are written beside the solve: two solvers, one of whose records are read between its solves, solve
  // the same, bit for bit (the flat parity golden of test_sqp_flat_parity holds the solve itself).
  const OptimalControlProblem problem = makeProblem();
  const DefaultInitializer initializer(kInputDim);
  SqpSolver read(baseSettings(), problem, initializer);
  SqpSolver unread(baseSettings(), problem, initializer);
  const PrimalSolution first = solveOnce(read);
  EXPECT_FALSE(read.getIterationRecords().empty());
  EXPECT_TRUE(sameTrajectories(first, solveOnce(unread)));
  EXPECT_TRUE(sameTrajectories(solveOnce(read), solveOnce(unread)));
}

}  // namespace
}  // namespace ocs2
