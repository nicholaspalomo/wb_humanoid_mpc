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

#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/RunningProblemFingerprint.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_sqp/SqpSettings.h"
#include "ocs2_sqp/SqpSolver.h"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodyLiveTuningFixture.h"
#include "humanoid_wb_mpc/WBMpcPreComputation.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsFootCost.h"
#include "humanoid_wb_mpc/cost/JointTorqueCostCppAd.h"

namespace ocs2::humanoid::live_tuning_test {
namespace {

// [rad] how far inside its upper limit every joint is put for the limit barrier, whose band (joint_limits.delta) is wider.
constexpr scalar_t kInsideJointLimit = 0.02;
// [s] Times in the ramps of the swings of WholeBodySolverStack's schedule (walkingSchedule(): the right foot over
// [0.1, 0.45], the left over [0.55, 0.9]), which the times of evaluationPoints() miss: early in each swing, in the first
// half of the swing spline and in the rise of the swing pitch, and late in the right one, in the fall of the pitch.
// LINT.IfChange(early_swing_times)
constexpr std::array<scalar_t, 3> kSwingRampTimes = {0.15, 0.6, 0.42};
// LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodySolverStack.cpp:walking_schedule)
// The constraint values the penalties of the soft constraints are evaluated at.
constexpr std::array<scalar_t, 12> kPenaltySweep = {-1.0, -0.1, -0.01, -0.001, 0.0005, 0.002, 0.005, 0.02, 0.05, 0.2, 2.0, 20.0};

/** Appends the entries of `values`, column by column, to `fingerprint`. */
void append(const vector_t& values, std::vector<scalar_t>& fingerprint) {
  fingerprint.insert(fingerprint.end(), values.data(), values.data() + values.size());
}

void append(const matrix_t& values, std::vector<scalar_t>& fingerprint) {
  fingerprint.insert(fingerprint.end(), values.data(), values.data() + values.size());
}

/** The names of the terms of `collection`, sorted, so that the fingerprint does not follow a map's order. */
template <typename Term>
std::vector<std::string> sortedTermNames(const Collection<Term>& collection) {
  std::vector<std::string> names;
  for (const std::pair<const std::string, size_t>& entry : collection.getTermNameMap()) names.push_back(entry.first);
  std::sort(names.begin(), names.end());
  return names;
}

/** The point of evaluationPoints() with every joint kInsideJointLimit inside its upper limit, in stance. */
EvaluationPoint pointInsideTheJointLimitBarrier(WBMpcInterface& interface) {
  EvaluationPoint point;
  point.time = -0.4;
  point.state = interface.getInitialState();
  point.input = vector_t::Zero(static_cast<Eigen::Index>(interface.getMpcRobotModel().getInputDim()));
  const Eigen::Index numJoints = static_cast<Eigen::Index>(interface.getMpcRobotModel().getJointDim());
  const pinocchio::Model& model = interface.getPinocchioInterface().getModel();
  const vector_t upper = model.upperPositionLimit.tail(numJoints);
  interface.getMpcRobotModel().setJointAngles(point.state, vector_t(upper.array() - kInsideJointLimit));
  return point;
}

void appendStateInputTerms(StateInputCostCollection& collection,
                           const EvaluationPoint& point,
                           const TargetTrajectories& target,
                           const PreComputation& preComputation,
                           std::vector<scalar_t>& fingerprint) {
  for (const std::string& name : sortedTermNames(collection)) {
    const StateInputCost& term = collection.get(name);
    fingerprint.push_back(term.getValue(point.time, point.state, point.input, target, preComputation));
    const ScalarFunctionQuadraticApproximation approximation =
        term.getQuadraticApproximation(point.time, point.state, point.input, target, preComputation);
    append(approximation.dfdx, fingerprint);
    append(approximation.dfdu, fingerprint);
  }
}

void appendStateTerms(StateCostCollection& collection,
                      const EvaluationPoint& point,
                      const TargetTrajectories& target,
                      const PreComputation& preComputation,
                      std::vector<scalar_t>& fingerprint) {
  for (const std::string& name : sortedTermNames(collection)) {
    const StateCost& term = collection.get(name);
    fingerprint.push_back(term.getValue(point.time, point.state, target, preComputation));
    append(term.getQuadraticApproximation(point.time, point.state, target, preComputation).dfdx, fingerprint);
  }
}

void appendEqualityConstraints(StateInputConstraintCollection& collection,
                               const EvaluationPoint& point,
                               const PreComputation& preComputation,
                               std::vector<scalar_t>& fingerprint) {
  for (const std::string& name : sortedTermNames(collection)) {
    const StateInputConstraint& term = collection.get(name);
    const VectorFunctionLinearApproximation linearization =
        term.getLinearApproximation(point.time, point.state, point.input, preComputation);
    append(linearization.f, fingerprint);
    append(linearization.dfdx, fingerprint);
    append(linearization.dfdu, fingerprint);
  }
}

void appendSwingFootCoefficients(const PreComputation& preComputation, std::vector<scalar_t>& fingerprint) {
  for (const EndEffectorDynamicsLinearAccConstraint::Config& config :
       cast<WBMpcPreComputation>(preComputation).getEeNormalAccelerationConstraintConfigs()) {
    append(config.b, fingerprint);
    append(config.Ax, fingerprint);
    append(config.Av, fingerprint);
    append(config.Aa, fingerprint);
  }
}

/** The penalty of `soft`, whose constraint has `numConstraints` rows, over kPenaltySweep. */
void appendPenaltySweep(const MultidimensionalPenalty& penalty, size_t numConstraints, scalar_t time, std::vector<scalar_t>& fingerprint) {
  for (const scalar_t value : kPenaltySweep) {
    fingerprint.push_back(penalty.getValue(time, vector_t::Constant(static_cast<Eigen::Index>(numConstraints), value)));
  }
}

void appendPenalties(OptimalControlProblem& problem, scalar_t time, std::vector<scalar_t>& fingerprint) {
  for (const std::string& name : sortedTermNames(*problem.softConstraintPtr)) {
    StateInputSoftConstraint& soft = problem.softConstraintPtr->get<StateInputSoftConstraint>(name);
    appendPenaltySweep(soft.getPenalty(), soft.get().getNumConstraints(time), time, fingerprint);
  }
  for (const std::string& name : sortedTermNames(*problem.stateSoftConstraintPtr)) {
    // The joint limits' barrier is not a soft constraint's penalty: the point inside it evaluates it.
    if (name == kJointLimitsTerm) continue;
    StateSoftConstraint& soft = problem.stateSoftConstraintPtr->get<StateSoftConstraint>(name);
    appendPenaltySweep(soft.getPenalty(), soft.get().getNumConstraints(time), time, fingerprint);
  }
}

void appendCppAdParameters(OptimalControlProblem& problem,
                           const WBMpcInterface& interface,
                           const TargetTrajectories& target,
                           std::vector<scalar_t>& fingerprint) {
  const scalar_t swingTime = 0.5 * (kSwingStart + kSwingEnd);
  for (const std::string& footName : interface.modelSettings().contactNames) {
    const std::string term = EndEffectorDynamicsFootCost::termName(footName);
    if (!carriesTerm(*problem.costPtr, term)) continue;
    append(problem.costPtr->get<EndEffectorDynamicsFootCost>(term).getParameters(swingTime, target, *problem.preComputationPtr),
           fingerprint);
  }
  if (carriesTerm(*problem.costPtr, JointTorqueCostCppAd::kTermName)) {
    append(problem.costPtr->get<JointTorqueCostCppAd>(JointTorqueCostCppAd::kTermName)
               .getParameters(swingTime, target, *problem.preComputationPtr),
           fingerprint);
  }
}

/** The settings the SQP solver reads at every solve that a reload writes: what they do is how the solver iterates. */
void appendSolverSettings(SqpSolver& solver, std::vector<scalar_t>& fingerprint) {
  const sqp::Settings& settings = solver.getSettings();
  fingerprint.insert(fingerprint.end(), {static_cast<scalar_t>(settings.sqpIteration), settings.deltaTol, settings.g_max, settings.g_min});
}

/** The swing trajectory planner's configuration as a reload left it, read back. */
void appendPlannerConfig(const SwingTrajectoryPlanner& planner, std::vector<scalar_t>& fingerprint) {
  const SwingTrajectoryPlanner::Config& config = planner.getConfig();
  fingerprint.insert(fingerprint.end(), {config.liftOffVelocity, config.touchDownVelocity, config.swingHeight, config.swingTimeScale,
                                         config.touchDownHeightOffset, config.impactProximityFactorLiftOffVelocity,
                                         config.impactProximityFactorTouchDownVelocity, config.impactProximityFactorMidPointValue,
                                         config.swingPitchAngle, config.swingPitchRiseFraction, config.swingPitchFallFraction});
}

}  // namespace

std::vector<scalar_t> runningProblemEffects(WholeBodySolverStack& stack) {
  // The swing trajectories the terms and the swing-foot coefficients read are planned before every solve.
  stack.replanReferences();
  WBMpcInterface& interface = stack.interface();
  SqpSolver& solver = *stack.mpc().getSolverPtr();
  OptimalControlProblem& problem = solver.getOcpDefinitions().front();
  const TargetTrajectories& target = interface.getReferenceManagerPtr()->getTargetTrajectories();
  std::vector<EvaluationPoint> points = evaluationPoints(interface, /*count=*/5);
  for (size_t i = 0; i < kSwingRampTimes.size(); ++i) {
    EvaluationPoint inRamp = points[i];
    inRamp.time = kSwingRampTimes[i];
    points.push_back(std::move(inRamp));
  }
  points.push_back(pointInsideTheJointLimitBarrier(interface));

  std::vector<scalar_t> fingerprint;
  for (const EvaluationPoint& point : points) {
    problem.preComputationPtr->request(Request::Cost + Request::Constraint + Request::SoftConstraint, point.time, point.state, point.input);
    const PreComputation& preComputation = *problem.preComputationPtr;
    appendStateInputTerms(*problem.costPtr, point, target, preComputation, fingerprint);
    appendStateInputTerms(*problem.softConstraintPtr, point, target, preComputation, fingerprint);
    appendStateTerms(*problem.stateCostPtr, point, target, preComputation, fingerprint);
    appendStateTerms(*problem.stateSoftConstraintPtr, point, target, preComputation, fingerprint);
    appendStateTerms(*problem.finalCostPtr, point, target, preComputation, fingerprint);
    appendEqualityConstraints(*problem.equalityConstraintPtr, point, preComputation, fingerprint);
    appendSwingFootCoefficients(preComputation, fingerprint);
  }
  appendPenalties(problem, /*time=*/0.0, fingerprint);
  appendSolverSettings(solver, fingerprint);
  return fingerprint;
}

std::vector<scalar_t> runningProblemFingerprint(WholeBodySolverStack& stack) {
  std::vector<scalar_t> fingerprint = runningProblemEffects(stack);
  WBMpcInterface& interface = stack.interface();
  OptimalControlProblem& problem = stack.mpc().getSolverPtr()->getOcpDefinitions().front();
  appendCppAdParameters(problem, interface, interface.getReferenceManagerPtr()->getTargetTrajectories(), fingerprint);
  appendPlannerConfig(*interface.getSwitchedModelReferenceManagerPtr()->getSwingTrajectoryPlanner(), fingerprint);
  return fingerprint;
}

}  // namespace ocs2::humanoid::live_tuning_test
