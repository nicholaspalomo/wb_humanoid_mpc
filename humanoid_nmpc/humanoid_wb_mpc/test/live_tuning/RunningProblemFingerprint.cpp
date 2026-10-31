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
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_sqp/SqpSettings.h"
#include "ocs2_sqp/SqpSolver.h"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/constraint/FootCollisionCbfConstraint.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodyLiveTuningFixture.h"
#include "humanoid_wb_mpc/WBMpcPreComputation.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsFootCost.h"
#include "humanoid_wb_mpc/cost/JointTorqueCostCppAd.h"

namespace ocs2::humanoid::live_tuning_test {
namespace {

// [rad] how far inside its upper limit every joint is put for the limit
// barrier, whose band (joint_limits.delta) is wider.
constexpr scalar_t kInsideJointLimit = 0.02;
// LINT.IfChange(cbf_band_margin)
// [m^2] The squared margin |p_a - p_b|^2 - (2 r)^2 of the closest pair at
// pointInsideTheFootCollisionCbfBarrier(): gamma * h is 0.001 at the variant's
// gamma of 10, inside the barrier's band (delta 0.05).
constexpr scalar_t kCbfBandMargin = 1.0e-4;
constexpr scalar_t kCbfBandMarginTolerance = 1.0e-6;
// LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodyLiveTuningFixture.cpp:variant_cbf)
// The most Newton steps pointInsideTheFootCollisionCbfBarrier() takes toward
// kCbfBandMargin.
constexpr int kMaxCbfBandSteps = 20;
// [s] Times in the ramps of the swings of WholeBodySolverStack's schedule
// (walkingSchedule(): the right foot over [0.1, 0.45], the left over [0.55,
// 0.9]), which the times of evaluationPoints() miss: early in each swing, in
// the first half of the swing spline and in the rise of the swing pitch, and
// late in the right one, in the fall of the pitch.
// LINT.IfChange(early_swing_times)
constexpr std::array<scalar_t, 3> kSwingRampTimes = {0.15, 0.6, 0.42};
// LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodySolverStack.cpp:walking_schedule)
// The constraint values the penalties of the soft constraints are evaluated at.
constexpr std::array<scalar_t, 12> kPenaltySweep = {
    -1.0,  -0.1, -0.01, -0.001, 0.0005, 0.002,
    0.005, 0.02, 0.05,  0.2,    2.0,    20.0};

/** Appends the entries of `values`, column by column, to `fingerprint`. */
void append(const vector_t& values, std::vector<scalar_t>& fingerprint) {
  fingerprint.insert(fingerprint.end(), values.data(),
                     values.data() + values.size());
}

void append(const matrix_t& values, std::vector<scalar_t>& fingerprint) {
  fingerprint.insert(fingerprint.end(), values.data(),
                     values.data() + values.size());
}

/** The names of the terms of `collection`, sorted, so that the fingerprint does
 * not follow a map's order. */
template <typename Term>
std::vector<std::string> sortedTermNames(const Collection<Term>& collection) {
  std::vector<std::string> names;
  for (const std::pair<const std::string, size_t>& entry :
       collection.getTermNameMap())
    names.push_back(entry.first);
  std::sort(names.begin(), names.end());
  return names;
}

/** The point of evaluationPoints() with every joint kInsideJointLimit inside
 * its upper limit, in stance. */
EvaluationPoint pointInsideTheJointLimitBarrier(WBMpcInterface& interface) {
  EvaluationPoint point;
  point.time = -0.4;
  point.state = interface.getInitialState();
  point.input = vector_t::Zero(
      static_cast<Eigen::Index>(interface.getMpcRobotModel().getInputDim()));
  const Eigen::Index numJoints =
      static_cast<Eigen::Index>(interface.getMpcRobotModel().getJointDim());
  const pinocchio::Model& model = interface.getPinocchioInterface().getModel();
  const vector_t upper = model.upperPositionLimit.tail(numJoints);
  interface.getMpcRobotModel().setJointAngles(
      point.state, vector_t(upper.array() - kInsideJointLimit));
  return point;
}

/**
 * The point at the start of the right foot's swing with the joint angles moved
 * until the closest pair of spheres of foot_collision_cbf has the squared
 * margin h = kCbfBandMargin, or nullopt when the problem has no such soft
 * constraint. The state's joint velocities are zero, so a row is gamma * h: the
 * point depends on the spheres alone, not on the gamma, mu and delta a reload
 * changes, and the shipped tuning puts gamma * h inside the barrier's band,
 * where the barrier is not zero.
 */
std::optional<EvaluationPoint> pointInsideTheFootCollisionCbfBarrier(
    WBMpcInterface& interface, OptimalControlProblem& problem) {
  size_t index = 0;
  if (!problem.softConstraintPtr->getTermIndex(kFootCollisionCbfTerm, index))
    return std::nullopt;
  StateInputSoftConstraint* absl_nullable soft =
      dynamic_cast<StateInputSoftConstraint*>(
          &problem.softConstraintPtr->get(kFootCollisionCbfTerm));
  if (soft == nullptr) return std::nullopt;
  const FootCollisionCbfConstraint* absl_nullable cbf =
      dynamic_cast<const FootCollisionCbfConstraint*>(&soft->get());
  if (cbf == nullptr) return std::nullopt;

  EvaluationPoint point;
  point.time = kSwingRampTimes[0];
  point.state = interface.getInitialState();
  point.input = vector_t::Zero(
      static_cast<Eigen::Index>(interface.getMpcRobotModel().getInputDim()));
  const MpcRobotModelBase<scalar_t>& model = interface.getMpcRobotModel();
  const Eigen::Index jointStart =
      static_cast<Eigen::Index>(model.getJointStartindex());
  const Eigen::Index jointDim = static_cast<Eigen::Index>(model.getJointDim());
  for (int step = 0; step < kMaxCbfBandSteps; ++step) {
    problem.preComputationPtr->request(
        Request::Constraint + Request::SoftConstraint + Request::Approximation,
        point.time, point.state, point.input);
    const VectorFunctionLinearApproximation linear =
        cbf->getLinearApproximation(point.time, point.state, point.input,
                                    *problem.preComputationPtr);
    EXPECT_GT(linear.f.size(), 0)
        << "foot_collision_cbf is inactive at " << point.time
        << " s, in the swing of the right foot";
    if (linear.f.size() == 0) return std::nullopt;
    Eigen::Index row = 0;
    const scalar_t smallestMargin = linear.f.minCoeff(&row) / cbf->getGamma();
    if (std::abs(smallestMargin - kCbfBandMargin) <= kCbfBandMarginTolerance)
      break;
    const vector_t gradient =
        linear.dfdx.row(row).segment(jointStart, jointDim).transpose() /
        cbf->getGamma();
    if (gradient.isZero(/*prec=*/0.0)) break;
    model.setJointAngles(point.state, model.getJointAngles(point.state) +
                                          ((kCbfBandMargin - smallestMargin) /
                                           gradient.squaredNorm()) *
                                              gradient);
  }
  return point;
}

void appendStateInputTerms(StateInputCostCollection& collection,
                           const EvaluationPoint& point,
                           const TargetTrajectories& target,
                           const PreComputation& preComputation,
                           std::vector<scalar_t>& fingerprint) {
  for (const std::string& name : sortedTermNames(collection)) {
    const StateInputCost& term = collection.get(name);
    fingerprint.push_back(term.getValue(point.time, point.state, point.input,
                                        target, preComputation));
    const ScalarFunctionQuadraticApproximation approximation =
        term.getQuadraticApproximation(point.time, point.state, point.input,
                                       target, preComputation);
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
    fingerprint.push_back(
        term.getValue(point.time, point.state, target, preComputation));
    append(term.getQuadraticApproximation(point.time, point.state, target,
                                          preComputation)
               .dfdx,
           fingerprint);
  }
}

void appendEqualityConstraints(StateInputConstraintCollection& collection,
                               const EvaluationPoint& point,
                               const PreComputation& preComputation,
                               std::vector<scalar_t>& fingerprint) {
  for (const std::string& name : sortedTermNames(collection)) {
    const StateInputConstraint& term = collection.get(name);
    const VectorFunctionLinearApproximation linearization =
        term.getLinearApproximation(point.time, point.state, point.input,
                                    preComputation);
    append(linearization.f, fingerprint);
    append(linearization.dfdx, fingerprint);
    append(linearization.dfdu, fingerprint);
  }
}

void appendSwingFootCoefficients(const PreComputation& preComputation,
                                 std::vector<scalar_t>& fingerprint) {
  for (const EndEffectorDynamicsLinearAccConstraint::Config& config :
       cast<WBMpcPreComputation>(preComputation)
           .getEeNormalAccelerationConstraintConfigs()) {
    append(config.b, fingerprint);
    append(config.Ax, fingerprint);
    append(config.Av, fingerprint);
    append(config.Aa, fingerprint);
  }
}

/** The penalty of `soft`, whose constraint has `numConstraints` rows, over
 * kPenaltySweep. */
void appendPenaltySweep(const MultidimensionalPenalty& penalty,
                        size_t numConstraints,
                        scalar_t time,
                        std::vector<scalar_t>& fingerprint) {
  for (const scalar_t value : kPenaltySweep) {
    fingerprint.push_back(penalty.getValue(
        time,
        vector_t::Constant(static_cast<Eigen::Index>(numConstraints), value)));
  }
}

void appendPenalties(OptimalControlProblem& problem,
                     scalar_t time,
                     std::vector<scalar_t>& fingerprint) {
  for (const std::string& name : sortedTermNames(*problem.softConstraintPtr)) {
    StateInputSoftConstraint& soft =
        problem.softConstraintPtr->get<StateInputSoftConstraint>(name);
    appendPenaltySweep(soft.getPenalty(), soft.get().getNumConstraints(time),
                       time, fingerprint);
  }
  for (const std::string& name :
       sortedTermNames(*problem.stateSoftConstraintPtr)) {
    // The joint limits' barrier is not a soft constraint's penalty: the point
    // inside it evaluates it.
    if (name == kJointLimitsTerm) continue;
    StateSoftConstraint& soft =
        problem.stateSoftConstraintPtr->get<StateSoftConstraint>(name);
    appendPenaltySweep(soft.getPenalty(), soft.get().getNumConstraints(time),
                       time, fingerprint);
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
    append(
        problem.costPtr->get<EndEffectorDynamicsFootCost>(term).getParameters(
            swingTime, target, *problem.preComputationPtr),
        fingerprint);
  }
  if (carriesTerm(*problem.costPtr, JointTorqueCostCppAd::kTermName)) {
    append(problem.costPtr
               ->get<JointTorqueCostCppAd>(JointTorqueCostCppAd::kTermName)
               .getParameters(swingTime, target, *problem.preComputationPtr),
           fingerprint);
  }
}

/** The settings the SQP solver reads at every solve that a reload writes: what
 * they do is how the solver iterates. */
void appendSolverSettings(SqpSolver& solver,
                          std::vector<scalar_t>& fingerprint) {
  const sqp::Settings& settings = solver.getSettings();
  fingerprint.insert(fingerprint.end(),
                     {static_cast<scalar_t>(settings.sqpIteration),
                      settings.deltaTol, settings.g_max, settings.g_min});
}

/** The swing trajectory planner's configuration as a reload left it, read back.
 */
void appendPlannerConfig(const SwingTrajectoryPlanner& planner,
                         std::vector<scalar_t>& fingerprint) {
  const SwingTrajectoryPlanner::Config& config = planner.getConfig();
  fingerprint.insert(
      fingerprint.end(),
      {config.liftoffLinvelFootRtWorldEwrtWorldZ,
       config.touchdownLinvelFootRtWorldEwrtWorldZ, config.swingHeight,
       config.swingTimeScale, config.touchDownHeightOffset,
       config.impactProximityFactorLiftoffSlope,
       config.impactProximityFactorTouchdownSlope,
       config.impactProximityFactorMidPointValue, config.swingPitchAngle,
       config.swingPitchRiseFraction, config.swingPitchFallFraction});
}

/**
 * Appends the state reference the running costs read at each of `points`
 * (SwitchedModelReferenceManager::getDesiredState()) with `target`'s knots
 * moving forward along their heading. The procedural arm swing is proportional
 * to the commanded forward speed, which the stack's standing target does not
 * command, so the arm_swing tuning acts on the problem only under such a
 * target; evaluated on the standing one it would change nothing, rightly.
 */
void appendWalkingStateReference(const WBMpcInterface& interface,
                                 const TargetTrajectories& target,
                                 const std::vector<EvaluationPoint>& points,
                                 std::vector<scalar_t>& fingerprint) {
  constexpr scalar_t kForwardSpeed = 0.5;  // [m/s]
  const WBAccelMpcRobotModel<scalar_t>& model = interface.getMpcRobotModel();
  TargetTrajectories walking = target;
  for (vector_t& knot : walking.stateTrajectory) {
    const scalar_t heading = model.getYawBaseRtWorld(knot);
    model.setBaseComLinearVelocity(
        knot, vector3_t(kForwardSpeed * std::cos(heading),
                        kForwardSpeed * std::sin(heading), 0.0));
  }
  const SwitchedModelReferenceManager& referenceManager =
      *interface.getSwitchedModelReferenceManagerPtr();
  for (const EvaluationPoint& point : points) {
    const vector_t reference =
        referenceManager.getDesiredState(walking, point.state, point.time);
    fingerprint.insert(fingerprint.end(), reference.data(),
                       reference.data() + reference.size());
  }
}

}  // namespace

std::vector<scalar_t> runningProblemEffects(WholeBodySolverStack& stack) {
  // The swing trajectories the terms and the swing-foot coefficients read are
  // planned before every solve.
  stack.replanReferences();
  WBMpcInterface& interface = stack.interface();
  SqpSolver& solver = *stack.mpc().getSolverPtr();
  OptimalControlProblem& problem = solver.getOcpDefinitions().front();
  const TargetTrajectories& target =
      interface.getReferenceManagerPtr()->getTargetTrajectories();
  std::vector<EvaluationPoint> points =
      evaluationPoints(interface, /*count=*/5);
  for (size_t i = 0; i < kSwingRampTimes.size(); ++i) {
    EvaluationPoint inRamp = points[i];
    inRamp.time = kSwingRampTimes[i];
    points.push_back(std::move(inRamp));
  }
  points.push_back(pointInsideTheJointLimitBarrier(interface));
  if (std::optional<EvaluationPoint> cbf =
          pointInsideTheFootCollisionCbfBarrier(interface, problem);
      cbf.has_value()) {
    points.push_back(*std::move(cbf));
  }

  std::vector<scalar_t> fingerprint;
  for (const EvaluationPoint& point : points) {
    problem.preComputationPtr->request(
        Request::Cost + Request::Constraint + Request::SoftConstraint,
        point.time, point.state, point.input);
    const PreComputation& preComputation = *problem.preComputationPtr;
    appendStateInputTerms(*problem.costPtr, point, target, preComputation,
                          fingerprint);
    appendStateInputTerms(*problem.softConstraintPtr, point, target,
                          preComputation, fingerprint);
    appendStateTerms(*problem.stateCostPtr, point, target, preComputation,
                     fingerprint);
    appendStateTerms(*problem.stateSoftConstraintPtr, point, target,
                     preComputation, fingerprint);
    appendStateTerms(*problem.finalCostPtr, point, target, preComputation,
                     fingerprint);
    appendEqualityConstraints(*problem.equalityConstraintPtr, point,
                              preComputation, fingerprint);
    appendSwingFootCoefficients(preComputation, fingerprint);
  }
  appendWalkingStateReference(interface, target, points, fingerprint);
  appendPenalties(problem, /*time=*/0.0, fingerprint);
  appendSolverSettings(solver, fingerprint);
  return fingerprint;
}

std::vector<scalar_t> runningProblemFingerprint(WholeBodySolverStack& stack) {
  std::vector<scalar_t> fingerprint = runningProblemEffects(stack);
  WBMpcInterface& interface = stack.interface();
  OptimalControlProblem& problem =
      stack.mpc().getSolverPtr()->getOcpDefinitions().front();
  appendCppAdParameters(
      problem, interface,
      interface.getReferenceManagerPtr()->getTargetTrajectories(), fingerprint);
  appendPlannerConfig(*interface.getSwitchedModelReferenceManagerPtr()
                           ->getSwingTrajectoryPlanner(),
                      fingerprint);
  return fingerprint;
}

}  // namespace ocs2::humanoid::live_tuning_test
