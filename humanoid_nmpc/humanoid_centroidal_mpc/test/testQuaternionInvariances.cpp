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
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/ComputationRequest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/constraint/StateConstraintCollection.h"
#include "ocs2_core/constraint/StateInputConstraintCollection.h"
#include "ocs2_core/cost/StateCostCollection.h"
#include "ocs2_core/cost/StateInputCostCollection.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/dynamics/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "support/ShippedCentroidalRobot.h"

/*
 * The invariances of the quaternion base orientation (design README sections 2.4, 2.8 and 5(c)) for every term the
 * centroidal MPC registers - every cost, soft constraint and constraint of every collection, intermediate and final -
 * and for its dynamics, on the problem of each shipped configuration as CentroidalMpcInterface builds it, at a tilted,
 * turned and moving state on a walking schedule:
 *  - tangent finite differences: the ambient Jacobian pulled back by the state manifold, J E(x), is the central
 *    difference of the term along x (+) eps e_i (eps = 1e-6) for every tangent direction, to 1e-5 relative;
 *  - radial invariance: g(c xi) = g(xi) for c in {0.5, 2, -1, -2} and grad_xi g . xi = 0 (a term reads only xi_hat); the
 *    dynamics' quaternion rows are homogeneous of degree one instead, f_xi(c xi) = c f_xi(xi) and J_xi xi = f_xi;
 *  - the double cover: g(-xi) = g(xi), and the dynamics' quaternion rate flips;
 *  - a pitch of 90 degrees: every value and derivative finite, and one SQP iteration from there finite;
 *  - the turn's representation: the state and its references turned about the vertical by alpha and by alpha + 2 pi k
 *    (the same rotation, the quaternion negated for an odd k), for alpha in {pi - 0.01, -pi + 0.01, 0.1} and up to ten
 *    half turns out, give every term the same value; and the dynamics turn with the state, f(R_z x, R_z u) = R_z f(x, u),
 *    for alpha in {pi - 0.01, -pi + 0.01, 10 pi + 0.1}. The costs are not invariant under the turn itself: the shipped
 *    world-frame weights differ between x and y (README section 5(c) asks for more than the formulation has, on either
 *    layout).
 * One documented exception, named where it applies: the zero-velocity constraints linearize with the foot at rest
 * (checked at a still point). The foot velocity constraints build their momentum map from the normalized quaternion, as
 * the kinematics do (CentroidalMpcInterface's velocity update callback), so they are held to every check.
 * Generates each configuration's CppAD libraries on a cold cache.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
constexpr scalar_t kFiniteDifferenceTolerance = 1.0e-5;
/** The intermediate times the terms are evaluated at: in left swing and in double support of the schedule below. */
constexpr std::array<scalar_t, 2> kTimes = {0.3, 0.6};
constexpr scalar_t kFinalTime = 1.5;

/** One term's value and, when asked for, the Jacobian of the value in the state's ambient coordinates. Passive data. */
struct TermValue {
  vector_t value;
  matrix_t stateJacobian;
};

/** Every active term of the problem by label (collection and term name), with "dynamics" for the flow map. */
using TermValues = std::map<std::string, TermValue>;

/** Whether evaluateProblem() computes the Jacobians as well. */
enum class Derivatives { kWith, kWithout };

/** A point of the problem: the time, the state and the input. Passive data. */
struct ProblemPoint {
  scalar_t time = 0.0;
  vector_t state;
  vector_t input;
};

void addStateInputCosts(absl::string_view label,
                        StateInputCostCollection& collection,
                        const ProblemPoint& point,
                        const TargetTrajectories& targets,
                        const PreComputation& preComputation,
                        Derivatives derivatives,
                        TermValues& values) {
  for (const std::pair<const std::string, size_t>& entry : collection.getTermNameMap()) {
    const StateInputCost& term = collection.get<StateInputCost>(entry.first);
    if (!term.isActive(point.time)) continue;
    TermValue& value = values[absl::StrCat(label, "/", entry.first)];
    value.value = vector_t::Constant(1, term.getValue(point.time, point.state, point.input, targets, preComputation));
    if (derivatives == Derivatives::kWith) {
      value.stateJacobian = term.getQuadraticApproximation(point.time, point.state, point.input, targets, preComputation).dfdx.transpose();
    }
  }
}

void addStateCosts(absl::string_view label,
                   StateCostCollection& collection,
                   const ProblemPoint& point,
                   const TargetTrajectories& targets,
                   const PreComputation& preComputation,
                   Derivatives derivatives,
                   TermValues& values) {
  for (const std::pair<const std::string, size_t>& entry : collection.getTermNameMap()) {
    const StateCost& term = collection.get<StateCost>(entry.first);
    if (!term.isActive(point.time)) continue;
    TermValue& value = values[absl::StrCat(label, "/", entry.first)];
    value.value = vector_t::Constant(1, term.getValue(point.time, point.state, targets, preComputation));
    if (derivatives == Derivatives::kWith) {
      value.stateJacobian = term.getQuadraticApproximation(point.time, point.state, targets, preComputation).dfdx.transpose();
    }
  }
}

void addStateInputConstraints(absl::string_view label,
                              StateInputConstraintCollection& collection,
                              const ProblemPoint& point,
                              const PreComputation& preComputation,
                              Derivatives derivatives,
                              TermValues& values) {
  for (const std::pair<const std::string, size_t>& entry : collection.getTermNameMap()) {
    const StateInputConstraint& term = collection.get<StateInputConstraint>(entry.first);
    if (!term.isActive(point.time)) continue;
    TermValue& value = values[absl::StrCat(label, "/", entry.first)];
    value.value = term.getValue(point.time, point.state, point.input, preComputation);
    if (derivatives == Derivatives::kWith) {
      value.stateJacobian = term.getLinearApproximation(point.time, point.state, point.input, preComputation).dfdx;
    }
  }
}

void addStateConstraints(absl::string_view label,
                         StateConstraintCollection& collection,
                         const ProblemPoint& point,
                         const PreComputation& preComputation,
                         Derivatives derivatives,
                         TermValues& values) {
  for (const std::pair<const std::string, size_t>& entry : collection.getTermNameMap()) {
    const StateConstraint& term = collection.get<StateConstraint>(entry.first);
    if (!term.isActive(point.time)) continue;
    TermValue& value = values[absl::StrCat(label, "/", entry.first)];
    value.value = term.getValue(point.time, point.state, preComputation);
    if (derivatives == Derivatives::kWith) {
      value.stateJacobian = term.getLinearApproximation(point.time, point.state, preComputation).dfdx;
    }
  }
}

/** Every intermediate term at `point` and the dynamics, as the solver's approximation requests them. */
TermValues evaluateIntermediate(OptimalControlProblem& problem, const ProblemPoint& point, Derivatives derivatives) {
  PreComputation& preComputation = *problem.preComputationPtr;
  preComputation.request(Request::Cost + Request::SoftConstraint + Request::Constraint + Request::Dynamics + Request::Approximation,
                         point.time, point.state, point.input);
  const TargetTrajectories& targets = *problem.targetTrajectoriesPtr;
  TermValues values;
  addStateInputCosts("cost", *problem.costPtr, point, targets, preComputation, derivatives, values);
  addStateInputCosts("softConstraint", *problem.softConstraintPtr, point, targets, preComputation, derivatives, values);
  addStateCosts("stateCost", *problem.stateCostPtr, point, targets, preComputation, derivatives, values);
  addStateCosts("stateSoftConstraint", *problem.stateSoftConstraintPtr, point, targets, preComputation, derivatives, values);
  addStateInputConstraints("equalityConstraint", *problem.equalityConstraintPtr, point, preComputation, derivatives, values);
  addStateInputConstraints("inequalityConstraint", *problem.inequalityConstraintPtr, point, preComputation, derivatives, values);
  addStateConstraints("stateEqualityConstraint", *problem.stateEqualityConstraintPtr, point, preComputation, derivatives, values);
  addStateConstraints("stateInequalityConstraint", *problem.stateInequalityConstraintPtr, point, preComputation, derivatives, values);
  TermValue& dynamics = values["dynamics"];
  if (derivatives == Derivatives::kWith) {
    const VectorFunctionLinearApproximation linear =
        problem.dynamicsPtr->linearApproximation(point.time, point.state, point.input, preComputation);
    dynamics.value = linear.f;
    dynamics.stateJacobian = linear.dfdx;
  } else {
    dynamics.value = problem.dynamicsPtr->computeFlowMap(point.time, point.state, point.input, preComputation);
  }
  return values;
}

/** Every final term at the state of `point` and kFinalTime. */
TermValues evaluateFinal(OptimalControlProblem& problem, const ProblemPoint& point, Derivatives derivatives) {
  PreComputation& preComputation = *problem.preComputationPtr;
  const ProblemPoint finalPoint{.time = kFinalTime, .state = point.state, .input = vector_t()};
  preComputation.requestFinal(Request::Cost + Request::SoftConstraint + Request::Constraint + Request::Approximation, finalPoint.time,
                              finalPoint.state);
  const TargetTrajectories& targets = *problem.targetTrajectoriesPtr;
  TermValues values;
  addStateCosts("finalCost", *problem.finalCostPtr, finalPoint, targets, preComputation, derivatives, values);
  addStateCosts("finalSoftConstraint", *problem.finalSoftConstraintPtr, finalPoint, targets, preComputation, derivatives, values);
  addStateConstraints("finalEqualityConstraint", *problem.finalEqualityConstraintPtr, finalPoint, preComputation, derivatives, values);
  addStateConstraints("finalInequalityConstraint", *problem.finalInequalityConstraintPtr, finalPoint, preComputation, derivatives, values);
  return values;
}

/** Every term, intermediate at `point` and final at its state, labeled apart. */
TermValues evaluateProblem(OptimalControlProblem& problem, const ProblemPoint& point, Derivatives derivatives) {
  TermValues values = evaluateIntermediate(problem, point, derivatives);
  for (std::pair<const std::string, TermValue>& entry : evaluateFinal(problem, point, derivatives)) {
    values.emplace(absl::StrCat("final:", entry.first), std::move(entry.second));
  }
  return values;
}

/** Whether `label` is the dynamics' flow map, whose quaternion rows read the raw xi; every term reads xi_hat alone. */
bool isFlowMap(const std::string& label) {
  return label == "dynamics";
}

/**
 * Whether `label` is a foot's zero-velocity constraint (ZeroVelocityConstraintCppAd on EndEffectorKinematicsTwistConstraint),
 * whose linearization holds the map from the foot's angular velocity to the rate of its orientation residual constant
 * (EndEffectorKinematicsTwistConstraint::getLinearApproximation(): the dropped term is the map's derivative times the
 * angular velocity). Its Jacobian is exact only where the foot does not turn, which a stance foot at the solution does
 * not: the tangent check holds it to its finite differences at a still point and leaves it out at a moving one.
 */
bool linearizesWithTheFootAtRest(const std::string& label) {
  return absl::StrContains(label, "_zeroVelocity");
}

/** The unit quaternion of a turn by `angle` about the world z axis, (x, y, z, w). */
vector4_t yawQuaternion(scalar_t angle) {
  return vector4_t(0.0, 0.0, std::sin(0.5 * angle), std::cos(0.5 * angle));
}

/** R_z(angle) v. */
vector3_t turnedAboutZ(scalar_t angle, const vector3_t& v) {
  return vector3_t(std::cos(angle) * v(0) - std::sin(angle) * v(1), std::sin(angle) * v(0) + std::cos(angle) * v(1), v(2));
}

/**
 * The state `state` turned by R_z(angle) about the world z axis: the world-frame momentum and the base position turned,
 * the base quaternion composed on the left, xi -> q_z(angle) xi, the joints unchanged. Linear in each block, so it also
 * turns a flow map's value (the rates of those blocks).
 */
vector_t turnedState(const MpcRobotModelBase<scalar_t>& model, const vector_t& state, scalar_t angle) {
  vector_t turned = state;
  turned.head<3>() = turnedAboutZ(angle, vector3_t(state.head<3>()));
  turned.segment<3>(3) = turnedAboutZ(angle, vector3_t(state.segment<3>(3)));
  const Eigen::Index baseStart = static_cast<Eigen::Index>(model.getBaseStartindex());
  turned.segment<3>(baseStart) = turnedAboutZ(angle, vector3_t(state.segment<3>(baseStart)));
  const Eigen::Index orientationStart = static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  turned.segment<4>(orientationStart) = quaternionProduct<scalar_t>(yawQuaternion(angle), vector4_t(state.segment<4>(orientationStart)));
  return turned;
}

/** max |a - b| <= tolerance max(1, max |b|), naming `label`. */
::testing::AssertionResult isCloseRelative(const std::string& label, const matrix_t& a, const matrix_t& b, scalar_t tolerance) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return ::testing::AssertionFailure() << label << ": " << a.rows() << "x" << a.cols() << " against " << b.rows() << "x" << b.cols();
  }
  if (a.size() == 0) return ::testing::AssertionSuccess();
  const scalar_t scale = std::max(1.0, b.cwiseAbs().maxCoeff());
  const scalar_t error = (a - b).cwiseAbs().maxCoeff();
  if (error <= tolerance * scale) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << label << ": off by " << error << " against " << tolerance << " x " << scale;
}

class QuaternionInvariances : public ::testing::TestWithParam<ShippedCentroidalRobot> {
 protected:
  void SetUp() override {
    const CentroidalRobotFiles files = shippedCentroidalRobotFiles(GetParam());
    absl::StatusOr<CentroidalMpcConfig> config = loadConfigOf(files);
    ASSERT_TRUE(config.ok()) << config.status();
    absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface = CentroidalMpcInterface::Create(*std::move(config), files.urdfFile);
    ASSERT_TRUE(interface.ok()) << interface.status();
    interface_ = *std::move(interface);
    manifold_ = interface_->getStateManifold();
    ASSERT_NE(manifold_, nullptr);
    setReferences(/*angle=*/0.0);
  }

  const MpcRobotModelBase<scalar_t>& model() const { return interface_->getMpcRobotModel(); }
  OptimalControlProblem& problem() { return interface_->getOptimalControlProblemRef(); }
  Eigen::Index orientationStart() const { return static_cast<Eigen::Index>(model().getBaseOrientationStartIndex()); }

  /**
   * The walking schedule and a forward-moving target from the initial state, all turned by R_z(angle), the references
   * built on them from a fresh reference manager (reset() first) as before a solve, and the problem pointed at them.
   */
  void setReferences(scalar_t angle) {
    const vector_t x0 = interface_->getInitialState();
    const ModeSchedule schedule({0.2, 0.55, 0.65, 1.0, 1.1, 1.45},
                                {ModeNumber::kStance, ModeNumber::kLf, ModeNumber::kStance, ModeNumber::kRf, ModeNumber::kStance,
                                 ModeNumber::kLf, ModeNumber::kStance});
    vector_t start = x0;
    start(0) = 0.4;  // forward normalized momentum: a walking command
    vector_t end = start;
    end(static_cast<Eigen::Index>(model().getBaseStartindex())) += 0.8;
    const vector_t u0 = vector_t::Zero(static_cast<Eigen::Index>(interface_->getEffectiveMpcRobotModel().getInputDim()));
    const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = interface_->getSwitchedModelReferenceManagerPtr();
    referenceManager->reset();
    referenceManager->getGaitSchedule()->updateModeSchedule(schedule);
    referenceManager->setTargetTrajectories(
        TargetTrajectories({0.0, 2.0}, {turnedState(model(), start, angle), turnedState(model(), end, angle)}, {u0, u0}));
    referenceManager->preSolverRun(/*initTime=*/0.0, /*finalTime=*/kFinalTime, turnedState(model(), x0, angle), ModeNumber::kStance);
    problem().targetTrajectoriesPtr = &referenceManager->getTargetTrajectories();
  }

  /**
   * The point at `time`: the initial state moving (momentum), turned past pi and tilted, its joints bent; the input the
   * weight on both feet, perturbed, every joint moving.
   */
  ProblemPoint pointAt(scalar_t time, const vector3_t& eulerZyx) const {
    const MpcRobotModelBase<scalar_t>& effectiveModel = interface_->getEffectiveMpcRobotModel();
    vector_t state = interface_->getInitialState();
    state.head<6>() << 0.3, 0.05, -0.02, 0.02, -0.01, 0.03;
    model().setBaseOrientationEulerZyx(state, eulerZyx);
    vector_t joints = model().getJointAngles(state);
    for (Eigen::Index i = 0; i < joints.size(); ++i) joints(i) += 0.05 * std::sin(1.3 * static_cast<scalar_t>(i) + 0.4);
    model().setJointAngles(state, joints);
    vector_t input = weightCompensatingInput(interface_->getCentroidalModelInfo(), contact_flag_t{true, true}, effectiveModel, state);
    for (Eigen::Index i = 0; i < input.size(); ++i) input(i) *= 1.0 + 0.1 * std::sin(0.7 * static_cast<scalar_t>(i));
    vector_t jointVelocities(joints.size());
    for (Eigen::Index i = 0; i < jointVelocities.size(); ++i) jointVelocities(i) = 0.3 * std::cos(0.9 * static_cast<scalar_t>(i));
    effectiveModel.setJointVelocities(state, input, jointVelocities);
    return ProblemPoint{.time = time, .state = std::move(state), .input = std::move(input)};
  }

  /** pointAt() with the robot still: no momentum and no joint velocity, so every foot's twist is zero. */
  ProblemPoint stillPointAt(scalar_t time, const vector3_t& eulerZyx) const {
    ProblemPoint point = pointAt(time, eulerZyx);
    point.state.head<6>().setZero();
    const MpcRobotModelBase<scalar_t>& effectiveModel = interface_->getEffectiveMpcRobotModel();
    effectiveModel.setJointVelocities(point.state, point.input, vector_t::Zero(static_cast<Eigen::Index>(model().getJointDim())));
    return point;
  }

  /** `point` with its base quaternion multiplied by `c`. */
  ProblemPoint withScaledQuaternion(const ProblemPoint& point, scalar_t c) const {
    ProblemPoint scaled = point;
    scaled.state.segment<4>(orientationStart()) *= c;
    return scaled;
  }

  /** `point`'s input turned by R_z(angle): the world-frame wrenches of the wrench input; the basis scalings are local. */
  vector_t turnedInput(const vector_t& input, scalar_t angle) const {
    const MpcRobotModelBase<scalar_t>& effectiveModel = interface_->getEffectiveMpcRobotModel();
    if (effectiveModel.getInputDim() != model().getInputDim()) return input;
    vector_t turned = input;
    for (size_t contact = 0; contact < kNumContacts; ++contact) {
      model().setContactForce(turned, turnedAboutZ(angle, model().getContactForce(input, contact)), contact);
      model().setContactMoment(turned, turnedAboutZ(angle, model().getContactMoment(input, contact)), contact);
    }
    return turned;
  }

  std::unique_ptr<CentroidalMpcInterface> interface_;
  std::shared_ptr<const StateManifold> manifold_;
};

/**
 * Expects the tangent Jacobian of every term at `point` but the ones `skip` names, J E(x) of the ambient one, to be the
 * central difference of the term along x (+) eps e_i; returns how many terms it checked.
 */
size_t expectTangentJacobiansAreFiniteDifferences(OptimalControlProblem& problem,
                                                  const StateManifold& manifold,
                                                  const ProblemPoint& point,
                                                  bool (*absl_nonnull skip)(const std::string&)) {
  const TermValues values = evaluateProblem(problem, point, Derivatives::kWith);
  const Eigen::Index tangentDim = static_cast<Eigen::Index>(manifold.getTangentDim());
  std::map<std::string, matrix_t> differences;
  for (Eigen::Index i = 0; i < tangentDim; ++i) {
    vector_t direction = vector_t::Zero(tangentDim);
    direction(i) = 1.0;
    ProblemPoint plus = point;
    ProblemPoint minus = point;
    manifold.retract(point.state, direction, kFiniteDifferenceStep, plus.state);
    manifold.retract(point.state, direction, -kFiniteDifferenceStep, minus.state);
    const TermValues plusValues = evaluateProblem(problem, plus, Derivatives::kWithout);
    const TermValues minusValues = evaluateProblem(problem, minus, Derivatives::kWithout);
    for (const std::pair<const std::string, TermValue>& entry : values) {
      matrix_t& difference = differences[entry.first];
      if (difference.size() == 0) difference.setZero(entry.second.value.size(), tangentDim);
      difference.col(i) = (plusValues.at(entry.first).value - minusValues.at(entry.first).value) / (2.0 * kFiniteDifferenceStep);
    }
  }
  size_t checked = 0;
  for (const std::pair<const std::string, TermValue>& entry : values) {
    if (skip(entry.first)) continue;
    // The flow map's rows are ambient rates, its quaternion rows compared as they are, a rate in R^4.
    matrix_t tangentJacobian = entry.second.stateJacobian;
    manifold.pullBackStateColumns(point.state, tangentJacobian);
    EXPECT_TRUE(isCloseRelative(entry.first, tangentJacobian, differences.at(entry.first), kFiniteDifferenceTolerance));
    ++checked;
  }
  return checked;
}

/** A skip of expectTangentJacobiansAreFiniteDifferences() that checks every term. */
bool noTerm(const std::string& /*label*/) {
  return false;
}

TEST_P(QuaternionInvariances, EveryTermsTangentJacobianIsItsFiniteDifferenceAlongTheRetraction) {
  // The procedural arm swing reads the REFERENCE's heading, not the node's (SwitchedModelReferenceManager::
  // getDesiredState()), so the state costs' Jacobians, which hold the reference constant, are exact with it on. It is
  // switched on wherever the robot names its arm joints - also on the robots whose shipped ACoM cost turns it off - so
  // that this check would catch a reference that went back to reading the state.
  interface_->getSwitchedModelReferenceManagerPtr()->setArmSwingReferenceActive(interface_->modelSettings().hasArmSwingJoints);
  size_t checked = 0;
  for (const scalar_t time : kTimes) {
    SCOPED_TRACE(absl::StrCat("t = ", time));
    // Moving, turned past pi and tilted: every term whose linearization is exact everywhere.
    {
      SCOPED_TRACE("moving");
      checked += expectTangentJacobiansAreFiniteDifferences(problem(), *manifold_, pointAt(time, vector3_t(3.3, 0.15, -0.1)),
                                                            linearizesWithTheFootAtRest);
    }
    // Still, at another attitude: every term, the zero-velocity constraints' linearization exact there.
    {
      SCOPED_TRACE("still");
      checked += expectTangentJacobiansAreFiniteDifferences(problem(), *manifold_, stillPointAt(time, vector3_t(-1.9, -0.2, 0.12)), noTerm);
    }
  }
  EXPECT_GT(checked, 40U) << "the problem registers too few terms for this to test anything";
  ::testing::Test::RecordProperty("terms_checked", static_cast<int>(checked));
}

TEST_P(QuaternionInvariances, EveryTermReadsOnlyTheAttitudeOfTheQuaternion) {
  for (const scalar_t time : kTimes) {
    SCOPED_TRACE(absl::StrCat("t = ", time));
    const ProblemPoint point = pointAt(time, vector3_t(-2.4, -0.2, 0.25));
    const TermValues values = evaluateProblem(problem(), point, Derivatives::kWith);
    const vector4_t xi = point.state.segment<4>(orientationStart());
    for (const scalar_t c : {0.5, 2.0, -1.0, -2.0}) {
      SCOPED_TRACE(absl::StrCat("xi scaled by ", c));
      const TermValues scaled = evaluateProblem(problem(), withScaledQuaternion(point, c), Derivatives::kWithout);
      ASSERT_EQ(scaled.size(), values.size());
      for (const std::pair<const std::string, TermValue>& entry : values) {
        vector_t expected = entry.second.value;
        if (isFlowMap(entry.first)) expected.segment<4>(orientationStart()) *= c;  // G(c xi) w_B / 2 = c G(xi) w_B / 2
        EXPECT_TRUE(isCloseRelative(entry.first, scaled.at(entry.first).value, expected, /*tolerance=*/1.0e-12));
      }
    }
    // Euler's theorem for the homogeneous functions: grad_xi g . xi = 0 for the terms (degree 0), J_xi xi = f_xi for
    // the quaternion rows of the flow map (degree 1) and 0 for its other rows.
    for (const std::pair<const std::string, TermValue>& entry : values) {
      const matrix_t& jacobian = entry.second.stateJacobian;
      const vector_t radial = jacobian.middleCols<4>(orientationStart()) * xi;
      vector_t expected = vector_t::Zero(radial.size());
      if (isFlowMap(entry.first)) expected.segment<4>(orientationStart()) = entry.second.value.segment<4>(orientationStart());
      const scalar_t scale = std::max(1.0, jacobian.middleCols<4>(orientationStart()).cwiseAbs().maxCoeff());
      EXPECT_LE((radial - expected).cwiseAbs().maxCoeff(), 1.0e-9 * scale) << entry.first;
    }
  }
}

TEST_P(QuaternionInvariances, AtAPitchOf90DegreesEveryTermAndOneSqpIterationAreFinite) {
  for (const vector3_t& eulerZyx : {vector3_t(0.3, M_PI / 2.0, 0.2), vector3_t(-2.0, -M_PI / 2.0, 0.0)}) {
    SCOPED_TRACE(eulerZyx.transpose());
    for (const scalar_t time : kTimes) {
      const ProblemPoint point = pointAt(time, eulerZyx);
      for (const std::pair<const std::string, TermValue>& entry : evaluateProblem(problem(), point, Derivatives::kWith)) {
        EXPECT_TRUE(entry.second.value.allFinite()) << entry.first << " at t = " << time;
        EXPECT_TRUE(entry.second.stateJacobian.allFinite()) << entry.first << " at t = " << time;
      }
    }
    // One SQP iteration from the robot at that attitude, on the references of the initial state.
    sqp::Settings settings = interface_->sqpSettings();
    settings.sqpIteration = 1;
    settings.printSolverStatus = false;
    settings.printSolverStatistics = false;
    settings.printLinesearch = false;
    SqpSolver solver(settings, interface_->getOptimalControlProblem(), interface_->getInitializer());
    solver.setReferenceManager(interface_->getReferenceManagerPtr());
    const ProblemPoint start = pointAt(/*time=*/0.0, eulerZyx);
    solver.run(/*initTime=*/0.0, start.state, ModeNumber::kStance, /*finalTime=*/interface_->mpcSettings().timeHorizon_);
    PrimalSolution solution;
    solver.getPrimalSolution(interface_->mpcSettings().timeHorizon_, &solution);
    ASSERT_FALSE(solution.stateTrajectory_.empty());
    for (size_t node = 0; node < solution.stateTrajectory_.size(); ++node) {
      EXPECT_TRUE(solution.stateTrajectory_[node].allFinite()) << "node " << node;
      EXPECT_TRUE(solution.inputTrajectory_[node].allFinite()) << "node " << node;
    }
    EXPECT_TRUE(std::isfinite(solver.getPerformanceIndeces().cost));
    setReferences(/*angle=*/0.0);
  }
}

TEST_P(QuaternionInvariances, NoTermReadsWhichAngleOrQuaternionRepresentsATurn) {
  // The state and its references turned about the vertical by alpha, and by alpha + 2 pi k: the same rotation, its
  // headings 2 pi k apart and its quaternions negated for odd k. Every term and the dynamics read them alike, across
  // +-pi and ten half turns out.
  const std::vector<std::pair<scalar_t, scalar_t>> sameTurns = {
      {M_PI - 0.01, -M_PI - 0.01}, {-M_PI + 0.01, M_PI + 0.01}, {0.1, 10.0 * M_PI + 0.1}, {0.1, -4.0 * M_PI + 0.1}};
  for (const scalar_t time : kTimes) {
    SCOPED_TRACE(absl::StrCat("t = ", time));
    const ProblemPoint point = pointAt(time, vector3_t(0.6, 0.12, -0.08));
    for (const std::pair<scalar_t, scalar_t>& turns : sameTurns) {
      SCOPED_TRACE(absl::StrCat("turned by ", turns.first, " and by ", turns.second));
      setReferences(turns.first);
      const TermValues first = evaluateProblem(
          problem(),
          ProblemPoint{
              .time = time, .state = turnedState(model(), point.state, turns.first), .input = turnedInput(point.input, turns.first)},
          Derivatives::kWithout);
      setReferences(turns.second);
      const TermValues second = evaluateProblem(
          problem(),
          ProblemPoint{
              .time = time, .state = turnedState(model(), point.state, turns.second), .input = turnedInput(point.input, turns.second)},
          Derivatives::kWithout);
      ASSERT_EQ(first.size(), second.size());
      for (const std::pair<const std::string, TermValue>& entry : first) {
        vector_t expected = entry.second.value;
        // The flow map's quaternion rate turns with the quaternion: negated with it for an odd k.
        if (isFlowMap(entry.first)) {
          const vector4_t firstQuaternion = yawQuaternion(turns.first);
          const vector4_t secondQuaternion = yawQuaternion(turns.second);
          expected.segment<4>(orientationStart()) *= firstQuaternion.dot(secondQuaternion) < 0.0 ? -1.0 : 1.0;
        }
        EXPECT_TRUE(isCloseRelative(entry.first, second.at(entry.first).value, expected, /*tolerance=*/1.0e-9));
      }
    }
  }
  setReferences(/*angle=*/0.0);
}

TEST_P(QuaternionInvariances, TheDynamicsTurnWithTheStateAboutTheVertical) {
  // The physics is isotropic: the flow map of the state and the input's world-frame wrenches turned by R_z(alpha) is the
  // flow map turned by R_z(alpha). (The costs are not invariant under the turn as a whole: the shipped weights of the
  // world-frame blocks differ between x and y, e.g. the SA01's normalized angular momentum 15 and 40.)
  for (const scalar_t time : kTimes) {
    SCOPED_TRACE(absl::StrCat("t = ", time));
    const ProblemPoint point = pointAt(time, vector3_t(0.6, 0.12, -0.08));
    PreComputation& preComputation = *problem().preComputationPtr;
    preComputation.request(Request::Dynamics, point.time, point.state, point.input);
    const vector_t flow = problem().dynamicsPtr->computeFlowMap(point.time, point.state, point.input, preComputation);
    for (const scalar_t angle : {M_PI - 0.01, -M_PI + 0.01, 10.0 * M_PI + 0.1}) {
      SCOPED_TRACE(absl::StrCat("turned by ", angle));
      const ProblemPoint turned{.time = time, .state = turnedState(model(), point.state, angle), .input = turnedInput(point.input, angle)};
      preComputation.request(Request::Dynamics, turned.time, turned.state, turned.input);
      const vector_t turnedFlow = problem().dynamicsPtr->computeFlowMap(turned.time, turned.state, turned.input, preComputation);
      EXPECT_TRUE(isCloseRelative("dynamics", turnedFlow, turnedState(model(), flow, angle), /*tolerance=*/1.0e-9));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ShippedRobots,
                         QuaternionInvariances,
                         ::testing::Values(ShippedCentroidalRobot::kDrcAtlas,
                                           ShippedCentroidalRobot::kEngineAiSa01,
                                           ShippedCentroidalRobot::kUnitreeG1,
                                           ShippedCentroidalRobot::kUnitreeR1),
                         shippedCentroidalRobotTestName);

}  // namespace
}  // namespace ocs2::humanoid
