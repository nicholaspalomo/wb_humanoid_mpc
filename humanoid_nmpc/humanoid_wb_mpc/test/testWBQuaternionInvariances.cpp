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

#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_sqp/SqpSettings.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "robot_core/ResourcePaths.h"

/*
 * Every registered term of the shipped G1 whole-body MPC - each cost, soft
 * constraint, equality and inequality constraint, running, state-only and final
 * - and its dynamics, on the quaternion layout (README 5(c)):
 *  - their Jacobians are their derivatives along the state manifold: J E(x)
 * against central differences of g(x (+) eps e_i), eps = 1e-6, to 1e-5 of the
 * Jacobian's size;
 *  - they read the base orientation as a rotation: g(c xi) = g(xi) and dg/dxi
 * xi = 0 for c in {0.5, 2}, and g(-xi) = g(xi) (the dynamics: but for the
 * quaternion rate, which is linear in xi);
 *  - at a pitch of 90 degrees every value and Jacobian is finite, and so is one
 * SQP iteration from there;
 *  - they are equivariant under a turn R_z(alpha) of the world about its
 * vertical, of the state, the input's world-frame wrenches and the target
 * alike, for alpha = pi - 0.01, -pi + 0.01 and 10 pi + 0.1: every cost and the
 * norm of every constraint is unchanged, and the dynamics turns with them. A
 * term this test cannot evaluate (the pre-jump collections) fails it, so a term
 * added to the MPC is checked here. Large: it tapes and compiles every CppAD
 * library of the MPC once, into a fresh directory.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
constexpr scalar_t kFiniteDifferenceTolerance = 1.0e-5;
// [s] The swing of the right foot in the mode schedule of every point, and a
// time in stance and one in the swing.
constexpr scalar_t kSwingStart = 0.0;
constexpr scalar_t kSwingEnd = 0.6;
constexpr scalar_t kStanceTime = -0.3;
constexpr scalar_t kSwingTime = 0.3;

std::unique_ptr<WBMpcInterface> createShippedG1WholeBodyMpc() {
  // LINT.IfChange(robot_files)
  const absl::StatusOr<std::string> taskFile = robot::resolveResourcePath(
      "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
  const absl::StatusOr<std::string> referenceFile = robot::resolveResourcePath(
      "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto");
  const absl::StatusOr<std::string> urdfFile = robot::resolveResourcePath(
      "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
  // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:quaternion_invariances_test_data)
  EXPECT_TRUE(taskFile.ok() && referenceFile.ok() && urdfFile.ok())
      << "the G1 whole-body files are not in the runfiles";
  if (!taskFile.ok() || !referenceFile.ok() || !urdfFile.ok()) return nullptr;
  // The libraries go to a folder relative to the working directory: a directory
  // made for this run is empty, so every library is taped and compiled from
  // this code (testWBMpcConstruction).
  std::string freshDirectory = (std::filesystem::path(testing::TempDir()) /
                                "testWBQuaternionInvariances_XXXXXX")
                                   .string();
  if (mkdtemp(freshDirectory.data()) == nullptr) {
    ADD_FAILURE() << "cannot make a fresh working directory from "
                  << freshDirectory;
    return nullptr;
  }
  std::filesystem::current_path(freshDirectory);
  absl::StatusOr<std::unique_ptr<WBMpcInterface>> created =
      WBMpcInterface::Create(*taskFile, *urdfFile, *referenceFile);
  EXPECT_TRUE(created.ok()) << created.status();
  return created.ok() ? *std::move(created) : nullptr;
}

/** The MPC of createShippedG1WholeBodyMpc(), built once for the test program;
 * never destroyed. */
WBMpcInterface* absl_nullable shippedG1WholeBodyMpc() {
  static WBMpcInterface* absl_nullable const kInterface =
      createShippedG1WholeBodyMpc().release();
  return kInterface;
}

/** The collection a registered term lives in. */
enum class TermCollection {
  kCost,
  kSoftConstraint,
  kStateCost,
  kStateSoftConstraint,
  kFinalCost,
  kFinalSoftConstraint,
  kEqualityConstraint,
  kStateEqualityConstraint,
  kFinalEqualityConstraint,
  kInequalityConstraint,
  kStateInequalityConstraint,
  kFinalInequalityConstraint,
  kDynamics,
};

/** Whether a term's values are costs (compared as they are) or constraint
 * values (compared by their norm). */
bool isCost(TermCollection collection) {
  switch (collection) {
    case TermCollection::kCost:
    case TermCollection::kSoftConstraint:
    case TermCollection::kStateCost:
    case TermCollection::kStateSoftConstraint:
    case TermCollection::kFinalCost:
    case TermCollection::kFinalSoftConstraint:
      return true;
    case TermCollection::kEqualityConstraint:
    case TermCollection::kStateEqualityConstraint:
    case TermCollection::kFinalEqualityConstraint:
    case TermCollection::kInequalityConstraint:
    case TermCollection::kStateInequalityConstraint:
    case TermCollection::kFinalInequalityConstraint:
    case TermCollection::kDynamics:
      return false;
  }
  ADD_FAILURE() << "unknown term collection " << static_cast<int>(collection);
  return false;
}

/** A registered term of an optimal control problem: its collection and its name
 * ("dynamics" for the dynamics). */
struct TermReference {
  TermCollection collection = TermCollection::kCost;
  std::string name;
};

/** The names of the terms of `collection`, sorted. */
template <typename Collection>
std::vector<std::string> sortedNames(const Collection& collection) {
  std::vector<std::string> names;
  for (const std::pair<const std::string, size_t>& entry :
       collection.getTermNameMap())
    names.push_back(entry.first);
  std::sort(names.begin(), names.end());
  return names;
}

/** Appends a reference to each of the terms `names` of `collection` to `terms`.
 */
void appendTerms(TermCollection collection,
                 const std::vector<std::string>& names,
                 std::vector<TermReference>& terms) {
  for (const std::string& name : names)
    terms.push_back(TermReference{.collection = collection, .name = name});
}

/** Every registered term of `problem` and its dynamics; a test failure for a
 * term in a collection this test does not evaluate. */
std::vector<TermReference> registeredTerms(
    const OptimalControlProblem& problem) {
  std::vector<TermReference> terms;
  appendTerms(TermCollection::kCost, sortedNames(*problem.costPtr), terms);
  appendTerms(TermCollection::kSoftConstraint,
              sortedNames(*problem.softConstraintPtr), terms);
  appendTerms(TermCollection::kStateCost, sortedNames(*problem.stateCostPtr),
              terms);
  appendTerms(TermCollection::kStateSoftConstraint,
              sortedNames(*problem.stateSoftConstraintPtr), terms);
  appendTerms(TermCollection::kFinalCost, sortedNames(*problem.finalCostPtr),
              terms);
  appendTerms(TermCollection::kFinalSoftConstraint,
              sortedNames(*problem.finalSoftConstraintPtr), terms);
  appendTerms(TermCollection::kEqualityConstraint,
              sortedNames(*problem.equalityConstraintPtr), terms);
  appendTerms(TermCollection::kStateEqualityConstraint,
              sortedNames(*problem.stateEqualityConstraintPtr), terms);
  appendTerms(TermCollection::kFinalEqualityConstraint,
              sortedNames(*problem.finalEqualityConstraintPtr), terms);
  appendTerms(TermCollection::kInequalityConstraint,
              sortedNames(*problem.inequalityConstraintPtr), terms);
  appendTerms(TermCollection::kStateInequalityConstraint,
              sortedNames(*problem.stateInequalityConstraintPtr), terms);
  appendTerms(TermCollection::kFinalInequalityConstraint,
              sortedNames(*problem.finalInequalityConstraintPtr), terms);
  terms.push_back(TermReference{.collection = TermCollection::kDynamics,
                                .name = "dynamics"});
  // The collections a term is evaluated in otherwise: none of the whole-body
  // MPC's are, and one that is must be added here.
  EXPECT_TRUE(problem.preJumpCostPtr->getTermNameMap().empty());
  EXPECT_TRUE(problem.preJumpSoftConstraintPtr->getTermNameMap().empty());
  EXPECT_TRUE(problem.preJumpEqualityConstraintPtr->getTermNameMap().empty());
  EXPECT_TRUE(problem.preJumpInequalityConstraintPtr->getTermNameMap().empty());
  return terms;
}

/** A time, a state and an input of the whole-body MPC. Passive data. */
struct EvaluationPoint {
  scalar_t time = 0.0;
  vector_t state;
  vector_t input;
};

/** The value of a term at a point and its Jacobian in the ambient state
 * coordinates. Passive data. */
struct TermEvaluation {
  bool active = false;
  vector_t value;
  matrix_t stateJacobian;
};

/** What evaluate() computes. */
enum class Derivatives {
  kValue,
  kValueAndStateJacobian,
};

/** The evaluation of a cost: its value as a vector of one entry and the
 * transpose of its state gradient. */
TermEvaluation costEvaluation(const StateInputCost& cost,
                              const EvaluationPoint& point,
                              const TargetTrajectories& target,
                              const PreComputation& preComputation,
                              Derivatives derivatives) {
  TermEvaluation evaluation{
      .active = true,
      .value = vector_t::Constant(
          1, cost.getValue(point.time, point.state, point.input, target,
                           preComputation)),
      .stateJacobian = matrix_t()};
  if (derivatives == Derivatives::kValueAndStateJacobian) {
    evaluation.stateJacobian =
        cost.getQuadraticApproximation(point.time, point.state, point.input,
                                       target, preComputation)
            .dfdx.transpose();
  }
  return evaluation;
}

/** costEvaluation() of a state-only cost. */
TermEvaluation costEvaluation(const StateCost& cost,
                              const EvaluationPoint& point,
                              const TargetTrajectories& target,
                              const PreComputation& preComputation,
                              Derivatives derivatives) {
  TermEvaluation evaluation{
      .active = true,
      .value = vector_t::Constant(
          1, cost.getValue(point.time, point.state, target, preComputation)),
      .stateJacobian = matrix_t()};
  if (derivatives == Derivatives::kValueAndStateJacobian) {
    evaluation.stateJacobian =
        cost.getQuadraticApproximation(point.time, point.state, target,
                                       preComputation)
            .dfdx.transpose();
  }
  return evaluation;
}

/**
 * Evaluates `term` of `problem` at `point` against the problem's target, after
 * the pre-computation's request at the point, as the solver does (an
 * intermediate node; the final terms at a final node). An inactive term is not
 * evaluated.
 */
TermEvaluation evaluate(
    OptimalControlProblem& problem,
    const TermReference& term,
    const EvaluationPoint& point,
    Derivatives derivatives = Derivatives::kValueAndStateJacobian) {
  PreComputation& preComputation = *problem.preComputationPtr;
  const TargetTrajectories& target = *problem.targetTrajectoriesPtr;
  const scalar_t t = point.time;
  const vector_t& x = point.state;
  const vector_t& u = point.input;
  const RequestSet request = Request::Cost + Request::SoftConstraint +
                             Request::Constraint + Request::Dynamics +
                             Request::Approximation;
  preComputation.request(request, t, x, u);
  const bool withJacobian = derivatives == Derivatives::kValueAndStateJacobian;
  TermEvaluation evaluation;
  switch (term.collection) {
    case TermCollection::kCost:
    case TermCollection::kSoftConstraint: {
      StateInputCostCollection& collection =
          term.collection == TermCollection::kCost ? *problem.costPtr
                                                   : *problem.softConstraintPtr;
      const StateInputCost& cost = collection.get(term.name);
      if (cost.isActive(t))
        evaluation =
            costEvaluation(cost, point, target, preComputation, derivatives);
      return evaluation;
    }
    case TermCollection::kStateCost:
    case TermCollection::kStateSoftConstraint:
    case TermCollection::kFinalCost:
    case TermCollection::kFinalSoftConstraint: {
      StateCostCollection* absl_nonnull collection = problem.stateCostPtr.get();
      if (term.collection == TermCollection::kStateSoftConstraint)
        collection = problem.stateSoftConstraintPtr.get();
      if (term.collection == TermCollection::kFinalCost)
        collection = problem.finalCostPtr.get();
      if (term.collection == TermCollection::kFinalSoftConstraint)
        collection = problem.finalSoftConstraintPtr.get();
      if (term.collection == TermCollection::kFinalCost ||
          term.collection == TermCollection::kFinalSoftConstraint) {
        preComputation.requestFinal(
            Request::Cost + Request::SoftConstraint + Request::Approximation, t,
            x);
      }
      const StateCost& cost = collection->get(term.name);
      if (cost.isActive(t))
        evaluation =
            costEvaluation(cost, point, target, preComputation, derivatives);
      return evaluation;
    }
    case TermCollection::kEqualityConstraint:
    case TermCollection::kInequalityConstraint: {
      StateInputConstraintCollection& collection =
          term.collection == TermCollection::kEqualityConstraint
              ? *problem.equalityConstraintPtr
              : *problem.inequalityConstraintPtr;
      const StateInputConstraint& constraint = collection.get(term.name);
      evaluation.active = constraint.isActive(t);
      if (evaluation.active) {
        evaluation.value = constraint.getValue(t, x, u, preComputation);
        if (withJacobian)
          evaluation.stateJacobian =
              constraint.getLinearApproximation(t, x, u, preComputation).dfdx;
      }
      return evaluation;
    }
    case TermCollection::kStateEqualityConstraint:
    case TermCollection::kStateInequalityConstraint:
    case TermCollection::kFinalEqualityConstraint:
    case TermCollection::kFinalInequalityConstraint: {
      StateConstraintCollection* absl_nonnull collection =
          problem.stateEqualityConstraintPtr.get();
      if (term.collection == TermCollection::kStateInequalityConstraint)
        collection = problem.stateInequalityConstraintPtr.get();
      if (term.collection == TermCollection::kFinalEqualityConstraint)
        collection = problem.finalEqualityConstraintPtr.get();
      if (term.collection == TermCollection::kFinalInequalityConstraint)
        collection = problem.finalInequalityConstraintPtr.get();
      if (term.collection == TermCollection::kFinalEqualityConstraint ||
          term.collection == TermCollection::kFinalInequalityConstraint) {
        preComputation.requestFinal(
            Request::Constraint + Request::Approximation, t, x);
      }
      const StateConstraint& constraint = collection->get(term.name);
      evaluation.active = constraint.isActive(t);
      if (evaluation.active) {
        evaluation.value = constraint.getValue(t, x, preComputation);
        if (withJacobian)
          evaluation.stateJacobian =
              constraint.getLinearApproximation(t, x, preComputation).dfdx;
      }
      return evaluation;
    }
    case TermCollection::kDynamics: {
      evaluation.active = true;
      evaluation.value =
          problem.dynamicsPtr->computeFlowMap(t, x, u, preComputation);
      if (withJacobian)
        evaluation.stateJacobian =
            problem.dynamicsPtr->linearApproximation(t, x, u, preComputation)
                .dfdx;
      return evaluation;
    }
  }
  ADD_FAILURE() << "unknown term collection "
                << static_cast<int>(term.collection);
  return evaluation;
}

/** max |a - b| / max(1, |a|_inf). */
scalar_t relativeDifference(const vector_t& a, const vector_t& b) {
  if (a.size() != b.size()) return std::numeric_limits<scalar_t>::infinity();
  if (a.size() == 0) return 0.0;
  return (a - b).cwiseAbs().maxCoeff() / std::max(1.0, a.cwiseAbs().maxCoeff());
}

/** `value` of `term` without the dynamics' quaternion rate, the rows at
 * `orientationStart`, which is linear in xi. */
vector_t withoutQuaternionRate(const TermReference& term,
                               Eigen::Index orientationStart,
                               vector_t value) {
  if (term.collection == TermCollection::kDynamics)
    value.segment<4>(orientationStart).setZero();
  return value;
}

/** The unit quaternion of a turn by `angle` about the world z axis. */
vector4_t headingRotation(scalar_t angle) {
  return vector4_t(0.0, 0.0, std::sin(0.5 * angle), std::cos(0.5 * angle));
}

/**
 * The robot, the reference manager and the problem of the shipped MPC at one
 * set of references: the target `target` and a mode schedule that swings the
 * right foot over [kSwingStart, kSwingEnd], the reference manager run from the
 * state `initState`, as the solver runs it before a solve.
 */
OptimalControlProblem problemAt(WBMpcInterface& interface,
                                const vector_t& initState,
                                const TargetTrajectories& target) {
  contact_flag_t rightSwing = makeFeetArray(true);
  rightSwing[1] = false;
  const ModeSchedule schedule(
      /*eventTimesInput=*/{-10.0, kSwingStart, kSwingEnd, 10.0},
      /*modeSequenceInput=*/{ModeNumber::kStance, ModeNumber::kStance,
                             stanceLeg2ModeNumber(rightSwing),
                             ModeNumber::kStance, ModeNumber::kStance});
  SwitchedModelReferenceManager& referenceManager =
      *interface.getSwitchedModelReferenceManagerPtr();
  referenceManager.setTargetTrajectories(target);
  referenceManager.getGaitSchedule()->updateModeSchedule(schedule);
  referenceManager.preSolverRun(kStanceTime - 0.2, kSwingEnd + 0.5, initState,
                                ModeNumber::kStance);
  OptimalControlProblem problem(interface.getOptimalControlProblem());
  problem.targetTrajectoriesPtr = &referenceManager.getTargetTrajectories();
  return problem;
}

/**
 * A walking-like state of the G1 at the base pose (x, y, z, yaw, pitch, roll)
 * `basePose`: the joints off the initial posture by a few hundredths of a
 * radian, the base moving and turning slowly, the joints moving.
 */
vector_t walkingState(const WBMpcInterface& interface,
                      const vector6_t& basePose) {
  const WBAccelMpcRobotModel<scalar_t>& model = interface.getMpcRobotModel();
  vector_t state = interface.getInitialState();
  model.setPoseeulerzyxBaseRtWorld(state, basePose);
  const Eigen::Index joints = static_cast<Eigen::Index>(model.getJointDim());
  vector_t jointAngles = model.getJointAngles(state);
  vector_t jointVelocities(joints);
  for (Eigen::Index j = 0; j < joints; ++j) {
    jointAngles(j) += 0.03 * std::sin(1.1 * static_cast<scalar_t>(j));
    jointVelocities(j) = 0.2 * std::cos(0.9 * static_cast<scalar_t>(j));
  }
  model.setJointAngles(state, jointAngles);
  vector_t input =
      vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));
  model.setJointVelocities(state, input, jointVelocities);
  model.setBaseLinearVelocity(state, vector3_t(0.3, -0.1, 0.02));
  model.setAngvelBaseRtWorldEwrtBase(state, vector3_t(0.05, -0.1, 0.3));
  return state;
}

/** The weight on both feet and the joints accelerating: an input near the
 * solver's. */
vector_t walkingInput(WBMpcInterface& interface) {
  const WBAccelMpcRobotModel<scalar_t>& model = interface.getMpcRobotModel();
  vector_t input = weightCompensatingInput(interface.getPinocchioInterface(),
                                           makeFeetArray(true), model);
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    vector6_t wrench = model.getWrenchContactEwrtParam(input, contact);
    wrench.head<2>() += vector2_t(15.0, -8.0);
    wrench.tail<3>() += vector3_t(2.0, -1.5, 0.5);
    model.setWrenchContactEwrtParam(input, wrench, contact);
  }
  const Eigen::Index joints = static_cast<Eigen::Index>(model.getJointDim());
  for (Eigen::Index j = 0; j < joints; ++j)
    input(input.size() - joints + j) = std::sin(0.4 * static_cast<scalar_t>(j));
  return input;
}

/** The target of the points: one knot level at a heading of 1.0, at the initial
 * posture, at rest. */
TargetTrajectories levelTarget(const WBMpcInterface& interface) {
  const WBAccelMpcRobotModel<scalar_t>& model = interface.getMpcRobotModel();
  vector_t knot = interface.getInitialState();
  model.setPoseeulerzyxBaseRtWorld(
      knot, (vector6_t() << 0.05, -0.02,
             model.getPosBaseRtWorldEwrtWorld(knot)(2), 1.0, 0.0, 0.0)
                .finished());
  return TargetTrajectories(
      {0.0}, {knot},
      {vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()))});
}

/** `state` turned about the world vertical by `angle`: the base position,
 * orientation and linear velocity. */
vector_t turnedState(const WBAccelMpcRobotModel<scalar_t>& model,
                     const vector_t& state,
                     scalar_t angle) {
  const matrix3_t rotation =
      Eigen::AngleAxis<scalar_t>(angle, vector3_t::UnitZ()).toRotationMatrix();
  vector_t turned = state;
  model.setPosBaseRtWorldEwrtWorld(
      turned, rotation * model.getPosBaseRtWorldEwrtWorld(state));
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  const vector4_t orientation = state.segment<4>(orientationStart);
  turned.segment<4>(orientationStart) =
      quaternionProduct<scalar_t>(headingRotation(angle), orientation);
  model.setBaseLinearVelocity(turned,
                              rotation * model.getBaseComLinearVelocity(state));
  return turned;
}

/** `input` turned about the world vertical by `angle`: the world-frame force
 * and moment of every contact. */
vector_t turnedInput(const WBAccelMpcRobotModel<scalar_t>& model,
                     const vector_t& input,
                     scalar_t angle) {
  const matrix3_t rotation =
      Eigen::AngleAxis<scalar_t>(angle, vector3_t::UnitZ()).toRotationMatrix();
  vector_t turned = input;
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    const vector6_t wrench = model.getWrenchContactEwrtParam(input, contact);
    vector6_t turnedWrench;
    turnedWrench << rotation * wrench.head<3>(), rotation * wrench.tail<3>();
    model.setWrenchContactEwrtParam(turned, turnedWrench, contact);
  }
  return turned;
}

/** The state derivative `derivative` turned with the state about the world
 * vertical by `angle` (turnedState()). */
vector_t turnedStateDerivative(const WBAccelMpcRobotModel<scalar_t>& model,
                               const vector_t& derivative,
                               scalar_t angle) {
  const matrix3_t rotation =
      Eigen::AngleAxis<scalar_t>(angle, vector3_t::UnitZ()).toRotationMatrix();
  vector_t turned = derivative;
  turned.head<3>() = rotation * derivative.head<3>();
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  const vector4_t rate = derivative.segment<4>(orientationStart);
  turned.segment<4>(orientationStart) =
      quaternionProduct<scalar_t>(headingRotation(angle), rate);
  const Eigen::Index linearVelocityStart =
      static_cast<Eigen::Index>(model.getConfigurationDim());
  turned.segment<3>(linearVelocityStart) =
      rotation * derivative.segment<3>(linearVelocityStart);
  return turned;
}

/** The points of the shipped MPC every property is checked at: in stance and in
 * the swing, at two attitudes. */
std::vector<EvaluationPoint> evaluationPoints(WBMpcInterface& interface) {
  std::vector<EvaluationPoint> points;
  for (const vector6_t& basePose :
       {(vector6_t() << 0.1, -0.05, 0.75, 1.3, 0.08, -0.05).finished(),
        (vector6_t() << -0.2, 0.1, 0.74, -2.9, -0.15, 0.2).finished()}) {
    for (const scalar_t time : {kStanceTime, kSwingTime}) {
      points.push_back(
          EvaluationPoint{.time = time,
                          .state = walkingState(interface, basePose),
                          .input = walkingInput(interface)});
    }
  }
  return points;
}

TEST(WBQuaternionInvariances,
     EveryJacobianIsTheDerivativeAlongTheStateManifold) {
  WBMpcInterface* absl_nullable const interface = shippedG1WholeBodyMpc();
  ASSERT_NE(interface, nullptr);
  const std::shared_ptr<const StateManifold>& manifold =
      interface->getStateManifold();
  ASSERT_NE(manifold, nullptr);
  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(manifold->getTangentDim());
  const TargetTrajectories target = levelTarget(*interface);
  size_t evaluated = 0;
  scalar_t largestError = 0.0;
  for (const EvaluationPoint& point : evaluationPoints(*interface)) {
    OptimalControlProblem problem = problemAt(*interface, point.state, target);
    for (const TermReference& term : registeredTerms(problem)) {
      SCOPED_TRACE(absl::StrCat(term.name, " at t = ", point.time));
      const TermEvaluation evaluation = evaluate(problem, term, point);
      if (!evaluation.active) continue;
      ++evaluated;
      matrix_t tangentJacobian = evaluation.stateJacobian;
      ASSERT_EQ(tangentJacobian.cols(), point.state.size());
      manifold->pullBackStateColumns(point.state, tangentJacobian);
      ASSERT_EQ(tangentJacobian.cols(), tangentDim);
      const scalar_t scale =
          std::max(1.0, tangentJacobian.cwiseAbs().maxCoeff());
      scalar_t error = 0.0;
      for (Eigen::Index i = 0; i < tangentDim; ++i) {
        vector_t step = vector_t::Zero(tangentDim);
        step(i) = 1.0;
        EvaluationPoint plus = point;
        EvaluationPoint minus = point;
        manifold->retract(point.state, step, kFiniteDifferenceStep, plus.state);
        manifold->retract(point.state, step, -kFiniteDifferenceStep,
                          minus.state);
        const vector_t column =
            (evaluate(problem, term, plus, Derivatives::kValue).value -
             evaluate(problem, term, minus, Derivatives::kValue).value) /
            (2.0 * kFiniteDifferenceStep);
        error = std::max(
            error,
            (column - tangentJacobian.col(i)).cwiseAbs().maxCoeff() / scale);
      }
      EXPECT_LE(error, kFiniteDifferenceTolerance);
      largestError = std::max(largestError, error);
    }
  }
  EXPECT_GT(evaluated, 20U)
      << "positive control: the MPC's terms were evaluated";
  RecordProperty("terms_evaluated", static_cast<int>(evaluated));
  RecordProperty("largest_relative_error", absl::StrCat(largestError));
}

TEST(WBQuaternionInvariances, EveryTermReadsTheBaseOrientationAsARotation) {
  WBMpcInterface* absl_nullable const interface = shippedG1WholeBodyMpc();
  ASSERT_NE(interface, nullptr);
  const WBAccelMpcRobotModel<scalar_t>& model = interface->getMpcRobotModel();
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  const TargetTrajectories target = levelTarget(*interface);
  size_t bitwiseUnderNegation = 0;
  size_t evaluated = 0;
  for (const EvaluationPoint& point : evaluationPoints(*interface)) {
    OptimalControlProblem problem = problemAt(*interface, point.state, target);
    const vector4_t xi = point.state.segment<4>(orientationStart);
    for (const TermReference& term : registeredTerms(problem)) {
      SCOPED_TRACE(absl::StrCat(term.name, " at t = ", point.time));
      const TermEvaluation evaluation = evaluate(problem, term, point);
      if (!evaluation.active) continue;
      ++evaluated;
      // The dynamics' quaternion rate 1/2 G(xi) w_B is linear in xi; every
      // other row and every other term reads xi_hat.
      const bool isDynamics = term.collection == TermCollection::kDynamics;
      const scalar_t jacobianScale =
          std::max(1.0, evaluation.stateJacobian.cwiseAbs().maxCoeff());
      const vector_t radial =
          evaluation.stateJacobian.middleCols<4>(orientationStart) * xi;
      EXPECT_LE(withoutQuaternionRate(term, orientationStart, radial)
                    .cwiseAbs()
                    .maxCoeff(),
                1.0e-9 * jacobianScale)
          << "dg/dxi xi";
      for (const scalar_t scale : {0.5, 2.0}) {
        EvaluationPoint scaled = point;
        scaled.state.segment<4>(orientationStart) *= scale;
        const vector_t value =
            evaluate(problem, term, scaled, Derivatives::kValue).value;
        EXPECT_LE(
            relativeDifference(
                withoutQuaternionRate(term, orientationStart, evaluation.value),
                withoutQuaternionRate(term, orientationStart, value)),
            1.0e-10)
            << "c = " << scale;
        if (isDynamics) {
          EXPECT_LE((value.segment<4>(orientationStart) -
                     scale * evaluation.value.segment<4>(orientationStart))
                        .cwiseAbs()
                        .maxCoeff(),
                    1.0e-12);
        }
      }
      EvaluationPoint negated = point;
      negated.state.segment<4>(orientationStart) *= -1.0;
      const vector_t negatedValue =
          evaluate(problem, term, negated, Derivatives::kValue).value;
      EXPECT_LE(
          relativeDifference(
              withoutQuaternionRate(term, orientationStart, evaluation.value),
              withoutQuaternionRate(term, orientationStart, negatedValue)),
          1.0e-12)
          << "-xi";
      if (isDynamics) {
        EXPECT_LE((negatedValue.segment<4>(orientationStart) +
                   evaluation.value.segment<4>(orientationStart))
                      .cwiseAbs()
                      .maxCoeff(),
                  1.0e-15);
      }
      if (withoutQuaternionRate(term, orientationStart, evaluation.value) ==
          withoutQuaternionRate(term, orientationStart, negatedValue)) {
        ++bitwiseUnderNegation;
      }
    }
  }
  RecordProperty("terms_evaluated", static_cast<int>(evaluated));
  RecordProperty("bitwise_under_negation",
                 static_cast<int>(bitwiseUnderNegation));
}

TEST(WBQuaternionInvariances,
     AtAPitchOf90DegreesEveryTermAndOneSqpIterationAreFinite) {
  WBMpcInterface* absl_nullable const interface = shippedG1WholeBodyMpc();
  ASSERT_NE(interface, nullptr);
  const WBAccelMpcRobotModel<scalar_t>& model = interface->getMpcRobotModel();
  const TargetTrajectories target = levelTarget(*interface);
  for (const vector6_t& basePose :
       {(vector6_t() << 0.0, 0.0, 0.75, 0.4, M_PI / 2.0, 0.2).finished(),
        (vector6_t() << 0.0, 0.0, 0.75, -1.0, -M_PI / 2.0, 0.0).finished()}) {
    SCOPED_TRACE(basePose.transpose());
    const vector_t state = walkingState(*interface, basePose);
    ASSERT_NEAR(std::abs(model.getEulerzyxBaseRtWorld(state)(1)), M_PI / 2.0,
                1.0e-6);
    for (const scalar_t time : {kStanceTime, kSwingTime}) {
      OptimalControlProblem problem = problemAt(*interface, state, target);
      const EvaluationPoint point{
          .time = time, .state = state, .input = walkingInput(*interface)};
      for (const TermReference& term : registeredTerms(problem)) {
        const TermEvaluation evaluation = evaluate(problem, term, point);
        if (!evaluation.active) continue;
        EXPECT_TRUE(evaluation.value.allFinite())
            << term.name << " at t = " << time;
        EXPECT_TRUE(evaluation.stateJacobian.allFinite())
            << term.name << " at t = " << time;
      }
    }

    // One SQP iteration of the shipped settings from there, towards the level
    // target.
    sqp::Settings settings = interface->sqpSettings();
    settings.sqpIteration = 1;
    settings.printSolverStatistics = false;
    settings.printSolverStatus = false;
    settings.printLinesearch = false;
    SqpSolver solver(settings, interface->getOptimalControlProblem(),
                     interface->getInitializer());
    problemAt(*interface, state, target);
    solver.setReferenceManager(interface->getReferenceManagerPtr());
    const scalar_t finalTime =
        kSwingStart + interface->mpcSettings().timeHorizon_;
    solver.run(kSwingStart, state, /*initMode=*/ModeNumber::kStance, finalTime);
    PrimalSolution solution;
    solver.getPrimalSolution(finalTime, &solution);
    ASSERT_FALSE(solution.stateTrajectory_.empty());
    for (size_t k = 0; k < solution.stateTrajectory_.size(); ++k) {
      EXPECT_TRUE(solution.stateTrajectory_[k].allFinite()) << "node " << k;
      EXPECT_NEAR(solution.stateTrajectory_[k]
                      .segment<4>(static_cast<Eigen::Index>(
                          model.getBaseOrientationStartIndex()))
                      .norm(),
                  1.0, 1.0e-12)
          << "node " << k;
    }
    for (size_t k = 0; k < solution.inputTrajectory_.size(); ++k)
      EXPECT_TRUE(solution.inputTrajectory_[k].allFinite()) << "node " << k;
    EXPECT_TRUE(std::isfinite(solver.getPerformanceIndeces().cost));
    EXPECT_TRUE(std::isfinite(solver.getPerformanceIndeces().merit));
  }
}

TEST(WBQuaternionInvariances,
     EveryTermIsEquivariantUnderATurnOfTheWorldAboutItsVertical) {
  WBMpcInterface* absl_nullable const interface = shippedG1WholeBodyMpc();
  ASSERT_NE(interface, nullptr);
  const WBAccelMpcRobotModel<scalar_t>& model = interface->getMpcRobotModel();
  const TargetTrajectories target = levelTarget(*interface);
  size_t compared = 0;
  for (const EvaluationPoint& point : evaluationPoints(*interface)) {
    // Each problem in its own references, as the solver would run them from the
    // state.
    std::vector<TermEvaluation> original;
    OptimalControlProblem problem = problemAt(*interface, point.state, target);
    const std::vector<TermReference> terms = registeredTerms(problem);
    for (const TermReference& term : terms)
      original.push_back(evaluate(problem, term, point));
    for (const scalar_t angle :
         {M_PI - 0.01, -M_PI + 0.01, 10.0 * M_PI + 0.1}) {
      SCOPED_TRACE(absl::StrCat("turned by ", angle, " at t = ", point.time));
      const EvaluationPoint turned{
          .time = point.time,
          .state = turnedState(model, point.state, angle),
          .input = turnedInput(model, point.input, angle)};
      TargetTrajectories turnedTarget = target;
      for (vector_t& knot : turnedTarget.stateTrajectory)
        knot = turnedState(model, knot, angle);
      OptimalControlProblem turnedProblem =
          problemAt(*interface, turned.state, turnedTarget);
      for (size_t i = 0; i < terms.size(); ++i) {
        SCOPED_TRACE(terms[i].name);
        const TermEvaluation evaluation =
            evaluate(turnedProblem, terms[i], turned, Derivatives::kValue);
        ASSERT_EQ(evaluation.active, original[i].active);
        if (!evaluation.active) continue;
        ++compared;
        if (terms[i].collection == TermCollection::kDynamics) {
          EXPECT_LE(relativeDifference(
                        turnedStateDerivative(model, original[i].value, angle),
                        evaluation.value),
                    1.0e-9);
        } else if (isCost(terms[i].collection)) {
          EXPECT_LE(relativeDifference(original[i].value, evaluation.value),
                    1.0e-9);
        } else {
          EXPECT_NEAR(evaluation.value.norm(), original[i].value.norm(),
                      1.0e-9 * std::max(1.0, original[i].value.norm()));
        }
      }
    }
  }
  RecordProperty("terms_compared", static_cast<int>(compared));
}

}  // namespace
}  // namespace ocs2::humanoid
