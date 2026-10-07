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
 * The SQP on a state manifold, end to end: a free rigid body whose attitude is a unit quaternion (ambient state 7,
 * tangent 6) tracks a heading reference that turns 720 degrees, in closed loop with a simulated plant (the policy rolled
 * out by a projected TimeTriggeredRollout between solves), as a real-time MPC would run it.
 *
 * What an Euler-angle state could not do and the manifold must: the heading crosses +-pi four times without a transient
 * (the initial gap x(t0) (-) x_init stays small: the state stores no angle, so nothing wraps), the heading increases
 * monotonically to 4 pi, every iterate stays on the unit sphere, and starting from -xi instead of xi (the same attitude)
 * produces the same inputs.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"

#include <ocs2_core/control/ManifoldLinearController.h>
#include <ocs2_core/cost/StateCost.h>
#include <ocs2_core/cost/StateInputCost.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_core/manifold/ProductStateManifold.h>
#include <ocs2_core/manifold/UnitQuaternionMath.h>
#include <ocs2_oc/oc_data/StateTrajectoryInterpolation.h>
#include <ocs2_oc/rollout/TimeTriggeredRollout.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_oc/test/RigidBodyAttitudeDynamics.h>

#include "ocs2_sqp/SqpSolver.h"

namespace ocs2 {
namespace {

using manifold_test::RigidBodyAttitudeDynamics;
using quaternion_t = quaternion_coeffs_t<scalar_t>;
using rotation_t = rotation_vector_t<scalar_t>;

constexpr scalar_t kPi = 3.14159265358979323846;
constexpr scalar_t kHeadingRate = 1.0;                        // [rad/s]
constexpr scalar_t kTurnDuration = 4.0 * kPi / kHeadingRate;  // 720 degrees
constexpr scalar_t kMpcPeriod = 0.05;
constexpr scalar_t kHorizon = 1.0;

/** The reference: a level attitude turning about the world z axis at kHeadingRate, then holding 4 pi. */
quaternion_t referenceAttitude(scalar_t t) {
  const scalar_t heading = kHeadingRate * std::min(std::max(t, 0.0), kTurnDuration);
  return quaternion_t(0.0, 0.0, std::sin(0.5 * heading), std::cos(0.5 * heading));
}

rotation_t referenceAngularVelocity(scalar_t t) {
  return rotation_t(/*x=*/0.0, /*y=*/0.0, /*z=*/(t < kTurnDuration) ? kHeadingRate : 0.0);
}

/**
 * The attitude error r = Log(xi_ref^-1 (x) xi) and the angular-velocity error, with their Gauss-Newton approximation.
 * The tangent Jacobian of r is Jr^-1(r); it is lifted to the ambient quaternion columns with E+(xi), so the cost returns
 * ambient derivatives as every term does, and sees the quaternion only through its rotation.
 */
ScalarFunctionQuadraticApproximation attitudeTrackingApproximation(scalar_t t,
                                                                   const vector_t& x,
                                                                   scalar_t attitudeWeight,
                                                                   scalar_t rateWeight) {
  const quaternion_t xi = x.head<4>();
  const rotation_t error = quaternionLog(quaternionProduct(quaternionConjugate(referenceAttitude(t)), xi));
  const rotation_t rateError = x.tail<3>() - referenceAngularVelocity(t);
  const Eigen::Matrix<scalar_t, 3, 4> errorJacobian = so3RightJacobianInverse(error) * quaternionTangentMapPseudoInverse(xi);

  ScalarFunctionQuadraticApproximation approximation;
  approximation.f = 0.5 * attitudeWeight * error.squaredNorm() + 0.5 * rateWeight * rateError.squaredNorm();
  approximation.dfdx.resize(7);
  approximation.dfdx.head<4>() = attitudeWeight * errorJacobian.transpose() * error;
  approximation.dfdx.tail<3>() = rateWeight * rateError;
  approximation.dfdxx.setZero(7, 7);
  approximation.dfdxx.topLeftCorner<4, 4>() = attitudeWeight * errorJacobian.transpose() * errorJacobian;
  approximation.dfdxx.bottomRightCorner<3, 3>() = rateWeight * Eigen::Matrix3d::Identity();
  return approximation;
}

class AttitudeTrackingCost final : public StateInputCost {
 public:
  AttitudeTrackingCost* absl_nonnull clone() const override { return new AttitudeTrackingCost(*this); }
  scalar_t getValue(scalar_t t,
                    const vector_t& x,
                    const vector_t& u,
                    const TargetTrajectories& /*targets*/,
                    const PreComputation& /*preComp*/) const override {
    return attitudeTrackingApproximation(t, x, kAttitudeWeight, kRateWeight).f + 0.5 * kInputWeight * u.squaredNorm();
  }
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t t,
                                                                 const vector_t& x,
                                                                 const vector_t& u,
                                                                 const TargetTrajectories& /*targets*/,
                                                                 const PreComputation& /*preComp*/) const override {
    ScalarFunctionQuadraticApproximation approximation = attitudeTrackingApproximation(t, x, kAttitudeWeight, kRateWeight);
    approximation.f += 0.5 * kInputWeight * u.squaredNorm();
    approximation.dfdu = kInputWeight * u;
    approximation.dfduu = kInputWeight * matrix_t::Identity(3, 3);
    approximation.dfdux.setZero(3, 7);
    return approximation;
  }

 private:
  static constexpr scalar_t kAttitudeWeight = 50.0;
  static constexpr scalar_t kRateWeight = 2.0;
  static constexpr scalar_t kInputWeight = 0.05;
};

class AttitudeTrackingFinalCost final : public StateCost {
 public:
  AttitudeTrackingFinalCost* absl_nonnull clone() const override { return new AttitudeTrackingFinalCost(*this); }
  scalar_t getValue(scalar_t t,
                    const vector_t& x,
                    const TargetTrajectories& /*targets*/,
                    const PreComputation& /*preComp*/) const override {
    return attitudeTrackingApproximation(t, x, kAttitudeWeight, kRateWeight).f;
  }
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t t,
                                                                 const vector_t& x,
                                                                 const TargetTrajectories& /*targets*/,
                                                                 const PreComputation& /*preComp*/) const override {
    return attitudeTrackingApproximation(t, x, kAttitudeWeight, kRateWeight);
  }

 private:
  static constexpr scalar_t kAttitudeWeight = 200.0;
  static constexpr scalar_t kRateWeight = 5.0;
};

sqp::Settings solverSettings() {
  sqp::Settings settings;
  settings.dt = kMpcPeriod;
  settings.sqpIteration = 3;
  settings.integratorType = SensitivityIntegratorType::RK4;
  settings.useFeedbackPolicy = true;
  settings.createValueFunction = false;
  settings.printSolverStatistics = false;
  settings.enableLogging = false;
  settings.nThreads = 1;
  return settings;
}

struct AttitudeProblem {
  AttitudeProblem() : dynamics(/*withPosition=*/false), manifold(dynamics.getStateManifold()) {
    problem.dynamicsPtr.reset(dynamics.clone());
    problem.costPtr->add("attitudeTracking", std::make_unique<AttitudeTrackingCost>());
    problem.finalCostPtr->add("attitudeTracking", std::make_unique<AttitudeTrackingFinalCost>());
    problem.stateManifoldPtr = manifold;
    problem.targetTrajectoriesPtr = &targets;
  }

  RigidBodyAttitudeDynamics dynamics;
  std::shared_ptr<const ProductStateManifold> manifold;
  TargetTrajectories targets = TargetTrajectories({0.0}, {vector_t::Zero(7)}, {vector_t::Zero(3)});
  OptimalControlProblem problem;
};

struct ClosedLoop {
  std::vector<scalar_t> times;                // the end of each MPC step
  std::vector<scalar_t> headings;             // unwrapped twist heading of the plant relative to the start, per MPC step
  std::vector<scalar_t> initialRotationGaps;  // rotation part of x(t0) (-) x_init of each warm-started solve
  std::vector<vector_t> solverInitialStateGaps;  // SqpSolver::getInitialStateGap() of the same solves
  vector_array_t plantInputs;                 // the policy's input at the start of each step
  vector_t finalState;
  scalar_t maximumNormError = 0.0;  // over every plan state of every solve
};

scalar_t twistHeading(const vector_t& x) {
  return 2.0 * std::atan2(x(2), x(3));
}

scalar_t wrapToPi(scalar_t angle) {
  return std::remainder(angle, 2.0 * kPi);
}

ClosedLoop runClosedLoop(const vector_t& initState, scalar_t duration) {
  AttitudeProblem attitude;
  const DefaultInitializer initializer(3);
  SqpSolver solver(solverSettings(), attitude.problem, initializer);
  solver.setReferenceManager(std::make_shared<ReferenceManager>(attitude.targets));
  rollout::Settings rolloutSettings;
  rolloutSettings.integratorType = IntegratorType::RK4;
  rolloutSettings.timeStep = 0.005;
  TimeTriggeredRollout plant(attitude.dynamics, rolloutSettings);
  plant.setStateManifold(attitude.manifold);

  ClosedLoop result;
  vector_t x = initState;
  scalar_t heading = 0.0;
  PrimalSolution previous;
  const size_t numSteps = static_cast<size_t>(std::round(duration / kMpcPeriod));
  for (size_t step = 0; step < numSteps; ++step) {
    const scalar_t t = step * kMpcPeriod;
    if (!previous.timeTrajectory_.empty()) {
      const vector_t warmStart =
          interpolateStateTrajectory(attitude.manifold.get(), t, previous.timeTrajectory_, previous.stateTrajectory_);
      result.initialRotationGaps.push_back(attitude.manifold->difference(warmStart, x).head<3>().norm());
    }
    const bool warmStarted = !previous.timeTrajectory_.empty();
    solver.run(t, x, /*initMode=*/0, t + kHorizon);
    if (warmStarted) result.solverInitialStateGaps.push_back(solver.getInitialStateGap());
    previous = solver.primalSolution(t + kHorizon);
    for (const vector_t& planState : previous.stateTrajectory_) {
      result.maximumNormError = std::max(result.maximumNormError, std::abs(planState.head<4>().norm() - 1.0));
    }
    EXPECT_EQ(previous.controllerPtr_->getType(), ControllerType::MANIFOLD_LINEAR);
    result.plantInputs.push_back(previous.controllerPtr_->computeInput(t, x));

    // The plant: the policy, rolled out on the projected rigid body.
    scalar_array_t timeTrajectory;
    size_array_t postEventIndices;
    vector_array_t stateTrajectory;
    vector_array_t inputTrajectory;
    ModeSchedule modeSchedule;
    x = plant.run(t, x, t + kMpcPeriod, previous.controllerPtr_.get(), modeSchedule, timeTrajectory, postEventIndices, stateTrajectory,
                  inputTrajectory);
    const scalar_t newHeading = twistHeading(x);
    heading += wrapToPi(newHeading - twistHeading(stateTrajectory.front()));
    result.headings.push_back(heading);
    result.times.push_back(t + kMpcPeriod);
  }
  result.finalState = x;
  return result;
}

vector_t initialState(scalar_t sign) {
  vector_t x = vector_t::Zero(7);
  x(3) = sign;  // identity attitude, xi = +-(0, 0, 0, 1), at rest
  return x;
}

TEST(SqpOnSO3, TracksA720DegreeTurnThroughEveryHalfTurnCrossing) {
  const ClosedLoop loop = runClosedLoop(initialState(1.0), kTurnDuration + 2.0);

  // Monotone heading while the reference turns, reaching 4 pi.
  ASSERT_FALSE(loop.headings.empty());
  for (size_t k = 1; k < loop.headings.size() && loop.times[k] <= kTurnDuration; ++k) {
    ASSERT_GE(loop.headings[k], loop.headings[k - 1] - 1e-9) << "the heading went back at t = " << loop.times[k];
  }
  EXPECT_NEAR(loop.headings.back(), 4.0 * kPi, 0.1);
  // ... and the attitude is the reference's (the identity) at the end.
  const quaternion_t finalAttitude = loop.finalState.head<4>();
  EXPECT_LT(quaternionLog(finalAttitude).norm(), 0.05);

  // No transient at the +-pi crossings: the warm start is always close to the measured state.
  scalar_t largestGap = 0.0;
  for (const scalar_t gap : loop.initialRotationGaps) {
    largestGap = std::max(largestGap, gap);
  }
  EXPECT_LT(largestGap, 0.1);

  // Every iterate on the unit sphere.
  EXPECT_LT(loop.maximumNormError, 1e-12);

  std::cout << "[SqpOnSO3] final heading " << loop.headings.back() << " rad (4 pi = " << 4.0 * kPi << "), largest initial rotation gap "
            << largestGap << " rad, largest |norm - 1| of a plan state " << loop.maximumNormError << ", " << loop.headings.size()
            << " solves\n";
}

TEST(SqpOnSO3, RecordsTheInitialStateGapOfTheFirstIterationOnTheManifold) {
  // getInitialStateGap() is delta_x0 = difference(x[0], x_init) with x[0] the warm start slerped at the initial time: the
  // tangent gap the QP eliminates, whose rotation part is the gap measured from outside.
  const ClosedLoop loop = runClosedLoop(initialState(1.0), /*duration=*/1.0);
  ASSERT_EQ(loop.solverInitialStateGaps.size(), loop.initialRotationGaps.size());
  ASSERT_FALSE(loop.solverInitialStateGaps.empty());
  for (size_t k = 0; k < loop.solverInitialStateGaps.size(); ++k) {
    ASSERT_EQ(loop.solverInitialStateGaps[k].size(), 6) << k;  // the tangent of the attitude and the body rate
    EXPECT_NEAR(loop.solverInitialStateGaps[k].head<3>().norm(), loop.initialRotationGaps[k], 1e-12) << k;
  }
}

TEST(SqpOnSO3, TheOtherQuaternionOfTheSameAttitudeGivesTheSameInputs) {
  const scalar_t duration = 4.0;  // through the first half-turn crossing
  const ClosedLoop plus = runClosedLoop(initialState(1.0), duration);
  const ClosedLoop minus = runClosedLoop(initialState(-1.0), duration);
  ASSERT_EQ(plus.plantInputs.size(), minus.plantInputs.size());
  for (size_t k = 0; k < plus.plantInputs.size(); ++k) {
    EXPECT_LT((plus.plantInputs[k] - minus.plantInputs[k]).cwiseAbs().maxCoeff(), 1e-12) << "step " << k;
  }
  EXPECT_NEAR(plus.headings.back(), minus.headings.back(), 1e-12);
}

TEST(SqpOnSO3, AWarmStartMoreThanAQuarterTurnAwayIsDiscarded) {
  AttitudeProblem attitude;
  const DefaultInitializer initializer(3);
  const std::shared_ptr<ReferenceManager> referenceManager = std::make_shared<ReferenceManager>(attitude.targets);
  SqpSolver warm(solverSettings(), attitude.problem, initializer);
  warm.setReferenceManager(referenceManager);
  warm.run(/*initTime=*/0.0, initialState(1.0), /*initMode=*/0, /*finalTime=*/kHorizon);

  // From an attitude 2.5 rad away from the plan, the warm-started solve is the solve of a fresh solver: the warm start
  // was dropped.
  vector_t far = initialState(1.0);
  far.head<4>() = quaternionExp(rotation_t(2.5, 0.0, 0.0));
  warm.run(/*initTime=*/kMpcPeriod, far, /*initMode=*/0, /*finalTime=*/kMpcPeriod + kHorizon);
  SqpSolver fresh(solverSettings(), attitude.problem, initializer);
  fresh.setReferenceManager(referenceManager);
  fresh.run(/*initTime=*/kMpcPeriod, far, /*initMode=*/0, /*finalTime=*/kMpcPeriod + kHorizon);
  EXPECT_EQ(warm.primalSolution(kMpcPeriod + kHorizon).stateTrajectory_, fresh.primalSolution(kMpcPeriod + kHorizon).stateTrajectory_);

  // From close by (0.3 rad), it is kept: the solves differ.
  SqpSolver warmNear(solverSettings(), attitude.problem, initializer);
  warmNear.setReferenceManager(referenceManager);
  warmNear.run(/*initTime=*/0.0, initialState(1.0), /*initMode=*/0, /*finalTime=*/kHorizon);
  vector_t near = initialState(1.0);
  near.head<4>() = quaternionExp(rotation_t(0.3, 0.0, 0.0));
  warmNear.run(/*initTime=*/kMpcPeriod, near, /*initMode=*/0, /*finalTime=*/kMpcPeriod + kHorizon);
  SqpSolver freshNear(solverSettings(), attitude.problem, initializer);
  freshNear.setReferenceManager(referenceManager);
  freshNear.run(/*initTime=*/kMpcPeriod, near, /*initMode=*/0, /*finalTime=*/kMpcPeriod + kHorizon);
  EXPECT_NE(warmNear.primalSolution(kMpcPeriod + kHorizon).stateTrajectory_,
            freshNear.primalSolution(kMpcPeriod + kHorizon).stateTrajectory_);
}

TEST(SqpOnSO3, TheValueFunctionIsRefusedOnAManifold) {
  AttitudeProblem attitude;
  const DefaultInitializer initializer(3);
  sqp::Settings settings = solverSettings();
  settings.createValueFunction = true;
  EXPECT_THROW({ SqpSolver solver(settings, attitude.problem, initializer); }, std::invalid_argument);
}

}  // namespace
}  // namespace ocs2
