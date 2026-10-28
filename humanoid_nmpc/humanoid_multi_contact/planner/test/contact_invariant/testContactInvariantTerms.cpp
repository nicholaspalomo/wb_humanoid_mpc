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

// The terms of the contact-invariant problem (design section 4.2) against
// finite differences of their values: the contact-invariant cost, the force
// gate, the activation rate, the quasi-static momentum rate, the activation
// bounds, the feature rows as constraints, the augmented dynamics and the
// wrapped physical terms; the contact features' Jacobian against finite
// differences of the features; and the Hessians' positive semidefiniteness.

#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "Eigen/Eigenvalues"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/ComputationRequest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_multi_contact/planner/contact_invariant/ActivationBoundsConstraint.h"
#include "humanoid_multi_contact/planner/contact_invariant/ActivationRateCost.h"
#include "humanoid_multi_contact/planner/contact_invariant/AugmentedDynamics.h"
#include "humanoid_multi_contact/planner/contact_invariant/AugmentedPreComputation.h"
#include "humanoid_multi_contact/planner/contact_invariant/AugmentedStateCost.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureConstraint.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureRows.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactForceGateCost.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantCost.h"
#include "humanoid_multi_contact/planner/contact_invariant/MomentumRateCost.h"
#include "humanoid_multi_contact/planner/contact_invariant/PsdProjection.h"
#include "humanoid_nmpc/humanoid_multi_contact/planner/test/contact_invariant/BallProblem.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr double kStep = 1.0e-6;
constexpr double kGradientTolerance = 1.0e-4;

/** The augmented pre-computation of the ball. */
std::unique_ptr<AugmentedPreComputation> ballPreComputation() {
  const std::string folder =
      (std::filesystem::path(::testing::TempDir()) / "cppad").string();
  absl::StatusOr<std::unique_ptr<ContactFeatureFunction>> features =
      ballFeatures(ballFloor(), "ball_terms_features", folder);
  EXPECT_TRUE(features.ok()) << features.status();
  return std::make_unique<AugmentedPreComputation>(
      ballLayout(), std::make_unique<PreComputation>(), *std::move(features));
}

/** An augmented state of the ball: center at `height`, moving, with c. */
vector_t ballState(double height, double activation) {
  vector_t x(ballLayout().stateDim());
  x << 0.1, -0.05, height, 0.3, -0.2, -0.4, activation;
  return x;
}

/** An augmented input: a force and an activation rate. */
vector_t ballInput() {
  vector_t u(ballLayout().inputDim());
  u << 1.5, -0.7, 8.0, 0.3;
  return u;
}

/** The value of a state-input cost at (x, u), requested first. */
using CostValue = std::function<double(const vector_t&, const vector_t&)>;

CostValue costValue(const StateInputCost& cost, PreComputation& pre) {
  return [&cost, &pre](const vector_t& x, const vector_t& u) {
    pre.request(Request::Cost, /*t=*/0.0, x, u);
    return cost.getValue(/*time=*/0.0, x, u, TargetTrajectories(), pre);
  };
}

/** Expects the gradient of `approximation` to be the central difference of
 * `value` at (x, u). */
void expectGradient(const CostValue& value,
                    const vector_t& x,
                    const vector_t& u,
                    const ScalarFunctionQuadraticApproximation& approximation) {
  for (int i = 0; i < x.size(); ++i) {
    vector_t plus = x;
    vector_t minus = x;
    plus(i) += kStep;
    minus(i) -= kStep;
    const double numeric = (value(plus, u) - value(minus, u)) / (2.0 * kStep);
    EXPECT_NEAR(approximation.dfdx(i), numeric,
                kGradientTolerance * (1.0 + std::abs(numeric)))
        << "state " << i;
  }
  for (int i = 0; i < u.size(); ++i) {
    vector_t plus = u;
    vector_t minus = u;
    plus(i) += kStep;
    minus(i) -= kStep;
    const double numeric = (value(x, plus) - value(x, minus)) / (2.0 * kStep);
    EXPECT_NEAR(approximation.dfdu(i), numeric,
                kGradientTolerance * (1.0 + std::abs(numeric)))
        << "input " << i;
  }
}

/** Expects the Hessian of `approximation` over (x, u) to be positive
 * semidefinite. */
void expectPositiveSemidefinite(
    const ScalarFunctionQuadraticApproximation& approximation) {
  const int nx = static_cast<int>(approximation.dfdx.size());
  const int nu = static_cast<int>(approximation.dfdu.size());
  matrix_t hessian(nx + nu, nx + nu);
  hessian << approximation.dfdxx, approximation.dfdux.transpose(),
      approximation.dfdux, approximation.dfduu;
  const Eigen::SelfAdjointEigenSolver<matrix_t> solver(hessian);
  EXPECT_GT(solver.eigenvalues().minCoeff(), -1.0e-9);
}

/** Checks a cost's gradient and Hessian at the ball's points. */
void checkCost(const StateInputCost& cost) {
  std::unique_ptr<AugmentedPreComputation> pre = ballPreComputation();
  for (const double height : {0.12, 0.35}) {
    for (const double activation : {0.0, 0.3, 0.9}) {
      const vector_t x = ballState(height, activation);
      const vector_t u = ballInput();
      pre->request(Request::Cost + Request::Approximation, /*t=*/0.0, x, u);
      const ScalarFunctionQuadraticApproximation approximation =
          cost.getQuadraticApproximation(/*time=*/0.0, x, u,
                                         TargetTrajectories(), *pre);
      EXPECT_NEAR(approximation.f,
                  cost.getValue(/*time=*/0.0, x, u, TargetTrajectories(), *pre),
                  1.0e-12);
      if (activation == 0.0) {
        // c = 0 is the kink of the clamp; only the other variables.
        vector_t xc = x;
        xc(ballLayout().activationIndex(0)) = 1.0e-3;
        pre->request(Request::Cost + Request::Approximation, /*t=*/0.0, xc, u);
        expectGradient(costValue(cost, *pre), xc, u,
                       cost.getQuadraticApproximation(
                           /*time=*/0.0, xc, u, TargetTrajectories(), *pre));
      } else {
        expectGradient(costValue(cost, *pre), x, u, approximation);
      }
      expectPositiveSemidefinite(approximation);
    }
  }
}

TEST(ContactInvariantTerms, TheContactInvariantCostMatchesFiniteDifferences) {
  checkCost(ContactInvariantCost(
      ballLayout(), ContactInvariantWeights{
                        .weight = 30.0, .slipWeight = 0.5, .tiltWeight = 2.0}));
}

TEST(ContactInvariantTerms, TheContactInvariantCostIsCOfTheResidual) {
  std::unique_ptr<AugmentedPreComputation> pre = ballPreComputation();
  const ContactInvariantCost cost(
      ballLayout(), ContactInvariantWeights{
                        .weight = 2.0, .slipWeight = 0.5, .tiltWeight = 0.0});
  const vector_t x = ballState(/*height=*/0.35, /*activation=*/0.25);
  const vector_t u = ballInput();
  pre->request(Request::Cost, /*t=*/0.0, x, u);
  // d = 0.35 - 0.1 = 0.25, v_t = (0.3, -0.2, 0).
  const double residual = 0.25 * 0.25 + 0.5 * (0.3 * 0.3 + 0.2 * 0.2);
  EXPECT_NEAR(cost.getValue(/*time=*/0.0, x, u, TargetTrajectories(), *pre),
              2.0 * 0.25 * residual, 1.0e-9);
  EXPECT_NEAR(cost.residual(pre->features().f, /*candidate=*/0), residual,
              1.0e-9);
  // A candidate that does not claim contact pays nothing.
  const vector_t free = ballState(/*height=*/0.35, /*activation=*/-0.1);
  pre->request(Request::Cost, /*t=*/0.0, free, u);
  EXPECT_EQ(cost.getValue(/*time=*/0.0, free, u, TargetTrajectories(), *pre),
            0.0);
}

TEST(ContactInvariantTerms, TheForceGateMatchesFiniteDifferences) {
  checkCost(ContactForceGateCost(
      ballLayout(),
      ForceGateWeights{
          .k0 = 3.0, .k1 = 0.05, .forceScale = 9.81, .momentArm = 0.1}));
}

TEST(ContactInvariantTerms, TheForceGateIsTheScaledForceOverTheGate) {
  const ContactForceGateCost cost(
      ballLayout(),
      ForceGateWeights{
          .k0 = 2.0, .k1 = 0.5, .forceScale = 2.0, .momentArm = 1.0});
  const PreComputation pre;
  const vector_t x = ballState(/*height=*/0.35, /*activation=*/0.5);
  const vector_t u = ballInput();
  const double scaled = (u.head<3>() / 2.0).squaredNorm();
  EXPECT_NEAR(cost.getValue(/*time=*/0.0, x, u, TargetTrajectories(), pre),
              2.0 * scaled / (0.25 + 0.5), 1.0e-12);
}

TEST(ContactInvariantTerms, TheActivationRateMatchesFiniteDifferences) {
  checkCost(ActivationRateCost(ballLayout(), /*weight=*/0.7));
}

TEST(ContactInvariantTerms, TheMomentumRateMatchesFiniteDifferences) {
  checkCost(MomentumRateCost(ballLayout(), /*linearWeight=*/0.3,
                             /*angularWeight=*/0.1));
}

TEST(ContactInvariantTerms, TheFeaturesJacobianMatchesFiniteDifferences) {
  const std::string folder =
      (std::filesystem::path(::testing::TempDir()) / "cppad").string();
  absl::StatusOr<std::unique_ptr<ContactFeatureFunction>> features =
      ballFeatures(ballFloor(), "ball_terms_jacobian", folder);
  ASSERT_TRUE(features.ok()) << features.status();
  const vector_t x =
      ballState(/*height=*/0.2, /*activation=*/0.0).head(kBallStateDim);
  const vector_t u = ballInput().head(kBallInputDim);
  const VectorFunctionLinearApproximation approximation =
      (*features)->linearApproximation(x, u);
  ASSERT_EQ(approximation.f.size(), numFeatureRows(1));
  for (int i = 0; i < x.size(); ++i) {
    vector_t plus = x;
    vector_t minus = x;
    plus(i) += kStep;
    minus(i) -= kStep;
    const vector_t numeric =
        ((*features)->value(plus, u) - (*features)->value(minus, u)) /
        (2.0 * kStep);
    EXPECT_LT((approximation.dfdx.col(i) - numeric).norm(), 1.0e-6)
        << "state " << i;
  }
  for (int i = 0; i < u.size(); ++i) {
    vector_t plus = u;
    vector_t minus = u;
    plus(i) += kStep;
    minus(i) -= kStep;
    const vector_t numeric =
        ((*features)->value(x, plus) - (*features)->value(x, minus)) /
        (2.0 * kStep);
    EXPECT_LT((approximation.dfdu.col(i) - numeric).norm(), 1.0e-6)
        << "input " << i;
  }
  // The rows of a ball 0.1 m above the floor: d, v_t, the cone, the
  // clearance and the acceleration.
  EXPECT_NEAR(approximation.f(featureRow(/*candidate=*/0, kDistanceFeature)),
              0.1, 1.0e-9);
  EXPECT_NEAR(approximation.f(featureRow(/*candidate=*/0, kClearanceFeature)),
              0.1, 1.0e-9);
  EXPECT_NEAR(approximation.f(featureRow(/*candidate=*/0, kSlipFeature) + 2),
              0.0, 1.0e-12);
  EXPECT_NEAR(approximation.f(momentumRateRow(1) + 2), 8.0 - 9.81, 1.0e-9);
}

TEST(ContactInvariantTerms, TheFeatureConstraintIsItsRows) {
  std::unique_ptr<AugmentedPreComputation> pre = ballPreComputation();
  const ContactFeatureConstraint constraint(
      {featureRow(/*candidate=*/0, kConeFeature),
       featureRow(/*candidate=*/0, kClearanceFeature)});
  const vector_t x = ballState(/*height=*/0.3, /*activation=*/0.5);
  const vector_t u = ballInput();
  pre->request(Request::Constraint + Request::Approximation, /*t=*/0.0, x, u);
  const VectorFunctionLinearApproximation linear =
      constraint.getLinearApproximation(/*time=*/0.0, x, u, *pre);
  ASSERT_EQ(linear.f.size(), 2);
  EXPECT_DOUBLE_EQ(linear.f(1), pre->features().f(featureRow(
                                    /*candidate=*/0, kClearanceFeature)));
  EXPECT_TRUE(linear.dfdx.row(0).isApprox(
      pre->features().dfdx.row(featureRow(/*candidate=*/0, kConeFeature))));
  EXPECT_EQ(linear.dfdx.cols(), ballLayout().stateDim());
  EXPECT_EQ(linear.dfdu.cols(), ballLayout().inputDim());
  // The cone of an upward force inside it is positive.
  EXPECT_GT(linear.f(0), 0.0);
}

TEST(ContactInvariantTerms, TheActivationBoundsAreCAndOneMinusC) {
  // The rows of one candidate: c, 1 - c, and the support.
  constexpr int kLower = 0;
  constexpr int kUpper = 1;
  constexpr int kSupport = 2;
  const ActivationBoundsConstraint bounds(ballLayout(), /*minimumSupport=*/0.0);
  const PreComputation pre;
  const vector_t x = ballState(/*height=*/0.3, /*activation=*/0.25);
  const VectorFunctionLinearApproximation linear =
      bounds.getLinearApproximation(/*time=*/0.0, x, pre);
  ASSERT_EQ(linear.f.size(), 2);
  EXPECT_DOUBLE_EQ(linear.f(0), 0.25);
  EXPECT_DOUBLE_EQ(linear.f(1), 0.75);
  EXPECT_DOUBLE_EQ(linear.dfdx.col(ballLayout().activationIndex(0))(kLower),
                   1.0);
  EXPECT_DOUBLE_EQ(linear.dfdx.col(ballLayout().activationIndex(0))(kUpper),
                   -1.0);
  // With a minimum support: sum c - 0.2.
  const ActivationBoundsConstraint supported(ballLayout(),
                                             /*minimumSupport=*/0.2);
  const VectorFunctionLinearApproximation withSupport =
      supported.getLinearApproximation(/*time=*/0.0, x, pre);
  ASSERT_EQ(withSupport.f.size(), 3);
  EXPECT_NEAR(withSupport.f(2), 0.05, 1.0e-12);
  EXPECT_DOUBLE_EQ(
      withSupport.dfdx.col(ballLayout().activationIndex(0))(kSupport), 1.0);
}

TEST(ContactInvariantTerms, TheAugmentedDynamicsAddTheActivationRates) {
  AugmentedDynamics dynamics(ballLayout(), std::make_unique<BallDynamics>());
  const PreComputation pre;
  const vector_t x = ballState(/*height=*/0.3, /*activation=*/0.25);
  const vector_t u = ballInput();
  const VectorFunctionLinearApproximation linear =
      dynamics.linearApproximation(/*time=*/0.0, x, u, pre);
  EXPECT_DOUBLE_EQ(linear.f(6), 0.3);
  EXPECT_NEAR(linear.f(5), 8.0 - 9.81, 1.0e-12);
  for (int i = 0; i < x.size(); ++i) {
    vector_t plus = x;
    vector_t minus = x;
    plus(i) += kStep;
    minus(i) -= kStep;
    const vector_t numeric =
        (dynamics.computeFlowMap(/*time=*/0.0, plus, u, pre) -
         dynamics.computeFlowMap(/*time=*/0.0, minus, u, pre)) /
        (2.0 * kStep);
    EXPECT_LT((linear.dfdx.col(i) - numeric).norm(), 1.0e-6);
  }
  for (int i = 0; i < u.size(); ++i) {
    vector_t plus = u;
    vector_t minus = u;
    plus(i) += kStep;
    minus(i) -= kStep;
    const vector_t numeric =
        (dynamics.computeFlowMap(/*time=*/0.0, x, plus, pre) -
         dynamics.computeFlowMap(/*time=*/0.0, x, minus, pre)) /
        (2.0 * kStep);
    EXPECT_LT((linear.dfdu.col(i) - numeric).norm(), 1.0e-6);
  }
}

TEST(ContactInvariantTerms, AWrappedStateCostReadsThePhysicalState) {
  vector_t goal(kBallStateDim);
  goal << 0.5, 0.0, 0.5, 0.0, 0.0, 0.0;
  const AugmentedStateCost cost(
      ballLayout(), std::make_unique<BallGoalCost>(goal, /*weight=*/2.0));
  std::unique_ptr<AugmentedPreComputation> pre = ballPreComputation();
  const vector_t x = ballState(/*height=*/0.3, /*activation=*/0.25);
  pre->requestFinal(Request::Cost + Request::Approximation, /*t=*/0.0, x);
  const ScalarFunctionQuadraticApproximation q = cost.getQuadraticApproximation(
      /*time=*/0.0, x, TargetTrajectories(), *pre);
  EXPECT_NEAR(q.f, 2.0 * (x.head(kBallStateDim) - goal).squaredNorm(), 1.0e-12);
  ASSERT_EQ(q.dfdx.size(), ballLayout().stateDim());
  EXPECT_EQ(q.dfdx(ballLayout().activationIndex(0)), 0.0);
  EXPECT_EQ(q.dfdxx.rows(), ballLayout().stateDim());
}

TEST(ContactInvariantTerms, TheProjectionClipsNegativeEigenvalues) {
  matrix_t indefinite(2, 2);
  indefinite << 0.0, 1.0, 1.0, 0.0;
  const matrix_t projected = projectToPositiveSemidefinite(indefinite);
  // Eigenvalues +1 and -1, along (1, 1) and (1, -1): the +1 part remains.
  EXPECT_NEAR(projected(0, 0), 0.5, 1.0e-12);
  EXPECT_NEAR(projected(0, 1), 0.5, 1.0e-12);
  matrix_t definite(2, 2);
  definite << 2.0, 0.5, 0.5, 1.0;
  EXPECT_TRUE(projectToPositiveSemidefinite(definite).isApprox(definite));
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
