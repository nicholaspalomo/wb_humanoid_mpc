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
 * The manifold transcription of a node is the exact Newton linearization of the shooting problem on the manifold:
 *   x_next (+) dx_next = Pi(F(x (+) dx, u + du)),   cost l(x (+) dx, u + du),   constraints g(x (+) dx, u + du),
 * around dx = du = 0. Each pull-back is compared with central finite differences of exactly these functions, through
 * the manifold's own retract and difference, on a rigid body with a position (x = [p, xi, omega]: Euclidean blocks
 * before and after the quaternion), nonlinear in every block.
 */

#include <gtest/gtest.h>

#include <functional>
#include <memory>

#include <ocs2_core/integration/SensitivityIntegrator.h>
#include <ocs2_core/manifold/EuclideanStateManifold.h>
#include <ocs2_core/manifold/ProductStateManifold.h>

#include "ocs2_oc/multiple_shooting/ManifoldProjection.h"
#include "ocs2_oc/multiple_shooting/MetricsComputation.h"
#include "ocs2_oc/multiple_shooting/Transcription.h"
#include "ocs2_oc/test/RigidBodyAttitudeDynamics.h"
#include "ocs2_oc/test/TapedRigidBodyTerms.h"

namespace ocs2 {
namespace {

using manifold_test::RigidBodyAttitudeDynamics;

constexpr scalar_t kTime = 0.3;
constexpr scalar_t kDt = 0.05;
constexpr scalar_t kFirstOrderTolerance = 1e-8;
constexpr scalar_t kSecondOrderTolerance = 1e-6;

class ManifoldProjectionTest : public ::testing::Test {
 protected:
  ManifoldProjectionTest() {
    problem_.dynamicsPtr = std::make_unique<RigidBodyAttitudeDynamics>(/*withPosition=*/true);
    problem_.costPtr->add("cost", std::make_unique<manifold_test::TapedStateInputCost>());
    problem_.equalityConstraintPtr->add("equality", std::make_unique<manifold_test::TapedStateInputConstraint>());
    problem_.stateInequalityConstraintPtr->add("inequality", std::make_unique<manifold_test::TapedStateConstraint>());
    problem_.preJumpCostPtr->add("cost", std::make_unique<manifold_test::TapedStateCost>());
    problem_.preJumpInequalityConstraintPtr->add("inequality", std::make_unique<manifold_test::TapedStateConstraint>());
    problem_.finalCostPtr->add("cost", std::make_unique<manifold_test::TapedStateCost>());
    problem_.finalEqualityConstraintPtr->add("equality", std::make_unique<manifold_test::TapedStateConstraint>());
    problem_.targetTrajectoriesPtr = &targets_;
    problem_.stateManifoldPtr = manifold_;

    x_ = randomState();
    u_ = vector_t::Random(6);
    // A next state near the flow, as in a converging solve, so the gap is a small rotation and Jr^-1 is not trivial.
    vector_t flow = discretizer_(*problem_.dynamicsPtr, kTime, x_, u_, kDt);
    vector_t offset = 0.2 * vector_t::Random(manifold_->getTangentDim());
    manifold_->project(flow);
    manifold_->retract(flow, offset, /*alpha=*/1.0, xNext_);
  }

  vector_t randomState() const {
    vector_t x = vector_t::Random(manifold_->getAmbientDim());
    manifold_->project(x);
    return x;
  }

  vector_t retracted(const vector_t& x, const vector_t& dx) const {
    vector_t xNew;
    manifold_->retract(x, dx, /*alpha=*/1.0, xNew);
    return xNew;
  }

  /** Central differences of a vector function of the tangent perturbation of x_. */
  matrix_t tangentJacobian(const std::function<vector_t(const vector_t&)>& function, scalar_t eps = 1e-6) const {
    const size_t ndx = manifold_->getTangentDim();
    matrix_t jacobian;
    for (size_t j = 0; j < ndx; ++j) {
      const vector_t step = eps * vector_t::Unit(ndx, j);
      const vector_t column = (function(retracted(x_, step)) - function(retracted(x_, -step))) / (2.0 * eps);
      if (j == 0) {
        jacobian.resize(column.size(), ndx);
      }
      jacobian.col(j) = column;
    }
    return jacobian;
  }

  /** Second central differences of a scalar function of the tangent perturbation of x_. */
  matrix_t tangentHessian(const std::function<scalar_t(const vector_t&)>& function, scalar_t eps = 1e-4) const {
    const size_t ndx = manifold_->getTangentDim();
    matrix_t hessian(ndx, ndx);
    for (size_t i = 0; i < ndx; ++i) {
      for (size_t j = 0; j < ndx; ++j) {
        const vector_t ei = eps * vector_t::Unit(ndx, i);
        const vector_t ej = eps * vector_t::Unit(ndx, j);
        hessian(i, j) = (function(retracted(x_, ei + ej)) - function(retracted(x_, ei - ej)) - function(retracted(x_, ej - ei)) +
                         function(retracted(x_, -ei - ej))) /
                        (4.0 * eps * eps);
      }
    }
    return hessian;
  }

  static scalar_t maxError(const matrix_t& a, const matrix_t& b) {
    EXPECT_EQ(a.rows(), b.rows());
    EXPECT_EQ(a.cols(), b.cols());
    return (a - b).cwiseAbs().maxCoeff();
  }

  std::shared_ptr<const ProductStateManifold> manifold_ = RigidBodyAttitudeDynamics(/*withPosition=*/true).getStateManifold();
  TargetTrajectories targets_ = TargetTrajectories({0.0}, {vector_t::Zero(10)}, {vector_t::Zero(6)});
  OptimalControlProblem problem_;
  DynamicsDiscretizer discretizer_ = selectDynamicsDiscretization(SensitivityIntegratorType::RK4);
  DynamicsSensitivityDiscretizer sensitivityDiscretizer_ = selectDynamicsSensitivityDiscretization(SensitivityIntegratorType::RK4);
  vector_t x_;
  vector_t u_;
  vector_t xNext_;
};

TEST_F(ManifoldProjectionTest, IntermediateNodeDynamicsAreTheNewtonLinearizationOfTheGap) {
  const multiple_shooting::Transcription transcription =
      multiple_shooting::setupIntermediateNode(problem_, sensitivityDiscretizer_, kTime, kDt, x_, xNext_, u_);
  const size_t ndx = manifold_->getTangentDim();

  // The gap and its sizes.
  const vector_t flow = discretizer_(*problem_.dynamicsPtr, kTime, x_, u_, kDt);
  EXPECT_LT((transcription.dynamics.f - manifold_->difference(xNext_, flow)).norm(), 1e-13);
  ASSERT_EQ(transcription.dynamics.dfdx.rows(), static_cast<Eigen::Index>(ndx));
  ASSERT_EQ(transcription.dynamics.dfdx.cols(), static_cast<Eigen::Index>(ndx));
  ASSERT_EQ(transcription.dynamics.dfdu.rows(), static_cast<Eigen::Index>(ndx));

  // A: d/d dx of difference(x_next, F(x (+) dx, u)).
  const matrix_t numericA = tangentJacobian(
      [&](const vector_t& x) { return manifold_->difference(xNext_, discretizer_(*problem_.dynamicsPtr, kTime, x, u_, kDt)); });
  EXPECT_LT(maxError(transcription.dynamics.dfdx, numericA), kFirstOrderTolerance);

  // B: d/d du of difference(x_next, F(x, u + du)).
  const scalar_t eps = 1e-6;
  matrix_t numericB(ndx, 6);
  for (int j = 0; j < 6; ++j) {
    const vector_t step = eps * vector_t::Unit(/*newSize=*/6, /*i=*/j);
    numericB.col(j) = (manifold_->difference(xNext_, discretizer_(*problem_.dynamicsPtr, kTime, x_, u_ + step, kDt)) -
                       manifold_->difference(xNext_, discretizer_(*problem_.dynamicsPtr, kTime, x_, u_ - step, kDt))) /
                      (2.0 * eps);
  }
  EXPECT_LT(maxError(transcription.dynamics.dfdu, numericB), kFirstOrderTolerance);

  // The metrics use the same gap.
  const Metrics metrics = multiple_shooting::computeIntermediateMetrics(problem_, discretizer_, kTime, kDt, x_, xNext_, u_);
  EXPECT_LT((metrics.dynamicsViolation - transcription.dynamics.f).norm(), 1e-13);
}

TEST_F(ManifoldProjectionTest, IntermediateNodeCostAndConstraintsArePulledBack) {
  const multiple_shooting::Transcription transcription =
      multiple_shooting::setupIntermediateNode(problem_, sensitivityDiscretizer_, kTime, kDt, x_, xNext_, u_);
  const PreComputation preComputation;
  const manifold_test::TapedStateInputCost costTerm;
  const manifold_test::TapedStateInputConstraint equality;
  const manifold_test::TapedStateConstraint inequality;

  // Cost (times dt, the transcription's Euler integral): gradient, Gauss-Newton-free exact Hessian, and cross term.
  const std::function<scalar_t(const vector_t&)> nodeCost = [&](const vector_t& x) {
    return kDt * costTerm.getValue(kTime, x, u_, targets_, preComputation);
  };
  const matrix_t numericGradient = tangentJacobian([&](const vector_t& x) {
    vector_t value(1);
    value << nodeCost(x);
    return value;
  });
  EXPECT_LT(maxError(transcription.cost.dfdx.transpose(), numericGradient), kFirstOrderTolerance);
  EXPECT_LT(maxError(transcription.cost.dfdxx, tangentHessian(nodeCost)), kSecondOrderTolerance);
  const matrix_t numericCross = tangentJacobian(
      [&](const vector_t& x) { return vector_t(kDt * costTerm.getQuadraticApproximation(kTime, x, u_, targets_, preComputation).dfdu); });
  EXPECT_LT(maxError(transcription.cost.dfdux, numericCross), kFirstOrderTolerance);
  EXPECT_NEAR(transcription.cost.f, nodeCost(x_), 1e-15);

  // Constraints.
  EXPECT_LT(maxError(transcription.stateInputEqConstraints.dfdx,
                     tangentJacobian([&](const vector_t& x) { return equality.getValue(kTime, x, u_, preComputation); })),
            kFirstOrderTolerance);
  EXPECT_EQ(transcription.stateInputEqConstraints.dfdu, equality.getLinearApproximation(kTime, x_, u_, preComputation).dfdu);
  EXPECT_LT(maxError(transcription.stateIneqConstraints.dfdx,
                     tangentJacobian([&](const vector_t& x) { return inequality.getValue(kTime, x, preComputation); })),
            kFirstOrderTolerance);
}

TEST_F(ManifoldProjectionTest, EventNode) {
  const multiple_shooting::EventTranscription transcription = multiple_shooting::setupEventNode(problem_, kTime, x_, xNext_);
  const size_t ndx = manifold_->getTangentDim();
  const PreComputation preComputation;

  const vector_t jumped = problem_.dynamicsPtr->computeJumpMap(kTime, x_);
  EXPECT_LT((transcription.dynamics.f - manifold_->difference(xNext_, jumped)).norm(), 1e-14);
  EXPECT_LT(maxError(transcription.dynamics.dfdx, tangentJacobian([&](const vector_t& x) {
                       return manifold_->difference(xNext_, problem_.dynamicsPtr->computeJumpMap(kTime, x));
                     })),
            kFirstOrderTolerance);
  EXPECT_EQ(transcription.dynamics.dfdu.rows(), static_cast<Eigen::Index>(ndx));
  EXPECT_EQ(transcription.dynamics.dfdu.cols(), 0);

  const manifold_test::TapedStateCost costTerm;
  const std::function<scalar_t(const vector_t&)> eventCost = [&](const vector_t& x) {
    return costTerm.getValue(kTime, x, targets_, preComputation);
  };
  EXPECT_LT(maxError(transcription.cost.dfdx.transpose(), tangentJacobian([&](const vector_t& x) {
                       vector_t value(1);
                       value << eventCost(x);
                       return value;
                     })),
            kFirstOrderTolerance);
  EXPECT_LT(maxError(transcription.cost.dfdxx, tangentHessian(eventCost)), kSecondOrderTolerance);
  const manifold_test::TapedStateConstraint inequality;
  EXPECT_LT(maxError(transcription.ineqConstraints.dfdx,
                     tangentJacobian([&](const vector_t& x) { return inequality.getValue(kTime, x, preComputation); })),
            kFirstOrderTolerance);

  const Metrics metrics = multiple_shooting::computeEventMetrics(problem_, kTime, x_, xNext_);
  EXPECT_LT((metrics.dynamicsViolation - transcription.dynamics.f).norm(), 1e-14);
}

TEST_F(ManifoldProjectionTest, TerminalNode) {
  const multiple_shooting::TerminalTranscription transcription = multiple_shooting::setupTerminalNode(problem_, kTime, x_);
  const PreComputation preComputation;
  const manifold_test::TapedStateCost costTerm;
  const std::function<scalar_t(const vector_t&)> finalCost = [&](const vector_t& x) {
    return costTerm.getValue(kTime, x, targets_, preComputation);
  };
  EXPECT_LT(maxError(transcription.cost.dfdx.transpose(), tangentJacobian([&](const vector_t& x) {
                       vector_t value(1);
                       value << finalCost(x);
                       return value;
                     })),
            kFirstOrderTolerance);
  EXPECT_LT(maxError(transcription.cost.dfdxx, tangentHessian(finalCost)), kSecondOrderTolerance);
  const manifold_test::TapedStateConstraint equality;
  EXPECT_LT(maxError(transcription.eqConstraints.dfdx,
                     tangentJacobian([&](const vector_t& x) { return equality.getValue(kTime, x, preComputation); })),
            kFirstOrderTolerance);
}

/** Bitwise equality of two linear approximations. */
void expectIdentical(const VectorFunctionLinearApproximation& a, const VectorFunctionLinearApproximation& b) {
  EXPECT_EQ(a.f, b.f);
  EXPECT_EQ(a.dfdx, b.dfdx);
  EXPECT_EQ(a.dfdu, b.dfdu);
}

void expectIdentical(const ScalarFunctionQuadraticApproximation& a, const ScalarFunctionQuadraticApproximation& b) {
  EXPECT_EQ(a.f, b.f);
  EXPECT_EQ(a.dfdx, b.dfdx);
  EXPECT_EQ(a.dfdu, b.dfdu);
  EXPECT_EQ(a.dfdxx, b.dfdxx);
  EXPECT_EQ(a.dfdux, b.dfdux);
  EXPECT_EQ(a.dfduu, b.dfduu);
}

TEST_F(ManifoldProjectionTest, WithoutAManifoldTheTranscriptionIsTheFlatOne) {
  // nullptr: the flat transcription, f = F(x, u) - x_next.
  OptimalControlProblem flat = problem_;
  flat.stateManifoldPtr = nullptr;
  const multiple_shooting::Transcription flatNode =
      multiple_shooting::setupIntermediateNode(flat, sensitivityDiscretizer_, kTime, kDt, x_, xNext_, u_);
  VectorFunctionLinearApproximation expected = sensitivityDiscretizer_(*flat.dynamicsPtr, kTime, x_, u_, kDt);
  expected.f -= xNext_;
  expectIdentical(flatNode.dynamics, expected);
  EXPECT_EQ(flatNode.cost.dfdxx.rows(), 10);

  // A Euclidean manifold of the ambient size gives the flat transcription bit for bit.
  OptimalControlProblem euclidean = problem_;
  euclidean.stateManifoldPtr = std::make_shared<EuclideanStateManifold>(10);
  const multiple_shooting::Transcription euclideanNode =
      multiple_shooting::setupIntermediateNode(euclidean, sensitivityDiscretizer_, kTime, kDt, x_, xNext_, u_);
  expectIdentical(euclideanNode.dynamics, flatNode.dynamics);
  expectIdentical(euclideanNode.cost, flatNode.cost);
  expectIdentical(euclideanNode.stateInputEqConstraints, flatNode.stateInputEqConstraints);
  expectIdentical(euclideanNode.stateIneqConstraints, flatNode.stateIneqConstraints);

  const multiple_shooting::EventTranscription flatEvent = multiple_shooting::setupEventNode(flat, kTime, x_, xNext_);
  const multiple_shooting::EventTranscription euclideanEvent = multiple_shooting::setupEventNode(euclidean, kTime, x_, xNext_);
  expectIdentical(euclideanEvent.dynamics, flatEvent.dynamics);
  expectIdentical(euclideanEvent.cost, flatEvent.cost);
  EXPECT_EQ(flatEvent.dynamics.dfdu.rows(), 10);
}

TEST_F(ManifoldProjectionTest, PullBacksLeaveUnsetTermsAlone) {
  ScalarFunctionQuadraticApproximation emptyCost;
  emptyCost.f = 1.0;
  multiple_shooting::pullBackCost(*manifold_, x_, emptyCost);
  EXPECT_EQ(emptyCost.dfdx.size(), 0);
  EXPECT_EQ(emptyCost.dfdxx.size(), 0);
  VectorFunctionLinearApproximation emptyConstraint;
  multiple_shooting::pullBackConstraint(*manifold_, x_, emptyConstraint);
  EXPECT_EQ(emptyConstraint.dfdx.size(), 0);
  // A constraint with no rows but ambient columns gets tangent columns.
  VectorFunctionLinearApproximation noRows = VectorFunctionLinearApproximation::Zero(0, 10, 6);
  multiple_shooting::pullBackConstraint(*manifold_, x_, noRows);
  EXPECT_EQ(noRows.dfdx.cols(), 9);
}

}  // namespace
}  // namespace ocs2
