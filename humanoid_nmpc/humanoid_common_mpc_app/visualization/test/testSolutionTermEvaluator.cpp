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

#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/ComputationRequest.h"
#include "ocs2_core/constraint/LinearStateConstraint.h"
#include "ocs2_core/constraint/LinearStateInputConstraint.h"
#include "ocs2_core/cost/QuadraticStateCost.h"
#include "ocs2_core/cost/QuadraticStateInputCost.h"
#include "ocs2_core/model_data/Metrics.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/approximate_model/LinearQuadraticApproximator.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_common_mpc_app/visualization/SolutionTermEvaluator.h"

namespace ocs2::humanoid::visualization {
namespace {

constexpr int kStateDim = 3;
constexpr int kInputDim = 2;
constexpr int kConstraintDim = 2;
constexpr scalar_t kTolerance = 1.0e-12;
/** [s] The quadratic cost "late" is active from here on: after the solution's
 * first node. */
constexpr scalar_t kLateStart = 0.25;

/** A quadratic state-input cost that is active only from kLateStart on. */
class LateCost final : public QuadraticStateInputCost {
 public:
  LateCost(matrix_t q, matrix_t r)
      : QuadraticStateInputCost(std::move(q), std::move(r)) {}
  LateCost* absl_nonnull clone() const override { return new LateCost(*this); }
  bool isActive(scalar_t time) const override { return time >= kLateStart; }
};

matrix_t randomSpd(std::mt19937& generator, int size) {
  std::normal_distribution<scalar_t> normal;
  matrix_t a(size, size);
  for (int row = 0; row < size; ++row) {
    for (int column = 0; column < size; ++column)
      a(row, column) = normal(generator);
  }
  return a * a.transpose() + matrix_t::Identity(size, size);
}

vector_t randomVector(std::mt19937& generator, int size) {
  std::normal_distribution<scalar_t> normal;
  vector_t v(size);
  for (int index = 0; index < size; ++index) v[index] = normal(generator);
  return v;
}

matrix_t randomMatrix(std::mt19937& generator, int rows, int columns) {
  std::normal_distribution<scalar_t> normal;
  matrix_t m(rows, columns);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column)
      m(row, column) = normal(generator);
  }
  return m;
}

/** A small problem with a term in the collections the evaluator reads, and a
 * solution and a target for it. */
struct Fixture {
  explicit Fixture(unsigned int seed) : generator(seed) {
    q = randomSpd(generator, kStateDim);
    r = randomSpd(generator, kInputDim);
    stateQ = randomSpd(generator, kStateDim);
    finalQ = randomSpd(generator, kStateDim);
    e = randomVector(generator, kConstraintDim);
    c = randomMatrix(generator, kConstraintDim, kStateDim);
    d = randomMatrix(generator, kConstraintDim, kInputDim);
    // The inequality g = e_ineq + C x + D u >= 0: one row violated, one
    // satisfied, at the first node (set below).
    h = randomVector(generator, kConstraintDim);
    f = randomMatrix(generator, kConstraintDim, kStateDim);
    problem.costPtr->add("tracking",
                         std::make_unique<QuadraticStateInputCost>(q, r));
    problem.costPtr->add("late", std::make_unique<LateCost>(q, r));
    problem.stateCostPtr->add("stateTerm",
                              std::make_unique<QuadraticStateCost>(stateQ));
    problem.finalCostPtr->add("terminal",
                              std::make_unique<QuadraticStateCost>(finalQ));
    problem.equalityConstraintPtr->add(
        "eq", std::make_unique<LinearStateInputConstraint>(e, c, d));
    problem.finalEqualityConstraintPtr->add(
        "finalEq", std::make_unique<LinearStateConstraint>(h, f));

    target = TargetTrajectories({0.0}, {randomVector(generator, kStateDim)},
                                {randomVector(generator, kInputDim)});
    problem.targetTrajectoriesPtr = &target;
    solution.timeTrajectory_ = {0.0, 0.5, 1.0};
    for (size_t node = 0; node < solution.timeTrajectory_.size(); ++node) {
      solution.stateTrajectory_.push_back(randomVector(generator, kStateDim));
      solution.inputTrajectory_.push_back(randomVector(generator, kInputDim));
    }
    const vector_t& x0 = solution.stateTrajectory_.front();
    const vector_t& u0 = solution.inputTrajectory_.front();
    // Rows of g at the first node: -0.7 (a violation of 0.7) and +0.4
    // (satisfied).
    const matrix_t inequalityC =
        randomMatrix(generator, kConstraintDim, kStateDim);
    const matrix_t inequalityD =
        randomMatrix(generator, kConstraintDim, kInputDim);
    const vector_t wanted = (vector_t(kConstraintDim) << -0.7, 0.4).finished();
    problem.inequalityConstraintPtr->add(
        "ineq", std::make_unique<LinearStateInputConstraint>(
                    wanted - inequalityC * x0 - inequalityD * u0, inequalityC,
                    inequalityD));
    performance.merit = 3.0;
    performance.cost = 2.5;
    performance.equalityConstraintsSSE = 0.125;
  }

  std::mt19937 generator;
  matrix_t q, r, stateQ, finalQ, c, d, f;
  vector_t e, h;
  OptimalControlProblem problem;
  TargetTrajectories target;
  PrimalSolution solution;
  PerformanceIndex performance;
};

scalar_t quadratic(const vector_t& deviation, const matrix_t& weight) {
  return 0.5 * deviation.dot(weight * deviation);
}

TEST(SolutionTermEvaluator, NamesEveryTermByCollectionInTheOrderItWasAdded) {
  Fixture fixture(/*seed=*/1);
  const absl::StatusOr<std::unique_ptr<SolutionTermEvaluator>> evaluator =
      SolutionTermEvaluator::Create(&fixture.problem);
  ASSERT_TRUE(evaluator.ok()) << evaluator.status();
  EXPECT_EQ(
      (*evaluator)->costNames(),
      (std::vector<std::string>{"intermediate/tracking", "intermediate/late",
                                "state/stateTerm", "final/terminal"}));
  EXPECT_EQ((*evaluator)->constraintNames(),
            (std::vector<std::string>{"equality/eq", "inequality/ineq",
                                      "final_equality/finalEq"}));
}

TEST(SolutionTermEvaluator, TheTermsAreTheirValuesAtTheFirstAndTheLastNode) {
  for (unsigned int seed = 0; seed < 20; ++seed) {
    Fixture fixture(seed);
    std::unique_ptr<SolutionTermEvaluator> evaluator =
        *SolutionTermEvaluator::Create(&fixture.problem);
    SolutionTerms terms;
    ASSERT_TRUE(evaluator
                    ->evaluate(fixture.solution, fixture.target,
                               fixture.performance, &terms)
                    .ok());
    const vector_t& x0 = fixture.solution.stateTrajectory_.front();
    const vector_t& u0 = fixture.solution.inputTrajectory_.front();
    const vector_t& xf = fixture.solution.stateTrajectory_.back();
    const vector_t dx0 = x0 - fixture.target.stateTrajectory.front();
    const vector_t du0 = u0 - fixture.target.inputTrajectory.front();
    const vector_t dxf = xf - fixture.target.stateTrajectory.front();
    ASSERT_EQ(terms.costs.size(), 4);
    // Independently of OCS2: the quadratic forms by hand, the late term
    // inactive at t0 = 0.
    EXPECT_NEAR(terms.costs[0],
                quadratic(dx0, fixture.q) + quadratic(du0, fixture.r),
                kTolerance);
    EXPECT_EQ(terms.costs[1], 0.0);
    EXPECT_NEAR(terms.costs[2], quadratic(dx0, fixture.stateQ), kTolerance);
    EXPECT_NEAR(terms.costs[3], quadratic(dxf, fixture.finalQ), kTolerance);
    ASSERT_EQ(terms.constraintViolations.size(), 3);
    EXPECT_NEAR(terms.constraintViolations[0],
                (fixture.e + fixture.c * x0 + fixture.d * u0).norm(),
                kTolerance);
    EXPECT_NEAR(terms.constraintViolations[1], 0.7,
                1.0e-9);  // only the violated row counts
    EXPECT_NEAR(terms.constraintViolations[2],
                (fixture.h + fixture.f * xf).norm(), kTolerance);
    EXPECT_EQ(terms.performance.merit, fixture.performance.merit);
    EXPECT_EQ(terms.performance.cost, fixture.performance.cost);
    EXPECT_EQ(terms.performance.equalityConstraintsSSE,
              fixture.performance.equalityConstraintsSSE);
  }
}

// The terms add up to what the solver's own sums give: OCS2's computeCost() /
// computeFinalCost() and its metrics.
TEST(SolutionTermEvaluator, TheTermsSumToTheSolversTotals) {
  Fixture fixture(/*seed=*/7);
  std::unique_ptr<SolutionTermEvaluator> evaluator =
      *SolutionTermEvaluator::Create(&fixture.problem);
  SolutionTerms terms;
  ASSERT_TRUE(evaluator
                  ->evaluate(fixture.solution, fixture.target,
                             fixture.performance, &terms)
                  .ok());
  const scalar_t t0 = fixture.solution.timeTrajectory_.front();
  const vector_t& x0 = fixture.solution.stateTrajectory_.front();
  const vector_t& u0 = fixture.solution.inputTrajectory_.front();
  constexpr RequestSet kRequest =
      Request::Cost + Request::SoftConstraint + Request::Constraint;
  fixture.problem.preComputationPtr->request(kRequest, t0, x0, u0);
  EXPECT_NEAR(terms.costs[0] + terms.costs[1] + terms.costs[2],
              computeCost(fixture.problem, t0, x0, u0), kTolerance);
  const Metrics metrics =
      computeIntermediateMetrics(fixture.problem, t0, x0, u0);
  EXPECT_NEAR(terms.constraintViolations[0],
              metrics.stateInputEqConstraint[0].norm(), kTolerance);
  EXPECT_NEAR(terms.constraintViolations[1],
              metrics.stateInputIneqConstraint[0].cwiseMin(0.0).norm(),
              kTolerance);
  const scalar_t tf = fixture.solution.timeTrajectory_.back();
  const vector_t& xf = fixture.solution.stateTrajectory_.back();
  fixture.problem.preComputationPtr->requestFinal(kRequest, tf, xf);
  EXPECT_NEAR(terms.costs[3], computeFinalCost(fixture.problem, tf, xf),
              kTolerance);
  EXPECT_NEAR(
      terms.constraintViolations[2],
      computeFinalMetrics(fixture.problem, tf, xf).stateEqConstraint[0].norm(),
      kTolerance);
}

TEST(SolutionTermEvaluator, ATermTheProblemNoLongerHasIsNotANumber) {
  Fixture fixture(/*seed=*/3);
  std::unique_ptr<SolutionTermEvaluator> evaluator =
      *SolutionTermEvaluator::Create(&fixture.problem);
  ASSERT_TRUE(fixture.problem.costPtr->erase("tracking"));
  ASSERT_TRUE(fixture.problem.equalityConstraintPtr->erase("eq"));
  SolutionTerms terms;
  ASSERT_TRUE(evaluator
                  ->evaluate(fixture.solution, fixture.target,
                             fixture.performance, &terms)
                  .ok());
  EXPECT_TRUE(std::isnan(terms.costs[0]));
  EXPECT_FALSE(std::isnan(terms.costs[2]));
  EXPECT_TRUE(std::isnan(terms.constraintViolations[0]));
  EXPECT_FALSE(std::isnan(terms.constraintViolations[1]));
}

TEST(SolutionTermEvaluator, AnEmptySolutionIsRefusedAndLeavesTheTerms) {
  Fixture fixture(/*seed=*/4);
  std::unique_ptr<SolutionTermEvaluator> evaluator =
      *SolutionTermEvaluator::Create(&fixture.problem);
  SolutionTerms terms;
  terms.costs = vector_t::Constant(/*rows=*/1, /*value=*/5.0);
  EXPECT_EQ(evaluator
                ->evaluate(PrimalSolution(), fixture.target,
                           fixture.performance, &terms)
                .code(),
            absl::StatusCode::kInvalidArgument);
  ASSERT_EQ(terms.costs.size(), 1);
  EXPECT_EQ(terms.costs[0], 5.0);
}

TEST(SolutionTermEvaluator, AProblemWithoutACollectionIsRefused) {
  OptimalControlProblem problem;
  problem.finalCostPtr.reset();
  EXPECT_EQ(SolutionTermEvaluator::Create(&problem).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid::visualization
