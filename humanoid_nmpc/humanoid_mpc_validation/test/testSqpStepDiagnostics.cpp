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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/constraint/LinearStateInputConstraint.h"
#include "ocs2_core/constraint/StateInputConstraintCollection.h"
#include "ocs2_core/model_data/Metrics.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/ProblemMetrics.h"
#include "ocs2_oc/search_strategy/FilterLinesearch.h"
#include "ocs2_sqp/SqpSolverStatus.h"

#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetrics.h"
#include "humanoid_mpc_validation/closed_loop/SqpStepDiagnostics.h"

/*
 * The SQP step of a solve as the closed-loop metrics sample it: the hard
 * normal-velocity terms found by their names, their largest residual over the
 * nodes where they are active, and the line search's verdicts read from the
 * solver's iteration record. On a hand-built constraint collection and
 * synthetic node values; builds no MPC.
 */

namespace ocs2::humanoid::validation {
namespace {

/** A one-row constraint, of a state and an input of one entry each, to name in
 * a collection. */
std::unique_ptr<LinearStateInputConstraint> oneRowConstraint() {
  return std::make_unique<LinearStateInputConstraint>(
      vector_t::Zero(1), matrix_t::Zero(1, 1), matrix_t::Zero(1, 1));
}

/** A node whose state-input equality terms have the values `values` (an empty
 * vector: the term is inactive there). */
Metrics nodeWith(vector_array_t values) {
  Metrics node;
  node.cost = 0.0;
  node.stateInputEqConstraint = std::move(values);
  return node;
}

vector_t valueOf(double entry) {
  return vector_t::Constant(1, entry);
}

TEST(SqpStepDiagnostics, TheNormalVelocityTermsAreFoundByTheirContactsNames) {
  StateInputConstraintCollection constraints;
  constraints.add("foot_l_contact_zeroVelocity", oneRowConstraint());
  constraints.add("foot_r_contact_normalVelocity", oneRowConstraint());
  constraints.add("foot_l_contact_normalVelocity", oneRowConstraint());
  const std::vector<std::string> contacts = {"foot_l_contact", "foot_r_contact",
                                             "hand_contact"};
  // In the contacts' order; the contact without one left out.
  EXPECT_EQ(normalVelocityTermIndices(constraints, contacts),
            (std::vector<size_t>{2, 1}));
  EXPECT_TRUE(
      normalVelocityTermIndices(StateInputConstraintCollection(), contacts)
          .empty())
      << "a problem without the hard rows";
}

TEST(SqpStepDiagnostics,
     TheResidualIsTheLargestOverTheNodesWhereTheTermsAreActive) {
  ProblemMetrics metrics;
  // Term 0 is another constraint, terms 1 and 2 the normal-velocity rows:
  // inactive at the first node (stance).
  metrics.intermediates.push_back(
      nodeWith({valueOf(9.0), vector_t(), vector_t()}));
  metrics.intermediates.push_back(
      nodeWith({valueOf(9.0), valueOf(-0.25), vector_t()}));
  metrics.intermediates.push_back(
      nodeWith({valueOf(9.0), valueOf(0.125), valueOf(0.0625)}));
  const std::vector<size_t> terms = {1, 2};
  ASSERT_TRUE(largestEqualityConstraintResidual(metrics, terms).has_value());
  EXPECT_EQ(largestEqualityConstraintResidual(metrics, terms),
            std::optional<double>(0.25))
      << "|-0.25|, not the other term's 9";
  EXPECT_EQ(largestEqualityConstraintResidual(metrics, std::vector<size_t>{2}),
            std::optional<double>(0.0625));

  // Nowhere active, no term, or an index past the nodes' terms: no residual.
  ProblemMetrics stance;
  stance.intermediates.push_back(
      nodeWith({valueOf(9.0), vector_t(), vector_t()}));
  EXPECT_FALSE(largestEqualityConstraintResidual(stance, terms).has_value());
  EXPECT_FALSE(largestEqualityConstraintResidual(metrics, std::vector<size_t>{})
                   .has_value());
  EXPECT_FALSE(
      largestEqualityConstraintResidual(metrics, std::vector<size_t>{7})
          .has_value());
  // The final and pre-jump nodes have no state-input constraints; they are not
  // read.
  ProblemMetrics finalOnly;
  finalOnly.final = nodeWith({valueOf(9.0), valueOf(1.0), valueOf(1.0)});
  EXPECT_FALSE(largestEqualityConstraintResidual(finalOnly, terms).has_value());
}

TEST(SqpStepDiagnostics, EveryLineSearchVerdictHasItsKind) {
  EXPECT_EQ(sqpStepKindOf(FilterLinesearch::StepType::UNKNOWN),
            SqpStepKind::kUnknown);
  EXPECT_EQ(sqpStepKindOf(FilterLinesearch::StepType::CONSTRAINT),
            SqpStepKind::kConstraint);
  EXPECT_EQ(sqpStepKindOf(FilterLinesearch::StepType::DUAL),
            SqpStepKind::kDual);
  EXPECT_EQ(sqpStepKindOf(FilterLinesearch::StepType::COST),
            SqpStepKind::kCost);
  EXPECT_EQ(sqpStepKindOf(FilterLinesearch::StepType::ZERO),
            SqpStepKind::kZero);
}

/** A performance index whose constraint violation
 * (FilterLinesearch::totalConstraintViolation()) is `violation`. */
PerformanceIndex performanceWithViolation(scalar_t violation, scalar_t merit) {
  PerformanceIndex performance;
  performance.merit = merit;
  performance.dynamicsViolationSSE = violation * violation;
  return performance;
}

// The bands SqpStepKind documents: kDual is every step not above g_max that the
// Armijo rule does not judge, below g_min too - after an iterate at or above
// g_min, or without a descent direction. The docs used to call it the band
// between g_min and g_max, and a near-converged step counted under
// accepted_by.dual read as one that ended above g_min.
TEST(
    SqpStepDiagnostics,
    AStepBelowGMinIsJudgedByTheArmijoRuleOnlyFromAnIterateBelowItOnADescentDirection) {
  FilterLinesearch filter;
  filter.g_min = 1.0e-6;
  filter.g_max = 1.0e-2;
  const scalar_t descent = -1.0;
  const scalar_t noDescent = 1.0;
  const PerformanceIndex belowGMin =
      performanceWithViolation(5.0e-7, /*merit=*/1.0);
  EXPECT_EQ(filter
                .acceptStep(performanceWithViolation(2.0e-6, /*merit=*/2.0),
                            belowGMin, descent)
                .second,
            FilterLinesearch::StepType::DUAL)
      << "from an iterate at or above g_min";
  EXPECT_EQ(filter
                .acceptStep(performanceWithViolation(8.0e-7, /*merit=*/2.0),
                            belowGMin, noDescent)
                .second,
            FilterLinesearch::StepType::DUAL)
      << "without a descent direction";
  EXPECT_EQ(filter
                .acceptStep(performanceWithViolation(8.0e-7, /*merit=*/2.0),
                            belowGMin, descent)
                .second,
            FilterLinesearch::StepType::COST);
  EXPECT_EQ(
      filter
          .acceptStep(performanceWithViolation(8.0e-7, /*merit=*/2.0),
                      performanceWithViolation(1.0e-4, /*merit=*/1.0), descent)
          .second,
      FilterLinesearch::StepType::DUAL)
      << "between g_min and g_max";
  EXPECT_EQ(
      filter
          .acceptStep(performanceWithViolation(1.0, /*merit=*/2.0),
                      performanceWithViolation(0.5, /*merit=*/1.0), descent)
          .second,
      FilterLinesearch::StepType::CONSTRAINT);
}

TEST(SqpStepDiagnostics, TheSampleIsTheIterationsStepAndTheResidualOfItsNodes) {
  // A full step refused above g_max and the half step taken: the reset of the
  // cycle the metrics count.
  sqp::IterationRecord iteration;
  iteration.baselineConstraintViolation = 6.0e-3;
  iteration.stepInfo.stepSize = 0.5;
  iteration.stepInfo.stepType = FilterLinesearch::StepType::DUAL;
  iteration.stepInfo.fullStepType = FilterLinesearch::StepType::CONSTRAINT;
  iteration.stepInfo.fullStepConstraintViolation = 1.1e-2;
  iteration.stepInfo.totalConstraintViolationAfterStep = 3.0e-3;
  ProblemMetrics metrics;
  metrics.intermediates.push_back(nodeWith({valueOf(0.5), valueOf(-0.03)}));
  const SqpStepSample sample =
      sqpStepSampleOf(iteration, metrics, std::vector<size_t>{1});
  EXPECT_EQ(sample.stepSize, 0.5);
  EXPECT_EQ(sample.stepKind, SqpStepKind::kDual);
  EXPECT_EQ(sample.fullStepKind, SqpStepKind::kConstraint);
  EXPECT_EQ(sample.fullStepConstraintViolation, 1.1e-2);
  EXPECT_EQ(sample.constraintViolation, 3.0e-3);
  ASSERT_TRUE(sample.swingNormalVelocityResidual.has_value());
  EXPECT_EQ(sample.swingNormalVelocityResidual, std::optional<double>(0.03));
  EXPECT_FALSE(sqpStepSampleOf(iteration, metrics, std::vector<size_t>{})
                   .swingNormalVelocityResidual.has_value())
      << "no swing row";
}

}  // namespace
}  // namespace ocs2::humanoid::validation
