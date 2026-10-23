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

#include <algorithm>
#include <cstddef>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "gtest/gtest.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"
#include "ocs2_core/control/ManifoldLinearController.h"
#include "ocs2_core/manifold/ProductStateManifold.h"
#include "ocs2_core/misc/LinearInterpolation.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_common_mpc/mrt/LookaheadPolicyEvaluator.h"
#include "humanoid_mpc_ipc/RealtimePolicyEvaluator.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"

/*
 * The policy an MRT joint controller executes a lookahead delta after its measurement at t: the feedforward and the
 * plan at t + delta, the feedback where the measurement is, at t (LookaheadPolicyEvaluator.h).
 *   - On a moving plan a measurement exactly on it gets exactly the feedforward u(t + delta, x*(t + delta)), at every
 *     delta, on a flat state and on a state manifold; the law evaluated at t + delta on that same measurement (what the
 *     controllers did) does not - the positive control.
 *   - Off the plan, for a flat LinearController u = b(t) + K(t) x, the input is b(t + delta) + K(t + delta) x*(t + delta)
 *     + K(t) (x - x*(t)), built here from the policy's own arrays by linear interpolation.
 */

namespace ocs2::humanoid {
namespace {

constexpr size_t kStateDim = 7;  // a position (3) and a unit quaternion (4) on the manifold; seven numbers when flat
constexpr size_t kInputDim = 4;
constexpr size_t kTangentDim = 6;
constexpr size_t kNodes = 4;
constexpr scalar_t kNodeSpacing = 0.1;  // [s]
constexpr scalar_t kLookaheads[] = {0.001, 0.002, 0.0073, 0.02};

ipc::ModelDimensions dimensions() {
  return ipc::ModelDimensions{.stateDim = kStateDim, .inputDim = kInputDim, .numModes = 1};
}

scalar_array_t nodeTimes() {
  scalar_array_t times;
  for (size_t k = 0; k < kNodes; ++k) times.push_back(static_cast<scalar_t>(k) * kNodeSpacing);
  return times;
}

/** A plan that moves fast: random states and inputs at every node, the quaternion block projected when on a manifold. */
PrimalSolution movingPlan(std::unique_ptr<ControllerBase> controller, const vector_array_t& states, const vector_array_t& inputs) {
  PrimalSolution policy;
  policy.timeTrajectory_ = nodeTimes();
  policy.stateTrajectory_ = states;
  policy.inputTrajectory_ = inputs;
  policy.modeSchedule_ = ModeSchedule(/*eventTimesInput=*/{}, /*modeSequenceInput=*/{0});
  policy.controllerPtr_ = std::move(controller);
  return policy;
}

vector_array_t randomVectors(size_t size, std::mt19937& generator, const ProductStateManifold* absl_nullable manifold = nullptr) {
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  vector_array_t vectors;
  for (size_t k = 0; k < kNodes; ++k) {
    vector_t v(static_cast<Eigen::Index>(size));
    for (Eigen::Index i = 0; i < v.size(); ++i) v(i) = normal(generator);
    if (manifold != nullptr) manifold->project(v);
    vectors.push_back(v);
  }
  return vectors;
}

matrix_array_t randomGains(Eigen::Index columns, std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal(0.0, 10.0);
  matrix_array_t gains;
  for (size_t k = 0; k < kNodes; ++k) {
    matrix_t gain(static_cast<Eigen::Index>(kInputDim), columns);
    for (Eigen::Index i = 0; i < gain.size(); ++i) gain.data()[i] = normal(generator);
    gains.push_back(gain);
  }
  return gains;
}

/** The times within the plan the tests measure at. */
std::vector<scalar_t> measurementTimes() {
  std::vector<scalar_t> times;
  for (int i = 0; i < 25; ++i) times.push_back(0.0123 + 0.0107 * i);
  return times;
}

/** On a moving plan, a measurement exactly on it (x = x*(t), bit for bit) gets exactly u(t + delta, x*(t + delta)). */
void expectOnPlanMeasurementGetsTheFeedforward(const PrimalSolution& policy, const StateManifold* absl_nullable manifold) {
  LookaheadPolicyEvaluator evaluator(dimensions(), manifold);
  ipc::RealtimePolicyEvaluator reference(dimensions(), manifold);
  vector_t planNow = vector_t::Zero(kStateDim);
  vector_t planLater = vector_t::Zero(kStateDim);
  vector_t unused = vector_t::Zero(kStateDim);
  vector_t feedforward = vector_t::Zero(kInputDim);
  vector_t lawAtLookahead = vector_t::Zero(kInputDim);
  vector_t input = vector_t::Zero(kInputDim);
  vector_t planned = vector_t::Zero(kStateDim);
  size_t mode = 0;
  scalar_t largestOldBias = 0.0;
  for (const scalar_t t : measurementTimes()) {
    for (const scalar_t delta : kLookaheads) {
      ASSERT_EQ(reference.evaluate(policy, t, unused, planNow, feedforward, mode), LookaheadPolicyEvaluator::Outcome::kEvaluated);
      const vector_t measured = planNow;  // x*(t), exactly
      ASSERT_EQ(reference.evaluate(policy, t + delta, measured, planLater, lawAtLookahead, mode),
                LookaheadPolicyEvaluator::Outcome::kEvaluated);
      ASSERT_EQ(reference.evaluate(policy, t + delta, planLater, unused, feedforward, mode), LookaheadPolicyEvaluator::Outcome::kEvaluated);

      ASSERT_EQ(evaluator.evaluate(policy, t, delta, measured, planned, input, mode), LookaheadPolicyEvaluator::Outcome::kEvaluated);
      EXPECT_EQ(input, feedforward) << "t = " << t << ", delta = " << delta;
      EXPECT_EQ(planned, planLater);
      largestOldBias = std::max(largestOldBias, (lawAtLookahead - feedforward).cwiseAbs().maxCoeff());
    }
  }
  // The law at t + delta on the state measured at t is biased by about -K xdot* delta on this plan.
  EXPECT_GT(largestOldBias, 1.0e-3);
}

TEST(LookaheadPolicyEvaluator, OnAMovingFlatPlanAMeasurementOnItGetsExactlyTheFeedforward) {
  std::mt19937 generator(/*sd=*/3);
  const vector_array_t states = randomVectors(kStateDim, generator);
  const vector_array_t inputs = randomVectors(kInputDim, generator);
  const matrix_array_t gains = randomGains(static_cast<Eigen::Index>(kStateDim), generator);
  const PrimalSolution policy = movingPlan(std::make_unique<LinearController>(nodeTimes(), inputs, gains), states, inputs);
  expectOnPlanMeasurementGetsTheFeedforward(policy, /*manifold=*/nullptr);
}

TEST(LookaheadPolicyEvaluator, OnAMovingManifoldPlanAMeasurementOnItGetsExactlyTheFeedforward) {
  const std::shared_ptr<const ProductStateManifold> manifold =
      checkedValue(ProductStateManifold::create({StateManifoldSegment::euclidean(3), StateManifoldSegment::unitQuaternion()}));
  std::mt19937 generator(/*sd=*/5);
  const vector_array_t states = randomVectors(kStateDim, generator, manifold.get());
  const vector_array_t inputs = randomVectors(kInputDim, generator);
  const matrix_array_t gains = randomGains(static_cast<Eigen::Index>(kTangentDim), generator);
  const PrimalSolution policy =
      movingPlan(std::make_unique<ManifoldLinearController>(nodeTimes(), states, inputs, gains, manifold), states, inputs);
  expectOnPlanMeasurementGetsTheFeedforward(policy, manifold.get());
}

TEST(LookaheadPolicyEvaluator, OffAFlatPlanTheFeedbackIsTheGainAtTheMeasurementTimeTimesTheErrorThere) {
  std::mt19937 generator(/*sd=*/11);
  std::normal_distribution<scalar_t> normal(0.0, 0.3);
  const vector_array_t states = randomVectors(kStateDim, generator);
  const vector_array_t biases = randomVectors(kInputDim, generator);
  const matrix_array_t gains = randomGains(static_cast<Eigen::Index>(kStateDim), generator);
  const PrimalSolution policy = movingPlan(std::make_unique<LinearController>(nodeTimes(), biases, gains), states, biases);
  LookaheadPolicyEvaluator evaluator(dimensions(), /*stateManifold=*/nullptr);
  vector_t input = vector_t::Zero(kInputDim);
  vector_t planned = vector_t::Zero(kStateDim);
  size_t mode = 0;
  for (const scalar_t t : measurementTimes()) {
    for (const scalar_t delta : kLookaheads) {
      const LinearInterpolation::index_alpha_t now = LinearInterpolation::timeSegment(t, policy.timeTrajectory_);
      const LinearInterpolation::index_alpha_t later = LinearInterpolation::timeSegment(t + delta, policy.timeTrajectory_);
      // The oracle of a flat state, which has no quaternion: linear interpolation is the plan's.
      const vector_t planNow = LinearInterpolation::interpolate(now, states);      // NOLINT(state-interpolation): a flat state
      const vector_t planLater = LinearInterpolation::interpolate(later, states);  // NOLINT(state-interpolation): a flat state
      vector_t measured = planNow;
      for (Eigen::Index i = 0; i < measured.size(); ++i) measured(i) += normal(generator);
      const vector_t biasLater = LinearInterpolation::interpolate(later, biases);  // NOLINT(state-interpolation): an input
      const matrix_t gainLater = LinearInterpolation::interpolate(later, gains);   // NOLINT(state-interpolation): a gain
      const matrix_t gainNow = LinearInterpolation::interpolate(now, gains);       // NOLINT(state-interpolation): a gain
      const vector_t expected = biasLater + gainLater * planLater + gainNow * (measured - planNow);
      ASSERT_EQ(evaluator.evaluate(policy, t, delta, measured, planned, input, mode), LookaheadPolicyEvaluator::Outcome::kEvaluated);
      EXPECT_TRUE(input.isApprox(expected, 1.0e-12)) << "t = " << t << ", delta = " << delta << "\n"
                                                     << input.transpose() << "\n"
                                                     << expected.transpose();
      EXPECT_TRUE(planned.isApprox(planLater, 1.0e-14));
    }
  }
}

TEST(LookaheadPolicyEvaluator, AFeedforwardPolicyGetsItsInputAtTheLookahead) {
  std::mt19937 generator(/*sd=*/13);
  const vector_array_t states = randomVectors(kStateDim, generator);
  const vector_array_t inputs = randomVectors(kInputDim, generator);
  const PrimalSolution policy = movingPlan(std::make_unique<FeedforwardController>(nodeTimes(), inputs), states, inputs);
  LookaheadPolicyEvaluator evaluator(dimensions(), /*stateManifold=*/nullptr);
  vector_t input = vector_t::Zero(kInputDim);
  vector_t planned = vector_t::Zero(kStateDim);
  size_t mode = 0;
  const vector_t measured = vector_t::Constant(kStateDim, 0.5);
  for (const scalar_t t : measurementTimes()) {
    const scalar_t delta = kLookaheads[1];
    ASSERT_EQ(evaluator.evaluate(policy, t, delta, measured, planned, input, mode), LookaheadPolicyEvaluator::Outcome::kEvaluated);
    EXPECT_TRUE(input.isApprox(
        LinearInterpolation::interpolate(LinearInterpolation::timeSegment(t + delta, policy.timeTrajectory_), inputs), 1.0e-14));
  }
}

TEST(LookaheadPolicyEvaluator, ARefusedPolicyLeavesTheOutputs) {
  PrimalSolution policy;  // no controller
  LookaheadPolicyEvaluator evaluator(dimensions(), /*stateManifold=*/nullptr);
  vector_t input = vector_t::Constant(kInputDim, 3.0);
  vector_t planned = vector_t::Constant(kStateDim, 4.0);
  size_t mode = 7;
  EXPECT_NE(evaluator.evaluate(policy, /*measurementTime=*/0.0, /*lookahead=*/0.002, vector_t::Zero(kStateDim), planned, input, mode),
            LookaheadPolicyEvaluator::Outcome::kEvaluated);
  EXPECT_EQ(input, vector_t::Constant(kInputDim, 3.0));
  EXPECT_EQ(planned, vector_t::Constant(kStateDim, 4.0));
  EXPECT_EQ(mode, 7u);
}

}  // namespace
}  // namespace ocs2::humanoid
