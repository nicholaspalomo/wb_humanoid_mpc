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

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include <ocs2_core/control/LinearController.h>
#include <ocs2_core/control/ManifoldLinearController.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_core/manifold/ProductStateManifold.h>
#include <ocs2_core/misc/LinearInterpolation.h>

#include "ocs2_oc/multiple_shooting/Helpers.h"
#include "ocs2_oc/multiple_shooting/Initialization.h"
#include "ocs2_oc/oc_data/StateTrajectoryInterpolation.h"
#include "ocs2_oc/oc_data/TimeDiscretization.h"

namespace ocs2 {
namespace {

/** [Q, E(2)]: ambient 6, tangent 5. */
std::shared_ptr<const ProductStateManifold> attitudeManifold() {
  return ProductStateManifold::create({StateManifoldSegment::unitQuaternion(), StateManifoldSegment::euclidean(2)}).value();
}

vector_array_t randomStates(const StateManifold& manifold, size_t count) {
  vector_array_t states;
  for (size_t i = 0; i < count; ++i) {
    vector_t x = vector_t::Random(manifold.getAmbientDim());
    manifold.project(x);
    states.push_back(x);
  }
  return states;
}

TEST(ManifoldShootingHelpers, RetractTrajectoryWithoutManifoldIsIncrementTrajectory) {
  const vector_array_t x = {vector_t::Random(3), vector_t::Random(3)};
  const vector_array_t dx = {vector_t::Random(3), vector_t::Random(3)};
  vector_array_t incremented(2);
  vector_array_t retracted(2);
  multiple_shooting::incrementTrajectory(x, dx, /*alpha=*/0.3, incremented);
  multiple_shooting::retractTrajectory(/*stateManifold=*/nullptr, x, dx, /*alpha=*/0.3, retracted);
  EXPECT_EQ(retracted, incremented);
}

TEST(ManifoldShootingHelpers, RetractTrajectoryStaysOnTheManifold) {
  const std::shared_ptr<const ProductStateManifold> manifold = attitudeManifold();
  const vector_array_t x = randomStates(*manifold, /*count=*/3);
  const vector_array_t dx = {vector_t::Random(5), vector_t::Random(5), vector_t::Random(5)};
  vector_array_t xNew(3);
  multiple_shooting::retractTrajectory(manifold.get(), x, dx, /*alpha=*/0.5, xNew);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(xNew[i].head<4>().norm(), 1.0, 1e-15);
    EXPECT_TRUE(manifold->difference(x[i], xNew[i]).isApprox(0.5 * dx[i], 1e-12));
  }
}

TEST(ManifoldShootingHelpers, InterpolateStateTrajectory) {
  const std::shared_ptr<const ProductStateManifold> manifold = attitudeManifold();
  const scalar_array_t times = {0.0, 0.1, 0.1, 0.3};
  const vector_array_t states = randomStates(*manifold, /*count=*/4);
  for (const scalar_t t : {-0.1, 0.0, 0.04, 0.1, 0.2, 0.3, 0.5}) {
    // nullptr: LinearInterpolation, bit for bit.
    EXPECT_EQ(interpolateStateTrajectory(/*stateManifold=*/nullptr, t, times, states), LinearInterpolation::interpolate(t, times, states))
        << t;
    // A manifold: the geodesic, on the manifold.
    EXPECT_NEAR(interpolateStateTrajectory(manifold.get(), t, times, states).head<4>().norm(), 1.0, 1e-15) << t;
  }
  // The nodes themselves, and the pre-event node at the event time.
  EXPECT_EQ(interpolateStateTrajectory(manifold.get(), /*time=*/0.0, times, states), states[0]);
  EXPECT_EQ(interpolateStateTrajectory(manifold.get(), /*time=*/0.1, times, states), states[1]);
  EXPECT_EQ(interpolateStateTrajectory(manifold.get(), /*time=*/0.3, times, states), states[3]);
  // Between nodes: the manifold interpolation at the time fraction.
  EXPECT_TRUE(interpolateStateTrajectory(manifold.get(), /*time=*/0.15, times, states)
                  .isApprox(manifold->interpolate(states[2], states[3], /*alpha=*/0.25), 1e-12));
  EXPECT_EQ(interpolateStateTrajectory(manifold.get(), /*time=*/0.7, {0.5}, {states[2]}), states[2]);
}

TEST(ManifoldShootingHelpers, WarmStartIsInterpolatedOnTheManifold) {
  const std::shared_ptr<const ProductStateManifold> manifold = attitudeManifold();
  // A previous solution on [0, 1] with distinct node times; the new discretization on [0.05, 1.05] falls between its nodes.
  PrimalSolution previous;
  previous.timeTrajectory_ = {0.0, 0.25, 0.5, 0.75, 1.0};
  previous.stateTrajectory_ = randomStates(*manifold, /*count=*/5);
  previous.inputTrajectory_ = vector_array_t(5, vector_t::Random(1));
  const std::vector<AnnotatedTime> discretization =
      timeDiscretizationWithEvents(/*initTime=*/0.05, /*finalTime=*/1.05, /*dt=*/0.2, /*eventTimes=*/{});
  DefaultInitializer initializer(1);
  const vector_t initState = randomStates(*manifold, /*count=*/1).front();

  vector_array_t flatX;
  vector_array_t flatU;
  multiple_shooting::initializeStateInputTrajectories(initState, discretization, previous, initializer, flatX, flatU);
  vector_array_t nullX;
  vector_array_t nullU;
  multiple_shooting::initializeStateInputTrajectories(initState, discretization, previous, initializer, nullX, nullU,
                                                      /*stateManifold=*/nullptr);
  vector_array_t x;
  vector_array_t u;
  multiple_shooting::initializeStateInputTrajectories(initState, discretization, previous, initializer, x, u, manifold.get());

  ASSERT_EQ(x.size(), flatX.size());
  EXPECT_EQ(nullX, flatX);
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(x[i].head<4>().norm(), 1.0, 1e-15) << i;
  }
  // The first node is the previous solution at the initial time, interpolated along the manifold; the flat
  // initialization interpolates it linearly, off the unit sphere.
  const scalar_t initTime = getIntervalStart(discretization.front());
  EXPECT_EQ(x[0], interpolateStateTrajectory(manifold.get(), initTime, previous.timeTrajectory_, previous.stateTrajectory_));
  EXPECT_EQ(flatX[0], LinearInterpolation::interpolate(initTime, previous.timeTrajectory_, previous.stateTrajectory_));
  EXPECT_GT(std::abs(flatX[0].head<4>().norm() - 1.0), 1e-6);
  EXPECT_EQ(u, flatU);
}

TEST(ManifoldShootingHelpers, PrimalSolutionCarriesAManifoldPolicyThatCopiesThePreEventNode) {
  const std::shared_ptr<const ProductStateManifold> manifold = attitudeManifold();
  // Nodes 0, 1, 2 = pre-event, 3 = post-event, 4 = terminal.
  const std::vector<AnnotatedTime> time = {
      AnnotatedTime(/*t=*/0.0, AnnotatedTime::Event::None), AnnotatedTime(/*t=*/0.1, AnnotatedTime::Event::None),
      AnnotatedTime(/*t=*/0.15, AnnotatedTime::Event::PreEvent), AnnotatedTime(/*t=*/0.15, AnnotatedTime::Event::PostEvent),
      AnnotatedTime(/*t=*/0.3, AnnotatedTime::Event::None)};
  const vector_array_t x = randomStates(*manifold, /*count=*/5);
  const vector_array_t u = {vector_t::Random(2), vector_t::Random(2), vector_t(), vector_t::Random(2)};
  const matrix_array_t K = {matrix_t::Random(2, 5), matrix_t::Random(2, 5), matrix_t::Random(2, 5), matrix_t::Random(2, 5)};

  PrimalSolution solution = multiple_shooting::toPrimalSolution(time, ModeSchedule({0.15}, {0, 1}), vector_array_t(x), vector_array_t(u),
                                                                matrix_array_t(K), manifold);
  ASSERT_NE(solution.controllerPtr_, nullptr);
  ASSERT_EQ(solution.controllerPtr_->getType(), ControllerType::MANIFOLD_LINEAR);
  const ManifoldLinearController& controller = static_cast<const ManifoldLinearController&>(*solution.controllerPtr_);
  ASSERT_EQ(controller.size(), 5);
  // Node 1 is its own law; the pre-event node 2 takes node 1's anchor, input and gain; the terminal node takes node 3's.
  EXPECT_EQ(controller.getAnchorStates()[1], x[1]);
  EXPECT_EQ(controller.getAnchorStates()[2], x[1]);
  EXPECT_EQ(controller.getNominalInputs()[2], u[1]);
  EXPECT_EQ(controller.getFeedbackGains()[2], K[1]);
  EXPECT_EQ(controller.getAnchorStates()[4], x[3]);
  EXPECT_EQ(controller.getFeedbackGains()[4], K[3]);
  // The input trajectory is completed as in the flat version.
  EXPECT_EQ(solution.inputTrajectory_[2], u[1]);
  EXPECT_EQ(solution.inputTrajectory_.size(), 5u);
  // On the plan, the policy returns the planned input.
  EXPECT_TRUE(solution.controllerPtr_->computeInput(/*t=*/0.1, x[1]).isApprox(u[1], 1e-14));

  // Off the plan, the gain acts on the tangent difference: here a Euclidean offset, the last two tangent entries.
  vector_t offPlan = x[0];
  offPlan.tail<2>() += vector_t::Constant(2, 1e-3);
  EXPECT_TRUE(
      solution.controllerPtr_->computeInput(/*t=*/0.0, offPlan).isApprox(u[0] + K[0].rightCols<2>() * vector_t::Constant(2, 1e-3), 1e-12));
}

}  // namespace
}  // namespace ocs2
