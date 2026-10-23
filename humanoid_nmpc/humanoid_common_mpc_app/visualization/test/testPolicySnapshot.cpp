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
#include <vector>

#include "Eigen/Geometry"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/ProductStateManifold.h"
#include "ocs2_core/manifold/StateManifoldSegment.h"
#include "ocs2_core/misc/LinearInterpolation.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/visualization/PolicySnapshot.h"

namespace ocs2::humanoid::visualization {
namespace {

// The toy state [p (2), xi (4: x, y, z, w), q (1)] and its manifold [E(2), Q, E(1)].
constexpr Eigen::Index kStateDim = 7;
constexpr Eigen::Index kQuaternionStart = 2;
constexpr size_t kInputDim = 2;

std::shared_ptr<const StateManifold> toyManifold() {
  const absl::StatusOr<std::shared_ptr<const ProductStateManifold>> manifold = ProductStateManifold::create(
      {StateManifoldSegment::euclidean(2), StateManifoldSegment::unitQuaternion(), StateManifoldSegment::euclidean(1)});
  EXPECT_TRUE(manifold.ok()) << manifold.status();
  return manifold.ok() ? *manifold : nullptr;
}

/** A toy state at `position`, `yaw` about z and joint `joint`. */
vector_t toyState(scalar_t position, scalar_t yaw, scalar_t joint) {
  vector_t state(kStateDim);
  state.head<2>() = vector2_t(position, -position);
  state.segment<4>(kQuaternionStart) = quaternion_t(Eigen::AngleAxis<scalar_t>(yaw, vector3_t::UnitZ())).coeffs();
  state(kStateDim - 1) = joint;
  return state;
}

/** Three nodes turning by 2 rad each, with an event (a repeated time) at 1 s between the second and the third. */
PolicySnapshot turningPlan() {
  PolicySnapshot plan;
  plan.time = {0.0, 1.0, 1.0, 2.0};
  plan.state = {toyState(0.0, 0.0, 0.1), toyState(1.0, 2.0, 0.2), toyState(1.0, 2.0, 0.3), toyState(2.0, 4.0, 0.4)};
  plan.input = {vector_t::Constant(kInputDim, 1.0), vector_t::Constant(kInputDim, 2.0), vector_t::Constant(kInputDim, 3.0),
                vector_t::Constant(kInputDim, 5.0)};
  return plan;
}

TEST(PolicySnapshotTest, ThePlansInputsAreInterpolatedLinearly) {
  const std::shared_ptr<const StateManifold> manifold = toyManifold();
  ASSERT_NE(manifold, nullptr);
  const PolicySnapshot plan = turningPlan();
  ASSERT_TRUE(isConsistentPlan(plan, kStateDim, kInputDim));
  for (const scalar_t time : {-1.0, 0.0, 0.3, 1.0, 1.5, 2.0, 3.0}) {
    vector_t state;
    vector_t input;
    samplePlan(plan, manifold.get(), time, &state, &input);
    EXPECT_EQ(input, LinearInterpolation::interpolate(time, plan.time, plan.input)) << time;
  }
}

TEST(PolicySnapshotTest, OnAManifoldTheStatesFollowItsGeodesicAndTheInputsStayLinear) {
  const std::shared_ptr<const StateManifold> manifold = toyManifold();
  ASSERT_NE(manifold, nullptr);
  const PolicySnapshot plan = turningPlan();
  vector_t state;
  vector_t input;

  // A quarter of the way from the first node to the second: the slerp turns by a quarter of 2 rad, the rest is linear.
  samplePlan(plan, manifold.get(), /*time=*/0.25, &state, &input);
  EXPECT_TRUE(state.isApprox(manifold->interpolate(plan.state[0], plan.state[1], /*alpha=*/0.25), 1.0e-15));
  EXPECT_NEAR(state.segment<4>(kQuaternionStart).norm(), 1.0, 1.0e-15);
  const quaternion_t expected(Eigen::AngleAxis<scalar_t>(0.5, vector3_t::UnitZ()));
  EXPECT_NEAR(quaternion_t(vector4_t(state.segment<4>(kQuaternionStart))).angularDistance(expected), 0.0, 1.0e-12);
  EXPECT_NEAR(state(0), 0.25, 1.0e-15);
  EXPECT_NEAR(state(kStateDim - 1), 0.125, 1.0e-15);
  EXPECT_EQ(input, LinearInterpolation::interpolate(/*enquiryTime=*/0.25, plan.time, plan.input));
  // Unlike the linear interpolation, whose quaternion is not unit.
  // NOLINTNEXTLINE(state-interpolation): the linear interpolation the manifold's is shown to differ from
  const vector_t linear = LinearInterpolation::interpolate(/*enquiryTime=*/0.25, plan.time, plan.state);
  EXPECT_GT(std::abs(linear.segment<4>(kQuaternionStart).norm() - 1.0), 1.0e-3);

  // At the event the node before it, held at both ends of the horizon.
  samplePlan(plan, manifold.get(), /*time=*/1.0, &state, &input);
  EXPECT_EQ(state, plan.state[1]);
  samplePlan(plan, manifold.get(), /*time=*/-1.0, &state, &input);
  EXPECT_EQ(state, plan.state.front());
  samplePlan(plan, manifold.get(), /*time=*/3.0, &state, &input);
  EXPECT_EQ(state, plan.state.back());
  EXPECT_EQ(input, plan.input.back());
}

TEST(PolicySnapshotTest, TheTargetIsSampledAsThePlan) {
  const std::shared_ptr<const StateManifold> manifold = toyManifold();
  ASSERT_NE(manifold, nullptr);
  const PolicySnapshot plan = turningPlan();
  const TargetTrajectories target({0.0, 1.0}, {plan.state[0], plan.state[1]}, {plan.input[0], plan.input[1]});
  vector_t state;
  vector_t input;
  ASSERT_EQ(sampleTarget(target, manifold.get(), /*time=*/0.25, kStateDim, kInputDim, &state, &input), TargetSample::kStateAndInput);
  EXPECT_TRUE(state.isApprox(manifold->interpolate(plan.state[0], plan.state[1], /*alpha=*/0.25), 1.0e-15));
  EXPECT_EQ(input, LinearInterpolation::interpolate(/*enquiryTime=*/0.25, target.timeTrajectory, target.inputTrajectory));

  // Without inputs, a state; with states of another size, nothing, and the outputs untouched.
  const TargetTrajectories statesOnly({0.0, 1.0}, {plan.state[0], plan.state[1]});
  EXPECT_EQ(sampleTarget(statesOnly, manifold.get(), /*time=*/0.5, kStateDim, kInputDim, &state, &input), TargetSample::kState);
  const vector_t before = state;
  EXPECT_EQ(sampleTarget(target, manifold.get(), /*time=*/0.5, kStateDim + 1, kInputDim, &state, &input), TargetSample::kNone);
  EXPECT_EQ(state, before);
}

}  // namespace
}  // namespace ocs2::humanoid::visualization
