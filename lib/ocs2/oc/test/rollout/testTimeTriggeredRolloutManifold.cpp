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

#include <memory>

#include <ocs2_core/control/FeedforwardController.h>
#include <ocs2_core/initialization/DefaultInitializer.h>

#include "ocs2_oc/rollout/InitializerRollout.h"
#include "ocs2_oc/rollout/StateTriggeredRollout.h"
#include "ocs2_oc/rollout/TimeTriggeredRollout.h"
#include "ocs2_oc/test/RigidBodyAttitudeDynamics.h"

namespace ocs2 {
namespace {

using manifold_test::RigidBodyAttitudeDynamics;

struct Rollout {
  scalar_array_t time;
  size_array_t postEventIndices;
  vector_array_t state;
  vector_array_t input;
};

Rollout rollOut(TimeTriggeredRollout& rollout, ModeSchedule modeSchedule) {
  vector_t torque(3);
  torque << 0.4, -0.3, 0.8;
  FeedforwardController controller({0.0, 10.0}, {torque, torque});
  vector_t initState(7);
  initState << 0.1, -0.2, 0.3, 0.9, 2.0, -1.0, 3.0;  // a fast spin, so the integrator's norm drift is visible
  initState.head<4>().normalize();
  Rollout result;
  rollout.run(/*initTime=*/0.0, initState, /*finalTime=*/3.0, &controller, modeSchedule, result.time, result.postEventIndices, result.state,
              result.input);
  return result;
}

TEST(TimeTriggeredRolloutManifold, ProjectsEveryOutputState) {
  const RigidBodyAttitudeDynamics dynamics(/*withPosition=*/false);
  rollout::Settings settings;
  settings.integratorType = IntegratorType::RK4;
  settings.timeStep = 0.05;
  TimeTriggeredRollout flat(dynamics, settings);
  TimeTriggeredRollout projected(dynamics, settings);
  projected.setStateManifold(dynamics.getStateManifold());

  const Rollout flatRollout = rollOut(flat, ModeSchedule());
  const Rollout projectedRollout = rollOut(projected, ModeSchedule());
  ASSERT_EQ(projectedRollout.state.size(), flatRollout.state.size());
  scalar_t flatDrift = 0.0;
  for (size_t k = 0; k < flatRollout.state.size(); ++k) {
    flatDrift = std::max(flatDrift, std::abs(flatRollout.state[k].head<4>().norm() - 1.0));
    EXPECT_NEAR(projectedRollout.state[k].head<4>().norm(), 1.0, 1e-15) << k;
    // Without an event the integration itself is unchanged: the output is the flat output, projected.
    vector_t expected = flatRollout.state[k];
    dynamics.getStateManifold()->project(expected);
    EXPECT_EQ(projectedRollout.state[k], expected) << k;
  }
  EXPECT_GT(flatDrift, 1e-12) << "RK4 should drift off the sphere at this spin rate and step";
}

TEST(TimeTriggeredRolloutManifold, ProjectsAcrossEventsAndKeepsTheManifoldWhenCloned) {
  const RigidBodyAttitudeDynamics dynamics(/*withPosition=*/false);
  TimeTriggeredRollout rollout(dynamics);
  rollout.setStateManifold(dynamics.getStateManifold());
  std::unique_ptr<RolloutBase> clone(rollout.clone());
  EXPECT_EQ(clone->getStateManifold(), rollout.getStateManifold());

  const Rollout result = rollOut(static_cast<TimeTriggeredRollout&>(*clone), ModeSchedule({1.0, 2.0}, {0, 1, 2}));
  ASSERT_EQ(result.postEventIndices.size(), 2u);
  for (const vector_t& state : result.state) {
    EXPECT_NEAR(state.head<4>().norm(), 1.0, 1e-15);
  }
}

TEST(TimeTriggeredRolloutManifold, RolloutsThatCannotHonorAManifoldRejectIt) {
  // Only TimeTriggeredRollout projects its states and keeps the manifold when cloned. The others must not accept one
  // and then silently integrate (and clone) as if the state were flat.
  const RigidBodyAttitudeDynamics dynamics(/*withPosition=*/false);
  TimeTriggeredRollout timeTriggered(dynamics);
  StateTriggeredRollout stateTriggered(dynamics);
  InitializerRollout initializer(DefaultInitializer(/*inputDim=*/3));
  EXPECT_TRUE(timeTriggered.supportsStateManifold());
  EXPECT_FALSE(stateTriggered.supportsStateManifold());
  EXPECT_FALSE(initializer.supportsStateManifold());

  // The flat default stays settable on every rollout.
  stateTriggered.setStateManifold(/*stateManifold=*/nullptr);
  initializer.setStateManifold(/*stateManifold=*/nullptr);
  EXPECT_EQ(stateTriggered.getStateManifold(), nullptr);

  EXPECT_DEATH(stateTriggered.setStateManifold(dynamics.getStateManifold()), "does not support a state manifold");
  EXPECT_DEATH(initializer.setStateManifold(dynamics.getStateManifold()), "does not support a state manifold");
}

}  // namespace
}  // namespace ocs2
