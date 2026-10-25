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

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/rollout/RolloutBase.h"

#include "humanoid_centroidal_mpc/dynamics/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "support/AtlasReferenceStack.h"
#include "support/CentroidalMpcLoop.h"
#include "support/ShippedCentroidalRobot.h"

/*
 * The unit norm of the base quaternion wherever the centroidal MPC makes one
 * (design README sections 2.4 and 5(c)): a retraction of the state manifold
 * leaves |xi| within 1e-15 of 1, from any state and step; the slerped targets
 * the solver reads are unit to 1e-15; the interface's rollout projects every
 * state it returns onto the unit sphere, to 1e-15, starting off it; and a 60 s
 * lockstep loop of the shipped EngineAI SA01 - standing, walking, turning -
 * keeps every planned and rolled-out quaternion within 1e-12 of 1
 * (support/CentroidalMpcLoop.h). Generates the SA01's CppAD libraries on a cold
 * cache.
 */

namespace ocs2::humanoid {
namespace {

/** | |xi| - 1 | of the base quaternion of `state`. */
scalar_t normError(const MpcRobotModelBase<scalar_t>& model,
                   const vector_t& state) {
  return std::abs(state
                      .segment<4>(static_cast<Eigen::Index>(
                          model.getBaseOrientationStartIndex()))
                      .norm() -
                  1.0);
}

TEST(UnitNorm, AfterARetractionTheQuaternionIsUnitToRoundOff) {
  absl::StatusOr<LoadedCentroidalRobot> robot = loadShippedCentroidalRobot(
      ShippedCentroidalRobot::kDrcAtlas, /*centroidalModel=*/"");
  ASSERT_TRUE(robot.ok()) << robot.status();
  const MpcRobotModelBase<scalar_t>& model = *robot->model;
  const absl::StatusOr<std::shared_ptr<const StateManifold>> manifold =
      stateManifoldOf(model);
  ASSERT_TRUE(manifold.ok()) << manifold.status();
  ASSERT_NE(*manifold, nullptr);
  std::mt19937 generator(21);
  std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  scalar_t largest = 0.0;
  for (int sample = 0; sample < 200; ++sample) {
    vector_t state =
        vector_t::Zero(static_cast<Eigen::Index>(model.getStateDim()));
    for (Eigen::Index i = 0; i < state.size(); ++i)
      state(i) = uniform(generator);
    // Unit, or off the sphere as a solver's iterate between projections can be.
    const scalar_t scale =
        sample % 3 == 0 ? 1.0 + 0.3 * uniform(generator) : 1.0;
    state.segment<4>(orientationStart) =
        scale * state.segment<4>(orientationStart).normalized();
    vector_t step =
        vector_t::Zero(static_cast<Eigen::Index>(model.getStateTangentDim()));
    for (Eigen::Index i = 0; i < step.size(); ++i)
      step(i) = 2.0 * uniform(generator);
    for (const scalar_t alpha : {1.0, 0.5, 1.0e-3, -0.7}) {
      vector_t retracted;
      (*manifold)->retract(state, step, alpha, retracted);
      largest = std::max(largest, normError(model, retracted));
    }
  }
  EXPECT_LE(largest, 1.0e-15);
  ::testing::Test::RecordProperty("largest_norm_error", absl::StrCat(largest));
}

TEST(UnitNorm, TheSlerpedTargetsAreUnitToRoundOff) {
  // A fast turning command's knots, read at every 5 ms of the horizon through
  // the robot model's slerp.
  AtlasReferenceStack stack;
  const MpcRobotModelBase<scalar_t>& model = stack.model();
  const vector_t initState =
      stack.standingAt(stack.initialState(), /*x=*/0.2, /*y=*/0.1, /*yaw=*/3.0);
  const BasePlanarMotion measured{.velocityInHeadingFrame = vector2_t(0.3, 0.1),
                                  .headingRate = 4.0};
  const TargetTrajectories targets =
      stack.targetCalculator().commandedVelocityToTargetTrajectories(
          vector4_t(0.5, 0.0, 0.0, 1.0), /*initTime=*/2.0, initState, measured);
  ASSERT_GT(targets.size(), 3U) << "the turn must be cut into knots";
  scalar_t largest = 0.0;
  for (scalar_t time = 1.9; time <= targets.timeTrajectory.back() + 0.1;
       time += 0.005) {
    largest = std::max(largest,
                       normError(model, model.getDesiredState(targets, time)));
  }
  for (const vector_t& knot : targets.stateTrajectory)
    largest = std::max(largest, normError(model, knot));
  EXPECT_LE(largest, 1.0e-15);
}

TEST(UnitNorm, TheRolloutProjectsEveryStateOntoTheUnitSphere) {
  absl::StatusOr<std::unique_ptr<CentroidalMpcLoop>> loop =
      CentroidalMpcLoop::Create(ShippedCentroidalRobot::kEngineAiSa01,
                                CentroidalMpcLoopOptions{});
  ASSERT_TRUE(loop.ok()) << loop.status();
  CentroidalMpcInterface& interface = (*loop)->interface();
  const MpcRobotModelBase<scalar_t>& model = interface.getMpcRobotModel();
  const MpcRobotModelBase<scalar_t>& effectiveModel =
      interface.getEffectiveMpcRobotModel();
  // From a spinning robot whose quaternion is off the sphere by 10 %, its
  // weight on both feet.
  vector_t state = interface.getInitialState();
  state.segment<3>(3) << 0.4, -0.3, 1.5;
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  state.segment<4>(orientationStart) =
      1.1 * quaternionFromEulerZyx(vector3_t(0.5, 0.1, -0.1));
  const vector_t input = weightCompensatingInput(
      interface.getCentroidalModelInfo(), contact_flag_t{true, true},
      effectiveModel, state);
  FeedforwardController controller({0.0, 2.0}, {input, input});
  ModeSchedule schedule(/*eventTimesInput=*/{}, {ModeNumber::kStance});
  scalar_array_t times;
  size_array_t postEventIndices;
  vector_array_t states;
  vector_array_t inputs;
  // A clone keeps the rollout's state manifold, as the solver's workers' clones
  // do.
  const std::unique_ptr<RolloutBase> rollout(interface.getRollout().clone());
  ASSERT_NE(rollout->getStateManifold(), nullptr);
  rollout->run(/*initTime=*/0.0, state, /*finalTime=*/1.0, &controller,
               schedule, times, postEventIndices, states, inputs);
  ASSERT_GT(states.size(), 10U);
  scalar_t largest = 0.0;
  for (size_t i = 1; i < states.size(); ++i)
    largest = std::max(largest, normError(model, states[i]));
  EXPECT_LE(largest, 1.0e-15);
  // Positive control: the base did turn.
  EXPECT_GT(std::abs(model.getBaseHeading(states.back()) -
                     model.getBaseHeading(states.front())),
            0.1);
}

TEST(UnitNorm, ASixtySecondLoopKeepsEveryQuaternionUnit) {
  absl::StatusOr<std::unique_ptr<CentroidalMpcLoop>> created =
      CentroidalMpcLoop::Create(ShippedCentroidalRobot::kEngineAiSa01,
                                CentroidalMpcLoopOptions{});
  ASSERT_TRUE(created.ok()) << created.status();
  CentroidalMpcLoop& loop = **created;
  // Standing for 5 s, then 20 s of walking forward, 20 s of walking and
  // turning, 15 s of turning in place.
  struct Phase {
    scalar_t until;
    scalar_t forward;
    scalar_t yawRate;
  };
  const std::vector<Phase> phases = {
      {5.0, 0.0, 0.0}, {25.0, 0.3, 0.0}, {45.0, 0.2, 0.4}, {60.0, 0.0, -0.6}};
  scalar_t largest = 0.0;
  size_t solves = 0;
  for (const Phase& phase : phases) {
    loop.command(phase.forward, /*lateral=*/0.0, phase.yawRate);
    while (loop.observation().time < phase.until) {
      const absl::StatusOr<CentroidalMpcLoopStep> step = loop.step();
      ASSERT_TRUE(step.ok())
          << "t = " << loop.observation().time << ": " << step.status();
      largest = std::max(largest, step->quaternionNormError);
      ++solves;
    }
  }
  EXPECT_LE(largest, 1.0e-12);
  EXPECT_GT(solves, 1000U);
  ::testing::Test::RecordProperty("largest_norm_error", absl::StrCat(largest));
  ::testing::Test::RecordProperty("solves", static_cast<int>(solves));
}

}  // namespace
}  // namespace ocs2::humanoid
