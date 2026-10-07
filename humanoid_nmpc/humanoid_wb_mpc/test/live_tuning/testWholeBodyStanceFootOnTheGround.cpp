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
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodyLiveTuningFixture.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodySolverStack.h"

/*
 * The whole-body MPC's stance foot stands on the ground the task file puts it on (terrain_height), as its swing foot
 * lands there: the stance constraint of a robot raised with its ground is the one of the robot on flat ground, from a
 * fresh start on the raised file and one solve after a live update to it alike. The constraint used to hold every stance
 * foot at z = 0 whatever the ground.
 */
namespace ocs2::humanoid::live_tuning_test {
namespace {

// [m] How far the raised ground is above the flat one.
constexpr scalar_t kGround = 0.05;
// [s] A time at which both feet stand (WholeBodySolverStack's schedule stands until 0.1 s).
constexpr scalar_t kStanceTime = 0.05;

/**
 * The stance constraint of `foot` of the first worker's problem of `stack`, at `state` and `input`; a test failure and
 * an empty vector when it is not a hard constraint there.
 */
vector_t stanceResidual(WholeBodySolverStack& stack, const std::string& foot, const vector_t& state, const vector_t& input) {
  OptimalControlProblem& problem = stack.mpc().getSolverPtr()->getOcpDefinitions().front();
  const std::string term = zeroVelocityTermName(foot);
  if (!carriesTerm(*problem.equalityConstraintPtr, term)) {
    ADD_FAILURE() << "the problem has no hard " << term;
    return vector_t();
  }
  problem.preComputationPtr->request(Request::Constraint, kStanceTime, state, input);
  return problem.equalityConstraintPtr->get(term).getValue(kStanceTime, state, input, *problem.preComputationPtr);
}

/** The largest difference of `a` and `b`, relative to max(1, |a_i|, |b_i|); infinite for vectors of different sizes. */
scalar_t relativeDifference(const vector_t& a, const vector_t& b) {
  if (a.size() != b.size()) return std::numeric_limits<scalar_t>::infinity();
  scalar_t largest = 0.0;
  for (Eigen::Index i = 0; i < a.size(); ++i) {
    largest = std::max(largest, std::abs(a(i) - b(i)) / std::max({1.0, std::abs(a(i)), std::abs(b(i))}));
  }
  return largest;
}

TEST(WholeBodyStanceFoot, AStanceFootIsHeldOnTheGroundFromAFreshStartAndAfterALiveUpdate) {
  const mpc_config::TaskFile flatFile = shippedTaskFile();
  ASSERT_EQ(flatFile.terrain_height, 0.0) << "the shipped file raises the ground; this test sets it";
  mpc_config::TaskFile raisedFile = flatFile;
  raisedFile.terrain_height = kGround;

  WholeBodySolverStack flat(flatFile, Updater::kRegistered);
  WholeBodySolverStack fresh(raisedFile, Updater::kRegistered);
  WholeBodySolverStack live(flatFile, Updater::kRegistered);
  ASSERT_TRUE(flat.ok() && fresh.ok() && live.ok());
  // A solve plans the swing trajectories, whose stance height the constraint follows; the live update reaches the
  // reference manager at the solve after it (humanoid_mpc_config/README.md, "One solve late").
  flat.solve(/*count=*/1);
  fresh.solve(/*count=*/1);
  live.applyNow(raisedFile);
  live.solve(/*count=*/1);

  const scalar_t heightGain = flatFile.model_settings.foot_constraint.position_error_gain_z;
  ASSERT_GT(heightGain, 0.0) << "without a position gain the constraint does not see the height";
  for (const EvaluationPoint& point : evaluationPoints(flat.interface(), /*count=*/3)) {
    vector_t raisedState = point.state;
    flat.interface().getMpcRobotModel().adaptBasePoseHeight(raisedState, kGround);
    for (const std::string& foot : flat.interface().modelSettings().contactNames) {
      SCOPED_TRACE(foot);
      const vector_t onFlatGround = stanceResidual(flat, foot, point.state, point.input);
      ASSERT_EQ(onFlatGround.size(), 6);
      const vector_t onRaisedGround = stanceResidual(fresh, foot, raisedState, point.input);
      EXPECT_LE(relativeDifference(onFlatGround, onRaisedGround), 1.0e-9) << "a robot raised with its ground is held otherwise";
      EXPECT_EQ(stanceResidual(live, foot, raisedState, point.input), onRaisedGround)
          << "a live update holds the foot elsewhere than a fresh start";
      // Positive control: the robot left where it was is the ground's height below the raised ground.
      vector_t expected = onFlatGround;
      expected(2) -= heightGain * kGround;
      EXPECT_LE(relativeDifference(stanceResidual(fresh, foot, point.state, point.input), expected), 1.0e-9);
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid::live_tuning_test
