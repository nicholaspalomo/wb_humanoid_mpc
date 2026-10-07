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

// The solution time window that the MPC node cuts every policy to before it sends it, whatever the solver does with the
// final time it is given (SqpSolver ignores it): upstream OCS2 GaussNewtonDDP's rule, on the solutions of the multiple-shooting
// solvers.

#include <cstddef>
#include <random>

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/control/ControllerType.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_mpc_ipc/SolutionTimeWindow.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/PolicyTestData.h"

namespace ocs2::humanoid::ipc {
namespace {

using test_data::PolicyShape;

TEST(SolutionWindowLengthTest, KeepsEveryTimeUpToTheEndAndOneBeyond) {
  const scalar_array_t times = {0.0, 0.1, 0.2, 0.3, 0.4};
  EXPECT_EQ(solutionWindowLength(times, /*finalTime=*/0.25), 4u) << "0.3 spans the window to 0.25";
  EXPECT_EQ(solutionWindowLength(times, /*finalTime=*/0.2), 4u) << "a node exactly at the end is kept, and one beyond it";
  EXPECT_EQ(solutionWindowLength(times, /*finalTime=*/0.4), 5u);
  EXPECT_EQ(solutionWindowLength(times, /*finalTime=*/7.0), 5u) << "a solution shorter than the window is kept whole";
  EXPECT_EQ(solutionWindowLength(times, /*finalTime=*/-1.0), 1u) << "a non-empty solution keeps its first node";
  EXPECT_EQ(solutionWindowLength(scalar_array_t(), /*finalTime=*/1.0), 0u);
}

TEST(SolutionWindowLengthTest, AnEventAtTheEndKeepsBothOfItsNodes) {
  // The pre- and post-event nodes of a multiple-shooting solution share one time.
  const scalar_array_t times = {0.0, 0.1, 0.2, 0.2, 0.3, 0.4};
  EXPECT_EQ(solutionWindowLength(times, /*finalTime=*/0.2), 5u);
}

TEST(TrimToSolutionWindowTest, CutsTheTrajectoriesTheEventsAndTheControllerAlike) {
  for (const ControllerType controllerType : {ControllerType::FEEDFORWARD, ControllerType::LINEAR}) {
    std::mt19937 generator(/*sd=*/7);
    const PolicyShape shape{.nodes = 40, .stateDim = 4, .inputDim = 3, .events = 6, .controllerType = controllerType};
    const PrimalSolution original = test_data::randomPrimalSolution(generator, shape);
    const scalar_t finalTime = original.timeTrajectory_[original.timeTrajectory_.size() / 2];
    PrimalSolution trimmed = original;
    trimToSolutionWindow(finalTime, &trimmed);

    const size_t length = solutionWindowLength(original.timeTrajectory_, finalTime);
    ASSERT_LT(length, original.timeTrajectory_.size()) << "positive control: the window cuts something off";
    ASSERT_EQ(trimmed.timeTrajectory_.size(), length);
    EXPECT_EQ(trimmed.stateTrajectory_.size(), length);
    EXPECT_EQ(trimmed.inputTrajectory_.size(), length);
    EXPECT_GT(trimmed.timeTrajectory_.back(), finalTime) << "the kept trajectory spans the window";
    for (size_t node = 0; node < length; ++node) {
      EXPECT_EQ(trimmed.timeTrajectory_[node], original.timeTrajectory_[node]);
      EXPECT_EQ(trimmed.stateTrajectory_[node], original.stateTrajectory_[node]);
      EXPECT_EQ(trimmed.inputTrajectory_[node], original.inputTrajectory_[node]);
    }
    // Every event whose post-event node is kept, and no other.
    for (const size_t index : trimmed.postEventIndices_) EXPECT_LT(index, length);
    size_t keptEvents = 0;
    for (const size_t index : original.postEventIndices_) keptEvents += index < length ? 1 : 0;
    EXPECT_EQ(trimmed.postEventIndices_.size(), keptEvents);
    EXPECT_EQ(trimmed.modeSchedule_.eventTimes, original.modeSchedule_.eventTimes) << "the mode schedule is kept whole";

    // The controller is the original one on the kept nodes: the same input at every kept node, of either type.
    ASSERT_NE(trimmed.controllerPtr_, nullptr);
    EXPECT_EQ(trimmed.controllerPtr_->size(), static_cast<int>(length));
    for (size_t node = 0; node + 1 < length; ++node) {
      const scalar_t time = 0.5 * (trimmed.timeTrajectory_[node] + trimmed.timeTrajectory_[node + 1]);
      const vector_t& state = trimmed.stateTrajectory_[node];
      EXPECT_EQ(trimmed.controllerPtr_->computeInput(time, state), original.controllerPtr_->computeInput(time, state)) << "node " << node;
    }
  }
}

TEST(TrimToSolutionWindowTest, ASolutionInsideTheWindowIsLeftAsItIs) {
  std::mt19937 generator(/*sd=*/8);
  const PolicyShape shape{.nodes = 12, .stateDim = 2, .inputDim = 2, .events = 2, .controllerType = ControllerType::LINEAR};
  const PrimalSolution original = test_data::randomPrimalSolution(generator, shape);
  PrimalSolution trimmed = original;
  trimToSolutionWindow(original.timeTrajectory_.back() + 1.0, &trimmed);
  EXPECT_EQ(trimmed.timeTrajectory_, original.timeTrajectory_);
  EXPECT_EQ(trimmed.postEventIndices_, original.postEventIndices_);
  EXPECT_EQ(trimmed.controllerPtr_->size(), original.controllerPtr_->size());
}

}  // namespace
}  // namespace ocs2::humanoid::ipc
