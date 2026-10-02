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

#include <vector>

#include "humanoid_mpc_validation/closed_loop/LockstepSolveSchedule.h"

/*
 * When the lockstep closed loop solves: the pacing and the failure back-off of InProcessMpcLink's solver thread on the
 * simulation's clock, which the runner keeps itself (Execution::kCaller).
 */

namespace ocs2::humanoid::validation {
namespace {

constexpr double kTolerance = 1e-9;
/** [s] A control cycle of 2 ms. */
constexpr double kCycle = 0.002;

/** The cycles in [start, end) at which `schedule` solves, with no reset requested and no pause asked for. */
std::vector<double> solveTimes(LockstepSolveSchedule& schedule, double start, double end) {
  std::vector<double> times;
  for (int cycle = 0; start + cycle * kCycle < end - kTolerance; ++cycle) {
    const double time = start + cycle * kCycle;
    if (!schedule.isDue(time, /*resetRequestedSinceFailure=*/false)) continue;
    times.push_back(time);
    schedule.onSolve(time, /*retryDelay=*/0.0);
  }
  return times;
}

TEST(LockstepSolveSchedule, SolvesInTheFirstCycleAtOrAfterEachPeriod) {
  // 60 Hz on 2 ms cycles: due at 1/60, 2/60, ...; each in the first cycle at or after it, on a grid that does not drift.
  LockstepSolveSchedule schedule(/*firstSolveTime=*/1.0 / 60.0, /*period=*/1.0 / 60.0, kTolerance);
  const std::vector<double> times = solveTimes(schedule, /*start=*/0.0, /*end=*/1.0);
  ASSERT_EQ(times.size(), 59u);
  for (size_t k = 0; k < times.size(); ++k) {
    const double due = static_cast<double>(k + 1) / 60.0;
    EXPECT_GE(times[k], due - kTolerance) << k;
    EXPECT_LT(times[k], due + kCycle - kTolerance) << k;
  }
  EXPECT_FALSE(schedule.isPaused());
}

TEST(LockstepSolveSchedule, APauseHoldsTheSolvesUntilItsDelayHasPassed) {
  LockstepSolveSchedule schedule(/*firstSolveTime=*/0.01, /*period=*/0.01, kTolerance);
  ASSERT_TRUE(schedule.isDue(/*time=*/0.01, /*resetRequestedSinceFailure=*/false));
  schedule.onSolve(/*time=*/0.01, /*retryDelay=*/0.1);
  EXPECT_TRUE(schedule.isPaused());
  // The solves the period would have made are held back ...
  EXPECT_FALSE(schedule.isDue(/*time=*/0.02, /*resetRequestedSinceFailure=*/false));
  EXPECT_FALSE(schedule.isDue(/*time=*/0.108, /*resetRequestedSinceFailure=*/false));
  // ... until the delay has passed: then at once, and the period counts from that solve.
  ASSERT_TRUE(schedule.isDue(/*time=*/0.11, /*resetRequestedSinceFailure=*/false));
  schedule.onSolve(/*time=*/0.11, /*retryDelay=*/0.0);
  EXPECT_FALSE(schedule.isPaused());
  EXPECT_FALSE(schedule.isDue(/*time=*/0.118, /*resetRequestedSinceFailure=*/false));
  EXPECT_TRUE(schedule.isDue(/*time=*/0.12, /*resetRequestedSinceFailure=*/false));
}

TEST(LockstepSolveSchedule, AResetRequestedAfterTheFailureEndsThePauseAtOnce) {
  // As MpcResetSupervisor::waitBeforeRetry() returns when a reset is requested after the failure: an operator
  // re-entering WB_MPC, or the gantry released, does not wait out the back-off.
  LockstepSolveSchedule schedule(/*firstSolveTime=*/0.01, /*period=*/0.01, kTolerance);
  schedule.onSolve(/*time=*/0.01, /*retryDelay=*/2.0);
  EXPECT_FALSE(schedule.isDue(/*time=*/0.5, /*resetRequestedSinceFailure=*/false));
  ASSERT_TRUE(schedule.isDue(/*time=*/0.5, /*resetRequestedSinceFailure=*/true));
  // That solve fails again: a new pause, which the request made before it does not end.
  schedule.onSolve(/*time=*/0.5, /*retryDelay=*/0.2);
  EXPECT_TRUE(schedule.isPaused());
  EXPECT_FALSE(schedule.isDue(/*time=*/0.6, /*resetRequestedSinceFailure=*/false));
  schedule.onSolve(/*time=*/0.7, /*retryDelay=*/0.0);
  EXPECT_FALSE(schedule.isPaused());
  EXPECT_TRUE(schedule.isDue(/*time=*/0.71, /*resetRequestedSinceFailure=*/false));
}

TEST(LockstepSolveSchedule, ARequestedResetDoesNotAdvanceASolveWithoutAPause) {
  // Outside a pause a reset waits for the next due solve, as the solver thread serves it at the start of its next
  // iteration.
  LockstepSolveSchedule schedule(/*firstSolveTime=*/0.01, /*period=*/0.01, kTolerance);
  schedule.onSolve(/*time=*/0.01, /*retryDelay=*/0.0);
  EXPECT_FALSE(schedule.isDue(/*time=*/0.014, /*resetRequestedSinceFailure=*/true));
  EXPECT_TRUE(schedule.isDue(/*time=*/0.02, /*resetRequestedSinceFailure=*/true));
}

TEST(LockstepSolveSchedule, WithoutAPeriodEveryCycleSolves) {
  LockstepSolveSchedule schedule(/*firstSolveTime=*/0.0, /*period=*/0.0, kTolerance);
  EXPECT_EQ(solveTimes(schedule, /*start=*/0.0, /*end=*/0.02).size(), 10u);
}

}  // namespace
}  // namespace ocs2::humanoid::validation
