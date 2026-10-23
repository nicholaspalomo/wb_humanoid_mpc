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
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "humanoid_mpc_validation/closed_loop/LockstepSolveSchedule.h"

/*
 * When the lockstep closed loop solves: the pacing and the failure back-off of InProcessMpcLink's solver thread on the
 * simulation's clock, which the runner keeps itself (Execution::kCaller).
 */

namespace ocs2::humanoid::validation {
namespace {

constexpr double kTolerance = 1.0e-9;
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

/** The grid time after `time` as the schedule once found it: one period at a time. */
double firstPeriodicTimeAfterByAddition(double scheduled, double period, double time) {
  while (scheduled <= time) scheduled += period;
  return scheduled;
}

TEST(FirstPeriodicTimeAfter, IsTheFirstGridTimeAfterTheGivenOne) {
  std::mt19937_64 generator(/*seed=*/11);
  std::uniform_real_distribution<double> uniform(/*a=*/0.0, /*b=*/1.0);
  for (int trial = 0; trial < 10000; ++trial) {
    const double scheduled = 10.0 * uniform(generator);
    const double period = 1.0e-3 + 0.1 * uniform(generator);
    const double time = scheduled + (trial % 4 == 0 ? -1.0 : 200.0) * period * uniform(generator);
    const double next = firstPeriodicTimeAfter(scheduled, period, time);
    const double oracle = firstPeriodicTimeAfterByAddition(scheduled, period, time);
    EXPECT_GT(next, time) << trial;
    if (scheduled > time) {
      EXPECT_EQ(next, scheduled) << trial;
    } else {
      EXPECT_LE(next - period, time + 1.0e-12) << trial << ": the first grid time after, not a later one";
    }
    // The same grid point as the addition, up to the rounding its sums accumulate.
    EXPECT_NEAR(next, oracle, 1.0e-12 * (1.0 + std::abs(time - scheduled) / period)) << trial;
  }
}

TEST(FirstPeriodicTimeAfter, OnePeriodAheadIsTheSumBitForBit) {
  std::mt19937_64 generator(/*seed=*/12);
  std::uniform_real_distribution<double> uniform(/*a=*/0.0, /*b=*/1.0);
  for (int trial = 0; trial < 10000; ++trial) {
    const double scheduled = 10.0 * uniform(generator);
    const double period = 1.0e-3 + 0.1 * uniform(generator);
    const double time = scheduled + 0.999 * period * uniform(generator);
    EXPECT_EQ(firstPeriodicTimeAfter(scheduled, period, time), scheduled + period) << trial;
  }
}

TEST(FirstPeriodicTimeAfter, SkipsAnyNumberOfPeriodsAtOnce) {
  // A billion periods, which the addition would take a billion steps for.
  const double period = 1.0 / 60.0;
  const double time = 1.0e9 * period + 0.25 * period;
  const double next = firstPeriodicTimeAfter(/*scheduled=*/0.0, period, time);
  EXPECT_GT(next, time);
  EXPECT_LE(next - period, time);
}

TEST(LockstepSolveSchedule, AfterALongGapTheNextSolveIsOnePeriodOnTheGrid) {
  LockstepSolveSchedule schedule(/*firstSolveTime=*/0.0, /*period=*/0.01, kTolerance);
  ASSERT_TRUE(schedule.isDue(/*time=*/0.0, /*resetRequestedSinceFailure=*/false));
  schedule.onSolve(/*time=*/1000.004, /*retryDelay=*/0.0);
  EXPECT_FALSE(schedule.isDue(/*time=*/1000.009, /*resetRequestedSinceFailure=*/false));
  EXPECT_TRUE(schedule.isDue(/*time=*/1000.010, /*resetRequestedSinceFailure=*/false));
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
