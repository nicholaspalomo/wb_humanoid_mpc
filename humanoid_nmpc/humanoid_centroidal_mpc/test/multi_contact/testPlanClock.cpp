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

#include <cstddef>
#include <functional>
#include <vector>

#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/multi_contact/PlanClock.h"

/*
 * The clock of a tracked plan (PlanClock): it runs with the solves, stops at
 * the hold point lead seconds before a contact change until the stance is
 * reached, never skips a change, and maps a solve's time to plan time by an
 * offset that grows by the time held; a periodic plan's changes repeat every
 * period and are held across its wrap.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kLead = 0.1;
constexpr scalar_t kTolerance = 1.0e-12;

TEST(PlanClock, runsWithTheSolvesFromTheFirstOne) {
  PlanClock clock({1.0, 2.0}, kLead, /*startTime=*/0.0);
  const std::function<bool(size_t, scalar_t)> ready = [](size_t, scalar_t) {
    return true;
  };
  EXPECT_NEAR(clock.advance(/*time=*/5.0, ready), 0.0, kTolerance);
  EXPECT_NEAR(clock.advance(/*time=*/5.5, ready), 0.5, kTolerance);
  EXPECT_NEAR(clock.offset(), 5.0, kTolerance);
  EXPECT_FALSE(clock.isHolding());
}

TEST(PlanClock, holdsBeforeAChangeUntilTheStanceIsReached) {
  PlanClock clock({1.0}, kLead, /*startTime=*/0.0);
  bool reached = false;
  std::vector<scalar_t> judgedAt;
  const std::function<bool(size_t, scalar_t)> ready = [&](size_t event,
                                                          scalar_t holdTime) {
    EXPECT_EQ(event, 0U);
    judgedAt.push_back(holdTime);
    return reached;
  };
  EXPECT_NEAR(clock.advance(/*time=*/0.0, ready), 0.0, kTolerance);
  EXPECT_NEAR(clock.advance(/*time=*/0.85, ready), 0.85, kTolerance);
  // Past the hold point, 0.9 s, and not reached: held there.
  EXPECT_NEAR(clock.advance(/*time=*/0.95, ready), 0.9, kTolerance);
  EXPECT_TRUE(clock.isHolding());
  EXPECT_NEAR(clock.advance(/*time=*/3.0, ready), 0.9, kTolerance);
  EXPECT_NEAR(clock.offset(), 2.1, kTolerance);
  ASSERT_FALSE(judgedAt.empty());
  EXPECT_NEAR(judgedAt.back(), 0.9, kTolerance);
  // Reached: released, and it runs on from the hold point.
  reached = true;
  EXPECT_NEAR(clock.advance(/*time=*/3.02, ready), 0.92, kTolerance);
  EXPECT_FALSE(clock.isHolding());
  EXPECT_EQ(clock.releasedEvents(), 1U);
  EXPECT_NEAR(clock.advance(/*time=*/3.5, ready), 1.4, kTolerance);
}

TEST(PlanClock, neverSkipsAChangeOnALongStep) {
  PlanClock clock({1.0, 1.2, 1.4}, kLead, /*startTime=*/0.0);
  size_t judged = 0;
  const std::function<bool(size_t, scalar_t)> ready = [&judged](size_t,
                                                                scalar_t) {
    ++judged;
    return true;
  };
  clock.advance(/*time=*/0.0, ready);
  // A stalled solve of 2 s releases one change and stops at the next hold
  // point.
  EXPECT_NEAR(clock.advance(/*time=*/2.0, ready), 1.1, kTolerance);
  EXPECT_EQ(clock.releasedEvents(), 1U);
  EXPECT_NEAR(clock.advance(/*time=*/4.0, ready), 1.3, kTolerance);
  EXPECT_EQ(clock.releasedEvents(), 2U);
  EXPECT_NEAR(clock.advance(/*time=*/6.0, ready), 1.3 + 2.0, kTolerance);
  EXPECT_EQ(clock.releasedEvents(), 3U);
  EXPECT_EQ(judged, 3U);
}

TEST(PlanClock, aChangeWithinTheLeadOfTheStartIsJudgedAtTheStart) {
  PlanClock clock({0.05}, kLead, /*startTime=*/0.0);
  scalar_t judgedAt = -1.0;
  const std::function<bool(size_t, scalar_t)> ready =
      [&judgedAt](size_t, scalar_t holdTime) {
        judgedAt = holdTime;
        return false;
      };
  EXPECT_NEAR(clock.advance(/*time=*/1.0, ready), 0.0, kTolerance);
  EXPECT_NEAR(judgedAt, 0.0, kTolerance);
  EXPECT_TRUE(clock.isHolding());
}

TEST(PlanClock, resetStartsThePlanAnew) {
  PlanClock clock({1.0}, kLead, /*startTime=*/0.0);
  const std::function<bool(size_t, scalar_t)> ready = [](size_t, scalar_t) {
    return true;
  };
  clock.advance(/*time=*/0.0, ready);
  clock.advance(/*time=*/2.0, ready);
  clock.reset();
  EXPECT_NEAR(clock.advance(/*time=*/10.0, ready), 0.0, kTolerance);
  EXPECT_EQ(clock.releasedEvents(), 0U);
}

TEST(PlanClock, aPeriodicPlansChangesRepeatEveryPeriod) {
  PlanClock clock({1.0, 2.0}, kLead, /*startTime=*/0.0, /*period=*/3.0);
  EXPECT_NEAR(clock.eventTime(2), 4.0, kTolerance);
  EXPECT_NEAR(clock.eventTime(5), 8.0, kTolerance);
  EXPECT_NEAR(clock.holdTime(4), 7.0 - kLead, kTolerance);
  const std::function<bool(size_t, scalar_t)> ready = [](size_t, scalar_t) {
    return true;
  };
  // Every change of three periods released, solve after solve, the clock
  // never held.
  for (int step = 0; step <= 180; ++step) {
    const scalar_t time = 0.05 * step;
    EXPECT_NEAR(clock.advance(time, ready), time, kTolerance);
    EXPECT_FALSE(clock.isHolding());
  }
  EXPECT_EQ(clock.releasedEvents(), 6U);
}

TEST(PlanClock, holdsAcrossTheWrapOfAPeriodicPlan) {
  PlanClock clock({1.0}, kLead, /*startTime=*/0.0, /*period=*/2.0);
  bool reached = true;
  std::vector<size_t> judged;
  const std::function<bool(size_t, scalar_t)> ready = [&](size_t event,
                                                          scalar_t holdTime) {
    judged.push_back(event);
    EXPECT_NEAR(holdTime, clock.eventTime(event) - kLead, kTolerance);
    return reached;
  };
  clock.advance(/*time=*/0.0, ready);
  clock.advance(/*time=*/1.0, ready);
  EXPECT_EQ(clock.releasedEvents(), 1U);
  // The second period's change, at plan time 3 s: held at 2.9 s until its
  // stance is reached.
  reached = false;
  EXPECT_NEAR(clock.advance(/*time=*/2.95, ready), 2.9, kTolerance);
  EXPECT_TRUE(clock.isHolding());
  EXPECT_NEAR(clock.advance(/*time=*/5.0, ready), 2.9, kTolerance);
  EXPECT_TRUE(clock.isHolding());
  ASSERT_FALSE(judged.empty());
  EXPECT_EQ(judged.back(), 1U);
  reached = true;
  EXPECT_NEAR(clock.advance(/*time=*/5.02, ready), 2.92, kTolerance);
  EXPECT_FALSE(clock.isHolding());
  EXPECT_EQ(clock.releasedEvents(), 2U);
  // And the third period's, at 5 s, a period later.
  EXPECT_NEAR(clock.holdTime(clock.releasedEvents()), 5.0 - kLead, kTolerance);
}

}  // namespace
}  // namespace ocs2::humanoid
