/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <chrono>

#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"

namespace ocs2::humanoid {

namespace {

using Throttle = ContactPlannerModule::SnapshotThrottle;

/** A steady-clock instant `seconds` after the epoch of the test. */
Throttle::Clock::time_point at(scalar_t seconds) {
  return Throttle::Clock::time_point{} + std::chrono::duration_cast<Throttle::Clock::duration>(std::chrono::duration<scalar_t>(seconds));
}

constexpr std::chrono::duration<scalar_t> kPeriod(0.1);  // planningFrequency 10 Hz

}  // namespace

TEST(SnapshotThrottle, FirstPostIsAlwaysAllowedAndShortPlansAreCappedAtThePeriod) {
  Throttle throttle;
  EXPECT_TRUE(throttle.allows(at(0.0), kPeriod));
  throttle.posted(at(0.0));
  throttle.taken();  // the worker starts at once and finishes well within the period
  EXPECT_FALSE(throttle.allows(at(0.03), kPeriod)) << "the period is a cap on the planning rate";
  EXPECT_FALSE(throttle.allows(at(0.09), kPeriod));
  EXPECT_TRUE(throttle.allows(at(0.1), kPeriod));
}

/**
 * A snapshot posted while the worker is busy is dropped when the worker hands its plan over (the next plan has to start
 * from a schedule that contains it). Counting the period from that dropped post held the next snapshot back for a
 * whole period while the worker sat idle, so with plans of 0.1-0.2 s the planner ran at 5 Hz instead of 10. The period
 * counts from the snapshot the worker actually took, or from the one still waiting for it.
 */
TEST(SnapshotThrottle, CountsThePeriodFromTheSnapshotTheWorkerTookNotFromADroppedPost) {
  Throttle throttle;
  throttle.posted(at(0.0));
  throttle.taken();  // a plan that takes 0.15 s
  EXPECT_TRUE(throttle.allows(at(0.1), kPeriod));
  throttle.posted(at(0.1));  // posted while the worker is busy: it waits
  EXPECT_FALSE(throttle.allows(at(0.12), kPeriod)) << "a waiting snapshot holds the period";
  throttle.dropped();  // the hand-over at 0.15 drops it
  EXPECT_TRUE(throttle.allows(at(0.15), kPeriod)) << "the last snapshot the worker took is 0.15 s old: post at once";
  throttle.posted(at(0.15));
  throttle.taken();
  EXPECT_FALSE(throttle.allows(at(0.2), kPeriod)) << "and the cap holds again from the new snapshot";
  EXPECT_TRUE(throttle.allows(at(0.25), kPeriod));
}

TEST(SnapshotThrottle, ResetForgetsTheHistory) {
  Throttle throttle;
  throttle.posted(at(5.0));
  throttle.taken();
  EXPECT_FALSE(throttle.allows(at(5.01), kPeriod));
  throttle = Throttle{};
  EXPECT_TRUE(throttle.allows(at(5.01), kPeriod)) << "a restarted worker takes the first snapshot regardless of the old timestamps";
}

}  // namespace ocs2::humanoid
