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

#include <utility>
#include <vector>

#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

namespace {

/** Alternating single supports of `period` seconds each, starting with the left foot in swing at t = 0. */
ModeSchedule alternatingSingleSupport(scalar_t period, size_t cycles) {
  // Built through the two-argument constructor, not by push_back onto a default-constructed schedule: the default
  // constructor is ModeSchedule({}, {0}), i.e. it already carries one FLY mode, so appending to it silently produces a
  // schedule whose first phase is a flight phase and whose modes and events no longer line up.
  std::vector<size_t> modes;
  std::vector<scalar_t> events;
  for (size_t i = 0; i < 2 * cycles; ++i) {
    modes.push_back(i % 2 == 0 ? RF : LF);
    events.push_back(static_cast<scalar_t>(i + 1) * period);
  }
  modes.push_back(STANCE);  // N modes, N - 1 events: the trailing phase is unbounded
  return ModeSchedule(std::move(events), std::move(modes));
}

}  // namespace

TEST(StanceDutyFactor, AlternatingSingleSupportIsHalfForEachFoot) {
  // The cadence Bledt's impulse scaling is about: each foot is down for half of the cycle, so while it IS down it has
  // to carry twice what static weight compensation would ask of it.
  const ModeSchedule schedule = alternatingSingleSupport(0.25, 4);
  EXPECT_NEAR(stanceDutyFactor(schedule, CONTACT_LEFT_INDEX, 0.0, 2.0), 0.5, 1e-9);
  EXPECT_NEAR(stanceDutyFactor(schedule, CONTACT_RIGHT_INDEX, 0.0, 2.0), 0.5, 1e-9);
}

TEST(StanceDutyFactor, StandingIsOneForBothFeet) {
  // A real standing schedule: both feet down throughout, with events so that the walk has something to measure.
  const ModeSchedule schedule(std::vector<scalar_t>{0.5, 1.0}, std::vector<size_t>{STANCE, STANCE, STANCE});
  // 1 is also the value at which the impulse-scaling correction is identically zero, so a robot that never lifts a
  // foot gets exactly the contact-force reference it had before the heuristic existed.
  EXPECT_NEAR(stanceDutyFactor(schedule, CONTACT_LEFT_INDEX, 0.0, 1.0), 1.0, 1e-12);
  EXPECT_NEAR(stanceDutyFactor(schedule, CONTACT_RIGHT_INDEX, 0.0, 1.0), 1.0, 1e-12);
}

TEST(StanceDutyFactor, DegenerateInputsReturnOneRatherThanZero) {
  // Every caller divides by beta, so the answer to "I cannot tell" has to be the one that makes the correction vanish
  // rather than the one that makes it infinite.
  //
  // The default-constructed schedule is the case that matters in practice: ModeSchedule() is ModeSchedule({}, {0}),
  // one FLY mode and no events, and it is what the reference manager holds until its first modifyReferences(). Read
  // literally it says no foot is ever in contact, which would clamp to minimumDutyFactor and multiply the
  // contact-force reference fivefold on the strength of a placeholder.
  const ModeSchedule placeholder;
  ASSERT_FALSE(placeholder.modeSequence.empty()) << "the default schedule is not empty; it carries one FLY mode";
  ASSERT_TRUE(placeholder.eventTimes.empty());
  EXPECT_NEAR(stanceDutyFactor(placeholder, CONTACT_LEFT_INDEX, 0.0, 1.0), 1.0, 1e-12);

  const ModeSchedule schedule = alternatingSingleSupport(0.25, 2);
  EXPECT_NEAR(stanceDutyFactor(schedule, CONTACT_LEFT_INDEX, 0.0, 0.0), 1.0, 1e-12);
  EXPECT_NEAR(stanceDutyFactor(schedule, CONTACT_LEFT_INDEX, 0.0, -1.0), 1.0, 1e-12);
}

TEST(StanceDutyFactor, IsMeasuredOverWhatTheScheduleCoversNotTheRequestedWindow) {
  // A window running past the end of the mode sequence must not report the missing tail as swing: that would scale
  // the contact force up on the strength of events nobody scheduled.
  const ModeSchedule schedule = alternatingSingleSupport(0.25, 2);  // events at 0.25 .. 1.0, then a trailing STANCE
  EXPECT_NEAR(stanceDutyFactor(schedule, CONTACT_LEFT_INDEX, 0.0, 1.0), 0.5, 1e-9);
  // The trailing phase is double support and is unbounded, so a longer window is mostly stance for both feet.
  EXPECT_GT(stanceDutyFactor(schedule, CONTACT_LEFT_INDEX, 0.0, 10.0), 0.9);
}

TEST(StanceDutyFactor, IsAlwaysAFraction) {
  const ModeSchedule schedule = alternatingSingleSupport(0.25, 4);
  for (const scalar_t time : {0.0, 0.1, 0.3, 0.7, 1.4, 5.0}) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const scalar_t beta = stanceDutyFactor(schedule, foot, time, 1.0);
      EXPECT_GE(beta, 0.0);
      EXPECT_LE(beta, 1.0);
    }
  }
}

}  // namespace ocs2::humanoid
