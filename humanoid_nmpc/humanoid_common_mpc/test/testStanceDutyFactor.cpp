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

#include <optional>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

namespace {

/**
 * The value of `optional`, or a failed expectation and a value-initialized T. A test reads an optional through it, or
 * after an `if (!x.has_value()) FAIL()`, because bugprone-unchecked-optional-access does not see googletest's ASSERTs.
 */
template <typename T>
T valueOrFailure(std::optional<T> optional) {
  if (!optional.has_value()) {
    ADD_FAILURE() << "the optional has no value";
    return T();
  }
  return *std::move(optional);
}

/** Alternating single supports of `period` seconds each, starting with the left foot in swing at t = 0. */
ModeSchedule alternatingSingleSupport(scalar_t period, size_t cycles) {
  // Built through the two-argument constructor, not by push_back onto a default-constructed schedule: the default
  // constructor is ModeSchedule({}, {0}), i.e. it already carries one FLY mode, so appending to it silently produces a
  // schedule whose first phase is a flight phase and whose modes and events no longer line up.
  std::vector<size_t> modes;
  std::vector<scalar_t> events;
  for (size_t i = 0; i < 2 * cycles; ++i) {
    modes.push_back(i % 2 == 0 ? ModeNumber::kRf : ModeNumber::kLf);
    events.push_back(static_cast<scalar_t>(i + 1) * period);
  }
  modes.push_back(ModeNumber::kStance);  // N modes, N - 1 events: the trailing phase is unbounded
  return ModeSchedule(std::move(events), std::move(modes));
}

/**
 * The shipped `walk` gait (gait.textproto): each foot swings 0.6 s, with 0.1 s of double support between, a 1.4 s stride.
 * Left in swing first. `leadIn` seconds of standing before it, and a trailing STANCE after `cycles` strides.
 */
ModeSchedule walk(size_t cycles, scalar_t leadIn = 0.0) {
  std::vector<size_t> modes;
  std::vector<scalar_t> events;
  scalar_t time = 0.0;
  if (leadIn > 0.0) {
    modes.push_back(ModeNumber::kStance);
    time += leadIn;
    events.push_back(time);
  }
  for (size_t i = 0; i < cycles; ++i) {
    for (const std::pair<size_t, scalar_t>& phase :
         {std::pair<size_t, scalar_t>{ModeNumber::kRf, 0.6}, std::pair<size_t, scalar_t>{ModeNumber::kStance, 0.1},
          std::pair<size_t, scalar_t>{ModeNumber::kLf, 0.6}, std::pair<size_t, scalar_t>{ModeNumber::kStance, 0.1}}) {
      modes.push_back(phase.first);
      time += phase.second;
      events.push_back(time);
    }
  }
  modes.push_back(ModeNumber::kStance);
  return ModeSchedule(std::move(events), std::move(modes));
}

}  // namespace

TEST(StanceDutyFactor, AlternatingSingleSupportIsHalfForEachFoot) {
  // The cadence Bledt's impulse scaling is about: each foot is down for half of its stride, so while it IS down it has
  // to carry twice what static weight compensation would ask of it.
  const ModeSchedule schedule = alternatingSingleSupport(0.25, 4);
  for (const scalar_t time : {0.3, 0.55, 0.9, 1.2, 1.6}) {
    EXPECT_NEAR(stanceDutyFactor(schedule, kContactLeftIndex, time), 0.5, 1.0e-9) << time;
    EXPECT_NEAR(stanceDutyFactor(schedule, kContactRightIndex, time), 0.5, 1.0e-9) << time;
  }
}

TEST(StanceDutyFactor, IsTheGaitsDutyFactorAtEveryNodeForBothFeet) {
  // The property the impulse budget rests on. It used to be the contact fraction over a window as long as the MPC
  // horizon, and a 1.0 s window over this 1.4 s stride gave 0.40 to 0.80 depending on the phase, with the two feet
  // disagreeing at the same node - which put 17% more vertical impulse than the weight into the reference.
  const ModeSchedule schedule = walk(6);
  const scalar_t expected = 0.8 / 1.4;  // 0.6 of swing in a 1.4 s stride
  for (scalar_t time = 1.4; time < 7.0; time += 0.037) {
    EXPECT_NEAR(stanceDutyFactor(schedule, kContactLeftIndex, time), expected, 1.0e-9) << "left at " << time;
    EXPECT_NEAR(stanceDutyFactor(schedule, kContactRightIndex, time), expected, 1.0e-9) << "right at " << time;
  }
}

TEST(StanceDutyFactor, StandingIsOneForBothFeet) {
  // A real standing schedule: both feet down throughout, with events so that the walk has something to measure.
  const ModeSchedule schedule(std::vector<scalar_t>{0.5, 1.0},
                              std::vector<size_t>{ModeNumber::kStance, ModeNumber::kStance, ModeNumber::kStance});
  // 1 is also the value at which the impulse-scaling correction is identically zero, so a robot that never lifts a
  // foot gets exactly the contact-force reference it had before the heuristic existed.
  EXPECT_NEAR(stanceDutyFactor(schedule, kContactLeftIndex, /*time=*/0.2), 1.0, 1.0e-12);
  EXPECT_NEAR(stanceDutyFactor(schedule, kContactRightIndex, /*time=*/0.7), 1.0, 1.0e-12);
}

TEST(StanceDutyFactor, StandingBeforeAWalkStartsIsOneAndTheFirstStepIsCarriedAtTheGaitsValue) {
  const ModeSchedule schedule = walk(4, /*leadIn=*/1.0);
  // Before any foot has lifted off, the robot is standing.
  EXPECT_NEAR(stanceDutyFactor(schedule, kContactLeftIndex, /*time=*/0.5), 1.0, 1.0e-12);
  EXPECT_NEAR(stanceDutyFactor(schedule, kContactRightIndex, /*time=*/0.5), 1.0, 1.0e-12);
  // During the first swing of the left foot the RIGHT foot is still in the stance it started in - it has no complete
  // stride yet - but it is carrying the robot through a step. Handing it the standing value would ask it for half the
  // weight in single support.
  EXPECT_NEAR(stanceDutyFactor(schedule, kContactRightIndex, /*time=*/1.3), 0.8 / 1.4, 1.0e-9);
}

TEST(StanceDutyFactor, OnceTheWholeScheduleHasEndedTheRobotIsStandingAgain) {
  const ModeSchedule schedule = walk(3);
  EXPECT_NEAR(stanceDutyFactor(schedule, kContactLeftIndex, schedule.eventTimes.back() + 0.5), 1.0, 1.0e-12);
  EXPECT_NEAR(stanceDutyFactor(schedule, kContactRightIndex, schedule.eventTimes.back() + 0.5), 1.0, 1.0e-12);
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
  EXPECT_NEAR(stanceDutyFactor(placeholder, kContactLeftIndex, /*time=*/0.0), 1.0, 1.0e-12);
  // One lift-off and nothing after it is no complete stride either.
  const ModeSchedule oneStep(std::vector<scalar_t>{0.5}, std::vector<size_t>{ModeNumber::kStance, ModeNumber::kRf});
  EXPECT_NEAR(stanceDutyFactor(oneStep, kContactRightIndex, /*time=*/0.7), 1.0, 1.0e-12);
}

TEST(StanceDutyFactor, IsAlwaysAFraction) {
  const ModeSchedule schedule = walk(4, 0.5);
  for (scalar_t time = -1.0; time < 8.0; time += 0.13) {
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      const scalar_t beta = stanceDutyFactor(schedule, foot, time);
      EXPECT_GT(beta, 0.0);
      EXPECT_LE(beta, 1.0);
    }
  }
}

TEST(UpcomingStanceDuration, IsTheStanceTheFootBeginsByLanding) {
  const ModeSchedule schedule = walk(4);
  // The left foot lands at 0.6 and lifts again at 1.4: 0.8 s of stance.
  EXPECT_NEAR(upcomingStanceDuration(schedule, kContactLeftIndex, /*touchDownTime=*/0.6), 0.8, 1.0e-9);
  // The right foot lands at 1.3 and lifts at 2.1.
  EXPECT_NEAR(upcomingStanceDuration(schedule, kContactRightIndex, /*touchDownTime=*/1.3), 0.8, 1.0e-9);
  // The last landing's stance runs past the schedule: the most recent complete stance stands in.
  const scalar_t lastLeftTouchDown = schedule.eventTimes[schedule.eventTimes.size() - 4];
  EXPECT_NEAR(upcomingStanceDuration(schedule, kContactLeftIndex, lastLeftTouchDown), 0.8, 1.0e-9);
  EXPECT_NEAR(upcomingStanceDuration(ModeSchedule(), kContactLeftIndex, /*touchDownTime=*/0.0), 0.0, 1.0e-12) << "no schedule, no stance";
}

TEST(PreviousTouchDownTime, IsTheLatestLandingAtOrBeforeTheTime) {
  const ModeSchedule schedule = walk(3);
  EXPECT_FALSE(previousTouchDownTime(schedule, kContactLeftIndex, /*time=*/0.3).has_value()) << "it has not landed yet";
  EXPECT_NEAR(valueOrFailure(previousTouchDownTime(schedule, kContactLeftIndex, /*time=*/0.6)), 0.6, 1.0e-12)
      << "an event at the time counts as passed";
  EXPECT_NEAR(valueOrFailure(previousTouchDownTime(schedule, kContactLeftIndex, /*time=*/1.9)), 0.6, 1.0e-12);
  EXPECT_NEAR(valueOrFailure(previousTouchDownTime(schedule, kContactRightIndex, /*time=*/2.0)), 1.3, 1.0e-12);
}

}  // namespace ocs2::humanoid
