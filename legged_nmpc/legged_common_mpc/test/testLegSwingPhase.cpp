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

// The schedule queries of one leg (gait/LegSwingPhase.h): the swing window,
// the swing phase and its rate of each leg of a four-contact trot, a leg of two
// contacts, degenerate schedules, and the upcoming stance duration, which on
// the biped's feet equals upcomingStanceDuration().

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "legged_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "legged_common_mpc/gait/BipedModes.h"
#include "legged_common_mpc/gait/LegSwingPhase.h"

namespace ocs2::legged {
namespace {

constexpr size_t kNumContacts = 4;
constexpr scalar_t kTolerance = 1.0e-12;

// Contacts in the quadruped order LF, RF, LH, RH (mode bits 8, 4, 2, 1), one
// contact per leg.
constexpr std::array<size_t, 1> kLeftFront = {0};
constexpr std::array<size_t, 1> kRightFront = {1};
constexpr std::array<size_t, 1> kLeftHind = {2};
constexpr std::array<size_t, 1> kRightHind = {3};

constexpr size_t kAllFour = 0b1111;
constexpr size_t kLeftFrontRightHind = 0b1001;
constexpr size_t kRightFrontLeftHind = 0b0110;

// The trot: all four until 0.2 s, LF and RH stand while RF and LH swing until
// 0.5 s, then RF and LH stand while LF and RH swing until 0.8 s, then the first
// diagonal again until 1.1 s, and all four after.
constexpr scalar_t kFirstLiftOffTime = 0.2;
constexpr scalar_t kDiagonalSwitchTime = 0.5;
constexpr scalar_t kSecondSwitchTime = 0.8;
constexpr scalar_t kLastTouchDownTime = 1.1;

ModeSchedule trot() {
  return ModeSchedule(
      /*eventTimesInput=*/{kFirstLiftOffTime, kDiagonalSwitchTime,
                           kSecondSwitchTime, kLastTouchDownTime},
      /*modeSequenceInput=*/{kAllFour, kLeftFrontRightHind, kRightFrontLeftHind,
                             kLeftFrontRightHind, kAllFour});
}

TEST(LegSwingPhase, eachLegOfATrotSweepsItsOwnSwing) {
  const ModeSchedule schedule = trot();
  const scalar_t swingDuration = kDiagonalSwitchTime - kFirstLiftOffTime;
  const scalar_t midFirstSwing =
      0.5 * (kFirstLiftOffTime + kDiagonalSwitchTime);
  const scalar_t midSecondSwing =
      0.5 * (kDiagonalSwitchTime + kSecondSwitchTime);
  // RF and LH swing first, LF and RH second.
  for (const std::array<size_t, 1>& leg : {kRightFront, kLeftHind}) {
    SCOPED_TRACE(leg[0]);
    EXPECT_EQ(
        legSwingPhaseVariable(schedule, kNumContacts, leg, kFirstLiftOffTime),
        0.0)
        << "the lift-off instant is still stance";
    EXPECT_NEAR(
        legSwingPhaseVariable(schedule, kNumContacts, leg, midFirstSwing), 0.5,
        kTolerance);
    EXPECT_NEAR(
        legSwingPhaseVariable(schedule, kNumContacts, leg, kDiagonalSwitchTime),
        1.0, kTolerance)
        << "the touch-down instant is the end of the swing";
    EXPECT_EQ(
        legSwingPhaseVariable(schedule, kNumContacts, leg, midSecondSwing), 0.0)
        << "in stance";
    EXPECT_NEAR(
        legSwingPhaseVariableRate(schedule, kNumContacts, leg, midFirstSwing),
        1.0 / swingDuration, kTolerance);
    EXPECT_NEAR(legSwingPhaseVariableRate(schedule, kNumContacts, leg,
                                          kFirstLiftOffTime + 0.01),
                1.0 / swingDuration, kTolerance)
        << "the rate is constant over the swing";
    EXPECT_EQ(
        legSwingPhaseVariableRate(schedule, kNumContacts, leg, midSecondSwing),
        0.0);
    const std::optional<LegSwingWindow> window =
        legSwingWindowAtTime(schedule, kNumContacts, leg, midFirstSwing);
    if (!window.has_value()) {
      GTEST_FAIL() << "no window";
    }
    EXPECT_EQ(window->liftOff, kFirstLiftOffTime);
    EXPECT_EQ(window->touchDown, kDiagonalSwitchTime);
  }
  for (const std::array<size_t, 1>& leg : {kLeftFront, kRightHind}) {
    SCOPED_TRACE(leg[0]);
    EXPECT_EQ(legSwingPhaseVariable(schedule, kNumContacts, leg, midFirstSwing),
              0.0);
    EXPECT_NEAR(
        legSwingPhaseVariable(schedule, kNumContacts, leg, midSecondSwing), 0.5,
        kTolerance);
    EXPECT_EQ(
        legSwingPhaseVariableRate(schedule, kNumContacts, leg, midFirstSwing),
        0.0);
  }
}

TEST(LegSwingPhase, aLegOfTwoContactsSwingsOnlyWhenBothAreUp) {
  // The left leg of a biped standing on its toe (contact 0) and heel
  // (contact 2): the heel lifts first, then the toe, then both land together.
  constexpr std::array<size_t, 2> kLeftLeg = {0, 2};
  const ModeSchedule schedule(
      /*eventTimesInput=*/{0.1, 0.3, 0.6},
      /*modeSequenceInput=*/{kAllFour, 0b1101, 0b0101, kAllFour});
  EXPECT_EQ(
      legSwingPhaseVariable(schedule, kNumContacts, kLeftLeg, /*time=*/0.2),
      0.0)
      << "the toe is still down";
  EXPECT_FALSE(
      legSwingWindowAtTime(schedule, kNumContacts, kLeftLeg, /*time=*/0.2)
          .has_value());
  const std::optional<LegSwingWindow> window =
      legSwingWindowAtTime(schedule, kNumContacts, kLeftLeg, /*time=*/0.45);
  if (!window.has_value()) {
    GTEST_FAIL() << "no window";
  }
  EXPECT_EQ(window->liftOff, 0.3);
  EXPECT_EQ(window->touchDown, 0.6);
  EXPECT_NEAR(
      legSwingPhaseVariable(schedule, kNumContacts, kLeftLeg, /*time=*/0.45),
      0.5, kTolerance);
}

TEST(LegSwingPhase, aSwingSpansThePhasesOfTheOtherLegs) {
  // LF swings from 0.1 s to 0.7 s while RF lifts off and lands in between.
  const ModeSchedule schedule(
      /*eventTimesInput=*/{0.1, 0.3, 0.5, 0.7},
      /*modeSequenceInput=*/{kAllFour, 0b0111, 0b0011, 0b0111, kAllFour});
  const std::optional<LegSwingWindow> window =
      legSwingWindowAtTime(schedule, kNumContacts, kLeftFront, /*time=*/0.4);
  if (!window.has_value()) {
    GTEST_FAIL() << "no window";
  }
  EXPECT_EQ(window->liftOff, 0.1);
  EXPECT_EQ(window->touchDown, 0.7);
  EXPECT_NEAR(
      legSwingPhaseVariable(schedule, kNumContacts, kLeftFront, /*time=*/0.4),
      0.5, kTolerance);
}

TEST(LegSwingPhase, aDegenerateScheduleHasNoPhase) {
  // No lift-off event: the schedule starts with the leg up.
  const ModeSchedule startsInSwing(/*eventTimesInput=*/{0.5},
                                   /*modeSequenceInput=*/{0b0111, kAllFour});
  EXPECT_EQ(legSwingPhaseVariable(startsInSwing, kNumContacts, kLeftFront,
                                  /*time=*/0.2),
            0.0);
  EXPECT_EQ(legSwingPhaseVariableRate(startsInSwing, kNumContacts, kLeftFront,
                                      /*time=*/0.2),
            0.0);
  // No touch-down event.
  const ModeSchedule endsInSwing(/*eventTimesInput=*/{0.5},
                                 /*modeSequenceInput=*/{kAllFour, 0b0111});
  EXPECT_EQ(legSwingPhaseVariable(endsInSwing, kNumContacts, kLeftFront,
                                  /*time=*/0.7),
            0.0);
  // The default schedule: one flight mode and no events.
  EXPECT_EQ(legSwingPhaseVariable(ModeSchedule(), kNumContacts, kLeftFront,
                                  /*time=*/0.0),
            0.0);
  // A swing of no duration.
  const ModeSchedule instant(
      /*eventTimesInput=*/{0.5, 0.5},
      /*modeSequenceInput=*/{kAllFour, 0b0111, kAllFour});
  EXPECT_EQ(
      legSwingPhaseVariable(instant, kNumContacts, kLeftFront, /*time=*/0.5),
      0.0);
  EXPECT_EQ(legSwingPhaseVariableRate(instant, kNumContacts, kLeftFront,
                                      /*time=*/0.5),
            0.0);
  // A leg of no contacts never swings.
  EXPECT_FALSE(legIsInSwingInMode(/*modeNumber=*/0, kNumContacts,
                                  /*legContactIndices=*/{}));
}

TEST(LegSwingPhase, theUpcomingStanceIsTheLegsOwn) {
  const ModeSchedule schedule = trot();
  // RF lands at 0.5 s and lifts off again at 0.8 s.
  EXPECT_NEAR(legUpcomingStanceDuration(schedule, kNumContacts, kRightFront,
                                        kDiagonalSwitchTime),
              kSecondSwitchTime - kDiagonalSwitchTime, kTolerance);
  // RF lands at 1.1 s and never lifts off again: its last complete stance,
  // [0.5, 0.8], stands in.
  EXPECT_NEAR(legUpcomingStanceDuration(schedule, kNumContacts, kRightFront,
                                        kLastTouchDownTime),
              kSecondSwitchTime - kDiagonalSwitchTime, kTolerance);
  // LF lands at 0.8 s and never lifts off again, with no complete stance
  // before.
  EXPECT_EQ(legUpcomingStanceDuration(schedule, kNumContacts, kLeftFront,
                                      kSecondSwitchTime),
            0.0);
}

TEST(LegSwingPhase, theBipedsFeetGetTheBipedsStanceAndTouchDowns) {
  constexpr size_t kBipedContacts = 2;
  constexpr std::array<size_t, 1> kLeftFoot = {0};
  constexpr std::array<size_t, 1> kRightFoot = {1};
  const ModeSchedule walk(
      /*eventTimesInput=*/{0.0, 0.4, 0.5, 0.9, 1.0, 1.4},
      /*modeSequenceInput=*/{kBipedModeStance, kBipedModeLf, kBipedModeStance,
                             kBipedModeRf, kBipedModeStance, kBipedModeLf,
                             kBipedModeStance});
  for (const scalar_t touchDownTime : {0.4, 0.9, 1.4, 2.0}) {
    SCOPED_TRACE(touchDownTime);
    EXPECT_EQ(legUpcomingStanceDuration(walk, kBipedContacts, kLeftFoot,
                                        touchDownTime),
              upcomingStanceDuration(walk, /*foot=*/0, touchDownTime));
    EXPECT_EQ(legUpcomingStanceDuration(walk, kBipedContacts, kRightFoot,
                                        touchDownTime),
              upcomingStanceDuration(walk, /*foot=*/1, touchDownTime));
    EXPECT_EQ(legPreviousTouchDownTime(walk, kBipedContacts, kLeftFoot,
                                       touchDownTime),
              previousTouchDownTime(walk, /*foot=*/0, touchDownTime));
    EXPECT_EQ(legPreviousTouchDownTime(walk, kBipedContacts, kRightFoot,
                                       touchDownTime),
              previousTouchDownTime(walk, /*foot=*/1, touchDownTime));
  }
}

}  // namespace
}  // namespace ocs2::legged
