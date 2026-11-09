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

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/gait/BipedModes.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/GaitScheduleUpdater.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

/*
 * updateGaitSchedule tiles a new gait into the schedule up to an absolute
 * time, 1.5 horizons ahead of the solve. It used to hand
 * GaitSchedule::insertModeSequenceTemplate the DURATION 1.5 * timeHorizon as
 * that final time, so a gait inserted later than 1.5 s into a run was not tiled
 * at all: the schedule stopped in stance where the new gait was to begin, until
 * the next getModeSchedule() tiled it from there.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kHorizon = 1.0;  // [s] the DRC Atlas horizon

// The shipped reference files: a stance schedule and a stance template of 0.5
// s; a walk of two 0.35 s swings.
ModeSchedule initialSchedule() {
  return ModeSchedule({0.5}, {kBipedModeStance, kBipedModeStance});
}
ModeSequenceTemplate stanceTemplate() {
  return ModeSequenceTemplate({0.0, 0.5}, {kBipedModeStance});
}
ModeSequenceTemplate walkTemplate() {
  return ModeSequenceTemplate({0.0, 0.35, 0.7}, {kBipedModeLf, kBipedModeRf});
}

bool hasSwing(const ModeSchedule& schedule, scalar_t from, scalar_t to) {
  for (scalar_t time = from; time <= to; time += 0.01) {
    if (schedule.modeAtTime(time) != kBipedModeStance) return true;
  }
  return false;
}

/** A stance schedule that the reference manager has queried at `initTime`, as
 * it does before every solve. */
std::shared_ptr<GaitSchedule> standingScheduleAt(
    scalar_t initTime, scalar_t phaseTransitionStanceTime) {
  std::shared_ptr<GaitSchedule> schedule = std::make_shared<GaitSchedule>(
      initialSchedule(), stanceTemplate(), phaseTransitionStanceTime,
      GaitModeLayout{});
  schedule->getModeSchedule(initTime - kHorizon, initTime + 2.0 * kHorizon);
  return schedule;
}

TEST(GaitScheduleUpdaterHorizon,
     aNewGaitIsTiledOverTheHorizonHoweverLateItIsInserted) {
  for (const scalar_t phaseTransitionStanceTime : {0.0, 0.2}) {
    for (const scalar_t initTime : {0.0, 2.0, 300.0}) {
      SCOPED_TRACE(absl::StrCat(
          "t = ", initTime,
          " s, phaseTransitionStanceTime = ", phaseTransitionStanceTime, " s"));
      std::shared_ptr<GaitSchedule> schedule =
          standingScheduleAt(initTime, phaseTransitionStanceTime);
      updateGaitSchedule(*schedule, walkTemplate(), initTime,
                         initTime + kHorizon);

      // What the schedule holds right after the insertion, before any other
      // query re-tiles it.
      const ModeSchedule inserted = schedule->getCurrentModeSchedule();
      ASSERT_FALSE(inserted.eventTimes.empty());
      EXPECT_GE(inserted.eventTimes.back(), initTime + 1.5 * kHorizon)
          << "the new gait must reach 1.5 horizons ahead";
      EXPECT_TRUE(hasSwing(inserted, initTime, initTime + 1.5 * kHorizon))
          << "the new gait must be in the schedule";
    }
  }
}

TEST(GaitScheduleUpdaterHorizon, theNextSolveSeesTheScheduleItSawBeforeTheFix) {
  // The next getModeSchedule() tiles from the last event, so the schedule a
  // solve reads was already right: the fix changes only what the schedule holds
  // between the insertion and that query. This pins that it changes nothing
  // else.
  for (const scalar_t phaseTransitionStanceTime : {0.0, 0.2}) {
    for (const scalar_t initTime : {0.0, 2.0, 300.0}) {
      SCOPED_TRACE(absl::StrCat(
          "t = ", initTime,
          " s, phaseTransitionStanceTime = ", phaseTransitionStanceTime, " s"));
      const scalar_t finalTime = initTime + kHorizon;
      std::shared_ptr<GaitSchedule> fixed =
          standingScheduleAt(initTime, phaseTransitionStanceTime);
      updateGaitSchedule(*fixed, walkTemplate(), initTime, finalTime);

      // The legacy insertion, with the start time the updater picks on a stance
      // schedule: the first event after 0.7 of the horizon.
      std::shared_ptr<GaitSchedule> legacy =
          standingScheduleAt(initTime, phaseTransitionStanceTime);
      const ModeSchedule before =
          legacy->getModeSchedule(initTime, finalTime + kHorizon);
      const std::vector<scalar_t>::const_iterator next =
          std::upper_bound(before.eventTimes.begin(), before.eventTimes.end(),
                           0.7 * finalTime + 0.3 * initTime);
      ASSERT_NE(next, before.eventTimes.end());
      legacy->insertModeSequenceTemplate(walkTemplate(), /*startTime=*/*next,
                                         /*finalTime=*/1.5 * kHorizon);
      if (initTime > 0.0) {
        // Positive control of the test above: the legacy insertion leaves the
        // horizon uncovered once the clock has run.
        const ModeSchedule legacyInserted = legacy->getCurrentModeSchedule();
        EXPECT_LT(legacyInserted.eventTimes.back(), initTime + 1.5 * kHorizon);
        EXPECT_FALSE(
            hasSwing(legacyInserted, initTime, initTime + 1.5 * kHorizon));
      }

      for (const scalar_t solveTime : {initTime + 0.01, initTime + 0.5}) {
        const ModeSchedule fixedSchedule = fixed->getModeSchedule(
            solveTime - kHorizon, solveTime + 2.0 * kHorizon);
        const ModeSchedule legacySchedule = legacy->getModeSchedule(
            solveTime - kHorizon, solveTime + 2.0 * kHorizon);
        EXPECT_EQ(fixedSchedule.eventTimes, legacySchedule.eventTimes)
            << "solve at " << solveTime;
        EXPECT_EQ(fixedSchedule.modeSequence, legacySchedule.modeSequence)
            << "solve at " << solveTime;
        EXPECT_TRUE(
            hasSwing(fixedSchedule, solveTime, solveTime + 2.0 * kHorizon))
            << "positive control: the gait changed";
      }
    }
  }
}

// The new gait starts at the first event after 0.7 of the horizon, 0.8 s here,
// and the phase before it is mode 2. On the biped set that is LF, the mode a
// walk starts with, so the switch moves back to where LF starts (0.5 s); on a
// set of four contacts 2 (0b0010) is the third contact alone, which is not the
// 0b1001 the trot inserted there starts with, and the switch stays.
constexpr scalar_t kPhaseTwoStart = 0.5;  // [s]
constexpr scalar_t kPhaseTwoEnd = 0.8;    // [s]
constexpr size_t kModeTwo = 0b0010;

std::shared_ptr<GaitSchedule> scheduleWithModeTwoBefore(
    const GaitModeLayout& layout) {
  std::shared_ptr<GaitSchedule> schedule = std::make_shared<GaitSchedule>(
      ModeSchedule({kPhaseTwoStart, kPhaseTwoEnd},
                   {layout.stanceMode, kModeTwo, layout.stanceMode}),
      ModeSequenceTemplate({0.0, 0.5}, {layout.stanceMode}),
      /*phaseTransitionStanceTime=*/0.0, layout);
  schedule->getModeSchedule(/*lowerBoundTime=*/-kHorizon,
                            /*upperBoundTime=*/2.0 * kHorizon);
  return schedule;
}

TEST(GaitScheduleUpdaterSwitchTime, aBipedSwitchMovesBackToTheStartOfLf) {
  static_assert(kModeTwo == kBipedModeLf);
  std::shared_ptr<GaitSchedule> schedule =
      scheduleWithModeTwoBefore(GaitModeLayout{});
  EXPECT_EQ(updateGaitSchedule(*schedule, walkTemplate(), /*initTime=*/0.0,
                               /*finalTime=*/kHorizon),
            kPhaseTwoStart);
}

TEST(GaitScheduleUpdaterSwitchTime, aFourContactSwitchIsNotMovedBack) {
  constexpr size_t kAllFour = 0b1111;
  std::shared_ptr<GaitSchedule> schedule = scheduleWithModeTwoBefore(
      GaitModeLayout{.numContacts = 4, .stanceMode = kAllFour});
  const ModeSequenceTemplate trot({0.0, 0.35, 0.7}, {0b1001, 0b0110});
  EXPECT_EQ(updateGaitSchedule(*schedule, trot, /*initTime=*/0.0,
                               /*finalTime=*/kHorizon),
            kPhaseTwoEnd);
}

// The gaits of the default gait switching table as the shipped biped gait file
// gives them.
struct NamedGait {
  std::string name;
  ModeSequenceTemplate gait;
};

std::vector<NamedGait> defaultTableGaits() {
  return {
      {"stance", ModeSequenceTemplate({0.0, 0.5}, {kBipedModeStance})},
      {"slow_walk", ModeSequenceTemplate({0.0, 0.65, 0.85, 1.5, 1.7},
                                         {kBipedModeLf, kBipedModeStance,
                                          kBipedModeRf, kBipedModeStance})},
      {"walk", ModeSequenceTemplate({0.0, 0.6, 0.7, 1.3, 1.4},
                                    {kBipedModeLf, kBipedModeStance,
                                     kBipedModeRf, kBipedModeStance})},
      {"slower_trot",
       ModeSequenceTemplate({0.0, 0.6, 1.2}, {kBipedModeLf, kBipedModeRf})},
      {"slow_trot",
       ModeSequenceTemplate({0.0, 0.55, 1.1}, {kBipedModeLf, kBipedModeRf})},
      {"trot",
       ModeSequenceTemplate({0.0, 0.5, 1.0}, {kBipedModeLf, kBipedModeRf})},
      {"run", ModeSequenceTemplate(
                  {0.0, 0.3, 0.4, 0.7, 0.8},
                  {kBipedModeLf, kBipedModeFly, kBipedModeRf, kBipedModeFly})},
  };
}

/** A biped stance schedule with one `mode` phase over [0.5, 0.8] s, queried
 * as before a solve. */
std::shared_ptr<GaitSchedule> bipedScheduleWithPhase(
    size_t mode, const GaitModeLayout& layout = GaitModeLayout{}) {
  std::shared_ptr<GaitSchedule> schedule = std::make_shared<GaitSchedule>(
      ModeSchedule({kPhaseTwoStart, kPhaseTwoEnd},
                   {kBipedModeStance, mode, kBipedModeStance}),
      stanceTemplate(), /*phaseTransitionStanceTime=*/0.0, layout);
  schedule->getModeSchedule(/*lowerBoundTime=*/-kHorizon,
                            /*upperBoundTime=*/2.0 * kHorizon);
  return schedule;
}

TEST(GaitScheduleUpdaterSwitchTime,
     EveryDefaultTableGaitStartsWhereItDidBefore) {
  // The rule before the generic cycle start: back to the start of an LF phase
  // that ends at the first event after 0.7 s, otherwise at that event. Each
  // gait the motion manager selects by default starts at the same literal time
  // and leaves the same schedule as an insertion at that time.
  struct Case {
    std::string schedule;
    std::shared_ptr<GaitSchedule> (*absl_nonnull make)();
    scalar_t expectedStart;  // [s]
  };
  const std::vector<Case> cases = {
      {"stance", [] { return standingScheduleAt(0.0, 0.0); }, 1.0},
      {"LF before", [] { return bipedScheduleWithPhase(kBipedModeLf); },
       kPhaseTwoStart},
      {"RF before", [] { return bipedScheduleWithPhase(kBipedModeRf); },
       kPhaseTwoEnd},
      {"FLY before", [] { return bipedScheduleWithPhase(kBipedModeFly); },
       kPhaseTwoEnd},
  };
  for (const NamedGait& named : defaultTableGaits()) {
    for (const Case& testCase : cases) {
      SCOPED_TRACE(absl::StrCat(named.name, " on ", testCase.schedule));
      std::shared_ptr<GaitSchedule> updated = testCase.make();
      EXPECT_EQ(updateGaitSchedule(*updated, named.gait, /*initTime=*/0.0,
                                   /*finalTime=*/kHorizon),
                testCase.expectedStart);
      std::shared_ptr<GaitSchedule> legacy = testCase.make();
      legacy->getModeSchedule(/*lowerBoundTime=*/0.0,
                              /*upperBoundTime=*/2.0 * kHorizon);
      legacy->insertModeSequenceTemplate(named.gait, testCase.expectedStart,
                                         /*finalTime=*/1.5 * kHorizon);
      const ModeSchedule updatedSchedule = updated->getCurrentModeSchedule();
      const ModeSchedule legacySchedule = legacy->getCurrentModeSchedule();
      EXPECT_EQ(updatedSchedule.eventTimes, legacySchedule.eventTimes);
      EXPECT_EQ(updatedSchedule.modeSequence, legacySchedule.modeSequence);
    }
  }
}

// The biped's two default legs, left (contact 0) and right (contact 1), as
// gaitModeLayoutOf() gives them.
GaitModeLayout bipedLeggedLayout() {
  return GaitModeLayout{.numContacts = 2,
                        .stanceMode = kBipedModeStance,
                        .legs = {{.name = "left", .modeBits = kBipedModeLf},
                                 {.name = "right", .modeBits = kBipedModeRf}},
                        .contactNames = {}};
}

// Four legs of one contact each, LF, RF, LH and RH.
GaitModeLayout quadrupedLeggedLayout() {
  return GaitModeLayout{.numContacts = 4,
                        .stanceMode = 0b1111,
                        .legs = {{.name = "LF", .modeBits = 0b1000},
                                 {.name = "RF", .modeBits = 0b0100},
                                 {.name = "LH", .modeBits = 0b0010},
                                 {.name = "RH", .modeBits = 0b0001}},
                        .contactNames = {}};
}

// The shipped biped gaits that do not open with LF: very_slow_walk, right_leg
// and jump.
std::vector<NamedGait> gaitsThatDoNotOpenWithLf() {
  return {
      {"very_slow_walk",
       ModeSequenceTemplate(
           {0.0, 0.7, 1.1, 1.8, 2.2},
           {kBipedModeRf, kBipedModeStance, kBipedModeLf, kBipedModeStance})},
      {"right_leg", ModeSequenceTemplate({0.0, 0.5}, {kBipedModeRf})},
      {"jump", ModeSequenceTemplate({0.0, 0.3, 0.5},
                                    {kBipedModeStance, kBipedModeFly})},
  };
}

TEST(GaitScheduleUpdaterSwitchTime,
     GaitsThatDoNotOpenWithLfStartWhereTheyDidBefore) {
  // The rule before the generic cycle start moved every biped switch back to
  // the start of an LF phase that ends at the first event after 0.7 s, whatever
  // the new gait opens with.
  struct Case {
    std::string schedule;
    size_t mode;
    scalar_t expectedStart;  // [s]
  };
  const std::vector<Case> cases = {
      {"LF before", kBipedModeLf, kPhaseTwoStart},
      {"RF before", kBipedModeRf, kPhaseTwoEnd},
      {"FLY before", kBipedModeFly, kPhaseTwoEnd},
  };
  for (const GaitModeLayout& layout : {GaitModeLayout{}, bipedLeggedLayout()}) {
    for (const NamedGait& named : gaitsThatDoNotOpenWithLf()) {
      for (const Case& testCase : cases) {
        SCOPED_TRACE(absl::StrCat(named.name, " on ", testCase.schedule, ", ",
                                  layout.legs.size(), " legs"));
        std::shared_ptr<GaitSchedule> updated =
            bipedScheduleWithPhase(testCase.mode, layout);
        EXPECT_EQ(updateGaitSchedule(*updated, named.gait, /*initTime=*/0.0,
                                     /*finalTime=*/kHorizon),
                  testCase.expectedStart);
      }
    }
  }
}

TEST(GaitScheduleUpdaterSwitchTime, TheCycleStartsWhereTheFirstLegStands) {
  for (const GaitModeLayout& biped : {GaitModeLayout{}, bipedLeggedLayout()}) {
    SCOPED_TRACE(absl::StrCat(biped.legs.size(), " legs"));
    EXPECT_EQ(cycleStartMode(walkTemplate(), biped), kBipedModeLf);
    EXPECT_EQ(
        cycleStartMode(ModeSequenceTemplate({0.0, 0.2, 0.6},
                                            {kBipedModeStance, kBipedModeRf}),
                       biped),
        kBipedModeLf)
        << "a gait without an LF phase falls back to LF";
    EXPECT_EQ(
        cycleStartMode(ModeSequenceTemplate({0.0, 0.3, 0.5},
                                            {kBipedModeStance, kBipedModeFly}),
                       biped),
        kBipedModeLf);
    EXPECT_EQ(
        cycleStartMode(ModeSequenceTemplate({0.0, 0.7, 1.1, 1.8, 2.2},
                                            {kBipedModeRf, kBipedModeStance,
                                             kBipedModeLf, kBipedModeStance}),
                       biped),
        kBipedModeLf)
        << "a gait that opens with RF still aligns on its LF phase";
    EXPECT_EQ(cycleStartMode(stanceTemplate(), biped), kBipedModeLf)
        << "a biped stance template keeps aligning to LF";
    EXPECT_EQ(defaultCycleStartMode(biped), kBipedModeLf);
  }

  constexpr size_t kAllFour = 0b1111;
  const GaitModeLayout quadruped = quadrupedLeggedLayout();
  EXPECT_EQ(
      cycleStartMode(ModeSequenceTemplate({0.0, 0.35, 0.7}, {0b1001, 0b0110}),
                     quadruped),
      0b1001U);
  EXPECT_EQ(
      cycleStartMode(ModeSequenceTemplate({0.0, 0.35, 0.7}, {0b0110, 0b1001}),
                     quadruped),
      0b1001U)
      << "the trot starts where LF stands, wherever the template opens";
  EXPECT_FALSE(
      cycleStartMode(ModeSequenceTemplate({0.0, 0.5}, {kAllFour}), quadruped)
          .has_value());
  EXPECT_FALSE(defaultCycleStartMode(quadruped).has_value());
  const GaitModeLayout leglessFour{.numContacts = 4, .stanceMode = kAllFour};
  EXPECT_FALSE(
      cycleStartMode(ModeSequenceTemplate({0.0, 0.35, 0.7}, {0b1001, 0b0110}),
                     leglessFour)
          .has_value())
      << "a set without legs has no first leg to align to";
}

// A quadruped stance schedule (contacts LF, RF, LH, RH) with one `mode` phase
// over [0.5, 0.8] s, queried as before a solve.
constexpr size_t kQuadrupedStance = 0b1111;
constexpr size_t kLfRh = 0b1001;
constexpr size_t kRfLh = 0b0110;

std::shared_ptr<GaitSchedule> quadrupedScheduleWithPhase(size_t mode) {
  const GaitModeLayout layout = quadrupedLeggedLayout();
  std::shared_ptr<GaitSchedule> schedule = std::make_shared<GaitSchedule>(
      ModeSchedule({kPhaseTwoStart, kPhaseTwoEnd},
                   {kQuadrupedStance, mode, kQuadrupedStance}),
      ModeSequenceTemplate({0.0, 0.5}, {kQuadrupedStance}),
      /*phaseTransitionStanceTime=*/0.0, layout);
  schedule->getModeSchedule(/*lowerBoundTime=*/-kHorizon,
                            /*upperBoundTime=*/2.0 * kHorizon);
  return schedule;
}

TEST(GaitScheduleUpdaterSwitchTime, AQuadrupedTrotAlignsToItsFirstDiagonal) {
  const ModeSequenceTemplate trot({0.0, 0.35, 0.7}, {kLfRh, kRfLh});
  std::shared_ptr<GaitSchedule> lfRhBefore = quadrupedScheduleWithPhase(kLfRh);
  EXPECT_EQ(updateGaitSchedule(*lfRhBefore, trot, /*initTime=*/0.0,
                               /*finalTime=*/kHorizon),
            kPhaseTwoStart)
      << "an LF_RH phase that ends past the switching time is redone";
  std::shared_ptr<GaitSchedule> rfLhBefore = quadrupedScheduleWithPhase(kRfLh);
  EXPECT_EQ(updateGaitSchedule(*rfLhBefore, trot, /*initTime=*/0.0,
                               /*finalTime=*/kHorizon),
            kPhaseTwoEnd)
      << "after RF_LH the trot goes on with LF_RH";
}

TEST(GaitScheduleUpdaterSwitchTime, AQuadrupedStanceTemplateIsNotMoved) {
  std::shared_ptr<GaitSchedule> schedule = quadrupedScheduleWithPhase(kLfRh);
  EXPECT_EQ(updateGaitSchedule(
                *schedule, ModeSequenceTemplate({0.0, 0.5}, {kQuadrupedStance}),
                /*initTime=*/0.0, /*finalTime=*/kHorizon),
            kPhaseTwoEnd);
}

}  // namespace
}  // namespace ocs2::humanoid
