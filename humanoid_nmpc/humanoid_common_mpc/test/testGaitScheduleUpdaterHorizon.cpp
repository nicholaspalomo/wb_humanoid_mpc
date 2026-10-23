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
#include <memory>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/GaitScheduleUpdater.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

/*
 * updateGaitSchedule tiles a new gait into the schedule up to an absolute time, 1.5 horizons ahead
 * of the solve. It used to hand GaitSchedule::insertModeSequenceTemplate the DURATION 1.5 * timeHorizon as that final
 * time, so a gait inserted later than 1.5 s into a run was not tiled at all: the schedule stopped in stance where the new
 * gait was to begin, until the next getModeSchedule() tiled it from there.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kHorizon = 1.0;  // [s] the DRC Atlas horizon

// The shipped reference files: a stance schedule and a stance template of 0.5 s; a walk of two 0.35 s swings.
ModeSchedule initialSchedule() {
  return ModeSchedule({0.5}, {ModeNumber::kStance, ModeNumber::kStance});
}
ModeSequenceTemplate stanceTemplate() {
  return ModeSequenceTemplate({0.0, 0.5}, {ModeNumber::kStance});
}
ModeSequenceTemplate walkTemplate() {
  return ModeSequenceTemplate({0.0, 0.35, 0.7}, {ModeNumber::kLf, ModeNumber::kRf});
}

bool hasSwing(const ModeSchedule& schedule, scalar_t from, scalar_t to) {
  for (scalar_t time = from; time <= to; time += 0.01) {
    if (schedule.modeAtTime(time) != ModeNumber::kStance) return true;
  }
  return false;
}

/** A stance schedule that the reference manager has queried at `initTime`, as it does before every solve. */
std::shared_ptr<GaitSchedule> standingScheduleAt(scalar_t initTime, scalar_t phaseTransitionStanceTime) {
  std::shared_ptr<GaitSchedule> schedule = std::make_shared<GaitSchedule>(initialSchedule(), stanceTemplate(), phaseTransitionStanceTime);
  schedule->getModeSchedule(initTime - kHorizon, initTime + 2.0 * kHorizon);
  return schedule;
}

TEST(GaitScheduleUpdaterHorizon, aNewGaitIsTiledOverTheHorizonHoweverLateItIsInserted) {
  for (const scalar_t phaseTransitionStanceTime : {0.0, 0.2}) {
    for (const scalar_t initTime : {0.0, 2.0, 300.0}) {
      SCOPED_TRACE(absl::StrCat("t = ", initTime, " s, phaseTransitionStanceTime = ", phaseTransitionStanceTime, " s"));
      std::shared_ptr<GaitSchedule> schedule = standingScheduleAt(initTime, phaseTransitionStanceTime);
      updateGaitSchedule(*schedule, walkTemplate(), initTime, initTime + kHorizon);

      // What the schedule holds right after the insertion, before any other query re-tiles it.
      const ModeSchedule inserted = schedule->getCurrentModeSchedule();
      ASSERT_FALSE(inserted.eventTimes.empty());
      EXPECT_GE(inserted.eventTimes.back(), initTime + 1.5 * kHorizon) << "the new gait must reach 1.5 horizons ahead";
      EXPECT_TRUE(hasSwing(inserted, initTime, initTime + 1.5 * kHorizon)) << "the new gait must be in the schedule";
    }
  }
}

TEST(GaitScheduleUpdaterHorizon, theNextSolveSeesTheScheduleItSawBeforeTheFix) {
  // The next getModeSchedule() tiles from the last event, so the schedule a solve reads was already right: the fix
  // changes only what the schedule holds between the insertion and that query. This pins that it changes nothing else.
  for (const scalar_t phaseTransitionStanceTime : {0.0, 0.2}) {
    for (const scalar_t initTime : {0.0, 2.0, 300.0}) {
      SCOPED_TRACE(absl::StrCat("t = ", initTime, " s, phaseTransitionStanceTime = ", phaseTransitionStanceTime, " s"));
      const scalar_t finalTime = initTime + kHorizon;
      std::shared_ptr<GaitSchedule> fixed = standingScheduleAt(initTime, phaseTransitionStanceTime);
      updateGaitSchedule(*fixed, walkTemplate(), initTime, finalTime);

      // The legacy insertion, with the start time the updater picks on a stance schedule: the first event after
      // 0.7 of the horizon.
      std::shared_ptr<GaitSchedule> legacy = standingScheduleAt(initTime, phaseTransitionStanceTime);
      const ModeSchedule before = legacy->getModeSchedule(initTime, finalTime + kHorizon);
      const std::vector<scalar_t>::const_iterator next =
          std::upper_bound(before.eventTimes.begin(), before.eventTimes.end(), 0.7 * finalTime + 0.3 * initTime);
      ASSERT_NE(next, before.eventTimes.end());
      legacy->insertModeSequenceTemplate(walkTemplate(), /*startTime=*/*next, /*finalTime=*/1.5 * kHorizon);
      if (initTime > 0.0) {
        // Positive control of the test above: the legacy insertion leaves the horizon uncovered once the clock has run.
        const ModeSchedule legacyInserted = legacy->getCurrentModeSchedule();
        EXPECT_LT(legacyInserted.eventTimes.back(), initTime + 1.5 * kHorizon);
        EXPECT_FALSE(hasSwing(legacyInserted, initTime, initTime + 1.5 * kHorizon));
      }

      for (const scalar_t solveTime : {initTime + 0.01, initTime + 0.5}) {
        const ModeSchedule fixedSchedule = fixed->getModeSchedule(solveTime - kHorizon, solveTime + 2.0 * kHorizon);
        const ModeSchedule legacySchedule = legacy->getModeSchedule(solveTime - kHorizon, solveTime + 2.0 * kHorizon);
        EXPECT_EQ(fixedSchedule.eventTimes, legacySchedule.eventTimes) << "solve at " << solveTime;
        EXPECT_EQ(fixedSchedule.modeSequence, legacySchedule.modeSequence) << "solve at " << solveTime;
        EXPECT_TRUE(hasSwing(fixedSchedule, solveTime, solveTime + 2.0 * kHorizon)) << "positive control: the gait changed";
      }
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid
