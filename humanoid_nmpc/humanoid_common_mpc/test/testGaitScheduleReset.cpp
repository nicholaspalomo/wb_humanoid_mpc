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
#include <vector>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/GaitScheduleUpdater.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

/*
 * The gait schedule after a reset of the controller. It used to keep every gait inserted before the reset, events in
 * the future included: after a clock that ran backwards, the robot stood still until the clock caught up with them.
 */

namespace ocs2::humanoid {
namespace {

// The shipped reference files: a stance schedule and a stance template of 0.5 s.
ModeSchedule initialSchedule() {
  return ModeSchedule({0.5}, {ModeNumber::kStance, ModeNumber::kStance});
}
ModeSequenceTemplate stanceTemplate() {
  return ModeSequenceTemplate({0.0, 0.5}, {ModeNumber::kStance});
}
ModeSequenceTemplate trotTemplate() {
  return ModeSequenceTemplate({0.0, 0.35, 0.7}, {ModeNumber::kLf, ModeNumber::kRf});
}

bool hasSwing(const ModeSchedule& schedule, scalar_t from, scalar_t to) {
  for (scalar_t time = from; time <= to; time += 0.01) {
    if (schedule.modeAtTime(time) != ModeNumber::kStance) return true;
  }
  return false;
}

TEST(GaitScheduleReset, AResetScheduleAnswersExactlyAsAFreshOne) {
  GaitSchedule used(initialSchedule(), stanceTemplate(), /*phaseTransitionStanceTime=*/0.0);
  GaitSchedule fresh(initialSchedule(), stanceTemplate(), /*phaseTransitionStanceTime=*/0.0);

  // Trotting from t = 5 s: the schedule carries swings up to the end of its horizon.
  used.getModeSchedule(4.0, 7.0);
  used.insertModeSequenceTemplate(trotTemplate(), /*startTime=*/5.0, /*finalTime=*/12.0);
  ASSERT_TRUE(hasSwing(used.getModeSchedule(6.0, 8.0), /*from=*/6.0, /*to=*/8.0)) << "positive control: the used schedule trots";

  used.reset();
  // Queried at the time of the reset, and after the clock ran back to 2 s.
  for (const scalar_t time : {6.0, 2.0}) {
    GaitSchedule reference(initialSchedule(), stanceTemplate(), /*phaseTransitionStanceTime=*/0.0);
    GaitSchedule reset(initialSchedule(), stanceTemplate(), /*phaseTransitionStanceTime=*/0.0);
    reset.insertModeSequenceTemplate(trotTemplate(), /*startTime=*/5.0, /*finalTime=*/12.0);
    reset.reset();
    const ModeSchedule afterReset = reset.getModeSchedule(time - 1.0, time + 2.0);
    const ModeSchedule afterConstruction = reference.getModeSchedule(time - 1.0, time + 2.0);
    EXPECT_EQ(afterReset.eventTimes, afterConstruction.eventTimes) << "at t = " << time;
    EXPECT_EQ(afterReset.modeSequence, afterConstruction.modeSequence) << "at t = " << time;
    EXPECT_FALSE(hasSwing(afterReset, time - 1.0, time + 2.0)) << "at t = " << time;
  }
  EXPECT_EQ(used.getModeSchedule(6.0, 8.0).modeSequence, fresh.getModeSchedule(6.0, 8.0).modeSequence);
}

TEST(UpdateGaitSchedule, AFirstEventThatClosesALeftSwingDoesNotReadBeforeTheSchedule) {
  // The first phase of the schedule is a left-foot swing that ends after the switching time: the phase before it, where
  // the new gait would start, does not exist. The updater used to read the event before the first one.
  GaitSchedule schedule(ModeSchedule({5.0, 5.3}, {ModeNumber::kLf, ModeNumber::kStance, ModeNumber::kStance}), stanceTemplate(),
                        /*phaseTransitionStanceTime=*/0.0);
  const scalar_t initTime = 1.0;
  const scalar_t finalTime = 2.0;
  const ModeSchedule before = schedule.getModeSchedule(initTime, finalTime + (finalTime - initTime));
  ASSERT_EQ(before.eventTimes.front(), 5.0);
  ASSERT_EQ(before.modeAtTime(before.eventTimes.front()), ModeNumber::kLf) << "the case under test: the first phase is a left swing";

  updateGaitSchedule(schedule, trotTemplate(), initTime, finalTime);

  // The new gait starts where that swing ends, the first event of the schedule: the trot is tiled from there.
  const ModeSchedule after = schedule.getModeSchedule(initTime, /*upperBoundTime=*/8.0);
  ASSERT_FALSE(after.eventTimes.empty());
  EXPECT_DOUBLE_EQ(after.eventTimes.front(), 5.0);
  EXPECT_EQ(after.modeAtTime(5.0 + 0.1), ModeNumber::kLf) << "the trot begins after the first event";
  EXPECT_EQ(after.modeAtTime(5.0 + 0.35 + 0.1), ModeNumber::kRf);
  for (const scalar_t eventTime : after.eventTimes) EXPECT_TRUE(std::isfinite(eventTime));
}

}  // namespace
}  // namespace ocs2::humanoid
