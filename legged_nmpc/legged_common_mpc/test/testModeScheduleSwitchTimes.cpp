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

// plannedContactSwitchTimes(): each contact's time since its last and until
// its next planned switch in a mode schedule, on a biped walk and on a
// four-contact trot, without a heap allocation.

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "legged_common_mpc/gait/BipedModes.h"
#include "legged_common_mpc/gait/ModeScheduleSwitchTimes.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

namespace ocs2::legged {
namespace {

constexpr scalar_t kUnknown = std::numeric_limits<scalar_t>::infinity();
constexpr scalar_t kTolerance = 1.0e-12;

/** The left foot stands from 0.5 s to 0.8 s while the right foot swings, both
 * stand until 0.9 s, then the right foot stands until 1.2 s. */
ModeSchedule bipedWalk() {
  return ModeSchedule(
      /*eventTimesInput=*/{0.5, 0.8, 0.9, 1.2},
      /*modeSequenceInput=*/{kBipedModeStance, kBipedModeLf, kBipedModeStance,
                             kBipedModeRf, kBipedModeStance});
}

/** The switch times of `numContacts` contacts at `time`. */
struct SwitchTimes {
  std::vector<scalar_t> since;
  std::vector<scalar_t> to;
};

SwitchTimes switchTimesAt(const ModeSchedule& schedule,
                          scalar_t time,
                          size_t numContacts) {
  SwitchTimes times{.since = std::vector<scalar_t>(numContacts, -1.0),
                    .to = std::vector<scalar_t>(numContacts, -1.0)};
  plannedContactSwitchTimes(schedule, time, numContacts,
                            absl::MakeSpan(times.since),
                            absl::MakeSpan(times.to));
  return times;
}

TEST(ModeScheduleSwitchTimesTest, ABipedWalk) {
  constexpr size_t kNumFeet = 2;
  constexpr size_t kLeft = 0;
  constexpr size_t kRight = 1;
  // In the left stance: the left foot stood from before the schedule and
  // lifts off at 0.9 s; the right foot lifted off at 0.5 s and lands at
  // 0.8 s.
  const SwitchTimes leftStance =
      switchTimesAt(bipedWalk(), /*time=*/0.6, kNumFeet);
  EXPECT_EQ(leftStance.since[kLeft], kUnknown);
  EXPECT_NEAR(leftStance.to[kLeft], 0.3, kTolerance);
  EXPECT_NEAR(leftStance.since[kRight], 0.1, kTolerance);
  EXPECT_NEAR(leftStance.to[kRight], 0.2, kTolerance);

  // At an event time the event is still ahead, as modeAtTime() reads it.
  const SwitchTimes atLiftoff =
      switchTimesAt(bipedWalk(), /*time=*/0.5, kNumFeet);
  EXPECT_EQ(atLiftoff.to[kRight], 0.0);
  EXPECT_NEAR(atLiftoff.to[kLeft], 0.4, kTolerance);
  EXPECT_EQ(atLiftoff.since[kLeft], kUnknown);
  EXPECT_EQ(atLiftoff.since[kRight], kUnknown);

  // After the last event no switch is ahead.
  const SwitchTimes after = switchTimesAt(bipedWalk(), /*time=*/2.0, kNumFeet);
  EXPECT_EQ(after.to[kLeft], kUnknown);
  EXPECT_EQ(after.to[kRight], kUnknown);
  EXPECT_NEAR(after.since[kLeft], 0.8, kTolerance);   // touchdown at 1.2 s
  EXPECT_NEAR(after.since[kRight], 1.2, kTolerance);  // touchdown at 0.8 s
}

TEST(ModeScheduleSwitchTimesTest, AFourContactTrot) {
  // Contacts LF, RF, LH, RH, the first the most significant bit: LF_RH is
  // 1001, RF_LH 0110, the stance 1111.
  constexpr size_t kNumContacts = 4;
  constexpr size_t kStance = 0b1111;
  constexpr size_t kLfRh = 0b1001;
  constexpr size_t kRfLh = 0b0110;
  const ModeSchedule trot(/*eventTimesInput=*/{0.0, 0.3, 0.6},
                          /*modeSequenceInput=*/{kStance, kLfRh, kRfLh, kLfRh});
  // In the RF_LH phase every contact switched at 0.3 s and switches at 0.6 s.
  const SwitchTimes mid = switchTimesAt(trot, /*time=*/0.45, kNumContacts);
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    EXPECT_NEAR(mid.since[contact], 0.15, kTolerance) << contact;
    EXPECT_NEAR(mid.to[contact], 0.15, kTolerance) << contact;
  }
  // In the first LF_RH phase LF and RH stood from the stance on; RF and LH
  // lifted off at 0 s.
  const SwitchTimes first = switchTimesAt(trot, /*time=*/0.1, kNumContacts);
  const std::array<scalar_t, kNumContacts> expectedSince = {kUnknown, 0.1, 0.1,
                                                            kUnknown};
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    if (expectedSince[contact] == kUnknown) {
      EXPECT_EQ(first.since[contact], kUnknown) << contact;
    } else {
      EXPECT_NEAR(first.since[contact], expectedSince[contact], kTolerance)
          << contact;
    }
    EXPECT_NEAR(first.to[contact], 0.2, kTolerance) << contact;
  }
  // In the last phase nothing switches ahead.
  const SwitchTimes last = switchTimesAt(trot, /*time=*/1.0, kNumContacts);
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    EXPECT_EQ(last.to[contact], kUnknown) << contact;
    EXPECT_NEAR(last.since[contact], 0.4, kTolerance) << contact;
  }
}

TEST(ModeScheduleSwitchTimesTest, AScheduleWithoutEventsShowsNoSwitch) {
  const SwitchTimes none =
      switchTimesAt(ModeSchedule(/*eventTimesInput=*/{},
                                 /*modeSequenceInput=*/{kBipedModeStance}),
                    /*time=*/0.3, /*numContacts=*/2);
  for (size_t contact = 0; contact < 2; ++contact) {
    EXPECT_EQ(none.since[contact], kUnknown);
    EXPECT_EQ(none.to[contact], kUnknown);
  }
}

TEST(ModeScheduleSwitchTimesTest,
     WritesOnlyWhatTheSpansHoldAndAllocatesNothing) {
  const ModeSchedule schedule = bipedWalk();
  std::vector<scalar_t> since(1, -1.0);
  std::vector<scalar_t> to(2, -1.0);
  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  for (int step = 0; step < 100; ++step) {
    plannedContactSwitchTimes(schedule, 0.02 * step, /*numContacts=*/2,
                              absl::MakeSpan(since), absl::MakeSpan(to));
  }
  EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0U);
  EXPECT_EQ(to[1], -1.0) << "past the shorter span: not written";
  EXPECT_NE(to[0], -1.0);
}

}  // namespace
}  // namespace ocs2::legged
