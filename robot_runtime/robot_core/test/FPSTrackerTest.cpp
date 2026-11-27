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

#include <chrono>

#include "gtest/gtest.h"

#include "robot_core/FPSTracker.h"

namespace robot {
namespace {

using std::chrono::milliseconds;
using std::chrono::steady_clock;

TEST(FPSTracker, TheRateIsOneOverThePeriodFromTheSecondTickOn) {
  FPSTracker tracker(/*alpha=*/1.0);
  const steady_clock::time_point start = steady_clock::now();
  tracker.tickAt(start);
  EXPECT_EQ(tracker.fps(), 0.0) << "one tick has no period";
  tracker.tickAt(start + milliseconds(10));
  EXPECT_NEAR(tracker.fps(), 100.0, 1.0e-9);
  tracker.tickAt(start + milliseconds(30));
  EXPECT_NEAR(tracker.fps(), 50.0, 1.0e-9)
      << "alpha 1 keeps only the newest period";
}

TEST(FPSTracker, TheRateIsSmoothedExponentially) {
  const double alpha = 0.25;
  FPSTracker tracker(alpha);
  const steady_clock::time_point start = steady_clock::now();
  tracker.tickAt(start);
  tracker.tickAt(start + milliseconds(10));  // 100 Hz, the first rate as it is
  tracker.tickAt(start + milliseconds(30));  // 50 Hz
  EXPECT_NEAR(tracker.fps(), alpha * 50.0 + (1.0 - alpha) * 100.0, 1.0e-9);
}

TEST(FPSTracker, AResetForgetsThePeriodAcrossIt) {
  // The simulator resets the tracker after a one-second pause; the pause must
  // not become a 1 Hz sample.
  FPSTracker tracker(/*alpha=*/0.5);
  const steady_clock::time_point start = steady_clock::now();
  tracker.tickAt(start);
  tracker.tickAt(start + milliseconds(1));
  tracker.reset();
  EXPECT_EQ(tracker.fps(), 0.0);
  const steady_clock::time_point resumed = start + milliseconds(1001);
  tracker.tickAt(resumed);
  EXPECT_EQ(tracker.fps(), 0.0)
      << "the first tick after a reset starts the clock";
  tracker.tickAt(resumed + milliseconds(2));
  EXPECT_NEAR(tracker.fps(), 500.0, 1.0e-9)
      << "the rate after the reset, not smoothed against the one before it";
}

TEST(FPSTracker, ATickWithinOneClockTickIsSkippedRatherThanAnInfiniteRate) {
  FPSTracker tracker(/*alpha=*/0.5);
  const steady_clock::time_point start = steady_clock::now();
  tracker.tickAt(start);
  tracker.tickAt(start + milliseconds(4));
  tracker.tickAt(start + milliseconds(4));
  EXPECT_NEAR(tracker.fps(), 250.0, 1.0e-9);
  tracker.tickAt(start + milliseconds(8));
  EXPECT_NEAR(tracker.fps(), 250.0, 1.0e-9)
      << "the period is measured from the last tick that counted";
}

}  // namespace
}  // namespace robot
