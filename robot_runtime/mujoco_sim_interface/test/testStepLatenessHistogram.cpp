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

#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "mujoco_sim_interface/StepLatenessHistogram.h"

/*
 * The physics loop's lateness histogram: its percentiles to the bin, its exact
 * maximum, its count of late steps, and what it does with early, far-too-late
 * and forgotten samples.
 */

namespace robot::mujoco_sim_interface {
namespace {

TEST(StepLatenessHistogram, IsEmptyAtFirst) {
  const StepLatenessHistogram histogram;
  EXPECT_EQ(histogram.count(), 0);
  EXPECT_EQ(histogram.percentile(/*fraction=*/0.99), absl::ZeroDuration());
  EXPECT_EQ(histogram.max(), absl::ZeroDuration());
  EXPECT_EQ(histogram.countLaterThan(absl::ZeroDuration()), 0);
}

TEST(StepLatenessHistogram, ReportsPercentilesToTheBinAndTheMaximumExactly) {
  StepLatenessHistogram histogram;
  // 98 steps in time, one 12 us late and one 3.2 ms late.
  for (int step = 0; step < 98; ++step)
    histogram.record(std::chrono::microseconds(-40));
  histogram.record(std::chrono::microseconds(12));
  histogram.record(std::chrono::microseconds(3200));
  EXPECT_EQ(histogram.count(), 100);
  EXPECT_EQ(histogram.percentile(/*fraction=*/0.5), absl::Microseconds(5))
      << "an early step is on time: the first bin, 0 to 5 us";
  EXPECT_EQ(histogram.percentile(/*fraction=*/0.99), absl::Microseconds(15))
      << "the upper edge of the 10-15 us bin";
  EXPECT_EQ(histogram.percentile(/*fraction=*/1.0), absl::Microseconds(3200));
  EXPECT_EQ(histogram.max(), absl::Microseconds(3200));
  EXPECT_EQ(histogram.countLaterThan(absl::Milliseconds(1)), 1);
  EXPECT_EQ(histogram.countLaterThan(absl::Microseconds(10)), 2);
}

TEST(StepLatenessHistogram, KeepsTheMaximumOfASamplePastTheLastBin) {
  StepLatenessHistogram histogram;
  histogram.record(std::chrono::seconds(1));
  EXPECT_EQ(histogram.max(), absl::Seconds(1));
  EXPECT_EQ(histogram.percentile(/*fraction=*/0.5), absl::Seconds(1))
      << "the last bin's upper edge is the maximum";
  EXPECT_EQ(histogram.countLaterThan(absl::Milliseconds(19)), 1);
  EXPECT_EQ(histogram.countLaterThan(absl::Seconds(5)), 1)
      << "every sample past the last bin shares it";
}

TEST(StepLatenessHistogram, ResetForgetsEverySample) {
  StepLatenessHistogram histogram;
  histogram.record(std::chrono::milliseconds(2));
  histogram.reset();
  EXPECT_EQ(histogram.count(), 0);
  EXPECT_EQ(histogram.max(), absl::ZeroDuration());
  EXPECT_EQ(histogram.countLaterThan(absl::ZeroDuration()), 0);
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
