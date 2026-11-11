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

#include <cstddef>

#include "gtest/gtest.h"

#include "mujoco_sim_interface/BaseAccelerometer.h"
#include "robot_core/Types.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/*
 * The simulated accelerometer's mean acceleration of the base over its window,
 * from the velocities the simulator records after each step.
 */

namespace robot::mujoco_sim_interface {
namespace {

constexpr double kStep = 0.0005;
constexpr double kTolerance = 1.0e-9;

/** The velocity of a base whose acceleration is 3 t^2 along x, -2 along y and
 * 1 along z: (t^3, -2 t, t). */
vector3_t linvelAt(double time) {
  return vector3_t(time * time * time, -2.0 * time, time);
}

TEST(BaseAccelerometer, TheWindowIsRoundedToWholeStepsOfAtLeastOne) {
  EXPECT_EQ(BaseAccelerometer::windowStepsFor(/*window=*/0.0, kStep), 1);
  EXPECT_EQ(BaseAccelerometer::windowStepsFor(/*window=*/0.0001, kStep), 1);
  EXPECT_EQ(BaseAccelerometer::windowStepsFor(/*window=*/0.01, kStep), 20);
  EXPECT_EQ(BaseAccelerometer::windowStepsFor(/*window=*/0.0124, kStep), 25);
  EXPECT_EQ(BaseAccelerometer(/*windowSteps=*/0).windowSteps(), 1);
  EXPECT_EQ(BaseAccelerometer(/*windowSteps=*/20).windowSteps(), 20);
}

TEST(BaseAccelerometer, ReadsZeroUntilItHasTwoSamples) {
  BaseAccelerometer accelerometer(/*windowSteps=*/4);
  EXPECT_EQ(accelerometer.linaccBaseRtWorldEwrtWorld(), vector3_t::Zero());
  accelerometer.restart(/*time=*/1.0, linvelAt(1.0));
  EXPECT_EQ(accelerometer.linaccBaseRtWorldEwrtWorld(), vector3_t::Zero());
}

TEST(BaseAccelerometer, OneStepReadsThatStepsChangeOfVelocity) {
  BaseAccelerometer accelerometer(/*windowSteps=*/1);
  accelerometer.restart(/*time=*/0.0, linvelAt(0.0));
  for (int step = 1; step <= 10; ++step) {
    const double time = step * kStep;
    accelerometer.record(time, linvelAt(time));
    const vector3_t expected =
        (linvelAt(time) - linvelAt(time - kStep)) / kStep;
    EXPECT_LT((accelerometer.linaccBaseRtWorldEwrtWorld() - expected).norm(),
              kTolerance)
        << "step " << step;
  }
}

TEST(BaseAccelerometer, AWindowReadsTheMeanOverItsStepsAndFewerAtTheStart) {
  constexpr int kWindowSteps = 4;
  BaseAccelerometer accelerometer(kWindowSteps);
  const double start = 2.0;
  accelerometer.restart(start, linvelAt(start));
  for (int step = 1; step <= 12; ++step) {
    const double time = start + step * kStep;
    accelerometer.record(time, linvelAt(time));
    // Fewer steps than the window since the restart: the mean since then.
    const int steps = step < kWindowSteps ? step : kWindowSteps;
    const double then = time - steps * kStep;
    const vector3_t expected =
        (linvelAt(time) - linvelAt(then)) / (time - then);
    EXPECT_LT((accelerometer.linaccBaseRtWorldEwrtWorld() - expected).norm(),
              kTolerance)
        << "step " << step;
  }
}

TEST(BaseAccelerometer, ARestartForgetsTheSamplesBeforeIt) {
  BaseAccelerometer accelerometer(/*windowSteps=*/4);
  accelerometer.restart(/*time=*/0.0, vector3_t::Zero());
  accelerometer.record(/*time=*/kStep, vector3_t(1.0, 0.0, 0.0));
  accelerometer.record(/*time=*/2.0 * kStep, vector3_t(2.0, 0.0, 0.0));
  // A teleport to another velocity, at the same time.
  accelerometer.restart(/*time=*/2.0 * kStep, vector3_t(0.0, 5.0, 0.0));
  EXPECT_EQ(accelerometer.linaccBaseRtWorldEwrtWorld(), vector3_t::Zero());
  accelerometer.record(/*time=*/3.0 * kStep, vector3_t(0.0, 6.0, 0.0));
  EXPECT_LT((accelerometer.linaccBaseRtWorldEwrtWorld() -
             vector3_t(0.0, 1.0 / kStep, 0.0))
                .norm(),
            kTolerance);
}

TEST(BaseAccelerometer, AStatePublishedAgainWithoutAStepReplacesTheNewest) {
  BaseAccelerometer accelerometer(/*windowSteps=*/2);
  accelerometer.restart(/*time=*/0.0, vector3_t::Zero());
  accelerometer.record(/*time=*/kStep, vector3_t(1.0, 0.0, 0.0));
  // Published again at the same time with what was set meanwhile: no step, so
  // no sample of its own.
  accelerometer.record(/*time=*/kStep, vector3_t(2.0, 0.0, 0.0));
  EXPECT_LT((accelerometer.linaccBaseRtWorldEwrtWorld() -
             vector3_t(2.0 / kStep, 0.0, 0.0))
                .norm(),
            kTolerance);
  accelerometer.record(/*time=*/2.0 * kStep, vector3_t(4.0, 0.0, 0.0));
  EXPECT_LT((accelerometer.linaccBaseRtWorldEwrtWorld() -
             vector3_t(4.0 / (2.0 * kStep), 0.0, 0.0))
                .norm(),
            kTolerance);
}

TEST(BaseAccelerometer, RecordingAndReadingAllocateNothing) {
  BaseAccelerometer accelerometer(/*windowSteps=*/20);
  const size_t before = realtime::heapAllocationCountOnThisThread();
  accelerometer.restart(/*time=*/0.0, linvelAt(0.0));
  vector3_t sum = vector3_t::Zero();
  for (int step = 1; step <= 100; ++step) {
    const double time = step * kStep;
    accelerometer.record(time, linvelAt(time));
    sum += accelerometer.linaccBaseRtWorldEwrtWorld();
  }
  EXPECT_EQ(realtime::heapAllocationCountOnThisThread() - before, 0U);
  EXPECT_TRUE(sum.allFinite());
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
