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
#include <vector>

#include <ocs2_core/reference/ModeSchedule.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/gait/GaitOptimizationSettings.h"
#include "humanoid_common_mpc/gait/GaitSwitchingTimeOptimizer.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

namespace {

static constexpr scalar_t kInitialEventTime1 = 0.5;
static constexpr scalar_t kInitialEventTime2 = 1.0;
static constexpr scalar_t kTolerance = 1e-4;

static constexpr scalar_t kMinSingleSupport = 0.25;
static constexpr scalar_t kMaxSingleSupport = 0.50;
static constexpr scalar_t kMinDoubleSupport = 0.05;
static constexpr scalar_t kMaxDoubleSupport = 0.20;

}  // namespace

TEST(TestGaitSwitchingTimeOptimization, DisabledConfig) {
  GaitOptimizationSettings settings;
  settings.enabled = false;
  GaitSwitchingTimeOptimizer optimizer(settings);

  ModeSchedule schedule({kInitialEventTime1, kInitialEventTime2}, {3, 1, 3});
  PrimalSolution primalSolution;
  primalSolution.timeTrajectory_ = {0.0, 0.25, 0.5, 0.75, 1.0, 1.25};
  primalSolution.stateTrajectory_.resize(6, vector_t::Zero(12));
  primalSolution.inputTrajectory_.resize(6, vector_t::Zero(12));

  const bool trajectoryOptimized =
      optimizer.optimizeEventTimes(primalSolution, schedule);
  EXPECT_FALSE(trajectoryOptimized);
  EXPECT_NEAR(schedule.eventTimes[0], kInitialEventTime1, kTolerance);
  EXPECT_NEAR(schedule.eventTimes[1], kInitialEventTime2, kTolerance);

  const contact_flag_t measuredContact = {true, true};
  const bool feedbackAdapted =
      optimizer.adaptFromContactFeedback(measuredContact, 0.45, schedule);
  EXPECT_FALSE(feedbackAdapted);
  EXPECT_NEAR(schedule.eventTimes[0], kInitialEventTime1, kTolerance);
  EXPECT_NEAR(schedule.eventTimes[1], kInitialEventTime2, kTolerance);
}

TEST(TestGaitSwitchingTimeOptimization, EnforceDurationBounds) {
  GaitOptimizationSettings settings;
  settings.enabled = true;
  settings.minSingleSupportDuration = kMinSingleSupport;
  settings.maxSingleSupportDuration = kMaxSingleSupport;
  settings.minDoubleSupportDuration = kMinDoubleSupport;
  settings.maxDoubleSupportDuration = kMaxDoubleSupport;
  GaitSwitchingTimeOptimizer optimizer(settings);

  // Single support mode 1 (left swing) with duration 0.1s (too short, should be
  // clamped to 0.25s) Double support mode 3 with duration 0.6s (too long,
  // should be clamped to 0.20s)
  ModeSchedule schedule({0.1, 0.7}, {1, 3, 1});
  optimizer.enforceDurationBounds(schedule);

  const scalar_t duration0 = schedule.eventTimes[0];
  EXPECT_GE(duration0, kMinSingleSupport - kTolerance);
  EXPECT_LE(duration0, kMaxSingleSupport + kTolerance);

  const scalar_t duration1 = schedule.eventTimes[1] - schedule.eventTimes[0];
  EXPECT_GE(duration1, kMinDoubleSupport - kTolerance);
  EXPECT_LE(duration1, kMaxDoubleSupport + kTolerance);
}

TEST(TestGaitSwitchingTimeOptimization, EarlyTouchdownFeedback) {
  GaitOptimizationSettings settings;
  settings.enabled = true;
  settings.enableContactFeedback = true;
  settings.earlyTouchDownTimeWindow = 0.15;
  settings.minSingleSupportDuration = kMinSingleSupport;
  settings.maxSingleSupportDuration = kMaxSingleSupport;
  settings.minDoubleSupportDuration = kMinDoubleSupport;
  settings.maxDoubleSupportDuration = kMaxDoubleSupport;
  GaitSwitchingTimeOptimizer optimizer(settings);

  // Mode 1: Left foot swing (stance = {false, true}) until t = 0.5s
  ModeSchedule schedule({kInitialEventTime1, kInitialEventTime2}, {1, 3, 2});

  // At t = 0.42s, left foot makes early touchdown (measured = {true, true})
  const scalar_t currentTime = 0.42;
  const contact_flag_t measuredContact = {true, true};

  const bool adapted = optimizer.adaptFromContactFeedback(
      measuredContact, currentTime, schedule);
  EXPECT_TRUE(adapted);
  // Event time should be moved earlier to currentTime (or bounded)
  EXPECT_LE(schedule.eventTimes[0], 0.45);
}

TEST(TestGaitSwitchingTimeOptimization, StatusOrSensitivityLookup) {
  GaitOptimizationSettings settings;
  settings.enabled = true;
  GaitSwitchingTimeOptimizer optimizer(settings);

  PrimalSolution primalSolution;
  primalSolution.timeTrajectory_ = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
  primalSolution.stateTrajectory_.resize(6, vector_t::Zero(12));
  primalSolution.inputTrajectory_.resize(6, vector_t::Zero(12));

  // Valid interior point
  const absl::StatusOr<size_t> interiorIndexStatus =
      GaitSwitchingTimeOptimizer::findInteriorTimeIndex(
          primalSolution.timeTrajectory_, 0.4);
  ASSERT_TRUE(interiorIndexStatus.ok());
  EXPECT_EQ(interiorIndexStatus.value(), 2);

  // Out of range (boundary point 0.0 or outside horizon)
  const absl::StatusOr<size_t> boundaryIndexStatus =
      GaitSwitchingTimeOptimizer::findInteriorTimeIndex(
          primalSolution.timeTrajectory_, 0.0);
  EXPECT_FALSE(boundaryIndexStatus.ok());
  EXPECT_EQ(boundaryIndexStatus.status().code(), absl::StatusCode::kOutOfRange);

  // Sensitivity computation with valid interior point
  const absl::StatusOr<scalar_t> sensitivityStatus =
      optimizer.computeSwitchingTimeSensitivity(0.4, 1, 3, primalSolution);
  EXPECT_TRUE(sensitivityStatus.ok());

  // Sensitivity computation with boundary point
  const absl::StatusOr<scalar_t> invalidSensitivityStatus =
      optimizer.computeSwitchingTimeSensitivity(0.0, 1, 3, primalSolution);
  EXPECT_FALSE(invalidSensitivityStatus.ok());
}

TEST(TestGaitSwitchingTimeOptimization, SensitivityOptimization) {
  GaitOptimizationSettings settings;
  settings.enabled = true;
  settings.enableTrajectorySensitivity = true;
  settings.stepSize = 0.05;
  settings.maxIterations = 2;
  settings.minSingleSupportDuration = kMinSingleSupport;
  settings.maxSingleSupportDuration = kMaxSingleSupport;
  settings.minDoubleSupportDuration = kMinDoubleSupport;
  settings.maxDoubleSupportDuration = kMaxDoubleSupport;
  GaitSwitchingTimeOptimizer optimizer(settings);

  ModeSchedule schedule({0.4, 0.8}, {1, 3, 2});

  PrimalSolution primalSolution;
  primalSolution.timeTrajectory_ = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
  primalSolution.stateTrajectory_.resize(6, vector_t::Zero(12));
  primalSolution.inputTrajectory_.resize(6, vector_t::Zero(12));

  // Introduce a jump in inputs before and after event time
  primalSolution.inputTrajectory_[1].setConstant(10.0);
  primalSolution.inputTrajectory_[2].setConstant(0.0);

  const bool modified = optimizer.optimizeEventTimes(primalSolution, schedule);
  EXPECT_TRUE(modified);

  // Durations must still be within bounds
  const scalar_t dur0 = schedule.eventTimes[0];
  EXPECT_GE(dur0, kMinSingleSupport - kTolerance);
  EXPECT_LE(dur0, kMaxSingleSupport + kTolerance);
}

}  // namespace ocs2::humanoid
