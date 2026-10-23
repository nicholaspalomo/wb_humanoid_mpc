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

// The realtime property of the state estimators: once built, the kalman estimator's estimate() and reset() make no
// heap allocation, through stance changes. A target of its own, because the allocation counter replaces malloc for
// the whole binary.

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_state_estimation/humanoid_state_estimator/test/AllocationCounter.h"
#include "humanoid_state_estimation/humanoid_state_estimator/test/MeasuredState.h"
#include "humanoid_state_estimation/humanoid_state_estimator/test/WalkingTrajectory.h"
#include "humanoid_state_estimator/StateEstimator.h"
#include "humanoid_state_estimator/StateEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid::estimation {
namespace {

constexpr scalar_t kDt = 0.002;

TEST(StateEstimatorAllocationTest, TheKalmanEstimatorEstimatesAndResetsWithoutAllocating) {
  const absl::StatusOr<robot::model::RobotDescription> description =
      robot::model::RobotDescription::Create(test_support::kPrismaticBipedUrdf);
  ASSERT_TRUE(description.ok()) << description.status();
  absl::StatusOr<std::unique_ptr<StateEstimator>> created =
      StateEstimatorRegistry().create("kalman", StateEstimatorContext{.urdfFile = test_support::kPrismaticBipedUrdf,
                                                                      .contactFrameNames = test_support::prismaticBipedFootFrames(),
                                                                      .contactParentJointNames = {},
                                                                      .description = &*description,
                                                                      .kalman = KalmanBaseEstimator::Config()});
  ASSERT_TRUE(created.ok()) << created.status();
  StateEstimator& estimator = **created;
  robot::model::RobotState measured(*description);
  // The samples are computed before counting: the trajectory generator allocates.
  constexpr int kSteps = 1000;
  std::vector<test_support::WalkingSample> samples;
  for (int step = 0; step <= kSteps; ++step) samples.push_back(test_support::walkingSample(step * kDt));
  test_support::writeMeasuredState(samples[0], measured);
  estimator.reset(measured);
  // One step first, so that the estimated state owns its storage as every later one does.
  test_support::writeMeasuredState(samples[1], measured);
  estimator.estimate(measured);

  bool sawStanceChange = false;
  const size_t before = heapAllocationCount();
  for (int step = 2; step <= kSteps; ++step) {
    sawStanceChange = sawStanceChange || samples[step].contactFlags != samples[step - 1].contactFlags;
    test_support::writeMeasuredState(samples[step], measured);
    estimator.estimate(measured);
    if (step == kSteps / 2) estimator.reset(measured);
  }
  const size_t allocations = heapAllocationCount() - before;
  EXPECT_TRUE(sawStanceChange) << "the run covers lift-offs and touch-downs";
  EXPECT_EQ(allocations, 0u);
  EXPECT_EQ(estimator.failedUpdates(), 0u);
}

}  // namespace
}  // namespace ocs2::humanoid::estimation
