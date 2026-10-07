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

// The real-time property of KalmanFilter: once its workspace is sized, a step allocates nothing on the heap. The
// allocation counter interposes malloc for this whole binary, which is why these tests have a target of their own.

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "humanoid_state_estimation/humanoid_state_estimator/test/AllocationCounter.h"
#include "humanoid_state_estimator/KalmanFilter.h"

namespace ocs2::humanoid::estimation {
namespace {

constexpr scalar_t kDt = 0.001;

matrix_t identity(Eigen::Index dim) {
  return matrix_t::Identity(dim, dim);
}

/**
 * The estimator the SA01 would run, held the way a real-time loop holds it: base position and velocity and one 3D
 * position per foot, the world-frame IMU acceleration as the input, and per stance foot its position relative to the
 * base, its velocity and its height. The model, input and measurement vectors are built once; a step only overwrites
 * their values.
 */
class KalmanFilterAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<KalmanFilter>> filter = KalmanFilter::Create(
        {{.name = "base_position", .state = (vector_t(3) << 0.0, 0.0, 0.8).finished(), .P_state_estimate = 1.0e-4 * identity(3)},
         {.name = "base_velocity", .state = vector_t::Zero(3), .P_state_estimate = identity(3)},
         {.name = "left_foot_position", .state = (vector_t(3) << 0.0, 0.1, 0.0).finished(), .P_state_estimate = 1.0e-2 * identity(3)},
         {.name = "right_foot_position", .state = (vector_t(3) << 0.0, -0.1, 0.0).finished(), .P_state_estimate = 1.0e-2 * identity(3)}});
    ASSERT_TRUE(filter.ok()) << filter.status();
    filter_ = std::move(*filter);

    process_model_ = {{.state_name = "base_position",
                       .A_state_transition = {{"base_velocity", kDt * identity(3)}},
                       .B_control_input = {{"imu_acceleration", 0.5 * kDt * kDt * identity(3)}}},
                      {.state_name = "base_velocity", .B_control_input = {{"imu_acceleration", kDt * identity(3)}}},
                      {.state_name = "left_foot_position", .Q_process_noise = 1.0e-8 * identity(3)},
                      {.state_name = "right_foot_position", .Q_process_noise = 1.0e-8 * identity(3)}};
    inputs_ = {{.name = "imu_acceleration", .input = vector_t::Zero(3), .Q_input_noise = 1.0e-4 * identity(3)}};

    matrix_t e_z = matrix_t::Zero(1, 3);
    e_z(0, 2) = 1.0;
    for (const std::string foot : {"left_foot", "right_foot"}) {
      const std::string position = absl::StrCat(foot, "_position");
      double_support_.push_back({.name = absl::StrCat(foot, "_relative_position"),
                                 .measurement = vector_t::Zero(3),
                                 .H_measurement_model = {{position, identity(3)}, {"base_position", -identity(3)}},
                                 .R_measurement_noise = 1.0e-6 * identity(3)});
      double_support_.push_back({.name = absl::StrCat(foot, "_velocity"),
                                 .measurement = vector_t::Zero(3),
                                 .H_measurement_model = {{"base_velocity", identity(3)}},
                                 .R_measurement_noise = 1.0e-4 * identity(3)});
      double_support_.push_back({.name = absl::StrCat(foot, "_height"),
                                 .measurement = vector_t::Zero(1),
                                 .H_measurement_model = {{position, e_z}},
                                 .R_measurement_noise = 1.0e-6 * identity(1)});
    }
    left_support_.assign(double_support_.begin(), double_support_.begin() + 3);

    // Read once, so that the caller's KalmanFilterState already owns storage of the right size.
    ASSERT_TRUE(filter_->getState("base_velocity", base_velocity_).ok());
  }

  /// One control tick: fresh sensor values written in place, then predict, correct and read. False if a call failed.
  bool step(std::vector<KalmanFilterMeasurement>& measurements, int tick) {
    inputs_[0].input(0) = 0.01 * tick;
    measurements[0].measurement << 0.0, 0.1, -0.8 - 1.0e-4 * tick;
    measurements[0].H_measurement_model.at("base_position") = -identity_;
    return filter_->predict(process_model_, inputs_).ok() && filter_->correct(measurements).ok() &&
           filter_->getState("base_velocity", base_velocity_).ok();
  }

  std::unique_ptr<KalmanFilter> filter_;
  std::vector<KalmanFilterProcessModel> process_model_;
  std::vector<KalmanFilterInput> inputs_;
  std::vector<KalmanFilterMeasurement> double_support_;
  std::vector<KalmanFilterMeasurement> left_support_;
  KalmanFilterState base_velocity_;
  const matrix_t identity_ = identity(3);
};

// Without this, a counter that saw nothing would make every test below pass.
TEST_F(KalmanFilterAllocationTest, theCounterSeesAllocationsMadeInsideTheFilterLibrary) {
  const size_t allocations_before = heapAllocationCount();
  const absl::StatusOr<KalmanFilterState> copy = filter_->getState("base_velocity");
  const size_t allocations = heapAllocationCount() - allocations_before;
  ASSERT_TRUE(copy.ok());
  EXPECT_GE(allocations, 2u) << "returning a KalmanFilterState by value allocates its state and its covariance";
}

TEST_F(KalmanFilterAllocationTest, stepsAfterTheFirstDoNotAllocate) {
  ASSERT_TRUE(step(double_support_, /*tick=*/0));  // sizes the workspace

  const size_t allocations_before = heapAllocationCount();
  bool all_succeeded = true;
  for (int tick = 1; tick <= 200; ++tick) {
    all_succeeded = step(double_support_, tick) && all_succeeded;
  }
  const size_t allocations = heapAllocationCount() - allocations_before;

  EXPECT_TRUE(all_succeeded);
  EXPECT_EQ(allocations, 0u);
}

TEST_F(KalmanFilterAllocationTest, reserveMakesEvenTheFirstStepAllocationFree) {
  ASSERT_TRUE(filter_->reserve(/*max_input_dim=*/3, /*max_measurement_dim=*/14).ok());

  const size_t allocations_before = heapAllocationCount();
  const bool succeeded = step(double_support_, /*tick=*/0);
  const size_t allocations = heapAllocationCount() - allocations_before;

  EXPECT_TRUE(succeeded);
  EXPECT_EQ(allocations, 0u);
}

// The measurement set shrinks and grows as the feet lift off and touch down; the workspace only grows, so once the
// largest set has been seen, switching does not allocate.
TEST_F(KalmanFilterAllocationTest, switchingBetweenSingleAndDoubleSupportDoesNotAllocate) {
  ASSERT_TRUE(step(double_support_, /*tick=*/0));

  const size_t allocations_before = heapAllocationCount();
  bool all_succeeded = true;
  for (int tick = 1; tick <= 200; ++tick) {
    all_succeeded = step(tick % 50 < 25 ? left_support_ : double_support_, tick) && all_succeeded;
  }
  const size_t allocations = heapAllocationCount() - allocations_before;

  EXPECT_TRUE(all_succeeded);
  EXPECT_EQ(allocations, 0u);
}

// reset() rebuilds the channels and may allocate, but it keeps the capacity that reserve() asked for.
TEST_F(KalmanFilterAllocationTest, theReservedCapacitySurvivesAReset) {
  ASSERT_TRUE(filter_->reserve(/*max_input_dim=*/3, /*max_measurement_dim=*/14).ok());
  const std::vector<KalmanFilterState> states = filter_->getStates();
  ASSERT_TRUE(filter_->reset(states).ok());

  const size_t allocations_before = heapAllocationCount();
  const bool succeeded = step(double_support_, /*tick=*/0);
  const size_t allocations = heapAllocationCount() - allocations_before;

  EXPECT_TRUE(succeeded);
  EXPECT_EQ(allocations, 0u);
}

}  // namespace
}  // namespace ocs2::humanoid::estimation
