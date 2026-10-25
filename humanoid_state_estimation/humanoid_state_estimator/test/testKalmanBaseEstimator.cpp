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

// KalmanBaseEstimator on the closed-form walk of the prismatic biped
// (WalkingTrajectory): with exact sensors it tracks the base, with noisy ones
// it stays within a few noise levels, a wrong initial velocity is corrected,
// the base height is observed through the contact height, and its refusals.

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_state_estimation/humanoid_state_estimator/test/WalkingTrajectory.h"
#include "humanoid_state_estimator/ContactKinematics.h"
#include "humanoid_state_estimator/KalmanBaseEstimator.h"

namespace ocs2::humanoid::estimation {
namespace {

using test_support::WalkingSample;
using test_support::walkingSample;

constexpr scalar_t kDt = 0.002;
constexpr scalar_t kDuration = 8.0;

std::unique_ptr<KalmanBaseEstimator> makeEstimator(
    const KalmanBaseEstimator::Config& config = KalmanBaseEstimator::Config()) {
  std::vector<ContactPointSpec> contacts;
  for (const std::string& frame : test_support::prismaticBipedFootFrames())
    contacts.push_back(ContactPointSpec{.frameNames = {frame}});
  absl::StatusOr<std::unique_ptr<ContactKinematics>> kinematics =
      ContactKinematics::Create(test_support::kPrismaticBipedUrdf, contacts,
                                test_support::prismaticBipedJointNames());
  EXPECT_TRUE(kinematics.ok()) << kinematics.status();
  if (!kinematics.ok()) return nullptr;
  absl::StatusOr<std::unique_ptr<KalmanBaseEstimator>> estimator =
      KalmanBaseEstimator::Create(config, *std::move(kinematics));
  EXPECT_TRUE(estimator.ok()) << estimator.status();
  return estimator.ok() ? *std::move(estimator) : nullptr;
}

/** Gaussian noise on what the sensors read: the IMU, the gyroscope and the
 * joint encoders. */
struct SensorNoise {
  scalar_t acceleration = 0.0;     // [m/s^2]
  scalar_t angularVelocity = 0.0;  // [rad/s]
  scalar_t jointPosition = 0.0;    // [m] (prismatic joints)
  scalar_t jointVelocity = 0.0;    // [m/s]
};

/** The largest and the root-mean-square errors of a run. */
struct Errors {
  scalar_t maxPosition = 0.0;
  scalar_t maxVelocity = 0.0;
  scalar_t rmsPosition = 0.0;
  scalar_t rmsVelocity = 0.0;
};

/** `value` with independent Gaussian noise of standard deviation `sigma` on
 * every entry. */
vector_t addNoise(const vector_t& value,
                  scalar_t sigma,
                  std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  vector_t result = value;
  for (Eigen::Index i = 0; i < result.size(); ++i)
    result[i] += sigma * normal(generator);
  return result;
}

/**
 * Runs `estimator` over the walk from `from` to kDuration, after reset() at the
 * truth with `velocityOffset` added to the base velocity, and measures its
 * errors from `settle` [s] on.
 */
Errors run(KalmanBaseEstimator& estimator,
           const SensorNoise& noise,
           const vector3_t& velocityOffset,
           scalar_t settle) {
  std::mt19937 generator(42);
  const WalkingSample start = walkingSample(0.0);
  EXPECT_TRUE(estimator
                  .reset(start.basePosition,
                         start.baseVelocity + velocityOffset, start.orientation,
                         start.jointPositions, start.jointVelocities)
                  .ok());
  Errors errors;
  int counted = 0;
  const int steps = static_cast<int>(std::round(kDuration / kDt));
  for (int step = 1; step <= steps; ++step) {
    const WalkingSample sample = walkingSample(step * kDt);
    const vector3_t acceleration =
        addNoise(sample.imuLinearAcceleration, noise.acceleration, generator);
    const vector3_t angularVelocity =
        addNoise(sample.angularVelocity, noise.angularVelocity, generator);
    const absl::Status updated = estimator.update(
        kDt, sample.orientation, angularVelocity, acceleration,
        addNoise(sample.jointPositions, noise.jointPosition, generator),
        addNoise(sample.jointVelocities, noise.jointVelocity, generator),
        sample.contactFlags);
    EXPECT_TRUE(updated.ok()) << updated;
    if (sample.time < settle) continue;
    const scalar_t positionError =
        (estimator.basePosition() - sample.basePosition).norm();
    const scalar_t velocityError =
        (estimator.baseLinearVelocity() - sample.baseVelocity).norm();
    errors.maxPosition = std::max(errors.maxPosition, positionError);
    errors.maxVelocity = std::max(errors.maxVelocity, velocityError);
    errors.rmsPosition += positionError * positionError;
    errors.rmsVelocity += velocityError * velocityError;
    ++counted;
  }
  errors.rmsPosition = std::sqrt(errors.rmsPosition / counted);
  errors.rmsVelocity = std::sqrt(errors.rmsVelocity / counted);
  return errors;
}

TEST(KalmanBaseEstimatorTest, TracksTheWalkWithExactSensors) {
  std::unique_ptr<KalmanBaseEstimator> estimator = makeEstimator();
  ASSERT_NE(estimator, nullptr);
  const Errors errors =
      run(*estimator, SensorNoise(), vector3_t::Zero(), /*settle=*/0.0);
  // Exact sensors: what remains is the prediction's rectangle rule on an
  // acceleration that changes (the sway and the bob, |a_dot| < 0.4 m/s^3: dt^2
  // |a_dot| / 2 < 1e-6 m/s per step), which the kinematic velocity measurement
  // pulls back within a few steps. Millimeters and centimeters per second would
  // be a wrong model, not rounding.
  EXPECT_LT(errors.maxPosition, 1.0e-3) << errors.rmsPosition;
  EXPECT_LT(errors.maxVelocity, 5.0e-3) << errors.rmsVelocity;
}

TEST(KalmanBaseEstimatorTest, StaysWithinTheNoiseWithNoisySensors) {
  std::unique_ptr<KalmanBaseEstimator> estimator = makeEstimator();
  ASSERT_NE(estimator, nullptr);
  const SensorNoise noise{.acceleration = 0.2,
                          .angularVelocity = 0.01,
                          .jointPosition = 0.001,
                          .jointVelocity = 0.02};
  const Errors errors =
      run(*estimator, noise, vector3_t::Zero(), /*settle=*/0.0);
  // The kinematic velocity has the joint velocity noise (0.02 m/s per axis)
  // plus omega x p_rel's (0.01 rad/s at 0.8 m), about 0.03 m/s per sample; the
  // filter averages it with the IMU's over many samples, so its RMS is well
  // below it. The position's x and y are odometry: they drift with the stance
  // feet's random walk and the velocity error's integral over 8 s, a few
  // millimeters; z is held by the contact height to the joint noise.
  EXPECT_LT(errors.rmsVelocity, 0.02);
  EXPECT_LT(errors.maxVelocity, 0.08);
  EXPECT_LT(errors.maxPosition, 0.02);
}

TEST(KalmanBaseEstimatorTest, AWrongInitialVelocityConverges) {
  std::unique_ptr<KalmanBaseEstimator> estimator = makeEstimator();
  ASSERT_NE(estimator, nullptr);
  // 0.6 m/s wrong at the start; the kinematic velocity measurement (0.05 m/s
  // against an initial 0.1 m/s) corrects it within a few steps, so after 0.5 s
  // the error is that of the exact run.
  const Errors errors =
      run(*estimator, SensorNoise(), vector3_t(0.5, -0.3, 0.2), /*settle=*/0.5);
  EXPECT_LT(errors.maxVelocity, 5.0e-3);
}

TEST(KalmanBaseEstimatorTest, TheHeightIsObservedThroughTheContactHeight) {
  std::unique_ptr<KalmanBaseEstimator> estimator = makeEstimator();
  ASSERT_NE(estimator, nullptr);
  const WalkingSample start = walkingSample(0.0);
  // Started 5 cm too high: the feet start 5 cm above the ground too, and the
  // contact height brings both down.
  ASSERT_TRUE(estimator
                  ->reset(start.basePosition + vector3_t(0.0, 0.0, 0.05),
                          start.baseVelocity, start.orientation,
                          start.jointPositions, start.jointVelocities)
                  .ok());
  for (int step = 1; step <= 1000; ++step) {
    const WalkingSample sample = walkingSample(step * kDt);
    ASSERT_TRUE(estimator
                    ->update(kDt, sample.orientation, sample.angularVelocity,
                             sample.imuLinearAcceleration,
                             sample.jointPositions, sample.jointVelocities,
                             sample.contactFlags)
                    .ok());
  }
  EXPECT_NEAR(estimator->basePosition().z(),
              walkingSample(1000 * kDt).basePosition.z(), 2.0e-3);
  const absl::StatusOr<vector3_t> foot = estimator->contactPosition(0);
  ASSERT_TRUE(foot.ok()) << foot.status();
  EXPECT_NEAR(foot->z(), walkingSample(1000 * kDt).footPositions[0].z(),
              2.0e-3);
}

TEST(KalmanBaseEstimatorTest, RefusesANoiseModelOutOfRangeNamingTheField) {
  KalmanBaseEstimator::Config config;
  config.kinematicVelocityNoise = -1.0;
  const absl::Status refused = KalmanBaseEstimator::validateConfig(config);
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(
      refused.message(), "kalman_state_estimator.kinematic_velocity_noise"))
      << refused;
  config = KalmanBaseEstimator::Config();
  config.contactHeight = std::nan("");
  EXPECT_FALSE(KalmanBaseEstimator::validateConfig(config).ok());
  EXPECT_TRUE(
      KalmanBaseEstimator::validateConfig(KalmanBaseEstimator::Config()).ok());
}

TEST(KalmanBaseEstimatorTest, ABadStepIsRefusedAndAFailedOneKeepsTheEstimate) {
  std::unique_ptr<KalmanBaseEstimator> estimator = makeEstimator();
  ASSERT_NE(estimator, nullptr);
  const WalkingSample sample = walkingSample(0.0);
  ASSERT_TRUE(estimator
                  ->reset(sample.basePosition, sample.baseVelocity,
                          sample.orientation, sample.jointPositions,
                          sample.jointVelocities)
                  .ok());
  const vector3_t position = estimator->basePosition();
  EXPECT_FALSE(estimator
                   ->update(/*dt=*/0.0, sample.orientation,
                            sample.angularVelocity,
                            sample.imuLinearAcceleration, sample.jointPositions,
                            sample.jointVelocities, sample.contactFlags)
                   .ok());
  EXPECT_FALSE(estimator
                   ->update(kDt, sample.orientation, sample.angularVelocity,
                            vector3_t::Constant(std::nan("")),
                            sample.jointPositions, sample.jointVelocities,
                            sample.contactFlags)
                   .ok());
  EXPECT_FALSE(estimator
                   ->update(kDt, sample.orientation, sample.angularVelocity,
                            sample.imuLinearAcceleration, sample.jointPositions,
                            sample.jointVelocities, std::vector<bool>(3, true))
                   .ok());
  EXPECT_EQ(estimator->basePosition(), position);
}

}  // namespace
}  // namespace ocs2::humanoid::estimation
