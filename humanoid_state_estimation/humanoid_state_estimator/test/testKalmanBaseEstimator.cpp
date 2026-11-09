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
// the base height is observed through the contact heights the caller gives
// (flat ground, a box top, or none for one contact), and its refusals.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_state_estimation/humanoid_state_estimator/test/WalkingTrajectory.h"
#include "humanoid_state_estimator/ContactEstimationInputs.h"
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
  for (const std::string& link : test_support::prismaticBipedFootLinks()) {
    contacts.push_back(ContactPointSpec{.linkName = link});
  }
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
  scalar_t accelerometer = 0.0;  // [m/s^2]
  scalar_t gyroscope = 0.0;      // [rad/s]
  scalar_t jointPosition = 0.0;  // [m] (prismatic joints)
  scalar_t jointVelocity = 0.0;  // [m/s]
};

/** The largest and the root-mean-square norms of the errors of a run. */
struct Errors {
  scalar_t maxErrPosBaseRtWorldEwrtWorld = 0.0;
  scalar_t maxErrLinvelBaseRtWorldEwrtWorld = 0.0;
  scalar_t rmsErrPosBaseRtWorldEwrtWorld = 0.0;
  scalar_t rmsErrLinvelBaseRtWorldEwrtWorld = 0.0;
};

/** `value` with independent Gaussian noise of standard deviation `sigma` on
 * every entry. */
vector_t addNoise(const vector_t& value,
                  scalar_t sigma,
                  std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  vector_t result = value;
  for (Eigen::Index i = 0; i < result.size(); ++i) {
    result[i] += sigma * normal(generator);
  }
  return result;
}

/**
 * Runs `estimator` over the walk from 0 to kDuration, after reset() at the
 * truth with `initErrLinvelBaseRtWorldEwrtWorld` added to the base velocity,
 * and measures its errors from `settle` [s] on.
 */
Errors run(KalmanBaseEstimator& estimator,
           const SensorNoise& noise,
           const vector3_t& initErrLinvelBaseRtWorldEwrtWorld,
           scalar_t settle) {
  const ContactEstimationInputs flatGround(test_support::kNumFeet);
  std::mt19937 generator(42);
  const WalkingSample start = walkingSample(0.0);
  EXPECT_TRUE(estimator
                  .reset(start.posBaseRtWorldEwrtWorld,
                         start.linvelBaseRtWorldEwrtWorld +
                             initErrLinvelBaseRtWorldEwrtWorld,
                         start.quatBaseRtWorld, start.jointPositions,
                         start.jointVelocities)
                  .ok());
  Errors errors;
  int counted = 0;
  const int steps = static_cast<int>(std::round(kDuration / kDt));
  for (int step = 1; step <= steps; ++step) {
    const WalkingSample sample = walkingSample(step * kDt);
    const vector3_t measSpecforceBaseRtWorldEwrtBase = addNoise(
        sample.specforceBaseRtWorldEwrtBase, noise.accelerometer, generator);
    const vector3_t measAngvelBaseRtWorldEwrtBase =
        addNoise(sample.angvelBaseRtWorldEwrtBase, noise.gyroscope, generator);
    const absl::Status updated = estimator.update(
        kDt, sample.quatBaseRtWorld, measAngvelBaseRtWorldEwrtBase,
        measSpecforceBaseRtWorldEwrtBase,
        addNoise(sample.jointPositions, noise.jointPosition, generator),
        addNoise(sample.jointVelocities, noise.jointVelocity, generator),
        sample.contactFlags, flatGround);
    EXPECT_TRUE(updated.ok()) << updated;
    if (sample.time < settle) continue;
    const scalar_t normErrPosBaseRtWorldEwrtWorld =
        (estimator.posBaseRtWorldEwrtWorld() - sample.posBaseRtWorldEwrtWorld)
            .norm();
    const scalar_t normErrLinvelBaseRtWorldEwrtWorld =
        (estimator.linvelBaseRtWorldEwrtWorld() -
         sample.linvelBaseRtWorldEwrtWorld)
            .norm();
    errors.maxErrPosBaseRtWorldEwrtWorld = std::max(
        errors.maxErrPosBaseRtWorldEwrtWorld, normErrPosBaseRtWorldEwrtWorld);
    errors.maxErrLinvelBaseRtWorldEwrtWorld =
        std::max(errors.maxErrLinvelBaseRtWorldEwrtWorld,
                 normErrLinvelBaseRtWorldEwrtWorld);
    errors.rmsErrPosBaseRtWorldEwrtWorld +=
        normErrPosBaseRtWorldEwrtWorld * normErrPosBaseRtWorldEwrtWorld;
    errors.rmsErrLinvelBaseRtWorldEwrtWorld +=
        normErrLinvelBaseRtWorldEwrtWorld * normErrLinvelBaseRtWorldEwrtWorld;
    ++counted;
  }
  errors.rmsErrPosBaseRtWorldEwrtWorld =
      std::sqrt(errors.rmsErrPosBaseRtWorldEwrtWorld / counted);
  errors.rmsErrLinvelBaseRtWorldEwrtWorld =
      std::sqrt(errors.rmsErrLinvelBaseRtWorldEwrtWorld / counted);
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
  EXPECT_LT(errors.maxErrPosBaseRtWorldEwrtWorld, 1.0e-3)
      << errors.rmsErrPosBaseRtWorldEwrtWorld;
  EXPECT_LT(errors.maxErrLinvelBaseRtWorldEwrtWorld, 5.0e-3)
      << errors.rmsErrLinvelBaseRtWorldEwrtWorld;
}

TEST(KalmanBaseEstimatorTest, StaysWithinTheNoiseWithNoisySensors) {
  std::unique_ptr<KalmanBaseEstimator> estimator = makeEstimator();
  ASSERT_NE(estimator, nullptr);
  const SensorNoise noise{.accelerometer = 0.2,
                          .gyroscope = 0.01,
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
  EXPECT_LT(errors.rmsErrLinvelBaseRtWorldEwrtWorld, 0.02);
  EXPECT_LT(errors.maxErrLinvelBaseRtWorldEwrtWorld, 0.08);
  EXPECT_LT(errors.maxErrPosBaseRtWorldEwrtWorld, 0.02);
}

TEST(KalmanBaseEstimatorTest, AWrongInitialVelocityConverges) {
  std::unique_ptr<KalmanBaseEstimator> estimator = makeEstimator();
  ASSERT_NE(estimator, nullptr);
  // 0.6 m/s wrong at the start; the kinematic velocity measurement (0.05 m/s
  // against an initial 0.1 m/s) corrects it within a few steps, so after 0.5 s
  // the error is that of the exact run.
  const Errors errors =
      run(*estimator, SensorNoise(), vector3_t(0.5, -0.3, 0.2), /*settle=*/0.5);
  EXPECT_LT(errors.maxErrLinvelBaseRtWorldEwrtWorld, 5.0e-3);
}

TEST(KalmanBaseEstimatorTest, TheHeightIsObservedThroughTheContactHeight) {
  std::unique_ptr<KalmanBaseEstimator> estimator = makeEstimator();
  ASSERT_NE(estimator, nullptr);
  const WalkingSample start = walkingSample(0.0);
  // Started 5 cm too high: the feet start 5 cm above the ground too, and the
  // contact height brings both down.
  const ContactEstimationInputs flatGround(test_support::kNumFeet);
  ASSERT_TRUE(
      estimator
          ->reset(start.posBaseRtWorldEwrtWorld + vector3_t(0.0, 0.0, 0.05),
                  start.linvelBaseRtWorldEwrtWorld, start.quatBaseRtWorld,
                  start.jointPositions, start.jointVelocities)
          .ok());
  for (int step = 1; step <= 1000; ++step) {
    const WalkingSample sample = walkingSample(step * kDt);
    ASSERT_TRUE(estimator
                    ->update(kDt, sample.quatBaseRtWorld,
                             sample.angvelBaseRtWorldEwrtBase,
                             sample.specforceBaseRtWorldEwrtBase,
                             sample.jointPositions, sample.jointVelocities,
                             sample.contactFlags, flatGround)
                    .ok());
  }
  EXPECT_NEAR(estimator->posBaseRtWorldEwrtWorld().z(),
              walkingSample(1000 * kDt).posBaseRtWorldEwrtWorld.z(), 2.0e-3);
  const absl::StatusOr<vector3_t> estPosLfootRtWorldEwrtWorld =
      estimator->posContactRtWorldEwrtWorld(0);
  ASSERT_TRUE(estPosLfootRtWorldEwrtWorld.ok())
      << estPosLfootRtWorldEwrtWorld.status();
  EXPECT_NEAR(estPosLfootRtWorldEwrtWorld->z(),
              walkingSample(1000 * kDt).posFeetRtWorldEwrtWorld[0].z(), 2.0e-3);
  // The allocation-free copy is the filter's state.
  ASSERT_EQ(estimator->estPosContactsRtWorldEwrtWorld().size(),
            test_support::kNumFeet);
  EXPECT_EQ(estimator->estPosContactsRtWorldEwrtWorld()[0],
            *estPosLfootRtWorldEwrtWorld);
}

/** What one exact-sensor run over the walk ends with. */
struct RunEnd {
  vector3_t posBaseRtWorldEwrtWorld = vector3_t::Zero();
  vector3_t linvelBaseRtWorldEwrtWorld = vector3_t::Zero();
  vector3_t truePosBaseRtWorldEwrtWorld = vector3_t::Zero();
  vector3_t trueLinvelBaseRtWorldEwrtWorld = vector3_t::Zero();
};

/** How a run treats the left foot (contact 0). */
enum class LeftFoot {
  kAsWalked,       // its flags and the run's heights
  kWithoutHeight,  // its flags, no height row
  kAlwaysInSwing,  // never in stance
};

/**
 * Runs a new estimator over the exact walk from `startTime` for `steps`
 * steps, with the whole walk raised by `posSurfaceRtWorldEwrtWorldZ` [m] (a
 * box top) and every contact given that surface height, the left foot as
 * `leftFoot` says. reset() is at the raised truth plus `initErrPos...` and
 * `initErrLinvel...`.
 */
RunEnd runOnGround(scalar_t posSurfaceRtWorldEwrtWorldZ,
                   LeftFoot leftFoot,
                   scalar_t startTime,
                   int steps,
                   const vector3_t& initErrPosBaseRtWorldEwrtWorld,
                   const vector3_t& initErrLinvelBaseRtWorldEwrtWorld,
                   bool explicitHeights = true) {
  std::unique_ptr<KalmanBaseEstimator> estimator = makeEstimator();
  EXPECT_NE(estimator, nullptr);
  if (estimator == nullptr) return RunEnd();
  const vector3_t raise(0.0, 0.0, posSurfaceRtWorldEwrtWorldZ);
  ContactEstimationInputs contactInputs(test_support::kNumFeet);
  if (explicitHeights) {
    contactInputs.setAllNomPosContactsRtWorldEwrtWorldZ(
        posSurfaceRtWorldEwrtWorldZ);
  }
  if (leftFoot == LeftFoot::kWithoutHeight) {
    contactInputs.clearNomPosContactRtWorldEwrtWorldZ(0);
  }
  const WalkingSample start = walkingSample(startTime);
  EXPECT_TRUE(estimator
                  ->reset(start.posBaseRtWorldEwrtWorld + raise +
                              initErrPosBaseRtWorldEwrtWorld,
                          start.linvelBaseRtWorldEwrtWorld +
                              initErrLinvelBaseRtWorldEwrtWorld,
                          start.quatBaseRtWorld, start.jointPositions,
                          start.jointVelocities)
                  .ok());
  WalkingSample sample = start;
  for (int step = 1; step <= steps; ++step) {
    sample = walkingSample(startTime + step * kDt);
    std::vector<bool> contactFlags = sample.contactFlags;
    if (leftFoot == LeftFoot::kAlwaysInSwing) contactFlags[0] = false;
    const absl::Status updated = estimator->update(
        kDt, sample.quatBaseRtWorld, sample.angvelBaseRtWorldEwrtBase,
        sample.specforceBaseRtWorldEwrtBase, sample.jointPositions,
        sample.jointVelocities, contactFlags, contactInputs);
    EXPECT_TRUE(updated.ok()) << updated;
  }
  return RunEnd{
      .posBaseRtWorldEwrtWorld = estimator->posBaseRtWorldEwrtWorld(),
      .linvelBaseRtWorldEwrtWorld = estimator->linvelBaseRtWorldEwrtWorld(),
      .truePosBaseRtWorldEwrtWorld = sample.posBaseRtWorldEwrtWorld + raise,
      .trueLinvelBaseRtWorldEwrtWorld = sample.linvelBaseRtWorldEwrtWorld};
}

TEST(KalmanBaseEstimatorTest, AFreshContactInputIsFlatGroundAtZero) {
  // The inputs a robot process starts with give every contact the height 0,
  // the flat ground the estimator assumed before it took per-contact heights:
  // bit for bit the run with every height set to 0 explicitly.
  const vector3_t initErrPosBaseRtWorldEwrtWorld(0.0, 0.0, 0.05);
  const RunEnd fresh = runOnGround(
      /*posSurfaceRtWorldEwrtWorldZ=*/0.0, LeftFoot::kAsWalked,
      /*startTime=*/0.0, /*steps=*/1000, initErrPosBaseRtWorldEwrtWorld,
      vector3_t::Zero(), /*explicitHeights=*/false);
  const RunEnd explicitZero = runOnGround(
      /*posSurfaceRtWorldEwrtWorldZ=*/0.0, LeftFoot::kAsWalked,
      /*startTime=*/0.0, /*steps=*/1000, initErrPosBaseRtWorldEwrtWorld,
      vector3_t::Zero(), /*explicitHeights=*/true);
  EXPECT_EQ(fresh.posBaseRtWorldEwrtWorld,
            explicitZero.posBaseRtWorldEwrtWorld);
  EXPECT_EQ(fresh.linvelBaseRtWorldEwrtWorld,
            explicitZero.linvelBaseRtWorldEwrtWorld);
  const ContactEstimationInputs inputs(test_support::kNumFeet);
  for (size_t contact = 0; contact < inputs.numContacts(); ++contact) {
    EXPECT_TRUE(inputs.hasNomPosContactRtWorldEwrtWorldZ(contact));
    EXPECT_EQ(inputs.nomPosContactRtWorldEwrtWorldZ(contact), 0.0);
  }
}

TEST(KalmanBaseEstimatorTest, ContactsOnABoxTopBringTheBaseToItsHeight) {
  // The robot walks on a 0.3 m box top, and the estimate starts at the floor's
  // height: the contacts' surface height lifts it to the box's.
  constexpr scalar_t kBoxTopHeight = 0.3;
  const RunEnd end = runOnGround(
      kBoxTopHeight, LeftFoot::kAsWalked, /*startTime=*/0.0,
      /*steps=*/1000, vector3_t(0.0, 0.0, -kBoxTopHeight), vector3_t::Zero());
  EXPECT_NEAR(end.posBaseRtWorldEwrtWorld.z(),
              end.truePosBaseRtWorldEwrtWorld.z(), 2.0e-3);
  EXPECT_LT((end.posBaseRtWorldEwrtWorld - end.truePosBaseRtWorldEwrtWorld)
                .head<2>()
                .norm(),
            2.0e-3);
}

TEST(KalmanBaseEstimatorTest, AContactWithoutAHeightKeepsItsOtherRows) {
  // Over the whole walk, the right foot's height alone brings a base started
  // 5 cm high down to the truth.
  const RunEnd heightThroughTheRightFoot = runOnGround(
      /*posSurfaceRtWorldEwrtWorldZ=*/0.0, LeftFoot::kWithoutHeight,
      /*startTime=*/0.0, /*steps=*/1000, vector3_t(0.0, 0.0, 0.05),
      vector3_t::Zero());
  EXPECT_NEAR(heightThroughTheRightFoot.posBaseRtWorldEwrtWorld.z(),
              heightThroughTheRightFoot.truePosBaseRtWorldEwrtWorld.z(),
              2.0e-3);

  // While the right foot swings (0.08 s to 0.32 s of the walk), the left foot
  // is the only stance contact. Without its height row, its velocity rows
  // still correct a velocity 0.6 m/s wrong at 0.1 s; with the left foot in
  // swing there is no correction, and the error stays.
  constexpr scalar_t kRightSwingStart = 0.1;
  constexpr int kStepsInRightSwing = 100;
  const vector3_t initErrLinvelBaseRtWorldEwrtWorld(0.5, -0.3, 0.2);
  const RunEnd withoutHeight = runOnGround(
      /*posSurfaceRtWorldEwrtWorldZ=*/0.0, LeftFoot::kWithoutHeight,
      kRightSwingStart, kStepsInRightSwing, vector3_t::Zero(),
      initErrLinvelBaseRtWorldEwrtWorld);
  const RunEnd inSwing = runOnGround(
      /*posSurfaceRtWorldEwrtWorldZ=*/0.0, LeftFoot::kAlwaysInSwing,
      kRightSwingStart, kStepsInRightSwing, vector3_t::Zero(),
      initErrLinvelBaseRtWorldEwrtWorld);
  EXPECT_LT((withoutHeight.linvelBaseRtWorldEwrtWorld -
             withoutHeight.trueLinvelBaseRtWorldEwrtWorld)
                .norm(),
            5.0e-3);
  EXPECT_GT((inSwing.linvelBaseRtWorldEwrtWorld -
             inSwing.trueLinvelBaseRtWorldEwrtWorld)
                .norm(),
            0.5);
  // The xy position follows the left foot's position row: within the drift of
  // the corrected velocity, not the wrong velocity's 0.2 s of integration.
  EXPECT_LT(
      (withoutHeight.posBaseRtWorldEwrtWorld -
       withoutHeight.truePosBaseRtWorldEwrtWorld)
          .head<2>()
          .norm(),
      (inSwing.posBaseRtWorldEwrtWorld - inSwing.truePosBaseRtWorldEwrtWorld)
          .head<2>()
          .norm());
}

/** What a run with given planned switch times ends with. */
struct TimedRunEnd {
  vector3_t posBaseRtWorldEwrtWorld = vector3_t::Zero();
  vector3_t linvelBaseRtWorldEwrtWorld = vector3_t::Zero();
  matrix_t covLfoot;  // contact 0's position covariance
  bool lfootAlwaysInStance = true;
};

/**
 * Runs a new estimator of `config` over the first `steps` steps of the exact
 * walk on flat ground, with every contact's planned switch `timeSinceSwitch`
 * [s] behind and `timeToSwitch` [s] ahead in every step, or none known when
 * `setTimes` is false.
 */
TimedRunEnd runWithSwitchTimes(const KalmanBaseEstimator::Config& config,
                               bool setTimes,
                               scalar_t timeSinceSwitch,
                               scalar_t timeToSwitch,
                               int steps) {
  std::unique_ptr<KalmanBaseEstimator> estimator = makeEstimator(config);
  EXPECT_NE(estimator, nullptr);
  if (estimator == nullptr) return TimedRunEnd();
  ContactEstimationInputs contactInputs(test_support::kNumFeet);
  if (setTimes) {
    for (size_t contact = 0; contact < contactInputs.numContacts(); ++contact) {
      contactInputs.setPlannedContactSwitchTimes(contact, timeSinceSwitch,
                                                 timeToSwitch);
    }
  }
  const WalkingSample start = walkingSample(0.0);
  EXPECT_TRUE(estimator
                  ->reset(start.posBaseRtWorldEwrtWorld,
                          start.linvelBaseRtWorldEwrtWorld,
                          start.quatBaseRtWorld, start.jointPositions,
                          start.jointVelocities)
                  .ok());
  TimedRunEnd end;
  for (int step = 1; step <= steps; ++step) {
    const WalkingSample sample = walkingSample(step * kDt);
    end.lfootAlwaysInStance = end.lfootAlwaysInStance && sample.contactFlags[0];
    const absl::Status updated = estimator->update(
        kDt, sample.quatBaseRtWorld, sample.angvelBaseRtWorldEwrtBase,
        sample.specforceBaseRtWorldEwrtBase, sample.jointPositions,
        sample.jointVelocities, sample.contactFlags, contactInputs);
    EXPECT_TRUE(updated.ok()) << updated;
  }
  end.posBaseRtWorldEwrtWorld = estimator->posBaseRtWorldEwrtWorld();
  end.linvelBaseRtWorldEwrtWorld = estimator->linvelBaseRtWorldEwrtWorld();
  const absl::StatusOr<matrix_t> covLfoot =
      estimator->contactStateCovariance(0);
  EXPECT_TRUE(covLfoot.ok()) << covLfoot.status();
  if (covLfoot.ok()) end.covLfoot = *covLfoot;
  return end;
}

TEST(KalmanBaseEstimatorTest, TheRampOffIgnoresThePlannedSwitchTimes) {
  // Off by default: switch times at the switch itself change nothing, bit
  // for bit.
  const KalmanBaseEstimator::Config config;
  ASSERT_EQ(config.contactTrustRampDuration, 0.0);
  const TimedRunEnd unknown =
      runWithSwitchTimes(config, /*setTimes=*/false, /*timeSinceSwitch=*/0.0,
                         /*timeToSwitch=*/0.0, /*steps=*/500);
  const TimedRunEnd atTheSwitch =
      runWithSwitchTimes(config, /*setTimes=*/true, /*timeSinceSwitch=*/0.0,
                         /*timeToSwitch=*/0.0, /*steps=*/500);
  EXPECT_EQ(unknown.posBaseRtWorldEwrtWorld,
            atTheSwitch.posBaseRtWorldEwrtWorld);
  EXPECT_EQ(unknown.linvelBaseRtWorldEwrtWorld,
            atTheSwitch.linvelBaseRtWorldEwrtWorld);
  EXPECT_EQ(unknown.covLfoot, atTheSwitch.covLfoot);
}

TEST(KalmanBaseEstimatorTest, TheRampOnTrustsAContactLessNearItsSwitch) {
  KalmanBaseEstimator::Config config;
  config.contactTrustRampDuration = 0.05;
  config.contactTrustRampMaxScale = 5.0;
  // The left foot stands through the first 0.1 s of the walk.
  constexpr int kSteps = 50;
  const TimedRunEnd near =
      runWithSwitchTimes(config, /*setTimes=*/true, /*timeSinceSwitch=*/0.2,
                         /*timeToSwitch=*/0.005, kSteps);
  const TimedRunEnd far =
      runWithSwitchTimes(config, /*setTimes=*/true, /*timeSinceSwitch=*/0.2,
                         /*timeToSwitch=*/0.2, kSteps);
  const TimedRunEnd unknown =
      runWithSwitchTimes(config, /*setTimes=*/false, /*timeSinceSwitch=*/0.0,
                         /*timeToSwitch=*/0.0, kSteps);
  ASSERT_TRUE(near.lfootAlwaysInStance);
  EXPECT_GT(near.covLfoot.trace(), 1.5 * far.covLfoot.trace())
      << "near " << near.covLfoot.trace() << ", far " << far.covLfoot.trace();
  // Far from a switch and with none known the scale is 1: the same filter.
  EXPECT_EQ(far.covLfoot, unknown.covLfoot);
  EXPECT_EQ(far.posBaseRtWorldEwrtWorld, unknown.posBaseRtWorldEwrtWorld);
  // And the estimate near the switch still tracks the exact walk.
  EXPECT_TRUE(near.posBaseRtWorldEwrtWorld.allFinite());
  EXPECT_TRUE(near.linvelBaseRtWorldEwrtWorld.allFinite());
  EXPECT_LT((near.posBaseRtWorldEwrtWorld -
             walkingSample(kSteps * kDt).posBaseRtWorldEwrtWorld)
                .norm(),
            5.0e-3);
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
  config.nomPosContactRtWorldEwrtWorldZ = std::nan("");
  EXPECT_FALSE(KalmanBaseEstimator::validateConfig(config).ok());
  config = KalmanBaseEstimator::Config();
  config.contactTrustRampDuration = -0.01;
  const absl::Status negativeDuration =
      KalmanBaseEstimator::validateConfig(config);
  EXPECT_TRUE(
      absl::StrContains(negativeDuration.message(),
                        "kalman_state_estimator.contact_trust_ramp_duration"))
      << negativeDuration;
  config.contactTrustRampDuration = std::numeric_limits<scalar_t>::infinity();
  EXPECT_FALSE(KalmanBaseEstimator::validateConfig(config).ok());
  config = KalmanBaseEstimator::Config();
  config.contactTrustRampMaxScale = 0.5;
  const absl::Status smallScale = KalmanBaseEstimator::validateConfig(config);
  EXPECT_TRUE(
      absl::StrContains(smallScale.message(),
                        "kalman_state_estimator.contact_trust_ramp_max_scale"))
      << smallScale;
  config.contactTrustRampMaxScale = std::nan("");
  EXPECT_FALSE(KalmanBaseEstimator::validateConfig(config).ok());
  EXPECT_TRUE(
      KalmanBaseEstimator::validateConfig(KalmanBaseEstimator::Config()).ok());
}

TEST(KalmanBaseEstimatorTest, ABadStepIsRefusedAndAFailedOneKeepsTheEstimate) {
  std::unique_ptr<KalmanBaseEstimator> estimator = makeEstimator();
  ASSERT_NE(estimator, nullptr);
  const WalkingSample sample = walkingSample(0.0);
  ASSERT_TRUE(estimator
                  ->reset(sample.posBaseRtWorldEwrtWorld,
                          sample.linvelBaseRtWorldEwrtWorld,
                          sample.quatBaseRtWorld, sample.jointPositions,
                          sample.jointVelocities)
                  .ok());
  const vector3_t posBaseRtWorldEwrtWorld =
      estimator->posBaseRtWorldEwrtWorld();
  const ContactEstimationInputs flatGround(test_support::kNumFeet);
  EXPECT_FALSE(estimator
                   ->update(/*dt=*/0.0, sample.quatBaseRtWorld,
                            sample.angvelBaseRtWorldEwrtBase,
                            sample.specforceBaseRtWorldEwrtBase,
                            sample.jointPositions, sample.jointVelocities,
                            sample.contactFlags, flatGround)
                   .ok());
  EXPECT_FALSE(estimator
                   ->update(kDt, sample.quatBaseRtWorld,
                            sample.angvelBaseRtWorldEwrtBase,
                            vector3_t::Constant(std::nan("")),
                            sample.jointPositions, sample.jointVelocities,
                            sample.contactFlags, flatGround)
                   .ok());
  EXPECT_FALSE(estimator
                   ->update(kDt, sample.quatBaseRtWorld,
                            sample.angvelBaseRtWorldEwrtBase,
                            sample.specforceBaseRtWorldEwrtBase,
                            sample.jointPositions, sample.jointVelocities,
                            std::vector<bool>(3, true), flatGround)
                   .ok());
  const absl::Status wrongInputs = estimator->update(
      kDt, sample.quatBaseRtWorld, sample.angvelBaseRtWorldEwrtBase,
      sample.specforceBaseRtWorldEwrtBase, sample.jointPositions,
      sample.jointVelocities, sample.contactFlags,
      ContactEstimationInputs(test_support::kNumFeet + 1));
  EXPECT_EQ(wrongInputs.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(wrongInputs.message(), "contact inputs for 3"))
      << wrongInputs;
  EXPECT_EQ(estimator->posBaseRtWorldEwrtWorld(), posBaseRtWorldEwrtWorld);
}

}  // namespace
}  // namespace ocs2::humanoid::estimation
