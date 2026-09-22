/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <cmath>

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"

namespace ocs2::humanoid {
namespace {

using Controller = CentroidalMpcMrtJointController;

constexpr scalar_t kTimeConstant = 0.5;

/// SAFETY hands out the full joint PD at the instant the mode is entered, so the torque does not jump on the switch.
TEST(SafetyModeDecay, startsAtFullAuthority) {
  EXPECT_DOUBLE_EQ(Controller::safetyDecayFactor(0.0, kTimeConstant), 1.0);
}

/// The decay is exponential, not linear: the shipped behaviour before this was a linear ramp that kept most of its
/// authority through the first half of the window, which is the opposite of "rapidly decrease down to 0".
TEST(SafetyModeDecay, isExponentialNotLinear) {
  for (int i = 1; i <= 4; ++i) {
    const scalar_t elapsed = i * kTimeConstant;
    const scalar_t expected = std::exp(-static_cast<scalar_t>(i));
    if (expected < 0.02) continue;
    EXPECT_NEAR(Controller::safetyDecayFactor(elapsed, kTimeConstant), expected, 1e-12);
  }
  // One time constant in, an exponential has shed 63% where a linear ramp over the same four-time-constant window
  // would still be holding 75%.
  EXPECT_LT(Controller::safetyDecayFactor(kTimeConstant, kTimeConstant), 0.5);
}

/// Monotonically non-increasing: the torque must never climb back up part-way through a safety stop.
TEST(SafetyModeDecay, decreasesMonotonically) {
  scalar_t previous = Controller::safetyDecayFactor(0.0, kTimeConstant);
  for (int i = 1; i <= 100; ++i) {
    const scalar_t alpha = Controller::safetyDecayFactor(0.05 * i, kTimeConstant);
    EXPECT_LE(alpha, previous);
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, 1.0);
    previous = alpha;
  }
}

/// A pure exponential never reaches zero, so the factor is snapped to zero below the cutoff. Without this the mode
/// would leave a small residual stiffness on the joints for ever instead of ending in true zero torque.
TEST(SafetyModeDecay, reachesExactlyZeroInFiniteTime) {
  EXPECT_GT(Controller::safetyDecayFactor(3.0 * kTimeConstant, kTimeConstant), 0.0);
  EXPECT_DOUBLE_EQ(Controller::safetyDecayFactor(4.0 * kTimeConstant, kTimeConstant), 0.0);
  EXPECT_DOUBLE_EQ(Controller::safetyDecayFactor(100.0, kTimeConstant), 0.0);
}

/// Time is read from the MPC observation, which can step backwards across an MPC reset. A negative elapsed must not
/// produce a factor above one, which would amplify the gains rather than decay them.
TEST(SafetyModeDecay, clampsNegativeElapsedTime) {
  EXPECT_DOUBLE_EQ(Controller::safetyDecayFactor(-1.0, kTimeConstant), 1.0);
}

/// A task file setting the time constant to zero must not produce a division by zero.
TEST(SafetyModeDecay, toleratesAZeroTimeConstant) {
  const scalar_t alpha = Controller::safetyDecayFactor(0.01, 0.0);
  EXPECT_TRUE(std::isfinite(alpha));
  EXPECT_DOUBLE_EQ(alpha, 0.0);
}

/// The commanded torque is alpha * (kp * (q_hold - q) - kd * qd): it opposes the velocity at every instant of the
/// decay, so the mode damps the joints on the way down rather than injecting energy.
TEST(SafetyModeDecay, dampingOpposesVelocityThroughoutTheDecay) {
  const scalar_t kp = 100.0;
  const scalar_t kd = 5.0;
  const scalar_t velocity = 2.0;
  for (int i = 0; i <= 40; ++i) {
    const scalar_t alpha = Controller::safetyDecayFactor(0.05 * i, kTimeConstant);
    // At the held posture the position error vanishes and only the damping term survives.
    const scalar_t torque = alpha * (kp * 0.0 - kd * velocity);
    EXPECT_LE(torque, 0.0) << "damping must oppose a positive velocity";
    EXPECT_LE(std::abs(torque), kd * velocity);
  }
}

}  // namespace
}  // namespace ocs2::humanoid
