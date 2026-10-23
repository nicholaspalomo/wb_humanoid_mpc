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

#include <cmath>
#include <limits>
#include <optional>
#include <random>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/visualization/BalanceQuantities.h"

namespace ocs2::humanoid::visualization {
namespace {

constexpr scalar_t kTolerance = 1.0e-9;
constexpr int kTrials = 200;
constexpr scalar_t kGravity = 9.81;

vector3_t randomVector(std::mt19937& generator, scalar_t scale) {
  std::uniform_real_distribution<scalar_t> distribution(-scale, scale);
  return vector3_t(distribution(generator), distribution(generator), distribution(generator));
}

TEST(PendulumFrequency, IsTheSquareRootOfGravityOverHeight) {
  std::mt19937 generator(/*seed=*/1);
  std::uniform_real_distribution<scalar_t> height(0.2, 1.5);
  for (int trial = 0; trial < kTrials; ++trial) {
    const scalar_t z0 = height(generator);
    const std::optional<scalar_t> omega = pendulumFrequency(kGravity, z0);
    if (!omega.has_value()) {
      FAIL() << "a positive height has a pendulum frequency";
    }
    // omega^2 z0 = g: the pendulum's defining relation, not the formula restated.
    EXPECT_NEAR(*omega * *omega * z0, kGravity, kTolerance);
    EXPECT_GT(*omega, 0.0);
  }
}

TEST(PendulumFrequency, IsUndefinedForANonPositiveOrNonFiniteInput) {
  const scalar_t nan = std::numeric_limits<scalar_t>::quiet_NaN();
  const scalar_t infinity = std::numeric_limits<scalar_t>::infinity();
  EXPECT_FALSE(pendulumFrequency(kGravity, /*pendulumHeight=*/0.0).has_value());
  EXPECT_FALSE(pendulumFrequency(kGravity, /*pendulumHeight=*/-0.8).has_value());
  EXPECT_FALSE(pendulumFrequency(/*gravity=*/0.0, /*pendulumHeight=*/0.8).has_value());
  EXPECT_FALSE(pendulumFrequency(kGravity, nan).has_value());
  EXPECT_FALSE(pendulumFrequency(nan, /*pendulumHeight=*/0.8).has_value());
  EXPECT_FALSE(pendulumFrequency(kGravity, infinity).has_value());
}

// On the linear inverted pendulum c'' = omega^2 (c - p) with a fixed ZMP p, c(t) = p + A e^(omega t) + B e^(-omega t):
// the DCM diverges alone, xi(t) = p + 2 A e^(omega t), and its convergent part B drops out.
TEST(DivergentComponentOfMotion, IsThePendulumsDivergentMode) {
  std::mt19937 generator(/*seed=*/2);
  std::uniform_real_distribution<scalar_t> time(0.0, 0.5);
  std::uniform_real_distribution<scalar_t> height(0.4, 1.2);
  for (int trial = 0; trial < kTrials; ++trial) {
    const std::optional<scalar_t> frequency = pendulumFrequency(kGravity, height(generator));
    if (!frequency.has_value()) {
      FAIL() << "a positive height has a pendulum frequency";
    }
    const scalar_t omega = *frequency;
    const vector3_t zmp = randomVector(generator, /*scale=*/1.0);
    const vector3_t divergent = randomVector(generator, /*scale=*/0.1);
    const vector3_t convergent = randomVector(generator, /*scale=*/0.1);
    const scalar_t t = time(generator);
    const vector3_t com = zmp + divergent * std::exp(omega * t) + convergent * std::exp(-omega * t);
    const vector3_t comVelocity = omega * (divergent * std::exp(omega * t) - convergent * std::exp(-omega * t));
    const vector2_t expected = (zmp + 2.0 * divergent * std::exp(omega * t)).head<kHorizontalPositionDim>();
    EXPECT_LT((divergentComponentOfMotion(com, comVelocity, omega) - expected).norm(), kTolerance) << trial;
  }
}

TEST(DivergentComponentOfMotion, IsTheComAtRestAndIgnoresTheVerticalAxis) {
  const vector3_t com(0.1, -0.2, 0.9);
  EXPECT_LT((divergentComponentOfMotion(com, vector3_t(0.0, 0.0, 0.7), /*omega=*/3.0) - com.head<kHorizontalPositionDim>()).norm(),
            kTolerance);
}

/** The net contact moment about `point`: sum_i (r_i - point) x f_i + tau_i. */
vector3_t momentAbout(const vector3_t& point, const feet_array_t<vector3_t>& positions, const feet_array_t<vector6_t>& wrenches) {
  vector3_t moment = vector3_t::Zero();
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    moment += (positions[contact] - point).cross(vector3_t(wrenches[contact].head<kForceDim>())) +
              vector3_t(wrenches[contact].tail<kTorqueDim>());
  }
  return moment;
}

feet_array_t<vector6_t> randomWrenches(std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> normal(50.0, 500.0);
  feet_array_t<vector6_t> wrenches;
  for (vector6_t& wrench : wrenches) {
    wrench.head<kForceDim>() = randomVector(generator, /*scale=*/100.0);
    wrench[kWrenchForceZIndex] = normal(generator);
    wrench.tail<kTorqueDim>() = randomVector(generator, /*scale=*/20.0);
  }
  return wrenches;
}

// The defining property: the net contact moment about the ZMP has no horizontal component, for feet at any heights.
TEST(ZeroMomentPoint, TheNetMomentAboutItIsVertical) {
  std::mt19937 generator(/*seed=*/3);
  for (int trial = 0; trial < kTrials; ++trial) {
    const feet_array_t<vector3_t> positions = {randomVector(generator, /*scale=*/0.5), randomVector(generator, /*scale=*/0.5)};
    const feet_array_t<vector6_t> wrenches = randomWrenches(generator);
    const std::optional<vector3_t> zmp = zeroMomentPoint(positions, wrenches, /*minNormalForce=*/1.0);
    if (!zmp.has_value()) {
      FAIL() << "loaded feet have a ZMP";
    }
    const vector3_t moment = momentAbout(*zmp, positions, wrenches);
    EXPECT_NEAR(moment.x(), 0.0, 1.0e-8) << trial;
    EXPECT_NEAR(moment.y(), 0.0, 1.0e-8) << trial;
    // The plane is the normal-force-weighted contact height.
    const scalar_t height = (wrenches[0][kWrenchForceZIndex] * positions[0].z() + wrenches[1][kWrenchForceZIndex] * positions[1].z()) /
                            (wrenches[0][kWrenchForceZIndex] + wrenches[1][kWrenchForceZIndex]);
    EXPECT_NEAR(zmp->z(), height, kTolerance) << trial;
  }
}

// On flat ground with one foot loaded, the ZMP is that foot's center of pressure, r + (-tau_y, tau_x) / f_z.
TEST(ZeroMomentPoint, OfOneFootOnFlatGroundIsItsCenterOfPressure) {
  std::mt19937 generator(/*seed=*/4);
  for (int trial = 0; trial < kTrials; ++trial) {
    feet_array_t<vector3_t> positions = {randomVector(generator, /*scale=*/0.5), randomVector(generator, /*scale=*/0.5)};
    positions[1].z() = positions[0].z();
    feet_array_t<vector6_t> wrenches = randomWrenches(generator);
    wrenches[1].setZero();
    const std::optional<vector3_t> zmp = zeroMomentPoint(positions, wrenches, /*minNormalForce=*/1.0);
    if (!zmp.has_value()) {
      FAIL() << "loaded feet have a ZMP";
    }
    const scalar_t normalForce = wrenches[0][kWrenchForceZIndex];
    const vector3_t centerOfPressure =
        positions[0] + vector3_t(-wrenches[0][kWrenchTorqueYIndex] / normalForce, wrenches[0][kWrenchTorqueXIndex] / normalForce, 0.0);
    EXPECT_LT((*zmp - centerOfPressure).norm(), kTolerance) << trial;
  }
}

// Moving every contact by a horizontal offset moves the ZMP by it, and the wrenches' torques are unchanged.
TEST(ZeroMomentPoint, MovesWithTheContacts) {
  std::mt19937 generator(/*seed=*/5);
  for (int trial = 0; trial < kTrials; ++trial) {
    feet_array_t<vector3_t> positions = {randomVector(generator, /*scale=*/0.5), randomVector(generator, /*scale=*/0.5)};
    const feet_array_t<vector6_t> wrenches = randomWrenches(generator);
    const vector3_t offset(randomVector(generator, /*scale=*/2.0).x(), randomVector(generator, /*scale=*/2.0).y(), 0.0);
    const std::optional<vector3_t> zmp = zeroMomentPoint(positions, wrenches, /*minNormalForce=*/1.0);
    for (vector3_t& position : positions) {
      position += offset;
    }
    const std::optional<vector3_t> moved = zeroMomentPoint(positions, wrenches, /*minNormalForce=*/1.0);
    if (!(zmp.has_value() && moved.has_value())) {
      FAIL() << "loaded feet have a ZMP";
    }
    EXPECT_LT((*moved - (*zmp + offset)).norm(), 1.0e-8) << trial;
  }
}

TEST(ZeroMomentPoint, IsUndefinedWithoutNormalForce) {
  const feet_array_t<vector3_t> positions = {vector3_t(0.0, 0.1, 0.0), vector3_t(0.0, -0.1, 0.0)};
  feet_array_t<vector6_t> wrenches = {vector6_t::Zero(), vector6_t::Zero()};
  EXPECT_FALSE(zeroMomentPoint(positions, wrenches, /*minNormalForce=*/1.0).has_value());
  wrenches[0][kWrenchForceZIndex] = 0.5;
  EXPECT_FALSE(zeroMomentPoint(positions, wrenches, /*minNormalForce=*/1.0).has_value());
  wrenches[1][kWrenchForceZIndex] = 0.6;
  EXPECT_TRUE(zeroMomentPoint(positions, wrenches, /*minNormalForce=*/1.0).has_value());
}

}  // namespace
}  // namespace ocs2::humanoid::visualization
