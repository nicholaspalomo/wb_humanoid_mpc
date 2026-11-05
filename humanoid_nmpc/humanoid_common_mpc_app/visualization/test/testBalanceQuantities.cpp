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
#include <cstddef>
#include <limits>
#include <optional>
#include <random>

#include "gtest/gtest.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/visualization/BalanceQuantities.h"

namespace ocs2::humanoid::visualization {
namespace {

constexpr scalar_t kTolerance = 1.0e-9;
constexpr int kTrials = 200;
constexpr scalar_t kGravity = 9.81;

vector3_t randomVector(std::mt19937& generator, scalar_t scale) {
  std::uniform_real_distribution<scalar_t> distribution(-scale, scale);
  return vector3_t(distribution(generator), distribution(generator),
                   distribution(generator));
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
    // omega^2 z0 = g: the pendulum's defining relation, not the formula
    // restated.
    EXPECT_NEAR(*omega * *omega * z0, kGravity, kTolerance);
    EXPECT_GT(*omega, 0.0);
  }
}

TEST(PendulumFrequency, IsUndefinedForANonPositiveOrNonFiniteInput) {
  const scalar_t nan = std::numeric_limits<scalar_t>::quiet_NaN();
  const scalar_t infinity = std::numeric_limits<scalar_t>::infinity();
  EXPECT_FALSE(
      pendulumFrequency(kGravity, /*proxyPosComRtFeetmeanEwrtWorldZ=*/0.0)
          .has_value());
  EXPECT_FALSE(
      pendulumFrequency(kGravity, /*proxyPosComRtFeetmeanEwrtWorldZ=*/-0.8)
          .has_value());
  EXPECT_FALSE(pendulumFrequency(/*gravityMagnitude=*/0.0,
                                 /*proxyPosComRtFeetmeanEwrtWorldZ=*/0.8)
                   .has_value());
  EXPECT_FALSE(pendulumFrequency(kGravity, nan).has_value());
  EXPECT_FALSE(pendulumFrequency(nan, /*proxyPosComRtFeetmeanEwrtWorldZ=*/0.8)
                   .has_value());
  EXPECT_FALSE(pendulumFrequency(kGravity, infinity).has_value());
}

// On the linear inverted pendulum c'' = omega^2 (c - p) with a fixed ZMP p,
// c(t) = p + A e^(omega t) + B e^(-omega t): the DCM diverges alone, xi(t) = p
// + 2 A e^(omega t), and its convergent part B drops out.
TEST(DivergentComponentOfMotion, IsThePendulumsDivergentMode) {
  std::mt19937 generator(/*seed=*/2);
  std::uniform_real_distribution<scalar_t> time(0.0, 0.5);
  std::uniform_real_distribution<scalar_t> height(0.4, 1.2);
  for (int trial = 0; trial < kTrials; ++trial) {
    const std::optional<scalar_t> frequency =
        pendulumFrequency(kGravity, height(generator));
    if (!frequency.has_value()) {
      FAIL() << "a positive height has a pendulum frequency";
    }
    const scalar_t omega = *frequency;
    const vector3_t posZmpRtWorldEwrtWorld =
        randomVector(generator, /*scale=*/1.0);
    const vector3_t divergentPosComRtZmpEwrtWorld =
        randomVector(generator, /*scale=*/0.1);
    const vector3_t convergentPosComRtZmpEwrtWorld =
        randomVector(generator, /*scale=*/0.1);
    const scalar_t t = time(generator);
    const vector3_t posComRtWorldEwrtWorld =
        posZmpRtWorldEwrtWorld +
        divergentPosComRtZmpEwrtWorld * std::exp(omega * t) +
        convergentPosComRtZmpEwrtWorld * std::exp(-omega * t);
    const vector3_t linvelComRtWorldEwrtWorld =
        omega * (divergentPosComRtZmpEwrtWorld * std::exp(omega * t) -
                 convergentPosComRtZmpEwrtWorld * std::exp(-omega * t));
    const vector2_t expected =
        (posZmpRtWorldEwrtWorld +
         2.0 * divergentPosComRtZmpEwrtWorld * std::exp(omega * t))
            .head<kHorizontalPositionDim>();
    EXPECT_LT((computePosDcmRtWorldEwrtWorldXy(
                   posComRtWorldEwrtWorld, linvelComRtWorldEwrtWorld, omega) -
               expected)
                  .norm(),
              kTolerance)
        << trial;
  }
}

TEST(DivergentComponentOfMotion, IsTheComAtRestAndIgnoresTheVerticalAxis) {
  const vector3_t posComRtWorldEwrtWorld(0.1, -0.2, 0.9);
  EXPECT_LT((computePosDcmRtWorldEwrtWorldXy(posComRtWorldEwrtWorld,
                                             vector3_t(0.0, 0.0, 0.7),
                                             /*omega=*/3.0) -
             posComRtWorldEwrtWorld.head<kHorizontalPositionDim>())
                .norm(),
            kTolerance);
}

/** The net contact moment about the point `posLocalRtWorldEwrtWorld`: sum_i
 * (r_i - point) x f_i + tau_i. */
vector3_t computeTorqueLocalEwrtWorld(
    const vector3_t& posLocalRtWorldEwrtWorld,
    const feet_array_t<vector3_t>& posFeetRtWorldEwrtWorld,
    const feet_array_t<vector6_t>& wrenchFeetEwrtWorld) {
  vector3_t torqueLocalEwrtWorld = vector3_t::Zero();
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    torqueLocalEwrtWorld +=
        (posFeetRtWorldEwrtWorld[contact] - posLocalRtWorldEwrtWorld)
            .cross(vector3_t(wrenchFeetEwrtWorld[contact].head<kForceDim>())) +
        vector3_t(wrenchFeetEwrtWorld[contact].tail<kTorqueDim>());
  }
  return torqueLocalEwrtWorld;
}

feet_array_t<vector6_t> randomWrenchFeetEwrtWorld(std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> normal(50.0, 500.0);
  feet_array_t<vector6_t> wrenchFeetEwrtWorld;
  for (vector6_t& wrenchFootEwrtWorld : wrenchFeetEwrtWorld) {
    wrenchFootEwrtWorld.head<kForceDim>() =
        randomVector(generator, /*scale=*/100.0);
    wrenchFootEwrtWorld[kWrenchForceZIndex] = normal(generator);
    wrenchFootEwrtWorld.tail<kTorqueDim>() =
        randomVector(generator, /*scale=*/20.0);
  }
  return wrenchFeetEwrtWorld;
}

// The defining property: the net contact moment about the ZMP has no horizontal
// component, for feet at any heights.
TEST(ZeroMomentPoint, TheNetMomentAboutItIsVertical) {
  std::mt19937 generator(/*seed=*/3);
  for (int trial = 0; trial < kTrials; ++trial) {
    const feet_array_t<vector3_t> posFeetRtWorldEwrtWorld = {
        randomVector(generator, /*scale=*/0.5),
        randomVector(generator, /*scale=*/0.5)};
    const feet_array_t<vector6_t> wrenchFeetEwrtWorld =
        randomWrenchFeetEwrtWorld(generator);
    const std::optional<vector3_t> posZmpRtWorldEwrtWorld =
        computePosZmpRtWorldEwrtWorld(posFeetRtWorldEwrtWorld,
                                      wrenchFeetEwrtWorld,
                                      /*minForceWorldEwrtWorldZ=*/1.0);
    if (!posZmpRtWorldEwrtWorld.has_value()) {
      FAIL() << "loaded feet have a ZMP";
    }
    const vector3_t torqueZmpEwrtWorld = computeTorqueLocalEwrtWorld(
        *posZmpRtWorldEwrtWorld, posFeetRtWorldEwrtWorld, wrenchFeetEwrtWorld);
    EXPECT_NEAR(torqueZmpEwrtWorld.x(), 0.0, 1.0e-8) << trial;
    EXPECT_NEAR(torqueZmpEwrtWorld.y(), 0.0, 1.0e-8) << trial;
    // The plane is the normal-force-weighted contact height.
    const scalar_t height = (wrenchFeetEwrtWorld[0][kWrenchForceZIndex] *
                                 posFeetRtWorldEwrtWorld[0].z() +
                             wrenchFeetEwrtWorld[1][kWrenchForceZIndex] *
                                 posFeetRtWorldEwrtWorld[1].z()) /
                            (wrenchFeetEwrtWorld[0][kWrenchForceZIndex] +
                             wrenchFeetEwrtWorld[1][kWrenchForceZIndex]);
    EXPECT_NEAR(posZmpRtWorldEwrtWorld->z(), height, kTolerance) << trial;
  }
}

// On flat ground with one foot loaded, the ZMP is that foot's center of
// pressure, r + (-tau_y, tau_x) / f_z.
TEST(ZeroMomentPoint, OfOneFootOnFlatGroundIsItsCenterOfPressure) {
  std::mt19937 generator(/*seed=*/4);
  for (int trial = 0; trial < kTrials; ++trial) {
    feet_array_t<vector3_t> posFeetRtWorldEwrtWorld = {
        randomVector(generator, /*scale=*/0.5),
        randomVector(generator, /*scale=*/0.5)};
    posFeetRtWorldEwrtWorld[1].z() = posFeetRtWorldEwrtWorld[0].z();
    feet_array_t<vector6_t> wrenchFeetEwrtWorld =
        randomWrenchFeetEwrtWorld(generator);
    wrenchFeetEwrtWorld[1].setZero();
    const std::optional<vector3_t> posZmpRtWorldEwrtWorld =
        computePosZmpRtWorldEwrtWorld(posFeetRtWorldEwrtWorld,
                                      wrenchFeetEwrtWorld,
                                      /*minForceWorldEwrtWorldZ=*/1.0);
    if (!posZmpRtWorldEwrtWorld.has_value()) {
      FAIL() << "loaded feet have a ZMP";
    }
    const scalar_t forceFootEwrtWorldZ =
        wrenchFeetEwrtWorld[0][kWrenchForceZIndex];
    const vector3_t posCopRtWorldEwrtWorld =
        posFeetRtWorldEwrtWorld[0] +
        vector3_t(
            -wrenchFeetEwrtWorld[0][kWrenchTorqueYIndex] / forceFootEwrtWorldZ,
            wrenchFeetEwrtWorld[0][kWrenchTorqueXIndex] / forceFootEwrtWorldZ,
            0.0);
    EXPECT_LT((*posZmpRtWorldEwrtWorld - posCopRtWorldEwrtWorld).norm(),
              kTolerance)
        << trial;
  }
}

// Moving every contact by a horizontal offset moves the ZMP by it, and the
// wrenches' torques are unchanged.
TEST(ZeroMomentPoint, MovesWithTheContacts) {
  std::mt19937 generator(/*seed=*/5);
  for (int trial = 0; trial < kTrials; ++trial) {
    feet_array_t<vector3_t> posFeetRtWorldEwrtWorld = {
        randomVector(generator, /*scale=*/0.5),
        randomVector(generator, /*scale=*/0.5)};
    const feet_array_t<vector6_t> wrenchFeetEwrtWorld =
        randomWrenchFeetEwrtWorld(generator);
    const vector3_t errPosFootRtWorldEwrtWorld(
        randomVector(generator, /*scale=*/2.0).x(),
        randomVector(generator, /*scale=*/2.0).y(), 0.0);
    const std::optional<vector3_t> posZmpRtWorldEwrtWorld =
        computePosZmpRtWorldEwrtWorld(posFeetRtWorldEwrtWorld,
                                      wrenchFeetEwrtWorld,
                                      /*minForceWorldEwrtWorldZ=*/1.0);
    for (vector3_t& posFootRtWorldEwrtWorld : posFeetRtWorldEwrtWorld) {
      posFootRtWorldEwrtWorld += errPosFootRtWorldEwrtWorld;
    }
    const std::optional<vector3_t> movedPosZmpRtWorldEwrtWorld =
        computePosZmpRtWorldEwrtWorld(posFeetRtWorldEwrtWorld,
                                      wrenchFeetEwrtWorld,
                                      /*minForceWorldEwrtWorldZ=*/1.0);
    if (!(posZmpRtWorldEwrtWorld.has_value() &&
          movedPosZmpRtWorldEwrtWorld.has_value())) {
      FAIL() << "loaded feet have a ZMP";
    }
    EXPECT_LT((*movedPosZmpRtWorldEwrtWorld -
               (*posZmpRtWorldEwrtWorld + errPosFootRtWorldEwrtWorld))
                  .norm(),
              1.0e-8)
        << trial;
  }
}

TEST(ZeroMomentPoint, IsUndefinedWithoutNormalForce) {
  const feet_array_t<vector3_t> posFeetRtWorldEwrtWorld = {
      vector3_t(0.0, 0.1, 0.0), vector3_t(0.0, -0.1, 0.0)};
  feet_array_t<vector6_t> wrenchFeetEwrtWorld = {vector6_t::Zero(),
                                                 vector6_t::Zero()};
  EXPECT_FALSE(computePosZmpRtWorldEwrtWorld(posFeetRtWorldEwrtWorld,
                                             wrenchFeetEwrtWorld,
                                             /*minForceWorldEwrtWorldZ=*/1.0)
                   .has_value());
  wrenchFeetEwrtWorld[0][kWrenchForceZIndex] = 0.5;
  EXPECT_FALSE(computePosZmpRtWorldEwrtWorld(posFeetRtWorldEwrtWorld,
                                             wrenchFeetEwrtWorld,
                                             /*minForceWorldEwrtWorldZ=*/1.0)
                   .has_value());
  wrenchFeetEwrtWorld[1][kWrenchForceZIndex] = 0.6;
  EXPECT_TRUE(computePosZmpRtWorldEwrtWorld(posFeetRtWorldEwrtWorld,
                                            wrenchFeetEwrtWorld,
                                            /*minForceWorldEwrtWorldZ=*/1.0)
                  .has_value());
}

}  // namespace
}  // namespace ocs2::humanoid::visualization
