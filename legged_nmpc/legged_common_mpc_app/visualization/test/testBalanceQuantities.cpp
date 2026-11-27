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

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>

#include "gtest/gtest.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/ContactArray.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc_app/visualization/BalanceQuantities.h"

namespace ocs2::legged::visualization {
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
    const ContactArray<vector3_t>& posContactsRtWorldEwrtWorld,
    const ContactArray<vector6_t>& wrenchContactsEwrtWorld) {
  vector3_t torqueLocalEwrtWorld = vector3_t::Zero();
  for (size_t contact = 0; contact < posContactsRtWorldEwrtWorld.size();
       ++contact) {
    torqueLocalEwrtWorld +=
        (posContactsRtWorldEwrtWorld[contact] - posLocalRtWorldEwrtWorld)
            .cross(
                vector3_t(wrenchContactsEwrtWorld[contact].head<kForceDim>())) +
        vector3_t(wrenchContactsEwrtWorld[contact].tail<kTorqueDim>());
  }
  return torqueLocalEwrtWorld;
}

ContactArray<vector3_t> randomPosContactsRtWorldEwrtWorld(
    std::mt19937& generator, size_t numContacts) {
  ContactArray<vector3_t> posContactsRtWorldEwrtWorld(numContacts);
  for (vector3_t& posContactRtWorldEwrtWorld : posContactsRtWorldEwrtWorld) {
    posContactRtWorldEwrtWorld = randomVector(generator, /*scale=*/0.5);
  }
  return posContactsRtWorldEwrtWorld;
}

ContactArray<vector6_t> randomWrenchContactsEwrtWorld(std::mt19937& generator,
                                                      size_t numContacts) {
  std::uniform_real_distribution<scalar_t> normal(50.0, 500.0);
  ContactArray<vector6_t> wrenchContactsEwrtWorld(numContacts);
  for (vector6_t& wrenchContactEwrtWorld : wrenchContactsEwrtWorld) {
    wrenchContactEwrtWorld.head<kForceDim>() =
        randomVector(generator, /*scale=*/100.0);
    wrenchContactEwrtWorld[kWrenchForceZIndex] = normal(generator);
    wrenchContactEwrtWorld.tail<kTorqueDim>() =
        randomVector(generator, /*scale=*/20.0);
  }
  return wrenchContactsEwrtWorld;
}

// The contact counts of the property tests: a biped's two feet, a
// quadruped's four feet and a set of six contacts.
constexpr std::array<size_t, 3> kContactCounts = {2, 4, 6};

// The defining property: the net contact moment about the ZMP has no horizontal
// component, for contacts at any heights.
TEST(ZeroMomentPoint, TheNetMomentAboutItIsVertical) {
  std::mt19937 generator(/*seed=*/3);
  for (const size_t numContacts : kContactCounts) {
    for (int trial = 0; trial < kTrials; ++trial) {
      const ContactArray<vector3_t> posContactsRtWorldEwrtWorld =
          randomPosContactsRtWorldEwrtWorld(generator, numContacts);
      const ContactArray<vector6_t> wrenchContactsEwrtWorld =
          randomWrenchContactsEwrtWorld(generator, numContacts);
      const std::optional<vector3_t> posZmpRtWorldEwrtWorld =
          computePosZmpRtWorldEwrtWorld(posContactsRtWorldEwrtWorld,
                                        wrenchContactsEwrtWorld,
                                        /*minForceWorldEwrtWorldZ=*/1.0);
      if (!posZmpRtWorldEwrtWorld.has_value()) {
        FAIL() << "loaded contacts have a ZMP";
      }
      const vector3_t torqueZmpEwrtWorld = computeTorqueLocalEwrtWorld(
          *posZmpRtWorldEwrtWorld, posContactsRtWorldEwrtWorld,
          wrenchContactsEwrtWorld);
      EXPECT_NEAR(torqueZmpEwrtWorld.x(), 0.0, 1.0e-8)
          << numContacts << " contacts, trial " << trial;
      EXPECT_NEAR(torqueZmpEwrtWorld.y(), 0.0, 1.0e-8)
          << numContacts << " contacts, trial " << trial;
      // The plane is the normal-force-weighted contact height.
      scalar_t weightedSumPosContactsRtWorldEwrtWorldZ = 0.0;
      scalar_t forceWorldEwrtWorldZ = 0.0;
      for (size_t contact = 0; contact < numContacts; ++contact) {
        weightedSumPosContactsRtWorldEwrtWorldZ +=
            wrenchContactsEwrtWorld[contact][kWrenchForceZIndex] *
            posContactsRtWorldEwrtWorld[contact].z();
        forceWorldEwrtWorldZ +=
            wrenchContactsEwrtWorld[contact][kWrenchForceZIndex];
      }
      EXPECT_NEAR(
          posZmpRtWorldEwrtWorld->z(),
          weightedSumPosContactsRtWorldEwrtWorldZ / forceWorldEwrtWorldZ,
          kTolerance)
          << numContacts << " contacts, trial " << trial;
    }
  }
}

// On flat ground with one contact loaded, the ZMP is that contact's center of
// pressure, r + (-tau_y, tau_x) / f_z, whichever contact of the set it is.
TEST(ZeroMomentPoint, OfOneContactOnFlatGroundIsItsCenterOfPressure) {
  std::mt19937 generator(/*seed=*/4);
  for (const size_t numContacts : kContactCounts) {
    for (int trial = 0; trial < kTrials; ++trial) {
      const size_t loaded = static_cast<size_t>(trial) % numContacts;
      ContactArray<vector3_t> posContactsRtWorldEwrtWorld =
          randomPosContactsRtWorldEwrtWorld(generator, numContacts);
      for (vector3_t& posContactRtWorldEwrtWorld :
           posContactsRtWorldEwrtWorld) {
        posContactRtWorldEwrtWorld.z() =
            posContactsRtWorldEwrtWorld[loaded].z();
      }
      ContactArray<vector6_t> wrenchContactsEwrtWorld =
          randomWrenchContactsEwrtWorld(generator, numContacts);
      for (size_t contact = 0; contact < numContacts; ++contact) {
        if (contact != loaded) {
          wrenchContactsEwrtWorld[contact].setZero();
        }
      }
      const std::optional<vector3_t> posZmpRtWorldEwrtWorld =
          computePosZmpRtWorldEwrtWorld(posContactsRtWorldEwrtWorld,
                                        wrenchContactsEwrtWorld,
                                        /*minForceWorldEwrtWorldZ=*/1.0);
      if (!posZmpRtWorldEwrtWorld.has_value()) {
        FAIL() << "a loaded contact has a ZMP";
      }
      const vector6_t& wrenchContactEwrtWorld = wrenchContactsEwrtWorld[loaded];
      const scalar_t forceContactEwrtWorldZ =
          wrenchContactEwrtWorld[kWrenchForceZIndex];
      const vector3_t posCopRtWorldEwrtWorld =
          posContactsRtWorldEwrtWorld[loaded] +
          vector3_t(-wrenchContactEwrtWorld[kWrenchTorqueYIndex] /
                        forceContactEwrtWorldZ,
                    wrenchContactEwrtWorld[kWrenchTorqueXIndex] /
                        forceContactEwrtWorldZ,
                    0.0);
      EXPECT_LT((*posZmpRtWorldEwrtWorld - posCopRtWorldEwrtWorld).norm(),
                kTolerance)
          << numContacts << " contacts, trial " << trial;
    }
  }
}

// Moving every contact by a horizontal offset moves the ZMP by it, and the
// wrenches' torques are unchanged.
TEST(ZeroMomentPoint, MovesWithTheContacts) {
  std::mt19937 generator(/*seed=*/5);
  for (const size_t numContacts : kContactCounts) {
    for (int trial = 0; trial < kTrials; ++trial) {
      ContactArray<vector3_t> posContactsRtWorldEwrtWorld =
          randomPosContactsRtWorldEwrtWorld(generator, numContacts);
      const ContactArray<vector6_t> wrenchContactsEwrtWorld =
          randomWrenchContactsEwrtWorld(generator, numContacts);
      const vector3_t errPosContactRtWorldEwrtWorld(
          randomVector(generator, /*scale=*/2.0).x(),
          randomVector(generator, /*scale=*/2.0).y(), 0.0);
      const std::optional<vector3_t> posZmpRtWorldEwrtWorld =
          computePosZmpRtWorldEwrtWorld(posContactsRtWorldEwrtWorld,
                                        wrenchContactsEwrtWorld,
                                        /*minForceWorldEwrtWorldZ=*/1.0);
      for (vector3_t& posContactRtWorldEwrtWorld :
           posContactsRtWorldEwrtWorld) {
        posContactRtWorldEwrtWorld += errPosContactRtWorldEwrtWorld;
      }
      const std::optional<vector3_t> movedPosZmpRtWorldEwrtWorld =
          computePosZmpRtWorldEwrtWorld(posContactsRtWorldEwrtWorld,
                                        wrenchContactsEwrtWorld,
                                        /*minForceWorldEwrtWorldZ=*/1.0);
      if (!(posZmpRtWorldEwrtWorld.has_value() &&
            movedPosZmpRtWorldEwrtWorld.has_value())) {
        FAIL() << "loaded contacts have a ZMP";
      }
      EXPECT_LT((*movedPosZmpRtWorldEwrtWorld -
                 (*posZmpRtWorldEwrtWorld + errPosContactRtWorldEwrtWorld))
                    .norm(),
                1.0e-8)
          << numContacts << " contacts, trial " << trial;
    }
  }
}

// Four point contacts at the corners of a rectangle, loaded equally, have
// their ZMP at its center: a quadruped standing on four feet.
TEST(ZeroMomentPoint, OfFourEquallyLoadedPointContactsIsTheirCentroid) {
  const ContactArray<vector3_t> posContactsRtWorldEwrtWorld = [] {
    ContactArray<vector3_t> posCornersRtWorldEwrtWorld(/*numContacts=*/4);
    posCornersRtWorldEwrtWorld[0] = vector3_t(0.3, 0.2, 0.0);
    posCornersRtWorldEwrtWorld[1] = vector3_t(0.3, -0.2, 0.0);
    posCornersRtWorldEwrtWorld[2] = vector3_t(-0.3, 0.2, 0.0);
    posCornersRtWorldEwrtWorld[3] = vector3_t(-0.3, -0.2, 0.0);
    return posCornersRtWorldEwrtWorld;
  }();
  vector6_t wrenchContactEwrtWorld = vector6_t::Zero();
  wrenchContactEwrtWorld[kWrenchForceZIndex] = 100.0;
  const ContactArray<vector6_t> wrenchContactsEwrtWorld(
      /*numContacts=*/4, wrenchContactEwrtWorld);
  const std::optional<vector3_t> posZmpRtWorldEwrtWorld =
      computePosZmpRtWorldEwrtWorld(posContactsRtWorldEwrtWorld,
                                    wrenchContactsEwrtWorld,
                                    /*minForceWorldEwrtWorldZ=*/1.0);
  if (!posZmpRtWorldEwrtWorld.has_value()) {
    FAIL() << "loaded contacts have a ZMP";
  }
  EXPECT_LT(posZmpRtWorldEwrtWorld->norm(), kTolerance);
}

TEST(ZeroMomentPoint, IsUndefinedWithoutNormalForce) {
  ContactArray<vector3_t> posContactsRtWorldEwrtWorld(/*numContacts=*/2);
  posContactsRtWorldEwrtWorld[0] = vector3_t(0.0, 0.1, 0.0);
  posContactsRtWorldEwrtWorld[1] = vector3_t(0.0, -0.1, 0.0);
  ContactArray<vector6_t> wrenchContactsEwrtWorld(/*numContacts=*/2,
                                                  vector6_t::Zero());
  EXPECT_FALSE(computePosZmpRtWorldEwrtWorld(posContactsRtWorldEwrtWorld,
                                             wrenchContactsEwrtWorld,
                                             /*minForceWorldEwrtWorldZ=*/1.0)
                   .has_value());
  wrenchContactsEwrtWorld[0][kWrenchForceZIndex] = 0.5;
  EXPECT_FALSE(computePosZmpRtWorldEwrtWorld(posContactsRtWorldEwrtWorld,
                                             wrenchContactsEwrtWorld,
                                             /*minForceWorldEwrtWorldZ=*/1.0)
                   .has_value());
  wrenchContactsEwrtWorld[1][kWrenchForceZIndex] = 0.6;
  EXPECT_TRUE(computePosZmpRtWorldEwrtWorld(posContactsRtWorldEwrtWorld,
                                            wrenchContactsEwrtWorld,
                                            /*minForceWorldEwrtWorldZ=*/1.0)
                  .has_value());
}

// An empty contact set carries no force and has no ZMP.
TEST(ZeroMomentPoint, IsUndefinedForAnEmptyContactSet) {
  EXPECT_FALSE(computePosZmpRtWorldEwrtWorld(ContactArray<vector3_t>(),
                                             ContactArray<vector6_t>(),
                                             /*minForceWorldEwrtWorldZ=*/1.0)
                   .has_value());
}

}  // namespace
}  // namespace ocs2::legged::visualization
