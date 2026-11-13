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

// The geometry of a rolling contact (contact/RollingContact.h): the contact is
// the wheel's lowest point, and the wheel's material there is at rest exactly
// when the wheel rolls without slipping.

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/automatic_differentiation/Types.h"

#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/contact/RollingContact.h"

namespace ocs2::legged {
namespace {

constexpr scalar_t kWheelRadius = 0.127;
constexpr scalar_t kTolerance = 1.0e-12;

TEST(RollingContact, TheContactIsTheWheelsLowestPointOnFlatGround) {
  const vector3_t normalSurfaceEwrtWorld = normalTerrainEwrtWorld<scalar_t>();
  EXPECT_EQ(normalSurfaceEwrtWorld, vector3_t(0.0, 0.0, 1.0));
  EXPECT_EQ(posContactRtWheelEwrtWorld(kWheelRadius, normalSurfaceEwrtWorld),
            vector3_t(0.0, 0.0, -kWheelRadius));
  const vector3_t posWheelRtWorldEwrtWorld(1.0, -2.0, kWheelRadius);
  EXPECT_TRUE(posContactRtWorldEwrtWorld(posWheelRtWorldEwrtWorld, kWheelRadius,
                                         normalSurfaceEwrtWorld)
                  .isApprox(vector3_t(1.0, -2.0, 0.0), kTolerance));
}

TEST(RollingContact, OnATiltedGroundTheContactIsAlongItsNormal) {
  const vector3_t normalSurfaceEwrtWorld =
      vector3_t(0.0, 0.6, 0.8);  // a unit normal
  EXPECT_TRUE(
      posContactRtWheelEwrtWorld(kWheelRadius, normalSurfaceEwrtWorld)
          .isApprox(-kWheelRadius * normalSurfaceEwrtWorld, kTolerance));
}

TEST(RollingContact, AWheelRollingForwardLeavesItsMaterialAtRestAtTheGround) {
  // A wheel turning about +y at w moves its lowest point by w x (-r z) =
  // (-w r, 0, 0): it rolls without slipping when the center moves forward at
  // w r.
  constexpr scalar_t kForwardSpeed = 0.5;
  const vector3_t angvelWheelRtWorldEwrtWorld(0.0, kForwardSpeed / kWheelRadius,
                                              0.0);
  const vector3_t linvelWheelRtWorldEwrtWorld(kForwardSpeed, 0.0, 0.0);
  EXPECT_LT(linvelContactRtWorldEwrtWorld(
                linvelWheelRtWorldEwrtWorld, angvelWheelRtWorldEwrtWorld,
                kWheelRadius, normalTerrainEwrtWorld<scalar_t>())
                .norm(),
            kTolerance);
}

TEST(RollingContact, AWheelSpinningInPlaceSlipsBackwardAtTheGround) {
  const vector3_t slipOfSpin = linvelContactRtWorldEwrtWorld(
      vector3_t(vector3_t::Zero()), vector3_t(0.0, 1.0, 0.0), kWheelRadius,
      normalTerrainEwrtWorld<scalar_t>());
  EXPECT_TRUE(
      slipOfSpin.isApprox(vector3_t(-kWheelRadius, 0.0, 0.0), kTolerance));
  // A wheel sliding sideways without turning slips at the center's speed.
  const vector3_t slipOfSlide = linvelContactRtWorldEwrtWorld(
      vector3_t(0.0, 0.3, 0.0), vector3_t(vector3_t::Zero()), kWheelRadius,
      normalTerrainEwrtWorld<scalar_t>());
  EXPECT_TRUE(slipOfSlide.isApprox(vector3_t(0.0, 0.3, 0.0), kTolerance));
}

TEST(RollingContact, AnyRollingTwistOfAWheelOnItsRimLeavesNoSlip) {
  // Rolling and turning at once (a wheel of a robot turning in place):
  // the center moves at w x (r n), whatever w is.
  const vector3_t normalSurfaceEwrtWorld = normalTerrainEwrtWorld<scalar_t>();
  const vector3_t angvelWheelRtWorldEwrtWorld(0.4, -3.0, 1.2);
  const vector3_t linvelWheelRtWorldEwrtWorld =
      angvelWheelRtWorldEwrtWorld.cross(kWheelRadius * normalSurfaceEwrtWorld);
  EXPECT_LT(linvelContactRtWorldEwrtWorld(linvelWheelRtWorldEwrtWorld,
                                          angvelWheelRtWorldEwrtWorld,
                                          kWheelRadius, normalSurfaceEwrtWorld)
                .norm(),
            kTolerance);
}

TEST(RollingContact, TheContactForceTurnsTheWheelByTheRadiusTimesItsTraction) {
  // A forward traction f_x and a normal force f_z at the lowest point of a
  // wheel whose axle is y: the moment about the center is (-r z) x f =
  // (0, -r f_x, 0), the axle torque, and the normal force passes through the
  // center.
  constexpr scalar_t kTraction = 12.0;
  constexpr scalar_t kForceContactEwrtWorldZ = 100.0;
  EXPECT_TRUE(
      torqueWheelEwrtWorld(vector3_t(kTraction, 0.0, kForceContactEwrtWorldZ),
                           kWheelRadius, normalTerrainEwrtWorld<scalar_t>())
          .isApprox(vector3_t(0.0, -kWheelRadius * kTraction, 0.0),
                    kTolerance));
  // A lateral force turns the wheel about the heading axis, not the axle.
  EXPECT_TRUE(
      torqueWheelEwrtWorld(vector3_t(0.0, kTraction, 0.0), kWheelRadius,
                           normalTerrainEwrtWorld<scalar_t>())
          .isApprox(vector3_t(kWheelRadius * kTraction, 0.0, 0.0), kTolerance));
}

TEST(RollingContact, TheHelpersEvaluateOnTheAutomaticDifferentiationScalar) {
  const ad_vector3_t adPosContactRtWorldEwrtWorld =
      posContactRtWorldEwrtWorld<ad_scalar_t>(
          ad_vector3_t(ad_scalar_t(1.0), ad_scalar_t(0.0),
                       ad_scalar_t(kWheelRadius)),
          ad_scalar_t(kWheelRadius), normalTerrainEwrtWorld<ad_scalar_t>());
  EXPECT_EQ(CppAD::Value(adPosContactRtWorldEwrtWorld.z()).getValue(), 0.0);
}

}  // namespace
}  // namespace ocs2::legged
