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

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/contact_swing/ContactSwingPlanner.h"
#include "legged_common_mpc/contact_swing/ContactSwingTrajectory.h"
#include "legged_nmpc/legged_common_mpc/test/support/StatusTestMacros.h"

/*
 * The generalized swing planner (ContactSwingTrajectory, ContactSwingPlanner;
 * legged_nmpc/docs/multi_contact_planning/README.md, section 7.2): a swing
 * starts and ends at rest on its two points, leaves along the normal of the
 * surface it lifts off from and arrives along the normal of the surface it
 * touches down on, and keeps the required clearance from the terrain, which a
 * step's riser forces it to raise its arc for.
 */
namespace ocs2::legged {
namespace {

constexpr scalar_t kTolerance = 1.0e-9;

/** The signed distance of `posQueryRtWorldEwrtWorld` to an axis-aligned box. */
scalar_t boxDistance(const vector3_t& posQueryRtWorldEwrtWorld,
                     const vector3_t& posBoxRtWorldEwrtWorld,
                     const vector3_t& halfExtents) {
  const vector3_t q =
      (posQueryRtWorldEwrtWorld - posBoxRtWorldEwrtWorld).cwiseAbs() -
      halfExtents;
  const scalar_t outside = q.cwiseMax(0.0).norm();
  const scalar_t inside = std::min(q.maxCoeff(), 0.0);
  return outside + inside;
}

/** The floor z = 0 and a step 0.15 m high from x = 0.2 to x = 0.7. */
scalar_t stepTerrain(const vector3_t& posQueryRtWorldEwrtWorld) {
  const scalar_t floor = posQueryRtWorldEwrtWorld.z();
  const scalar_t step =
      boxDistance(posQueryRtWorldEwrtWorld, vector3_t(0.45, 0.0, 0.075),
                  vector3_t(0.25, 1.0, 0.075));
  return std::min(floor, step);
}

/** The swing of a foot from the floor onto the step. */
ContactSwingTrajectory stepUp(const ContactSwingSettings& settings) {
  return checkedValue(ContactSwingTrajectory::Create(
      /*liftOffTime=*/1.0, /*touchDownTime=*/2.0,
      ContactSwingEndpoint{
          .posContactRtWorldEwrtWorld = vector3_t(0.0, 0.1, 0.0),
          .normalSurfaceEwrtWorld = vector3_t::UnitZ()},
      ContactSwingEndpoint{
          .posContactRtWorldEwrtWorld = vector3_t(0.4, 0.1, 0.15),
          .normalSurfaceEwrtWorld = vector3_t::UnitZ()},
      settings, &stepTerrain));
}

TEST(ContactSwingTrajectory, startsAndEndsAtRestOnItsPoints) {
  const ContactSwingTrajectory swing = stepUp(ContactSwingSettings{});
  for (const scalar_t time : {0.5, 1.0}) {
    const ContactSwingReference start = swing.reference(time);
    EXPECT_TRUE(start.posContactRtWorldEwrtWorld.isApprox(
        vector3_t(0.0, 0.1, 0.0), kTolerance));
    EXPECT_TRUE(start.linvelContactRtWorldEwrtWorld.isZero(kTolerance));
  }
  for (const scalar_t time : {2.0, 3.0}) {
    const ContactSwingReference end = swing.reference(time);
    EXPECT_TRUE(end.posContactRtWorldEwrtWorld.isApprox(
        vector3_t(0.4, 0.1, 0.15), kTolerance));
    EXPECT_TRUE(end.linvelContactRtWorldEwrtWorld.isZero(kTolerance));
  }
  EXPECT_TRUE(swing.contains(1.5));
  EXPECT_FALSE(swing.contains(2.5));
}

TEST(ContactSwingTrajectory, theVelocityIsTheDerivativeOfThePosition) {
  const ContactSwingTrajectory swing = stepUp(ContactSwingSettings{});
  constexpr scalar_t kStep = 1.0e-6;
  for (const scalar_t time : {1.1, 1.37, 1.5, 1.83}) {
    const vector3_t fdLinvelContactRtWorldEwrtWorld =
        (swing.reference(time + kStep).posContactRtWorldEwrtWorld -
         swing.reference(time - kStep).posContactRtWorldEwrtWorld) /
        (2.0 * kStep);
    EXPECT_TRUE(swing.reference(time).linvelContactRtWorldEwrtWorld.isApprox(
        fdLinvelContactRtWorldEwrtWorld, 1.0e-6))
        << time;
  }
}

TEST(ContactSwingTrajectory, leavesAndArrivesAlongTheSurfaceNormals) {
  // A hand from a table top to a wall: it leaves upwards and arrives moving
  // into the wall, against the wall's outward normal.
  const vector3_t normalSurfaceEwrtWorld(-1.0, 0.0, 0.0);
  const ContactSwingTrajectory swing =
      checkedValue(ContactSwingTrajectory::Create(
          /*liftOffTime=*/0.0, /*touchDownTime=*/1.0,
          ContactSwingEndpoint{
              .posContactRtWorldEwrtWorld = vector3_t(0.3, 0.2, 0.8),
              .normalSurfaceEwrtWorld = vector3_t::UnitZ()},
          ContactSwingEndpoint{
              .posContactRtWorldEwrtWorld = vector3_t(0.6, 0.2, 1.2),
              .normalSurfaceEwrtWorld = normalSurfaceEwrtWorld},
          ContactSwingSettings{.minDistControlpointRtContact = 0.1,
                               .clearance = 0.0},
          [](const vector3_t& posQueryRtWorldEwrtWorld) {
            return posQueryRtWorldEwrtWorld.z();
          }));
  constexpr scalar_t kNearTheEnds = 1.0e-4;
  const vector3_t liftoffDirContactEwrtWorld =
      swing.reference(kNearTheEnds).linvelContactRtWorldEwrtWorld.normalized();
  const vector3_t touchdownDirContactEwrtWorld =
      swing.reference(1.0 - kNearTheEnds)
          .linvelContactRtWorldEwrtWorld.normalized();
  EXPECT_GT(liftoffDirContactEwrtWorld.dot(vector3_t::UnitZ()), 0.999)
      << liftoffDirContactEwrtWorld.transpose();
  EXPECT_GT(touchdownDirContactEwrtWorld.dot(-normalSurfaceEwrtWorld), 0.999)
      << touchdownDirContactEwrtWorld.transpose();
  // The arc's tangents at its ends are the normals themselves.
  EXPECT_TRUE(swing.tangentAt(0.0).normalized().isApprox(vector3_t::UnitZ(),
                                                         kTolerance));
  EXPECT_TRUE(swing.tangentAt(1.0).normalized().isApprox(
      -normalSurfaceEwrtWorld, kTolerance));
}

TEST(ContactSwingTrajectory, raisesItsArcOverTheRiser) {
  const ContactSwingSettings settings{.minDistControlpointRtContact = 0.01,
                                      .clearance = 0.05};
  const ContactSwingTrajectory swing = stepUp(settings);
  EXPECT_TRUE(swing.clearanceSatisfied());
  EXPECT_GE(swing.clearanceMargin(), 0.0);
  EXPECT_GT(swing.getDistControlpointRtContact(),
            settings.minDistControlpointRtContact)
      << "the first arc grazes the riser";
  // Away from its ends, every checked point keeps the full clearance.
  for (int k = 1; k < ContactSwingTrajectory::kClearanceSamples; ++k) {
    const scalar_t s =
        static_cast<scalar_t>(k) / ContactSwingTrajectory::kClearanceSamples;
    if (s < ContactSwingTrajectory::kClearanceRampFraction ||
        s > 1.0 - ContactSwingTrajectory::kClearanceRampFraction) {
      continue;
    }
    EXPECT_GE(stepTerrain(swing.evaluatePosContactRtWorldEwrtWorld(s)),
              settings.clearance - kTolerance)
        << "at s = " << s;
  }
}

TEST(ContactSwingTrajectory, saysSoWhenNoArcClearsTheTerrain) {
  // A wall 100 m high between the two points.
  const ContactSwingTrajectory swing =
      checkedValue(ContactSwingTrajectory::Create(
          /*liftOffTime=*/0.0, /*touchDownTime=*/1.0,
          ContactSwingEndpoint{
              .posContactRtWorldEwrtWorld = vector3_t(0.0, 0.0, 0.0),
              .normalSurfaceEwrtWorld = vector3_t::UnitZ()},
          ContactSwingEndpoint{
              .posContactRtWorldEwrtWorld = vector3_t(1.0, 0.0, 0.0),
              .normalSurfaceEwrtWorld = vector3_t::UnitZ()},
          ContactSwingSettings{.minDistControlpointRtContact = 0.1,
                               .clearance = 0.05},
          [](const vector3_t& posQueryRtWorldEwrtWorld) {
            return std::min(
                posQueryRtWorldEwrtWorld.z(),
                boxDistance(posQueryRtWorldEwrtWorld, vector3_t(0.5, 0.0, 50.0),
                            vector3_t(0.05, 10.0, 50.0)));
          }));
  EXPECT_FALSE(swing.clearanceSatisfied());
  EXPECT_LT(swing.clearanceMargin(), 0.0);
  EXPECT_DOUBLE_EQ(swing.getDistControlpointRtContact(),
                   ContactSwingTrajectory::kMaxControlDistanceGrowth * 0.1);
}

TEST(ContactSwingTrajectory, refusesWhatIsNotASwing) {
  const ContactSwingEndpoint floor{
      .posContactRtWorldEwrtWorld = vector3_t::Zero(),
      .normalSurfaceEwrtWorld = vector3_t::UnitZ()};
  const ContactSwingSettings settings;
  const absl::StatusOr<ContactSwingTrajectory> backwards =
      ContactSwingTrajectory::Create(/*liftOffTime=*/1.0,
                                     /*touchDownTime=*/1.0, floor, floor,
                                     settings, &stepTerrain);
  EXPECT_EQ(backwards.status().code(), absl::StatusCode::kInvalidArgument);
  ContactSwingEndpoint notUnit = floor;
  notUnit.normalSurfaceEwrtWorld = vector3_t(0.0, 0.0, 2.0);
  EXPECT_FALSE(ContactSwingTrajectory::Create(/*liftOffTime=*/0.0,
                                              /*touchDownTime=*/1.0, notUnit,
                                              floor, settings, &stepTerrain)
                   .ok());
  EXPECT_FALSE(ContactSwingTrajectory::Create(
                   /*liftOffTime=*/0.0, /*touchDownTime=*/1.0, floor, floor,
                   ContactSwingSettings{.minDistControlpointRtContact = 0.1,
                                        .clearance = -0.1},
                   &stepTerrain)
                   .ok());
  EXPECT_FALSE(ContactSwingTrajectory::Create(
                   /*liftOffTime=*/0.0, /*touchDownTime=*/1.0, floor, floor,
                   ContactSwingSettings{.minDistControlpointRtContact = 0.0,
                                        .clearance = 0.0},
                   &stepTerrain)
                   .ok());
}

TEST(ContactSwingPlanner, answersTheSwingWhoseWindowHoldsTheTime) {
  ContactSwingPlanner planner(/*numContacts=*/3);
  std::vector<ContactSwingTrajectory> swings;
  swings.push_back(stepUp(ContactSwingSettings{}));
  planner.setSwings(/*contactIndex=*/1, std::move(swings));
  EXPECT_TRUE(planner.reference(/*contactIndex=*/1, /*time=*/1.5).has_value());
  EXPECT_FALSE(planner.reference(/*contactIndex=*/1, /*time=*/2.5).has_value());
  EXPECT_FALSE(planner.reference(/*contactIndex=*/0, /*time=*/1.5).has_value());
  EXPECT_FALSE(planner.reference(/*contactIndex=*/7, /*time=*/1.5).has_value());
  planner.clear();
  EXPECT_FALSE(planner.reference(/*contactIndex=*/1, /*time=*/1.5).has_value());
}

}  // namespace
}  // namespace ocs2::legged
