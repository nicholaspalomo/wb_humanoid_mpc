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

// sampleContacts(): the 5 cm grid of a face shrunk by the pad, the reach,
// the faces a kind accepts, the samples under another primitive, and a
// capsule's top and side.

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/planner/ContactSampler.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr double kTolerance = 1.0e-9;
// [m] The sampler's grid by default.
constexpr double kGrid = 0.05;

config::TerrainScene::Primitive boxPrimitive(absl::string_view name,
                                             const Eigen::Vector3d& center,
                                             const Eigen::Vector3d& half) {
  config::TerrainScene::Primitive primitive;
  primitive.name = std::string(name);
  primitive.friction = 0.8;
  config::TerrainScene::Box box;
  box.center = center;
  box.half_extents = half;
  primitive.shape = box;
  return primitive;
}

/** The floor and a 0.18 m stair 0.28 m deep, its center at x = 0.64. */
TerrainModel stairTerrain() {
  config::TerrainScene scene;
  scene.name = "stair";
  scene.primitives.push_back(boxPrimitive("ground", Eigen::Vector3d(0, 0, -0.5),
                                          Eigen::Vector3d(20, 20, 0.5)));
  scene.primitives.push_back(boxPrimitive("step",
                                          Eigen::Vector3d(0.64, 0, 0.09),
                                          Eigen::Vector3d(0.14, 0.6, 0.09)));
  config::TerrainScene::Primitive rail;
  rail.name = "rail";
  rail.friction = 0.6;
  config::TerrainScene::Capsule capsule;
  capsule.start = Eigen::Vector3d(0.0, 0.7, 1.0);
  capsule.end = Eigen::Vector3d(1.0, 0.7, 1.0);
  capsule.radius = 0.025;
  rail.shape = capsule;
  scene.primitives.push_back(rail);
  absl::StatusOr<TerrainModel> terrain = TerrainModel::Create(scene);
  EXPECT_TRUE(terrain.ok()) << terrain.status();
  return *std::move(terrain);
}

ContactCandidate foot() {
  ContactCandidate candidate;
  candidate.name = "foot";
  candidate.kind = ContactKind::kSurface;
  candidate.rectangle = ContactRectangle{
      .xMin = -0.12, .xMax = 0.12, .yMin = -0.055, .yMax = 0.055};
  return candidate;
}

ContactCandidate hand() {
  ContactCandidate candidate;
  candidate.name = "hand";
  candidate.kind = ContactKind::kPoint;
  candidate.geomRadius = 0.04;
  return candidate;
}

TEST(ContactSampler, AFootFitsTheTreadOnlyAtItsMiddle) {
  const TerrainModel terrain = stairTerrain();
  const std::vector<ContactSample> samples = sampleContacts(
      terrain, foot(), /*candidateIndex=*/0, Eigen::Vector3d(0.64, 0, 0.9),
      /*reach=*/1.0, Eigen::Vector3d::UnitX(), ContactSamplerOptions());
  int onStep = 0;
  for (const ContactSample& sample : samples) {
    EXPECT_LE((sample.position - Eigen::Vector3d(0.64, 0, 0.9)).norm(),
              1.0 + kTolerance);
    EXPECT_TRUE(std::abs(sample.normal.z() - 1.0) < kTolerance)
        << "a foot only on level faces";
    if (sample.primitive != 1) continue;
    ++onStep;
    // 0.28 m deep: a 0.24 m sole fits with 1.5 cm to spare only centered.
    EXPECT_NEAR(sample.position.x(), 0.64, kTolerance);
    EXPECT_NEAR(sample.position.z(), 0.18, kTolerance);
    // On the 5 cm grid of the face, its edges 0.055 + 0.015 m away.
    EXPECT_NEAR(std::remainder(sample.position.y(), kGrid), 0.0, kTolerance);
    EXPECT_LE(std::abs(sample.position.y()), 0.6 - 0.07 + kTolerance);
  }
  EXPECT_GT(onStep, 0);
}

TEST(ContactSampler, DropsSamplesUnderOtherPrimitives) {
  const TerrainModel terrain = stairTerrain();
  const std::vector<ContactSample> samples = sampleContacts(
      terrain, foot(), /*candidateIndex=*/0, Eigen::Vector3d(0.5, 0, 0.5),
      /*reach=*/1.0, Eigen::Vector3d::UnitX(), ContactSamplerOptions());
  for (const ContactSample& sample : samples) {
    // Beside the stair, which is 1.2 m wide, a sole may be anywhere.
    if (sample.primitive != 0 || std::abs(sample.position.y()) > 0.6 + 0.055) {
      continue;
    }
    // A sole on the floor stays out of the stair: its toe before the riser
    // at x = 0.5, or its heel beyond the stair's back at x = 0.78.
    EXPECT_TRUE(sample.position.x() + 0.12 <= 0.5 + 0.004 ||
                sample.position.x() - 0.12 >= 0.78 - 0.004)
        << sample.position.transpose();
  }
}

TEST(ContactSampler, AHandAlsoUsesWallsAndTheRail) {
  const TerrainModel terrain = stairTerrain();
  const std::vector<ContactSample> samples = sampleContacts(
      terrain, hand(), /*candidateIndex=*/3, Eigen::Vector3d(0.4, 0.3, 1.0),
      /*reach=*/1.0, Eigen::Vector3d::UnitX(), ContactSamplerOptions());
  bool riser = false;
  bool railTop = false;
  bool railSide = false;
  for (const ContactSample& sample : samples) {
    EXPECT_EQ(sample.candidate, 3);
    if (sample.primitive == 1 && sample.normal.x() < -0.99) riser = true;
    if (sample.primitive == 2) {
      // On the rail's surface.
      EXPECT_NEAR(terrain.primitiveQuery(/*index=*/2, sample.position).distance,
                  0.0, 1.0e-6);
      if (sample.normal.z() > 0.99) railTop = true;
      if (sample.normal.y() < -0.5) railSide = true;
    }
  }
  EXPECT_TRUE(riser);
  EXPECT_TRUE(railTop);
  EXPECT_TRUE(railSide);
}

TEST(ContactSampler, TangentAxis) {
  EXPECT_TRUE(tangentAxis(Eigen::Vector3d(1, 0, 1), Eigen::Vector3d::UnitZ())
                  .isApprox(Eigen::Vector3d::UnitX()));
  const Eigen::Vector3d degenerate =
      tangentAxis(Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ());
  EXPECT_NEAR(degenerate.dot(Eigen::Vector3d::UnitZ()), 0.0, kTolerance);
  EXPECT_NEAR(degenerate.norm(), 1.0, kTolerance);
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
