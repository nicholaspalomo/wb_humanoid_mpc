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

// PlanFile (design section 7.1): the hashes, save and load round trips, and the
// refusal of a plan made for another contact set or terrain scene.

#include <cstdlib>
#include <filesystem>
#include <string>
#include <variant>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_multi_contact/contact_set/contact_set.nproto.h"
#include "humanoid_multi_contact/plan/PlanFile.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/terrain/terrain_scene.nproto.h"
#include "humanoid_multi_contact/terrain/terrain_scene.nproto.pb.h"
#include "humanoid_multi_contact/terrain/terrain_scene.pb.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid::multi_contact {
namespace {

config::ContactSet contactSet() {
  config::ContactSet set;
  set.robot = "test";
  config::ContactSet::Candidate candidate;
  candidate.name = "knee";
  candidate.kind = config::ContactSet::Kind::kPoint;
  candidate.parent_joint = "knee_joint";
  candidate.offset = Eigen::Vector3d(0.05, 0.0, 0.0);
  candidate.pad_normal = Eigen::Vector3d::UnitX();
  set.candidates = {candidate};
  return set;
}

config::TerrainScene scene() {
  config::TerrainScene terrain;
  terrain.name = "floor";
  config::TerrainScene::Primitive floor;
  floor.name = "ground";
  floor.friction = 0.8;
  config::TerrainScene::Box box;
  box.center = Eigen::Vector3d(0.0, 0.0, -0.5);
  box.half_extents = Eigen::Vector3d(10.0, 10.0, 0.5);
  floor.shape = box;
  terrain.primitives = {floor};
  return terrain;
}

config::MultiContactPlan plan() {
  config::MultiContactPlan plan;
  plan.scenario = "test";
  plan.planner = "hand_written";
  plan.joint_names = {"knee_joint"};
  for (int k = 0; k < 3; ++k) {
    config::MultiContactPlan::Knot knot;
    knot.time = 0.5 * k;
    knot.base_position = Eigen::Vector3d(0.1 * k, 0.0, 0.9);
    knot.base_orientation_xyzw = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    knot.joint_positions = Eigen::VectorXd::Constant(1, 0.2);
    knot.active_contacts = 1;
    plan.knots.push_back(knot);
  }
  stampPlanHashes(contactSet(), scene(), plan);
  return plan;
}

std::string temporaryPath(const std::string& name) {
  const char* absl_nullable directory = std::getenv("TEST_TMPDIR");
  return absl::StrCat(directory != nullptr ? directory : "/tmp", "/", name);
}

TEST(PlanFileTest, HashesIgnoreHowDefaultsAreSpelled) {
  // A field written at its default hashes like a field left out.
  const absl::StatusOr<humanoid_multi_contact::TerrainScene> spelled =
      nproto::ParseTextproto<humanoid_multi_contact::TerrainScene>(
          R"(name: "floor"
             primitives { name: "ground" friction: 0.8
                          box { center: [0, 0, -0.5] half_extents: [10, 10, 0.5]
                                orientation_xyzw: [] } })",
          "spelled");
  ASSERT_TRUE(spelled.ok()) << spelled.status();
  config::TerrainScene parsed;
  ASSERT_TRUE(FromProto(*spelled, &parsed).ok());
  EXPECT_EQ(terrainSceneHash(parsed), terrainSceneHash(scene()));
  // A changed value does not.
  config::TerrainScene moved = scene();
  std::get<config::TerrainScene::Primitive::kBoxIndex>(
      moved.primitives.front().shape)
      .center.z() = -0.49;
  EXPECT_NE(terrainSceneHash(moved), terrainSceneHash(scene()));
  config::ContactSet renamed = contactSet();
  renamed.candidates.front().name = "elbow";
  EXPECT_NE(contactSetHash(renamed), contactSetHash(contactSet()));
}

TEST(PlanFileTest, SavesAndLoadsAPlanWithItsSummary) {
  const std::string path = temporaryPath("plan.binpb");
  ASSERT_TRUE(savePlanFile(plan(), path).ok());
  const absl::StatusOr<config::MultiContactPlan> loaded =
      loadPlanFile(path, contactSet(), scene());
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_EQ(*loaded, plan());
  // The summary is the plan without its knots, a textproto with its header.
  EXPECT_EQ(planSummaryPath(path), temporaryPath("plan.summary.textproto"));
  const absl::StatusOr<config::MultiContactPlan> summary =
      loadPlanFile(planSummaryPath(path), contactSet(), scene());
  ASSERT_TRUE(summary.ok()) << summary.status();
  EXPECT_TRUE(summary->knots.empty());
  EXPECT_EQ(summary->contact_set_hash, plan().contact_set_hash);
  EXPECT_EQ(savePlanFile(plan(), temporaryPath("plan.pb")).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(loadPlanFile(temporaryPath("missing.binpb"), contactSet(), scene())
                .status()
                .code(),
            absl::StatusCode::kNotFound);
}

TEST(PlanFileTest, RefusesAPlanMadeForAnotherContactSetOrScene) {
  const std::string path = temporaryPath("refused.binpb");
  ASSERT_TRUE(savePlanFile(plan(), path).ok());
  config::ContactSet otherSet = contactSet();
  otherSet.candidates.front().offset = Eigen::Vector3d(0.06, 0.0, 0.0);
  const absl::StatusOr<config::MultiContactPlan> wrongSet =
      loadPlanFile(path, otherSet, scene());
  EXPECT_EQ(wrongSet.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(absl::StrContains(wrongSet.status().message(), "contact set"))
      << wrongSet.status();
  config::TerrainScene otherScene = scene();
  otherScene.primitives.front().friction = 0.5;
  const absl::StatusOr<config::MultiContactPlan> wrongScene =
      loadPlanFile(path, contactSet(), otherScene);
  EXPECT_EQ(wrongScene.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(absl::StrContains(wrongScene.status().message(), "terrain scene"))
      << wrongScene.status();
  EXPECT_TRUE(checkPlanHashes(plan(), contactSet(), scene()).ok());
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
