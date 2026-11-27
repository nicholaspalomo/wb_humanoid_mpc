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

// ContactSet (design section 3.1): validation, masks and names, and the shipped
// contact set of the DRC Atlas.

#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/LU"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "legged_multi_contact/contact_set/ContactSet.h"
#include "legged_multi_contact/contact_set/contact_set.nproto.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::legged::multi_contact {
namespace {

constexpr char kAtlasContactSet[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/"
    "contact_set.textproto";

config::ContactSet::Candidate pointCandidate(std::string name) {
  config::ContactSet::Candidate candidate;
  candidate.name = std::move(name);
  candidate.kind = config::ContactSet::Kind::kPoint;
  candidate.parent_joint = "joint";
  candidate.pos_pad_rt_link_ewrt_link = Eigen::Vector3d(0.1, 0.0, 0.0);
  candidate.normal_pad_ewrt_link = Eigen::Vector3d::UnitX();
  candidate.geom_radius = 0.05;
  return candidate;
}

config::ContactSet::Candidate surfaceCandidate(std::string name) {
  config::ContactSet::Candidate candidate = pointCandidate(std::move(name));
  candidate.kind = config::ContactSet::Kind::kSurface;
  candidate.normal_pad_ewrt_link = -Eigen::Vector3d::UnitZ();
  candidate.x_axis_pad_ewrt_link = Eigen::Vector3d::UnitX();
  candidate.rectangle =
      config::ContactSet::Rectangle{.min_pos_corners_rt_pad_ewrt_pad_x = -0.1,
                                    .max_pos_corners_rt_pad_ewrt_pad_x = 0.1,
                                    .min_pos_corners_rt_pad_ewrt_pad_y = -0.05,
                                    .max_pos_corners_rt_pad_ewrt_pad_y = 0.05};
  return candidate;
}

absl::StatusOr<ContactSet> setOf(
    std::vector<config::ContactSet::Candidate> candidates) {
  config::ContactSet config;
  config.robot = "test";
  config.candidates = std::move(candidates);
  return ContactSet::Create(config);
}

void expectRefused(std::vector<config::ContactSet::Candidate> candidates,
                   absl::string_view message) {
  const absl::StatusOr<ContactSet> set = setOf(std::move(candidates));
  ASSERT_FALSE(set.ok());
  EXPECT_EQ(set.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(set.status().message(), message))
      << set.status();
}

TEST(ContactSetTest, PadFrames) {
  const absl::StatusOr<ContactSet> set =
      setOf({pointCandidate("knee"), surfaceCandidate("foot")});
  ASSERT_TRUE(set.ok()) << set.status();
  ASSERT_EQ(set->size(), 2);
  for (const ContactCandidate& candidate : set->candidates()) {
    EXPECT_NEAR(candidate.rotmatPadRtLink.determinant(), 1.0, 1.0e-12);
    EXPECT_LT(
        (candidate.rotmatPadRtLink.col(2) + candidate.normalPadEwrtLink).norm(),
        1.0e-12);
  }
  const ContactCandidate& foot = set->candidate(1);
  EXPECT_EQ(foot.kind, ContactKind::kSurface);
  EXPECT_TRUE(foot.rotmatPadRtLink.isIdentity(1.0e-12));
  // An if, not ASSERT_TRUE, here and below: bugprone-unchecked-optional-access
  // sees a check only through an if.
  if (!foot.rectangle.has_value()) {
    FAIL() << "foot.rectangle has no value";
  }
  EXPECT_DOUBLE_EQ(foot.rectangle->maxPosCornersRtPadEwrtPadX, 0.1);
  EXPECT_FALSE(set->candidate(0).rectangle.has_value());
}

TEST(ContactSetTest, MasksAndNames) {
  const absl::StatusOr<ContactSet> set =
      setOf({pointCandidate("a"), pointCandidate("b"), surfaceCandidate("c")});
  ASSERT_TRUE(set.ok()) << set.status();
  const absl::StatusOr<ContactMask> mask = set->maskOf({"c", "a", "c"});
  ASSERT_TRUE(mask.ok());
  EXPECT_EQ(*mask, ContactMask{5});
  EXPECT_TRUE(isContactActive(*mask, /*index=*/0));
  EXPECT_FALSE(isContactActive(*mask, /*index=*/1));
  const absl::StatusOr<std::vector<std::string>> names = set->namesOf(*mask);
  ASSERT_TRUE(names.ok());
  EXPECT_EQ(*names, (std::vector<std::string>{"a", "c"}));
  EXPECT_EQ(set->maskOf({"z"}).status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(set->activeIndices(ContactMask{8}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(*set->indexOf("b"), 1);
}

TEST(ContactSetTest, RefusesInvalidCandidates) {
  expectRefused(/*candidates=*/{}, "needs 1 to 64");
  expectRefused({pointCandidate("a"), pointCandidate("a")}, "listed twice");
  config::ContactSet::Candidate kindless = pointCandidate("a");
  kindless.kind = config::ContactSet::Kind::kUnspecified;
  expectRefused({kindless}, "kind must be");
  config::ContactSet::Candidate orphan = pointCandidate("a");
  orphan.parent_joint = "";
  expectRefused({orphan}, "has no parent_joint");
  config::ContactSet::Candidate shortOffset = pointCandidate("a");
  shortOffset.pos_pad_rt_link_ewrt_link = Eigen::Vector2d(0.1, 0.0);
  expectRefused({shortOffset}, "pos_pad_rt_link_ewrt_link must hold 3");
  config::ContactSet::Candidate longNormal = pointCandidate("a");
  longNormal.normal_pad_ewrt_link = Eigen::Vector3d(2.0, 0.0, 0.0);
  expectRefused({longNormal}, "normal_pad_ewrt_link must be a unit vector");
  config::ContactSet::Candidate skewAxis = surfaceCandidate("a");
  skewAxis.x_axis_pad_ewrt_link = Eigen::Vector3d::UnitZ();
  expectRefused({skewAxis}, "orthogonal");
  config::ContactSet::Candidate noRectangle = surfaceCandidate("a");
  noRectangle.rectangle.reset();
  expectRefused({noRectangle}, "needs a rectangle");
  config::ContactSet::Candidate pointRectangle = surfaceCandidate("a");
  pointRectangle.kind = config::ContactSet::Kind::kPoint;
  expectRefused({pointRectangle}, "has no rectangle");
  config::ContactSet::Candidate emptyRectangle = surfaceCandidate("a");
  if (!emptyRectangle.rectangle.has_value()) {
    FAIL() << "a surface candidate has a rectangle";
  }
  emptyRectangle.rectangle->max_pos_corners_rt_pad_ewrt_pad_x = -0.2;
  expectRefused(
      {emptyRectangle},
      "min_pos_corners_rt_pad_ewrt_pad_x < max_pos_corners_rt_pad_ewrt_pad_x");
  config::ContactSet::Candidate negativeRadius = pointCandidate("a");
  negativeRadius.geom_radius = -0.01;
  expectRefused({negativeRadius}, "geom_radius");
}

TEST(ContactSetTest, AtlasContactSet) {
  const absl::StatusOr<std::string> path =
      robot::resolveResourcePath(kAtlasContactSet);
  ASSERT_TRUE(path.ok()) << path.status();
  const absl::StatusOr<config::ContactSet> config = loadContactSetFile(*path);
  ASSERT_TRUE(config.ok()) << config.status();
  const absl::StatusOr<ContactSet> set = ContactSet::Create(*config);
  ASSERT_TRUE(set.ok()) << set.status();
  // The seventeen candidates of the design's table, in its order.
  const std::vector<std::string> expected = {"foot_l_contact",
                                             "foot_r_contact",
                                             "knee_l_contact",
                                             "knee_r_contact",
                                             "hand_l_contact",
                                             "hand_r_contact",
                                             "forearm_l_contact",
                                             "forearm_r_contact",
                                             "seat_contact",
                                             "thigh_l_contact",
                                             "thigh_r_contact",
                                             "torso_back_contact",
                                             "torso_front_contact",
                                             "torso_edge_r_low_25_contact",
                                             "torso_edge_r_high_25_contact",
                                             "torso_edge_r_low_65_contact",
                                             "torso_edge_r_high_65_contact"};
  ASSERT_EQ(set->size(), static_cast<int>(expected.size()));
  int surfaces = 0;
  for (int index = 0; index < set->size(); ++index) {
    const ContactCandidate& candidate = set->candidate(index);
    EXPECT_EQ(candidate.name, expected[index]);
    if (candidate.kind == ContactKind::kSurface) ++surfaces;
  }
  // The feet, the seat and the two torso pads.
  EXPECT_EQ(surfaces, 5);
  // A sole is 0.24 x 0.11 m, the torso pads 0.36 x 0.30 m.
  const ContactCandidate& foot = set->candidate(0);
  if (!foot.rectangle.has_value()) {
    FAIL() << "the sole has no rectangle";
  }
  EXPECT_NEAR(foot.rectangle->maxPosCornersRtPadEwrtPadX -
                  foot.rectangle->minPosCornersRtPadEwrtPadX,
              0.24, 1.0e-12);
  EXPECT_NEAR(foot.rectangle->maxPosCornersRtPadEwrtPadY -
                  foot.rectangle->minPosCornersRtPadEwrtPadY,
              0.11, 1.0e-12);
  const ContactCandidate& back = set->candidate(11);
  if (!back.rectangle.has_value()) {
    FAIL() << "the torso pad has no rectangle";
  }
  EXPECT_NEAR(back.rectangle->maxPosCornersRtPadEwrtPadX -
                  back.rectangle->minPosCornersRtPadEwrtPadX,
              0.36, 1.0e-12);
  EXPECT_NEAR(back.rectangle->maxPosCornersRtPadEwrtPadY -
                  back.rectangle->minPosCornersRtPadEwrtPadY,
              0.30, 1.0e-12);
  // The torso pad's x axis runs up the torso, and its z axis into the body.
  EXPECT_LT((back.rotmatPadRtLink.col(0) - Eigen::Vector3d::UnitZ()).norm(),
            1.0e-12);
  EXPECT_LT((back.rotmatPadRtLink.col(2) - Eigen::Vector3d::UnitX()).norm(),
            1.0e-12);
  // A round pad lies at its radius from the geometry: the thigh's from the
  // capsule axis from (0, 0, 0) to (-0.05, 0, -0.37).
  const ContactCandidate& thigh = set->candidate(9);
  const Eigen::Vector3d axis = Eigen::Vector3d(-0.05, 0.0, -0.37).normalized();
  const Eigen::Vector3d fromAxis =
      thigh.posPadRtLinkEwrtLink - thigh.posPadRtLinkEwrtLink.dot(axis) * axis;
  EXPECT_NEAR(fromAxis.norm(), thigh.geomRadius, 1.0e-3);
  EXPECT_LT((fromAxis.normalized() - thigh.normalPadEwrtLink).norm(), 1.0e-3);
}

}  // namespace
}  // namespace ocs2::legged::multi_contact
