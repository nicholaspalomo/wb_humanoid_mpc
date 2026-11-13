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

// ContactKinematics against closed forms (the prismatic biped, a continuous
// joint with an offset contact point) and, on the Atlas URDF, its velocity
// against a central finite difference of its position; the child links of
// joints (childLinkNamesOfJoints()).

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_state_estimation/humanoid_state_estimator/test/WalkingTrajectory.h"
#include "humanoid_state_estimator/ContactKinematics.h"
#include "robot_model/RobotDescription.h"

namespace ocs2::humanoid::estimation {
namespace {

constexpr char kAtlasUrdf[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

std::vector<ContactPointSpec> specsOf(absl::Span<const std::string> links) {
  std::vector<ContactPointSpec> specs;
  for (const std::string& link : links) {
    specs.push_back(ContactPointSpec{.linkName = link});
  }
  return specs;
}

std::unique_ptr<ContactKinematics> prismaticBiped() {
  absl::StatusOr<std::unique_ptr<ContactKinematics>> kinematics =
      ContactKinematics::Create(
          test_support::kPrismaticBipedUrdf,
          specsOf(test_support::prismaticBipedFootLinks()),
          test_support::prismaticBipedJointNames());
  EXPECT_TRUE(kinematics.ok()) << kinematics.status();
  return kinematics.ok() ? *std::move(kinematics) : nullptr;
}

TEST(ContactKinematicsTest,
     ThePrismaticBipedsFeetAreAtTheLegOriginPlusTheJoints) {
  std::unique_ptr<ContactKinematics> kinematics = prismaticBiped();
  ASSERT_NE(kinematics, nullptr);
  EXPECT_EQ(kinematics->numContacts(), 2u);
  EXPECT_EQ(kinematics->numJoints(), 6u);
  EXPECT_TRUE(
      kinematics->gravityEwrtWorld().isApprox(vector3_t(0.0, 0.0, -9.81)));
  std::mt19937 generator(7);
  std::uniform_real_distribution<scalar_t> uniform(-0.5, 0.5);
  const std::vector<vector3_t> posHipsRtBaseEwrtBase = {
      vector3_t(0.0, 0.1, -0.8), vector3_t(0.0, -0.1, -0.8)};
  for (int trial = 0; trial < 20; ++trial) {
    vector_t q(6);
    vector_t v(6);
    for (int i = 0; i < 6; ++i) {
      q[i] = uniform(generator);
      v[i] = uniform(generator);
    }
    kinematics->update(q, v);
    for (size_t foot = 0; foot < 2; ++foot) {
      const Eigen::Index first = static_cast<Eigen::Index>(3 * foot);
      EXPECT_TRUE(kinematics->posContactRtBaseEwrtBase(foot).isApprox(
          posHipsRtBaseEwrtBase[foot] + q.segment<3>(first), 1.0e-12))
          << trial;
      EXPECT_TRUE(kinematics->linvelContactRtBaseEwrtBase(foot).isApprox(
          v.segment<3>(first), 1.0e-12))
          << trial;
    }
  }
}

/** A one-joint arm in TEST_TMPDIR: a continuous hinge `spin` from `base` to
 * `arm`, and `tip` fixed 1 m along the arm's x axis by `tip_mount`. */
std::string armUrdf() {
  const std::string urdf =
      (std::filesystem::path(std::getenv("TEST_TMPDIR")) / "arm.urdf").string();
  std::ofstream(urdf) << R"(<?xml version="1.0"?>
<robot name="arm">
  <link name="base"><inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
  <link name="arm"><inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
  <link name="tip"/>
  <joint name="spin" type="continuous"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/></joint>
  <joint name="tip_mount" type="fixed"><parent link="arm"/><child link="tip"/><origin xyz="1 0 0"/></joint>
</robot>)";
  return urdf;
}

TEST(ContactKinematicsTest,
     AContinuousJointAndAnOffsetPointFollowTheClosedForm) {
  // The tip's frame, and a point 0.5 m along its y axis: R(theta) (1, 0.5, 0).
  absl::StatusOr<std::unique_ptr<ContactKinematics>> kinematics =
      ContactKinematics::Create(
          armUrdf(),
          {ContactPointSpec{
              .linkName = "tip",
              .posContactRtLinkEwrtLink = vector3_t(0.0, 0.5, 0.0)}},
          {"spin"});
  ASSERT_TRUE(kinematics.ok()) << kinematics.status();
  EXPECT_EQ((*kinematics)->contactLinkName(0), "tip");
  const scalar_t theta = 0.7;
  const scalar_t rate = -1.3;
  (*kinematics)
      ->update((vector_t(1) << theta).finished(),
               (vector_t(1) << rate).finished());
  // The contact is the spec's offset from its link, `tip`, which the fixed
  // tip_mount holds 1 m along the hinge child's x axis, with the child's axes.
  const vector3_t posContactRtLinkEwrtLink(0.0, 0.5, 0.0);
  const matrix3_t rotmatLinkRtBase =
      Eigen::AngleAxisd(theta, vector3_t::UnitZ()).toRotationMatrix();
  const vector3_t posLinkRtBaseEwrtBase =
      rotmatLinkRtBase * vector3_t(1.0, 0.0, 0.0);
  const vector3_t posContactRtBaseEwrtBase =
      posLinkRtBaseEwrtBase + rotmatLinkRtBase * posContactRtLinkEwrtLink;
  EXPECT_TRUE((*kinematics)
                  ->posContactRtBaseEwrtBase(0)
                  .isApprox(posContactRtBaseEwrtBase, 1.0e-12));
  EXPECT_TRUE(
      (*kinematics)
          ->linvelContactRtBaseEwrtBase(0)
          .isApprox(vector3_t(0.0, 0.0, rate).cross(posContactRtBaseEwrtBase),
                    1.0e-12));
}

TEST(ContactKinematicsTest, RefusesAContactWhoseLinkIsNotInTheUrdf) {
  // An MPC contact frame is not a link of the URDF, and neither is a joint.
  for (const std::string& notALink :
       std::vector<std::string>{"foot_l_contact", "left_z"}) {
    const absl::StatusOr<std::unique_ptr<ContactKinematics>> kinematics =
        ContactKinematics::Create(test_support::kPrismaticBipedUrdf,
                                  specsOf({notALink}),
                                  test_support::prismaticBipedJointNames());
    ASSERT_EQ(kinematics.status().code(), absl::StatusCode::kInvalidArgument)
        << notALink;
    EXPECT_TRUE(absl::StrContains(kinematics.status().message(), notALink))
        << kinematics.status();
    EXPECT_TRUE(absl::StrContains(kinematics.status().message(), "left_foot"))
        << "lists the links: " << kinematics.status();
  }
  EXPECT_EQ(ContactKinematics::Create("no/such.urdf", specsOf({"left_foot"}),
                                      /*jointNames=*/{})
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ContactKinematicsTest, RefusesARollingContact) {
  // A wheel's lowest point is no point of its link; a contact without an MPC
  // name is named by its link.
  std::vector<ContactPointSpec> specs = specsOf({"left_foot", "right_foot"});
  specs[0].kind = ContactPointKind::kRolling;
  specs[0].wheelRadius = 0.1;
  const absl::StatusOr<std::unique_ptr<ContactKinematics>> kinematics =
      ContactKinematics::Create(test_support::kPrismaticBipedUrdf, specs,
                                test_support::prismaticBipedJointNames());
  ASSERT_EQ(kinematics.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(kinematics.status().message(),
                                "the rolling contact on the link 'left_foot'"))
      << kinematics.status();
  EXPECT_TRUE(checkFixedContactPoints(specsOf({"left_foot"}), "test").ok());
}

TEST(ContactKinematicsTest, AJointsChildLinkIsTheLinkItPlaces) {
  const absl::StatusOr<std::vector<std::string>> links = childLinkNamesOfJoints(
      armUrdf(), {kMpcRootJointName, "spin", "tip_mount", "spin"});
  ASSERT_TRUE(links.ok()) << links.status();
  EXPECT_EQ(*links, (std::vector<std::string>{"base", "arm", "tip", "arm"}))
      << "the MPC's free-flyer is the URDF's root link; a fixed joint has a "
         "child link too";
  const absl::StatusOr<std::vector<std::string>> biped = childLinkNamesOfJoints(
      test_support::kPrismaticBipedUrdf, {"left_z", "right_x"});
  ASSERT_TRUE(biped.ok()) << biped.status();
  EXPECT_EQ(*biped, (std::vector<std::string>{"left_foot", "right_x_link"}));
  EXPECT_TRUE(childLinkNamesOfJoints(test_support::kPrismaticBipedUrdf,
                                     /*jointNames=*/{})
                  .ok());
}

TEST(ContactKinematicsTest, AJointTheUrdfDoesNotHaveHasNoChildLink) {
  // A link's name is not a joint's.
  for (const std::string& notAJoint :
       std::vector<std::string>{"no_such_joint", "left_foot"}) {
    const absl::StatusOr<std::vector<std::string>> links =
        childLinkNamesOfJoints(test_support::kPrismaticBipedUrdf,
                               {"left_x", notAJoint});
    ASSERT_EQ(links.status().code(), absl::StatusCode::kInvalidArgument)
        << notAJoint;
    EXPECT_TRUE(absl::StrContains(links.status().message(), notAJoint))
        << links.status();
  }
  EXPECT_EQ(childLinkNamesOfJoints("no/such.urdf", {"left_x"}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ContactKinematicsTest, TheAtlasFootVelocityIsTheDerivativeOfItsPosition) {
  const absl::StatusOr<robot::model::RobotDescription> description =
      robot::model::RobotDescription::Create(kAtlasUrdf);
  ASSERT_TRUE(description.ok()) << description.status();
  absl::StatusOr<std::unique_ptr<ContactKinematics>> created =
      ContactKinematics::Create(kAtlasUrdf, specsOf({"l_foot", "r_foot"}),
                                description->getJointNames());
  ASSERT_TRUE(created.ok()) << created.status();
  ContactKinematics& kinematics = **created;
  const Eigen::Index joints =
      static_cast<Eigen::Index>(description->getNumJoints());
  std::mt19937 generator(3);
  std::uniform_real_distribution<scalar_t> uniform(-0.6, 0.6);
  constexpr scalar_t kStep = 1.0e-6;
  for (int trial = 0; trial < 10; ++trial) {
    vector_t q(joints);
    vector_t v(joints);
    for (Eigen::Index i = 0; i < joints; ++i) {
      q[i] = uniform(generator);
      v[i] = uniform(generator);
    }
    kinematics.update(q, v);
    const std::vector<vector3_t> linvelFeetRtBaseEwrtBase = {
        kinematics.linvelContactRtBaseEwrtBase(0),
        kinematics.linvelContactRtBaseEwrtBase(1)};
    kinematics.update(q + kStep * v, v);
    const std::vector<vector3_t> nextPosFeetRtBaseEwrtBase = {
        kinematics.posContactRtBaseEwrtBase(0),
        kinematics.posContactRtBaseEwrtBase(1)};
    kinematics.update(q - kStep * v, v);
    for (size_t foot = 0; foot < 2; ++foot) {
      const vector3_t fdLinvelFootRtBaseEwrtBase =
          (nextPosFeetRtBaseEwrtBase[foot] -
           kinematics.posContactRtBaseEwrtBase(foot)) /
          (2.0 * kStep);
      // A central difference errs by O(kStep^2) and the rounding of positions
      // of about 1 m, ~1e-16 / 1e-6.
      EXPECT_LT(
          (fdLinvelFootRtBaseEwrtBase - linvelFeetRtBaseEwrtBase[foot]).norm(),
          1.0e-7)
          << "trial " << trial << ", foot " << foot;
      EXPECT_GT(linvelFeetRtBaseEwrtBase[foot].norm(), 0.0);
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid::estimation
