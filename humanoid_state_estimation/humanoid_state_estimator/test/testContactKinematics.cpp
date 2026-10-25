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
// against a central finite difference of its position.

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_state_estimation/humanoid_state_estimator/test/WalkingTrajectory.h"
#include "humanoid_state_estimator/ContactKinematics.h"
#include "robot_model/RobotDescription.h"

namespace ocs2::humanoid::estimation {
namespace {

constexpr char kAtlasUrdf[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

std::vector<ContactPointSpec> specsOf(const std::vector<std::string>& frames) {
  std::vector<ContactPointSpec> specs;
  for (const std::string& frame : frames)
    specs.push_back(ContactPointSpec{.frameNames = {frame}});
  return specs;
}

std::unique_ptr<ContactKinematics> prismaticBiped() {
  absl::StatusOr<std::unique_ptr<ContactKinematics>> kinematics =
      ContactKinematics::Create(
          test_support::kPrismaticBipedUrdf,
          specsOf(test_support::prismaticBipedFootFrames()),
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
  EXPECT_TRUE(kinematics->gravity().isApprox(vector3_t(0.0, 0.0, -9.81)));
  std::mt19937 generator(7);
  std::uniform_real_distribution<scalar_t> uniform(-0.5, 0.5);
  const std::vector<vector3_t> origins = {vector3_t(0.0, 0.1, -0.8),
                                          vector3_t(0.0, -0.1, -0.8)};
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
      EXPECT_TRUE(kinematics->contactPositionInBase(foot).isApprox(
          origins[foot] + q.segment<3>(first), 1.0e-12))
          << trial;
      EXPECT_TRUE(kinematics->contactVelocityInBase(foot).isApprox(
          v.segment<3>(first), 1.0e-12))
          << trial;
    }
  }
}

TEST(ContactKinematicsTest,
     AContinuousJointAndAnOffsetPointFollowTheClosedForm) {
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
  // The tip's frame, and a point 0.5 m along its y axis: R(theta) (1, 0.5, 0).
  absl::StatusOr<std::unique_ptr<ContactKinematics>> kinematics =
      ContactKinematics::Create(
          urdf,
          {ContactPointSpec{.frameNames = {"not_a_frame", "tip"},
                            .offsetInFrame = vector3_t(0.0, 0.5, 0.0)}},
          {"spin"});
  ASSERT_TRUE(kinematics.ok()) << kinematics.status();
  EXPECT_EQ((*kinematics)->contactFrameName(0), "tip")
      << "the first frame of the list the URDF has";
  const scalar_t theta = 0.7;
  const scalar_t rate = -1.3;
  (*kinematics)
      ->update((vector_t(1) << theta).finished(),
               (vector_t(1) << rate).finished());
  const vector3_t point(1.0, 0.5, 0.0);
  const matrix3_t rotation =
      Eigen::AngleAxisd(theta, vector3_t::UnitZ()).toRotationMatrix();
  EXPECT_TRUE((*kinematics)
                  ->contactPositionInBase(0)
                  .isApprox(rotation * point, 1.0e-12));
  EXPECT_TRUE((*kinematics)
                  ->contactVelocityInBase(0)
                  .isApprox(vector3_t(0.0, 0.0, rate).cross(rotation * point),
                            1.0e-12));
}

TEST(ContactKinematicsTest, RefusesAContactNoneOfWhoseFramesIsInTheUrdf) {
  const absl::StatusOr<std::unique_ptr<ContactKinematics>> kinematics =
      ContactKinematics::Create(test_support::kPrismaticBipedUrdf,
                                specsOf({"foot_l_contact"}),
                                test_support::prismaticBipedJointNames());
  ASSERT_EQ(kinematics.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(
      absl::StrContains(kinematics.status().message(), "foot_l_contact"))
      << kinematics.status();
  EXPECT_TRUE(absl::StrContains(kinematics.status().message(), "left_foot"))
      << "lists the frames: " << kinematics.status();
  EXPECT_EQ(ContactKinematics::Create("no/such.urdf", specsOf({"left_foot"}),
                                      /*jointNames=*/{})
                .status()
                .code(),
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
    const std::vector<vector3_t> velocity = {
        kinematics.contactVelocityInBase(0),
        kinematics.contactVelocityInBase(1)};
    kinematics.update(q + kStep * v, v);
    const std::vector<vector3_t> ahead = {kinematics.contactPositionInBase(0),
                                          kinematics.contactPositionInBase(1)};
    kinematics.update(q - kStep * v, v);
    for (size_t foot = 0; foot < 2; ++foot) {
      const vector3_t difference =
          (ahead[foot] - kinematics.contactPositionInBase(foot)) /
          (2.0 * kStep);
      // A central difference errs by O(kStep^2) and the rounding of positions
      // of about 1 m, ~1e-16 / 1e-6.
      EXPECT_LT((difference - velocity[foot]).norm(), 1.0e-7)
          << "trial " << trial << ", foot " << foot;
      EXPECT_GT(velocity[foot].norm(), 0.0);
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid::estimation
