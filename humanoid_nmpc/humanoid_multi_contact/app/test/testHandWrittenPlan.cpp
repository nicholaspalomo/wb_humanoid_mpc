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

// The certifier end to end on the hand-written plan testdata/box_climb_sway
// (Atlas swaying on both feet in the box_climb scenario): the plan's hashes,
// a passing report, and the failures of edited copies of it.

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "gtest/gtest.h"

#include "humanoid_multi_contact/app/CertifyPlan.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/plan/PlanFile.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.pb.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.pb.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"
#include "humanoid_multi_contact/stability/StabilityCertifier.h"
#include "humanoid_multi_contact/stability/certifier_report.nproto.h"
#include "nproto/Textproto.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr char kScenario[] =
    "humanoid_nmpc/humanoid_multi_contact/scenarios/config/box_climb.textproto";
constexpr char kPlan[] =
    "humanoid_nmpc/humanoid_multi_contact/app/testdata/"
    "box_climb_sway.textproto";

std::string resolved(const char* absl_nonnull path) {
  const absl::StatusOr<std::string> resolvedPath =
      robot::resolveResourcePath(path);
  EXPECT_TRUE(resolvedPath.ok()) << resolvedPath.status();
  return resolvedPath.ok() ? *resolvedPath : std::string();
}

TEST(HandWrittenPlanTest, ItsHashesAreThoseOfTheScenario) {
  const absl::StatusOr<LoadedScenario> loaded =
      loadScenario(resolved(kScenario));
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  const absl::StatusOr<config::MultiContactPlan> plan =
      nproto::LoadTextprotoFile<config::MultiContactPlan,
                                humanoid_multi_contact::MultiContactPlan>(
          resolved(kPlan));
  ASSERT_TRUE(plan.ok()) << plan.status();
  // When the contact set or the scene changes, write these into the plan.
  EXPECT_EQ(plan->contact_set_hash, contactSetHash(loaded->contactSetConfig))
      << "contact_set_hash: " << contactSetHash(loaded->contactSetConfig);
  EXPECT_EQ(plan->terrain_scene_hash,
            terrainSceneHash(loaded->scenario.terrain_scene))
      << "terrain_scene_hash: "
      << terrainSceneHash(loaded->scenario.terrain_scene);
}

TEST(HandWrittenPlanTest, EveryKnotPasses) {
  const absl::StatusOr<config::MultiContactPlan> plan =
      certifyPlanFile(resolved(kScenario), resolved(kPlan),
                      /*samplesBetweenKnots=*/0);
  ASSERT_TRUE(plan.ok()) << plan.status();
  ASSERT_TRUE(plan->report.has_value());
  const config::CertifierReport& report = *plan->report;
  EXPECT_TRUE(report.passed) << formatReport(report);
  ASSERT_EQ(report.knots.size(), plan->knots.size());
  EXPECT_EQ(report.static_margin, 0.03);
  for (const config::CertifierReport::Knot& knot : report.knots) {
    EXPECT_TRUE(knot.passed) << absl::StrJoin(knot.failures, "; ");
    EXPECT_EQ(knot.static_condition, true);
    EXPECT_TRUE(knot.dynamic_condition);
    EXPECT_FALSE(knot.transition_condition.has_value());
    ASSERT_TRUE(knot.static_margin.has_value());
    EXPECT_GE(*knot.static_margin, 0.03);
    EXPECT_GE(knot.support_polygon.size(), 4u);
    EXPECT_EQ(knot.active_contacts.size(), 2u);
    EXPECT_EQ(knot.com.size(), 3);
  }
  // The sway moves the CoM 6 cm toward the toes; the margin, bounded by the
  // outer edges of the soles at the sides, stays.
  EXPECT_NEAR(report.knots[2].com(0) - report.knots[0].com(0), 0.06, 1.0e-9);
  EXPECT_LE(*report.knots[2].static_margin,
            *report.knots[0].static_margin + 1.0e-9);
  EXPECT_FALSE(formatReport(report).empty());
}

TEST(HandWrittenPlanTest, TheMotionBetweenTheKnotsPassesToo) {
  // Three configurations between every two knots, as the MPC tracks them.
  const absl::StatusOr<config::MultiContactPlan> plan =
      certifyPlanFile(resolved(kScenario), resolved(kPlan),
                      /*samplesBetweenKnots=*/3);
  ASSERT_TRUE(plan.ok()) << plan.status();
  ASSERT_TRUE(plan->report.has_value());
  EXPECT_TRUE(plan->report->passed) << formatReport(*plan->report);
  EXPECT_EQ(certifyPlanFile(resolved(kScenario), resolved(kPlan),
                            /*samplesBetweenKnots=*/-1)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(HandWrittenPlanTest, KnotsBetweenKnotsInterpolateTheReference) {
  config::MultiContactPlan plan;
  config::MultiContactPlan::Knot first;
  first.time = 1.0;
  first.base_position = Eigen::Vector3d(0.0, 0.0, 1.0);
  first.base_orientation_xyzw = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  first.joint_positions = Eigen::Vector2d(0.0, 1.0);
  first.active_contacts = 3;
  config::MultiContactPlan::Knot second = first;
  second.time = 2.0;
  second.base_position = Eigen::Vector3d(0.4, 0.0, 1.0);
  // A half turn about z.
  second.base_orientation_xyzw = Eigen::Vector4d(0.0, 0.0, 1.0, 0.0);
  second.joint_positions = Eigen::Vector2d(0.8, -1.0);
  second.active_contacts = 1;
  plan.knots = {first, second};
  const config::MultiContactPlan dense = planBetweenKnots(plan, /*samples=*/3);
  ASSERT_EQ(dense.knots.size(), 5u);
  EXPECT_EQ(dense.knots.front().time, 1.0);
  EXPECT_EQ(dense.knots.back().time, 2.0);
  const config::MultiContactPlan::Knot& middle = dense.knots[2];
  EXPECT_DOUBLE_EQ(middle.time, 1.5);
  EXPECT_DOUBLE_EQ(middle.base_position[0], 0.2);
  EXPECT_DOUBLE_EQ(middle.joint_positions[0], 0.4);
  EXPECT_DOUBLE_EQ(middle.joint_positions[1], 0.0);
  // A quarter turn about z, unit; the first knot's contacts.
  EXPECT_NEAR(middle.base_orientation_xyzw[2], std::sqrt(0.5), 1.0e-12);
  EXPECT_NEAR(middle.base_orientation_xyzw[3], std::sqrt(0.5), 1.0e-12);
  EXPECT_EQ(middle.active_contacts, 3u);
}

/** The plan, its scenario and a certifier for it. */
struct Fixture {
  LoadedScenario loaded;
  config::MultiContactPlan plan;
  std::unique_ptr<StabilityCertifier> certifier;
};

absl::StatusOr<Fixture> fixture() {
  absl::StatusOr<LoadedScenario> loaded = loadScenario(resolved(kScenario));
  if (!loaded.ok()) return loaded.status();
  absl::StatusOr<config::MultiContactPlan> plan =
      loadPlanFile(resolved(kPlan), loaded->contactSetConfig,
                   loaded->scenario.terrain_scene);
  if (!plan.ok()) return plan.status();
  absl::StatusOr<std::unique_ptr<StabilityCertifier>> certifier =
      makeCertifier(*loaded);
  if (!certifier.ok()) return certifier.status();
  return Fixture{.loaded = *std::move(loaded),
                 .plan = *std::move(plan),
                 .certifier = *std::move(certifier)};
}

TEST(HandWrittenPlanTest, ALungeLeavesTheContactWrenchCone) {
  absl::StatusOr<Fixture> test = fixture();
  ASSERT_TRUE(test.ok()) << test.status();
  // 1500 kg m/s in 0.25 s: about 6000 N forward, far beyond the friction.
  test->plan.knots[2].linear_momentum = Eigen::Vector3d(1500.0, 0.0, 0.0);
  const absl::StatusOr<config::CertifierReport> report =
      test->certifier->certify(test->plan);
  ASSERT_TRUE(report.ok()) << report.status();
  EXPECT_FALSE(report->passed);
  EXPECT_FALSE(report->knots[1].momentum_rate_feasible);
  EXPECT_FALSE(report->knots[1].dynamic_condition);
  EXPECT_GT(report->knots[1].momentum_rate_residual, 1.0e-3);
  // Knot 2 is a peak of the momentum, whose central difference is zero.
  EXPECT_TRUE(report->knots[2].dynamic_condition);
}

TEST(HandWrittenPlanTest,
     LiftingAFootUnderTheCenterOfMassFailsConditionsOneAndThree) {
  absl::StatusOr<Fixture> test = fixture();
  ASSERT_TRUE(test.ok()) << test.status();
  const ContactMask leftFoot =
      contactBit(*test->loaded.contactSet.indexOf("foot_l_contact"));
  test->plan.knots[3].active_contacts = leftFoot;
  const absl::StatusOr<config::CertifierReport> report =
      test->certifier->certify(test->plan);
  ASSERT_TRUE(report.ok()) << report.status();
  EXPECT_FALSE(report->passed);
  // The CoM is between the feet, not above the left one.
  EXPECT_EQ(report->knots[3].static_condition, false);
  ASSERT_TRUE(report->knots[3].static_margin.has_value());
  EXPECT_LT(*report->knots[3].static_margin, 0.0);
  // The stance changes by one contact into and out of knot 3, and the left
  // foot alone does not hold the robot at either side of the change.
  EXPECT_EQ(report->knots[3].transition_condition, false);
  EXPECT_EQ(report->knots[4].transition_condition, false);
  EXPECT_TRUE(report->knots[2].passed);
}

TEST(HandWrittenPlanTest, AnActiveContactOffTheTerrainFails) {
  absl::StatusOr<Fixture> test = fixture();
  ASSERT_TRUE(test.ok()) << test.status();
  test->plan.knots[0].active_contacts |=
      contactBit(*test->loaded.contactSet.indexOf("knee_l_contact"));
  const absl::StatusOr<config::CertifierReport> report =
      test->certifier->certify(test->plan);
  ASSERT_TRUE(report.ok()) << report.status();
  EXPECT_FALSE(report->knots[0].dynamic_condition);
  EXPECT_GT(report->knots[0].max_active_contact_distance, 0.2);
}

TEST(HandWrittenPlanTest, WithoutAStaticMarginOnlyConditionTwoIsJudged) {
  absl::StatusOr<Fixture> test = fixture();
  ASSERT_TRUE(test.ok()) << test.status();
  test->loaded.scenario.static_margin.reset();
  const absl::StatusOr<std::unique_ptr<StabilityCertifier>> certifier =
      makeCertifier(test->loaded);
  ASSERT_TRUE(certifier.ok()) << certifier.status();
  test->plan.knots[3].active_contacts =
      contactBit(*test->loaded.contactSet.indexOf("foot_l_contact"));
  const absl::StatusOr<config::CertifierReport> report =
      (*certifier)->certify(test->plan);
  ASSERT_TRUE(report.ok()) << report.status();
  EXPECT_FALSE(report->static_margin.has_value());
  for (const config::CertifierReport::Knot& knot : report->knots) {
    EXPECT_FALSE(knot.static_condition.has_value());
    EXPECT_FALSE(knot.transition_condition.has_value());
  }
  // On one foot with the CoM beside it, the momentum rate of standing still
  // is not in the cone either: the plan still fails, by condition 2.
  EXPECT_FALSE(report->knots[3].dynamic_condition);
  EXPECT_TRUE(report->knots[0].passed);
}

TEST(HandWrittenPlanTest, TheTorquesAreJudgedOnlyUnderPositionAndEffort) {
  absl::StatusOr<Fixture> test = fixture();
  ASSERT_TRUE(test.ok()) << test.status();
  // A torque margin the sway cannot keep: standing takes more than 1 % of
  // some joint's effort limit.
  constexpr double kTinyTorqueMargin = 0.01;
  test->loaded.scenario.torque_margin = kTinyTorqueMargin;

  test->loaded.scenario.joint_limits = "position_and_effort";
  const absl::StatusOr<std::unique_ptr<StabilityCertifier>> enforcing =
      makeCertifier(test->loaded);
  ASSERT_TRUE(enforcing.ok()) << enforcing.status();
  const absl::StatusOr<config::CertifierReport> judged =
      (*enforcing)->certify(test->plan);
  ASSERT_TRUE(judged.ok()) << judged.status();
  EXPECT_FALSE(judged->passed);
  EXPECT_EQ(judged->joint_limits, "position_and_effort");
  EXPECT_EQ(judged->torque_margin, kTinyTorqueMargin);
  for (const config::CertifierReport::Knot& knot : judged->knots) {
    EXPECT_EQ(knot.static_condition, false);
    ASSERT_FALSE(knot.failures.empty());
    EXPECT_TRUE(absl::StrContains(knot.failures.back(), "torque utilization"))
        << knot.failures.back();
  }
  EXPECT_TRUE(absl::StrContains(formatReport(*judged), "torque margin 0.01"));

  test->loaded.scenario.joint_limits = "position_only";
  const absl::StatusOr<std::unique_ptr<StabilityCertifier>> relaxed =
      makeCertifier(test->loaded);
  ASSERT_TRUE(relaxed.ok()) << relaxed.status();
  const absl::StatusOr<config::CertifierReport> reported =
      (*relaxed)->certify(test->plan);
  ASSERT_TRUE(reported.ok()) << reported.status();
  EXPECT_TRUE(reported->passed) << formatReport(*reported);
  EXPECT_EQ(reported->joint_limits, "position_only");
  EXPECT_FALSE(reported->torque_margin.has_value());
  ASSERT_EQ(reported->knots.size(), judged->knots.size());
  for (size_t k = 0; k < reported->knots.size(); ++k) {
    const config::CertifierReport::Knot& knot = reported->knots[k];
    EXPECT_EQ(knot.static_condition, true);
    // The utilization is still reported, the same as when it is judged.
    ASSERT_TRUE(knot.torque_utilization.has_value());
    EXPECT_GT(*knot.torque_utilization, kTinyTorqueMargin);
    EXPECT_EQ(knot.torque_utilization, judged->knots[k].torque_utilization);
  }
  EXPECT_TRUE(absl::StrContains(formatReport(*reported),
                                "torques not judged: joint limits "
                                "position_only"));
}

TEST(HandWrittenPlanTest, PositionOnlyStillJudgesTheStaticMargin) {
  absl::StatusOr<Fixture> test = fixture();
  ASSERT_TRUE(test.ok()) << test.status();
  test->loaded.scenario.joint_limits = "position_only";
  const absl::StatusOr<std::unique_ptr<StabilityCertifier>> certifier =
      makeCertifier(test->loaded);
  ASSERT_TRUE(certifier.ok()) << certifier.status();
  test->plan.knots[3].active_contacts =
      contactBit(*test->loaded.contactSet.indexOf("foot_l_contact"));
  const absl::StatusOr<config::CertifierReport> report =
      (*certifier)->certify(test->plan);
  ASSERT_TRUE(report.ok()) << report.status();
  EXPECT_FALSE(report->passed);
  EXPECT_EQ(report->knots[3].static_condition, false);
  EXPECT_EQ(report->knots[3].transition_condition, false);
  ASSERT_FALSE(report->knots[3].failures.empty());
  for (const std::string& failure : report->knots[3].failures) {
    EXPECT_FALSE(absl::StrContains(failure, "torque")) << failure;
  }
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
