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

// The shipped scenarios (design section 8.1): every file parses, references
// existing contacts and primitives and carries the design's static margin, and
// its initial and goal stances pass the stability certifier.

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"
#include "humanoid_multi_contact/scenarios/scenario.nproto.h"
#include "humanoid_multi_contact/stability/JointLimits.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_multi_contact/stability/StabilityCertifier.h"
#include "humanoid_multi_contact/stability/StaticEquilibrium.h"
#include "humanoid_multi_contact/stability/certifier_report.nproto.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr char kScenarioDirectory[] =
    "humanoid_nmpc/humanoid_multi_contact/scenarios/config/";

/** A scenario and its expected static margin (design section 8.1): none for
 * the stairs, 0.03 m elsewhere. */
struct ExpectedScenario {
  std::string name;
  std::optional<double> staticMargin;
};

const std::vector<ExpectedScenario>& expectedScenarios() {
  static const std::vector<ExpectedScenario>* absl_nonnull const kScenarios =
      new std::vector<ExpectedScenario>{
          {.name = "stairs", .staticMargin = std::nullopt},
          {.name = "stairs_steep", .staticMargin = std::nullopt},
          {.name = "box_climb", .staticMargin = 0.03},
          {.name = "box_climb_high", .staticMargin = 0.03},
          {.name = "car_egress", .staticMargin = 0.03},
          {.name = "get_up_supine", .staticMargin = 0.03},
          {.name = "get_up_prone", .staticMargin = 0.03}};
  return *kScenarios;
}

absl::StatusOr<LoadedScenario> loadNamedScenario(const std::string& name) {
  ASSIGN_OR_RETURN(const std::string path,
                   robot::resolveResourcePath(
                       absl::StrCat(kScenarioDirectory, name, ".textproto")));
  return loadScenario(path);
}

/** The certifier's report on `pose` held still. */
absl::StatusOr<config::CertifierReport::Knot> certifyPose(
    StabilityCertifier& certifier,
    const config::Scenario::Pose& pose,
    const ContactSet& contactSet) {
  CertifierKnot knot;
  ASSIGN_OR_RETURN(knot.configuration, configurationOf(pose));
  ASSIGN_OR_RETURN(knot.activeContacts, contactsOf(pose, contactSet));
  ASSIGN_OR_RETURN(const config::CertifierReport report,
                   certifier.certifyKnots({knot}));
  return report.knots.front();
}

void expectStable(const config::CertifierReport::Knot& knot,
                  const std::string& what) {
  SCOPED_TRACE(what);
  EXPECT_TRUE(knot.passed) << absl::StrJoin(knot.failures, "; ");
  EXPECT_TRUE(knot.dynamic_condition);
  EXPECT_LE(knot.max_active_contact_distance, kDefaultContactDistanceTolerance);
  ASSERT_TRUE(knot.static_margin.has_value());
  // Every shipped pose stands with the design's margin, whether or not its
  // scenario judges it.
  EXPECT_GE(*knot.static_margin, 0.03);
  ASSERT_TRUE(knot.torque_utilization.has_value());
  EXPECT_LE(*knot.torque_utilization, 0.9) << knot.max_torque_joint;
  EXPECT_TRUE(knot.collision_free.value_or(false))
      << knot.min_sphere_clearance_link;
  // No inactive pad's geometry sinks into the terrain by more than 1 cm, and
  // no collision sphere by more than the contact tolerance (`passed`).
  if (knot.min_inactive_clearance.has_value()) {
    EXPECT_GE(*knot.min_inactive_clearance, -0.01)
        << knot.min_inactive_clearance_contact;
  }
}

TEST(ScenariosTest, EveryScenarioParsesAndReferencesItsContactsAndTerrain) {
  for (const ExpectedScenario& expected : expectedScenarios()) {
    SCOPED_TRACE(expected.name);
    const absl::StatusOr<LoadedScenario> loaded =
        loadNamedScenario(expected.name);
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    EXPECT_EQ(loaded->scenario.name, expected.name);
    EXPECT_TRUE(
        validateScenario(loaded->scenario, loaded->contactSet, loaded->terrain)
            .ok())
        << validateScenario(loaded->scenario, loaded->contactSet,
                            loaded->terrain);
    EXPECT_EQ(loaded->scenario.static_margin, expected.staticMargin);
    EXPECT_GT(loaded->scenario.torque_margin, 0.0);
    // Every shipped scenario plans within the position limits only (design
    // section 11, decision 6), the box climb's and the get-up from the front's
    // widened by 0.1 rad (decision 8), the get-up from the back's by 0.05 rad
    // (decision 11).
    if (expected.name == "box_climb" || expected.name == "get_up_prone" ||
        expected.name == "get_up_supine") {
      EXPECT_EQ(loaded->scenario.joint_limits, "relaxed_position");
      EXPECT_EQ(loaded->scenario.joint_limit_relaxation,
                expected.name == "get_up_supine" ? 0.05 : 0.1);
    } else {
      EXPECT_EQ(loaded->scenario.joint_limits, "position_only");
      EXPECT_EQ(*jointLimitsOf(loaded->scenario.joint_limits),
                JointLimits::kPositionOnly);
    }
    EXPECT_FALSE(
        enforcesEffortLimits(*jointLimitsOf(loaded->scenario.joint_limits)));
    // The kneels on the floor let the thigh's end rest; every other scenario
    // exempts it (design section 5.4).
    EXPECT_EQ(loaded->scenario.point_pad_link_above,
              absl::StartsWith(expected.name, "get_up_") ? "rests" : "exempt");
    // Only the roll on the torso's edge needs the refined polygon.
    EXPECT_EQ(loaded->scenario.support_polygon,
              expected.name == "get_up_supine" ? "refined" : "sampled");
    // Only the get-up from the back keeps the spheres farther off the floor
    // than the planner's own 1 cm.
    EXPECT_EQ(loaded->scenario.sphere_clearance.has_value(),
              expected.name == "get_up_supine");
    // The roll's contacts are the get-up from the back's alone.
    for (const std::string& contact : loaded->scenario.allowed_contacts) {
      const bool roll = absl::StartsWith(contact, "torso_edge_");
      EXPECT_TRUE(!roll || expected.name == "get_up_supine") << contact;
    }
  }
}

TEST(ScenariosTest, TheJointLimitsDefaultToEnforcingTheEffortLimits) {
  const config::Scenario scenario;
  EXPECT_EQ(scenario.joint_limits, "position_and_effort");
  const absl::StatusOr<CertifierOptions> options = certifierOptionsOf(scenario);
  ASSERT_TRUE(options.ok()) << options.status();
  EXPECT_EQ(options->jointLimits, JointLimits::kPositionAndEffort);
  config::Scenario positionOnly;
  positionOnly.joint_limits = "position_only";
  positionOnly.torque_margin = 0.5;
  const absl::StatusOr<CertifierOptions> relaxed =
      certifierOptionsOf(positionOnly);
  ASSERT_TRUE(relaxed.ok()) << relaxed.status();
  EXPECT_EQ(relaxed->jointLimits, JointLimits::kPositionOnly);
  EXPECT_EQ(relaxed->torqueMargin, 0.5);
  positionOnly.joint_limits = "velocity_only";
  EXPECT_EQ(certifierOptionsOf(positionOnly).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ScenariosTest, TheCertifierTakesTheScenariosSupportPolygon) {
  config::Scenario scenario;
  const absl::StatusOr<CertifierOptions> sampled = certifierOptionsOf(scenario);
  ASSERT_TRUE(sampled.ok()) << sampled.status();
  EXPECT_EQ(sampled->equilibrium.supportPolygon, SupportPolygon::kSampled);
  scenario.support_polygon = "refined";
  const absl::StatusOr<CertifierOptions> refined = certifierOptionsOf(scenario);
  ASSERT_TRUE(refined.ok()) << refined.status();
  EXPECT_EQ(refined->equilibrium.supportPolygon, SupportPolygon::kRefined);
  scenario.support_polygon = "exact";
  EXPECT_EQ(certifierOptionsOf(scenario).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ScenariosTest, TheAllowedContactsAreTheDesigns) {
  const absl::StatusOr<LoadedScenario> egress = loadNamedScenario("car_egress");
  ASSERT_TRUE(egress.ok()) << egress.status();
  // All but the two torso pads.
  EXPECT_EQ(egress->scenario.allowed_contacts.size(), 11u);
  for (const std::string& contact : egress->scenario.allowed_contacts) {
    EXPECT_FALSE(absl::StartsWith(contact, "torso_")) << contact;
  }
  // The get-up from the back may use every candidate, the roll's included;
  // every other scenario none of the roll's (the thirteen candidates before
  // them).
  constexpr int kRollContacts = 4;
  for (const std::string& name :
       std::vector<std::string>{"get_up_supine", "get_up_prone"}) {
    const absl::StatusOr<LoadedScenario> getUp = loadNamedScenario(name);
    ASSERT_TRUE(getUp.ok()) << getUp.status();
    EXPECT_EQ(static_cast<int>(getUp->scenario.allowed_contacts.size()),
              name == "get_up_supine"
                  ? getUp->contactSet.size()
                  : getUp->contactSet.size() - kRollContacts);
    EXPECT_EQ(getUp->scenario.simulation_start.tilt_catch_arming,
              "when_standing");
    EXPECT_EQ(getUp->scenario.simulation_start.gantry_arming, "when_standing");
  }
  const absl::StatusOr<LoadedScenario> stairs = loadNamedScenario("stairs");
  ASSERT_TRUE(stairs.ok());
  EXPECT_EQ(stairs->scenario.allowed_contacts.size(), 2u);
  EXPECT_EQ(stairs->scenario.simulation_start.tilt_catch_arming, "at_start");
}

TEST(ScenariosTest, InitialAndGoalStancesPassTheCertifier) {
  for (const ExpectedScenario& expected : expectedScenarios()) {
    SCOPED_TRACE(expected.name);
    const absl::StatusOr<LoadedScenario> loaded =
        loadNamedScenario(expected.name);
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    const absl::StatusOr<std::unique_ptr<StabilityCertifier>> certifier =
        makeCertifier(*loaded);
    ASSERT_TRUE(certifier.ok()) << certifier.status();
    const absl::StatusOr<config::CertifierReport::Knot> initial = certifyPose(
        **certifier, loaded->scenario.initial_pose, loaded->contactSet);
    ASSERT_TRUE(initial.ok()) << initial.status();
    expectStable(*initial, "initial pose");
    EXPECT_EQ(initial->static_condition.has_value(),
              expected.staticMargin.has_value());
    ASSERT_TRUE(loaded->scenario.goal.pose.has_value());
    const absl::StatusOr<config::CertifierReport::Knot> goal = certifyPose(
        **certifier, *loaded->scenario.goal.pose, loaded->contactSet);
    ASSERT_TRUE(goal.ok()) << goal.status();
    expectStable(*goal, "goal pose");
  }
}

TEST(ScenariosTest, AKneelingWaypointRestsTheEndOfItsThighBesideTheKnee) {
  // box_climb's left knee on the box: the end of the thigh, at the knee, rests
  // on the top beside the knee's pad, which exempts it in contact and while
  // it rests there with its contact broken (CollisionModel::
  // terrainClearance()).
  const absl::StatusOr<LoadedScenario> loaded = loadNamedScenario("box_climb");
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  const absl::StatusOr<std::unique_ptr<StabilityCertifier>> certifier =
      makeCertifier(*loaded);
  ASSERT_TRUE(certifier.ok()) << certifier.status();
  const config::Scenario::Pose* absl_nullable kneeling = nullptr;
  for (const config::Scenario::Goal& waypoint : loaded->scenario.waypoints) {
    if (waypoint.pose.has_value() &&
        waypoint.pose->name == "left_knee_on_the_box") {
      kneeling = &*waypoint.pose;
    }
  }
  ASSERT_NE(kneeling, nullptr);
  const absl::StatusOr<RobotConfiguration> configuration =
      configurationOf(*kneeling);
  ASSERT_TRUE(configuration.ok()) << configuration.status();
  const absl::StatusOr<ContactMask> contacts =
      contactsOf(*kneeling, loaded->contactSet);
  ASSERT_TRUE(contacts.ok()) << contacts.status();
  CertifierKnot knot;
  knot.configuration = *configuration;
  knot.activeContacts = *contacts;
  const absl::StatusOr<ContactMask> knee =
      loaded->contactSet.maskOf({"knee_l_contact"});
  ASSERT_TRUE(knee.ok()) << knee.status();
  CertifierKnot broken = knot;
  broken.activeContacts &= ~*knee;
  for (const CertifierKnot& judged : {knot, broken}) {
    const absl::StatusOr<config::CertifierReport> report =
        (*certifier)->certifyKnots({judged});
    ASSERT_TRUE(report.ok()) << report.status();
    EXPECT_TRUE(report->knots.front().collision_free.value_or(false))
        << report->knots.front().min_sphere_clearance_link;
  }
}

void expectRefused(const LoadedScenario& loaded,
                   const config::Scenario& scenario,
                   const std::string& message) {
  const absl::Status status =
      validateScenario(scenario, loaded.contactSet, loaded.terrain);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(status.message(), message)) << status;
}

TEST(ScenariosTest, ValidationRefusesBadScenarios) {
  const absl::StatusOr<LoadedScenario> loaded = loadNamedScenario("box_climb");
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  config::Scenario planner = loaded->scenario;
  planner.planner = "teleportation";
  expectRefused(*loaded, planner, "unknown planner");

  config::Scenario linkAbove = loaded->scenario;
  linkAbove.point_pad_link_above = "whole";
  expectRefused(*loaded, linkAbove, "unknown point_pad_link_above");
  config::Scenario clearance = loaded->scenario;
  clearance.sphere_clearance = 0.0;
  expectRefused(*loaded, clearance, "sphere_clearance must be positive");
  config::Scenario polygon = loaded->scenario;
  polygon.support_polygon = "exact";
  expectRefused(*loaded, polygon, "unknown support_polygon");
  config::Scenario contact = loaded->scenario;
  contact.allowed_contacts.push_back("tail_contact");
  expectRefused(*loaded, contact, "not a candidate");
  config::Scenario notAllowed = loaded->scenario;
  notAllowed.initial_pose.active_contacts.push_back("seat_contact");
  expectRefused(*loaded, notAllowed, "is not allowed");
  config::Scenario primitive = loaded->scenario;
  primitive.goal.contacts.front().primitive = "roof";
  expectRefused(*loaded, primitive, "unknown primitive");
  config::Scenario waypoint = loaded->scenario;
  waypoint.waypoints.push_back(loaded->scenario.goal);
  EXPECT_TRUE(
      validateScenario(waypoint, loaded->contactSet, loaded->terrain).ok());
  waypoint.waypoints.front().contacts.front().primitive = "roof";
  expectRefused(*loaded, waypoint, "waypoints[0]: contact");
  waypoint.waypoints.front() = loaded->scenario.goal;
  waypoint.waypoints.front().contacts.front().contact = "seat_contact";
  expectRefused(*loaded, waypoint, "waypoints[0].contacts: 'seat_contact'");
  waypoint.waypoints.front().contacts.clear();
  expectRefused(*loaded, waypoint, "waypoints[0] names no contact");
  config::Scenario margin = loaded->scenario;
  margin.static_margin = -0.01;
  expectRefused(*loaded, margin, "static_margin");
  config::Scenario arming = loaded->scenario;
  arming.simulation_start.gantry_arming = "when_standing";
  expectRefused(*loaded, arming, "standing_base_height");
  arming.simulation_start.gantry_arming = "never";
  expectRefused(*loaded, arming, "unknown arming");
  config::Scenario limits = loaded->scenario;
  limits.joint_limits = "none";
  expectRefused(*loaded, limits, "unknown joint limits");
  config::Scenario relaxed = loaded->scenario;
  relaxed.joint_limits = "relaxed_position";
  relaxed.joint_limit_relaxation = 0.0;
  expectRefused(*loaded, relaxed, "joint_limit_relaxation");
  relaxed.joint_limit_relaxation = 0.5;
  expectRefused(*loaded, relaxed, "joint_limit_relaxation");
  relaxed.joint_limit_relaxation = 0.1;
  EXPECT_TRUE(
      validateScenario(relaxed, loaded->contactSet, loaded->terrain).ok());
  relaxed.joint_limits = "position_only";
  expectRefused(*loaded, relaxed, "joint_limit_relaxation");
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
