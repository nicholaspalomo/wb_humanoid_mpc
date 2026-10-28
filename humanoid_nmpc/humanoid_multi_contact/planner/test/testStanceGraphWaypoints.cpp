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

// The stance graph planner's waypoints (design section 5.5): Atlas, standing
// in front of the box_climb scenario's box, plans to stand where it starts
// through a waypoint with both hands on the box. Without the waypoint the goal
// is the start; with it the plan puts both hands on the box, takes them off,
// and ends on the feet, every knot certified. A waypoint the robot cannot
// reach ends the search with NotFound naming it.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/planner/StanceGraphPlanner.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr char kBoxClimbScenario[] =
    "humanoid_nmpc/humanoid_multi_contact/scenarios/config/box_climb.textproto";

/** The box_climb scene with the goal of standing where the robot starts, on
 * the floor, and no waypoint. */
LoadedScenario standingInFrontOfTheBox() {
  const absl::StatusOr<std::string> path =
      robot::resolveResourcePath(kBoxClimbScenario);
  EXPECT_TRUE(path.ok()) << path.status();
  absl::StatusOr<LoadedScenario> loaded = loadScenario(*path);
  EXPECT_TRUE(loaded.ok()) << loaded.status();
  config::Scenario& scenario = loaded->scenario;
  scenario.waypoints.clear();
  scenario.goal.contacts.clear();
  for (const char* absl_nonnull foot : {"foot_l_contact", "foot_r_contact"}) {
    config::Scenario::GoalContact contact;
    contact.contact = foot;
    contact.primitive = "ground";
    scenario.goal.contacts.push_back(contact);
  }
  scenario.goal.pose = scenario.initial_pose;
  config::Scenario::BaseRegion region;
  region.center = Eigen::Vector3d(scenario.initial_pose.base_position);
  region.half_extents = Eigen::Vector3d::Constant(0.1);
  scenario.goal.base_region = region;
  return *std::move(loaded);
}

/** A waypoint of `stance`'s contacts on their primitives. */
config::Scenario::Goal waypointOf(
    const std::vector<std::pair<std::string, std::string>>& stance) {
  config::Scenario::Goal waypoint;
  for (const std::pair<std::string, std::string>& entry : stance) {
    config::Scenario::GoalContact contact;
    contact.contact = entry.first;
    contact.primitive = entry.second;
    waypoint.contacts.push_back(contact);
  }
  return waypoint;
}

TEST(StanceGraphWaypoints, aPlanPassesThroughItsWaypoint) {
  LoadedScenario loaded = standingInFrontOfTheBox();
  loaded.scenario.waypoints.push_back(waypointOf({{"foot_l_contact", "ground"},
                                                  {"foot_r_contact", "ground"},
                                                  {"hand_l_contact", "box"},
                                                  {"hand_r_contact", "box"}}));
  ASSERT_TRUE(
      validateScenario(loaded.scenario, loaded.contactSet, loaded.terrain)
          .ok());
  absl::StatusOr<std::unique_ptr<StanceGraphPlanner>> planner =
      StanceGraphPlanner::Create(StanceGraphPlannerOptions());
  ASSERT_TRUE(planner.ok()) << planner.status();
  const absl::StatusOr<config::MultiContactPlan> plan =
      (*planner)->plan(loaded);
  ASSERT_TRUE(plan.ok()) << plan.status();
  ASSERT_TRUE(plan->report.has_value());
  EXPECT_TRUE(plan->report->passed);

  const absl::StatusOr<ContactMask> waypoint = loaded.contactSet.maskOf(
      {"foot_l_contact", "foot_r_contact", "hand_l_contact", "hand_r_contact"});
  const absl::StatusOr<ContactMask> feet =
      loaded.contactSet.maskOf({"foot_l_contact", "foot_r_contact"});
  ASSERT_TRUE(waypoint.ok() && feet.ok());
  bool passed = false;
  for (const config::MultiContactPlan::Knot& knot : plan->knots) {
    passed = passed || knot.active_contacts == *waypoint;
  }
  EXPECT_TRUE(passed) << "no knot holds both hands on the box";
  EXPECT_EQ(plan->knots.front().active_contacts, *feet);
  EXPECT_EQ(plan->knots.back().active_contacts, *feet);
  // The hands were on the box, not anywhere.
  for (const config::MultiContactPlan::ContactPhase& phase : plan->phases) {
    if (absl::StartsWith(phase.contact, "hand_")) {
      EXPECT_EQ(phase.primitive, "box") << phase.contact;
    }
  }
  // Planner B alone breaks a contact at the end of its dwell: the knot that
  // first lacks it is where the one before it was, at rest
  // (ContactBreak::kAtDwellEnd), and the contact's phase ends there.
  int breaks = 0;
  for (size_t k = 1; k < plan->knots.size(); ++k) {
    const ContactMask before = plan->knots[k - 1].active_contacts;
    const ContactMask after = plan->knots[k].active_contacts;
    if ((before & ~after) == 0) continue;
    ++breaks;
    EXPECT_EQ(plan->knots[k].joint_positions,
              plan->knots[k - 1].joint_positions)
        << "t = " << plan->knots[k].time;
    bool phaseEnds = false;
    for (const config::MultiContactPlan::ContactPhase& phase : plan->phases) {
      phaseEnds = phaseEnds || phase.end_time == plan->knots[k].time;
    }
    EXPECT_TRUE(phaseEnds) << "t = " << plan->knots[k].time;
  }
  EXPECT_EQ(breaks, 2);  // The two hands.
}

TEST(StanceGraphWaypoints, anUnreachableWaypointIsNamed) {
  LoadedScenario loaded = standingInFrontOfTheBox();
  // A knee on the box's top while both feet stay on the floor: no
  // configuration holds it.
  loaded.scenario.waypoints.push_back(waypointOf({{"foot_l_contact", "ground"},
                                                  {"foot_r_contact", "ground"},
                                                  {"knee_l_contact", "box"},
                                                  {"knee_r_contact", "box"}}));
  StanceGraphPlannerOptions options;
  options.search.maxValidations = 40;
  absl::StatusOr<std::unique_ptr<StanceGraphPlanner>> planner =
      StanceGraphPlanner::Create(options);
  ASSERT_TRUE(planner.ok()) << planner.status();
  const absl::StatusOr<config::MultiContactPlan> plan =
      (*planner)->plan(loaded);
  EXPECT_EQ(plan.status().code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(plan.status().message(), "waypoint 1 of 1"))
      << plan.status();
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
