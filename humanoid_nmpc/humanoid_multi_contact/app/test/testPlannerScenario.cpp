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

// A multi-contact planner, named by the environment variable
// MULTI_CONTACT_PLANNER (a registry name), on one shipped scenario, named by
// MULTI_CONTACT_SCENARIO (the stem of its file in scenarios/config): it plans,
// the certifier passes at every knot, the plan is stamped with the scenario's
// hashes, and its last knot reaches the goal: the goal stance on its
// primitives, the base in the goal region. On the box the plan uses more than
// the feet (a knee, a forearm or a hand); on the stairs only the feet (design
// section 9). Planner A's plans last the scenario's duration.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_multi_contact/app/PlanMultiContact.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/plan/PlanFile.h"
#include "humanoid_multi_contact/planner/ContactInvariantPlanner.h"
#include "humanoid_multi_contact/planner/MultiContactPlannerRegistry.h"
#include "humanoid_multi_contact/planner/StanceGraphPlanner.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid::multi_contact {
namespace {

/** The environment variable `variable`; empty when it is not set. */
std::string environment(const char* absl_nonnull variable) {
  const char* absl_nullable value = std::getenv(variable);
  return value == nullptr ? std::string() : std::string(value);
}

TEST(PlannerScenario, PlansACertifiedMotionToTheGoal) {
  const std::string name = environment("MULTI_CONTACT_SCENARIO");
  ASSERT_FALSE(name.empty()) << "set MULTI_CONTACT_SCENARIO";
  const std::string plannerName = environment("MULTI_CONTACT_PLANNER");
  ASSERT_FALSE(plannerName.empty()) << "set MULTI_CONTACT_PLANNER";
  const absl::StatusOr<std::string> path = robot::resolveResourcePath(
      absl::StrCat("humanoid_nmpc/humanoid_multi_contact/scenarios/config/",
                   name, ".textproto"));
  ASSERT_TRUE(path.ok()) << path.status();
  const absl::StatusOr<LoadedScenario> loaded = loadScenario(*path);
  ASSERT_TRUE(loaded.ok()) << loaded.status();

  const MultiContactPlannerRegistry registry;
  absl::StatusOr<std::unique_ptr<MultiContactPlanner>> planner =
      registry.create(plannerName);
  ASSERT_TRUE(planner.ok()) << planner.status();
  const absl::StatusOr<config::MultiContactPlan> plan =
      (*planner)->plan(*loaded);
  ASSERT_TRUE(plan.ok()) << plan.status();

  // Certified, at every knot, for this scenario's contact set and scene.
  ASSERT_TRUE(plan->report.has_value());
  EXPECT_TRUE(plan->report->passed);
  for (const config::CertifierReport::Knot& knot : plan->report->knots) {
    EXPECT_TRUE(knot.passed) << "t = " << knot.time;
  }
  EXPECT_TRUE(checkPlanHashes(*plan, loaded->contactSetConfig,
                              loaded->scenario.terrain_scene)
                  .ok());
  EXPECT_EQ(plan->planner, plannerName);
  if (plannerName != kStanceGraphPlannerName) {
    // Planner A optimizes the motion over the scenario's duration.
    EXPECT_NEAR(plan->knots.back().time - plan->knots.front().time,
                loaded->scenario.duration, 1.0e-6);
  }

  // The last knot is the goal stance, each contact on its primitive.
  std::vector<std::string> goalNames;
  for (const config::Scenario::GoalContact& goal :
       loaded->scenario.goal.contacts) {
    goalNames.push_back(goal.contact);
    bool onPrimitive = false;
    for (const config::MultiContactPlan::ContactPhase& phase : plan->phases) {
      if (phase.contact == goal.contact &&
          phase.end_time == plan->knots.back().time) {
        onPrimitive = phase.primitive == goal.primitive;
      }
    }
    EXPECT_TRUE(onPrimitive) << goal.contact << " ends on " << goal.primitive;
  }
  const absl::StatusOr<ContactMask> goalMask =
      loaded->contactSet.maskOf(goalNames);
  ASSERT_TRUE(goalMask.ok()) << goalMask.status();
  EXPECT_EQ(plan->knots.back().active_contacts, *goalMask);
  if (loaded->scenario.goal.base_region.has_value()) {
    const Eigen::Vector3d offset =
        (Eigen::Vector3d(plan->knots.back().base_position) -
         Eigen::Vector3d(loaded->scenario.goal.base_region->center))
            .cwiseAbs();
    EXPECT_TRUE(
        (offset.array() <=
         Eigen::Vector3d(loaded->scenario.goal.base_region->half_extents)
             .array())
            .all())
        << "the base ends at "
        << Eigen::Vector3d(plan->knots.back().base_position).transpose();
  }

  const absl::StatusOr<PlanSummary> summary =
      summarizePlan(*plan, loaded->contactSet);
  ASSERT_TRUE(summary.ok()) << summary.status();
  std::cerr << name << ": " << formatPlanSummary(*summary) << "\n";
  bool beyondFeet = false;
  for (const std::string& contact : summary->contactsUsed) {
    if (!absl::StartsWith(contact, "foot_")) beyondFeet = true;
  }
  if (absl::StartsWith(name, "box_climb")) {
    // More than walking: a knee, a forearm or a hand.
    EXPECT_TRUE(beyondFeet) << formatPlanSummary(*summary);
  }
  if (name == "stairs") {
    EXPECT_FALSE(beyondFeet) << formatPlanSummary(*summary);
  }
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
