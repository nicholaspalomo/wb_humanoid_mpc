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

// MultiContactPlannerRegistry: the built-in planners, every name of the
// design's list built and every built name a design name, the refusal of an
// unknown name, and a planner added by name.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_multi_contact/planner/ContactInvariantPlanner.h"
#include "humanoid_multi_contact/planner/MultiContactPlanner.h"
#include "humanoid_multi_contact/planner/MultiContactPlannerRegistry.h"
#include "humanoid_multi_contact/planner/StanceGraphPlanner.h"
#include "humanoid_multi_contact/planner/StanceGraphThenContactInvariantPlanner.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"

namespace ocs2::humanoid::multi_contact {
namespace {

TEST(MultiContactPlannerRegistry, BuildsTheStanceGraphPlanner) {
  const MultiContactPlannerRegistry registry;
  EXPECT_TRUE(registry.has("stance_graph"));
  const absl::StatusOr<std::unique_ptr<MultiContactPlanner>> planner =
      registry.create("stance_graph");
  ASSERT_TRUE(planner.ok()) << planner.status();
  EXPECT_EQ((*planner)->name(), kStanceGraphPlannerName);
}

TEST(MultiContactPlannerRegistry, BuildsEveryPlannerOfTheDesign) {
  const MultiContactPlannerRegistry registry;
  for (const absl::string_view name : kPlannerNames) {
    const absl::StatusOr<std::unique_ptr<MultiContactPlanner>> planner =
        registry.create(name);
    ASSERT_TRUE(planner.ok()) << name << ": " << planner.status();
    EXPECT_EQ((*planner)->name(), name);
  }
  EXPECT_EQ(kContactInvariantPlannerName, "contact_invariant");
  EXPECT_EQ(kStanceGraphThenContactInvariantPlannerName,
            "stance_graph_then_contact_invariant");
}

TEST(MultiContactPlannerRegistry, EveryNameIsADesignName) {
  const MultiContactPlannerRegistry registry;
  for (const std::string& name : registry.names()) {
    EXPECT_NE(std::find(kPlannerNames.begin(), kPlannerNames.end(), name),
              kPlannerNames.end())
        << name;
  }
}

TEST(MultiContactPlannerRegistry, RefusesUnknownNames) {
  const MultiContactPlannerRegistry registry;
  const absl::StatusOr<std::unique_ptr<MultiContactPlanner>> unknown =
      registry.create("rrt");
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(unknown.status().message(), "stance_graph"));
}

TEST(MultiContactPlannerRegistry, AddsAPlannerByName) {
  MultiContactPlannerRegistry registry;
  registry.add("contact_invariant", "a stand-in", []() {
    return absl::StatusOr<std::unique_ptr<MultiContactPlanner>>(
        absl::UnimplementedError("a stand-in"));
  });
  EXPECT_TRUE(registry.has("contact_invariant"));
  EXPECT_EQ(registry.create("contact_invariant").status().code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_TRUE(absl::StrContains(registry.availableNames(), "a stand-in"));
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
