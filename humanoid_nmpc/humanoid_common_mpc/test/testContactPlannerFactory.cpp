/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"

namespace ocs2::humanoid {
namespace {

ContactPlanningConfig makeConfig() {
  ContactPlanningConfig config;
  config.planner.dt = 0.05;
  config.planner.numNodes = 24;
  config.validate();
  return config;
}

TEST(ContactPlannerFactory, buildsEveryKnownPlanner) {
  for (const std::string& name : knownPlannerNames()) {
    ContactPlanningConfig config = makeConfig();
    config.planner.type = name;
    const absl::StatusOr<std::unique_ptr<ContactPlannerInterface>> planner = makeContactPlanner(config);
    ASSERT_TRUE(planner.ok()) << name << ": " << planner.status().message();
    EXPECT_NE(*planner, nullptr);
    EXPECT_FALSE((*planner)->getFormulationSummary().empty());
  }
}

TEST(ContactPlannerFactory, nameMatchingIgnoresCaseAndSeparators) {
  EXPECT_EQ(canonicalPlannerName("HLIP"), planner::kHlip);
  EXPECT_EQ(canonicalPlannerName("lipMiqp"), planner::kLipMiqp);
  EXPECT_EQ(canonicalPlannerName("LIP-MIQP"), planner::kLipMiqp);
  EXPECT_TRUE(canonicalPlannerName("nope").empty());
}

TEST(ContactPlannerFactory, unknownPlannerListsTheOnesThatExist) {
  ContactPlanningConfig config = makeConfig();
  config.planner.type = "hilp";  // a typo of hlip
  const absl::StatusOr<std::unique_ptr<ContactPlannerInterface>> planner = makeContactPlanner(config);
  ASSERT_FALSE(planner.ok());
  const std::string message(planner.status().message());
  EXPECT_NE(message.find("hilp"), std::string::npos);
  for (const std::string& name : knownPlannerNames()) {
    EXPECT_NE(message.find(name), std::string::npos) << "the error must name " << name;
  }
}

TEST(ContactPlannerFactory, summaryMatchesTheBuiltPlanner) {
  ContactPlanningConfig config = makeConfig();
  config.planner.type = planner::kHlip;
  const absl::StatusOr<std::string> summary = contactPlannerSummary(config);
  ASSERT_TRUE(summary.ok());
  const absl::StatusOr<std::unique_ptr<ContactPlannerInterface>> built = makeContactPlanner(config);
  ASSERT_TRUE(built.ok());
  EXPECT_EQ(*summary, (*built)->getFormulationSummary());
}

}  // namespace
}  // namespace ocs2::humanoid
