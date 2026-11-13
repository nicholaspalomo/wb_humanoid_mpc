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

#include <cstddef>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "legged_common_mpc/contact_planning/ContactPlannerInterface.h"
#include "legged_common_mpc/contact_planning/ContactPlanningConfig.h"

namespace ocs2::legged {
namespace {

ContactPlanningConfig makeConfig() {
  ContactPlanningConfig config;
  config.planner.dt = 0.05;
  config.planner.numNodes = 24;
  // No robot model derives the pendulum here (the library default 0 means "from
  // the model"), so it is set.
  config.shared.nomPosComRtFeetmeanEwrtWorldZ = 0.85;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  return config;
}

TEST(ContactPlannerFactory, buildsEveryKnownPlanner) {
  for (const std::string& name : knownPlannerNames()) {
    ContactPlanningConfig config = makeConfig();
    config.planner.type = name;
    const absl::StatusOr<std::unique_ptr<ContactPlannerInterface>> planner =
        makeContactPlanner(config, kBipedContactSetSummary);
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
  const absl::StatusOr<std::unique_ptr<ContactPlannerInterface>> planner =
      makeContactPlanner(config, kBipedContactSetSummary);
  ASSERT_FALSE(planner.ok());
  const std::string message(planner.status().message());
  EXPECT_NE(message.find("hilp"), std::string::npos);
  for (const std::string& name : knownPlannerNames()) {
    EXPECT_NE(message.find(name), std::string::npos)
        << "the error must name " << name;
  }
}

// An invalid configuration comes back as a Status that names the key to change,
// for every planner, and nothing throws: makeContactPlanner returns a StatusOr,
// and its callers (ContactPlannerModule, also on the planning thread) handle
// the Status, not an exception. The mixed-integer planner used to validate in
// its constructor by throwing, and the H-LIP one did not validate at all, so
// the same invalid key used to throw for one and build the other.
TEST(ContactPlannerFactory,
     invalidConfigurationIsAStatusThatNamesTheKeyForEveryPlanner) {
  for (const std::string& name : knownPlannerNames()) {
    ContactPlanningConfig config = makeConfig();
    config.planner.type = name;
    config.planner.dt = -0.05;
    absl::StatusOr<std::unique_ptr<ContactPlannerInterface>> planner =
        absl::InternalError("not built");
    EXPECT_NO_THROW(planner =
                        makeContactPlanner(config, kBipedContactSetSummary))
        << name;
    ASSERT_FALSE(planner.ok())
        << name << ": an invalid planner.dt must not build a planner";
    EXPECT_EQ(planner.status().code(), absl::StatusCode::kInvalidArgument)
        << name;
    EXPECT_NE(std::string(planner.status().message()).find("planner.dt"),
              std::string::npos)
        << name << ": " << planner.status().message();

    absl::StatusOr<std::string> summary = std::string();
    EXPECT_NO_THROW(summary =
                        contactPlannerSummary(config, kBipedContactSetSummary))
        << name;
    ASSERT_FALSE(summary.ok()) << name;
    EXPECT_NE(std::string(summary.status().message()).find("planner.dt"),
              std::string::npos)
        << name << ": " << summary.status().message();

    // Positive control: the same configuration with the key repaired builds.
    config.planner.dt = 0.05;
    EXPECT_TRUE(makeContactPlanner(config, kBipedContactSetSummary).ok())
        << name;
  }
}

TEST(ContactPlannerFactory, summaryMatchesTheBuiltPlanner) {
  ContactPlanningConfig config = makeConfig();
  config.planner.type = planner::kHlip;
  const absl::StatusOr<std::string> summary =
      contactPlannerSummary(config, kBipedContactSetSummary);
  ASSERT_TRUE(summary.ok());
  const absl::StatusOr<std::unique_ptr<ContactPlannerInterface>> built =
      makeContactPlanner(config, kBipedContactSetSummary);
  ASSERT_TRUE(built.ok());
  EXPECT_EQ(*summary, (*built)->getFormulationSummary());
}

// Both planners are written for a biped's two feet. A model with any other
// contact set (shared.numContacts, which
// ContactPlanningModelParameters::applyTo() writes from the model) is refused
// by the planner's name and the count, both when the planner is built and when
// its summary is asked for.
TEST(ContactPlannerFactory, everyPlannerRefusesAContactSetThatIsNotTwoFeet) {
  constexpr size_t kFourContacts = 4;
  for (const std::string& name : knownPlannerNames()) {
    ContactPlanningConfig config = makeConfig();
    config.planner.type = name;
    config.shared.numContacts = kFourContacts;
    const std::string expected = absl::StrCat(
        name == planner::kHlip ? "HlipContactPlanner" : "LipContactPlanner",
        " is written for two contacts; the contact set has 4 contacts");
    const absl::Status built =
        makeContactPlanner(config, kBipedContactSetSummary).status();
    EXPECT_EQ(built.code(), absl::StatusCode::kInvalidArgument) << name;
    EXPECT_TRUE(absl::StrContains(built.message(), expected)) << built;
    const absl::Status summarized =
        contactPlannerSummary(config, kBipedContactSetSummary).status();
    EXPECT_EQ(summarized.code(), absl::StatusCode::kInvalidArgument) << name;
    EXPECT_TRUE(absl::StrContains(summarized.message(), expected))
        << summarized;

    config.shared.numContacts = static_cast<size_t>(kBipedFootCount);
    EXPECT_TRUE(makeContactPlanner(config, kBipedContactSetSummary).ok())
        << name;
  }
}

// The contact set the factory is given: a set that is not the biped's two
// 6-DoF feet - four point feet, or two point feet, whose count the planners'
// own check accepts - is refused by planner.type and the contact count before
// anything is built, and with it the planners' step widths, which are positive
// by default.
TEST(ContactPlannerFactory, everyPlannerRefusesANonBipedContactSetByName) {
  constexpr ContactSetSummary kFourPointFeet = {.numContacts = 4,
                                                .isBiped = false};
  constexpr ContactSetSummary kTwoPointFeet = {.numContacts = 2,
                                               .isBiped = false};
  for (const std::string& name :
       {std::string(planner::kHlip), std::string(planner::kLipMiqp)}) {
    ContactPlanningConfig config = makeConfig();
    config.planner.type = name;
    ASSERT_GT(config.hlip.nomPosLfootRtRfootEwrtHeadingY, 0.0);
    ASSERT_GT(config.stepWidth.nomPosLfootRtRfootEwrtHeadingY, 0.0);
    for (const ContactSetSummary contactSet : {kFourPointFeet, kTwoPointFeet}) {
      const std::string expected = absl::StrCat(
          "planner.type '", name, "' is written for two 6-DoF feet");
      const std::string count = absl::StrCat(
          "the contact set has ", contactSet.numContacts, " contacts");
      const absl::Status built =
          makeContactPlanner(config, contactSet).status();
      EXPECT_EQ(built.code(), absl::StatusCode::kInvalidArgument) << name;
      EXPECT_TRUE(absl::StrContains(built.message(), expected)) << built;
      EXPECT_TRUE(absl::StrContains(built.message(), count)) << built;
      const absl::Status summarized =
          contactPlannerSummary(config, contactSet).status();
      EXPECT_EQ(summarized.code(), absl::StatusCode::kInvalidArgument) << name;
      EXPECT_TRUE(absl::StrContains(summarized.message(), expected))
          << summarized;
    }
    EXPECT_TRUE(makeContactPlanner(config, kBipedContactSetSummary).ok())
        << name;
  }
}

}  // namespace
}  // namespace ocs2::legged
