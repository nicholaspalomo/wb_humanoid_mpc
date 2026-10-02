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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/PropertyTree.h>

#include "absl/types/span.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"

namespace ocs2::humanoid {

// Found by argument-dependent lookup, so it lives in the namespace of ModelSettings rather than in the anonymous one.
void PrintTo(const ModelSettings::ContactImplicitKey& key, std::ostream* os) {
  *os << key.name;
}

namespace {

/** Writes a task file with the given lists (and a cost, which every real file has) under the test's temp directory. */
std::string writeTaskFile(absl::string_view name,
                          const std::vector<std::string>& hardConstraints,
                          const std::vector<std::string>& softConstraints) {
  const std::string path = (std::filesystem::path(testing::TempDir()) / absl::StrCat("testMpcFormulationConfig_", name, ".yaml")).string();
  std::ofstream out(path);
  out << "hard_constraints:";
  if (hardConstraints.empty()) out << " []";
  out << "\n";
  for (const std::string& entry : hardConstraints) out << "  - " << entry << "\n";
  out << "\nsoft_constraints:";
  if (softConstraints.empty()) out << " []";
  out << "\n";
  for (const std::string& entry : softConstraints) out << "  - " << entry << "\n";
  out << "\ncosts:\n  - state_quadratic_cost\n";
  return path;
}

absl::StatusOr<MpcFormulationTasks> load(absl::string_view name,
                                         const std::vector<std::string>& hardConstraints,
                                         const std::vector<std::string>& softConstraints) {
  return loadMpcFormulationTasks(writeTaskFile(name, hardConstraints, softConstraints), /*verbose=*/false);
}

/** The shipped, schedule-gated arrangement. */
const std::vector<std::string> kGatedHard = {"zero_wrench", "zero_velocity", "normal_velocity"};

/** The complete contact-implicit formulation: no schedule-gated hard constraint, a cone, the soft servo, the three terms. */
std::vector<std::string> contactImplicitSoft(absl::string_view cone = "contact_wrench_cone") {
  return {"joint_limits", std::string(cone), "normal_velocity", "contact_complementarity", "force_weighted_slip", "ground_penetration"};
}

std::vector<std::string> without(std::vector<std::string> entries, absl::string_view entry) {
  entries.erase(std::remove(entries.begin(), entries.end(), std::string(entry)), entries.end());
  return entries;
}

std::vector<std::string> with(std::vector<std::string> entries, absl::string_view entry) {
  entries.emplace_back(entry);
  return entries;
}

::testing::AssertionResult refusedMentioning(const absl::StatusOr<MpcFormulationTasks>& tasks, const std::vector<std::string>& phrases) {
  if (tasks.ok()) return ::testing::AssertionFailure() << "the task file was accepted";
  if (tasks.status().code() != absl::StatusCode::kInvalidArgument) {
    return ::testing::AssertionFailure() << "expected InvalidArgument, got " << tasks.status();
  }
  for (const std::string& phrase : phrases) {
    if (!absl::StrContains(tasks.status().message(), phrase)) {
      return ::testing::AssertionFailure() << "the message does not mention '" << phrase << "': " << tasks.status().message();
    }
  }
  return ::testing::AssertionSuccess();
}

std::string readWholeFile(const std::string& path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/** Writes `content` under the test's temp directory as testMpcFormulationConfig_<name>.yaml and returns its path. */
std::string writeContent(absl::string_view name, const std::string& content) {
  const std::string path = (std::filesystem::path(testing::TempDir()) / absl::StrCat("testMpcFormulationConfig_", name, ".yaml")).string();
  std::ofstream out(path);
  out << content;
  return path;
}

/** The names an unknown-name message offers: the comma-separated list after "are: ", without the closing period. */
std::vector<std::string> offeredNames(absl::string_view message) {
  const absl::string_view::size_type listStart = message.find("are: ");
  if (listStart == absl::string_view::npos) return {};
  std::string listed(message.substr(listStart + std::string("are: ").size()));
  if (!listed.empty() && listed.back() == '.') listed.pop_back();
  return absl::StrSplit(listed, ", ");
}

/**
 * Every canonical name of an enum, found by walking its values (none of these enums has explicit values) until
 * `toString` knows no more.
 */
template <typename Type>
std::vector<std::string> everyCanonicalName(absl::StatusOr<std::string> (*toString)(Type)) {
  std::vector<std::string> names;
  for (int value = 0;; ++value) {
    const absl::StatusOr<std::string> name = toString(static_cast<Type>(value));
    if (!name.ok()) break;
    names.push_back(*name);
  }
  return names;
}

// ---------------------------------------------------------------------------------------------------------------
// The registries.
// ---------------------------------------------------------------------------------------------------------------

TEST(MpcFormulationNames, costNamesResolveInEitherSpelling) {
  EXPECT_EQ(*stringToMpcCostType("state_input_quadratic_cost"), MpcCostType::StateInputQuadraticCost);
  EXPECT_EQ(*stringToMpcCostType("stateInputQuadraticCost"), MpcCostType::StateInputQuadraticCost);
  EXPECT_EQ(*stringToMpcCostType("state_quadratic_cost"), MpcCostType::StateQuadraticCost);
  EXPECT_EQ(*stringToMpcCostType("input_quadratic_cost"), MpcCostType::InputQuadraticCost);
  EXPECT_EQ(*stringToMpcCostType("terminal_cost"), MpcCostType::TerminalCost);
  EXPECT_EQ(*stringToMpcCostType("icp_cost"), MpcCostType::IcpCost);
  EXPECT_EQ(*stringToMpcCostType("task_space_foot_cost"), MpcCostType::TaskSpaceFootCost);
  EXPECT_EQ(*stringToMpcCostType("task_space_torso_cost"), MpcCostType::TaskSpaceTorsoCost);
  EXPECT_EQ(*stringToMpcCostType("external_torque_cost"), MpcCostType::ExternalTorqueCost);
  EXPECT_EQ(*stringToMpcCostType("joint_torque_cost"), MpcCostType::JointTorqueCost);
  EXPECT_EQ(*stringToMpcCostType("dcm_terminal_cost"), MpcCostType::DcmTerminalCost);
  EXPECT_EQ(*stringToMpcCostType("com_and_acom_tracking_cost"), MpcCostType::ComAndAcomTrackingCost);
  EXPECT_EQ(*stringToMpcCostType("comAndAcomTrackingCost"), MpcCostType::ComAndAcomTrackingCost);
}

TEST(MpcFormulationNames, everyEnumeratorRoundTripsThroughItsName) {
  // The canonical name each enumerator prints is one the loader reads back as the same enumerator, so a name in a log
  // line can always be pasted into a task file.
  for (MpcSoftConstraintType type :
       {MpcSoftConstraintType::JointLimits, MpcSoftConstraintType::FootCollision, MpcSoftConstraintType::FrictionForceCone,
        MpcSoftConstraintType::ContactMomentXY, MpcSoftConstraintType::ContactWrenchCone, MpcSoftConstraintType::ZeroVelocity,
        MpcSoftConstraintType::NormalVelocity, MpcSoftConstraintType::ContactComplementarity, MpcSoftConstraintType::ForceWeightedSlip,
        MpcSoftConstraintType::GroundPenetration}) {
    const absl::StatusOr<std::string> name = mpcSoftConstraintTypeToString(type);
    ASSERT_TRUE(name.ok()) << static_cast<int>(type);
    EXPECT_EQ(*stringToMpcSoftConstraintType(*name), type) << *name;
  }
  for (MpcHardConstraintType type : {MpcHardConstraintType::ZeroWrench, MpcHardConstraintType::ZeroVelocity,
                                     MpcHardConstraintType::NormalVelocity, MpcHardConstraintType::KneeJointMimic}) {
    const absl::StatusOr<std::string> name = mpcHardConstraintTypeToString(type);
    ASSERT_TRUE(name.ok()) << static_cast<int>(type);
    EXPECT_EQ(*stringToMpcHardConstraintType(*name), type) << *name;
  }
  for (MpcCostType type :
       {MpcCostType::StateInputQuadraticCost, MpcCostType::StateQuadraticCost, MpcCostType::InputQuadraticCost, MpcCostType::TerminalCost,
        MpcCostType::IcpCost, MpcCostType::TaskSpaceFootCost, MpcCostType::TaskSpaceTorsoCost, MpcCostType::ExternalTorqueCost,
        MpcCostType::JointTorqueCost, MpcCostType::DcmTerminalCost, MpcCostType::ComAndAcomTrackingCost}) {
    const absl::StatusOr<std::string> name = mpcCostTypeToString(type);
    ASSERT_TRUE(name.ok()) << static_cast<int>(type);
    EXPECT_EQ(*stringToMpcCostType(*name), type) << *name;
  }
}

TEST(MpcFormulationNames, anUnknownNameIsAnInvalidArgumentThatListsTheValidOnes) {
  const absl::StatusOr<MpcSoftConstraintType> soft = stringToMpcSoftConstraintType("invalid_soft_name");
  ASSERT_FALSE(soft.ok());
  EXPECT_EQ(soft.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(soft.status().message(), "ground_penetration")) << soft.status().message();

  const absl::StatusOr<MpcHardConstraintType> hard = stringToMpcHardConstraintType("invalid_hard_name");
  ASSERT_FALSE(hard.ok());
  EXPECT_EQ(hard.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(hard.status().message(), "zero_wrench")) << hard.status().message();

  const absl::StatusOr<MpcCostType> cost = stringToMpcCostType("invalid_cost_name");
  ASSERT_FALSE(cost.ok());
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kInvalidArgument);
  // Every registered cost is offered, the newest included: the list is generated, not written out by hand.
  for (const char* registered :
       {"state_quadratic_cost", "state_input_quadratic_cost", "terminal_cost", "dcm_terminal_cost", "com_and_acom_tracking_cost"}) {
    EXPECT_TRUE(absl::StrContains(cost.status().message(), registered)) << registered << " is missing from: " << cost.status().message();
  }
  // And so is every enumerator that has a name, and nothing else: the enum is walked by value (it has no explicit
  // values) until mpcCostTypeToString() knows no more, so a cost added to the enum and to its name but not to the
  // generated list fails here rather than quietly going unoffered.
  const std::string message(cost.status().message());
  const std::string::size_type listStart = message.find("are: ");
  ASSERT_NE(listStart, std::string::npos) << message;
  std::string listed = message.substr(listStart + std::string("are: ").size());
  if (!listed.empty() && listed.back() == '.') listed.pop_back();
  const std::vector<std::string> offered = absl::StrSplit(listed, ", ");
  int named = 0;
  for (int value = 0;; ++value) {
    const absl::StatusOr<std::string> name = mpcCostTypeToString(static_cast<MpcCostType>(value));
    if (!name.ok()) break;
    ++named;
    EXPECT_NE(std::find(offered.begin(), offered.end(), *name), offered.end()) << *name << " is missing from: " << message;
  }
  EXPECT_GT(named, static_cast<int>(MpcCostType::ComAndAcomTrackingCost)) << "the walk stopped early";
  EXPECT_EQ(offered.size(), static_cast<size_t>(named)) << message;

  EXPECT_FALSE(mpcCostTypeToString(static_cast<MpcCostType>(999)).ok());
  EXPECT_FALSE(mpcSoftConstraintTypeToString(static_cast<MpcSoftConstraintType>(999)).ok());
  EXPECT_FALSE(mpcHardConstraintTypeToString(static_cast<MpcHardConstraintType>(999)).ok());

  // And an unknown name inside a task file fails the load rather than being skipped.
  EXPECT_TRUE(refusedMentioning(load("unknownName", kGatedHard, {"joint_limits", "no_such_term"}), {"no_such_term"}));
}

TEST(MpcFormulationNames, anUnknownConstraintNameOffersEveryRegisteredConstraintAndNothingElse) {
  // The soft and hard lists used to be written out by hand in the messages, and a term added to a registry was not
  // offered until someone remembered the second list. They are generated from the registries now; every enumerator
  // that has a canonical name must be offered, exactly once.
  const absl::StatusOr<MpcSoftConstraintType> soft = stringToMpcSoftConstraintType("invalid_soft_name");
  ASSERT_FALSE(soft.ok());
  const std::vector<std::string> softNames = everyCanonicalName(&mpcSoftConstraintTypeToString);
  ASSERT_GT(softNames.size(), static_cast<size_t>(MpcSoftConstraintType::GroundPenetration)) << "the walk stopped early";
  const std::vector<std::string> offeredSoft = offeredNames(soft.status().message());
  for (const std::string& name : softNames) {
    EXPECT_EQ(std::count(offeredSoft.begin(), offeredSoft.end(), name), 1) << name << " in: " << soft.status().message();
  }
  EXPECT_EQ(offeredSoft.size(), softNames.size()) << soft.status().message();

  const absl::StatusOr<MpcHardConstraintType> hard = stringToMpcHardConstraintType("invalid_hard_name");
  ASSERT_FALSE(hard.ok());
  const std::vector<std::string> hardNames = everyCanonicalName(&mpcHardConstraintTypeToString);
  ASSERT_GT(hardNames.size(), static_cast<size_t>(MpcHardConstraintType::KneeJointMimic)) << "the walk stopped early";
  const std::vector<std::string> offeredHard = offeredNames(hard.status().message());
  for (const std::string& name : hardNames) {
    EXPECT_EQ(std::count(offeredHard.begin(), offeredHard.end(), name), 1) << name << " in: " << hard.status().message();
  }
  EXPECT_EQ(offeredHard.size(), hardNames.size()) << hard.status().message();

  // Every offered name resolves, so a name pasted from the message into a task file is accepted.
  for (const std::string& name : offeredSoft) EXPECT_TRUE(stringToMpcSoftConstraintType(name).ok()) << name;
  for (const std::string& name : offeredHard) EXPECT_TRUE(stringToMpcHardConstraintType(name).ok()) << name;
  const absl::StatusOr<MpcCostType> cost = stringToMpcCostType("invalid_cost_name");
  ASSERT_FALSE(cost.ok());
  for (const std::string& name : offeredNames(cost.status().message())) EXPECT_TRUE(stringToMpcCostType(name).ok()) << name;
}

// ---------------------------------------------------------------------------------------------------------------
// Loading.
// ---------------------------------------------------------------------------------------------------------------

TEST(MpcFormulationLoader, aMissingFileIsNotFound) {
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks("/non/existent/path/task.yaml");
  ASSERT_FALSE(tasks.ok());
  EXPECT_EQ(tasks.status().code(), absl::StatusCode::kNotFound);
}

TEST(MpcFormulationLoader, theListsLoadAsWritten) {
  const std::string path = writeTaskFile("lists", kGatedHard, {"joint_limits", "friction_force_cone", "contact_moment_xy"});
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(path, /*verbose=*/true);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_EQ(tasks->hardConstraints.size(), 3U);
  EXPECT_EQ(tasks->softConstraints.size(), 3U);
  EXPECT_EQ(tasks->costs.size(), 1U);
  EXPECT_TRUE(tasks->hasHardConstraint(MpcHardConstraintType::ZeroWrench));
  EXPECT_TRUE(tasks->hasHardConstraint(MpcHardConstraintType::NormalVelocity));
  EXPECT_TRUE(tasks->hasSoftConstraint(MpcSoftConstraintType::ContactMomentXY));
  EXPECT_TRUE(tasks->hasCost(MpcCostType::StateQuadraticCost));
  EXPECT_TRUE(contactConstraintsAreScheduleGated(*tasks));
  EXPECT_FALSE(usesContactImplicitFormulation(*tasks));

  // The absl::string_view overload reads the same file.
  const absl::string_view view = path;
  EXPECT_EQ(loadMpcFormulationTasks(view, /*verbose=*/false)->softConstraints, tasks->softConstraints);
}

TEST(MpcFormulationLoader, theRetiredAcomSwitchIsRefusedNamingTheCostThatReplacedItWhateverItsValue) {
  // useComAndAcomTracking was a top-level boolean; CoM + ACoM tracking is now the cost com_and_acom_tracking_cost. A
  // file that still carries the key must not load, even with `false`: ignoring it would let a stale file run a
  // different formulation from the one it asks for, silently.
  const std::string base = writeTaskFile("retiredAcomBase", kGatedHard, {"joint_limits"});
  std::ifstream in(base);
  const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  for (const char* value : {"true", "false"}) {
    const std::string path =
        (std::filesystem::path(testing::TempDir()) / absl::StrCat("testMpcFormulationConfig_retiredAcom_", value, ".yaml")).string();
    {
      std::ofstream out(path);
      out << "useComAndAcomTracking: " << value << "\n" << content;
    }
    EXPECT_TRUE(refusedMentioning(loadMpcFormulationTasks(path, /*verbose=*/false),
                                  {"useComAndAcomTracking", "com_and_acom_tracking_cost", "costs"}))
        << "useComAndAcomTracking: " << value;
  }

  // Positive control: the same file without the key loads, and the cost that replaced it is read from `costs`.
  const absl::StatusOr<MpcFormulationTasks> without = loadMpcFormulationTasks(base, /*verbose=*/false);
  ASSERT_TRUE(without.ok()) << without.status();
  EXPECT_FALSE(without->hasCost(MpcCostType::ComAndAcomTrackingCost));
  const std::string listed = (std::filesystem::path(testing::TempDir()) / "testMpcFormulationConfig_acomListed.yaml").string();
  {
    std::ofstream out(listed);
    out << content << "  - com_and_acom_tracking_cost\n";
  }
  const absl::StatusOr<MpcFormulationTasks> with = loadMpcFormulationTasks(listed, /*verbose=*/false);
  ASSERT_TRUE(with.ok()) << with.status();
  EXPECT_TRUE(with->hasCost(MpcCostType::ComAndAcomTrackingCost));
  EXPECT_TRUE(with->hasCost(MpcCostType::StateQuadraticCost));
}

TEST(MpcFormulationLoader, theRetiredDcmSwitchIsRefusedNamingTheCostThatReplacedItWhateverItsValue) {
  // useDcmTerminalCost was a top-level boolean; ending the horizon on the DCM is now the cost dcm_terminal_cost, in
  // place of terminal_cost. The key is refused even when false: ignoring it would let a stale file end its horizon on a
  // different cost from the one it asks for, silently.
  const std::string content = readWholeFile(writeTaskFile("retiredDcmBase", kGatedHard, {"joint_limits"}));
  for (const char* value : {"true", "false"}) {
    const std::string path = writeContent(absl::StrCat("retiredDcm_", value), absl::StrCat("useDcmTerminalCost: ", value, "\n", content));
    EXPECT_TRUE(refusedMentioning(loadMpcFormulationTasks(path, /*verbose=*/false),
                                  {"useDcmTerminalCost", "dcm_terminal_cost", "costs", "terminal_cost"}))
        << "useDcmTerminalCost: " << value;
  }

  // Positive control: the same file without the key loads, and the cost that replaced it is read from `costs`.
  const absl::StatusOr<MpcFormulationTasks> listed =
      loadMpcFormulationTasks(writeContent("dcmListed", content + "  - dcm_terminal_cost\n"), /*verbose=*/false);
  ASSERT_TRUE(listed.ok()) << listed.status();
  EXPECT_TRUE(listed->hasCost(MpcCostType::DcmTerminalCost));
  EXPECT_FALSE(listed->hasCost(MpcCostType::TerminalCost));
}

TEST(MpcFormulationLoader, aFileThatIsNotAMapIsAStatusNotAnException) {
  // The loader looks keys up on the document root, which yaml-cpp answers with an exception for a scalar. It returns a
  // Status, so that has to be one too.
  absl::StatusOr<MpcFormulationTasks> scalar;
  EXPECT_NO_THROW(scalar = loadMpcFormulationTasks(writeContent("notAMap", "just a string\n"), /*verbose=*/false));
  ASSERT_FALSE(scalar.ok());
  EXPECT_EQ(scalar.status().code(), absl::StatusCode::kInvalidArgument) << scalar.status();
  EXPECT_TRUE(absl::StrContains(scalar.status().message(), "not a map")) << scalar.status();
  // Positive control: an empty file is an empty map, which lists nothing and is refused only for what it lacks.
  absl::StatusOr<MpcFormulationTasks> empty;
  EXPECT_NO_THROW(empty = loadMpcFormulationTasks(writeContent("emptyFile", ""), /*verbose=*/false));
  EXPECT_FALSE(absl::StrContains(empty.status().message(), "not a map")) << empty.status();
}

TEST(MpcFormulationLoader, theTwoTerminalCostsAreAlternativesAndAListNamingBothIsRefusedNamingBoth) {
  // When the DCM cost was a boolean, a `terminal_cost` entry beside it was silently ignored. With the list as the switch
  // a list naming both is ambiguous, whatever spelling it uses, and refused rather than resolved by a precedence.
  const std::string content = readWholeFile(writeTaskFile("terminalBase", kGatedHard, {"joint_limits"}));
  for (const char* entry : {"terminal_cost", "dcm_terminal_cost"}) {
    const absl::StatusOr<MpcFormulationTasks> alone =
        loadMpcFormulationTasks(writeContent(absl::StrCat("alone_", entry), absl::StrCat(content, "  - ", entry, "\n")), /*verbose=*/false);
    EXPECT_TRUE(alone.ok()) << entry << ": " << alone.status();
  }
  for (const char* spelling : {"  - terminal_cost\n  - dcm_terminal_cost\n", "  - dcmTerminalCost\n  - terminalCost\n"}) {
    const std::string path = writeContent(absl::StrCat("bothTerminal_", std::string(spelling).size()), content + spelling);
    EXPECT_TRUE(refusedMentioning(loadMpcFormulationTasks(path, /*verbose=*/false), {"'terminal_cost'", "'dcm_terminal_cost'"}))
        << spelling;
  }
}

TEST(MpcFormulationLoader, zeroVelocityMayBeSoftButNotBothHardAndSoft) {
  const absl::StatusOr<MpcFormulationTasks> soft = load("softZeroVelocity", {"zero_wrench", "normal_velocity"}, {"zero_velocity"});
  ASSERT_TRUE(soft.ok()) << soft.status();
  EXPECT_TRUE(soft->hasSoftConstraint(MpcSoftConstraintType::ZeroVelocity));
  EXPECT_TRUE(refusedMentioning(load("bothZeroVelocity", {"zero_wrench", "zero_velocity"}, {"zero_velocity"}), {"zero_velocity"}));
}

TEST(MpcFormulationLoader, normalVelocityMayNotBeBothHardAndSoft) {
  EXPECT_TRUE(refusedMentioning(load("bothNormalVelocity", kGatedHard, {"normal_velocity"}), {"normal_velocity", "simultaneously"}));
}

// ---------------------------------------------------------------------------------------------------------------
// The contact-implicit formulation: the one arrangement that loads, and every half-way one the loader refuses.
// Each refusal is reached with every EARLIER check satisfied, and asserts a phrase only its own message carries, so
// deleting any single check turns exactly its own test red.
// ---------------------------------------------------------------------------------------------------------------

TEST(ContactImplicitLoader, theCompleteFormulationLoadsWithEitherCone) {
  for (const char* cone : {"contact_wrench_cone", "friction_force_cone"}) {
    const absl::StatusOr<MpcFormulationTasks> tasks = load(absl::StrCat("complete_", cone), {}, contactImplicitSoft(cone));
    ASSERT_TRUE(tasks.ok()) << cone << ": " << tasks.status();
    EXPECT_TRUE(usesContactImplicitFormulation(*tasks)) << cone;
    EXPECT_FALSE(contactConstraintsAreScheduleGated(*tasks)) << cone << ": without zero_wrench the cones must be un-gated";
    EXPECT_TRUE(tasks->hasSoftConstraint(MpcSoftConstraintType::NormalVelocity)) << cone;
  }
}

TEST(ContactImplicitLoader, anyOneTermMissingIsRefusedNamingIt) {
  // "Listed together or not at all" is what usesContactImplicitFormulation() and every caller of it assume. The loader
  // used to enforce it for contact_complementarity only, so a lone ground_penetration or force_weighted_slip loaded -
  // and then met a later check whose explanation did not fit it.
  const std::vector<std::string> terms = {"contact_complementarity", "force_weighted_slip", "ground_penetration"};
  for (const std::string& missing : terms) {
    const absl::StatusOr<MpcFormulationTasks> tasks = load(absl::StrCat("missing_", missing), {}, without(contactImplicitSoft(), missing));
    EXPECT_TRUE(refusedMentioning(tasks, {"together or not at all", absl::StrCat("not '", missing, "'")})) << missing;
  }
}

TEST(ContactImplicitLoader, aLoneComplementarityTermIsRefusedNamingBothMissingTerms) {
  // The third lone term; the other two are below. Both missing terms are named, each with the hole it leaves.
  EXPECT_TRUE(refusedMentioning(
      load("loneComplementarity", {}, {"joint_limits", "contact_wrench_cone", "normal_velocity", "contact_complementarity"}),
      {"together or not at all", "'force_weighted_slip' (", "'ground_penetration' ("}));
}

TEST(ContactImplicitLoader, complementarityWithoutThePenetrationHingeIsRefused) {
  // The complementarity product is also satisfied by a NEGATIVE height, so without h >= 0 a foot pushed through the
  // floor carries load for free.
  EXPECT_TRUE(refusedMentioning(load("noPenetration", {}, without(contactImplicitSoft(), "ground_penetration")),
                                {"'ground_penetration' (without it nothing stops a foot being pushed through the ground"}));
}

TEST(ContactImplicitLoader, complementarityWithoutTheSlipTermIsRefused) {
  EXPECT_TRUE(refusedMentioning(load("noSlip", {}, without(contactImplicitSoft(), "force_weighted_slip")),
                                {"'force_weighted_slip' (without it nothing holds a loaded foot still"}));
}

TEST(ContactImplicitLoader, aLoneGroundPenetrationBesideTheShippedListsGetsTheRightExplanation) {
  // The case the audit traced: an engineer adds ground_penetration to the shipped soft list to keep the toes out of
  // the floor. It used to load past the partial-list check and be refused by the hard normal_velocity one, whose
  // message - about "a touch-down it is being asked to choose" - has nothing to do with it.
  const absl::StatusOr<MpcFormulationTasks> tasks =
      load("lonePenetration", kGatedHard, {"joint_limits", "contact_wrench_cone", "ground_penetration"});
  EXPECT_TRUE(refusedMentioning(tasks, {"together or not at all", "'contact_complementarity' (", "'force_weighted_slip' ("}));
  EXPECT_FALSE(absl::StrContains(tasks.status().message(), "touch-down")) << tasks.status().message();
}

TEST(ContactImplicitLoader, aLoneSlipTermIsRefused) {
  EXPECT_TRUE(refusedMentioning(load("loneSlip", {}, {"joint_limits", "contact_wrench_cone", "normal_velocity", "force_weighted_slip"}),
                                {"together or not at all", "'contact_complementarity' (", "'ground_penetration' ("}));
}

TEST(ContactImplicitLoader, complementarityBesideTheHardZeroWrenchIsRefused) {
  EXPECT_TRUE(refusedMentioning(load("withZeroWrench", {"zero_wrench"}, contactImplicitSoft()),
                                {"'contact_complementarity' and the hard 'zero_wrench' constraint are mutually exclusive"}));
}

TEST(ContactImplicitLoader, droppingZeroWrenchWithoutAConeIsRefusedWhateverElseIsListed) {
  // f_n >= 0 is the first of the three conditions of rigid contact, and it is the cone's: the complementarity product
  // is zero at h = 0 for ANY f_n, adhesion included. The cones gate themselves off `zero_wrench`, so dropping it
  // without listing one leaves every foot's wrench unbounded - with or without the contact-implicit terms.
  EXPECT_TRUE(refusedMentioning(load("noCone", {}, without(contactImplicitSoft(), "contact_wrench_cone")),
                                {"'contact_wrench_cone' or 'friction_force_cone' must be listed"}));
  EXPECT_TRUE(refusedMentioning(load("bareUngated", {"zero_velocity"}, {"joint_limits"}),
                                {"'contact_wrench_cone' or 'friction_force_cone' must be listed"}));

  // While zero_wrench is listed a missing cone is merely redundant, and the shipped arrangement stays loadable.
  const absl::StatusOr<MpcFormulationTasks> gated = load("gatedNoCone", {"zero_wrench", "zero_velocity"}, {"joint_limits"});
  ASSERT_TRUE(gated.ok()) << gated.status();
  EXPECT_TRUE(contactConstraintsAreScheduleGated(*gated));
}

TEST(ContactImplicitLoader, theSlipTermBesideEitherZeroVelocityIsRefused) {
  // The soft zero_velocity is refused as well as the hard one: both hold the same foot still from the schedule.
  EXPECT_TRUE(refusedMentioning(load("slipWithSoftZeroVelocity", {}, with(contactImplicitSoft(), "zero_velocity")),
                                {"'force_weighted_slip' replaces 'zero_velocity'"}));
  EXPECT_TRUE(refusedMentioning(load("slipWithHardZeroVelocity", {"zero_velocity"}, contactImplicitSoft()),
                                {"'force_weighted_slip' replaces 'zero_velocity'"}));
}

TEST(ContactImplicitLoader, theHardNormalVelocityIsRefusedBesideTheFormulation) {
  // Without the soft normal_velocity in the list, so the duplicate check (hard AND soft) cannot be what refuses it.
  EXPECT_TRUE(refusedMentioning(load("hardNormalVelocity", {"normal_velocity"}, without(contactImplicitSoft(), "normal_velocity")),
                                {"the hard 'normal_velocity' constraint are mutually exclusive", "list it in soft_constraints"}));
}

TEST(ContactImplicitLoader, theFormulationWithoutTheSoftNormalVelocityIsRefused) {
  // With the hard servo refused, the soft one is the only term with the authority to lift a swing foot; the README and
  // the task files call it "not optional" because the robot shuffled without it. It used to load.
  EXPECT_TRUE(refusedMentioning(load("noSoftNormalVelocity", {}, without(contactImplicitSoft(), "normal_velocity")),
                                {"needs 'normal_velocity' in soft_constraints", "shuffles"}));
}

// ---------------------------------------------------------------------------------------------------------------
// The contact_implicit block's values.
// ---------------------------------------------------------------------------------------------------------------

class ContactImplicitConfigValidation : public ::testing::TestWithParam<ModelSettings::ContactImplicitKey> {
 protected:
  /** Whether the key is a penalty weight, told from its NAME rather than from the list's own flag it is checked against. */
  bool namedAsAWeight() const { return absl::EndsWith(GetParam().name, "Weight"); }
};

TEST(ContactImplicitKeys, everyKeyIsNamedOnceSetsAFieldOfItsOwnAndIsAWeightExactlyWhenItsNameSaysSo) {
  // The list ModelSettings loads the block with, the validator checks it against and the parameter updater reloads it
  // from. ModelSettings.cpp static_asserts that it covers every field of ContactImplicitConfig; this checks that no two
  // entries share a name or a field, and that the weight/divisor split the validator relies on is the one the names
  // state.
  const absl::Span<const ModelSettings::ContactImplicitKey> keys = ModelSettings::contactImplicitKeys();
  ASSERT_FALSE(keys.empty());
  for (size_t i = 0; i < keys.size(); ++i) {
    SCOPED_TRACE(keys[i].name);
    EXPECT_EQ(keys[i].isWeight, absl::EndsWith(keys[i].name, "Weight"));
    for (size_t j = 0; j < i; ++j) {
      EXPECT_NE(keys[i].name, keys[j].name);
      EXPECT_NE(keys[i].field, keys[j].field);
    }
  }
  EXPECT_EQ(sizeof(ModelSettings::ContactImplicitConfig), keys.size() * sizeof(scalar_t));
}

TEST(ContactImplicitConfigDefaults, theDefaultBlockIsValid) {
  // Positive control for every refusal below: the one change each of them makes is what gets refused.
  EXPECT_TRUE(validateContactImplicitConfig(ModelSettings::ContactImplicitConfig{}).ok());
}

TEST_P(ContactImplicitConfigValidation, anOutOfRangeValueIsRefusedNamingItsKey) {
  const std::string expectedKey = absl::StrCat("contact_implicit.", GetParam().name);
  const std::vector<scalar_t> refused =
      namedAsAWeight()
          ? std::vector<scalar_t>{-1.0, std::numeric_limits<scalar_t>::quiet_NaN(), std::numeric_limits<scalar_t>::infinity()}
          : std::vector<scalar_t>{0.0, -1.0e-3, std::numeric_limits<scalar_t>::quiet_NaN(), std::numeric_limits<scalar_t>::infinity()};
  for (const scalar_t value : refused) {
    ModelSettings::ContactImplicitConfig config;
    config.*GetParam().field = value;
    const absl::Status status = validateContactImplicitConfig(config);
    ASSERT_FALSE(status.ok()) << expectedKey << " = " << value;
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(status.message(), expectedKey)) << status.message();
  }
}

TEST_P(ContactImplicitConfigValidation, aWeightMayBeZeroButAReferenceMayNot) {
  ModelSettings::ContactImplicitConfig config;
  config.*GetParam().field = 0.0;
  EXPECT_EQ(validateContactImplicitConfig(config).ok(), namedAsAWeight()) << GetParam().name;
}

INSTANTIATE_TEST_SUITE_P(EveryKey,
                         ContactImplicitConfigValidation,
                         ::testing::ValuesIn(ModelSettings::contactImplicitKeys().begin(), ModelSettings::contactImplicitKeys().end()),
                         [](const ::testing::TestParamInfo<ModelSettings::ContactImplicitKey>& info) {
                           return std::string(info.param.name);
                         });

// ---------------------------------------------------------------------------------------------------------------
// The contact_implicit block's keys.
// ---------------------------------------------------------------------------------------------------------------

/** Parses `yaml` the way the MPC interfaces and the updater parse a task file. */
PropertyTree parsedTaskFile(absl::string_view yaml) {
  PropertyTree tree;
  loadData::readPropertyTreeFromString(yaml, tree);
  return tree;
}

/** A `contact_implicit` block with every key the code reads, `renamed` spelled `as` instead. */
std::string contactImplicitBlock(absl::string_view renamed = "", absl::string_view as = "") {
  std::string block = "contact_implicit:\n";
  for (const ModelSettings::ContactImplicitKey& key : ModelSettings::contactImplicitKeys()) {
    absl::StrAppend(&block, "  ", key.name == renamed ? as : key.name, ": 1.0\n");
  }
  return block;
}

TEST(ContactImplicitBlockKeys, aBlockOfTheKnownKeysOrNoBlockAtAllIsAccepted) {
  EXPECT_TRUE(checkContactImplicitBlockKeys(parsedTaskFile(contactImplicitBlock())).ok());
  EXPECT_TRUE(checkContactImplicitBlockKeys(parsedTaskFile("terrainHeight: 0.0\n")).ok());
}

TEST(ContactImplicitBlockKeys, aRenamedKeyIsRefusedNamingItAndListingTheKeys) {
  // A key nothing reads is a renamed or misspelled one: the term it was meant for would run on its default and its tuning
  // slider would reach nothing, so it is refused rather than skipped.
  for (const ModelSettings::ContactImplicitKey& key : ModelSettings::contactImplicitKeys()) {
    SCOPED_TRACE(key.name);
    const std::string renamed = absl::StrCat(key.name, "Renamed");
    const absl::Status status = checkContactImplicitBlockKeys(parsedTaskFile(contactImplicitBlock(key.name, renamed)));
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(status.message(), absl::StrCat("contact_implicit.", renamed))) << status.message();
    EXPECT_TRUE(absl::StrContains(status.message(), absl::StrCat(key.name, ","))  // the list of the keys the block may carry
                || absl::StrContains(status.message(), absl::StrCat(key.name, ".")))
        << status.message();
  }
}

TEST(ContactImplicitBlockKeys, theGroundThatMovedOutOfTheBlockIsRefusedPointingAtWhereItWent) {
  const absl::Status status = checkContactImplicitBlockKeys(parsedTaskFile(absl::StrCat(contactImplicitBlock(), "  terrainHeight: 0.0\n")));
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.message(), "contact_implicit.terrainHeight")) << status.message();
  EXPECT_TRUE(absl::StrContains(status.message(), "top-level `terrainHeight`")) << status.message();
}

// ---------------------------------------------------------------------------------------------------------------
// interface.verbose, which the MPC interfaces read before their model settings.
// ---------------------------------------------------------------------------------------------------------------

TEST(InterfaceVerbose, isReadFromTheTaskFileAndAValueThatIsNotABoolIsRefusedNamingTheKey) {
  const absl::StatusOr<bool> absent = ModelSettings::loadInterfaceVerbose(writeContent("verboseAbsent", "interface:\n  other: 1\n"));
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_FALSE(*absent);
  for (const bool value : {true, false}) {
    const absl::StatusOr<bool> read = ModelSettings::loadInterfaceVerbose(
        writeContent(absl::StrCat("verbose_", value), absl::StrCat("interface:\n  verbose: ", value ? "true" : "false", "\n")));
    ASSERT_TRUE(read.ok()) << read.status();
    EXPECT_EQ(*read, value);
  }
  const absl::StatusOr<bool> notABool =
      ModelSettings::loadInterfaceVerbose(writeContent("verboseNotABool", "interface:\n  verbose: loud\n"));
  ASSERT_FALSE(notABool.ok());
  EXPECT_EQ(notABool.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(notABool.status().message(), ModelSettings::kInterfaceVerboseKey)) << notABool.status();
  EXPECT_TRUE(absl::StrContains(notABool.status().message(), "'loud'")) << notABool.status();

  const absl::StatusOr<bool> missing = ModelSettings::loadInterfaceVerbose("/non/existent/path/task.yaml");
  ASSERT_FALSE(missing.ok());
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kNotFound);
}

// ---------------------------------------------------------------------------------------------------------------
// The contact schedule source: where the mode schedule and the footholds come from, selected by name.
// ---------------------------------------------------------------------------------------------------------------

TEST(ContactScheduleSourceRegistry, everyNameRoundTripsAndTheDefaultIsTheGaitSchedule) {
  const std::vector<std::string> names = contactScheduleSourceNames();
  ASSERT_FALSE(names.empty());
  for (const std::string& name : names) {
    EXPECT_EQ(std::count(names.begin(), names.end(), name), 1) << name << " is registered twice";
    const absl::StatusOr<ContactScheduleSource> source = contactScheduleSourceFromName(name);
    ASSERT_TRUE(source.ok()) << source.status();
    EXPECT_EQ(contactScheduleSourceName(*source), name);
  }
  // Every enumerator is registered: each has a name the registry lists.
  for (const ContactScheduleSource source : {ContactScheduleSource::kGaitSchedule, ContactScheduleSource::kContactPlanner}) {
    const std::string name(contactScheduleSourceName(source));
    EXPECT_NE(std::find(names.begin(), names.end(), name), names.end()) << name;
  }
  // The default is what every robot ran before the key existed: the gait schedule, not the planner.
  EXPECT_EQ(kDefaultContactScheduleSource, ContactScheduleSource::kGaitSchedule);
  EXPECT_EQ(contactScheduleSourceName(kDefaultContactScheduleSource), kGaitScheduleContactScheduleSource);
}

TEST(ContactScheduleSourceRegistry, anUnknownNameIsRefusedNamingTheKeyAndListingEveryValidName) {
  for (const char* unknown : {"planner", "true", "Gait_Schedule", ""}) {
    const absl::StatusOr<ContactScheduleSource> source = contactScheduleSourceFromName(unknown);
    ASSERT_FALSE(source.ok()) << "'" << unknown << "' was accepted";
    EXPECT_EQ(source.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(source.status().message(), kContactScheduleSourceKey)) << source.status();
    for (const std::string& name : contactScheduleSourceNames()) {
      EXPECT_TRUE(absl::StrContains(source.status().message(), name)) << name << " is not offered: " << source.status();
    }
  }
}

TEST(ContactScheduleSourceLoader, anAbsentKeyIsTheGaitScheduleAndANamedSourceIsRead) {
  const absl::StatusOr<ContactScheduleSource> absent =
      loadContactScheduleSource(writeContent("sourceAbsent", "costs:\n  - state_quadratic_cost\n"));
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_EQ(*absent, kDefaultContactScheduleSource);
  const absl::StatusOr<ContactScheduleSource> empty = loadContactScheduleSource(writeContent("sourceEmptyFile", ""));
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_EQ(*empty, kDefaultContactScheduleSource);
  for (const std::string& name : contactScheduleSourceNames()) {
    const absl::StatusOr<ContactScheduleSource> named =
        loadContactScheduleSource(writeContent(absl::StrCat("source_", name), absl::StrCat(kContactScheduleSourceKey, ": ", name, "\n")));
    ASSERT_TRUE(named.ok()) << named.status();
    EXPECT_EQ(contactScheduleSourceName(*named), name);
  }
}

TEST(ContactScheduleSourceLoader, theRetiredBooleanIsRefusedWhateverItsValueNamingItsReplacement) {
  // useContactPlanning: false meant the gait schedule, which is also the default - but a file that still spells the
  // switch the old way was written against the old loader, so it is refused either way.
  for (const char* value : {"true", "false"}) {
    const absl::StatusOr<ContactScheduleSource> source =
        loadContactScheduleSource(writeContent(absl::StrCat("retiredPlanning_", value), absl::StrCat("useContactPlanning: ", value, "\n")));
    ASSERT_FALSE(source.ok()) << "useContactPlanning: " << value << " was accepted";
    EXPECT_EQ(source.status().code(), absl::StatusCode::kInvalidArgument);
    for (const std::string& phrase :
         {std::string(kRetiredContactPlanningKey), absl::StrCat(kContactScheduleSourceKey, ": ", kContactPlannerContactScheduleSource),
          absl::StrCat(kContactScheduleSourceKey, ": ", kGaitScheduleContactScheduleSource)}) {
      EXPECT_TRUE(absl::StrContains(source.status().message(), phrase)) << phrase << " is missing from: " << source.status();
    }
  }
  // Also beside the key that replaced it: the retired one is never read, so it must not silently lose to the name.
  const absl::StatusOr<ContactScheduleSource> both = loadContactScheduleSource(
      writeContent("retiredBesideName", absl::StrCat("useContactPlanning: true\n", kContactScheduleSourceKey, ": gait_schedule\n")));
  EXPECT_FALSE(both.ok());
}

TEST(ContactScheduleSourceLoader, aValueThatIsNotANameAndAFileThatCannotBeReadAreStatuses) {
  const absl::StatusOr<ContactScheduleSource> sequence =
      loadContactScheduleSource(writeContent("sourceSequence", absl::StrCat(kContactScheduleSourceKey, ":\n  - contact_planner\n")));
  ASSERT_FALSE(sequence.ok());
  EXPECT_EQ(sequence.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(sequence.status().message(), kContactScheduleSourceKey)) << sequence.status();

  const absl::StatusOr<ContactScheduleSource> scalarFile = loadContactScheduleSource(writeContent("sourceScalarFile", "just a string\n"));
  EXPECT_FALSE(scalarFile.ok());

  const absl::StatusOr<ContactScheduleSource> missing = loadContactScheduleSource("/non/existent/path/task.yaml");
  ASSERT_FALSE(missing.ok());
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kNotFound);
}

// ---------------------------------------------------------------------------------------------------------------
// The shipped task files keep the formulation OFF.
//
// It changes the closed loop and has not been validated in simulation on hardware-like conditions, so the repository
// rule is that it ships disabled. testContactImplicitFormulation deliberately does not assert that - an engineer
// experimenting with the formulation switches it on in the same task file that suite reads, and the whole suite would
// go red - so this is where the rule is pinned. The files are the ones this test's `data` carries, i.e. the task
// files as they stand in the tree, which is what gets committed: an experiment in progress turns these cases red, and
// only these, until the file is switched back off.
// ---------------------------------------------------------------------------------------------------------------

// Every MPC task file of every robot. The BUILD target's `data` has to list the same packages, and
// theTaskFilesInTheRunfilesAreExactlyTheList closes the other direction.
// LINT.IfChange(formulation_shipped_task_files)
constexpr const char* kShippedTaskFiles[] = {
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
    "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml",
    "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml",
    "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml",
    "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml",
};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:formulation_shipped_task_files_data)
// clang-format on

std::vector<std::filesystem::path> runfilesRoots() {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
    roots.emplace_back(std::filesystem::path(srcDir) / "wb_humanoid_mpc");
  }
  roots.emplace_back(std::filesystem::current_path());
  return roots;
}

std::string runfilePath(absl::string_view relativePath) {
  for (const std::filesystem::path& root : runfilesRoots()) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/** Every robot_models/<robot>/<package>/config/mpc/task.yaml in the runfiles, relative to the runfiles root, sorted. */
std::vector<std::string> taskFilesInRunfiles() {
  std::vector<std::string> found;
  for (const std::filesystem::path& root : runfilesRoots()) {
    const std::filesystem::path robotModels = root / "robot_models";
    if (!std::filesystem::is_directory(robotModels)) continue;
    for (const std::filesystem::directory_entry& robot : std::filesystem::directory_iterator(robotModels)) {
      if (!robot.is_directory()) continue;
      for (const std::filesystem::directory_entry& package : std::filesystem::directory_iterator(robot.path())) {
        if (!package.is_directory()) continue;
        const std::string relativePath = absl::StrCat("robot_models/", robot.path().filename().string(), "/",
                                                      package.path().filename().string(), "/config/mpc/task.yaml");
        if (std::filesystem::exists(root / relativePath)) found.push_back(relativePath);
      }
    }
    break;
  }
  std::sort(found.begin(), found.end());
  return found;
}

class ShippedTaskFileFormulation : public ::testing::TestWithParam<const char*> {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath(GetParam());
    ASSERT_FALSE(taskFile_.empty()) << GetParam() << " is not in the runfiles; add its package to this test's `data`.";
  }
  std::string taskFile_;
};

TEST_P(ShippedTaskFileFormulation, loadsWithTheContactImplicitFormulationOffAndTheConesGated) {
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile_, /*verbose=*/false);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_FALSE(usesContactImplicitFormulation(*tasks))
      << GetParam() << " switches the contact-implicit formulation on. It changes the closed loop and ships disabled until it has "
      << "been validated in simulation (humanoid_nmpc/docs/contact_implicit_mpc/README.md); commit it switched off.";
  EXPECT_TRUE(contactConstraintsAreScheduleGated(*tasks))
      << GetParam() << " drops the hard 'zero_wrench' constraint, which un-gates every contact cone of the robot.";
}

TEST_P(ShippedTaskFileFormulation, theSameFileSwitchedOnIsSeenAsOn) {
  // Positive control: the file's own lists, rewritten the way README section 5 says, make the predicates above flip.
  // Without this a loader that read no lists at all would pass the test above on every file.
  YAML::Node root = YAML::LoadFile(taskFile_);
  YAML::Node soft(YAML::NodeType::Sequence);
  for (const std::string& entry : contactImplicitSoft()) soft.push_back(entry);
  root["hard_constraints"] = YAML::Node(YAML::NodeType::Sequence);
  root["soft_constraints"] = soft;
  const std::string switchedOn =
      (std::filesystem::path(testing::TempDir()) /
       absl::StrCat("switchedOn_", std::filesystem::path(GetParam()).parent_path().parent_path().parent_path().filename().string(),
                    ".yaml"))
          .string();
  {
    std::ofstream out(switchedOn);
    out << root;
  }
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(switchedOn, /*verbose=*/false);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_TRUE(usesContactImplicitFormulation(*tasks));
  EXPECT_FALSE(contactConstraintsAreScheduleGated(*tasks));
}

TEST_P(ShippedTaskFileFormulation, aContactImplicitBlockCarriesExactlyTheKeysTheCodeReads) {
  // A key renamed in the task file alone - or in the code alone - is caught here, on the shipped files, rather than by
  // a term that quietly runs on its default. Every file passes the check the interfaces run at start-up; a file that
  // carries the block carries every key of it, so that none is left to a default nobody chose.
  PropertyTree tree;
  loadData::readPropertyTree(taskFile_, tree);
  EXPECT_TRUE(checkContactImplicitBlockKeys(tree).ok()) << checkContactImplicitBlockKeys(tree);
  const YAML::Node block = YAML::LoadFile(taskFile_)[std::string(ModelSettings::kContactImplicitBlock)];
  if (!block) return;
  std::vector<std::string> carried;
  for (const YAML::detail::iterator_value& entry : block) carried.push_back(entry.first.as<std::string>());
  std::vector<std::string> known;
  for (const ModelSettings::ContactImplicitKey& key : ModelSettings::contactImplicitKeys()) known.emplace_back(key.name);
  std::sort(carried.begin(), carried.end());
  std::sort(known.begin(), known.end());
  EXPECT_EQ(carried, known) << GetParam();
}

TEST_P(ShippedTaskFileFormulation, namesTheGaitScheduleAndCarriesNoRetiredFormulationSwitch) {
  // The online contact planner changes the closed loop, and ships switched off on every robot, by name: every file
  // spells the key out rather than leaving it to the default, so that the choice is visible where it is made.
  const absl::StatusOr<ContactScheduleSource> source = loadContactScheduleSource(taskFile_);
  ASSERT_TRUE(source.ok()) << source.status();
  EXPECT_EQ(*source, ContactScheduleSource::kGaitSchedule) << GetParam();
  const YAML::Node root = YAML::LoadFile(taskFile_);
  EXPECT_TRUE(root[std::string(kContactScheduleSourceKey)]) << GetParam() << " does not name its " << kContactScheduleSourceKey;
  for (const char* retired : {"useContactPlanning", "useDcmTerminalCost", "useComAndAcomTracking", "useContactBasisVectorInputs"}) {
    EXPECT_FALSE(root[retired]) << GetParam() << " still carries the retired key " << retired;
  }
  // And it ends its horizon on one terminal cost at most (the loader refuses both; this says which file broke it).
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile_, /*verbose=*/false);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_FALSE(tasks->hasCost(MpcCostType::TerminalCost) && tasks->hasCost(MpcCostType::DcmTerminalCost)) << GetParam();
}

TEST(ShippedTaskFileScan, theRobotsThatConfigureTheFormulationCarryTheBlock) {
  // Positive control for the check above, which returns early for a file without the block.
  size_t filesWithTheBlock = 0;
  for (const char* relativePath : kShippedTaskFiles) {
    const std::string path = runfilePath(relativePath);
    ASSERT_FALSE(path.empty()) << relativePath;
    if (YAML::LoadFile(path)[std::string(ModelSettings::kContactImplicitBlock)]) ++filesWithTheBlock;
  }
  EXPECT_GE(filesWithTheBlock, 2U) << "the DRC Atlas and the EngineAI SA01 both configure the contact_implicit block";
}

INSTANTIATE_TEST_SUITE_P(
    EveryRobot, ShippedTaskFileFormulation, ::testing::ValuesIn(kShippedTaskFiles), [](const ::testing::TestParamInfo<const char*>& info) {
      std::string name = std::filesystem::path(info.param).parent_path().parent_path().parent_path().filename().string();
      std::replace(name.begin(), name.end(), '-', '_');
      return name;
    });

TEST(ShippedTaskFileScan, theTaskFilesInTheRunfilesAreExactlyTheList) {
  // A robot whose package is in `data` but not in kShippedTaskFiles would never be checked, and nothing would say so.
  const std::vector<std::string> found = taskFilesInRunfiles();
  ASSERT_GE(found.size(), std::size(kShippedTaskFiles)) << "the runfiles walk found only: " << absl::StrJoin(found, ", ");
  std::vector<std::string> listed(std::begin(kShippedTaskFiles), std::end(kShippedTaskFiles));
  std::sort(listed.begin(), listed.end());
  EXPECT_EQ(found, listed) << "the task files in this test's runfiles are not the ones kShippedTaskFiles names";
}

}  // namespace
}  // namespace ocs2::humanoid
