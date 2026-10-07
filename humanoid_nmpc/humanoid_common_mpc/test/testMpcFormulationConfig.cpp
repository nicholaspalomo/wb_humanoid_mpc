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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {

// Found by argument-dependent lookup, so it lives in the namespace of ModelSettings rather than in the anonymous one.
// NOLINTNEXTLINE(misc-use-anonymous-namespace): an unnamed namespace would hide it from that lookup; static keeps it local.
static void PrintTo(const ModelSettings::ContactImplicitKey& key, std::ostream* absl_nonnull os) {
  *os << key.fieldName;
}

namespace {

/** A task file with the given lists (and a cost, which every real file has). */
mpc_config::TaskFile taskWith(const std::vector<std::string>& hardConstraints, const std::vector<std::string>& softConstraints) {
  mpc_config::TaskFile task;
  task.hard_constraints = hardConstraints;
  task.soft_constraints = softConstraints;
  task.costs = {"state_quadratic_cost"};
  return task;
}

/** The formulation of the task file with the given lists, as the MPC interfaces read it. */
absl::StatusOr<MpcFormulationTasks> load(const std::vector<std::string>& hardConstraints, const std::vector<std::string>& softConstraints) {
  return mpcFormulationTasksFromConfig(taskWith(hardConstraints, softConstraints), FormulationLogging::kQuiet);
}

/** The textproto `text` as the parser reads a task file. */
absl::StatusOr<humanoid_mpc_config::TaskFile> parse(absl::string_view text) {
  return nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(text, "task.textproto");
}

/** The schedule-gated arrangement. */
const absl::NoDestructor<std::vector<std::string>> kGatedHard(std::vector<std::string>{"zero_wrench", "zero_velocity", "normal_velocity"});

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

/** Whether `status`, or the status of `tasks`, is an InvalidArgument whose message mentions every one of `phrases`. */
::testing::AssertionResult refusedMentioning(const absl::Status& status, const std::vector<std::string>& phrases) {
  if (status.ok()) return ::testing::AssertionFailure() << "the task file was accepted";
  if (status.code() != absl::StatusCode::kInvalidArgument) {
    return ::testing::AssertionFailure() << "expected InvalidArgument, got " << status;
  }
  for (const std::string& phrase : phrases) {
    if (!absl::StrContains(status.message(), phrase)) {
      return ::testing::AssertionFailure() << "the message does not mention '" << phrase << "': " << status.message();
    }
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult refusedMentioning(const absl::StatusOr<MpcFormulationTasks>& tasks, const std::vector<std::string>& phrases) {
  return refusedMentioning(tasks.status(), phrases);
}

/** Writes `content` under the test's temp directory as testMpcFormulationConfig_<name>.textproto and returns its path. */
std::string writeContent(absl::string_view name, const std::string& content) {
  const std::string path =
      (std::filesystem::path(testing::TempDir()) / absl::StrCat("testMpcFormulationConfig_", name, ".textproto")).string();
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
std::vector<std::string> everyCanonicalName(absl::StatusOr<std::string> (*absl_nonnull toString)(Type)) {
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
  EXPECT_EQ(*stringToMpcCostType("state_input_quadratic_cost"), MpcCostType::kStateInputQuadraticCost);
  EXPECT_EQ(*stringToMpcCostType("stateInputQuadraticCost"), MpcCostType::kStateInputQuadraticCost);
  EXPECT_EQ(*stringToMpcCostType("state_quadratic_cost"), MpcCostType::kStateQuadraticCost);
  EXPECT_EQ(*stringToMpcCostType("input_quadratic_cost"), MpcCostType::kInputQuadraticCost);
  EXPECT_EQ(*stringToMpcCostType("terminal_cost"), MpcCostType::kTerminalCost);
  EXPECT_EQ(*stringToMpcCostType("icp_cost"), MpcCostType::kIcpCost);
  EXPECT_EQ(*stringToMpcCostType("task_space_foot_cost"), MpcCostType::kTaskSpaceFootCost);
  EXPECT_EQ(*stringToMpcCostType("task_space_torso_cost"), MpcCostType::kTaskSpaceTorsoCost);
  EXPECT_EQ(*stringToMpcCostType("external_torque_cost"), MpcCostType::kExternalTorqueCost);
  EXPECT_EQ(*stringToMpcCostType("joint_torque_cost"), MpcCostType::kJointTorqueCost);
  EXPECT_EQ(*stringToMpcCostType("dcm_terminal_cost"), MpcCostType::kDcmTerminalCost);
  EXPECT_EQ(*stringToMpcCostType("com_and_acom_tracking_cost"), MpcCostType::kComAndAcomTrackingCost);
  EXPECT_EQ(*stringToMpcCostType("comAndAcomTrackingCost"), MpcCostType::kComAndAcomTrackingCost);
}

TEST(MpcFormulationNames, everyEnumeratorRoundTripsThroughItsName) {
  // The canonical name each enumerator prints is one the loader reads back as the same enumerator, so a name in a log
  // line can always be pasted into a task file.
  for (MpcSoftConstraintType type :
       {MpcSoftConstraintType::kJointLimits, MpcSoftConstraintType::kFootCollision, MpcSoftConstraintType::kFrictionForceCone,
        MpcSoftConstraintType::kContactMomentXy, MpcSoftConstraintType::kContactWrenchCone, MpcSoftConstraintType::kZeroVelocity,
        MpcSoftConstraintType::kNormalVelocity, MpcSoftConstraintType::kContactComplementarity, MpcSoftConstraintType::kForceWeightedSlip,
        MpcSoftConstraintType::kGroundPenetration}) {
    const absl::StatusOr<std::string> name = mpcSoftConstraintTypeToString(type);
    ASSERT_TRUE(name.ok()) << static_cast<int>(type);
    EXPECT_EQ(*stringToMpcSoftConstraintType(*name), type) << *name;
  }
  for (MpcHardConstraintType type : {MpcHardConstraintType::kZeroWrench, MpcHardConstraintType::kZeroVelocity,
                                     MpcHardConstraintType::kNormalVelocity, MpcHardConstraintType::kKneeJointMimic}) {
    const absl::StatusOr<std::string> name = mpcHardConstraintTypeToString(type);
    ASSERT_TRUE(name.ok()) << static_cast<int>(type);
    EXPECT_EQ(*stringToMpcHardConstraintType(*name), type) << *name;
  }
  for (MpcCostType type : {MpcCostType::kStateInputQuadraticCost, MpcCostType::kStateQuadraticCost, MpcCostType::kInputQuadraticCost,
                           MpcCostType::kTerminalCost, MpcCostType::kIcpCost, MpcCostType::kTaskSpaceFootCost,
                           MpcCostType::kTaskSpaceTorsoCost, MpcCostType::kExternalTorqueCost, MpcCostType::kJointTorqueCost,
                           MpcCostType::kDcmTerminalCost, MpcCostType::kComAndAcomTrackingCost}) {
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
  for (const char* absl_nonnull registered :
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
  EXPECT_GT(named, static_cast<int>(MpcCostType::kComAndAcomTrackingCost)) << "the walk stopped early";
  EXPECT_EQ(offered.size(), static_cast<size_t>(named)) << message;

  EXPECT_FALSE(mpcCostTypeToString(static_cast<MpcCostType>(999)).ok());
  EXPECT_FALSE(mpcSoftConstraintTypeToString(static_cast<MpcSoftConstraintType>(999)).ok());
  EXPECT_FALSE(mpcHardConstraintTypeToString(static_cast<MpcHardConstraintType>(999)).ok());

  // And an unknown name inside a task file fails the conversion rather than being skipped.
  EXPECT_TRUE(refusedMentioning(load(*kGatedHard, {"joint_limits", "no_such_term"}), {"no_such_term"}));
}

TEST(MpcFormulationNames, anUnknownConstraintNameOffersEveryRegisteredConstraintAndNothingElse) {
  // The soft and hard lists used to be written out by hand in the messages, and a term added to a registry was not
  // offered until someone remembered the second list. They are generated from the registries now; every enumerator
  // that has a canonical name must be offered, exactly once.
  const absl::StatusOr<MpcSoftConstraintType> soft = stringToMpcSoftConstraintType("invalid_soft_name");
  ASSERT_FALSE(soft.ok());
  const std::vector<std::string> softNames = everyCanonicalName(&mpcSoftConstraintTypeToString);
  ASSERT_GT(softNames.size(), static_cast<size_t>(MpcSoftConstraintType::kGroundPenetration)) << "the walk stopped early";
  const std::vector<std::string> offeredSoft = offeredNames(soft.status().message());
  for (const std::string& name : softNames) {
    EXPECT_EQ(std::count(offeredSoft.begin(), offeredSoft.end(), name), 1) << name << " in: " << soft.status().message();
  }
  EXPECT_EQ(offeredSoft.size(), softNames.size()) << soft.status().message();

  const absl::StatusOr<MpcHardConstraintType> hard = stringToMpcHardConstraintType("invalid_hard_name");
  ASSERT_FALSE(hard.ok());
  const std::vector<std::string> hardNames = everyCanonicalName(&mpcHardConstraintTypeToString);
  ASSERT_GT(hardNames.size(), static_cast<size_t>(MpcHardConstraintType::kKneeJointMimic)) << "the walk stopped early";
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

TEST(MpcFormulationNames, theNameListsAreTheCanonicalNamesInTheOrderOfTheEnums) {
  // What the tuning GUI offers for the three lists (tools/config_registries) is what the messages offer.
  EXPECT_EQ(mpcCostNames(), everyCanonicalName(&mpcCostTypeToString));
  EXPECT_EQ(mpcSoftConstraintNames(), everyCanonicalName(&mpcSoftConstraintTypeToString));
  EXPECT_EQ(mpcHardConstraintNames(), everyCanonicalName(&mpcHardConstraintTypeToString));
  const absl::StatusOr<MpcCostType> cost = stringToMpcCostType("invalid_cost_name");
  ASSERT_FALSE(cost.ok());
  EXPECT_EQ(offeredNames(cost.status().message()), mpcCostNames());
}

// ---------------------------------------------------------------------------------------------------------------
// Loading: the task file through loadTaskFile() and the formulation conversion, as the MPC interfaces read it.
// ---------------------------------------------------------------------------------------------------------------

TEST(MpcFormulationLoader, aMissingFileIsNotFound) {
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile("/non/existent/path/task.textproto");
  ASSERT_FALSE(task.ok());
  EXPECT_EQ(task.status().code(), absl::StatusCode::kNotFound);
}

TEST(MpcFormulationLoader, theListsLoadAsWritten) {
  const std::string path = writeContent("lists",
                                        "hard_constraints: [\"zero_wrench\", \"zero_velocity\", \"normal_velocity\"]\n"
                                        "soft_constraints: [\"joint_limits\", \"friction_force_cone\", \"contact_moment_xy\"]\n"
                                        "costs: \"state_quadratic_cost\"\n");
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(path);
  ASSERT_TRUE(task.ok()) << task.status();
  const absl::StatusOr<MpcFormulationTasks> tasks = mpcFormulationTasksFromConfig(*task, FormulationLogging::kLogSummary);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_EQ(tasks->hardConstraints.size(), 3U);
  EXPECT_EQ(tasks->softConstraints.size(), 3U);
  EXPECT_EQ(tasks->costs.size(), 1U);
  EXPECT_TRUE(tasks->hasHardConstraint(MpcHardConstraintType::kZeroWrench));
  EXPECT_TRUE(tasks->hasHardConstraint(MpcHardConstraintType::kNormalVelocity));
  EXPECT_TRUE(tasks->hasSoftConstraint(MpcSoftConstraintType::kContactMomentXy));
  EXPECT_TRUE(tasks->hasCost(MpcCostType::kStateQuadraticCost));
  EXPECT_TRUE(contactConstraintsAreScheduleGated(*tasks));
  EXPECT_FALSE(usesContactImplicitFormulation(*tasks));
}

TEST(MpcFormulationLoader, theRetiredAcomSwitchIsRefusedNamingTheCostThatReplacedItWhateverItsValue) {
  // useComAndAcomTracking was a top-level boolean; CoM + ACoM tracking is now the cost com_and_acom_tracking_cost. A
  // file that still carries the key must not load, even with `false`: ignoring it would let a stale file run a
  // different formulation from the one it asks for, silently.
  const std::string lists =
      "hard_constraints: [\"zero_wrench\", \"zero_velocity\", \"normal_velocity\"]\nsoft_constraints: \"joint_limits\"\n";
  for (const char* absl_nonnull value : {"true", "false"}) {
    EXPECT_TRUE(refusedMentioning(parse(absl::StrCat("useComAndAcomTracking: ", value, "\n", lists)).status(),
                                  {"useComAndAcomTracking", "is retired", "com_and_acom_tracking_cost", "costs"}))
        << "useComAndAcomTracking: " << value;
  }

  // Positive control: the same file without the key loads, and the cost that replaced it is read from `costs`.
  const absl::StatusOr<humanoid_mpc_config::TaskFile> listed =
      parse(absl::StrCat(lists, "costs: [\"state_quadratic_cost\", \"com_and_acom_tracking_cost\"]\n"));
  ASSERT_TRUE(listed.ok()) << listed.status();
  mpc_config::TaskFile task;
  ASSERT_TRUE(mpc_config::FromProto(*listed, &task).ok());
  const absl::StatusOr<MpcFormulationTasks> with = mpcFormulationTasksFromConfig(task, FormulationLogging::kQuiet);
  ASSERT_TRUE(with.ok()) << with.status();
  EXPECT_TRUE(with->hasCost(MpcCostType::kComAndAcomTrackingCost));
  EXPECT_TRUE(with->hasCost(MpcCostType::kStateQuadraticCost));
}

TEST(MpcFormulationLoader, theRetiredDcmSwitchIsRefusedNamingTheCostThatReplacedItWhateverItsValue) {
  // useDcmTerminalCost was a top-level boolean; ending the horizon on the DCM is now the cost dcm_terminal_cost, in
  // place of terminal_cost. The key is refused even when false: ignoring it would let a stale file end its horizon on a
  // different cost from the one it asks for, silently.
  for (const char* absl_nonnull value : {"true", "false"}) {
    EXPECT_TRUE(refusedMentioning(parse(absl::StrCat("useDcmTerminalCost: ", value, "\ncosts: \"state_quadratic_cost\"\n")).status(),
                                  {"useDcmTerminalCost", "is retired", "dcm_terminal_cost", "costs", "terminal_cost"}))
        << "useDcmTerminalCost: " << value;
  }

  // Positive control: the cost that replaced it is read from `costs`.
  mpc_config::TaskFile task = taskWith(*kGatedHard, {"joint_limits"});
  task.costs.emplace_back("dcm_terminal_cost");
  const absl::StatusOr<MpcFormulationTasks> listed = mpcFormulationTasksFromConfig(task, FormulationLogging::kQuiet);
  ASSERT_TRUE(listed.ok()) << listed.status();
  EXPECT_TRUE(listed->hasCost(MpcCostType::kDcmTerminalCost));
  EXPECT_FALSE(listed->hasCost(MpcCostType::kTerminalCost));
}

TEST(MpcFormulationLoader, theTwoTerminalCostsAreAlternativesAndAListNamingBothIsRefusedNamingBoth) {
  // When the DCM cost was a boolean, a `terminal_cost` entry beside it was silently ignored. With the list as the switch
  // a list naming both is ambiguous, whatever spelling it uses, and refused rather than resolved by a precedence.
  for (const char* absl_nonnull entry : {"terminal_cost", "dcm_terminal_cost"}) {
    mpc_config::TaskFile alone = taskWith(*kGatedHard, {"joint_limits"});
    alone.costs.emplace_back(entry);
    const absl::StatusOr<MpcFormulationTasks> tasks = mpcFormulationTasksFromConfig(alone, FormulationLogging::kQuiet);
    EXPECT_TRUE(tasks.ok()) << entry << ": " << tasks.status();
  }
  for (const std::vector<std::string>& spelling :
       {std::vector<std::string>{"terminal_cost", "dcm_terminal_cost"}, std::vector<std::string>{"dcmTerminalCost", "terminalCost"}}) {
    mpc_config::TaskFile both = taskWith(*kGatedHard, {"joint_limits"});
    both.costs.insert(both.costs.end(), spelling.begin(), spelling.end());
    EXPECT_TRUE(
        refusedMentioning(mpcFormulationTasksFromConfig(both, FormulationLogging::kQuiet), {"'terminal_cost'", "'dcm_terminal_cost'"}))
        << absl::StrJoin(spelling, ", ");
  }
}

TEST(MpcFormulationLoader, zeroVelocityMayBeSoftButNotBothHardAndSoft) {
  const absl::StatusOr<MpcFormulationTasks> soft = load({"zero_wrench", "normal_velocity"}, {"zero_velocity"});
  ASSERT_TRUE(soft.ok()) << soft.status();
  EXPECT_TRUE(soft->hasSoftConstraint(MpcSoftConstraintType::kZeroVelocity));
  EXPECT_TRUE(refusedMentioning(load({"zero_wrench", "zero_velocity"}, {"zero_velocity"}), {"zero_velocity"}));
}

TEST(MpcFormulationLoader, normalVelocityMayNotBeBothHardAndSoft) {
  EXPECT_TRUE(refusedMentioning(load(*kGatedHard, {"normal_velocity"}),
                                {"normal_velocity", "listed in both hard_constraints and soft_constraints"}));
}

// ---------------------------------------------------------------------------------------------------------------
// The contact-implicit formulation: the one arrangement that converts, and every half-way one the conversion refuses.
// Each refusal is reached with every EARLIER check satisfied, and asserts a phrase only its own message carries, so
// deleting any single check turns exactly its own test red.
// ---------------------------------------------------------------------------------------------------------------

TEST(ContactImplicitLoader, theCompleteFormulationLoadsWithEitherCone) {
  for (const char* absl_nonnull cone : {"contact_wrench_cone", "friction_force_cone"}) {
    const absl::StatusOr<MpcFormulationTasks> tasks = load(/*hardConstraints=*/{}, contactImplicitSoft(cone));
    ASSERT_TRUE(tasks.ok()) << cone << ": " << tasks.status();
    EXPECT_TRUE(usesContactImplicitFormulation(*tasks)) << cone;
    EXPECT_FALSE(contactConstraintsAreScheduleGated(*tasks)) << cone << ": without zero_wrench the cones must be un-gated";
    EXPECT_TRUE(tasks->hasSoftConstraint(MpcSoftConstraintType::kNormalVelocity)) << cone;
  }
}

TEST(ContactImplicitLoader, anyOneTermMissingIsRefusedNamingIt) {
  // "Listed together or not at all" is what usesContactImplicitFormulation() and every caller of it assume. The loader
  // used to enforce it for contact_complementarity only, so a lone ground_penetration or force_weighted_slip loaded -
  // and then met a later check whose explanation did not fit it.
  const std::vector<std::string> terms = {"contact_complementarity", "force_weighted_slip", "ground_penetration"};
  for (const std::string& missing : terms) {
    const absl::StatusOr<MpcFormulationTasks> tasks = load(/*hardConstraints=*/{}, without(contactImplicitSoft(), missing));
    EXPECT_TRUE(refusedMentioning(tasks, {"together or not at all", absl::StrCat("not '", missing, "'")})) << missing;
  }
}

TEST(ContactImplicitLoader, aLoneComplementarityTermIsRefusedNamingBothMissingTerms) {
  // The third lone term; the other two are below. Both missing terms are named, each with the hole it leaves.
  EXPECT_TRUE(
      refusedMentioning(load(/*hardConstraints=*/{}, {"joint_limits", "contact_wrench_cone", "normal_velocity", "contact_complementarity"}),
                        {"together or not at all", "'force_weighted_slip' (", "'ground_penetration' ("}));
}

TEST(ContactImplicitLoader, complementarityWithoutThePenetrationHingeIsRefused) {
  // The complementarity product is also satisfied by a NEGATIVE height, so without h >= 0 a foot pushed through the
  // floor carries load for free.
  EXPECT_TRUE(refusedMentioning(load(/*hardConstraints=*/{}, without(contactImplicitSoft(), "ground_penetration")),
                                {"'ground_penetration' (without it nothing stops a foot being pushed through the ground"}));
}

TEST(ContactImplicitLoader, complementarityWithoutTheSlipTermIsRefused) {
  EXPECT_TRUE(refusedMentioning(load(/*hardConstraints=*/{}, without(contactImplicitSoft(), "force_weighted_slip")),
                                {"'force_weighted_slip' (without it nothing holds a loaded foot still"}));
}

TEST(ContactImplicitLoader, aLoneGroundPenetrationBesideTheShippedListsGetsTheRightExplanation) {
  // The case the audit traced: an engineer adds ground_penetration to the shipped soft list to keep the toes out of
  // the floor. It used to load past the partial-list check and be refused by the hard normal_velocity one, whose
  // message - about "a touch-down it is being asked to choose" - has nothing to do with it.
  const absl::StatusOr<MpcFormulationTasks> tasks = load(*kGatedHard, {"joint_limits", "contact_wrench_cone", "ground_penetration"});
  EXPECT_TRUE(refusedMentioning(tasks, {"together or not at all", "'contact_complementarity' (", "'force_weighted_slip' ("}));
  EXPECT_FALSE(absl::StrContains(tasks.status().message(), "touch-down")) << tasks.status().message();
}

TEST(ContactImplicitLoader, aLoneSlipTermIsRefused) {
  EXPECT_TRUE(
      refusedMentioning(load(/*hardConstraints=*/{}, {"joint_limits", "contact_wrench_cone", "normal_velocity", "force_weighted_slip"}),
                        {"together or not at all", "'contact_complementarity' (", "'ground_penetration' ("}));
}

TEST(ContactImplicitLoader, complementarityBesideTheHardZeroWrenchIsRefused) {
  EXPECT_TRUE(refusedMentioning(load({"zero_wrench"}, contactImplicitSoft()),
                                {"'contact_complementarity' and the hard 'zero_wrench' constraint are mutually exclusive"}));
}

TEST(ContactImplicitLoader, droppingZeroWrenchWithoutAConeIsRefusedWhateverElseIsListed) {
  // f_n >= 0 is the first of the three conditions of rigid contact, and it is the cone's: the complementarity product
  // is zero at h = 0 for ANY f_n, adhesion included. The cones gate themselves off `zero_wrench`, so dropping it
  // without listing one leaves every foot's wrench unbounded - with or without the contact-implicit terms.
  EXPECT_TRUE(refusedMentioning(load(/*hardConstraints=*/{}, without(contactImplicitSoft(), "contact_wrench_cone")),
                                {"'contact_wrench_cone' or 'friction_force_cone' must be listed"}));
  EXPECT_TRUE(
      refusedMentioning(load({"zero_velocity"}, {"joint_limits"}), {"'contact_wrench_cone' or 'friction_force_cone' must be listed"}));

  // While zero_wrench is listed a missing cone is merely redundant, and the shipped arrangement stays loadable.
  const absl::StatusOr<MpcFormulationTasks> gated = load({"zero_wrench", "zero_velocity"}, {"joint_limits"});
  ASSERT_TRUE(gated.ok()) << gated.status();
  EXPECT_TRUE(contactConstraintsAreScheduleGated(*gated));
}

TEST(ContactImplicitLoader, theSlipTermBesideEitherZeroVelocityIsRefused) {
  // The soft zero_velocity is refused as well as the hard one: both hold the same foot still from the schedule.
  EXPECT_TRUE(refusedMentioning(load(/*hardConstraints=*/{}, with(contactImplicitSoft(), "zero_velocity")),
                                {"'force_weighted_slip' replaces 'zero_velocity'"}));
  EXPECT_TRUE(refusedMentioning(load({"zero_velocity"}, contactImplicitSoft()), {"'force_weighted_slip' replaces 'zero_velocity'"}));
}

TEST(ContactImplicitLoader, theHardNormalVelocityIsRefusedBesideTheFormulation) {
  // Without the soft normal_velocity in the list, so the duplicate check (hard AND soft) cannot be what refuses it.
  EXPECT_TRUE(refusedMentioning(load({"normal_velocity"}, without(contactImplicitSoft(), "normal_velocity")),
                                {"the hard 'normal_velocity' constraint are mutually exclusive", "list it in soft_constraints"}));
}

TEST(ContactImplicitLoader, theFormulationWithoutTheSoftNormalVelocityIsRefused) {
  // With the hard servo refused, the soft one is the only term with the authority to lift a swing foot; the README and
  // the task files call it "not optional" because the robot shuffled without it. It used to load.
  EXPECT_TRUE(refusedMentioning(load(/*hardConstraints=*/{}, without(contactImplicitSoft(), "normal_velocity")),
                                {"needs 'normal_velocity' in soft_constraints", "shuffles"}));
}

// ---------------------------------------------------------------------------------------------------------------
// The contact_implicit block's values.
// ---------------------------------------------------------------------------------------------------------------

class ContactImplicitConfigValidation : public ::testing::TestWithParam<ModelSettings::ContactImplicitKey> {
 protected:
  /** Whether the field is a penalty weight, told from its NAME rather than from the list's own flag it is checked against. */
  bool namedAsAWeight() const { return absl::EndsWith(GetParam().fieldName, "_weight"); }
};

TEST(ContactImplicitKeys, everyKeyIsNamedOnceSetsAFieldOfItsOwnAndIsAWeightExactlyWhenItsNameSaysSo) {
  // The list the validator checks the block against and the parameter updater reloads it from. ModelSettings.cpp
  // static_asserts that it covers every field of ContactImplicitConfig; this checks that no two entries share a name or
  // a field, and that the weight/divisor split the validator relies on is the one the names state.
  const absl::Span<const ModelSettings::ContactImplicitKey> keys = ModelSettings::contactImplicitKeys();
  ASSERT_FALSE(keys.empty());
  for (size_t i = 0; i < keys.size(); ++i) {
    SCOPED_TRACE(keys[i].fieldName);
    EXPECT_EQ(keys[i].isWeight, absl::EndsWith(keys[i].fieldName, "_weight"));
    for (size_t j = 0; j < i; ++j) {
      EXPECT_NE(keys[i].fieldName, keys[j].fieldName);
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
  const std::string expectedKey = absl::StrCat("contact_implicit.", GetParam().fieldName);
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
  EXPECT_EQ(validateContactImplicitConfig(config).ok(), namedAsAWeight()) << GetParam().fieldName;
}

INSTANTIATE_TEST_SUITE_P(EveryKey,
                         ContactImplicitConfigValidation,
                         ::testing::ValuesIn(ModelSettings::contactImplicitKeys().begin(), ModelSettings::contactImplicitKeys().end()),
                         [](const ::testing::TestParamInfo<ModelSettings::ContactImplicitKey>& info) {
                           return std::string(info.param.fieldName);
                         });

// ---------------------------------------------------------------------------------------------------------------
// The contact_implicit block's fields.
// ---------------------------------------------------------------------------------------------------------------

TEST(ContactImplicitBlockFields, everyKeyIsAFieldOfTheSchemaAndEveryFieldOfTheSchemaIsAKey) {
  // The list validateContactImplicitConfig() names its fields from and the schema the task file is parsed with: a field
  // renamed on one side only would make the refusals name a field the file cannot carry.
  const google::protobuf::Descriptor* absl_nonnull block = humanoid_mpc_config::ContactImplicitConfig::descriptor();
  std::vector<std::string> schemaFields;
  for (int i = 0; i < block->field_count(); ++i) schemaFields.emplace_back(block->field(i)->name());
  std::vector<std::string> keyFields;
  for (const ModelSettings::ContactImplicitKey& key : ModelSettings::contactImplicitKeys()) keyFields.emplace_back(key.fieldName);
  std::sort(schemaFields.begin(), schemaFields.end());
  std::sort(keyFields.begin(), keyFields.end());
  EXPECT_EQ(keyFields, schemaFields);
}

TEST(ContactImplicitBlockFields, aRenamedFieldIsRefusedByTheParserAtItsLine) {
  // A field nothing reads is a renamed or misspelled one: the term it was meant for would run on its default and its
  // tuning slider would reach nothing, so the parser refuses it rather than skipping it.
  const absl::StatusOr<humanoid_mpc_config::TaskFile> renamed = parse("contact_implicit {\n  complementarity_weights: 1.0\n}\n");
  EXPECT_TRUE(refusedMentioning(renamed.status(), {"task.textproto:2:", "complementarity_weights"}));
}

TEST(ContactImplicitBlockFields, theGroundThatMovedOutOfTheBlockIsRefusedPointingAtWhereItWent) {
  const absl::StatusOr<humanoid_mpc_config::TaskFile> moved = parse("contact_implicit {\n  terrain_height: 0.0\n}\n");
  EXPECT_TRUE(refusedMentioning(moved.status(), {"'terrain_height' is retired", "top-level terrain_height"}));
}

// ---------------------------------------------------------------------------------------------------------------
// interface.verbose, which the MPC interfaces read before their model settings.
// ---------------------------------------------------------------------------------------------------------------

TEST(InterfaceVerbose, isFalseUnlessTheTaskFileSetsItAndAValueThatIsNotABoolIsRefusedAtItsLine) {
  EXPECT_FALSE(mpc_config::TaskFile{}.interface.verbose);
  for (const bool value : {true, false}) {
    const absl::StatusOr<humanoid_mpc_config::TaskFile> read =
        parse(absl::StrCat("interface {\n  verbose: ", value ? "true" : "false", "\n}\n"));
    ASSERT_TRUE(read.ok()) << read.status();
    EXPECT_EQ(read->interface().verbose(), value);
  }
  EXPECT_TRUE(refusedMentioning(parse("interface {\n  verbose: loud\n}\n").status(), {"task.textproto:2:", "loud"}));
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
  for (const char* absl_nonnull unknown : {"planner", "true", "Gait_Schedule", ""}) {
    const absl::StatusOr<ContactScheduleSource> source = contactScheduleSourceFromName(unknown);
    ASSERT_FALSE(source.ok()) << "'" << unknown << "' was accepted";
    EXPECT_EQ(source.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(source.status().message(), kContactScheduleSourceKey)) << source.status();
    for (const std::string& name : contactScheduleSourceNames()) {
      EXPECT_TRUE(absl::StrContains(source.status().message(), name)) << name << " is not offered: " << source.status();
    }
  }
}

TEST(ContactScheduleSourceField, anAbsentFieldIsTheGaitScheduleAndANamedSourceIsRead) {
  const absl::StatusOr<ContactScheduleSource> absent = contactScheduleSourceFromConfig(mpc_config::TaskFile{});
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_EQ(*absent, kDefaultContactScheduleSource);
  for (const std::string& name : contactScheduleSourceNames()) {
    mpc_config::TaskFile task;
    task.contact_schedule_source = name;
    const absl::StatusOr<ContactScheduleSource> named = contactScheduleSourceFromConfig(task);
    ASSERT_TRUE(named.ok()) << named.status();
    EXPECT_EQ(contactScheduleSourceName(*named), name);
  }
}

TEST(ContactScheduleSourceField, theRetiredBooleanIsRefusedWhateverItsValueNamingItsReplacement) {
  // useContactPlanning: false meant the gait schedule, which is also the default - but a file that still spells the
  // switch the old way was written against the old loader, so it is refused either way, also beside the field that
  // replaced it: the retired one is never read, so it must not silently lose to the name.
  for (const char* absl_nonnull value : {"true", "false"}) {
    const absl::StatusOr<humanoid_mpc_config::TaskFile> refused =
        parse(absl::StrCat(kRetiredContactPlanningKey, ": ", value, "\ncontact_schedule_source: \"gait_schedule\"\n"));
    EXPECT_TRUE(
        refusedMentioning(refused.status(), {std::string(kRetiredContactPlanningKey), "is retired",
                                             absl::StrCat("contact_schedule_source: \"", kContactPlannerContactScheduleSource, "\""),
                                             absl::StrCat("\"", kGaitScheduleContactScheduleSource, "\"")}))
        << kRetiredContactPlanningKey << ": " << value;
  }
}

TEST(ContactScheduleSourceField, aValueThatIsNotANameIsRefused) {
  mpc_config::TaskFile task;
  task.contact_schedule_source = "planner";
  EXPECT_TRUE(
      refusedMentioning(contactScheduleSourceFromConfig(task).status(), {"contact_schedule_source", "gait_schedule, contact_planner"}));
  // A list where the one name belongs is the parser's to refuse, at its line.
  EXPECT_TRUE(refusedMentioning(parse("\ncontact_schedule_source: [\"contact_planner\"]\n").status(), {"task.textproto:2:"}));
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
constexpr const char* absl_nonnull kShippedTaskFiles[] = {
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto",
    "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto",
    "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
    "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
    "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto",
};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:formulation_shipped_task_files_data)
// clang-format on

std::vector<std::filesystem::path> runfilesRoots() {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
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

/** Every robot_models/<robot>/<package>/config/mpc/task.textproto in the runfiles, relative to the runfiles root, sorted. */
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
                                                      package.path().filename().string(), "/config/mpc/task.textproto");
        if (std::filesystem::exists(root / relativePath)) found.push_back(relativePath);
      }
    }
    break;
  }
  std::sort(found.begin(), found.end());
  return found;
}

class ShippedTaskFileFormulation : public ::testing::TestWithParam<const char* absl_nonnull> {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath(GetParam());
    ASSERT_FALSE(taskFile_.empty()) << GetParam() << " is not in the runfiles; add its package to this test's `data`.";
    absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFile_);
    ASSERT_TRUE(task.ok()) << task.status();
    task_ = *std::move(task);
    absl::StatusOr<humanoid_mpc_config::TaskFile> parsed = nproto::ParseTextprotoFile<humanoid_mpc_config::TaskFile>(taskFile_);
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    parsed_ = *std::move(parsed);
  }

  std::string taskFile_;
  // The file as the MPC reads it, and as the parser reads it, for the presence of its fields.
  mpc_config::TaskFile task_;
  humanoid_mpc_config::TaskFile parsed_;
};

TEST_P(ShippedTaskFileFormulation, loadsWithTheContactImplicitFormulationOffAndTheConesGated) {
  const absl::StatusOr<MpcFormulationTasks> tasks = mpcFormulationTasksFromConfig(task_, FormulationLogging::kQuiet);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_FALSE(usesContactImplicitFormulation(*tasks))
      << GetParam() << " switches the contact-implicit formulation on. It changes the closed loop and ships disabled until it has "
      << "been validated in simulation (humanoid_nmpc/docs/contact_implicit_mpc/README.md); commit it switched off.";
  EXPECT_TRUE(contactConstraintsAreScheduleGated(*tasks))
      << GetParam() << " drops the hard 'zero_wrench' constraint, which un-gates every contact cone of the robot.";
}

TEST_P(ShippedTaskFileFormulation, theSameFileSwitchedOnIsSeenAsOn) {
  // Positive control: the file's own lists, rewritten the way the README's section 5 says, make the predicates above
  // flip. Without this a conversion that read no lists at all would pass the test above on every file.
  mpc_config::TaskFile switchedOn = task_;
  switchedOn.hard_constraints.clear();
  switchedOn.soft_constraints = contactImplicitSoft();
  const absl::StatusOr<MpcFormulationTasks> tasks = mpcFormulationTasksFromConfig(switchedOn, FormulationLogging::kQuiet);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_TRUE(usesContactImplicitFormulation(*tasks));
  EXPECT_FALSE(contactConstraintsAreScheduleGated(*tasks));
}

TEST_P(ShippedTaskFileFormulation, aContactImplicitBlockCarriesEveryFieldOfTheBlock) {
  // A file that carries the block carries every field of it, so that none is left to a default nobody chose for the
  // robot. The parser has already refused a field the schema does not know.
  if (!parsed_.has_contact_implicit()) return;
  const google::protobuf::Message& block = parsed_.contact_implicit();
  for (int i = 0; i < block.GetDescriptor()->field_count(); ++i) {
    const google::protobuf::FieldDescriptor* absl_nonnull field = block.GetDescriptor()->field(i);
    EXPECT_TRUE(block.GetReflection()->HasField(block, field)) << GetParam() << " leaves contact_implicit." << field->name() << " out";
  }
}

TEST_P(ShippedTaskFileFormulation, namesTheGaitScheduleAndEndsItsHorizonOnOneTerminalCostAtMost) {
  // The online contact planner changes the closed loop, and ships switched off on every robot, by name: every file
  // spells the field out rather than leaving it to the default, so that the choice is visible where it is made. The
  // retired formulation switches cannot be in it: the parser refuses them.
  EXPECT_TRUE(parsed_.has_contact_schedule_source()) << GetParam() << " does not name its contact_schedule_source";
  const absl::StatusOr<ContactScheduleSource> source = contactScheduleSourceFromConfig(task_);
  ASSERT_TRUE(source.ok()) << source.status();
  EXPECT_EQ(*source, ContactScheduleSource::kGaitSchedule) << GetParam();
  // The conversion refuses both; this says which file broke it.
  const absl::StatusOr<MpcFormulationTasks> tasks = mpcFormulationTasksFromConfig(task_, FormulationLogging::kQuiet);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_FALSE(tasks->hasCost(MpcCostType::kTerminalCost) && tasks->hasCost(MpcCostType::kDcmTerminalCost)) << GetParam();
}

TEST(ShippedTaskFileScan, theRobotsThatConfigureTheFormulationCarryTheBlock) {
  // Positive control for the check above, which returns early for a file without the block.
  size_t filesWithTheBlock = 0;
  for (const char* absl_nonnull relativePath : kShippedTaskFiles) {
    const std::string path = runfilePath(relativePath);
    ASSERT_FALSE(path.empty()) << relativePath;
    const absl::StatusOr<humanoid_mpc_config::TaskFile> parsed = nproto::ParseTextprotoFile<humanoid_mpc_config::TaskFile>(path);
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    if (parsed->has_contact_implicit()) ++filesWithTheBlock;
  }
  EXPECT_GE(filesWithTheBlock, 2U) << "the DRC Atlas and the EngineAI SA01 both configure the contact_implicit block";
}

INSTANTIATE_TEST_SUITE_P(EveryRobot,
                         ShippedTaskFileFormulation,
                         ::testing::ValuesIn(kShippedTaskFiles),
                         [](const ::testing::TestParamInfo<const char* absl_nonnull>& info) {
                           std::string name =
                               std::filesystem::path(info.param).parent_path().parent_path().parent_path().filename().string();
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
