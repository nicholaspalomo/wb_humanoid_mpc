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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <system_error>
#include <vector>

#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include <ocs2_core/cost/StateCostCollection.h>

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "support/ProblemFingerprint.h"

/**
 * The two formulation choices of the centroidal MPC that were top-level booleans and are selected by name now
 * (humanoid_nmpc/docs/README.md, sections 1 and 2), and the verbosity the interface reads before its model settings:
 *  - `useDcmTerminalCost: true` became `dcm_terminal_cost` in `costs`, in place of `terminal_cost`. The DRC Atlas and
 *    the EngineAI SA01 shipped the boolean, and the problem each assembles from the name must be bit for bit the one the
 *    boolean assembled from the old list;
 *  - `useContactPlanning` became `contactScheduleSource`, `gait_schedule` or `contact_planner`;
 *  - a task file still carrying either retired boolean is refused at start-up, naming the replacement.
 */
namespace ocs2::humanoid {
namespace {

/** A centroidal MPC package, located in the runfiles by its path in the repository. */
struct CentroidalRobot {
  const char* name;
  const char* mpcDirectory;  // holds config/mpc/task.yaml and config/command/reference.yaml
  const char* urdf;
};

void PrintTo(const CentroidalRobot& robot, std::ostream* os) {
  *os << robot.name;
}

// The robots that shipped `useDcmTerminalCost: true` beside a `terminal_cost` entry the boolean ignored. The BUILD
// target's `data` lists the same packages.
// LINT.IfChange(dcm_terminal_cost_robots)
constexpr CentroidalRobot kDcmTerminalCostRobots[] = {
    {"drc_atlas", "robot_models/drc_atlas/drc_atlas_centroidal_mpc", "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf"},
    {"engineai_sa01", "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc",
     "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf"},
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/BUILD.bazel:formulation_names_data)

constexpr absl::string_view kDcmTerminalCostEntry = "\n  - dcm_terminal_cost\n";
constexpr absl::string_view kQuadraticTerminalCostEntry = "\n  - terminal_cost\n";

/** The absolute path of a data file of this test, or empty when the runfiles do not contain it. */
std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
    roots.emplace_back(std::filesystem::path(srcDir) / "wb_humanoid_mpc");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/** `content` with `from` replaced by `to` exactly once; fails the test when `from` does not occur exactly once. */
std::string replacedOnce(const std::string& content, absl::string_view from, absl::string_view to) {
  const std::string::size_type position = content.find(from);
  EXPECT_NE(position, std::string::npos) << "'" << from << "' not found";
  if (position == std::string::npos) return content;
  EXPECT_EQ(content.find(from, position + 1), std::string::npos) << "'" << from << "' occurs more than once";
  std::string result = content;
  result.replace(position, from.size(), std::string(to));
  return result;
}

/** `content` with interface.verbose set to `verbose`. */
std::string withInterfaceVerbose(const std::string& content, bool verbose) {
  const std::string key = "\ninterface:\n  verbose: ";
  const std::string::size_type position = content.find(key);
  EXPECT_NE(position, std::string::npos) << "interface.verbose not found";
  if (position == std::string::npos) return content;
  const std::string::size_type valueStart = position + key.size();
  const std::string::size_type valueEnd = content.find_first_of(" \n#", valueStart);
  std::string result = content;
  result.replace(valueStart, valueEnd - valueStart, verbose ? "true" : "false");
  return result;
}

class FormulationNamesTest : public ::testing::TestWithParam<CentroidalRobot> {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath(absl::StrCat(GetParam().mpcDirectory, "/config/mpc/task.yaml"));
    referenceFile_ = runfilePath(absl::StrCat(GetParam().mpcDirectory, "/config/command/reference.yaml"));
    urdfFile_ = runfilePath(GetParam().urdf);
    ASSERT_FALSE(taskFile_.empty() || referenceFile_.empty() || urdfFile_.empty())
        << GetParam().name << ": the robot's files are not in the runfiles; add its packages to `data`";
    shipped_ = readFile(taskFile_);
    tmpDir_ = (std::filesystem::path(testing::TempDir()) / "formulation_names" / GetParam().name).string();
    std::filesystem::create_directories(tmpDir_);
  }

  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove_all(tmpDir_, ignored);
  }

  /** Writes a variant of the task file in a directory of its own, so that nothing beside it is shared by accident. */
  std::string writeTaskFile(absl::string_view name, const std::string& content) const {
    const std::filesystem::path directory = std::filesystem::path(tmpDir_) / std::string(name);
    std::filesystem::create_directories(directory);
    const std::string path = (directory / "task.yaml").string();
    std::ofstream out(path);
    out << content;
    return path;
  }

  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> create(const std::string& taskFile) const {
    return CentroidalMpcInterface::Create(taskFile, urdfFile_, referenceFile_);
  }

  /** Creates the interface of a variant that must be refused, and returns the refusal. */
  absl::Status refusal(absl::string_view name, const std::string& content) const {
    const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = create(writeTaskFile(name, content));
    if (created.ok()) return absl::OkStatus();
    return created.status();
  }

  std::string taskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::string shipped_;
  std::string tmpDir_;
};

}  // namespace

/**
 * `useDcmTerminalCost: true` beside a `terminal_cost` entry, as both robots shipped it, built the DCM viability cost as
 * the one terminal cost and ignored the entry. The file now lists `dcm_terminal_cost` in its place. The problem the name
 * assembles must be bit for bit the one the boolean assembled: the boolean's effect is re-applied by hand, from the code
 * that applied it, to the problem built from the old list (terminal_cost), and the two are compared term by term,
 * derivative by derivative, at points along a walking schedule.
 */
TEST_P(FormulationNamesTest, theNamedDcmTerminalCostAssemblesExactlyTheProblemTheBooleanDid) {
  ASSERT_NE(shipped_.find(kDcmTerminalCostEntry), std::string::npos) << GetParam().name << " no longer lists dcm_terminal_cost";
  ASSERT_EQ(shipped_.find(kQuadraticTerminalCostEntry), std::string::npos) << GetParam().name << " lists both terminal costs";

  const std::string listedFile = writeTaskFile("listed", shipped_);
  const std::string oldListFile = writeTaskFile("oldList", replacedOnce(shipped_, kDcmTerminalCostEntry, kQuadraticTerminalCostEntry));
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> listed = create(listedFile);
  ASSERT_TRUE(listed.ok()) << listed.status();
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> oldList = create(oldListFile);
  ASSERT_TRUE(oldList.ok()) << oldList.status();
  // The shipped contact schedule is the gait schedule, by name.
  EXPECT_EQ((*listed)->contactScheduleSource(), ContactScheduleSource::kGaitSchedule);
  EXPECT_FALSE((*listed)->usesContactPlanning());

  const test::Fingerprint listedFingerprint = test::fingerprintOf(**listed);
  // The comparison below is only meaningful if evaluating a problem twice gives the same numbers.
  ASSERT_TRUE(test::identical(test::fingerprintOf(**listed), listedFingerprint));
  // Positive control: the terminal cost changes the problem, so the comparison can fail.
  ASSERT_FALSE(test::identical(test::fingerprintOf(**oldList), listedFingerprint));

  // The effect of `useDcmTerminalCost: true`, re-applied by hand to the problem built from the old list: the final cost
  // was the DCM cost alone. terminal_cost was ignored, and so was the terminal CoM + ACoM instance that is built only
  // beside it.
  CentroidalMpcInterface& interface = **oldList;
  ASSERT_EQ(interface.getOptimalControlProblem().finalCostPtr->getTermNameMap().count("terminalCost"), 1U);
  const absl::StatusOr<DcmTerminalCost::Config> config =
      DcmTerminalCost::loadConfig(oldListFile, DcmTerminalCost::kConfigPrefix, /*verbose=*/false);
  ASSERT_TRUE(config.ok()) << config.status();
  absl::StatusOr<std::unique_ptr<DcmTerminalCost>> dcmTerminalCost = DcmTerminalCost::Create(
      *interface.getSwitchedModelReferenceManagerPtr(), *config, interface.getNominalComHeight(), interface.getPinocchioInterface(),
      interface.getEffectiveMpcRobotModelAD(), DcmTerminalCost::kTermName, interface.modelSettings());
  ASSERT_TRUE(dcmTerminalCost.ok()) << dcmTerminalCost.status();
  OptimalControlProblem& problem = interface.getOptimalControlProblemRef();
  problem.finalCostPtr = std::make_unique<StateCostCollection>();
  problem.finalCostPtr->add(DcmTerminalCost::kTermName, *std::move(dcmTerminalCost));

  EXPECT_TRUE(test::identical(test::fingerprintOf(interface), listedFingerprint));
}

/** The retired boolean is refused whatever its value, naming the entry that replaced it, before any term is built. */
TEST_P(FormulationNamesTest, theRetiredDcmBooleanIsRefusedNamingTheCostThatReplacedIt) {
  for (const char* value : {"true", "false"}) {
    SCOPED_TRACE(value);
    const absl::Status refused = refusal(absl::StrCat("retiredDcm_", value), absl::StrCat("useDcmTerminalCost: ", value, "\n", shipped_));
    ASSERT_FALSE(refused.ok()) << "useDcmTerminalCost: " << value << " was accepted";
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), "useDcmTerminalCost")) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), "dcm_terminal_cost")) << refused;
  }
}

/** Listing both terminal costs is ambiguous now that the list is the switch, and refused naming both. */
TEST_P(FormulationNamesTest, bothTerminalCostsAreRefusedNamingBoth) {
  const absl::Status refused =
      refusal("bothTerminalCosts", replacedOnce(shipped_, kDcmTerminalCostEntry, "\n  - dcm_terminal_cost\n  - terminal_cost\n"));
  ASSERT_FALSE(refused.ok()) << "a costs list naming both terminal costs was accepted";
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "'terminal_cost'")) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "'dcm_terminal_cost'")) << refused;
}

INSTANTIATE_TEST_SUITE_P(ShippedTheBoolean,
                         FormulationNamesTest,
                         ::testing::ValuesIn(kDcmTerminalCostRobots),
                         [](const ::testing::TestParamInfo<CentroidalRobot>& info) { return std::string(info.param.name); });

class AtlasFormulationNamesTest : public FormulationNamesTest {};

/** `useContactPlanning` is refused whatever its value, naming the key that replaced it and both of its names. */
TEST_P(AtlasFormulationNamesTest, theRetiredContactPlanningBooleanIsRefusedNamingTheSource) {
  for (const char* value : {"true", "false"}) {
    SCOPED_TRACE(value);
    const absl::Status refused =
        refusal(absl::StrCat("retiredPlanning_", value), absl::StrCat("useContactPlanning: ", value, "\n", shipped_));
    ASSERT_FALSE(refused.ok()) << "useContactPlanning: " << value << " was accepted";
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), kRetiredContactPlanningKey)) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), absl::StrCat(kContactScheduleSourceKey, ": ", kContactPlannerContactScheduleSource)))
        << refused;
  }
}

/** An unknown source is refused, naming the key and listing every registered name. */
TEST_P(AtlasFormulationNamesTest, anUnknownContactScheduleSourceIsRefusedListingTheNames) {
  const absl::Status refused =
      refusal("unknownSource", replacedOnce(shipped_, "\ncontactScheduleSource: gait_schedule\n", "\ncontactScheduleSource: planner\n"));
  ASSERT_FALSE(refused.ok()) << "an unknown contact schedule source was accepted";
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), kContactScheduleSourceKey)) << refused;
  for (const std::string& name : contactScheduleSourceNames()) {
    EXPECT_TRUE(absl::StrContains(refused.message(), name)) << name << " is not offered: " << refused;
  }
}

/**
 * The interface used to hand its model settings the string literal "true" as their `verbose` flag, which converts to
 * the bool true, so every start-up printed the settings banner whatever interface.verbose said. The settings are loaded
 * by the constructor, before the contact schedule source is read, so a file refused for its source has already shown
 * whether the banner followed the flag.
 */
TEST_P(AtlasFormulationNamesTest, theModelSettingsBannerFollowsInterfaceVerbose) {
  const std::string refusedSource =
      replacedOnce(shipped_, "\ncontactScheduleSource: gait_schedule\n", "\ncontactScheduleSource: not_a_source\n");
  for (const bool verbose : {false, true}) {
    SCOPED_TRACE(verbose);
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr("Robot Model Settings")))
        .Times(verbose ? testing::AtLeast(1) : testing::Exactly(0));
    log.StartCapturingLogs();
    const absl::Status refused = refusal(absl::StrCat("verbose_", verbose), withInterfaceVerbose(refusedSource, verbose));
    log.StopCapturingLogs();
    EXPECT_FALSE(refused.ok());
  }
}

/** A value of interface.verbose that is not a bool is refused by Create(), naming the key. */
TEST_P(AtlasFormulationNamesTest, anInterfaceVerboseThatIsNotABoolIsRefusedNamingTheKey) {
  const std::string content = withInterfaceVerbose(shipped_, /*verbose=*/true);
  const absl::Status refused =
      refusal("verboseNotABool", replacedOnce(content, "\ninterface:\n  verbose: true", "\ninterface:\n  verbose: loud"));
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), ModelSettings::kInterfaceVerboseKey)) << refused;
}

INSTANTIATE_TEST_SUITE_P(DrcAtlas,
                         AtlasFormulationNamesTest,
                         ::testing::Values(kDcmTerminalCostRobots[0]),
                         [](const ::testing::TestParamInfo<CentroidalRobot>& info) { return std::string(info.param.name); });

}  // namespace ocs2::humanoid
