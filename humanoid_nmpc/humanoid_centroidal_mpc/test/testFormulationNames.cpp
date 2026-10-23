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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <algorithm>
#include <filesystem>
#include <memory>
#include <ostream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/cost/StateCostCollection.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/config/costs/DcmTerminalCostFromConfig.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "robot_core/ResourcePaths.h"
#include "support/ProblemFingerprint.h"
#include "support/TypedConfigFiles.h"

/**
 * The two formulation choices of the centroidal MPC that were top-level booleans and are selected by name now
 * (humanoid_nmpc/docs/README.md, sections 1 and 2), and the verbosity the interface reads before its model settings:
 *  - `useDcmTerminalCost: true` became `dcm_terminal_cost` in `costs`, in place of `terminal_cost`. The DRC Atlas and
 *    the EngineAI SA01 shipped the boolean, and the problem each assembles from the name must be bit for bit the one the
 *    boolean assembled from the old list;
 *  - `useContactPlanning` became `contact_schedule_source`, "gait_schedule" or "contact_planner";
 *  - a task file still carrying either retired boolean is refused when it is parsed, naming the replacement.
 */
namespace ocs2::humanoid {
namespace {

/** A centroidal MPC package, located in the runfiles by its path in the repository. */
struct CentroidalRobot {
  const char* absl_nonnull name;
  const char* absl_nonnull mpcDirectory;  // holds config/mpc/task.textproto and config/command/reference.textproto
  const char* absl_nonnull urdf;
};

void PrintTo(const CentroidalRobot& robot, std::ostream* absl_nonnull os) {
  *os << robot.name;
}

// The robots that shipped `useDcmTerminalCost: true` beside a `terminal_cost` entry the boolean ignored. The BUILD
// target's `data` lists the same packages.
// LINT.IfChange(dcm_terminal_cost_robots)
constexpr CentroidalRobot kDcmTerminalCostRobots[] = {
    {.name = "drc_atlas",
     .mpcDirectory = "robot_models/drc_atlas/drc_atlas_centroidal_mpc",
     .urdf = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf"},
    {.name = "engineai_sa01",
     .mpcDirectory = "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc",
     .urdf = "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf"},
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/BUILD.bazel:formulation_names_data)

/** The runfiles path of `relativePath`; a missing file fails the test and is empty. */
std::string runfilePath(absl::string_view relativePath) {
  absl::StatusOr<std::string> path = robot::resolveResourcePath(relativePath);
  EXPECT_TRUE(path.ok()) << path.status();
  return path.ok() ? *std::move(path) : std::string();
}

/** Whether `list` names `entry`. */
bool lists(const std::vector<std::string>& list, absl::string_view entry) {
  return std::find(list.begin(), list.end(), entry) != list.end();
}

class FormulationNamesTest : public ::testing::TestWithParam<CentroidalRobot> {
 protected:
  void SetUp() override {
    files_.taskFile = runfilePath(absl::StrCat(GetParam().mpcDirectory, "/config/mpc/task.textproto"));
    files_.referenceFile = runfilePath(absl::StrCat(GetParam().mpcDirectory, "/config/command/reference.textproto"));
    files_.urdfFile = runfilePath(GetParam().urdf);
    ASSERT_FALSE(files_.taskFile.empty() || files_.referenceFile.empty() || files_.urdfFile.empty())
        << GetParam().name << ": the robot's files are not in the runfiles; add its packages to `data`";
    absl::StatusOr<CentroidalMpcConfig> config = loadConfigOf(files_);
    ASSERT_TRUE(config.ok()) << config.status();
    shipped_ = *std::move(config);
    tmpDir_ = (std::filesystem::path(testing::TempDir()) / "formulation_names" / GetParam().name).string();
  }

  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove_all(tmpDir_, ignored);
  }

  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> create(const CentroidalMpcConfig& config) const {
    return CentroidalMpcInterface::Create(config, files_.urdfFile);
  }

  /** The refusal of the shipped configuration with `task` as its task file; OK when it is accepted. */
  absl::Status refusal(const mpc_config::TaskFile& task) const {
    CentroidalMpcConfig config = shipped_;
    config.task = task;
    const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = create(config);
    return created.ok() ? absl::OkStatus() : created.status();
  }

  /** The refusal of the robot's files with `taskText` as the text of its task file, read by path; OK when accepted. */
  absl::Status textRefusal(absl::string_view name, absl::string_view taskText) const {
    absl::StatusOr<CentroidalRobotFiles> files = writeConfig(absl::StrCat(tmpDir_, "/", name), shipped_, files_.urdfFile);
    if (!files.ok()) return files.status();
    absl::Status written = writeTextFile(files->taskFile, taskText);
    if (!written.ok()) return written;
    const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created =
        CentroidalMpcInterface::Create(files->taskFile, files->urdfFile, files->referenceFile);
    return created.ok() ? absl::OkStatus() : created.status();
  }

  CentroidalRobotFiles files_;
  CentroidalMpcConfig shipped_;
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
  ASSERT_TRUE(lists(shipped_.task.costs, "dcm_terminal_cost")) << GetParam().name << " no longer lists dcm_terminal_cost";
  ASSERT_FALSE(lists(shipped_.task.costs, "terminal_cost")) << GetParam().name << " lists both terminal costs";

  CentroidalMpcConfig oldListConfig = shipped_;
  std::replace(oldListConfig.task.costs.begin(), oldListConfig.task.costs.end(), std::string("dcm_terminal_cost"),
               std::string("terminal_cost"));
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> listed = create(shipped_);
  ASSERT_TRUE(listed.ok()) << listed.status();
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> oldList = create(oldListConfig);
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
  const absl::StatusOr<DcmTerminalCost::Config> config = dcmTerminalCostConfigFromConfig(oldListConfig.task.dcm_terminal_cost);
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
  for (const char* absl_nonnull value : {"true", "false"}) {
    SCOPED_TRACE(value);
    const absl::Status refused =
        textRefusal(absl::StrCat("retiredDcm_", value), absl::StrCat(taskFileText(shipped_.task), "useDcmTerminalCost: ", value, "\n"));
    ASSERT_FALSE(refused.ok()) << "useDcmTerminalCost: " << value << " was accepted";
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), "is retired")) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), "dcm_terminal_cost")) << refused;
  }
}

/** Listing both terminal costs is ambiguous now that the list is the switch, and refused naming both. */
TEST_P(FormulationNamesTest, bothTerminalCostsAreRefusedNamingBoth) {
  mpc_config::TaskFile task = shipped_.task;
  task.costs.emplace_back("terminal_cost");
  const absl::Status refused = refusal(task);
  ASSERT_FALSE(refused.ok()) << "a costs list naming both terminal costs was accepted";
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "terminal_cost")) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "dcm_terminal_cost")) << refused;
}

// googletest's macro defines a static function and reads std::tuple_size<...>::value.
// NOLINTNEXTLINE(misc-use-anonymous-namespace): expanded from INSTANTIATE_TEST_SUITE_P.
INSTANTIATE_TEST_SUITE_P(ShippedTheBoolean,
                         FormulationNamesTest,
                         ::testing::ValuesIn(kDcmTerminalCostRobots),
                         [](const ::testing::TestParamInfo<CentroidalRobot>& info) { return std::string(info.param.name); });

class AtlasFormulationNamesTest : public FormulationNamesTest {};

/** `useContactPlanning` is refused whatever its value, naming the field that replaced it. */
TEST_P(AtlasFormulationNamesTest, theRetiredContactPlanningBooleanIsRefusedNamingTheSource) {
  for (const char* absl_nonnull value : {"true", "false"}) {
    SCOPED_TRACE(value);
    const absl::Status refused = textRefusal(absl::StrCat("retiredPlanning_", value),
                                             absl::StrCat(taskFileText(shipped_.task), "useContactPlanning: ", value, "\n"));
    ASSERT_FALSE(refused.ok()) << "useContactPlanning: " << value << " was accepted";
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), "is retired")) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), "contact_schedule_source")) << refused;
  }
}

/** An unknown source is refused, naming the field and listing every registered name. */
TEST_P(AtlasFormulationNamesTest, anUnknownContactScheduleSourceIsRefusedListingTheNames) {
  mpc_config::TaskFile task = shipped_.task;
  task.contact_schedule_source = "planner";
  const absl::Status refused = refusal(task);
  ASSERT_FALSE(refused.ok()) << "an unknown contact schedule source was accepted";
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "contact_schedule_source")) << refused;
  for (const std::string& name : contactScheduleSourceNames()) {
    EXPECT_TRUE(absl::StrContains(refused.message(), name)) << name << " is not offered: " << refused;
  }
}

/**
 * The interface used to hand its model settings the string literal "true" as their `verbose` flag, which converts to
 * the bool true, so every start-up printed the settings banner whatever interface.verbose said. The settings are
 * converted before the contact schedule source is, so a configuration refused for its source has already shown whether
 * the banner followed the flag.
 */
TEST_P(AtlasFormulationNamesTest, theModelSettingsBannerFollowsInterfaceVerbose) {
  mpc_config::TaskFile task = shipped_.task;
  task.contact_schedule_source = "not_a_source";
  for (const bool verbose : {false, true}) {
    SCOPED_TRACE(verbose);
    task.interface.verbose = verbose;
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr("Robot Model Settings")))
        .Times(verbose ? testing::AtLeast(1) : testing::Exactly(0));
    log.StartCapturingLogs();
    const absl::Status refused = refusal(task);
    log.StopCapturingLogs();
    EXPECT_FALSE(refused.ok());
  }
}

/** A value of interface.verbose that is not a bool does not parse, and the parser names the field and its position. */
TEST_P(AtlasFormulationNamesTest, anInterfaceVerboseThatIsNotABoolIsRefusedNamingTheField) {
  mpc_config::TaskFile task = shipped_.task;
  task.interface.verbose = true;
  const std::string text = taskFileText(task);
  const std::string broken = absl::StrReplaceAll(text, {{"verbose: true", "verbose: loud"}});
  ASSERT_NE(broken, text);
  const absl::Status refused = textRefusal("verboseNotABool", broken);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "task.textproto:")) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "verbose")) << refused;
}

// googletest's macro defines a static function and reads std::tuple_size<...>::value.
// NOLINTNEXTLINE(misc-use-anonymous-namespace): expanded from INSTANTIATE_TEST_SUITE_P.
INSTANTIATE_TEST_SUITE_P(DrcAtlas,
                         AtlasFormulationNamesTest,
                         ::testing::Values(kDcmTerminalCostRobots[0]),
                         [](const ::testing::TestParamInfo<CentroidalRobot>& info) { return std::string(info.param.name); });

}  // namespace ocs2::humanoid
