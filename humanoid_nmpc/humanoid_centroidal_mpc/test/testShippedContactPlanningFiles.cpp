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

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/config/costs/DcmTerminalCostFromConfig.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/contact_planning/ContactPlanningFromConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::humanoid {

namespace {

/**
 * The contact_planning.textproto of every robot that ships one, next to the task.textproto it belongs to, read from the
 * files on disk rather than from values copied into a test.
 *
 * Neither robot runs the planner in its shipped closed loop (both task files select contact_schedule_source:
 * "gait_schedule"), which is exactly why a file could drift into a configuration its own checks reject - SA01's H-LIP
 * block clipped the trailing step of every full-stick sidestep - with nothing noticing until the planner was switched on.
 * The previous stand-in was a test that copied four numbers per robot by hand.
 */
struct ShippedPlannerFile {
  const char* absl_nonnull robot;
  const char* absl_nonnull directory;  // under the runfiles root, i.e. the repository root
  const char* absl_nonnull urdf;       // likewise; the pendulum a shared block without com_height stands for is read off this model
};

void PrintTo(const ShippedPlannerFile& file, std::ostream* absl_nonnull os) {
  *os << file.robot;
}

// The BUILD target's `data` has to list the same packages, or SetUp cannot find the files.
// LINT.IfChange(shipped_contact_planning_files)
constexpr ShippedPlannerFile kShippedPlannerFiles[] = {
    {.robot = "drc_atlas",
     .directory = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc",
     .urdf = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf"},
    {.robot = "engineai_sa01",
     .directory = "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc",
     .urdf = "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf"},
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/BUILD.bazel:shipped_contact_planning_files_data)

/** The directories the main repository's runfiles may be rooted at, most specific first. */
std::vector<std::filesystem::path> runfilesRoots() {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
    roots.emplace_back(std::filesystem::path(srcDir) / "wb_humanoid_mpc");
  }
  roots.emplace_back(std::filesystem::current_path());
  return roots;
}

/** The absolute path of a data file of this test, or empty when the runfiles do not contain it. */
std::string runfilePath(absl::string_view relativePath) {
  for (const std::filesystem::path& root : runfilesRoots()) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

class ShippedContactPlanningFile : public ::testing::TestWithParam<ShippedPlannerFile> {
 protected:
  void SetUp() override {
    plannerFile_ = runfilePath(absl::StrCat(GetParam().directory, "/", kContactPlanningFileName));
    taskFile_ = runfilePath(absl::StrCat(GetParam().directory, "/task.textproto"));
    ASSERT_FALSE(plannerFile_.empty()) << GetParam().robot << " ships no contact_planning.textproto in the runfiles; add its "
                                       << "*_centroidal_mpc filegroup to this test's `data`.";
    ASSERT_FALSE(taskFile_.empty()) << GetParam().robot << ": task.textproto is not in the runfiles";
    ASSERT_EQ(contactPlanningFileBeside(taskFile_), plannerFile_) << "the interface must find this file beside the task file";
    const std::string referenceFile = runfilePath(absl::StrCat(GetParam().directory, "/../command/reference.textproto"));
    ASSERT_FALSE(referenceFile.empty()) << GetParam().robot << ": reference.textproto is not in the runfiles";
    // The files as CentroidalMpcInterface reads them: the task, the reference, and the planner's beside the task file.
    absl::StatusOr<CentroidalMpcConfig> files = loadCentroidalMpcConfig(taskFile_, referenceFile);
    ASSERT_TRUE(files.ok()) << files.status();
    files_ = *std::move(files);
    if (!files_.contactPlanning.has_value()) GTEST_FAIL() << GetParam().robot << ": the planner's file beside the task file was not read";
    planner_ = *files_.contactPlanning;
    // validate = false: the conversion must not be what decides whether this suite can inspect the file.
    const absl::StatusOr<ContactPlanningConfig> converted =
        contactPlanningConfigFromConfig(planner_, ContactPlanningValidation::kDeferUntilModelParametersApplied);
    ASSERT_TRUE(converted.ok()) << converted.status().message();
    fileConfig_ = *converted;

    // The pendulum a com_height left out stands for, read off the robot model exactly as CentroidalMpcInterface reads it.
    const std::string urdfFile = runfilePath(GetParam().urdf);
    ASSERT_FALSE(urdfFile.empty()) << GetParam().robot << ": the URDF is not in the runfiles";
    const ModelSettings modelSettings =
        ModelSettings::Create(files_.task, urdfFile, "testShippedContactPlanningFiles_", /*verbose=*/false).value();
    PinocchioInterface pinocchioInterface =
        loadCustomPinocchioInterface(files_.task, urdfFile, modelSettings, /*scaleTotalMass=*/false).value();
    const CentroidalModelInfo info = centroidalModelInfoOf(files_, pinocchioInterface, modelSettings).value();
    const CentroidalMpcRobotModel<scalar_t> robotModel(modelSettings, pinocchioInterface, info);
    const vector_t initialState = initialStateOf(files_.task, modelSettings).value();
    modelComHeight_ = computeComHeightAboveFeet(robotModel.getGeneralizedCoordinates(initialState), pinocchioInterface, robotModel);
    ASSERT_GT(modelComHeight_, 0.4) << GetParam().robot;

    // The configuration the planner runs: the file's, with the unset comHeight of a file without shared.com_height
    // resolved as ContactPlanningModelParameters resolves it.
    config_ = fileConfig_;
    if (!config_.shared.comHeight.has_value()) config_.shared.comHeight = modelComHeight_;
  }

  /** The planner's file as it is written. */
  const mpc_config::ContactPlanningFile& plannerFile() const { return planner_; }

  std::string plannerFile_;
  std::string taskFile_;
  CentroidalMpcConfig files_;                // the typed files, the planner's included
  mpc_config::ContactPlanningFile planner_;  // the planner's file of files_
  ContactPlanningConfig fileConfig_;         // as the file writes it
  ContactPlanningConfig config_;             // with an unset comHeight resolved to the model's pendulum
  scalar_t modelComHeight_ = 0.0;
};

std::string joinWarnings(const std::vector<std::string>& warnings) {
  std::string out;
  for (const std::string& warning : warnings) absl::StrAppend(&out, "\n  - ", warning);
  return out;
}

}  // namespace

/** The conversion reads what the file says: a few fields compared with the file as written, so a default cannot pass. */
TEST_P(ShippedContactPlanningFile, TheConversionReadsTheValuesTheFileWrites) {
  EXPECT_EQ(config_.planner.type, plannerFile().planner.type);
  EXPECT_EQ(config_.planner.numNodes, plannerFile().planner.num_nodes);
  EXPECT_EQ(fileConfig_.shared.comHeight, plannerFile().shared.com_height) << "absent: the model's, left unset";
  EXPECT_DOUBLE_EQ(config_.hlip.sspDuration, plannerFile().hlip.ssp_duration);
  EXPECT_DOUBLE_EQ(config_.hlip.stepWidth, plannerFile().hlip.step_width);
  EXPECT_EQ(config_.formulation.execution.size(), plannerFile().execution.size());
}

/**
 * Each shipped file must be a configuration with nothing documented to fall or to block, under the planner it selects:
 * enabling the planner on the robot is meant to be a switch, not a tuning exercise.
 */
TEST_P(ShippedContactPlanningFile, ValidatesWithNothingDocumentedToFallOrBlock) {
  // On the pendulum the planner actually runs: the H-LIP's first-step demand and its sidestep checks depend on omega.
  ASSERT_GT(config_.pendulumHeight(), 0.0);
  EXPECT_TRUE(config_.validateStatus().ok()) << config_.validateStatus().message();
  EXPECT_TRUE(config_.warnings().empty()) << GetParam().robot << ":" << joinWarnings(config_.warnings());

  if (canonicalPlannerName(config_.planner.type) == planner::kHlip) {
    EXPECT_TRUE(config_.formulation.hasExecutionRule(term::kPlannedComOverride)) << "README 3c: without it the robot falls";
    EXPECT_LE(HlipContactPlanner::startUpLateralStep(config_), config_.hlip.maxStepWidth) << "README 3b: the first step must fit";
    EXPECT_FALSE(config_.planner.runInBackgroundThread) << "README 6: a feedback law must not run on a stale state";
  }

  // Control: the same file with its trailing sidestep step pushed just inside hlip.minStepWidth is reported, so the
  // empty list above is a finding and not a check that cannot fail. This is the defect SA01's 0.18 m step width had.
  ContactPlanningConfig narrow = config_;
  narrow.planner.type = planner::kHlip;
  const scalar_t drift = narrow.hlip.blend.maxCommandedVelocityY * (narrow.hlip.sspDuration + narrow.hlip.dspDuration);
  narrow.hlip.stepWidth = narrow.hlip.minStepWidth + drift - 0.001;
  EXPECT_FALSE(narrow.warnings().empty());
}

/**
 * The task file's swing_trajectory_config.swing_time_scale scales every swing shorter than itself down in height and
 * velocity, and the planner decides how short the swings are: hlip.ssp_duration under hlip, the quantized minimum swing under
 * lip_miqp. Both planners are checked, because switching planner.type is the one edit the file invites.
 */
TEST_P(ShippedContactPlanningFile, NoPlannedSwingIsShorterThanTheSwingTimeScale) {
  const scalar_t swingTimeScale = files_.task.swing_trajectory_config.swing_time_scale;
  ASSERT_GT(swingTimeScale, 0.0);
  for (const std::string& name : knownPlannerNames()) {
    ContactPlanningConfig switched = config_;
    switched.planner.type = name;
    const std::optional<std::string> warning = switched.swingTimeScaleWarning(swingTimeScale);
    EXPECT_FALSE(warning.has_value()) << GetParam().robot << " under " << name << ": " << warning.value_or("");
  }
  // Control: a swing time scale just above the shortest planned swing is reported.
  EXPECT_TRUE(config_.swingTimeScaleWarning(config_.shortestPlannedSwingDuration() + 0.01).has_value());
}

/**
 * The terminal DCM cost's pendulum is the planner's: both files leave com_height out, which each consumer resolves with
 * the same function of the model (computeComHeightAboveFeet), or write the same explicit override. The two keys live in
 * different files (LINT: <robot>_pendulum <-> <robot>_dcm_pendulum). Checked on what each consumer RESOLVES, not only
 * on what the files write, so a consumer that resolved the model's pendulum differently would fail here.
 */
TEST_P(ShippedContactPlanningFile, ThePlannerPendulumIsTheDcmCostPendulum) {
  EXPECT_EQ(plannerFile().shared.com_height, files_.task.dcm_terminal_cost.com_height);
  EXPECT_DOUBLE_EQ(fileConfig_.shared.gravity, files_.task.dcm_terminal_cost.gravity);
  const absl::StatusOr<DcmTerminalCost::Config> dcmFile = dcmTerminalCostConfigFromConfig(files_.task.dcm_terminal_cost);
  ASSERT_TRUE(dcmFile.ok()) << dcmFile.status();
  const absl::StatusOr<DcmTerminalCost::Config> dcm = DcmTerminalCost::resolveConfig(*dcmFile, modelComHeight_);
  ASSERT_TRUE(dcm.ok()) << dcm.status();
  EXPECT_NEAR(dcm->omega(), config_.omega(), 1.0e-12) << "the DCM cost and the planner would run on different pendulums";
}

// googletest's macro defines a static function and reads std::tuple_size<...>::value.
// NOLINTNEXTLINE(misc-use-anonymous-namespace): expanded from INSTANTIATE_TEST_SUITE_P.
INSTANTIATE_TEST_SUITE_P(ShippedRobots,
                         ShippedContactPlanningFile,
                         ::testing::ValuesIn(kShippedPlannerFiles),
                         [](const ::testing::TestParamInfo<ShippedPlannerFile>& info) { return std::string(info.param.robot); });

}  // namespace ocs2::humanoid
