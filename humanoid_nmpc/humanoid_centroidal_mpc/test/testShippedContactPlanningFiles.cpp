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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/PropertyTree.h>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"

namespace ocs2::humanoid {

namespace {

/**
 * The contact_planning.yaml of every robot that ships one, next to the task.yaml it belongs to, read from the files on
 * disk rather than from values copied into a test.
 *
 * Neither robot runs the planner in its shipped closed loop (both task files select contactScheduleSource:
 * gait_schedule), which is
 * exactly why a file could drift into a configuration its own checks reject - SA01's H-LIP block clipped the trailing
 * step of every full-stick sidestep - with nothing noticing until the planner was switched on. The previous stand-in
 * was a test that copied four numbers per robot by hand.
 */
struct ShippedPlannerFile {
  const char* robot;
  const char* directory;  // under the runfiles root, i.e. the repository root
  const char* urdf;       // likewise; the pendulum a shared.comHeight of 0 stands for is read off this model
};

void PrintTo(const ShippedPlannerFile& file, std::ostream* os) {
  *os << file.robot;
}

// The BUILD target's `data` has to list the same packages, or SetUp cannot find the files.
// LINT.IfChange(shipped_contact_planning_files)
constexpr ShippedPlannerFile kShippedPlannerFiles[] = {
    {"drc_atlas", "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc",
     "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf"},
    {"engineai_sa01", "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc",
     "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf"},
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/BUILD.bazel:shipped_contact_planning_files_data)

/** The directories the main repository's runfiles may be rooted at, most specific first. */
std::vector<std::filesystem::path> runfilesRoots() {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
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
    plannerFile_ = runfilePath(absl::StrCat(GetParam().directory, "/", kContactPlanningConfigFileName));
    taskFile_ = runfilePath(absl::StrCat(GetParam().directory, "/task.yaml"));
    ASSERT_FALSE(plannerFile_.empty()) << GetParam().robot << " ships no contact_planning.yaml in the runfiles; add its "
                                       << "*_centroidal_mpc filegroup to this test's `data`.";
    ASSERT_FALSE(taskFile_.empty()) << GetParam().robot << ": task.yaml is not in the runfiles";
    ASSERT_EQ(resolveContactPlanningConfigFile(taskFile_), plannerFile_) << "the interface must find this file beside the task file";
    loadData::readPropertyTree(plannerFile_, plannerTree_);
    loadData::readPropertyTree(taskFile_, taskTree_);
    // validate = false: the loader must not be what decides whether this suite can inspect the file.
    const absl::StatusOr<ContactPlanningConfig> loaded =
        loadContactPlanningConfigStatus(plannerFile_, "contact_planning.", /*verbose=*/false, /*validate=*/false);
    ASSERT_TRUE(loaded.ok()) << loaded.status().message();
    fileConfig_ = *loaded;

    // The pendulum a comHeight of 0 stands for, read off the robot model exactly as CentroidalMpcInterface reads it.
    const std::string urdfFile = runfilePath(GetParam().urdf);
    const std::string referenceFile = runfilePath(absl::StrCat(GetParam().directory, "/../command/reference.yaml"));
    ASSERT_FALSE(urdfFile.empty()) << GetParam().robot << ": the URDF is not in the runfiles";
    ASSERT_FALSE(referenceFile.empty()) << GetParam().robot << ": reference.yaml is not in the runfiles";
    const ModelSettings modelSettings(taskFile_, urdfFile, "testShippedContactPlanningFiles_", /*verbose=*/false);
    PinocchioInterface pinocchioInterface = createCustomPinocchioInterface(taskFile_, urdfFile, modelSettings, /*scaleTotalMass=*/false);
    const CentroidalModelInfo info = centroidal_model::createCentroidalModelInfo(
        pinocchioInterface, centroidal_model::loadCentroidalType(taskFile_),
        centroidal_model::loadDefaultJointState(pinocchioInterface.getModel().nq - 6, referenceFile), modelSettings.contactNames3DoF,
        modelSettings.contactNames6DoF);
    const CentroidalMpcRobotModel<scalar_t> robotModel(modelSettings, pinocchioInterface, info);
    vector_t initialState = vector_t::Zero(info.stateDim);
    loadData::loadEigenMatrix(taskFile_, "initialState", initialState);
    modelComHeight_ = computeComHeightAboveFeet(robotModel.getGeneralizedCoordinates(initialState), pinocchioInterface, robotModel);
    ASSERT_GT(modelComHeight_, 0.4) << GetParam().robot;

    // The configuration the planner runs: the file's, with a comHeight of 0 resolved as ContactPlanningModelParameters
    // resolves it.
    config_ = fileConfig_;
    if (config_.shared.comHeight <= 0.0) config_.shared.comHeight = modelComHeight_;
  }

  std::string plannerFile_;
  std::string taskFile_;
  PropertyTree plannerTree_;
  PropertyTree taskTree_;
  ContactPlanningConfig fileConfig_;  // as the file writes it
  ContactPlanningConfig config_;      // with a comHeight of 0 resolved to the model's pendulum
  scalar_t modelComHeight_ = 0.0;
};

std::string joinWarnings(const std::vector<std::string>& warnings) {
  std::string out;
  for (const std::string& warning : warnings) absl::StrAppend(&out, "\n  - ", warning);
  return out;
}

}  // namespace

/** The loader reads what the file says: a few keys compared against the raw YAML tree, so a default cannot pass. */
TEST_P(ShippedContactPlanningFile, TheLoaderReadsTheValuesTheFileWrites) {
  EXPECT_EQ(config_.planner.type, plannerTree_.get<std::string>("contact_planning.planner.type"));
  EXPECT_EQ(config_.planner.numNodes, plannerTree_.get<int>("contact_planning.planner.numNodes"));
  EXPECT_DOUBLE_EQ(fileConfig_.shared.comHeight, plannerTree_.get<scalar_t>("contact_planning.shared.comHeight"));
  EXPECT_DOUBLE_EQ(config_.hlip.sspDuration, plannerTree_.get<scalar_t>("contact_planning.hlip.sspDuration"));
  EXPECT_DOUBLE_EQ(config_.hlip.stepWidth, plannerTree_.get<scalar_t>("contact_planning.hlip.stepWidth"));
  EXPECT_EQ(config_.formulation.execution.size(), plannerTree_.getChild("contact_planning.execution").size());
}

/**
 * Each shipped file must be a configuration with nothing documented to fall or to block, under the planner it selects:
 * enabling the planner on the robot is meant to be a switch, not a tuning exercise.
 */
TEST_P(ShippedContactPlanningFile, ValidatesWithNothingDocumentedToFallOrBlock) {
  // On the pendulum the planner actually runs: the H-LIP's first-step demand and its sidestep checks depend on omega.
  ASSERT_GT(config_.shared.comHeight, 0.0);
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
 * task.yaml swing_trajectory_config.swingTimeScale scales every swing shorter than itself down in height and velocity,
 * and the planner decides how short the swings are: hlip.sspDuration under hlip, the quantized minimum swing under
 * lip_miqp. Both planners are checked, because switching planner.type is the one edit the file invites.
 */
TEST_P(ShippedContactPlanningFile, NoPlannedSwingIsShorterThanTheSwingTimeScale) {
  const scalar_t swingTimeScale = taskTree_.get<scalar_t>("swing_trajectory_config.swingTimeScale");
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
 * The terminal DCM cost's pendulum is the planner's: both files leave it at 0, which each consumer resolves with the same
 * function of the model (computeComHeightAboveFeet), or write the same explicit override. The two keys live in different
 * files (LINT: <robot>_pendulum <-> dcm_terminal_cost_config). Checked on what each consumer RESOLVES, not only on what
 * the files write, so a consumer that resolved 0 differently would fail here.
 */
TEST_P(ShippedContactPlanningFile, ThePlannerPendulumIsTheDcmCostPendulum) {
  EXPECT_DOUBLE_EQ(fileConfig_.shared.comHeight, taskTree_.get<scalar_t>("dcm_terminal_cost.comHeight"));
  EXPECT_DOUBLE_EQ(fileConfig_.shared.gravity, taskTree_.get<scalar_t>("dcm_terminal_cost.gravity"));
  const absl::StatusOr<DcmTerminalCost::Config> dcmFile = DcmTerminalCost::loadConfig(taskFile_);
  ASSERT_TRUE(dcmFile.ok()) << dcmFile.status();
  const absl::StatusOr<DcmTerminalCost::Config> dcm = DcmTerminalCost::resolveConfig(*dcmFile, modelComHeight_);
  ASSERT_TRUE(dcm.ok()) << dcm.status();
  EXPECT_NEAR(dcm->omega(), config_.omega(), 1e-12) << "the DCM cost and the planner would run on different pendulums";
}

INSTANTIATE_TEST_SUITE_P(ShippedRobots,
                         ShippedContactPlanningFile,
                         ::testing::ValuesIn(kShippedPlannerFiles),
                         [](const ::testing::TestParamInfo<ShippedPlannerFile>& info) { return std::string(info.param.robot); });

}  // namespace ocs2::humanoid
