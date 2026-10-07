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

// The entry points of the MPC's configuration in the common package (humanoid_nmpc/humanoid_mpc_config/README.md, "Reading a file") on the
// typed files: every shipped robot's files load and build its model settings, Pinocchio model, gait schedule, target calculator, motion
// manager and the factory's terms; a file that does not parse, or a block that does not convert, is refused naming the file or the field;
// the factory serves the constructor it was built with.

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/cost/StateCost.h"
#include "ocs2_core/cost/StateInputCost.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/contact_planning/ContactPlanningFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/LayoutRobotModel.h"

namespace ocs2::humanoid {
namespace {

using ::testing::HasSubstr;
using ::testing::StartsWith;

/** A shipped robot configuration, from the repository root (string literals). */
struct Robot {
  const char* absl_nonnull name;
  const char* absl_nonnull package;
  const char* absl_nonnull urdfFile;
  StateInputLayout::Mpc mpc;
  bool hasContactPlanningFile;
};

// LINT.IfChange(shipped_robots)
constexpr std::array<Robot, 5> kRobots = {{
    {.name = "drc_atlas_centroidal_mpc",
     .package = "robot_models/drc_atlas/drc_atlas_centroidal_mpc",
     .urdfFile = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
     .mpc = StateInputLayout::Mpc::kCentroidal,
     .hasContactPlanningFile = true},
    {.name = "engineai_sa01_centroidal_mpc",
     .package = "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc",
     .urdfFile = "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf",
     .mpc = StateInputLayout::Mpc::kCentroidal,
     .hasContactPlanningFile = true},
    {.name = "g1_centroidal_mpc",
     .package = "robot_models/unitree_g1/g1_centroidal_mpc",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
     .mpc = StateInputLayout::Mpc::kCentroidal,
     .hasContactPlanningFile = false},
    {.name = "g1_wb_mpc",
     .package = "robot_models/unitree_g1/g1_wb_mpc",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
     .mpc = StateInputLayout::Mpc::kWholeBody,
     .hasContactPlanningFile = false},
    {.name = "unitree_r1_centroidal_mpc",
     .package = "robot_models/unitree_r1/unitree_r1_centroidal_mpc",
     .urdfFile = "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf",
     .mpc = StateInputLayout::Mpc::kCentroidal,
     .hasContactPlanningFile = false},
}};
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:config_test_data)

constexpr absl::string_view kGaitFile = "humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto";

/** `relativePath` in the test's runfiles. */
std::string runfilesPath(absl::string_view relativePath) {
  const char* absl_nullable directory = std::getenv("TEST_SRCDIR");
  return directory == nullptr ? std::string(relativePath) : absl::StrCat(directory, "/_main/", relativePath);
}

std::string taskFileOf(const Robot& robot) {
  return runfilesPath(absl::StrCat(robot.package, "/config/mpc/task.textproto"));
}

std::string referenceFileOf(const Robot& robot) {
  return runfilesPath(absl::StrCat(robot.package, "/config/command/reference.textproto"));
}

std::string mpcNameOf(const Robot& robot) {
  return robot.mpc == StateInputLayout::Mpc::kWholeBody ? "wb_mpc_" : "centroidal_mpc_";
}

/** Writes `content` to `name` in the test's scratch directory and returns its path. */
std::string writeFile(absl::string_view name, absl::string_view content) {
  const std::string path = absl::StrCat(::testing::TempDir(), "/", name);
  std::ofstream(path) << content;
  return path;
}

/** The target calculator base, which the formulations derive from, on its own. */
class CommandLimitsProbe final : public TargetTrajectoriesCalculatorBase {
 public:
  using TargetTrajectoriesCalculatorBase::TargetTrajectoriesCalculatorBase;

  TargetTrajectories commandedPositionToTargetTrajectories(const vector4_t& /*commandedVelocities*/,
                                                           scalar_t /*initTime*/,
                                                           const vector_t& /*initState*/) override {
    return TargetTrajectories();
  }
  TargetTrajectories commandedVelocityToTargetTrajectories(const vector4_t& /*commandedVelocities*/,
                                                           scalar_t /*initTime*/,
                                                           const vector_t& /*initState*/) override {
    return TargetTrajectories();
  }

  scalar_t maxDisplacementVelocityX() const { return maxDisplacementVelocityX_; }
  scalar_t targetRotationVelocity() const { return targetRotationVelocity_; }
  const vector_t& targetJointState() const { return targetJointState_; }
};

/** Everything a robot's files build in the common package: the roots' products. */
class RobotStack {
 public:
  static absl::StatusOr<std::unique_ptr<RobotStack>> Create(const Robot& robot) {
    std::unique_ptr<RobotStack> stack = std::make_unique<RobotStack>();
    absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFileOf(robot));
    if (!task.ok()) return task.status();
    stack->task = *std::move(task);
    absl::StatusOr<mpc_config::ReferenceFile> reference = loadReferenceFile(referenceFileOf(robot));
    if (!reference.ok()) return reference.status();
    stack->reference = *std::move(reference);
    absl::StatusOr<mpc_config::GaitFile> gait = loadGaitFile(runfilesPath(kGaitFile));
    if (!gait.ok()) return gait.status();
    stack->gait = *std::move(gait);

    const std::string urdf = runfilesPath(robot.urdfFile);
    absl::StatusOr<ModelSettings> settings = ModelSettings::Create(stack->task, urdf, mpcNameOf(robot), /*verbose=*/false);
    if (!settings.ok()) return settings.status();
    stack->settings = std::make_unique<ModelSettings>(*std::move(settings));
    absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(stack->task, urdf, *stack->settings);
    if (!pinocchioInterface.ok()) return pinocchioInterface.status();
    stack->pinocchioInterface = std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    stack->model = std::make_unique<test::LayoutRobotModel<scalar_t>>(*stack->settings, robot.mpc);
    stack->modelAd = std::make_unique<test::LayoutRobotModel<ad_scalar_t>>(*stack->settings, robot.mpc);
    absl::StatusOr<std::shared_ptr<GaitSchedule>> gaitSchedule = GaitSchedule::Create(stack->reference, *stack->settings);
    if (!gaitSchedule.ok()) return gaitSchedule.status();
    stack->referenceManager = std::make_shared<SwitchedModelReferenceManager>(
        *gaitSchedule, std::make_shared<SwingTrajectoryPlanner>(SwingTrajectoryPlanner::Config(), /*numFeet=*/2),
        *stack->pinocchioInterface, *stack->model);
    return stack;
  }

  mpc_config::TaskFile task;
  mpc_config::ReferenceFile reference;
  mpc_config::GaitFile gait;
  std::unique_ptr<ModelSettings> settings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;
  std::unique_ptr<test::LayoutRobotModel<scalar_t>> model;
  std::unique_ptr<test::LayoutRobotModel<ad_scalar_t>> modelAd;
  std::shared_ptr<SwitchedModelReferenceManager> referenceManager;
};

class ShippedRobotRootsTest : public ::testing::TestWithParam<Robot> {};

TEST_P(ShippedRobotRootsTest, EveryFileLoadsAndBuildsEveryRoot) {
  const Robot& robot = GetParam();
  const absl::StatusOr<std::unique_ptr<RobotStack>> stack = RobotStack::Create(robot);
  ASSERT_TRUE(stack.ok()) << stack.status();
  const ModelSettings& settings = *(*stack)->settings;
  EXPECT_FALSE(settings.robotName.empty());
  EXPECT_EQ(settings.contactNames.size(), 2U);

  // The path forms of the roots read the same files.
  const absl::StatusOr<ModelSettings> byPath = ModelSettings::Create(taskFileOf(robot), runfilesPath(robot.urdfFile), mpcNameOf(robot));
  ASSERT_TRUE(byPath.ok()) << byPath.status();
  EXPECT_EQ(byPath->mpcModelJointNames, settings.mpcModelJointNames);
  const absl::StatusOr<PinocchioInterface> pinocchioByPath =
      loadCustomPinocchioInterface(taskFileOf(robot), runfilesPath(robot.urdfFile), settings);
  ASSERT_TRUE(pinocchioByPath.ok()) << pinocchioByPath.status();
  EXPECT_EQ(pinocchioByPath->getModel().nframes, (*stack)->pinocchioInterface->getModel().nframes);
  ASSERT_TRUE(GaitSchedule::Create(referenceFileOf(robot), settings).ok());

  // The target calculator and the motion manager on the reference file's checked settings.
  const absl::StatusOr<ReferenceSettings> referenceSettings = referenceSettingsFromConfig((*stack)->reference);
  ASSERT_TRUE(referenceSettings.ok()) << referenceSettings.status();
  const absl::StatusOr<vector_t> defaultJointState =
      defaultJointStateFromConfig((*stack)->reference, settings.mpcModelJointNames, settings.fixedJointNames);
  ASSERT_TRUE(defaultJointState.ok()) << defaultJointState.status();
  CommandLimitsProbe calculator(*referenceSettings, *defaultJointState, *(*stack)->model, /*mpcHorizon=*/1.0);
  EXPECT_EQ(calculator.maxDisplacementVelocityX(), referenceSettings->maxDisplacementVelocityX);
  EXPECT_EQ(calculator.commandedBaseHeight(/*commandedPelvisHeight=*/0.0), settings.terrainHeight + referenceSettings->defaultBaseHeight);
  EXPECT_EQ(calculator.targetJointState(), *defaultJointState);
  // A hot reload of the reference file replaces every limit it sets, the yaw rate a command reset included.
  calculator.setTargetRotationVelocity(referenceSettings->targetRotationVelocity + 1.0);
  calculator.applyCommandLimits(*referenceSettings);
  EXPECT_EQ(calculator.targetRotationVelocity(), referenceSettings->targetRotationVelocity);

  const absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> manager =
      ProceduralMpcMotionManager::Create((*stack)->gait, *referenceSettings, (*stack)->referenceManager, *(*stack)->model,
                                         [](const vector4_t& /*velocityTarget*/, scalar_t /*initTime*/, scalar_t /*finalTime*/,
                                            const vector_t& /*initState*/) { return TargetTrajectories(); });
  ASSERT_TRUE(manager.ok()) << manager.status();
  EXPECT_FALSE((*manager)->gaitMap().empty());
  (*manager)->setAndScaleVelocityCommand(WalkingVelocityCommand(/*v_x=*/1.0, /*v_y=*/1.0, /*desired_pelvis_h=*/0.0, /*v_yaw=*/1.0));
  const WalkingVelocityCommand fullStick = (*manager)->getScaledWalkingVelocityCommand();
  EXPECT_EQ(fullStick.linear_velocity_x, referenceSettings->maxDisplacementVelocityX);
  EXPECT_EQ(fullStick.angular_velocity_z, referenceSettings->maxRotationVelocity);
  EXPECT_EQ((*manager)->getMaxLinearAcceleration(), referenceSettings->maxLinearAcceleration);
  EXPECT_TRUE((*manager)->applyCommandLimits(*referenceSettings).ok());

  // The factory's terms that are not taped with CppAD; every shipped file gives what they read.
  const HumanoidCostConstraintFactory factory(&(*stack)->task, robot.mpc, *(*stack)->referenceManager, *(*stack)->pinocchioInterface,
                                              *(*stack)->model, *(*stack)->modelAd, settings);
  EXPECT_TRUE(factory.makeStateInputQuadraticCost().ok());
  EXPECT_TRUE(factory.makeStateQuadraticCost().ok());
  EXPECT_TRUE(factory.makeInputQuadraticCost().ok());
  EXPECT_TRUE(factory.makeTerminalCost().ok());
  EXPECT_TRUE(factory.makeJointLimitsConstraint().ok());
  EXPECT_TRUE(factory.makeFrictionForceConeConstraint(/*contactPointIndex=*/0).ok());
}

INSTANTIATE_TEST_SUITE_P(Robots, ShippedRobotRootsTest, ::testing::ValuesIn(kRobots), [](const ::testing::TestParamInfo<Robot>& info) {
  return std::string(info.param.name);
});

TEST(ConfigFilesTest, TheContactPlannersFileIsBesideTheTaskFileOrAbsent) {
  for (const Robot& robot : kRobots) {
    SCOPED_TRACE(robot.name);
    EXPECT_THAT(contactPlanningFileBeside(taskFileOf(robot)), ::testing::EndsWith("/config/mpc/contact_planning.textproto"));
    const absl::StatusOr<std::optional<mpc_config::ContactPlanningFile>> file = loadContactPlanningFileBeside(taskFileOf(robot));
    ASSERT_TRUE(file.ok()) << file.status();
    EXPECT_EQ(file->has_value(), robot.hasContactPlanningFile);
    // Its configuration converts, and a robot without one plans with the library's defaults.
    const absl::StatusOr<ContactPlanningConfig> config = contactPlanningConfigFromOptionalFile(
        file->has_value() ? &**file : nullptr, ContactPlanningValidation::kDeferUntilModelParametersApplied);
    ASSERT_TRUE(config.ok()) << config.status();
    if (!robot.hasContactPlanningFile) {
      EXPECT_EQ(config->planner.type, ContactPlanningConfig().planner.type);
      EXPECT_EQ(config->formulation.dynamics, ContactPlanningConfig().formulation.dynamics);
    }
  }
}

TEST(ConfigFilesTest, TheJointPdGainsFileIsInTheConfigDirectoryOfTheTaskFile) {
  for (const Robot& robot : kRobots) {
    SCOPED_TRACE(robot.name);
    const std::string gains = jointPdGainsFileBeside(taskFileOf(robot));
    EXPECT_THAT(gains, ::testing::EndsWith("/config/controller/joint_pd_gains.textproto"));
    const absl::StatusOr<mpc_config::JointPdGainsFile> file = loadJointPdGainsFile(gains);
    EXPECT_TRUE(file.ok()) << file.status();
    const absl::StatusOr<std::string> existing = existingJointPdGainsFileBeside(taskFileOf(robot));
    ASSERT_TRUE(existing.ok()) << existing.status();
    EXPECT_EQ(*existing, gains);
  }
  // A robot directory without the file: the robot process does not start.
  const std::string missing = absl::StrCat(::testing::TempDir(), "/no_gains_robot/config/mpc/task.textproto");
  const absl::StatusOr<std::string> refused = existingJointPdGainsFileBeside(missing);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(refused.status().message(), HasSubstr("no_gains_robot/config/controller/joint_pd_gains.textproto"));
}

TEST(ConfigFilesTest, AContactPlannersFileThatCannotBeLookedAtIsAnErrorNotAbsent) {
  const std::filesystem::path directory = std::filesystem::path(::testing::TempDir()) / "unreadable_contact_planning";
  std::error_code error;
  std::filesystem::create_directories(directory / "mpc", error);
  ASSERT_FALSE(error) << error.message();
  const std::string taskFile = (directory / "mpc" / "task.textproto").string();
  // A link to itself: looking at it fails (ELOOP), which is not "the robot has no such file".
  const std::filesystem::path planner = contactPlanningFileBeside(taskFile);
  std::filesystem::remove(planner, error);
  std::filesystem::create_symlink(planner, planner, error);
  ASSERT_FALSE(error) << error.message();
  absl::StatusOr<std::optional<mpc_config::ContactPlanningFile>> file = loadContactPlanningFileBeside(taskFile);
  EXPECT_EQ(file.status().code(), absl::StatusCode::kUnavailable);
  EXPECT_THAT(file.status().message(), HasSubstr(planner.string()));
  // A directory of that name is not a file either.
  std::filesystem::remove(planner, error);
  std::filesystem::create_directory(planner, error);
  ASSERT_FALSE(error) << error.message();
  file = loadContactPlanningFileBeside(taskFile);
  EXPECT_EQ(file.status().code(), absl::StatusCode::kFailedPrecondition);
  // And no file at all is the robot without one.
  std::filesystem::remove(planner, error);
  ASSERT_FALSE(error) << error.message();
  file = loadContactPlanningFileBeside(taskFile);
  ASSERT_TRUE(file.ok()) << file.status();
  EXPECT_FALSE(file->has_value());
}

TEST(ConfigFilesTest, AFileThatDoesNotParseIsRefusedNamingItsLine) {
  const std::string path = writeFile("unknown_field.textproto",
                                     "# proto-file: humanoid_nmpc/humanoid_mpc_config/task_file.proto\n"
                                     "# proto-message: humanoid_mpc_config.TaskFile\n"
                                     "no_such_field: 1\n");
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(path);
  EXPECT_EQ(task.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(task.status().message(), HasSubstr(absl::StrCat(path, ":3:")));
  // A reference file is not a task file.
  EXPECT_EQ(loadTaskFile(referenceFileOf(kRobots[0])).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(loadReferenceFile(absl::StrCat(::testing::TempDir(), "/no_such_reference.textproto")).ok());
}

TEST(ConfigFilesTest, WithConfigFilePrefixesAnErrorAndLeavesOkAlone) {
  EXPECT_TRUE(withConfigFile(absl::OkStatus(), "task.textproto").ok());
  const absl::Status prefixed = withConfigFile(absl::InvalidArgumentError("state_weights misses a block"), "task.textproto");
  EXPECT_EQ(prefixed.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(prefixed.message(), "task.textproto: state_weights misses a block");
}

TEST(ConfigRootsTest, AConversionErrorOfAPathRootNamesTheFile) {
  // Atlas's task file with an arm joint the model does not have.
  absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFileOf(kRobots[0]));
  ASSERT_TRUE(task.ok()) << task.status();
  std::ifstream shipped(taskFileOf(kRobots[0]));
  const std::string text((std::istreambuf_iterator<char>(shipped)), std::istreambuf_iterator<char>());
  const std::string arm = task->model_settings.arm_joint_names.left_elbow_y;
  ASSERT_FALSE(arm.empty());
  std::string misspelled = text;
  const size_t at = misspelled.find(absl::StrCat("\"", arm, "\""));
  ASSERT_NE(at, std::string::npos);
  misspelled.replace(at, arm.size() + 2, "\"no_such_joint\"");
  const std::string path = writeFile("misspelled_arm_joint.textproto", misspelled);
  const absl::StatusOr<ModelSettings> settings = ModelSettings::Create(path, runfilesPath(kRobots[0].urdfFile), "centroidal_mpc_");
  EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(settings.status().message(), StartsWith(absl::StrCat(path, ": ")));
  EXPECT_THAT(settings.status().message(), HasSubstr("no_such_joint"));
}

class CostConstraintFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<RobotStack>> stack = RobotStack::Create(kRobots[0]);
    ASSERT_TRUE(stack.ok()) << stack.status();
    stack_ = *std::move(stack);
  }

  std::unique_ptr<RobotStack> stack_;
};

TEST_F(CostConstraintFactoryTest, ABlockThatDoesNotConvertIsRefusedNamingIt) {
  mpc_config::TaskFile task = stack_->task;
  task.state_weights = mpc_config::StateWeights();
  task.terminal_cost_scaling.reset();
  const HumanoidCostConstraintFactory factory(&task, StateInputLayout::Mpc::kCentroidal, *stack_->referenceManager,
                                              *stack_->pinocchioInterface, *stack_->model, *stack_->modelAd, *stack_->settings);
  const absl::StatusOr<std::unique_ptr<StateInputCost>> state = factory.makeStateQuadraticCost();
  EXPECT_EQ(state.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(state.status().message(), HasSubstr("state_weights"));
  const absl::StatusOr<std::unique_ptr<StateCost>> terminal = factory.makeTerminalCost();
  EXPECT_EQ(terminal.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(terminal.status().message(), HasSubstr("terminal_cost_scaling"));
  // The blocks the cost does not read are not its concern.
  EXPECT_TRUE(factory.makeInputQuadraticCost().ok());
}

TEST_F(CostConstraintFactoryTest, WeightsOfAnotherMpcThanTheModelsAreRefused) {
  // The whole-body layout of Atlas's joints on the centroidal model: neither the blocks nor the sizes are its.
  const HumanoidCostConstraintFactory factory(&stack_->task, StateInputLayout::Mpc::kWholeBody, *stack_->referenceManager,
                                              *stack_->pinocchioInterface, *stack_->model, *stack_->modelAd, *stack_->settings);
  EXPECT_EQ(factory.makeStateQuadraticCost().status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(CostConstraintFactoryTest, AnExternalTorqueCostIsALegs) {
  const HumanoidCostConstraintFactory factory(&stack_->task, StateInputLayout::Mpc::kCentroidal, *stack_->referenceManager,
                                              *stack_->pinocchioInterface, *stack_->model, *stack_->modelAd, *stack_->settings);
  EXPECT_EQ(factory.makeExternalTorqueQuadraticCost(/*contactPointIndex=*/2).status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid
