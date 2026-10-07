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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/dynamics/LinearSystemDynamics.h"
#include "ocs2_core/initialization/DefaultInitializer.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_sqp/SqpMpc.h"
#include "ocs2_sqp/SqpSettings.h"

#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/parameter_update/ReferenceManagerApplier.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/mpc_parameter_update.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"

/**
 * The base-height reference against the ground, end to end: the whole-body MPC's reference manager, its target
 * calculator and the procedural motion manager that publishes a target every solve, run in the order the solver runs
 * them. `default_base_height` and the commanded pelvis height are heights above the ground, and the ground is the
 * reference manager's (`terrain_height`, moved on a hot reload). The targets used to be written at an absolute
 * default_base_height, so the base reference ignored the ground at launch and snapped back to it one solve after a reload,
 * while the reference manager lifted the initial target - a measured, absolute pose - by the whole terrain height.
 *
 * Every property is a difference against the same run on flat ground, or against the height the target was published
 * at, so none of them depends on the tuned heights. One test delivers the ground as the tuning GUI does, through the
 * MPC's parameter updater, which runs after the reference manager in a solve. The last test covers the contact planner's
 * reference manager, which keeps a copy of the published target of its own.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kHorizon = 1.0;
constexpr scalar_t kCyclePeriod = 0.01;
// Cycles run on each ground.
constexpr int kSettlingCycles = 10;
// [m] how far below default_base_height the robot's measured base starts.
constexpr scalar_t kCrouch = 0.05;

constexpr char kTaskFile[] = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
constexpr char kUrdfFile[] = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr char kReferenceFile[] = "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto";
constexpr char kGaitFile[] = "humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto";

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/** The shipped G1 whole-body task file; a test failure and an empty file when it does not load. */
mpc_config::TaskFile shippedTaskFile() {
  absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(runfilePath(kTaskFile));
  EXPECT_TRUE(task.ok()) << task.status();
  return task.ok() ? *std::move(task) : mpc_config::TaskFile{};
}

/** The shipped G1 whole-body task file on a ground at `terrainHeight`. */
mpc_config::TaskFile taskFileOnGround(scalar_t terrainHeight) {
  mpc_config::TaskFile task = shippedTaskFile();
  EXPECT_EQ(task.terrain_height, 0.0) << "the shipped file raises the ground; this test sets it";
  task.terrain_height = terrainHeight;
  return task;
}

/** The initial state of `task` as the whole-body MPC reads it; a test failure and the zero state when it does not convert. */
vector_t initialStateOf(const mpc_config::TaskFile& task, const ModelSettings& modelSettings) {
  const StateInputLayout layout = stateInputLayout(modelSettings, StateInputLayout::Mpc::kWholeBody);
  absl::StatusOr<vector_t> state = stateValuesFromConfig(task.initial_state, layout, "initial_state");
  EXPECT_TRUE(state.ok()) << state.status();
  return state.ok() ? *std::move(state) : vector_t(vector_t::Zero(static_cast<Eigen::Index>(stateDimension(layout))));
}

/** The swing trajectory planner of `task`; a test failure and the default planner when its block does not convert. */
std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlannerOf(const mpc_config::TaskFile& task) {
  absl::StatusOr<SwingTrajectoryPlanner::Config> config = swingTrajectorySettingsFromConfig(task.swing_trajectory_config);
  EXPECT_TRUE(config.ok()) << config.status();
  return std::make_unique<SwingTrajectoryPlanner>(config.ok() ? *std::move(config) : SwingTrajectoryPlanner::Config(), kNumContacts);
}

/** A plan that stands on both feet for `numNodes` nodes of `dt` from `startTime`, the center of mass at rest. */
ContactPlan standingPlan(scalar_t startTime, scalar_t dt, size_t numNodes) {
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = startTime;
  plan.dt = dt;
  plan.committedUntil = startTime + dt;
  plan.contacts.assign(numNodes, makeFeetArray(true));
  plan.footholds.assign(numNodes + 1, makeFeetArray(vector2_t(vector2_t::Zero())));
  plan.comPosition.assign(numNodes + 1, vector2_t::Zero());
  plan.comVelocity.assign(numNodes + 1, vector2_t::Zero());
  plan.zmp.assign(numNodes, vector2_t::Zero());
  return plan;
}

/** [m] the base height of every knot of the target `referenceManager` holds. */
std::vector<scalar_t> targetBaseHeights(const ReferenceManagerInterface& referenceManager, const MpcRobotModelBase<scalar_t>& model) {
  std::vector<scalar_t> heights;
  for (const vector_t& state : referenceManager.getTargetTrajectories().stateTrajectory) {
    heights.push_back(model.getBasePose(state)(2));
  }
  return heights;
}

/** Where the target calculator takes the ground from. */
enum class CalculatorGround {
  kReferenceManager,  // the ground the reference manager applied in this solve, as the MPC nodes wire it
  kTaskFile,          // unwired: the task file's `terrain_height`, all a command node outside the MPC knows
};

/** The whole-body MPC's references, wired as the MPC node wires them, on the task file's ground. */
class WholeBodyReferences {
 public:
  explicit WholeBodyReferences(scalar_t terrainHeight, CalculatorGround calculatorGround = CalculatorGround::kReferenceManager) {
    const mpc_config::TaskFile task = taskFileOnGround(terrainHeight);
    const std::string urdfFile = runfilePath(kUrdfFile);
    const std::string referenceFile = runfilePath(kReferenceFile);
    const std::string gaitFile = runfilePath(kGaitFile);
    EXPECT_FALSE(urdfFile.empty() || referenceFile.empty() || gaitFile.empty()) << "the G1 whole-body files are not in the runfiles";
    modelSettings_ = std::make_unique<ModelSettings>(ModelSettings::Create(task, urdfFile, "wb_mpc_", /*verbose=*/false).value());
    absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(task, urdfFile, *modelSettings_);
    EXPECT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
    referenceManager_ =
        std::make_shared<SwitchedModelReferenceManager>(GaitSchedule::Create(referenceFile, *modelSettings_, /*verbose=*/false).value(),
                                                        swingTrajectoryPlannerOf(task), *pinocchioInterface_, *model_);
    calculator_ = WBMpcTargetTrajectoriesCalculator::Create(referenceFile, *model_, kHorizon).value();
    if (calculatorGround == CalculatorGround::kReferenceManager) {
      // What the MPC node wires: the calculator builds on the ground the reference manager applied in this solve.
      calculator_->setTerrainHeightSource([referenceManager = referenceManager_]() { return referenceManager->getAppliedTerrainHeight(); });
    }
    motionManager_ =
        ProceduralMpcMotionManager::Create(gaitFile, referenceFile, referenceManager_, *model_,
                                           [calculator = calculator_.get()](const vector4_t& velocityTarget, scalar_t initTime,
                                                                            scalar_t /*finalTime*/, const vector_t& initState) {
                                             return calculator->commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState);
                                           })
            .value();
    initialState_ = initialStateOf(task, *modelSettings_);
    // The robot stands on this ground, a little crouched, so that the measured base is not the commanded one.
    model_->adaptBasePoseHeight(initialState_, terrainHeight - kCrouch);
    // No pelvis height in the command: every target stands default_base_height above the ground. (The motion manager's
    // command filter ramps against the wall clock, so a commanded height would differ between two runs.)
    motionManager_->setAndScaleVelocityCommand(WalkingVelocityCommand(/*v_x=*/0.0, /*v_y=*/0.0, /*desired_pelvis_h=*/0.0, /*v_yaw=*/0.0));
    // What the MPC node does on its reset: a target at the measured initial state, a world pose already.
    referenceManager_->setTargetTrajectories(TargetTrajectories({0.0}, {initialState_}, {vector_t::Zero(model_->getInputDim())}));
  }

  /**
   * One solver cycle: the reference manager's pre-solve hook, then the motion manager's and the parameter updater's, as
   * the solver orders them (the updater is registered after the motion manager).
   */
  void cycle() {
    const scalar_t time = kCyclePeriod * static_cast<scalar_t>(numCycles_++);
    referenceManager_->preSolverRun(time, time + kHorizon, initialState_, ModeNumber::kStance);
    motionManager_->preSolverRun(time, time + kHorizon, initialState_, *referenceManager_);
    if (parameterUpdater_ != nullptr) parameterUpdater_->preSolverRun(time, time + kHorizon, initialState_, *referenceManager_);
  }

  /** Runs `updater` in every cycle from now on, after the motion manager; it must outlive this object's cycles. */
  void setParameterUpdater(MpcParameterUpdaterModule* absl_nonnull updater) { parameterUpdater_ = updater; }

  /** [m] the base height of every knot of the target the solver tracks in this cycle. */
  std::vector<scalar_t> targetBaseHeights() const { return ocs2::humanoid::targetBaseHeights(*referenceManager_, *model_); }

  SwitchedModelReferenceManager& referenceManager() { return *referenceManager_; }
  WBMpcTargetTrajectoriesCalculator& calculator() { return *calculator_; }
  const ModelSettings& modelSettings() const { return *modelSettings_; }
  const vector_t& initialState() const { return initialState_; }
  const WBAccelMpcRobotModel<scalar_t>& model() const { return *model_; }

 private:
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model_;
  std::shared_ptr<SwitchedModelReferenceManager> referenceManager_;
  std::unique_ptr<WBMpcTargetTrajectoriesCalculator> calculator_;
  std::unique_ptr<ProceduralMpcMotionManager> motionManager_;
  vector_t initialState_;
  int numCycles_ = 0;
  // Run after the motion manager in every cycle when set (setParameterUpdater()).
  MpcParameterUpdaterModule* absl_nullable parameterUpdater_ = nullptr;
};

/**
 * The parameter updater of the whole-body MPC as its node builds it for what the reference manager owns
 * (ReferenceManagerApplier: the ground and the swing trajectories) of `references`, with `task` as the file the MPC
 * started from. The updater writes the problems of an SQP solver; the solver of a one-state stand-in problem holds them,
 * which none of the appliers here writes, so that no CppAD model is built. Not thread-safe.
 */
class GroundUpdater {
 public:
  GroundUpdater(WholeBodyReferences& references, const mpc_config::TaskFile& task) : initializer_(/*inputDim=*/1) {
    problem_.dynamicsPtr = std::make_unique<LinearSystemDynamics>(matrix_t::Zero(1, 1), matrix_t::Identity(1, 1));
    sqp::Settings sqpSettings;
    sqpSettings.nThreads = 1;
    mpc_ = std::make_unique<SqpMpc>(mpc::Settings(), sqpSettings, problem_, initializer_);
    MpcParameterUpdaterModule::Options options;
    options.runningTask = task;
    options.layout = stateInputLayout(references.modelSettings(), StateInputLayout::Mpc::kWholeBody);
    options.inputDim = references.model().getInputDim();
    options.referenceManager = &references.referenceManager();
    options.appliers.push_back(std::make_unique<ReferenceManagerApplier>());
    absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> created = MpcParameterUpdaterModule::Create(mpc_.get(), std::move(options));
    EXPECT_TRUE(created.ok()) << created.status();
    if (created.ok()) updater_ = *std::move(created);
  }

  /** The updater; null when it was refused (a test failure). */
  MpcParameterUpdaterModule* absl_nullable updater() { return updater_.get(); }

 private:
  OptimalControlProblem problem_;
  DefaultInitializer initializer_;
  std::unique_ptr<SqpMpc> mpc_;
  std::unique_ptr<MpcParameterUpdaterModule> updater_;
};

void expectRaisedBy(const std::vector<scalar_t>& raised, const std::vector<scalar_t>& flat, scalar_t height, absl::string_view when) {
  ASSERT_EQ(raised.size(), flat.size()) << when;
  ASSERT_FALSE(raised.empty()) << when;
  for (size_t knot = 0; knot < raised.size(); ++knot) {
    EXPECT_NEAR(raised[knot] - flat[knot], height, 1.0e-12) << when << ", knot " << knot;
  }
}

}  // namespace

TEST(BaseHeightFollowsTerrainTest, aTerrainHeightRaisesTheWholeBaseReferenceByItAndAReloadMovesItOnce) {
  constexpr scalar_t kGround = 0.3;
  constexpr scalar_t kReloadedGround = -0.12;
  WholeBodyReferences flat(0.0);
  WholeBodyReferences raised(kGround);
  // Unwired, a calculator builds on the task file's ground - all a command node outside the MPC can know.
  EXPECT_EQ(WBMpcTargetTrajectoriesCalculator::Create(runfilePath(kReferenceFile), raised.model(), kHorizon).value()->getTerrainHeight(),
            kGround);

  // The first solve tracks the reset target, which is the measured state: a world pose, which the ground must not lift
  // a second time.
  flat.cycle();
  raised.cycle();
  expectRaisedBy(raised.targetBaseHeights(), flat.targetBaseHeights(), kGround, "the reset target");
  EXPECT_EQ(raised.targetBaseHeights().front(), raised.model().getBasePose(raised.initialState())(2));

  // Every target the motion manager publishes stands the commanded height above the ground.
  for (int i = 0; i < kSettlingCycles; ++i) {
    flat.cycle();
    raised.cycle();
    expectRaisedBy(raised.targetBaseHeights(), flat.targetBaseHeights(), kGround, absl::StrCat("published target ", i));
  }
  // Positive control: the published base reference is the commanded height, not the measured one, so the difference
  // above is not the measured state's.
  EXPECT_NEAR(flat.targetBaseHeights().back() - flat.model().getBasePose(flat.initialState())(2), kCrouch, 1.0e-12);

  // A hot reload of the ground moves the target in use once, and every target published after it stands on the new
  // ground; nothing is moved a second time in the cycles that follow.
  raised.referenceManager().setTerrainHeight(kReloadedGround);
  for (int i = 0; i < kSettlingCycles; ++i) {
    flat.cycle();
    raised.cycle();
    expectRaisedBy(raised.targetBaseHeights(), flat.targetBaseHeights(), kReloadedGround, absl::StrCat("after the reload, cycle ", i));
  }
  EXPECT_EQ(raised.referenceManager().getAppliedTerrainHeight(), kReloadedGround);
  EXPECT_EQ(raised.calculator().getTerrainHeight(), kReloadedGround);
}

/**
 * The ground of the tuning GUI's live update, through the parameter updater as the MPC node runs it: the updater runs
 * after the reference manager has built the references of a solve, so the solve the update arrives before still stands
 * on the old ground, and the next one moves the target in use by the change, once; every later target stands on the new
 * ground (humanoid_nmpc/humanoid_mpc_config/README.md, "Live updates").
 */
TEST(BaseHeightFollowsTerrainTest, aLiveGroundChangeThroughTheUpdaterTakesEffectAtTheNextSolveAndMovesTheTargetOnce) {
  constexpr scalar_t kGround = 0.3;
  constexpr scalar_t kReloadedGround = -0.12;
  WholeBodyReferences flat(0.0);
  WholeBodyReferences raised(kGround);
  GroundUpdater groundUpdater(raised, taskFileOnGround(kGround));
  MpcParameterUpdaterModule* absl_nullable const updater = groundUpdater.updater();
  ASSERT_NE(updater, nullptr);
  raised.setParameterUpdater(updater);
  for (int i = 0; i <= kSettlingCycles; ++i) {
    flat.cycle();
    raised.cycle();
    expectRaisedBy(raised.targetBaseHeights(), flat.targetBaseHeights(), kGround, absl::StrCat("launch, cycle ", i));
  }

  mpc_config::MpcParameterUpdate update;
  update.task = taskFileOnGround(kReloadedGround);
  updater->enqueueParameterUpdate(update);
  flat.cycle();
  raised.cycle();
  expectRaisedBy(raised.targetBaseHeights(), flat.targetBaseHeights(), kGround, "the solve the update arrived before");
  EXPECT_EQ(raised.referenceManager().getTerrainHeight(), kReloadedGround) << "the updater handed the reference manager the ground";
  EXPECT_EQ(raised.referenceManager().getAppliedTerrainHeight(), kGround) << "the references of that solve were built before";

  for (int i = 0; i < kSettlingCycles; ++i) {
    flat.cycle();
    raised.cycle();
    expectRaisedBy(raised.targetBaseHeights(), flat.targetBaseHeights(), kReloadedGround, absl::StrCat("after the update, cycle ", i));
  }
  EXPECT_EQ(raised.referenceManager().getAppliedTerrainHeight(), kReloadedGround);
  EXPECT_EQ(raised.calculator().getTerrainHeight(), kReloadedGround);
}

/**
 * The control of the reload above: a calculator that is not handed the reference manager's ground builds on the task
 * file's. The target in use at the reload still moves once, but every target published after it stands on the launch
 * ground again - which is why the MPC nodes wire the calculator (setTerrainHeightSource).
 */
TEST(BaseHeightFollowsTerrainTest, anUnwiredCalculatorStaysOnTheTaskFilesGround) {
  constexpr scalar_t kGround = 0.3;
  constexpr scalar_t kReloadedGround = -0.12;
  WholeBodyReferences flat(/*terrainHeight=*/0.0, CalculatorGround::kTaskFile);
  WholeBodyReferences unwired(kGround, CalculatorGround::kTaskFile);
  for (int i = 0; i <= kSettlingCycles; ++i) {
    flat.cycle();
    unwired.cycle();
    expectRaisedBy(unwired.targetBaseHeights(), flat.targetBaseHeights(), kGround, absl::StrCat("launch, cycle ", i));
  }
  unwired.referenceManager().setTerrainHeight(kReloadedGround);
  flat.cycle();
  unwired.cycle();
  expectRaisedBy(unwired.targetBaseHeights(), flat.targetBaseHeights(), kReloadedGround, "the target in use at the reload");
  for (int i = 0; i < kSettlingCycles; ++i) {
    flat.cycle();
    unwired.cycle();
    expectRaisedBy(unwired.targetBaseHeights(), flat.targetBaseHeights(), kGround, absl::StrCat("published after the reload, cycle ", i));
  }
}

/**
 * The contact planner's reference manager keeps a second copy of the target, the operator's as published, and fills
 * the reference past the plan's horizon from it (ContactPlanningReferenceManager::modifyReferences, whose planned_*
 * overrides rewrite the target in use). That copy stands on the same ground as the target in use, so a hot reload of
 * the ground moves it as well. Left behind, it put the tail of the reference back on the launch ground from the run
 * after the reload on, for as long as the operator published nothing new (for good under a pose command).
 */
TEST(BaseHeightFollowsTerrainTest, theContactPlannersCopyOfThePublishedTargetFollowsAReload) {
  constexpr scalar_t kReloadedGround = 0.2;
  const mpc_config::TaskFile task = shippedTaskFile();
  const std::string urdfFile = runfilePath(kUrdfFile);
  const std::string referenceFile = runfilePath(kReferenceFile);
  ASSERT_FALSE(urdfFile.empty() || referenceFile.empty()) << "the G1 whole-body files are not in the runfiles";
  const ModelSettings modelSettings = ModelSettings::Create(task, urdfFile, "wb_mpc_", /*verbose=*/false).value();
  const absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(task, urdfFile, modelSettings);
  ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
  const WBAccelMpcRobotModel<scalar_t> model(modelSettings);
  ContactPlanningConfig config;
  config.shared.comHeight = 0.85;  // no model derives the pendulum here (the library default 0 means "from the model")
  config.formulation.setExecutionRule(term::kPlannedComOverride, /*on=*/true);
  ASSERT_EQ(config.validateStatus(), absl::OkStatus());
  const absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>> created =
      ContactPlanningReferenceManager::Create(GaitSchedule::Create(referenceFile, modelSettings, /*verbose=*/false).value(),
                                              swingTrajectoryPlannerOf(task), *pinocchioInterface, model, config);
  ASSERT_TRUE(created.ok()) << created.status();
  ContactPlanningReferenceManager& referenceManager = **created;

  vector_t state = initialStateOf(task, modelSettings);
  const scalar_t publishedHeight = model.getBasePose(state)(2);
  const vector_t zeroInput = vector_t::Zero(model.getInputDim());
  // The operator publishes once: a standing target that reaches past every horizon below.
  constexpr scalar_t kPublishedUntil = 3.0;
  referenceManager.setTargetTrajectories(TargetTrajectories({0.0, kPublishedUntil}, {state, state}, {zeroInput, zeroInput}));
  // A plan whose horizon ends inside the solver's at every cycle below, so the reference past it is read from the
  // operator's copy, and that is still in use at the last of them.
  const ContactPlan plan = standingPlan(/*startTime=*/0.0, config.planner.dt, /*numNodes=*/5);
  ASSERT_LT(plan.endTime(), kHorizon);
  ASSERT_GT(plan.endTime(), 2 * kSettlingCycles * kCyclePeriod);
  referenceManager.setContactPlan(plan);

  int cycle = 0;
  for (int i = 0; i < kSettlingCycles; ++i, ++cycle) {
    const scalar_t time = kCyclePeriod * static_cast<scalar_t>(cycle);
    referenceManager.preSolverRun(time, time + kHorizon, state, ModeNumber::kStance);
    // Positive control: the plan is in use and its target is the densified one - a knot per plan node, then the
    // operator's own knot past the plan - so the tail below is read from the operator's copy.
    const TargetTrajectories& target = referenceManager.getTargetTrajectories();
    ASSERT_TRUE(referenceManager.planReferencesUsableAt(time));
    ASSERT_EQ(target.timeTrajectory.size(), plan.comPosition.size() + 1);
    ASSERT_EQ(target.timeTrajectory.back(), kPublishedUntil);
    for (const scalar_t height : targetBaseHeights(referenceManager, model)) {
      EXPECT_NEAR(height, publishedHeight, 1.0e-12) << "launch, cycle " << i;
    }
  }

  // The reload moves the whole reference, the tail from the operator's copy included, and it stays moved.
  referenceManager.setTerrainHeight(kReloadedGround);
  for (int i = 0; i < kSettlingCycles; ++i, ++cycle) {
    const scalar_t time = kCyclePeriod * static_cast<scalar_t>(cycle);
    referenceManager.preSolverRun(time, time + kHorizon, state, ModeNumber::kStance);
    ASSERT_TRUE(referenceManager.planReferencesUsableAt(time));
    ASSERT_EQ(referenceManager.getTargetTrajectories().timeTrajectory.back(), kPublishedUntil);
    const std::vector<scalar_t> heights = targetBaseHeights(referenceManager, model);
    for (size_t knot = 0; knot < heights.size(); ++knot) {
      EXPECT_NEAR(heights[knot], publishedHeight + kReloadedGround, 1.0e-12) << "after the reload, cycle " << i << ", knot " << knot;
    }
  }
}

}  // namespace ocs2::humanoid
