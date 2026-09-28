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

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/reference/TargetTrajectories.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"

/**
 * The base-height reference against the ground, end to end: the whole-body MPC's reference manager, its target
 * calculator and the procedural motion manager that publishes a target every solve, run in the order the solver runs
 * them. `defaultBaseHeight` and the commanded pelvis height are heights above the ground, and the ground is the
 * reference manager's (`terrainHeight`, moved on a hot reload). The targets used to be written at an absolute
 * defaultBaseHeight, so the base reference ignored the ground at launch and snapped back to it one solve after a reload,
 * while the reference manager lifted the initial target - a measured, absolute pose - by the whole terrain height.
 *
 * Every property is a difference against the same run on flat ground, or against the height the target was published
 * at, so none of them depends on the tuned heights. The last test covers the contact planner's reference manager, which
 * keeps a copy of the published target of its own.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kHorizon = 1.0;
constexpr scalar_t kCyclePeriod = 0.01;
// Cycles run on each ground.
constexpr int kSettlingCycles = 10;
// [m] how far below defaultBaseHeight the robot's measured base starts.
constexpr scalar_t kCrouch = 0.05;

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
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

/** The shipped G1 whole-body task file on a ground at `terrainHeight`. */
std::string writeTaskFileOnGround(scalar_t terrainHeight) {
  const std::string shipped = readFile(runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml"));
  EXPECT_EQ(shipped.find("\nterrainHeight:"), std::string::npos) << "the shipped file sets the ground; this test appends it";
  const std::string path =
      (std::filesystem::path(testing::TempDir()) / absl::StrCat("testBaseHeightFollowsTerrain_", terrainHeight, ".yaml")).string();
  std::ofstream out(path);
  out << shipped << "\nterrainHeight: " << terrainHeight << "\n";
  return path;
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
  kTaskFile,          // unwired: the task file's `terrainHeight`, all a command node outside the MPC knows
};

/** The whole-body MPC's references, wired as the MPC node wires them, on the task file's ground. */
class WholeBodyReferences {
 public:
  explicit WholeBodyReferences(scalar_t terrainHeight, CalculatorGround calculatorGround = CalculatorGround::kReferenceManager) {
    const std::string taskFile = writeTaskFileOnGround(terrainHeight);
    const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    const std::string referenceFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml");
    const std::string gaitFile = runfilePath("humanoid_nmpc/humanoid_common_mpc/config/command/gait.yaml");
    EXPECT_FALSE(urdfFile.empty() || referenceFile.empty() || gaitFile.empty()) << "the G1 whole-body files are not in the runfiles";
    modelSettings_ = std::make_unique<ModelSettings>(taskFile, urdfFile, "wb_mpc_", /*verbose=*/false);
    absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(taskFile, urdfFile, *modelSettings_);
    EXPECT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
    referenceManager_ = std::make_shared<SwitchedModelReferenceManager>(
        GaitSchedule::loadGaitSchedule(referenceFile, *modelSettings_, /*verbose=*/false),
        std::make_unique<SwingTrajectoryPlanner>(loadSwingTrajectorySettings(taskFile, "swing_trajectory_config", /*verbose=*/false),
                                                 N_CONTACTS),
        *pinocchioInterface_, *model_);
    calculator_ = std::make_unique<WBMpcTargetTrajectoriesCalculator>(referenceFile, *model_, kHorizon);
    if (calculatorGround == CalculatorGround::kReferenceManager) {
      // What the MPC node wires: the calculator builds on the ground the reference manager applied in this solve.
      calculator_->setTerrainHeightSource([referenceManager = referenceManager_]() { return referenceManager->getAppliedTerrainHeight(); });
    }
    motionManager_ = std::make_unique<ProceduralMpcMotionManager>(
        gaitFile, referenceFile, referenceManager_, *model_,
        [calculator = calculator_.get()](const vector4_t& velocityTarget, scalar_t initTime, scalar_t /*finalTime*/,
                                         const vector_t& initState) {
          return calculator->commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState);
        });
    initialState_ = vector_t::Zero(model_->getStateDim());
    loadData::loadEigenMatrix(taskFile, "initialState", initialState_);
    // The robot stands on this ground, a little crouched, so that the measured base is not the commanded one.
    model_->adaptBasePoseHeight(initialState_, terrainHeight - kCrouch);
    // No pelvis height in the command: every target stands defaultBaseHeight above the ground. (The motion manager's
    // command filter ramps against the wall clock, so a commanded height would differ between two runs.)
    motionManager_->setAndScaleVelocityCommand(WalkingVelocityCommand(/*v_x=*/0.0, /*v_y=*/0.0, /*desired_pelvis_h=*/0.0, /*v_yaw=*/0.0));
    // What the MPC node does on its reset: a target at the measured initial state, a world pose already.
    referenceManager_->setTargetTrajectories(TargetTrajectories({0.0}, {initialState_}, {vector_t::Zero(model_->getInputDim())}));
  }

  /** One solver cycle: the reference manager's pre-solve hook, then the motion manager's, as the solver orders them. */
  void cycle() {
    const scalar_t time = kCyclePeriod * static_cast<scalar_t>(numCycles_++);
    referenceManager_->preSolverRun(time, time + kHorizon, initialState_, ModeNumber::STANCE);
    motionManager_->preSolverRun(time, time + kHorizon, initialState_, *referenceManager_);
  }

  /** [m] the base height of every knot of the target the solver tracks in this cycle. */
  std::vector<scalar_t> targetBaseHeights() const { return ocs2::humanoid::targetBaseHeights(*referenceManager_, *model_); }

  SwitchedModelReferenceManager& referenceManager() { return *referenceManager_; }
  WBMpcTargetTrajectoriesCalculator& calculator() { return *calculator_; }
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
};

void expectRaisedBy(const std::vector<scalar_t>& raised, const std::vector<scalar_t>& flat, scalar_t height, absl::string_view when) {
  ASSERT_EQ(raised.size(), flat.size()) << when;
  ASSERT_FALSE(raised.empty()) << when;
  for (size_t knot = 0; knot < raised.size(); ++knot) {
    EXPECT_NEAR(raised[knot] - flat[knot], height, 1e-12) << when << ", knot " << knot;
  }
}

}  // namespace

TEST(BaseHeightFollowsTerrainTest, aTerrainHeightRaisesTheWholeBaseReferenceByItAndAReloadMovesItOnce) {
  constexpr scalar_t kGround = 0.3;
  constexpr scalar_t kReloadedGround = -0.12;
  WholeBodyReferences flat(0.0);
  WholeBodyReferences raised(kGround);
  // Unwired, a calculator builds on the task file's ground - all a command node outside the MPC can know.
  EXPECT_EQ(WBMpcTargetTrajectoriesCalculator(runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml"),
                                              raised.model(), kHorizon)
                .getTerrainHeight(),
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
  EXPECT_NEAR(flat.targetBaseHeights().back() - flat.model().getBasePose(flat.initialState())(2), kCrouch, 1e-12);

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
  const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml");
  const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
  const std::string referenceFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml");
  ASSERT_FALSE(taskFile.empty() || urdfFile.empty() || referenceFile.empty()) << "the G1 whole-body files are not in the runfiles";
  const ModelSettings modelSettings(taskFile, urdfFile, "wb_mpc_", /*verbose=*/false);
  const absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(taskFile, urdfFile, modelSettings);
  ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
  const WBAccelMpcRobotModel<scalar_t> model(modelSettings);
  ContactPlanningConfig config;
  config.shared.comHeight = 0.85;  // no model derives the pendulum here (the library default 0 means "from the model")
  config.formulation.setExecutionRule(term::kPlannedComOverride, /*on=*/true);
  ASSERT_EQ(config.validateStatus(), absl::OkStatus());
  const absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>> created = ContactPlanningReferenceManager::Create(
      GaitSchedule::loadGaitSchedule(referenceFile, modelSettings, /*verbose=*/false),
      std::make_shared<SwingTrajectoryPlanner>(loadSwingTrajectorySettings(taskFile, "swing_trajectory_config", /*verbose=*/false),
                                               N_CONTACTS),
      *pinocchioInterface, model, config);
  ASSERT_TRUE(created.ok()) << created.status();
  ContactPlanningReferenceManager& referenceManager = **created;

  vector_t state = vector_t::Zero(model.getStateDim());
  loadData::loadEigenMatrix(taskFile, "initialState", state);
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
    referenceManager.preSolverRun(time, time + kHorizon, state, ModeNumber::STANCE);
    // Positive control: the plan is in use and its target is the densified one - a knot per plan node, then the
    // operator's own knot past the plan - so the tail below is read from the operator's copy.
    const TargetTrajectories& target = referenceManager.getTargetTrajectories();
    ASSERT_TRUE(referenceManager.planReferencesUsableAt(time));
    ASSERT_EQ(target.timeTrajectory.size(), plan.comPosition.size() + 1);
    ASSERT_EQ(target.timeTrajectory.back(), kPublishedUntil);
    for (const scalar_t height : targetBaseHeights(referenceManager, model)) {
      EXPECT_NEAR(height, publishedHeight, 1e-12) << "launch, cycle " << i;
    }
  }

  // The reload moves the whole reference, the tail from the operator's copy included, and it stays moved.
  referenceManager.setTerrainHeight(kReloadedGround);
  for (int i = 0; i < kSettlingCycles; ++i, ++cycle) {
    const scalar_t time = kCyclePeriod * static_cast<scalar_t>(cycle);
    referenceManager.preSolverRun(time, time + kHorizon, state, ModeNumber::STANCE);
    ASSERT_TRUE(referenceManager.planReferencesUsableAt(time));
    ASSERT_EQ(referenceManager.getTargetTrajectories().timeTrajectory.back(), kPublishedUntil);
    const std::vector<scalar_t> heights = targetBaseHeights(referenceManager, model);
    for (size_t knot = 0; knot < heights.size(); ++knot) {
      EXPECT_NEAR(heights[knot], publishedHeight + kReloadedGround, 1e-12) << "after the reload, cycle " << i << ", knot " << knot;
    }
  }
}

}  // namespace ocs2::humanoid
