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

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_core/misc/LoadData.h>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid {

namespace {

/** A reference manager with a planned DCM of its own choosing, so that the cost can be shown what it does under a plan. */
class PlannedDcmReferenceManager final : public SwitchedModelReferenceManager {
 public:
  using SwitchedModelReferenceManager::SwitchedModelReferenceManager;
  std::optional<PlannedDcm> getPlannedDcm(scalar_t /*time*/) const override { return planned; }
  std::optional<PlannedDcm> planned;
};

}  // namespace

class DcmTerminalCostTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml").value();
    referenceFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml").value();
    urdfFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf").value();

    modelSettings_ = std::make_unique<ModelSettings>(taskFile_, urdfFile_, "testDcmTerminalCost_", /*verbose=*/false);
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(
        createCustomPinocchioInterface(taskFile_, urdfFile_, *modelSettings_, /*scaleTotalMass=*/false));
    info_ = centroidal_model::createCentroidalModelInfo(
        *pinocchioInterface_, centroidal_model::loadCentroidalType(taskFile_),
        centroidal_model::loadDefaultJointState(pinocchioInterface_->getModel().nq - 6, referenceFile_), modelSettings_->contactNames3DoF,
        modelSettings_->contactNames6DoF);
    robotModel_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, info_);
    robotModelAd_ =
        std::make_unique<CentroidalMpcRobotModel<ad_scalar_t>>(*modelSettings_, pinocchioInterface_->toCppAd(), info_.toCppAd());

    std::unique_ptr<SwingTrajectoryPlanner> swingPlanner(
        new SwingTrajectoryPlanner(loadSwingTrajectorySettings(taskFile_, "swing_trajectory_config", /*verbose=*/false), N_CONTACTS));
    std::shared_ptr<GaitSchedule> gaitSchedule = GaitSchedule::loadGaitSchedule(referenceFile_, *modelSettings_, /*verbose=*/false);
    referenceManager_ = std::make_shared<SwitchedModelReferenceManager>(std::move(gaitSchedule), std::move(swingPlanner),
                                                                        *pinocchioInterface_, *robotModel_);
    // Right-foot single support on [1.0, 1.5) (and [2.0, 2.5)): the reference manager rebuilds its schedule from the gait
    // schedule on every preSolverRun, so the phase has to be inserted there.
    referenceManager_->getGaitSchedule()->insertModeSequenceTemplate(
        ModeSequenceTemplate({0.0, 0.5, 1.0}, {ModeNumber::RF, ModeNumber::STANCE}), /*startTime=*/1.0, /*finalTime=*/3.0);
    initialState_.setZero(info_.stateDim);
    loadData::loadEigenMatrix(taskFile_, "initialState", initialState_);
    // The reference manager needs one preSolverRun to swap in the buffered mode schedule.
    referenceManager_->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, initialState_, ModeNumber::STANCE);

    // An explicit height: the tests below are about the cost's arithmetic, on a pendulum they choose.
    config_.comHeight = 0.85;
    config_.weights = vector2_t(400.0, 100.0);
    config_.velocityOffsetFactor = 1.0;
    config_.supportBlendTime = 0.1;
    modelSettings_->recompileLibrariesCppAd = false;
    modelComHeight_ = independentComHeight(initialState_);
    cost_ = createCost(*referenceManager_, config_);
    ASSERT_NE(cost_, nullptr);
  }

  std::unique_ptr<DcmTerminalCost> createCost(const SwitchedModelReferenceManager& referenceManager,
                                              const DcmTerminalCost::Config& config) {
    absl::StatusOr<std::unique_ptr<DcmTerminalCost>> created = DcmTerminalCost::Create(
        referenceManager, config, modelComHeight_, *pinocchioInterface_, *robotModelAd_, "testDcmTerminalCost", *modelSettings_);
    EXPECT_TRUE(created.ok()) << created.status();
    return created.ok() ? *std::move(created) : nullptr;
  }

  /**
   * The model's center of mass above the mean height of its contact frames at `state`, computed here from Pinocchio
   * directly - forward kinematics and the frames looked up by name - rather than through computeComHeightAboveFeet().
   */
  scalar_t independentComHeight(const vector_t& state) const {
    PinocchioInterface pinocchio(*pinocchioInterface_);
    const pinocchio::ModelTpl<scalar_t>& model = pinocchio.getModel();
    pinocchio::DataTpl<scalar_t>& data = pinocchio.getData();
    const vector_t q = robotModel_->getGeneralizedCoordinates(state);
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    const scalar_t comZ = pinocchio::centerOfMass(model, data, q)(2);
    scalar_t footZ = 0.0;
    for (const std::string& frame : modelSettings_->contactNames) {
      footZ += data.oMf[model.getFrameId(frame)].translation()(2) / static_cast<scalar_t>(modelSettings_->contactNames.size());
    }
    return comZ - footZ;
  }

  vector2_t supportCenter(const vector_t& state, const contact_flag_t& contacts) const {
    const vector_t q = robotModel_->getGeneralizedCoordinates(state);
    PinocchioInterface pinocchio(*pinocchioInterface_);
    const std::vector<vector3_t> feet = computeContactPositions<scalar_t>(q, pinocchio, *robotModel_);
    vector2_t center = vector2_t::Zero();
    int count = 0;
    for (size_t i = 0; i < N_CONTACTS; ++i) {
      if (contacts[i]) {
        center += feet[i].head<2>();
        ++count;
      }
    }
    return center / static_cast<scalar_t>(count);
  }

  vector2_t comXy(const vector_t& state) const {
    PinocchioInterface pinocchio(*pinocchioInterface_);
    const vector_t q = robotModel_->getGeneralizedCoordinates(state);
    pinocchio::centerOfMass(pinocchio.getModel(), pinocchio.getData(), q, /*computeSubtreeComs=*/false);
    return pinocchio.getData().com[0].head<2>();
  }

  std::string taskFile_, referenceFile_, urdfFile_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  CentroidalModelInfo info_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> robotModel_;
  std::unique_ptr<CentroidalMpcRobotModel<ad_scalar_t>> robotModelAd_;
  std::shared_ptr<SwitchedModelReferenceManager> referenceManager_;
  vector_t initialState_;
  scalar_t modelComHeight_ = 0.0;
  DcmTerminalCost::Config config_;
  std::unique_ptr<DcmTerminalCost> cost_;
};

TEST_F(DcmTerminalCostTest, DcmErrorMatchesAnalyticDefinition) {
  const TargetTrajectories emptyTargets;
  // Double support at t = 0.5: support center is the mean of both feet.
  vector_t state = initialState_;
  state(0) = 0.3;   // v_com_x
  state(1) = -0.1;  // v_com_y
  const vector_t params = cost_->getParameters(/*time=*/0.5, emptyTargets);
  EXPECT_DOUBLE_EQ(params(0), 1.0);
  EXPECT_DOUBLE_EQ(params(1), 1.0);
  const scalar_t omega = std::sqrt(9.81 / 0.85);
  EXPECT_NEAR(params(2), omega, 1e-12);
  const vector2_t expected = comXy(state) + state.head<2>() / omega - supportCenter(state, {true, true});
  const vector2_t error = cost_->computeDcmError(state, params);
  EXPECT_LT((error - expected).norm(), 1e-6) << "error " << error.transpose() << " expected " << expected.transpose();

  // Value consistent with the weights.
  const scalar_t value = cost_->getValue(/*time=*/0.5, state, emptyTargets, PreComputation());
  EXPECT_NEAR(value, 0.5 * (400.0 * expected(0) * expected(0) + 100.0 * expected(1) * expected(1)), 1e-6);
}

TEST_F(DcmTerminalCostTest, SingleSupportUsesStanceFoot) {
  const TargetTrajectories emptyTargets;
  const vector_t& state = initialState_;
  // Right foot single support on [1.0, 1.5).
  ASSERT_EQ(referenceManager_->getModeSchedule().modeAtTime(1.25), static_cast<size_t>(ModeNumber::RF));
  const vector_t paramsRf = cost_->getParameters(/*time=*/1.25, emptyTargets);
  EXPECT_DOUBLE_EQ(paramsRf(0), 0.0);
  EXPECT_DOUBLE_EQ(paramsRf(1), 1.0);
  const vector2_t errorRf = cost_->computeDcmError(state, paramsRf);
  const vector2_t expectedRf = comXy(state) - supportCenter(state, {false, true});
  EXPECT_LT((errorRf - expectedRf).norm(), 1e-6);
  // Versus double support: the two references differ by half the foot separation (laterally ~0.1 m for Atlas).
  const vector2_t errorDs = cost_->computeDcmError(state, cost_->getParameters(/*time=*/0.5, emptyTargets));
  EXPECT_GT(std::abs(errorRf(1) - errorDs(1)), 0.05);
}

TEST_F(DcmTerminalCostTest, SupportWeightsBlendThroughTransitions) {
  // Right-foot single support on [1.0, 1.5): the left foot's weight ramps out before 1.0 and back in after 1.5, the
  // right foot keeps weight 1 (its contact phase spans the whole interval).
  EXPECT_NEAR(cost_->computeSupportWeights(0.5)(0), 1.0, 1e-12);
  EXPECT_NEAR(cost_->computeSupportWeights(0.95)(0), 0.5, 1e-9);
  EXPECT_NEAR(cost_->computeSupportWeights(1.25)(0), 0.0, 1e-12);
  EXPECT_NEAR(cost_->computeSupportWeights(1.55)(0), 0.5, 1e-9);
  EXPECT_NEAR(cost_->computeSupportWeights(1.7)(0), 1.0, 1e-12);
  for (const scalar_t t : {0.5, 0.95, 1.25, 1.55, 1.7}) {
    EXPECT_NEAR(cost_->computeSupportWeights(t)(1), 1.0, 1e-12) << "t=" << t;
  }
  // The reference is continuous across the transition: parameters at 0.999 and 1.001 are close.
  const TargetTrajectories emptyTargets;
  const vector2_t before = cost_->computeDcmError(initialState_, cost_->getParameters(/*time=*/0.999, emptyTargets));
  const vector2_t after = cost_->computeDcmError(initialState_, cost_->getParameters(/*time=*/1.001, emptyTargets));
  EXPECT_LT((before - after).norm(), 5e-3);
  // Blending off reproduces the hard switch.
  DcmTerminalCost::Config hard = config_;
  hard.supportBlendTime = 0.0;
  ASSERT_EQ(cost_->setConfig(hard), absl::OkStatus());
  EXPECT_NEAR(cost_->computeSupportWeights(0.95)(0), 1.0, 1e-12);
  EXPECT_NEAR(cost_->computeSupportWeights(1.05)(0), 0.0, 1e-12);
  ASSERT_EQ(cost_->setConfig(config_), absl::OkStatus());
}

TEST_F(DcmTerminalCostTest, VelocityCommandShiftsReference) {
  const scalar_t omega = config_.omega();
  vector_t targetState = vector_t::Zero(info_.stateDim);
  targetState(0) = 0.4;
  const TargetTrajectories targets({0.0}, {targetState}, {vector_t::Zero(info_.inputDim)});
  const vector2_t withCommand = cost_->computeDcmError(initialState_, cost_->getParameters(/*time=*/0.5, targets));
  const vector2_t without = cost_->computeDcmError(initialState_, cost_->getParameters(/*time=*/0.5, TargetTrajectories()));
  EXPECT_NEAR(withCommand(0) - without(0), -0.4 / omega, 1e-9);
  EXPECT_NEAR(withCommand(1) - without(1), 0.0, 1e-9);

  DcmTerminalCost::Config noOffset = config_;
  noOffset.velocityOffsetFactor = 0.0;
  ASSERT_EQ(cost_->setConfig(noOffset), absl::OkStatus());
  const vector2_t pureCapturability = cost_->computeDcmError(initialState_, cost_->getParameters(/*time=*/0.5, targets));
  EXPECT_LT((pureCapturability - without).norm(), 1e-9);
}

TEST_F(DcmTerminalCostTest, GaussNewtonApproximationMatchesFiniteDifferences) {
  vector_t state = initialState_;
  state(0) = 0.25;
  state(1) = -0.15;
  state(9) = 0.2;   // base yaw
  state(10) = 0.1;  // base pitch
  state(13) = 0.3;  // a torso joint
  const TargetTrajectories emptyTargets;
  const ScalarFunctionQuadraticApproximation approximation =
      cost_->getQuadraticApproximation(/*time=*/0.5, state, emptyTargets, PreComputation());
  ASSERT_EQ(approximation.dfdx.size(), state.size());
  ASSERT_EQ(approximation.dfdxx.rows(), state.size());
  EXPECT_NEAR(approximation.f, cost_->getValue(/*time=*/0.5, state, emptyTargets, PreComputation()), 1e-9);

  const scalar_t h = 1e-6;
  for (Eigen::Index i = 0; i < state.size(); ++i) {
    vector_t plus = state, minus = state;
    plus(i) += h;
    minus(i) -= h;
    const scalar_t fd = (cost_->getValue(/*time=*/0.5, plus, emptyTargets, PreComputation()) -
                         cost_->getValue(/*time=*/0.5, minus, emptyTargets, PreComputation())) /
                        (2.0 * h);
    EXPECT_NEAR(approximation.dfdx(i), fd, 1e-4 * std::max(1.0, std::abs(fd))) << "gradient entry " << i;
  }
  const Eigen::SelfAdjointEigenSolver<matrix_t> eigen(approximation.dfdxx);
  EXPECT_GE(eigen.eigenvalues().minCoeff(), -1e-9);
}

TEST_F(DcmTerminalCostTest, AComHeightOfZeroIsTheModelsPendulumAndAPositiveOneIsAnOverride) {
  // The model's pendulum, computed in this test independently of computeComHeightAboveFeet(); a humanoid's center of
  // mass stands somewhere between half a meter and a meter and a half above its soles.
  ASSERT_GT(modelComHeight_, 0.5);
  ASSERT_LT(modelComHeight_, 1.5);
  DcmTerminalCost::Config derived = config_;
  derived.comHeight = 0.0;
  absl::StatusOr<std::unique_ptr<DcmTerminalCost>> created = DcmTerminalCost::Create(
      *referenceManager_, derived, modelComHeight_, *pinocchioInterface_, *robotModelAd_, "testDcmTerminalCost", *modelSettings_);
  ASSERT_TRUE(created.ok()) << created.status();
  const std::unique_ptr<DcmTerminalCost> cost = *std::move(created);
  EXPECT_NEAR(cost->getConfig().comHeight, modelComHeight_, 1e-12) << "0 means the model's center of mass above its feet";
  EXPECT_NEAR(cost->getParameters(/*time=*/0.5, TargetTrajectories())(2), std::sqrt(9.81 / modelComHeight_), 1e-12);

  // Positive control: an explicit height is used as given, and the model's is still what a later 0 means - a hot reload
  // (MpcParameterUpdaterModule) hands the block to setConfig() exactly as the file writes it.
  DcmTerminalCost::Config explicitHeight = derived;
  explicitHeight.comHeight = 0.9;
  ASSERT_EQ(cost->setConfig(explicitHeight), absl::OkStatus());
  EXPECT_NEAR(cost->getParameters(/*time=*/0.5, TargetTrajectories())(2), std::sqrt(9.81 / 0.9), 1e-12);
  ASSERT_GT(std::abs(0.9 - modelComHeight_), 0.05) << "the override must differ from the model, or the checks above prove nothing";
  ASSERT_EQ(cost->setConfig(derived), absl::OkStatus());
  EXPECT_NEAR(cost->getConfig().comHeight, modelComHeight_, 1e-12);
  EXPECT_NEAR(cost->getModelComHeight(), modelComHeight_, 1e-12);
}

TEST_F(DcmTerminalCostTest, ARejectedConfigurationNamesItsKeyAndKeepsTheRunningOne) {
  DcmTerminalCost::Config negativeWeight = config_;
  negativeWeight.weights(1) = -1.0;
  const absl::Status weight = cost_->setConfig(negativeWeight);
  EXPECT_EQ(weight.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(weight.message(), "dcm_terminal_cost.weight_y")) << weight;
  EXPECT_DOUBLE_EQ(cost_->getConfig().weights(1), config_.weights(1)) << "a refused configuration must leave the running one in force";

  DcmTerminalCost::Config negativeHeight = config_;
  negativeHeight.comHeight = -0.9;
  const absl::Status height = cost_->setConfig(negativeHeight);
  EXPECT_EQ(height.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(height.message(), "dcm_terminal_cost.comHeight")) << height;
  EXPECT_DOUBLE_EQ(cost_->getConfig().comHeight, config_.comHeight);

  // A model that gives no pendulum cannot stand in for a comHeight of 0.
  DcmTerminalCost::Config derived = config_;
  derived.comHeight = 0.0;
  const absl::StatusOr<DcmTerminalCost::Config> noModel = DcmTerminalCost::resolveConfig(derived, /*modelComHeight=*/0.0);
  EXPECT_EQ(noModel.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(noModel.status().message(), "dcm_terminal_cost.comHeight")) << noModel.status();
  EXPECT_TRUE(DcmTerminalCost::resolveConfig(derived, /*modelComHeight=*/1.0).ok()) << "the control: the same configuration with a model";
}

TEST_F(DcmTerminalCostTest, TheLoaderReadsTheBlockAsWrittenAndNamesAKeyThatIsNotANumber) {
  const absl::StatusOr<DcmTerminalCost::Config> shipped = DcmTerminalCost::loadConfig(taskFile_);
  ASSERT_TRUE(shipped.ok()) << shipped.status();
  EXPECT_GT(shipped->weights(0), 0.0);
  EXPECT_GE(shipped->comHeight, 0.0) << "0 (the model's) or an explicit height";

  const std::string file = absl::StrCat(testing::TempDir(), "/dcm_terminal_cost_not_a_number.yaml");
  {
    std::ofstream out(file);
    out << "dcm_terminal_cost:\n  comHeight: tall\n  weight_x: 10\n";
  }
  const absl::StatusOr<DcmTerminalCost::Config> notANumber = DcmTerminalCost::loadConfig(file);
  EXPECT_EQ(notANumber.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(notANumber.status().message(), "dcm_terminal_cost.comHeight")) << notANumber.status();
  {
    std::ofstream out(file);
    out << "dcm_terminal_cost:\n  comHeight: 0\n  gravity: -9.81\n";
  }
  const absl::StatusOr<DcmTerminalCost::Config> negativeGravity = DcmTerminalCost::loadConfig(file);
  EXPECT_EQ(negativeGravity.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(negativeGravity.status().message(), "dcm_terminal_cost.gravity")) << negativeGravity.status();
  {
    std::ofstream out(file);
    out << "dcm_terminal_cost:\n  comHeight: 0\n";
  }
  const absl::StatusOr<DcmTerminalCost::Config> derived = DcmTerminalCost::loadConfig(file);
  ASSERT_TRUE(derived.ok()) << derived.status();
  EXPECT_EQ(derived->comHeight, 0.0) << "the loader leaves 0 for the cost to resolve against the model";
  std::remove(file.c_str());
}

TEST_F(DcmTerminalCostTest, UnderAPlanTheRobotsDcmIsTakenOnThePlansPendulum) {
  std::shared_ptr<PlannedDcmReferenceManager> planning = std::make_shared<PlannedDcmReferenceManager>(
      GaitSchedule::loadGaitSchedule(referenceFile_, *modelSettings_, /*verbose=*/false),
      std::make_shared<SwingTrajectoryPlanner>(loadSwingTrajectorySettings(taskFile_, "swing_trajectory_config", /*verbose=*/false),
                                               N_CONTACTS),
      *pinocchioInterface_, *robotModel_);
  planning->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, initialState_, ModeNumber::STANCE);
  const std::unique_ptr<DcmTerminalCost> cost = createCost(*planning, config_);
  ASSERT_NE(cost, nullptr);

  // Control: without a plan the cost measures the DCM on its own pendulum.
  const scalar_t ownOmega = std::sqrt(9.81 / config_.comHeight);
  vector_t parameters = cost->getParameters(/*time=*/0.5, TargetTrajectories());
  EXPECT_NEAR(parameters(2), ownOmega, 1e-12);
  EXPECT_EQ(parameters(8), 0.0);

  // A plan made on another pendulum: its DCM is only its DCM with its own omega, so the robot's is taken with that one
  // too, and the residual compares two points of one pendulum.
  SwitchedModelReferenceManager::PlannedDcm plan;
  plan.dcm = vector2_t(0.12, -0.04);
  plan.omega = 2.5;
  ASSERT_GT(std::abs(plan.omega - ownOmega), 0.5);
  planning->planned = plan;
  parameters = cost->getParameters(/*time=*/0.5, TargetTrajectories());
  EXPECT_NEAR(parameters(2), plan.omega, 1e-12) << "the robot's DCM must be taken on the plan's pendulum, not the cost's";
  EXPECT_EQ(parameters(8), 1.0);
  EXPECT_NEAR(parameters(9), plan.dcm(0), 1e-12);
  EXPECT_NEAR(parameters(10), plan.dcm(1), 1e-12);
  vector_t state = initialState_;
  state(0) = 0.3;
  state(1) = -0.2;
  const vector2_t expected = comXy(state) + state.head<2>() / plan.omega - plan.dcm;
  EXPECT_LT((cost->computeDcmError(state, parameters) - expected).norm(), 1e-6);
}

}  // namespace ocs2::humanoid
