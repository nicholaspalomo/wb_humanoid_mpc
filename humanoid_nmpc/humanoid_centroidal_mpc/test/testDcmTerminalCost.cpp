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
#include <fstream>
#include <memory>
#include <string>

#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_core/misc/LoadData.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

namespace ocs2::humanoid {

class DcmTerminalCostTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string configDir = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
    const std::string descriptionDir = ament_index_cpp::get_package_share_directory("drc_atlas_description");
    taskFile_ = configDir + "/config/mpc/task.yaml";
    referenceFile_ = configDir + "/config/command/reference.yaml";
    urdfFile_ = descriptionDir + "/urdf/atlas.urdf";

    modelSettings_ = std::make_unique<ModelSettings>(taskFile_, urdfFile_, "testDcmTerminalCost_", false);
    pinocchioInterface_ =
        std::make_unique<PinocchioInterface>(createCustomPinocchioInterface(taskFile_, urdfFile_, *modelSettings_, false));
    info_ = centroidal_model::createCentroidalModelInfo(
        *pinocchioInterface_, centroidal_model::loadCentroidalType(taskFile_),
        centroidal_model::loadDefaultJointState(pinocchioInterface_->getModel().nq - 6, referenceFile_), modelSettings_->contactNames3DoF,
        modelSettings_->contactNames6DoF);
    robotModel_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, info_);
    robotModelAd_ =
        std::make_unique<CentroidalMpcRobotModel<ad_scalar_t>>(*modelSettings_, pinocchioInterface_->toCppAd(), info_.toCppAd());

    std::unique_ptr<SwingTrajectoryPlanner> swingPlanner(
        new SwingTrajectoryPlanner(loadSwingTrajectorySettings(taskFile_, "swing_trajectory_config", false), N_CONTACTS));
    auto gaitSchedule = GaitSchedule::loadGaitSchedule(referenceFile_, *modelSettings_, false);
    referenceManager_ = std::make_shared<SwitchedModelReferenceManager>(std::move(gaitSchedule), std::move(swingPlanner),
                                                                        *pinocchioInterface_, *robotModel_);
    // Right-foot single support on [1.0, 1.5) (and [2.0, 2.5)): the reference manager rebuilds its schedule from the gait
    // schedule on every preSolverRun, so the phase has to be inserted there.
    referenceManager_->getGaitSchedule()->insertModeSequenceTemplate(
        ModeSequenceTemplate({0.0, 0.5, 1.0}, {ModeNumber::RF, ModeNumber::STANCE}), 1.0, 3.0);
    initialState_.setZero(info_.stateDim);
    loadData::loadEigenMatrix(taskFile_, "initialState", initialState_);
    // The reference manager needs one preSolverRun to swap in the buffered mode schedule.
    referenceManager_->preSolverRun(0.0, 1.0, initialState_, ModeNumber::STANCE);

    config_.comHeight = 0.85;
    config_.weights = vector2_t(400.0, 100.0);
    config_.velocityOffsetFactor = 1.0;
    config_.supportBlendTime = 0.1;
    modelSettings_->recompileLibrariesCppAd = false;
    cost_ = std::make_unique<DcmTerminalCost>(*referenceManager_, config_, *pinocchioInterface_, *robotModelAd_, "testDcmTerminalCost",
                                              *modelSettings_);
  }

  vector2_t supportCentre(const vector_t& state, const contact_flag_t& contacts) const {
    const vector_t q = robotModel_->getGeneralizedCoordinates(state);
    PinocchioInterface pinocchio(*pinocchioInterface_);
    const std::vector<vector3_t> feet = computeContactPositions<scalar_t>(q, pinocchio, *robotModel_);
    vector2_t centre = vector2_t::Zero();
    int count = 0;
    for (size_t i = 0; i < N_CONTACTS; ++i) {
      if (contacts[i]) {
        centre += feet[i].head<2>();
        ++count;
      }
    }
    return centre / static_cast<scalar_t>(count);
  }

  vector2_t comXy(const vector_t& state) const {
    PinocchioInterface pinocchio(*pinocchioInterface_);
    const vector_t q = robotModel_->getGeneralizedCoordinates(state);
    pinocchio::centerOfMass(pinocchio.getModel(), pinocchio.getData(), q, false);
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
  DcmTerminalCost::Config config_;
  std::unique_ptr<DcmTerminalCost> cost_;
};

TEST_F(DcmTerminalCostTest, DcmErrorMatchesAnalyticDefinition) {
  const TargetTrajectories emptyTargets;
  // Double support at t = 0.5: support centre is the mean of both feet.
  vector_t state = initialState_;
  state(0) = 0.3;   // v_com_x
  state(1) = -0.1;  // v_com_y
  const vector_t params = cost_->getParameters(0.5, emptyTargets);
  EXPECT_DOUBLE_EQ(params(0), 1.0);
  EXPECT_DOUBLE_EQ(params(1), 1.0);
  const scalar_t omega = std::sqrt(9.81 / 0.85);
  EXPECT_NEAR(params(2), omega, 1e-12);
  const vector2_t expected = comXy(state) + state.head<2>() / omega - supportCentre(state, {true, true});
  const vector2_t error = cost_->computeDcmError(state, params);
  EXPECT_LT((error - expected).norm(), 1e-6) << "error " << error.transpose() << " expected " << expected.transpose();

  // Value consistent with the weights.
  const scalar_t value = cost_->getValue(0.5, state, emptyTargets, PreComputation());
  EXPECT_NEAR(value, 0.5 * (400.0 * expected(0) * expected(0) + 100.0 * expected(1) * expected(1)), 1e-6);
}

TEST_F(DcmTerminalCostTest, SingleSupportUsesStanceFoot) {
  const TargetTrajectories emptyTargets;
  const vector_t& state = initialState_;
  // Right foot single support on [1.0, 1.5).
  ASSERT_EQ(referenceManager_->getModeSchedule().modeAtTime(1.25), static_cast<size_t>(ModeNumber::RF));
  const vector_t paramsRf = cost_->getParameters(1.25, emptyTargets);
  EXPECT_DOUBLE_EQ(paramsRf(0), 0.0);
  EXPECT_DOUBLE_EQ(paramsRf(1), 1.0);
  const vector2_t errorRf = cost_->computeDcmError(state, paramsRf);
  const vector2_t expectedRf = comXy(state) - supportCentre(state, {false, true});
  EXPECT_LT((errorRf - expectedRf).norm(), 1e-6);
  // Versus double support: the two references differ by half the foot separation (laterally ~0.1 m for Atlas).
  const vector2_t errorDs = cost_->computeDcmError(state, cost_->getParameters(0.5, emptyTargets));
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
  const vector2_t before = cost_->computeDcmError(initialState_, cost_->getParameters(0.999, emptyTargets));
  const vector2_t after = cost_->computeDcmError(initialState_, cost_->getParameters(1.001, emptyTargets));
  EXPECT_LT((before - after).norm(), 5e-3);
  // Blending off reproduces the hard switch.
  DcmTerminalCost::Config hard = config_;
  hard.supportBlendTime = 0.0;
  cost_->setConfig(hard);
  EXPECT_NEAR(cost_->computeSupportWeights(0.95)(0), 1.0, 1e-12);
  EXPECT_NEAR(cost_->computeSupportWeights(1.05)(0), 0.0, 1e-12);
  cost_->setConfig(config_);
}

TEST_F(DcmTerminalCostTest, VelocityCommandShiftsReference) {
  const scalar_t omega = config_.omega();
  vector_t targetState = vector_t::Zero(info_.stateDim);
  targetState(0) = 0.4;
  const TargetTrajectories targets({0.0}, {targetState}, {vector_t::Zero(info_.inputDim)});
  const vector2_t withCommand = cost_->computeDcmError(initialState_, cost_->getParameters(0.5, targets));
  const vector2_t without = cost_->computeDcmError(initialState_, cost_->getParameters(0.5, TargetTrajectories()));
  EXPECT_NEAR(withCommand(0) - without(0), -0.4 / omega, 1e-9);
  EXPECT_NEAR(withCommand(1) - without(1), 0.0, 1e-9);

  DcmTerminalCost::Config noOffset = config_;
  noOffset.velocityOffsetFactor = 0.0;
  cost_->setConfig(noOffset);
  const vector2_t pureCapturability = cost_->computeDcmError(initialState_, cost_->getParameters(0.5, targets));
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
  const auto approximation = cost_->getQuadraticApproximation(0.5, state, emptyTargets, PreComputation());
  ASSERT_EQ(approximation.dfdx.size(), state.size());
  ASSERT_EQ(approximation.dfdxx.rows(), state.size());
  EXPECT_NEAR(approximation.f, cost_->getValue(0.5, state, emptyTargets, PreComputation()), 1e-9);

  const scalar_t h = 1e-6;
  for (Eigen::Index i = 0; i < state.size(); ++i) {
    vector_t plus = state, minus = state;
    plus(i) += h;
    minus(i) -= h;
    const scalar_t fd =
        (cost_->getValue(0.5, plus, emptyTargets, PreComputation()) - cost_->getValue(0.5, minus, emptyTargets, PreComputation())) /
        (2.0 * h);
    EXPECT_NEAR(approximation.dfdx(i), fd, 1e-4 * std::max(1.0, std::abs(fd))) << "gradient entry " << i;
  }
  const Eigen::SelfAdjointEigenSolver<matrix_t> eigen(approximation.dfdxx);
  EXPECT_GE(eigen.eigenvalues().minCoeff(), -1e-9);
}

TEST_F(DcmTerminalCostTest, ConfigLoadsFromTaskFileAndValidates) {
  const DcmTerminalCost::Config loaded = DcmTerminalCost::loadConfig(taskFile_, "dcm_terminal_cost.", false);
  EXPECT_GT(loaded.weights(0), 0.0);
  EXPECT_GT(loaded.comHeight, 0.0);
  DcmTerminalCost::Config invalid = config_;
  invalid.weights(1) = -1.0;
  EXPECT_THROW(cost_->setConfig(invalid), std::invalid_argument);
  invalid = config_;
  invalid.comHeight = 0.0;
  EXPECT_THROW(invalid.validate(), std::invalid_argument);
}

}  // namespace ocs2::humanoid
