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

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_sqp/SqpMpc.h>
#include <ocs2_sqp/SqpSolver.h>

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"
#include "humanoid_centroidal_mpc/constraint/ZeroVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_centroidal_mpc/mrt/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/HumanoidPreComputation.h"
#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"
#include "humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h"
#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"
#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

#include <ocs2_core/penalties/penalties/QuadraticPenalty.h>
#include <ocs2_sqp/SqpSettings.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace ocs2::humanoid {

/**
 * Test fixture for MpcParameterUpdaterModule.
 *
 * Constructs a real CentroidalMpcInterface + SqpMpc from test robot config,
 * so all OCP cost/constraint names match the production configuration.
 */
class MpcParameterUpdaterModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Resolve config paths from the installed drc_atlas package
    const std::string configDir = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
    const std::string descriptionDir = ament_index_cpp::get_package_share_directory("drc_atlas_description");

    taskFile_ = configDir + "/config/mpc/task.yaml";
    referenceFile_ = configDir + "/config/command/reference.yaml";
    urdfFile_ = descriptionDir + "/urdf/atlas.urdf";

    // Create the interface
    auto status = CentroidalMpcInterface::Create(taskFile_, urdfFile_, referenceFile_);
    ASSERT_TRUE(status.ok()) << status.status().message();
    interface_ = *std::move(status);

    stateDim_ = interface_->getMpcRobotModel().getStateDim();
    // The updater writes gains into the OCP, so it must be sized to the input layout the solver optimizes over. With
    // basis-vector contact inputs that is the decorator's (basis-space) dimension, and the wrench-space R of task.yaml
    // has to be transformed the same way the OCP factory did it.
    inputDim_ = interface_->getEffectiveMpcRobotModel().getInputDim();
    basisCostTransform_ = interface_->getBasisInputsCostTransformConfig();
    contactNames_ = interface_->modelSettings().contactNames;

    // Create SqpMpc
    mpc_ = std::make_unique<SqpMpc>(interface_->mpcSettings(), interface_->sqpSettings(), interface_->getOptimalControlProblem(),
                                    interface_->getInitializer());
    mpc_->getSolverPtr()->setReferenceManager(interface_->getReferenceManagerPtr());

    // Write a temporary task.yaml copy for mutation tests
    tmpTaskFile_ = testing::TempDir() + "/test_task.yaml";
    std::ifstream src(taskFile_);
    std::ofstream dst(tmpTaskFile_);
    dst << src.rdbuf();
  }

  void TearDown() override { std::remove(tmpTaskFile_.c_str()); }

  /** Helper: get the SqpSolver pointer from the MPC */
  SqpSolver* getSqpSolver() { return dynamic_cast<SqpSolver*>(mpc_->getSolverPtr()); }

  /** Helper: touch the temp task file and drive the updater past its ~1 Hz file-check threshold. */
  void touchTaskFileAndRunUpdater(MpcParameterUpdaterModule& updater) {
    {
      std::ofstream touch(tmpTaskFile_, std::ios_base::app);
      touch << "\n# trigger update\n";
    }
    // Wait briefly for filesystem timestamp granularity
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const vector_t dummyState = vector_t::Zero(stateDim_);
    for (size_t i = 0; i < 101; ++i) {
      updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
    }
  }

  /**
   * Helper: a synthetic basis-space cost transform that is deliberately independent of the interface configuration, so
   * the test checks the updater's arithmetic rather than re-deriving the interface's own map.
   */
  BasisInputsCostTransformConfig makeSyntheticBasisCostTransform(size_t numBasisPerFoot, scalar_t lambdaRegularization) const {
    const size_t wrenchInputDim = interface_->getWrenchInputDim();
    const size_t numJoints = wrenchInputDim - 6 * N_CONTACTS;
    BasisInputsCostTransformConfig config;
    config.wrenchInputDim = wrenchInputDim;
    config.numBasisInputs = numBasisPerFoot * N_CONTACTS;
    config.lambdaRegularization = lambdaRegularization;
    config.basisToWrenchMap = matrix_t::Random(wrenchInputDim, config.numBasisInputs + numJoints);
    return config;
  }

  /** Helper: the wrench-space R exactly as written in the temp task file. */
  matrix_t loadWrenchSpaceR() const {
    matrix_t R_wrench = matrix_t::Zero(interface_->getWrenchInputDim(), interface_->getWrenchInputDim());
    loadData::loadEigenMatrix(tmpTaskFile_, "R", R_wrench);
    return R_wrench;
  }

  std::string taskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::string tmpTaskFile_;
  std::unique_ptr<CentroidalMpcInterface> interface_;
  std::unique_ptr<SqpMpc> mpc_;
  size_t stateDim_;
  size_t inputDim_;
  std::vector<std::string> contactNames_;
  std::optional<BasisInputsCostTransformConfig> basisCostTransform_;
};

/******************************************************************************************************/
// Test: Verify that construction succeeds and initial state is sane.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, ConstructionSucceeds) {
  EXPECT_NO_THROW({
    MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                      basisCostTransform_);
  });
}

/******************************************************************************************************/
// Test: Verify that quadratic cost weights (Q/R) are updated in-place after file change.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, QuadraticCostWeightsUpdatedInPlace) {
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);

  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // Read original Q(0,0) value
  matrix_t origQ, origR, origP;
  try {
    sqp->getOcpDefinitions().front().costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost").getGains(origQ, origR, origP);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "stateInputQuadraticCost not present in this configuration: " << e.what();
  }

  // Mutate task.yaml: multiply Q scaling by 2
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    // Find "Q:" section and modify scaling
    auto pos = content.find("Q:");
    ASSERT_NE(pos, std::string::npos) << "Could not find Q: section in task.yaml";
    auto scalingPos = content.find("scaling:", pos);
    ASSERT_NE(scalingPos, std::string::npos) << "Could not find Q scaling in task.yaml";

    // Extract original scaling value
    auto valueStart = scalingPos + std::string("scaling:").size();
    auto lineEnd = content.find('\n', valueStart);
    std::string origValue = content.substr(valueStart, lineEnd - valueStart);
    double origScaling = std::stod(origValue);

    // Replace with doubled value
    std::string newValue = " " + std::to_string(origScaling * 2.0);
    content.replace(valueStart, lineEnd - valueStart, newValue);

    std::ofstream out(tmpTaskFile_);
    out << content;
  }

  // Wait briefly for filesystem timestamp granularity
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Force the updater to run by simulating many preSolverRun calls past the counter threshold
  vector_t dummyState = vector_t::Zero(stateDim_);
  for (size_t i = 0; i < 101; ++i) {
    updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
  }

  // Read updated Q value
  matrix_t newQ, newR, newP;
  sqp->getOcpDefinitions().front().costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost").getGains(newQ, newR, newP);

  // The Q matrix should have changed (approximately doubled scaling)
  EXPECT_GT(newQ.norm(), 0.0) << "Updated Q should be non-zero";
  if (origQ.norm() > 0.0) {
    // The ratio should be approximately 2.0 (since we doubled the scaling)
    double ratio = newQ.norm() / origQ.norm();
    EXPECT_NEAR(ratio, 2.0, 0.1) << "Q matrix should have approximately doubled";
  }
}

/******************************************************************************************************/
// Test: Verify that no crash occurs when cost terms are missing from the collection.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, NoCrashOnMissingCostTerms) {
  // Use a minimal task file that might not have all cost terms configured
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);

  // Touch the file to trigger an update
  {
    std::ofstream touch(tmpTaskFile_, std::ios_base::app);
    touch << "\n# trigger update\n";
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Should not crash even if some cost terms don't exist in the collection
  vector_t dummyState = vector_t::Zero(stateDim_);
  EXPECT_NO_THROW({
    for (size_t i = 0; i < 101; ++i) {
      updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
    }
  });
}

/******************************************************************************************************/
// Test: Verify that repeated updates don't crash or cause memory issues.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, RepeatedUpdatesAreStable) {
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);

  vector_t dummyState = vector_t::Zero(stateDim_);

  // Simulate multiple rapid file changes + updates
  for (int round = 0; round < 5; ++round) {
    {
      std::ofstream touch(tmpTaskFile_, std::ios_base::app);
      touch << "# round " << round << "\n";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    EXPECT_NO_THROW({
      for (size_t i = 0; i < 101; ++i) {
        updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
      }
    });
  }
}

/******************************************************************************************************/
// Test: Verify that terminal cost scaling is applied correctly.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, TerminalCostScalingApplied) {
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);

  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // Read original terminal cost
  matrix_t origQ_final;
  try {
    sqp->getOcpDefinitions().front().finalCostPtr->get<QuadraticStateCost>("terminalCost").getGains(origQ_final);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "terminalCost not present in this configuration: " << e.what();
  }

  // Mutate terminalCostScaling
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    auto pos = content.find("terminalCostScaling:");
    ASSERT_NE(pos, std::string::npos);
    auto valueStart = pos + std::string("terminalCostScaling:").size();
    auto lineEnd = content.find('\n', valueStart);
    // Replace with a known value
    content.replace(valueStart, lineEnd - valueStart, " 100.0");

    std::ofstream out(tmpTaskFile_);
    out << content;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Trigger update
  vector_t dummyState = vector_t::Zero(stateDim_);
  for (size_t i = 0; i < 101; ++i) {
    updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
  }

  // Read updated terminal cost
  matrix_t newQ_final;
  sqp->getOcpDefinitions().front().finalCostPtr->get<QuadraticStateCost>("terminalCost").getGains(newQ_final);

  // With scaling=100.0, the terminal cost should be significantly different from original
  EXPECT_GT(newQ_final.norm(), 0.0) << "Updated Q_final should be non-zero";
  EXPECT_NE(newQ_final.norm(), origQ_final.norm()) << "Terminal cost should have changed after scaling update";
}

/******************************************************************************************************/
// Test: Verify joint limits barrier parameters are updated.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, JointLimitsBarrierUpdated) {
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);

  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // Read original joint limits gains
  scalar_t origMu, origDelta;
  try {
    sqp->getOcpDefinitions().front().stateSoftConstraintPtr->get<JointLimitsSoftConstraint>("jointLimits").getGains(origMu, origDelta);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "jointLimits not present in this configuration: " << e.what();
  }

  // Mutate jointLimits.mu
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    auto pos = content.find("jointLimits:");
    ASSERT_NE(pos, std::string::npos);
    auto muPos = content.find("mu:", pos);
    ASSERT_NE(muPos, std::string::npos);
    auto valueStart = muPos + std::string("mu:").size();
    auto lineEnd = content.find('\n', valueStart);
    content.replace(valueStart, lineEnd - valueStart, " 999.0");

    std::ofstream out(tmpTaskFile_);
    out << content;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Trigger update
  vector_t dummyState = vector_t::Zero(stateDim_);
  for (size_t i = 0; i < 101; ++i) {
    updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
  }

  // Read updated gains
  scalar_t newMu, newDelta;
  sqp->getOcpDefinitions().front().stateSoftConstraintPtr->get<JointLimitsSoftConstraint>("jointLimits").getGains(newMu, newDelta);

  EXPECT_NEAR(newMu, 999.0, 1e-6) << "Joint limits mu should have been updated to 999.0";
}

/******************************************************************************************************/
// Test: Verify that no file change means no update is applied.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, NoUpdateWhenFileUnchanged) {
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);

  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // Read original Q
  matrix_t origQ, origR, origP;
  try {
    sqp->getOcpDefinitions().front().costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost").getGains(origQ, origR, origP);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "stateInputQuadraticCost not present: " << e.what();
  }

  // Run without touching the file
  vector_t dummyState = vector_t::Zero(stateDim_);
  for (size_t i = 0; i < 101; ++i) {
    updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
  }

  // Q should be unchanged
  matrix_t sameQ, sameR, sameP;
  sqp->getOcpDefinitions().front().costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost").getGains(sameQ, sameR, sameP);

  EXPECT_DOUBLE_EQ(origQ.norm(), sameQ.norm()) << "Q should not change when file is untouched";
}

// Test: Verify that SQP solver settings are updated at runtime.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, SqpSettingsUpdated) {
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);

  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // Read original sqpIteration
  size_t origIter = sqp->getSettings().sqpIteration;

  // Mutate multiple_shooting.sqpIteration
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    // Anchored to the start of an indented line, not a bare substring: a COMMENT elsewhere in the task file that
    // happens to mention `sqpIteration:` would otherwise be found first and rewritten instead of the setting, which
    // is exactly what happened once.
    const std::string key = "\n  sqpIteration:";
    const size_t pos = content.find(key);
    ASSERT_NE(pos, std::string::npos);
    const size_t valueStart = pos + key.size();
    const size_t lineEnd = content.find('\n', valueStart);
    content.replace(valueStart, lineEnd - valueStart, " 15");

    std::ofstream out(tmpTaskFile_);
    out << content;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Trigger update
  vector_t dummyState = vector_t::Zero(stateDim_);
  for (size_t i = 0; i < 101; ++i) {
    updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
  }

  // Verify
  EXPECT_EQ(sqp->getSettings().sqpIteration, 15u) << "sqpIteration should have been updated to 15";
  EXPECT_NE(sqp->getSettings().sqpIteration, origIter) << "sqpIteration should differ from original";
}

/******************************************************************************************************/
// Test: Verify that zero velocity soft constraint weight (QuadraticPenalty scale) is updated.
/******************************************************************************************************/
/******************************************************************************************************/
// Test: the stance-foot yaw-rate row is switched on a live reload and applied to the constraint in every OCP copy.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, StanceFootYawRateFlagAppliedOnReload) {
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);
  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // Collect the twist constraints behind every foot's zeroVelocity term, hard or soft.
  const auto twistConstraints = [&]() {
    std::vector<EndEffectorKinematicsTwistConstraint*> found;
    for (auto& ocp : sqp->getOcpDefinitions()) {
      for (const auto& footName : contactNames_) {
        const std::string name = footName + "_zeroVelocity";
        size_t index = 0;
        if (ocp.equalityConstraintPtr->getTermIndex(name, index)) {
          if (auto* con = dynamic_cast<ZeroVelocityConstraintCppAd*>(&ocp.equalityConstraintPtr->get(name)))
            found.push_back(&con->getTwistConstraint());
        }
        if (ocp.softConstraintPtr->getTermIndex(name, index)) {
          if (auto* soft = dynamic_cast<StateInputSoftConstraint*>(&ocp.softConstraintPtr->get(name))) {
            if (auto* con = dynamic_cast<ZeroVelocityConstraintCppAd*>(soft->getConstraintPtr().get()))
              found.push_back(&con->getTwistConstraint());
          }
        }
      }
    }
    return found;
  };
  const std::vector<EndEffectorKinematicsTwistConstraint*> before = twistConstraints();
  if (before.empty()) {
    GTEST_SKIP() << "zero_velocity is not part of this configuration";
  }

  const bool shipped = interface_->modelSettings().footConstraintConfig.constrainYawRateAboutContactNormal;
  for (const auto* twist : before) {
    EXPECT_EQ(twist->getConstrainYawRateAboutNormal(), shipped) << "every OCP copy starts from the shipped task file";
  }

  // Flip the flag in the task file and reload.
  const bool flipped = !shipped;
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const std::string key = "constrainYawRateAboutContactNormal:";
    const auto pos = content.find(key);
    ASSERT_NE(pos, std::string::npos) << "the task file must carry the key so that the reload can set it";
    const auto lineEnd = content.find('\n', pos);
    content.replace(pos, lineEnd - pos, key + (flipped ? " true" : " false"));
    std::ofstream out(tmpTaskFile_);
    out << content;
  }
  touchTaskFileAndRunUpdater(updater);

  const std::vector<EndEffectorKinematicsTwistConstraint*> after = twistConstraints();
  ASSERT_EQ(after.size(), before.size());
  for (const auto* twist : after) {
    EXPECT_EQ(twist->getConstrainYawRateAboutNormal(), flipped) << "the reload must apply the flag to every OCP copy";
  }

  // The loader of the model settings reads the same key (the interface uses it at construction).
  const ModelSettings reloaded(tmpTaskFile_, urdfFile_, "centroidal_mpc_", false);  // the name the interface uses
  EXPECT_EQ(reloaded.footConstraintConfig.constrainYawRateAboutContactNormal, flipped);
  EXPECT_EQ(interface_->modelSettings().footConstraintConfig.constrainYawRateAboutContactNormal, shipped)
      << "the interface still holds the settings it was built with";
}

TEST_F(MpcParameterUpdaterModuleTest, SoftConstraintWeightUpdated) {
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);

  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // softConstraintWeight only reaches a penalty when zero_velocity is configured
  // as a SOFT constraint. Listing it under hard_constraints, as the Atlas task
  // file does, puts it in equalityConstraintPtr where there is no penalty to
  // scale, so there is nothing for this test to observe.
  try {
    sqp->getOcpDefinitions().front().softConstraintPtr->get<StateInputSoftConstraint>(contactNames_.front() + "_zeroVelocity");
  } catch (const std::exception&) {
    GTEST_SKIP() << "zero_velocity is a hard constraint in this configuration, so softConstraintWeight has no penalty to update.";
  }

  // Mutate model_settings.foot_constraint.softConstraintWeight to 12345.0
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    auto pos = content.find("softConstraintWeight:");
    if (pos == std::string::npos) {
      GTEST_SKIP() << "softConstraintWeight not found in task.yaml";
    }
    auto valueStart = pos + std::string("softConstraintWeight:").size();
    auto lineEnd = content.find('\n', valueStart);
    content.replace(valueStart, lineEnd - valueStart, " 12345.0");

    std::ofstream out(tmpTaskFile_);
    out << content;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Trigger update
  vector_t dummyState = vector_t::Zero(stateDim_);
  for (size_t i = 0; i < 101; ++i) {
    updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
  }

  // Check that at least one foot's zeroVelocity soft constraint has QuadraticPenalty scale = 12345
  bool foundUpdated = false;
  for (auto& ocp : sqp->getOcpDefinitions()) {
    for (const auto& footName : contactNames_) {
      try {
        auto& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_zeroVelocity");
        for (auto& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
          auto* qp = dynamic_cast<QuadraticPenalty*>(penalty.get());
          if (qp != nullptr && std::abs(qp->getScale() - 12345.0) < 1e-3) {
            foundUpdated = true;
          }
        }
      } catch (...) {
      }
    }
    if (foundUpdated) break;
  }
  EXPECT_TRUE(foundUpdated) << "QuadraticPenalty scale should have been updated to 12345.0";
}

/******************************************************************************************************/
// Test: Verify that foot constraint gains (Ax/Av matrices) are updated.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, FootConstraintGainsUpdated) {
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);

  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // The gains live on the zero-velocity twist constraint, which the contact-implicit formulation replaces with the
  // relaxed complementarity terms (humanoid_nmpc/docs/contact_implicit_mpc/README.md). With that formulation selected
  // there is no such constraint to update, and the hot reload of these gains is simply not applicable.
  const bool hasZeroVelocityConstraint = [&]() {
    for (auto& ocp : sqp->getOcpDefinitions()) {
      for (const auto& footName : contactNames_) {
        try {
          ocp.equalityConstraintPtr->get<ZeroVelocityConstraintCppAd>(footName + "_zeroVelocity");
          return true;
        } catch (...) {
        }
        try {
          ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_zeroVelocity");
          return true;
        } catch (...) {
        }
      }
    }
    return false;
  }();
  if (!hasZeroVelocityConstraint) {
    GTEST_SKIP() << "zero_velocity is not in this configuration's constraint lists (contact-implicit formulation)";
  }

  // Mutate model_settings.foot_constraint.linearVelocityErrorGain_xy to 99.0
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    auto pos = content.find("linearVelocityErrorGain_xy:");
    if (pos == std::string::npos) {
      GTEST_SKIP() << "linearVelocityErrorGain_xy not found in task.yaml";
    }
    auto valueStart = pos + std::string("linearVelocityErrorGain_xy:").size();
    auto lineEnd = content.find('\n', valueStart);
    content.replace(valueStart, lineEnd - valueStart, " 99.0");

    std::ofstream out(tmpTaskFile_);
    out << content;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Trigger update
  vector_t dummyState = vector_t::Zero(stateDim_);
  for (size_t i = 0; i < 101; ++i) {
    updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
  }

  // Check that at least one constraint's Av(0,0) was updated
  bool foundUpdated = false;
  for (auto& ocp : sqp->getOcpDefinitions()) {
    for (const auto& footName : contactNames_) {
      // Check hard constraint path
      try {
        auto& con = ocp.equalityConstraintPtr->get<ZeroVelocityConstraintCppAd>(footName + "_zeroVelocity");
        auto& cfg = con.getTwistConstraint().getConfig();
        if (std::abs(cfg.Av(0, 0) - 99.0) < 1e-3) {
          foundUpdated = true;
        }
      } catch (...) {
      }
      // Check soft constraint path
      try {
        auto& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_zeroVelocity");
        auto* zeroVelCon = dynamic_cast<ZeroVelocityConstraintCppAd*>(softCon.getConstraintPtr().get());
        if (zeroVelCon != nullptr) {
          auto& cfg = zeroVelCon->getTwistConstraint().getConfig();
          if (std::abs(cfg.Av(0, 0) - 99.0) < 1e-3) {
            foundUpdated = true;
          }
        }
      } catch (...) {
      }
    }
    if (foundUpdated) break;
  }
  EXPECT_TRUE(foundUpdated) << "Av(0,0) should have been updated to 99.0 (linearVelocityErrorGain_xy)";
}

/******************************************************************************************************/
// Test: a change to reference.yaml reaches the registered reloaders, so the command limits and ramps can be tuned on
// a running controller instead of needing a restart (the remote control's Command Limits tab writes that file).
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, ReferenceFileChangeReachesTheRegisteredReloaders) {
  const std::string tmpReferenceFile = (std::filesystem::path(testing::TempDir()) / "updater_reference.yaml").string();
  {
    std::ifstream in(referenceFile_);
    std::ofstream out(tmpReferenceFile);
    out << std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  }

  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, tmpReferenceFile, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);
  std::vector<std::string> reloadedWith;
  updater.addReferenceFileReloader([&reloadedWith](const std::string& file) { reloadedWith.push_back(file); });

  const vector_t dummyState = vector_t::Zero(stateDim_);
  // The file is polled once every hundred solves; nothing has changed yet, so nothing is reloaded.
  for (size_t i = 0; i < 101; ++i) {
    updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
  }
  EXPECT_TRUE(reloadedWith.empty()) << "an untouched reference file must not trigger a reload";

  {
    std::ofstream out(tmpReferenceFile, std::ios::app);
    out << "\n# touched by the test\n";
  }
  std::filesystem::last_write_time(tmpReferenceFile, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(1));
  for (size_t i = 0; i < 101; ++i) {
    updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
  }
  ASSERT_EQ(reloadedWith.size(), 1U) << "a changed reference file must reload exactly once";
  EXPECT_EQ(reloadedWith.front(), tmpReferenceFile);

  std::remove(tmpReferenceFile.c_str());
}

/******************************************************************************************************/
// Test: the values a reload reads are the ones the command path then uses. The pelvis height is the observable one:
// commandedPositionToTargetTrajectories clamps the commanded delta to maxDeltaPelvisHeight and adds defaultBaseHeight,
// so both reloaded values appear directly in the target it returns.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, ReloadingCommandLimitsChangesTheTargetItProduces) {
  const std::string tmpReferenceFile = (std::filesystem::path(testing::TempDir()) / "updater_limits.yaml").string();
  const auto writeWith = [&tmpReferenceFile, this](scalar_t maxDeltaPelvisHeight, scalar_t defaultBaseHeight) {
    std::ifstream in(referenceFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    content = std::regex_replace(content, std::regex("maxDeltaPelvisHeight: *[0-9.]+"),
                                 "maxDeltaPelvisHeight: " + std::to_string(maxDeltaPelvisHeight));
    content =
        std::regex_replace(content, std::regex("defaultBaseHeight: *[0-9.]+"), "defaultBaseHeight: " + std::to_string(defaultBaseHeight));
    std::ofstream out(tmpReferenceFile);
    out << content;
  };

  writeWith(0.10, 0.90);
  CentroidalMpcTargetTrajectoriesCalculator calculator(tmpReferenceFile, interface_->getEffectiveMpcRobotModel(),
                                                       interface_->getPinocchioInterface(), interface_->getCentroidalModelInfo(),
                                                       interface_->mpcSettings().timeHorizon_);

  // A crouch far beyond the limit, so the returned height is the clamp itself: defaultBaseHeight - maxDeltaPelvisHeight.
  const vector_t state = interface_->getInitialState();
  const vector4_t deepCrouch(0.0, 0.0, -10.0, 0.0);
  const TargetTrajectories before = calculator.commandedPositionToTargetTrajectories(deepCrouch, 0.0, state);
  ASSERT_FALSE(before.stateTrajectory.empty());
  const scalar_t heightBefore = interface_->getEffectiveMpcRobotModel().getBasePosition(before.stateTrajectory.back())(2);
  EXPECT_NEAR(heightBefore, 0.90 - 0.10, 1e-6);

  writeWith(0.25, 0.80);
  calculator.reloadCommandLimits(tmpReferenceFile);
  const TargetTrajectories after = calculator.commandedPositionToTargetTrajectories(deepCrouch, 0.0, state);
  ASSERT_FALSE(after.stateTrajectory.empty());
  const scalar_t heightAfter = interface_->getEffectiveMpcRobotModel().getBasePosition(after.stateTrajectory.back())(2);
  EXPECT_NEAR(heightAfter, 0.80 - 0.25, 1e-6) << "the reloaded limits must be the ones the target is built from";

  std::remove(tmpReferenceFile.c_str());
}

/******************************************************************************************************/
// Test: With a basis-space cost transform the constructor rejects an inputDim that is not the basis-space
// dimension. Passing the wrench-space dimension is exactly the mistake this guards against.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, BasisCostTransformRejectsMismatchedInputDim) {
  const auto config = makeSyntheticBasisCostTransform(/*numBasisPerFoot=*/8, /*lambdaRegularization=*/0.1);
  ASSERT_NE(config.basisInputDim(), interface_->getWrenchInputDim());

  EXPECT_THROW(
      {
        MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, interface_->getWrenchInputDim(),
                                          contactNames_, nullptr, config);
      },
      std::invalid_argument);
  EXPECT_THROW(
      {
        MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, config.basisInputDim() + 1,
                                          contactNames_, nullptr, config);
      },
      std::invalid_argument);
  EXPECT_NO_THROW({
    MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, config.basisInputDim(), contactNames_,
                                      nullptr, config);
  });
}

/******************************************************************************************************/
// Test: With a synthetic basis-space cost transform, the R written into the OCP equals Mᵀ R_wrench M with the
// λ regularization on the leading diagonal only — i.e. the yaml R is read in wrench space and transformed.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, BasisCostTransformAppliedToInputCost) {
  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  try {
    matrix_t Q, R, P;
    sqp->getOcpDefinitions().front().costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost").getGains(Q, R, P);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "stateInputQuadraticCost not present in this configuration: " << e.what();
  }

  constexpr scalar_t lambdaRegularization = 0.25;
  const auto config = makeSyntheticBasisCostTransform(/*numBasisPerFoot=*/8, lambdaRegularization);
  const size_t basisInputDim = config.basisInputDim();

  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, basisInputDim, contactNames_, nullptr,
                                    config);
  touchTaskFileAndRunUpdater(updater);

  const matrix_t R_wrench = loadWrenchSpaceR();
  ASSERT_GT(R_wrench.norm(), 0.0) << "task.yaml R should be non-zero";
  const matrix_t expectedR = transformWrenchInputCostToBasisSpace(R_wrench, config);

  for (auto& ocp : sqp->getOcpDefinitions()) {
    matrix_t Q, R, P;
    ocp.costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost").getGains(Q, R, P);
    ASSERT_EQ(static_cast<size_t>(R.rows()), basisInputDim);
    ASSERT_EQ(static_cast<size_t>(R.cols()), basisInputDim);
    EXPECT_TRUE(R.isApprox(expectedR, 1e-9)) << "R must equal the basis-space transform of the wrench-space yaml R";

    // The regularization must land on the λ diagonal only, never on the joint-velocity block.
    const matrix_t unregularized = config.basisToWrenchMap.transpose() * R_wrench * config.basisToWrenchMap;
    const vector_t diagonalDelta = (R - unregularized).diagonal();
    EXPECT_TRUE(diagonalDelta.head(config.numBasisInputs).isApproxToConstant(lambdaRegularization, 1e-9));
    EXPECT_NEAR(diagonalDelta.tail(basisInputDim - config.numBasisInputs).norm(), 0.0, 1e-9);
    EXPECT_TRUE((R - unregularized - diagonalDelta.asDiagonal().toDenseMatrix()).isZero(1e-9))
        << "Only the diagonal may differ from Mᵀ R_wrench M";
  }
}

/******************************************************************************************************/
// Test: With the interface's real basis-vector configuration, an online update from an unchanged task.yaml must
// reproduce exactly the R the OCP factory built, i.e. the updater and the factory apply the same transform.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, BasisCostTransformMatchesOcpFactory) {
  if (!interface_->usesContactBasisVectorInputs()) {
    GTEST_SKIP() << "useContactBasisVectorInputs is disabled in this configuration";
  }
  ASSERT_TRUE(basisCostTransform_.has_value());
  EXPECT_EQ(inputDim_, basisCostTransform_->basisInputDim());
  EXPECT_NE(inputDim_, interface_->getWrenchInputDim()) << "basis-space and wrench-space input dimensions should differ";

  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  matrix_t origQ, origR, origP;
  try {
    sqp->getOcpDefinitions().front().costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost").getGains(origQ, origR, origP);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "stateInputQuadraticCost not present in this configuration: " << e.what();
  }
  ASSERT_EQ(static_cast<size_t>(origR.rows()), inputDim_) << "OCP factory R should already be in basis space";

  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);
  touchTaskFileAndRunUpdater(updater);

  matrix_t newQ, newR, newP;
  sqp->getOcpDefinitions().front().costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost").getGains(newQ, newR, newP);
  ASSERT_EQ(static_cast<size_t>(newR.rows()), inputDim_);
  ASSERT_EQ(static_cast<size_t>(newR.cols()), inputDim_);

  EXPECT_TRUE(newR.isApprox(origR, 1e-9)) << "Online update must reproduce the OCP factory's basis-space R for an unchanged yaml";
  EXPECT_TRUE(newR.isApprox(transformWrenchInputCostToBasisSpace(loadWrenchSpaceR(), *basisCostTransform_), 1e-9));
}

/******************************************************************************************************/
// Test: ComAndAcomTrackingWeightsUpdated
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, ComAndAcomTrackingWeightsUpdated) {
  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  auto& ocp0 = sqp->getOcpDefinitions().front();
  if (ocp0.stateCostPtr == nullptr) {
    GTEST_SKIP() << "stateCostPtr is null";
  }
  try {
    ocp0.stateCostPtr->get<ComAndAcomTrackingCost>("comAndAcomTrackingCost");
  } catch (...) {
    GTEST_SKIP() << "comAndAcomTrackingCost not present in this configuration";
  }

  // Mutate one entry of Q_com and one of Q_acom in tmpTaskFile_.
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    const auto replaceEntry = [&content](const std::string& matrixName, const std::string& entry, const std::string& newValue) {
      const auto matrixPos = content.find(matrixName + ":");
      ASSERT_NE(matrixPos, std::string::npos) << matrixName << " not found in task file";
      const auto entryPos = content.find("\"" + entry + "\":", matrixPos);
      ASSERT_NE(entryPos, std::string::npos) << entry << " not found under " << matrixName;
      const auto valueStart = entryPos + entry.size() + 3;  // Past the quoted key and colon.
      const auto lineEnd = content.find('\n', valueStart);
      content.replace(valueStart, lineEnd - valueStart, " " + newValue);
    };
    replaceEntry("Q_com", "(2,2)", "999.0");
    replaceEntry("Q_acom", "(0,0)", "777.0");

    std::ofstream out(tmpTaskFile_);
    out << content;
  }

  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);
  touchTaskFileAndRunUpdater(updater);

  // Each matrix is premultiplied by its own `scaling` entry, and the two are not the same number in every robot's
  // task file. Read them rather than hard-coding, or this test rots the next time either is retuned.
  scalar_t comScaling = 1.0;
  scalar_t acomScaling = 1.0;
  loadData::loadCppDataType(tmpTaskFile_, "Q_com.scaling", comScaling);
  loadData::loadCppDataType(tmpTaskFile_, "Q_acom.scaling", acomScaling);

  // Every per-thread clone of the problem must see the update, not just the first.
  for (auto& ocp : sqp->getOcpDefinitions()) {
    auto& cost = ocp.stateCostPtr->get<ComAndAcomTrackingCost>("comAndAcomTrackingCost");
    EXPECT_NEAR(cost.getQCom()(2, 2), comScaling * 999.0, 1e-3);
    // Row 0 of Q_acom is yaw, in the centroidal state's ZYX Euler convention.
    EXPECT_NEAR(cost.getQAcom()(0, 0), acomScaling * 777.0, 1e-3);
  }
}

/******************************************************************************************************/
// Test: BasePoseWeightsZeroedInBothRunningAndTerminalCost
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, BasePoseWeightsZeroedInBothRunningAndTerminalCost) {
  if (!interface_->modelSettings().useComAndAcomTracking) {
    GTEST_SKIP() << "useComAndAcomTracking is disabled in this configuration";
  }
  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // CoM + aCOM tracking regulates base pose in its own coordinates, so the base
  // pose block must be zero in the running AND the terminal state cost. Zeroing
  // only one of them leaves the end of the horizon pulled toward a base pose
  // target while every other node follows the aCOM coordinate.
  constexpr Eigen::Index kBasePoseStateIndex = 6;
  constexpr Eigen::Index kBasePoseDim = 6;

  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);
  touchTaskFileAndRunUpdater(updater);

  bool useDcmTerminalCost = false;
  loadData::loadCppDataType(taskFile_, "useDcmTerminalCost", useDcmTerminalCost);
  for (auto& ocp : sqp->getOcpDefinitions()) {
    if (useDcmTerminalCost) {
      // The DCM terminal cost replaces the quadratic terminal cost, so there is no Q_final to zero: the base pose is not
      // regulated at the horizon end at all.
      EXPECT_THROW(ocp.finalCostPtr->get<QuadraticStateCost>("terminalCost"), std::out_of_range);
      EXPECT_NO_THROW(ocp.finalCostPtr->get<DcmTerminalCost>("dcmTerminalCost"));
      continue;
    }
    matrix_t Q_final;
    ocp.finalCostPtr->get<QuadraticStateCost>("terminalCost").getGains(Q_final);
    EXPECT_TRUE(Q_final.block(kBasePoseStateIndex, kBasePoseStateIndex, kBasePoseDim, kBasePoseDim).isZero(1e-12))
        << "Terminal cost still penalises base pose while aCOM tracking is active.";
    // The rest of the terminal cost must survive, otherwise this would pass
    // trivially for an all-zero matrix.
    EXPECT_GT(Q_final.norm(), 0.0);
  }
}

/******************************************************************************************************/
// Test: BasisNonNegativityBarrierUpdated
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, BasisNonNegativityBarrierUpdated) {
  if (!interface_->usesContactBasisVectorInputs()) {
    GTEST_SKIP() << "useContactBasisVectorInputs is disabled in this configuration";
  }
  auto* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // Mutate contacts.basisNonNegativityBarrier.mu
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    auto pos = content.find("basisNonNegativityBarrier:");
    ASSERT_NE(pos, std::string::npos);
    auto pMu = content.find("mu:", pos);
    ASSERT_NE(pMu, std::string::npos);
    auto valueStart = pMu + std::string("mu:").size();
    auto lineEnd = content.find('\n', valueStart);
    content.replace(valueStart, lineEnd - valueStart, " 0.42");

    std::ofstream out(tmpTaskFile_);
    out << content;
  }

  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);
  touchTaskFileAndRunUpdater(updater);

  for (auto& ocp : sqp->getOcpDefinitions()) {
    for (const auto& footName : contactNames_) {
      auto& con = ocp.costPtr->get<BasisScalingNonNegativityConstraint>(footName + "_basisNonNegativity");
      EXPECT_NEAR(con.getBarrierConfig().mu, 0.42, 1e-4);
    }
  }
}

/******************************************************************************************************/
// Test: ContactImplicitTuningReachesEveryTerm
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, ContactImplicitTuningReachesEveryTerm) {
  // The contact-implicit block has three terms and two of them share the terrain height: one says a foot may not carry
  // load above the ground, the other that it may not go below it. A height applied to one and not the other leaves the
  // formulation with two disagreeing definitions of where the ground is. That was a real bug, and nothing caught it
  // because live tuning of this block had no coverage at all - which is the gap this test closes.
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  const std::string probeFoot = contactNames_.front();
  try {
    sqp->getOcpDefinitions().front().softConstraintPtr->get<StateInputSoftConstraint>(probeFoot + "_contactComplementarity");
  } catch (...) {
    GTEST_SKIP() << "the contact-implicit terms are not enabled in this robot's task file";
  }

  const scalar_t newTerrainHeight = 0.017;
  const scalar_t newHeightReference = 0.123;
  const scalar_t newVelocityReference = 0.456;
  const scalar_t newAngularVelocityReference = 0.789;
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    // `terrainHeight` is a TOP-LEVEL key of the task file, while the rest of this block is indented under
    // `contact_implicit`. Searching only for the indented spelling silently stopped finding it when the ground height
    // was made a single source of truth, so the setter accepts either indentation and asserts it found one.
    const std::function<void(const std::string&, scalar_t)> setKey = [&content](const std::string& key, scalar_t value) {
      size_t indent = 2;
      size_t keyPos = content.find("\n  " + key + ":");
      if (keyPos == std::string::npos) {
        indent = 0;
        keyPos = content.find("\n" + key + ":");
      }
      ASSERT_NE(keyPos, std::string::npos) << key << " not found at the top level or under contact_implicit";
      const size_t valueStart = keyPos + 1 + indent + key.size() + 1;  // past the newline, the indent, the key and the colon
      const size_t lineEnd = content.find('\n', valueStart);
      content.replace(valueStart, lineEnd - valueStart, " " + std::to_string(value));
    };
    setKey("terrainHeight", newTerrainHeight);
    setKey("heightReference", newHeightReference);
    setKey("velocityReference", newVelocityReference);
    setKey("angularVelocityReference", newAngularVelocityReference);

    std::ofstream out(tmpTaskFile_);
    out << content;
  }

  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);
  touchTaskFileAndRunUpdater(updater);

  // Every per-thread clone of the problem, and every foot, must see all of it.
  for (OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
    for (const std::string& footName : contactNames_) {
      const ContactComplementarityConstraint& complementarity =
          ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_contactComplementarity")
              .get<ContactComplementarityConstraint>();
      EXPECT_NEAR(complementarity.getTerrainHeight(), newTerrainHeight, 1e-9) << footName;
      EXPECT_NEAR(complementarity.getHeightReference(), newHeightReference, 1e-9) << footName;

      const ForceWeightedSlipConstraint& slip =
          ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_forceWeightedSlip").get<ForceWeightedSlipConstraint>();
      EXPECT_NEAR(slip.getInverseTwistReference()(0), 1.0 / newVelocityReference, 1e-9) << footName;
      EXPECT_NEAR(slip.getInverseTwistReference()(1), 1.0 / newVelocityReference, 1e-9) << footName;
      EXPECT_NEAR(slip.getInverseTwistReference()(2), 1.0 / newAngularVelocityReference, 1e-9) << footName;

      const GroundPenetrationConstraint& penetration =
          ocp.stateSoftConstraintPtr->get<StateSoftConstraint>(footName + "_groundPenetration").get<GroundPenetrationConstraint>();
      EXPECT_NEAR(penetration.getTerrainHeight(), newTerrainHeight, 1e-9)
          << footName << ": the penetration barrier still places the ground somewhere else than the complementarity term does";
    }
  }
}

TEST_F(MpcParameterUpdaterModuleTest, ThePositionErrorGainReachesTheSoftNormalVelocityTerm) {
  // positionErrorGain_z was loaded into a local FootConstraintConfig and written into the zeroVelocity twist config -
  // but under the contact-implicit formulation zeroVelocity is not built at all, and the term that DOES use the gain,
  // the soft normal-velocity servo, reads it from the PreComputation. So the slider moved a term that did not exist
  // while the one that did kept its launch value. The gain enters the residual v_z - zdot_ref + k (z - z_ref)
  // linearly, hence the curvature SQUARED: it is the strongest single knob on swing-foot tracking, and it was inert.
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  const scalar_t newGain = 3.5;
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const std::string key = "\n    positionErrorGain_z:";
    const size_t keyPos = content.find(key);
    ASSERT_NE(keyPos, std::string::npos) << "the task file has no model_settings.foot_constraint.positionErrorGain_z";
    const size_t valueStart = keyPos + key.size();
    const size_t lineEnd = content.find('\n', valueStart);
    content.replace(valueStart, lineEnd - valueStart, " " + std::to_string(newGain));
    std::ofstream out(tmpTaskFile_);
    out << content;
  }

  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);
  touchTaskFileAndRunUpdater(updater);

  // Every per-thread clone has to see it: the gain lives on the PreComputation precisely because each worker owns one.
  for (OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
    const HumanoidPreComputation* preComputationPtr = dynamic_cast<const HumanoidPreComputation*>(ocp.preComputationPtr.get());
    ASSERT_NE(preComputationPtr, nullptr) << "the centroidal MPC is expected to use HumanoidPreComputation";
    EXPECT_NEAR(preComputationPtr->getNormalVelocityPositionErrorGain(), newGain, 1e-9);
  }
}

/******************************************************************************************************/
// Test: ContactPlanningReloadAppliesModelParametersBeforeValidating
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, ContactPlanningReloadAppliesModelParametersBeforeValidating) {
  // A robot may write 0 for the planner parameters that are properties of the model rather than tuning - shared.comHeight
  // and the two zmp_support_region half widths - and have them derived from the URDF and from the wrench cone instead
  // (ContactPlanningModelParameters::applyTo, documented in ContactPlanningConfig.h and in both shipped
  // contact_planning.yaml files). The start-up path in CentroidalMpcInterface therefore loads the file with the loader's
  // own validation switched OFF, applies the derived parameters and validates only afterwards.
  //
  // The hot reload did the opposite: it left the loader's `validate` argument at its default of true, so the file was
  // validated while those three fields were still 0 - and 0 in those three fields is exactly what validate() rejects.
  // Every reload of such a file therefore threw inside the loader, before ContactPlannerModule::setConfig could run
  // applyTo, the throw was caught and became a single warning, and because the file watcher in preSolverRun had already
  // stored the new modification time the edit was gone for good: the operator moved a slider, the tuning GUI reported the
  // change as applied, and the planner ran on its launch configuration for the rest of the session. No shipped robot uses
  // the markers today, which is why nothing noticed; the first one to take the documented option would have lost every
  // reload. This test therefore writes a planner file that takes that option and drives the real watcher over it.
  const std::string shippedPlanningFile = resolveContactPlanningConfigFile(taskFile_);
  if (shippedPlanningFile == taskFile_) {
    GTEST_SKIP() << "this robot keeps the contact_planning block inside its task file, so there is no separate file to watch";
  }

  // A directory of its own: the planner file is found by its fixed name next to the task file, and the fixture's own
  // temporary task file must not suddenly acquire one.
  const std::filesystem::path planningDir = std::filesystem::path(testing::TempDir()) / "mpc_parameter_updater_contact_planning";
  std::error_code ec;
  std::filesystem::remove_all(planningDir, ec);
  std::filesystem::create_directories(planningDir, ec);
  ASSERT_FALSE(ec) << "could not create " << planningDir.string() << ": " << ec.message();
  const std::string planningTaskFile = (planningDir / "task.yaml").string();
  const std::string planningFile = (planningDir / kContactPlanningConfigFileName).string();
  {
    std::ifstream src(taskFile_);
    std::ofstream dst(planningTaskFile);
    dst << src.rdbuf();
  }

  std::string planning;
  {
    std::ifstream in(shippedPlanningFile);
    ASSERT_TRUE(in.is_open()) << "could not read " << shippedPlanningFile;
    planning.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  // Replaces the value of one key, addressed with its exact indentation so that a key of the same name in another block
  // (or the same word inside a comment) cannot be hit by accident.
  const std::function<void(const std::string&, const std::string&)> setKey = [&planning](const std::string& key, const std::string& value) {
    const std::string needle = "\n" + key + ":";
    const size_t keyPos = planning.find(needle);
    ASSERT_NE(keyPos, std::string::npos) << key << " is not a key of the shipped contact_planning.yaml";
    const size_t valueStart = keyPos + needle.size();
    const size_t lineEnd = planning.find('\n', valueStart);
    planning.replace(valueStart, lineEnd - valueStart, " " + value);
  };
  const std::function<void()> writePlanningFile = [&planning, &planningFile]() {
    std::ofstream out(planningFile);
    out << planning;
  };

  setKey("    comHeight", "0.0");   // shared.comHeight: from the model
  setKey("    halfWidthX", "0.0");  // zmp_support_region: from the sole's footprint
  setKey("    halfWidthY", "0.0");
  setKey("    runInBackgroundThread", "false");  // plan in the pre-solve hook, so this test starts no worker thread
  writePlanningFile();
  ASSERT_EQ(resolveContactPlanningConfigFile(planningTaskFile), planningFile);

  // The file is the documented "derive it from the model" case: on its own it does NOT validate. That is the whole point
  // - it is the reason the reload may not ask the loader to validate it before the model parameters have been applied.
  EXPECT_THROW(loadContactPlanningConfig(planningFile, "contact_planning.", false), std::invalid_argument);
  ContactPlanningConfig config = loadContactPlanningConfig(planningFile, "contact_planning.", false, /*validate=*/false);
  ASSERT_EQ(config.shared.comHeight, 0.0);
  ASSERT_EQ(config.zmpSupportRegion.halfWidthX, 0.0);

  // Synthetic model parameters rather than the ones derived from the Atlas model: what is under test is the ORDER in
  // which the reload applies and validates them, so the expected values are better off being constants this test owns.
  ContactPlanningModelParameters modelParameters;
  modelParameters.totalMass = 80.0;
  modelParameters.comHeight = 0.93;
  modelParameters.zmpHalfWidthX = 0.11;
  modelParameters.zmpHalfWidthY = 0.055;
  modelParameters.torsionalFrictionTorque = 12.0;
  modelParameters.doubleSupportYawCouple = 34.0;
  modelParameters.footYawOffsetLower = makeFeetArray(static_cast<scalar_t>(-0.4));
  modelParameters.footYawOffsetUpper = makeFeetArray(static_cast<scalar_t>(0.4));
  modelParameters.hipYawJoints.assign(N_CONTACTS, std::string());
  modelParameters.applyTo(config);
  ASSERT_NO_THROW(config.validate()) << "with the model parameters applied the very same file is valid";

  std::shared_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner =
      std::make_shared<SwingTrajectoryPlanner>(loadSwingTrajectorySettings(taskFile_, "swing_trajectory_config", false), N_CONTACTS);
  std::shared_ptr<ContactPlanningReferenceManager> planningReferenceManager = std::make_shared<ContactPlanningReferenceManager>(
      GaitSchedule::loadGaitSchedule(referenceFile_, interface_->modelSettings(), false), swingTrajectoryPlanner,
      interface_->getPinocchioInterface(), interface_->getEffectiveMpcRobotModel(), config);
  std::shared_ptr<ContactPlannerModule> plannerModule = std::make_shared<ContactPlannerModule>(planningReferenceManager, config);
  plannerModule->setModelParameters(modelParameters);
  ASSERT_NEAR(plannerModule->getConfig().shared.comHeight, modelParameters.comHeight, 1e-12) << "the module knows its model parameters";

  // No MPC: this path updates the planner only, and the task file is never touched, so nothing else in the updater runs.
  MpcParameterUpdaterModule updater(nullptr, planningTaskFile, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, nullptr,
                                    basisCostTransform_);
  updater.setContactPlannerModule(plannerModule);

  // The operator moves a slider: the tuning GUI saves the planner file, and the watcher inside preSolverRun has to pick
  // it up. The sleeps bracket the write because the watcher compares modification times, whose granularity is coarser
  // than this test's own timing.
  const scalar_t reloadedMaxSolveTime = 0.234;
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  setKey("    maxSolveTime", std::to_string(reloadedMaxSolveTime));
  writePlanningFile();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const vector_t dummyState = vector_t::Zero(stateDim_);
  for (size_t i = 0; i < 101; ++i) {  // the file is stat-ed once every 100 pre-solve hooks
    updater.preSolverRun(0.0, 1.0, dummyState, *interface_->getReferenceManagerPtr());
  }

  const ContactPlanningConfig applied = plannerModule->getConfig();
  EXPECT_NEAR(applied.planner.maxSolveTime, reloadedMaxSolveTime, 1e-9)
      << "the edited planner file never reached the module: the reload was rejected inside the loader, before the model-derived "
         "comHeight and ZMP box could be applied, and the warning that followed is the only trace. The watcher has already stored the "
         "new modification time, so this edit and every later one is lost for the rest of the run.";
  EXPECT_NEAR(applied.shared.comHeight, modelParameters.comHeight, 1e-12) << "the reload has to re-apply the model-derived CoM height";
  EXPECT_NEAR(applied.zmpSupportRegion.halfWidthX, modelParameters.zmpHalfWidthX, 1e-12);
  EXPECT_NEAR(applied.zmpSupportRegion.halfWidthY, modelParameters.zmpHalfWidthY, 1e-12);
  EXPECT_TRUE(applied.hasModelParameters());
  EXPECT_NO_THROW(applied.validate()) << "skipping the loader's validation may never hand the planner an unvalidated configuration";

  std::filesystem::remove_all(planningDir, ec);
}

}  // namespace ocs2::humanoid
