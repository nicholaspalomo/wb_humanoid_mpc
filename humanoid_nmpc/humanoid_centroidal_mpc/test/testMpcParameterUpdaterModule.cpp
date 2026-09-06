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

#include <fstream>
#include <string>
#include <thread>

#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_sqp/SqpMpc.h>
#include <ocs2_sqp/SqpSolver.h>

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_centroidal_mpc/mrt/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"

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
    inputDim_ = interface_->getMpcRobotModel().getInputDim();
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

  std::string taskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::string tmpTaskFile_;
  std::unique_ptr<CentroidalMpcInterface> interface_;
  std::unique_ptr<SqpMpc> mpc_;
  size_t stateDim_;
  size_t inputDim_;
  std::vector<std::string> contactNames_;
};

/******************************************************************************************************/
// Test: Verify that construction succeeds and initial state is sane.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, ConstructionSucceeds) {
  EXPECT_NO_THROW(
      { MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_); });
}

/******************************************************************************************************/
// Test: Verify that quadratic cost weights (Q/R) are updated in-place after file change.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, QuadraticCostWeightsUpdatedInPlace) {
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_);

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
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_);

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
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_);

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
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_);

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
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_);

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
  MpcParameterUpdaterModule updater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_);

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

}  // namespace ocs2::humanoid
