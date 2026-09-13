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
#include <optional>
#include <stdexcept>
#include <string>
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
#include "humanoid_centroidal_mpc/constraint/ZeroVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_centroidal_mpc/mrt/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"

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

    auto pos = content.find("sqpIteration:");
    ASSERT_NE(pos, std::string::npos);
    auto valueStart = pos + std::string("sqpIteration:").size();
    auto lineEnd = content.find('\n', valueStart);
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

  // Every per-thread clone of the problem must see the update, not just the first.
  for (auto& ocp : sqp->getOcpDefinitions()) {
    auto& cost = ocp.stateCostPtr->get<ComAndAcomTrackingCost>("comAndAcomTrackingCost");
    // Both matrices are premultiplied by their respective `scaling` entry, 85.
    EXPECT_NEAR(cost.getQCom()(2, 2), 85.0 * 999.0, 1e-3);
    // Row 0 of Q_acom is yaw, in the centroidal state's ZYX Euler convention.
    EXPECT_NEAR(cost.getQAcom()(0, 0), 85.0 * 777.0, 1e-3);
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

  for (auto& ocp : sqp->getOcpDefinitions()) {
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

}  // namespace ocs2::humanoid
