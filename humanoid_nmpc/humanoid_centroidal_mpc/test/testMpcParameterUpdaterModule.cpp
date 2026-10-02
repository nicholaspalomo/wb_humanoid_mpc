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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/reference/TargetTrajectories.h>
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
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcParameterUpdater.h"
#include "humanoid_centroidal_mpc/mrt/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/HumanoidPreComputation.h"
#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
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
#include "humanoid_common_mpc/contact_planning/TargetContactPose.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/locomotion_heuristics/BasePoseHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "robot_core/ResourcePaths.h"

#include <ocs2_core/penalties/penalties/QuadraticPenalty.h>
#include <ocs2_sqp/SqpSettings.h>

namespace ocs2::humanoid {

namespace acom_test {

// The base pose in the centroidal state x = [h_norm(6), p_base(3), euler_zyx(3), q_j], written out rather than taken
// from ComAndAcomTrackingCost so that a drift of the one definition fails here instead of moving the expectation.
constexpr Eigen::Index kBasePoseIndex = 6;
constexpr Eigen::Index kBasePoseDim = 6;

/** `content` with the value of the diagonal entry (i,i) of the top-level matrix `matrixName` replaced by `value`. */
std::string withDiagonalEntry(const std::string& content, const std::string& matrixName, Eigen::Index i, scalar_t value) {
  const std::string::size_type blockStart = content.find(absl::StrCat("\n", matrixName, ":\n"));
  EXPECT_NE(blockStart, std::string::npos) << matrixName << " not found";
  if (blockStart == std::string::npos) return content;
  const std::string key = absl::StrCat("\"(", i, ",", i, ")\":");
  const std::string::size_type entry = content.find(key, blockStart + 1);
  EXPECT_NE(entry, std::string::npos) << matrixName << key << " not found";
  if (entry == std::string::npos) return content;
  std::string result = content;
  result.replace(entry, result.find('\n', entry) - entry, absl::StrCat(key, " ", value));
  return result;
}

/** `content` with every base-pose weight (6..11) of `matrixName` set to a distinct non-zero value, offset by `offset`. */
std::string withBasePoseWeights(const std::string& content, const std::string& matrixName, scalar_t offset) {
  std::string result = content;
  for (Eigen::Index i = kBasePoseIndex; i < kBasePoseIndex + kBasePoseDim; ++i) {
    result = withDiagonalEntry(result, matrixName, i, offset + static_cast<scalar_t>(i));
  }
  return result;
}

/** `content` with `from` replaced by `to` exactly once. */
std::string replacedOnce(const std::string& content, const std::string& from, const std::string& to) {
  const std::string::size_type position = content.find(from);
  EXPECT_NE(position, std::string::npos) << "'" << from << "' not found";
  if (position == std::string::npos) return content;
  std::string result = content;
  result.replace(position, from.size(), to);
  return result;
}

const std::string kAcomCostEntry = "\n  - com_and_acom_tracking_cost\n";

// The two terminal costs of the costs list, as the shipped Atlas file spells its entry.
const std::string kDcmTerminalCostEntry = "\n  - dcm_terminal_cost\n";
const std::string kQuadraticTerminalCostEntry = "\n  - terminal_cost\n";

/** `content` ending its horizon on the quadratic Q_final cost instead of the DCM cost the shipped Atlas lists. */
std::string withQuadraticTerminalCost(const std::string& content) {
  return replacedOnce(content, kDcmTerminalCostEntry, kQuadraticTerminalCostEntry);
}

/** `content` ending its horizon on the DCM cost instead of the quadratic Q_final cost. */
std::string withDcmTerminalCost(const std::string& content) {
  return replacedOnce(content, kQuadraticTerminalCostEntry, kDcmTerminalCostEntry);
}

/** The base-pose block of a matrix. */
matrix_t basePoseBlock(const matrix_t& Q) {
  return Q.block(kBasePoseIndex, kBasePoseIndex, kBasePoseDim, kBasePoseDim);
}

}  // namespace acom_test

/**
 * MpcParameterUpdaterModule::Create() with arguments the calling test expects to be accepted: the updater, or nullptr
 * after a failure that prints the refusal.
 */
std::unique_ptr<MpcParameterUpdaterModule> createUpdater(MPC_BASE* mpc,
                                                         const std::string& taskFile,
                                                         const std::string& urdfFile,
                                                         const std::string& referenceFile,
                                                         size_t stateDim,
                                                         size_t inputDim,
                                                         const std::vector<std::string>& contactNames,
                                                         SwitchedModelReferenceManager* referenceManager,
                                                         std::optional<BasisInputsCostTransformConfig> basisCostTransform) {
  absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> created = MpcParameterUpdaterModule::Create(
      mpc, taskFile, urdfFile, referenceFile, stateDim, inputDim, contactNames, referenceManager, std::move(basisCostTransform));
  EXPECT_TRUE(created.ok()) << created.status();
  return created.ok() ? *std::move(created) : nullptr;
}

/**
 * Test fixture for MpcParameterUpdaterModule.
 *
 * Constructs a real CentroidalMpcInterface + SqpMpc from test robot config,
 * so all OCP cost/constraint names match the production configuration.
 */
class MpcParameterUpdaterModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // The DRC Atlas files, from the test's runfiles.
    taskFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml").value();
    referenceFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml").value();
    urdfFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf").value();

    // Create the interface
    absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> status = CentroidalMpcInterface::Create(taskFile_, urdfFile_, referenceFile_);
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
      updater.preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
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

  std::string readTmpTaskFile() const {
    std::ifstream in(tmpTaskFile_);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  }

  void writeTmpTaskFile(const std::string& content) const {
    std::ofstream out(tmpTaskFile_);
    out << content;
  }

  /** A second interface and its MPC, built from a variant of the task file, for the cases the shipped one cannot show. */
  struct Built {
    std::unique_ptr<CentroidalMpcInterface> interface;
    std::unique_ptr<SqpMpc> mpc;
    SqpSolver& solver() const { return dynamic_cast<SqpSolver&>(*mpc->getSolverPtr()); }
  };

  Built build(const std::string& taskFile) const {
    Built built;
    absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = CentroidalMpcInterface::Create(taskFile, urdfFile_, referenceFile_);
    EXPECT_TRUE(created.ok()) << created.status();
    if (!created.ok()) return built;
    built.interface = *std::move(created);
    built.mpc = std::make_unique<SqpMpc>(built.interface->mpcSettings(), built.interface->sqpSettings(),
                                         built.interface->getOptimalControlProblem(), built.interface->getInitializer());
    built.mpc->getSolverPtr()->setReferenceManager(built.interface->getReferenceManagerPtr());
    return built;
  }

  /** Touches `taskFile` and drives `updater` past its ~1 Hz file check, as touchTaskFileAndRunUpdater does for the fixture. */
  static void touchAndRun(const std::string& taskFile, MpcParameterUpdaterModule& updater, const CentroidalMpcInterface& interface) {
    {
      std::ofstream touch(taskFile, std::ios_base::app);
      touch << "\n# trigger update\n";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const vector_t state = interface.getInitialState();
    for (size_t i = 0; i < 101; ++i) {
      updater.preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, state, *interface.getReferenceManagerPtr());
    }
  }

  static std::unique_ptr<MpcParameterUpdaterModule> makeUpdater(const Built& built,
                                                                const std::string& taskFile,
                                                                const std::string& urdfFile,
                                                                const std::string& referenceFile) {
    return createUpdater(built.mpc.get(), taskFile, urdfFile, referenceFile, built.interface->getMpcRobotModel().getStateDim(),
                         built.interface->getEffectiveMpcRobotModel().getInputDim(), built.interface->modelSettings().contactNames,
                         /*referenceManager=*/nullptr, built.interface->getBasisInputsCostTransformConfig());
  }

  /**
   * The first of `candidates` the problem's cost collection carries, or an empty string. The shipped Atlas lists
   * state_quadratic_cost and input_quadratic_cost, so Q and R live in two terms there rather than in one
   * stateInputQuadraticCost; the tests look the term up instead of assuming one layout, and ASSERT that one was found.
   */
  static std::string firstCostTerm(const OptimalControlProblem& ocp, const std::vector<std::string>& candidates) {
    for (const std::string& name : candidates) {
      size_t index = 0;
      if (ocp.costPtr->getTermIndex(name, index)) {
        return name;
      }
    }
    return std::string();
  }
  /** The term that carries the input weight R. */
  static std::string inputCostTerm(const OptimalControlProblem& ocp) {
    return firstCostTerm(ocp, {"stateInputQuadraticCost", "inputQuadraticCost"});
  }
  /** The term that carries the state weight Q. */
  static std::string stateCostTerm(const OptimalControlProblem& ocp) {
    return firstCostTerm(ocp, {"stateInputQuadraticCost", "stateQuadraticCost"});
  }
  /** The (Q, R) gains of a quadratic cost term. */
  static std::pair<matrix_t, matrix_t> gains(const OptimalControlProblem& ocp, const std::string& term) {
    matrix_t Q;
    matrix_t R;
    matrix_t P;
    ocp.costPtr->get<QuadraticStateInputCost>(term).getGains(Q, R, P);
    return {Q, R};
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
  const absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> created =
      MpcParameterUpdaterModule::Create(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_,
                                        /*referenceManager=*/nullptr, basisCostTransform_);
  ASSERT_TRUE(created.ok()) << created.status();
  EXPECT_NE(*created, nullptr);
}

/******************************************************************************************************/
// Test: Verify that quadratic cost weights (Q/R) are updated in-place after file change.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, QuadraticCostWeightsUpdatedInPlace) {
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);

  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // Read the original Q from whichever term carries it (stateQuadraticCost on the shipped Atlas).
  const std::string qTerm = stateCostTerm(sqp->getOcpDefinitions().front());
  ASSERT_FALSE(qTerm.empty()) << "the problem carries no quadratic state cost";
  const matrix_t origQ = gains(sqp->getOcpDefinitions().front(), qTerm).first;

  // Mutate task.yaml: multiply Q scaling by 2
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    // Find "Q:" section and modify scaling
    const std::string::size_type pos = content.find("Q:");
    ASSERT_NE(pos, std::string::npos) << "Could not find Q: section in task.yaml";
    const std::string::size_type scalingPos = content.find("scaling:", pos);
    ASSERT_NE(scalingPos, std::string::npos) << "Could not find Q scaling in task.yaml";

    // Extract original scaling value
    const std::string::size_type valueStart = scalingPos + std::string("scaling:").size();
    const std::string::size_type lineEnd = content.find('\n', valueStart);
    std::string origValue = content.substr(valueStart, lineEnd - valueStart);
    double origScaling = std::stod(origValue);

    // Replace with doubled value
    const std::string newValue = absl::StrCat(" ", origScaling * 2.0);
    content.replace(valueStart, lineEnd - valueStart, newValue);

    std::ofstream out(tmpTaskFile_);
    out << content;
  }

  // Wait briefly for filesystem timestamp granularity
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Force the updater to run by simulating many preSolverRun calls past the counter threshold
  vector_t dummyState = vector_t::Zero(stateDim_);
  for (size_t i = 0; i < 101; ++i) {
    updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
  }

  // Read updated Q value
  const matrix_t newQ = gains(sqp->getOcpDefinitions().front(), qTerm).first;

  // The Q matrix should have changed (approximately doubled scaling)
  ASSERT_GT(origQ.norm(), 0.0) << "the shipped Q is zero, so doubling its scaling would show nothing";
  EXPECT_GT(newQ.norm(), 0.0) << "Updated Q should be non-zero";
  // The ratio should be approximately 2.0 (since we doubled the scaling)
  const scalar_t ratio = newQ.norm() / origQ.norm();
  EXPECT_NEAR(ratio, 2.0, 0.1) << "Q matrix should have approximately doubled";
}

/******************************************************************************************************/
// Test: Verify that no crash occurs when cost terms are missing from the collection.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, NoCrashOnMissingCostTerms) {
  // Use a minimal task file that might not have all cost terms configured
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);

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
      updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
    }
  });
}

/******************************************************************************************************/
// Test: Verify that repeated updates don't crash or cause memory issues.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, RepeatedUpdatesAreStable) {
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);

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
        updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
      }
    });
  }
}

/******************************************************************************************************/
// Test: Q_final and terminalCostScaling reach a running quadratic terminal cost, whatever terminal cost the file lists
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, TerminalCostScalingApplied) {
  // The shipped Atlas ends its horizon on the DCM cost and has no quadratic terminal cost to scale, so the problem is
  // built from a variant that ends on Q_final. Which terminal cost the problem carries is structural: a reload that also
  // switches the costs list back to dcm_terminal_cost must still reach the Q_final the running problem has, and says
  // that the switch waits for a restart. Deciding it from the file, as the updater used to, froze Q_final here.
  const std::string variantFile = absl::StrCat(testing::TempDir(), "/test_task_terminal_scaling.yaml");
  const std::string quadratic = acom_test::withQuadraticTerminalCost(readTmpTaskFile());
  {
    std::ofstream out(variantFile);
    out << quadratic;
  }
  const Built built = build(variantFile);
  ASSERT_NE(built.interface, nullptr);
  size_t index = 0;
  ASSERT_TRUE(built.solver().getOcpDefinitions().front().finalCostPtr->getTermIndex("terminalCost", index));
  ASSERT_FALSE(built.solver().getOcpDefinitions().front().finalCostPtr->getTermIndex(DcmTerminalCost::kTermName, index));

  constexpr scalar_t kScaling = 100.0;
  constexpr scalar_t kJointWeight = 321.0;
  std::string edited = acom_test::withDiagonalEntry(quadratic, "Q_final", /*i=*/12, kJointWeight);
  edited = std::regex_replace(edited, std::regex("\nterminalCostScaling: [^\n]*"), absl::StrCat("\nterminalCostScaling: ", kScaling));
  edited = acom_test::withDcmTerminalCost(edited);
  {
    std::ofstream out(variantFile);
    out << edited;
  }
  scalar_t qFinalScaling = 1.0;
  loadData::loadCppDataType(variantFile, "Q_final.scaling", qFinalScaling);
  const scalar_t expected = kScaling * qFinalScaling * kJointWeight;
  {
    matrix_t QFinal;
    built.solver().getOcpDefinitions().front().finalCostPtr->get<QuadraticStateCost>("terminalCost").getGains(QFinal);
    ASSERT_GT(std::abs(QFinal(12, 12) - expected), 1.0) << "the launch value must differ from the reloaded one";
  }

  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built, variantFile, urdfFile_, referenceFile_);
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("dcm_terminal_cost"), testing::HasSubstr("structural"))))
      .Times(testing::AtLeast(1));
  log.StartCapturingLogs();
  touchAndRun(variantFile, *updater, *built.interface);
  log.StopCapturingLogs();
  for (OptimalControlProblem& ocp : built.solver().getOcpDefinitions()) {
    matrix_t QFinal;
    ocp.finalCostPtr->get<QuadraticStateCost>("terminalCost").getGains(QFinal);
    EXPECT_NEAR(QFinal(12, 12), expected, 1e-9) << "the running Q_final did not follow the reload";
  }
  std::remove(variantFile.c_str());
}

/******************************************************************************************************/
// Test: a reload that switches the terminal cost is reported, and the running DCM cost still follows its block
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, AReloadThatSwitchesTheTerminalCostIsReportedAndTheRunningDcmCostFollowsItsBlock) {
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  size_t index = 0;
  ASSERT_TRUE(sqp->getOcpDefinitions().front().finalCostPtr->getTermIndex(DcmTerminalCost::kTermName, index))
      << "the shipped Atlas ends its horizon on the DCM cost";
  const std::string shipped = readTmpTaskFile();
  const std::string::size_type block = shipped.find("\ndcm_terminal_cost:\n");
  ASSERT_NE(block, std::string::npos);
  std::string edited = shipped.substr(0, block) + std::regex_replace(shipped.substr(block), std::regex("\n  weight_x: [^\n]*"),
                                                                     "\n  weight_x: 123", std::regex_constants::format_first_only);
  edited = acom_test::withQuadraticTerminalCost(edited);
  writeTmpTaskFile(edited);

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("dcm_terminal_cost"), testing::HasSubstr("structural"))))
      .Times(testing::AtLeast(1));
  log.StartCapturingLogs();
  touchTaskFileAndRunUpdater(*updater);
  log.StopCapturingLogs();
  for (OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
    EXPECT_FALSE(ocp.finalCostPtr->getTermIndex("terminalCost", index)) << "a reload never adds a term";
    EXPECT_NEAR(ocp.finalCostPtr->get<DcmTerminalCost>(DcmTerminalCost::kTermName).getConfig().weights(0), 123.0, 1e-12)
        << "the running DCM cost must keep following its block";
  }
}

/******************************************************************************************************/
// Test: Verify joint limits barrier parameters are updated.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, JointLimitsBarrierUpdated) {
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);

  SqpSolver* sqp = getSqpSolver();
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

    const std::string::size_type pos = content.find("jointLimits:");
    ASSERT_NE(pos, std::string::npos);
    const std::string::size_type muPos = content.find("mu:", pos);
    ASSERT_NE(muPos, std::string::npos);
    const std::string::size_type valueStart = muPos + std::string("mu:").size();
    const std::string::size_type lineEnd = content.find('\n', valueStart);
    content.replace(valueStart, lineEnd - valueStart, " 999.0");

    std::ofstream out(tmpTaskFile_);
    out << content;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Trigger update
  vector_t dummyState = vector_t::Zero(stateDim_);
  for (size_t i = 0; i < 101; ++i) {
    updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
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
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);

  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // Read original Q and R, from whichever terms carry them.
  const std::string qTerm = stateCostTerm(sqp->getOcpDefinitions().front());
  const std::string rTerm = inputCostTerm(sqp->getOcpDefinitions().front());
  ASSERT_FALSE(qTerm.empty() || rTerm.empty()) << "the problem carries no quadratic state or input cost";
  const matrix_t origQ = gains(sqp->getOcpDefinitions().front(), qTerm).first;
  const matrix_t origR = gains(sqp->getOcpDefinitions().front(), rTerm).second;

  // Run without touching the file, past the ~1 Hz file check. Reloading the unchanged file would write back the very Q
  // and R it holds, so the gains alone cannot show a spurious reload; the updater's own log line can.
  const std::string reloadLine = "Applying in-place parameter updates";
  {
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr(reloadLine))).Times(0);
    log.StartCapturingLogs();
    const vector_t dummyState = vector_t::Zero(stateDim_);
    for (size_t i = 0; i < 101; ++i) {
      updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
    }
    log.StopCapturingLogs();
  }

  // Q and R should be unchanged
  const matrix_t sameQ = gains(sqp->getOcpDefinitions().front(), qTerm).first;
  const matrix_t sameR = gains(sqp->getOcpDefinitions().front(), rTerm).second;
  EXPECT_TRUE(sameQ == origQ) << "Q should not change when file is untouched";
  EXPECT_TRUE(sameR == origR) << "R should not change when file is untouched";

  // Positive control: the same loop after touching the file does reload, and logs the line watched above.
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr(reloadLine))).Times(testing::AtLeast(1));
  log.StartCapturingLogs();
  touchTaskFileAndRunUpdater(*updater);
  log.StopCapturingLogs();
}

// Test: Verify that SQP solver settings are updated at runtime.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, SqpSettingsUpdated) {
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);

  SqpSolver* sqp = getSqpSolver();
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
    updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
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
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // Collect the twist constraints behind every foot's zeroVelocity term, hard or soft.
  const std::function<std::vector<EndEffectorKinematicsTwistConstraint*>()> twistConstraints = [&]() {
    std::vector<EndEffectorKinematicsTwistConstraint*> found;
    for (OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
      for (const std::string& footName : contactNames_) {
        const std::string name = footName + "_zeroVelocity";
        size_t index = 0;
        if (ocp.equalityConstraintPtr->getTermIndex(name, index)) {
          if (ZeroVelocityConstraintCppAd* con = dynamic_cast<ZeroVelocityConstraintCppAd*>(&ocp.equalityConstraintPtr->get(name)))
            found.push_back(&con->getTwistConstraint());
        }
        if (ocp.softConstraintPtr->getTermIndex(name, index)) {
          if (StateInputSoftConstraint* soft = dynamic_cast<StateInputSoftConstraint*>(&ocp.softConstraintPtr->get(name))) {
            if (ZeroVelocityConstraintCppAd* con = dynamic_cast<ZeroVelocityConstraintCppAd*>(soft->getConstraintPtr().get()))
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
  for (const EndEffectorKinematicsTwistConstraint* twist : before) {
    EXPECT_EQ(twist->getConstrainYawRateAboutNormal(), shipped) << "every OCP copy starts from the shipped task file";
  }

  // Flip the flag in the task file and reload.
  const bool flipped = !shipped;
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const std::string key = "constrainYawRateAboutContactNormal:";
    const std::string::size_type pos = content.find(key);
    ASSERT_NE(pos, std::string::npos) << "the task file must carry the key so that the reload can set it";
    const std::string::size_type lineEnd = content.find('\n', pos);
    content.replace(pos, lineEnd - pos, key + (flipped ? " true" : " false"));
    std::ofstream out(tmpTaskFile_);
    out << content;
  }
  touchTaskFileAndRunUpdater(*updater);

  const std::vector<EndEffectorKinematicsTwistConstraint*> after = twistConstraints();
  ASSERT_EQ(after.size(), before.size());
  for (const EndEffectorKinematicsTwistConstraint* twist : after) {
    EXPECT_EQ(twist->getConstrainYawRateAboutNormal(), flipped) << "the reload must apply the flag to every OCP copy";
  }

  // The loader of the model settings reads the same key (the interface uses it at construction).
  const ModelSettings reloaded(tmpTaskFile_, urdfFile_, "centroidal_mpc_", /*verbose=*/false);  // the name the interface uses
  EXPECT_EQ(reloaded.footConstraintConfig.constrainYawRateAboutContactNormal, flipped);
  EXPECT_EQ(interface_->modelSettings().footConstraintConfig.constrainYawRateAboutContactNormal, shipped)
      << "the interface still holds the settings it was built with";
}

TEST_F(MpcParameterUpdaterModuleTest, SoftConstraintWeightUpdated) {
  // softConstraintWeight only reaches a penalty when zero_velocity is a SOFT constraint. The shipped Atlas lists it
  // under hard_constraints, where there is no penalty to scale, so the problem is built from a variant that lists it
  // under soft_constraints.
  const std::string variantFile = absl::StrCat(testing::TempDir(), "/test_task_soft_zero_velocity.yaml");
  std::string soft = acom_test::replacedOnce(
      readTmpTaskFile(), "\n  - zero_velocity                     # remove for the contact-implicit formulation\n", "\n");
  soft = acom_test::replacedOnce(soft, "\nsoft_constraints:\n", "\nsoft_constraints:\n  - zero_velocity\n");
  {
    std::ofstream out(variantFile);
    out << soft;
  }
  const Built built = build(variantFile);
  ASSERT_NE(built.interface, nullptr);

  // The penalty's weight, read off its value at a unit residual: 0.5 * weight for the quadratic penalty.
  const std::function<scalar_t(OptimalControlProblem&, const std::string&)> installedWeight = [](OptimalControlProblem& ocp,
                                                                                                 const std::string& footName) {
    StateInputSoftConstraint& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(footName, "_zeroVelocity"));
    return 2.0 * softCon.getPenalty().getValue(/*t=*/0.0, vector_t::Ones(1));
  };
  scalar_t shippedWeight = 0.0;
  loadData::loadCppDataType(variantFile, "model_settings.foot_constraint.softConstraintWeight", shippedWeight);
  constexpr scalar_t kWeight = 12345.0;
  for (const std::string& footName : contactNames_) {
    ASSERT_NEAR(installedWeight(built.solver().getOcpDefinitions().front(), footName), shippedWeight, 1e-9)
        << "the variant does not carry a soft zero_velocity with the file's weight";
  }

  {
    std::ofstream out(variantFile);
    out << std::regex_replace(soft, std::regex("\n    softConstraintWeight: [^\n]*"),
                              absl::StrCat("\n    softConstraintWeight: ", kWeight));
  }
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built, variantFile, urdfFile_, referenceFile_);
  ASSERT_NE(updater, nullptr);
  touchAndRun(variantFile, *updater, *built.interface);
  for (OptimalControlProblem& ocp : built.solver().getOcpDefinitions()) {
    for (const std::string& footName : contactNames_) {
      EXPECT_NEAR(installedWeight(ocp, footName), kWeight, 1e-9) << footName << ": the soft zero_velocity weight did not follow the reload";
    }
  }
  std::remove(variantFile.c_str());
}

/******************************************************************************************************/
// Test: a reload of the shipped file reports no failed update: a term the problem does not carry is not a failure
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, AShippedReloadReportsNoFailedUpdate) {
  // The shipped Atlas lists zero_velocity as a hard constraint and carries model_settings.foot_constraint's
  // softConstraintWeight, which only a soft zero_velocity reads. The soft lookup used to log a "Failed to update" per
  // foot per worker on every reload, and so did the foot-gain lookups of whichever of the hard and soft terms is absent.
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr("Failed to update"))).Times(0);
  // Positive control: the reload ran to the end.
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr("Successfully applied in-place parameter updates")))
      .Times(testing::AtLeast(1));
  log.StartCapturingLogs();
  touchTaskFileAndRunUpdater(*updater);
  log.StopCapturingLogs();
}

/******************************************************************************************************/
// Test: a value that does not parse, outside the cost matrices, is refused by its key and the rest of the reload applies
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, AnUnparsableSettingIsRefusedByItsKeyAndTheRestOfTheReloadApplies) {
  // These were read with loadPtreeValue, which throws on a value it cannot parse, and a file-watch reload calls the
  // updater from preSolverRun without a handler: one half-typed number in a saved task.yaml threw out of the solver.
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  const std::string qTerm = stateCostTerm(sqp->getOcpDefinitions().front());
  ASSERT_FALSE(qTerm.empty());
  const size_t runningIterations = sqp->getSettings().sqpIteration;
  const scalar_t runningPositionGain = interface_->modelSettings().footConstraintConfig.positionErrorGain_z;

  constexpr scalar_t kJointWeight = 654.0;
  std::string content = acom_test::withDiagonalEntry(readTmpTaskFile(), "Q", /*i=*/12, kJointWeight);
  content = acom_test::replacedOnce(content, "\n    softConstraintWeight: 250", "\n    softConstraintWeight: heavy");
  content = std::regex_replace(content, std::regex("\n    positionErrorGain_z: [^\n]*"), "\n    positionErrorGain_z: stiff",
                               std::regex_constants::format_first_only);
  content = std::regex_replace(content, std::regex("\n  sqpIteration: [^\n]*"), "\n  sqpIteration: many",
                               std::regex_constants::format_first_only);
  writeTmpTaskFile(content);
  scalar_t qScaling = 1.0;
  loadData::loadCppDataType(tmpTaskFile_, "Q.scaling", qScaling);

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  for (const char* key : {"model_settings.foot_constraint.softConstraintWeight", "model_settings.foot_constraint.positionErrorGain_z",
                          "multiple_shooting.sqpIteration"}) {
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr(key))).Times(testing::AtLeast(1));
  }
  log.StartCapturingLogs();
  touchTaskFileAndRunUpdater(*updater);
  log.StopCapturingLogs();

  EXPECT_EQ(sqp->getSettings().sqpIteration, runningIterations) << "a refused multiple_shooting block changed the solver";
  for (OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
    EXPECT_NEAR(gains(ocp, qTerm).first(12, 12), qScaling * kJointWeight, 1e-9) << "the rest of the reload was not applied";
    const HumanoidPreComputation* preComputation = dynamic_cast<const HumanoidPreComputation*>(ocp.preComputationPtr.get());
    ASSERT_NE(preComputation, nullptr);
    EXPECT_EQ(preComputation->getNormalVelocityPositionErrorGain(), runningPositionGain)
        << "a refused foot-constraint block changed the running gain";
  }
}

/******************************************************************************************************/
// Test: a cost weight that does not parse is refused by its key, the running one is kept, and nothing is thrown
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, AnUnparsableCostWeightIsRefusedByItsKeyAndTheRestOfTheReloadApplies) {
  // The weight matrices were read with a `get` that silently put a default in place of a value it could not parse (a
  // mistyped Q entry became 0 on the running problem), the sections through loaders whose exceptions an empty catch
  // swallowed without a word, and nothing guarded preSolverRun against the rest. Each is now refused by the key the
  // operator has to fix, and the running value is kept.
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  const std::string qTerm = stateCostTerm(sqp->getOcpDefinitions().front());
  ASSERT_FALSE(qTerm.empty());
  const scalar_t runningJointWeight = gains(sqp->getOcpDefinitions().front(), qTerm).first(12, 12);
  const size_t runningIterations = sqp->getSettings().sqpIteration;
  const size_t reloadedIterations = runningIterations + 2;

  std::string content = std::regex_replace(readTmpTaskFile(), std::regex("\n  sqpIteration: [^\n]*"),
                                           absl::StrCat("\n  sqpIteration: ", reloadedIterations), std::regex_constants::format_first_only);
  const std::string::size_type qBlock = content.find("\nQ:\n");
  ASSERT_NE(qBlock, std::string::npos);
  const std::string::size_type qEntry = content.find("\"(12,12)\":", qBlock);
  ASSERT_NE(qEntry, std::string::npos);
  content.replace(qEntry, content.find('\n', qEntry) - qEntry, "\"(12,12)\": heavy");
  content = acom_test::replacedOnce(content, "\n  icpErrorWeight: 0\n", "\n  icpErrorWeight: none\n");
  content = std::regex_replace(content, std::regex("\nterminalCostScaling: [^\n]*"), "\nterminalCostScaling: big",
                               std::regex_constants::format_first_only);
  writeTmpTaskFile(content);

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  for (const char* key : {"Q.(12,12) is 'heavy'", "icp_cost_weights.icpErrorWeight is 'none'", "terminalCostScaling is 'big'"}) {
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr(key))).Times(testing::AtLeast(1));
  }
  log.StartCapturingLogs();
  EXPECT_NO_THROW(touchTaskFileAndRunUpdater(*updater));
  log.StopCapturingLogs();

  // Positive control: the rest of the file was applied.
  EXPECT_EQ(sqp->getSettings().sqpIteration, reloadedIterations) << "the rest of the reload was not applied";
  for (OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
    EXPECT_EQ(gains(ocp, qTerm).first(12, 12), runningJointWeight) << "a refused Q changed the running state cost";
  }
}

/******************************************************************************************************/
// Test: the contact schedule source is structural; a reloaded file that changes it, or carries the retired boolean, is
// reported, not applied
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, AReloadedContactScheduleSourceIsReportedAsStructural) {
  ASSERT_FALSE(interface_->usesContactPlanning());
  const std::string shipped = readTmpTaskFile();
  struct Case {
    std::string name;
    std::string content;
    std::string phrase;
  };
  const std::vector<Case> cases = {
      {"switched to the planner",
       acom_test::replacedOnce(shipped, "\ncontactScheduleSource: gait_schedule\n", "\ncontactScheduleSource: contact_planner\n"),
       "contactScheduleSource is contact_planner"},
      {"retired contact planning boolean", absl::StrCat("useContactPlanning: true\n", shipped), "useContactPlanning"},
      {"retired DCM boolean", absl::StrCat("useDcmTerminalCost: true\n", shipped), "useDcmTerminalCost"},
  };
  for (const Case& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    writeTmpTaskFile(testCase.content);
    std::unique_ptr<MpcParameterUpdaterModule> updater =
        createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_,
                      /*referenceManager=*/nullptr, basisCostTransform_);
    ASSERT_NE(updater, nullptr);
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr(testCase.phrase))).Times(testing::AtLeast(1));
    log.StartCapturingLogs();
    touchTaskFileAndRunUpdater(*updater);
    log.StopCapturingLogs();
  }
  // Positive control: the shipped file itself is not reported.
  writeTmpTaskFile(shipped);
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr(std::string(kContactScheduleSourceKey)))).Times(0);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr("structural"))).Times(0);
  log.StartCapturingLogs();
  touchTaskFileAndRunUpdater(*updater);
  log.StopCapturingLogs();
}

/******************************************************************************************************/
// Test: Verify that foot constraint gains (Ax/Av matrices) are updated.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, FootConstraintGainsUpdated) {
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);

  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // The gains live on the zero-velocity twist constraint, which the contact-implicit formulation replaces with the
  // relaxed complementarity terms (humanoid_nmpc/docs/contact_implicit_mpc/README.md). With that formulation selected
  // there is no such constraint to update, and the hot reload of these gains is simply not applicable.
  const bool hasZeroVelocityConstraint = [&]() {
    for (OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
      for (const std::string& footName : contactNames_) {
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

    const std::string::size_type pos = content.find("linearVelocityErrorGain_xy:");
    ASSERT_NE(pos, std::string::npos) << "the shipped task file carries linearVelocityErrorGain_xy";
    const std::string::size_type valueStart = pos + std::string("linearVelocityErrorGain_xy:").size();
    const std::string::size_type lineEnd = content.find('\n', valueStart);
    content.replace(valueStart, lineEnd - valueStart, " 99.0");

    std::ofstream out(tmpTaskFile_);
    out << content;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Trigger update
  vector_t dummyState = vector_t::Zero(stateDim_);
  for (size_t i = 0; i < 101; ++i) {
    updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
  }

  // Check that at least one constraint's Av(0,0) was updated
  bool foundUpdated = false;
  for (OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
    for (const std::string& footName : contactNames_) {
      // Check hard constraint path
      try {
        ZeroVelocityConstraintCppAd& con = ocp.equalityConstraintPtr->get<ZeroVelocityConstraintCppAd>(footName + "_zeroVelocity");
        const EndEffectorKinematicsTwistConstraint::Config& cfg = con.getTwistConstraint().getConfig();
        if (std::abs(cfg.Av(0, 0) - 99.0) < 1e-3) {
          foundUpdated = true;
        }
      } catch (...) {
      }
      // Check soft constraint path
      try {
        StateInputSoftConstraint& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_zeroVelocity");
        ZeroVelocityConstraintCppAd* zeroVelCon = dynamic_cast<ZeroVelocityConstraintCppAd*>(softCon.getConstraintPtr().get());
        if (zeroVelCon != nullptr) {
          const EndEffectorKinematicsTwistConstraint::Config& cfg = zeroVelCon->getTwistConstraint().getConfig();
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

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, tmpReferenceFile, stateDim_, inputDim_, contactNames_,
                    /*referenceManager=*/nullptr, basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  std::vector<std::string> reloadedWith;
  updater->addReferenceFileReloader([&reloadedWith](const std::string& file) { reloadedWith.push_back(file); });

  const vector_t dummyState = vector_t::Zero(stateDim_);
  // The file is polled once every hundred solves; nothing has changed yet, so nothing is reloaded.
  for (size_t i = 0; i < 101; ++i) {
    updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
  }
  EXPECT_TRUE(reloadedWith.empty()) << "an untouched reference file must not trigger a reload";

  {
    std::ofstream out(tmpReferenceFile, std::ios::app);
    out << "\n# touched by the test\n";
  }
  std::filesystem::last_write_time(tmpReferenceFile, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(1));
  for (size_t i = 0; i < 101; ++i) {
    updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
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
  const std::function<void(scalar_t, scalar_t)> writeWith = [&tmpReferenceFile, this](scalar_t maxDeltaPelvisHeight,
                                                                                      scalar_t defaultBaseHeight) {
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
  const TargetTrajectories before = calculator.commandedPositionToTargetTrajectories(deepCrouch, /*initTime=*/0.0, state);
  ASSERT_FALSE(before.stateTrajectory.empty());
  const scalar_t heightBefore = interface_->getEffectiveMpcRobotModel().getBasePosition(before.stateTrajectory.back())(2);
  EXPECT_NEAR(heightBefore, 0.90 - 0.10, 1e-6);

  writeWith(0.25, 0.80);
  calculator.reloadCommandLimits(tmpReferenceFile);
  const TargetTrajectories after = calculator.commandedPositionToTargetTrajectories(deepCrouch, /*initTime=*/0.0, state);
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
  const BasisInputsCostTransformConfig config = makeSyntheticBasisCostTransform(/*numBasisPerFoot=*/8, /*lambdaRegularization=*/0.1);
  ASSERT_NE(config.basisInputDim(), interface_->getWrenchInputDim());

  // Create() returns the mismatch as an InvalidArgument naming inputDim, where the constructor used to throw.
  for (const size_t wrongInputDim : {interface_->getWrenchInputDim(), config.basisInputDim() + 1}) {
    const absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> created = MpcParameterUpdaterModule::Create(
        mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, wrongInputDim, contactNames_, /*referenceManager=*/nullptr, config);
    ASSERT_FALSE(created.ok()) << "inputDim " << wrongInputDim << " was accepted";
    EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
    EXPECT_NE(created.status().message().find("inputDim"), absl::string_view::npos) << created.status();
  }
  // Positive control: the basis-space dimension itself is accepted.
  const absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> created =
      MpcParameterUpdaterModule::Create(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, config.basisInputDim(),
                                        contactNames_, /*referenceManager=*/nullptr, config);
  EXPECT_TRUE(created.ok()) << created.status();
}

/******************************************************************************************************/
// Test: With a synthetic basis-space cost transform, the R written into the OCP equals Mᵀ R_wrench M with the
// λ regularization on the leading diagonal only — i.e. the yaml R is read in wrench space and transformed, with the
// regularization weight the reloaded file carries (contacts.basisScalingRegularization), not the one the updater was
// constructed with.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, BasisCostTransformAppliedToInputCost) {
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  const std::string rTerm = inputCostTerm(sqp->getOcpDefinitions().front());
  ASSERT_FALSE(rTerm.empty()) << "the problem carries no quadratic input cost";

  constexpr scalar_t constructedRegularization = 0.25;
  const BasisInputsCostTransformConfig config = makeSyntheticBasisCostTransform(/*numBasisPerFoot=*/8, constructedRegularization);
  const size_t basisInputDim = config.basisInputDim();
  // The weight the file carries is the one a reload applies; it differs from the constructed one, so the check below
  // tells the two apart.
  scalar_t lambdaRegularization = -1.0;
  loadData::loadCppDataType(tmpTaskFile_, std::string(kBasisScalingRegularizationKey), lambdaRegularization);
  ASSERT_GT(lambdaRegularization, 0.0);
  ASSERT_NE(lambdaRegularization, constructedRegularization);
  BasisInputsCostTransformConfig reloaded = config;
  reloaded.lambdaRegularization = lambdaRegularization;

  std::unique_ptr<MpcParameterUpdaterModule> updater = createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_,
                                                                     basisInputDim, contactNames_, /*referenceManager=*/nullptr, config);
  ASSERT_NE(updater, nullptr);
  touchTaskFileAndRunUpdater(*updater);

  const matrix_t R_wrench = loadWrenchSpaceR();
  ASSERT_GT(R_wrench.norm(), 0.0) << "task.yaml R should be non-zero";
  const matrix_t expectedR = transformWrenchInputCostToBasisSpace(R_wrench, reloaded);

  for (const OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
    const std::pair<matrix_t, matrix_t> QR = gains(ocp, rTerm);
    const matrix_t& R = QR.second;
    if (rTerm == "inputQuadraticCost") {
      EXPECT_TRUE(QR.first.isZero(0.0)) << "the input cost must carry no state weight";
    }
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
  // The shipped Atlas runs basis-vector contact inputs; this test is about exactly that configuration, so it asserts it
  // rather than skipping without it.
  ASSERT_TRUE(interface_->usesContactBasisVectorInputs()) << "the shipped Atlas selects contactInputParameterization: basis_vectors";
  ASSERT_TRUE(basisCostTransform_.has_value());
  EXPECT_EQ(inputDim_, basisCostTransform_->basisInputDim());
  EXPECT_NE(inputDim_, interface_->getWrenchInputDim()) << "basis-space and wrench-space input dimensions should differ";

  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  const std::string rTerm = inputCostTerm(sqp->getOcpDefinitions().front());
  ASSERT_FALSE(rTerm.empty()) << "the problem carries no quadratic input cost";
  const matrix_t origR = gains(sqp->getOcpDefinitions().front(), rTerm).second;
  ASSERT_EQ(static_cast<size_t>(origR.rows()), inputDim_) << "OCP factory R should already be in basis space";

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  touchTaskFileAndRunUpdater(*updater);

  const matrix_t newR = gains(sqp->getOcpDefinitions().front(), rTerm).second;
  ASSERT_EQ(static_cast<size_t>(newR.rows()), inputDim_);
  ASSERT_EQ(static_cast<size_t>(newR.cols()), inputDim_);

  EXPECT_TRUE(newR.isApprox(origR, 1e-9)) << "Online update must reproduce the OCP factory's basis-space R for an unchanged yaml";
  EXPECT_TRUE(newR.isApprox(transformWrenchInputCostToBasisSpace(loadWrenchSpaceR(), *basisCostTransform_), 1e-9));
}

/******************************************************************************************************/
// Test: ComAndAcomTrackingWeightsUpdated
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, ComAndAcomTrackingWeightsUpdated) {
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // The shipped Atlas problem lists com_and_acom_tracking_cost, so this runs on the configuration the robot runs.
  const OptimalControlProblem& ocp0 = sqp->getOcpDefinitions().front();
  ASSERT_NE(ocp0.stateCostPtr, nullptr);
  ASSERT_EQ(ocp0.stateCostPtr->getTermNameMap().count(std::string(ComAndAcomTrackingCost::kRunningTermName)), 1u);

  // Mutate one entry of Q_com and one of Q_acom in tmpTaskFile_.
  writeTmpTaskFile(acom_test::withDiagonalEntry(acom_test::withDiagonalEntry(readTmpTaskFile(), "Q_com", /*i=*/2, /*value=*/999.0),
                                                "Q_acom", /*i=*/0, /*value=*/777.0));

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  touchTaskFileAndRunUpdater(*updater);

  // Each matrix is premultiplied by its own `scaling` entry, and the two are not the same number in every robot's
  // task file. Read them rather than hard-coding, or this test rots the next time either is retuned.
  scalar_t comScaling = 1.0;
  scalar_t acomScaling = 1.0;
  loadData::loadCppDataType(tmpTaskFile_, "Q_com.scaling", comScaling);
  loadData::loadCppDataType(tmpTaskFile_, "Q_acom.scaling", acomScaling);

  // Every per-thread clone of the problem must see the update, not just the first.
  for (OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
    const ComAndAcomTrackingCost& cost =
        ocp.stateCostPtr->get<ComAndAcomTrackingCost>(std::string(ComAndAcomTrackingCost::kRunningTermName));
    EXPECT_NEAR(cost.getQCom()(2, 2), comScaling * 999.0, 1e-3);
    // Row 0 of Q_acom is yaw, in the centroidal state's ZYX Euler convention.
    EXPECT_NEAR(cost.getQAcom()(0, 0), acomScaling * 777.0, 1e-3);
  }
}

/******************************************************************************************************/
// Tests: the base-pose weights of a reloaded Q and Q_final follow the RUNNING problem (findings A92, A104, A111)
/******************************************************************************************************/
/**
 * On the shipped Atlas problem, which lists com_and_acom_tracking_cost, a reload must zero Q's base-pose block - the
 * ACoM cost regulates the base pose there - and must go on doing so when the reloaded file no longer lists the cost:
 * the cost list is structural, fixed when the problem was built, and the problem still carries the ACoM cost. Atlas
 * ships its base-pose weights at 0, so the file's are made non-zero here, or the zeroing could not be seen. This used to
 * be decided from the file's own `useComAndAcomTracking`, and its test skipped on the shipped configuration.
 */
TEST_F(MpcParameterUpdaterModuleTest, BasePoseWeightsFollowTheRunningProblemNotTheReloadedFile) {
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  ASSERT_EQ(sqp->getOcpDefinitions().front().stateCostPtr->getTermNameMap().count(std::string(ComAndAcomTrackingCost::kRunningTermName)),
            1u)
      << "the shipped Atlas problem carries the CoM + ACoM cost";

  // Non-zero base-pose weights, a changed joint weight to show the reload was applied, and the cost dropped from the list.
  std::string content = acom_test::withBasePoseWeights(readTmpTaskFile(), "Q", /*offset=*/30.0);
  content = acom_test::withDiagonalEntry(content, "Q", /*i=*/12, /*value=*/123.0);
  content = acom_test::replacedOnce(content, acom_test::kAcomCostEntry, "\n");
  writeTmpTaskFile(content);
  scalar_t qScaling = 1.0;
  loadData::loadCppDataType(tmpTaskFile_, "Q.scaling", qScaling);

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr("com_and_acom_tracking_cost is not listed")))
      .Times(testing::AtLeast(1));
  log.StartCapturingLogs();
  touchTaskFileAndRunUpdater(*updater);
  log.StopCapturingLogs();

  for (OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
    matrix_t Q;
    matrix_t R;
    matrix_t P;
    ocp.costPtr->get<QuadraticStateInputCost>("stateQuadraticCost").getGains(Q, R, P);
    EXPECT_NEAR(Q(12, 12), qScaling * 123.0, 1e-9) << "the reload was not applied";
    EXPECT_TRUE(acom_test::basePoseBlock(Q).isZero(0.0)) << "a reload re-introduced base-pose tracking beside the running ACoM cost:\n"
                                                         << acom_test::basePoseBlock(Q);
    EXPECT_EQ(ocp.stateCostPtr->getTermNameMap().count(std::string(ComAndAcomTrackingCost::kRunningTermName)), 1u);
  }
}

/**
 * The reverse: a problem built WITHOUT the ACoM cost keeps its base-pose weights live across a reload of a file that
 * lists it - zeroing them would leave the base pose unregulated, with no cost to take over.
 */
TEST_F(MpcParameterUpdaterModuleTest, BasePoseWeightsStayLiveInAProblemBuiltWithoutTheAcomCost) {
  const std::string variantFile = absl::StrCat(testing::TempDir(), "/test_task_without_acom.yaml");
  {
    std::ofstream out(variantFile);
    out << acom_test::replacedOnce(acom_test::withBasePoseWeights(readTmpTaskFile(), "Q", /*offset=*/30.0), acom_test::kAcomCostEntry,
                                   "\n");
  }
  const Built built = build(variantFile);
  ASSERT_NE(built.interface, nullptr);
  ASSERT_EQ(built.solver().getOcpDefinitions().front().stateCostPtr->getTermNameMap().count(
                std::string(ComAndAcomTrackingCost::kRunningTermName)),
            0u);

  // The file now lists the cost again, with other base-pose weights.
  {
    std::ifstream in(variantFile);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    content = acom_test::withBasePoseWeights(content, "Q", /*offset=*/60.0);
    content =
        acom_test::replacedOnce(content, "\n  - state_quadratic_cost\n", "\n  - state_quadratic_cost\n  - com_and_acom_tracking_cost\n");
    std::ofstream out(variantFile);
    out << content;
  }
  const Eigen::Index stateDim = built.interface->getInitialState().size();
  matrix_t fileQ(stateDim, stateDim);
  loadData::loadEigenMatrix(variantFile, "Q", fileQ);
  ASSERT_GT(acom_test::basePoseBlock(fileQ).minCoeff() + acom_test::basePoseBlock(fileQ).maxCoeff(), 0.0);

  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built, variantFile, urdfFile_, referenceFile_);
  ASSERT_NE(updater, nullptr);
  touchAndRun(variantFile, *updater, *built.interface);
  for (OptimalControlProblem& ocp : built.solver().getOcpDefinitions()) {
    matrix_t Q;
    matrix_t R;
    matrix_t P;
    ocp.costPtr->get<QuadraticStateInputCost>("stateQuadraticCost").getGains(Q, R, P);
    EXPECT_TRUE(acom_test::basePoseBlock(Q).isApprox(acom_test::basePoseBlock(fileQ), 1e-12))
        << "a reload zeroed the base-pose weights of a problem that has no ACoM cost to replace them:\n"
        << acom_test::basePoseBlock(Q);
  }
  std::remove(variantFile.c_str());
}

/**
 * The terminal branch, which the shipped Atlas cannot show because its horizon ends on the DCM cost: with the quadratic
 * terminal cost, a reload keeps Q_final's base-pose block zero, and the terminal CoM + ACoM instance (finding A115)
 * follows a Q_com / Q_acom reload with terminalCostScaling applied, as at start-up.
 */
TEST_F(MpcParameterUpdaterModuleTest, TheQuadraticTerminalCostAndItsAcomInstanceFollowAReload) {
  const std::string variantFile = absl::StrCat(testing::TempDir(), "/test_task_quadratic_terminal.yaml");
  {
    std::ofstream out(variantFile);
    out << acom_test::withQuadraticTerminalCost(acom_test::withBasePoseWeights(readTmpTaskFile(), "Q_final", /*offset=*/40.0));
  }
  const Built built = build(variantFile);
  ASSERT_NE(built.interface, nullptr);
  scalar_t terminalCostScaling = 0.0;
  loadData::loadCppDataType(variantFile, "terminalCostScaling", terminalCostScaling);
  ASSERT_NE(terminalCostScaling, 1.0);
  {
    // At start-up already.
    matrix_t QFinal;
    built.solver().getOcpDefinitions().front().finalCostPtr->get<QuadraticStateCost>("terminalCost").getGains(QFinal);
    ASSERT_TRUE(acom_test::basePoseBlock(QFinal).isZero(0.0));
  }

  {
    std::ifstream in(variantFile);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    content = acom_test::withBasePoseWeights(content, "Q_final", /*offset=*/70.0);
    content = acom_test::withDiagonalEntry(content, "Q_final", /*i=*/12, /*value=*/321.0);
    content = acom_test::withDiagonalEntry(content, "Q_com", /*i=*/2, /*value=*/999.0);
    std::ofstream out(variantFile);
    out << content;
  }
  scalar_t qFinalScaling = 1.0;
  scalar_t comScaling = 1.0;
  loadData::loadCppDataType(variantFile, "Q_final.scaling", qFinalScaling);
  loadData::loadCppDataType(variantFile, "Q_com.scaling", comScaling);

  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built, variantFile, urdfFile_, referenceFile_);
  ASSERT_NE(updater, nullptr);
  touchAndRun(variantFile, *updater, *built.interface);
  for (OptimalControlProblem& ocp : built.solver().getOcpDefinitions()) {
    matrix_t QFinal;
    ocp.finalCostPtr->get<QuadraticStateCost>("terminalCost").getGains(QFinal);
    EXPECT_NEAR(QFinal(12, 12), terminalCostScaling * qFinalScaling * 321.0, 1e-9) << "the reload was not applied";
    EXPECT_TRUE(acom_test::basePoseBlock(QFinal).isZero(0.0)) << acom_test::basePoseBlock(QFinal);
    const ComAndAcomTrackingCost& running =
        ocp.stateCostPtr->get<ComAndAcomTrackingCost>(std::string(ComAndAcomTrackingCost::kRunningTermName));
    const ComAndAcomTrackingCost& terminal =
        ocp.finalCostPtr->get<ComAndAcomTrackingCost>(std::string(ComAndAcomTrackingCost::kTerminalTermName));
    EXPECT_NEAR(running.getQCom()(2, 2), comScaling * 999.0, 1e-9);
    EXPECT_NEAR(terminal.getQCom()(2, 2), terminalCostScaling * comScaling * 999.0, 1e-9);
    EXPECT_TRUE(terminal.getQAcom().isApprox(terminalCostScaling * running.getQAcom(), 1e-12));
  }
  std::remove(variantFile.c_str());
}

/******************************************************************************************************/
// Test: BasisNonNegativityBarrierUpdated
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, BasisNonNegativityBarrierUpdated) {
  ASSERT_TRUE(interface_->usesContactBasisVectorInputs()) << "the shipped Atlas selects contactInputParameterization: basis_vectors";
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);

  // Mutate contacts.basisNonNegativityBarrier.mu
  {
    std::ifstream in(tmpTaskFile_);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    const std::string::size_type pos = content.find("basisNonNegativityBarrier:");
    ASSERT_NE(pos, std::string::npos);
    const std::string::size_type pMu = content.find("mu:", pos);
    ASSERT_NE(pMu, std::string::npos);
    const std::string::size_type valueStart = pMu + std::string("mu:").size();
    const std::string::size_type lineEnd = content.find('\n', valueStart);
    content.replace(valueStart, lineEnd - valueStart, " 0.42");

    std::ofstream out(tmpTaskFile_);
    out << content;
  }

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  touchTaskFileAndRunUpdater(*updater);

  for (OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
    for (const std::string& footName : contactNames_) {
      BasisScalingNonNegativityConstraint& con =
          ocp.costPtr->get<BasisScalingNonNegativityConstraint>(absl::StrCat(footName, "_basisNonNegativity"));
      EXPECT_NEAR(con.getBarrierConfig().mu, 0.42, 1e-4);
    }
  }
}

/******************************************************************************************************/
// Test: a barrier value that is not a number is refused by its key, and the rest of the reload still applies
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, AnUnparsableBarrierValueIsRefusedByKeyAndTheRestOfTheReloadApplies) {
  // It used to throw PropertyTreeBadData out of applyParameterUpdates, and on a file-watch reload out of preSolverRun.
  ASSERT_TRUE(interface_->usesContactBasisVectorInputs()) << "the shipped Atlas selects contactInputParameterization: basis_vectors";
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  const std::string qTerm = stateCostTerm(sqp->getOcpDefinitions().front());
  ASSERT_FALSE(qTerm.empty());
  const std::string barrierTerm = absl::StrCat(contactNames_.front(), "_basisNonNegativity");
  const scalar_t runningMu =
      sqp->getOcpDefinitions().front().costPtr->get<BasisScalingNonNegativityConstraint>(barrierTerm).getBarrierConfig().mu;

  // A joint weight of Q, which shows that the rest of the file was applied.
  constexpr scalar_t kJointWeight = 654.0;
  std::string content = acom_test::replacedOnce(readTmpTaskFile(), "  basisNonNegativityBarrier:\n    mu: 0.01\n",
                                                "  basisNonNegativityBarrier:\n    mu: stiff\n");
  content = acom_test::withDiagonalEntry(content, "Q", /*i=*/12, kJointWeight);
  writeTmpTaskFile(content);
  scalar_t qScaling = 1.0;
  loadData::loadCppDataType(tmpTaskFile_, "Q.scaling", qScaling);

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("contacts.basisNonNegativityBarrier.mu"), testing::HasSubstr("'stiff'"))))
      .Times(testing::AtLeast(1));
  log.StartCapturingLogs();
  touchTaskFileAndRunUpdater(*updater);
  log.StopCapturingLogs();

  for (const OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
    EXPECT_EQ(ocp.costPtr->get<BasisScalingNonNegativityConstraint>(barrierTerm).getBarrierConfig().mu, runningMu)
        << "a refused barrier changed the running one";
    EXPECT_NEAR(gains(ocp, qTerm).first(12, 12), qScaling * kJointWeight, 1e-9) << "the rest of the reload was not applied";
  }
}

/******************************************************************************************************/
// Test: the regularization of the basis-space input cost is hot-reloaded by name
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, ABasisRegularizationReloadAppliesTheNamedShape) {
  ASSERT_TRUE(interface_->usesContactBasisVectorInputs()) << "the shipped Atlas selects contactInputParameterization: basis_vectors";
  ASSERT_TRUE(basisCostTransform_.has_value());
  ASSERT_EQ(basisCostTransform_->regularization, kFullDiagonalBasisRegularization) << "the shipped regularization is the full diagonal";
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  const std::string rTerm = inputCostTerm(sqp->getOcpDefinitions().front());
  ASSERT_FALSE(rTerm.empty());
  const matrix_t shippedR = gains(sqp->getOcpDefinitions().front(), rTerm).second;

  writeTmpTaskFile(acom_test::replacedOnce(readTmpTaskFile(), absl::StrCat("basisRegularization: ", kFullDiagonalBasisRegularization),
                                           absl::StrCat("basisRegularization: ", kNullSpaceBasisRegularization)));
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  touchTaskFileAndRunUpdater(*updater);

  BasisInputsCostTransformConfig nullSpace = *basisCostTransform_;
  nullSpace.regularization = std::string(kNullSpaceBasisRegularization);
  const matrix_t expectedR = transformWrenchInputCostToBasisSpace(loadWrenchSpaceR(), nullSpace);
  // Positive control: the two shapes give different input costs, so equality below is the reload's doing.
  ASSERT_GT((expectedR - shippedR).cwiseAbs().maxCoeff(), 1e-9);
  for (const OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
    EXPECT_TRUE(gains(ocp, rTerm).second.isApprox(expectedR, 1e-9)) << "the reload did not apply basisRegularization: null_space";
  }
}

/******************************************************************************************************/
// Test: a basis-space R the QP cannot solve with is refused, naming its key, and the running R is kept
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, ARejectedBasisRegularizationKeepsTheRunningInputCost) {
  ASSERT_TRUE(interface_->usesContactBasisVectorInputs()) << "the shipped Atlas selects contactInputParameterization: basis_vectors";
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  const std::string rTerm = inputCostTerm(sqp->getOcpDefinitions().front());
  const std::string qTerm = stateCostTerm(sqp->getOcpDefinitions().front());
  ASSERT_FALSE(rTerm.empty() || qTerm.empty());
  const matrix_t shippedR = gains(sqp->getOcpDefinitions().front(), rTerm).second;
  const std::string shipped = readTmpTaskFile();

  struct Case {
    std::string name;
    std::string from;
    std::string to;
    std::string key;
  };
  const std::vector<Case> cases = {
      {"unknown shape", "basisRegularization: full_diagonal", "basisRegularization: no_such_shape", std::string(kBasisRegularizationKey)},
      {"negative weight", "basisScalingRegularization: 0.0001", "basisScalingRegularization: -1.0",
       std::string(kBasisScalingRegularizationKey)},
      // A zero weight passes the range check but leaves the λ block singular under full_diagonal's M^T R M.
      {"zero weight", "basisScalingRegularization: 0.0001", "basisScalingRegularization: 0.0", std::string(kBasisScalingRegularizationKey)},
      // Not a number: it used to throw out of the Q/R parse, which dropped the WHOLE reload with a message naming no key.
      {"unparsable weight", "basisScalingRegularization: 0.0001", "basisScalingRegularization: small",
       std::string(kBasisScalingRegularizationKey)},
  };
  scalar_t jointWeight = 321.0;
  for (const Case& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    // The rest of the file still applies: a joint weight of Q, different in every case, shows this reload ran.
    jointWeight += 1.0;
    std::string content = acom_test::replacedOnce(shipped, testCase.from, testCase.to);
    content = acom_test::withDiagonalEntry(content, "Q", /*i=*/12, jointWeight);
    writeTmpTaskFile(content);
    scalar_t qScaling = 1.0;
    loadData::loadCppDataType(tmpTaskFile_, "Q.scaling", qScaling);

    std::unique_ptr<MpcParameterUpdaterModule> updater =
        createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_,
                      /*referenceManager=*/nullptr, basisCostTransform_);
    ASSERT_NE(updater, nullptr);
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                         testing::AllOf(testing::HasSubstr("R was not applied"), testing::HasSubstr(testCase.key))))
        .Times(testing::AtLeast(1));
    log.StartCapturingLogs();
    touchTaskFileAndRunUpdater(*updater);
    log.StopCapturingLogs();

    for (const OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
      EXPECT_TRUE(gains(ocp, rTerm).second == shippedR) << "a refused reload changed the running R";
      EXPECT_NEAR(gains(ocp, qTerm).first(12, 12), qScaling * jointWeight, 1e-9) << "the rest of the reload was not applied";
    }
  }
}

/******************************************************************************************************/
// Test: a reload of the shipped basis-mode file reports no failed update of a cone term the problem does not carry
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, AReloadReportsNoFailureForConeTermsTheProblemDoesNotCarry) {
  // Under basis-vector inputs contact_wrench_cone builds nothing, and friction_force_cone / contact_moment_xy are not
  // listed on Atlas, yet all three sections are in the file. Their absence is the normal case and used to log one
  // "Failed to update" warning per foot per worker on every reload.
  ASSERT_TRUE(interface_->usesContactBasisVectorInputs());
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  for (const std::string& footName : contactNames_) {
    size_t index = 0;
    ASSERT_FALSE(sqp->getOcpDefinitions().front().softConstraintPtr->getTermIndex(absl::StrCat(footName, "_contactWrenchCone"), index));
  }
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("Failed to update"),
                                      testing::AnyOf(testing::HasSubstr("_contactWrenchCone"), testing::HasSubstr("_frictionForceCone"),
                                                     testing::HasSubstr("_contactMomentXY")))))
      .Times(0);
  // Positive control: the reload ran to the end.
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr("Successfully applied in-place parameter updates")))
      .Times(testing::AtLeast(1));
  log.StartCapturingLogs();
  touchTaskFileAndRunUpdater(*updater);
  log.StopCapturingLogs();
}

/******************************************************************************************************/
// Test: the contact input parameterization is structural; a reloaded file that changes it is reported, not applied
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, AReloadedContactInputParameterizationIsReportedAsStructural) {
  ASSERT_TRUE(interface_->usesContactBasisVectorInputs());
  const std::string shipped = readTmpTaskFile();
  struct Case {
    std::string name;
    std::string content;
    std::string phrase;
  };
  const std::vector<Case> cases = {
      {"switched to wrench",
       acom_test::replacedOnce(shipped, "\ncontactInputParameterization: basis_vectors\n", "\ncontactInputParameterization: wrench\n"),
       "takes effect at the next start"},
      {"retired boolean", absl::StrCat("useContactBasisVectorInputs: true\n", shipped), "useContactBasisVectorInputs"},
  };
  for (const Case& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    writeTmpTaskFile(testCase.content);
    std::unique_ptr<MpcParameterUpdaterModule> updater =
        createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_,
                      /*referenceManager=*/nullptr, basisCostTransform_);
    ASSERT_NE(updater, nullptr);
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr(testCase.phrase))).Times(testing::AtLeast(1));
    log.StartCapturingLogs();
    touchTaskFileAndRunUpdater(*updater);
    log.StopCapturingLogs();
  }
  // Positive control: the shipped file itself is not reported.
  writeTmpTaskFile(shipped);
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr(std::string(kContactInputParameterizationKey))))
      .Times(0);
  log.StartCapturingLogs();
  touchTaskFileAndRunUpdater(*updater);
  log.StopCapturingLogs();
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

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  touchTaskFileAndRunUpdater(*updater);

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
  // change as applied, and the planner ran on its launch configuration for the rest of the session. No shipped robot used
  // the markers then, which is why nothing noticed; both shipped robots now leave shared.comHeight at 0, so every reload
  // of either would have been lost. This test therefore writes a planner file that takes that option and drives the real
  // watcher over it.
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
  EXPECT_EQ(loadContactPlanningConfigStatus(planningFile, "contact_planning.", /*verbose=*/false).status().code(),
            absl::StatusCode::kInvalidArgument);
  absl::StatusOr<ContactPlanningConfig> loaded =
      loadContactPlanningConfigStatus(planningFile, "contact_planning.", /*verbose=*/false, /*validate=*/false);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  ContactPlanningConfig config = *loaded;
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
  ASSERT_EQ(config.validateStatus(), absl::OkStatus()) << "with the model parameters applied the very same file is valid";

  std::shared_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner = std::make_shared<SwingTrajectoryPlanner>(
      loadSwingTrajectorySettings(taskFile_, "swing_trajectory_config", /*verbose=*/false), N_CONTACTS);
  absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>> createdManager = ContactPlanningReferenceManager::Create(
      GaitSchedule::loadGaitSchedule(referenceFile_, interface_->modelSettings(), /*verbose=*/false), swingTrajectoryPlanner,
      interface_->getPinocchioInterface(), interface_->getEffectiveMpcRobotModel(), config);
  ASSERT_TRUE(createdManager.ok()) << createdManager.status();
  const std::shared_ptr<ContactPlanningReferenceManager> planningReferenceManager = *createdManager;
  absl::StatusOr<std::shared_ptr<ContactPlannerModule>> createdModule =
      ContactPlannerModule::Create(planningReferenceManager, config, modelParameters);
  ASSERT_TRUE(createdModule.ok()) << createdModule.status();
  const std::shared_ptr<ContactPlannerModule> plannerModule = *createdModule;
  ASSERT_NEAR(plannerModule->getConfig().shared.comHeight, modelParameters.comHeight, 1e-12) << "the module knows its model parameters";

  // No MPC: this path updates the planner only, and the task file is never touched, so nothing else in the updater runs.
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(/*mpc=*/nullptr, planningTaskFile, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_,
                    /*referenceManager=*/nullptr, basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  updater->setContactPlannerModule(plannerModule);

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
    updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
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
  EXPECT_EQ(applied.validateStatus(), absl::OkStatus())
      << "skipping the loader's validation may never hand the planner an unvalidated configuration";

  std::filesystem::remove_all(planningDir, ec);
}

namespace {

/**
 * The position of the newline that ends the YAML block whose key line starts right after `keyNewline`: the next line
 * indented by at most `indent` spaces that is neither blank nor a comment, or the end of the text.
 */
size_t endOfYamlBlock(const std::string& content, size_t keyNewline, size_t indent) {
  size_t lineStart = content.find('\n', keyNewline + 1);
  while (lineStart != std::string::npos) {
    const size_t firstChar = content.find_first_not_of(' ', lineStart + 1);
    if (firstChar == std::string::npos) return content.size();
    const char c = content[firstChar];
    if (c != '\n' && c != '#' && firstChar - (lineStart + 1) <= indent) return lineStart;
    lineStart = content.find('\n', lineStart + 1);
  }
  return content.size();
}

/** The position of the newline before the top-level `locomotion_heuristics:` key, and the end of that block. */
::testing::AssertionResult findLocomotionHeuristicsBlock(const std::string& content, size_t& blockStart, size_t& blockEnd) {
  blockStart = content.find("\nlocomotion_heuristics:");
  if (blockStart == std::string::npos) {
    return ::testing::AssertionFailure() << "the task file has no top-level locomotion_heuristics block";
  }
  blockEnd = endOfYamlBlock(content, blockStart, /*indent=*/0);
  return ::testing::AssertionSuccess();
}

/**
 * Replaces the list `locomotion_heuristics.<list>` (base_pose, foothold or wrench) with `names`, whatever the file had
 * there - entries, commented-out candidates, or a flow `[]`.
 */
::testing::AssertionResult setLocomotionHeuristicList(std::string& content,
                                                      const std::string& list,
                                                      const std::vector<std::string>& names) {
  size_t blockStart = 0;
  size_t blockEnd = 0;
  const ::testing::AssertionResult found = findLocomotionHeuristicsBlock(content, blockStart, blockEnd);
  if (!found) return found;
  const std::string keyLine = absl::StrCat("\n  ", list, ":");
  const size_t keyPos = content.find(keyLine, blockStart);
  if (keyPos == std::string::npos || keyPos >= blockEnd) {
    return ::testing::AssertionFailure() << "locomotion_heuristics." << list << " is not a key of the task file";
  }
  std::string replacement = keyLine;
  for (const std::string& name : names) absl::StrAppend(&replacement, "\n    - ", name);
  content.replace(keyPos, endOfYamlBlock(content, keyPos, /*indent=*/2) - keyPos, replacement);
  return ::testing::AssertionSuccess();
}

/** One coefficient of the locomotion_heuristics block, written verbatim - which lets a test write text that is not a number. */
struct LocomotionHeuristicCoefficient {
  std::string heuristic;
  std::string key;
  std::string value;
};

/**
 * Sets `locomotion_heuristics.<heuristic>.<key>`, addressed by its block and its exact indentation, so that a key of the
 * same name in another block, or the same word in a comment, cannot be hit instead.
 */
::testing::AssertionResult setLocomotionHeuristicCoefficient(std::string& content, const LocomotionHeuristicCoefficient& coefficient) {
  size_t blockStart = 0;
  size_t blockEnd = 0;
  const ::testing::AssertionResult found = findLocomotionHeuristicsBlock(content, blockStart, blockEnd);
  if (!found) return found;
  const std::string heuristicLine = absl::StrCat("\n  ", coefficient.heuristic, ":");
  const size_t heuristicPos = content.find(heuristicLine, blockStart);
  if (heuristicPos == std::string::npos || heuristicPos >= blockEnd) {
    return ::testing::AssertionFailure() << "locomotion_heuristics." << coefficient.heuristic << " is not a block of the task file";
  }
  const std::string keyLine = absl::StrCat("\n    ", coefficient.key, ":");
  const size_t keyPos = content.find(keyLine, heuristicPos);
  if (keyPos == std::string::npos || keyPos >= endOfYamlBlock(content, heuristicPos, /*indent=*/2)) {
    return ::testing::AssertionFailure() << "locomotion_heuristics." << coefficient.heuristic << "." << coefficient.key
                                         << " is not a key of the task file";
  }
  const size_t valueStart = keyPos + keyLine.size();
  const size_t lineEnd = content.find('\n', valueStart);
  content.replace(valueStart, lineEnd - valueStart, absl::StrCat(" ", coefficient.value));
  return ::testing::AssertionSuccess();
}

/** Sets the SQP iteration count, which the updater applies on every reload, to `iterations`. */
::testing::AssertionResult setSqpIteration(std::string& content, size_t iterations) {
  // Anchored to its indentation, as in SqpSettingsUpdated: a comment that mentions the key must not be hit instead.
  const std::string keyLine = "\n  sqpIteration:";
  const size_t keyPos = content.find(keyLine);
  if (keyPos == std::string::npos) return ::testing::AssertionFailure() << "the task file has no multiple_shooting.sqpIteration";
  const size_t valueStart = keyPos + keyLine.size();
  const size_t lineEnd = content.find('\n', valueStart);
  content.replace(valueStart, lineEnd - valueStart, absl::StrCat(" ", iterations));
  return ::testing::AssertionSuccess();
}

/// [m] The speed-independent height offset composeHeuristicTaskFile() pins for height_compensation.
constexpr scalar_t kPinnedHeightCompensationOffset = 0.03;

/**
 * `shipped` with the locomotion_heuristics lists (wrench always empty), the gain
 * orientation_compensation.pitchPerForwardVelocity and multiple_shooting.sqpIteration replaced.
 *
 * The coefficients an expectation depends on are pinned to values the tests own, so that retuning the shipped block
 * cannot move them: roll is zeroed so that pitch is the only channel orientation_compensation moves, its offset is
 * zeroed so that the pitch is exactly the gain times the forward command, its clamp is opened so that the clamp is never
 * what an assertion measures, and height_compensation is given a speed-independent kPinnedHeightCompensationOffset so
 * that it is visible the moment it is evaluated.
 *
 * The SQP iteration count is how a test proves that the updater applied a particular version of the file: the updater
 * applies it past every early return of applyParameterUpdates(), just ahead of the locomotion_heuristics block. An
 * assertion that a coefficient did NOT change is otherwise indistinguishable from an updater that never looked.
 */
::testing::AssertionResult composeHeuristicTaskFile(const std::string& shipped,
                                                    const std::vector<std::string>& basePose,
                                                    const std::vector<std::string>& foothold,
                                                    const std::string& pitchPerForwardVelocity,
                                                    size_t sqpIteration,
                                                    std::string& content) {
  content = shipped;
  const std::vector<LocomotionHeuristicCoefficient> coefficients{
      {"orientation_compensation", "rollPerLateralVelocity", "0.0"},
      {"orientation_compensation", "rollOffset", "0.0"},
      {"orientation_compensation", "pitchPerForwardVelocity", pitchPerForwardVelocity},
      {"orientation_compensation", "pitchOffset", "0.0"},
      {"orientation_compensation", "maximumTilt", "0.5"},
      {"height_compensation", "heightPerSpeedSquared", "0.0"},
      {"height_compensation", "heightPerSpeed", "0.0"},
      {"height_compensation", "heightOffset", absl::StrCat(kPinnedHeightCompensationOffset)},
      {"height_compensation", "maximumHeightOffset", "0.05"},
  };
  for (const LocomotionHeuristicCoefficient& coefficient : coefficients) {
    const ::testing::AssertionResult set = setLocomotionHeuristicCoefficient(content, coefficient);
    if (!set) return set;
  }
  const ::testing::AssertionResult basePoseSet = setLocomotionHeuristicList(content, "base_pose", basePose);
  if (!basePoseSet) return basePoseSet;
  const ::testing::AssertionResult footholdSet = setLocomotionHeuristicList(content, "foothold", foothold);
  if (!footholdSet) return footholdSet;
  const ::testing::AssertionResult wrenchSet = setLocomotionHeuristicList(content, "wrench", {});
  if (!wrenchSet) return wrenchSet;
  return setSqpIteration(content, sqpIteration);
}

/** The whole text of `path`, or an empty string when it cannot be read. */
std::string readWholeFile(const std::string& path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/** The pitch offset `layer` produces for a forward command of `forwardSpeed` in the base's own yaw frame. */
BasePoseOffset basePoseOffsetAtForwardSpeed(const LocomotionHeuristicLayer& layer, scalar_t forwardSpeed) {
  BasePoseHeuristicContext context;
  context.commandedVelocityInBaseFrame = vector2_t(forwardSpeed, 0.0);
  return layer.basePoseOffset(context);
}

}  // namespace

/******************************************************************************************************/
// Test: the locomotion-heuristic coefficients follow the task file on a running controller, a malformed one is
// rejected without disturbing them, and the lists are not reloaded at all.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, LocomotionHeuristicCoefficientsHotReloadButTheirListsDoNot) {
  // The tuning GUI's sliders for every heuristic coefficient reach the controller through exactly one path: the file
  // watcher (or the topic, which writes a temp file and takes the same function) -> applyParameterUpdates() ->
  // loadLocomotionHeuristicConfig() -> LocomotionHeuristicLayer::reconfigure(). That block sits at the very END of
  // applyParameterUpdates(), behind every early return the function has, and no test called setLocomotionHeuristicLayer()
  // at all - so moving it behind a return, gating it on something the task file does not set, or dropping it, would
  // have left every slider inert while every test passed. This drives the real watcher over a real layer, built by
  // the real start-up path from a task file that lists a heuristic.
  const std::string shipped = readWholeFile(taskFile_);
  ASSERT_FALSE(shipped.empty()) << "could not read " << taskFile_;

  // Every write gets a new SQP iteration count (see composeHeuristicTaskFile) and a modification time of its own rather
  // than a sleep: the watcher compares times for inequality, and two writes inside the file system's timestamp
  // granularity would otherwise look like one.
  constexpr size_t kFirstSqpIteration = 10;
  size_t writeCount = 0;
  const std::filesystem::file_time_type firstWriteTime = std::filesystem::file_time_type::clock::now();
  const std::function<::testing::AssertionResult(const std::vector<std::string>&, const std::vector<std::string>&, const std::string&)>
      writeTaskFile = [&](const std::vector<std::string>& basePose, const std::vector<std::string>& foothold,
                          const std::string& pitchPerForwardVelocity) -> ::testing::AssertionResult {
    ++writeCount;
    std::string content;
    const ::testing::AssertionResult composed =
        composeHeuristicTaskFile(shipped, basePose, foothold, pitchPerForwardVelocity, kFirstSqpIteration + writeCount, content);
    if (!composed) return composed;
    {
      std::ofstream out(tmpTaskFile_, std::ios::trunc);
      out << content;
      if (!out) return ::testing::AssertionFailure() << "could not write " << tmpTaskFile_;
    }
    std::error_code ec;
    std::filesystem::last_write_time(tmpTaskFile_, firstWriteTime + std::chrono::seconds(writeCount), ec);
    if (ec) return ::testing::AssertionFailure() << "could not set the modification time of " << tmpTaskFile_ << ": " << ec.message();
    return ::testing::AssertionSuccess();
  };

  const std::vector<std::string> startUpBasePose{"orientation_compensation"};
  const std::vector<std::string> noHeuristics{};
  ASSERT_TRUE(writeTaskFile(startUpBasePose, noHeuristics, "0.05"));

  // The layer the production node registers: the one the interface built from this task file at start-up.
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = CentroidalMpcInterface::Create(tmpTaskFile_, urdfFile_, referenceFile_);
  ASSERT_TRUE(created.ok()) << created.status().message();
  const std::unique_ptr<CentroidalMpcInterface> heuristicInterface = *std::move(created);
  const std::shared_ptr<LocomotionHeuristicLayer> layer = heuristicInterface->getLocomotionHeuristicLayerPtr();
  ASSERT_NE(layer, nullptr);
  ASSERT_FALSE(layer->basePoseEmpty()) << "the test's task file did not list orientation_compensation, so nothing below can be observed";

  // What the layer says for a commanded forward velocity. With the offset and the roll pinned to zero this is exactly
  // pitchPerForwardVelocity * v_x, so two speeds tell the gain from an offset.
  const std::function<BasePoseOffset(scalar_t)> offsetAtForwardSpeed = [&layer](scalar_t forwardSpeed) {
    return basePoseOffsetAtForwardSpeed(*layer, forwardSpeed);
  };
  EXPECT_NEAR(offsetAtForwardSpeed(1.0).pitch, 0.05, 1e-12) << "the start-up path did not load the listed heuristic's coefficient";

  // The costs do not hold the layer, the reference manager does: unless it holds this very object, a reload retunes a
  // layer that nothing evaluates. Checked through the seam itself as well. The reference manager's own target commands
  // 1 m/s forward at zero yaw, while the state handed to shapeBasePose() carries NO velocity in its momentum channel:
  // shapeBasePose() must take the operator's command from the reference manager (getCommandedVelocity), not from the
  // state it shapes. Were the two the same state, as they once were here, a seam that had gone back to reading the
  // momentum channel of its argument - which under online contact planning is the plan's swaying CoM velocity rather
  // than the command - would have produced the very same pitch and passed every assertion below.
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = heuristicInterface->getSwitchedModelReferenceManagerPtr();
  ASSERT_EQ(referenceManager->getLocomotionHeuristicLayer().get(), layer.get())
      << "the interface hands the updater a different layer from the one its reference manager evaluates";
  const MpcRobotModelBase<scalar_t>& robotModel = heuristicInterface->getEffectiveMpcRobotModel();
  vector_t commandedState = heuristicInterface->getInitialState();
  ASSERT_NEAR(robotModel.getBasePose(commandedState)(3), 0.0, 1e-12) << "the expectations below assume the reference faces +x";
  robotModel.setBaseComLinearVelocity(commandedState, vector3_t(1.0, 0.0, 0.0));
  const vector_t zeroInput = vector_t::Zero(robotModel.getInputDim());
  referenceManager->setTargetTrajectories(TargetTrajectories({0.0, 1.0}, {commandedState, commandedState}, {zeroInput, zeroInput}));
  referenceManager->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, heuristicInterface->getInitialState(), ModeNumber::STANCE);
  // The positive control for the zero-velocity state below: the command does reach the reference manager.
  const vector2_t commandedVelocity = referenceManager->getCommandedVelocity(0.5);
  ASSERT_NEAR(commandedVelocity(0), 1.0, 1e-12) << "the reference manager does not see the 1 m/s its target commands";
  ASSERT_NEAR(commandedVelocity(1), 0.0, 1e-12);
  vector_t unshapedState = heuristicInterface->getInitialState();
  robotModel.setBaseComLinearVelocity(unshapedState, vector3_t::Zero());
  const vector6_t unshapedBasePose = robotModel.getBasePose(unshapedState);
  const std::function<vector6_t()> shapedBasePose = [&referenceManager, &robotModel, &unshapedState]() {
    return robotModel.getBasePose(referenceManager->shapeBasePose(/*time=*/0.5, unshapedState));
  };
  // Index 4 is PITCH: the base pose is Euler ZYX with yaw first.
  EXPECT_NEAR(shapedBasePose()(4) - unshapedBasePose(4), 0.05, 1e-12)
      << "the reference manager does not evaluate the listed heuristic, or takes the command from the state it shapes";
  // And the seam shapes the state it is HANDED: only the base pose moves, so the zero momentum survives. A seam that
  // shaped its own target instead would carry the target's 1 m/s through, while matching every pitch above.
  EXPECT_EQ(robotModel.getBaseComLinearVelocity(referenceManager->shapeBasePose(/*time=*/0.5, unshapedState)).norm(), 0.0)
      << "shapeBasePose() returned something other than the state it was handed with its base pose shaped";

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  updater->setLocomotionHeuristicLayer(layer);
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  const vector_t dummyState = vector_t::Zero(stateDim_);
  const std::function<void()> runUpdater = [&]() {
    for (size_t i = 0; i < 101; ++i) {  // the file is stat-ed once every 100 pre-solve hooks
      updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
    }
  };

  // 1. A slider moves the coefficient: the new value is the one in force, in the layer and at the seam.
  ASSERT_TRUE(writeTaskFile(startUpBasePose, noHeuristics, "0.09"));
  runUpdater();
  ASSERT_EQ(sqp->getSettings().sqpIteration, kFirstSqpIteration + writeCount) << "the updater never applied the edited task file";
  EXPECT_NEAR(offsetAtForwardSpeed(1.0).pitch, 0.09, 1e-12) << "the edited coefficient did not reach the running layer";
  EXPECT_NEAR(offsetAtForwardSpeed(0.5).pitch, 0.045, 1e-12) << "the reload must set the GAIN, which scales with the command";
  EXPECT_NEAR(shapedBasePose()(4) - unshapedBasePose(4), 0.09, 1e-12) << "the reloaded coefficient did not reach the pitch reference";

  // 2. A half-typed value. The file is still valid YAML, so the updater does run - and the loader rejects it, naming
  // the key. The layer must go on running the last good value: neither the struct default (0) nor the start-up value.
  ASSERT_TRUE(writeTaskFile(startUpBasePose, noHeuristics, "0.0.1"));
  const absl::StatusOr<LocomotionHeuristicConfig> malformed = loadLocomotionHeuristicConfig(tmpTaskFile_, /*verbose=*/false);
  ASSERT_FALSE(malformed.ok()) << "0.0.1 was accepted as a number, so this step would test nothing";
  EXPECT_NE(std::string(malformed.status().message()).find("orientation_compensation.pitchPerForwardVelocity"), std::string::npos)
      << malformed.status().message();
  runUpdater();
  ASSERT_EQ(sqp->getSettings().sqpIteration, kFirstSqpIteration + writeCount)
      << "the updater never applied the malformed task file, so the value surviving it proves nothing";
  EXPECT_NEAR(offsetAtForwardSpeed(1.0).pitch, 0.09, 1e-12) << "a rejected reload changed the running coefficient";
  EXPECT_NEAR(shapedBasePose()(4) - unshapedBasePose(4), 0.09, 1e-12);

  // 3. The next good value is applied: a rejected reload must not wedge the watcher or the layer.
  ASSERT_TRUE(writeTaskFile(startUpBasePose, noHeuristics, "0.07"));
  runUpdater();
  ASSERT_EQ(sqp->getSettings().sqpIteration, kFirstSqpIteration + writeCount);
  EXPECT_NEAR(offsetAtForwardSpeed(1.0).pitch, 0.07, 1e-12) << "the first good reload after a rejected one was not applied";

  // 4. The LISTS change on disk: orientation_compensation is removed and height_compensation added to base_pose, and a
  // foothold heuristic appears, while the running heuristic's coefficient moves in the same edit. Which heuristics are
  // listed is wired in at start-up, so the running layer must keep exactly its start-up list - with the coefficient of
  // the heuristic it does run following the file, which is what the tuning GUI relies on.
  const std::vector<std::string> editedBasePose{"height_compensation"};
  const std::vector<std::string> editedFoothold{"capture_point"};
  ASSERT_TRUE(writeTaskFile(editedBasePose, editedFoothold, "0.06"));
  runUpdater();
  ASSERT_EQ(sqp->getSettings().sqpIteration, kFirstSqpIteration + writeCount);
  const BasePoseOffset afterListEdit = offsetAtForwardSpeed(1.0);
  EXPECT_NEAR(afterListEdit.pitch, 0.06, 1e-12)
      << "a changed list on disk must not stop the running heuristic's coefficients from reloading, nor remove the heuristic";
  EXPECT_EQ(afterListEdit.height, 0.0) << "height_compensation, newly listed on disk, was switched on by a hot reload";
  EXPECT_TRUE(layer->footholdEmpty()) << "capture_point, newly listed on disk, was switched on by a hot reload";
  EXPECT_FALSE(layer->basePoseEmpty());
  const std::string summary = layer->summary();
  EXPECT_NE(summary.find("orientation_compensation"), std::string::npos) << summary;
  EXPECT_EQ(summary.find("height_compensation"), std::string::npos) << summary;
  EXPECT_EQ(summary.find("capture_point"), std::string::npos) << summary;
  const vector6_t shapedAfterListEdit = shapedBasePose();
  EXPECT_NEAR(shapedAfterListEdit(4) - unshapedBasePose(4), 0.06, 1e-12);
  EXPECT_EQ(shapedAfterListEdit(2), unshapedBasePose(2)) << "the newly listed height_compensation reached the height reference";

  // The positive control for the zeros above: a controller STARTED from this very file does run height_compensation,
  // with exactly the pinned offset, and a foothold heuristic - so the running layer's zeros come from the list being
  // held, not from the new heuristics' coefficients failing to load.
  const absl::StatusOr<LocomotionHeuristicConfig> onDisk = loadLocomotionHeuristicConfig(tmpTaskFile_, /*verbose=*/false);
  ASSERT_TRUE(onDisk.ok()) << onDisk.status().message();
  const absl::StatusOr<LocomotionHeuristicModelParameters> model = deriveLocomotionHeuristicModelParameters(
      heuristicInterface->getPinocchioInterface(), heuristicInterface->getEffectiveMpcRobotModel(), heuristicInterface->getInitialState());
  ASSERT_TRUE(model.ok()) << model.status().message();
  LocomotionHeuristicEnvironment environment;
  environment.nominalStepWidth = 0.3;  // any positive width: capture_point is listed without hip_centered_stepping
  const absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> fresh = LocomotionHeuristicLayer::Create(*onDisk, *model, environment);
  ASSERT_TRUE(fresh.ok()) << fresh.status().message();
  const BasePoseOffset freshOffset = basePoseOffsetAtForwardSpeed(**fresh, /*forwardSpeed=*/1.0);
  EXPECT_NEAR(freshOffset.height, kPinnedHeightCompensationOffset, 1e-12);
  EXPECT_EQ(freshOffset.pitch, 0.0) << "orientation_compensation is not listed in the edited file";
  EXPECT_FALSE((*fresh)->footholdEmpty());

  // 5. And with the edited lists still on disk - already reported, so the layer takes its quiet path - a further
  // coefficient edit is applied as usual.
  ASSERT_TRUE(writeTaskFile(editedBasePose, editedFoothold, "0.08"));
  runUpdater();
  ASSERT_EQ(sqp->getSettings().sqpIteration, kFirstSqpIteration + writeCount);
  EXPECT_NEAR(offsetAtForwardSpeed(1.0).pitch, 0.08, 1e-12) << "reloads stopped once a changed list had been reported";
  EXPECT_EQ(offsetAtForwardSpeed(1.0).height, 0.0);
  EXPECT_TRUE(layer->footholdEmpty());
}

/******************************************************************************************************/
// Test: the locomotion-heuristic coefficients follow the YAML the tuning GUI publishes, as enqueued text.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, LocomotionHeuristicCoefficientsFollowTheParameterTopic) {
  // The tuning GUI does not save the task file when a slider moves: it publishes the whole edited file on
  // operator/mpc_parameters (mpc_params_tab.py), which the MPC node forwards to enqueueParameterUpdate(),
  // and the updater writes that into a temp file of its own and runs applyParameterUpdates() on THAT. A heuristics block
  // that re-read the watched task file instead of the file it is handed, or that sat on a branch only the file watcher
  // takes, would leave every heuristic slider inert in the mode the GUI actually uses, while the file-watcher case above
  // went on passing. The bus side of the topic is tested by humanoid_common_mpc_app/node's test_mpc_node_runtime.
  const std::string shipped = readWholeFile(taskFile_);
  ASSERT_FALSE(shipped.empty()) << "could not read " << taskFile_;
  const std::vector<std::string> basePose{"orientation_compensation"};
  const std::vector<std::string> noHeuristics{};
  // The published SQP iteration count is the proof that the published YAML, and only it, was applied: the watched file
  // carries a different one and is never touched after the updater is built.
  constexpr size_t kWatchedSqpIteration = 10;
  constexpr size_t kPublishedSqpIteration = 11;
  std::string watchedContent;
  ASSERT_TRUE(composeHeuristicTaskFile(shipped, basePose, noHeuristics, "0.05", kWatchedSqpIteration, watchedContent));
  {
    std::ofstream out(tmpTaskFile_, std::ios::trunc);
    out << watchedContent;
    ASSERT_TRUE(out.good()) << "could not write " << tmpTaskFile_;
  }

  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = CentroidalMpcInterface::Create(tmpTaskFile_, urdfFile_, referenceFile_);
  ASSERT_TRUE(created.ok()) << created.status().message();
  const std::unique_ptr<CentroidalMpcInterface> heuristicInterface = *std::move(created);
  const std::shared_ptr<LocomotionHeuristicLayer> layer = heuristicInterface->getLocomotionHeuristicLayerPtr();
  ASSERT_NE(layer, nullptr);
  ASSERT_NEAR(basePoseOffsetAtForwardSpeed(*layer, /*forwardSpeed=*/1.0).pitch, 0.05, 1e-12)
      << "the test's task file did not list orientation_compensation";

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  updater->setLocomotionHeuristicLayer(layer);

  std::string published;
  ASSERT_TRUE(composeHeuristicTaskFile(shipped, basePose, noHeuristics, "0.11", kPublishedSqpIteration, published));
  updater->enqueueParameterUpdate(published);

  // Applied by the next pre-solve hook.
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  const vector_t dummyState = vector_t::Zero(stateDim_);
  updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
  ASSERT_EQ(sqp->getSettings().sqpIteration, kPublishedSqpIteration) << "the published YAML never reached the updater";
  EXPECT_NEAR(basePoseOffsetAtForwardSpeed(*layer, /*forwardSpeed=*/1.0).pitch, 0.11, 1e-12)
      << "the published YAML reached the updater but its locomotion_heuristics coefficients did not reach the layer";

  // And the value came from the message: the watched file still carries the start-up gain.
  const absl::StatusOr<LocomotionHeuristicConfig> watched = loadLocomotionHeuristicConfig(tmpTaskFile_, /*verbose=*/false);
  ASSERT_TRUE(watched.ok()) << watched.status().message();
  EXPECT_EQ(watched->orientationCompensation.pitchPerForwardVelocity, 0.05);

  std::remove((tmpTaskFile_ + ".live.yaml").c_str());
}

/******************************************************************************************************/
// Test: enqueueParameterUpdate() hands a whole task file to the next preSolverRun(), from any thread, and the newest
// document wins.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, EnqueuedParameterUpdatesAreAppliedByTheNextSolveNewestFirst) {
  // The transport-agnostic entry point the ROS node forwards /mpc_parameter_updates to and the bus's communication
  // thread will call: thread-safe, nothing applied until the solver's pre-solve hook runs, and a document enqueued
  // before the previous one was applied replaces it (the GUI sends the whole file on every slider move).
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_, /*referenceManager=*/nullptr,
                    basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  const size_t runningIterations = sqp->getSettings().sqpIteration;
  const std::string watched = readTmpTaskFile();
  const std::function<std::string(size_t)> withIterations = [&watched](size_t iterations) {
    return std::regex_replace(watched, std::regex("\n  sqpIteration: [^\n]*"), absl::StrCat("\n  sqpIteration: ", iterations),
                              std::regex_constants::format_first_only);
  };
  ASSERT_NE(withIterations(runningIterations + 1), watched) << "the task file has no multiple_shooting.sqpIteration";
  const vector_t dummyState = vector_t::Zero(stateDim_);
  const std::function<void()> solve = [&]() {
    updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, dummyState, *interface_->getReferenceManagerPtr());
  };

  // Two documents from another thread, as a ROS callback would enqueue them, before any solve.
  std::thread producer([&]() {
    updater->enqueueParameterUpdate(withIterations(runningIterations + 1));
    updater->enqueueParameterUpdate(withIterations(runningIterations + 2));
  });
  producer.join();
  EXPECT_EQ(sqp->getSettings().sqpIteration, runningIterations) << "an enqueued document was applied before the next solve";
  solve();
  EXPECT_EQ(sqp->getSettings().sqpIteration, runningIterations + 2) << "the newest document was not the one applied";

  // Documents enqueued while solves run: each solve applies one of them, and the solve after the last enqueue applies
  // the last one.
  constexpr size_t kNumDocuments = 10;
  std::thread streamer([&]() {
    for (size_t k = 0; k < kNumDocuments; ++k) {
      updater->enqueueParameterUpdate(withIterations(runningIterations + 3 + k));
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  });
  for (size_t k = 0; k < kNumDocuments; ++k) {
    solve();
    const size_t applied = sqp->getSettings().sqpIteration;
    EXPECT_GE(applied, runningIterations + 2) << "a solve applied something that was never enqueued";
    EXPECT_LT(applied, runningIterations + 3 + kNumDocuments) << "a solve applied something that was never enqueued";
  }
  streamer.join();
  solve();
  EXPECT_EQ(sqp->getSettings().sqpIteration, runningIterations + 3 + kNumDocuments - 1) << "the last document was not applied";

  // The watched task file is not what was applied: it still carries the start-up value.
  EXPECT_EQ(readTmpTaskFile(), watched) << "an enqueued update wrote the watched task file";
  std::remove((tmpTaskFile_ + ".live.yaml").c_str());
}

namespace contact_implicit_test {

/** `content` with the top-level list `key` - everything up to the next blank line - replaced by `entries`. */
std::string withListBlock(const std::string& content, const std::string& key, const std::vector<std::string>& entries) {
  const std::string::size_type keyPos = content.find(absl::StrCat("\n", key, ":"));
  EXPECT_NE(keyPos, std::string::npos) << "the task file has no " << key << " list";
  if (keyPos == std::string::npos) return content;
  const std::string::size_type blockStart = keyPos + 1;
  const std::string::size_type blockEnd = content.find("\n\n", blockStart);
  std::string block = absl::StrCat(key, ":", entries.empty() ? " []" : "", "\n");
  for (const std::string& entry : entries) absl::StrAppend(&block, "  - ", entry, "\n");
  const std::string tail = blockEnd == std::string::npos ? std::string() : content.substr(blockEnd + 1);
  return absl::StrCat(content.substr(0, blockStart), block, tail);
}

/**
 * The shipped task file with the contact-implicit formulation switched on, the way README section 5 says: no
 * schedule-gated hard constraint, the soft normal_velocity, a cone, and the three terms.
 */
std::string withContactImplicitFormulation(const std::string& shipped) {
  const std::string content = withListBlock(shipped, "hard_constraints", {});
  return withListBlock(content, "soft_constraints",
                       {"joint_limits", "foot_collision", "contact_wrench_cone", "normal_velocity", "contact_complementarity",
                        "force_weighted_slip", "ground_penetration"});
}

/** `content` with the value of the line `<indent><key>:` that follows `anchor` replaced by `value`, its comment included. */
std::string withValueAfter(const std::string& content, const std::string& anchor, const std::string& keyLine, const std::string& value) {
  const std::string::size_type anchorPos = content.find(anchor);
  EXPECT_NE(anchorPos, std::string::npos) << "'" << anchor << "' not found";
  if (anchorPos == std::string::npos) return content;
  const std::string::size_type keyPos = content.find(keyLine, anchorPos);
  EXPECT_NE(keyPos, std::string::npos) << "'" << keyLine << "' not found after '" << anchor << "'";
  if (keyPos == std::string::npos) return content;
  const std::string::size_type valueStart = keyPos + keyLine.size();
  std::string result = content;
  result.replace(valueStart, result.find('\n', valueStart) - valueStart, absl::StrCat(" ", value));
  return result;
}

/** `content` with `contact_implicit.<key>` set to `value`. */
std::string withContactImplicitValue(const std::string& content, absl::string_view key, const std::string& value) {
  return withValueAfter(content, absl::StrCat("\n", ModelSettings::kContactImplicitBlock, ":"), absl::StrCat("\n  ", key, ":"), value);
}

/** `content` with the top-level scalar `key` set to `value`. */
std::string withTopLevelValue(const std::string& content, const std::string& key, const std::string& value) {
  return withValueAfter(content, absl::StrCat("\n", key, ":"), absl::StrCat("\n", key, ":"), value);
}

/** `content` with the line of `contact_implicit.<key>` removed, so that the block no longer carries the key. */
std::string withoutContactImplicitKey(const std::string& content, absl::string_view key) {
  const std::string::size_type blockPos = content.find(absl::StrCat("\n", ModelSettings::kContactImplicitBlock, ":"));
  EXPECT_NE(blockPos, std::string::npos) << "the task file has no contact_implicit block";
  if (blockPos == std::string::npos) return content;
  const std::string::size_type keyPos = content.find(absl::StrCat("\n  ", key, ":"), blockPos);
  EXPECT_NE(keyPos, std::string::npos) << "contact_implicit." << key << " not found";
  if (keyPos == std::string::npos) return content;
  std::string result = content;
  result.erase(keyPos, result.find('\n', keyPos + 1) - keyPos);
  return result;
}

/** Writes `content` to `path`, after a pause long enough for the file watcher to see a new modification time. */
void writeAfterAPause(const std::string& path, const std::string& content) {
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::ofstream out(path, std::ios::trunc);
  out << content;
}

/** The parameters of the penalty a soft constraint carries. */
template <typename SoftConstraint>
vector_t penaltyParameters(SoftConstraint& softConstraint) {
  vector_t parameters;
  std::vector<std::unique_ptr<augmented::AugmentedPenaltyBase>>& penalties = softConstraint.getPenalty().getPenaltyPtrArray();
  EXPECT_FALSE(penalties.empty());
  if (!penalties.empty()) penalties.front()->getParameters(parameters);
  return parameters;
}

/** Whether the penalty of a soft constraint carries exactly `expected` as its parameters. */
template <typename SoftConstraint>
::testing::AssertionResult hasPenaltyParameters(SoftConstraint& softConstraint, const std::vector<scalar_t>& expected) {
  const vector_t actual = penaltyParameters(softConstraint);
  if (static_cast<size_t>(actual.size()) == expected.size() && std::equal(expected.begin(), expected.end(), actual.data())) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << "penalty parameters [" << actual.transpose() << "], expected [" << absl::StrJoin(expected, ", ")
                                       << "]";
}

StateInputSoftConstraint& complementarityTerm(OptimalControlProblem& ocp, const std::string& footName) {
  return ocp.softConstraintPtr->get<StateInputSoftConstraint>(contact_term::name(footName, contact_term::kContactComplementarity));
}
StateInputSoftConstraint& slipTerm(OptimalControlProblem& ocp, const std::string& footName) {
  return ocp.softConstraintPtr->get<StateInputSoftConstraint>(contact_term::name(footName, contact_term::kForceWeightedSlip));
}
StateSoftConstraint& penetrationTerm(OptimalControlProblem& ocp, const std::string& footName) {
  return ocp.stateSoftConstraintPtr->get<StateSoftConstraint>(contact_term::name(footName, contact_term::kGroundPenetration));
}

/**
 * [m] The ground of the complementarity and penetration terms of every foot of every worker's problem, or NaN when
 * they do not all agree - which is itself the failure the tests below guard against.
 */
scalar_t contactImplicitTermsTerrainHeight(SqpSolver& solver, const std::vector<std::string>& contactNames) {
  std::vector<scalar_t> heights;
  for (OptimalControlProblem& ocp : solver.getOcpDefinitions()) {
    for (const std::string& footName : contactNames) {
      heights.push_back(complementarityTerm(ocp, footName).get<ContactComplementarityConstraint>().getTerrainHeight());
      heights.push_back(penetrationTerm(ocp, footName).get<GroundPenetrationConstraint>().getTerrainHeight());
    }
  }
  if (heights.empty()) return std::numeric_limits<scalar_t>::quiet_NaN();
  for (const scalar_t height : heights) {
    if (height != heights.front()) return std::numeric_limits<scalar_t>::quiet_NaN();
  }
  return heights.front();
}

/** One reference-manager-then-modules cycle, in the order SolverBase::preRun calls them. */
void solverOrderCycle(scalar_t time, const CentroidalMpcInterface& interface, const std::vector<SolverSynchronizedModule*>& modules) {
  const vector_t state = interface.getInitialState();
  const scalar_t horizon = interface.mpcSettings().timeHorizon_;
  SwitchedModelReferenceManager& referenceManager = *interface.getSwitchedModelReferenceManagerPtr();
  referenceManager.preSolverRun(time, time + horizon, state, ModeNumber::STANCE);
  for (SolverSynchronizedModule* module : modules) module->preSolverRun(time, time + horizon, state, referenceManager);
}

}  // namespace contact_implicit_test

/******************************************************************************************************/
// Test: every key of the contact_implicit block reaches every term of every worker's problem (audit findings A5/A14)
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, EveryContactImplicitKeyReachesEveryTermOfEveryWorker) {
  // The formulation ships switched off, and the one test of this path used to build its problem from the shipped file,
  // find no contact-implicit term and GTEST_SKIP() - on every run. The file is switched on here instead, and every key
  // the block defines is rewritten, so that a key the updater forgot, a term name it misspelled or a setter it dropped
  // turns this red.
  const std::string taskFile = absl::StrCat(testing::TempDir(), "/contact_implicit_updater_task.yaml");
  const std::string enabled = contact_implicit_test::withContactImplicitFormulation(readWholeFile(taskFile_));
  contact_implicit_test::writeAfterAPause(taskFile, enabled);
  Built built = build(taskFile);
  ASSERT_NE(built.interface, nullptr);
  const std::vector<std::string>& contactNames = built.interface->modelSettings().contactNames;
  const ModelSettings::ContactImplicitConfig& launched = built.interface->modelSettings().contactImplicitConfig;

  // A value per key that is neither the shipped one nor ModelSettings' default. Looked up for EVERY key the block
  // defines, so that a key added to ModelSettings::contactImplicitKeys() without a check here fails the test.
  const std::vector<std::pair<std::string, scalar_t>> sentinels = {
      {"complementarityWeight", 61.5}, {"slipWeight", 173.25},      {"penetrationWeight", 4.25e4},
      {"heightReference", 0.0725},     {"velocityReference", 0.35}, {"angularVelocityReference", 1.25},
      {"gapSmoothing", 1.5e-3},
  };
  ModelSettings::ContactImplicitConfig expected;
  std::string reloaded = enabled;
  for (const ModelSettings::ContactImplicitKey& key : ModelSettings::contactImplicitKeys()) {
    const std::vector<std::pair<std::string, scalar_t>>::const_iterator sentinel = std::find_if(
        sentinels.begin(), sentinels.end(), [&key](const std::pair<std::string, scalar_t>& entry) { return entry.first == key.name; });
    ASSERT_NE(sentinel, sentinels.end()) << "contact_implicit." << key.name << " has no sentinel in this test";
    ASSERT_NE(launched.*key.field, sentinel->second) << "the sentinel of " << key.name << " is the launched value";
    expected.*key.field = sentinel->second;
    reloaded = contact_implicit_test::withContactImplicitValue(reloaded, key.name, absl::StrCat(sentinel->second));
  }
  constexpr scalar_t kReloadedTerrainHeight = 0.013;
  const scalar_t launchedTerrainHeight = built.interface->modelSettings().terrainHeight;
  ASSERT_NE(launchedTerrainHeight, kReloadedTerrainHeight);
  reloaded = contact_implicit_test::withTopLevelValue(reloaded, "terrainHeight", absl::StrCat(kReloadedTerrainHeight));

  SwitchedModelReferenceManager& referenceManager = *built.interface->getSwitchedModelReferenceManagerPtr();
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(built.mpc.get(), taskFile, urdfFile_, referenceFile_, built.interface->getMpcRobotModel().getStateDim(),
                    built.interface->getEffectiveMpcRobotModel().getInputDim(), contactNames, &referenceManager,
                    built.interface->getBasisInputsCostTransformConfig());
  ASSERT_NE(updater, nullptr);
  contact_implicit_test::writeAfterAPause(taskFile, reloaded);
  touchAndRun(taskFile, *updater, *built.interface);

  SqpSolver& solver = built.solver();
  ASSERT_GT(solver.getOcpDefinitions().size(), 1U) << "the check has to cover more than one worker's copy of the problem";
  for (OptimalControlProblem& ocp : solver.getOcpDefinitions()) {
    for (const std::string& footName : contactNames) {
      SCOPED_TRACE(footName);
      StateInputSoftConstraint& complementarity = contact_implicit_test::complementarityTerm(ocp, footName);
      EXPECT_TRUE(contact_implicit_test::hasPenaltyParameters(complementarity, {expected.complementarityWeight}));
      EXPECT_NEAR(complementarity.get<ContactComplementarityConstraint>().getHeightReference(), expected.heightReference, 1e-12);
      EXPECT_DOUBLE_EQ(complementarity.get<ContactComplementarityConstraint>().getGapSmoothing(), expected.gapSmoothing);

      StateInputSoftConstraint& slip = contact_implicit_test::slipTerm(ocp, footName);
      EXPECT_TRUE(contact_implicit_test::hasPenaltyParameters(slip, {expected.slipWeight}));
      const vector3_t inverseReferences = slip.get<ForceWeightedSlipConstraint>().getInverseTwistReference();
      EXPECT_NEAR(inverseReferences(0), 1.0 / expected.velocityReference, 1e-12);
      EXPECT_NEAR(inverseReferences(1), 1.0 / expected.velocityReference, 1e-12);
      EXPECT_NEAR(inverseReferences(2), 1.0 / expected.angularVelocityReference, 1e-12);

      // The hinge: the reloaded weight, and a delta that stays 0 so its zero stays on the ground.
      StateSoftConstraint& penetration = contact_implicit_test::penetrationTerm(ocp, footName);
      EXPECT_EQ(penetration.getPenalty().getPenaltyPtrArray().front()->name(), "SquaredHingePenalty");
      EXPECT_TRUE(contact_implicit_test::hasPenaltyParameters(penetration, {expected.penetrationWeight, 0.0}));
    }
  }

  // The ground went to the reference manager, which owns it - and NOT yet to the terms: this solve's references were
  // built on the launched ground before the updater ran, and the terms must describe the same one.
  EXPECT_DOUBLE_EQ(referenceManager.getTerrainHeight(), kReloadedTerrainHeight);
  EXPECT_DOUBLE_EQ(referenceManager.getAppliedTerrainHeight(), launchedTerrainHeight);
  EXPECT_DOUBLE_EQ(contact_implicit_test::contactImplicitTermsTerrainHeight(solver, contactNames), launchedTerrainHeight)
      << "the terms moved to a ground the references the solver tracks were not built on";

  // The next solve, in the solver's order: the swing trajectories are rebuilt on the new ground, and the terms follow.
  const vector_t state = built.interface->getInitialState();
  referenceManager.setTargetTrajectories(
      TargetTrajectories({0.0}, {state}, {vector_t::Zero(built.interface->getEffectiveMpcRobotModel().getInputDim())}));
  contact_implicit_test::solverOrderCycle(/*time=*/0.1, *built.interface, {updater.get()});
  EXPECT_DOUBLE_EQ(referenceManager.getAppliedTerrainHeight(), kReloadedTerrainHeight);
  EXPECT_DOUBLE_EQ(contact_implicit_test::contactImplicitTermsTerrainHeight(solver, contactNames), kReloadedTerrainHeight);
  size_t stanceFeet = 0;
  for (size_t foot = 0; foot < contactNames.size(); ++foot) {
    if (!referenceManager.isInContact(/*time=*/0.1, foot)) continue;
    EXPECT_DOUBLE_EQ(referenceManager.getSwingTrajectoryPlanner()->getZpositionConstraint(foot, /*time=*/0.1), kReloadedTerrainHeight)
        << contactNames[foot] << ": a stance foot's height reference is the ground";
    ++stanceFeet;
  }
  EXPECT_GT(stanceFeet, 0U);
}

/******************************************************************************************************/
// Test: a contact_implicit block the start-up path would refuse is refused as a whole, naming the key
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, ARefusedContactImplicitBlockLeavesEveryTermAsItWasAndTheRestOfTheFileApplies) {
  // Audit finding A25 and A11/A22. A zero divisor or a negative weight used to be skipped without a word by `> 0` guards
  // (and +inf passed them), and a key renamed in the file was simply never read. Each is now refused with the message the
  // start-up path gives, and nothing of the block is applied - not the valid complementarityWeight edit made alongside
  // - while the rest of the file still is (the SQP iteration count proves the reload ran).
  const std::string taskFile = absl::StrCat(testing::TempDir(), "/contact_implicit_refused_task.yaml");
  const std::string enabled = contact_implicit_test::withContactImplicitFormulation(readWholeFile(taskFile_));
  contact_implicit_test::writeAfterAPause(taskFile, enabled);
  Built built = build(taskFile);
  ASSERT_NE(built.interface, nullptr);
  const std::vector<std::string>& contactNames = built.interface->modelSettings().contactNames;
  const ModelSettings::ContactImplicitConfig& launched = built.interface->modelSettings().contactImplicitConfig;
  constexpr scalar_t kEditedWeight = 61.5;
  ASSERT_NE(launched.complementarityWeight, kEditedWeight);
  const std::string edited = contact_implicit_test::withContactImplicitValue(enabled, "complementarityWeight", absl::StrCat(kEditedWeight));

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(built.mpc.get(), taskFile, urdfFile_, referenceFile_, built.interface->getMpcRobotModel().getStateDim(),
                    built.interface->getEffectiveMpcRobotModel().getInputDim(), contactNames,
                    built.interface->getSwitchedModelReferenceManagerPtr().get(), built.interface->getBasisInputsCostTransformConfig());
  ASSERT_NE(updater, nullptr);
  SqpSolver& solver = built.solver();
  const std::function<std::vector<scalar_t>()> complementarityWeights = [&solver, &contactNames]() {
    std::vector<scalar_t> weights;
    for (OptimalControlProblem& ocp : solver.getOcpDefinitions()) {
      for (const std::string& footName : contactNames) {
        weights.push_back(contact_implicit_test::penaltyParameters(contact_implicit_test::complementarityTerm(ocp, footName))(0));
      }
    }
    return weights;
  };

  struct Case {
    std::string name;
    std::string content;
    std::string phrase;  // what the warning has to name
  };
  const std::vector<Case> cases = {
      {"a divisor of zero", contact_implicit_test::withContactImplicitValue(edited, "gapSmoothing", "0"), "contact_implicit.gapSmoothing"},
      {"a negative weight", contact_implicit_test::withContactImplicitValue(edited, "penetrationWeight", "-5.0e4"),
       "contact_implicit.penetrationWeight"},
      {"a value that is not a number", contact_implicit_test::withContactImplicitValue(edited, "heightReference", "high"),
       "contact_implicit.heightReference"},
      {"a renamed key", acom_test::replacedOnce(edited, "\n  gapSmoothing:", "\n  gap_smoothing:"), "contact_implicit.gap_smoothing"},
  };
  size_t sqpIteration = 20;
  for (const Case& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    ++sqpIteration;
    std::string content = testCase.content;
    ASSERT_TRUE(setSqpIteration(content, sqpIteration));
    contact_implicit_test::writeAfterAPause(taskFile, content);
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                         testing::AllOf(testing::HasSubstr("contact_implicit not applied"), testing::HasSubstr(testCase.phrase))))
        .Times(testing::AtLeast(1));
    log.StartCapturingLogs();
    touchAndRun(taskFile, *updater, *built.interface);
    log.StopCapturingLogs();
    EXPECT_EQ(solver.getSettings().sqpIteration, sqpIteration) << "the rest of the file was not applied";
    for (const scalar_t weight : complementarityWeights()) {
      EXPECT_EQ(weight, launched.complementarityWeight) << "part of a refused contact_implicit block was applied";
    }
  }

  // Positive control: the same edit, without the defect, is applied - so the refusals above are not an updater that
  // never reaches the block.
  contact_implicit_test::writeAfterAPause(taskFile, edited);
  touchAndRun(taskFile, *updater, *built.interface);
  for (const scalar_t weight : complementarityWeights()) {
    EXPECT_EQ(weight, kEditedWeight);
  }
}

/******************************************************************************************************/
// Test: a reloaded terrainHeight moves the swing references, the landing targets and both terms in the same solve
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, AReloadedGroundMovesTheSwingReferencesTheLandingTargetsAndBothTermsInTheSameSolve) {
  // Audit findings A1/A13. The updater used to write a reloaded terrainHeight into the complementarity and penetration
  // terms only, while the swing trajectories, the landing targets and the base height kept reading the launch value
  // from ModelSettings: the MPC planned every touch-down on one ground and priced contact against another. The reference
  // manager now owns the ground and the terms follow the ground it applied. Checked on the configuration where every
  // one of them exists: the contact-implicit terms AND the online contact planner, which draws the landing targets.
  const std::filesystem::path directory = std::filesystem::path(testing::TempDir()) / "mpc_parameter_updater_ground";
  std::error_code ec;
  std::filesystem::remove_all(directory, ec);
  std::filesystem::create_directories(directory, ec);
  ASSERT_FALSE(ec) << ec.message();
  const std::string taskFile = (directory / "task.yaml").string();
  std::string enabled = contact_implicit_test::withContactImplicitFormulation(readWholeFile(taskFile_));
  enabled = acom_test::replacedOnce(enabled, "\ncontactScheduleSource: gait_schedule\n", "\ncontactScheduleSource: contact_planner\n");
  contact_implicit_test::writeAfterAPause(taskFile, enabled);
  std::string planning = readWholeFile(resolveContactPlanningConfigFile(taskFile_));
  ASSERT_NE(planning.find("contact_planning:"), std::string::npos) << "the shipped planner configuration was not found";
  // Planned in the pre-solve hook, so that the test decides when a plan is made and activated.
  planning = std::regex_replace(planning, std::regex("runInBackgroundThread: *(true|false)"), "runInBackgroundThread: false");
  {
    std::ofstream out((directory / kContactPlanningConfigFileName).string());
    out << planning;
  }

  Built built = build(taskFile);
  ASSERT_NE(built.interface, nullptr);
  const std::vector<std::string>& contactNames = built.interface->modelSettings().contactNames;
  const std::shared_ptr<ContactPlanningReferenceManager> referenceManager =
      std::dynamic_pointer_cast<ContactPlanningReferenceManager>(built.interface->getSwitchedModelReferenceManagerPtr());
  ASSERT_NE(referenceManager, nullptr) << "the test's task file did not switch the contact planner on";
  const std::shared_ptr<ContactPlannerModule> plannerModule = built.interface->getContactPlannerModulePtr();
  ASSERT_NE(plannerModule, nullptr);
  const scalar_t launchedTerrainHeight = built.interface->modelSettings().terrainHeight;
  constexpr scalar_t kReloadedTerrainHeight = 0.021;
  ASSERT_NE(launchedTerrainHeight, kReloadedTerrainHeight);

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(built.mpc.get(), taskFile, urdfFile_, referenceFile_, built.interface->getMpcRobotModel().getStateDim(),
                    built.interface->getEffectiveMpcRobotModel().getInputDim(), contactNames, referenceManager.get(),
                    built.interface->getBasisInputsCostTransformConfig());
  ASSERT_NE(updater, nullptr);
  updater->setContactPlannerModule(plannerModule);
  const std::vector<SolverSynchronizedModule*> modules = {plannerModule.get(), updater.get()};

  // Walk forward at 0.5 m/s: plan at 0, activate at 0.02.
  const vector_t state = built.interface->getInitialState();
  vector_t target = vector_t::Zero(state.size());
  target.segment(6, 6) = state.segment(6, 6);
  target(0) = 0.5;
  referenceManager->setTargetTrajectories(
      TargetTrajectories({0.0}, {target}, {vector_t::Zero(built.interface->getEffectiveMpcRobotModel().getInputDim())}));
  contact_implicit_test::solverOrderCycle(/*time=*/0.0, *built.interface, modules);
  contact_implicit_test::solverOrderCycle(/*time=*/0.02, *built.interface, modules);
  ASSERT_TRUE(referenceManager->hasActivePlan());

  // Everything that has an opinion about the ground: the height reference of a foot in stance and at the touch-down of
  // its next swing, the landing targets, and the two terms.
  struct Grounds {
    std::vector<scalar_t> stanceHeights;
    std::vector<scalar_t> touchDownHeights;
    std::vector<scalar_t> landingHeights;
  };
  const std::function<Grounds(scalar_t)> groundsAfter = [&referenceManager, &contactNames](scalar_t time) {
    Grounds grounds;
    const ModeSchedule& schedule = referenceManager->getModeSchedule();
    const SwingTrajectoryPlanner& swingPlanner = *referenceManager->getSwingTrajectoryPlanner();
    for (size_t foot = 0; foot < contactNames.size(); ++foot) {
      for (size_t i = 0; i + 1 < schedule.modeSequence.size() && i < schedule.eventTimes.size(); ++i) {
        const scalar_t liftOff = schedule.eventTimes[i];
        if (liftOff <= time || !modeNumber2StanceLeg(schedule.modeSequence[i])[foot] ||
            modeNumber2StanceLeg(schedule.modeSequence[i + 1])[foot]) {
          continue;
        }
        grounds.stanceHeights.push_back(swingPlanner.getZpositionConstraint(foot, liftOff - 1.0e-3));
        for (size_t j = i + 1; j + 1 < schedule.modeSequence.size() && j < schedule.eventTimes.size(); ++j) {
          if (modeNumber2StanceLeg(schedule.modeSequence[j + 1])[foot]) {
            grounds.touchDownHeights.push_back(swingPlanner.getZpositionConstraint(foot, schedule.eventTimes[j] - 1.0e-9) -
                                               swingPlanner.getConfig().touchDownHeightOffset);
            break;
          }
        }
        break;
      }
    }
    for (const TargetContactPose& pose : referenceManager->getTargetContactPoses()) {
      if (pose.valid && pose.kind != TargetContactPose::Kind::STANCE) grounds.landingHeights.push_back(pose.height);
    }
    return grounds;
  };
  const std::function<void(const Grounds&, scalar_t)> expectGround = [](const Grounds& grounds, scalar_t ground) {
    ASSERT_FALSE(grounds.stanceHeights.empty()) << "the plan has no swing to check";
    ASSERT_FALSE(grounds.touchDownHeights.empty());
    ASSERT_FALSE(grounds.landingHeights.empty()) << "the plan has no landing target to check";
    for (const scalar_t height : grounds.stanceHeights) EXPECT_NEAR(height, ground, 1e-12) << "the stance height reference";
    for (const scalar_t height : grounds.touchDownHeights) EXPECT_NEAR(height, ground, 1e-6) << "the touch-down height reference";
    for (const scalar_t height : grounds.landingHeights) EXPECT_NEAR(height, ground, 1e-12) << "a landing target";
  };
  // Positive control: before the reload, everything is on the launched ground.
  expectGround(groundsAfter(0.02), launchedTerrainHeight);
  EXPECT_DOUBLE_EQ(contact_implicit_test::contactImplicitTermsTerrainHeight(built.solver(), contactNames), launchedTerrainHeight);

  // The operator moves the terrainHeight slider.
  contact_implicit_test::writeAfterAPause(
      taskFile, contact_implicit_test::withTopLevelValue(enabled, "terrainHeight", absl::StrCat(kReloadedTerrainHeight)));
  touchAndRun(taskFile, *updater, *built.interface);
  EXPECT_DOUBLE_EQ(referenceManager->getTerrainHeight(), kReloadedTerrainHeight);
  // Until the reference manager rebuilds its references, nothing moves: not the references, and not the terms.
  EXPECT_DOUBLE_EQ(contact_implicit_test::contactImplicitTermsTerrainHeight(built.solver(), contactNames), launchedTerrainHeight)
      << "the terms moved to a ground the references the solver tracks were not built on";

  // The next solve: all of them, together.
  contact_implicit_test::solverOrderCycle(/*time=*/0.04, *built.interface, modules);
  expectGround(groundsAfter(0.04), kReloadedTerrainHeight);
  EXPECT_DOUBLE_EQ(contact_implicit_test::contactImplicitTermsTerrainHeight(built.solver(), contactNames), kReloadedTerrainHeight);

  std::filesystem::remove_all(directory, ec);
}

/******************************************************************************************************/
// Test: without a reference manager, a reloaded terrainHeight reaches both contact-implicit terms at once
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, WithoutAReferenceManagerAReloadedGroundReachesBothTermsAtOnce) {
  // With a reference manager the reloaded ground goes to it, and the terms follow at the next solve (the test above).
  // Without one there are no swing trajectories or landing targets for the terms to agree with, so the updater moves
  // them itself, at once - the only path to them that test cannot reach.
  const std::string taskFile = absl::StrCat(testing::TempDir(), "/contact_implicit_no_reference_manager_task.yaml");
  const std::string enabled = contact_implicit_test::withContactImplicitFormulation(readWholeFile(taskFile_));
  contact_implicit_test::writeAfterAPause(taskFile, enabled);
  Built built = build(taskFile);
  ASSERT_NE(built.interface, nullptr);
  const std::vector<std::string>& contactNames = built.interface->modelSettings().contactNames;
  const scalar_t launchedTerrainHeight = built.interface->modelSettings().terrainHeight;
  constexpr scalar_t kReloadedTerrainHeight = 0.017;
  ASSERT_NE(launchedTerrainHeight, kReloadedTerrainHeight);
  // Positive control: the terms start on the launched ground.
  ASSERT_DOUBLE_EQ(contact_implicit_test::contactImplicitTermsTerrainHeight(built.solver(), contactNames), launchedTerrainHeight);

  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built, taskFile, urdfFile_, referenceFile_);
  ASSERT_NE(updater, nullptr);  // no reference manager
  contact_implicit_test::writeAfterAPause(
      taskFile, contact_implicit_test::withTopLevelValue(enabled, "terrainHeight", absl::StrCat(kReloadedTerrainHeight)));
  touchAndRun(taskFile, *updater, *built.interface);

  EXPECT_DOUBLE_EQ(contact_implicit_test::contactImplicitTermsTerrainHeight(built.solver(), contactNames), kReloadedTerrainHeight)
      << "without a reference manager the reloaded ground never reached the terms";
  EXPECT_DOUBLE_EQ(built.interface->getSwitchedModelReferenceManagerPtr()->getTerrainHeight(), launchedTerrainHeight)
      << "the updater was given no reference manager, so it cannot have moved the one the interface holds";
}

/******************************************************************************************************/
// Test: a contact_implicit block that carries only some of its keys applies those and leaves the rest running
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, APartialContactImplicitBlockAppliesOnlyTheKeysItCarries) {
  // The reloaded block is read over ModelSettings' defaults, so that it can be validated as a whole, but only the keys
  // the file carries are applied: a key it leaves out keeps the value the running term holds, NOT the default. The
  // problem is launched on values that are neither the defaults nor the shipped ones, so that a reload which applied a
  // default for a key the file does not carry would show.
  const std::string taskFile = absl::StrCat(testing::TempDir(), "/contact_implicit_partial_block_task.yaml");
  const std::vector<std::pair<std::string, scalar_t>> launchValues = {
      {"complementarityWeight", 61.5}, {"slipWeight", 173.25},      {"penetrationWeight", 4.25e4},
      {"heightReference", 0.0725},     {"velocityReference", 0.35}, {"angularVelocityReference", 1.25},
      {"gapSmoothing", 1.5e-3},
  };
  const ModelSettings::ContactImplicitConfig defaults;
  std::string launchedContent = contact_implicit_test::withContactImplicitFormulation(readWholeFile(taskFile_));
  for (const ModelSettings::ContactImplicitKey& key : ModelSettings::contactImplicitKeys()) {
    const std::vector<std::pair<std::string, scalar_t>>::const_iterator value =
        std::find_if(launchValues.begin(), launchValues.end(),
                     [&key](const std::pair<std::string, scalar_t>& entry) { return entry.first == key.name; });
    ASSERT_NE(value, launchValues.end()) << "contact_implicit." << key.name << " has no launch value in this test";
    ASSERT_NE(defaults.*key.field, value->second) << "the launch value of " << key.name << " is its default";
    launchedContent = contact_implicit_test::withContactImplicitValue(launchedContent, key.name, absl::StrCat(value->second));
  }
  contact_implicit_test::writeAfterAPause(taskFile, launchedContent);
  Built built = build(taskFile);
  ASSERT_NE(built.interface, nullptr);
  const std::vector<std::string>& contactNames = built.interface->modelSettings().contactNames;
  const ModelSettings::ContactImplicitConfig& launched = built.interface->modelSettings().contactImplicitConfig;
  for (const ModelSettings::ContactImplicitKey& key : ModelSettings::contactImplicitKeys()) {
    ASSERT_NE(launched.*key.field, defaults.*key.field) << "the problem was not launched on the test's value of " << key.name;
  }

  // The reload carries slipWeight alone.
  constexpr scalar_t kReloadedSlipWeight = 212.5;
  ASSERT_NE(launched.slipWeight, kReloadedSlipWeight);
  std::string partial = contact_implicit_test::withContactImplicitValue(launchedContent, "slipWeight", absl::StrCat(kReloadedSlipWeight));
  for (const ModelSettings::ContactImplicitKey& key : ModelSettings::contactImplicitKeys()) {
    if (key.name != "slipWeight") partial = contact_implicit_test::withoutContactImplicitKey(partial, key.name);
  }

  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(built.mpc.get(), taskFile, urdfFile_, referenceFile_, built.interface->getMpcRobotModel().getStateDim(),
                    built.interface->getEffectiveMpcRobotModel().getInputDim(), contactNames,
                    built.interface->getSwitchedModelReferenceManagerPtr().get(), built.interface->getBasisInputsCostTransformConfig());
  ASSERT_NE(updater, nullptr);
  contact_implicit_test::writeAfterAPause(taskFile, partial);
  touchAndRun(taskFile, *updater, *built.interface);

  SqpSolver& solver = built.solver();
  for (OptimalControlProblem& ocp : solver.getOcpDefinitions()) {
    for (const std::string& footName : contactNames) {
      SCOPED_TRACE(footName);
      // Positive control: the reload ran and reached the block.
      StateInputSoftConstraint& slip = contact_implicit_test::slipTerm(ocp, footName);
      EXPECT_TRUE(contact_implicit_test::hasPenaltyParameters(slip, {kReloadedSlipWeight}));
      const vector3_t inverseReferences = slip.get<ForceWeightedSlipConstraint>().getInverseTwistReference();
      EXPECT_NEAR(inverseReferences(0), 1.0 / launched.velocityReference, 1e-12);
      EXPECT_NEAR(inverseReferences(2), 1.0 / launched.angularVelocityReference, 1e-12);

      StateInputSoftConstraint& complementarity = contact_implicit_test::complementarityTerm(ocp, footName);
      EXPECT_TRUE(contact_implicit_test::hasPenaltyParameters(complementarity, {launched.complementarityWeight}));
      EXPECT_NEAR(complementarity.get<ContactComplementarityConstraint>().getHeightReference(), launched.heightReference, 1e-12);
      EXPECT_DOUBLE_EQ(complementarity.get<ContactComplementarityConstraint>().getGapSmoothing(), launched.gapSmoothing);

      StateSoftConstraint& penetration = contact_implicit_test::penetrationTerm(ocp, footName);
      EXPECT_TRUE(contact_implicit_test::hasPenaltyParameters(penetration, {launched.penetrationWeight, 0.0}));
    }
  }
}

/******************************************************************************************************/
// Test: the updater both MPC nodes build with makeCentroidalMpcParameterUpdater() reaches everything the interface built.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, TheSharedNodeWiringReloadsTheHeuristicLayerThePlannerAndTheReferenceFile) {
  // CentroidalMpcSqpNode and CentroidalMpcRobotSim used to repeat this wiring by hand, and it was tested nowhere: a node
  // that forgot setLocomotionHeuristicLayer() or setContactPlannerModule() built and ran with those sliders inert, and
  // the simulator node did forget the reference.yaml reloaders. This builds a controller with a listed heuristic and the
  // contact planner on, wires its updater with the one function both nodes now call, and edits each file on disk.
  const std::filesystem::path dir = std::filesystem::path(testing::TempDir()) / "shared_updater_wiring";
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);
  ASSERT_FALSE(ec) << ec.message();
  const std::string taskFile = (dir / "task.yaml").string();
  const std::string planningFile = (dir / kContactPlanningConfigFileName).string();
  const std::string referenceFile = (dir / "reference.yaml").string();
  std::filesystem::copy_file(referenceFile_, referenceFile, std::filesystem::copy_options::overwrite_existing, ec);
  ASSERT_FALSE(ec) << ec.message();

  const std::string shipped = readWholeFile(taskFile_);
  std::string planning = readWholeFile(resolveContactPlanningConfigFile(taskFile_));
  ASSERT_FALSE(shipped.empty());
  ASSERT_FALSE(planning.empty());
  // Every write gets a modification time of its own, as in LocomotionHeuristicCoefficientsHotReloadButTheirListsDoNot.
  const std::filesystem::file_time_type firstWriteTime = std::filesystem::file_time_type::clock::now();
  size_t writeCount = 0;
  const std::function<void(const std::string&, const std::string&)> writeFile = [&](const std::string& path, const std::string& text) {
    ++writeCount;
    {
      std::ofstream out(path, std::ios::trunc);
      out << text;
    }
    std::filesystem::last_write_time(path, firstWriteTime + std::chrono::seconds(writeCount), ec);
  };
  const std::function<::testing::AssertionResult(const std::string&, size_t)> writeTask = [&](const std::string& pitch,
                                                                                              size_t sqpIteration) {
    std::string content;
    const ::testing::AssertionResult composed =
        composeHeuristicTaskFile(shipped, {"orientation_compensation"}, {}, pitch, sqpIteration, content);
    if (!composed) return composed;
    content = acom_test::replacedOnce(content, "\ncontactScheduleSource: gait_schedule\n", "\ncontactScheduleSource: contact_planner\n");
    writeFile(taskFile, content);
    return ::testing::AssertionSuccess();
  };
  // The planner's own file, planned synchronously, with its hlip.stepWidth set: the one value checked below.
  const std::function<void(const std::string&)> writePlanning = [&](const std::string& stepWidth) {
    std::string content = std::regex_replace(planning, std::regex("runInBackgroundThread: *(true|false)"), "runInBackgroundThread: false");
    content = std::regex_replace(content, std::regex("\n    stepWidth: [0-9.]+"), absl::StrCat("\n    stepWidth: ", stepWidth));
    writeFile(planningFile, content);
  };
  ASSERT_TRUE(writeTask("0.05", /*sqpIteration=*/11));
  writePlanning("0.25");
  ASSERT_EQ(resolveContactPlanningConfigFile(taskFile), planningFile);

  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = CentroidalMpcInterface::Create(taskFile, urdfFile_, referenceFile);
  ASSERT_TRUE(created.ok()) << created.status();
  const std::unique_ptr<CentroidalMpcInterface> interface = *std::move(created);
  ASSERT_NE(interface->getContactPlannerModulePtr(), nullptr) << "the test's task file did not switch the contact planner on";
  ASSERT_FALSE(interface->getLocomotionHeuristicLayerPtr()->basePoseEmpty()) << "the test's task file lists no heuristic";
  SqpMpc mpc(interface->mpcSettings(), interface->sqpSettings(), interface->getOptimalControlProblem(), interface->getInitializer());
  mpc.getSolverPtr()->setReferenceManager(interface->getReferenceManagerPtr());

  std::vector<std::string> reloadedReferenceFiles;
  absl::StatusOr<std::shared_ptr<MpcParameterUpdaterModule>> wired =
      makeCentroidalMpcParameterUpdater(&mpc, *interface, taskFile, urdfFile_, referenceFile,
                                        {[&reloadedReferenceFiles](const std::string& file) { reloadedReferenceFiles.push_back(file); }});
  ASSERT_TRUE(wired.ok()) << wired.status();
  const std::shared_ptr<MpcParameterUpdaterModule> updater = *wired;
  const std::function<void()> runUpdater = [&]() {
    for (size_t i = 0; i < 101; ++i) {  // the files are stat-ed once every 100 pre-solve hooks
      updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, interface->getInitialState(), *interface->getReferenceManagerPtr());
    }
  };
  const std::function<scalar_t()> pitchGain = [&interface]() {
    return basePoseOffsetAtForwardSpeed(*interface->getLocomotionHeuristicLayerPtr(), /*forwardSpeed=*/1.0).pitch;
  };
  // Positive controls: the launch values are in force, and differ from what the edits below write.
  ASSERT_NEAR(pitchGain(), 0.05, 1e-12);
  ASSERT_NEAR(interface->getContactPlannerModulePtr()->getConfig().hlip.stepWidth, 0.25, 1e-12);

  // A heuristic coefficient, into the very layer the interface's reference manager evaluates.
  ASSERT_TRUE(writeTask("0.09", /*sqpIteration=*/12));
  runUpdater();
  ASSERT_EQ(dynamic_cast<SqpSolver&>(*mpc.getSolverPtr()).getSettings().sqpIteration, 12U) << "the task file was not reloaded at all";
  EXPECT_NEAR(pitchGain(), 0.09, 1e-12) << "the wired updater does not reach the interface's locomotion-heuristic layer";

  // A contact_planning value, into the planner module.
  writePlanning("0.27");
  runUpdater();
  EXPECT_NEAR(interface->getContactPlannerModulePtr()->getConfig().hlip.stepWidth, 0.27, 1e-12)
      << "the wired updater does not reach the contact planner module";

  // reference.yaml, into the reloaders the node registered.
  writeFile(referenceFile, readWholeFile(referenceFile_));
  runUpdater();
  EXPECT_EQ(reloadedReferenceFiles, std::vector<std::string>({referenceFile})) << "the reference.yaml reloaders were not registered";
  std::filesystem::remove_all(dir, ec);
}

/******************************************************************************************************/
// Test: a hot reload of dcm_terminal_cost.comHeight keeps 0 meaning the model's pendulum.
/******************************************************************************************************/
TEST_F(MpcParameterUpdaterModuleTest, AReloadedDcmComHeightOfZeroIsTheModelsPendulumAndAPositiveOneAnOverride) {
  // The shipped Atlas ends its horizon on the DCM cost with comHeight 0, i.e. the model's center of mass above its feet.
  // Every reload hands the block to the running costs as the file writes it, so 0 must go on meaning the model's -
  // on every worker - while an explicit height is used as given and a negative one is refused by its key.
  const std::string shipped = readTmpTaskFile();
  ASSERT_NE(shipped.find("\n  comHeight: 0"), std::string::npos) << "the shipped DCM cost no longer derives its pendulum";
  const scalar_t model = interface_->getNominalComHeight();
  ASSERT_GT(model, 0.5);
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      createUpdater(mpc_.get(), tmpTaskFile_, urdfFile_, referenceFile_, stateDim_, inputDim_, contactNames_,
                    interface_->getSwitchedModelReferenceManagerPtr().get(), basisCostTransform_);
  ASSERT_NE(updater, nullptr);
  SqpSolver* sqp = getSqpSolver();
  ASSERT_NE(sqp, nullptr);
  const std::function<void(scalar_t)> expectOmegaOnEveryWorker = [sqp](scalar_t height) {
    for (OptimalControlProblem& ocp : sqp->getOcpDefinitions()) {
      const DcmTerminalCost& cost = ocp.finalCostPtr->get<DcmTerminalCost>("dcmTerminalCost");
      EXPECT_NEAR(cost.getConfig().comHeight, height, 1e-12);
      EXPECT_NEAR(cost.getParameters(/*time=*/0.5, TargetTrajectories())(2), std::sqrt(9.81 / height), 1e-12);
    }
  };
  const std::function<void(const std::string&)> reloadWithComHeight = [&](const std::string& value) {
    writeTmpTaskFile(std::regex_replace(shipped, std::regex("\n  comHeight: [^\n]*"), absl::StrCat("\n  comHeight: ", value)));
    touchTaskFileAndRunUpdater(*updater);
  };
  expectOmegaOnEveryWorker(model);

  reloadWithComHeight("0.9");
  ASSERT_GT(std::abs(0.9 - model), 0.05) << "the override must differ from the model, or the checks prove nothing";
  expectOmegaOnEveryWorker(0.9);

  reloadWithComHeight("0");
  expectOmegaOnEveryWorker(model);

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::HasSubstr("dcm_terminal_cost.comHeight")))
      .Times(testing::AtLeast(1));
  log.StartCapturingLogs();
  reloadWithComHeight("-0.9");
  log.StopCapturingLogs();
  expectOmegaOnEveryWorker(model);
}

}  // namespace ocs2::humanoid
