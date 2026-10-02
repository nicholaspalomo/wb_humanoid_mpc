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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_oc/approximate_model/LinearQuadraticApproximator.h>
#include <ocs2_oc/oc_problem/OptimalControlProblemHelperFunction.h>

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "robot_core/ResourcePaths.h"
#include "support/ProblemFingerprint.h"

/**
 * The wiring of CoM + ACoM tracking into the centroidal MPC (humanoid_learning/acom/README.md, section 4).
 *
 * CoM + ACoM tracking used to be the top-level boolean `useComAndAcomTracking`, whose four effects lived in four places
 * and did not check one another. It is now the cost `com_and_acom_tracking_cost` of the `costs` list, and these tests
 * pin what listing it means: the problem it assembles is exactly the one the boolean assembled (Atlas ships it), the
 * base-pose block is zeroed whichever quadratic state cost carries Q, the terminal node keeps CoM and orientation
 * regulation beside the quadratic terminal cost, a task file still carrying the boolean is refused, and the contact
 * planner's heading model - the other consumer of the ACoM network - follows a hot reload.
 */
namespace ocs2::humanoid {
namespace {

// The base pose in the centroidal state x = [h_norm(6), p_base(3), euler_zyx(3), q_j], written out here rather than
// taken from ComAndAcomTrackingCost, so that a drift of the one definition shows up as a failure instead of moving the
// expectation with it.
constexpr Eigen::Index kBasePoseIndex = 6;
constexpr Eigen::Index kBasePoseDim = 6;
constexpr Eigen::Index kBasePositionZIndex = 8;
constexpr Eigen::Index kBasePitchIndex = 10;

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/** `content` with `from` replaced by `to` exactly once; fails the test when `from` does not occur exactly once. */
std::string replacedOnce(const std::string& content, absl::string_view from, absl::string_view to) {
  const std::string::size_type position = content.find(from);
  EXPECT_NE(position, std::string::npos) << "'" << from << "' not found";
  if (position == std::string::npos) return content;
  EXPECT_EQ(content.find(from, position + 1), std::string::npos) << "'" << from << "' occurs more than once";
  std::string result = content;
  result.replace(position, from.size(), std::string(to));
  return result;
}

/**
 * `content` with the diagonal entries (i,i), i = 6..11, of the top-level matrix `matrixName` set to non-zero values.
 * Atlas ships its base-pose weights at 0, so without this the zeroing these tests are about would be invisible.
 */
std::string withBasePoseWeights(const std::string& content, absl::string_view matrixName) {
  const std::string header = absl::StrCat("\n", matrixName, ":\n");
  const std::string::size_type blockStart = content.find(header);
  EXPECT_NE(blockStart, std::string::npos) << matrixName << " not found";
  if (blockStart == std::string::npos) return content;
  std::string result = content;
  for (Eigen::Index i = kBasePoseIndex; i < kBasePoseIndex + kBasePoseDim; ++i) {
    const std::string key = absl::StrCat("\"(", i, ",", i, ")\":");
    const std::string::size_type entry = result.find(key, blockStart + 1);
    EXPECT_NE(entry, std::string::npos) << matrixName << key << " not found";
    if (entry == std::string::npos) return content;
    const std::string::size_type lineEnd = result.find('\n', entry);
    result.replace(entry, lineEnd - entry, absl::StrCat(key, " ", 20 + 3 * i, "  # non-zero for testAcomWiring"));
  }
  return result;
}

std::string withoutAcomCost(const std::string& content) {
  return replacedOnce(content, "\n  - com_and_acom_tracking_cost\n", "\n");
}

// The assembled problem, compared exactly (support/ProblemFingerprint.h).
using test::Fingerprint;
using test::fingerprintOf;
using test::identical;
using test::setWalkingReferences;

}  // namespace

class AcomWiringTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shippedTaskFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml").value();
    referenceFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml").value();
    urdfFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf").value();
    shipped_ = readFile(shippedTaskFile_);
    ASSERT_NE(shipped_.find("\n  - com_and_acom_tracking_cost\n"), std::string::npos)
        << "Atlas, the robot with the one validated ACoM network, no longer lists com_and_acom_tracking_cost";
    tmpDir_ = (std::filesystem::path(testing::TempDir()) / "acom_wiring").string();
    std::filesystem::create_directories(tmpDir_);
  }

  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove_all(tmpDir_, ignored);
  }

  /** Writes a variant of the task file in a directory of its own, so that nothing beside it is shared by accident. */
  std::string writeTaskFile(absl::string_view name, const std::string& content) const {
    const std::filesystem::path directory = std::filesystem::path(tmpDir_) / std::string(name);
    std::filesystem::create_directories(directory);
    const std::string path = (directory / "task.yaml").string();
    std::ofstream out(path);
    out << content;
    return path;
  }

  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> create(const std::string& taskFile) const {
    return CentroidalMpcInterface::Create(taskFile, urdfFile_, referenceFile_);
  }

  std::string shippedTaskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::string shipped_;
  std::string tmpDir_;
};

/**
 * Findings A91/A103: `useComAndAcomTracking: true` became `com_and_acom_tracking_cost` in `costs`, and Atlas ships it.
 * The problem the name assembles must be bit for bit the one the boolean assembled. The boolean's four effects are
 * re-applied here by hand, from the code that used to apply them, to a problem built WITHOUT the name, and the two
 * problems are then compared term by term, derivative by derivative, at points along a walking schedule:
 *  - Q's base-pose block (6..11) zeroed in state_quadratic_cost (the old HumanoidCostConstraintFactory);
 *  - ComAndAcomTrackingCost on Q_com / Q_acom added to the state cost as "comAndAcomTrackingCost";
 *  - the procedural arm swing off (the old SwitchedModelReferenceManager::getDesiredState);
 *  - Q_final's block zeroed - absent here, as on the shipped Atlas, whose horizon ends on the DCM terminal cost.
 * Q's base-pose weights are made non-zero first: Atlas ships them at 0, which would hide the zeroing.
 */
TEST_F(AcomWiringTest, theNamedCostAssemblesExactlyTheProblemTheBooleanDid) {
  const std::string base = withBasePoseWeights(shipped_, "Q");
  const std::string listedFile = writeTaskFile("listed", base);
  const std::string unlistedFile = writeTaskFile("unlisted", withoutAcomCost(base));
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> listed = create(listedFile);
  ASSERT_TRUE(listed.ok()) << listed.status();
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> unlisted = create(unlistedFile);
  ASSERT_TRUE(unlisted.ok()) << unlisted.status();

  const Fingerprint listedFingerprint = fingerprintOf(**listed);
  // The comparison below is only meaningful if evaluating a problem twice gives the same numbers.
  ASSERT_TRUE(identical(fingerprintOf(**listed), listedFingerprint));
  // Positive control: the name changes the problem, so the comparison can fail.
  ASSERT_FALSE(identical(fingerprintOf(**unlisted), listedFingerprint));

  // The effects of `useComAndAcomTracking: true`, re-applied by hand to the problem built without the name.
  OptimalControlProblem& problem = (*unlisted)->getOptimalControlProblemRef();
  QuadraticStateInputCost& stateCost = problem.costPtr->get<QuadraticStateInputCost>("stateQuadraticCost");
  matrix_t Q;
  matrix_t R;
  matrix_t P;
  stateCost.getGains(Q, R, P);
  ASSERT_GT(Q.block(kBasePoseIndex, kBasePoseIndex, kBasePoseDim, kBasePoseDim).norm(), 0.0);
  Q.block(kBasePoseIndex, kBasePoseIndex, kBasePoseDim, kBasePoseDim).setZero();
  stateCost.setGains(Q, R, P);
  matrix_t Q_com(3, 3);
  matrix_t Q_acom(3, 3);
  loadData::loadEigenMatrix(unlistedFile, "Q_com", Q_com);
  loadData::loadEigenMatrix(unlistedFile, "Q_acom", Q_acom);
  absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> acom = ComAndAcomTrackingCost::Create(
      Q_com, Q_acom, (*unlisted)->getPinocchioInterface(), (*unlisted)->getCentroidalModelInfo(), (*unlisted)->modelSettings().robotName);
  ASSERT_TRUE(acom.ok()) << acom.status();
  problem.stateCostPtr->add("comAndAcomTrackingCost", *std::move(acom));
  ASSERT_TRUE((*unlisted)->getSwitchedModelReferenceManagerPtr()->isArmSwingReferenceActive());
  (*unlisted)->getSwitchedModelReferenceManagerPtr()->setArmSwingReferenceActive(false);
  ASSERT_TRUE(problem.finalCostPtr->getTermNameMap().count("dcmTerminalCost") > 0) << "the shipped Atlas ends on the DCM cost";

  EXPECT_TRUE(identical(fingerprintOf(**unlisted), listedFingerprint));
  EXPECT_FALSE((*listed)->getSwitchedModelReferenceManagerPtr()->isArmSwingReferenceActive());
}

/**
 * Finding A91/A103, scenario (a): with state_input_quadratic_cost in place of state_quadratic_cost the ACoM cost used to
 * be dropped - it was only added inside the state_quadratic_cost branch - while Q_final was still zeroed and the arm
 * swing still switched off. The name now works there: the cost is added, and the Q this cost carries loses its
 * base-pose block like state_quadratic_cost's does.
 */
TEST_F(AcomWiringTest, theNamedCostReplacesTheBasePoseInWhicheverQuadraticStateCostCarriesQ) {
  const std::string base =
      replacedOnce(withBasePoseWeights(shipped_, "Q"), "\n  - state_quadratic_cost\n", "\n  - state_input_quadratic_cost\n");
  const std::string listedFile = writeTaskFile("stateInputListed", base);
  const std::string unlistedFile = writeTaskFile("stateInputUnlisted", withoutAcomCost(base));

  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> listed = create(listedFile);
  ASSERT_TRUE(listed.ok()) << listed.status();
  const OptimalControlProblem& problem = (*listed)->getOptimalControlProblem();
  EXPECT_EQ(problem.costPtr->getTermNameMap().count("stateQuadraticCost"), 0u);
  EXPECT_EQ(problem.stateCostPtr->getTermNameMap().count(std::string(ComAndAcomTrackingCost::kRunningTermName)), 1u)
      << "the ACoM cost was dropped beside state_input_quadratic_cost";
  const Eigen::Index stateDim = (*listed)->getInitialState().size();
  matrix_t fileQ(stateDim, stateDim);
  loadData::loadEigenMatrix(listedFile, "Q", fileQ);
  ASSERT_GT(fileQ.block(kBasePoseIndex, kBasePoseIndex, kBasePoseDim, kBasePoseDim).norm(), 0.0);
  matrix_t Q;
  matrix_t R;
  matrix_t P;
  problem.costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost").getGains(Q, R, P);
  EXPECT_TRUE(Q.block(kBasePoseIndex, kBasePoseIndex, kBasePoseDim, kBasePoseDim).isZero(0.0));
  matrix_t expected = fileQ;
  expected.block(kBasePoseIndex, kBasePoseIndex, kBasePoseDim, kBasePoseDim).setZero();
  EXPECT_TRUE(Q == expected) << "the rest of Q must be the file's";

  // Positive control: without the name the same file's base-pose block reaches the cost untouched.
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> unlisted = create(unlistedFile);
  ASSERT_TRUE(unlisted.ok()) << unlisted.status();
  (*unlisted)->getOptimalControlProblem().costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost").getGains(Q, R, P);
  EXPECT_TRUE(Q == fileQ);
  EXPECT_EQ(
      (*unlisted)->getOptimalControlProblem().stateCostPtr->getTermNameMap().count(std::string(ComAndAcomTrackingCost::kRunningTermName)),
      0u);
}

/**
 * Finding A115: with the quadratic terminal cost, zeroing Q_final's base-pose block used to leave the terminal node with
 * no CoM, height or orientation regulation at all - the ACoM cost was added to the running state cost only - although
 * terminalCostScaling makes the terminal node the heaviest of the horizon. The terminal node now carries its own
 * ComAndAcomTrackingCost, weighted by terminalCostScaling like Q_final.
 */
TEST_F(AcomWiringTest, theTerminalNodeIsRegulatedInCoMAndAcomCoordinatesBesideTheQuadraticTerminalCost) {
  const std::string content = replacedOnce(withBasePoseWeights(shipped_, "Q_final"), "\n  - dcm_terminal_cost\n", "\n  - terminal_cost\n");
  const std::string taskFile = writeTaskFile("quadraticTerminal", content);
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = create(taskFile);
  ASSERT_TRUE(created.ok()) << created.status();
  CentroidalMpcInterface& interface = **created;
  OptimalControlProblem& problem = interface.getOptimalControlProblemRef();
  ASSERT_EQ(problem.finalCostPtr->getTermNameMap().count("terminalCost"), 1u);
  ASSERT_EQ(problem.finalCostPtr->getTermNameMap().count(std::string(ComAndAcomTrackingCost::kTerminalTermName)), 1u)
      << "the terminal node has no CoM + ACoM regulation";

  // Q_final's base-pose block is zeroed, as before...
  const Eigen::Index stateDim = interface.getInitialState().size();
  matrix_t fileQFinal(stateDim, stateDim);
  loadData::loadEigenMatrix(taskFile, "Q_final", fileQFinal);
  ASSERT_GT(fileQFinal.block(kBasePoseIndex, kBasePoseIndex, kBasePoseDim, kBasePoseDim).norm(), 0.0);
  matrix_t QFinal;
  problem.finalCostPtr->get<QuadraticStateCost>("terminalCost").getGains(QFinal);
  EXPECT_TRUE(QFinal.block(kBasePoseIndex, kBasePoseIndex, kBasePoseDim, kBasePoseDim).isZero(0.0));
  // ...and the terminal instance is weighted like Q_final: by terminalCostScaling.
  scalar_t terminalCostScaling = 0.0;
  loadData::loadCppDataType(taskFile, "terminalCostScaling", terminalCostScaling);
  ASSERT_NE(terminalCostScaling, 1.0) << "a scaling of 1 would not tell the running weights from the terminal ones";
  matrix_t Q_com(3, 3);
  matrix_t Q_acom(3, 3);
  loadData::loadEigenMatrix(taskFile, "Q_com", Q_com);
  loadData::loadEigenMatrix(taskFile, "Q_acom", Q_acom);
  const ComAndAcomTrackingCost& terminal =
      problem.finalCostPtr->get<ComAndAcomTrackingCost>(std::string(ComAndAcomTrackingCost::kTerminalTermName));
  EXPECT_TRUE(terminal.getQCom().isApprox(terminalCostScaling * Q_com, 1e-14));
  EXPECT_TRUE(terminal.getQAcom().isApprox(terminalCostScaling * Q_acom, 1e-14));

  // The property itself: the terminal cost pulls on the base height and pitch...
  setWalkingReferences(interface);
  vector_t state = interface.getInitialState();
  state(kBasePositionZIndex) += 0.03;
  state(kBasePitchIndex) += 0.1;
  const scalar_t finalTime = 1.5;
  MultiplierCollection multipliers;
  initializeFinalMultiplierCollection(problem, finalTime, multipliers);
  const ModelData withTerminalAcom = approximateFinalLQ(problem, finalTime, state, multipliers);
  EXPECT_GT(std::abs(withTerminalAcom.cost.dfdx(kBasePitchIndex)), 1.0);
  EXPECT_GT(std::abs(withTerminalAcom.cost.dfdx(kBasePositionZIndex)), 1.0);

  // ...which, positive control, nothing else in the terminal cost does: without the terminal instance both vanish.
  OptimalControlProblem withoutTerminalAcom(problem);
  ASSERT_TRUE(withoutTerminalAcom.finalCostPtr->erase(std::string(ComAndAcomTrackingCost::kTerminalTermName)));
  const ModelData bare = approximateFinalLQ(withoutTerminalAcom, finalTime, state, multipliers);
  EXPECT_EQ(bare.cost.dfdx(kBasePitchIndex), 0.0);
  EXPECT_EQ(bare.cost.dfdx(kBasePositionZIndex), 0.0);
}

/**
 * A task file that still carries the retired boolean is refused at start-up with a Status naming the cost that replaced
 * it, so that a stale file cannot silently run a different formulation - whatever the key's value.
 */
TEST_F(AcomWiringTest, aTaskFileStillCarryingTheRetiredBooleanIsRefusedAtStartUp) {
  for (const char* value : {"true", "false"}) {
    const std::string taskFile =
        writeTaskFile(absl::StrCat("retired_", value), absl::StrCat("useComAndAcomTracking: ", value, "\n", shipped_));
    const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = create(taskFile);
    ASSERT_FALSE(created.ok()) << "useComAndAcomTracking: " << value << " was accepted";
    EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(created.status().message(), "com_and_acom_tracking_cost")) << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), "useComAndAcomTracking")) << created.status();
  }
}

/**
 * Listing the cost without the weights it reads - the state the G1 and SA01 task files are in, which carry no Q_com /
 * Q_acom - is a configuration error. Create() reports it as a Status naming the missing key; the task-file loader's
 * exception used to escape the Status-returning set-up instead.
 */
TEST_F(AcomWiringTest, aListedCostWithoutItsWeightsIsRefusedNamingTheMissingKey) {
  for (const char* key : {"Q_com", "Q_acom"}) {
    const std::string taskFile = writeTaskFile(
        absl::StrCat("without_", key), replacedOnce(shipped_, absl::StrCat("\n", key, ":\n"), absl::StrCat("\n", key, "_unread:\n")));
    absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = absl::UnknownError("not created");
    ASSERT_NO_THROW(created = create(taskFile)) << key;
    ASSERT_FALSE(created.ok()) << "the cost was built without " << key;
    EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
    // Quoted, because the advice in the message names both matrices whichever one is missing.
    EXPECT_TRUE(absl::StrContains(created.status().message(), absl::StrCat("'", key, "'"))) << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), "com_and_acom_tracking_cost")) << created.status();
  }
}

/**
 * Finding AC4: the contact planner's heading model reads the whole-body heading off the ACoM network. The evaluator used
 * to be installed only at start-up, and only when the INITIAL configuration listed heading_double_integrator, although
 * that list is hot-reloadable: switching the heading model on by a reload left the planner taking the base yaw for the
 * heading, while a restart with the same file gave the ACoM heading. The reference manager is built here the way the
 * interface builds it, without the rest of the MPC.
 */
class HeadingModelReloadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml").value();
    referenceFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml").value();
    urdfFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf").value();

    modelSettings_ = std::make_unique<ModelSettings>(taskFile_, urdfFile_, "testAcomWiring", /*verbose=*/false);
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(
        createCustomPinocchioInterface(taskFile_, urdfFile_, *modelSettings_, /*scaleTotalMass=*/false));
    info_ = centroidal_model::createCentroidalModelInfo(
        *pinocchioInterface_, centroidal_model::loadCentroidalType(taskFile_),
        centroidal_model::loadDefaultJointState(pinocchioInterface_->getModel().nq - 6, referenceFile_), modelSettings_->contactNames3DoF,
        modelSettings_->contactNames6DoF);
    robotModel_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, info_);
    initialState_.setZero(info_.stateDim);
    loadData::loadEigenMatrix(taskFile_, "initialState", initialState_);

    // The shipped planner configuration with the heading model on, its model parameters derived as the interface
    // derives them, and the same configuration with the heading model off.
    absl::StatusOr<ContactPlanningConfig> loaded = loadContactPlanningConfigStatus(
        resolveContactPlanningConfigFile(taskFile_), "contact_planning.", /*verbose=*/false, /*validate=*/false);
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    headingOn_ = *std::move(loaded);
    headingOn_.setHeadingModel(true);
    ContactPlanningGroundParameters ground;
    PinocchioInterface pinocchioForDerivation(*pinocchioInterface_);
    deriveContactPlanningModelParameters(pinocchioForDerivation, *robotModel_, initialState_, modelSettings_->contactParentJointNames,
                                         ground, headingOn_.shared.gravity, headingOn_.stepWidth.nominalStepWidth)
        .applyTo(headingOn_);
    ASSERT_TRUE(headingOn_.validateStatus().ok()) << headingOn_.validateStatus();
    headingOff_ = headingOn_;
    headingOff_.setHeadingModel(false);
    ASSERT_TRUE(headingOff_.validateStatus().ok()) << headingOff_.validateStatus();
    ASSERT_TRUE(headingOn_.usesHeadingModel());
    ASSERT_FALSE(headingOff_.usesHeadingModel());
  }

  std::shared_ptr<ContactPlanningReferenceManager> makeReferenceManager(const MpcRobotModelBase<scalar_t>& robotModel,
                                                                        const ContactPlanningConfig& config) const {
    std::shared_ptr<SwingTrajectoryPlanner> swingPlanner = std::make_shared<SwingTrajectoryPlanner>(
        loadSwingTrajectorySettings(taskFile_, "swing_trajectory_config", /*verbose=*/false), N_CONTACTS);
    absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>> manager =
        ContactPlanningReferenceManager::Create(GaitSchedule::loadGaitSchedule(referenceFile_, *modelSettings_, /*verbose=*/false),
                                                std::move(swingPlanner), *pinocchioInterface_, robotModel, config);
    EXPECT_TRUE(manager.ok()) << manager.status();
    return manager.ok() ? *manager : nullptr;
  }

  /** The initial state with every joint moved, so that the whole-body heading and the base yaw differ. */
  vector_t bentState() const {
    vector_t state = initialState_;
    for (Eigen::Index i = 12; i < state.size(); ++i) state(i) += 0.3 * std::sin(1.7 * static_cast<scalar_t>(i));
    return state;
  }

  std::string taskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  CentroidalModelInfo info_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> robotModel_;
  vector_t initialState_;
  ContactPlanningConfig headingOn_;
  ContactPlanningConfig headingOff_;
};

TEST_F(HeadingModelReloadTest, aReloadThatSwitchesTheHeadingModelOnInstallsTheAcomEvaluator) {
  const std::shared_ptr<ContactPlanningReferenceManager> manager = makeReferenceManager(*robotModel_, headingOff_);
  ASSERT_TRUE(manager->loadHeadingModelEvaluator().ok());
  EXPECT_FALSE(manager->hasAngularCenterOfMass()) << "nothing needs the network while the heading model is off";

  const vector_t state = bentState();
  const scalar_t baseYaw = robotModel_->getBaseOrientationEulerZYX(state)(0);
  absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
      AngularCenterOfMass::Create(modelSettings_->robotName, modelSettings_->mpcModelJointNames);
  ASSERT_TRUE(acom.ok()) << acom.status();
  const scalar_t acomYaw = (*acom)->computeAcomOrientation(robotModel_->getGeneralizedCoordinates(state))(0);
  // Positive control: at this state the two headings are different quantities, so the check below can fail.
  ASSERT_GT(std::abs(acomYaw - baseYaw), 1e-3);
  EXPECT_EQ(manager->computeHeading(state), baseYaw);

  ASSERT_TRUE(manager->setConfigStatus(headingOn_).ok());
  ASSERT_TRUE(manager->getConfig().usesHeadingModel());
  ASSERT_TRUE(manager->hasAngularCenterOfMass()) << "the reload switched the heading model on without its evaluator";
  EXPECT_NEAR(manager->computeHeading(state), acomYaw, 1e-12);

  // Switching it off again keeps the evaluator, which only the heading model reads.
  ASSERT_TRUE(manager->setConfigStatus(headingOff_).ok());
  EXPECT_TRUE(manager->hasAngularCenterOfMass());
}

TEST_F(HeadingModelReloadTest, theEvaluatorIsLeftToTheStatusReturningSetUpUntilItHasRun) {
  // While the interface is being constructed, the reference manager's Create() and ContactPlannerModule::Create() hand
  // the configuration over through setConfigStatus() before the interface has called loadHeadingModelEvaluator(); the
  // evaluator is left to that one call.
  const std::shared_ptr<ContactPlanningReferenceManager> manager = makeReferenceManager(*robotModel_, headingOff_);
  ASSERT_TRUE(manager->setConfigStatus(headingOn_).ok());
  EXPECT_FALSE(manager->hasAngularCenterOfMass());
  ASSERT_TRUE(manager->loadHeadingModelEvaluator().ok());
  EXPECT_TRUE(manager->hasAngularCenterOfMass());
}

TEST_F(HeadingModelReloadTest, aNetworkTrainedOnOtherJointsRefusesTheReloadAndTheStart) {
  // The MPC model's joints relabeled - two of them swapped - so that the network no longer matches them name by name.
  ModelSettings permuted(taskFile_, urdfFile_, "testAcomWiring", /*verbose=*/false);
  ASSERT_GE(permuted.mpcModelJointNames.size(), 2u);
  std::swap(permuted.mpcModelJointNames[0], permuted.mpcModelJointNames[1]);
  const CentroidalMpcRobotModel<scalar_t> permutedModel(permuted, *pinocchioInterface_, info_);

  const std::shared_ptr<ContactPlanningReferenceManager> manager = makeReferenceManager(permutedModel, headingOff_);
  ASSERT_TRUE(manager->loadHeadingModelEvaluator().ok()) << "with the heading model off the network is not needed";
  const absl::Status refused = manager->setConfigStatus(headingOn_);
  EXPECT_EQ(refused.code(), absl::StatusCode::kFailedPrecondition) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), permuted.mpcModelJointNames[0])) << refused;
  EXPECT_FALSE(manager->getConfig().usesHeadingModel()) << "a refused reload must leave the running configuration in place";
  EXPECT_FALSE(manager->hasAngularCenterOfMass());

  const std::shared_ptr<ContactPlanningReferenceManager> startedOn = makeReferenceManager(permutedModel, headingOn_);
  EXPECT_EQ(startedOn->loadHeadingModelEvaluator().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_FALSE(startedOn->hasAngularCenterOfMass());
}

TEST_F(HeadingModelReloadTest, aRobotWithoutANetworkKeepsTheBaseYawAsItsHeading) {
  ModelSettings unregistered(taskFile_, urdfFile_, "testAcomWiring", /*verbose=*/false);
  unregistered.robotName = "robot_without_an_acom_network";
  const CentroidalMpcRobotModel<scalar_t> model(unregistered, *pinocchioInterface_, info_);
  const std::shared_ptr<ContactPlanningReferenceManager> manager = makeReferenceManager(model, headingOff_);
  ASSERT_TRUE(manager->loadHeadingModelEvaluator().ok());
  ASSERT_TRUE(manager->setConfigStatus(headingOn_).ok()) << "no network is a warning, not an error";
  EXPECT_TRUE(manager->getConfig().usesHeadingModel());
  EXPECT_FALSE(manager->hasAngularCenterOfMass());
  const vector_t state = bentState();
  EXPECT_EQ(manager->computeHeading(state), model.getBaseOrientationEulerZYX(state)(0));
}

}  // namespace ocs2::humanoid
