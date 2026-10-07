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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/cost/QuadraticStateCost.h"
#include "ocs2_core/cost/QuadraticStateInputCost.h"
#include "ocs2_core/misc/Collection.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"
#include "ocs2_mpc/MPC_BASE.h"
#include "ocs2_sqp/SqpMpc.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"
#include "humanoid_centroidal_mpc/constraint/ZeroVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcParameterUpdater.h"
#include "humanoid_centroidal_mpc/parameter_update/BasisInputsCostApplier.h"
#include "humanoid_common_mpc/HumanoidPreComputation.h"
#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/contact_planning/ContactPlanningFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
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
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/locomotion_heuristics/BasePoseHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/joint_value.nproto.h"
#include "humanoid_mpc_config/mpc_parameter_update.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "support/TypedConfigFiles.h"

/**
 * MpcParameterUpdaterModule on the DRC Atlas MPC: what a reload of the typed task file and of the contact planner's file
 * applies to the running problem - its RELOAD_HOT fields, as that file's values (a reload is the file, so a block it
 * leaves out is its defaults) - what it reports and leaves for the next start, and what it refuses by
 * field while the rest of the file still applies. Each case edits the typed configuration the shipped files load into,
 * so every other value is the one the robot runs.
 */
namespace ocs2::humanoid {
namespace {

/** Whether `collection` carries a term named `name`. */
template <typename Term>
bool hasTerm(const Collection<Term>& collection, const std::string& name) {
  size_t index = 0;
  return collection.getTermIndex(name, index);
}

/** The entry of `joint` in `joints` (a block of joint values by name); a missing one fails the test and is appended. */
double& jointValue(std::vector<mpc_config::JointValue>& joints, absl::string_view joint) {
  for (mpc_config::JointValue& entry : joints) {
    if (entry.joint == joint) return entry.value;
  }
  ADD_FAILURE() << joint << " has no entry";
  joints.push_back(mpc_config::JointValue{.joint = std::string(joint), .value = 0.0});
  return joints.back().value;
}

/** `list` without `entry`; fails the test when it has none. */
std::vector<std::string> without(std::vector<std::string> list, absl::string_view entry) {
  const std::vector<std::string>::iterator found = std::find(list.begin(), list.end(), entry);
  EXPECT_NE(found, list.end()) << entry << " is not listed";
  if (found != list.end()) list.erase(found);
  return list;
}

/** `list` with `from` replaced by `to`; fails the test when it does not list `from`. */
std::vector<std::string> replaced(std::vector<std::string> list, absl::string_view from, absl::string_view to) {
  const std::vector<std::string>::iterator found = std::find(list.begin(), list.end(), from);
  EXPECT_NE(found, list.end()) << from << " is not listed";
  if (found != list.end()) *found = std::string(to);
  return list;
}

// The base pose in the centroidal state x = [h_norm(6), p_base(3), euler_zyx(3), q_j], written out rather than taken
// from ComAndAcomTrackingCost so that a drift of the one definition fails here instead of moving the expectation.
constexpr Eigen::Index kBasePoseIndex = 6;
constexpr Eigen::Index kBasePoseDim = 6;
// The first joint of the centroidal state.
constexpr Eigen::Index kFirstJointIndex = 12;

/** The base-pose block of a state weight matrix. */
matrix_t basePoseBlock(const matrix_t& Q) {
  return Q.block(kBasePoseIndex, kBasePoseIndex, kBasePoseDim, kBasePoseDim);
}

/** `weights` with every base-pose weight non-zero and distinct, from `offset`. */
void setBasePoseWeights(mpc_config::StateWeights& weights, scalar_t offset) {
  weights.base_position = mpc_config::Xyz{.x = offset + 1.0, .y = offset + 2.0, .z = offset + 3.0};
  weights.base_orientation = mpc_config::YawPitchRoll{.yaw = offset + 4.0, .pitch = offset + 5.0, .roll = offset + 6.0};
}

/** The contact-implicit formulation: no hard constraint, its three terms and the soft normal velocity. */
void selectContactImplicitFormulation(mpc_config::TaskFile& task) {
  task.hard_constraints.clear();
  task.soft_constraints = {"joint_limits",        "foot_collision",    "contact_wrench_cone", "normal_velocity", "contact_complementarity",
                           "force_weighted_slip", "ground_penetration"};
}

/** The parameters of the first penalty of a soft constraint. */
template <typename SoftConstraint>
vector_t penaltyParameters(SoftConstraint& softConstraint) {
  vector_t parameters;
  std::vector<std::unique_ptr<augmented::AugmentedPenaltyBase>>& penalties = softConstraint.getPenalty().getPenaltyPtrArray();
  EXPECT_FALSE(penalties.empty());
  if (!penalties.empty()) penalties.front()->getParameters(parameters);
  return parameters;
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

/** The one ground every contact-implicit term of every worker is on, or NaN when they disagree or there is none. */
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

/** The base-pose offset the layer's heuristics give a forward walk at `forwardSpeed`. */
BasePoseOffset basePoseOffsetAtForwardSpeed(const LocomotionHeuristicLayer& layer, scalar_t forwardSpeed) {
  BasePoseHeuristicContext context;
  context.commandedVelocityInBaseFrame = vector2_t(forwardSpeed, 0.0);
  return layer.basePoseOffset(context);
}

/** An MPC built from a configuration, with the SQP solver a node runs it with. */
struct Built {
  std::unique_ptr<CentroidalMpcInterface> interface;
  std::unique_ptr<SqpMpc> mpc;

  SqpSolver& solver() const { return dynamic_cast<SqpSolver&>(*mpc->getSolverPtr()); }
  OptimalControlProblem& firstProblem() const { return solver().getOcpDefinitions().front(); }
  const std::vector<std::string>& contactNames() const { return interface->modelSettings().contactNames; }
  /** The name of the first joint of the state. */
  const std::string& firstJoint() const { return interface->modelSettings().mpcModelJointNames.front(); }
};

/** The MPC of `config` and the Atlas URDF; fails the test and has no interface when it is refused. */
Built build(const CentroidalMpcConfig& config, const std::string& urdfFile) {
  Built built;
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = CentroidalMpcInterface::Create(config, urdfFile);
  EXPECT_TRUE(created.ok()) << created.status();
  if (!created.ok()) return built;
  built.interface = *std::move(created);
  built.mpc = std::make_unique<SqpMpc>(built.interface->mpcSettings(), built.interface->sqpSettings(),
                                       built.interface->getOptimalControlProblem(), built.interface->getInitializer());
  built.mpc->getSolverPtr()->setReferenceManager(built.interface->getReferenceManagerPtr());
  return built;
}

/** What makeUpdater() watches; every path empty: nothing is watched. */
struct WatchedFiles {
  std::string taskFile;
  std::string referenceFile;
  std::string contactPlanningFile;
};

/** The task file `path` watched, alone. */
WatchedFiles watchingTaskFile(const std::string& path) {
  WatchedFiles watched;
  watched.taskFile = path;
  return watched;
}

/** The reference file `path` watched, alone. */
WatchedFiles watchingReferenceFile(const std::string& path) {
  WatchedFiles watched;
  watched.referenceFile = path;
  return watched;
}

/** The contact planner's file `path` watched, alone. */
WatchedFiles watchingContactPlanningFile(const std::string& path) {
  WatchedFiles watched;
  watched.contactPlanningFile = path;
  return watched;
}

/**
 * The updater of `built`, with the appliers makeCentroidalMpcParameterUpdater() wires (centroidalHotFieldAppliers()), in
 * `order`, but without the planner, on the reference manager `referenceManager` (nullptr: none), watching `watched`;
 * fails the test and is null when it is refused.
 */
/** In which order an updater of makeUpdater() runs the centroidal appliers. */
enum class ApplierOrder {
  kListed,
  kReversed,
};

std::unique_ptr<MpcParameterUpdaterModule> makeUpdater(const Built& built,
                                                       SwitchedModelReferenceManager* absl_nullable referenceManager = nullptr,
                                                       const WatchedFiles& watched = WatchedFiles{},
                                                       ApplierOrder order = ApplierOrder::kListed) {
  MpcParameterUpdaterModule::Options options;
  options.taskFile = watched.taskFile;
  options.referenceFile = watched.referenceFile;
  options.contactPlanningFile = watched.contactPlanningFile;
  options.runningTask = built.interface->config().task;
  options.layout = stateInputLayout(built.interface->modelSettings(), StateInputLayout::Mpc::kCentroidal);
  options.inputDim = built.interface->getEffectiveMpcRobotModel().getInputDim();
  options.referenceManager = referenceManager;
  absl::StatusOr<std::vector<std::unique_ptr<HotFieldApplier>>> appliers = centroidalHotFieldAppliers(*built.interface);
  EXPECT_TRUE(appliers.ok()) << appliers.status();
  if (!appliers.ok()) return nullptr;
  options.appliers = *std::move(appliers);
  if (order == ApplierOrder::kReversed) std::reverse(options.appliers.begin(), options.appliers.end());
  absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> created =
      MpcParameterUpdaterModule::Create(built.mpc.get(), std::move(options));
  EXPECT_TRUE(created.ok()) << created.status();
  return created.ok() ? *std::move(created) : nullptr;
}

/** Enqueues `task` (and `contactPlanning`) as the GUI's update and runs the one pre-solve hook that applies it. */
void reload(MpcParameterUpdaterModule& updater,
            const Built& built,
            const mpc_config::TaskFile& task,
            std::optional<mpc_config::ContactPlanningFile> contactPlanning = std::nullopt) {
  mpc_config::MpcParameterUpdate update;
  update.task = task;
  update.contact_planning = std::move(contactPlanning);
  updater.enqueueParameterUpdate(update);
  updater.preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, built.interface->getInitialState(), *built.interface->getReferenceManagerPtr());
}

/** Drives `updater` past its ~1 Hz poll of the watched files (every 100th pre-solve hook). */
void runFileWatch(MpcParameterUpdaterModule& updater, const Built& built) {
  for (size_t i = 0; i < 101; ++i) {
    updater.preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, built.interface->getInitialState(),
                         *built.interface->getReferenceManagerPtr());
  }
}

/** Writes `text` to `path` with a modification time `seconds` after the test started, so that every write is seen. */
void writeWatched(const std::string& path, absl::string_view text, int seconds) {
  ASSERT_TRUE(writeTextFile(path, text).ok());
  std::error_code error;
  std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(seconds), error);
  ASSERT_FALSE(error) << error.message();
}

/** The term of `ocp` that carries Q: stateInputQuadraticCost, or stateQuadraticCost (the shipped Atlas); empty: none. */
std::string stateCostTerm(const OptimalControlProblem& ocp) {
  for (const std::string name : {"stateInputQuadraticCost", "stateQuadraticCost"}) {
    if (hasTerm(*ocp.costPtr, name)) return name;
  }
  return std::string();
}

/** The term of `ocp` that carries R: stateInputQuadraticCost, or inputQuadraticCost (the shipped Atlas); empty: none. */
std::string inputCostTerm(const OptimalControlProblem& ocp) {
  for (const std::string name : {"stateInputQuadraticCost", "inputQuadraticCost"}) {
    if (hasTerm(*ocp.costPtr, name)) return name;
  }
  return std::string();
}

/** Q of the term `term` of `ocp`. */
matrix_t stateWeightsOf(const OptimalControlProblem& ocp, const std::string& term) {
  matrix_t Q;
  matrix_t R;
  matrix_t P;
  ocp.costPtr->get<QuadraticStateInputCost>(term).getGains(Q, R, P);
  return Q;
}

/** R of the term `term` of `ocp`. */
matrix_t inputWeightsOf(const OptimalControlProblem& ocp, const std::string& term) {
  matrix_t Q;
  matrix_t R;
  matrix_t P;
  ocp.costPtr->get<QuadraticStateInputCost>(term).getGains(Q, R, P);
  return R;
}

/** The quadratic terminal cost's Q_final of `ocp`. */
matrix_t finalStateWeightsOf(const OptimalControlProblem& ocp) {
  matrix_t Q_final;
  ocp.finalCostPtr->get<QuadraticStateCost>("terminalCost").getGains(Q_final);
  return Q_final;
}

/** The shipped DRC Atlas MPC and its typed configuration, built once per test. */
class MpcParameterUpdaterModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    files_ = atlasFiles();
    ASSERT_FALSE(files_.taskFile.empty() || files_.referenceFile.empty() || files_.urdfFile.empty());
    absl::StatusOr<CentroidalMpcConfig> config = loadConfigOf(files_);
    ASSERT_TRUE(config.ok()) << config.status();
    shipped_ = *std::move(config);
    atlas_ = build(shipped_, files_.urdfFile);
    ASSERT_NE(atlas_.interface, nullptr);
  }

  /** The shipped task file, to edit. */
  mpc_config::TaskFile shippedTask() const { return shipped_.task; }

  /** A directory of the test's own, emptied. */
  static std::string freshDirectory(absl::string_view name) {
    const std::filesystem::path directory = std::filesystem::path(testing::TempDir()) / std::string(name);
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    std::filesystem::create_directories(directory, error);
    EXPECT_FALSE(error) << error.message();
    return directory.string();
  }

  CentroidalRobotFiles files_;
  CentroidalMpcConfig shipped_;
  Built atlas_;
};

/******************************************************************************************************/
// Construction
/******************************************************************************************************/

TEST_F(MpcParameterUpdaterModuleTest, TheNodeWiringAppliesTheCentroidalFieldsAndANullApplierIsRefused) {
  // The robot's textprotos as a node wires them: the updater applies exactly the centroidal formulation's hot fields.
  const absl::StatusOr<std::shared_ptr<MpcParameterUpdaterModule>> wired = makeCentroidalMpcParameterUpdater(
      atlas_.mpc.get(), *atlas_.interface, files_.taskFile, files_.referenceFile, /*referenceFileReloaders=*/{});
  ASSERT_TRUE(wired.ok()) << wired.status();
  EXPECT_EQ((*wired)->appliedFields(), centroidalHotFieldNames());
  // An applier that is not there is refused by its index.
  MpcParameterUpdaterModule::Options options;
  options.layout = stateInputLayout(atlas_.interface->modelSettings(), StateInputLayout::Mpc::kCentroidal);
  options.inputDim = atlas_.interface->getEffectiveMpcRobotModel().getInputDim();
  options.appliers.push_back(std::unique_ptr<HotFieldApplier>());
  const absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> withNull =
      MpcParameterUpdaterModule::Create(atlas_.mpc.get(), std::move(options));
  EXPECT_EQ(withNull.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(withNull.status().message(), testing::HasSubstr("applier 0"));
}

TEST_F(MpcParameterUpdaterModuleTest, BasisCostTransformRejectsMismatchedInputDim) {
  // The R of the task file is indexed in wrench space, so the only sane inputDim for an updater carrying a basis-space
  // transform is the transform's basis-space dimension: the wrench-space one is exactly the mistake this guards against.
  std::optional<BasisInputsCostTransformConfig> transform = atlas_.interface->getBasisInputsCostTransformConfig();
  if (!transform.has_value()) GTEST_FAIL() << "the shipped Atlas runs basis-vector contact inputs";
  for (const size_t wrongInputDim : {atlas_.interface->getWrenchInputDim(), transform->basisInputDim() + 1}) {
    const absl::StatusOr<std::unique_ptr<BasisInputsCostApplier>> created = BasisInputsCostApplier::Create(transform, wrongInputDim);
    ASSERT_FALSE(created.ok()) << "inputDim " << wrongInputDim << " was accepted";
    EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(created.status().message(), testing::HasSubstr("inputDim"));
  }
  EXPECT_NE(makeUpdater(atlas_), nullptr) << "positive control: the basis-space dimension is accepted";
}

/******************************************************************************************************/
// The weights of the costs
/******************************************************************************************************/

TEST_F(MpcParameterUpdaterModuleTest, StateWeightsAreAppliedInPlaceToEveryWorker) {
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  const std::string qTerm = stateCostTerm(atlas_.firstProblem());
  ASSERT_FALSE(qTerm.empty()) << "the problem carries no quadratic state cost";
  const matrix_t shippedQ = stateWeightsOf(atlas_.firstProblem(), qTerm);
  ASSERT_GT(shippedQ.norm(), 0.0) << "the shipped Q is zero, so doubling its scaling would show nothing";

  mpc_config::TaskFile task = shippedTask();
  task.state_weights.scaling *= 2.0;
  reload(*updater, atlas_, task);
  ASSERT_GT(atlas_.solver().getOcpDefinitions().size(), 1U) << "the check has to cover more than one worker's problem";
  for (const OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
    EXPECT_TRUE(stateWeightsOf(ocp, qTerm).isApprox(2.0 * shippedQ, 1.0e-12)) << "the doubled scaling did not reach every worker";
  }
}

TEST_F(MpcParameterUpdaterModuleTest, AReloadOfTheShippedFileReproducesTheStartUpProblemAndReportsNothing) {
  // The payload is the file (humanoid_nmpc/humanoid_mpc_config/README.md, "Live updates"): the shipped file reloaded
  // writes back the very weights the factory built, and no field differs from the file the MPC started with. A term the
  // problem does not carry (the soft zero_velocity weight of a hard zero_velocity, the cones the basis-vector inputs do
  // not build) is not a failure.
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  const OptimalControlProblem& ocp = atlas_.firstProblem();
  const std::string qTerm = stateCostTerm(ocp);
  const std::string rTerm = inputCostTerm(ocp);
  ASSERT_FALSE(qTerm.empty() || rTerm.empty());
  const matrix_t shippedQ = stateWeightsOf(ocp, qTerm);
  const matrix_t shippedR = inputWeightsOf(ocp, rTerm);

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_, testing::_)).Times(0);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_, testing::_)).Times(0);
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr("Successfully applied in-place parameter updates")))
      .Times(1);
  log.StartCapturingLogs();
  reload(*updater, atlas_, shippedTask(), shipped_.contactPlanning);
  log.StopCapturingLogs();
  EXPECT_TRUE(stateWeightsOf(ocp, qTerm) == shippedQ) << "a reload of the shipped file changed Q";
  EXPECT_TRUE(inputWeightsOf(ocp, rTerm).isApprox(shippedR, 1.0e-12)) << "a reload of the shipped file changed R";
}

TEST_F(MpcParameterUpdaterModuleTest, ARefusedBlockIsReportedByItsFieldAndTheRestOfTheReloadApplies) {
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  const std::string qTerm = stateCostTerm(atlas_.firstProblem());
  ASSERT_FALSE(qTerm.empty());
  const matrix_t shippedQ = stateWeightsOf(atlas_.firstProblem(), qTerm);
  const size_t runningIterations = atlas_.solver().getSettings().sqpIteration;

  // A state weight that is not finite, and a joint block without one of the robot's joints: state_weights does not
  // convert. multiple_shooting.sqp_iteration shows that the rest of the file still applies.
  mpc_config::TaskFile task = shippedTask();
  jointValue(task.state_weights.joint_positions, atlas_.firstJoint()) = std::numeric_limits<double>::quiet_NaN();
  task.multiple_shooting.sqp_iteration = static_cast<int32_t>(runningIterations) + 2;
  task.icp_cost_weights.icp_error_weight = std::numeric_limits<double>::infinity();

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("state_weights"), testing::HasSubstr("was not applied"))))
      .Times(1);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("icp_cost_weights"), testing::HasSubstr("was not applied"))))
      .Times(1);
  log.StartCapturingLogs();
  reload(*updater, atlas_, task);
  log.StopCapturingLogs();
  EXPECT_EQ(atlas_.solver().getSettings().sqpIteration, runningIterations + 2) << "the rest of the reload was not applied";
  for (const OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
    EXPECT_TRUE(stateWeightsOf(ocp, qTerm) == shippedQ) << "a refused state_weights changed the running state cost";
  }
}

TEST_F(MpcParameterUpdaterModuleTest, ATermOfAnotherTypeIsReportedAndTheRestOfTheReloadApplies) {
  // Every worker's problem gets a quadratic cost under the DCM terminal cost's name; Collection::get then throws
  // std::bad_cast for it, which may not escape the reload or stop it part-way.
  const size_t stateDim = atlas_.interface->getMpcRobotModel().getStateDim();
  for (OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
    ASSERT_TRUE(hasTerm(*ocp.finalCostPtr, DcmTerminalCost::kTermName)) << "the shipped Atlas lists dcm_terminal_cost";
    ASSERT_NE(ocp.finalCostPtr->extract(DcmTerminalCost::kTermName), nullptr);
    ocp.finalCostPtr->add(DcmTerminalCost::kTermName, std::make_unique<QuadraticStateCost>(matrix_t::Identity(stateDim, stateDim)));
  }
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("Failed to update"), testing::HasSubstr(DcmTerminalCost::kTermName))))
      .Times(testing::AtLeast(1));
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_, testing::HasSubstr("stopped part-way"))).Times(0);
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr("Successfully applied in-place parameter updates")))
      .Times(1);
  log.StartCapturingLogs();
  reload(*updater, atlas_, shippedTask());
  log.StopCapturingLogs();
}

TEST_F(MpcParameterUpdaterModuleTest, ComAndAcomTrackingWeightsReachEveryWorker) {
  ASSERT_TRUE(hasTerm(*atlas_.firstProblem().stateCostPtr, std::string(ComAndAcomTrackingCost::kRunningTermName)))
      << "the shipped Atlas lists com_and_acom_tracking_cost";
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  mpc_config::TaskFile task = shippedTask();
  task.com_weights.z = 999.0;
  task.acom_weights.yaw = 777.0;
  reload(*updater, atlas_, task);
  for (OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
    const ComAndAcomTrackingCost& cost =
        ocp.stateCostPtr->get<ComAndAcomTrackingCost>(std::string(ComAndAcomTrackingCost::kRunningTermName));
    EXPECT_NEAR(cost.getQCom()(2, 2), task.com_weights.scaling * 999.0, 1.0e-9);
    // Row 0 of the ACoM weight is yaw, in the centroidal state's ZYX Euler convention.
    EXPECT_NEAR(cost.getQAcom()(0, 0), task.acom_weights.scaling * 777.0, 1.0e-9);
  }
}

TEST_F(MpcParameterUpdaterModuleTest, BasePoseWeightsFollowTheRunningProblemNotTheReloadedFile) {
  // The shipped problem lists com_and_acom_tracking_cost, which regulates the base pose: a reload must zero Q's base-pose
  // block, and must go on doing so when the reloaded file no longer lists the cost - the costs list is structural, fixed
  // when the problem was built, and the change is reported as waiting for the next start.
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  const std::string qTerm = stateCostTerm(atlas_.firstProblem());
  ASSERT_FALSE(qTerm.empty());
  mpc_config::TaskFile task = shippedTask();
  setBasePoseWeights(task.state_weights, /*offset=*/30.0);
  jointValue(task.state_weights.joint_positions, atlas_.firstJoint()) = 123.0;
  task.costs = without(task.costs, "com_and_acom_tracking_cost");

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kWarning, testing::_, testing::AllOf(testing::HasSubstr("next start"), testing::HasSubstr("costs"))))
      .Times(1);
  log.StartCapturingLogs();
  reload(*updater, atlas_, task);
  log.StopCapturingLogs();
  for (OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
    const matrix_t Q = stateWeightsOf(ocp, qTerm);
    EXPECT_NEAR(Q(kFirstJointIndex, kFirstJointIndex), task.state_weights.scaling * 123.0, 1.0e-9) << "the reload was not applied";
    EXPECT_TRUE(basePoseBlock(Q).isZero(0.0)) << "a reload re-introduced base-pose tracking beside the running ACoM cost:\n"
                                              << basePoseBlock(Q);
    EXPECT_TRUE(hasTerm(*ocp.stateCostPtr, std::string(ComAndAcomTrackingCost::kRunningTermName))) << "a reload never removes a term";
  }
}

TEST_F(MpcParameterUpdaterModuleTest, BasePoseWeightsStayLiveInAProblemBuiltWithoutTheAcomCost) {
  // The reverse: a problem built WITHOUT the ACoM cost keeps its base-pose weights live across a reload of a file that
  // lists it - zeroing them would leave the base pose unregulated, with no cost to take over.
  CentroidalMpcConfig config = shipped_;
  config.task.costs = without(config.task.costs, "com_and_acom_tracking_cost");
  setBasePoseWeights(config.task.state_weights, /*offset=*/30.0);
  const Built built = build(config, files_.urdfFile);
  ASSERT_NE(built.interface, nullptr);
  ASSERT_FALSE(hasTerm(*built.firstProblem().stateCostPtr, std::string(ComAndAcomTrackingCost::kRunningTermName)));
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built);
  ASSERT_NE(updater, nullptr);

  mpc_config::TaskFile task = config.task;
  task.costs = shippedTask().costs;
  setBasePoseWeights(task.state_weights, /*offset=*/60.0);
  reload(*updater, built, task);
  for (OptimalControlProblem& ocp : built.solver().getOcpDefinitions()) {
    const vector_t basePose = basePoseBlock(stateWeightsOf(ocp, "stateQuadraticCost")).diagonal();
    for (Eigen::Index i = 0; i < kBasePoseDim; ++i) {
      EXPECT_NEAR(basePose(i), task.state_weights.scaling * (60.0 + static_cast<scalar_t>(i + 1)), 1.0e-9)
          << "a reload zeroed the base-pose weights of a problem that has no ACoM cost to replace them";
    }
  }
}

/** `config` ending its horizon on the quadratic terminal cost (final_state_weights) instead of the DCM cost. */
CentroidalMpcConfig withQuadraticTerminalCost(CentroidalMpcConfig config) {
  config.task.costs = replaced(config.task.costs, "dcm_terminal_cost", "terminal_cost");
  return config;
}

TEST_F(MpcParameterUpdaterModuleTest, TheQuadraticTerminalCostAndItsAcomInstanceFollowAReload) {
  // The shipped Atlas ends on the DCM cost, so the problem is built from a variant that ends on final_state_weights. A
  // reload applies terminal_cost_scaling times final_state_weights with the base-pose block zeroed first under the ACoM
  // cost, and terminal_cost_scaling times the ACoM weights to the terminal ACoM instance, as at start-up.
  CentroidalMpcConfig config = withQuadraticTerminalCost(shipped_);
  setBasePoseWeights(config.task.final_state_weights, /*offset=*/40.0);
  const Built built = build(config, files_.urdfFile);
  ASSERT_NE(built.interface, nullptr);
  ASSERT_TRUE(hasTerm(*built.firstProblem().finalCostPtr, "terminalCost"));
  ASSERT_TRUE(basePoseBlock(finalStateWeightsOf(built.firstProblem())).isZero(0.0)) << "zeroed at start-up already";
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built);
  ASSERT_NE(updater, nullptr);

  mpc_config::TaskFile task = config.task;
  if (!task.terminal_cost_scaling.has_value()) GTEST_FAIL() << "the shipped task file has no terminal_cost_scaling";
  const scalar_t terminalCostScaling = 2.5 * *task.terminal_cost_scaling;
  task.terminal_cost_scaling = terminalCostScaling;
  setBasePoseWeights(task.final_state_weights, /*offset=*/70.0);
  jointValue(task.final_state_weights.joint_positions, built.firstJoint()) = 321.0;
  task.com_weights.z = 999.0;
  // The reload switches back to the DCM cost: structural, so reported, and the running quadratic cost keeps following.
  task.costs = shippedTask().costs;
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kWarning, testing::_, testing::AllOf(testing::HasSubstr("next start"), testing::HasSubstr("costs"))))
      .Times(1);
  log.StartCapturingLogs();
  reload(*updater, built, task);
  log.StopCapturingLogs();
  for (OptimalControlProblem& ocp : built.solver().getOcpDefinitions()) {
    const matrix_t Q_final = finalStateWeightsOf(ocp);
    EXPECT_NEAR(Q_final(kFirstJointIndex, kFirstJointIndex), terminalCostScaling * task.final_state_weights.scaling * 321.0, 1.0e-9);
    EXPECT_TRUE(basePoseBlock(Q_final).isZero(0.0)) << basePoseBlock(Q_final);
    EXPECT_FALSE(hasTerm(*ocp.finalCostPtr, DcmTerminalCost::kTermName)) << "a reload never adds a term";
    const ComAndAcomTrackingCost& running =
        ocp.stateCostPtr->get<ComAndAcomTrackingCost>(std::string(ComAndAcomTrackingCost::kRunningTermName));
    const ComAndAcomTrackingCost& terminal =
        ocp.finalCostPtr->get<ComAndAcomTrackingCost>(std::string(ComAndAcomTrackingCost::kTerminalTermName));
    EXPECT_NEAR(running.getQCom()(2, 2), task.com_weights.scaling * 999.0, 1.0e-9);
    EXPECT_NEAR(terminal.getQCom()(2, 2), terminalCostScaling * task.com_weights.scaling * 999.0, 1.0e-9);
    EXPECT_TRUE(terminal.getQAcom().isApprox(terminalCostScaling * running.getQAcom(), 1.0e-12));
  }
}

TEST_F(MpcParameterUpdaterModuleTest, WithoutATerminalCostScalingTheTerminalWeightsAreRefused) {
  const Built built = build(withQuadraticTerminalCost(shipped_), files_.urdfFile);
  ASSERT_NE(built.interface, nullptr);
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built);
  ASSERT_NE(updater, nullptr);
  const matrix_t running = finalStateWeightsOf(built.firstProblem());
  mpc_config::TaskFile task = built.interface->config().task;
  task.terminal_cost_scaling.reset();
  jointValue(task.final_state_weights.joint_positions, built.firstJoint()) = 321.0;
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("terminal_cost_scaling"), testing::HasSubstr("was not applied"))))
      .Times(1);
  log.StartCapturingLogs();
  reload(*updater, built, task);
  log.StopCapturingLogs();
  EXPECT_TRUE(finalStateWeightsOf(built.firstProblem()) == running) << "the terminal weights were applied without their scaling";
}

TEST_F(MpcParameterUpdaterModuleTest, TheRunningDcmCostFollowsItsBlockAndALeftOutComHeightIsTheModelsPendulum) {
  const scalar_t model = atlas_.interface->getNominalComHeight();
  ASSERT_GT(model, 0.5);
  ASSERT_FALSE(shippedTask().dcm_terminal_cost.com_height.has_value()) << "the shipped DCM cost no longer derives its pendulum";
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_, atlas_.interface->getSwitchedModelReferenceManagerPtr().get());
  ASSERT_NE(updater, nullptr);
  const std::function<void(scalar_t, scalar_t)> expectOnEveryWorker = [this](scalar_t height, scalar_t weightX) {
    for (OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
      const DcmTerminalCost& cost = ocp.finalCostPtr->get<DcmTerminalCost>(DcmTerminalCost::kTermName);
      EXPECT_THAT(cost.getConfig().comHeight, ::testing::Optional(::testing::DoubleNear(height, /*max_abs_error=*/1.0e-12)));
      EXPECT_NEAR(cost.getParameters(/*time=*/0.5, TargetTrajectories())(2), std::sqrt(9.81 / height), 1.0e-12);
      EXPECT_NEAR(cost.getConfig().weights(0), weightX, 1.0e-12);
    }
  };
  const scalar_t shippedWeight = shippedTask().dcm_terminal_cost.weight_x;
  expectOnEveryWorker(model, shippedWeight);
  mpc_config::TaskFile task = shippedTask();
  task.dcm_terminal_cost.com_height = 0.9;
  task.dcm_terminal_cost.weight_x = 123.0;
  ASSERT_GT(std::abs(0.9 - model), 0.05) << "the override must differ from the model, or the checks prove nothing";
  reload(*updater, atlas_, task);
  expectOnEveryWorker(/*height=*/0.9, /*weightX=*/123.0);
  task.dcm_terminal_cost.com_height.reset();
  reload(*updater, atlas_, task);
  expectOnEveryWorker(model, /*weightX=*/123.0);
  // A height that is no height is refused by the conversion, and the running cost kept: a negative one, and the 0 that
  // stood for the model's pendulum before com_height was optional.
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("dcm_terminal_cost"), testing::HasSubstr("was not applied"))))
      .Times(2);
  log.StartCapturingLogs();
  for (const scalar_t refused : {-0.9, 0.0}) {
    task.dcm_terminal_cost.com_height = refused;
    reload(*updater, atlas_, task);
  }
  log.StopCapturingLogs();
  expectOnEveryWorker(model, /*weightX=*/123.0);
}

TEST_F(MpcParameterUpdaterModuleTest, TheTaskSpaceIcpAndTorqueCostsFollowAReload) {
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  mpc_config::TaskFile task = shippedTask();
  task.task_space_foot_cost.weights.pos_x = 11.0;
  const bool shippedActiveInStance = task.task_space_foot_cost.active_phases == "swing_and_stance";
  task.task_space_foot_cost.active_phases = shippedActiveInStance ? "swing" : "swing_and_stance";
  ASSERT_FALSE(task.task_space_costs.empty()) << "the shipped Atlas tracks its torso";
  task.task_space_costs.front().weights.orientation_z = 12.0;
  task.icp_cost_weights.icp_error_weight = 13.0;
  ASSERT_FALSE(task.left_leg_torque_cost.joints.empty());
  task.left_leg_torque_cost.joints.front().value = 14.0;
  reload(*updater, atlas_, task);
  const std::string linkCost = absl::StrCat(task.task_space_costs.front().name, "_TaskSpaceKinematicsCost");
  for (OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
    for (const std::string& footName : atlas_.contactNames()) {
      const CentroidalMpcEndEffectorFootCost& footCost =
          ocp.costPtr->get<CentroidalMpcEndEffectorFootCost>(absl::StrCat(footName, "_TaskSpaceKinematicsCost"));
      vector12_t weights;
      footCost.getWeights(weights);
      EXPECT_NEAR(weights(0), 11.0, 1.0e-9) << footName;
      EXPECT_EQ(footCost.getActiveInStance(), !shippedActiveInStance) << footName;
    }
    if (hasTerm(*ocp.costPtr, linkCost)) {
      vector12_t weights;
      ocp.costPtr->get<EndEffectorKinematicsQuadraticCost>(linkCost).getWeights(weights);
      EXPECT_NEAR(weights(5), 12.0, 1.0e-9);
    } else {
      ADD_FAILURE() << "the problem has no " << linkCost;
    }
    vector2_t icpWeights;
    ocp.costPtr->get<ICPCost>("icp_Cost").getWeights(icpWeights);
    EXPECT_NEAR(icpWeights(0), 13.0, 1.0e-9);
    vector_t torqueWeights;
    ocp.costPtr->get<ExternalTorqueQuadraticCostAD>(absl::StrCat(atlas_.contactNames().front(), "_ExternalTorqueQuadraticCost"))
        .getWeights(torqueWeights);
    ASSERT_GT(torqueWeights.size(), 0);
    EXPECT_NEAR(torqueWeights(0), task.left_leg_torque_cost.scaling * 14.0, 1.0e-9);
  }
}

TEST_F(MpcParameterUpdaterModuleTest, AnAddedOrRemovedLinkCostIsReportedForTheNextStartAndChangesNoTerm) {
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  const mpc_config::TaskFile shipped = shippedTask();
  ASSERT_EQ(shipped.task_space_costs.size(), 1u) << "the shipped Atlas tracks its torso alone";
  const std::string torso = absl::StrCat(shipped.task_space_costs.front().name, "_TaskSpaceKinematicsCost");
  const size_t numCosts = atlas_.firstProblem().costPtr->getTermNameMap().size();

  // An added entry, on the same link so that its conversion accepts it, beside a new weight of the carried one.
  mpc_config::TaskFile added = shipped;
  mpc_config::TaskSpaceCostConfig copy = added.task_space_costs.front();
  copy.name = "torso_copy";
  added.task_space_costs.push_back(copy);
  added.task_space_costs.front().weights.orientation_z = 21.0;
  {
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                         testing::AllOf(testing::HasSubstr("task_space_costs entry of torso_copy_TaskSpaceKinematicsCost"),
                                        testing::HasSubstr("next start"))))
        .Times(1);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                         testing::AllOf(testing::HasSubstr("MPC's next start"), testing::HasSubstr("task_space_costs[1].name"))))
        .Times(1);
    log.StartCapturingLogs();
    reload(*updater, atlas_, added);
    log.StopCapturingLogs();
  }
  for (OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
    EXPECT_FALSE(hasTerm(*ocp.costPtr, "torso_copy_TaskSpaceKinematicsCost")) << "a reload added a term";
    EXPECT_EQ(ocp.costPtr->getTermNameMap().size(), numCosts);
    vector12_t weights;
    ocp.costPtr->get<EndEffectorKinematicsQuadraticCost>(torso).getWeights(weights);
    EXPECT_NEAR(weights(5), 21.0, 1.0e-9) << "the carried entry's weights still apply";
  }

  // A removed entry keeps the running link cost and its weights.
  mpc_config::TaskFile removed = shipped;
  removed.task_space_costs.clear();
  {
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log,
                Log(absl::LogSeverity::kWarning, testing::_,
                    testing::AllOf(testing::HasSubstr(absl::StrCat("removal of the link cost ", torso)), testing::HasSubstr("next start"))))
        .Times(1);
    log.StartCapturingLogs();
    reload(*updater, atlas_, removed);
    log.StopCapturingLogs();
  }
  for (OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
    ASSERT_TRUE(hasTerm(*ocp.costPtr, torso)) << "a reload removed a term";
    vector12_t weights;
    ocp.costPtr->get<EndEffectorKinematicsQuadraticCost>(torso).getWeights(weights);
    EXPECT_NEAR(weights(5), 21.0, 1.0e-9) << "a removed entry's weights are the running ones";
  }
}

/******************************************************************************************************/
// The input cost under basis-vector contact inputs
/******************************************************************************************************/

TEST_F(MpcParameterUpdaterModuleTest, AnUnchangedReloadReproducesTheFactorysBasisSpaceInputCost) {
  ASSERT_TRUE(atlas_.interface->usesContactBasisVectorInputs()) << "the shipped Atlas selects basis-vector contact inputs";
  const std::optional<BasisInputsCostTransformConfig> transform = atlas_.interface->getBasisInputsCostTransformConfig();
  if (!transform.has_value()) GTEST_FAIL() << "the basis-mode interface has no cost transform";
  const std::string rTerm = inputCostTerm(atlas_.firstProblem());
  ASSERT_FALSE(rTerm.empty());
  const matrix_t factoryR = inputWeightsOf(atlas_.firstProblem(), rTerm);
  ASSERT_EQ(static_cast<size_t>(factoryR.rows()), transform->basisInputDim()) << "the factory's R is in basis space";
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  reload(*updater, atlas_, shippedTask());
  for (const OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
    EXPECT_TRUE(inputWeightsOf(ocp, rTerm).isApprox(factoryR, 1.0e-12)) << "the updater and the factory transform R differently";
  }
}

TEST_F(MpcParameterUpdaterModuleTest, TheRegularizationOfTheBasisSpaceInputCostIsReloadedByName) {
  ASSERT_TRUE(atlas_.interface->usesContactBasisVectorInputs());
  const std::optional<BasisInputsCostTransformConfig> transform = atlas_.interface->getBasisInputsCostTransformConfig();
  if (!transform.has_value()) GTEST_FAIL() << "the basis-mode interface has no cost transform";
  ASSERT_EQ(transform->regularization, kFullDiagonalBasisRegularization) << "the shipped regularization is the full diagonal";
  const std::string rTerm = inputCostTerm(atlas_.firstProblem());
  ASSERT_FALSE(rTerm.empty());
  const matrix_t shippedR = inputWeightsOf(atlas_.firstProblem(), rTerm);
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);

  mpc_config::TaskFile task = shippedTask();
  task.contacts.basis_regularization = std::string(kNullSpaceBasisRegularization);
  task.contacts.basis_scaling_regularization *= 3.0;
  reload(*updater, atlas_, task);
  BasisInputsCostTransformConfig expected = *transform;
  expected.regularization = std::string(kNullSpaceBasisRegularization);
  expected.lambdaRegularization = task.contacts.basis_scaling_regularization;
  // The wrench-space R is the factory's R undone: transform it with the shipped regularization and with the reloaded one.
  const matrix_t wrenchR = [&]() {
    absl::StatusOr<matrix_t> converted = inputWeightsFromConfig(
        task.input_weights, stateInputLayout(atlas_.interface->modelSettings(), StateInputLayout::Mpc::kCentroidal), "input_weights");
    EXPECT_TRUE(converted.ok()) << converted.status();
    return converted.ok() ? *converted : matrix_t();
  }();
  ASSERT_TRUE(transformWrenchInputCostToBasisSpace(wrenchR, *transform).isApprox(shippedR, 1.0e-12));
  const matrix_t expectedR = transformWrenchInputCostToBasisSpace(wrenchR, expected);
  ASSERT_GT((expectedR - shippedR).cwiseAbs().maxCoeff(), 1.0e-9) << "the two shapes give the same cost, so the check proves nothing";
  for (const OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
    EXPECT_TRUE(inputWeightsOf(ocp, rTerm).isApprox(expectedR, 1.0e-12)) << "the reload did not apply the null_space regularization";
  }
}

TEST_F(MpcParameterUpdaterModuleTest, ABasisSpaceInputCostTheQpCannotSolveWithIsRefusedAndTheRunningOneKept) {
  ASSERT_TRUE(atlas_.interface->usesContactBasisVectorInputs());
  const OptimalControlProblem& ocp = atlas_.firstProblem();
  const std::string rTerm = inputCostTerm(ocp);
  const std::string qTerm = stateCostTerm(ocp);
  ASSERT_FALSE(rTerm.empty() || qTerm.empty());
  const matrix_t shippedR = inputWeightsOf(ocp, rTerm);
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  struct Case {
    std::string name;
    std::function<void(mpc_config::TaskFile&)> edit;
    std::string field;
  };
  const std::vector<Case> cases = {
      {"unknown shape", [](mpc_config::TaskFile& task) { task.contacts.basis_regularization = "no_such_shape"; }, "unknown"},
      {"negative weight", [](mpc_config::TaskFile& task) { task.contacts.basis_scaling_regularization = -1.0; }, "non-negative"},
      // A zero weight passes the range check but leaves the lambda block singular under full_diagonal's M^T R M.
      {"zero weight", [](mpc_config::TaskFile& task) { task.contacts.basis_scaling_regularization = 0.0; }, "not positive definite"},
  };
  scalar_t jointWeight = 321.0;
  for (const Case& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    // The rest of the file still applies: a joint weight of Q, different in every case, shows this reload ran.
    jointWeight += 1.0;
    mpc_config::TaskFile task = shippedTask();
    testCase.edit(task);
    jointValue(task.state_weights.joint_positions, atlas_.firstJoint()) = jointWeight;
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                         testing::AllOf(testing::HasSubstr("input_weights"), testing::HasSubstr(testCase.field))))
        .Times(1);
    log.StartCapturingLogs();
    reload(*updater, atlas_, task);
    log.StopCapturingLogs();
    for (const OptimalControlProblem& worker : atlas_.solver().getOcpDefinitions()) {
      EXPECT_TRUE(inputWeightsOf(worker, rTerm) == shippedR) << "a refused reload changed the running R";
      EXPECT_NEAR(stateWeightsOf(worker, qTerm)(kFirstJointIndex, kFirstJointIndex), task.state_weights.scaling * jointWeight, 1.0e-9)
          << "the rest of the reload was not applied";
    }
  }
}

TEST_F(MpcParameterUpdaterModuleTest, TheAppliersThatShareTheStateInputCostWriteTheSameGainsInEitherOrder) {
  // Under basis-vector contact inputs, stateInputQuadraticCost carries Q of state_weights (QuadraticCostWeightsApplier)
  // and R of input_weights in the basis space (BasisInputsCostApplier): each keeps the other's part.
  ASSERT_TRUE(atlas_.interface->usesContactBasisVectorInputs());
  CentroidalMpcConfig config = shipped_;
  config.task.costs = without(without(config.task.costs, "state_quadratic_cost"), "input_quadratic_cost");
  config.task.costs.emplace_back("state_input_quadratic_cost");
  const Built listed = build(config, files_.urdfFile);
  const Built reversed = build(config, files_.urdfFile);
  ASSERT_TRUE(listed.interface != nullptr && reversed.interface != nullptr);
  ASSERT_TRUE(hasTerm(*listed.firstProblem().costPtr, kStateInputQuadraticCostTerm));
  std::unique_ptr<MpcParameterUpdaterModule> listedUpdater = makeUpdater(listed);
  std::unique_ptr<MpcParameterUpdaterModule> reversedUpdater =
      makeUpdater(reversed, /*referenceManager=*/nullptr, /*watched=*/WatchedFiles{}, ApplierOrder::kReversed);
  ASSERT_TRUE(listedUpdater != nullptr && reversedUpdater != nullptr);
  const matrix_t shippedQ = stateWeightsOf(listed.firstProblem(), kStateInputQuadraticCostTerm);
  const matrix_t shippedR = inputWeightsOf(listed.firstProblem(), kStateInputQuadraticCostTerm);

  mpc_config::TaskFile task = config.task;
  task.state_weights.scaling *= 2.0;
  task.contacts.basis_scaling_regularization *= 3.0;
  reload(*listedUpdater, listed, task);
  reload(*reversedUpdater, reversed, task);
  const matrix_t Q = stateWeightsOf(listed.firstProblem(), kStateInputQuadraticCostTerm);
  const matrix_t R = inputWeightsOf(listed.firstProblem(), kStateInputQuadraticCostTerm);
  ASSERT_FALSE(Q == shippedQ) << "the reload did not reach Q, so the order proves nothing";
  ASSERT_FALSE(R == shippedR) << "the reload did not reach R, so the order proves nothing";
  for (const OptimalControlProblem& worker : reversed.solver().getOcpDefinitions()) {
    EXPECT_TRUE(stateWeightsOf(worker, kStateInputQuadraticCostTerm) == Q) << "the order of the appliers changed Q";
    EXPECT_TRUE(inputWeightsOf(worker, kStateInputQuadraticCostTerm) == R) << "the order of the appliers changed R";
  }
}

/******************************************************************************************************/
// The constraints
/******************************************************************************************************/

TEST_F(MpcParameterUpdaterModuleTest, TheBarriersOfTheJointLimitsTheFootCollisionAndTheBasisScalingsFollowAReload) {
  ASSERT_TRUE(hasTerm(*atlas_.firstProblem().stateSoftConstraintPtr, "jointLimits")) << "the shipped Atlas lists joint_limits";
  ASSERT_TRUE(hasTerm(*atlas_.firstProblem().stateSoftConstraintPtr, "FootCollisionSoftConstraint"));
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  mpc_config::TaskFile task = shippedTask();
  task.joint_limits.mu = 999.0;
  task.collision_constraint.mu = 0.75;
  task.collision_constraint.delta = 0.0025;
  task.contacts.basis_non_negativity_barrier.mu = 0.42;
  reload(*updater, atlas_, task);
  for (OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
    scalar_t mu = 0.0;
    scalar_t delta = 0.0;
    ocp.stateSoftConstraintPtr->get<JointLimitsSoftConstraint>("jointLimits").getGains(mu, delta);
    EXPECT_NEAR(mu, 999.0, 1.0e-12);
    EXPECT_EQ(penaltyParameters(ocp.stateSoftConstraintPtr->get<StateSoftConstraint>("FootCollisionSoftConstraint")),
              (vector_t(2) << 0.75, 0.0025).finished());
    for (const std::string& footName : atlas_.contactNames()) {
      EXPECT_NEAR(
          ocp.costPtr->get<BasisScalingNonNegativityConstraint>(absl::StrCat(footName, "_basisNonNegativity")).getBarrierConfig().mu, 0.42,
          1.0e-12);
    }
  }
}

/** The twist constraints behind every foot's zeroVelocity term of every worker, hard or soft. */
std::vector<EndEffectorKinematicsTwistConstraint* absl_nonnull> zeroVelocityTwists(SqpSolver& solver,
                                                                                   const std::vector<std::string>& contactNames) {
  std::vector<EndEffectorKinematicsTwistConstraint* absl_nonnull> found;
  for (OptimalControlProblem& ocp : solver.getOcpDefinitions()) {
    for (const std::string& footName : contactNames) {
      const std::string name = absl::StrCat(footName, "_zeroVelocity");
      if (hasTerm(*ocp.equalityConstraintPtr, name)) {
        found.push_back(&ocp.equalityConstraintPtr->get<ZeroVelocityConstraintCppAd>(name).getTwistConstraint());
      }
      if (hasTerm(*ocp.softConstraintPtr, name)) {
        found.push_back(
            &ocp.softConstraintPtr->get<StateInputSoftConstraint>(name).get<ZeroVelocityConstraintCppAd>().getTwistConstraint());
      }
    }
  }
  return found;
}

TEST_F(MpcParameterUpdaterModuleTest, TheFootConstraintGainsReachTheTwistConstraintsAndThePreComputation) {
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  const std::vector<EndEffectorKinematicsTwistConstraint* absl_nonnull> twists = zeroVelocityTwists(atlas_.solver(), atlas_.contactNames());
  ASSERT_FALSE(twists.empty()) << "the shipped Atlas lists zero_velocity as a hard constraint";
  mpc_config::TaskFile task = shippedTask();
  mpc_config::ModelSettingsConfig::FootConstraintConfig& gains = task.model_settings.foot_constraint;
  gains.linear_velocity_error_gain_xy = 99.0;
  // The other stance constraint with the orientation rows: the yaw-rate row about the contact normal toggles.
  const bool shippedYawRate = gains.stance_constraint == "position_and_orientation";
  gains.stance_constraint = shippedYawRate ? "position_and_tilt" : "position_and_orientation";
  // positionErrorGain_z has to reach the pre-computation too: under the contact-implicit formulation the soft
  // normal-velocity servo reads it from there, and zeroVelocity is not built at all.
  gains.position_error_gain_z = 3.5;
  reload(*updater, atlas_, task);
  for (EndEffectorKinematicsTwistConstraint* absl_nonnull twist : twists) {
    EXPECT_NEAR(twist->getConfig().Av(0, 0), 99.0, 1.0e-12);
    EXPECT_EQ(twist->getConstrainYawRateAboutNormal(), !shippedYawRate);
    EXPECT_EQ(twist->getNumConstraints(/*time=*/0.0), 6U) << "both stance constraints have the orientation rows";
  }
  for (OptimalControlProblem& ocp : atlas_.solver().getOcpDefinitions()) {
    const HumanoidPreComputation* absl_nullable preComputation = dynamic_cast<const HumanoidPreComputation*>(ocp.preComputationPtr.get());
    ASSERT_NE(preComputation, nullptr);
    EXPECT_NEAR(preComputation->getNormalVelocityPositionErrorGain(), 3.5, 1.0e-12);
  }
  EXPECT_EQ(atlas_.interface->modelSettings().footConstraintConfig.constrainYawRateAboutContactNormal, shippedYawRate)
      << "the interface still holds the settings it was built with";
}

TEST_F(MpcParameterUpdaterModuleTest, TheWeightOfASoftZeroVelocityFollowsAReload) {
  // soft_constraint_weight only reaches a penalty when zero_velocity is a SOFT constraint; the shipped Atlas lists it as a
  // hard one, so the problem is built from a variant that lists it among the soft constraints.
  CentroidalMpcConfig config = shipped_;
  config.task.hard_constraints = without(config.task.hard_constraints, "zero_velocity");
  config.task.soft_constraints.emplace_back("zero_velocity");
  const Built built = build(config, files_.urdfFile);
  ASSERT_NE(built.interface, nullptr);
  const std::function<scalar_t(OptimalControlProblem&, const std::string&)> installedWeight = [](OptimalControlProblem& ocp,
                                                                                                 const std::string& footName) {
    // 0.5 * weight at a unit residual, for the quadratic penalty.
    return 2.0 * ocp.softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(footName, "_zeroVelocity"))
                     .getPenalty()
                     .getValue(
                         /*t=*/0.0, vector_t::Ones(1));
  };
  for (const std::string& footName : built.contactNames()) {
    ASSERT_NEAR(installedWeight(built.firstProblem(), footName), config.task.model_settings.foot_constraint.soft_constraint_weight, 1.0e-9);
  }
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built);
  ASSERT_NE(updater, nullptr);
  mpc_config::TaskFile task = config.task;
  task.model_settings.foot_constraint.soft_constraint_weight = 12345.0;
  reload(*updater, built, task);
  for (OptimalControlProblem& ocp : built.solver().getOcpDefinitions()) {
    for (const std::string& footName : built.contactNames()) {
      EXPECT_NEAR(installedWeight(ocp, footName), 12345.0, 1.0e-9) << footName;
    }
  }
}

TEST_F(MpcParameterUpdaterModuleTest, ASoftFootWeightThatIsNotPositiveIsRefusedAtStartUpAndOnAReloadAlike) {
  CentroidalMpcConfig config = shipped_;
  config.task.hard_constraints = without(config.task.hard_constraints, "zero_velocity");
  config.task.soft_constraints.emplace_back("zero_velocity");
  const Built built = build(config, files_.urdfFile);
  ASSERT_NE(built.interface, nullptr);
  const std::string term = absl::StrCat(built.contactNames().front(), "_zeroVelocity");
  const scalar_t running = built.firstProblem().softConstraintPtr->get<StateInputSoftConstraint>(term).getPenalty().getValue(
      /*t=*/0.0, vector_t::Ones(1));

  CentroidalMpcConfig zero = config;
  zero.task.model_settings.foot_constraint.soft_constraint_weight = 0.0;
  // A fresh start on the file refuses it, naming the field.
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> started = CentroidalMpcInterface::Create(zero, files_.urdfFile);
  EXPECT_EQ(started.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(started.status().message(), testing::HasSubstr("model_settings.foot_constraint.soft_constraint_weight is 0"));

  // A reload of it is refused by the same conversion, and the running weight is kept.
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built);
  ASSERT_NE(updater, nullptr);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kWarning, testing::_,
                  testing::AllOf(testing::HasSubstr("model_settings.foot_constraint"), testing::HasSubstr("soft_constraint_weight is 0"))))
      .Times(1);
  log.StartCapturingLogs();
  reload(*updater, built, zero.task);
  log.StopCapturingLogs();
  for (OptimalControlProblem& ocp : built.solver().getOcpDefinitions()) {
    EXPECT_EQ(ocp.softConstraintPtr->get<StateInputSoftConstraint>(term).getPenalty().getValue(/*t=*/0.0, vector_t::Ones(1)), running);
  }
}

/******************************************************************************************************/
// The contact-implicit terms and the ground
/******************************************************************************************************/

/** The Atlas configuration under the contact-implicit formulation, with every contact_implicit field off its default. */
CentroidalMpcConfig contactImplicitConfig(CentroidalMpcConfig config) {
  selectContactImplicitFormulation(config.task);
  config.task.contact_implicit = mpc_config::ContactImplicitConfig{.complementarity_weight = 61.5,
                                                                   .slip_weight = 173.25,
                                                                   .penetration_weight = 4.25e4,
                                                                   .height_reference = 0.0725,
                                                                   .velocity_reference = 0.35,
                                                                   .angular_velocity_reference = 1.25,
                                                                   .gap_smoothing = 1.5e-3};
  return config;
}

TEST_F(MpcParameterUpdaterModuleTest, EveryContactImplicitFieldReachesEveryTermOfEveryWorker) {
  const Built built = build(contactImplicitConfig(shipped_), files_.urdfFile);
  ASSERT_NE(built.interface, nullptr);
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built, built.interface->getSwitchedModelReferenceManagerPtr().get());
  ASSERT_NE(updater, nullptr);
  const mpc_config::ContactImplicitConfig expected{.complementarity_weight = 71.5,
                                                   .slip_weight = 183.25,
                                                   .penetration_weight = 5.25e4,
                                                   .height_reference = 0.0825,
                                                   .velocity_reference = 0.45,
                                                   .angular_velocity_reference = 1.35,
                                                   .gap_smoothing = 2.5e-3};
  mpc_config::TaskFile task = built.interface->config().task;
  task.contact_implicit = expected;
  reload(*updater, built, task);
  ASSERT_GT(built.solver().getOcpDefinitions().size(), 1U) << "the check has to cover more than one worker's copy of the problem";
  for (OptimalControlProblem& ocp : built.solver().getOcpDefinitions()) {
    for (const std::string& footName : built.contactNames()) {
      SCOPED_TRACE(footName);
      StateInputSoftConstraint& complementarity = complementarityTerm(ocp, footName);
      EXPECT_EQ(penaltyParameters(complementarity), (vector_t(1) << expected.complementarity_weight).finished());
      EXPECT_EQ(complementarity.get<ContactComplementarityConstraint>().getHeightReference(), expected.height_reference);
      EXPECT_EQ(complementarity.get<ContactComplementarityConstraint>().getGapSmoothing(), expected.gap_smoothing);
      StateInputSoftConstraint& slip = slipTerm(ocp, footName);
      EXPECT_EQ(penaltyParameters(slip), (vector_t(1) << expected.slip_weight).finished());
      const vector3_t inverseReferences = slip.get<ForceWeightedSlipConstraint>().getInverseTwistReference();
      EXPECT_NEAR(inverseReferences(0), 1.0 / expected.velocity_reference, 1.0e-12);
      EXPECT_NEAR(inverseReferences(2), 1.0 / expected.angular_velocity_reference, 1.0e-12);
      StateSoftConstraint& penetration = penetrationTerm(ocp, footName);
      // A hinge whose zero stays on the ground: delta 0.
      EXPECT_EQ(penaltyParameters(penetration), (vector_t(2) << expected.penetration_weight, 0.0).finished());
    }
  }
}

TEST_F(MpcParameterUpdaterModuleTest, AContactImplicitBlockTheReloadLeavesOutIsItsDefaults) {
  // The payload is the file (humanoid_nmpc/humanoid_mpc_config/README.md, "Live updates"): a reload that sets only the
  // slip weight runs every other field at its default, as a start-up from that file would - not the running values, as
  // a YAML reload of a partial block used to.
  const Built built = build(contactImplicitConfig(shipped_), files_.urdfFile);
  ASSERT_NE(built.interface, nullptr);
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built, built.interface->getSwitchedModelReferenceManagerPtr().get());
  ASSERT_NE(updater, nullptr);
  mpc_config::TaskFile task = built.interface->config().task;
  task.contact_implicit = mpc_config::ContactImplicitConfig{};
  task.contact_implicit.slip_weight = 212.5;
  const ModelSettings::ContactImplicitConfig defaults;
  reload(*updater, built, task);
  for (OptimalControlProblem& ocp : built.solver().getOcpDefinitions()) {
    for (const std::string& footName : built.contactNames()) {
      EXPECT_EQ(penaltyParameters(slipTerm(ocp, footName)), (vector_t(1) << 212.5).finished());
      EXPECT_EQ(penaltyParameters(complementarityTerm(ocp, footName)), (vector_t(1) << defaults.complementarityWeight).finished());
      EXPECT_EQ(complementarityTerm(ocp, footName).get<ContactComplementarityConstraint>().getHeightReference(), defaults.heightReference);
    }
  }
}

TEST_F(MpcParameterUpdaterModuleTest, ARefusedContactImplicitBlockLeavesEveryTermAsItWasAndTheRestOfTheFileApplies) {
  const CentroidalMpcConfig config = contactImplicitConfig(shipped_);
  const Built built = build(config, files_.urdfFile);
  ASSERT_NE(built.interface, nullptr);
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built, built.interface->getSwitchedModelReferenceManagerPtr().get());
  ASSERT_NE(updater, nullptr);
  const scalar_t launchedWeight = config.task.contact_implicit.complementarity_weight;
  struct Case {
    std::string name;
    std::function<void(mpc_config::ContactImplicitConfig&)> edit;
    std::string field;
  };
  const std::vector<Case> cases = {
      {"a divisor of zero", [](mpc_config::ContactImplicitConfig& block) { block.gap_smoothing = 0.0; }, "gap"},
      {"a negative weight", [](mpc_config::ContactImplicitConfig& block) { block.penetration_weight = -5.0e4; }, "penetration"},
  };
  int32_t sqpIteration = 20;
  for (const Case& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    ++sqpIteration;
    mpc_config::TaskFile task = config.task;
    task.contact_implicit.complementarity_weight = 81.5;
    testCase.edit(task.contact_implicit);
    task.multiple_shooting.sqp_iteration = sqpIteration;
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                         testing::AllOf(testing::HasSubstr("contact_implicit"), testing::HasSubstr("was not applied"),
                                        testing::HasSubstr(testCase.field))))
        .Times(1);
    log.StartCapturingLogs();
    reload(*updater, built, task);
    log.StopCapturingLogs();
    EXPECT_EQ(built.solver().getSettings().sqpIteration, static_cast<size_t>(sqpIteration)) << "the rest of the file was not applied";
    for (OptimalControlProblem& ocp : built.solver().getOcpDefinitions()) {
      for (const std::string& footName : built.contactNames()) {
        EXPECT_EQ(penaltyParameters(complementarityTerm(ocp, footName))(0), launchedWeight) << "part of a refused block was applied";
      }
    }
  }
}

TEST_F(MpcParameterUpdaterModuleTest, AReloadedGroundReachesTheContactImplicitTermsInTheSolveWhoseReferencesStandOnIt) {
  const Built built = build(contactImplicitConfig(shipped_), files_.urdfFile);
  ASSERT_NE(built.interface, nullptr);
  SwitchedModelReferenceManager& referenceManager = *built.interface->getSwitchedModelReferenceManagerPtr();
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built, &referenceManager);
  ASSERT_NE(updater, nullptr);
  const scalar_t launched = built.interface->modelSettings().terrainHeight;
  constexpr scalar_t kReloaded = 0.013;
  ASSERT_NE(launched, kReloaded);
  mpc_config::TaskFile task = built.interface->config().task;
  task.terrain_height = kReloaded;
  reload(*updater, built, task);
  // The ground went to the reference manager, which applies it at its next pre-solve hook; the terms follow it in that
  // same solve, never before the references stand on it.
  EXPECT_DOUBLE_EQ(referenceManager.getTerrainHeight(), kReloaded);
  EXPECT_DOUBLE_EQ(referenceManager.getAppliedTerrainHeight(), launched);
  EXPECT_DOUBLE_EQ(contactImplicitTermsTerrainHeight(built.solver(), built.contactNames()), launched);

  const vector_t& state = built.interface->getInitialState();
  referenceManager.setTargetTrajectories(
      TargetTrajectories({0.0}, {state}, {vector_t::Zero(built.interface->getEffectiveMpcRobotModel().getInputDim())}));
  const scalar_t horizon = built.interface->mpcSettings().timeHorizon_;
  referenceManager.preSolverRun(/*initTime=*/0.1, /*finalTime=*/0.1 + horizon, state, ModeNumber::kStance);
  updater->preSolverRun(/*initTime=*/0.1, /*finalTime=*/0.1 + horizon, state, referenceManager);
  EXPECT_DOUBLE_EQ(referenceManager.getAppliedTerrainHeight(), kReloaded);
  EXPECT_DOUBLE_EQ(contactImplicitTermsTerrainHeight(built.solver(), built.contactNames()), kReloaded);
  for (size_t foot = 0; foot < built.contactNames().size(); ++foot) {
    if (!referenceManager.isInContact(/*time=*/0.1, foot)) continue;
    EXPECT_DOUBLE_EQ(referenceManager.getSwingTrajectoryPlanner()->getZpositionConstraint(foot, /*time=*/0.1), kReloaded)
        << built.contactNames()[foot] << ": a stance foot's height reference is the ground";
  }
}

TEST_F(MpcParameterUpdaterModuleTest, WithoutAReferenceManagerAReloadedGroundReachesBothTermsAtOnce) {
  const Built built = build(contactImplicitConfig(shipped_), files_.urdfFile);
  ASSERT_NE(built.interface, nullptr);
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built);
  ASSERT_NE(updater, nullptr);
  const scalar_t launched = built.interface->modelSettings().terrainHeight;
  mpc_config::TaskFile task = built.interface->config().task;
  task.terrain_height = 0.017;
  reload(*updater, built, task);
  EXPECT_DOUBLE_EQ(contactImplicitTermsTerrainHeight(built.solver(), built.contactNames()), 0.017);
  EXPECT_DOUBLE_EQ(built.interface->getSwitchedModelReferenceManagerPtr()->getTerrainHeight(), launched)
      << "the updater was given no reference manager, so it cannot have moved the one the interface holds";
}

/******************************************************************************************************/
// The solver, the references and the planner
/******************************************************************************************************/

TEST_F(MpcParameterUpdaterModuleTest, OnlyTheHotSettingsOfTheSqpSolverAreApplied) {
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  const sqp::Settings running = atlas_.solver().getSettings();
  mpc_config::TaskFile task = shippedTask();
  task.multiple_shooting.sqp_iteration = static_cast<int32_t>(running.sqpIteration) + 5;
  task.multiple_shooting.g_max = 2.0 * running.g_max;
  // Not hot: the shooting interval decides the problem's discretization.
  task.multiple_shooting.dt = 2.0 * running.dt;
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("next start"), testing::HasSubstr("multiple_shooting.dt"))))
      .Times(1);
  log.StartCapturingLogs();
  reload(*updater, atlas_, task);
  log.StopCapturingLogs();
  EXPECT_EQ(atlas_.solver().getSettings().sqpIteration, running.sqpIteration + 5);
  EXPECT_EQ(atlas_.solver().getSettings().g_max, 2.0 * running.g_max);
  EXPECT_EQ(atlas_.solver().getSettings().dt, running.dt) << "a start-up setting of the solver was applied";
}

TEST_F(MpcParameterUpdaterModuleTest, TheStructuralFieldsOfAReloadAreReportedByNameAndNotApplied) {
  // The formulation's lists, its contact input parameterization and its contact schedule source decide which terms the
  // problem was assembled from and its input dimension: a reload reports each that differs from the running file, in
  // one warning, and applies none of them.
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  mpc_config::TaskFile task = shippedTask();
  task.contact_schedule_source = "contact_planner";
  task.contact_input_parameterization = "wrench";
  task.soft_constraints = without(task.soft_constraints, "foot_collision");
  task.mpc.time_horizon *= 2.0;
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("next start"), testing::HasSubstr("contact_schedule_source"),
                                      testing::HasSubstr("contact_input_parameterization"), testing::HasSubstr("soft_constraints"),
                                      testing::HasSubstr("mpc.time_horizon"))))
      .Times(1);
  log.StartCapturingLogs();
  reload(*updater, atlas_, task);
  log.StopCapturingLogs();
  EXPECT_FALSE(atlas_.interface->usesContactPlanning());
  EXPECT_TRUE(hasTerm(*atlas_.firstProblem().stateSoftConstraintPtr, "FootCollisionSoftConstraint")) << "a reload removed a term";
}

TEST_F(MpcParameterUpdaterModuleTest, TheSwingTrajectoriesFollowAReload) {
  SwitchedModelReferenceManager& referenceManager = *atlas_.interface->getSwitchedModelReferenceManagerPtr();
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_, &referenceManager);
  ASSERT_NE(updater, nullptr);
  mpc_config::TaskFile task = shippedTask();
  task.swing_trajectory_config.swing_height = 0.2345;
  reload(*updater, atlas_, task);
  EXPECT_EQ(referenceManager.getSwingTrajectoryPlanner()->getConfig().swingHeight, 0.2345);
}

TEST_F(MpcParameterUpdaterModuleTest, LocomotionHeuristicCoefficientsHotReloadButTheirListsDoNot) {
  CentroidalMpcConfig config = shipped_;
  config.task.locomotion_heuristics.base_pose = {"orientation_compensation"};
  config.task.locomotion_heuristics.orientation_compensation.pitch_per_forward_velocity = 0.05;
  const Built built = build(config, files_.urdfFile);
  ASSERT_NE(built.interface, nullptr);
  const std::shared_ptr<LocomotionHeuristicLayer> layer = built.interface->getLocomotionHeuristicLayerPtr();
  ASSERT_NE(layer, nullptr);
  ASSERT_NEAR(basePoseOffsetAtForwardSpeed(*layer, /*forwardSpeed=*/1.0).pitch, 0.05, 1.0e-12) << "the start-up did not list the heuristic";
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(built);
  ASSERT_NE(updater, nullptr);

  mpc_config::TaskFile task = config.task;
  task.locomotion_heuristics.orientation_compensation.pitch_per_forward_velocity = 0.09;
  reload(*updater, built, task);
  EXPECT_NEAR(basePoseOffsetAtForwardSpeed(*layer, /*forwardSpeed=*/1.0).pitch, 0.09, 1.0e-12) << "the coefficient did not reload";
  EXPECT_NEAR(basePoseOffsetAtForwardSpeed(*layer, /*forwardSpeed=*/0.5).pitch, 0.045, 1.0e-12) << "the reload sets the GAIN";

  // A coefficient the layer refuses keeps the running one.
  task.locomotion_heuristics.orientation_compensation.maximum_tilt = -1.0;
  task.locomotion_heuristics.orientation_compensation.pitch_per_forward_velocity = 0.11;
  reload(*updater, built, task);
  EXPECT_NEAR(basePoseOffsetAtForwardSpeed(*layer, /*forwardSpeed=*/1.0).pitch, 0.09, 1.0e-12) << "a refused reload changed the layer";

  // A changed list is structural: reported, not applied, and the running heuristic's coefficients still reload.
  task.locomotion_heuristics.orientation_compensation.maximum_tilt =
      config.task.locomotion_heuristics.orientation_compensation.maximum_tilt;
  task.locomotion_heuristics.orientation_compensation.pitch_per_forward_velocity = 0.06;
  task.locomotion_heuristics.base_pose = {"height_compensation"};
  task.locomotion_heuristics.height_compensation.height_offset = 0.03;
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                       testing::AllOf(testing::HasSubstr("next start"), testing::HasSubstr("locomotion_heuristics.base_pose"))))
      .Times(1);
  log.StartCapturingLogs();
  reload(*updater, built, task);
  log.StopCapturingLogs();
  const BasePoseOffset offset = basePoseOffsetAtForwardSpeed(*layer, /*forwardSpeed=*/1.0);
  EXPECT_NEAR(offset.pitch, 0.06, 1.0e-12) << "a changed list stopped the running heuristic's coefficients from reloading";
  EXPECT_EQ(offset.height, 0.0) << "height_compensation, newly listed, was switched on by a hot reload";
}

/** The Atlas planner's file as the start-up and the reload derive it from the model: no com_height, the ZMP box at 0. */
mpc_config::ContactPlanningFile derivedPlanningFile(mpc_config::ContactPlanningFile file) {
  file.shared.com_height.reset();
  file.zmp_support_region.half_width_x = 0.0;
  file.zmp_support_region.half_width_y = 0.0;
  // Plan in the pre-solve hook, so that the test starts no worker thread.
  file.planner.threading = "pre_solve_hook";
  return file;
}

TEST_F(MpcParameterUpdaterModuleTest, AContactPlanningReloadAppliesTheModelParametersBeforeValidating) {
  // A robot may leave out shared.com_height and write 0 for the ZMP box, the planner parameters that are properties of
  // its model, and have them derived. Such a file does not validate on its own; the reload converts it without validation and leaves
  // ContactPlannerModule::setConfig() to apply the model parameters first, exactly as the start-up does.
  if (!shipped_.contactPlanning.has_value()) GTEST_FAIL() << "the shipped Atlas has a contact planner's file";
  const mpc_config::ContactPlanningFile file = derivedPlanningFile(*shipped_.contactPlanning);
  absl::StatusOr<ContactPlanningConfig> unvalidated =
      contactPlanningConfigFromConfig(file, ContactPlanningValidation::kDeferUntilModelParametersApplied);
  ASSERT_TRUE(unvalidated.ok()) << unvalidated.status();
  EXPECT_FALSE(contactPlanningConfigFromConfig(file, ContactPlanningValidation::kValidate).ok())
      << "the file validates on its own: the test proves nothing";

  ContactPlanningModelParameters modelParameters;
  modelParameters.totalMass = 80.0;
  modelParameters.comHeight = 0.93;
  modelParameters.zmpHalfWidthX = 0.11;
  modelParameters.zmpHalfWidthY = 0.055;
  modelParameters.torsionalFrictionTorque = 12.0;
  modelParameters.doubleSupportYawCouple = 34.0;
  modelParameters.footYawOffsetLower = makeFeetArray(static_cast<scalar_t>(-0.4));
  modelParameters.footYawOffsetUpper = makeFeetArray(static_cast<scalar_t>(0.4));
  modelParameters.hipYawJoints.assign(kNumContacts, std::string());
  ContactPlanningConfig config = *unvalidated;
  modelParameters.applyTo(config);
  ASSERT_TRUE(config.validateStatus().ok()) << "with the model parameters applied the very same file is valid";
  std::shared_ptr<SwingTrajectoryPlanner> swingPlanner =
      std::make_shared<SwingTrajectoryPlanner>(SwingTrajectoryPlanner::Config{}, kNumContacts);
  absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>> manager = ContactPlanningReferenceManager::Create(
      GaitSchedule::Create(shipped_.reference, atlas_.interface->modelSettings()).value(), swingPlanner,
      atlas_.interface->getPinocchioInterface(), atlas_.interface->getEffectiveMpcRobotModel(), config);
  ASSERT_TRUE(manager.ok()) << manager.status();
  absl::StatusOr<std::shared_ptr<ContactPlannerModule>> module = ContactPlannerModule::Create(*manager, config, modelParameters);
  ASSERT_TRUE(module.ok()) << module.status();

  // The planner's file is watched beside the task file: the operator saves it, and the watcher picks it up.
  const std::string directory = freshDirectory("updater_contact_planning");
  const std::string planningFile = (std::filesystem::path(directory) / std::string(kContactPlanningFileName)).string();
  writeWatched(planningFile, contactPlanningFileText(file), /*seconds=*/1);
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      makeUpdater(atlas_, /*referenceManager=*/nullptr, watchingContactPlanningFile(planningFile));
  ASSERT_NE(updater, nullptr);
  updater->setContactPlannerModule(*module);
  mpc_config::ContactPlanningFile edited = file;
  edited.planner.max_solve_time = 0.234;
  writeWatched(planningFile, contactPlanningFileText(edited), /*seconds=*/2);
  runFileWatch(*updater, atlas_);

  const ContactPlanningConfig applied = (*module)->getConfig();
  EXPECT_NEAR(applied.planner.maxSolveTime, 0.234, 1.0e-12) << "the edited planner file never reached the module";
  EXPECT_THAT(applied.shared.comHeight, ::testing::Optional(::testing::DoubleNear(modelParameters.comHeight, /*max_abs_error=*/1.0e-12)))
      << "the reload has to re-apply the model-derived height";
  EXPECT_NEAR(applied.zmpSupportRegion.halfWidthX, modelParameters.zmpHalfWidthX, 1.0e-12);
  EXPECT_NEAR(applied.zmpSupportRegion.halfWidthY, modelParameters.zmpHalfWidthY, 1.0e-12);
  EXPECT_TRUE(applied.validateStatus().ok()) << "the planner was handed an unvalidated configuration";

  // The planner's file of an enqueued update reaches it too.
  edited.planner.max_solve_time = 0.345;
  reload(*updater, atlas_, shippedTask(), edited);
  EXPECT_NEAR((*module)->getConfig().planner.maxSolveTime, 0.345, 1.0e-12);
}

/******************************************************************************************************/
// The pathways: the enqueued update and the watched files
/******************************************************************************************************/

TEST_F(MpcParameterUpdaterModuleTest, EnqueuedUpdatesAreAppliedByTheNextSolveNewestFirst) {
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_);
  ASSERT_NE(updater, nullptr);
  const size_t running = atlas_.solver().getSettings().sqpIteration;
  const std::function<mpc_config::MpcParameterUpdate(size_t)> withIterations = [this](size_t iterations) {
    mpc_config::MpcParameterUpdate update;
    update.task = shippedTask();
    update.task.multiple_shooting.sqp_iteration = static_cast<int32_t>(iterations);
    return update;
  };
  const std::function<void()> solve = [&]() {
    updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, atlas_.interface->getInitialState(),
                          *atlas_.interface->getReferenceManagerPtr());
  };
  // Two updates from another thread before a solve: nothing applies before the solve, and the newest replaces the first.
  std::thread producer([&]() {
    updater->enqueueParameterUpdate(withIterations(running + 1));
    updater->enqueueParameterUpdate(withIterations(running + 2));
  });
  producer.join();
  EXPECT_EQ(atlas_.solver().getSettings().sqpIteration, running) << "an enqueued update was applied before the next solve";
  solve();
  EXPECT_EQ(atlas_.solver().getSettings().sqpIteration, running + 2) << "the newest update was not the one applied";

  // Updates streamed while the solver runs: every solve applies one that was enqueued, and the last one is applied.
  constexpr size_t kNumUpdates = 10;
  std::thread streamer([&]() {
    for (size_t k = 0; k < kNumUpdates; ++k) {
      updater->enqueueParameterUpdate(withIterations(running + 3 + k));
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  });
  for (size_t k = 0; k < kNumUpdates; ++k) {
    solve();
    const size_t applied = atlas_.solver().getSettings().sqpIteration;
    EXPECT_GE(applied, running + 2);
    EXPECT_LT(applied, running + 3 + kNumUpdates);
  }
  streamer.join();
  solve();
  EXPECT_EQ(atlas_.solver().getSettings().sqpIteration, running + 3 + kNumUpdates - 1) << "the last update was not applied";
}

TEST_F(MpcParameterUpdaterModuleTest, TheWatchedTaskFileIsReloadedWhenItChangesAndOnlyThen) {
  const std::string directory = freshDirectory("updater_task_file");
  const std::string taskFile = (std::filesystem::path(directory) / "task.textproto").string();
  writeWatched(taskFile, taskFileText(shippedTask()), /*seconds=*/1);
  std::unique_ptr<MpcParameterUpdaterModule> updater = makeUpdater(atlas_, /*referenceManager=*/nullptr, watchingTaskFile(taskFile));
  ASSERT_NE(updater, nullptr);
  const size_t running = atlas_.solver().getSettings().sqpIteration;
  const std::string reloadLine = "Applying in-place parameter updates";
  {
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr(reloadLine))).Times(0);
    log.StartCapturingLogs();
    runFileWatch(*updater, atlas_);
    log.StopCapturingLogs();
  }
  mpc_config::TaskFile task = shippedTask();
  task.multiple_shooting.sqp_iteration = static_cast<int32_t>(running) + 4;
  writeWatched(taskFile, taskFileText(task), /*seconds=*/2);
  runFileWatch(*updater, atlas_);
  EXPECT_EQ(atlas_.solver().getSettings().sqpIteration, running + 4) << "the edited task file was not reloaded";

  // A file that does not parse is reported with its position and applies nothing.
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       testing::AllOf(testing::HasSubstr("was not reloaded"), testing::HasSubstr("task.textproto:1:"))))
      .Times(1);
  log.StartCapturingLogs();
  writeWatched(taskFile, "multiple_shooting { sqp_iteration: many }\n", /*seconds=*/3);
  runFileWatch(*updater, atlas_);
  log.StopCapturingLogs();
  EXPECT_EQ(atlas_.solver().getSettings().sqpIteration, running + 4);
}

TEST_F(MpcParameterUpdaterModuleTest, AChangedReferenceFileReachesTheRegisteredReloadersAndARefusalIsReported) {
  const std::string directory = freshDirectory("updater_reference_file");
  const std::string referenceFile = (std::filesystem::path(directory) / "reference.textproto").string();
  writeWatched(referenceFile, referenceFileText(shipped_.reference), /*seconds=*/1);
  std::unique_ptr<MpcParameterUpdaterModule> updater =
      makeUpdater(atlas_, /*referenceManager=*/nullptr, watchingReferenceFile(referenceFile));
  ASSERT_NE(updater, nullptr);
  // As ProceduralMpcMotionManager::applyCommandLimits() refuses a value it cannot use.
  updater->addReferenceFileReloader(
      [](const ReferenceSettings& /*settings*/) { return absl::InvalidArgumentError("max_rotation_velocity is -1"); });
  std::vector<scalar_t> reloadedRotationVelocities;
  updater->addReferenceFileReloader([&reloadedRotationVelocities](const ReferenceSettings& settings) {
    reloadedRotationVelocities.push_back(settings.maxRotationVelocity);
    return absl::OkStatus();
  });
  runFileWatch(*updater, atlas_);
  EXPECT_TRUE(reloadedRotationVelocities.empty()) << "an untouched reference file triggered a reload";

  // The edited file's limits reach every reloader, the ones after a refusing one included, once.
  mpc_config::ReferenceFile edited = shipped_.reference;
  edited.max_rotation_velocity = 0.75;
  {
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                         testing::AllOf(testing::HasSubstr("Failed to reload"), testing::HasSubstr(referenceFile),
                                        testing::HasSubstr("max_rotation_velocity is -1"))))
        .Times(1);
    log.StartCapturingLogs();
    writeWatched(referenceFile, referenceFileText(edited), /*seconds=*/2);
    runFileWatch(*updater, atlas_);
    log.StopCapturingLogs();
  }
  EXPECT_EQ(reloadedRotationVelocities, std::vector<scalar_t>({0.75})) << "the reloader after the refusing one did not run once";

  // A file whose limits do not convert reaches no typed reloader: it is reported once, naming the file and the field,
  // and every running limit is kept.
  mpc_config::ReferenceFile refused = shipped_.reference;
  refused.max_linear_acceleration = std::numeric_limits<scalar_t>::quiet_NaN();
  {
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                         testing::AllOf(testing::HasSubstr("was not reloaded"), testing::HasSubstr(referenceFile),
                                        testing::HasSubstr("max_linear_acceleration"))))
        .Times(1);
    log.StartCapturingLogs();
    writeWatched(referenceFile, referenceFileText(refused), /*seconds=*/3);
    runFileWatch(*updater, atlas_);
    log.StopCapturingLogs();
  }
  EXPECT_EQ(reloadedRotationVelocities.size(), 1U) << "a reference file that does not convert reached a reloader";
}

TEST_F(MpcParameterUpdaterModuleTest, ReloadingCommandLimitsChangesTheTargetItProduces) {
  // The pelvis height is the observable one: commandedPositionToTargetTrajectories clamps the commanded delta to
  // max_delta_pelvis_height and adds default_base_height, so both reloaded values appear in the target it returns.
  const std::function<mpc_config::ReferenceFile(scalar_t, scalar_t)> referenceWith = [this](scalar_t maxDeltaPelvisHeight,
                                                                                            scalar_t defaultBaseHeight) {
    mpc_config::ReferenceFile reference = shipped_.reference;
    reference.max_delta_pelvis_height = maxDeltaPelvisHeight;
    reference.default_base_height = defaultBaseHeight;
    return reference;
  };
  const std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator> calculator =
      CentroidalMpcTargetTrajectoriesCalculator::Create(
          referenceWith(0.10, 0.90), atlas_.interface->getEffectiveMpcRobotModel(), atlas_.interface->getPinocchioInterface(),
          atlas_.interface->getCentroidalModelInfo(), atlas_.interface->mpcSettings().timeHorizon_)
          .value();
  const vector_t& state = atlas_.interface->getInitialState();
  const vector4_t deepCrouch(0.0, 0.0, -10.0, 0.0);
  const std::function<scalar_t()> commandedHeight = [&]() {
    const TargetTrajectories target = calculator->commandedPositionToTargetTrajectories(deepCrouch, /*initTime=*/0.0, state);
    return atlas_.interface->getEffectiveMpcRobotModel().getBasePosition(target.stateTrajectory.back())(2);
  };
  EXPECT_NEAR(commandedHeight(), 0.90 - 0.10, 1.0e-6);
  // What a reload of the reference file hands the calculator (MpcParameterUpdaterModule::ReferenceFileReloader).
  const absl::StatusOr<ReferenceSettings> reloaded = referenceSettingsFromConfig(referenceWith(0.25, 0.80));
  ASSERT_TRUE(reloaded.ok()) << reloaded.status();
  calculator->applyCommandLimits(*reloaded);
  EXPECT_NEAR(commandedHeight(), 0.80 - 0.25, 1.0e-6) << "the reloaded limits must be the ones the target is built from";
}

/******************************************************************************************************/
// The wiring every node uses
/******************************************************************************************************/

TEST_F(MpcParameterUpdaterModuleTest, TheSharedNodeWiringReloadsTheHeuristicLayerThePlannerAndTheReferenceFile) {
  // The robot's files in a directory of the test's own, with the planner and a heuristic switched on.
  CentroidalMpcConfig config = shipped_;
  if (!config.contactPlanning.has_value()) GTEST_FAIL() << "the shipped Atlas has a contact planner's file";
  config.task.contact_schedule_source = "contact_planner";
  config.task.locomotion_heuristics.base_pose = {"orientation_compensation"};
  config.task.locomotion_heuristics.orientation_compensation.pitch_per_forward_velocity = 0.05;
  config.contactPlanning->planner.threading = "pre_solve_hook";
  config.contactPlanning->hlip.step_width = 0.25;
  const std::string directory = freshDirectory("updater_shared_wiring");
  absl::StatusOr<CentroidalRobotFiles> files = writeConfig(directory, config, files_.urdfFile);
  ASSERT_TRUE(files.ok()) << files.status();
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created =
      CentroidalMpcInterface::Create(files->taskFile, files->urdfFile, files->referenceFile);
  ASSERT_TRUE(created.ok()) << created.status();
  const std::unique_ptr<CentroidalMpcInterface> interface = *std::move(created);
  ASSERT_NE(interface->getContactPlannerModulePtr(), nullptr) << "the planner's file beside the task file was not read";
  SqpMpc mpc(interface->mpcSettings(), interface->sqpSettings(), interface->getOptimalControlProblem(), interface->getInitializer());
  mpc.getSolverPtr()->setReferenceManager(interface->getReferenceManagerPtr());
  std::vector<scalar_t> reloadedRotationVelocities;
  absl::StatusOr<std::shared_ptr<MpcParameterUpdaterModule>> wired = makeCentroidalMpcParameterUpdater(
      &mpc, *interface, files->taskFile, files->referenceFile, {[&reloadedRotationVelocities](const ReferenceSettings& settings) {
        reloadedRotationVelocities.push_back(settings.maxRotationVelocity);
        return absl::OkStatus();
      }});
  ASSERT_TRUE(wired.ok()) << wired.status();
  MpcParameterUpdaterModule& updater = **wired;
  const std::function<void()> runUpdater = [&]() {
    for (size_t i = 0; i < 101; ++i) {
      updater.preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, interface->getInitialState(), *interface->getReferenceManagerPtr());
    }
  };
  const std::function<scalar_t()> pitchGain = [&interface]() {
    return basePoseOffsetAtForwardSpeed(*interface->getLocomotionHeuristicLayerPtr(), /*forwardSpeed=*/1.0).pitch;
  };
  ASSERT_NEAR(pitchGain(), 0.05, 1.0e-12);

  CentroidalMpcConfig edited = config;
  edited.task.locomotion_heuristics.orientation_compensation.pitch_per_forward_velocity = 0.09;
  edited.task.multiple_shooting.sqp_iteration = 12;
  writeWatched(files->taskFile, taskFileText(edited.task), /*seconds=*/1);
  runUpdater();
  ASSERT_EQ(dynamic_cast<SqpSolver&>(*mpc.getSolverPtr()).getSettings().sqpIteration, 12U) << "the task file was not reloaded at all";
  EXPECT_NEAR(pitchGain(), 0.09, 1.0e-12) << "the wired updater does not reach the interface's locomotion-heuristic layer";
  if (!edited.contactPlanning.has_value()) GTEST_FAIL() << "the copy of the configuration lost the planner's file";
  edited.contactPlanning->hlip.step_width = 0.27;
  writeWatched((std::filesystem::path(directory) / std::string(kContactPlanningFileName)).string(),
               contactPlanningFileText(*edited.contactPlanning), /*seconds=*/2);
  runUpdater();
  EXPECT_NEAR(interface->getContactPlannerModulePtr()->getConfig().hlip.stepWidth, 0.27, 1.0e-12)
      << "the wired updater does not reach the contact planner module";
  mpc_config::ReferenceFile reference = config.reference;
  reference.max_rotation_velocity = 0.75;
  writeWatched(files->referenceFile, referenceFileText(reference), /*seconds=*/3);
  runUpdater();
  EXPECT_EQ(reloadedRotationVelocities, std::vector<scalar_t>({0.75})) << "the reference file's reloaders were not registered";
}

}  // namespace
}  // namespace ocs2::humanoid
