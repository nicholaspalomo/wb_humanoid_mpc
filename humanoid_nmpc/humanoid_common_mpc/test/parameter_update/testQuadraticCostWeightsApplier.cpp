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

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_centroidal_model/FactoryFunctions.h"
#include "ocs2_core/cost/QuadraticStateCost.h"
#include "ocs2_core/initialization/DefaultInitializer.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_sqp/SqpSettings.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/InputQuadraticCost.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateCost.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateInputCost.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"
#include "humanoid_common_mpc/parameter_update/QuadraticCostWeightsApplier.h"
#include "humanoid_common_mpc/parameter_update/ReferenceManagerApplier.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/ArmSwing.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/yaw_pitch_roll.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/LayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/**
 * QuadraticCostWeightsApplier on the quadratic costs the start-up factory
 * builds (the Tuning* costs and the input-only InputQuadraticCost): a hot
 * reload of state_weights, input_weights, final_state_weights and
 * terminal_cost_scaling reaches every one of them in every worker's problem -
 * no cast to another type, so no WARNING - and leaves each with exactly the
 * weights a fresh start with the reloaded file builds, through the same one
 * conversion (trackingStateWeightsFromConfig(),
 * terminalStateWeightsFromConfig()), the CoM + ACoM tracking cost's zeroed
 * base-pose block included. The reload moves every orientation row to a nonzero
 * value, the yaw that every shipped file leaves at 0 included. The robot model
 * is the centroidal quaternion layout (test/support/LayoutRobotModel.h): the
 * state has one entry more than the tuning layout the weights are written in,
 * which is the state's tangent. Builds no CppAD model.
 */
namespace ocs2::humanoid {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::HasSubstr;

constexpr char kTaskFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr char kReferenceFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/"
    "reference.textproto";
constexpr char kUrdfFile[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

/** The value of `converted`; a refusal fails the test. */
template <typename T>
T converted(absl::StatusOr<T> converted) {
  EXPECT_TRUE(converted.ok()) << converted.status();
  return converted.ok() ? *std::move(converted) : T();
}

/** The task file `task` with every weight the applier reloads moved, the
 * orientation rows to nonzero values. */
mpc_config::TaskFile reloadedTask(const mpc_config::TaskFile& task) {
  mpc_config::TaskFile reloaded = task;
  reloaded.state_weights.scaling *= 2.0;
  reloaded.state_weights.base_orientation =
      mpc_config::YawPitchRoll{.yaw = 7.0, .pitch = 11.0, .roll = 13.0};
  reloaded.final_state_weights.scaling *= 3.0;
  reloaded.final_state_weights.base_orientation =
      mpc_config::YawPitchRoll{.yaw = 17.0, .pitch = 19.0, .roll = 23.0};
  reloaded.input_weights.scaling *= 1.5;
  reloaded.terminal_cost_scaling =
      2.0 * task.terminal_cost_scaling.value_or(1.0);
  return reloaded;
}

/**
 * The shipped DRC Atlas's centroidal layout with a reference manager, and the
 * factory's quadratic terms on it in a problem of two workers.
 */
class QuadraticCostWeightsApplierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const absl::StatusOr<std::string> taskFile =
        robot::resolveResourcePath(kTaskFile);
    const absl::StatusOr<std::string> referenceFile =
        robot::resolveResourcePath(kReferenceFile);
    const absl::StatusOr<std::string> urdfFile =
        robot::resolveResourcePath(kUrdfFile);
    ASSERT_TRUE(taskFile.ok() && referenceFile.ok() && urdfFile.ok())
        << "the shipped DRC Atlas files are not in the runfiles";
    task_ = converted(loadTaskFile(*taskFile));
    ASSERT_TRUE(task_.terminal_cost_scaling.has_value())
        << "the shipped Atlas has no terminal_cost_scaling";
    absl::StatusOr<ModelSettings> settings = ModelSettings::Create(
        task_, *urdfFile, "testQuadraticCostWeightsApplier", /*verbose=*/false);
    ASSERT_TRUE(settings.ok()) << settings.status();
    settings_ = std::make_unique<ModelSettings>(*std::move(settings));
    absl::StatusOr<PinocchioInterface> pinocchioInterface =
        loadCustomPinocchioInterface(task_, *urdfFile, *settings_);
    ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ =
        std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    model_ = std::make_unique<test::LayoutRobotModel<scalar_t>>(
        *settings_, StateInputLayout::Mpc::kCentroidal);
    modelAd_ = std::make_unique<test::LayoutRobotModel<ad_scalar_t>>(
        *settings_, StateInputLayout::Mpc::kCentroidal);
    // The quaternion layout: the base orientation's four state slots have a
    // three-row tangent.
    ASSERT_EQ(model_->getStateDim(), model_->getStateTangentDim() + 1);
    ASSERT_EQ(model_->getStateTangentDim(),
              tuningDimension(stateInputLayout(
                  *settings_, StateInputLayout::Mpc::kCentroidal)));
    referenceManager_ = std::make_unique<SwitchedModelReferenceManager>(
        converted(GaitSchedule::Create(*referenceFile, *settings_)),
        std::make_unique<SwingTrajectoryPlanner>(
            SwingTrajectoryPlanner::Config(), /*numFeet=*/2),
        *pinocchioInterface_, *model_);
    layout_ = stateInputLayout(*settings_, StateInputLayout::Mpc::kCentroidal);
  }

  /** The factory of `task` on the fixture's objects, the CoM + ACoM tracking
   * cost listed when `comAndAcomListed`. */
  std::unique_ptr<HumanoidCostConstraintFactory> factoryOf(
      const mpc_config::TaskFile& task, bool comAndAcomListed) const {
    std::unique_ptr<HumanoidCostConstraintFactory> factory =
        std::make_unique<HumanoidCostConstraintFactory>(
            &task, StateInputLayout::Mpc::kCentroidal, *referenceManager_,
            *pinocchioInterface_, *model_, *modelAd_, *settings_);
    factory->setComAndAcomTrackingCostListed(comAndAcomListed);
    return factory;
  }

  /**
   * The problem of the shipped file's four quadratic terms, as the factory
   * builds them, and the CoM + ACoM tracking cost when `comAndAcomListed`.
   */
  OptimalControlProblem problemOf(bool comAndAcomListed) const {
    const std::unique_ptr<HumanoidCostConstraintFactory> factory =
        factoryOf(task_, comAndAcomListed);
    OptimalControlProblem problem;
    problem.costPtr->add(kStateInputQuadraticCostTerm,
                         converted(factory->makeStateInputQuadraticCost()));
    problem.costPtr->add(kStateQuadraticCostTerm,
                         converted(factory->makeStateQuadraticCost()));
    problem.costPtr->add(kInputQuadraticCostTerm,
                         converted(factory->makeInputQuadraticCost()));
    problem.finalCostPtr->add(kTerminalCostTerm,
                              converted(factory->makeTerminalCost()));
    if (comAndAcomListed) {
      const CentroidalModelInfo info =
          centroidal_model::createCentroidalModelInfo(
              *pinocchioInterface_, CentroidalModelType::FullCentroidalDynamics,
              vector_t::Zero(static_cast<Eigen::Index>(
                  settings_->mpcModelJointNames.size())),
              settings_->contactNames3DoF, settings_->contactNames6DoF);
      problem.stateCostPtr->add(
          std::string(ComAndAcomTrackingCost::kRunningTermName),
          converted(factory->makeComAndAcomTrackingCost(info)));
    }
    return problem;
  }

  /** A solver of two workers on `problem`, each with its own clone of it. */
  std::unique_ptr<SqpSolver> solverOf(
      const OptimalControlProblem& problem) const {
    sqp::Settings settings;
    settings.nThreads = 2;
    settings.enableLogging = false;
    const DefaultInitializer initializer(model_->getInputDim());
    return std::make_unique<SqpSolver>(settings, problem, initializer);
  }

  /** Applies `reloaded` with QuadraticCostWeightsApplier to `solver`, expecting
   * no WARNING. */
  void applyWithoutWarning(const mpc_config::TaskFile& reloaded,
                           SqpSolver& solver) const {
    HotUpdateTarget target(&reloaded, &solver, /*referenceManager=*/nullptr,
                           &layout_, model_->getInputDim(), "test reload");
    QuadraticCostWeightsApplier applier(
        QuadraticCostWeightsApplier::InputCost::kInputWeights);
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _, _)).Times(0);
    log.StartCapturingLogs();
    applier.apply(target);
    log.StopCapturingLogs();
  }

  /** Expects every worker of `solver` to weigh with the weights a fresh start
   * with `task` builds. */
  void expectFreshStartWeights(const mpc_config::TaskFile& task,
                               bool comAndAcomListed,
                               SqpSolver& solver) const {
    const std::unique_ptr<HumanoidCostConstraintFactory> fresh =
        factoryOf(task, comAndAcomListed);
    const std::unique_ptr<StateInputCost> freshStateInput =
        converted(fresh->makeStateInputQuadraticCost());
    const std::unique_ptr<StateInputCost> freshInput =
        converted(fresh->makeInputQuadraticCost());
    const std::unique_ptr<StateCost> freshTerminal =
        converted(fresh->makeTerminalCost());
    const matrix_t& Q =
        dynamic_cast<const TuningQuadraticStateInputCost&>(*freshStateInput)
            .getStateWeights();
    const matrix_t& R =
        dynamic_cast<const InputQuadraticCost&>(*freshInput).getInputWeights();
    const matrix_t& Q_final =
        dynamic_cast<const TuningQuadraticStateCost&>(*freshTerminal)
            .getStateWeights();

    // The one conversion both paths use.
    const BasePoseWeights basePose =
        comAndAcomListed ? BasePoseWeights::kZeroedForComAndAcomTracking
                         : BasePoseWeights::kKept;
    EXPECT_TRUE(Q ==
                converted(trackingStateWeightsFromConfig(
                    task.state_weights, layout_, "state_weights", basePose)));
    EXPECT_TRUE(Q_final == converted(terminalStateWeightsFromConfig(
                               task.final_state_weights,
                               task.terminal_cost_scaling.value_or(1.0),
                               layout_, "final_state_weights", basePose)));

    // Written in the tuning layout: the state's tangent, not its slots.
    EXPECT_EQ(static_cast<size_t>(Q.rows()), model_->getStateTangentDim());
    EXPECT_EQ(static_cast<size_t>(Q_final.rows()),
              model_->getStateTangentDim());
    ASSERT_EQ(solver.getOcpDefinitions().size(), 2U);
    for (OptimalControlProblem& worker : solver.getOcpDefinitions()) {
      const TuningQuadraticStateInputCost& stateInput =
          worker.costPtr->get<TuningQuadraticStateInputCost>(
              kStateInputQuadraticCostTerm);
      EXPECT_TRUE(stateInput.getStateWeights() == Q)
          << "stateInputQuadraticCost's Q";
      EXPECT_TRUE(stateInput.getInputWeights() == R)
          << "stateInputQuadraticCost's R";
      const TuningQuadraticStateInputCost& state =
          worker.costPtr->get<TuningQuadraticStateInputCost>(
              kStateQuadraticCostTerm);
      EXPECT_TRUE(state.getStateWeights() == Q) << "stateQuadraticCost's Q";
      EXPECT_TRUE(state.getInputWeights().isZero(0.0))
          << "stateQuadraticCost weighs the input";
      EXPECT_TRUE(
          worker.costPtr->get<InputQuadraticCost>(kInputQuadraticCostTerm)
              .getInputWeights() == R)
          << "inputQuadraticCost's R";
      EXPECT_TRUE(
          worker.finalCostPtr->get<TuningQuadraticStateCost>(kTerminalCostTerm)
              .getStateWeights() == Q_final)
          << "Q_final";
    }
  }

  mpc_config::TaskFile task_;
  std::unique_ptr<ModelSettings> settings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<test::LayoutRobotModel<scalar_t>> model_;
  std::unique_ptr<test::LayoutRobotModel<ad_scalar_t>> modelAd_;
  std::unique_ptr<SwitchedModelReferenceManager> referenceManager_;
  StateInputLayout layout_;
};

TEST_F(QuadraticCostWeightsApplierTest,
       AReloadReachesEveryTuningCostWithoutAWarningAsAFreshStart) {
  const std::unique_ptr<SqpSolver> solver =
      solverOf(problemOf(/*comAndAcomListed=*/false));
  const mpc_config::TaskFile reloaded = reloadedTask(task_);
  // Positive control: the reload moves every weight, so a term it did not reach
  // keeps a different one.
  const matrix_t shippedQ = converted(trackingStateWeightsFromConfig(
      task_.state_weights, layout_, "state_weights", BasePoseWeights::kKept));
  ASSERT_FALSE(shippedQ == converted(trackingStateWeightsFromConfig(
                               reloaded.state_weights, layout_, "state_weights",
                               BasePoseWeights::kKept)));

  applyWithoutWarning(reloaded, *solver);
  expectFreshStartWeights(reloaded, /*comAndAcomListed=*/false, *solver);

  // The orientation rows, yaw included, reach the costs as written.
  const absl::StatusOr<size_t> orientationRow =
      tuningSegmentStart(tuningLayoutOf(layout_), "base_orientation");
  ASSERT_TRUE(orientationRow.ok()) << orientationRow.status();
  const Eigen::Index yawRow = static_cast<Eigen::Index>(*orientationRow);
  const matrix_t& Q =
      solver->getOcpDefinitions()
          .front()
          .costPtr->get<TuningQuadraticStateInputCost>(kStateQuadraticCostTerm)
          .getStateWeights();
  EXPECT_EQ(Q(yawRow, yawRow), reloaded.state_weights.scaling * 7.0);
  EXPECT_EQ(Q(yawRow + 2, yawRow + 2), reloaded.state_weights.scaling * 13.0);
}

TEST_F(
    QuadraticCostWeightsApplierTest,
    UnderTheComAndAcomTrackingCostTheReloadZeroesTheBasePoseBlockAsAFreshStart) {
  const std::unique_ptr<SqpSolver> solver =
      solverOf(problemOf(/*comAndAcomListed=*/true));
  const mpc_config::TaskFile reloaded = reloadedTask(task_);
  applyWithoutWarning(reloaded, *solver);
  expectFreshStartWeights(reloaded, /*comAndAcomListed=*/true, *solver);
  const absl::StatusOr<size_t> positionRow =
      tuningSegmentStart(tuningLayoutOf(layout_), "base_position");
  ASSERT_TRUE(positionRow.ok()) << positionRow.status();
  const matrix_t& Q =
      solver->getOcpDefinitions()
          .back()
          .costPtr
          ->get<TuningQuadraticStateInputCost>(kStateInputQuadraticCostTerm)
          .getStateWeights();
  EXPECT_TRUE(Q.block(static_cast<Eigen::Index>(*positionRow),
                      static_cast<Eigen::Index>(*positionRow), 6, 6)
                  .isZero(0.0));
}

TEST_F(QuadraticCostWeightsApplierTest,
       ANegativeOrNanWeightIsRefusedLiveAndTheRunningWeightsAreKept) {
  const std::unique_ptr<SqpSolver> solver =
      solverOf(problemOf(/*comAndAcomListed=*/false));
  // A weight below 0 would make the Gauss-Newton Hessian indefinite while the
  // robot runs: every corruption is refused with a warning, and every worker
  // keeps the weights of a fresh start with the shipped file (the block refused
  // keeps its running value, the others apply the shipped ones again).
  const std::vector<std::function<void(mpc_config::TaskFile&)>> corruptions = {
      [](mpc_config::TaskFile& t) { t.state_weights.scaling = -1.0; },
      [](mpc_config::TaskFile& t) {
        t.state_weights.scaling = std::numeric_limits<double>::quiet_NaN();
      },
      [](mpc_config::TaskFile& t) {
        t.state_weights.joint_positions.front().value = -5.0;
      },
      [](mpc_config::TaskFile& t) { t.final_state_weights.scaling = -1.0; },
      [](mpc_config::TaskFile& t) { t.terminal_cost_scaling = -2.0; },
      [](mpc_config::TaskFile& t) { t.input_weights.scaling = -1.0; },
  };
  for (const std::function<void(mpc_config::TaskFile&)>& corrupt :
       corruptions) {
    // From the shipped file, so that the blocks the corruption leaves valid
    // apply the shipped weights again.
    mpc_config::TaskFile refused = task_;
    corrupt(refused);
    HotUpdateTarget target(&refused, solver.get(), /*referenceManager=*/nullptr,
                           &layout_, model_->getInputDim(), "test reload");
    QuadraticCostWeightsApplier applier(
        QuadraticCostWeightsApplier::InputCost::kInputWeights);
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    // A negative value is refused as below 0, NaN as not a finite number;
    // either way the block is not applied.
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                         AllOf(HasSubstr("was not applied"),
                               AnyOf(HasSubstr("at least 0"),
                                     HasSubstr("not a finite number")))))
        .Times(::testing::AtLeast(1));
    log.StartCapturingLogs();
    applier.apply(target);
    log.StopCapturingLogs();
    expectFreshStartWeights(task_, /*comAndAcomListed=*/false, *solver);
  }
}

// The arm swing's generator, a name and no number, which the live-update parity
// tests do not perturb: applied live (ReferenceManagerApplier), it is the
// generator a fresh start converts from the same file, either way.
TEST_F(QuadraticCostWeightsApplierTest,
       TheArmSwingGeneratorIsAppliedLiveAsAFreshStartReadsIt) {
  const std::unique_ptr<SqpSolver> solver =
      solverOf(problemOf(/*comAndAcomListed=*/false));
  for (const char* absl_nonnull generator : {"none", "procedural"}) {
    mpc_config::TaskFile edited = task_;
    edited.arm_swing.generator = generator;
    HotUpdateTarget target(&edited, solver.get(), referenceManager_.get(),
                           &layout_, model_->getInputDim(), "test reload");
    ReferenceManagerApplier applier;
    applier.apply(target);
    const ArmSwingSettings fresh =
        converted(armSwingSettingsFromConfig(edited.arm_swing));
    EXPECT_EQ(referenceManager_->getArmSwingSettings().generator,
              fresh.generator)
        << generator;
  }
}

/** Negative control: a term of another type under a quadratic cost's name is
 * the WARNING path, which the tests above exclude. */
TEST_F(QuadraticCostWeightsApplierTest,
       ATermOfAnotherTypeIsReportedAsAFailedUpdate) {
  OptimalControlProblem problem = problemOf(/*comAndAcomListed=*/false);
  ASSERT_NE(problem.finalCostPtr->extract(kTerminalCostTerm), nullptr);
  const Eigen::Index stateDim =
      static_cast<Eigen::Index>(model_->getStateDim());
  problem.finalCostPtr->add(kTerminalCostTerm,
                            std::make_unique<QuadraticStateCost>(
                                matrix_t::Identity(stateDim, stateDim)));
  const std::unique_ptr<SqpSolver> solver = solverOf(problem);
  const mpc_config::TaskFile reloaded = reloadedTask(task_);
  HotUpdateTarget target(&reloaded, solver.get(), /*referenceManager=*/nullptr,
                         &layout_, model_->getInputDim(), "test reload");
  QuadraticCostWeightsApplier applier(
      QuadraticCostWeightsApplier::InputCost::kInputWeights);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       AllOf(HasSubstr("Failed to update"),
                             HasSubstr(kTerminalCostTerm))))
      .Times(::testing::AtLeast(1));
  log.StartCapturingLogs();
  applier.apply(target);
  log.StopCapturingLogs();
}

}  // namespace
}  // namespace ocs2::humanoid
