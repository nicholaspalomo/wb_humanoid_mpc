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

#include "pinocchio/fwd.hpp"

#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/MpcEquivalenceFixture.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/globals.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/NumericTraits.h"
#include "ocs2_core/control/ControllerBase.h"
#include "ocs2_core/control/ControllerType.h"
#include "ocs2_core/control/ManifoldLinearController.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_data/TimeDiscretization.h"
#include "ocs2_sqp/SqpSolver.h"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/EulerChartStateManifold.h"

namespace ocs2::humanoid {
namespace {

using validation::EulerStateLayout;
using validation::EulerStateSegment;
using validation::EulerStateSegmentKind;

/** [m] How far ahead along the world x axis the walking reference's target is,
 * and [s] when. */
constexpr scalar_t kWalkingTargetDistance = 1.0;
constexpr scalar_t kWalkingTargetTime = 2.0;
/** The walking reference's single supports. */
constexpr size_t kWalkingSteps = 4;

/** Uniform in [low, high]; exactly `low` for an empty interval, which a zero
 * bound gives. */
scalar_t uniform(scalar_t low, scalar_t high, std::mt19937& generator) {
  if (!(low < high)) return low;
  std::uniform_real_distribution<scalar_t> distribution(low, high);
  return distribution(generator);
}

/** One expected block of the Euler state the oracle evaluates: its kind and
 * rows (0: the joints'). */
struct ExpectedSegment {
  EulerStateSegmentKind kind;
  size_t dim;
};

/** OK when `layout` is the oracle's Euler state (EulerMpcOracle.h) of
 * `jointDim` joints. */
absl::Status checkOracleLayout(const EulerStateLayout& layout,
                               bool wholeBody,
                               size_t jointDim) {
  constexpr EulerStateSegmentKind kE = EulerStateSegmentKind::kEuclidean;
  const std::vector<ExpectedSegment> expected =
      wholeBody
          ? std::vector<
                ExpectedSegment>{{kE, 3},
                                 {EulerStateSegmentKind::kEulerZyxAngles, 3},
                                 {kE, jointDim},
                                 {kE, 3},
                                 {EulerStateSegmentKind::kEulerZyxRates, 3},
                                 {kE, jointDim}}
          : std::vector<ExpectedSegment>{
                {kE, 3},
                {kE, 3},
                {kE, 3},
                {EulerStateSegmentKind::kEulerZyxAngles, 3},
                {kE, jointDim}};
  bool matches = layout.segments.size() == expected.size();
  for (size_t i = 0; matches && i < expected.size(); ++i) {
    matches = layout.segments[i].kind == expected[i].kind &&
              layout.segments[i].dim == expected[i].dim;
  }
  if (!matches) {
    return absl::InvalidArgumentError(
        absl::StrCat("[MpcEquivalenceFixture] the ", layout.name,
                     " tuning layout of ", validation::getEulerStateDim(layout),
                     " rows is not the oracle's Euler state"));
  }
  return absl::OkStatus();
}

/**
 * Whether a Euclidean block of the Euler state is a world-frame vector (the
 * momentum, the base position and linear velocity) or a joint block; ABSL_CHECK
 * on any other name, which a new block of the tuning layout would need sorted.
 */
bool isWorldFrameSegment(absl::string_view name) {
  if (name == "normalized_linear_momentum" ||
      name == "normalized_angular_momentum" || name == "base_position" ||
      name == "base_linear_velocity") {
    return true;
  }
  ABSL_CHECK(name == "joint_positions" || name == "joint_velocities")
      << "an Euler state block the world action does not know: " << name;
  return false;
}

::testing::AssertionResult shapesDiffer(const matrix_t& actual,
                                        const matrix_t& expected) {
  return ::testing::AssertionFailure()
         << "size " << actual.rows() << " x " << actual.cols() << ", expected "
         << expected.rows() << " x " << expected.cols();
}

}  // namespace

std::vector<std::string> mpcEquivalenceConfigurationNames() {
  const char* absl_nullable name =
      std::getenv(kMpcEquivalenceConfigurationVariable);
  if (name != nullptr && name[0] != '\0') return {std::string(name)};
  std::vector<std::string> names;
  for (const validation::RobotConfiguration& robot :
       validation::robotConfigurations())
    names.push_back(robot.name);
  return names;
}

std::string mpcEquivalenceTestName(
    const ::testing::TestParamInfo<std::string>& info) {
  return info.param;
}

absl::StatusOr<std::unique_ptr<MpcEquivalenceFixture>>
MpcEquivalenceFixture::Create(absl::string_view robotName) {
  // WrapUnique: the constructor is private (ToTW #134).
  std::unique_ptr<MpcEquivalenceFixture> fixture =
      absl::WrapUnique(new MpcEquivalenceFixture());
  ASSIGN_OR_RETURN(fixture->robot_,
                   validation::findRobotConfiguration(robotName));
  const validation::RobotConfiguration& robot = fixture->robot_;
  const pinocchio::Model* absl_nullable pinocchioModel = nullptr;
  switch (robot.formulation) {
    case validation::MpcFormulation::kCentroidal: {
      ASSIGN_OR_RETURN(fixture->centroidal_, CentroidalMpcInterface::Create(
                                                 robot.taskFile, robot.urdfFile,
                                                 robot.referenceFile));
      fixture->manifold_ = fixture->centroidal_->getStateManifold();
      pinocchioModel =
          &fixture->centroidal_->getPinocchioInterface().getModel();
      const BasisInputsModelDecorator<scalar_t>* absl_nullable basis =
          fixture->centroidal_->getBasisDecoratorPtr();
      if (basis != nullptr) {
        fixture->oracleOptions_.contactInput = OracleContactInput::kLocalBasis;
        fixture->oracleOptions_.localBasisMatrices = basis->getBasisMatrices();
      }
      break;
    }
    case validation::MpcFormulation::kWholeBody: {
      ASSIGN_OR_RETURN(fixture->wholeBody_,
                       WBMpcInterface::Create(robot.taskFile, robot.urdfFile,
                                              robot.referenceFile));
      fixture->manifold_ = fixture->wholeBody_->getStateManifold();
      pinocchioModel = &fixture->wholeBody_->getPinocchioInterface().getModel();
      fixture->oracleOptions_.formulation = OracleFormulation::kWholeBody;
      break;
    }
  }
  if (fixture->manifold_ == nullptr || pinocchioModel == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("[MpcEquivalenceFixture] ", robot.name,
                     " has no state manifold: not the quaternion layout"));
  }
  const ModelSettings& settings = fixture->modelSettings();
  if (settings.contactNames.size() != kNumContacts) {
    return absl::InvalidArgumentError(
        absl::StrCat("[MpcEquivalenceFixture] ", robot.name, " has ",
                     settings.contactNames.size(), " contacts"));
  }
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    fixture->oracleOptions_.contactFrameNames[contact] =
        settings.contactNames[contact];
  }
  fixture->eulerLayout_ = validation::eulerStateLayoutOf(stateInputLayout(
      settings, fixture->isWholeBody() ? StateInputLayout::Mpc::kWholeBody
                                       : StateInputLayout::Mpc::kCentroidal));
  RETURN_IF_ERROR(checkOracleLayout(fixture->eulerLayout_,
                                    fixture->isWholeBody(),
                                    fixture->model().getJointDim()));
  ASSIGN_OR_RETURN(
      fixture->oracle_,
      EulerMpcOracle::Create(*pinocchioModel, fixture->oracleOptions_));
  const MpcRobotModelBase<scalar_t>& model = fixture->model();
  if (static_cast<size_t>(fixture->oracle_->stateDim()) !=
          validation::getEulerStateDim(fixture->eulerLayout_) ||
      validation::getPhysicalStateDim(fixture->eulerLayout_) !=
          model.getStateDim() ||
      static_cast<size_t>(fixture->oracle_->inputDim()) !=
          model.getInputDim() ||
      fixture->manifold_->getAmbientDim() != model.getStateDim() ||
      fixture->manifold_->getTangentDim() !=
          validation::getEulerStateDim(fixture->eulerLayout_)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[MpcEquivalenceFixture] the sizes of ", robot.name,
        " do not agree: Euler state ", fixture->oracle_->stateDim(),
        ", MPC state ", model.getStateDim(), ", oracle input ",
        fixture->oracle_->inputDim(), ", MPC input ", model.getInputDim()));
  }
  return fixture;
}

const ModelSettings& MpcEquivalenceFixture::modelSettings() const {
  return centroidal_ != nullptr ? centroidal_->modelSettings()
                                : wholeBody_->modelSettings();
}

const MpcRobotModelBase<scalar_t>& MpcEquivalenceFixture::model() const {
  return centroidal_ != nullptr ? centroidal_->getEffectiveMpcRobotModel()
                                : wholeBody_->getMpcRobotModel();
}

const PinocchioInterface& MpcEquivalenceFixture::pinocchioInterface() const {
  return centroidal_ != nullptr ? centroidal_->getPinocchioInterface()
                                : wholeBody_->getPinocchioInterface();
}

std::shared_ptr<const StateManifold> MpcEquivalenceFixture::eulerChartPtr()
    const {
  return std::make_shared<const EulerChartStateManifold>(manifold_,
                                                         eulerLayout_);
}

const OptimalControlProblem& MpcEquivalenceFixture::problem() const {
  return centroidal_ != nullptr ? centroidal_->getOptimalControlProblem()
                                : wholeBody_->getOptimalControlProblem();
}

absl::StatusOr<std::unique_ptr<EulerMpcOracle>>
MpcEquivalenceFixture::oracleUnderGravity(const vector3_t& gravity) const {
  EulerMpcOracleOptions options = oracleOptions_;
  options.gravity = gravity;
  pinocchio::Model quaternionModel;
  if (centroidal_ != nullptr) {
    quaternionModel = centroidal_->getPinocchioInterface().getModel();
  } else {
    quaternionModel = wholeBody_->getPinocchioInterface().getModel();
  }
  return EulerMpcOracle::Create(quaternionModel, std::move(options));
}

scalar_t MpcEquivalenceFixture::nodeTimeStep() const {
  return sqpSettings().dt;
}

scalar_t MpcEquivalenceFixture::timeHorizon() const {
  return centroidal_ != nullptr ? centroidal_->mpcSettings().timeHorizon_
                                : wholeBody_->mpcSettings().timeHorizon_;
}

const sqp::Settings& MpcEquivalenceFixture::sqpSettings() const {
  return centroidal_ != nullptr ? centroidal_->sqpSettings()
                                : wholeBody_->sqpSettings();
}

vector_t MpcEquivalenceFixture::quaternionStateOf(
    const vector_t& eulerState) const {
  absl::StatusOr<vector_t> state =
      validation::physicalStateFromEulerState(eulerState, eulerLayout_);
  ABSL_CHECK(state.ok()) << state.status();
  return *std::move(state);
}

matrix_t MpcEquivalenceFixture::chartJacobian(
    const vector_t& eulerState) const {
  absl::StatusOr<matrix_t> jacobian =
      validation::eulerChartJacobian(eulerState, eulerLayout_);
  ABSL_CHECK(jacobian.ok()) << jacobian.status();
  return *std::move(jacobian);
}

matrix_t MpcEquivalenceFixture::physicalTangentJacobian(
    const vector_t& eulerState) const {
  absl::StatusOr<matrix_t> jacobian =
      validation::physicalTangentJacobian(eulerState, eulerLayout_);
  ABSL_CHECK(jacobian.ok()) << jacobian.status();
  return *std::move(jacobian);
}

vector_t MpcEquivalenceFixture::eulerStateOf(
    const vector_t& quaternionState) const {
  absl::StatusOr<vector_t> eulerState =
      validation::eulerStateFromPhysicalState(quaternionState, eulerLayout_);
  ABSL_CHECK(eulerState.ok()) << eulerState.status();
  return *std::move(eulerState);
}

matrix_t MpcEquivalenceFixture::ambientWorldRotation(
    const vector4_t& worldRotation) const {
  return worldAction(worldRotation, WorldActionSpace::kAmbient);
}

matrix_t MpcEquivalenceFixture::tangentWorldRotation(
    const vector4_t& worldRotation) const {
  return worldAction(worldRotation, WorldActionSpace::kTangent);
}

matrix_t MpcEquivalenceFixture::inputWorldRotation(
    const vector4_t& worldRotation) const {
  const matrix3_t rotation = quaternionRotationMatrix(worldRotation);
  matrix_t action =
      matrix_t::Identity(oracle_->inputDim(), oracle_->inputDim());
  if (oracleOptions_.contactInput == OracleContactInput::kLocalBasis)
    return action;
  for (Eigen::Index contact = 0;
       contact < static_cast<Eigen::Index>(kNumContacts); ++contact) {
    action.block<3, 3>(6 * contact, 6 * contact) = rotation;
    action.block<3, 3>(6 * contact + 3, 6 * contact + 3) = rotation;
  }
  return action;
}

matrix_t MpcEquivalenceFixture::worldAction(const vector4_t& worldRotation,
                                            WorldActionSpace space) const {
  const matrix3_t rotation = quaternionRotationMatrix(worldRotation);
  const Eigen::Index dim =
      space == WorldActionSpace::kAmbient
          ? static_cast<Eigen::Index>(
                validation::getPhysicalStateDim(eulerLayout_))
          : static_cast<Eigen::Index>(
                validation::getEulerStateDim(eulerLayout_));
  matrix_t action = matrix_t::Identity(dim, dim);
  Eigen::Index row = 0;
  for (const EulerStateSegment& segment : eulerLayout_.segments) {
    const Eigen::Index segmentDim = static_cast<Eigen::Index>(segment.dim);
    if (segment.kind == EulerStateSegmentKind::kEulerZyxAngles &&
        space == WorldActionSpace::kAmbient) {
      for (Eigen::Index column = 0; column < 4; ++column) {
        action.block<4, 1>(row, row + column) = quaternionProduct(
            worldRotation, vector4_t(vector4_t::Unit(column)));
      }
      row += 4;
      continue;
    }
    const bool worldFrame = segment.kind == EulerStateSegmentKind::kEuclidean &&
                            isWorldFrameSegment(segment.name);
    if (worldFrame) action.block<3, 3>(row, row) = rotation;
    row += segmentDim;
  }
  return action;
}

vector_t MpcEquivalenceFixture::randomEulerState(
    const EulerSamplingBounds& bounds, std::mt19937& generator) const {
  const pinocchio::Model& eulerModel = oracle_->eulerModel();
  const Eigen::Index nj = oracle_->jointDim();
  const vector_t lower = eulerModel.lowerPositionLimit.tail(nj);
  const vector_t upper = eulerModel.upperPositionLimit.tail(nj);
  vector_t state(oracle_->stateDim());
  Eigen::Index row = 0;
  for (const EulerStateSegment& segment : eulerLayout_.segments) {
    const Eigen::Index dim = static_cast<Eigen::Index>(segment.dim);
    if (segment.kind == EulerStateSegmentKind::kEulerZyxAngles) {
      state.segment<3>(row) << uniform(-M_PI, M_PI, generator),
          uniform(-bounds.maxAbsPitch, bounds.maxAbsPitch, generator),
          uniform(-bounds.maxAbsRoll, bounds.maxAbsRoll, generator);
    } else if (segment.kind == EulerStateSegmentKind::kEulerZyxRates) {
      for (Eigen::Index i = 0; i < 3; ++i)
        state(row + i) =
            uniform(-bounds.maxAbsEulerRate, bounds.maxAbsEulerRate, generator);
    } else if (segment.name == "base_position") {
      state.segment<3>(row) << uniform(/*low=*/-1.0, /*high=*/1.0, generator),
          uniform(/*low=*/-1.0, /*high=*/1.0, generator),
          uniform(/*low=*/0.5, /*high=*/1.1, generator);
    } else if (segment.name == "joint_positions") {
      for (Eigen::Index joint = 0; joint < nj; ++joint) {
        const bool limited = std::isfinite(lower(joint)) &&
                             std::isfinite(upper(joint)) &&
                             upper(joint) > lower(joint);
        const scalar_t middle =
            limited ? 0.5 * (lower(joint) + upper(joint)) : 0.0;
        const scalar_t halfWidth =
            limited ? std::min(bounds.maxAbsJointOffset,
                               0.5 * (upper(joint) - lower(joint)))
                    : bounds.maxAbsJointOffset;
        state(row + joint) = middle + uniform(-halfWidth, halfWidth, generator);
      }
    } else {
      scalar_t halfWidth = bounds.maxAbsLinearVelocity;
      if (segment.name == "normalized_angular_momentum")
        halfWidth = bounds.maxAbsNormalizedAngularMomentum;
      if (segment.name == "joint_velocities")
        halfWidth = bounds.maxAbsJointVelocity;
      for (Eigen::Index i = 0; i < dim; ++i)
        state(row + i) = uniform(-halfWidth, halfWidth, generator);
    }
    row += dim;
  }
  return state;
}

vector_t MpcEquivalenceFixture::randomInput(scalar_t maxAbsJointEntry,
                                            std::mt19937& generator) const {
  const scalar_t halfWeight = 0.5 * oracle_->totalMass() * 9.81;
  const Eigen::Index nj = oracle_->jointDim();
  vector_t input(oracle_->inputDim());
  Eigen::Index row = 0;
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    if (oracleOptions_.contactInput == OracleContactInput::kWorldWrench) {
      input.segment<6>(row)
          << uniform(/*low=*/-0.25, /*high=*/0.25, generator) * halfWeight,
          uniform(/*low=*/-0.25, /*high=*/0.25, generator) * halfWeight,
          uniform(/*low=*/0.5, /*high=*/1.5, generator) * halfWeight,
          uniform(/*low=*/-20.0, /*high=*/20.0, generator),
          uniform(/*low=*/-20.0, /*high=*/20.0, generator),
          uniform(/*low=*/-20.0, /*high=*/20.0, generator);
      row += 6;
      continue;
    }
    const matrix_t& basis = oracleOptions_.localBasisMatrices[contact];
    // The scaling of every generator that carries half the weight on a level
    // foot: row 2 of a local wrench is its normal force.
    const scalar_t nominal = halfWeight / basis.row(2).sum();
    for (Eigen::Index i = 0; i < basis.cols(); ++i)
      input(row + i) = uniform(/*low=*/0.0, /*high=*/2.0, generator) * nominal;
    row += basis.cols();
  }
  for (Eigen::Index joint = 0; joint < nj; ++joint)
    input(row + joint) =
        uniform(-maxAbsJointEntry, maxAbsJointEntry, generator);
  return input;
}

const vector_t& MpcEquivalenceFixture::initialState() const {
  return centroidal_ != nullptr ? centroidal_->getInitialState()
                                : wholeBody_->getInitialState();
}

vector_t MpcEquivalenceFixture::weightCompensatingInput(
    const vector_t& state, const contact_flag_t& contactFlags) const {
  return humanoid::weightCompensatingInput(pinocchioInterface(), contactFlags,
                                           model(), state);
}

EquivalenceReference MpcEquivalenceFixture::walkingReference() const {
  std::vector<scalar_t> eventTimes;
  std::vector<size_t> modes = {ModeNumber::kStance};
  scalar_t liftOff = kWalkingFirstLiftOff;
  for (size_t step = 0; step < kWalkingSteps; ++step) {
    const scalar_t touchDown = liftOff + kWalkingSingleSupport;
    eventTimes.push_back(liftOff);
    eventTimes.push_back(touchDown);
    modes.push_back(step % 2 == 0 ? ModeNumber::kLf : ModeNumber::kRf);
    modes.push_back(ModeNumber::kStance);
    liftOff = touchDown + kWalkingDoubleSupport;
  }
  const vector_t& start = initialState();
  vector_t ahead = start;
  model().setBasePosition(
      ahead, vector3_t(model().getBasePosition(start) +
                       kWalkingTargetDistance * vector3_t::UnitX()));
  const vector_t zeroInput =
      vector_t::Zero(static_cast<Eigen::Index>(model().getInputDim()));
  return EquivalenceReference{
      .modeSchedule = ModeSchedule(std::move(eventTimes), std::move(modes)),
      .targetTrajectories = TargetTrajectories(
          {0.0, kWalkingTargetTime}, {start, ahead}, {zeroInput, zeroInput}),
      .initTime = 0.0,
      .initState = start,
      .initMode = ModeNumber::kStance};
}

SqpCase MpcEquivalenceFixture::restingWarmStart() const {
  const vector_t& state = initialState();
  return SqpCase{.initState = state,
                 .warmStartState = state,
                 .warmStartInput = weightCompensatingInput(
                     state, modeNumber2StanceLeg(ModeNumber::kStance))};
}

SwitchedModelReferenceManager& MpcEquivalenceFixture::referenceManager() const {
  return centroidal_ != nullptr
             ? *centroidal_->getSwitchedModelReferenceManagerPtr()
             : *wholeBody_->getSwitchedModelReferenceManagerPtr();
}

void MpcEquivalenceFixture::installReference(
    const EquivalenceReference& reference) const {
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager =
      centroidal_ != nullptr
          ? centroidal_->getSwitchedModelReferenceManagerPtr()
          : wholeBody_->getSwitchedModelReferenceManagerPtr();
  referenceManager->reset();
  referenceManager->getGaitSchedule()->updateModeSchedule(
      reference.modeSchedule);
  referenceManager->setTargetTrajectories(reference.targetTrajectories);
  referenceManager->preSolverRun(reference.initTime,
                                 reference.initTime + timeHorizon(),
                                 reference.initState, reference.initMode);
}

std::vector<AnnotatedTime> MpcEquivalenceFixture::sqpNodeGrid(
    const EquivalenceReference& reference) const {
  installReference(reference);
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager =
      centroidal_ != nullptr
          ? centroidal_->getSwitchedModelReferenceManagerPtr()
          : wholeBody_->getSwitchedModelReferenceManagerPtr();
  // The solver's run calls preSolverRun() once more (SolverBase::preRun) before
  // it discretizes.
  const scalar_t finalTime = reference.initTime + timeHorizon();
  referenceManager->preSolverRun(reference.initTime, finalTime,
                                 reference.initState, reference.initMode);
  // SqpSolver::runImpl's discretization: the node spacing, cut at the events of
  // the reference manager's schedule.
  return timeDiscretizationWithEvents(
      reference.initTime, finalTime, sqpSettings().dt,
      referenceManager->getModeSchedule().eventTimes);
}

sqp::Settings MpcEquivalenceFixture::solveSettings(
    const SqpRunOptions& options) const {
  sqp::Settings settings = sqpSettings();
  settings.sqpIteration = options.iterations;
  if (options.convergenceTolerance.has_value()) {
    settings.deltaTol = *options.convergenceTolerance;
    settings.costTol = *options.convergenceTolerance;
  }
  // One thread: the performance index is summed per worker, so its bits would
  // depend on how the nodes were shared out.
  settings.nThreads = 1;
  settings.useFeedbackPolicy = true;
  settings.printSolverStatus = false;
  settings.printSolverStatistics = false;
  settings.printLinesearch = false;
  // What the tests read of an iteration the solver records
  // (SqpSolver::getIterationRecords()); it writes no log.
  settings.enableLogging = false;
  return settings;
}

SqpRun MpcEquivalenceFixture::sqpRun(
    std::shared_ptr<const StateManifold> manifold,
    const EquivalenceReference& reference,
    const vector_t& initState,
    PrimalSolution warmStart,
    const SqpRunOptions& options) const {
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager =
      centroidal_ != nullptr
          ? centroidal_->getSwitchedModelReferenceManagerPtr()
          : wholeBody_->getSwitchedModelReferenceManagerPtr();
  const Initializer& initializer =
      centroidal_ != nullptr
          ? static_cast<const Initializer&>(centroidal_->getInitializer())
          : wholeBody_->getInitializer();
  const scalar_t finalTime = reference.initTime + timeHorizon();
  installReference(reference);

  OptimalControlProblem problem = this->problem();
  problem.stateManifoldPtr = std::move(manifold);
  warmStart.modeSchedule_ = referenceManager->getModeSchedule();

  SqpSolver solver(solveSettings(options), problem, initializer);
  solver.setReferenceManager(referenceManager);
  if (centroidal_ != nullptr &&
      centroidal_->getContactPlannerModulePtr() != nullptr) {
    solver.addSynchronizedModule(centroidal_->getContactPlannerModulePtr());
  }
  solver.run(reference.initTime, initState, reference.initMode, finalTime,
             warmStart);
  return SqpRun{.solution = solver.primalSolution(finalTime),
                .iterations = solver.getIterationRecords(),
                .metrics = solver.getSolutionMetrics()};
}

PrimalSolution MpcEquivalenceFixture::sqpSolution(
    std::shared_ptr<const StateManifold> manifold,
    const EquivalenceReference& reference,
    const vector_t& initState,
    PrimalSolution warmStart,
    const SqpRunOptions& options) const {
  return sqpRun(std::move(manifold), reference, initState, std::move(warmStart),
                options)
      .solution;
}

std::vector<RealTimeSolve> MpcEquivalenceFixture::realTimeSolves(
    const EquivalenceReference& reference,
    PrimalSolution plan,
    size_t solves) const {
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager =
      centroidal_ != nullptr
          ? centroidal_->getSwitchedModelReferenceManagerPtr()
          : wholeBody_->getSwitchedModelReferenceManagerPtr();
  const Initializer& initializer =
      centroidal_ != nullptr
          ? static_cast<const Initializer&>(centroidal_->getInitializer())
          : wholeBody_->getInitializer();
  installReference(reference);
  SqpSolver solver(solveSettings(SqpRunOptions{
                       .iterations = 1, .convergenceTolerance = std::nullopt}),
                   problem(), initializer);
  solver.setReferenceManager(referenceManager);
  if (centroidal_ != nullptr &&
      centroidal_->getContactPlannerModulePtr() != nullptr) {
    solver.addSynchronizedModule(centroidal_->getContactPlannerModulePtr());
  }

  std::vector<RealTimeSolve> sequence;
  for (size_t solve = 0; solve < solves; ++solve) {
    // One node after the last plan, from its state there: node 1, or the node
    // after it where node 1 is an event's (the post-event node, at the same
    // time), in the mode that follows.
    ABSL_CHECK_GE(plan.timeTrajectory_.size(), size_t{3})
        << "a plan of fewer than three nodes";
    const size_t next =
        plan.timeTrajectory_[2] == plan.timeTrajectory_[1] ? 2 : 1;
    const scalar_t initTime = plan.timeTrajectory_[next];
    const vector_t initState = plan.stateTrajectory_[next];
    const size_t initMode = referenceManager->getModeSchedule().modeAtTime(
        initTime + numeric_traits::weakEpsilon<scalar_t>());
    const scalar_t finalTime = initTime + timeHorizon();
    if (solve == 0) {
      solver.run(initTime, initState, initMode, finalTime, plan);
    } else {
      solver.run(initTime, initState, initMode,
                 finalTime);  // warm-started from its own last solution
    }
    ABSL_CHECK_EQ(solver.getIterationRecords().size(), size_t{1})
        << "a real-time solve runs one iteration";
    sequence.push_back(
        RealTimeSolve{.initTime = initTime,
                      .iteration = solver.getIterationRecords().front(),
                      .metrics = solver.getSolutionMetrics()});
    plan = solver.primalSolution(finalTime);
  }
  return sequence;
}

RealTimeIteration MpcEquivalenceFixture::realTimeIteration(
    std::shared_ptr<const StateManifold> manifold,
    const EquivalenceReference& reference,
    const SqpCase& sqpCase) const {
  const scalar_t finalTime = reference.initTime + timeHorizon();
  PrimalSolution warmStart;
  for (scalar_t time = reference.initTime;
       time <= finalTime + 2.0 * sqpSettings().dt; time += sqpSettings().dt) {
    warmStart.timeTrajectory_.push_back(time);
    warmStart.stateTrajectory_.push_back(sqpCase.warmStartState);
    warmStart.inputTrajectory_.push_back(sqpCase.warmStartInput);
  }
  // Every node of a held warm start is its one state (the solution has one node
  // per node of the grid).
  const vector_array_t warmStartNodes(sqpNodeGrid(reference).size(),
                                      sqpCase.warmStartState);
  return realTimeIteration(std::move(manifold), reference, sqpCase.initState,
                           warmStart, warmStartNodes, sqpCase.warmStartInput);
}

RealTimeIteration MpcEquivalenceFixture::realTimeIteration(
    std::shared_ptr<const StateManifold> manifold,
    const EquivalenceReference& reference,
    const vector_t& initState,
    const PrimalSolution& warmStart,
    const vector_array_t& warmStartNodes,
    const vector_t& warmStartInput) const {
  const std::shared_ptr<const StateManifold> iterationManifold = manifold;
  const SqpRun run = sqpRun(
      std::move(manifold), reference, initState, warmStart,
      SqpRunOptions{.iterations = 1, .convergenceTolerance = std::nullopt});
  const PrimalSolution& solution = run.solution;
  ABSL_CHECK_EQ(run.iterations.size(), size_t{1})
      << "a real-time iteration is one iteration";

  const Eigen::Index nodes =
      static_cast<Eigen::Index>(solution.timeTrajectory_.size());
  ABSL_CHECK_GE(warmStartNodes.size(), solution.timeTrajectory_.size())
      << "fewer warm-start nodes than the solver's";
  RealTimeIteration iteration;
  iteration.stateStep.resize(
      nodes, static_cast<Eigen::Index>(iterationManifold->getTangentDim()));
  iteration.inputStep.resize(nodes, warmStartInput.size());
  for (Eigen::Index k = 0; k < nodes; ++k) {
    const size_t node = static_cast<size_t>(k);
    iteration.stateStep.row(k) =
        iterationManifold
            ->difference(warmStartNodes[node], solution.stateTrajectory_[node])
            .transpose();
    const vector_t& input = solution.inputTrajectory_[node];
    if (input.size() == warmStartInput.size()) {
      iteration.inputStep.row(k) = (input - warmStartInput).transpose();
    } else {
      iteration.inputStep.row(k).setConstant(
          std::numeric_limits<scalar_t>::quiet_NaN());
    }
  }
  const ControllerBase* absl_nullable controller =
      solution.controllerPtr_.get();
  ABSL_CHECK(controller != nullptr &&
             controller->getType() == ControllerType::MANIFOLD_LINEAR)
      << "an iteration on a state manifold has a manifold linear policy";
  iteration.firstGain = static_cast<const ManifoldLinearController*>(controller)
                            ->getFeedbackGains()
                            .front();
  iteration.stepSize = run.iterations.front().stepInfo.stepSize;
  return iteration;
}

vector_t MpcEquivalenceFixture::torqueFreeInput(const vector_t& eulerState,
                                                const vector_t& input) const {
  // The rows of the flow that the base's moment drives: the normalized angular
  // momentum's rate (centroidal) or the angular base acceleration (whole body).
  // Both are affine in the contact entries, so unit differences give their map
  // exactly. f_e = [hbar_dot (6), ...] (centroidal) or [v_e (6 + n_j), a_b,e
  // (6), qdd_j] (whole body), the rotation's rows last of each block of three.
  const Eigen::Index rotationRows =
      isWholeBody() ? 6 + oracle_->jointDim() + 3 : 3;
  const Eigen::Index contactEntries = oracle_->inputDim() - oracle_->jointDim();
  const vector3_t moment =
      oracle_->flowMap(eulerState, input).segment<3>(rotationRows);
  matrix_t map(3, contactEntries);
  for (Eigen::Index column = 0; column < contactEntries; ++column) {
    const vector_t unit = vector_t::Unit(oracle_->inputDim(), column);
    map.col(column) = 0.5 * (oracle_->flowMap(eulerState, input + unit) -
                             oracle_->flowMap(eulerState, input - unit))
                                .segment<3>(rotationRows);
  }
  vector_t corrected = input;
  corrected.head(contactEntries) -=
      map.completeOrthogonalDecomposition().solve(moment);
  return corrected;
}

void MpcEquivalenceTest::SetUp() {
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kWarning);
  absl::StatusOr<std::unique_ptr<MpcEquivalenceFixture>> fixture =
      MpcEquivalenceFixture::Create(GetParam());
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  mpc_ = *std::move(fixture);
}

::testing::AssertionResult isWithinEntrywiseBound(const matrix_t& actual,
                                                  const matrix_t& expected,
                                                  const matrix_t& bound) {
  if (actual.rows() != expected.rows() || actual.cols() != expected.cols())
    return shapesDiffer(actual, expected);
  if (bound.rows() != expected.rows() || bound.cols() != expected.cols())
    return shapesDiffer(bound, expected);
  Eigen::Index worstRow = 0;
  Eigen::Index worstColumn = 0;
  scalar_t worstRatio = 0.0;
  for (Eigen::Index row = 0; row < actual.rows(); ++row) {
    for (Eigen::Index column = 0; column < actual.cols(); ++column) {
      const scalar_t difference =
          std::abs(actual(row, column) - expected(row, column));
      if (!(difference <= bound(row, column))) {  // A NaN fails.
        return ::testing::AssertionFailure() << absl::StrFormat(
                   "entry (%d, %d): %.17g vs %.17g, |difference| %.3e above "
                   "its bound %.3e",
                   row, column, actual(row, column), expected(row, column),
                   difference, bound(row, column));
      }
      const scalar_t ratio =
          bound(row, column) > 0.0 ? difference / bound(row, column) : 0.0;
      if (ratio > worstRatio) {
        worstRatio = ratio;
        worstRow = row;
        worstColumn = column;
      }
    }
  }
  return ::testing::AssertionSuccess()
         << absl::StrFormat("worst entry (%d, %d) at %.3f of its bound",
                            worstRow, worstColumn, worstRatio);
}

::testing::AssertionResult isCloseToScale(const matrix_t& actual,
                                          const matrix_t& expected,
                                          scalar_t relativeTolerance,
                                          scalar_t floor) {
  if (actual.rows() != expected.rows() || actual.cols() != expected.cols())
    return shapesDiffer(actual, expected);
  if (actual.size() == 0) return ::testing::AssertionSuccess();
  const scalar_t bound =
      relativeTolerance * std::max(floor, expected.cwiseAbs().maxCoeff());
  return isWithinEntrywiseBound(
      actual, expected,
      matrix_t::Constant(actual.rows(), actual.cols(), bound));
}

}  // namespace ocs2::humanoid
