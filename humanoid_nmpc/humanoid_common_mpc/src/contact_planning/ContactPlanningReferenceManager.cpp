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

#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/ContactArray.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningTermFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/TargetContactPose.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionContext.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionRule.h"
#include "humanoid_common_mpc/contact_planning/execution/PlanCoverage.h"
#include "humanoid_common_mpc/contact_planning/execution/PlannedComOverride.h"
#include "humanoid_common_mpc/contact_planning/execution/PlannedHeadingOverride.h"
#include "humanoid_common_mpc/contact_planning/execution/ScheduleAdaptationPipeline.h"
#include "humanoid_common_mpc/contact_planning/problem/TermCollection.h"
#include "humanoid_common_mpc/gait/BipedModes.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

namespace ocs2::humanoid {

namespace {
constexpr scalar_t kShiftLogAge =
    2.0;  // [s] schedule shifts older than this cannot concern a pending plan
          // any more
constexpr scalar_t kPredictionTimeTolerance =
    1.0e-6;  // [s] slack when checking that a prediction covers the solver time
// [s] a plan whose commit boundary is no more than this behind the solver time
// has not passed it.
constexpr scalar_t kStalePlanTolerance = 1.0e-6;
// One in this many dropped plans is logged, per reason.
constexpr int kDroppedPlanLogInterval = 50;
// The fewest knots of a prediction that can be interpolated.
constexpr int kMinPredictionKnots = 2;
// The fewest entries of a target state that holds a velocity command, read
// through getProxyLinvelComRtWorldEwrtWorld() (linear) and getRefPlanarMotion()
// (yaw rate).
constexpr int kMinCommandTargetStateDim = kTwistDim;
// [rad] The largest rotation of the base between consecutive knots of the dense
// resample of the target, the spacing of the target calculators' knots
// (TargetTrajectoriesCalculatorBase::kKnotMaxErrYawBaseRtWorld): the
// shortest-path slerp between two knots then follows the target the way it
// turns (design README section 2.9).
constexpr scalar_t kMaxResampleAngleSpacing = 0.5 * std::numbers::pi;

/** [rad] The angle |Log(xi_a^-1 xi_b)| in [0, pi] between the base orientations
 * of the states `a` and `b` of `model`. */
scalar_t baseRotationBetween(const MpcRobotModelBase<scalar_t>& model,
                             const vector_t& a,
                             const vector_t& b) {
  const vector4_t relative = quaternionProduct<scalar_t>(
      quaternionConjugate<scalar_t>(model.getQuatBaseRtWorldXyzw(a)),
      model.getQuatBaseRtWorldXyzw(b));
  return quaternionLog<scalar_t>(relative).norm();
}

/**
 * The times strictly between `startTime` and `endTime` at which the dense
 * resample of `source` must add knots so that consecutive knots are at most
 * kMaxResampleAngleSpacing apart, in increasing order; `startState` and
 * `endState` are source's desired states at the two times. None while the base
 * turns by no more along `source` (the sum of the angles between its knots
 * in the interval), the case of every command up to 63 rad/s. Otherwise every
 * knot of `source` in the interval, and each piece of `source` between them cut
 * into the fewest equal parts in time for the spacing: within a piece the slerp
 * turns at a constant rate, so equal parts in time are equal parts of its
 * rotation.
 */
scalar_array_t boundedAngleResampleTimes(
    const MpcRobotModelBase<scalar_t>& model,
    const TargetTrajectories& source,
    scalar_t startTime,
    const vector_t& startState,
    scalar_t endTime,
    const vector_t& endState) {
  scalar_array_t times;
  // The pieces of `source` between its knots in the interval, by their ends;
  // the states are not copied.
  scalar_array_t breakpoints{startTime};
  std::vector<const vector_t* absl_nonnull> states{&startState};
  for (size_t i = 0; i < source.timeTrajectory.size(); ++i) {
    const scalar_t time = source.timeTrajectory[i];
    if (time > startTime && time < endTime) {
      breakpoints.push_back(time);
      states.push_back(&source.stateTrajectory[i]);
    }
  }
  breakpoints.push_back(endTime);
  states.push_back(&endState);
  std::vector<scalar_t> pieceAngles(breakpoints.size() - 1);
  scalar_t totalAngle = 0.0;
  for (size_t i = 0; i + 1 < breakpoints.size(); ++i) {
    pieceAngles[i] = baseRotationBetween(model, *states[i], *states[i + 1]);
    totalAngle += pieceAngles[i];
  }
  if (totalAngle <= kMaxResampleAngleSpacing) return times;
  for (size_t i = 0; i + 1 < breakpoints.size(); ++i) {
    if (i > 0) times.push_back(breakpoints[i]);
    const scalar_t parts = std::ceil(pieceAngles[i] / kMaxResampleAngleSpacing);
    for (scalar_t part = 1.0; part < parts; part += 1.0) {
      times.push_back(breakpoints[i] +
                      (part / parts) * (breakpoints[i + 1] - breakpoints[i]));
    }
  }
  return times;
}

/**
 * Returns the input of a tail knot at `time` (targetOnPlanGrid()): the
 * operator's when the published target carries one, the live target's
 * otherwise.
 */
vector_t tailInput(const TargetTrajectories& operatorTarget,
                   const TargetTrajectories& targetTrajectories,
                   scalar_t time) {
  return operatorTarget.inputTrajectory.empty()
             ? targetTrajectories.getDesiredInput(time)
             : operatorTarget.getDesiredInput(time);
}

/**
 * Appends a sample of the live target `targetTrajectories` at `time` with
 * `state` to `denseTarget`, with the live target's input when it has inputs.
 */
void appendLiveSample(const TargetTrajectories& targetTrajectories,
                      scalar_t time,
                      vector_t state,
                      TargetTrajectories& denseTarget) {
  denseTarget.timeTrajectory.push_back(time);
  denseTarget.stateTrajectory.push_back(std::move(state));
  if (!targetTrajectories.inputTrajectory.empty()) {
    denseTarget.inputTrajectory.push_back(
        targetTrajectories.getDesiredInput(time));
  }
}

}  // namespace

absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>>
ContactPlanningReferenceManager::Create(
    std::shared_ptr<GaitSchedule> gaitSchedulePtr,
    std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const ContactPlanningConfig& config) {
  // absl::WrapUnique: the constructor is private.
  std::shared_ptr<ContactPlanningReferenceManager> manager =
      absl::WrapUnique(new ContactPlanningReferenceManager(
          std::move(gaitSchedulePtr), std::move(swingTrajectoryPtr),
          pinocchioInterface, mpcRobotModel, config));
  // The execution rules hold pointers into this object (the robot model, the
  // ACoM slot), so they are built once it exists, by the same call that builds
  // them on every reload.
  RETURN_IF_ERROR(manager->setConfigStatus(config));
  return manager;
}

ContactPlanningReferenceManager::ContactPlanningReferenceManager(
    std::shared_ptr<GaitSchedule> gaitSchedulePtr,
    std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    ContactPlanningConfig config)
    : SwitchedModelReferenceManager(std::move(gaitSchedulePtr),
                                    std::move(swingTrajectoryPtr),
                                    pinocchioInterface,
                                    mpcRobotModel),
      config_(std::move(config)) {
  totalMass_ = pinocchio::computeTotalMass(pinocchioInterface_.getModel());
}

absl::StatusOr<TermCollection<ExecutionRule>>
ContactPlanningReferenceManager::buildExecutionRules(
    const ContactPlanningConfig& config) const {
  // The two rules that need this manager's robot model (and, for the heading,
  // its ACoM evaluator slot) are made here; every other rule comes from the
  // factory.
  const ContactPlanningTermFactory::ExtraRuleMaker makeModelRule =
      [this](absl::string_view name) -> std::unique_ptr<ExecutionRule> {
    if (name == term::kPlannedHeadingOverride) {
      return std::make_unique<PlannedHeadingOverride>(*mpcRobotModelPtr_,
                                                      &acom_);
    }
    if (name == term::kPlannedComOverride) {
      return std::make_unique<PlannedComOverride>(*mpcRobotModelPtr_);
    }
    return nullptr;
  };
  return ContactPlanningTermFactory::buildExecutionRulesStatus(config,
                                                               makeModelRule);
}

void ContactPlanningReferenceManager::logSwingTimeScaleWarning(
    const ContactPlanningConfig& config) {
  const scalar_t swingTimeScale =
      swingTrajectoryPtr_->getConfig().swingTimeScale;
  checkedSwingTimeScale_ = swingTimeScale;
  const std::optional<std::string> warning =
      config.swingTimeScaleWarning(swingTimeScale);
  if (warning.has_value()) {
    LOG(WARNING) << "[ContactPlanningReferenceManager] " << *warning;
  }
}

bool ContactPlanningReferenceManager::rulesNeedPredictedTrajectory() const {
  for (const std::unique_ptr<ExecutionRule>& rule : executionRules_) {
    if (rule->needsPredictedTrajectory()) return true;
  }
  return false;
}

bool ContactPlanningReferenceManager::rulesNeedComState() const {
  for (const std::unique_ptr<ExecutionRule>& rule : executionRules_) {
    if (rule->needsComState()) return true;
  }
  return false;
}

bool ContactPlanningReferenceManager::rulesRewriteTarget() const {
  for (const std::unique_ptr<ExecutionRule>& rule : executionRules_) {
    if (rule->rewritesTarget()) return true;
  }
  return false;
}

namespace {

/**
 * The ACoM evaluator a configuration's heading model needs, if it needs one
 * that is not installed yet: nullptr when there is nothing to install (heading
 * model off, evaluator present, or no network registered for the robot, which
 * is warned about and leaves the base yaw as the heading).
 */
absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> headingModelEvaluatorFor(
    const ContactPlanningConfig& config,
    const AngularCenterOfMass* absl_nullable installed,
    const ModelSettings& modelSettings) {
  if (!config.usesHeadingModel() || installed != nullptr) {
    return std::unique_ptr<AngularCenterOfMass>();
  }
  absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
      AngularCenterOfMass::Create(modelSettings.robotName,
                                  modelSettings.mpcModelJointNames);
  if (acom.ok()) {
    LOG(INFO) << "[ContactPlanningReferenceManager] contact planner heading "
                 "model: angular center of mass of '"
              << modelSettings.robotName << "'.";
    return *std::move(acom);
  }
  if (acom.status().code() == absl::StatusCode::kNotFound) {
    LOG(WARNING)
        << "[ContactPlanningReferenceManager] contact planner heading model: "
        << acom.status().message() << " The base yaw is the heading.";
    return std::unique_ptr<AngularCenterOfMass>();
  }
  return acom.status();
}

}  // namespace

absl::Status ContactPlanningReferenceManager::loadHeadingModelEvaluator() {
  loadsHeadingModelEvaluator_ = true;
  absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
      headingModelEvaluatorFor(getConfig(), acom_.get(),
                               mpcRobotModelPtr_->modelSettings);
  if (!acom.ok()) return acom.status();
  if (*acom != nullptr) acom_ = *std::move(acom);
  return absl::OkStatus();
}

std::optional<scalar_t> ContactPlanningReferenceManager::computeYawAcomRtWorld(
    const vector_t& state) const {
  if (!acom_) return std::nullopt;
  // The twist heading psi_h(xi_a) of the ACoM attitude (design section 2.9),
  // in (-pi, pi].
  return yawOf(acom_->computeQuatAcomRtWorldXyzw(
      mpcRobotModelPtr_->getQuatBaseRtWorldXyzw(state),
      mpcRobotModelPtr_->getJointAngles(state)));
}

BipedFeetArray<scalar_t>
ContactPlanningReferenceManager::readEulerzyxFeetRtWorldYaw() const {
  BipedFeetArray<scalar_t> eulerzyxFeetRtWorldYaw = makeBipedFeetArray(0.0);
  const PinocchioInterface::Data& data = pinocchioInterface_.getData();
  for (size_t i = 0; i < static_cast<size_t>(kBipedFootCount); ++i) {
    const matrix3_t rotmatFootRtWorld =
        data.oMf[getContactFrameIndex(pinocchioInterface_, *mpcRobotModelPtr_,
                                      i)]
            .rotation();
    eulerzyxFeetRtWorldYaw[i] =
        std::atan2(rotmatFootRtWorld(1, 0), rotmatFootRtWorld(0, 0));
  }
  return eulerzyxFeetRtWorldYaw;
}

matrix3_t ContactPlanningReferenceManager::computeInertiaComEwrtWorld(
    const vector_t& state) {
  const PinocchioInterface::Model& model = pinocchioInterface_.getModel();
  PinocchioInterface::Data& data = pinocchioInterface_.getData();
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  pinocchio::ccrba(
      model, data, q,
      vector_t::Zero(
          model.nv));  // the composite inertia about the CoM in the world frame
  return data.Ig.inertia().matrix();
}

scalar_t ContactPlanningReferenceManager::computeInertiaComEwrtWorldZz(
    const vector_t& state) {
  return computeInertiaComEwrtWorld(state)(kZAxisIndex, kZAxisIndex);
}

void ContactPlanningReferenceManager::setTargetTrajectories(
    const TargetTrajectories& targetTrajectories) {
  operatorTarget_.setBuffer(targetTrajectories);
  SwitchedModelReferenceManager::setTargetTrajectories(targetTrajectories);
}

void ContactPlanningReferenceManager::setTargetTrajectories(
    TargetTrajectories&& targetTrajectories) {
  operatorTarget_.setBuffer(targetTrajectories);
  SwitchedModelReferenceManager::setTargetTrajectories(
      std::move(targetTrajectories));
}

void ContactPlanningReferenceManager::captureOperatorCommand(
    scalar_t initTime, const vector_t& initState) {
  // Both commands are read off the momentum channel of the target: the linear
  // part is the commanded CoM velocity, and the angular part carries the yaw
  // rate as the momentum of a rigid turn about the vertical, h_z = I_zz omega /
  // m, with the same locked inertia this manager derives from the model. The
  // target's base yaw is not usable for the latter - it blends the measured yaw
  // rate into its first stretch, and the planned heading override rewrites it.
  //
  // The copy read here is the one the operator published, not the live target:
  // the rules rewrite the live one (the heading override its base yaw, the
  // center-of-mass override the very momentum channel read here) and that
  // rewrite survives into the next solver run, so reading the live target would
  // feed each planner its own previous output.
  cmdRateYawBaseRtWorld_ = 0.0;
  cmdLinvelBaseRtWorldEwrtWorldXy_.setZero();
  cmdLatchTime_ = initTime;
  operatorTarget_.updateFromBuffer();
  const TargetTrajectories& operatorTarget = operatorTarget_.get();
  if (operatorTarget.empty()) return;
  const vector_t desiredState =
      mpcRobotModelPtr_->getDesiredState(operatorTarget, initTime);
  if (desiredState.size() < kMinCommandTargetStateDim) return;
  // The linear part of the same channel, the velocity the target's knot
  // stores (getProxyLinvelComRtWorldEwrtWorld(): the CoM's on a centroidal
  // knot, the base's on a whole-body one), is the operator's commanded base
  // velocity, which the target calculator writes there. It has to be
  // read here, before the rules run, for the same reason the yaw rate
  // does and then some: planned_com_override rewrites exactly this channel with
  // the plan's own CoM velocity, so a planner that read the command back off
  // the target afterwards would be fed its own output. At rest that reads as a
  // zero command however far the operator pushes the stick, the standing blend
  // never crosses its half point, and the robot never starts walking.
  cmdLinvelBaseRtWorldEwrtWorldXy_ =
      getProxyLinvelComRtWorldEwrtWorld(*mpcRobotModelPtr_, desiredState)
          .head<kHorizontalPositionDim>();
  const scalar_t inertiaComEwrtWorldZz =
      computeInertiaComEwrtWorldZz(initState);
  if (inertiaComEwrtWorldZz <= 0.0) return;
  // Through the layout: a centroidal knot answers m h_z / I_zz, a knot that
  // stores the base velocity its world yaw rate.
  cmdRateYawBaseRtWorld_ = mpcRobotModelPtr_->getRefPlanarMotion(
      desiredState, inertiaComEwrtWorldZz)(
      MpcRobotModelBase<scalar_t>::kRefPlanarMotionYawRateIndex);
}

vector2_t ContactPlanningReferenceManager::getCmdLinvelBaseRtWorldEwrtWorldXy(
    scalar_t time) const {
  // The command is fixed in the commanded heading's axes, and that heading
  // turns at the commanded rate from the latch. Answering the latched
  // world-axes vector instead would leave it at the start heading, which over
  // a turning horizon reads as a lateral command of |v| sin(omega (t - t0)).
  const scalar_t cmdErrYawBaseRtWorld =
      cmdRateYawBaseRtWorld_ * std::max(time - cmdLatchTime_, 0.0);
  return Eigen::Rotation2D<scalar_t>(cmdErrYawBaseRtWorld) *
         cmdLinvelBaseRtWorldEwrtWorldXy_;
}

std::optional<SwitchedModelReferenceManager::PlannedDcm>
ContactPlanningReferenceManager::getPlannedDcm(scalar_t time) const {
  // planReferencesUsable(), not hasActivePlan(): an expired plan clamps to its
  // last node rather than reporting that the query ran off its horizon, so it
  // would keep aiming the terminal DCM cost at a stale target forever. And the
  // plan being usable NOW says nothing about `time`: the terminal cost asks at
  // the end of the solver horizon, which lies past the end of any plan older
  // than (plan horizon - MPC horizon), and the lookups below would answer there
  // with the last node's DCM, lagging the robot by (time - endTime) * v.
  // Outside the plan the cost keeps its own support-center reference instead.
  const ContactPlan* absl_nullable plan = usablePlanAt(lastSolveTime_);
  if (plan == nullptr || !planCoversTime(*plan, time)) return std::nullopt;
  // The plan's center of mass is only half of its DCM; the other half is the
  // pendulum it was made on. A plan that does not say which (one built by hand)
  // is taken to be on the running configuration's.
  scalar_t omega = plan->omega;
  if (!(omega > 0.0)) {
    absl::MutexLock lock(configMutex_);
    omega = config_.omega();
  }
  if (!(omega > 0.0)) return std::nullopt;
  const std::optional<vector2_t> posComRtWorldEwrtWorldXy =
      plan->evaluatePosComRtWorldEwrtWorldXy(time);
  const std::optional<vector2_t> linvelComRtWorldEwrtWorldXy =
      plan->evaluateLinvelComRtWorldEwrtWorldXy(time);
  if (!posComRtWorldEwrtWorldXy.has_value() ||
      !linvelComRtWorldEwrtWorldXy.has_value()) {
    return std::nullopt;
  }
  PlannedDcm planned;
  planned.desPosDcmRtWorldEwrtWorldXy =
      *posComRtWorldEwrtWorldXy + *linvelComRtWorldEwrtWorldXy / omega;
  planned.omega = omega;
  return planned;
}

void ContactPlanningReferenceManager::setContactPlan(const ContactPlan& plan) {
  absl::MutexLock lock(planMutex_);
  pendingPlan_ = plan;
}

bool ContactPlanningReferenceManager::setContactPlan(const ContactPlan& plan,
                                                     uint64_t planEpoch) {
  absl::MutexLock lock(planMutex_);
  if (planEpoch != planEpoch_) return false;
  pendingPlan_ = plan;
  return true;
}

uint64_t ContactPlanningReferenceManager::planEpoch() const {
  absl::MutexLock lock(planMutex_);
  return planEpoch_;
}

void ContactPlanningReferenceManager::reset() {
  // The base class's reset() runs resetRuntimeState(), i.e. this class's
  // override, after its own buffers.
  SwitchedModelReferenceManager::reset();
  // The operator's copy of the target is a buffered reference like the base
  // class's: what was published before the reset is dropped, and the target set
  // by the reset itself arrives through setTargetTrajectories() after it.
  operatorTarget_.updateFromBuffer();
  operatorTarget_.get() = TargetTrajectories();
}

void ContactPlanningReferenceManager::resetRuntimeState() {
  SwitchedModelReferenceManager::resetRuntimeState();
  {
    absl::MutexLock lock(planMutex_);
    pendingPlan_.reset();
    ++planEpoch_;
  }
  activePlan_.reset();
  appliedSchedule_ = ModeSchedule();
  hasAppliedSchedule_ = false;
  liftOffHistory_ = LiftOffHistory();
  lastSolveTime_ = std::numeric_limits<scalar_t>::lowest();

  cmdRateYawBaseRtWorld_ = 0.0;
  cmdLinvelBaseRtWorldEwrtWorldXy_.setZero();
  cmdLatchTime_ = 0.0;
  eulerzyxFeetRtWorldYaw_ = makeBipedFeetArray(0.0);
  liftoffEulerzyxFeetRtWorldYaw_ = makeBipedFeetArray(0.0);
  posFeetRtWorldEwrtWorld_ = makeBipedFeetArray(vector3_t(vector3_t::Zero()));
  liftoffPosFeetRtWorldEwrtWorld_ =
      makeBipedFeetArray(vector3_t(vector3_t::Zero()));
  footBookkeepingInitialized_ = false;

  swingLatches_ = makeBipedFeetArray(SwingTimingLatch{});
  lastContactEvents_ = makeBipedFeetArray(ContactEventReport{});
  cadenceTouchDownShift_ = makeBipedFeetArray(0.0);
  errPosFeetRtWorldEwrtWorldXy_ =
      makeBipedFeetArray(vector2_t(vector2_t::Zero()));
  comState_ = HorizontalComState{};
  {
    absl::MutexLock lock(predictionMutex_);
    predictedTimes_.clear();
    predictedStates_.clear();
  }
  hasPredictedComState_ = false;
  predictedComState_ = HorizontalComState{};
  scheduleShiftLog_.clear();
  replanRequested_.store(false);
  {
    absl::MutexLock lock(targetContactMutex_);
    desPoseFeetRtWorld_ = makeBipedFeetArray(TargetContactPose{});
  }
}

bool ContactPlanningReferenceManager::hasPendingPlan() const {
  absl::MutexLock lock(planMutex_);
  return pendingPlan_.has_value();
}

absl::Status ContactPlanningReferenceManager::setConfigStatus(
    const ContactPlanningConfig& config) {
  // Everything is built before anything is replaced, so a rejection leaves the
  // running configuration and rules alone.
  RETURN_IF_ERROR(config.validateStatus());
  ASSIGN_OR_RETURN(TermCollection<ExecutionRule> rules,
                   buildExecutionRules(config));
  // A reload that switches the heading model on needs the evaluator a start
  // with this file would have installed; without it the planner would silently
  // take the base yaw for the heading while its heading rate stays a whole-body
  // quantity. Resolved before anything is replaced, so a network that does not
  // fit refuses the reload as a whole. Before loadHeadingModelEvaluator() has
  // run - while the interface is still being constructed, and Create() and
  // ContactPlannerModule::Create() hand the configuration over through here -
  // the evaluator is left to that call, which the interface makes once from
  // setupOptimalControlProblem(), after the planner module exists.
  if (loadsHeadingModelEvaluator_) {
    ASSIGN_OR_RETURN(
        std::unique_ptr<AngularCenterOfMass> acom,
        headingModelEvaluatorFor(config, acom_.get(),
                                 mpcRobotModelPtr_->modelSettings));
    if (acom != nullptr) acom_ = std::move(acom);
  }
  logSwingTimeScaleWarning(config);
  absl::MutexLock lock(configMutex_);
  config_ = config;
  executionRules_ = std::move(rules);
  return absl::OkStatus();
}

ContactPlanningConfig ContactPlanningReferenceManager::getConfig() const {
  absl::MutexLock lock(configMutex_);
  return config_;
}

BipedFeetArray<vector3_t>
ContactPlanningReferenceManager::computePosFeetRtWorldEwrtWorld(
    const vector_t& state) {
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(q, pinocchioInterface_,
                                                   *mpcRobotModelPtr_);
  BipedFeetArray<vector3_t> posFeetRtWorldEwrtWorld;
  for (size_t i = 0; i < static_cast<size_t>(kBipedFootCount); ++i) {
    posFeetRtWorldEwrtWorld[i] = posContactsRtWorldEwrtWorld[i];
  }
  return posFeetRtWorldEwrtWorld;
}

ContactPlanningReferenceManager::HorizontalComState
ContactPlanningReferenceManager::computeComState(const vector_t& state) {
  const PinocchioInterface::Model& model = pinocchioInterface_.getModel();
  PinocchioInterface::Data& data = pinocchioInterface_.getData();
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  pinocchio::centerOfMass(model, data, q, /*computeSubtreeComs=*/false);
  const vector2_t posComRtWorldEwrtWorldXy =
      data.com[0].head<kHorizontalPositionDim>();
  const vector2_t proxyLinvelComRtWorldEwrtWorldXy =
      getProxyLinvelComRtWorldEwrtWorld(*mpcRobotModelPtr_, state)
          .head<kHorizontalPositionDim>();
  return {.posComRtWorldEwrtWorldXy = posComRtWorldEwrtWorldXy,
          .proxyLinvelComRtWorldEwrtWorldXy = proxyLinvelComRtWorldEwrtWorldXy};
}

void ContactPlanningReferenceManager::updateFootBookkeeping(
    scalar_t initTime, const vector_t& initState) {
  posFeetRtWorldEwrtWorld_ = computePosFeetRtWorldEwrtWorld(initState);
  eulerzyxFeetRtWorldYaw_ = readEulerzyxFeetRtWorldYaw();
  const BipedFeetArray<bool> contacts =
      contactFlagsAtTime(appliedSchedule_, initTime);
  for (size_t i = 0; i < static_cast<size_t>(kBipedFootCount); ++i) {
    if (contacts[i] || !footBookkeepingInitialized_) {
      liftoffPosFeetRtWorldEwrtWorld_[i] = posFeetRtWorldEwrtWorld_[i];
      liftoffEulerzyxFeetRtWorldYaw_[i] = eulerzyxFeetRtWorldYaw_[i];
    }
  }
  footBookkeepingInitialized_ = true;
}

void ContactPlanningReferenceManager::activatePendingPlan(scalar_t initTime) {
  std::optional<ContactPlan> candidate;
  {
    absl::MutexLock lock(planMutex_);
    if (pendingPlan_.has_value()) {
      candidate = std::move(*pendingPlan_);
      pendingPlan_.reset();
    }
  }
  if (candidate.has_value() && candidate->valid) {
    // The planner snapshot is taken after this manager ran in the same cycle,
    // so the plan has seen every shift logged at or before its start time;
    // shifts logged later re-timed the schedule it was built on and are applied
    // to the plan too.
    applyScheduleShiftsToPlan(*candidate, scheduleShiftLog_);
    const ContactPlanningConfig config = getConfig();
    // The plan may only change the schedule after its own commit boundary. A
    // boundary that has already passed means the plan's first decisions lie in
    // the past (the planner latency exceeded commitTime, or a touch-down the
    // boundary was extended to happened while the plan was being computed): the
    // plan is dropped and the executed schedule keeps running until a fresh
    // plan arrives. Shifting such a plan forward onto the current boundary
    // instead delayed it by up to a whole swing and re-lifted the foot that had
    // just landed. Likewise a plan that has a foot down where the executed
    // schedule already has it in flight at the merge point was built before
    // that swing was activated and would land the foot there and lift it again.
    const scalar_t mergeTime = std::max(initTime, candidate->committedUntil);
    if (candidate->committedUntil < initTime - kStalePlanTolerance) {
      if (++stalePlanCount_ % kDroppedPlanLogInterval == 1) {
        LOG(INFO)
            << "[ContactPlanningReferenceManager] the contact plan is stale by "
            << (initTime - candidate->committedUntil) << " s (commitTime "
            << config.planner.commitTime
            << " s does not cover the planner latency); keeping the executed "
               "schedule.";
      }
    } else if (hasAppliedSchedule_ &&
               !planAgreesWithSwingsInFlight(appliedSchedule_, *candidate,
                                             mergeTime)) {
      if (++inconsistentPlanCount_ % kDroppedPlanLogInterval == 1) {
        LOG(INFO)
            << "[ContactPlanningReferenceManager] the contact plan was made "
               "before a swing that is in flight at its merge point ("
            << mergeTime << " s) was committed; keeping the executed schedule.";
      }
    } else {
      activePlan_ = std::move(candidate);
    }
  }
  while (!scheduleShiftLog_.empty() &&
         scheduleShiftLog_.front().time < initTime - kShiftLogAge) {
    scheduleShiftLog_.pop_front();
  }
}

void ContactPlanningReferenceManager::setPredictedTrajectory(
    absl::Span<const scalar_t> times, absl::Span<const vector_t> states) {
  absl::MutexLock lock(predictionMutex_);
  if (times.size() != states.size()) {
    predictedTimes_.clear();
    predictedStates_.clear();
    return;
  }
  predictedTimes_.assign(times.begin(), times.end());
  predictedStates_.assign(states.begin(), states.end());
}

void ContactPlanningReferenceManager::updatePredictedComState(
    scalar_t initTime) {
  hasPredictedComState_ = false;
  vector_t predictedState;
  {
    absl::MutexLock lock(predictionMutex_);
    if (predictedTimes_.size() < kMinPredictionKnots ||
        initTime < predictedTimes_.front() - kPredictionTimeTolerance ||
        initTime > predictedTimes_.back() + kPredictionTimeTolerance) {
      return;
    }
    predictedState = mpcRobotModelPtr_->interpolateStateTrajectory(
        initTime, predictedTimes_, predictedStates_);
  }
  // A diverged solve hands over a non-finite trajectory; measuring against it
  // would poison the foot reference.
  if (predictedState.size() !=
          static_cast<Eigen::Index>(mpcRobotModelPtr_->getStateDim()) ||
      !predictedState.allFinite()) {
    return;
  }
  predictedComState_ = computeComState(predictedState);
  hasPredictedComState_ =
      predictedComState_.posComRtWorldEwrtWorldXy.allFinite() &&
      predictedComState_.proxyLinvelComRtWorldEwrtWorldXy.allFinite();
}

void ContactPlanningReferenceManager::handleContactEvents(
    ExecutionContext& ctx) {
  lastContactEvents_.fill(ContactEventReport{});
  cadenceTouchDownShift_.fill(0.0);
  if (!hasAppliedSchedule_ || !activePlan_.has_value()) {
    return;  // only the schedule of the planning path is adapted
  }

  for (const std::unique_ptr<ExecutionRule>& rule : executionRules_) {
    rule->beginCycle(ctx);
  }
  cadenceTouchDownShift_ = ctx.cadenceTouchDownShift;
  lastContactEvents_ = adaptScheduleWithRules(appliedSchedule_, ctx,
                                              executionRules_, swingLatches_);

  scalar_t totalShift = 0.0;
  bool replan = false;
  for (const ContactEventReport& report : lastContactEvents_) {
    switch (report.type) {
      case ContactEventReport::Type::kEarlyTouchDown:
        replan = true;
        break;
      case ContactEventReport::Type::kLateTouchDown:
        replan = true;
        totalShift += report.timeShift;
        break;
      case ContactEventReport::Type::kCadenceShift:
        totalShift += report.timeShift;
        break;
      case ContactEventReport::Type::kNone:
        break;
    }
  }
  if (std::abs(totalShift) > 0.0) {
    // Later events moved: keep the plan aligned with the executed schedule so
    // that the merge does not cut phases short.
    if (activePlan_->valid) activePlan_->shiftInTime(totalShift);
    scheduleShiftLog_.push_back({.time = ctx.time, .shift = totalShift});
  }
  if (replan) replanRequested_.store(true);
}

void ContactPlanningReferenceManager::modifyReferences(
    scalar_t initTime,
    scalar_t finalTime,
    const vector_t& initState,
    size_t initMode,
    TargetTrajectories& targetTrajectories,
    ModeSchedule& modeSchedule) {
  const scalar_t timeHorizon = finalTime - initTime;
  const scalar_t lowerBoundTime = initTime - timeHorizon;
  const scalar_t upperBoundTime = finalTime + timeHorizon;
  const ContactPlanningConfig config = getConfig();

  // The swing trajectory planner's configuration is replaced by the task file's
  // reload, which does not pass through this manager, so a swingTimeScale moved
  // from the GUI is checked against the planned swings here, once per change.
  if (checkedSwingTimeScale_ !=
      swingTrajectoryPtr_->getConfig().swingTimeScale) {
    logSwingTimeScaleWarning(config);
  }

  activatePendingPlan(initTime);

  ExecutionContext ctx(config);
  fillExecutionContext(initTime, initState, initMode, ctx);

  // Adapt the schedule executed so far to the measured contact state before it
  // is merged with the plan.
  handleContactEvents(ctx);

  const ModeSchedule schedule =
      scheduleToExecute(config, initTime, lowerBoundTime, upperBoundTime);

  captureOperatorCommand(initTime, initState);
  // shapePoseBaseRtWorld() turns a base-pose heuristic's tilt rate into angular
  // momentum with the composite inertia the base class latches in
  // captureMeasuredState(), which this override does not call; left at the
  // zero of resetRuntimeState(), the tilt rate never reached the momentum
  // reference. Only the matrix is latched, in the measured heading's axes, so
  // that it can be turned to each node's heading: its yaw entry, which the
  // base class keeps beside it, would change the command its terms read
  // (targetPlanarCommandAt()).
  if (!heuristicLayerPtr_->basePoseHeuristicsEmpty()) {
    latchInertiaComEwrtHeading(computeInertiaComEwrtWorld(initState),
                               mpcRobotModelPtr_->getYawBaseRtWorld(initState));
  }

  // The operator's target is resampled onto the plan's node grid so that a rule
  // which rewrites the state at a node has a knot there to rewrite. Three
  // things about this used to be wrong.
  //
  // It was gated on `!executionRules_.empty()`, i.e. on ANY rule being listed,
  // while ExecutionRule::overrideTarget is a no-op by default and only the two
  // planned_*_override rules implement it - so a configuration listing only
  // schedule rules (phase_resetting, energy_cadence_modulation,
  // dcm_step_adjustment) paid for the resample, and for the truncation below,
  // in exchange for nothing at all. It now asks whether a listed rule really
  // rewrites.
  //
  // It REPLACED the target with a grid spanning the plan's horizon alone.
  // Nothing intersected that with the solver horizon, and OCS2
  // zero-order-extrapolates past the last knot, so whenever the plan ended
  // before finalTime - which is any time the plan is older than (plan horizon -
  // MPC horizon), the same regime as an expiring plan - the whole reference
  // tail froze at the plan-end sample instead of following the command. The
  // grid is now EXTENDED past the plan's end with the operator's knots, and
  // with a final sample at finalTime when even those fall short, and the rules
  // leave every knot outside the plan alone (planCoversTime).
  //
  // And that extension read the LIVE target, which
  // ReferenceManager::preSolverRun keeps from the previous cycle - the copy the
  // rules had already rewritten. From the second cycle without a newly
  // published target the "operator's" tail was therefore the previous plan's
  // clamped end again. The extension now reads the target as the operator
  // published it (the copy setTargetTrajectories keeps for the command), and
  // falls back to the live one only when nothing was ever published through
  // this manager.
  const ContactPlan* absl_nullable usablePlan = usablePlanAt(initTime);
  if (usablePlan != nullptr && !targetTrajectories.empty() &&
      rulesRewriteTarget()) {
    targetTrajectories =
        targetOnPlanGrid(*usablePlan, targetTrajectories, finalTime);
  }

  for (const std::unique_ptr<ExecutionRule>& rule : executionRules_) {
    rule->overrideTarget(ctx, targetTrajectories);
  }
  const scalar_t prevRefPosSurfaceRtWorldEwrtWorldZ =
      refPosSurfaceRtWorldEwrtWorldZ_;
  const scalar_t posSurfaceRtWorldEwrtWorldZ =
      applyPosSurfaceRtWorldEwrtWorldZ(targetTrajectories, initState, initMode);
  // The operator's copy of the target stands on the same ground as the live one
  // - both are the target as published - and the tail of the densified target
  // above is read from it, so it moves with the ground as well. Left behind, a
  // hot reload of `posSurfaceRtWorldEwrtWorldZ` moved the plan's knots but put
  // every tail knot back on the old ground from the next run on, for as long as
  // the operator published nothing new (for good, under a pose command).
  const scalar_t errPosSurfaceRtWorldEwrtWorldZ =
      posSurfaceRtWorldEwrtWorldZ - prevRefPosSurfaceRtWorldEwrtWorldZ;
  if (errPosSurfaceRtWorldEwrtWorldZ != 0.0) {
    for (vector_t& operatorState : operatorTarget_.get().stateTrajectory) {
      mpcRobotModelPtr_->adaptPosBaseRtWorldEwrtWorldZ(
          operatorState, errPosSurfaceRtWorldEwrtWorldZ);
    }
  }
  updateSwingTrajectories(schedule, ctx, posSurfaceRtWorldEwrtWorldZ);

  modeSchedule = schedule;
  modeSchedule_ = schedule;
  appliedSchedule_ = schedule;
  hasAppliedSchedule_ = true;
  lastSolveTime_ = initTime;
  updateFootBookkeeping(initTime, initState);
  updateDcmStepAdjustment(ctx);
  updateDesPoseFeetRtWorld(initTime, posSurfaceRtWorldEwrtWorldZ);
}

void ContactPlanningReferenceManager::fillExecutionContext(
    scalar_t initTime,
    const vector_t& initState,
    size_t initMode,
    ExecutionContext& ctx) {
  ctx.time = initTime;
  ctx.measuredContact = bipedFootContactsOfMode(initMode);
  ctx.totalMass = totalMass_;
  // An expired plan is withheld from the execution rules for the same reason
  // the schedule merge refuses it: every ContactPlan lookup clamps, so
  // PlannedComOverride and PlannedHeadingOverride would steer the whole-body
  // MPC towards the last node of a horizon that has already passed.
  ctx.activePlan = usablePlanAt(initTime);
  // Only the rules that compare the center of mass with the NMPC's prediction
  // need the kinematics; skip them when none is listed so that the default
  // configuration does exactly the work it did before they existed.
  if (rulesNeedComState()) {
    comState_ = computeComState(initState);
  }
  if (rulesNeedPredictedTrajectory()) {
    updatePredictedComState(initTime);
  } else {
    hasPredictedComState_ = false;
  }
  ctx.hasPredictedComState = hasPredictedComState_;
  ctx.posComRtWorldEwrtWorldXy = comState_.posComRtWorldEwrtWorldXy;
  ctx.proxyLinvelComRtWorldEwrtWorldXy =
      comState_.proxyLinvelComRtWorldEwrtWorldXy;
  ctx.posBaseRtWorldEwrtWorldXy =
      mpcRobotModelPtr_->getPosBaseRtWorldEwrtWorld(initState)
          .head<kHorizontalPositionDim>();
  ctx.yawBaseRtWorld = mpcRobotModelPtr_->getYawBaseRtWorld(initState);
  ctx.predictedPosComRtWorldEwrtWorldXy =
      predictedComState_.posComRtWorldEwrtWorldXy;
  ctx.predictedProxyLinvelComRtWorldEwrtWorldXy =
      predictedComState_.proxyLinvelComRtWorldEwrtWorldXy;
}

ModeSchedule ContactPlanningReferenceManager::scheduleToExecute(
    const ContactPlanningConfig& config,
    scalar_t initTime,
    scalar_t lowerBoundTime,
    scalar_t upperBoundTime) const {
  // The plan honored the applied schedule up to its own commit boundary, and
  // that boundary already covers every swing that was in flight or about to
  // start when the plan was made (commitBoundary extends it to their
  // touch-downs). So the merge happens exactly there. Merging any later, e.g.
  // at the boundary computed now, cut the plan's first lift-off short by the
  // plan's age: the merged swing began at the later merge time but kept the
  // plan's touch-down, and reached the controller shorter than
  // minSwingDuration. Once the active plan's boundary has passed (no fresh plan
  // for longer than commitTime) the applied schedule keeps running;
  // activatePendingPlan() never lets such a plan in.
  scalar_t commitTime = initTime;
  bool planFresh = false;
  const ContactPlan* absl_nullable validPlan =
      activePlan_.has_value() && activePlan_->valid ? &*activePlan_ : nullptr;
  if (validPlan != nullptr) {
    planFresh = validPlan->committedUntil >= initTime - kStalePlanTolerance;
    commitTime = std::max(initTime, validPlan->committedUntil);
  }
  const bool planUsable =
      validPlan != nullptr && planFresh &&
      validPlan->endTime() > commitTime + config.planner.dt &&
      validPlan->startTime <= commitTime;

  ModeSchedule schedule;
  if (planUsable) {
    const ModeSchedule planSchedule = validPlan->toModeSchedule();
    const ModeSchedule& applied =
        hasAppliedSchedule_
            ? appliedSchedule_
            : gaitSchedulePtr_->getModeSchedule(lowerBoundTime, upperBoundTime);
    schedule = mergeModeSchedules(applied, planSchedule, commitTime,
                                  lowerBoundTime, upperBoundTime);
  } else if (hasAppliedSchedule_ && activePlan_.has_value()) {
    // The plan ran out (planner stalled): keep executing the applied schedule,
    // which ends in STANCE.
    schedule = mergeModeSchedules(
        appliedSchedule_,
        ModeSchedule(/*eventTimesInput=*/{}, {kBipedModeStance}),
        upperBoundTime, lowerBoundTime, upperBoundTime);
  } else {
    schedule =
        gaitSchedulePtr_->getModeSchedule(lowerBoundTime, upperBoundTime);
  }
  return schedule;
}

TargetTrajectories ContactPlanningReferenceManager::targetOnPlanGrid(
    const ContactPlan& plan,
    const TargetTrajectories& targetTrajectories,
    scalar_t finalTime) {
  const TargetTrajectories& operatorTarget = operatorTarget_.get().empty()
                                                 ? targetTrajectories
                                                 : operatorTarget_.get();
  TargetTrajectories denseTarget;
  const size_t numNodes = plan.posComRtWorldEwrtWorldXyTraj.size();
  const size_t numOperatorKnots = operatorTarget.timeTrajectory.size();
  denseTarget.timeTrajectory.reserve(numNodes + numOperatorKnots + 1);
  denseTarget.stateTrajectory.reserve(numNodes + numOperatorKnots + 1);
  const bool hasInput = !targetTrajectories.inputTrajectory.empty();
  if (hasInput) {
    denseTarget.inputTrajectory.reserve(numNodes + numOperatorKnots + 1);
  }
  for (size_t i = 0; i < numNodes; ++i) {
    const scalar_t time = plan.startTime + static_cast<scalar_t>(i) * plan.dt;
    vector_t state =
        mpcRobotModelPtr_->getDesiredState(targetTrajectories, time);
    if (i > 0) {
      // The nodes are omega dt apart in yaw, above the bound only for a turn
      // faster than 63 rad/s at dt = 0.025 s; the knots of the bound
      // (boundedAngleResampleTimes()) go before the node.
      for (const scalar_t boundTime : boundedAngleResampleTimes(
               *mpcRobotModelPtr_, targetTrajectories,
               denseTarget.timeTrajectory.back(),
               denseTarget.stateTrajectory.back(), time, state)) {
        appendLiveSample(
            targetTrajectories, boundTime,
            mpcRobotModelPtr_->getDesiredState(targetTrajectories, boundTime),
            denseTarget);
      }
    }
    appendLiveSample(targetTrajectories, time, std::move(state), denseTarget);
  }
  const scalar_t lastDenseTime = denseTarget.timeTrajectory.empty()
                                     ? plan.startTime
                                     : denseTarget.timeTrajectory.back();
  for (size_t i = 0; i < numOperatorKnots; ++i) {
    const scalar_t time = operatorTarget.timeTrajectory[i];
    if (time <= lastDenseTime) continue;
    denseTarget.timeTrajectory.push_back(time);
    denseTarget.stateTrajectory.push_back(operatorTarget.stateTrajectory[i]);
    if (hasInput) {
      denseTarget.inputTrajectory.push_back(
          tailInput(operatorTarget, targetTrajectories, time));
    }
  }
  if (denseTarget.timeTrajectory.empty() ||
      denseTarget.timeTrajectory.back() < finalTime) {
    denseTarget.timeTrajectory.push_back(finalTime);
    denseTarget.stateTrajectory.push_back(
        mpcRobotModelPtr_->getDesiredState(operatorTarget, finalTime));
    if (hasInput) {
      denseTarget.inputTrajectory.push_back(
          tailInput(operatorTarget, targetTrajectories, finalTime));
    }
  }
  return denseTarget;
}

void ContactPlanningReferenceManager::updateDesPoseFeetRtWorld(
    scalar_t initTime, scalar_t posSurfaceRtWorldEwrtWorldZ) {
  BipedFeetArray<TargetContactPose> desPoseFeetRtWorld =
      makeBipedFeetArray(TargetContactPose{});
  const ContactPlan* absl_nullable usablePlan = usablePlanAt(initTime);
  if (usablePlan != nullptr && footBookkeepingInitialized_) {
    TargetContactPoseInputs inputs;
    inputs.time = initTime;
    inputs.posFeetRtWorldEwrtWorld = posFeetRtWorldEwrtWorld_;
    inputs.eulerzyxFeetRtWorldYaw = eulerzyxFeetRtWorldYaw_;
    inputs.errPosFeetRtWorldEwrtWorldXy = errPosFeetRtWorldEwrtWorldXy_;
    inputs.posSurfaceRtWorldEwrtWorldZ = posSurfaceRtWorldEwrtWorldZ;
    desPoseFeetRtWorld =
        computeDesPoseFeetRtWorld(*usablePlan, appliedSchedule_, inputs);
  }
  absl::MutexLock lock(targetContactMutex_);
  desPoseFeetRtWorld_ = desPoseFeetRtWorld;
}

BipedFeetArray<TargetContactPose>
ContactPlanningReferenceManager::getDesPoseFeetRtWorld() const {
  absl::MutexLock lock(targetContactMutex_);
  return desPoseFeetRtWorld_;
}

void ContactPlanningReferenceManager::updateSwingTrajectories(
    const ModeSchedule& schedule,
    const ExecutionContext& ctx,
    scalar_t posSurfaceRtWorldEwrtWorldZ) {
  const SwingTrajectoryPlanner::Config& swingConfig =
      swingTrajectoryPtr_->getConfig();
  const size_t numPhases = schedule.modeSequence.size();
  // The planner's contacts are the biped's two feet: the contact planner is
  // refused for any other set (checkCentroidalContactSet()).
  const size_t numFeet = swingTrajectoryPtr_->numFeet();
  const ContactArray<scalar_array_t> liftoffPosFeetRtWorldEwrtWorldZTraj(
      numFeet, scalar_array_t(numPhases, posSurfaceRtWorldEwrtWorldZ));
  const ContactArray<scalar_array_t> touchdownPosFeetRtWorldEwrtWorldZTraj(
      numFeet,
      scalar_array_t(numPhases,
                     posSurfaceRtWorldEwrtWorldZ +
                         swingConfig.touchdownPosFootRtSurfaceEwrtWorldZ));

  // A rule may have a foot searching for the ground (a late touch-down): its
  // height reference then continues the planned swing as a straight descent
  // instead of hovering.
  ContactArray<std::optional<SwingTrajectoryPlanner::GroundSearch>>
      groundSearches(
          numFeet,
          /*value=*/std::optional<SwingTrajectoryPlanner::GroundSearch>{});
  for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
    for (const std::unique_ptr<ExecutionRule>& rule : executionRules_) {
      const std::optional<GroundSearchRequest> search =
          rule->groundSearch(ctx, foot, schedule, swingLatches_[foot]);
      if (search.has_value()) {
        groundSearches[foot] = SwingTrajectoryPlanner::GroundSearch{
            .liftOffTime = search->liftOffTime,
            .plannedTouchDownTime = search->plannedTouchDownTime,
            .descentSpeedFootRtWorld = search->searchSpeedFootRtWorld};
        break;
      }
    }
  }
  swingTrajectoryPtr_->update(schedule, liftoffPosFeetRtWorldEwrtWorldZTraj,
                              touchdownPosFeetRtWorldEwrtWorldZTraj,
                              groundSearches);
}

void ContactPlanningReferenceManager::updateDcmStepAdjustment(
    const ExecutionContext& ctx) {
  errPosFeetRtWorldEwrtWorldXy_.fill(vector2_t::Zero());
  for (const std::unique_ptr<ExecutionRule>& rule : executionRules_) {
    const BipedFeetArray<vector2_t> errPosFeetRtWorldEwrtWorldXy =
        rule->correctFootholds(ctx, appliedSchedule_);
    for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
      errPosFeetRtWorldEwrtWorldXy_[foot] += errPosFeetRtWorldEwrtWorldXy[foot];
    }
  }
}

scalar_t ContactPlanningReferenceManager::commitBoundary(scalar_t time) const {
  return commitBoundaryForSchedule(appliedSchedule_, time,
                                   getConfig().planner.commitTime,
                                   getConfig().planner.maxCommitExtension);
}

std::optional<SwingWindow> ContactPlanningReferenceManager::swingPhase(
    size_t contactIndex, scalar_t time) const {
  return swingPhaseAtTime(appliedSchedule_, contactIndex, time);
}

std::optional<SwingFootReference>
ContactPlanningReferenceManager::getSwingFootReference(size_t contactIndex,
                                                       scalar_t time) const {
  const ContactPlan* absl_nullable plan = usablePlanAt(lastSolveTime_);
  if (plan == nullptr || !footBookkeepingInitialized_) return std::nullopt;
  const std::optional<SwingWindow> phase = swingPhase(contactIndex, time);
  if (!phase.has_value()) return std::nullopt;
  const scalar_t liftOffTime = phase->liftOff;
  const scalar_t touchDownTime = phase->touchDown;
  const std::optional<vector2_t> touchdownDesPosFootRtWorldEwrtWorldXy =
      plan->getDesPosFootRtWorldEwrtWorldXy(contactIndex, touchDownTime);
  if (!touchdownDesPosFootRtWorldEwrtWorldXy.has_value()) return std::nullopt;

  // The xy motion is timed on the swing without its late touch-down extension:
  // a foot searching for the ground holds its landing target instead of
  // drifting back along the step.
  scalar_t xyTouchDownTime = touchDownTime;
  const SwingTimingLatch& latch = swingLatches_[contactIndex];
  if (latch.active &&
      std::abs(latch.liftOffTime - liftOffTime) <= kSameSwingTolerance &&
      latch.lateExtension > 0.0) {
    xyTouchDownTime = touchDownTime - latch.lateExtension;
  }
  const scalar_t duration = xyTouchDownTime - liftOffTime;
  if (duration <= kMinSwingReferenceDuration) return std::nullopt;

  // The xy interpolation starts where the foot stands at lift-off. For the
  // swing that ends the foot's current contact phase (in flight now, or the
  // next to lift) that is the latched measured position. For a later swing of
  // the same foot inside the horizon the foot first lands somewhere else, so
  // the start is the planned foot position at the node BEFORE that lift-off,
  // i.e. the stance position before the foot moves.
  //
  // getPrevDesPosFootRtWorldEwrtWorldXy(), not
  // getDesPosFootRtWorldEwrtWorldXy() with the argument nudged back half a
  // step: the nudge rounds back up to the lift-off node itself, and under the
  // shipped `hlip` planner that node already carries the swing's LANDING
  // position, so `initDesPosFootRtWorldEwrtWorldXy` became
  // `*touchdownDesPosFootRtWorldEwrtWorldXy`, `desErrPosFootRtWorldEwrtWorldXy`
  // became zero, and this whole reference collapsed to a constant at the
  // landing target with zero commanded velocity for every later swing in the
  // horizon. See ContactPlan.h.
  vector2_t initDesPosFootRtWorldEwrtWorldXy =
      liftoffPosFeetRtWorldEwrtWorld_[contactIndex]
          .head<kHorizontalPositionDim>();
  const std::optional<scalar_t> currentLiftOff =
      currentOrNextLiftOffTime(appliedSchedule_, contactIndex, lastSolveTime_);
  const bool endsCurrentContactPhase =
      currentLiftOff.has_value() &&
      std::abs(*currentLiftOff - liftOffTime) <= kSameSwingTolerance;
  if (!endsCurrentContactPhase) {
    const std::optional<vector2_t> prevDesPosFootRtWorldEwrtWorldXy =
        plan->getPrevDesPosFootRtWorldEwrtWorldXy(contactIndex, liftOffTime);
    if (prevDesPosFootRtWorldEwrtWorldXy.has_value()) {
      initDesPosFootRtWorldEwrtWorldXy = *prevDesPosFootRtWorldEwrtWorldXy;
    }
  }
  const scalar_t tau = std::clamp((time - liftOffTime) / duration, 0.0, 1.0);

  // Use a cubic spline with p'(0) = 1 and p'(1) = 0 to command an initial
  // velocity matching the step velocity. This prevents the swing foot from
  // kicking backward relative to the moving body at lift-off.
  const scalar_t tau2 = tau * tau;
  const scalar_t blend = -tau2 * tau + tau2 + tau;
  const scalar_t blendRate =
      (-kSwingBlendCubicDegree * tau2 + 2.0 * tau + 1.0) / duration;

  // The landing target offset of the foothold rules (zero without them) is
  // blended in with the same profile, so the target moves smoothly. It is
  // computed for the swing in flight at the last solve and belongs to that
  // swing alone: a later swing of the same foot inside the horizon lands on its
  // planned foothold.
  const std::optional<SwingWindow> swingInFlight =
      swingPhase(contactIndex, lastSolveTime_);
  const bool isTheSwingInFlight =
      swingInFlight.has_value() &&
      std::abs(swingInFlight->liftOff - liftOffTime) <= kSameSwingTolerance;
  const vector2_t errPosFootRtWorldEwrtWorldXy =
      isTheSwingInFlight ? errPosFeetRtWorldEwrtWorldXy_[contactIndex]
                         : vector2_t(vector2_t::Zero());
  // The landing (with its offset) minus the start.
  const vector2_t desErrPosFootRtWorldEwrtWorldXy =
      *touchdownDesPosFootRtWorldEwrtWorldXy + errPosFootRtWorldEwrtWorldXy -
      initDesPosFootRtWorldEwrtWorldXy;

  SwingFootReference reference;
  reference.desPosFootRtWorldEwrtWorld.head<kHorizontalPositionDim>() =
      initDesPosFootRtWorldEwrtWorldXy +
      blend * desErrPosFootRtWorldEwrtWorldXy;
  reference.desPosFootRtWorldEwrtWorld(kZAxisIndex) =
      swingTrajectoryPtr_->getDesPosFootRtWorldEwrtWorldZ(contactIndex, time);
  reference.desLinvelFootRtWorldEwrtWorld.head<kHorizontalPositionDim>() =
      blendRate * desErrPosFootRtWorldEwrtWorldXy;
  reference.desLinvelFootRtWorldEwrtWorld(kZAxisIndex) =
      swingTrajectoryPtr_->getDesLinvelFootRtWorldEwrtWorldZ(contactIndex,
                                                             time);
  // Heading model: the foot yaw turns from its lift-off yaw to the planned
  // landing yaw with the same profile.
  if (plan->hasHeading()) {
    const std::optional<scalar_t> touchdownDesEulerzyxFootRtWorldYaw =
        plan->getDesEulerzyxFootRtWorldYaw(contactIndex, touchDownTime);
    if (touchdownDesEulerzyxFootRtWorldYaw.has_value()) {
      scalar_t initDesEulerzyxFootRtWorldYaw =
          liftoffEulerzyxFeetRtWorldYaw_[contactIndex];
      if (!endsCurrentContactPhase) {
        const std::optional<scalar_t> prevDesEulerzyxFootRtWorldYaw =
            plan->getPrevDesEulerzyxFootRtWorldYaw(contactIndex, liftOffTime);
        if (prevDesEulerzyxFootRtWorldYaw.has_value()) {
          initDesEulerzyxFootRtWorldYaw = *prevDesEulerzyxFootRtWorldYaw;
        }
      }
      reference.desYawFootRtWorld =
          initDesEulerzyxFootRtWorldYaw +
          blend * (moduloAngleWithReference(*touchdownDesEulerzyxFootRtWorldYaw,
                                            initDesEulerzyxFootRtWorldYaw) -
                   initDesEulerzyxFootRtWorldYaw);
    }
  }
  return reference;
}

ContactPlannerInput ContactPlanningReferenceManager::makePlannerInput(
    scalar_t initTime,
    const vector_t& initState,
    const vector2_t& cmdLinvelBaseRtWorldEwrtWorldXy) {
  ContactPlannerInput input;
  input.time = initTime;
  input.cmdLinvelBaseRtWorldEwrtWorldXy = cmdLinvelBaseRtWorldEwrtWorldXy;

  const HorizontalComState com = computeComState(initState);
  input.posComRtWorldEwrtWorldXy = com.posComRtWorldEwrtWorldXy;
  // The planner's LIP starts from the CoM velocity, which a whole-body state
  // stands in for with its base velocity (computeComState()).
  input.proxyLinvelComRtWorldEwrtWorldXy = com.proxyLinvelComRtWorldEwrtWorldXy;
  const scalar_t yawBaseRtWorld =
      mpcRobotModelPtr_->getYawBaseRtWorld(initState);
  input.yawHeadingRtWorld = yawBaseRtWorld;

  const BipedFeetArray<vector3_t> posFeetRtWorldEwrtWorld =
      computePosFeetRtWorldEwrtWorld(initState);
  for (size_t i = 0; i < static_cast<size_t>(kBipedFootCount); ++i) {
    input.posFeetRtWorldEwrtWorldXy[i] =
        posFeetRtWorldEwrtWorld[i].head<kHorizontalPositionDim>();
  }

  // The operator's commanded yaw rate is filled whether or not the heading
  // model is on. It is not part of the heading MODEL - it is part of the
  // COMMAND, and the standing/walking blend reads it to decide whether the
  // robot should be stepping at all. Left inside the heading-model branch, a
  // robot without that model contributed nothing from the yaw stick to the
  // blend's activity, so `alpha` never crossed its half point on yaw alone and
  // the robot would not start stepping to turn in place however hard it was
  // asked.
  input.cmdRateYawBaseRtWorld = cmdRateYawBaseRtWorld();

  const ContactPlanningConfig config = getConfig();
  if (config.usesHeadingModel()) {
    // Heading model: the whole-body heading - the ACoM's yaw when this manager
    // has the ACoM evaluator, the base's otherwise - its rate from the angular
    // momentum about the vertical, and the foot yaws unwrapped near the
    // heading. The planning frame is the heading.
    const BipedFeetArray<scalar_t> eulerzyxFeetRtWorldYaw =
        readEulerzyxFeetRtWorldYaw();
    const std::optional<scalar_t> yawAcomRtWorld =
        computeYawAcomRtWorld(initState);
    input.initYawHeadingRtWorld = yawAcomRtWorld.value_or(yawBaseRtWorld);
    input.yawBaseRtWorld =
        moduloAngleWithReference(yawBaseRtWorld, input.initYawHeadingRtWorld);
    input.yawHeadingRtWorld = input.initYawHeadingRtWorld;
    input.inertiaComEwrtWorldZz = computeInertiaComEwrtWorldZz(initState);
    // Through the layout, as the command: m h_z / I_zz on the centroidal state
    // (0 while I_zz is not positive), the base's world yaw rate on a state that
    // stores the base velocity.
    input.initRateYawHeadingRtWorld = mpcRobotModelPtr_->getRefPlanarMotion(
        initState, input.inertiaComEwrtWorldZz)(
        MpcRobotModelBase<scalar_t>::kRefPlanarMotionYawRateIndex);
    for (size_t i = 0; i < static_cast<size_t>(kBipedFootCount); ++i) {
      input.eulerzyxFeetRtWorldYaw[i] = moduloAngleWithReference(
          eulerzyxFeetRtWorldYaw[i], input.initYawHeadingRtWorld);
    }
  }

  // Events at exactly `initTime` count as passed (a touch-down placed at the
  // current time by the phase resetting is a contact for the planner),
  // consistently with every other schedule query of this manager.
  const ModeSchedule& schedule =
      hasAppliedSchedule_ ? appliedSchedule_ : this->getModeSchedule();
  input.committedUntil = hasAppliedSchedule_
                             ? commitBoundary(initTime)
                             : initTime + config.planner.commitTime;
  // At least one node stays free of the committed window. The last swung foot
  // comes from liftOffHistory_, which outlives the schedule's history window.
  fillPlannerInputFromSchedule(schedule, config.planner.dt,
                               std::max(0, config.planner.numNodes - 1),
                               liftOffHistory_, input);
  return input;
}

}  // namespace ocs2::humanoid
