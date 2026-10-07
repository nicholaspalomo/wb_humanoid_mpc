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
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/misc/LinearInterpolation.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningTermFactory.h"
#include "humanoid_common_mpc/contact_planning/execution/PlanCoverage.h"
#include "humanoid_common_mpc/contact_planning/execution/PlannedComOverride.h"
#include "humanoid_common_mpc/contact_planning/execution/PlannedHeadingOverride.h"
#include "humanoid_common_mpc/contact_planning/execution/ScheduleAdaptationPipeline.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

namespace {
constexpr scalar_t kShiftLogAge = 2.0;                 // [s] schedule shifts older than this cannot concern a pending plan any more
constexpr scalar_t kSameSwingTolerance = 1.0e-6;       // [s] lift-off times closer than this identify the same swing
constexpr scalar_t kPredictionTimeTolerance = 1.0e-6;  // [s] slack when checking that a prediction covers the solver time
}  // namespace

absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>> ContactPlanningReferenceManager::Create(
    std::shared_ptr<GaitSchedule> gaitSchedulePtr,
    std::shared_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const ContactPlanningConfig& config) {
  // absl::WrapUnique: the constructor is private.
  std::shared_ptr<ContactPlanningReferenceManager> manager = absl::WrapUnique(new ContactPlanningReferenceManager(
      std::move(gaitSchedulePtr), std::move(swingTrajectoryPtr), pinocchioInterface, mpcRobotModel, config));
  // The execution rules hold pointers into this object (the robot model, the ACoM slot), so they are built once it
  // exists, by the same call that builds them on every reload.
  RETURN_IF_ERROR(manager->setConfigStatus(config));
  return manager;
}

ContactPlanningReferenceManager::ContactPlanningReferenceManager(std::shared_ptr<GaitSchedule> gaitSchedulePtr,
                                                                 std::shared_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
                                                                 const PinocchioInterface& pinocchioInterface,
                                                                 const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                 ContactPlanningConfig config)
    : SwitchedModelReferenceManager(std::move(gaitSchedulePtr), std::move(swingTrajectoryPtr), pinocchioInterface, mpcRobotModel),
      config_(std::move(config)) {
  totalMass_ = pinocchio::computeTotalMass(pinocchioInterface_.getModel());
}

absl::StatusOr<TermCollection<ExecutionRule>> ContactPlanningReferenceManager::buildExecutionRules(
    const ContactPlanningConfig& config) const {
  // The two rules that need this manager's robot model (and, for the heading, its ACoM evaluator slot) are made here;
  // every other rule comes from the factory.
  const ContactPlanningTermFactory::ExtraRuleMaker makeModelRule = [this](const std::string& name) -> std::unique_ptr<ExecutionRule> {
    if (name == term::kPlannedHeadingOverride) return std::make_unique<PlannedHeadingOverride>(*mpcRobotModelPtr_, &acom_);
    if (name == term::kPlannedComOverride) return std::make_unique<PlannedComOverride>(*mpcRobotModelPtr_);
    return nullptr;
  };
  return ContactPlanningTermFactory::buildExecutionRulesStatus(config, makeModelRule);
}

void ContactPlanningReferenceManager::logSwingTimeScaleWarning(const ContactPlanningConfig& config) {
  const scalar_t swingTimeScale = swingTrajectoryPtr_->getConfig().swingTimeScale;
  checkedSwingTimeScale_ = swingTimeScale;
  const std::optional<std::string> warning = config.swingTimeScaleWarning(swingTimeScale);
  if (warning.has_value()) LOG(WARNING) << "[ContactPlanningReferenceManager] " << *warning;
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
 * The ACoM evaluator a configuration's heading model needs, if it needs one that is not installed yet: nullptr when
 * there is nothing to install (heading model off, evaluator present, or no network registered for the robot, which is
 * warned about and leaves the base yaw as the heading).
 */
absl::StatusOr<std::shared_ptr<AngularCenterOfMass>> headingModelEvaluatorFor(const ContactPlanningConfig& config,
                                                                              const AngularCenterOfMass* absl_nullable installed,
                                                                              const ModelSettings& modelSettings) {
  if (!config.usesHeadingModel() || installed != nullptr) return std::shared_ptr<AngularCenterOfMass>();
  absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
      AngularCenterOfMass::Create(modelSettings.robotName, modelSettings.mpcModelJointNames);
  if (acom.ok()) {
    LOG(INFO) << "[ContactPlanningReferenceManager] contact planner heading model: angular center of mass of '" << modelSettings.robotName
              << "'.";
    return std::shared_ptr<AngularCenterOfMass>(*std::move(acom));
  }
  if (acom.status().code() == absl::StatusCode::kNotFound) {
    LOG(WARNING) << "[ContactPlanningReferenceManager] contact planner heading model: " << acom.status().message()
                 << " The base yaw is the heading.";
    return std::shared_ptr<AngularCenterOfMass>();
  }
  return acom.status();
}

}  // namespace

absl::Status ContactPlanningReferenceManager::loadHeadingModelEvaluator() {
  loadsHeadingModelEvaluator_ = true;
  const absl::StatusOr<std::shared_ptr<AngularCenterOfMass>> acom =
      headingModelEvaluatorFor(getConfig(), acom_.get(), mpcRobotModelPtr_->modelSettings);
  if (!acom.ok()) return acom.status();
  if (*acom != nullptr) acom_ = *acom;
  return absl::OkStatus();
}

scalar_t ContactPlanningReferenceManager::computeHeading(const vector_t& state) const {
  if (acom_) {
    return acom_->computeAcomOrientation(mpcRobotModelPtr_->getGeneralizedCoordinates(state))(0);
  }
  return mpcRobotModelPtr_->getBaseOrientationEulerZYX(state)(0);
}

feet_array_t<scalar_t> ContactPlanningReferenceManager::readFootYaws() const {
  feet_array_t<scalar_t> yaws = makeFeetArray(0.0);
  const PinocchioInterface::Data& data = pinocchioInterface_.getData();
  for (size_t i = 0; i < kNumContacts; ++i) {
    const matrix3_t rotation = data.oMf[getContactFrameIndex(pinocchioInterface_, *mpcRobotModelPtr_, i)].rotation();
    yaws[i] = std::atan2(rotation(1, 0), rotation(0, 0));
  }
  return yaws;
}

scalar_t ContactPlanningReferenceManager::computeYawInertia(const vector_t& state) {
  const PinocchioInterface::Model& model = pinocchioInterface_.getModel();
  PinocchioInterface::Data& data = pinocchioInterface_.getData();
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  pinocchio::ccrba(model, data, q, vector_t::Zero(model.nv));  // the composite inertia about the CoM in the world frame
  return data.Ig.inertia().matrix()(2, 2);
}

void ContactPlanningReferenceManager::setTargetTrajectories(const TargetTrajectories& targetTrajectories) {
  operatorTarget_.setBuffer(targetTrajectories);
  SwitchedModelReferenceManager::setTargetTrajectories(targetTrajectories);
}

void ContactPlanningReferenceManager::setTargetTrajectories(TargetTrajectories&& targetTrajectories) {
  operatorTarget_.setBuffer(targetTrajectories);
  SwitchedModelReferenceManager::setTargetTrajectories(std::move(targetTrajectories));
}

void ContactPlanningReferenceManager::captureOperatorCommand(scalar_t initTime, const vector_t& initState) {
  // Both commands are read off the momentum channel of the target: the linear part is the commanded CoM velocity, and
  // the angular part carries the yaw rate as the momentum of a rigid turn about the vertical, h_z = I_zz omega / m,
  // with the same locked inertia this manager derives from the model. The target's base yaw is not usable for the
  // latter - it blends the measured yaw rate into its first stretch, and the planned heading override rewrites it.
  //
  // The copy read here is the one the operator published, not the live target: the rules rewrite the live one (the
  // heading override its base yaw, the center-of-mass override the very momentum channel read here) and that rewrite
  // survives into the next solver run, so reading the live target would feed each planner its own previous output.
  commandedYawRate_ = 0.0;
  commandedVelocity_.setZero();
  operatorTarget_.updateFromBuffer();
  const TargetTrajectories& operatorTarget = operatorTarget_.get();
  if (operatorTarget.empty()) return;
  const vector_t desiredState = operatorTarget.getDesiredState(initTime);
  if (desiredState.size() < 6) return;
  // The linear part of the same channel is the commanded CoM velocity. It has to be read here, before the rules run,
  // for the same reason the yaw rate does and then some: planned_com_override rewrites exactly this channel with the
  // plan's own CoM velocity, so a planner that read the command back off the target afterwards would be fed its own
  // output. At rest that reads as a zero command however far the operator pushes the stick, the standing blend never
  // crosses its half point, and the robot never starts walking.
  commandedVelocity_ = mpcRobotModelPtr_->getBaseComLinearVelocity(desiredState).head<2>();
  const scalar_t yawInertia = computeYawInertia(initState);
  if (yawInertia <= 0.0) return;
  commandedYawRate_ = totalMass_ * mpcRobotModelPtr_->getBaseComVelocity(desiredState)(5) / yawInertia;
}

std::optional<SwitchedModelReferenceManager::PlannedDcm> ContactPlanningReferenceManager::getPlannedDcm(scalar_t time) const {
  // planReferencesUsable(), not hasActivePlan(): an expired plan clamps to its last node rather than reporting that
  // the query ran off its horizon, so it would keep aiming the terminal DCM cost at a stale target forever. And the
  // plan being usable NOW says nothing about `time`: the terminal cost asks at the end of the solver horizon, which
  // lies past the end of any plan older than (plan horizon - MPC horizon), and the lookups below would answer there
  // with the last node's DCM, lagging the robot by (time - endTime) * v. Outside the plan the cost keeps its own
  // support-center reference instead.
  const ContactPlan* absl_nullable plan = usablePlanAt(lastSolveTime_);
  if (plan == nullptr || !planCoversTime(*plan, time)) return std::nullopt;
  // The plan's center of mass is only half of its DCM; the other half is the pendulum it was made on. A plan that does
  // not say which (one built by hand) is taken to be on the running configuration's.
  scalar_t omega = plan->omega;
  if (!(omega > 0.0)) {
    std::lock_guard<std::mutex> lock(configMutex_);
    omega = config_.omega();
  }
  if (!(omega > 0.0)) return std::nullopt;
  const std::optional<vector2_t> position = plan->comPositionAtTime(time);
  const std::optional<vector2_t> velocity = plan->comVelocityAtTime(time);
  if (!position.has_value() || !velocity.has_value()) return std::nullopt;
  PlannedDcm planned;
  planned.dcm = *position + *velocity / omega;
  planned.omega = omega;
  return planned;
}

void ContactPlanningReferenceManager::setContactPlan(const ContactPlan& plan) {
  std::lock_guard<std::mutex> lock(planMutex_);
  pendingPlan_ = plan;
}

bool ContactPlanningReferenceManager::setContactPlan(const ContactPlan& plan, uint64_t planEpoch) {
  std::lock_guard<std::mutex> lock(planMutex_);
  if (planEpoch != planEpoch_) return false;
  pendingPlan_ = plan;
  return true;
}

uint64_t ContactPlanningReferenceManager::planEpoch() const {
  std::lock_guard<std::mutex> lock(planMutex_);
  return planEpoch_;
}

void ContactPlanningReferenceManager::reset() {
  // The base class's reset() runs resetRuntimeState(), i.e. this class's override, after its own buffers.
  SwitchedModelReferenceManager::reset();
  // The operator's copy of the target is a buffered reference like the base class's: what was published before the
  // reset is dropped, and the target set by the reset itself arrives through setTargetTrajectories() after it.
  operatorTarget_.updateFromBuffer();
  operatorTarget_.get() = TargetTrajectories();
}

void ContactPlanningReferenceManager::resetRuntimeState() {
  SwitchedModelReferenceManager::resetRuntimeState();
  {
    std::lock_guard<std::mutex> lock(planMutex_);
    pendingPlan_.reset();
    ++planEpoch_;
  }
  activePlan_.reset();
  appliedSchedule_ = ModeSchedule();
  hasAppliedSchedule_ = false;
  liftOffHistory_ = LiftOffHistory();
  lastSolveTime_ = std::numeric_limits<scalar_t>::lowest();

  commandedYawRate_ = 0.0;
  commandedVelocity_.setZero();
  footYaws_ = makeFeetArray(0.0);
  liftOffYaws_ = makeFeetArray(0.0);
  footPositions_ = makeFeetArray(vector3_t(vector3_t::Zero()));
  liftOffPositions_ = makeFeetArray(vector3_t(vector3_t::Zero()));
  footBookkeepingInitialized_ = false;

  swingLatches_ = makeFeetArray(SwingTimingLatch{});
  lastContactEvents_ = makeFeetArray(ContactEventReport{});
  cadenceTouchDownShift_ = makeFeetArray(0.0);
  dcmStepAdjustment_ = makeFeetArray(vector2_t(vector2_t::Zero()));
  comState_[0].setZero();
  comState_[1].setZero();
  {
    std::lock_guard<std::mutex> lock(predictionMutex_);
    predictedTimes_.clear();
    predictedStates_.clear();
  }
  hasPredictedComState_ = false;
  predictedComState_[0].setZero();
  predictedComState_[1].setZero();
  scheduleShiftLog_.clear();
  replanRequested_.store(false);
  {
    std::lock_guard<std::mutex> lock(targetPoseMutex_);
    targetContactPoses_ = makeFeetArray(TargetContactPose{});
  }
}

bool ContactPlanningReferenceManager::hasPendingPlan() const {
  std::lock_guard<std::mutex> lock(planMutex_);
  return pendingPlan_.has_value();
}

absl::Status ContactPlanningReferenceManager::setConfigStatus(const ContactPlanningConfig& config) {
  // Everything is built before anything is replaced, so a rejection leaves the running configuration and rules alone.
  RETURN_IF_ERROR(config.validateStatus());
  ASSIGN_OR_RETURN(TermCollection<ExecutionRule> rules, buildExecutionRules(config));
  // A reload that switches the heading model on needs the evaluator a start with this file would have installed;
  // without it the planner would silently take the base yaw for the heading while its heading rate stays a whole-body
  // quantity. Resolved before anything is replaced, so a network that does not fit refuses the reload as a whole.
  // Before loadHeadingModelEvaluator() has run - while the interface is still being constructed, and Create() and
  // ContactPlannerModule::Create() hand the configuration over through here - the evaluator is left to that call, which
  // the interface makes once from setupOptimalControlProblem(), after the planner module exists.
  if (loadsHeadingModelEvaluator_) {
    ASSIGN_OR_RETURN(const std::shared_ptr<AngularCenterOfMass> acom,
                     headingModelEvaluatorFor(config, acom_.get(), mpcRobotModelPtr_->modelSettings));
    if (acom != nullptr) acom_ = acom;
  }
  logSwingTimeScaleWarning(config);
  std::lock_guard<std::mutex> lock(configMutex_);
  config_ = config;
  executionRules_ = std::move(rules);
  return absl::OkStatus();
}

ContactPlanningConfig ContactPlanningReferenceManager::getConfig() const {
  std::lock_guard<std::mutex> lock(configMutex_);
  return config_;
}

feet_array_t<vector3_t> ContactPlanningReferenceManager::computeFootPositions(const vector_t& state) {
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  const std::vector<vector3_t> positions = computeContactPositions<scalar_t>(q, pinocchioInterface_, *mpcRobotModelPtr_);
  feet_array_t<vector3_t> feet;
  for (size_t i = 0; i < kNumContacts; ++i) {
    feet[i] = positions[i];
  }
  return feet;
}

std::pair<vector2_t, vector2_t> ContactPlanningReferenceManager::computeComState(const vector_t& state) {
  const PinocchioInterface::Model& model = pinocchioInterface_.getModel();
  PinocchioInterface::Data& data = pinocchioInterface_.getData();
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  pinocchio::centerOfMass(model, data, q, /*computeSubtreeComs=*/false);
  const vector2_t com = data.com[0].head<2>();
  const vector2_t comVelocity = mpcRobotModelPtr_->getBaseComLinearVelocity(state).head<2>();
  return {com, comVelocity};
}

void ContactPlanningReferenceManager::updateFootBookkeeping(scalar_t initTime, const vector_t& initState) {
  footPositions_ = computeFootPositions(initState);
  footYaws_ = readFootYaws();
  const contact_flag_t contacts = contactFlagsAtTime(appliedSchedule_, initTime);
  for (size_t i = 0; i < kNumContacts; ++i) {
    if (contacts[i] || !footBookkeepingInitialized_) {
      liftOffPositions_[i] = footPositions_[i];
      liftOffYaws_[i] = footYaws_[i];
    }
  }
  footBookkeepingInitialized_ = true;
}

void ContactPlanningReferenceManager::activatePendingPlan(scalar_t initTime) {
  std::optional<ContactPlan> candidate;
  {
    std::lock_guard<std::mutex> lock(planMutex_);
    if (pendingPlan_.has_value()) {
      candidate = std::move(*pendingPlan_);
      pendingPlan_.reset();
    }
  }
  if (candidate.has_value() && candidate->valid) {
    // The planner snapshot is taken after this manager ran in the same cycle, so the plan has seen every shift logged at
    // or before its start time; shifts logged later re-timed the schedule it was built on and are applied to the plan too.
    applyScheduleShiftsToPlan(*candidate, scheduleShiftLog_);
    const ContactPlanningConfig config = getConfig();
    // The plan may only change the schedule after its own commit boundary. A boundary that has already passed means the
    // plan's first decisions lie in the past (the planner latency exceeded commitTime, or a touch-down the boundary was
    // extended to happened while the plan was being computed): the plan is dropped and the executed schedule keeps
    // running until a fresh plan arrives. Shifting such a plan forward onto the current boundary instead delayed it by up
    // to a whole swing and re-lifted the foot that had just landed. Likewise a plan that has a foot down where the
    // executed schedule already has it in flight at the merge point was built before that swing was activated and would
    // land the foot there and lift it again.
    const scalar_t mergeTime = std::max(initTime, candidate->committedUntil);
    if (candidate->committedUntil < initTime - 1.0e-6) {
      if (++stalePlanCount_ % 50 == 1) {
        LOG(INFO) << "[ContactPlanningReferenceManager] the contact plan is stale by " << (initTime - candidate->committedUntil)
                  << " s (commitTime " << config.planner.commitTime
                  << " s does not cover the planner latency); keeping the executed schedule.";
      }
    } else if (hasAppliedSchedule_ && !planAgreesWithSwingsInFlight(appliedSchedule_, *candidate, mergeTime)) {
      if (++inconsistentPlanCount_ % 50 == 1) {
        LOG(INFO) << "[ContactPlanningReferenceManager] the contact plan was made before a swing that is in flight at its merge point ("
                  << mergeTime << " s) was committed; keeping the executed schedule.";
      }
    } else {
      activePlan_ = std::move(candidate);
    }
  }
  while (!scheduleShiftLog_.empty() && scheduleShiftLog_.front().first < initTime - kShiftLogAge) {
    scheduleShiftLog_.pop_front();
  }
}

void ContactPlanningReferenceManager::setPredictedTrajectory(const scalar_array_t& times, const vector_array_t& states) {
  std::lock_guard<std::mutex> lock(predictionMutex_);
  if (times.size() != states.size()) {
    predictedTimes_.clear();
    predictedStates_.clear();
    return;
  }
  predictedTimes_ = times;
  predictedStates_ = states;
}

void ContactPlanningReferenceManager::updatePredictedComState(scalar_t initTime) {
  hasPredictedComState_ = false;
  vector_t predictedState;
  {
    std::lock_guard<std::mutex> lock(predictionMutex_);
    if (predictedTimes_.size() < 2 || initTime < predictedTimes_.front() - kPredictionTimeTolerance ||
        initTime > predictedTimes_.back() + kPredictionTimeTolerance) {
      return;
    }
    predictedState = LinearInterpolation::interpolate(initTime, predictedTimes_, predictedStates_);
  }
  // A diverged solve hands over a non-finite trajectory; measuring against it would poison the foot reference.
  if (predictedState.size() != static_cast<Eigen::Index>(mpcRobotModelPtr_->getStateDim()) || !predictedState.allFinite()) return;
  std::tie(predictedComState_[0], predictedComState_[1]) = computeComState(predictedState);
  hasPredictedComState_ = predictedComState_[0].allFinite() && predictedComState_[1].allFinite();
}

void ContactPlanningReferenceManager::handleContactEvents(ExecutionContext& ctx) {
  lastContactEvents_.fill(ContactEventReport{});
  cadenceTouchDownShift_.fill(0.0);
  if (!hasAppliedSchedule_ || !activePlan_.has_value()) return;  // only the schedule of the planning path is adapted

  for (const std::unique_ptr<ExecutionRule>& rule : executionRules_) rule->beginCycle(ctx);
  cadenceTouchDownShift_ = ctx.cadenceTouchDownShift;
  lastContactEvents_ = adaptScheduleWithRules(appliedSchedule_, ctx, executionRules_, swingLatches_);

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
    // Later events moved: keep the plan aligned with the executed schedule so that the merge does not cut phases short.
    if (activePlan_->valid) activePlan_->shiftInTime(totalShift);
    scheduleShiftLog_.emplace_back(ctx.time, totalShift);
  }
  if (replan) replanRequested_.store(true);
}

void ContactPlanningReferenceManager::modifyReferences(scalar_t initTime,
                                                       scalar_t finalTime,
                                                       const vector_t& initState,
                                                       size_t initMode,
                                                       TargetTrajectories& targetTrajectories,
                                                       ModeSchedule& modeSchedule) {
  const scalar_t timeHorizon = finalTime - initTime;
  const scalar_t lowerBoundTime = initTime - timeHorizon;
  const scalar_t upperBoundTime = finalTime + timeHorizon;
  const ContactPlanningConfig config = getConfig();

  // The swing trajectory planner's configuration is replaced by the task file's reload, which does not pass through this
  // manager, so a swingTimeScale moved from the GUI is checked against the planned swings here, once per change.
  if (checkedSwingTimeScale_ != swingTrajectoryPtr_->getConfig().swingTimeScale) logSwingTimeScaleWarning(config);

  activatePendingPlan(initTime);

  ExecutionContext ctx(config);
  ctx.time = initTime;
  ctx.measuredContact = modeNumber2StanceLeg(initMode);
  ctx.totalMass = totalMass_;
  // An expired plan is withheld from the execution rules for the same reason the schedule merge refuses it: every
  // ContactPlan lookup clamps, so PlannedComOverride and PlannedHeadingOverride would steer the whole-body MPC
  // towards the last node of a horizon that has already passed.
  ctx.activePlan = usablePlanAt(initTime);
  // Only the rules that compare the center of mass with the NMPC's prediction need the kinematics; skip them when none is
  // listed so that the default configuration does exactly the work it did before they existed.
  if (rulesNeedComState()) {
    std::tie(comState_[0], comState_[1]) = computeComState(initState);
  }
  if (rulesNeedPredictedTrajectory()) {
    updatePredictedComState(initTime);
  } else {
    hasPredictedComState_ = false;
  }
  ctx.hasPredictedComState = hasPredictedComState_;
  ctx.com = comState_[0];
  ctx.comVelocity = comState_[1];
  ctx.basePosition = mpcRobotModelPtr_->getBasePosition(initState).head<2>();
  ctx.predictedCom = predictedComState_[0];
  ctx.predictedComVelocity = predictedComState_[1];

  // Adapt the schedule executed so far to the measured contact state before it is merged with the plan.
  handleContactEvents(ctx);

  // The plan honored the applied schedule up to its own commit boundary, and that boundary already covers every swing
  // that was in flight or about to start when the plan was made (commitBoundary extends it to their touch-downs). So
  // the merge happens exactly there. Merging any later, e.g. at the boundary computed now, cut the plan's first
  // lift-off short by the plan's age: the merged swing began at the later merge time but kept the plan's touch-down,
  // and reached the controller shorter than minSwingDuration. Once the active plan's boundary has passed (no fresh plan
  // for longer than commitTime) the applied schedule keeps running; activatePendingPlan() never lets such a plan in.
  scalar_t commitTime = initTime;
  bool planFresh = false;
  const ContactPlan* absl_nullable validPlan = activePlan_.has_value() && activePlan_->valid ? &*activePlan_ : nullptr;
  if (validPlan != nullptr) {
    planFresh = validPlan->committedUntil >= initTime - 1.0e-6;
    commitTime = std::max(initTime, validPlan->committedUntil);
  }
  const bool planUsable =
      validPlan != nullptr && planFresh && validPlan->endTime() > commitTime + config.planner.dt && validPlan->startTime <= commitTime;

  ModeSchedule schedule;
  if (planUsable) {
    const ModeSchedule planSchedule = validPlan->toModeSchedule();
    const ModeSchedule& applied =
        hasAppliedSchedule_ ? appliedSchedule_ : gaitSchedulePtr_->getModeSchedule(lowerBoundTime, upperBoundTime);
    schedule = mergeModeSchedules(applied, planSchedule, commitTime, lowerBoundTime, upperBoundTime);
  } else if (hasAppliedSchedule_ && activePlan_.has_value()) {
    // The plan ran out (planner stalled): keep executing the applied schedule, which ends in STANCE.
    schedule = mergeModeSchedules(appliedSchedule_, ModeSchedule(/*eventTimesInput=*/{}, {ModeNumber::kStance}), upperBoundTime,
                                  lowerBoundTime, upperBoundTime);
  } else {
    schedule = gaitSchedulePtr_->getModeSchedule(lowerBoundTime, upperBoundTime);
  }

  captureOperatorCommand(initTime, initState);

  // The operator's target is resampled onto the plan's node grid so that a rule which rewrites the state at a node has
  // a knot there to rewrite. Three things about this used to be wrong.
  //
  // It was gated on `!executionRules_.empty()`, i.e. on ANY rule being listed, while ExecutionRule::overrideTarget is
  // a no-op by default and only the two planned_*_override rules implement it - so a configuration listing only
  // schedule rules (phase_resetting, energy_cadence_modulation, dcm_step_adjustment) paid for the resample, and for
  // the truncation below, in exchange for nothing at all. It now asks whether a listed rule really rewrites.
  //
  // It REPLACED the target with a grid spanning the plan's horizon alone. Nothing intersected that with the solver
  // horizon, and OCS2 zero-order-extrapolates past the last knot, so whenever the plan ended before finalTime - which
  // is any time the plan is older than (plan horizon - MPC horizon), the same regime as an expiring plan - the whole
  // reference tail froze at the plan-end sample instead of following the command. The grid is now EXTENDED past the
  // plan's end with the operator's knots, and with a final sample at finalTime when even those fall short, and the
  // rules leave every knot outside the plan alone (planCoversTime).
  //
  // And that extension read the LIVE target, which ReferenceManager::preSolverRun keeps from the previous cycle - the
  // copy the rules had already rewritten. From the second cycle without a newly published target the "operator's"
  // tail was therefore the previous plan's clamped end again. The extension now reads the target as the operator
  // published it (the copy setTargetTrajectories keeps for the command), and falls back to the live one only when
  // nothing was ever published through this manager.
  const ContactPlan* absl_nullable usablePlan = usablePlanAt(initTime);
  if (usablePlan != nullptr && !targetTrajectories.empty() && rulesRewriteTarget()) {
    const ContactPlan& plan = *usablePlan;
    const TargetTrajectories& operatorTarget = operatorTarget_.get().empty() ? targetTrajectories : operatorTarget_.get();
    TargetTrajectories denseTarget;
    const size_t numNodes = plan.comPosition.size();
    const size_t numOperatorKnots = operatorTarget.timeTrajectory.size();
    denseTarget.timeTrajectory.reserve(numNodes + numOperatorKnots + 1);
    denseTarget.stateTrajectory.reserve(numNodes + numOperatorKnots + 1);
    const bool hasInput = !targetTrajectories.inputTrajectory.empty();
    if (hasInput) denseTarget.inputTrajectory.reserve(numNodes + numOperatorKnots + 1);
    // The input of a tail knot: the operator's when the published target carries one, the live target's otherwise.
    const std::function<vector_t(scalar_t)> tailInput = [&](scalar_t time) -> vector_t {
      return operatorTarget.inputTrajectory.empty() ? targetTrajectories.getDesiredInput(time) : operatorTarget.getDesiredInput(time);
    };
    for (size_t i = 0; i < numNodes; ++i) {
      const scalar_t time = plan.startTime + static_cast<scalar_t>(i) * plan.dt;
      denseTarget.timeTrajectory.push_back(time);
      denseTarget.stateTrajectory.push_back(targetTrajectories.getDesiredState(time));
      if (hasInput) denseTarget.inputTrajectory.push_back(targetTrajectories.getDesiredInput(time));
    }
    const scalar_t lastDenseTime = denseTarget.timeTrajectory.empty() ? plan.startTime : denseTarget.timeTrajectory.back();
    for (size_t i = 0; i < numOperatorKnots; ++i) {
      const scalar_t time = operatorTarget.timeTrajectory[i];
      if (time <= lastDenseTime) continue;
      denseTarget.timeTrajectory.push_back(time);
      denseTarget.stateTrajectory.push_back(operatorTarget.stateTrajectory[i]);
      if (hasInput) denseTarget.inputTrajectory.push_back(tailInput(time));
    }
    if (denseTarget.timeTrajectory.empty() || denseTarget.timeTrajectory.back() < finalTime) {
      denseTarget.timeTrajectory.push_back(finalTime);
      denseTarget.stateTrajectory.push_back(operatorTarget.getDesiredState(finalTime));
      if (hasInput) denseTarget.inputTrajectory.push_back(tailInput(finalTime));
    }
    targetTrajectories = std::move(denseTarget);
  }

  for (const std::unique_ptr<ExecutionRule>& rule : executionRules_) rule->overrideTarget(ctx, targetTrajectories);
  const scalar_t previousTerrainHeight = targetTerrainHeight_;
  const scalar_t terrainHeight = adaptToCurrentGroundHeight(targetTrajectories, initState, initMode);
  // The operator's copy of the target stands on the same ground as the live one - both are the target as published -
  // and the tail of the densified target above is read from it, so it moves with the ground as well. Left behind, a
  // hot reload of `terrainHeight` moved the plan's knots but put every tail knot back on the old ground from the next
  // run on, for as long as the operator published nothing new (for good, under a pose command).
  const scalar_t heightChange = terrainHeight - previousTerrainHeight;
  if (heightChange != 0.0) {
    for (vector_t& operatorState : operatorTarget_.get().stateTrajectory) {
      mpcRobotModelPtr_->adaptBasePoseHeight(operatorState, heightChange);
    }
  }
  updateSwingTrajectories(schedule, ctx, terrainHeight);

  modeSchedule = schedule;
  modeSchedule_ = schedule;
  appliedSchedule_ = schedule;
  hasAppliedSchedule_ = true;
  lastSolveTime_ = initTime;
  updateFootBookkeeping(initTime, initState);
  updateDcmStepAdjustment(ctx);
  updateTargetContactPoses(initTime, terrainHeight);
}

void ContactPlanningReferenceManager::updateTargetContactPoses(scalar_t initTime, scalar_t terrainHeight) {
  feet_array_t<TargetContactPose> poses = makeFeetArray(TargetContactPose{});
  const ContactPlan* absl_nullable usablePlan = usablePlanAt(initTime);
  if (usablePlan != nullptr && footBookkeepingInitialized_) {
    TargetContactPoseInputs inputs;
    inputs.time = initTime;
    inputs.footPositions = footPositions_;
    inputs.footYaws = footYaws_;
    inputs.dcmStepAdjustment = dcmStepAdjustment_;
    inputs.terrainHeight = terrainHeight;
    poses = computeTargetContactPoses(*usablePlan, appliedSchedule_, inputs);
  }
  std::lock_guard<std::mutex> lock(targetPoseMutex_);
  targetContactPoses_ = poses;
}

feet_array_t<TargetContactPose> ContactPlanningReferenceManager::getTargetContactPoses() const {
  std::lock_guard<std::mutex> lock(targetPoseMutex_);
  return targetContactPoses_;
}

void ContactPlanningReferenceManager::updateSwingTrajectories(const ModeSchedule& schedule,
                                                              const ExecutionContext& ctx,
                                                              scalar_t terrainHeight) {
  const SwingTrajectoryPlanner::Config& swingConfig = swingTrajectoryPtr_->getConfig();
  const size_t numPhases = schedule.modeSequence.size();
  feet_array_t<scalar_array_t> liftOffHeights = makeFeetArray(scalar_array_t(numPhases, terrainHeight));
  feet_array_t<scalar_array_t> touchDownHeights =
      makeFeetArray(scalar_array_t(numPhases, terrainHeight + swingConfig.touchDownHeightOffset));

  // A rule may have a foot searching for the ground (a late touch-down): its height reference then continues the planned
  // swing as a straight descent instead of hovering.
  feet_array_t<std::optional<SwingTrajectoryPlanner::GroundSearch>> groundSearches =
      makeFeetArray(std::optional<SwingTrajectoryPlanner::GroundSearch>{});
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    for (const std::unique_ptr<ExecutionRule>& rule : executionRules_) {
      const std::optional<GroundSearchRequest> search = rule->groundSearch(ctx, foot, schedule, swingLatches_[foot]);
      if (search.has_value()) {
        groundSearches[foot] = SwingTrajectoryPlanner::GroundSearch{.liftOffTime = search->liftOffTime,
                                                                    .plannedTouchDownTime = search->plannedTouchDownTime,
                                                                    .descentVelocity = search->searchVelocity};
        break;
      }
    }
  }
  swingTrajectoryPtr_->update(schedule, liftOffHeights, touchDownHeights, groundSearches);
}

void ContactPlanningReferenceManager::updateDcmStepAdjustment(const ExecutionContext& ctx) {
  dcmStepAdjustment_.fill(vector2_t::Zero());
  for (const std::unique_ptr<ExecutionRule>& rule : executionRules_) {
    const feet_array_t<vector2_t> correction = rule->correctFootholds(ctx, appliedSchedule_);
    for (size_t foot = 0; foot < kNumContacts; ++foot) dcmStepAdjustment_[foot] += correction[foot];
  }
}

scalar_t ContactPlanningReferenceManager::commitBoundary(scalar_t time) const {
  return commitBoundaryForSchedule(appliedSchedule_, time, getConfig().planner.commitTime, getConfig().planner.maxCommitExtension);
}

std::optional<std::pair<scalar_t, scalar_t>> ContactPlanningReferenceManager::swingPhase(size_t contactIndex, scalar_t time) const {
  return swingPhaseAtTime(appliedSchedule_, contactIndex, time);
}

std::optional<SwingFootReference> ContactPlanningReferenceManager::getSwingFootReference(size_t contactIndex, scalar_t time) const {
  const ContactPlan* absl_nullable plan = usablePlanAt(lastSolveTime_);
  if (plan == nullptr || !footBookkeepingInitialized_) return std::nullopt;
  const std::optional<std::pair<scalar_t, scalar_t>> phase = swingPhase(contactIndex, time);
  if (!phase.has_value()) return std::nullopt;
  const scalar_t liftOffTime = phase->first;
  const scalar_t touchDownTime = phase->second;
  const std::optional<vector2_t> landing = plan->footholdAtTime(contactIndex, touchDownTime);
  if (!landing.has_value()) return std::nullopt;

  // The xy motion is timed on the swing without its late touch-down extension: a foot searching for the ground holds its
  // landing target instead of drifting back along the step.
  scalar_t xyTouchDownTime = touchDownTime;
  const SwingTimingLatch& latch = swingLatches_[contactIndex];
  if (latch.active && std::abs(latch.liftOffTime - liftOffTime) <= kSameSwingTolerance && latch.lateExtension > 0.0) {
    xyTouchDownTime = touchDownTime - latch.lateExtension;
  }
  const scalar_t duration = xyTouchDownTime - liftOffTime;
  if (duration <= 1.0e-6) return std::nullopt;

  // The xy interpolation starts where the foot stands at lift-off. For the swing that ends the foot's current contact
  // phase (in flight now, or the next to lift) that is the latched measured position. For a later swing of the same foot
  // inside the horizon the foot first lands somewhere else, so the start is the planned foot position at the node
  // BEFORE that lift-off, i.e. the stance position before the foot moves.
  //
  // footholdBeforeTime(), not footholdAtTime() with the argument nudged back half a step: the nudge rounds back up to
  // the lift-off node itself, and under the shipped `hlip` planner that node already carries the swing's LANDING
  // position, so `start` became `*landing`, `delta` became zero, and this whole reference collapsed to a constant at
  // the landing target with zero commanded velocity for every later swing in the horizon. See ContactPlan.h.
  vector2_t start = liftOffPositions_[contactIndex].head<2>();
  const std::optional<scalar_t> currentLiftOff = currentOrNextLiftOffTime(appliedSchedule_, contactIndex, lastSolveTime_);
  const bool endsCurrentContactPhase = currentLiftOff.has_value() && std::abs(*currentLiftOff - liftOffTime) <= kSameSwingTolerance;
  if (!endsCurrentContactPhase) {
    const std::optional<vector2_t> plannedStance = plan->footholdBeforeTime(contactIndex, liftOffTime);
    if (plannedStance.has_value()) start = *plannedStance;
  }
  const scalar_t tau = std::clamp((time - liftOffTime) / duration, 0.0, 1.0);

  // Use a cubic spline with p'(0) = 1 and p'(1) = 0 to command an initial velocity matching the step velocity.
  // This prevents the swing foot from kicking backward relative to the moving body at lift-off.
  const scalar_t tau2 = tau * tau;
  const scalar_t blend = -tau2 * tau + tau2 + tau;
  const scalar_t blendRate = (-3.0 * tau2 + 2.0 * tau + 1.0) / duration;

  // The landing target offset of the foothold rules (zero without them) is blended in with the same profile, so the
  // target moves smoothly. It is computed for the swing in flight at the last solve and belongs to that swing alone: a
  // later swing of the same foot inside the horizon lands on its planned foothold.
  const std::optional<std::pair<scalar_t, scalar_t>> swingInFlight = swingPhase(contactIndex, lastSolveTime_);
  const bool isTheSwingInFlight = swingInFlight.has_value() && std::abs(swingInFlight->first - liftOffTime) <= kSameSwingTolerance;
  const vector2_t adjustment = isTheSwingInFlight ? dcmStepAdjustment_[contactIndex] : vector2_t(vector2_t::Zero());
  const vector2_t delta = *landing + adjustment - start;

  SwingFootReference reference;
  reference.position.head<2>() = start + blend * delta;
  reference.position(2) = swingTrajectoryPtr_->getZpositionConstraint(contactIndex, time);
  reference.linearVelocity.head<2>() = blendRate * delta;
  reference.linearVelocity(2) = swingTrajectoryPtr_->getZvelocityConstraint(contactIndex, time);
  // Heading model: the foot yaw turns from its lift-off yaw to the planned landing yaw with the same profile.
  if (plan->hasHeading()) {
    const std::optional<scalar_t> landingYaw = plan->footYawAtTime(contactIndex, touchDownTime);
    if (landingYaw.has_value()) {
      scalar_t startYaw = liftOffYaws_[contactIndex];
      if (!endsCurrentContactPhase) {
        const std::optional<scalar_t> plannedYaw = plan->footYawBeforeTime(contactIndex, liftOffTime);
        if (plannedYaw.has_value()) startYaw = *plannedYaw;
      }
      reference.yaw = startYaw + blend * (moduloAngleWithReference(*landingYaw, startYaw) - startYaw);
    }
  }
  return reference;
}

ContactPlannerInput ContactPlanningReferenceManager::makePlannerInput(scalar_t initTime,
                                                                      const vector_t& initState,
                                                                      const vector2_t& velocityCommand) {
  ContactPlannerInput input;
  input.time = initTime;
  input.velocityCommand = velocityCommand;

  std::tie(input.comPosition, input.comVelocity) = computeComState(initState);
  input.yaw = mpcRobotModelPtr_->getBaseOrientationEulerZYX(initState)(0);

  const feet_array_t<vector3_t> feet = computeFootPositions(initState);
  for (size_t i = 0; i < kNumContacts; ++i) {
    input.footPositions[i] = feet[i].head<2>();
  }

  // The operator's commanded yaw rate is filled whether or not the heading model is on. It is not part of the heading
  // MODEL - it is part of the COMMAND, and the standing/walking blend reads it to decide whether the robot should be
  // stepping at all. Left inside the heading-model branch, a robot without that model contributed nothing from the yaw
  // stick to the blend's activity, so `alpha` never crossed its half point on yaw alone and the robot would not start
  // stepping to turn in place however hard it was asked.
  input.headingRateCommand = commandedYawRate();

  const ContactPlanningConfig config = getConfig();
  if (config.usesHeadingModel()) {
    // Heading model: the whole-body heading, its rate from the angular momentum about the vertical, and the foot yaws
    // unwrapped near the heading. The planning frame is the heading.
    const feet_array_t<scalar_t> yaws = readFootYaws();
    input.heading = computeHeading(initState);
    input.yaw = input.heading;
    input.yawInertia = computeYawInertia(initState);
    const scalar_t angularMomentumZ = totalMass_ * mpcRobotModelPtr_->getBaseComVelocity(initState)(5);
    input.headingRate = input.yawInertia > 0.0 ? angularMomentumZ / input.yawInertia : 0.0;
    for (size_t i = 0; i < kNumContacts; ++i) {
      input.footYaws[i] = moduloAngleWithReference(yaws[i], input.heading);
    }
  }

  // Events at exactly `initTime` count as passed (a touch-down placed at the current time by the phase resetting is a
  // contact for the planner), consistently with every other schedule query of this manager.
  const ModeSchedule& schedule = hasAppliedSchedule_ ? appliedSchedule_ : this->getModeSchedule();
  input.committedUntil = hasAppliedSchedule_ ? commitBoundary(initTime) : initTime + config.planner.commitTime;
  // At least one node stays free of the committed window. The last swung foot comes from liftOffHistory_, which
  // outlives the schedule's history window.
  fillPlannerInputFromSchedule(schedule, config.planner.dt, std::max(0, config.planner.numNodes - 1), liftOffHistory_, input);
  return input;
}

}  // namespace ocs2::humanoid
