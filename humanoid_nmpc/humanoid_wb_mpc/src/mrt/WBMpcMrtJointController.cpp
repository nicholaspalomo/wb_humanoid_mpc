/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/robot/JointPdGainsFromConfig.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/ControlMode.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/mrt/JointActionAccess.h"
#include "humanoid_common_mpc/mrt/LoggingControllerEventSink.h"
#include "humanoid_common_mpc/mrt/LookaheadPolicyEvaluator.h"
#include "humanoid_common_mpc/mrt/SafetyDecay.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"
#include "humanoid_wb_mpc/mrt/WBMpcResetTarget.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotStateContactEstimator.h"

namespace ocs2::humanoid {

namespace {

/** The class name the events carry. */
constexpr char kControllerName[] = "WBMpcMrtJointController";

/** The fraction of their JOINT_PD position gain the joints outside the MPC keep
 * in GRAVITY_COMP. */
constexpr scalar_t kGravityCompOtherJointKpScale = 0.5;

/** The message of Create() for a link factory that made no link. */
constexpr char kNoMpcLinkMessage[] =
    "[WBMpcMrtJointController] the MPC link factory made no link.";

/** The link of an MPC in this process, as the in-process overload makes it. */
MpcLinkFactory inProcessMpcLinkFactory(
    MPC_BASE& mpc,
    MPC_MRT_Interface::SolverObservationObserver solverObservationObserver,
    scalar_t mpcDesiredFrequency) {
  InProcessMpcLink::Config config;
  config.mpcDesiredFrequency = mpcDesiredFrequency;
  config.solverThreadName = "WB MPC Solver Thread";
  config.solverObservationObserver = std::move(solverObservationObserver);
  return InProcessMpcLink::factory(mpc, std::move(config));
}

}  // namespace

absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>>
WBMpcMrtJointController::Create(
    const ::robot::model::RobotDescription& robotDescription,
    const ModelSettings& modelSettings,
    MPC_BASE& mpc,
    MPC_MRT_Interface::SolverObservationObserver solverObservationObserver,
    PinocchioInterface pinocchioInterface,
    scalar_t mpcDesiredFrequency,
    const std::string& pdGainsFile) {
  return Create(
      robotDescription, modelSettings,
      inProcessMpcLinkFactory(mpc, std::move(solverObservationObserver),
                              mpcDesiredFrequency),
      std::move(pinocchioInterface), pdGainsFile);
}

absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>>
WBMpcMrtJointController::Create(
    const ::robot::model::RobotDescription& robotDescription,
    const ModelSettings& modelSettings,
    const MpcLinkFactory& mpcLinkFactory,
    PinocchioInterface pinocchioInterface,
    const std::string& pdGainsFile) {
  // The joints the control cycle indexes without a check: refused here, by
  // name, rather than by the constructor's check.
  RETURN_IF_ERROR(
      robotDescription.findJointIndices(modelSettings.mpcModelJointNames)
          .status());
  RETURN_IF_ERROR(
      robotDescription.findJointIndices(modelSettings.fixedJointNames)
          .status());
  // The sole frames the inverse dynamics of a policy applies its contact
  // wrenches at, which it then reads unchecked.
  RETURN_IF_ERROR(checkContactFramesExist(pinocchioInterface.getModel(),
                                          modelSettings.contactNames));
  ASSIGN_OR_RETURN(InitialPdGains initialPdGains,
                   loadInitialPdGains(modelSettings, pdGainsFile));
  // The manifold of the state the controller writes and the policy acts on,
  // which the evaluator retains.
  ASSIGN_OR_RETURN(
      std::shared_ptr<const StateManifold> stateManifold,
      stateManifoldOf(WBAccelMpcRobotModel<scalar_t>(modelSettings)));
  // The gravity torques and the inverse dynamics of the control cycle, on a
  // FreeFlyer twin of the model.
  ASSIGN_OR_RETURN(std::unique_ptr<RealtimeRigidBodyDynamics> rigidBodyDynamics,
                   RealtimeRigidBodyDynamics::Create(
                       pinocchioInterface.getModel(),
                       getContactFrameIndexArray<scalar_t>(
                           pinocchioInterface,
                           WBAccelMpcRobotModel<scalar_t>(modelSettings))));
  std::unique_ptr<WBMpcMrtJointController> controller =
      absl::WrapUnique(new WBMpcMrtJointController(
          robotDescription, modelSettings, mpcLinkFactory,
          std::move(pinocchioInterface), pdGainsFile, std::move(initialPdGains),
          std::move(stateManifold), std::move(rigidBodyDynamics)));
  if (controller->mpcLink_ == nullptr) {
    return absl::InvalidArgumentError(kNoMpcLinkMessage);
  }
  return controller;
}

WBMpcMrtJointController::WBMpcMrtJointController(
    const ::robot::model::RobotDescription& robotDescription,
    const ModelSettings& modelSettings,
    const MpcLinkFactory& mpcLinkFactory,
    PinocchioInterface pinocchioInterface,
    const std::string& pdGainsFile,
    InitialPdGains initialPdGains,
    std::shared_ptr<const StateManifold> stateManifold,
    std::unique_ptr<RealtimeRigidBodyDynamics> rigidBodyDynamics)
    : mpcLink_(mpcLinkFactory([this](const SystemObservation& observation) {
        return currentObservationToResetTrajectory(observation);
      })),
      contactEstimator_(
          std::make_shared<::robot::model::RobotStateContactEstimator>()),
      contactEstimateIntake_(kControllerName),
      pinocchioInterface_(std::move(pinocchioInterface)),
      mpcRobotModel_(modelSettings),
      eventSink_(&LoggingControllerEventSink::instance()),
      stateManifold_(std::move(stateManifold)),
      policyEvaluator_(
          ipc::ModelDimensions{.stateDim = mpcRobotModel_.getStateDim(),
                               .inputDim = mpcRobotModel_.getInputDim(),
                               .numModes = kNumModeNumbers},
          stateManifold_.get()),
      rigidBodyDynamics_(std::move(rigidBodyDynamics)),
      robotWeight_(computeRobotWeight(pinocchioInterface_)),
      mpcPolicyState_(vector_t::Zero(mpcRobotModel_.getStateDim())),
      mpcPolicyInput_(vector_t::Zero(mpcRobotModel_.getInputDim())),
      pdGainsFile_(pdGainsFile),
      robotJointNames_(robotDescription.getJointNames()),
      mpcModelJointNames_(modelSettings.mpcModelJointNames),
      fixedJointNames_(modelSettings.fixedJointNames),
      pdGainsLastWriteTime_(initialPdGains.fileWriteTime),
      pdGains_(std::move(initialPdGains.gains)),
      pdGainsMailbox_(pdGains_) {
  mpcJointIndices_ =
      robotDescription.getJointIndices(modelSettings.mpcModelJointNames);
  otherJointIndices_ =
      robotDescription.getJointIndices(modelSettings.fixedJointNames);
  // Once, here, what the control cycle then takes for granted
  // (jointActionUnchecked()): every joint it writes carries an action in a
  // RobotJointAction of the robot description.
  checkJointIndices(robotDescription, mpcJointIndices_, kControllerName);
  checkJointIndices(robotDescription, otherJointIndices_, kControllerName);
  currentMpcObservation_.state = vector_t::Zero(mpcRobotModel_.getStateDim());
  currentMpcObservation_.input = vector_t::Zero(mpcRobotModel_.getInputDim());
  latestPolicyInput_ = vector_t::Zero(mpcRobotModel_.getInputDim());

  // The control thread's workspaces, at the sizes it writes them with.
  mpcJointPositions_ = vector_t::Zero(mpcJointIndices_.size());
  mpcJointVelocities_ = vector_t::Zero(mpcJointIndices_.size());
  estimatedContactFlags_.assign(kNumContacts, true);
  gravityCoordinates_ = vector_t::Zero(mpcRobotModel_.getConfigurationDim());
  policyCoordinates_ = vector_t::Zero(mpcRobotModel_.getConfigurationDim());
  safetyHoldMpcJointPositions_ = vector_t::Zero(mpcJointIndices_.size());
  safetyHoldOtherJointPositions_ = vector_t::Zero(otherJointIndices_.size());
  nominalJointPositions_.reserve(robotDescription.getNumJoints());
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::StatusOr<WBMpcMrtJointController::InitialPdGains>
WBMpcMrtJointController::loadInitialPdGains(const ModelSettings& modelSettings,
                                            const std::string& pdGainsFile) {
  // The write time first, then the gains: a save that lands between the two is
  // reloaded by the next poll.
  InitialPdGains initial;
  initial.fileWriteTime = jointPdGainsFileWriteTime(pdGainsFile);
  // At start-up there are no gains in use to keep, and the controller has none
  // of its own: no file, a missing one or one it refuses, and it does not
  // start.
  ASSIGN_OR_RETURN(
      initial.gains,
      loadRequiredJointPdGains(pdGainsFile, modelSettings.mpcModelJointNames,
                               modelSettings.fixedJointNames, kControllerName));
  LOG(INFO) << "[WBMpcMrtJointController] Loaded joint PD gains from "
            << pdGainsFile;
  return initial;
}

absl::Status WBMpcMrtJointController::postPdGains(
    const absl::StatusOr<JointPdGains>& gains,
    absl::string_view source,
    uint64_t ticket) {
  if (!gains.ok()) {
    LOG(WARNING)
        << "[WBMpcMrtJointController] Warning: Refused the PD gains from "
        << source
        << "; the gains in use are kept: " << gains.status().message();
    return gains.status();
  }
  LOG(INFO) << "[WBMpcMrtJointController] Loaded joint PD gains from "
            << source;
  return pdGainsMailbox_.post(*gains, ticket);
}

absl::Status WBMpcMrtJointController::setPdGains(
    const mpc_config::JointPdGainsFile& gains) {
  uint64_t ticket = 0;
  {
    // Its place in line and its arrival before anything else: gains handed in
    // after these win, however fast they are resolved, and a gains file written
    // before them does not undo them (pollPdGainsFile()).
    absl::MutexLock lock(&pdGainsFileMutex_);
    ticket = pdGainsMailbox_.takeTicket();
    liveGainsArrival_ = std::filesystem::file_time_type::clock::now();
  }
  LOG(INFO) << "[WBMpcMrtJointController] setPdGains received gains for "
            << gains.joint_gains.size() << " named joints";
  RETURN_IF_ERROR(postPdGains(
      jointPdGainsFromConfig(gains, mpcModelJointNames_, fixedJointNames_),
      "setPdGains", ticket));
  LOG(INFO) << "[WBMpcMrtJointController] PD gains from setPdGains are applied "
               "at the next control cycle.";
  return absl::OkStatus();
}

void WBMpcMrtJointController::pollPdGainsFile() {
  if (pdGainsFile_.empty()) return;
  absl::MutexLock lock(&pdGainsFileMutex_);
  std::error_code ec;
  const std::filesystem::file_time_type last_write =
      std::filesystem::last_write_time(pdGainsFile_, ec);
  if (ec || last_write == pdGainsLastWriteTime_) return;
  pdGainsLastWriteTime_ = last_write;
  if (liveGainsArrival_.has_value() && last_write < *liveGainsArrival_) {
    LOG(INFO) << "[WBMpcMrtJointController] " << pdGainsFile_
              << " was written before the newest gains of setPdGains(), which "
                 "supersede it; it was not applied.";
    return;
  }
  const uint64_t ticket = pdGainsMailbox_.takeTicket();
  absl::StatusOr<std::optional<JointPdGains>> loaded =
      loadJointPdGains(pdGainsFile_, mpcModelJointNames_, fixedJointNames_);
  // A refused file is logged by postPdGains() and leaves the gains in use.
  if (!loaded.ok()) {
    postPdGains(loaded.status(), pdGainsFile_, ticket).IgnoreError();
    return;
  }
  if (!loaded->has_value()) {
    return;  // gone again (an editor that saves by renaming); the next poll
             // sees the new file
  }
  postPdGains(**std::move(loaded), pdGainsFile_, ticket).IgnoreError();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

WBMpcMrtJointController::~WBMpcMrtJointController() {
  // Stop the solver before anything its reset target reads is destroyed. Its
  // thread used to loop on `while (true)` and ignore the stop flag, so the join
  // never returned and a controller could not be destroyed.
  if (mpcLink_ != nullptr) mpcLink_->stop();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::startMpcThread(
    const ::robot::model::RobotState& initRobotState) {
  updateMpcObservation(currentMpcObservation_, initRobotState);
  // Set observation to MPC and start serving policies.
  mpcLink_->start(currentMpcObservation_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::updateMpcState(
    vector_t& mpcState, const ::robot::model::RobotState& robotState) {
  // The whole-body state is [q, v] (WBAccelMpcRobotModel), written in place by
  // the robot model's measurement helpers: q of the measured base pose and
  // joint positions, v of the base velocities the robot measures in the base
  // frame and of the joint velocities.
  const quaternion_t rootOrientation = robotState.getQuatBaseRtWorld();
  robotState.getJointPositions(mpcJointIndices_, mpcJointPositions_);
  robotState.getJointVelocities(mpcJointIndices_, mpcJointVelocities_);
  mpcRobotModel_.fillGeneralizedCoordinatesFromMeasurement(
      robotState.getPosBaseRtWorldEwrtWorld(), rootOrientation.coeffs(),
      mpcJointPositions_, mpcState.head(mpcRobotModel_.getConfigurationDim()));
  mpcRobotModel_.fillGeneralizedVelocitiesFromMeasurement(
      rootOrientation.coeffs(), robotState.getLinvelBaseRtWorldEwrtBase(),
      robotState.getAngvelBaseRtWorldEwrtBase(), mpcJointVelocities_,
      mpcState.tail(mpcRobotModel_.getVelocityDim()));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::updateMpcObservation(
    ocs2::SystemObservation& mpcObservation,
    const ::robot::model::RobotState& robotState) {
  updateMpcState(mpcObservation.state, robotState);
  mpcObservation.time = robotState.getTime();
  mpcObservation.input = vector_t::Zero(
      mpcRobotModel_.getInputDim());  // Add contact forces later.
  // The measured contact state of this cycle: the observation mode of the MPC,
  // and the gate of the contact wrenches in the inverse dynamics
  // (computeJointControlAction).
  contactEstimator_->estimateContactFlags(robotState, estimatedContactFlags_);
  // An estimate without one flag per contact point is refused and the measured
  // contact state stays the last one; the first refusal of a run is reported
  // through the event sink (ContactEstimateIntake).
  contactEstimateIntake_.take(estimatedContactFlags_, measuredContactFlags_,
                              *eventSink_);
  mpcObservation.mode = stanceLeg2ModeNumber(measuredContactFlags_);
  contactWrenchGate_.update(mpcObservation.time, measuredContactFlags_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::setContactWrenchGateConfig(
    const ContactWrenchGate::Config& config) {
  if (contactWrenchGate_.setConfig(config)) {
    postEvent(ControllerEventCode::kContactWrenchGateChanged,
              config.debounceTime, config.rampTime);
  } else {
    postEvent(ControllerEventCode::kContactWrenchGateRefused,
              config.debounceTime, config.rampTime);
  }
}

void WBMpcMrtJointController::setEventSink(
    ControllerEventSink* absl_nullable eventSink) {
  eventSink_ = eventSink != nullptr ? eventSink
                                    : &LoggingControllerEventSink::instance();
}

void WBMpcMrtJointController::postEvent(ControllerEventCode code,
                                        scalar_t value0,
                                        scalar_t value1,
                                        absl::string_view text) {
  eventSink_->post(
      makeControllerEvent(code, kControllerName, value0, value1, text));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::setContactEstimator(
    std::shared_ptr<::robot::model::ContactEstimator> contactEstimator) {
  contactEstimator_ =
      contactEstimator
          ? std::move(contactEstimator)
          : std::make_shared<::robot::model::RobotStateContactEstimator>();
  const absl::string_view name = contactEstimator_->getName();
  contactEstimateIntake_.resetEstimator(name);
  postEvent(ControllerEventCode::kContactEstimatorChanged, /*value0=*/0.0,
            /*value1=*/0.0, name);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::setControlMode(absl::string_view mode) {
  // Called every cycle; compared and assigned through string_views, so that
  // nothing is allocated.
  const absl::string_view newMode = mode;
  if (newMode == controlMode_) return;
  noPolicyReported_ = false;
  if (newMode == control_mode::kSafety) {
    // The posture to hold and the clock origin are captured on the first cycle
    // in the mode. A pending hold for WB_MPC is abandoned: SAFETY must not be
    // held off waiting for a solver.
    safetyDecayStartTime_.reset();
    awaitingPostResetPolicy_.store(false);
  }
  if (control_mode::isPassive(controlMode_) && control_mode::isMpc(newMode)) {
    // The MPC starts again from the robot as it is now, and nothing solved
    // before reaches the robot. JOINT_PD and GRAVITY_COMP are held as they are;
    // ZERO_TORQUE (no posture hold) and SAFETY (a hold that decays to zero
    // torque) are held like JOINT_PD.
    requestMpcResetAndHold(controlMode_ == control_mode::kGravityComp);
  }
  controlMode_.assign(mode.data(), mode.size());
}

void WBMpcMrtJointController::requestMpcResetAndHold(bool holdGravityComp) {
  // The hold is armed before the reset is requested, so that no cycle sees the
  // request without the hold.
  armHold(holdGravityComp);
  requestMpcReset();
}

void WBMpcMrtJointController::armHold(bool holdGravityComp) {
  holdGravityComp_.store(holdGravityComp);
  awaitingPostResetPolicy_.store(true);
}

void WBMpcMrtJointController::setSafetyDecayTimeConstant(scalar_t seconds) {
  safetyDecayTimeConstant_ = std::max(safety_decay::kMinTimeConstant, seconds);
}

void WBMpcMrtJointController::handleClockRewind(scalar_t rewind) {
  postEvent(ControllerEventCode::kClockRewind, rewind,
            currentMpcObservation_.time);
  // SAFETY keeps the time it has already decayed for: restarting its clock
  // would put the gains back to full authority. A rewind past the entry shifts
  // the start time below zero, which is a valid start time on the new clock.
  if (safetyDecayStartTime_.has_value()) *safetyDecayStartTime_ -= rewind;
  previousObservationTime_ = currentMpcObservation_.time;
  // MpcResetSupervisor::observeTime() has requested the reset; the hold is all
  // that is left to arm.
  armHold(/*holdGravityComp=*/false);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

const vector_t& WBMpcMrtJointController::computeGravityCompensation(
    const ::robot::model::RobotState& robotState) {
  // In the workspaces, sized by the constructor: the Pinocchio configuration q
  // of the measured robot, as updateMpcState() writes it into the head of the
  // MPC state.
  const quaternion_t rootOrientation = robotState.getQuatBaseRtWorld();
  robotState.getJointPositions(mpcJointIndices_, mpcJointPositions_);
  mpcRobotModel_.fillGeneralizedCoordinatesFromMeasurement(
      robotState.getPosBaseRtWorldEwrtWorld(), rootOrientation.coeffs(),
      mpcJointPositions_, gravityCoordinates_);
  // The joint rows of g(q): nle(q, 0) of the model, without its allocations on
  // the composite root.
  return rigidBodyDynamics_->gravityJointTorques(gravityCoordinates_);
}

scalar_t WBMpcMrtJointController::nominalJointPosition(
    const ::robot::model::RobotState& robotState, size_t index) const {
  return index < nominalJointPositions_.size()
             ? nominalJointPositions_[index]
             : robotState.getJointPosition(index);
}

scalar_t WBMpcMrtJointController::clampedMpcJointFeedforward(
    size_t mpcJoint, scalar_t torque) const {
  return std::clamp(torque, -pdGains_.mpcJointTorqueLimit[mpcJoint],
                    pdGains_.mpcJointTorqueLimit[mpcJoint]);
}

void WBMpcMrtJointController::fillOtherJointsHold(
    const ::robot::model::RobotState& robotState,
    scalar_t kpScale,
    ::robot::model::RobotJointAction& robotJointAction) {
  for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
    const size_t index = otherJointIndices_[i];
    robot::model::JointAction& action =
        jointActionUnchecked(robotJointAction, index);
    action.q_des = nominalJointPosition(robotState, index);
    action.qd_des = 0.0;
    action.kp = pdGains_.otherJointKp[i] * kpScale;
    action.kd = pdGains_.otherJointKd[i];
    action.feed_forward_effort = 0.0;
  }
}

void WBMpcMrtJointController::reportPolicyUpdateLockContention() {
  // Once per run of failed try-locks: the count rises by one per cycle while
  // the run lasts.
  const MRT_BASE& mrt = mpcLink_->mrt();
  if (mrt.getConsecutiveFailedPolicyUpdateLocks() ==
      MRT_BASE::kPolicyUpdateLockContentionThreshold + 1) {
    postEvent(
        ControllerEventCode::kPolicyUpdateLockContended,
        static_cast<scalar_t>(mrt.getConsecutiveFailedPolicyUpdateLocks()),
        static_cast<scalar_t>(mrt.getFailedPolicyUpdateLocks()));
  }
}

void WBMpcMrtJointController::fillZeroTorqueAction(
    const ::robot::model::RobotState& robotState,
    ::robot::model::RobotJointAction& robotJointAction) {
  const std::array<const std::vector<size_t>* absl_nonnull, kNumMrtJointGroups>
      jointGroups{&mpcJointIndices_, &otherJointIndices_};
  for (const std::vector<size_t>* absl_nonnull indices : jointGroups) {
    for (size_t index : *indices) {
      robot::model::JointAction& action =
          jointActionUnchecked(robotJointAction, index);
      action.q_des = robotState.getJointPosition(index);
      action.qd_des = 0.0;
      action.kp = 0.0;
      action.kd = 0.0;
      action.feed_forward_effort = 0.0;
    }
  }
}

void WBMpcMrtJointController::fillJointPdAction(
    const ::robot::model::RobotState& robotState,
    ::robot::model::RobotJointAction& robotJointAction) {
  const vector_t& gravityTorques = computeGravityCompensation(robotState);
  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    const size_t index = mpcJointIndices_[i];
    robot::model::JointAction& action =
        jointActionUnchecked(robotJointAction, index);
    action.q_des = nominalJointPosition(robotState, index);
    action.qd_des = 0.0;
    action.kp = pdGains_.mpcJointKp[i];
    action.kd = pdGains_.mpcJointKd[i];
    action.feed_forward_effort =
        clampedMpcJointFeedforward(i, gravityTorques[i]);
  }
  fillOtherJointsHold(robotState, /*kpScale=*/1.0, robotJointAction);
}

void WBMpcMrtJointController::fillGravityCompAction(
    const ::robot::model::RobotState& robotState,
    ::robot::model::RobotJointAction& robotJointAction) {
  // The base-held gravity torques: GRAVITY_COMP is operated with the robot
  // suspended from the gantry (see the centroidal controller's
  // fillGravityCompAction for why this is not enough for a robot bearing its
  // own weight).
  const vector_t& gravityTorques = computeGravityCompensation(robotState);
  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    const size_t index = mpcJointIndices_[i];
    robot::model::JointAction& action =
        jointActionUnchecked(robotJointAction, index);
    action.q_des = robotState.getJointPosition(index);
    action.qd_des = 0.0;
    action.kp = 0.0;
    action.kd = pdGains_.mpcJointKd[i] * pdGains_.gravityCompDampingFraction;
    action.feed_forward_effort =
        clampedMpcJointFeedforward(i, gravityTorques[i]);
  }
  // The other joints keep their posture with half their position gain.
  fillOtherJointsHold(robotState, kGravityCompOtherJointKpScale,
                      robotJointAction);
}

void WBMpcMrtJointController::fillSafetyAction(
    const ::robot::model::RobotState& robotState,
    ::robot::model::RobotJointAction& robotJointAction) {
  // The posture held is the MEASURED one at entry: SAFETY is entered when
  // something has already gone wrong, and a return to the nominal stance at
  // full gain would be a lunge, not a safe stop.
  if (!safetyDecayStartTime_.has_value()) {
    // Sized by the constructor: no allocation.
    for (size_t i = 0; i < mpcJointIndices_.size(); ++i)
      safetyHoldMpcJointPositions_[i] =
          robotState.getJointPosition(mpcJointIndices_[i]);
    for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
      safetyHoldOtherJointPositions_[i] =
          robotState.getJointPosition(otherJointIndices_[i]);
    }
    safetyDecayStartTime_ = currentMpcObservation_.time;
    safetyDecayLatchedFactor_ = 1.0;
    postEvent(ControllerEventCode::kSafetyEntered, safetyDecayTimeConstant_);
  }
  // Captured above: value_or() only spares the reader a second check. Latched:
  // alpha never rises within this stay in SAFETY
  // (safety_decay::latchedFactor()).
  const scalar_t startTime =
      safetyDecayStartTime_.value_or(currentMpcObservation_.time);
  const scalar_t alpha = safety_decay::latchedFactor(
      safetyDecayLatchedFactor_, currentMpcObservation_.time - startTime,
      safetyDecayTimeConstant_);
  safetyDecayLatchedFactor_ = alpha;
  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    robot::model::JointAction& action =
        jointActionUnchecked(robotJointAction, mpcJointIndices_[i]);
    action.q_des = safetyHoldMpcJointPositions_[i];
    action.qd_des = 0.0;
    action.kp = alpha * pdGains_.mpcJointKp[i];
    action.kd = alpha * pdGains_.mpcJointKd[i];
    action.feed_forward_effort = 0.0;
  }
  for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
    robot::model::JointAction& action =
        jointActionUnchecked(robotJointAction, otherJointIndices_[i]);
    action.q_des = safetyHoldOtherJointPositions_[i];
    action.qd_des = 0.0;
    action.kp = alpha * pdGains_.otherJointKp[i];
    action.kd = alpha * pdGains_.otherJointKd[i];
    action.feed_forward_effort = 0.0;
  }
}

void WBMpcMrtJointController::fillHoldAction(
    const ::robot::model::RobotState& robotState,
    ::robot::model::RobotJointAction& robotJointAction) {
  if (holdGravityComp_.load()) {
    fillGravityCompAction(robotState, robotJointAction);
  } else {
    fillJointPdAction(robotState, robotJointAction);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::computeJointControlAction(
    scalar_t /*time*/,
    const ::robot::model::RobotState& robotState,
    ::robot::model::RobotJointAction& robotJointAction) {
  // The newest PD gains posted by setPdGains() or pollPdGainsFile(), parsed on
  // their callers' threads: a copy between preallocated vectors, without a
  // lock.
  pdGainsMailbox_.receive(pdGains_);

  // Set observation to MPC, in every mode, so that the solver keeps tracking
  // the robot and its clock.
  updateMpcObservation(currentMpcObservation_, robotState);
  const scalar_t clockRewind =
      mpcLink_->observeTime(currentMpcObservation_.time);
  if (clockRewind > 0.0) handleClockRewind(clockRewind);
  mpcLink_->setCurrentObservation(currentMpcObservation_);

  // The passive modes: independent of the MPC and of the solver.
  if (control_mode::isPassive(controlMode_)) {
    if (controlMode_ == control_mode::kZeroJointTorque) {
      fillZeroTorqueAction(robotState, robotJointAction);
    } else if (controlMode_ == control_mode::kJointPd) {
      fillJointPdAction(robotState, robotJointAction);
    } else if (controlMode_ == control_mode::kGravityComp) {
      fillGravityCompAction(robotState, robotJointAction);
    } else {
      fillSafetyAction(robotState, robotJointAction);
    }
    previousObservationTime_ = currentMpcObservation_.time;
    return;
  }

  // WB_MPC. The policy in use counts as post-reset once it was solved after the
  // last reset the solver thread served and no reset is outstanding: the two
  // checks together are race-free (MpcResetSupervisor).
  mpcLink_->updatePolicy();
  reportPolicyUpdateLockContention();
  const bool postResetPolicyActive =
      mpcLink_->isActivePolicyCurrent() && !mpcLink_->hasOutstandingReset();
  policyActivated_.store(postResetPolicyActive);

  // A solver that keeps failing leaves a stale policy in use: hold the robot
  // with the JOINT_PD action instead.
  if (!mpcLink_->isHealthy() && !awaitingPostResetPolicy_.load()) {
    holdGravityComp_.store(false);
    awaitingPostResetPolicy_.store(true);
  }
  bool handedOverThisCycle = false;
  if (awaitingPostResetPolicy_.load()) {
    if (!postResetPolicyActive || !mpcLink_->isHealthy()) {
      fillHoldAction(robotState, robotJointAction);
      previousObservationTime_ = currentMpcObservation_.time;
      return;
    }
    awaitingPostResetPolicy_.store(false);
    handedOverThisCycle = true;
  }

  // Sized once for the model, so that evaluating the policy into them allocates
  // nothing.
  vector_t& mpcPolicyState = mpcPolicyState_;
  vector_t& mpcPolicyInput = mpcPolicyInput_;
  size_t mpcPolicyMode = 0;

  if (mpcLink_->initialPolicyReceived()) {
    // Compute actual sim dt from elapsed simulation time (respects RTF)
    scalar_t simDt = currentMpcObservation_.time - previousObservationTime_;
    // Clamp to sane range: avoid zero/negative (first call, time resets) and
    // excessive lookahead
    simDt = std::clamp(simDt, kMinPolicyLookahead, kMaxPolicyLookahead);

    // The policy, without MRT_BASE::evaluatePolicy()'s heap allocations: the
    // plan, the joint targets and the feedforward one lookahead ahead (simDt,
    // about the control period the command is in force), the feedback where the
    // measurement is (LookaheadPolicyEvaluator). A policy the evaluator refuses
    // is not left to MRT_BASE::evaluatePolicy(), which allocates, logs and
    // aborts on a malformed policy: the robot is held as for a solver that
    // keeps failing - the entry's hold on the cycle that handed over to the
    // policy, the JOINT_PD action otherwise - until a policy the evaluator
    // takes is in use.
    const LookaheadPolicyEvaluator::Outcome outcome = policyEvaluator_.evaluate(
        mpcLink_->getPolicy(), currentMpcObservation_.time, simDt,
        currentMpcObservation_.state, mpcPolicyState, mpcPolicyInput,
        mpcPolicyMode);
    if (outcome != LookaheadPolicyEvaluator::Outcome::kEvaluated) {
      if (!refusedPolicyReported_) {
        const PolicyRefusal refusal =
            outcome == LookaheadPolicyEvaluator::Outcome::kUnsupportedController
                ? PolicyRefusal::kUnsupportedController
                : PolicyRefusal::kMismatchedPolicy;
        postEvent(ControllerEventCode::kPolicyRefused,
                  static_cast<scalar_t>(refusal));
        refusedPolicyReported_ = true;
      }
      if (!handedOverThisCycle) holdGravityComp_.store(false);
      awaitingPostResetPolicy_.store(true);
      fillHoldAction(robotState, robotJointAction);
      previousObservationTime_ = currentMpcObservation_.time;
      return;
    }
    refusedPolicyReported_ = false;
    latestPolicyInput_ = mpcPolicyInput;
    executePolicy(mpcPolicyState, mpcPolicyInput, robotJointAction);
  }

  else {
    if (!noPolicyReported_) {
      postEvent(ControllerEventCode::kNoPolicyWeightCompensation);
      noPolicyReported_ = true;
    }
    fillWeightCompensationAction(robotState, mpcPolicyState, mpcPolicyInput,
                                 robotJointAction);
  }

  // The joints outside the MPC hold the nominal posture as in JOINT_PD, so that
  // the hand-over between the modes does not move them (they used to be sent to
  // 0 here).
  fillOtherJointsHold(robotState, /*kpScale=*/1.0, robotJointAction);

  // Track observation time for next call's dt computation
  previousObservationTime_ = currentMpcObservation_.time;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::executePolicy(
    const vector_t& mpcPolicyState,
    const vector_t& mpcPolicyInput,
    ::robot::model::RobotJointAction& robotJointAction) {
  // The policy carries a wrench wherever its own schedule expects contact.
  // Whether a foot can actually transmit it is decided by the measured contact
  // state of this cycle, not by the plan: the wrench of a foot that is not
  // touching is dropped, and after touch-down it is debounced and ramped in as
  // configured (ContactWrenchGate). World-frame wrenches for the
  // LOCAL_WORLD_ALIGNED Jacobians; the whole-body input carries them as they
  // are (fixed-size copies).
  const std::array<vector6_t, kNumContacts> footWrenches =
      contactWrenchGate_.apply(
          {mpcRobotModel_.getWrenchContactEwrtWorld(
               mpcPolicyState, mpcPolicyInput, /*contactIndex=*/0),
           mpcRobotModel_.getWrenchContactEwrtWorld(
               mpcPolicyState, mpcPolicyInput, /*contactIndex=*/1)});
  // The inverse dynamics of the planned state [q, v] and joint accelerations,
  // on views of the policy's state and input and on preallocated workspaces: q
  // with its quaternion safely normalized, as getGeneralizedCoordinates()
  // returns it.
  const Eigen::Index velocityDim =
      static_cast<Eigen::Index>(mpcRobotModel_.getVelocityDim());
  const Eigen::Index jointDim =
      static_cast<Eigen::Index>(mpcRobotModel_.getJointDim());
  fillPolicyCoordinates(mpcPolicyState);
  const vector_t& mpcJointTorques = rigidBodyDynamics_->freeBaseJointTorques(
      policyCoordinates_, mpcPolicyState.tail(velocityDim),
      mpcPolicyInput.tail(jointDim), footWrenches);
  // [q, v]: the joint angles follow the base pose in q, the joint velocities
  // end v.
  const Eigen::Index jointStart =
      static_cast<Eigen::Index>(mpcRobotModel_.getJointStartindex());
  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    const Eigen::Index joint = static_cast<Eigen::Index>(i);
    robot::model::JointAction& action =
        jointActionUnchecked(robotJointAction, mpcJointIndices_[i]);
    action.q_des = mpcPolicyState[jointStart + joint];
    action.qd_des = mpcPolicyState[mpcPolicyState.size() - jointDim + joint];
    action.kp = pdGains_.mpcJointKp[i];
    action.kd = pdGains_.mpcJointKd[i];
    action.feed_forward_effort =
        clampedMpcJointFeedforward(i, mpcJointTorques[joint]);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::fillPolicyCoordinates(const vector_t& state) {
  const Eigen::Index configurationDim =
      static_cast<Eigen::Index>(mpcRobotModel_.getConfigurationDim());
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(mpcRobotModel_.getBaseOrientationStartIndex());
  policyCoordinates_ = state.head(configurationDim);
  policyCoordinates_.segment<kQuaternionCoefficients>(orientationStart) =
      safelyNormalizedQuaternion(
          vector4_t(state.segment<kQuaternionCoefficients>(orientationStart)));
}

void WBMpcMrtJointController::fillWeightCompensationAction(
    const ::robot::model::RobotState& robotState,
    vector_t& mpcPolicyState,
    vector_t& mpcPolicyInput,
    ::robot::model::RobotJointAction& robotJointAction) {
  // The weight-compensating input around the current state
  // (weightCompensatingInput(), into the workspace): the weight shared evenly
  // by the feet measured in contact, as vertical world-frame forces. With none
  // (the robot hangs on the gantry) no contact force is compensated and the
  // feedforward is the gravity and Coriolis term of the free legs, the base
  // being held (computeBaseHeldJointTorques(); the free-base inverse dynamics
  // would let the base fall freely).
  mpcPolicyState = currentMpcObservation_.state;
  mpcPolicyInput.setZero();
  const size_t numStanceLegs = numberOfLegsInContacts(measuredContactFlags_);
  feet_array_t<vector6_t> footWrenches;
  for (size_t i = 0; i < kNumContacts; ++i) {
    if (numStanceLegs > 0 && measuredContactFlags_[i]) {
      mpcRobotModel_.setForceContactEwrtParam(
          mpcPolicyInput,
          vector3_t(0.0, 0.0,
                    robotWeight_ / static_cast<scalar_t>(numStanceLegs)),
          i);
    }
    footWrenches[i] = mpcRobotModel_.getWrenchContactEwrtWorld(
        mpcPolicyState, mpcPolicyInput, i);
  }
  latestPolicyInput_ = mpcPolicyInput;
  fillPolicyCoordinates(mpcPolicyState);
  const Eigen::Index velocityDim =
      static_cast<Eigen::Index>(mpcRobotModel_.getVelocityDim());
  const Eigen::Index jointDim =
      static_cast<Eigen::Index>(mpcRobotModel_.getJointDim());
  const vector_t& weightCompensatingTorques =
      rigidBodyDynamics_->heldBaseJointTorques(
          policyCoordinates_, mpcPolicyState.tail(velocityDim),
          mpcPolicyInput.tail(jointDim), footWrenches);

  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    const size_t index = mpcJointIndices_[i];
    robot::model::JointAction& action =
        jointActionUnchecked(robotJointAction, index);
    action.q_des = robotState.getJointPosition(index);
    action.qd_des = 0.0;
    action.kp = pdGains_.mpcJointKp[i];
    action.kd = pdGains_.mpcJointKd[i];
    action.feed_forward_effort = clampedMpcJointFeedforward(
        i, weightCompensatingTorques[static_cast<Eigen::Index>(i)]);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

TargetTrajectories WBMpcMrtJointController::currentObservationToResetTrajectory(
    const SystemObservation& currentObservation) {
  // The one definition of the reset target, which the MPC node serves its
  // resets from as well.
  const TargetTrajectories resetTargetTrajectories =
      wbMpcResetTargetTrajectories(currentObservation, mpcRobotModel_,
                                   pinocchioInterface_);

  if (mpcLink_->isHealthy()) {
    LOG(INFO) << "Resetting MPC to current state: base pose "
              << mpcRobotModel_
                     .getPoseeulerzyxBaseRtWorld(
                         resetTargetTrajectories.stateTrajectory.front())
                     .transpose();
  }
  return resetTargetTrajectories;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::optional<contact_flag_t> WBMpcMrtJointController::getPlannedContactFlags(
    scalar_t time) const {
  if (!policyActivated_.load()) return std::nullopt;
  return modeNumber2StanceLeg(
      mpcLink_->getPolicy().modeSchedule_.modeAtTime(time));
}

}  // namespace ocs2::humanoid
