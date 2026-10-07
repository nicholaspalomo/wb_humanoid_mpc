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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "ocs2_robotic_tools/common/RotationDerivativesTransforms.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/rnea.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/robot/JointPdGainsFromConfig.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/ControlMode.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/mrt/JointActionAccess.h"
#include "humanoid_common_mpc/mrt/LoggingControllerEventSink.h"
#include "humanoid_common_mpc/mrt/SafetyDecay.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"
#include "humanoid_wb_mpc/mrt/WBMpcResetTarget.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotStateContactEstimator.h"

namespace ocs2::humanoid {

namespace {

/** The modes a policy may carry: the contact flags of the two feet, FLY to STANCE (MotionPhaseDefinition.h). */
constexpr size_t kNumPolicyModes = static_cast<size_t>(ModeNumber::kStance) + 1;

/** The class name the events carry. */
constexpr char kControllerName[] = "WBMpcMrtJointController";

/** The message of Create() for a link factory that made no link. */
constexpr char kNoMpcLinkMessage[] = "[WBMpcMrtJointController] the MPC link factory made no link.";

/** The link of an MPC in this process, as both in-process overloads make it. */
MpcLinkFactory inProcessMpcLinkFactory(MPC_BASE& mpc, scalar_t mpcDesiredFrequency) {
  InProcessMpcLink::Config config;
  config.mpcDesiredFrequency = mpcDesiredFrequency;
  config.solverThreadName = "WB MPC Solver Thread";
  return InProcessMpcLink::factory(mpc, std::move(config));
}

}  // namespace

absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> WBMpcMrtJointController::Create(
    const ::robot::model::RobotDescription& robotDescription,
    const ModelSettings& modelSettings,
    MPC_BASE& mpc,
    PinocchioInterface pinocchioInterface,
    scalar_t mpcDesiredFrequency,
    const std::string& pdGainsFile) {
  return Create(robotDescription, modelSettings, inProcessMpcLinkFactory(mpc, mpcDesiredFrequency), std::move(pinocchioInterface),
                pdGainsFile);
}

absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> WBMpcMrtJointController::Create(
    const ::robot::model::RobotDescription& robotDescription,
    const ModelSettings& modelSettings,
    const MpcLinkFactory& mpcLinkFactory,
    PinocchioInterface pinocchioInterface,
    const std::string& pdGainsFile) {
  // The joints the control cycle indexes without a check: refused here, by name, rather than by the constructor's check.
  RETURN_IF_ERROR(robotDescription.findJointIndices(modelSettings.mpcModelJointNames).status());
  RETURN_IF_ERROR(robotDescription.findJointIndices(modelSettings.fixedJointNames).status());
  ASSIGN_OR_RETURN(InitialPdGains initialPdGains, loadInitialPdGains(modelSettings, pdGainsFile));
  std::unique_ptr<WBMpcMrtJointController> controller = absl::WrapUnique(new WBMpcMrtJointController(
      robotDescription, modelSettings, mpcLinkFactory, std::move(pinocchioInterface), pdGainsFile, std::move(initialPdGains)));
  if (controller->mpcLink_ == nullptr) {
    return absl::InvalidArgumentError(kNoMpcLinkMessage);
  }
  return controller;
}

WBMpcMrtJointController::WBMpcMrtJointController(const ::robot::model::RobotDescription& robotDescription,
                                                 const ModelSettings& modelSettings,
                                                 const MpcLinkFactory& mpcLinkFactory,
                                                 PinocchioInterface pinocchioInterface,
                                                 const std::string& pdGainsFile,
                                                 InitialPdGains initialPdGains)
    : mpcLink_(mpcLinkFactory([this](const SystemObservation& observation) { return currentObservationToResetTrajectory(observation); })),
      contactEstimator_(std::make_shared<::robot::model::RobotStateContactEstimator>()),
      contactEstimateIntake_(kControllerName),
      pinocchioInterface_(std::move(pinocchioInterface)),
      mpcRobotModel_(modelSettings),
      eventSink_(&LoggingControllerEventSink::instance()),
      policyEvaluator_(ipc::ModelDimensions{
          .stateDim = mpcRobotModel_.getStateDim(), .inputDim = mpcRobotModel_.getInputDim(), .numModes = kNumPolicyModes}),
      mpcPolicyState_(vector_t::Zero(mpcRobotModel_.getStateDim())),
      mpcPolicyInput_(vector_t::Zero(mpcRobotModel_.getInputDim())),
      pdGainsFile_(pdGainsFile),
      robotJointNames_(robotDescription.getJointNames()),
      modelSettings_(modelSettings),
      pdGainsLastWriteTime_(initialPdGains.fileWriteTime),
      pdGains_(std::move(initialPdGains.gains)),
      pdGainsMailbox_(pdGains_) {
  mpcJointIndices_ = robotDescription.getJointIndices(modelSettings.mpcModelJointNames);
  otherJointIndices_ = robotDescription.getJointIndices(modelSettings.fixedJointNames);
  // Once, here, what the control cycle then takes for granted (jointActionUnchecked()): every joint it writes carries an
  // action in a RobotJointAction of the robot description.
  checkJointIndices(robotDescription, mpcJointIndices_, kControllerName);
  checkJointIndices(robotDescription, otherJointIndices_, kControllerName);
  currentMpcObservation_.state = vector_t::Zero(mpcRobotModel_.getStateDim());
  currentMpcObservation_.input = vector_t::Zero(mpcRobotModel_.getInputDim());
  latestPolicyInput_ = vector_t::Zero(mpcRobotModel_.getInputDim());

  // The control thread's workspaces, at the sizes it writes them with.
  mpcJointPositions_ = vector_t::Zero(mpcJointIndices_.size());
  mpcJointVelocities_ = vector_t::Zero(mpcJointIndices_.size());
  zeroInput_ = vector_t::Zero(mpcRobotModel_.getInputDim());
  estimatedContactFlags_.assign(kNumContacts, true);
  gravityState_ = vector_t::Zero(mpcRobotModel_.getStateDim());
  gravityCoordinates_ = vector_t::Zero(6 + mpcRobotModel_.getJointDim());
  zeroGeneralizedVelocity_ = vector_t::Zero(pinocchioInterface_.getModel().nv);
  gravityTorques_ = vector_t::Zero(mpcRobotModel_.getJointDim());
  safetyHoldMpcJointPositions_ = vector_t::Zero(mpcJointIndices_.size());
  safetyHoldOtherJointPositions_ = vector_t::Zero(otherJointIndices_.size());
  nominalJointPositions_.reserve(robotDescription.getNumJoints());
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

JointPdGainsDefaults WBMpcMrtJointController::pdGainsDefaults() {
  JointPdGainsDefaults defaults;
  defaults.kp = 150.0;
  defaults.kd = 8.0;
  // The whole-body controller commands no torque limit, so the documents' torque_limit keys are not read at all.
  defaults.torqueLimit = std::nullopt;
  return defaults;
}

absl::StatusOr<WBMpcMrtJointController::InitialPdGains> WBMpcMrtJointController::loadInitialPdGains(const ModelSettings& modelSettings,
                                                                                                    const std::string& pdGainsFile) {
  // The write time first, then the gains: a save that lands between the two is reloaded by the next poll.
  InitialPdGains initial{.fileWriteTime = jointPdGainsFileWriteTime(pdGainsFile),
                         .gains = defaultJointPdGains(pdGainsDefaults(), modelSettings.mpcModelJointNames, modelSettings.fixedJointNames)};
  absl::StatusOr<std::optional<JointPdGains>> loaded =
      loadJointPdGains(pdGainsFile, pdGainsDefaults(), modelSettings.mpcModelJointNames, modelSettings.fixedJointNames);
  if (!loaded.ok()) {
    // At start-up there are no gains in use to keep, and the hard-coded defaults are nobody's choice for this robot.
    return absl::InvalidArgumentError(absl::StrCat("[WBMpcMrtJointController] The joint PD gains file ", pdGainsFile,
                                                   " was refused, so the controller does not start: ", loaded.status().message()));
  }
  if (!loaded->has_value()) {
    return initial;
  }
  initial.gains = **std::move(loaded);
  LOG(INFO) << "[WBMpcMrtJointController] Loaded joint PD gains from " << pdGainsFile;
  return initial;
}

absl::Status WBMpcMrtJointController::postPdGains(const absl::StatusOr<JointPdGains>& gains, absl::string_view source, uint64_t ticket) {
  if (!gains.ok()) {
    LOG(WARNING) << "[WBMpcMrtJointController] Warning: Refused the PD gains from " << source
                 << "; the gains in use are kept: " << gains.status().message();
    return gains.status();
  }
  LOG(INFO) << "[WBMpcMrtJointController] Loaded joint PD gains from " << source;
  return pdGainsMailbox_.post(*gains, ticket);
}

absl::Status WBMpcMrtJointController::setPdGains(const mpc_config::JointPdGainsFile& gains) {
  // Its place in line before anything else: gains handed in after these win, however fast they are resolved.
  const uint64_t ticket = pdGainsMailbox_.takeTicket();
  LOG(INFO) << "[WBMpcMrtJointController] setPdGains received gains for " << gains.joint_gains.size() << " named joints";
  RETURN_IF_ERROR(
      postPdGains(jointPdGainsFromConfig(gains, pdGainsDefaults(), modelSettings_.mpcModelJointNames, modelSettings_.fixedJointNames),
                  "setPdGains", ticket));
  LOG(INFO) << "[WBMpcMrtJointController] PD gains from setPdGains are applied at the next control cycle.";
  return absl::OkStatus();
}

void WBMpcMrtJointController::pollPdGainsFile() {
  if (pdGainsFile_.empty()) return;
  absl::MutexLock lock(&pdGainsFileMutex_);
  std::error_code ec;
  const std::filesystem::file_time_type last_write = std::filesystem::last_write_time(pdGainsFile_, ec);
  if (ec || last_write == pdGainsLastWriteTime_) return;
  pdGainsLastWriteTime_ = last_write;
  const uint64_t ticket = pdGainsMailbox_.takeTicket();
  absl::StatusOr<std::optional<JointPdGains>> loaded =
      loadJointPdGains(pdGainsFile_, pdGainsDefaults(), modelSettings_.mpcModelJointNames, modelSettings_.fixedJointNames);
  // A refused file is logged by postPdGains() and leaves the gains in use.
  if (!loaded.ok()) {
    postPdGains(loaded.status(), pdGainsFile_, ticket).IgnoreError();
    return;
  }
  if (!loaded->has_value()) {
    return;  // gone again (an editor that saves by renaming); the next poll sees the new file
  }
  postPdGains(**std::move(loaded), pdGainsFile_, ticket).IgnoreError();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

WBMpcMrtJointController::~WBMpcMrtJointController() {
  // Stop the solver before anything its reset target reads is destroyed. Its thread used to loop on `while (true)` and
  // ignore the stop flag, so the join never returned and a controller could not be destroyed.
  if (mpcLink_ != nullptr) mpcLink_->stop();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::startMpcThread(const ::robot::model::RobotState& initRobotState) {
  updateMpcObservation(currentMpcObservation_, initRobotState);
  // Set observation to MPC and start serving policies.
  mpcLink_->start(currentMpcObservation_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::updateMpcState(vector_t& mpcState, const ::robot::model::RobotState& robotState) {
  mpcRobotModel_.setBasePosition(mpcState, robotState.getRootPositionInWorldFrame());
  mpcRobotModel_.setBaseOrientationEulerZYX(mpcState, quaternionToEulerZYX(robotState.getRootRotationLocalToWorldFrame()));

  robotState.getJointPositions(mpcJointIndices_, mpcJointPositions_);
  mpcRobotModel_.setJointAngles(mpcState, mpcJointPositions_);

  // currently we send local angular and linear velocity
  mpcRobotModel_.setBaseLinearVelocity(mpcState,
                                       robotState.getRootRotationLocalToWorldFrame() * robotState.getRootLinearVelocityInLocalFrame());
  mpcRobotModel_.setBaseOrientationEulerZYXDerivatives(
      mpcState, getEulerAnglesZyxDerivativesFromLocalAngularVelocity<scalar_t>(mpcRobotModel_.getBaseOrientationEulerZYX(mpcState),
                                                                               robotState.getRootAngularVelocityInLocalFrame()));

  zeroInput_.setZero();
  robotState.getJointVelocities(mpcJointIndices_, mpcJointVelocities_);
  mpcRobotModel_.setJointVelocities(mpcState, zeroInput_, mpcJointVelocities_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::updateMpcObservation(ocs2::SystemObservation& mpcObservation, const ::robot::model::RobotState& robotState) {
  updateMpcState(mpcObservation.state, robotState);
  mpcObservation.time = robotState.getTime();
  mpcObservation.input = vector_t::Zero(mpcRobotModel_.getInputDim());  // Add contact forces later.
  // The measured contact state of this cycle: the observation mode of the MPC, and the gate of the contact wrenches in
  // the inverse dynamics (computeJointControlAction).
  contactEstimator_->estimateContactFlags(robotState, estimatedContactFlags_);
  // An estimate without one flag per contact point is refused and the measured contact state stays the last one; the
  // first refusal of a run is reported through the event sink (ContactEstimateIntake).
  contactEstimateIntake_.take(estimatedContactFlags_, measuredContactFlags_, *eventSink_);
  mpcObservation.mode = stanceLeg2ModeNumber(measuredContactFlags_);
  contactWrenchGate_.update(mpcObservation.time, measuredContactFlags_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::setContactWrenchGateConfig(const ContactWrenchGate::Config& config) {
  if (contactWrenchGate_.setConfig(config)) {
    postEvent(ControllerEventCode::kContactWrenchGateChanged, config.debounceTime, config.rampTime);
  } else {
    postEvent(ControllerEventCode::kContactWrenchGateRefused, config.debounceTime, config.rampTime);
  }
}

void WBMpcMrtJointController::setEventSink(ControllerEventSink* absl_nullable eventSink) {
  eventSink_ = eventSink != nullptr ? eventSink : &LoggingControllerEventSink::instance();
}

void WBMpcMrtJointController::postEvent(ControllerEventCode code, scalar_t value0, scalar_t value1, absl::string_view text) {
  eventSink_->post(makeControllerEvent(code, kControllerName, value0, value1, text));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::setContactEstimator(std::shared_ptr<::robot::model::ContactEstimator> contactEstimator) {
  contactEstimator_ = contactEstimator ? std::move(contactEstimator) : std::make_shared<::robot::model::RobotStateContactEstimator>();
  const std::string name = contactEstimator_->getName();
  contactEstimateIntake_.resetEstimator(name);
  postEvent(ControllerEventCode::kContactEstimatorChanged, /*value0=*/0.0, /*value1=*/0.0, name);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::setControlMode(absl::string_view mode) {
  // Called every cycle; compared and assigned through string_views, so that nothing is allocated.
  const absl::string_view newMode = mode;
  if (newMode == controlMode_) return;
  noPolicyReported_ = false;
  if (newMode == control_mode::kSafety) {
    // The posture to hold and the clock origin are captured on the first cycle in the mode. A pending hold for WB_MPC
    // is abandoned: SAFETY must not be held off waiting for a solver.
    safetyDecayStartTime_ = -1.0;
    awaitingPostResetPolicy_.store(false);
  }
  if (control_mode::isPassive(controlMode_) && control_mode::isMpc(newMode)) {
    // The MPC starts again from the robot as it is now, and nothing solved before reaches the robot.
    requestMpcResetAndHold(controlMode_ == control_mode::kGravityComp);
  }
  controlMode_.assign(mode.data(), mode.size());
}

void WBMpcMrtJointController::requestMpcResetAndHold(bool holdGravityComp) {
  // The hold is armed before the reset is requested, so that no cycle sees the request without the hold.
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
  postEvent(ControllerEventCode::kClockRewind, rewind, currentMpcObservation_.time);
  // SAFETY keeps the time it has already decayed for: restarting its clock would put the gains back to full authority.
  if (safetyDecayStartTime_ >= 0.0) safetyDecayStartTime_ -= rewind;
  previousObservationTime_ = currentMpcObservation_.time;
  // MpcResetSupervisor::observeTime() has requested the reset; the hold is all that is left to arm.
  armHold(/*holdGravityComp=*/false);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

const vector_t& WBMpcMrtJointController::computeGravityCompensation(const ::robot::model::RobotState& robotState) {
  // In the workspaces, sized by the constructor: the generalized coordinates are the head of the MPC state
  // (WBAccelMpcRobotModel::getGeneralizedCoordinates()).
  gravityState_.setZero();
  updateMpcState(gravityState_, robotState);
  gravityCoordinates_ = gravityState_.head(gravityCoordinates_.size());
  const PinocchioInterface::Model& model = pinocchioInterface_.getModel();
  PinocchioInterface::Data& data = pinocchioInterface_.getData();
  pinocchio::nonLinearEffects(model, data, gravityCoordinates_, zeroGeneralizedVelocity_);
  gravityTorques_ = data.nle.tail(mpcRobotModel_.getJointDim());
  return gravityTorques_;
}

scalar_t WBMpcMrtJointController::nominalJointPosition(const ::robot::model::RobotState& robotState, size_t index) const {
  return index < nominalJointPositions_.size() ? nominalJointPositions_[index] : robotState.getJointPosition(index);
}

void WBMpcMrtJointController::fillZeroTorqueAction(const ::robot::model::RobotState& robotState,
                                                   ::robot::model::RobotJointAction& robotJointAction) {
  const std::array<const std::vector<size_t>* absl_nonnull, 2> jointGroups{&mpcJointIndices_, &otherJointIndices_};
  for (const std::vector<size_t>* absl_nonnull indices : jointGroups) {
    for (size_t index : *indices) {
      robot::model::JointAction& action = jointActionUnchecked(robotJointAction, index);
      action.q_des = robotState.getJointPosition(index);
      action.qd_des = 0.0;
      action.kp = 0.0;
      action.kd = 0.0;
      action.feed_forward_effort = 0.0;
    }
  }
}

void WBMpcMrtJointController::fillJointPdAction(const ::robot::model::RobotState& robotState,
                                                ::robot::model::RobotJointAction& robotJointAction) {
  const vector_t& gravityTorques = computeGravityCompensation(robotState);
  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    const size_t index = mpcJointIndices_[i];
    robot::model::JointAction& action = jointActionUnchecked(robotJointAction, index);
    action.q_des = nominalJointPosition(robotState, index);
    action.qd_des = 0.0;
    action.kp = pdGains_.mpcJointKp[i];
    action.kd = pdGains_.mpcJointKd[i];
    action.feed_forward_effort = gravityTorques[i];
  }
  for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
    const size_t index = otherJointIndices_[i];
    robot::model::JointAction& action = jointActionUnchecked(robotJointAction, index);
    action.q_des = nominalJointPosition(robotState, index);
    action.qd_des = 0.0;
    action.kp = pdGains_.otherJointKp[i];
    action.kd = pdGains_.otherJointKd[i];
    action.feed_forward_effort = 0.0;
  }
}

void WBMpcMrtJointController::fillGravityCompAction(const ::robot::model::RobotState& robotState,
                                                    ::robot::model::RobotJointAction& robotJointAction) {
  // The base-held gravity torques: GRAVITY_COMP is operated with the robot suspended from the gantry (see the centroidal
  // controller's fillGravityCompAction for why this is not enough for a robot bearing its own weight).
  const vector_t& gravityTorques = computeGravityCompensation(robotState);
  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    const size_t index = mpcJointIndices_[i];
    robot::model::JointAction& action = jointActionUnchecked(robotJointAction, index);
    action.q_des = robotState.getJointPosition(index);
    action.qd_des = 0.0;
    action.kp = 0.0;
    action.kd = pdGains_.mpcJointKd[i] * 0.2;  // Soft damping to prevent free-fall oscillation
    action.feed_forward_effort = gravityTorques[i];
  }
  for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
    const size_t index = otherJointIndices_[i];
    robot::model::JointAction& action = jointActionUnchecked(robotJointAction, index);
    action.q_des = nominalJointPosition(robotState, index);
    action.qd_des = 0.0;
    action.kp = pdGains_.otherJointKp[i] * 0.5;
    action.kd = pdGains_.otherJointKd[i];
    action.feed_forward_effort = 0.0;
  }
}

void WBMpcMrtJointController::fillSafetyAction(const ::robot::model::RobotState& robotState,
                                               ::robot::model::RobotJointAction& robotJointAction) {
  // The posture held is the MEASURED one at entry: SAFETY is entered when something has already gone wrong, and a
  // return to the nominal stance at full gain would be a lunge, not a safe stop.
  if (safetyDecayStartTime_ < 0.0) {
    // Sized by the constructor: no allocation.
    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) safetyHoldMpcJointPositions_[i] = robotState.getJointPosition(mpcJointIndices_[i]);
    for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
      safetyHoldOtherJointPositions_[i] = robotState.getJointPosition(otherJointIndices_[i]);
    }
    safetyDecayStartTime_ = currentMpcObservation_.time;
    postEvent(ControllerEventCode::kSafetyEntered, safetyDecayTimeConstant_);
  }
  const scalar_t alpha = safety_decay::factor(currentMpcObservation_.time - safetyDecayStartTime_, safetyDecayTimeConstant_);
  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    robot::model::JointAction& action = jointActionUnchecked(robotJointAction, mpcJointIndices_[i]);
    action.q_des = safetyHoldMpcJointPositions_[i];
    action.qd_des = 0.0;
    action.kp = alpha * pdGains_.mpcJointKp[i];
    action.kd = alpha * pdGains_.mpcJointKd[i];
    action.feed_forward_effort = 0.0;
  }
  for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
    robot::model::JointAction& action = jointActionUnchecked(robotJointAction, otherJointIndices_[i]);
    action.q_des = safetyHoldOtherJointPositions_[i];
    action.qd_des = 0.0;
    action.kp = alpha * pdGains_.otherJointKp[i];
    action.kd = alpha * pdGains_.otherJointKd[i];
    action.feed_forward_effort = 0.0;
  }
}

void WBMpcMrtJointController::fillHoldAction(const ::robot::model::RobotState& robotState,
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

void WBMpcMrtJointController::computeJointControlAction(scalar_t /*time*/,
                                                        const ::robot::model::RobotState& robotState,
                                                        ::robot::model::RobotJointAction& robotJointAction) {
  // The newest PD gains posted by setPdGains() or pollPdGainsFile(), parsed on their callers' threads: a copy
  // between preallocated vectors, without a lock.
  pdGainsMailbox_.receive(pdGains_);

  // Set observation to MPC, in every mode, so that the solver keeps tracking the robot and its clock.
  updateMpcObservation(currentMpcObservation_, robotState);
  const scalar_t clockRewind = mpcLink_->observeTime(currentMpcObservation_.time);
  if (clockRewind > 0.0) handleClockRewind(clockRewind);
  mpcLink_->setCurrentObservation(currentMpcObservation_);

  // The passive modes: independent of the MPC and of the solver.
  if (control_mode::isPassive(controlMode_)) {
    if (controlMode_ == control_mode::kZeroTorque) {
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

  // WB_MPC. The policy in use counts as post-reset once it was solved after the last reset the solver thread served and
  // no reset is outstanding: the two checks together are race-free (MpcResetSupervisor).
  mpcLink_->updatePolicy();
  const bool postResetPolicyActive = mpcLink_->isActivePolicyCurrent() && !mpcLink_->hasOutstandingReset();
  policyActivated_.store(postResetPolicyActive);

  // A solver that keeps failing leaves a stale policy in use: hold the robot with the JOINT_PD action instead.
  if (!mpcLink_->isHealthy() && !awaitingPostResetPolicy_.load()) {
    holdGravityComp_.store(false);
    awaitingPostResetPolicy_.store(true);
  }
  if (awaitingPostResetPolicy_.load()) {
    if (!postResetPolicyActive || !mpcLink_->isHealthy()) {
      fillHoldAction(robotState, robotJointAction);
      previousObservationTime_ = currentMpcObservation_.time;
      return;
    }
    awaitingPostResetPolicy_.store(false);
  }

  // Sized once for the model, so that evaluating the policy into them allocates nothing.
  vector_t& mpcPolicyState = mpcPolicyState_;
  vector_t& mpcPolicyInput = mpcPolicyInput_;
  size_t mpcPolicyMode = 0;

  if (mpcLink_->initialPolicyReceived()) {
    // Compute actual sim dt from elapsed simulation time (respects RTF)
    scalar_t simDt = currentMpcObservation_.time - previousObservationTime_;
    // Clamp to sane range: avoid zero/negative (first call, time resets) and excessive lookahead
    simDt = std::clamp(simDt, 0.001, 0.02);

    // Evaluate policy with feedback if activated in config: what MRT_BASE::evaluatePolicy() computes, without its heap
    // allocations. A policy the evaluator does not take (another controller type) goes through OCS2's own evaluation.
    const scalar_t evaluationTime = currentMpcObservation_.time + simDt;
    if (policyEvaluator_.evaluate(mpcLink_->getPolicy(), evaluationTime, currentMpcObservation_.state, mpcPolicyState, mpcPolicyInput,
                                  mpcPolicyMode) != ipc::RealtimePolicyEvaluator::Outcome::kEvaluated) {
      mpcLink_->evaluatePolicy(evaluationTime, currentMpcObservation_.state, mpcPolicyState, mpcPolicyInput, mpcPolicyMode);
    }
    latestPolicyInput_ = mpcPolicyInput;

    // The policy carries a wrench wherever its own schedule expects contact. Whether a foot can actually transmit it is
    // decided by the measured contact state of this cycle, not by the plan: the wrench of a foot that is not touching is
    // dropped, and after touch-down it is debounced and ramped in as configured (ContactWrenchGate). World-frame
    // wrenches for the LOCAL_WORLD_ALIGNED Jacobians.
    const std::array<vector6_t, 2> footWrenches =
        contactWrenchGate_.apply({mpcRobotModel_.getContactWrenchInWorldFrame(mpcPolicyState, mpcPolicyInput, /*contactIndex=*/0),
                                  mpcRobotModel_.getContactWrenchInWorldFrame(mpcPolicyState, mpcPolicyInput, /*contactIndex=*/1)});
    vector_t mpcJointTorques = computeJointTorques<scalar_t>(
        mpcRobotModel_.getGeneralizedCoordinates(mpcPolicyState), mpcRobotModel_.getGeneralizedVelocities(mpcPolicyState, mpcPolicyInput),
        mpcRobotModel_.getJointAccelerations(mpcPolicyInput), footWrenches, pinocchioInterface_);
    vector_t mpc_q_desired = mpcRobotModel_.getJointAngles(mpcPolicyState);
    vector_t mpc_qd_desired = mpcRobotModel_.getJointVelocities(mpcPolicyState, mpcPolicyInput);

    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
      size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action = jointActionUnchecked(robotJointAction, index);

      action.q_des = mpc_q_desired[i];
      action.qd_des = mpc_qd_desired[i];
      action.kp = pdGains_.mpcJointKp[i];
      action.kd = pdGains_.mpcJointKd[i];
      action.feed_forward_effort = mpcJointTorques[i];
    }
  }

  else {
    if (!noPolicyReported_) {
      postEvent(ControllerEventCode::kNoPolicyWeightCompensation);
      noPolicyReported_ = true;
    }
    //   Apply weight compensated input around current state
    mpcPolicyState = currentMpcObservation_.state;
    // The weight is carried by the feet measured in contact; with none (the robot hangs on the gantry) no contact force
    // is compensated and the feedforward is the gravity and Coriolis term of the free legs, the base being held
    // (computeBaseHeldJointTorques; computeJointTorques would let the base fall freely).
    mpcPolicyInput = weightCompensatingInput(pinocchioInterface_, measuredContactFlags_, mpcRobotModel_);
    latestPolicyInput_ = mpcPolicyInput;
    vector_t weightCompensatingTorques =
        computeBaseHeldJointTorques<scalar_t>(mpcPolicyState, mpcPolicyInput, pinocchioInterface_, mpcRobotModel_);

    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
      size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action = jointActionUnchecked(robotJointAction, index);

      action.q_des = robotState.getJointPosition(index);
      action.qd_des = 0.0;
      action.kp = pdGains_.mpcJointKp[i];
      action.kd = pdGains_.mpcJointKd[i];
      action.feed_forward_effort = weightCompensatingTorques[i];
    }
  }

  for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
    size_t index = otherJointIndices_[i];
    robot::model::JointAction& action = jointActionUnchecked(robotJointAction, index);

    action.q_des = 0;
    action.qd_des = 0;
    action.kp = pdGains_.otherJointKp[i];
    action.kd = pdGains_.otherJointKd[i];
    action.feed_forward_effort = 0.0;
  }

  // Track observation time for next call's dt computation
  previousObservationTime_ = currentMpcObservation_.time;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

TargetTrajectories WBMpcMrtJointController::currentObservationToResetTrajectory(const SystemObservation& currentObservation) {
  // The one definition of the reset target, which the MPC node serves its resets from as well.
  const TargetTrajectories resetTargetTrajectories = wbMpcResetTargetTrajectories(currentObservation, mpcRobotModel_, pinocchioInterface_);

  if (mpcLink_->isHealthy()) {
    LOG(INFO) << "Resetting MPC to current state: base pose "
              << mpcRobotModel_.getBasePose(resetTargetTrajectories.stateTrajectory.front()).transpose();
  }
  return resetTargetTrajectories;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::optional<contact_flag_t> WBMpcMrtJointController::getPlannedContactFlags(scalar_t time) const {
  if (!policyActivated_.load()) return std::nullopt;
  return modeNumber2StanceLeg(mpcLink_->getPolicy().modeSchedule_.modeAtTime(time));
}

}  // namespace ocs2::humanoid
