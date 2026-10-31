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

#include "pinocchio/fwd.hpp"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
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
#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcResetTarget.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/robot/JointPdGainsFromConfig.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/ControlMode.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/mrt/JointActionAccess.h"
#include "humanoid_common_mpc/mrt/JointActionBlend.h"
#include "humanoid_common_mpc/mrt/LoggingControllerEventSink.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "robot_model/RobotStateContactEstimator.h"

namespace ocs2::humanoid {

namespace {

/** The class name the events carry. */
constexpr char kControllerName[] = "CentroidalMpcMrtJointController";

}  // namespace

absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>>
CentroidalMpcMrtJointController::Create(
    const ::robot::model::RobotDescription& robotDescription,
    const ModelSettings& modelSettings,
    const CentroidalMpcRobotModel<scalar_t>& mpcRobotModel,
    MPC_BASE& mpc,
    MPC_MRT_Interface::SolverObservationObserver solverObservationObserver,
    PinocchioInterface pinocchioInterface,
    scalar_t mpcDesiredFrequency,
    const std::string& pdGainsFile,
    const MpcRobotModelBase<scalar_t>* absl_nullable effectiveMpcRobotModel) {
  InProcessMpcLink::Config linkConfig;
  linkConfig.mpcDesiredFrequency = mpcDesiredFrequency;
  linkConfig.solverThreadName = "Centroidal MPC Solver Thread";
  linkConfig.solverObservationObserver = std::move(solverObservationObserver);
  return Create(robotDescription, modelSettings, mpcRobotModel,
                InProcessMpcLink::factory(mpc, std::move(linkConfig)),
                std::move(pinocchioInterface), pdGainsFile,
                effectiveMpcRobotModel);
}

absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>>
CentroidalMpcMrtJointController::Create(
    const ::robot::model::RobotDescription& robotDescription,
    const ModelSettings& modelSettings,
    const CentroidalMpcRobotModel<scalar_t>& mpcRobotModel,
    const MpcLinkFactory& mpcLinkFactory,
    PinocchioInterface pinocchioInterface,
    const std::string& pdGainsFile,
    const MpcRobotModelBase<scalar_t>* absl_nullable effectiveMpcRobotModel) {
  // The joints the control cycle indexes without a check: refused here, by
  // name, rather than by the constructor's check.
  RETURN_IF_ERROR(
      robotDescription.findJointIndices(modelSettings.mpcModelJointNames)
          .status());
  RETURN_IF_ERROR(
      robotDescription.findJointIndices(modelSettings.fixedJointNames)
          .status());
  ASSIGN_OR_RETURN(InitialPdGains initialPdGains,
                   loadInitialPdGains(pdGainsFile, modelSettings));
  // The manifold of the state the controller writes and the policy acts on,
  // which the evaluator retains. The basis-input decorator keeps the wrapped
  // model's state, so the manifold is the same for both.
  ASSIGN_OR_RETURN(std::shared_ptr<const StateManifold> stateManifold,
                   stateManifoldOf(mpcRobotModel));
  // The rigid-body dynamics of the control cycle, on a FreeFlyer twin of the
  // model, at the sole frames it then reads unchecked.
  RETURN_IF_ERROR(checkContactFramesExist(pinocchioInterface.getModel(),
                                          modelSettings.contactNames));
  ASSIGN_OR_RETURN(std::unique_ptr<RealtimeRigidBodyDynamics> rigidBodyDynamics,
                   RealtimeRigidBodyDynamics::Create(
                       pinocchioInterface.getModel(),
                       getContactFrameIndexArray<scalar_t>(pinocchioInterface,
                                                           mpcRobotModel)));
  std::unique_ptr<CentroidalMpcMrtJointController> controller =
      absl::WrapUnique(new CentroidalMpcMrtJointController(
          robotDescription, modelSettings, mpcRobotModel, mpcLinkFactory,
          std::move(pinocchioInterface), pdGainsFile, effectiveMpcRobotModel,
          std::move(initialPdGains), std::move(stateManifold),
          std::move(rigidBodyDynamics)));
  RETURN_IF_ERROR(controller->checkConstruction());
  return controller;
}

CentroidalMpcMrtJointController::CentroidalMpcMrtJointController(
    const ::robot::model::RobotDescription& robotDescription,
    const ModelSettings& modelSettings,
    const CentroidalMpcRobotModel<scalar_t>& mpcRobotModel,
    const MpcLinkFactory& mpcLinkFactory,
    PinocchioInterface pinocchioInterface,
    const std::string& pdGainsFile,
    const MpcRobotModelBase<scalar_t>* absl_nullable effectiveMpcRobotModel,
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
      mpcRobotModelPtr_(mpcRobotModel.clone()),
      effectiveModelPtr_(effectiveMpcRobotModel
                             ? std::unique_ptr<MpcRobotModelBase<scalar_t>>(
                                   effectiveMpcRobotModel->clone())
                             : std::unique_ptr<MpcRobotModelBase<scalar_t>>(
                                   mpcRobotModel.clone())),
      eventSink_(&LoggingControllerEventSink::instance()),
      holdAction_(robotDescription),
      stateManifold_(std::move(stateManifold)),
      policyEvaluator_(
          ipc::ModelDimensions{.stateDim = effectiveModelPtr_->getStateDim(),
                               .inputDim = effectiveModelPtr_->getInputDim(),
                               .numModes = kNumModeNumbers},
          stateManifold_.get()),
      mpcPolicyState_(vector_t::Zero(effectiveModelPtr_->getStateDim())),
      mpcPolicyInput_(vector_t::Zero(effectiveModelPtr_->getInputDim())),
      rigidBodyDynamics_(std::move(rigidBodyDynamics)),
      robotWeight_(computeRobotWeight(pinocchioInterface_)),
      pdGainsFile_(pdGainsFile),
      robotJointNames_(robotDescription.getJointNames()),
      mpcModelJointNames_(modelSettings.mpcModelJointNames),
      fixedJointNames_(modelSettings.fixedJointNames),
      pdGainsLastWriteTime_(initialPdGains.writeTime),
      pdGains_(std::move(initialPdGains.gains)),
      pdGainsMailbox_(pdGains_) {
  mpcJointIndices_ =
      robotDescription.getJointIndices(modelSettings.mpcModelJointNames);
  otherJointIndices_ =
      robotDescription.getJointIndices(modelSettings.fixedJointNames);
  // Once, here, what the control cycle then takes for granted
  // (jointActionUnchecked()): every joint it writes carries an action in a
  // RobotJointAction of the robot description, as holdAction_ is one.
  checkJointIndices(robotDescription, mpcJointIndices_, kControllerName);
  checkJointIndices(robotDescription, otherJointIndices_, kControllerName);
  currentMpcObservation_.state =
      vector_t::Zero(effectiveModelPtr_->getStateDim());
  currentMpcObservation_.input =
      vector_t::Zero(effectiveModelPtr_->getInputDim());
  latestPolicyInput_ = vector_t::Zero(effectiveModelPtr_->getInputDim());

  // The control thread's workspaces, at the sizes it writes them with:
  // Pinocchio's configuration q (nq) and velocity v (nv), which the robot
  // model's measurement helpers fill.
  qPinocchio_ = vector_t::Zero(mpcRobotModelPtr_->getConfigurationDim());
  vPinocchio_ = vector_t::Zero(mpcRobotModelPtr_->getVelocityDim());
  mpcJointPositions_ = vector_t::Zero(mpcJointIndices_.size());
  mpcJointVelocities_ = vector_t::Zero(mpcJointIndices_.size());
  estimatedContactFlags_.assign(kNumContacts, true);
  gravityCoordinates_ =
      vector_t::Zero(mpcRobotModelPtr_->getConfigurationDim());
  gravityJointPositions_ = vector_t::Zero(mpcJointIndices_.size());
  plannedCoordinates_ =
      vector_t::Zero(mpcRobotModelPtr_->getConfigurationDim());
  policyJointPositions_ = vector_t::Zero(effectiveModelPtr_->getJointDim());
  policyJointVelocities_ = vector_t::Zero(effectiveModelPtr_->getJointDim());
  zeroJointAccelerations_ = vector_t::Zero(effectiveModelPtr_->getJointDim());
  safetyHoldMpcJointPositions_ = vector_t::Zero(mpcJointIndices_.size());
  safetyHoldOtherJointPositions_ = vector_t::Zero(otherJointIndices_.size());
  nominalJointPositions_.reserve(robotDescription.getNumJoints());
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status CentroidalMpcMrtJointController::checkConstruction() const {
  if (mpcLink_ == nullptr) {
    return absl::InvalidArgumentError(
        "[CentroidalMpcMrtJointController] the MPC link factory made no link.");
  }
  return absl::OkStatus();
}

absl::StatusOr<CentroidalMpcMrtJointController::InitialPdGains>
CentroidalMpcMrtJointController::loadInitialPdGains(
    const std::string& pdGainsFile, const ModelSettings& modelSettings) {
  const std::vector<std::string>& mpcJointNames =
      modelSettings.mpcModelJointNames;
  const std::vector<std::string>& fixedJointNames =
      modelSettings.fixedJointNames;
  InitialPdGains initial;
  // The write time first, then the gains: a save that lands between the two is
  // reloaded by the next poll.
  initial.writeTime = jointPdGainsFileWriteTime(pdGainsFile);
  // At start-up there are no gains in use to keep, and the controller has none
  // of its own: no file, a missing one or one it refuses, and it does not
  // start.
  ASSIGN_OR_RETURN(initial.gains,
                   loadRequiredJointPdGains(pdGainsFile, mpcJointNames,
                                            fixedJointNames, kControllerName));
  LOG(INFO) << "[CentroidalMpcMrtJointController] Loaded joint PD gains from "
            << pdGainsFile;

  // Diagnostic: print actual gain values after loading
  LOG(INFO) << "[PD_GAINS_DEBUG] default_kp=" << initial.gains.defaults.kp
            << " default_kd=" << initial.gains.defaults.kd;
  for (size_t i = 0; i < mpcJointNames.size(); ++i) {
    LOG(INFO) << "  mpc_joint[" << i << "] " << mpcJointNames[i]
              << " kp=" << initial.gains.mpcJointKp[i]
              << " kd=" << initial.gains.mpcJointKd[i];
  }
  return initial;
}

absl::Status CentroidalMpcMrtJointController::postPdGains(
    const absl::StatusOr<JointPdGains>& gains,
    absl::string_view source,
    uint64_t ticket) {
  if (!gains.ok()) {
    LOG(WARNING)
        << "[CentroidalMpcMrtJointController] Refused the PD gains from "
        << source
        << "; the gains in use are kept: " << gains.status().message();
    return gains.status();
  }
  LOG(INFO) << "[CentroidalMpcMrtJointController] Loaded joint PD gains from "
            << source;
  // Diagnostic: print actual gain values after loading
  LOG(INFO) << "[PD_GAINS_DEBUG] default_kp=" << gains->defaults.kp
            << " default_kd=" << gains->defaults.kd;
  for (size_t i = 0; i < mpcModelJointNames_.size(); ++i) {
    LOG(INFO) << "  mpc_joint[" << i << "] " << mpcModelJointNames_[i]
              << " kp=" << gains->mpcJointKp[i]
              << " kd=" << gains->mpcJointKd[i];
  }
  return pdGainsMailbox_.post(*gains, ticket);
}

absl::Status CentroidalMpcMrtJointController::setPdGains(
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
  LOG(INFO)
      << "[CentroidalMpcMrtJointController] setPdGains received gains for "
      << gains.joint_gains.size() << " named joints";
  RETURN_IF_ERROR(postPdGains(
      jointPdGainsFromConfig(gains, mpcModelJointNames_, fixedJointNames_),
      "setPdGains", ticket));
  LOG(INFO) << "[CentroidalMpcMrtJointController] PD gains from setPdGains are "
               "applied at the next control cycle.";
  return absl::OkStatus();
}

void CentroidalMpcMrtJointController::pollPdGainsFile() {
  if (pdGainsFile_.empty()) return;
  absl::MutexLock lock(&pdGainsFileMutex_);
  std::error_code ec;
  const std::filesystem::file_time_type last_write =
      std::filesystem::last_write_time(pdGainsFile_, ec);
  if (ec || last_write == pdGainsLastWriteTime_) return;
  LOG(INFO) << "[CentroidalMpcMrtJointController] PD gains file changed (old="
            << pdGainsLastWriteTime_.time_since_epoch().count()
            << " new=" << last_write.time_since_epoch().count()
            << "). Reloading...";
  pdGainsLastWriteTime_ = last_write;
  if (liveGainsArrival_.has_value() && last_write < *liveGainsArrival_) {
    LOG(INFO) << "[CentroidalMpcMrtJointController] " << pdGainsFile_
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

CentroidalMpcMrtJointController::~CentroidalMpcMrtJointController() {
  // Stop the solver before anything its reset target reads is destroyed.
  if (mpcLink_ != nullptr) mpcLink_->stop();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void CentroidalMpcMrtJointController::startMpcThread(
    const ::robot::model::RobotState& initRobotState) {
  updateMpcObservation(currentMpcObservation_, initRobotState);
  // Set observation to MPC and start serving policies.
  mpcLink_->start(currentMpcObservation_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void CentroidalMpcMrtJointController::updateMpcState(
    vector_t& mpcState, const ::robot::model::RobotState& robotState) {
  const CentroidalModelInfo& info = mpcRobotModelPtr_->getCentroidalModelInfo();
  // The measured base orientation, whose coefficients (x, y, z, w) the robot
  // model's measurement helpers read.
  const quaternion_t orientation = robotState.getQuatBaseRtWorld();

  // Pinocchio's q and v of the measurement, into the workspaces at their sizes:
  // no allocation.
  robotState.getJointPositions(mpcJointIndices_, mpcJointPositions_);
  mpcRobotModelPtr_->fillGeneralizedCoordinatesFromMeasurement(
      robotState.getPosBaseRtWorldEwrtWorld(), orientation.coeffs(),
      mpcJointPositions_, qPinocchio_);
  robotState.getJointVelocities(mpcJointIndices_, mpcJointVelocities_);
  mpcRobotModelPtr_->fillGeneralizedVelocitiesFromMeasurement(
      orientation.coeffs(), robotState.getLinvelBaseRtWorldEwrtBase(),
      robotState.getAngvelBaseRtWorldEwrtBase(), mpcJointVelocities_,
      vPinocchio_);

  // h = A(q) v / m, A of updateCentroidalDynamics() for the model type, without
  // its allocations: the whole robot's momentum (ccrba() on the twin) or the
  // single rigid body's, whose joint block is zero.
  if (info.centroidalModelType ==
      CentroidalModelType::SingleRigidBodyDynamics) {
    centroidal_model::getMnormCmomComRtWorldEwrtWorld(mpcState, info)
        .noalias() = getSingleRigidBodyBaseMomentumMatrix(info, qPinocchio_) *
                     vPinocchio_.head<centroidal_model::kBaseVelocityDim>() /
                     info.robotMass;
  } else {
    centroidal_model::getMnormCmomComRtWorldEwrtWorld(mpcState, info) =
        rigidBodyDynamics_->computeCmomComRtWorldEwrtWorld(qPinocchio_,
                                                           vPinocchio_) /
        info.robotMass;
  }
  centroidal_model::getGeneralizedCoordinates(mpcState, info) = qPinocchio_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void CentroidalMpcMrtJointController::updateMpcObservation(
    ocs2::SystemObservation& mpcObservation,
    const ::robot::model::RobotState& robotState) {
  updateMpcState(mpcObservation.state, robotState);
  mpcObservation.time = robotState.getTime();
  mpcObservation.input = vector_t::Zero(effectiveModelPtr_->getInputDim());
  // The contact block of the input differs between the wrench-space (6 per
  // foot) and basis-vector (numBasisPerFoot) layouts, so the joint velocities
  // are written through the effective model rather than by a fixed-offset
  // slice.
  robotState.getJointVelocities(mpcJointIndices_, mpcJointVelocities_,
                                /*defaultValue=*/0.0);
  effectiveModelPtr_->setJointVelocities(
      mpcObservation.state, mpcObservation.input, mpcJointVelocities_);
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

void CentroidalMpcMrtJointController::setContactWrenchGateConfig(
    const ContactWrenchGate::Config& config) {
  if (contactWrenchGate_.setConfig(config)) {
    postEvent(ControllerEventCode::kContactWrenchGateChanged,
              config.debounceTime, config.rampTime);
  } else {
    postEvent(ControllerEventCode::kContactWrenchGateRefused,
              config.debounceTime, config.rampTime);
  }
}

void CentroidalMpcMrtJointController::setEventSink(
    ControllerEventSink* absl_nullable eventSink) {
  eventSink_ = eventSink != nullptr ? eventSink
                                    : &LoggingControllerEventSink::instance();
}

void CentroidalMpcMrtJointController::postEvent(ControllerEventCode code,
                                                scalar_t value0,
                                                scalar_t value1,
                                                absl::string_view text) {
  eventSink_->post(
      makeControllerEvent(code, kControllerName, value0, value1, text));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void CentroidalMpcMrtJointController::setContactEstimator(
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

void CentroidalMpcMrtJointController::computeJointControlAction(
    scalar_t /*time*/,
    const ::robot::model::RobotState& robotState,
    ::robot::model::RobotJointAction& robotJointAction) {
  // The newest PD gains posted by setPdGains() or pollPdGainsFile(), parsed on
  // their callers' threads: a copy between preallocated vectors, without a
  // lock.
  pdGainsMailbox_.receive(pdGains_);

  // Always update MPC observation so the solver continues tracking robot state
  // and time in all modes.
  updateMpcObservation(currentMpcObservation_, robotState);
  // A clock that runs backwards leaves every plan and every time-keyed action
  // timed on the old clock; the supervisor requests the reset, and the hold
  // keeps the policies planned on the old clock away from the robot.
  const scalar_t clockRewind =
      mpcLink_->observeTime(currentMpcObservation_.time);
  if (clockRewind > 0.0) handleClockRewind(clockRewind);
  mpcLink_->setCurrentObservation(currentMpcObservation_);

  // ZERO_TORQUE: nothing is commanded, and nothing about the MPC is checked -
  // the policy is not executed, so neither a hold nor the divergence check has
  // anything to protect, and the check used to request a reset at every cycle
  // against a limp robot.
  if (controlMode_ == control_mode::kZeroJointTorque) {
    fillZeroTorqueAction(robotState, robotJointAction);
    previousObservationTime_ = currentMpcObservation_.time;
    return;
  }

  // JOINT_PD mode: PD tracking to nominal positions + Pinocchio gravity
  // compensation. This code path is shared between sim and real hardware.
  if (controlMode_ == control_mode::kJointPd) {
    fillJointPdAction(robotState, robotJointAction);
    // Keep the observation time current so that the policy lookahead is one
    // control cycle when WB_MPC resumes, not the clamp maximum after a long
    // stay in this mode.
    previousObservationTime_ = currentMpcObservation_.time;
    return;
  }

  // GRAVITY_COMP mode: Zero-G compliant mode using pure gravity compensation
  // torques + light damping. Limbs can be moved compliantly by hand or external
  // forces.
  if (controlMode_ == control_mode::kGravityComp) {
    fillGravityCompAction(robotState, robotJointAction);
    previousObservationTime_ = currentMpcObservation_.time;
    return;
  }

  // SAFETY mode: damped joint PD about the posture at mode entry, both gains
  // scaled by alpha(t) = exp(-t / tau) so the torques ramp out over a few time
  // constants rather than being cut in a single cycle. Deliberately ahead of
  // the hand-over check below and independent of the model and the solver:
  // SAFETY is what runs when those are the problem.
  if (controlMode_ == control_mode::kSafety) {
    fillSafetyAction(robotState, robotJointAction);
    previousObservationTime_ = currentMpcObservation_.time;
    return;
  }

  // Active MPC control path. The policy in use counts as post-reset once it was
  // solved after the last reset the solver thread served and no reset is
  // outstanding: the two checks together are race-free (MpcResetSupervisor).
  mpcLink_->updatePolicy();
  reportPolicyUpdateLockContention();
  const bool postResetPolicyActive =
      mpcLink_->isActivePolicyCurrent() && !mpcLink_->hasOutstandingReset();
  policyActivated_.store(postResetPolicyActive);

  // A solver that keeps failing leaves a stale policy in use; the robot is held
  // with the JOINT_PD action instead, until a solve succeeds again and its
  // policy is ramped in like an entry.
  if (!mpcLink_->isHealthy() && !awaitingPostResetPolicy_.load()) {
    entryHoldGravityComp_.store(false);
    entryBlendStartTime_.store(-1.0);
    awaitingPostResetPolicy_.store(true);
  }

  // Hand-over after a hold (setControlMode(), requestMpcResetAndHold()): the
  // previous mode's action is sent until a policy solved after the reset is in
  // use, and the ramp starts on the first cycle that runs on it.
  bool handedOverThisCycle = false;
  if (awaitingPostResetPolicy_.load()) {
    if (!postResetPolicyActive || !mpcLink_->isHealthy()) {
      fillEntryHoldAction(robotState, robotJointAction);
      previousObservationTime_ = currentMpcObservation_.time;
      return;
    }
    awaitingPostResetPolicy_.store(false);
    entryBlendStartTime_.store(
        mpcEntryBlendTime_ > 0.0 ? currentMpcObservation_.time : -1.0);
    handedOverThisCycle = true;
  }

  // Sized once for the effective model, so that evaluating the policy into them
  // allocates nothing.
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
      if (!handedOverThisCycle) entryHoldGravityComp_.store(false);
      entryBlendStartTime_.store(-1.0);
      awaitingPostResetPolicy_.store(true);
      fillEntryHoldAction(robotState, robotJointAction);
      previousObservationTime_ = currentMpcObservation_.time;
      return;
    }
    refusedPolicyReported_ = false;
    latestPolicyInput_ = mpcPolicyInput;

    // The feedforward below is the floating-base inverse dynamics at the
    // MEASURED state with zero joint acceleration (the centroidal input has
    // joint velocities, not accelerations) and the planned contact wrenches as
    // gated by the measured contact: tau_j = M_jb a_b + nle_j - (J^T W)_j with
    // a_b the base acceleration those wrenches produce
    // (RealtimeRigidBodyDynamics::freeBaseJointTorques(), computeJointTorques()
    // without its allocations, pinned against Pinocchio's RNEA by
    // testRealtimeRigidBodyDynamics).
    //
    // The planned joint angles and velocities (getJointAngles(),
    // getJointVelocities(): the state's and the input's joint blocks), into
    // workspaces, and the measured joint angles, the observation's.
    const Eigen::Index jointDim =
        static_cast<Eigen::Index>(effectiveModelPtr_->getJointDim());
    const Eigen::Index jointStart =
        static_cast<Eigen::Index>(effectiveModelPtr_->getJointStartindex());
    const Eigen::Index jointVelocitiesStart = static_cast<Eigen::Index>(
        effectiveModelPtr_->getJointVelocitiesStartindex());
    vector_t& mpc_q_j_des = policyJointPositions_;
    vector_t& mpc_qd_j_des = policyJointVelocities_;
    mpc_q_j_des = mpcPolicyState.segment(jointStart, jointDim);
    mpc_qd_j_des = mpcPolicyInput.segment(jointVelocitiesStart, jointDim);
    const Eigen::Ref<const vector_t> q_j =
        currentMpcObservation_.state.segment(jointStart, jointDim);

    // Sanity check: detect divergence in MPC policy state to prevent violent
    // actuator thrashing
    scalar_t maxJointError = (mpc_q_j_des - q_j).cwiseAbs().maxCoeff();
    bool policyDiverged = !mpcPolicyState.allFinite() ||
                          !mpcPolicyInput.allFinite() ||
                          maxJointError > pdGains_.maxPolicyJointError;
    if (policyDiverged) {
      // One reset per policy solved after the last one, and at most one per
      // kDivergenceResetInterval. This used to request a reset at every cycle:
      // the policy in use stays the diverged one until a post-reset policy
      // replaces it, so the check fired again at once, and every reset threw
      // away the solution that would have replaced it.
      const scalar_t time = currentMpcObservation_.time;
      const bool intervalElapsed =
          lastDivergenceResetTime_ < 0.0 ||
          time - lastDivergenceResetTime_ >= kDivergenceResetInterval;
      if (postResetPolicyActive && intervalElapsed) {
        postEvent(ControllerEventCode::kPolicyDiverged, maxJointError);
        // The solver alone: the schedule in execution stays. A full reset
        // restarted the gait in stance under a robot in mid-stride, which a
        // divergence while walking is likely to be, and the robot fell.
        policyActivated_.store(false);
        mpcLink_->requestReset(MpcResetSupervisor::ResetKind::kSolver);
        lastDivergenceResetTime_ = time;
      }
      for (int k = 0; k < mpc_q_j_des.size(); ++k) {
        mpc_q_j_des[k] = std::clamp(
            mpc_q_j_des[k], q_j[k] - pdGains_.divergedPolicyJointTargetBound,
            q_j[k] + pdGains_.divergedPolicyJointTargetBound);
      }
      mpc_qd_j_des.setZero();
    }

    // Feedforward joint acceleration is set to zero (zeroJointAccelerations_)
    // so feedforward torques act as pure dynamic cancellation (gravity,
    // Coriolis, and contact forces) and do not fight actuator PD loops.

    // The inverse dynamics projects the wrenches with LOCAL_WORLD_ALIGNED
    // Jacobians, so it needs WORLD-frame wrenches. With basis-vector inputs the
    // input-only accessor returns the LOCAL contact-frame wrench B*lambda; the
    // world-frame accessor rotates it with the contact frame orientation of the
    // planned state, which is the orientation the OCP dynamics used when it
    // chose lambda. A non-finite policy state (divergence) would poison the
    // rotation, so fall back to the measured state in that case. The contact
    // frames' orientations are read off the twin's kinematics at that state's
    // generalized coordinates (getGeneralizedCoordinates(): its quaternion
    // safely normalized), which the measured state's are.
    const vector_t& wrenchFrameState = mpcPolicyState.allFinite()
                                           ? mpcPolicyState
                                           : currentMpcObservation_.state;
    fillPlannedCoordinates(wrenchFrameState);
    const feet_array_t<matrix3_t>& contactFrameRotations =
        rigidBodyDynamics_->computeRotmatContactsRtWorld(plannedCoordinates_);
    // The policy carries a wrench wherever its own schedule expects contact.
    // Whether a foot can actually transmit it is decided by the measured
    // contact state of this cycle, not by the plan: the wrench of a foot that
    // is not touching is dropped, and after touch-down it is debounced and
    // ramped in as configured (ContactWrenchGate).
    const feet_array_t<vector6_t> footWrenches = contactWrenchGate_.apply(
        {effectiveModelPtr_->getWrenchContactEwrtWorld(
             mpcPolicyInput, contactFrameRotations[0], /*contactIndex=*/0),
         effectiveModelPtr_->getWrenchContactEwrtWorld(
             mpcPolicyInput, contactFrameRotations[1], /*contactIndex=*/1)});

    // Evaluate inverse dynamics using measured robot state for physical
    // consistency: the measurement's q and v, of which the observation is made
    // (getGeneralizedCoordinates() and getGeneralizedVelocities() of the
    // observation give them back, the latter through the centroidal momentum).
    const vector_t& mpcJointTorques = rigidBodyDynamics_->freeBaseJointTorques(
        qPinocchio_, vPinocchio_, zeroJointAccelerations_, footWrenches);

    // Gravity-comp fallback: use pure gravity compensation instead of full ID
    // torques (`wb_mpc_feedforward: "gravity_compensation"` in the task file),
    // to isolate ID issues.
    const vector_t& feedforwardTorques =
        useGravityCompFeedforward_ ? computeGravityCompensation(robotState)
                                   : mpcJointTorques;

    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
      const size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action =
          jointActionUnchecked(robotJointAction, index);

      action.q_des = mpc_q_j_des[i];
      action.qd_des = mpc_qd_j_des[i];
      action.kp = pdGains_.mpcJointKp[i];
      action.kd = pdGains_.mpcJointKd[i];
      action.feed_forward_effort =
          clampedMpcJointFeedforward(i, feedforwardTorques[i]);
    }
    // The torques this decomposes into (the PD and the feedforward of every
    // joint, the planned and measured base) are in the telemetry (robot/state),
    // which the Rerun bridge plots: the control thread logs none of it.
  }

  else {
    if (!noPolicyReported_) {
      postEvent(ControllerEventCode::kNoPolicyWeightCompensation);
      noPolicyReported_ = true;
    }
    // Apply the weight-compensating wrenches around the current state: the
    // weight shared evenly by the feet measured in contact, as vertical
    // world-frame forces (weightCompensatingInput(), whose wrenches these are,
    // and which the basis-vector input reproduces up to its non-negative
    // least-squares fit). With none (the robot hangs on the gantry) no contact
    // force is compensated and the feedforward is the gravity and Coriolis term
    // of the free legs, the base being held (heldBaseJointTorques(); the
    // free-base inverse dynamics would let the base fall freely).
    const size_t numStanceLegs = numberOfLegsInContacts(measuredContactFlags_);
    feet_array_t<vector6_t> footWrenches;
    for (size_t i = 0; i < kNumContacts; ++i) {
      footWrenches[i].setZero();
      if (numStanceLegs > 0 && measuredContactFlags_[i]) {
        footWrenches[i](kZAxisIndex) =
            robotWeight_ / static_cast<scalar_t>(numStanceLegs);
      }
    }
    const vector_t& weightCompensatingTorques =
        rigidBodyDynamics_->heldBaseJointTorques(
            qPinocchio_, vPinocchio_, zeroJointAccelerations_, footWrenches);

    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
      const size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action =
          jointActionUnchecked(robotJointAction, index);

      action.q_des = nominalJointPositions_.empty()
                         ? robotState.getJointPosition(index)
                         : nominalJointPositions_[index];
      action.qd_des = 0.0;
      action.kp = pdGains_.mpcJointKp[i];
      action.kd = pdGains_.mpcJointKd[i];
      action.feed_forward_effort =
          clampedMpcJointFeedforward(i, weightCompensatingTorques[i]);
    }
  }

  for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
    const size_t index = otherJointIndices_[i];
    robot::model::JointAction& action =
        jointActionUnchecked(robotJointAction, index);

    action.q_des =
        nominalJointPositions_.empty() ? 0.0 : nominalJointPositions_[index];
    action.qd_des = 0;
    action.kp = pdGains_.otherJointKp[i];
    action.kd = pdGains_.otherJointKd[i];
    action.feed_forward_effort = 0.0;
  }

  applyEntryBlend(robotState, robotJointAction);

  // Track observation time for next call's dt computation
  previousObservationTime_ = currentMpcObservation_.time;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void CentroidalMpcMrtJointController::setControlMode(absl::string_view mode) {
  // Called every cycle; compared and assigned through string_views, so that
  // nothing is allocated (controlMode_ holds any mode name in its small-string
  // buffer).
  const absl::string_view newMode = mode;
  if (newMode == controlMode_) return;
  if (newMode == control_mode::kSafety) {
    // Arm the decay. The clock origin and the posture to hold are captured on
    // the first control cycle in the mode, where the observation time and a
    // RobotState are in hand. A pending hand-over into WB_MPC is abandoned:
    // SAFETY must not be held off waiting for a solver.
    safetyDecayCaptured_.store(false);
    safetyDecayComplete_.store(false);
    awaitingPostResetPolicy_.store(false);
    entryBlendStartTime_.store(-1.0);
  }
  noPolicyReported_ = false;
  if (control_mode::isPassive(controlMode_) && control_mode::isMpc(newMode)) {
    // The MPC starts again from the robot as it is now; a passive action is
    // held until a policy solved after that is in use. JOINT_PD and
    // GRAVITY_COMP are held as they are; ZERO_TORQUE (no posture hold) and
    // SAFETY (a hold that decays to zero torque) are held like JOINT_PD.
    requestMpcResetAndHold(controlMode_ == control_mode::kGravityComp);
  }
  controlMode_.assign(mode.data(), mode.size());
}

void CentroidalMpcMrtJointController::requestMpcResetAndHold(
    bool holdGravityComp) {
  // The hold is armed before the reset is requested, so that no cycle sees the
  // request without the hold.
  armHold(holdGravityComp);
  requestMpcReset();
}

void CentroidalMpcMrtJointController::armHold(bool holdGravityComp) {
  entryHoldGravityComp_.store(holdGravityComp);
  entryBlendStartTime_.store(-1.0);
  awaitingPostResetPolicy_.store(true);
}

void CentroidalMpcMrtJointController::handleClockRewind(scalar_t rewind) {
  postEvent(ControllerEventCode::kClockRewind, rewind,
            currentMpcObservation_.time);
  // SAFETY keeps the time it has already decayed for: restarting its clock
  // would put the gains back to full authority. A rewind past the entry shifts
  // the start time below zero, which is a valid start time on the new clock.
  if (safetyDecayCaptured_.load())
    safetyDecayStartTime_.store(safetyDecayStartTime_.load() - rewind);
  lastDivergenceResetTime_ = -1.0;
  previousObservationTime_ = currentMpcObservation_.time;
  // MpcResetSupervisor::observeTime() has requested the reset; the hold is all
  // that is left to arm.
  armHold(/*holdGravityComp=*/false);
}

void CentroidalMpcMrtJointController::fillJointPdAction(
    const ::robot::model::RobotState& robotState,
    ::robot::model::RobotJointAction& robotJointAction) {
  const vector_t& gravTorques = computeGravityCompensation(robotState);

  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    const size_t index = mpcJointIndices_[i];
    robot::model::JointAction& action =
        jointActionUnchecked(robotJointAction, index);
    action.q_des =
        nominalJointPositions_.empty() ? 0.0 : nominalJointPositions_[index];
    action.qd_des = 0.0;
    action.kp = pdGains_.mpcJointKp[i];
    action.kd = pdGains_.mpcJointKd[i];
    action.feed_forward_effort = clampedMpcJointFeedforward(i, gravTorques[i]);
  }

  for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
    const size_t index = otherJointIndices_[i];
    robot::model::JointAction& action =
        jointActionUnchecked(robotJointAction, index);
    action.q_des =
        nominalJointPositions_.empty() ? 0.0 : nominalJointPositions_[index];
    action.qd_des = 0.0;
    action.kp = pdGains_.otherJointKp[i];
    action.kd = pdGains_.otherJointKd[i];
    action.feed_forward_effort = 0.0;  // Non-MPC joints don't get gravity comp
  }
  // The posture errors and the gravity torques of the settling robot are in the
  // telemetry (robot/state): the control thread logs none of it.
}

void CentroidalMpcMrtJointController::fillGravityCompAction(
    const ::robot::model::RobotState& robotState,
    ::robot::model::RobotJointAction& robotJointAction) {
  // The base-held gravity torques g_j(q). This mode is operated with the robot
  // SUSPENDED FROM THE GANTRY, which holds the base externally, so g_j(q) is
  // the correct compensation and the feet carry nothing.
  //
  // Off the gantry it would not be: for a floating base static equilibrium is
  // g(q) = S^T tau + J_c^T f, so a robot standing on its own feet needs tau =
  // g_j(q) - J_{c,j}^T f, and the contact term dominates at a bent knee (about
  // 135 Nm against 7 Nm at the shipped Atlas stance, 23 Nm against 3.4 Nm on
  // the SA01 crouch). With kp = 0 the feedforward is all that holds the robot,
  // so commanding g_j(q) alone on the ground leaves the ground reaction's
  // moment unopposed at the knee and the crouch runs away. Do not run this mode
  // with the robot bearing its own weight without adding that term back -
  // testContactConstraintScheduleGating pins the magnitudes.
  const vector_t& gravTorques = computeGravityCompensation(robotState);

  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    const size_t index = mpcJointIndices_[i];
    robot::model::JointAction& action =
        jointActionUnchecked(robotJointAction, index);
    action.q_des = robotState.getJointPosition(index);
    action.qd_des = 0.0;
    action.kp = 0.0;
    action.kd = pdGains_.mpcJointKd[i] * pdGains_.gravityCompDampingFraction;
    action.feed_forward_effort = clampedMpcJointFeedforward(i, gravTorques[i]);
  }

  for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
    const size_t index = otherJointIndices_[i];
    robot::model::JointAction& action =
        jointActionUnchecked(robotJointAction, index);
    action.q_des =
        nominalJointPositions_.empty() ? 0.0 : nominalJointPositions_[index];
    action.qd_des = 0.0;
    action.kp = pdGains_.otherJointKp[i] * 0.5;
    action.kd = pdGains_.otherJointKd[i];
    action.feed_forward_effort = 0.0;
  }
}

void CentroidalMpcMrtJointController::fillZeroTorqueAction(
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

scalar_t CentroidalMpcMrtJointController::currentSafetyDecayFactor(
    scalar_t time) const {
  if (!safetyDecayCaptured_.load()) return 1.0;
  return safety_decay::latchedFactor(safetyDecayLatchedFactor_,
                                     time - safetyDecayStartTime_.load(),
                                     safetyDecayTimeConstant_);
}

scalar_t CentroidalMpcMrtJointController::clampedMpcJointFeedforward(
    size_t mpcJoint, scalar_t torque) const {
  return std::clamp(torque, -pdGains_.mpcJointTorqueLimit[mpcJoint],
                    pdGains_.mpcJointTorqueLimit[mpcJoint]);
}

void CentroidalMpcMrtJointController::reportPolicyUpdateLockContention() {
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

void CentroidalMpcMrtJointController::fillSafetyAction(
    const ::robot::model::RobotState& robotState,
    ::robot::model::RobotJointAction& robotJointAction) {
  // Capture the clock origin and the posture to hold on the first cycle in the
  // mode. setControlMode() only arms the decay: neither the observation time of
  // the cycle nor a RobotState is in hand there.
  //
  // The posture held is the MEASURED one at entry, not the nominal: SAFETY is
  // entered when something has already gone wrong, and commanding a return to
  // the nominal stance at full gain would be a lunge, not a safe stop.
  if (!safetyDecayCaptured_.load()) {
    // Sized by the constructor: no allocation.
    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
      safetyHoldMpcJointPositions_[i] =
          robotState.getJointPosition(mpcJointIndices_[i]);
    }
    for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
      safetyHoldOtherJointPositions_[i] =
          robotState.getJointPosition(otherJointIndices_[i]);
    }
    safetyDecayStartTime_.store(currentMpcObservation_.time);
    safetyDecayLatchedFactor_ = 1.0;
    safetyDecayCaptured_.store(true);
    postEvent(ControllerEventCode::kSafetyEntered, safetyDecayTimeConstant_);
  }

  // Latched: alpha never rises within this stay in SAFETY
  // (safety_decay::latchedFactor()).
  const scalar_t alpha = currentSafetyDecayFactor(currentMpcObservation_.time);
  safetyDecayLatchedFactor_ = alpha;
  if (alpha == 0.0 && !safetyDecayComplete_.exchange(true)) {
    postEvent(ControllerEventCode::kSafetyDecayComplete);
  }

  // alpha * (kp * (q_hold - q) - kd * qd), assembled through the gains so that
  // RobotJointAction's own getTotalFeedbackJointTorque() produces it. No
  // feedforward: nothing here depends on the model or the policy.
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

void CentroidalMpcMrtJointController::fillEntryHoldAction(
    const ::robot::model::RobotState& robotState,
    ::robot::model::RobotJointAction& robotJointAction) {
  if (entryHoldGravityComp_.load()) {
    fillGravityCompAction(robotState, robotJointAction);
  } else {
    fillJointPdAction(robotState, robotJointAction);
  }
}

void CentroidalMpcMrtJointController::applyEntryBlend(
    const ::robot::model::RobotState& robotState,
    ::robot::model::RobotJointAction& robotJointAction) {
  const scalar_t startTime = entryBlendStartTime_.load();
  if (startTime < 0.0) return;
  const scalar_t alpha =
      mpcEntryBlendTime_ > 0.0
          ? (currentMpcObservation_.time - startTime) / mpcEntryBlendTime_
          : 1.0;
  if (alpha >= 1.0) {
    entryBlendStartTime_.store(-1.0);
    return;
  }
  // The total torque kp (q_des - q) + kd (qd_des - qd) + ff moves linearly from
  // the held action to the MPC action at every measured q and qd
  // (blendJointActionLinearInTorque(): the gains blend linearly, the targets as
  // gain-weighted means). Blending every field on its own, as this did, is not
  // linear in the torque - kp q_des is the product of two blended fields - and
  // blending the feedforward alone would leave the PD term to jump. The held
  // action goes into a workspace of the robot's size: a copy of equal sizes, no
  // allocation.
  holdAction_ = robotJointAction;
  fillEntryHoldAction(robotState, holdAction_);
  const scalar_t a = std::max(alpha, 0.0);
  for (size_t index : mpcJointIndices_)
    blendJointAction(index, a, robotJointAction);
  for (size_t index : otherJointIndices_)
    blendJointAction(index, a, robotJointAction);
}

void CentroidalMpcMrtJointController::blendJointAction(
    size_t index,
    scalar_t a,
    ::robot::model::RobotJointAction& robotJointAction) const {
  blendJointActionLinearInTorque(jointActionUnchecked(holdAction_, index), a,
                                 jointActionUnchecked(robotJointAction, index));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
TargetTrajectories
CentroidalMpcMrtJointController::currentObservationToResetTrajectory(
    const SystemObservation& currentObservation) {
  // The one definition of the reset target, which the MPC node serves its
  // resets from as well.
  const TargetTrajectories resetTargetTrajectories =
      centroidalMpcResetTargetTrajectories(
          currentObservation, mpcRobotModelPtr_->getCentroidalModelInfo(),
          *effectiveModelPtr_, pinocchioInterface_);

  if (mpcLink_->isHealthy()) {
    const vector_t& targetState =
        resetTargetTrajectories.stateTrajectory.front();
    const vector_t& targetInput =
        resetTargetTrajectories.inputTrajectory.front();
    const vector3_t basePosition =
        effectiveModelPtr_->getPosBaseRtWorldEwrtWorld(targetState);
    LOG(INFO) << "[CentroidalMPC] Resetting MPC target trajectory. Base pos: "
              << basePosition.transpose()
              << " Base z: " << basePosition(kZAxisIndex)
              << " Input forces (world): "
              << effectiveModelPtr_
                     ->getForceContactEwrtWorld(targetState, targetInput,
                                                /*contactIndex=*/0)
                     .transpose()
              << " / "
              << effectiveModelPtr_
                     ->getForceContactEwrtWorld(targetState, targetInput,
                                                /*contactIndex=*/1)
                     .transpose();
  }
  return resetTargetTrajectories;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void CentroidalMpcMrtJointController::fillPlannedCoordinates(
    const vector_t& state) {
  const Eigen::Index configurationDim =
      static_cast<Eigen::Index>(mpcRobotModelPtr_->getConfigurationDim());
  const Eigen::Index baseStart =
      static_cast<Eigen::Index>(effectiveModelPtr_->getBaseStartindex());
  const Eigen::Index orientationStart = static_cast<Eigen::Index>(
      effectiveModelPtr_->getBaseOrientationStartIndex());
  plannedCoordinates_ = state.segment(baseStart, configurationDim);
  plannedCoordinates_.segment<kQuaternionCoefficients>(orientationStart -
                                                       baseStart) =
      quaternionSafeNormalize<scalar_t>(quaternion_coeffs_t<scalar_t>(
          state.segment<kQuaternionCoefficients>(orientationStart)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
const vector_t& CentroidalMpcMrtJointController::computeGravityCompensation(
    const ::robot::model::RobotState& robotState) {
  // Pinocchio's configuration q of the measurement, in the workspaces (sized by
  // the constructor).
  const quaternion_t orientation = robotState.getQuatBaseRtWorld();
  robotState.getJointPositions(mpcJointIndices_, gravityJointPositions_);
  mpcRobotModelPtr_->fillGeneralizedCoordinatesFromMeasurement(
      robotState.getPosBaseRtWorldEwrtWorld(), orientation.coeffs(),
      gravityJointPositions_, gravityCoordinates_);

  // The joint rows of g(q), nle(q, 0) of the model, without its allocations on
  // the composite root.
  return rigidBodyDynamics_->gravityJointTorques(gravityCoordinates_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::optional<contact_flag_t>
CentroidalMpcMrtJointController::getPlannedContactFlags(scalar_t time) const {
  if (!policyActivated_.load()) return std::nullopt;
  return modeNumber2StanceLeg(
      mpcLink_->getPolicy().modeSchedule_.modeAtTime(time));
}

}  // namespace ocs2::humanoid
