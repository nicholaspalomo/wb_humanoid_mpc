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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <tuple>

#include <yaml-cpp/yaml.h>
#include <pinocchio/algorithm/rnea.hpp>

#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <humanoid_common_mpc/common/ThreadAffinity.h>
#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>
#include <humanoid_common_mpc/mrt/ControlMode.h>
#include <humanoid_common_mpc/mrt/SafetyDecay.h>
#include <humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h>
#include <humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h>
#include <robot_model/RobotStateContactEstimator.h>
#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

WBMpcMrtJointController::WBMpcMrtJointController(const ::robot::model::RobotDescription& robotDescription,
                                                 const ModelSettings& modelSettings,
                                                 MPC_BASE& mpc,
                                                 PinocchioInterface pinocchioInterface,
                                                 scalar_t mpcDesiredFrequency,
                                                 std::shared_ptr<DummyObserver> rVizVisualizerPtr,
                                                 const std::string& pdGainsFile)
    : mcpMrtInterface_(mpc),
      contactEstimator_(std::make_shared<::robot::model::RobotStateContactEstimator>()),
      pinocchioInterface_(pinocchioInterface),
      mpcRobotModel_(modelSettings),
      mpcDeltaTMicroSeconds_(1000000 / mpcDesiredFrequency),
      realtime_(mpcDesiredFrequency <= 0),
      visualizerPtr_(rVizVisualizerPtr),
      pdGainsFile_(pdGainsFile),
      modelSettings_(modelSettings) {
  mpcJointIndices_ = robotDescription.getJointIndices(modelSettings.mpcModelJointNames);
  otherJointIndices_ = robotDescription.getJointIndices(modelSettings.fixedJointNames);
  currentMpcObservation_.state = vector_t::Zero(mpcRobotModel_.getStateDim());
  currentMpcObservation_.input = vector_t::Zero(mpcRobotModel_.getInputDim());
  latestPolicyInput_ = vector_t::Zero(mpcRobotModel_.getInputDim());

  if (!pdGainsFile_.empty() && std::filesystem::exists(pdGainsFile_)) {
    std::error_code ec;
    pdGainsLastWriteTime_ = std::filesystem::last_write_time(pdGainsFile_, ec);
  }

  loadPdGains(pdGainsFile, modelSettings);
}

void WBMpcMrtJointController::loadPdGains(const std::string& pdGainsFile, const ModelSettings& modelSettings) {
  mpcJointKp_.resize(mpcJointIndices_.size());
  mpcJointKd_.resize(mpcJointIndices_.size());
  otherJointKp_.resize(otherJointIndices_.size());
  otherJointKd_.resize(otherJointIndices_.size());

  scalar_t defaultKp = 150.0;
  scalar_t defaultKd = 8.0;
  absl::flat_hash_map<std::string, std::pair<scalar_t, scalar_t>> jointGainsMap;

  if (!pdGainsFile.empty() && std::filesystem::exists(pdGainsFile)) {
    try {
      YAML::Node root = YAML::LoadFile(pdGainsFile);
      if (root["default_gains"]) {
        if (root["default_gains"]["kp"]) defaultKp = root["default_gains"]["kp"].as<scalar_t>();
        if (root["default_gains"]["kd"]) defaultKd = root["default_gains"]["kd"].as<scalar_t>();
      }
      if (root["joint_gains"]) {
        const YAML::Node jointGains = root["joint_gains"];
        for (YAML::const_iterator kv = jointGains.begin(); kv != jointGains.end(); ++kv) {
          std::string jname = kv->first.as<std::string>();
          scalar_t kp = defaultKp;
          scalar_t kd = defaultKd;
          if (kv->second["kp"]) kp = kv->second["kp"].as<scalar_t>();
          if (kv->second["kd"]) kd = kv->second["kd"].as<scalar_t>();
          jointGainsMap[jname] = {kp, kd};
        }
      }
      LOG(INFO) << "[WBMpcMrtJointController] Loaded joint PD gains from " << pdGainsFile;
    } catch (const std::exception& e) {
      LOG(WARNING) << "[WBMpcMrtJointController] Warning: Failed to parse " << pdGainsFile << ": " << e.what();
    }
  }

  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    const std::string& jname = modelSettings.mpcModelJointNames[i];
    const absl::flat_hash_map<std::string, std::pair<scalar_t, scalar_t>>::const_iterator it = jointGainsMap.find(jname);
    if (it != jointGainsMap.end()) {
      mpcJointKp_[i] = it->second.first;
      mpcJointKd_[i] = it->second.second;
    } else {
      mpcJointKp_[i] = defaultKp;
      mpcJointKd_[i] = defaultKd;
    }
  }

  for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
    const std::string& jname = modelSettings.fixedJointNames[i];
    const absl::flat_hash_map<std::string, std::pair<scalar_t, scalar_t>>::const_iterator it = jointGainsMap.find(jname);
    if (it != jointGainsMap.end()) {
      otherJointKp_[i] = it->second.first;
      otherJointKd_[i] = it->second.second;
    } else {
      otherJointKp_[i] = defaultKp * 0.3;
      otherJointKd_[i] = defaultKd * 0.3;
    }
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::subscribePdGains(rclcpp::Node::SharedPtr node) {
  rclcpp::QoS qos(1);
  qos.best_effort();
  pdGainsSubscription_ =
      node->create_subscription<std_msgs::msg::String>("/pd_gains_updates", qos, [this](const std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(pdGainsPendingMutex_);
        pdGainsPendingYamlContent_ = msg->data;
        hasNewPdGainsTopicData_.store(true);
        LOG(INFO) << "[WBMpcMrtJointController] topicCallback received " << msg->data.size() << " chars";
      });
  LOG(INFO) << "[WBMpcMrtJointController] Subscribed to /pd_gains_updates topic.";
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

WBMpcMrtJointController::~WBMpcMrtJointController() {
  // Signal the solver thread to terminate. It used to loop on `while (true)` and ignore this flag, so the join below
  // never returned and a controller could not be destroyed.
  terminateThread_.store(true);

  // Wait for the solver thread to finish if it's joinable
  if (solver_worker_.joinable()) {
    solver_worker_.join();
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::startMpcThread(const ::robot::model::RobotState& initRobotState) {
  updateMpcObservation(currentMpcObservation_, initRobotState);
  // Set observation to MPC
  mcpMrtInterface_.setCurrentObservation(currentMpcObservation_);
  solver_worker_ = std::jthread(&WBMpcMrtJointController::solverWorker, this);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::updateMpcState(vector_t& mpcState, const ::robot::model::RobotState& robotState) {
  mpcRobotModel_.setBasePosition(mpcState, robotState.getRootPositionInWorldFrame());
  mpcRobotModel_.setBaseOrientationEulerZYX(mpcState, quaternionToEulerZYX(robotState.getRootRotationLocalToWorldFrame()));

  mpcRobotModel_.setJointAngles(mpcState, robotState.getJointPositions(mpcJointIndices_));

  // currently we send local angular and linear velocity
  mpcRobotModel_.setBaseLinearVelocity(mpcState,
                                       robotState.getRootRotationLocalToWorldFrame() * robotState.getRootLinearVelocityInLocalFrame());
  mpcRobotModel_.setBaseOrientationEulerZYXDerivatives(
      mpcState, getEulerAnglesZyxDerivativesFromLocalAngularVelocity<scalar_t>(mpcRobotModel_.getBaseOrientationEulerZYX(mpcState),
                                                                               robotState.getRootAngularVelocityInLocalFrame()));

  vector_t dummyInput = vector_t::Zero(mpcRobotModel_.getInputDim());
  mpcRobotModel_.setJointVelocities(mpcState, dummyInput, robotState.getJointVelocities(mpcJointIndices_));
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
  const std::vector<bool> measuredContacts = contactEstimator_->estimateContactFlags(robotState);
  if (measuredContacts.size() != N_CONTACTS) {
    throw std::runtime_error("[WBMpcMrtJointController] contact estimator '" + contactEstimator_->getName() + "' reported " +
                             std::to_string(measuredContacts.size()) + " contact flags, expected " + std::to_string(N_CONTACTS));
  }
  std::copy(measuredContacts.begin(), measuredContacts.end(), measuredContactFlags_.begin());
  mpcObservation.mode = stanceLeg2ModeNumber(measuredContactFlags_);
  contactWrenchGate_.update(mpcObservation.time, measuredContactFlags_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::setContactWrenchGateConfig(const ContactWrenchGate::Config& config) {
  contactWrenchGate_.setConfig(config);
  LOG(INFO) << "[WBMpcMrtJointController] contact wrench gate: debounceTime=" << config.debounceTime << " s, rampTime=" << config.rampTime
            << " s.";
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::setContactEstimator(std::shared_ptr<::robot::model::ContactEstimator> contactEstimator) {
  contactEstimator_ = contactEstimator ? std::move(contactEstimator) : std::make_shared<::robot::model::RobotStateContactEstimator>();
  LOG(INFO) << "[WBMpcMrtJointController] measured contact state from " << contactEstimator_->getName() << ".";
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::setControlMode(absl::string_view mode) {
  const std::string newMode(mode);
  if (newMode == controlMode_) return;
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
  controlMode_ = newMode;
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
  LOG(WARNING) << "[WBMpcMrtJointController] The observation time went backwards by " << rewind << " s, to " << currentMpcObservation_.time
               << " s. The MPC is reset, and WB_MPC holds the robot with the JOINT_PD action until a policy planned on the new clock "
               << "is in use.";
  // SAFETY keeps the time it has already decayed for: restarting its clock would put the gains back to full authority.
  if (safetyDecayStartTime_ >= 0.0) safetyDecayStartTime_ -= rewind;
  previousObservationTime_ = currentMpcObservation_.time;
  // MpcResetSupervisor::observeTime() has requested the reset; the hold is all that is left to arm.
  armHold(false);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t WBMpcMrtJointController::computeGravityCompensation(const ::robot::model::RobotState& robotState) {
  vector_t state = vector_t::Zero(mpcRobotModel_.getStateDim());
  updateMpcState(state, robotState);
  const vector_t q = mpcRobotModel_.getGeneralizedCoordinates(state);
  const PinocchioInterface::Model& model = pinocchioInterface_.getModel();
  PinocchioInterface::Data& data = pinocchioInterface_.getData();
  pinocchio::nonLinearEffects(model, data, q, vector_t::Zero(model.nv));
  return data.nle.tail(mpcRobotModel_.getJointDim());
}

scalar_t WBMpcMrtJointController::nominalJointPosition(const ::robot::model::RobotState& robotState, size_t index) const {
  return index < nominalJointPositions_.size() ? nominalJointPositions_[index] : robotState.getJointPosition(index);
}

void WBMpcMrtJointController::fillZeroTorqueAction(const ::robot::model::RobotState& robotState,
                                                   ::robot::model::RobotJointAction& robotJointAction) {
  const std::array<const std::vector<size_t>*, 2> jointGroups{&mpcJointIndices_, &otherJointIndices_};
  for (const std::vector<size_t>* indices : jointGroups) {
    for (size_t index : *indices) {
      robot::model::JointAction& action = robotJointAction.at(index).value();
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
  const vector_t gravityTorques = computeGravityCompensation(robotState);
  for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
    const size_t index = mpcJointIndices_[i];
    robot::model::JointAction& action = robotJointAction.at(index).value();
    action.q_des = nominalJointPosition(robotState, index);
    action.qd_des = 0.0;
    action.kp = mpcJointKp_[i];
    action.kd = mpcJointKd_[i];
    action.feed_forward_effort = gravityTorques[i];
  }
  for (size_t i = 0; i < otherJointIndices_.size(); i++) {
    const size_t index = otherJointIndices_[i];
    robot::model::JointAction& action = robotJointAction.at(index).value();
    action.q_des = nominalJointPosition(robotState, index);
    action.qd_des = 0.0;
    action.kp = otherJointKp_[i];
    action.kd = otherJointKd_[i];
    action.feed_forward_effort = 0.0;
  }
}

void WBMpcMrtJointController::fillGravityCompAction(const ::robot::model::RobotState& robotState,
                                                    ::robot::model::RobotJointAction& robotJointAction) {
  // The base-held gravity torques: GRAVITY_COMP is operated with the robot suspended from the gantry (see the centroidal
  // controller's fillGravityCompAction for why this is not enough for a robot bearing its own weight).
  const vector_t gravityTorques = computeGravityCompensation(robotState);
  for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
    const size_t index = mpcJointIndices_[i];
    robot::model::JointAction& action = robotJointAction.at(index).value();
    action.q_des = robotState.getJointPosition(index);
    action.qd_des = 0.0;
    action.kp = 0.0;
    action.kd = mpcJointKd_[i] * 0.2;  // Soft damping to prevent free-fall oscillation
    action.feed_forward_effort = gravityTorques[i];
  }
  for (size_t i = 0; i < otherJointIndices_.size(); i++) {
    const size_t index = otherJointIndices_[i];
    robot::model::JointAction& action = robotJointAction.at(index).value();
    action.q_des = nominalJointPosition(robotState, index);
    action.qd_des = 0.0;
    action.kp = otherJointKp_[i] * 0.5;
    action.kd = otherJointKd_[i];
    action.feed_forward_effort = 0.0;
  }
}

void WBMpcMrtJointController::fillSafetyAction(const ::robot::model::RobotState& robotState,
                                               ::robot::model::RobotJointAction& robotJointAction) {
  // The posture held is the MEASURED one at entry: SAFETY is entered when something has already gone wrong, and a
  // return to the nominal stance at full gain would be a lunge, not a safe stop.
  if (safetyDecayStartTime_ < 0.0) {
    safetyHoldMpcJointPositions_.resize(mpcJointIndices_.size());
    for (size_t i = 0; i < mpcJointIndices_.size(); i++) safetyHoldMpcJointPositions_[i] = robotState.getJointPosition(mpcJointIndices_[i]);
    safetyHoldOtherJointPositions_.resize(otherJointIndices_.size());
    for (size_t i = 0; i < otherJointIndices_.size(); i++) {
      safetyHoldOtherJointPositions_[i] = robotState.getJointPosition(otherJointIndices_[i]);
    }
    safetyDecayStartTime_ = currentMpcObservation_.time;
    LOG(WARNING) << "SAFETY mode entered: holding the measured posture and decaying the joint PD gains to zero with a "
                 << safetyDecayTimeConstant_ << " s time constant.";
  }
  const scalar_t alpha = safety_decay::factor(currentMpcObservation_.time - safetyDecayStartTime_, safetyDecayTimeConstant_);
  for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
    robot::model::JointAction& action = robotJointAction.at(mpcJointIndices_[i]).value();
    action.q_des = safetyHoldMpcJointPositions_[i];
    action.qd_des = 0.0;
    action.kp = alpha * mpcJointKp_[i];
    action.kd = alpha * mpcJointKd_[i];
    action.feed_forward_effort = 0.0;
  }
  for (size_t i = 0; i < otherJointIndices_.size(); i++) {
    robot::model::JointAction& action = robotJointAction.at(otherJointIndices_[i]).value();
    action.q_des = safetyHoldOtherJointPositions_[i];
    action.qd_des = 0.0;
    action.kp = alpha * otherJointKp_[i];
    action.kd = alpha * otherJointKd_[i];
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

void WBMpcMrtJointController::computeJointControlAction(scalar_t time,
                                                        const ::robot::model::RobotState& robotState,
                                                        ::robot::model::RobotJointAction& robotJointAction) {
  // Check for ROS topic-based PD gains update (takes priority over file-watcher)
  if (hasNewPdGainsTopicData_.load()) {
    hasNewPdGainsTopicData_.store(false);
    std::string yamlContent;
    {
      std::lock_guard<std::mutex> lock(pdGainsPendingMutex_);
      yamlContent = std::move(pdGainsPendingYamlContent_);
    }
    if (!yamlContent.empty()) {
      // Write to a temp file and call loadPdGains
      std::string tempFile = pdGainsFile_ + ".live.yaml";
      {
        std::ofstream ofs(tempFile);
        ofs << yamlContent;
      }
      loadPdGains(tempFile, modelSettings_);
      LOG(INFO) << "[WBMpcMrtJointController] Applied PD gains from topic.";
    }
  }

  // Hot-reload Joint PD Gains at 1Hz (assuming 500Hz control loop)
  if (!pdGainsFile_.empty() && fileCheckCounter_++ % 500 == 0) {
    std::error_code ec;
    const std::filesystem::file_time_type last_write = std::filesystem::last_write_time(pdGainsFile_, ec);
    if (!ec && last_write != pdGainsLastWriteTime_) {
      pdGainsLastWriteTime_ = last_write;
      loadPdGains(pdGainsFile_, modelSettings_);
    }
  }

  // Set observation to MPC, in every mode, so that the solver keeps tracking the robot and its clock.
  updateMpcObservation(currentMpcObservation_, robotState);
  const scalar_t clockRewind = resetSupervisor_.observeTime(currentMpcObservation_.time);
  if (clockRewind > 0.0) handleClockRewind(clockRewind);
  mcpMrtInterface_.setCurrentObservation(currentMpcObservation_);

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
  mcpMrtInterface_.updatePolicy();
  const bool postResetPolicyActive = mcpMrtInterface_.isActivePolicyCurrent() && !resetSupervisor_.hasOutstandingReset();
  policyActivated_.store(postResetPolicyActive);

  // A solver that keeps failing leaves a stale policy in use: hold the robot with the JOINT_PD action instead.
  if (!resetSupervisor_.isHealthy() && !awaitingPostResetPolicy_.load()) {
    holdGravityComp_.store(false);
    awaitingPostResetPolicy_.store(true);
  }
  if (awaitingPostResetPolicy_.load()) {
    if (!postResetPolicyActive || !resetSupervisor_.isHealthy()) {
      fillHoldAction(robotState, robotJointAction);
      previousObservationTime_ = currentMpcObservation_.time;
      return;
    }
    awaitingPostResetPolicy_.store(false);
  }

  vector_t mpcPolicyState;
  vector_t mpcPolicyInput;
  size_t mpcPolicyMode;

  if (mcpMrtInterface_.initialPolicyReceived()) {
    // Compute actual sim dt from elapsed simulation time (respects RTF)
    scalar_t simDt = currentMpcObservation_.time - previousObservationTime_;
    // Clamp to sane range: avoid zero/negative (first call, time resets) and excessive lookahead
    simDt = std::clamp(simDt, 0.001, 0.02);

    // Evaluate policy with feedback if activated in config
    mcpMrtInterface_.evaluatePolicy(currentMpcObservation_.time + simDt, currentMpcObservation_.state, mpcPolicyState, mpcPolicyInput,
                                    mpcPolicyMode);
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

    for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
      size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action = robotJointAction.at(index).value();

      action.q_des = mpc_q_desired[i];
      action.qd_des = mpc_qd_desired[i];
      action.kp = mpcJointKp_[i];
      action.kd = mpcJointKd_[i];
      action.feed_forward_effort = mpcJointTorques[i];
    };

    static size_t vizCounter = 0;
    if (visualizerPtr_ != nullptr && (++vizCounter % 16 == 0)) {
      visualizerPtr_->update(currentMpcObservation_, mcpMrtInterface_.getPolicy(), mcpMrtInterface_.getCommand());
    }
  }

  else {
    LOG_EVERY_N_SEC(INFO, 1.0) << "Apply weight compensating torque...";
    //   Apply weight compensated input around current state
    mpcPolicyState = currentMpcObservation_.state;
    // The weight is carried by the feet measured in contact; with none (the robot hangs on the gantry) no contact force
    // is compensated and the feedforward is the gravity and Coriolis term of the free legs, the base being held
    // (computeBaseHeldJointTorques; computeJointTorques would let the base fall freely).
    mpcPolicyInput = weightCompensatingInput(pinocchioInterface_, measuredContactFlags_, mpcRobotModel_);
    latestPolicyInput_ = mpcPolicyInput;
    vector_t weightCompensatingTorques =
        computeBaseHeldJointTorques<scalar_t>(mpcPolicyState, mpcPolicyInput, pinocchioInterface_, mpcRobotModel_);

    for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
      size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action = robotJointAction.at(index).value();

      action.q_des = robotState.getJointPosition(index);
      action.qd_des = 0.0;
      action.kp = mpcJointKp_[i];
      action.kd = mpcJointKd_[i];
      action.feed_forward_effort = weightCompensatingTorques[i];
    };
  }

  for (size_t i = 0; i < otherJointIndices_.size(); i++) {
    size_t index = otherJointIndices_[i];
    robot::model::JointAction& action = robotJointAction.at(index).value();

    action.q_des = 0;
    action.qd_des = 0;
    action.kp = otherJointKp_[i];
    action.kd = otherJointKd_[i];
    action.feed_forward_effort = 0.0;
  };

  // Track observation time for next call's dt computation
  previousObservationTime_ = currentMpcObservation_.time;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::solverWorker() {
  const SystemCoreAllocation coreAlloc = ocs2::humanoid::getDefaultCoreAllocation();
  ocs2::humanoid::setThreadCpuAffinity(coreAlloc.mpcCores, pthread_self(), "WB MPC Solver Thread");

  resetMpcToCurrentObservation(/*full=*/true, "start-up");
  LOG(INFO) << "MPC is reset. NMPC solver started!";

  size_t slowWarningCount = 0;
  while (!terminateThread_.load()) {
    const std::chrono::steady_clock::time_point targetTimeForNextIteration =
        std::chrono::steady_clock::now() + std::chrono::microseconds(mpcDeltaTMicroSeconds_);

    // Serve a requested reset (a mode change, a discontinuity of the plant, a failed solve) before solving again.
    if (const std::optional<MpcResetSupervisor::ResetTicket> ticket = resetSupervisor_.takeResetRequest()) {
      resetMpcToCurrentObservation(ticket->full, "requested");
      resetSupervisor_.completeReset(*ticket);
    }

    // A failed solve used to keep the previous solution and retry at the solve rate for ever, logging each failure. It
    // now requests a reset and, once the failures persist, a pause before the next attempt; the supervisor logs once.
    const absl::Status mpcStatus = mcpMrtInterface_.advanceMpc();
    const std::chrono::duration<scalar_t> retryDelay = resetSupervisor_.onSolveResult(mpcStatus);
    if (retryDelay.count() > 0.0) {
      resetSupervisor_.waitBeforeRetry(retryDelay, [this]() { return terminateThread_.load(); });
      continue;
    }

    if (!realtime_) {
      const std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
      if (currentTime > targetTimeForNextIteration) {
        const int64_t delay = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - targetTimeForNextIteration).count();
        if (delay > 1000 && (++slowWarningCount % 20 == 0)) {
          LOG(WARNING) << "MPC loop running slow by " << delay << " microseconds.";
        }
      } else {
        // Sleep in case sim loop is faster than specified
        std::this_thread::sleep_until(targetTimeForNextIteration);
      }
    }
  }
  LOG(INFO) << "Shutting down NMPC";
}

void WBMpcMrtJointController::resetMpcToCurrentObservation(bool full, absl::string_view reason) {
  const SystemObservation observation = mcpMrtInterface_.getCurrentObservation();
  if (full) {
    mcpMrtInterface_.resetMpcNode(currentObservationToResetTrajectory(observation));
  } else {
    mcpMrtInterface_.resetMpcSolver(currentObservationToResetTrajectory(observation));
  }
  // While the solver keeps failing it is reset before every attempt; the supervisor has said so once already.
  if (resetSupervisor_.isHealthy()) {
    LOG(INFO) << (full ? "MPC reset" : "MPC solver reset") << " to the observation at t = " << observation.time << " s (" << reason << ").";
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

TargetTrajectories WBMpcMrtJointController::currentObservationToResetTrajectory(const SystemObservation& currentObservation) {
  vector_t targetState = currentObservation.state;

  // zero out velocities
  targetState.tail(mpcRobotModel_.getGenCoordinatesDim()) = vector_t::Zero(mpcRobotModel_.getGenCoordinatesDim());

  // zero out pitch + roll angles
  targetState.segment<2>(4) = vector_t::Zero(2);

  // The weight carried on both feet, like the centroidal controller's reset target: a zero input asked the first solve
  // after a reset to hold the robot up with no contact force at all. Two nodes, so that the target holds over the
  // horizon rather than being a single knot extrapolated.
  const vector_t targetInput = weightCompensatingInput(pinocchioInterface_, {true, true}, mpcRobotModel_);
  const scalar_t t0 = currentObservation.time;
  const TargetTrajectories resetTargetTrajectories({t0, t0 + 2.0}, {targetState, targetState}, {targetInput, targetInput});

  if (resetSupervisor_.isHealthy()) {
    LOG(INFO) << "Resetting MPC to current state: base pose " << mpcRobotModel_.getBasePose(targetState).transpose();
  }
  return resetTargetTrajectories;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::optional<contact_flag_t> WBMpcMrtJointController::getPlannedContactFlags(scalar_t time) const {
  if (!policyActivated_.load()) return std::nullopt;
  return modeNumber2StanceLeg(mcpMrtInterface_.getPolicy().modeSchedule_.modeAtTime(time));
}

}  // namespace ocs2::humanoid
