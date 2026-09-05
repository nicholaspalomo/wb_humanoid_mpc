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

#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"

#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <humanoid_common_mpc/common/ThreadAffinity.h>
#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>
#include <humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h>
#include <humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h>
#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"

#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "absl/log/log.h"
#include "absl/status/status.h"

namespace ocs2::humanoid {

WBMpcMrtJointController::WBMpcMrtJointController(const ::robot::model::RobotDescription& robotDescription,
                                                 const ModelSettings& modelSettings,
                                                 MPC_BASE& mpc,
                                                 PinocchioInterface pinocchioInterface,
                                                 scalar_t mpcDesiredFrequency,
                                                 std::shared_ptr<DummyObserver> rVizVisualizerPtr,
                                                 const std::string& pdGainsFile)
    : mcpMrtInterface_(mpc),
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
  std::unordered_map<std::string, std::pair<scalar_t, scalar_t>> jointGainsMap;

  if (!pdGainsFile.empty() && std::filesystem::exists(pdGainsFile)) {
    try {
      YAML::Node root = YAML::LoadFile(pdGainsFile);
      if (root["default_gains"]) {
        if (root["default_gains"]["kp"]) defaultKp = root["default_gains"]["kp"].as<scalar_t>();
        if (root["default_gains"]["kd"]) defaultKd = root["default_gains"]["kd"].as<scalar_t>();
      }
      if (root["joint_gains"]) {
        for (const auto& kv : root["joint_gains"]) {
          std::string jname = kv.first.as<std::string>();
          scalar_t kp = defaultKp;
          scalar_t kd = defaultKd;
          if (kv.second["kp"]) kp = kv.second["kp"].as<scalar_t>();
          if (kv.second["kd"]) kd = kv.second["kd"].as<scalar_t>();
          jointGainsMap[jname] = {kp, kd};
        }
      }
      std::cout << "[WBMpcMrtJointController] Loaded joint PD gains from " << pdGainsFile << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "[WBMpcMrtJointController] Warning: Failed to parse " << pdGainsFile << ": " << e.what() << std::endl;
    }
  }

  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    const std::string& jname = modelSettings.mpcModelJointNames[i];
    auto it = jointGainsMap.find(jname);
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
    auto it = jointGainsMap.find(jname);
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

WBMpcMrtJointController::~WBMpcMrtJointController() {
  // Signal the solver thread to terminate
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
  std::vector<bool> configContacts = robotState.getContactFlags();
  assert(configContacts.size() == 2);
  contact_flag_t contactFlags;
  std::copy(configContacts.begin(), configContacts.end(), contactFlags.begin());
  mpcObservation.mode = stanceLeg2ModeNumber(contactFlags);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::computeJointControlAction(scalar_t time,
                                                        const ::robot::model::RobotState& robotState,
                                                        ::robot::model::RobotJointAction& robotJointAction) {
  // Hot-reload Joint PD Gains at 1Hz (assuming 500Hz control loop)
  if (!pdGainsFile_.empty() && fileCheckCounter_++ % 500 == 0) {
    std::error_code ec;
    auto last_write = std::filesystem::last_write_time(pdGainsFile_, ec);
    if (!ec && last_write != pdGainsLastWriteTime_) {
      pdGainsLastWriteTime_ = last_write;
      loadPdGains(pdGainsFile_, modelSettings_);
    }
  }

  // Set observation to MPC
  updateMpcObservation(currentMpcObservation_, robotState);
  mcpMrtInterface_.setCurrentObservation(currentMpcObservation_);

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

    vector_t mpcJointTorques = computeJointTorques<scalar_t>(mpcPolicyState, mpcPolicyInput, pinocchioInterface_, mpcRobotModel_);
    vector_t mpc_q_desired = mpcRobotModel_.getJointAngles(mpcPolicyState);
    vector_t mpc_qd_desired = mpcRobotModel_.getJointVelocities(mpcPolicyState, mpcPolicyInput);

    // std::cout << "mpcJointTorques: " << mpcJointTorques.transpose() << std::endl;

    for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
      size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action = robotJointAction.at(index).value();

      action.q_des = mpc_q_desired[i];
      action.qd_des = mpc_qd_desired[i];
      action.kp = mpcJointKp_[i];
      action.kd = mpcJointKd_[i];
      action.feed_forward_effort = mpcJointTorques[i];

      // std::cerr << "MPCtorque!: " << mpcJointTorques[i] << std::endl;
    };

    if (visualizerPtr_ != nullptr) {
      visualizerPtr_->update(currentMpcObservation_, mcpMrtInterface_.getPolicy(), mcpMrtInterface_.getCommand());
    }
  }

  else {
    std::cerr << "Apply weight compensating torque..." << std::endl;
    //   Apply weight compensated input around current state
    mpcPolicyState = currentMpcObservation_.state;
    mpcPolicyInput = weightCompensatingInput(pinocchioInterface_, {true, true}, mpcRobotModel_);
    latestPolicyInput_ = mpcPolicyInput;
    vector_t weightCompensatingTorques = computeJointTorques<scalar_t>(mpcPolicyState, mpcPolicyInput, pinocchioInterface_, mpcRobotModel_);

    for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
      size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action = robotJointAction.at(index).value();

      action.q_des = 0;
      action.qd_des = 0;
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
  const auto coreAlloc = ocs2::humanoid::getDefaultCoreAllocation();
  ocs2::humanoid::setThreadCpuAffinity(coreAlloc.mpcCores, pthread_self(), "WB MPC Solver Thread");

  mcpMrtInterface_.resetMpcNode(currentObservationToResetTrajectory(mcpMrtInterface_.getCurrentObservation()));
  std::cerr << "MPC is reset. NMPC solver started!" << std::endl;

  size_t slowWarningCount = 0;
  while (true) {
    auto targetTimeForNextIteration = std::chrono::steady_clock::now() + std::chrono::microseconds(mpcDeltaTMicroSeconds_);

    absl::Status mpcStatus = mcpMrtInterface_.advanceMpc();
    if (!mpcStatus.ok()) {
      // MPC solver failed — log and continue with previous solution.
      LOG(ERROR) << "MPC solver error in WB worker: " << mpcStatus.message() << " — retaining previous solution and retrying.";
    } else {
      // Update active policy buffer
      mcpMrtInterface_.updatePolicy();
    }

    if (!realtime_) {
      auto currentTime = std::chrono::steady_clock::now();
      if (currentTime > targetTimeForNextIteration) {
        auto delay = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - targetTimeForNextIteration).count();
        if (delay > 1000 && (++slowWarningCount % 20 == 0)) {
          std::cerr << "Warning: MPC loop running slow by " << delay << " microseconds." << std::endl;
        }
      } else {
        // Sleep in case sim loop is faster than specified
        std::this_thread::sleep_until(targetTimeForNextIteration);
      }
    }
  }
  std::cerr << "Shutting down NMPC" << std::endl;
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

  const TargetTrajectories resetTargetTrajectories({currentObservation.time}, {targetState},
                                                   {vector_t::Zero(currentObservation.input.size())});

  std::cerr << "Resetting MPC to current state: \n" << targetState << std::endl;
  return resetTargetTrajectories;
}

}  // namespace ocs2::humanoid
