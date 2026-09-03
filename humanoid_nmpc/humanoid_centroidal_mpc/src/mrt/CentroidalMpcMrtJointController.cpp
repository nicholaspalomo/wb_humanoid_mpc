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

#include <pinocchio/fwd.hpp>

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"

#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include "ocs2_centroidal_model/ModelHelperFunctions.h"

#include "ocs2_centroidal_model/AccessHelperFunctions.h"

#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>
#include <humanoid_common_mpc/common/ThreadAffinity.h>
#include <humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h>
#include <humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h>
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

#include <yaml-cpp/yaml.h>
#include <filesystem>

// Pinocchio algorithm headers (must come after pinocchio/fwd.hpp)
#include <pinocchio/algorithm/rnea.hpp>

namespace ocs2::humanoid {

CentroidalMpcMrtJointController::CentroidalMpcMrtJointController(const ::robot::model::RobotDescription& robotDescription,
                                                                 const ModelSettings& modelSettings,
                                                                 const CentroidalMpcRobotModel<scalar_t>& mpcRobotModel,
                                                                 MPC_BASE& mpc,
                                                                 PinocchioInterface pinocchioInterface,
                                                                 scalar_t mpcDesiredFrequency,
                                                                 std::shared_ptr<DummyObserver> rVizVisualizerPtr,
                                                                 const std::string& pdGainsFile)
    : mcpMrtInterface_(mpc),
      pinocchioInterface_(pinocchioInterface),
      mpcRobotModelPtr_(mpcRobotModel.clone()),
      mpcDeltaTMicroSeconds_(1000000 / mpcDesiredFrequency),
      realtime_(mpcDesiredFrequency <= 0),
      visualizerPtr_(rVizVisualizerPtr),
      inverse_dynamics_kp_(mpcRobotModel.getJointDim()),
      inverse_dynamics_kd_(mpcRobotModel.getJointDim()) {
  mpcJointIndices_ = robotDescription.getJointIndices(modelSettings.mpcModelJointNames);
  otherJointIndices_ = robotDescription.getJointIndices(modelSettings.fixedJointNames);
  currentMpcObservation_.state = vector_t::Zero(mpcRobotModelPtr_->getStateDim());
  currentMpcObservation_.input = vector_t::Zero(mpcRobotModelPtr_->getInputDim());

  // Currently set to 0. There is still a bug in the momentum computation of the inverse dynamics.
  inverse_dynamics_kp_.fill(0.0);
  inverse_dynamics_kd_.fill(0.0);

  loadPdGains(pdGainsFile, modelSettings);
}

void CentroidalMpcMrtJointController::loadPdGains(const std::string& pdGainsFile, const ModelSettings& modelSettings) {
  mpcJointKp_.resize(mpcJointIndices_.size());
  mpcJointKd_.resize(mpcJointIndices_.size());
  mpcJointTorqueLimit_.resize(mpcJointIndices_.size());
  otherJointKp_.resize(otherJointIndices_.size());
  otherJointKd_.resize(otherJointIndices_.size());
  otherJointTorqueLimit_.resize(otherJointIndices_.size());

  scalar_t defaultKp = 250.0;
  scalar_t defaultKd = 15.0;
  scalar_t defaultTorqueLimit = 500.0;
  std::unordered_map<std::string, std::tuple<scalar_t, scalar_t, scalar_t>> jointGainsMap;

  if (!pdGainsFile.empty() && std::filesystem::exists(pdGainsFile)) {
    try {
      YAML::Node root = YAML::LoadFile(pdGainsFile);
      if (root["default_gains"]) {
        if (root["default_gains"]["kp"]) defaultKp = root["default_gains"]["kp"].as<scalar_t>();
        if (root["default_gains"]["kd"]) defaultKd = root["default_gains"]["kd"].as<scalar_t>();
        if (root["default_gains"]["torque_limit"]) defaultTorqueLimit = root["default_gains"]["torque_limit"].as<scalar_t>();
      }
      if (root["joint_gains"]) {
        for (const auto& kv : root["joint_gains"]) {
          std::string jname = kv.first.as<std::string>();
          scalar_t kp = defaultKp;
          scalar_t kd = defaultKd;
          scalar_t tl = defaultTorqueLimit;
          if (kv.second["kp"]) kp = kv.second["kp"].as<scalar_t>();
          if (kv.second["kd"]) kd = kv.second["kd"].as<scalar_t>();
          if (kv.second["torque_limit"]) tl = kv.second["torque_limit"].as<scalar_t>();
          jointGainsMap[jname] = {kp, kd, tl};
        }
      }
      std::cout << "[CentroidalMpcMrtJointController] Loaded joint PD gains from " << pdGainsFile << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "[CentroidalMpcMrtJointController] Warning: Failed to parse " << pdGainsFile << ": " << e.what() << std::endl;
    }
  }

  for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
    const std::string& jname = modelSettings.mpcModelJointNames[i];
    auto it = jointGainsMap.find(jname);
    if (it != jointGainsMap.end()) {
      mpcJointKp_[i] = std::get<0>(it->second);
      mpcJointKd_[i] = std::get<1>(it->second);
      mpcJointTorqueLimit_[i] = std::get<2>(it->second);
    } else {
      mpcJointKp_[i] = defaultKp;
      mpcJointKd_[i] = defaultKd;
      mpcJointTorqueLimit_[i] = defaultTorqueLimit;
    }
  }

  for (size_t i = 0; i < otherJointIndices_.size(); ++i) {
    const std::string& jname = modelSettings.fixedJointNames[i];
    auto it = jointGainsMap.find(jname);
    if (it != jointGainsMap.end()) {
      otherJointKp_[i] = std::get<0>(it->second);
      otherJointKd_[i] = std::get<1>(it->second);
      otherJointTorqueLimit_[i] = std::get<2>(it->second);
    } else {
      otherJointKp_[i] = defaultKp * 0.3;
      otherJointKd_[i] = defaultKd * 0.3;
      otherJointTorqueLimit_[i] = defaultTorqueLimit;
    }
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

CentroidalMpcMrtJointController::~CentroidalMpcMrtJointController() {
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

void CentroidalMpcMrtJointController::startMpcThread(const ::robot::model::RobotState& initRobotState) {
  updateMpcObservation(currentMpcObservation_, initRobotState);
  // Set observation to MPC
  mcpMrtInterface_.setCurrentObservation(currentMpcObservation_);
  solver_worker_ = std::jthread(&CentroidalMpcMrtJointController::solverWorker, this);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void CentroidalMpcMrtJointController::updateMpcState(vector_t& mpcState, const ::robot::model::RobotState& robotState) {
  const auto& info = mpcRobotModelPtr_->getCentroidalModelInfo();

  const vector3_t euler_zyx = quaternionToEulerZYX(robotState.getRootRotationLocalToWorldFrame());

  vector_t qPinocchio(info.generalizedCoordinatesNum);
  qPinocchio.head<3>() = robotState.getRootPositionInWorldFrame();
  qPinocchio.segment<3>(3) = euler_zyx;
  qPinocchio.tail(mpcRobotModelPtr_->getJointDim()) = robotState.getJointPositions(mpcJointIndices_);

  vector_t vPinocchio(info.generalizedCoordinatesNum);
  vPinocchio.head<3>() = robotState.getRootRotationLocalToWorldFrame() * robotState.getRootLinearVelocityInLocalFrame();
  vPinocchio.segment<3>(3) =
      getEulerAnglesZyxDerivativesFromLocalAngularVelocity<scalar_t>(euler_zyx, robotState.getRootAngularVelocityInLocalFrame());
  vPinocchio.tail(mpcRobotModelPtr_->getJointDim()) = robotState.getJointVelocities(mpcJointIndices_);

  updateCentroidalDynamics(pinocchioInterface_, info, qPinocchio);
  const auto& A = getCentroidalMomentumMatrix(pinocchioInterface_);

  centroidal_model::getNormalizedMomentum(mpcState, info).noalias() = A * vPinocchio / info.robotMass;
  centroidal_model::getGeneralizedCoordinates(mpcState, info) = qPinocchio;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void CentroidalMpcMrtJointController::updateMpcObservation(ocs2::SystemObservation& mpcObservation,
                                                           const ::robot::model::RobotState& robotState) {
  updateMpcState(mpcObservation.state, robotState);
  mpcObservation.time = robotState.getTime();
  mpcObservation.input = vector_t::Zero(mpcRobotModelPtr_->getInputDim());
  mpcObservation.input.tail(mpcRobotModelPtr_->getJointDim()) = robotState.getJointVelocities(mpcJointIndices_, 0.0);
  std::vector<bool> configContacts = robotState.getContactFlags();
  assert(configContacts.size() == 2);
  contact_flag_t contactFlags;
  std::copy(configContacts.begin(), configContacts.end(), contactFlags.begin());
  mpcObservation.mode = stanceLeg2ModeNumber(contactFlags);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void CentroidalMpcMrtJointController::computeJointControlAction(scalar_t time,
                                                                const ::robot::model::RobotState& robotState,
                                                                ::robot::model::RobotJointAction& robotJointAction) {
  // Always update MPC observation so the solver continues tracking robot state and time in all modes.
  updateMpcObservation(currentMpcObservation_, robotState);
  mcpMrtInterface_.setCurrentObservation(currentMpcObservation_);

  // JOINT_PD mode: PD tracking to nominal positions + Pinocchio gravity compensation.
  // This code path is shared between sim and real hardware.
  if (controlMode_ == "JOINT_PD") {
    vector_t gravTorques = computeGravityCompensation(robotState);

    for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
      size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action = robotJointAction.at(index).value();
      action.q_des = nominalJointPositions_.empty() ? 0.0 : nominalJointPositions_[index];
      action.qd_des = 0.0;
      action.kp = mpcJointKp_[i];
      action.kd = mpcJointKd_[i];
      action.feed_forward_effort = gravTorques[i];
    }

    for (size_t i = 0; i < otherJointIndices_.size(); i++) {
      size_t index = otherJointIndices_[i];
      robot::model::JointAction& action = robotJointAction.at(index).value();
      action.q_des = nominalJointPositions_.empty() ? 0.0 : nominalJointPositions_[index];
      action.qd_des = 0.0;
      action.kp = otherJointKp_[i];
      action.kd = otherJointKd_[i];
      action.feed_forward_effort = 0.0;  // Non-MPC joints don't get gravity comp
    }

    // Debug: print gravity comp torques and position errors (throttled)
    static size_t debugCount = 0;
    if (++debugCount % 500 == 1) {
      std::cerr << "[JOINT_PD] gravTorques: " << gravTorques.transpose() << std::endl;
      for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
        size_t index = mpcJointIndices_[i];
        double q_cur = robotState.getJointPosition(index);
        double q_des = nominalJointPositions_.empty() ? 0.0 : nominalJointPositions_[index];
        if (std::abs(q_des - q_cur) > 0.05) {
          std::cerr << "  joint[" << index << "] err=" << (q_des - q_cur)
                    << " q_cur=" << q_cur << " q_des=" << q_des
                    << " kp=" << mpcJointKp_[i] << " gravFF=" << gravTorques[i] << std::endl;
        }
      }
    }

    return;
  }

  // GRAVITY_COMP mode: Zero-G compliant mode using pure gravity compensation torques + light damping.
  // Limbs can be moved compliantly by hand or external forces.
  if (controlMode_ == "GRAVITY_COMP") {
    vector_t gravTorques = computeGravityCompensation(robotState);

    for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
      size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action = robotJointAction.at(index).value();
      action.q_des = robotState.getJointPosition(index);
      action.qd_des = 0.0;
      action.kp = 0.0;
      action.kd = mpcJointKd_[i] * 0.2;  // Soft damping to prevent free-fall oscillation
      action.feed_forward_effort = std::clamp(gravTorques[i], -mpcJointTorqueLimit_[i], mpcJointTorqueLimit_[i]);
    }

    for (size_t i = 0; i < otherJointIndices_.size(); i++) {
      size_t index = otherJointIndices_[i];
      robot::model::JointAction& action = robotJointAction.at(index).value();
      action.q_des = nominalJointPositions_.empty() ? 0.0 : nominalJointPositions_[index];
      action.qd_des = 0.0;
      action.kp = otherJointKp_[i] * 0.5;
      action.kd = otherJointKd_[i];
      action.feed_forward_effort = 0.0;
    }

    return;
  }

  // Active MPC control path
  mcpMrtInterface_.updatePolicy();

  vector_t mpcPolicyState;
  vector_t mpcPolicyInput;
  size_t mpcPolicyMode;

  if (mcpMrtInterface_.initialPolicyReceived()) {
    // Evaluate policy with feedback if activated in config
    mcpMrtInterface_.evaluatePolicy(currentMpcObservation_.time + 0.005, currentMpcObservation_.state, mpcPolicyState, mpcPolicyInput,
                                    mpcPolicyMode);

    // TODO something seems wrong with the inverse dynamics. You should correct that.
    vector_t mpc_q_j_des = mpcRobotModelPtr_->getJointAngles(mpcPolicyState);
    vector_t mpc_qd_j_des = mpcRobotModelPtr_->getJointVelocities(mpcPolicyState, mpcPolicyInput);
    vector_t q_j = mpcRobotModelPtr_->getJointAngles(currentMpcObservation_.state);
    vector_t qd_j = mpcRobotModelPtr_->getJointVelocities(currentMpcObservation_.state, currentMpcObservation_.input);
    vector_t qdd_j_des = inverse_dynamics_kp_ * (mpc_q_j_des - q_j) + inverse_dynamics_kd_ * (mpc_qd_j_des - qd_j);

    std::array<vector6_t, 2> footWrenches{mpcRobotModelPtr_->getContactWrench(mpcPolicyInput, 0),
                                          mpcRobotModelPtr_->getContactWrench(mpcPolicyInput, 1)};

    // Evaluate inverse dynamics using MPC planned state and input for dynamical consistency with footWrenches
    vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(mpcPolicyState);
    vector_t qd = mpcRobotModelPtr_->getGeneralizedVelocities(mpcPolicyState, mpcPolicyInput);

    vector_t mpcJointTorques = computeJointTorques<scalar_t>(q, qd, qdd_j_des, footWrenches, pinocchioInterface_);

    // std::cout << "mpcJointTorques: " << mpcJointTorques.transpose() << std::endl;

    for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
      size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action = robotJointAction.at(index).value();

      action.q_des = mpc_q_j_des[i];
      action.qd_des = mpc_qd_j_des[i];
      action.kp = mpcJointKp_[i];
      action.kd = mpcJointKd_[i];
      action.feed_forward_effort = std::clamp(mpcJointTorques[i], -mpcJointTorqueLimit_[i], mpcJointTorqueLimit_[i]);
    };

    static size_t mpcDebugCount = 0;
    if (++mpcDebugCount % 200 == 1) {
      std::cerr << "[ACTIVE_MPC] Foot wrenches: L=" << footWrenches[0].transpose()
                << " R=" << footWrenches[1].transpose() << std::endl;
      std::cerr << "[ACTIVE_MPC] Torques: " << mpcJointTorques.transpose() << std::endl;
      for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
        size_t index = mpcJointIndices_[i];
        double q_cur = robotState.getJointPosition(index);
        double q_des = mpc_q_j_des[i];
        if (std::abs(q_des - q_cur) > 0.05 || std::abs(mpcJointTorques[i]) > 100.0) {
          std::cerr << "  mpc_joint[" << index << "] des=" << q_des << " cur=" << q_cur
                    << " err=" << (q_des - q_cur) << " tau=" << mpcJointTorques[i] << std::endl;
        }
      }
    }

    if (visualizerPtr_ != nullptr) {
      try {
        visualizerPtr_->update(currentMpcObservation_, mcpMrtInterface_.getPolicy(), mcpMrtInterface_.getCommand());
      } catch (const std::exception& e) {
        // Suppress transient visualization exceptions during mode switches to protect real-time loop
      }
    }
  }

  else {
    std::cerr << "Apply weight compensating torque..." << std::endl;
    //   Apply weight compensated input around current state
    vector_t qdd_j_des = vector_t::Zero(mpcRobotModelPtr_->getJointDim());
    mpcPolicyInput = weightCompensatingInput(pinocchioInterface_, {true, true}, *mpcRobotModelPtr_);
    std::array<vector6_t, 2> footWrenches{mpcRobotModelPtr_->getContactWrench(mpcPolicyInput, 0),
                                          mpcRobotModelPtr_->getContactWrench(mpcPolicyInput, 1)};
    vector_t weightCompensatingTorques = computeJointTorques<scalar_t>(
        mpcRobotModelPtr_->getGeneralizedCoordinates(currentMpcObservation_.state),
        mpcRobotModelPtr_->getGeneralizedVelocities(currentMpcObservation_.state, currentMpcObservation_.input), qdd_j_des, footWrenches,
        pinocchioInterface_);

    for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
      size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action = robotJointAction.at(index).value();

      action.q_des = nominalJointPositions_.empty() ? robotState.getJointPosition(index) : nominalJointPositions_[index];
      action.qd_des = 0.0;
      action.kp = mpcJointKp_[i];
      action.kd = mpcJointKd_[i];
      action.feed_forward_effort = std::clamp(weightCompensatingTorques[i], -mpcJointTorqueLimit_[i], mpcJointTorqueLimit_[i]);
    };
  }

  for (size_t i = 0; i < otherJointIndices_.size(); i++) {
    size_t index = otherJointIndices_[i];
    robot::model::JointAction& action = robotJointAction.at(index).value();

    action.q_des = nominalJointPositions_.empty() ? 0.0 : nominalJointPositions_[index];
    action.qd_des = 0;
    action.kp = otherJointKp_[i];
    action.kd = otherJointKd_[i];
    action.feed_forward_effort = 0.0;
  };
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void CentroidalMpcMrtJointController::solverWorker() {
  const auto coreAlloc = ocs2::humanoid::getDefaultCoreAllocation();
  ocs2::humanoid::setThreadCpuAffinity(coreAlloc.mpcCores, pthread_self(), "Centroidal MPC Solver Thread");

  mcpMrtInterface_.resetMpcNode(currentObservationToResetTrajectory(mcpMrtInterface_.getCurrentObservation()));
  std::cerr << "MPC is reset. NMPC solver started!" << std::endl;

  size_t slowWarningCount = 0;
  while (!terminateThread_.load()) {
    auto targetTimeForNextIteration = std::chrono::steady_clock::now() + std::chrono::microseconds(mpcDeltaTMicroSeconds_);

    mcpMrtInterface_.advanceMpc();

    // Update active policy buffer immediately after solve finishes
    mcpMrtInterface_.updatePolicy();

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
TargetTrajectories CentroidalMpcMrtJointController::currentObservationToResetTrajectory(const SystemObservation& currentObservation) {
  const auto& info = mpcRobotModelPtr_->getCentroidalModelInfo();
  vector_t targetState = currentObservation.state;

  // Zero out normalized momentum (linear and angular momentum: first 6 DOFs)
  centroidal_model::getNormalizedMomentum(targetState, info).setZero();

  // Zero out base pitch (idx 10) and roll (idx 11) so target base is upright
  targetState(10) = 0.0;
  targetState(11) = 0.0;

  // Set target joint positions to nominal (if available), preserving upright stance
  if (!nominalJointPositions_.empty()) {
    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
      centroidal_model::getJointAngles(targetState, info)[i] = nominalJointPositions_[mpcJointIndices_[i]];
    }
  }

  // Weight-compensating vertical contact forces (forces = mg/2 per foot in stance)
  vector_t targetInput = weightCompensatingInput(pinocchioInterface_, {true, true}, *mpcRobotModelPtr_);

  scalar_t t0 = currentObservation.time;
  scalar_t t1 = t0 + 2.0;

  const TargetTrajectories resetTargetTrajectories({t0, t1}, {targetState, targetState}, {targetInput, targetInput});

  std::cerr << "[CentroidalMPC] Resetting MPC target trajectory. Base pos: "
            << targetState.segment<3>(6).transpose()
            << " Base z: " << targetState(8)
            << " Input forces: " << targetInput.head(3).transpose()
            << " / " << targetInput.segment<3>(6).transpose() << std::endl;
  return resetTargetTrajectories;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t CentroidalMpcMrtJointController::computeGravityCompensation(const ::robot::model::RobotState& robotState) {
  const auto& info = mpcRobotModelPtr_->getCentroidalModelInfo();
  const auto& model = pinocchioInterface_.getModel();
  auto& data = pinocchioInterface_.getData();

  // Build Pinocchio generalized coordinates from robot state
  const vector3_t euler_zyx = quaternionToEulerZYX(robotState.getRootRotationLocalToWorldFrame());
  vector_t q(info.generalizedCoordinatesNum);
  q.head<3>() = robotState.getRootPositionInWorldFrame();
  q.segment<3>(3) = euler_zyx;
  q.tail(mpcRobotModelPtr_->getJointDim()) = robotState.getJointPositions(mpcJointIndices_);

  // Compute gravity torques: nonLinearEffects with zero velocity gives pure gravity terms
  vector_t zeroVelocity = vector_t::Zero(info.generalizedCoordinatesNum);
  pinocchio::nonLinearEffects(model, data, q, zeroVelocity);

  // data.nle now contains gravity torques for all generalized coordinates.
  // Return only the joint portion (skip the 6 floating-base DOFs).
  return data.nle.tail(mpcRobotModelPtr_->getJointDim());
}

}  // namespace ocs2::humanoid
