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

#include "humanoid_centroidal_mpc/mrt/MpcParameterUpdaterModule.h"

#include <fstream>
#include <iostream>

#include <absl/log/log.h>

#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/Numerics.h>
#include <ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_sqp/SqpSettings.h>
#include <ocs2_sqp/SqpSolver.h>

#include "humanoid_centroidal_mpc/constraint/ZeroVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

namespace ocs2::humanoid {

MpcParameterUpdaterModule::MpcParameterUpdaterModule(MPC_BASE* mpcPtr,
                                                     const std::string& taskFile,
                                                     const std::string& urdfFile,
                                                     const std::string& referenceFile,
                                                     size_t stateDim,
                                                     size_t inputDim,
                                                     const std::vector<std::string>& contactNames,
                                                     const SwitchedModelReferenceManager* referenceManager)
    : mpcPtr_(mpcPtr),
      taskFile_(taskFile),
      urdfFile_(urdfFile),
      referenceFile_(referenceFile),
      stateDim_(stateDim),
      inputDim_(inputDim),
      contactNames_(contactNames),
      referenceManagerPtr_(referenceManager) {
  if (!taskFile_.empty() && std::filesystem::exists(taskFile_)) {
    std::error_code ec;
    taskFileLastWriteTime_ = std::filesystem::last_write_time(taskFile_, ec);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MpcParameterUpdaterModule::preSolverRun(scalar_t initTime,
                                             scalar_t finalTime,
                                             const vector_t& currentState,
                                             const ReferenceManagerInterface& referenceManager) {
  // Pathway 1: Check for ROS topic data (takes priority — no file I/O needed for the source YAML)
  if (hasNewTopicData_.load(std::memory_order_acquire)) {
    std::string yamlContent;
    {
      std::lock_guard<std::mutex> lock(pendingMutex_);
      yamlContent = std::move(pendingYamlContent_);
      hasNewTopicData_.store(false, std::memory_order_release);
    }
    // Write to a temp file so we can reuse the existing loadData parsing pipeline.
    // Keep the .yaml extension so readPropertyTree dispatches to the YAML parser.
    const std::string tempFile = taskFile_ + ".live.yaml";
    try {
      std::ofstream ofs(tempFile, std::ios::trunc);
      if (ofs.is_open()) {
        ofs << yamlContent;
        ofs.close();
        applyParameterUpdates(tempFile);
      } else {
        LOG(ERROR) << "[MpcParameterUpdaterModule] Failed to write temp file: " << tempFile;
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "[MpcParameterUpdaterModule] Error writing temp file: " << e.what();
    }
  }

  // Pathway 2: Check task.yaml modification time at roughly 1Hz (assuming solver runs around 100Hz)
  if (!taskFile_.empty() && checkCounter_++ % 100 == 0) {
    std::error_code ec;
    auto last_write = std::filesystem::last_write_time(taskFile_, ec);
    if (!ec && last_write != taskFileLastWriteTime_) {
      taskFileLastWriteTime_ = last_write;
      applyParameterUpdates(taskFile_);
    }
  }
}

void MpcParameterUpdaterModule::subscribe(rclcpp::Node::SharedPtr node) {
  subscription_ = node->create_subscription<std_msgs::msg::String>(
      "/mpc_parameter_updates", rclcpp::QoS(1).best_effort(), [this](const std_msgs::msg::String::SharedPtr msg) { topicCallback(msg); });
  LOG(INFO) << "[MpcParameterUpdaterModule] Subscribed to /mpc_parameter_updates";
}

void MpcParameterUpdaterModule::topicCallback(const std_msgs::msg::String::SharedPtr msg) {
  std::cerr << "[MpcParameterUpdaterModule] topicCallback received " << msg->data.size() << " chars" << std::endl;
  std::lock_guard<std::mutex> lock(pendingMutex_);
  pendingYamlContent_ = msg->data;
  hasNewTopicData_.store(true, std::memory_order_release);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MpcParameterUpdaterModule::applyParameterUpdates(const std::string& yamlFile) {
  LOG(INFO) << "[MpcParameterUpdaterModule] Applying in-place parameter updates from " << yamlFile << "...";

  if (mpcPtr_ == nullptr) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] mpcPtr_ is null.";
    return;
  }
  auto* solverBasePtr = mpcPtr_->getSolverPtr();
  if (solverBasePtr == nullptr) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] getSolverPtr() returned null.";
    return;
  }
  auto* sqpSolverPtr = dynamic_cast<SqpSolver*>(solverBasePtr);
  if (sqpSolverPtr == nullptr) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] Underlying solver is not SqpSolver. Cannot update parameters.";
    return;
  }

  // ────────────────────────────────────────────────────────────────
  // 1. Parse weight matrices and scalars from task.yaml
  // ────────────────────────────────────────────────────────────────
  matrix_t Q = matrix_t::Zero(stateDim_, stateDim_);
  matrix_t R = matrix_t::Zero(inputDim_, inputDim_);
  matrix_t Q_final = matrix_t::Zero(stateDim_, stateDim_);
  scalar_t terminalCostScaling = 1.0;

  try {
    loadData::loadEigenMatrix(yamlFile, "Q", Q);
    loadData::loadEigenMatrix(yamlFile, "R", R);
    loadData::loadEigenMatrix(yamlFile, "Q_final", Q_final);
    loadData::loadCppDataType<scalar_t>(yamlFile, "terminalCostScaling", terminalCostScaling);
    Q_final *= terminalCostScaling;
  } catch (const std::exception& e) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] Error parsing Q/R/Q_final: " << e.what();
    return;
  }

  // ────────────────────────────────────────────────────────────────
  // 2. Parse task-space tracking cost weights
  // ────────────────────────────────────────────────────────────────
  EndEffectorKinematicsWeights footTrackingWeights;
  vector12_t footTrackingWeightsVec = vector12_t::Zero();
  bool hasFootTrackingWeights = false;
  try {
    footTrackingWeights = EndEffectorKinematicsWeights::getWeights(yamlFile, "task_space_foot_cost_weights.", false);
    footTrackingWeightsVec = footTrackingWeights.toVector();
    hasFootTrackingWeights = true;
  } catch (...) {
  }

  vector2_t icpWeights = vector2_t::Zero();
  bool hasIcpWeights = false;
  try {
    icpWeights = ICPCost::getWeights(yamlFile, "icp_cost_weights.", false);
    hasIcpWeights = true;
  } catch (...) {
  }

  // Parse task-space torso/body tracking cost weights
  boost::property_tree::ptree pt;
  std::vector<std::pair<std::string, vector12_t>> taskSpaceCostUpdates;
  try {
    loadData::readPropertyTree(yamlFile, pt);
    auto taskSpaceCostsIt = pt.find("task_space_costs");
    if (taskSpaceCostsIt != pt.not_found()) {
      for (auto& task_space_cost : taskSpaceCostsIt->second) {
        std::string costName = task_space_cost.first;
        try {
          EndEffectorKinematicsWeights weights =
              EndEffectorKinematicsWeights::getWeights(yamlFile, "task_space_costs." + costName + ".weights.", false);
          taskSpaceCostUpdates.emplace_back(costName + "_TaskSpaceKinematicsCost", weights.toVector());
        } catch (...) {
        }
      }
    }
  } catch (...) {
  }

  // Parse external torque cost weights
  std::vector<std::pair<std::string, ExternalTorqueQuadraticCostAD::Config>> extTorqueConfigs;
  for (size_t i = 0; i < contactNames_.size(); ++i) {
    try {
      std::string fieldName = (i == 0) ? "left_leg_torque_cost." : "right_leg_torque_cost.";
      auto config = ExternalTorqueQuadraticCostAD::loadConfigFromFile(yamlFile, fieldName, false);
      extTorqueConfigs.emplace_back(contactNames_[i] + "_ExternalTorqueQuadraticCost", std::move(config));
    } catch (...) {
    }
  }

  // ────────────────────────────────────────────────────────────────
  // 3. Parse barrier penalty configs
  // ────────────────────────────────────────────────────────────────
  RelaxedBarrierPenalty::Config wrenchConeBarrier, frictionConeBarrier, contactMomentBarrier;
  PieceWisePolynomialBarrierPenalty::Config jointLimitsBarrier, collisionBarrier;

  try {
    loadData::loadPtreeValue(pt, wrenchConeBarrier.mu, "contacts.contactWrenchConeSoftConstraint.mu", false);
    loadData::loadPtreeValue(pt, wrenchConeBarrier.delta, "contacts.contactWrenchConeSoftConstraint.delta", false);
  } catch (...) {
  }

  try {
    loadData::loadPtreeValue(pt, frictionConeBarrier.mu, "contacts.frictionForceConeSoftConstraint.mu", false);
    loadData::loadPtreeValue(pt, frictionConeBarrier.delta, "contacts.frictionForceConeSoftConstraint.delta", false);
  } catch (...) {
  }

  try {
    loadData::loadPtreeValue(pt, contactMomentBarrier.mu, "contacts.contactMomentXYSoftConstraint.mu", false);
    loadData::loadPtreeValue(pt, contactMomentBarrier.delta, "contacts.contactMomentXYSoftConstraint.delta", false);
  } catch (...) {
  }

  try {
    loadData::loadPtreeValue(pt, jointLimitsBarrier.mu, "jointLimits.mu", false);
    loadData::loadPtreeValue(pt, jointLimitsBarrier.delta, "jointLimits.delta", false);
  } catch (...) {
  }

  try {
    loadData::loadPtreeValue(pt, collisionBarrier.mu, "collision_constraint.mu", false);
    loadData::loadPtreeValue(pt, collisionBarrier.delta, "collision_constraint.delta", false);
  } catch (...) {
  }

  // LINT.IfChange(softConstraintWeight_yaml_path)
  scalar_t zeroVelWeight = -1.0;
  try {
    loadData::loadPtreeValue(pt, zeroVelWeight, "model_settings.foot_constraint.softConstraintWeight", false);
  } catch (...) {
  }
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:foot_constraint_section,
  // //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml:foot_constraint_section)
  // ────────────────────────────────────────────────────────────────
  // 3b. Parse foot constraint error gains
  // ────────────────────────────────────────────────────────────────
  ModelSettings::FootConstraintConfig footCfg;
  bool hasFootConstraintGains = false;
  try {
    const std::string fcPrefix = "model_settings.foot_constraint.";
    loadData::loadPtreeValue(pt, footCfg.positionErrorGain_z, fcPrefix + "positionErrorGain_z", false);
    loadData::loadPtreeValue(pt, footCfg.orientationErrorGain, fcPrefix + "orientationErrorGain", false);
    loadData::loadPtreeValue(pt, footCfg.linearVelocityErrorGain_z, fcPrefix + "linearVelocityErrorGain_z", false);
    loadData::loadPtreeValue(pt, footCfg.linearVelocityErrorGain_xy, fcPrefix + "linearVelocityErrorGain_xy", false);
    loadData::loadPtreeValue(pt, footCfg.angularVelocityErrorGain, fcPrefix + "angularVelocityErrorGain", false);
    loadData::loadPtreeValue(pt, footCfg.linearAccelerationErrorGain_z, fcPrefix + "linearAccelerationErrorGain_z", false);
    loadData::loadPtreeValue(pt, footCfg.linearAccelerationErrorGain_xy, fcPrefix + "linearAccelerationErrorGain_xy", false);
    loadData::loadPtreeValue(pt, footCfg.angularAccelerationErrorGain, fcPrefix + "angularAccelerationErrorGain", false);
    hasFootConstraintGains = true;
  } catch (...) {
  }

  // Build the Ax/Av config from foot constraint gains (mirrors CentroidalMpcInterface::getStanceFootConstraint)
  EndEffectorKinematicsTwistConstraint::Config footTwistConfig;
  if (hasFootConstraintGains) {
    footTwistConfig.b.setZero(6);
    footTwistConfig.Ax.setZero(6, 6);
    footTwistConfig.Av.setZero(6, 6);
    if (!numerics::almost_eq(footCfg.positionErrorGain_z, 0.0)) {
      footTwistConfig.Ax(2, 2) = footCfg.positionErrorGain_z;
    }
    if (!numerics::almost_eq(footCfg.orientationErrorGain, 0.0)) {
      footTwistConfig.Ax.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * footCfg.orientationErrorGain;
    }
    footTwistConfig.Av(0, 0) = footCfg.linearVelocityErrorGain_xy;
    footTwistConfig.Av(1, 1) = footCfg.linearVelocityErrorGain_xy;
    footTwistConfig.Av(2, 2) = footCfg.linearVelocityErrorGain_z;
    footTwistConfig.Av(3, 3) = footCfg.angularVelocityErrorGain;
    footTwistConfig.Av(4, 4) = footCfg.angularVelocityErrorGain;
    footTwistConfig.Av(5, 5) = footCfg.angularVelocityErrorGain;
  }

  // ────────────────────────────────────────────────────────────────
  // 3c. Parse SQP solver settings (safe subset)
  // ────────────────────────────────────────────────────────────────
  sqp::Settings sqpUpdates = sqpSolverPtr->getSettings();
  bool hasSqpUpdates = false;
  try {
    size_t sqpIter = sqpUpdates.sqpIteration;
    loadData::loadPtreeValue(pt, sqpIter, "multiple_shooting.sqpIteration", false);
    sqpUpdates.sqpIteration = sqpIter;
    loadData::loadPtreeValue(pt, sqpUpdates.deltaTol, "multiple_shooting.deltaTol", false);
    loadData::loadPtreeValue(pt, sqpUpdates.g_max, "multiple_shooting.g_max", false);
    loadData::loadPtreeValue(pt, sqpUpdates.g_min, "multiple_shooting.g_min", false);
    loadData::loadPtreeValue(pt, sqpUpdates.inequalityConstraintMu, "multiple_shooting.inequalityConstraintMu", false);
    loadData::loadPtreeValue(pt, sqpUpdates.inequalityConstraintDelta, "multiple_shooting.inequalityConstraintDelta", false);
    hasSqpUpdates = true;
  } catch (...) {
  }

  // ────────────────────────────────────────────────────────────────
  // 3d. Parse swing trajectory config
  // ────────────────────────────────────────────────────────────────
  SwingTrajectoryPlanner::Config swingConfig;
  bool hasSwingConfig = false;
  if (referenceManagerPtr_ != nullptr) {
    try {
      swingConfig = loadSwingTrajectorySettings(yamlFile, "swing_trajectory_config", false);
      hasSwingConfig = true;
    } catch (...) {
    }
  }

  // ────────────────────────────────────────────────────────────────
  // 4. Apply updates in-place to every thread-local OCP
  // ────────────────────────────────────────────────────────────────
  const matrix_t zeroQ = matrix_t::Zero(stateDim_, stateDim_);
  const matrix_t zeroR = matrix_t::Zero(inputDim_, inputDim_);

  for (auto& ocp : sqpSolverPtr->getOcpDefinitions()) {
    // ── Quadratic costs ──
    try {
      ocp.costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost").setGains(Q, R);
    } catch (const std::out_of_range&) {
      // Expected if not used in task.yaml
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to update stateInputQuadraticCost: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to update stateInputQuadraticCost: unknown exception";
    }

    try {
      ocp.costPtr->get<QuadraticStateInputCost>("stateQuadraticCost").setGains(Q, zeroR);
    } catch (const std::out_of_range&) {
      // Expected if not used in task.yaml
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to update stateQuadraticCost: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to update stateQuadraticCost: unknown exception";
    }

    try {
      ocp.costPtr->get<QuadraticStateInputCost>("inputQuadraticCost").setGains(zeroQ, R);
    } catch (const std::out_of_range&) {
      // Expected if not used in task.yaml
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to update inputQuadraticCost: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to update inputQuadraticCost: unknown exception";
    }

    try {
      ocp.finalCostPtr->get<QuadraticStateCost>("terminalCost").setGains(Q_final);
    } catch (const std::out_of_range&) {
      // Expected if not used in task.yaml
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to update terminalCost: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to update terminalCost: unknown exception";
    }

    // ── Foot tracking costs ──
    if (hasFootTrackingWeights) {
      for (const auto& footName : contactNames_) {
        try {
          ocp.costPtr->get<CentroidalMpcEndEffectorFootCost>(footName + "_TaskSpaceKinematicsCost").setWeights(footTrackingWeightsVec);
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_TaskSpaceKinematicsCost: " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_TaskSpaceKinematicsCost: unknown exception";
        }
      }
    }

    // ── Task-space body tracking costs (torso, etc.) ──
    for (const auto& [costName, weightsVec] : taskSpaceCostUpdates) {
      try {
        ocp.costPtr->get<EndEffectorKinematicsQuadraticCost>(costName).setWeights(weightsVec);
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update " << costName << ": " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update " << costName << ": unknown exception";
      }
    }

    // ── ICP cost ──
    if (hasIcpWeights) {
      try {
        ocp.costPtr->get<ICPCost>("icp_Cost").setWeights(icpWeights);
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update icp_Cost: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update icp_Cost: unknown exception";
      }
    }

    // ── External torque costs ──
    for (const auto& [costName, config] : extTorqueConfigs) {
      try {
        ocp.costPtr->get<ExternalTorqueQuadraticCostAD>(costName).setWeights(config.weights);
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update " << costName << ": " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update " << costName << ": unknown exception";
      }
    }

    // ── Soft constraints: wrench cone, friction cone, contact moment XY ──
    // Penalties are wrapped in PenaltyBaseWrapper (AugmentedPenaltyBase), so we use
    // setParameters(vector_t{mu, delta}) which delegates through to the inner PenaltyBase.
    const vector_t wrenchConeParams = (vector_t(2) << wrenchConeBarrier.mu, wrenchConeBarrier.delta).finished();
    const vector_t frictionConeParams = (vector_t(2) << frictionConeBarrier.mu, frictionConeBarrier.delta).finished();
    const vector_t contactMomentParams = (vector_t(2) << contactMomentBarrier.mu, contactMomentBarrier.delta).finished();

    for (const auto& footName : contactNames_) {
      // Contact wrench cone
      try {
        auto& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_contactWrenchCone");
        for (auto& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
          penalty->setParameters(wrenchConeParams);
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update " << footName << "_contactWrenchCone: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update " << footName << "_contactWrenchCone: unknown exception";
      }

      // Friction force cone
      try {
        auto& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_frictionForceCone");
        for (auto& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
          penalty->setParameters(frictionConeParams);
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update " << footName << "_frictionForceCone: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update " << footName << "_frictionForceCone: unknown exception";
      }

      // Contact moment XY
      try {
        auto& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_contactMomentXY");
        for (auto& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
          penalty->setParameters(contactMomentParams);
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update " << footName << "_contactMomentXY: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update " << footName << "_contactMomentXY: unknown exception";
      }
    }

    // ── Joint limits ──
    try {
      ocp.stateSoftConstraintPtr->get<JointLimitsSoftConstraint>("jointLimits").setGains(jointLimitsBarrier.mu, jointLimitsBarrier.delta);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to update jointLimits: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to update jointLimits: unknown exception";
    }

    // ── Foot collision ──
    try {
      auto& softCon = ocp.stateSoftConstraintPtr->get<StateSoftConstraint>("FootCollisionSoftConstraint");
      const vector_t collisionParams = (vector_t(2) << collisionBarrier.mu, collisionBarrier.delta).finished();
      for (auto& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
        penalty->setParameters(collisionParams);
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to update FootCollisionSoftConstraint: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to update FootCollisionSoftConstraint: unknown exception";
    }

    // ── Zero velocity soft constraint weight ──
    if (zeroVelWeight > 0.0) {
      // The QuadraticPenalty is wrapped inside a PenaltyBaseWrapper (AugmentedPenaltyBase).
      // Use setParameters() which delegates through the wrapper to QuadraticPenalty::setParameters().
      vector_t scaleParam(1);
      scaleParam[0] = zeroVelWeight;
      for (const auto& footName : contactNames_) {
        try {
          auto& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_zeroVelocity");
          for (auto& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
            penalty->setParameters(scaleParam);
          }
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_zeroVelocity (soft): " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_zeroVelocity (soft): unknown exception";
        }
      }
    }

    // ── Foot constraint error gains ──
    if (hasFootConstraintGains) {
      for (const auto& footName : contactNames_) {
        // Hard constraint path
        try {
          auto& con = ocp.equalityConstraintPtr->get<ZeroVelocityConstraintCppAd>(footName + "_zeroVelocity");
          con.getTwistConstraint().configure(EndEffectorKinematicsTwistConstraint::Config(footTwistConfig));
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_zeroVelocity (hard): " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_zeroVelocity (hard): unknown exception";
        }
        // Soft constraint path: the inner constraint is wrapped in StateInputSoftConstraint.
        // ZeroVelocityConstraintCppAd is reached via dynamic_cast through the soft constraint wrapper.
        try {
          auto& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_zeroVelocity");
          auto* zeroVelCon = dynamic_cast<ZeroVelocityConstraintCppAd*>(softCon.getConstraintPtr().get());
          if (zeroVelCon != nullptr) {
            zeroVelCon->getTwistConstraint().configure(EndEffectorKinematicsTwistConstraint::Config(footTwistConfig));
          }
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_zeroVelocity (soft config): " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_zeroVelocity (soft config): unknown exception";
        }
      }
    }
  }

  // ── SQP solver settings (applied once, not per-thread OCP) ──
  if (hasSqpUpdates) {
    sqpSolverPtr->getSettings() = sqpUpdates;
  }

  // ── Swing trajectory config ──
  if (hasSwingConfig && referenceManagerPtr_ != nullptr) {
    auto swingPlanner = referenceManagerPtr_->getSwingTrajectoryPlanner();
    if (swingPlanner) {
      swingPlanner->setConfig(swingConfig);
    }
  }

  LOG(INFO) << "[MpcParameterUpdaterModule] Successfully applied in-place parameter updates to SqpSolver.";
}

}  // namespace ocs2::humanoid
