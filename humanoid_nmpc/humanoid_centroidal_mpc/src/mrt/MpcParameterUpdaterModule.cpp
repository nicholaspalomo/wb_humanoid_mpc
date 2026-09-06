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

#include <absl/log/log.h>

#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_sqp/SqpSolver.h>

#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"

namespace ocs2::humanoid {

MpcParameterUpdaterModule::MpcParameterUpdaterModule(MPC_BASE* mpcPtr,
                                                     const std::string& taskFile,
                                                     const std::string& urdfFile,
                                                     const std::string& referenceFile,
                                                     size_t stateDim,
                                                     size_t inputDim,
                                                     const std::vector<std::string>& contactNames)
    : mpcPtr_(mpcPtr),
      taskFile_(taskFile),
      urdfFile_(urdfFile),
      referenceFile_(referenceFile),
      stateDim_(stateDim),
      inputDim_(inputDim),
      contactNames_(contactNames) {
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
  // Check task.yaml modification time at roughly 1Hz (assuming solver runs around 100Hz)
  if (!taskFile_.empty() && checkCounter_++ % 100 == 0) {
    std::error_code ec;
    auto last_write = std::filesystem::last_write_time(taskFile_, ec);
    if (!ec && last_write != taskFileLastWriteTime_) {
      taskFileLastWriteTime_ = last_write;
      applyParameterUpdates();
    }
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MpcParameterUpdaterModule::applyParameterUpdates() {
  LOG(INFO) << "[MpcParameterUpdaterModule] Detected changes in " << taskFile_ << ". Applying in-place parameter updates...";

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
    loadData::loadEigenMatrix(taskFile_, "Q", Q);
    loadData::loadEigenMatrix(taskFile_, "R", R);
    loadData::loadEigenMatrix(taskFile_, "Q_final", Q_final);
    loadData::loadCppDataType<scalar_t>(taskFile_, "terminalCostScaling", terminalCostScaling);
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
    footTrackingWeights = EndEffectorKinematicsWeights::getWeights(taskFile_, "task_space_foot_cost_weights.", false);
    footTrackingWeightsVec = footTrackingWeights.toVector();
    hasFootTrackingWeights = true;
  } catch (...) {
  }

  vector2_t icpWeights = vector2_t::Zero();
  bool hasIcpWeights = false;
  try {
    icpWeights = ICPCost::getWeights(taskFile_, "icp_cost_weights.", false);
    hasIcpWeights = true;
  } catch (...) {
  }

  // Parse task-space torso/body tracking cost weights
  boost::property_tree::ptree pt;
  std::vector<std::pair<std::string, vector12_t>> taskSpaceCostUpdates;
  try {
    loadData::readPropertyTree(taskFile_, pt);
    auto taskSpaceCostsIt = pt.find("task_space_costs");
    if (taskSpaceCostsIt != pt.not_found()) {
      for (auto& task_space_cost : taskSpaceCostsIt->second) {
        std::string costName = task_space_cost.first;
        try {
          EndEffectorKinematicsWeights weights =
              EndEffectorKinematicsWeights::getWeights(taskFile_, "task_space_costs." + costName + ".weights.", false);
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
      auto config = ExternalTorqueQuadraticCostAD::loadConfigFromFile(taskFile_, fieldName, false);
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

  scalar_t zeroVelWeight = -1.0;
  try {
    loadData::loadPtreeValue(pt, zeroVelWeight, "contacts.footConstraintConfig.softConstraintWeight", false);
  } catch (...) {
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
    } catch (...) {
    }

    try {
      ocp.costPtr->get<QuadraticStateInputCost>("stateQuadraticCost").setGains(Q, zeroR);
    } catch (...) {
    }

    try {
      ocp.costPtr->get<QuadraticStateInputCost>("inputQuadraticCost").setGains(zeroQ, R);
    } catch (...) {
    }

    try {
      ocp.finalCostPtr->get<QuadraticStateCost>("terminalCost").setGains(Q_final);
    } catch (...) {
    }

    // ── Foot tracking costs ──
    if (hasFootTrackingWeights) {
      for (const auto& footName : contactNames_) {
        try {
          ocp.costPtr->get<CentroidalMpcEndEffectorFootCost>(footName + "_TaskSpaceKinematicsCost").setWeights(footTrackingWeightsVec);
        } catch (...) {
        }
      }
    }

    // ── Task-space body tracking costs (torso, etc.) ──
    for (const auto& [costName, weightsVec] : taskSpaceCostUpdates) {
      try {
        ocp.costPtr->get<EndEffectorKinematicsQuadraticCost>(costName).setWeights(weightsVec);
      } catch (...) {
      }
    }

    // ── ICP cost ──
    if (hasIcpWeights) {
      try {
        ocp.costPtr->get<ICPCost>("icp_Cost").setWeights(icpWeights);
      } catch (...) {
      }
    }

    // ── External torque costs ──
    for (const auto& [costName, config] : extTorqueConfigs) {
      try {
        ocp.costPtr->get<ExternalTorqueQuadraticCostAD>(costName).setWeights(config.weights);
      } catch (...) {
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
      } catch (...) {
      }

      // Friction force cone
      try {
        auto& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_frictionForceCone");
        for (auto& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
          penalty->setParameters(frictionConeParams);
        }
      } catch (...) {
      }

      // Contact moment XY
      try {
        auto& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_contactMomentXY");
        for (auto& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
          penalty->setParameters(contactMomentParams);
        }
      } catch (...) {
      }
    }

    // ── Joint limits ──
    try {
      ocp.stateSoftConstraintPtr->get<JointLimitsSoftConstraint>("jointLimits").setGains(jointLimitsBarrier.mu, jointLimitsBarrier.delta);
    } catch (...) {
    }

    // ── Foot collision ──
    try {
      auto& softCon = ocp.stateSoftConstraintPtr->get<StateSoftConstraint>("FootCollisionSoftConstraint");
      const vector_t collisionParams = (vector_t(2) << collisionBarrier.mu, collisionBarrier.delta).finished();
      for (auto& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
        penalty->setParameters(collisionParams);
      }
    } catch (...) {
    }
  }

  LOG(INFO) << "[MpcParameterUpdaterModule] Successfully applied in-place parameter updates to SqpSolver.";
}

}  // namespace ocs2::humanoid
