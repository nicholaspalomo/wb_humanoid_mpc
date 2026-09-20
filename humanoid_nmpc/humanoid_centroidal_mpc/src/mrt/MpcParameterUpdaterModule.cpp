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

#include <cmath>
#include <fstream>
#include <functional>
#include <limits>

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
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_common_mpc/HumanoidPreComputation.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"
#include "humanoid_common_mpc/constraint/ContactMomentXYConstraintCppAd.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

namespace ocs2::humanoid {

namespace {

/// In the centroidal state x = [h_norm(6), p_base(3), euler_zyx(3), q_j], the
/// base pose occupies the 6x6 block starting at index 6.
constexpr Eigen::Index kBasePoseStateIndex = 6;
constexpr Eigen::Index kBasePoseDim = 6;

/**
 * Loads a weight matrix out of an already-parsed property tree.
 *
 * loadData::loadEigenMatrix re-reads and re-parses the whole task file on every
 * call. This runs on the solver thread inside preSolverRun, at the rate the
 * operator drags a slider, so the tree is parsed once by the caller and reused.
 *
 * Mirrors loadEigenMatrix's semantics: a `scaling` key multiplies every entry, a
 * `default` key fills the entries that are absent.
 *
 * @throws std::runtime_error if the matrix is not present at all.
 */
void loadEigenMatrixFromPtree(const boost::property_tree::ptree& pt, const std::string& matrixName, matrix_t& matrix) {
  const scalar_t scaling = pt.get<scalar_t>(matrixName + ".scaling", 1.0);
  const scalar_t defaultValue = pt.get<scalar_t>(matrixName + ".default", 0.0);

  Eigen::Index numFailed = 0;
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
      const auto entry = pt.get_optional<scalar_t>(matrixName + ".(" + std::to_string(i) + "," + std::to_string(j) + ")");
      if (entry) {
        matrix(i, j) = scaling * (*entry);
      } else {
        matrix(i, j) = scaling * defaultValue;
        ++numFailed;
      }
    }
  }
  if (numFailed == matrix.size()) {
    throw std::runtime_error("[MpcParameterUpdaterModule] Could not load matrix \"" + matrixName + "\" from the task file.");
  }
}

/**
 * Zeroes the base pose block of a state weight matrix.
 *
 * Mirrors HumanoidCostConstraintFactory::zeroBasePoseWeights, which does the same
 * at construction time. Both the running and the terminal cost must be treated
 * identically, otherwise a live update re-introduces base pose tracking that
 * fights the CoM + aCOM cost.
 */
void zeroBasePoseWeights(matrix_t& Q) {
  Q.block(kBasePoseStateIndex, kBasePoseStateIndex, kBasePoseDim, kBasePoseDim).setZero();
}

}  // namespace

MpcParameterUpdaterModule::MpcParameterUpdaterModule(MPC_BASE* mpcPtr,
                                                     const std::string& taskFile,
                                                     const std::string& urdfFile,
                                                     const std::string& referenceFile,
                                                     size_t stateDim,
                                                     size_t inputDim,
                                                     const std::vector<std::string>& contactNames,
                                                     const SwitchedModelReferenceManager* referenceManager,
                                                     std::optional<BasisInputsCostTransformConfig> basisCostTransform)
    : mpcPtr_(mpcPtr),
      taskFile_(taskFile),
      urdfFile_(urdfFile),
      referenceFile_(referenceFile),
      contactPlanningFile_(taskFile.empty() ? std::string() : resolveContactPlanningConfigFile(taskFile)),
      stateDim_(stateDim),
      inputDim_(inputDim),
      contactNames_(contactNames),
      referenceManagerPtr_(referenceManager),
      basisCostTransform_(std::move(basisCostTransform)) {
  if (basisCostTransform_.has_value()) {
    // The transformed R is written straight into the OCP with setGains(), which performs no size check. Catch a
    // wrench-vs-basis dimension mix-up here, at construction, instead of silently corrupting the input cost online.
    const auto& cfg = *basisCostTransform_;
    if (inputDim_ != cfg.basisInputDim()) {
      throw std::invalid_argument("[MpcParameterUpdaterModule] inputDim (" + std::to_string(inputDim_) +
                                  ") must equal the basis-space input dimension of the cost transform (" +
                                  std::to_string(cfg.basisInputDim()) + ").");
    }
    if (static_cast<size_t>(cfg.basisToWrenchMap.rows()) != cfg.wrenchInputDim) {
      throw std::invalid_argument("[MpcParameterUpdaterModule] basisToWrenchMap has " + std::to_string(cfg.basisToWrenchMap.rows()) +
                                  " rows but wrenchInputDim is " + std::to_string(cfg.wrenchInputDim) + ".");
    }
    if (cfg.numBasisInputs > inputDim_) {
      throw std::invalid_argument("[MpcParameterUpdaterModule] numBasisInputs (" + std::to_string(cfg.numBasisInputs) +
                                  ") exceeds inputDim (" + std::to_string(inputDim_) + ").");
    }
  }

  if (!taskFile_.empty() && std::filesystem::exists(taskFile_)) {
    std::error_code ec;
    taskFileLastWriteTime_ = std::filesystem::last_write_time(taskFile_, ec);
  }
  if (!referenceFile_.empty() && std::filesystem::exists(referenceFile_)) {
    std::error_code ec;
    referenceFileLastWriteTime_ = std::filesystem::last_write_time(referenceFile_, ec);
  }
  if (contactPlanningFile_ != taskFile_ && std::filesystem::exists(contactPlanningFile_)) {
    std::error_code ec;
    contactPlanningFileLastWriteTime_ = std::filesystem::last_write_time(contactPlanningFile_, ec);
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

  // Pathway 2: Check the modification times of task.yaml and of the planner's own file at roughly 1Hz (assuming the
  // solver runs around 100Hz). The planner's file is watched separately only when it is a file of its own; a
  // `contact_planning` block inside the task file is applied by applyParameterUpdates() with the rest.
  if (!taskFile_.empty() && checkCounter_++ % 100 == 0) {
    std::error_code ec;
    auto last_write = std::filesystem::last_write_time(taskFile_, ec);
    if (!ec && last_write != taskFileLastWriteTime_) {
      taskFileLastWriteTime_ = last_write;
      applyParameterUpdates(taskFile_);
    }
    if (contactPlanningFile_ != taskFile_) {
      auto planningLastWrite = std::filesystem::last_write_time(contactPlanningFile_, ec);
      if (!ec && planningLastWrite != contactPlanningFileLastWriteTime_) {
        contactPlanningFileLastWriteTime_ = planningLastWrite;
        applyContactPlanningUpdates(contactPlanningFile_);
      }
    }
    // The command limits and ramps live in reference.yaml, which nothing else watches: without this a change to it
    // needed a restart of the controller (the Command Limits tab of the remote control writes exactly this file).
    if (!referenceFile_.empty() && !referenceFileReloaders_.empty()) {
      auto referenceLastWrite = std::filesystem::last_write_time(referenceFile_, ec);
      if (!ec && referenceLastWrite != referenceFileLastWriteTime_) {
        referenceFileLastWriteTime_ = referenceLastWrite;
        for (const std::function<void(const std::string&)>& reload : referenceFileReloaders_) {
          try {
            reload(referenceFile_);
          } catch (const std::exception& e) {
            LOG(WARNING) << "[MpcParameterUpdaterModule] Failed to reload " << referenceFile_ << ": " << e.what();
          }
        }
        LOG(INFO) << "[MpcParameterUpdaterModule] Reloaded command limits from " << referenceFile_;
      }
    }
  }
}

void MpcParameterUpdaterModule::subscribe(rclcpp::Node::SharedPtr node) {
  subscription_ = node->create_subscription<std_msgs::msg::String>(
      "/mpc_parameter_updates", rclcpp::QoS(1).best_effort(), [this](const std_msgs::msg::String::SharedPtr msg) { topicCallback(msg); });
  LOG(INFO) << "[MpcParameterUpdaterModule] Subscribed to /mpc_parameter_updates";
}

void MpcParameterUpdaterModule::topicCallback(const std_msgs::msg::String::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(pendingMutex_);
  pendingYamlContent_ = msg->data;
  hasNewTopicData_.store(true, std::memory_order_release);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::optional<std::string> MpcParameterUpdaterModule::takeContactEstimatorUpdate() {
  std::lock_guard<std::mutex> lock(controllerSettingsMutex_);
  std::optional<std::string> update = std::move(pendingContactEstimator_);
  pendingContactEstimator_.reset();
  return update;
}

std::optional<ContactWrenchGate::Config> MpcParameterUpdaterModule::takeContactWrenchGateUpdate() {
  std::lock_guard<std::mutex> lock(controllerSettingsMutex_);
  std::optional<ContactWrenchGate::Config> update = pendingContactWrenchGate_;
  pendingContactWrenchGate_.reset();
  return update;
}

void MpcParameterUpdaterModule::recordControllerSettings(const std::string& yamlFile) {
  boost::property_tree::ptree pt;
  try {
    loadData::readPropertyTree(yamlFile, pt);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] Could not parse " << yamlFile << " for the controller settings: " << e.what();
    return;
  }
  const boost::optional<std::string> name = pt.get_optional<std::string>("contactEstimator");
  std::optional<ContactWrenchGate::Config> gate;
  if (const auto block = pt.get_child_optional("contact_wrench_gate")) {
    ContactWrenchGate::Config config;
    config.debounceTime = block->get<scalar_t>("debounceTime", config.debounceTime);
    config.rampTime = block->get<scalar_t>("rampTime", config.rampTime);
    if (config.debounceTime >= 0.0 && config.rampTime >= 0.0) {
      gate = config;
    } else {
      LOG(ERROR) << "[MpcParameterUpdaterModule] contact_wrench_gate.debounceTime and rampTime must be non-negative; the block is ignored.";
    }
  }
  std::lock_guard<std::mutex> lock(controllerSettingsMutex_);
  if (name) pendingContactEstimator_ = *name;
  if (gate) pendingContactWrenchGate_ = gate;
}

void MpcParameterUpdaterModule::applyParameterUpdates(const std::string& yamlFile) {
  LOG(INFO) << "[MpcParameterUpdaterModule] Applying in-place parameter updates from " << yamlFile << "...";

  // The controller-side settings do not touch the solver; they are recorded even when there is no solver to update.
  recordControllerSettings(yamlFile);

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
  // 1. Parse property tree and weight matrices from task.yaml
  // ────────────────────────────────────────────────────────────────
  boost::property_tree::ptree pt;
  try {
    loadData::readPropertyTree(yamlFile, pt);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] Could not parse " << yamlFile << ": " << e.what() << ". No parameters were updated.";
    return;
  }

  matrix_t Q = matrix_t::Zero(stateDim_, stateDim_);
  matrix_t R = matrix_t::Zero(inputDim_, inputDim_);
  matrix_t Q_final = matrix_t::Zero(stateDim_, stateDim_);
  scalar_t terminalCostScaling = 1.0;
  bool hasQFinal = false;

  const bool useComAndAcom = pt.get<bool>("useComAndAcomTracking", false);

  try {
    loadEigenMatrixFromPtree(pt, "Q", Q);

    if (basisCostTransform_.has_value()) {
      // Hot-reload the basis scaling regularization if the contacts section has it.
      // R_basis = M^T R_wrench M + reg * I is only positive definite for a
      // non-negative regularization, and an indefinite R stalls the QP solver.
      scalar_t basisReg = basisCostTransform_->lambdaRegularization;
      loadData::loadPtreeValue(pt, basisReg, "contacts.basisScalingRegularization", false);
      if (basisReg < 0.0) {
        LOG(ERROR) << "[MpcParameterUpdaterModule] contacts.basisScalingRegularization must be non-negative, got " << basisReg
                   << ". Keeping the previous value " << basisCostTransform_->lambdaRegularization << ".";
      } else {
        basisCostTransform_->lambdaRegularization = basisReg;
      }
      // The R indices in task.yaml refer to wrench-space inputs (forces/moments/joint velocities). In basis-vector mode
      // the OCP input is [λ, joint velocities], so loading R directly at inputDim_ would put force weights on λ entries.
      // Load at the wrench dimension and apply the same transform the OCP factory used: R_basis = Mᵀ R_wrench M + reg.
      matrix_t R_wrench = matrix_t::Zero(basisCostTransform_->wrenchInputDim, basisCostTransform_->wrenchInputDim);
      loadEigenMatrixFromPtree(pt, "R", R_wrench);
      R = transformWrenchInputCostToBasisSpace(R_wrench, *basisCostTransform_);
    } else {
      loadEigenMatrixFromPtree(pt, "R", R);
    }
    // Q_final is optional, and ignored while useDcmTerminalCost is on (the OCP then has no quadratic terminal cost).
    hasQFinal = pt.get_child_optional("Q_final").is_initialized() && !pt.get<bool>("useDcmTerminalCost", false);
    if (hasQFinal) {
      loadEigenMatrixFromPtree(pt, "Q_final", Q_final);
      terminalCostScaling = pt.get<scalar_t>("terminalCostScaling", terminalCostScaling);
    }

    // The factory zeroes the base pose block of both the running and the terminal
    // state cost when CoM + aCOM tracking is on, so live updates must do the same
    // or a slider drag silently re-introduces base pose tracking.
    if (useComAndAcom) {
      zeroBasePoseWeights(Q);
      zeroBasePoseWeights(Q_final);
    }
    Q_final *= terminalCostScaling;
  } catch (const std::exception& e) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] Error parsing Q/R/Q_final: " << e.what();
    return;
  }

  // CoM and ACoM tracking weights
  matrix_t Q_com = matrix_t::Zero(3, 3);
  matrix_t Q_acom = matrix_t::Zero(3, 3);
  bool hasComAcom = false;
  try {
    loadEigenMatrixFromPtree(pt, "Q_com", Q_com);
    loadEigenMatrixFromPtree(pt, "Q_acom", Q_acom);
    hasComAcom = true;
  } catch (const std::exception& e) {
    if (useComAndAcom) {
      LOG(WARNING) << "[MpcParameterUpdaterModule] CoM + aCOM tracking is enabled but its weights could not be read: " << e.what();
    }
  }

  // ────────────────────────────────────────────────────────────────
  // 2. Parse task-space tracking cost weights
  // ────────────────────────────────────────────────────────────────

  // The getWeights and loadConfigFromFile helpers below throw when their section
  // is absent, which is the normal case for a task file that does not configure
  // that cost. The catch is therefore deliberately silent: the corresponding
  // has* flag stays false and the apply step is skipped, leaving the running
  // value untouched.
  EndEffectorKinematicsWeights footTrackingWeights;
  vector12_t footTrackingWeightsVec = vector12_t::Zero();
  bool hasFootTrackingWeights = false;
  try {
    footTrackingWeights = EndEffectorKinematicsWeights::getWeights(yamlFile, "task_space_foot_cost_weights.", false);
    footTrackingWeightsVec = footTrackingWeights.toVector();
    hasFootTrackingWeights = true;
  } catch (...) {
  }

  // loadPtreeValue leaves its target untouched for a missing key rather than
  // throwing, so presence has to be probed directly. Without the probe an absent
  // key would be applied as its default and silently switch the flag off.
  bool footActiveInStance = false;
  const bool hasFootActiveInStance = pt.get_optional<bool>("task_space_foot_cost_weights.activeInStance").is_initialized();
  if (hasFootActiveInStance) {
    loadData::loadPtreeValue(pt, footActiveInStance, "task_space_foot_cost_weights.activeInStance", false);
  }

  vector2_t icpWeights = vector2_t::Zero();
  bool hasIcpWeights = false;
  try {
    icpWeights = ICPCost::getWeights(yamlFile, "icp_cost_weights.", false);
    hasIcpWeights = true;
  } catch (...) {
  }

  // DCM terminal cost (weights, CoM height, velocity offset); the CppAD model is parameterised, no recompilation needed.
  std::optional<DcmTerminalCost::Config> dcmTerminalConfig;
  if (pt.get_child_optional("dcm_terminal_cost")) {
    try {
      dcmTerminalConfig = DcmTerminalCost::loadConfig(yamlFile, "dcm_terminal_cost.", false);
    } catch (const std::exception& e) {
      LOG(WARNING) << "[MpcParameterUpdaterModule] dcm_terminal_cost section could not be applied: " << e.what();
    }
  }

  // Parse task-space torso/body tracking cost weights
  std::vector<std::pair<std::string, vector12_t>> taskSpaceCostUpdates;
  try {
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
  // Every barrier below is default-constructed, so an absent section must leave
  // the running value alone rather than be applied as a library default.
  // loadPtreeValue does not throw for a missing key, so presence is probed on the
  // section itself and the corresponding apply step is skipped when it is absent.
  RelaxedBarrierPenalty::Config wrenchConeBarrier, frictionConeBarrier, contactMomentBarrier;
  PieceWisePolynomialBarrierPenalty::Config jointLimitsBarrier, collisionBarrier, basisNonNegativityBarrier;

  // Only `mu` and `delta` are hot-reloadable. The geometric coefficients in the
  // same yaml sections (frictionCoefficient, minNormalForce, ...) are baked into
  // the CppAD-compiled constraints at build time and cannot be updated here.
  // LINT.IfChange(hot_reloadable_barrier_keys)
  const auto loadBarrier = [&pt](const std::string& section, scalar_t& mu, scalar_t& delta) {
    if (!pt.get_child_optional(section)) {
      return false;
    }
    loadData::loadPtreeValue(pt, mu, section + ".mu", false);
    loadData::loadPtreeValue(pt, delta, section + ".delta", false);
    return true;
  };

  const bool hasWrenchConeBarrier = loadBarrier("contacts.contactWrenchConeSoftConstraint", wrenchConeBarrier.mu, wrenchConeBarrier.delta);
  const bool hasFrictionConeBarrier =
      loadBarrier("contacts.frictionForceConeSoftConstraint", frictionConeBarrier.mu, frictionConeBarrier.delta);
  const bool hasContactMomentBarrier =
      loadBarrier("contacts.contactMomentXYSoftConstraint", contactMomentBarrier.mu, contactMomentBarrier.delta);
  const bool hasBasisNonNegativityBarrier =
      loadBarrier("contacts.basisNonNegativityBarrier", basisNonNegativityBarrier.mu, basisNonNegativityBarrier.delta);
  const bool hasJointLimitsBarrier = loadBarrier("jointLimits", jointLimitsBarrier.mu, jointLimitsBarrier.delta);
  const bool hasCollisionBarrier = loadBarrier("collision_constraint", collisionBarrier.mu, collisionBarrier.delta);
  // LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/tk_app/mpc_params_tab.py:build_time_contact_keys)

  // ────────────────────────────────────────────────────────────────
  // 3a. Parse contact-implicit config
  // ────────────────────────────────────────────────────────────────
  scalar_t complementarityWeight = -1.0;
  scalar_t slipWeight = -1.0;
  // The residual references and the terrain height, retuned live alongside the weights. Every key of this block
  // becomes a slider in the tuning dashboard, so one that quietly did nothing until the next launch would be a trap
  // rather than a parameter; the negative sentinel marks a key the file does not carry.
  scalar_t heightReference = -1.0;
  scalar_t velocityReference = -1.0;
  scalar_t angularVelocityReference = -1.0;
  // The length scale the footprint corners' minimum is blended over; see ContactImplicitConfig::gapSmoothing.
  scalar_t gapSmoothing = -1.0;
  // Where the ground is: one top-level key, shared with the swing trajectories, rather than a second definition
  // inside the contact_implicit block. NaN marks a file that does not carry it.
  // LINT.IfChange(terrain_height_updater_yaml_path)
  scalar_t terrainHeight = std::numeric_limits<scalar_t>::quiet_NaN();
  loadData::loadPtreeValue(pt, terrainHeight, "terrainHeight", false);
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:terrain_height_config)
  // The hinge's two parameters. `delta` is structurally zero: it is the offset of the hinge's zero, and the whole
  // point of using a hinge for h >= 0 is that its zero sits exactly on the ground. A positive delta would demand
  // clearance from a foot that is supposed to be resting on the floor.
  scalar_t penetrationWeight = -1.0;
  bool hasPenetrationWeight = false;
  if (pt.get_child_optional("contact_implicit")) {
    const std::string ciPrefix = "contact_implicit.";
    loadData::loadPtreeValue(pt, complementarityWeight, ciPrefix + "complementarityWeight", false);
    loadData::loadPtreeValue(pt, slipWeight, ciPrefix + "slipWeight", false);
    loadData::loadPtreeValue(pt, heightReference, ciPrefix + "heightReference", false);
    loadData::loadPtreeValue(pt, velocityReference, ciPrefix + "velocityReference", false);
    loadData::loadPtreeValue(pt, angularVelocityReference, ciPrefix + "angularVelocityReference", false);
    loadData::loadPtreeValue(pt, gapSmoothing, ciPrefix + "gapSmoothing", false);

    if (pt.get_optional<scalar_t>(ciPrefix + "penetrationWeight")) {
      loadData::loadPtreeValue(pt, penetrationWeight, ciPrefix + "penetrationWeight", false);
      hasPenetrationWeight = penetrationWeight >= 0.0;
    }
  }

  // LINT.IfChange(softConstraintWeight_yaml_path)
  // The negative sentinel is what marks the weight absent: loadPtreeValue leaves
  // it untouched for a missing key, and the apply step below only runs for a
  // positive value.
  scalar_t zeroVelWeight = -1.0;
  loadData::loadPtreeValue(pt, zeroVelWeight, "model_settings.foot_constraint.softConstraintWeight", false);
  // The weight of the SOFT normal-velocity term, which is what shapes the swing under the contact-implicit
  // formulation and is therefore the knob an operator reaches for first. It was missing here while its neighbour
  // above was present, so the task file told the operator to tune a key that took effect only on the next launch.
  scalar_t normalVelSoftWeight = -1.0;
  loadData::loadPtreeValue(pt, normalVelSoftWeight, "model_settings.foot_constraint.normalVelocitySoftConstraintWeight", false);
  // clang-format off
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:foot_constraint_section, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml:foot_constraint_section)
  // clang-format on
  // ────────────────────────────────────────────────────────────────
  // 3b. Parse foot constraint error gains
  // ────────────────────────────────────────────────────────────────
  ModelSettings::FootConstraintConfig footCfg;
  bool hasFootConstraintGains = false;
  if (pt.get_child_optional("model_settings.foot_constraint")) {
    const std::string fcPrefix = "model_settings.foot_constraint.";
    loadData::loadPtreeValue(pt, footCfg.positionErrorGain_z, fcPrefix + "positionErrorGain_z", false);
    loadData::loadPtreeValue(pt, footCfg.orientationErrorGain, fcPrefix + "orientationErrorGain", false);
    loadData::loadPtreeValue(pt, footCfg.linearVelocityErrorGain_z, fcPrefix + "linearVelocityErrorGain_z", false);
    loadData::loadPtreeValue(pt, footCfg.linearVelocityErrorGain_xy, fcPrefix + "linearVelocityErrorGain_xy", false);
    loadData::loadPtreeValue(pt, footCfg.angularVelocityErrorGain, fcPrefix + "angularVelocityErrorGain", false);
    loadData::loadPtreeValue(pt, footCfg.linearAccelerationErrorGain_z, fcPrefix + "linearAccelerationErrorGain_z", false);
    loadData::loadPtreeValue(pt, footCfg.linearAccelerationErrorGain_xy, fcPrefix + "linearAccelerationErrorGain_xy", false);
    loadData::loadPtreeValue(pt, footCfg.angularAccelerationErrorGain, fcPrefix + "angularAccelerationErrorGain", false);
    loadData::loadPtreeValue(pt, footCfg.constrainOrientation, fcPrefix + "constrainOrientation", false);
    loadData::loadPtreeValue(pt, footCfg.constrainYawRateAboutContactNormal, fcPrefix + "constrainYawRateAboutContactNormal", false);
    hasFootConstraintGains = true;
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
  if (pt.get_child_optional("multiple_shooting")) {
    size_t sqpIter = sqpUpdates.sqpIteration;
    loadData::loadPtreeValue(pt, sqpIter, "multiple_shooting.sqpIteration", false);
    sqpUpdates.sqpIteration = sqpIter;
    loadData::loadPtreeValue(pt, sqpUpdates.deltaTol, "multiple_shooting.deltaTol", false);
    loadData::loadPtreeValue(pt, sqpUpdates.g_max, "multiple_shooting.g_max", false);
    loadData::loadPtreeValue(pt, sqpUpdates.g_min, "multiple_shooting.g_min", false);
    loadData::loadPtreeValue(pt, sqpUpdates.inequalityConstraintMu, "multiple_shooting.inequalityConstraintMu", false);
    loadData::loadPtreeValue(pt, sqpUpdates.inequalityConstraintDelta, "multiple_shooting.inequalityConstraintDelta", false);
    hasSqpUpdates = true;
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

    if (hasQFinal) {
      try {
        ocp.finalCostPtr->get<QuadraticStateCost>("terminalCost").setGains(Q_final);
      } catch (const std::out_of_range&) {
        // Expected if not used in task.yaml
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update terminalCost: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update terminalCost: unknown exception";
      }
    }

    // ── DCM terminal cost ──
    if (dcmTerminalConfig.has_value()) {
      try {
        ocp.finalCostPtr->get<DcmTerminalCost>("dcmTerminalCost").setConfig(*dcmTerminalConfig);
      } catch (const std::out_of_range&) {
        // Expected if dcm_terminal_cost is not in the cost list
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update dcmTerminalCost: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update dcmTerminalCost: unknown exception";
      }
    }

    // ── CoM and ACoM tracking cost ──
    if (hasComAcom && ocp.stateCostPtr != nullptr) {
      try {
        ocp.stateCostPtr->get<ComAndAcomTrackingCost>("comAndAcomTrackingCost").setWeights(Q_com, Q_acom);
      } catch (const std::out_of_range&) {
        // Expected if useComAndAcomTracking is false
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update comAndAcomTrackingCost: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update comAndAcomTrackingCost: unknown exception";
      }
    }

    // ── Foot tracking costs ──
    if (hasFootTrackingWeights || hasFootActiveInStance) {
      for (const auto& footName : contactNames_) {
        try {
          auto& footCost = ocp.costPtr->get<CentroidalMpcEndEffectorFootCost>(footName + "_TaskSpaceKinematicsCost");
          if (hasFootTrackingWeights) {
            footCost.setWeights(footTrackingWeightsVec);
          }
          if (hasFootActiveInStance) {
            footCost.setActiveInStance(footActiveInStance);
          }
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
    //
    // WHICH delta is written depends on the penalty that is actually installed, not on the YAML. A schedule-gated cone
    // is wrapped in a RelaxedBarrierPenalty, whose `delta` is the width of its quadratic relaxation and is read
    // straight from the file. An UN-gated cone - the contact-implicit formulation - is wrapped in a
    // SquaredHingePenalty built with delta = 0, because a hinge's delta is the OFFSET of its zero, and the whole point
    // of the hinge there is that its zero sits exactly on the cone that a foot at zero wrench lies on. Writing the
    // barrier's delta into it would move that zero into the interior and reinstate the very force floor that dropping
    // `minNormalForce` and the friction cone's parabolic margin exists to remove: with the shipped friction settings
    // (mu 0.2, delta 5) a foot in flight would be charged 2.5 with a gradient of -1.0 pushing its normal force up.
    const std::function<vector_t(const RelaxedBarrierPenalty::Config&, bool)> coneParameters =
        [](const RelaxedBarrierPenalty::Config& barrier, bool scheduleGated) {
          return vector_t((vector_t(2) << barrier.mu, scheduleGated ? barrier.delta : 0.0).finished());
        };

    for (const std::string& footName : contactNames_) {
      // Contact wrench cone
      if (hasWrenchConeBarrier) {
        try {
          StateInputSoftConstraint& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_contactWrenchCone");
          const vector_t parameters = coneParameters(wrenchConeBarrier, softCon.get<ContactWrenchConeConstraint>().isScheduleGated());
          for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
            penalty->setParameters(parameters);
          }
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_contactWrenchCone: " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_contactWrenchCone: unknown exception";
        }
      }

      // Friction force cone
      if (hasFrictionConeBarrier) {
        try {
          StateInputSoftConstraint& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_frictionForceCone");
          const vector_t parameters = coneParameters(frictionConeBarrier, softCon.get<FrictionForceConeConstraint>().isScheduleGated());
          for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
            penalty->setParameters(parameters);
          }
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_frictionForceCone: " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_frictionForceCone: unknown exception";
        }
      }

      // Contact moment XY
      if (hasContactMomentBarrier) {
        try {
          StateInputSoftConstraint& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_contactMomentXY");
          const vector_t parameters = coneParameters(contactMomentBarrier, softCon.get<ContactMomentXYConstraintCppAd>().isScheduleGated());
          for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
            penalty->setParameters(parameters);
          }
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_contactMomentXY: " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_contactMomentXY: unknown exception";
        }
      }

      // Basis scaling non-negativity barrier (λ ≥ 0)
      if (hasBasisNonNegativityBarrier) {
        try {
          ocp.costPtr->get<BasisScalingNonNegativityConstraint>(footName + "_basisNonNegativity")
              .setBarrierPenalty(basisNonNegativityBarrier);
        } catch (const std::out_of_range&) {
          // Expected if not in basis-vector mode
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_basisNonNegativity: " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_basisNonNegativity: unknown exception";
        }
      }
    }

    // ── Joint limits ──
    if (hasJointLimitsBarrier) {
      try {
        ocp.stateSoftConstraintPtr->get<JointLimitsSoftConstraint>("jointLimits").setGains(jointLimitsBarrier.mu, jointLimitsBarrier.delta);
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update jointLimits: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update jointLimits: unknown exception";
      }
    }

    // ── Foot collision ──
    if (hasCollisionBarrier) {
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
    }

    // ── Contact implicit soft constraints ──
    if (complementarityWeight >= 0.0 || heightReference > 0.0 || gapSmoothing > 0.0 || std::isfinite(terrainHeight)) {
      vector_t param(1);
      param[0] = complementarityWeight;
      for (const std::string& footName : contactNames_) {
        try {
          StateInputSoftConstraint& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_contactComplementarity");
          if (complementarityWeight >= 0.0) {
            for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
              penalty->setParameters(param);
            }
          }
          ContactComplementarityConstraint& constraint = softCon.get<ContactComplementarityConstraint>();
          if (heightReference > 0.0) {
            constraint.setHeightReference(heightReference);
          }
          if (gapSmoothing > 0.0) {
            constraint.setGapSmoothing(gapSmoothing);
          }
          if (std::isfinite(terrainHeight)) {
            constraint.setTerrainHeight(terrainHeight);
          }
        } catch (const std::out_of_range&) {
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_contactComplementarity: " << e.what();
        } catch (...) {
        }
      }
    }

    if (slipWeight >= 0.0 || velocityReference > 0.0 || angularVelocityReference > 0.0) {
      vector_t param(1);
      param[0] = slipWeight;
      for (const std::string& footName : contactNames_) {
        try {
          StateInputSoftConstraint& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_forceWeightedSlip");
          if (slipWeight >= 0.0) {
            for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
              penalty->setParameters(param);
            }
          }
          // The setter takes both references at once, so a file carrying only one of them keeps the other at the value
          // the term already holds rather than dropping the update on the floor.
          if (velocityReference > 0.0 || angularVelocityReference > 0.0) {
            ForceWeightedSlipConstraint& constraint = softCon.get<ForceWeightedSlipConstraint>();
            const vector3_t inverseCurrent = constraint.getInverseTwistReference();
            const scalar_t velocity = velocityReference > 0.0 ? velocityReference : 1.0 / inverseCurrent(0);
            const scalar_t angularVelocity = angularVelocityReference > 0.0 ? angularVelocityReference : 1.0 / inverseCurrent(2);
            constraint.setTwistReferences(velocity, angularVelocity);
          }
        } catch (const std::out_of_range&) {
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_forceWeightedSlip: " << e.what();
        } catch (...) {
        }
      }
    }

    // The terrain height has to reach this term as well as the complementarity term. The two are a pair - one says a
    // foot may not carry load above the ground, the other that it may not go below it - so a height applied to only
    // one of them leaves the formulation with two disagreeing definitions of where the ground is.
    if (hasPenetrationWeight || std::isfinite(terrainHeight)) {
      // SquaredHingePenalty::setParameters takes (mu, delta); delta stays 0 so the hinge's zero stays on the ground.
      const vector_t penetrationParams = (vector_t(2) << penetrationWeight, 0.0).finished();
      for (const std::string& footName : contactNames_) {
        try {
          StateSoftConstraint& softCon = ocp.stateSoftConstraintPtr->get<StateSoftConstraint>(footName + "_groundPenetration");
          if (std::isfinite(terrainHeight)) {
            softCon.get<GroundPenetrationConstraint>().setTerrainHeight(terrainHeight);
          }
          if (!hasPenetrationWeight) {
            continue;
          }
          for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
            penalty->setParameters(penetrationParams);
          }
        } catch (const std::out_of_range&) {
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_groundPenetration: " << e.what();
        } catch (...) {
        }
      }
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

    // ── Normal velocity soft constraint weight ──
    if (normalVelSoftWeight > 0.0) {
      vector_t scaleParam(1);
      scaleParam[0] = normalVelSoftWeight;
      for (const std::string& footName : contactNames_) {
        try {
          StateInputSoftConstraint& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(footName + "_normalVelocitySoft");
          for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
            penalty->setParameters(scaleParam);
          }
        } catch (const std::out_of_range&) {
          // The term is absent whenever normal_velocity is not listed as a soft constraint, which is every
          // schedule-gated configuration. That is not a failure and must not warn, or the log fills up once per foot
          // per slider drag.
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_normalVelocitySoft: " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_normalVelocitySoft: unknown exception";
        }
      }
    }

    // ── Foot constraint error gains ──
    if (hasFootConstraintGains) {
      // positionErrorGain_z has to reach the PRE-COMPUTATION as well as the zeroVelocity twist config below, and
      // that is not a refinement: under the contact-implicit formulation zeroVelocity is not built at all, so the
      // block below writes the gain into a term that does not exist while the term that does - the soft
      // normal-velocity servo - reads it from here. Without this the task-file key, and the slider the dashboard
      // renders for it, silently did nothing until the next launch.
      //
      // Each worker thread owns its own PreComputation, so writing it per OCP is also what keeps this race-free;
      // mutating the shared ModelSettings would not be.
      HumanoidPreComputation* preComputationPtr = dynamic_cast<HumanoidPreComputation*>(ocp.preComputationPtr.get());
      if (preComputationPtr != nullptr) {
        preComputationPtr->setNormalVelocityPositionErrorGain(footCfg.positionErrorGain_z);
      }

      for (const auto& footName : contactNames_) {
        // Hard constraint path
        try {
          auto& con = ocp.equalityConstraintPtr->get<ZeroVelocityConstraintCppAd>(footName + "_zeroVelocity");
          con.getTwistConstraint().setNumConstraints(footCfg.constrainOrientation ? 6 : 3);
          con.getTwistConstraint().setConstrainYawRateAboutNormal(footCfg.constrainYawRateAboutContactNormal);
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
            zeroVelCon->getTwistConstraint().setNumConstraints(footCfg.constrainOrientation ? 6 : 3);
            zeroVelCon->getTwistConstraint().setConstrainYawRateAboutNormal(footCfg.constrainYawRateAboutContactNormal);
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

  // ── Contact planning config, when the file carries the block (a task file with it inline, or the YAML published on
  // the topic, which the GUI assembles from both files) ──
  if (pt.get_child_optional("contact_planning")) {
    applyContactPlanningUpdates(yamlFile);
  }

  LOG(INFO) << "[MpcParameterUpdaterModule] Successfully applied in-place parameter updates to SqpSolver.";
}

void MpcParameterUpdaterModule::applyContactPlanningUpdates(const std::string& yamlFile) {
  if (contactPlannerModulePtr_ == nullptr) return;
  // Applied before the planner's next run.
  try {
    contactPlannerModulePtr_->setConfig(loadContactPlanningConfig(yamlFile, "contact_planning.", false));
    LOG(INFO) << "[MpcParameterUpdaterModule] Applied the contact_planning configuration from " << yamlFile << ".";
  } catch (const std::exception& e) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] contact_planning configuration of " << yamlFile << " could not be applied: " << e.what();
  }
}

}  // namespace ocs2::humanoid
