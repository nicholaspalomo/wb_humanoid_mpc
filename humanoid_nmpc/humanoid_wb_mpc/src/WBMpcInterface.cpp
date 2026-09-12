/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

// Pinocchio forward declarations must be included first
#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_wb_mpc/WBMpcInterface.h"

#include <ocs2_core/misc/Display.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/Numerics.h>
#include <ocs2_core/penalties/Penalties.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>

#include <humanoid_common_mpc/pinocchio_model/createPinocchioModel.h>
#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/initialization/WeightCompInitializer.h"

#include "humanoid_wb_mpc/WBMpcPreComputation.h"
#include "humanoid_wb_mpc/constraint/JointMimicDynamicsConstraint.h"
#include "humanoid_wb_mpc/constraint/SwingLegVerticalConstraintCppAd.h"
#include "humanoid_wb_mpc/constraint/ZeroAccelerationConstraintCppAd.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsFootCost.h"
#include "humanoid_wb_mpc/cost/JointTorqueCostCppAd.h"
#include "humanoid_wb_mpc/dynamics/WBAccelDynamicsAD.h"
#include "humanoid_wb_mpc/end_effector/PinocchioEndEffectorDynamicsCppAd.h"

// Boost
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
WBMpcInterface::WBMpcInterface(const std::string& taskFile, const std::string& urdfFile, const std::string& referenceFile, bool setupOCP)
    : taskFile_(taskFile), urdfFile_(urdfFile), referenceFile_(referenceFile), modelSettings_(taskFile, urdfFile, "wb_mpc_", "true") {
  // check that task file exists
  boost::filesystem::path taskFilePath(taskFile);
  if (boost::filesystem::exists(taskFilePath)) {
    LOG(INFO) << "[WBMpcInterface] Loading task file: " << taskFilePath;
  } else {
    throw std::invalid_argument(absl::StrCat("[WBMpcInterface] Task file not found: ", taskFilePath.string()));
  }
  // check that urdf file exists
  boost::filesystem::path urdfFilePath(urdfFile);
  if (boost::filesystem::exists(urdfFilePath)) {
    LOG(INFO) << "[WBMpcInterface] Loading Pinocchio model from: " << urdfFilePath;
  } else {
    throw std::invalid_argument(absl::StrCat("[WBMpcInterface] URDF file not found: ", urdfFilePath.string()));
  }
  // check that targetCommand file exists
  boost::filesystem::path referenceFilePath(referenceFile);
  if (boost::filesystem::exists(referenceFilePath)) {
    LOG(INFO) << "[WBMpcInterface] Loading target command settings from: " << referenceFilePath;
  } else {
    throw std::invalid_argument(absl::StrCat("[WBMpcInterface] targetCommand file not found: ", referenceFilePath.string()));
  }

  loadData::loadCppDataType(taskFile, "interface.verbose", verbose_);

  // load setting from loading file
  ddpSettings_ = ddp::loadSettings(taskFile, "ddp", verbose_);
  mpcSettings_ = mpc::loadSettings(taskFile, "mpc", verbose_);
  rolloutSettings_ = rollout::loadSettings(taskFile, "rollout", verbose_);
  sqpSettings_ = sqp::loadSettings(taskFile, "multiple_shooting", verbose_);

  // PinocchioInterface
  pinocchioInterfacePtr_.reset(new PinocchioInterface(createCustomPinocchioInterface(taskFile, urdfFile, modelSettings_)));

  // Setup WB State Input Mapping
  mpcRobotModelPtr_.reset(new WBAccelMpcRobotModel<scalar_t>(modelSettings_));
  mpcRobotModelADPtr_.reset(new WBAccelMpcRobotModel<ad_scalar_t>(modelSettings_));

  // Swing trajectory planner
  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(
      new SwingTrajectoryPlanner(loadSwingTrajectorySettings(taskFile, "swing_trajectory_config", verbose_), N_CONTACTS));

  // Mode schedule manager
  referenceManagerPtr_ =
      std::make_shared<SwitchedModelReferenceManager>(GaitSchedule::loadGaitSchedule(referenceFile, modelSettings_, verbose_),
                                                      std::move(swingTrajectoryPlanner), *pinocchioInterfacePtr_, *mpcRobotModelPtr_);
  referenceManagerPtr_->setArmSwingReferenceActive(true);

  // initial state
  initialState_.setZero(mpcRobotModelPtr_->getStateDim());
  loadData::loadEigenMatrix(taskFile, "initialState", initialState_);

  if (setupOCP) {
    absl::Status status = setupOptimalControlProblem();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::StatusOr<std::unique_ptr<WBMpcInterface>> WBMpcInterface::Create(const std::string& taskFile,
                                                                       const std::string& urdfFile,
                                                                       const std::string& referenceFile) {
  std::unique_ptr<WBMpcInterface> interface(new WBMpcInterface(taskFile, urdfFile, referenceFile, /*setupOCP=*/false));
  RETURN_IF_ERROR(interface->setupOptimalControlProblem());
  return interface;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status WBMpcInterface::setupOptimalControlProblem() {
  HumanoidCostConstraintFactory factory =
      HumanoidCostConstraintFactory(taskFile_, referenceFile_, *referenceManagerPtr_, *pinocchioInterfacePtr_, *mpcRobotModelPtr_,
                                    *mpcRobotModelADPtr_, modelSettings_, verbose_);

  // Optimal control problem
  problemPtr_.reset(new OptimalControlProblem);

  // Dynamics
  std::unique_ptr<SystemDynamicsBase> dynamicsPtr;
  const std::string modelName = "dynamics";
  dynamicsPtr.reset(new WBAccelDynamicsAD(*pinocchioInterfacePtr_, *mpcRobotModelADPtr_, modelName, modelSettings_));

  problemPtr_->dynamicsPtr = std::move(dynamicsPtr);

  // Load configured MPC formulation tasks
  ASSIGN_OR_RETURN(const MpcFormulationTasks formulationTasks, loadMpcFormulationTasks(taskFile_, verbose_));

  // Cost terms
  if (formulationTasks.hasCost(MpcCostType::StateInputQuadraticCost)) {
    problemPtr_->costPtr->add("stateInputQuadraticCost", factory.getStateInputQuadraticCost());
  }
  if (formulationTasks.hasCost(MpcCostType::StateQuadraticCost)) {
    problemPtr_->costPtr->add("stateQuadraticCost", factory.getStateQuadraticCost());
  }
  if (formulationTasks.hasCost(MpcCostType::InputQuadraticCost)) {
    problemPtr_->costPtr->add("inputQuadraticCost", factory.getInputQuadraticCost());
  }
  if (formulationTasks.hasCost(MpcCostType::JointTorqueCost)) {
    problemPtr_->costPtr->add("jointTorqueCost", getJointTorqueCost(taskFile_));
  }
  if (formulationTasks.hasCost(MpcCostType::TerminalCost)) {
    problemPtr_->finalCostPtr->add("terminalCost", factory.getTerminalCost());
  }

  // Soft constraints
  if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::JointLimits)) {
    problemPtr_->stateSoftConstraintPtr->add("jointLimits", factory.getJointLimitsConstraint());
  }
  if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::FootCollision)) {
    problemPtr_->stateSoftConstraintPtr->add("FootCollisionSoftConstraint", factory.getFootCollisionConstraint());
  }

  // Foot tracking cost weights
  EndEffectorDynamicsWeights footTrackingCostWeights;
  if (formulationTasks.hasCost(MpcCostType::TaskSpaceFootCost)) {
    footTrackingCostWeights = EndEffectorDynamicsWeights::getWeights(taskFile_, "task_space_foot_cost_weights.", verbose_);
  }

  for (size_t i = 0; i < N_CONTACTS; i++) {
    const std::string& footName = modelSettings_.contactNames[i];

    std::unique_ptr<EndEffectorDynamics<scalar_t>> eeDynamicsPtr;
    bool needsEeDynamics = formulationTasks.hasHardConstraint(MpcHardConstraintType::ZeroVelocity) ||
                           formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ZeroVelocity) ||
                           formulationTasks.hasHardConstraint(MpcHardConstraintType::NormalVelocity) ||
                           formulationTasks.hasCost(MpcCostType::TaskSpaceFootCost);
    if (needsEeDynamics) {
      eeDynamicsPtr.reset(new PinocchioEndEffectorDynamicsCppAd(*pinocchioInterfacePtr_, *mpcRobotModelADPtr_, {footName}, footName,
                                                                modelSettings_.modelFolderCppAd, modelSettings_.recompileLibrariesCppAd,
                                                                modelSettings_.verboseCppAd));
    }

    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactWrenchCone)) {
      problemPtr_->softConstraintPtr->add(absl::StrCat(footName, "_contactWrenchCone"), factory.getContactWrenchConeConstraint(i));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::FrictionForceCone)) {
      problemPtr_->softConstraintPtr->add(absl::StrCat(footName, "_frictionForceCone"), factory.getFrictionForceConeConstraint(i));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactMomentXY)) {
      problemPtr_->softConstraintPtr->add(absl::StrCat(footName, "_contactMomentXY"),
                                          factory.getContactMomentXYConstraint(i, absl::StrCat(footName, "_contact_moment_XY_constraint")));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ZeroVelocity) && eeDynamicsPtr) {
      auto stanceConstraint = getStanceFootConstraint(*eeDynamicsPtr, i);
      auto penalty = std::make_unique<QuadraticPenalty>(modelSettings_.footConstraintConfig.softConstraintWeight);
      problemPtr_->softConstraintPtr->add(absl::StrCat(footName, "_zeroVelocity"),
                                          std::make_unique<StateInputSoftConstraint>(std::move(stanceConstraint), std::move(penalty)));
    }

    if (formulationTasks.hasHardConstraint(MpcHardConstraintType::ZeroWrench)) {
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_zeroWrench"), factory.getZeroWrenchConstraint(i));
    }
    if (formulationTasks.hasHardConstraint(MpcHardConstraintType::ZeroVelocity) && eeDynamicsPtr) {
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_zeroVelocity"), getStanceFootConstraint(*eeDynamicsPtr, i));
    }
    if (formulationTasks.hasHardConstraint(MpcHardConstraintType::NormalVelocity) && eeDynamicsPtr) {
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_normalVelocity"), getNormalVelocityConstraint(*eeDynamicsPtr, i));
    }
    if (formulationTasks.hasHardConstraint(MpcHardConstraintType::KneeJointMimic)) {
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_kneeJointMimic"), getJointMimicConstraint(i));
    }

    if (formulationTasks.hasCost(MpcCostType::TaskSpaceFootCost) && eeDynamicsPtr) {
      std::string footTrackingCostName = absl::StrCat(footName, "_TaskSpaceTrackingCost");
      problemPtr_->costPtr->add(footTrackingCostName, std::unique_ptr<StateInputCost>(new EndEffectorDynamicsFootCost(
                                                          *referenceManagerPtr_, footTrackingCostWeights, *pinocchioInterfacePtr_,
                                                          *eeDynamicsPtr, *mpcRobotModelADPtr_, i, footTrackingCostName, modelSettings_)));
    }
  }

  // Pre-computation
  problemPtr_->preComputationPtr.reset(
      new WBMpcPreComputation(*pinocchioInterfacePtr_, *referenceManagerPtr_->getSwingTrajectoryPlanner(), *mpcRobotModelPtr_));

  // Rollout
  rolloutPtr_.reset(new TimeTriggeredRollout(*problemPtr_->dynamicsPtr, rolloutSettings_));

  // Initialization
  initializerPtr_.reset(new WeightCompInitializer(*pinocchioInterfacePtr_, *referenceManagerPtr_, *mpcRobotModelPtr_));

  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputConstraint> WBMpcInterface::getStanceFootConstraint(const EndEffectorDynamics<scalar_t>& eeDynamics,
                                                                              size_t contactPointIndex) {
  const ModelSettings::FootConstraintConfig& footCfg = modelSettings_.footConstraintConfig;

  EndEffectorDynamicsAccelerationsConstraint::Config config;
  config.b.setZero(6);
  config.Ax.setZero(6, 6);
  config.Av.setIdentity(6, 6);
  config.Aa.setIdentity(6, 6);
  if (!numerics::almost_eq(footCfg.positionErrorGain_z, 0.0)) {
    config.Ax(2, 2) = footCfg.positionErrorGain_z;
  }
  if (!numerics::almost_eq(footCfg.orientationErrorGain, 0.0)) {
    config.Ax.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * footCfg.orientationErrorGain;
  }
  config.Av.block(0, 0, 2, 2) = Eigen::MatrixXd::Identity(2, 2) * footCfg.linearVelocityErrorGain_xy;
  config.Av(2, 2) = footCfg.linearVelocityErrorGain_z;
  config.Av.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * footCfg.angularVelocityErrorGain;
  config.Aa.block(0, 0, 2, 2) = Eigen::MatrixXd::Identity(2, 2) * footCfg.linearAccelerationErrorGain_xy;
  config.Aa(2, 2) = footCfg.linearAccelerationErrorGain_z;
  config.Aa.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * footCfg.angularAccelerationErrorGain;

  return std::unique_ptr<StateInputConstraint>(
      new ZeroAccelerationConstraintCppAd(*referenceManagerPtr_, eeDynamics, contactPointIndex, config));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputConstraint> WBMpcInterface::getJointMimicConstraint(size_t mimicIndex) {
  boost::property_tree::ptree pt;
  loadData::readPropertyTree(taskFile_, pt);
  std::string prefix;
  if (mimicIndex == 0) {
    prefix = "mimicJoints.left_knee.";
  } else if (mimicIndex == 1) {
    prefix = "mimicJoints.right_knee.";
  } else {
    throw std::runtime_error(absl::StrCat("No mimic joint for index: ", mimicIndex));
  }

  std::string parentJointName;
  std::string childJointName;
  scalar_t multiplier;  // q_child = multiplier* q_parent
  scalar_t positionGain;
  scalar_t velocityGain;

  if (verbose_) {
    LOG(INFO) << "\n #### Joint Mimic Kinematic Constraint Config: \n"
              << " #### =============================================================================";
  }
  loadData::loadPtreeValue(pt, parentJointName, absl::StrCat(prefix, "parentJointName"), verbose_);
  loadData::loadPtreeValue(pt, childJointName, absl::StrCat(prefix, "childJointName"), verbose_);
  loadData::loadPtreeValue(pt, multiplier, absl::StrCat(prefix, "multiplier"), verbose_);
  loadData::loadPtreeValue(pt, positionGain, absl::StrCat(prefix, "positionGain"), verbose_);
  loadData::loadPtreeValue(pt, velocityGain, absl::StrCat(prefix, "velocityGain"), verbose_);
  if (verbose_) {
    LOG(INFO) << " #### =============================================================================";
  }

  JointMimicDynamicsConstraint::Config config(*mpcRobotModelPtr_, parentJointName, childJointName, multiplier, positionGain, velocityGain);

  return std::unique_ptr<StateInputConstraint>(new JointMimicDynamicsConstraint(*mpcRobotModelPtr_, config));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputConstraint> WBMpcInterface::getNormalVelocityConstraint(const EndEffectorDynamics<scalar_t>& eeDynamics,
                                                                                  size_t contactPointIndex) {
  return std::unique_ptr<StateInputConstraint>(new SwingLegVerticalConstraintCppAd(*referenceManagerPtr_, eeDynamics, contactPointIndex));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputCost> WBMpcInterface::getJointTorqueCost(const std::string& taskFile) {
  vector_t jointTorqueWeights(mpcRobotModelPtr_->getJointDim());
  loadData::loadEigenMatrix(taskFile, "joint_torque_weights", jointTorqueWeights);
  return std::unique_ptr<StateInputCost>(
      new JointTorqueCostCppAd(jointTorqueWeights, *pinocchioInterfacePtr_, *mpcRobotModelADPtr_, "jointTorqueCost", modelSettings_));
}

}  // namespace ocs2::humanoid
