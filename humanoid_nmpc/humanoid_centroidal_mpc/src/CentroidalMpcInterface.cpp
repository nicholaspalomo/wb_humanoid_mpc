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

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/Numerics.h>
#include <ocs2_core/penalties/Penalties.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>

#include <humanoid_common_mpc/HumanoidCostConstraintFactory.h>
#include <humanoid_common_mpc/HumanoidPreComputation.h>
#include <humanoid_common_mpc/common/MpcFormulationConfig.h>
#include <humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h>
#include <humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h>
#include <humanoid_common_mpc/pinocchio_model/createPinocchioModel.h>
#include "humanoid_common_mpc/common/StatusMacros.h"

#include "humanoid_centroidal_mpc/constraint/JointMimicKinematicConstraint.h"
#include "humanoid_centroidal_mpc/constraint/NormalVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/constraint/ZeroVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsAD.h"

// Boost
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

CentroidalMpcInterface::CentroidalMpcInterface(const std::string& taskFile,
                                               const std::string& urdfFile,
                                               const std::string& referenceFile,
                                               bool setupOCP)
    : taskFile_(taskFile),
      urdfFile_(urdfFile),
      referenceFile_(referenceFile),
      modelSettings_(taskFile, urdfFile, "centroidal_mpc_", "true") {
  // check that task file exists
  boost::filesystem::path taskFilePath(taskFile);
  if (boost::filesystem::exists(taskFilePath)) {
    LOG(INFO) << "[CentroidalMpcInterface] Loading task file: " << taskFilePath;
  } else {
    throw std::invalid_argument(absl::StrCat("[CentroidalMpcInterface] Task file not found: ", taskFilePath.string()));
  }
  // check that urdf file exists
  boost::filesystem::path urdfFilePath(urdfFile);
  if (boost::filesystem::exists(urdfFilePath)) {
    LOG(INFO) << "[CentroidalMpcInterface] Loading Pinocchio model from: " << urdfFilePath;
  } else {
    throw std::invalid_argument(absl::StrCat("[CentroidalMpcInterface] URDF file not found: ", urdfFilePath.string()));
  }
  // check that targetCommand file exists
  boost::filesystem::path referenceFilePath(referenceFile);
  if (boost::filesystem::exists(referenceFilePath)) {
    LOG(INFO) << "[CentroidalMpcInterface] Loading target command settings from: " << referenceFilePath;
  } else {
    throw std::invalid_argument(absl::StrCat("[CentroidalMpcInterface] targetCommand file not found: ", referenceFilePath.string()));
  }

  loadData::loadCppDataType(taskFile, "interface.verbose", verbose_);

  // load setting from loading file
  ddpSettings_ = ddp::loadSettings(taskFile, "ddp", verbose_);
  mpcSettings_ = mpc::loadSettings(taskFile, "mpc", verbose_);
  rolloutSettings_ = rollout::loadSettings(taskFile, "rollout", verbose_);
  sqpSettings_ = sqp::loadSettings(taskFile, "multiple_shooting", verbose_);

  // PinocchioInterface
  pinocchioInterfacePtr_.reset(new PinocchioInterface(createCustomPinocchioInterface(taskFile, urdfFile, modelSettings_, false)));

  // CentroidalModelInfo
  centroidalModelInfo_ = centroidal_model::createCentroidalModelInfo(
      *pinocchioInterfacePtr_, centroidal_model::loadCentroidalType(taskFile),
      centroidal_model::loadDefaultJointState(pinocchioInterfacePtr_->getModel().nq - 6, referenceFile), modelSettings_.contactNames3DoF,
      modelSettings_.contactNames6DoF);

  LOG(INFO) << "centroidalModelInfo_.numSixDofContacts: " << centroidalModelInfo_.numSixDofContacts;
  for (int i = 0; i < centroidalModelInfo_.numSixDofContacts; i++) {
    LOG(INFO) << "frameIndices: " << centroidalModelInfo_.endEffectorFrameIndices[i];
  }

  // Setup Centroidal State Input Mapping
  mpcRobotModelPtr_.reset(new CentroidalMpcRobotModel<scalar_t>(modelSettings_, *pinocchioInterfacePtr_, centroidalModelInfo_));
  mpcRobotModelADPtr_.reset(
      new CentroidalMpcRobotModel<ad_scalar_t>(modelSettings_, (*pinocchioInterfacePtr_).toCppAd(), centroidalModelInfo_.toCppAd()));

  // Swing trajectory planner
  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(
      new SwingTrajectoryPlanner(loadSwingTrajectorySettings(taskFile, "swing_trajectory_config", verbose_), N_CONTACTS));

  referenceManagerPtr_ =
      std::make_shared<SwitchedModelReferenceManager>(GaitSchedule::loadGaitSchedule(referenceFile, modelSettings_, verbose_),
                                                      std::move(swingTrajectoryPlanner), *pinocchioInterfacePtr_, *mpcRobotModelPtr_);
  referenceManagerPtr_->setArmSwingReferenceActive(true);

  // initial state
  initialState_.setZero(centroidalModelInfo_.stateDim);
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

absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> CentroidalMpcInterface::Create(const std::string& taskFile,
                                                                                       const std::string& urdfFile,
                                                                                       const std::string& referenceFile) {
  std::unique_ptr<CentroidalMpcInterface> interface(new CentroidalMpcInterface(taskFile, urdfFile, referenceFile, /*setupOCP=*/false));
  RETURN_IF_ERROR(interface->setupOptimalControlProblem());
  return interface;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status CentroidalMpcInterface::setupOptimalControlProblem() {
  HumanoidCostConstraintFactory factory =
      HumanoidCostConstraintFactory(taskFile_, referenceFile_, *referenceManagerPtr_, *pinocchioInterfacePtr_, *mpcRobotModelPtr_,
                                    *mpcRobotModelADPtr_, modelSettings_, verbose_);

  // Optimal control problem
  problemPtr_.reset(new OptimalControlProblem);

  // Dynamics
  std::unique_ptr<SystemDynamicsBase> dynamicsPtr;
  const std::string modelName = "dynamics";
  dynamicsPtr.reset(new CentroidalDynamicsAD(*pinocchioInterfacePtr_, centroidalModelInfo_, modelName, modelSettings_));
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
  if (formulationTasks.hasCost(MpcCostType::TerminalCost)) {
    problemPtr_->finalCostPtr->add("terminalCost", factory.getTerminalCost());
  }

  const auto infoCppAd = centroidalModelInfo_.toCppAd();
  const CentroidalModelPinocchioMappingCppAd pinocchioMappingCppAd(infoCppAd);

  auto velocityUpdateCallback = [&infoCppAd](const ad_vector_t& state, PinocchioInterfaceCppAd& pinocchioInterfaceAd) {
    const ad_vector_t q = centroidal_model::getGeneralizedCoordinates(state, infoCppAd);
    updateCentroidalDynamics(pinocchioInterfaceAd, infoCppAd, q);
  };

  if (formulationTasks.hasCost(MpcCostType::TaskSpaceTorsoCost)) {
    addTaskSpaceKinematicsCosts(pinocchioMappingCppAd, velocityUpdateCallback);
  }

  if (formulationTasks.hasCost(MpcCostType::IcpCost)) {
    const vector2_t icpWeights = ICPCost::getWeights(taskFile_, "icp_cost_weights.", verbose_);
    problemPtr_->costPtr->add(
        "icp_Cost", std::unique_ptr<StateInputCost>(new ICPCost(*referenceManagerPtr_, std::move(icpWeights), *pinocchioInterfacePtr_,
                                                                *mpcRobotModelADPtr_, "icp_Cost", modelSettings_)));
  }

  // Soft constraints
  if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::JointLimits)) {
    problemPtr_->stateSoftConstraintPtr->add("jointLimits", factory.getJointLimitsConstraint());
  }
  if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::FootCollision)) {
    problemPtr_->stateSoftConstraintPtr->add("FootCollisionSoftConstraint", factory.getFootCollisionConstraint());
  }

  // Constraint terms
  EndEffectorKinematicsWeights footTrackingCostWeights;
  if (formulationTasks.hasCost(MpcCostType::TaskSpaceFootCost)) {
    footTrackingCostWeights = EndEffectorKinematicsWeights::getWeights(taskFile_, "task_space_foot_cost_weights.", verbose_);
  }

  for (size_t i = 0; i < N_CONTACTS; i++) {
    const std::string& footName = modelSettings_.contactNames[i];

    std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKinematicsPtr;
    bool needsEeKinematics = formulationTasks.hasHardConstraint(MpcHardConstraintType::ZeroVelocity) ||
                             formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ZeroVelocity) ||
                             formulationTasks.hasHardConstraint(MpcHardConstraintType::NormalVelocity);
    if (needsEeKinematics) {
      eeKinematicsPtr.reset(new PinocchioEndEffectorKinematicsCppAd(*pinocchioInterfacePtr_, pinocchioMappingCppAd, {footName},
                                                                    centroidalModelInfo_.stateDim, centroidalModelInfo_.inputDim,
                                                                    velocityUpdateCallback, footName, modelSettings_.modelFolderCppAd,
                                                                    modelSettings_.recompileLibrariesCppAd, modelSettings_.verboseCppAd));
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
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ZeroVelocity) && eeKinematicsPtr) {
      auto stanceConstraint = getStanceFootConstraint(*eeKinematicsPtr, i);
      auto penalty = std::make_unique<QuadraticPenalty>(modelSettings_.footConstraintConfig.softConstraintWeight);
      problemPtr_->softConstraintPtr->add(absl::StrCat(footName, "_zeroVelocity"),
                                          std::make_unique<StateInputSoftConstraint>(std::move(stanceConstraint), std::move(penalty)));
    }

    if (formulationTasks.hasHardConstraint(MpcHardConstraintType::ZeroWrench)) {
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_zeroWrench"), factory.getZeroWrenchConstraint(i));
    }
    if (formulationTasks.hasHardConstraint(MpcHardConstraintType::ZeroVelocity) && eeKinematicsPtr) {
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_zeroVelocity"), getStanceFootConstraint(*eeKinematicsPtr, i));
    }
    if (formulationTasks.hasHardConstraint(MpcHardConstraintType::NormalVelocity) && eeKinematicsPtr) {
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_normalVelocity"), getNormalVelocityConstraint(*eeKinematicsPtr, i));
    }
    if (formulationTasks.hasHardConstraint(MpcHardConstraintType::KneeJointMimic)) {
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_kneeJointMimic"), getJointMimicConstraint(i));
    }

    if (formulationTasks.hasCost(MpcCostType::TaskSpaceFootCost)) {
      std::string footTrackingCostName = absl::StrCat(footName, "_TaskSpaceKinematicsCost");
      problemPtr_->costPtr->add(footTrackingCostName, std::unique_ptr<StateInputCost>(new CentroidalMpcEndEffectorFootCost(
                                                          *referenceManagerPtr_, footTrackingCostWeights, *pinocchioInterfacePtr_,
                                                          *mpcRobotModelADPtr_, i, footTrackingCostName, modelSettings_)));
    }
    if (formulationTasks.hasCost(MpcCostType::ExternalTorqueCost)) {
      problemPtr_->costPtr->add(absl::StrCat(footName, "_ExternalTorqueQuadraticCost"), factory.getExternalTorqueQuadraticCost(i));
    }
  }

  // Pre-computation
  problemPtr_->preComputationPtr.reset(
      new HumanoidPreComputation(*pinocchioInterfacePtr_, *referenceManagerPtr_->getSwingTrajectoryPlanner(), *mpcRobotModelPtr_));

  // Rollout
  rolloutPtr_.reset(new TimeTriggeredRollout(*problemPtr_->dynamicsPtr, rolloutSettings_));

  // Initialization
  constexpr bool extendNormalizedMomentum = true;
  initializerPtr_.reset(
      new CentroidalWeightCompInitializer(centroidalModelInfo_, *referenceManagerPtr_, *mpcRobotModelPtr_, extendNormalizedMomentum));

  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputConstraint> CentroidalMpcInterface::getStanceFootConstraint(const EndEffectorKinematics<scalar_t>& eeKinematics,
                                                                                      size_t contactPointIndex) {
  auto eeZeroVelConConfig = [](const ModelSettings::FootConstraintConfig& footConfig) {
    EndEffectorKinematicsTwistConstraint::Config config;
    config.b.setZero(6);
    config.Ax.setZero(6, 6);
    config.Av.setZero(6, 6);

    // Position error gain: only z-axis (foot height tracking during stance)
    if (!numerics::almost_eq(footConfig.positionErrorGain_z, 0.0)) {
      config.Ax(2, 2) = footConfig.positionErrorGain_z;
    }
    // Orientation error gain: all 3 rotation axes
    if (!numerics::almost_eq(footConfig.orientationErrorGain, 0.0)) {
      config.Ax.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * footConfig.orientationErrorGain;
    }

    // Linear velocity gains: xy and z separately
    config.Av(0, 0) = footConfig.linearVelocityErrorGain_xy;
    config.Av(1, 1) = footConfig.linearVelocityErrorGain_xy;
    config.Av(2, 2) = footConfig.linearVelocityErrorGain_z;

    // Angular velocity gain: all 3 rotation axes
    config.Av(3, 3) = footConfig.angularVelocityErrorGain;
    config.Av(4, 4) = footConfig.angularVelocityErrorGain;
    config.Av(5, 5) = footConfig.angularVelocityErrorGain;

    return config;
  };

  return std::unique_ptr<StateInputConstraint>(new ZeroVelocityConstraintCppAd(*referenceManagerPtr_, eeKinematics, contactPointIndex,
                                                                               eeZeroVelConConfig(modelSettings_.footConstraintConfig)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputConstraint> CentroidalMpcInterface::getNormalVelocityConstraint(
    const EndEffectorKinematics<scalar_t>& eeKinematics, size_t contactPointIndex) {
  return std::unique_ptr<StateInputConstraint>(new NormalVelocityConstraintCppAd(*referenceManagerPtr_, eeKinematics, contactPointIndex));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputConstraint> CentroidalMpcInterface::getJointMimicConstraint(size_t mimicIndex) {
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

  if (verbose_) {
    LOG(INFO) << "\n #### Joint Mimic Kinematic Constraint Config: \n"
              << " #### =============================================================================";
  }
  loadData::loadPtreeValue(pt, parentJointName, absl::StrCat(prefix, "parentJointName"), verbose_);
  loadData::loadPtreeValue(pt, childJointName, absl::StrCat(prefix, "childJointName"), verbose_);
  loadData::loadPtreeValue(pt, multiplier, absl::StrCat(prefix, "multiplier"), verbose_);
  loadData::loadPtreeValue(pt, positionGain, absl::StrCat(prefix, "positionGain"), verbose_);
  if (verbose_) {
    LOG(INFO) << " #### =============================================================================";
  }

  JointMimicKinematicConstraint::Config config(*mpcRobotModelPtr_, parentJointName, childJointName, multiplier, positionGain);

  return std::unique_ptr<StateInputConstraint>(new JointMimicKinematicConstraint(*mpcRobotModelPtr_, config));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void CentroidalMpcInterface::addTaskSpaceKinematicsCosts(
    const CentroidalModelPinocchioMappingCppAd& pinocchioMappingCppAd,
    const PinocchioEndEffectorKinematicsCppAd::update_pinocchio_interface_callback& velocityUpdateCallback) {
  boost::property_tree::ptree pt;
  loadData::readPropertyTree(taskFile_, pt);

  boost::property_tree::ptree task_space_costs_pt = pt.get_child("task_space_costs");

  for (auto& task_space_cost : task_space_costs_pt) {
    std::string costName = task_space_cost.first;
    std::string linkName;

    loadData::loadPtreeValue(task_space_costs_pt, linkName, absl::StrCat(costName, ".link_name"), verbose_);

    std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKinematicsPtr;

    eeKinematicsPtr.reset(new PinocchioEndEffectorKinematicsCppAd(*pinocchioInterfacePtr_, pinocchioMappingCppAd, {linkName},
                                                                  centroidalModelInfo_.stateDim, centroidalModelInfo_.inputDim,
                                                                  velocityUpdateCallback, linkName, modelSettings_.modelFolderCppAd,
                                                                  modelSettings_.recompileLibrariesCppAd, modelSettings_.verboseCppAd));

    EndEffectorKinematicsWeights weights =
        EndEffectorKinematicsWeights::getWeights(taskFile_, absl::StrCat("task_space_costs.", costName, ".weights."), verbose_);

    std::unique_ptr<StateInputCost> cost = std::make_unique<EndEffectorKinematicsQuadraticCost>(
        weights, *pinocchioInterfacePtr_, *eeKinematicsPtr, *mpcRobotModelADPtr_, linkName, modelSettings_);

    problemPtr_->costPtr->add(absl::StrCat(costName, "_TaskSpaceKinematicsCost"), std::move(cost));

    LOG(INFO) << "Initialized Task Space Kinematics Cost for link: " << linkName;
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<std::string> CentroidalMpcInterface::getCostNames() const {
  std::vector<std::string> costNames;
  for (const auto& [costName, index] : problemPtr_->costPtr->getTermNameMap()) {
    costNames.emplace_back(costName);
  }
  return costNames;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<std::string> CentroidalMpcInterface::getTerminalCostNames() const {
  std::vector<std::string> terminalCostNames;
  for (const auto& [costName, index] : problemPtr_->finalCostPtr->getTermNameMap()) {
    terminalCostNames.emplace_back(costName);
  }
  return terminalCostNames;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<std::string> CentroidalMpcInterface::getStateSoftConstraintNames() const {
  std::vector<std::string> costNames;
  for (const auto& [costName, index] : problemPtr_->stateSoftConstraintPtr->getTermNameMap()) {
    costNames.emplace_back(costName);
  }
  return costNames;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<std::string> CentroidalMpcInterface::getSoftConstraintNames() const {
  std::vector<std::string> costNames;
  for (const auto& [costName, index] : problemPtr_->softConstraintPtr->getTermNameMap()) {
    costNames.emplace_back(costName);
  }
  return costNames;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<std::string> CentroidalMpcInterface::getEqualityConstraintNames() const {
  std::vector<std::string> costNames;
  for (const auto& [costName, index] : problemPtr_->equalityConstraintPtr->getTermNameMap()) {
    costNames.emplace_back(costName);
  }
  return costNames;
}

}  // namespace ocs2::humanoid
