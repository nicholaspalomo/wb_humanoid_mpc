/******************************************************************************
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

#include <iostream>
#include <string>

// Pinocchio forward declarations must be included first
#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/Numerics.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>

#include <humanoid_common_mpc/HumanoidCostConstraintFactory.h>
#include <humanoid_common_mpc/HumanoidPreComputation.h>
#include <humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h>
#include <humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h>
#include <humanoid_common_mpc/gait/GaitOptimizationSettings.h>
#include <humanoid_common_mpc/pinocchio_model/createPinocchioModel.h>
#include <humanoid_common_mpc/problem/MpcProblemBuilder.h>
#include <humanoid_common_mpc/problem/MpcProblemDefinition.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

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
    std::cerr << "[CentroidalMpcInterface] Loading task file: " << taskFilePath << std::endl;
  } else {
    throw std::invalid_argument(absl::StrCat("[CentroidalMpcInterface] Task file not found: ", taskFilePath.string()));
  }
  // check that urdf file exists
  boost::filesystem::path urdfFilePath(urdfFile);
  if (boost::filesystem::exists(urdfFilePath)) {
    std::cerr << "[CentroidalMpcInterface] Loading Pinocchio model from: " << urdfFilePath << std::endl;
  } else {
    throw std::invalid_argument(absl::StrCat("[CentroidalMpcInterface] URDF file not found: ", urdfFilePath.string()));
  }
  // check that targetCommand file exists
  boost::filesystem::path referenceFilePath(referenceFile);
  if (boost::filesystem::exists(referenceFilePath)) {
    std::cerr << "[CentroidalMpcInterface] Loading target command settings from: " << referenceFilePath << std::endl;
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

  std::cout << "centroidalModelInfo_.numSixDofContacts: " << centroidalModelInfo_.numSixDofContacts << std::endl;
  for (int i = 0; i < centroidalModelInfo_.numSixDofContacts; i++) {
    std::cout << "frameIndices: " << centroidalModelInfo_.endEffectorFrameIndices[i] << std::endl;
  }

  // Setup Centroidal State Input Mapping
  mpcRobotModelPtr_.reset(new CentroidalMpcRobotModel<scalar_t>(modelSettings_, *pinocchioInterfacePtr_, centroidalModelInfo_));
  mpcRobotModelADPtr_.reset(
      new CentroidalMpcRobotModel<ad_scalar_t>(modelSettings_, (*pinocchioInterfacePtr_).toCppAd(), centroidalModelInfo_.toCppAd()));

  // Swing trajectory planner
  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(
      new SwingTrajectoryPlanner(loadSwingTrajectorySettings(taskFile, "swing_trajectory_config", verbose_), N_CONTACTS));

  const absl::StatusOr<GaitOptimizationSettings> gaitOptimizationSettingsStatus =
      loadGaitOptimizationSettings(taskFile, "gait_optimization", verbose_);
  const GaitOptimizationSettings gaitOptimizationSettings =
      gaitOptimizationSettingsStatus.ok() ? gaitOptimizationSettingsStatus.value() : GaitOptimizationSettings();

  referenceManagerPtr_ = std::make_shared<SwitchedModelReferenceManager>(
      GaitSchedule::loadGaitSchedule(referenceFile, modelSettings_, verbose_), std::move(swingTrajectoryPlanner), *pinocchioInterfacePtr_,
      *mpcRobotModelPtr_, gaitOptimizationSettings);
  referenceManagerPtr_->setArmSwingReferenceActive(true);

  // initial state
  initialState_.setZero(centroidalModelInfo_.stateDim);
  loadData::loadEigenMatrix(taskFile, "initialState", initialState_);

  if (setupOCP) {
    setupOptimalControlProblem();
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void CentroidalMpcInterface::setupOptimalControlProblem() {
  const HumanoidCostConstraintFactory factory(taskFile_, referenceFile_, *referenceManagerPtr_, *pinocchioInterfacePtr_, *mpcRobotModelPtr_,
                                              *mpcRobotModelADPtr_, modelSettings_, verbose_);

  // Optimal control problem
  problemPtr_.reset(new OptimalControlProblem);

  // Dynamics
  std::unique_ptr<SystemDynamicsBase> dynamicsPtr;
  const std::string modelName = "dynamics";
  dynamicsPtr.reset(new CentroidalDynamicsAD(*pinocchioInterfacePtr_, centroidalModelInfo_, modelName, modelSettings_));
  problemPtr_->dynamicsPtr = std::move(dynamicsPtr);

  const auto infoCppAd = centroidalModelInfo_.toCppAd();
  const CentroidalModelPinocchioMappingCppAd pinocchioMappingCppAd(infoCppAd);

  auto velocityUpdateCallback = [&infoCppAd](const ad_vector_t& state, PinocchioInterfaceCppAd& pinocchioInterfaceAd) {
    const ad_vector_t q = centroidal_model::getGeneralizedCoordinates(state, infoCppAd);
    updateCentroidalDynamics(pinocchioInterfaceAd, infoCppAd, q);
  };

  const EndEffectorKinematicsWeights footTrackingCostWeights =
      EndEffectorKinematicsWeights::getWeights(taskFile_, "task_space_foot_cost_weights.", verbose_);

  // Cache EE kinematics per foot
  std::vector<std::shared_ptr<EndEffectorKinematics<scalar_t>>> eeKinematicsList(N_CONTACTS);
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    const absl::string_view footName = modelSettings_.contactNames[i];
    eeKinematicsList[i] = std::make_shared<PinocchioEndEffectorKinematicsCppAd>(
        *pinocchioInterfacePtr_, pinocchioMappingCppAd, std::vector<std::string>{std::string(footName)}, centroidalModelInfo_.stateDim,
        centroidalModelInfo_.inputDim, velocityUpdateCallback, std::string(footName), modelSettings_.modelFolderCppAd,
        modelSettings_.recompileLibrariesCppAd, modelSettings_.verboseCppAd);
  }

  // Define custom builders
  MpcProblemBuilder::CustomBuilders customBuilders;
  customBuilders.footTrackingCostBuilder =
      [this, &footTrackingCostWeights](size_t contactIndex, absl::string_view name) -> absl::StatusOr<std::unique_ptr<StateInputCost>> {
    return std::unique_ptr<StateInputCost>(new CentroidalMpcEndEffectorFootCost(*referenceManagerPtr_, footTrackingCostWeights,
                                                                                *pinocchioInterfacePtr_, *mpcRobotModelADPtr_, contactIndex,
                                                                                std::string(name), modelSettings_));
  };

  customBuilders.stanceFootConstraintBuilder =
      [this, &eeKinematicsList](size_t contactIndex, absl::string_view) -> absl::StatusOr<std::unique_ptr<StateInputConstraint>> {
    if (contactIndex >= eeKinematicsList.size()) {
      return absl::OutOfRangeError(absl::StrFormat("Contact index %zu exceeds kinematics list size.", contactIndex));
    }
    return getStanceFootConstraint(*eeKinematicsList[contactIndex], contactIndex);
  };

  customBuilders.normalVelocityConstraintBuilder =
      [this, &eeKinematicsList](size_t contactIndex, absl::string_view) -> absl::StatusOr<std::unique_ptr<StateInputConstraint>> {
    if (contactIndex >= eeKinematicsList.size()) {
      return absl::OutOfRangeError(absl::StrFormat("Contact index %zu exceeds kinematics list size.", contactIndex));
    }
    return getNormalVelocityConstraint(*eeKinematicsList[contactIndex], contactIndex);
  };

  customBuilders.jointMimicConstraintBuilder = [this](size_t mimicIndex,
                                                      absl::string_view) -> absl::StatusOr<std::unique_ptr<StateInputConstraint>> {
    return getJointMimicConstraint(mimicIndex);
  };

  customBuilders.taskSpaceKinematicsCostBuilder = [this, &pinocchioMappingCppAd,
                                                   &velocityUpdateCallback](OptimalControlProblem&) -> absl::Status {
    addTaskSpaceKinematicsCosts(pinocchioMappingCppAd, velocityUpdateCallback);
    return absl::OkStatus();
  };

  customBuilders.icpCostBuilder = [this]() -> absl::StatusOr<std::unique_ptr<StateInputCost>> {
    const vector2_t icpWeights = ICPCost::getWeights(taskFile_, "icp_cost_weights.", verbose_);
    return std::unique_ptr<StateInputCost>(new ICPCost(*referenceManagerPtr_, std::move(icpWeights), *pinocchioInterfacePtr_,
                                                       *mpcRobotModelADPtr_, "icp_Cost", modelSettings_));
  };

  // Load problem definition from YAML
  const absl::StatusOr<MpcProblemDefinition> problemDefStatus = loadMpcProblemDefinition(taskFile_, "problem_definition", verbose_);
  if (!problemDefStatus.ok()) {
    throw std::runtime_error(
        absl::StrCat("Failed to load problem_definition from ", taskFile_, ": ", problemDefStatus.status().ToString()));
  }

  const MpcProblemBuilder problemBuilder(problemDefStatus.value(), factory, modelSettings_, std::move(customBuilders));
  const absl::Status buildStatus = problemBuilder.buildProblem(*problemPtr_);
  if (!buildStatus.ok()) {
    throw std::runtime_error(absl::StrCat("Failed to build OptimalControlProblem: ", buildStatus.ToString()));
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
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputConstraint> CentroidalMpcInterface::getStanceFootConstraint(const EndEffectorKinematics<scalar_t>& eeKinematics,
                                                                                      size_t contactPointIndex) {
  auto eeZeroVelConConfig = [](scalar_t positionErrorGain, scalar_t orientationErrorGain) {
    EndEffectorKinematicsTwistConstraint::Config config;
    config.b.setZero(6);
    config.Ax.setZero(6, 6);
    config.Av.setIdentity(6, 6);
    if (!numerics::almost_eq(positionErrorGain, 0.0)) {
      config.Ax(2, 2) = positionErrorGain;
    }
    if (!numerics::almost_eq(orientationErrorGain, 0.0)) {
      config.Ax.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * orientationErrorGain;
    }

    return config;
  };

  return std::unique_ptr<StateInputConstraint>(
      new ZeroVelocityConstraintCppAd(*referenceManagerPtr_, eeKinematics, contactPointIndex,
                                      eeZeroVelConConfig(modelSettings_.footConstraintConfig.positionErrorGain_z,
                                                         modelSettings_.footConstraintConfig.orientationErrorGain)));
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
  absl::string_view prefix;
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
    std::cerr << "\n #### Joint Mimic Kinematic Constraint Config: ";
    std::cerr << "\n #### "
                 "============================================================="
                 "================\n";
  }
  loadData::loadPtreeValue(pt, parentJointName, std::string(absl::StrCat(prefix, "parentJointName")), verbose_);
  loadData::loadPtreeValue(pt, childJointName, std::string(absl::StrCat(prefix, "childJointName")), verbose_);
  loadData::loadPtreeValue(pt, multiplier, std::string(absl::StrCat(prefix, "multiplier")), verbose_);
  loadData::loadPtreeValue(pt, positionGain, std::string(absl::StrCat(prefix, "positionGain")), verbose_);
  if (verbose_) {
    std::cerr << " #### "
                 "============================================================="
                 "================\n";
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
    const absl::string_view costName = task_space_cost.first;
    std::string linkName;

    loadData::loadPtreeValue(task_space_costs_pt, linkName, std::string(absl::StrCat(costName, ".link_name")), verbose_);

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

    std::cout << "Initialized Task Space Kinematics Cost for link: " << linkName << std::endl;
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
