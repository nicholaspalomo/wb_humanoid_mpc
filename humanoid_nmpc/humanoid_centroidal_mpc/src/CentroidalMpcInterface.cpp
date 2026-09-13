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
#include <humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h>
#include <humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h>
#include <humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h>
#include <humanoid_common_mpc/pinocchio_model/createPinocchioModel.h>
#include "humanoid_common_mpc/common/StatusMacros.h"

#include "humanoid_centroidal_mpc/constraint/JointMimicKinematicConstraint.h"
#include "humanoid_centroidal_mpc/constraint/NormalVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/constraint/ZeroVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsAD.h"
#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsBasisInputsAD.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"

#include "humanoid_common_mpc/common/BasisInputsMappingDecorator.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"

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

  // Optionally wrap models with basis-vector decorator
  useContactBasisVectorInputs_ = false;
  try {
    loadData::loadCppDataType(taskFile, "useContactBasisVectorInputs", useContactBasisVectorInputs_);
  } catch (...) {
    useContactBasisVectorInputs_ = false;
  }

  if (useContactBasisVectorInputs_) {
    LOG(INFO) << "[CentroidalMpcInterface] Using basis-vector contact inputs";

    // Load wrench cone config to build basis matrices (reuses contactWrenchConeSoftConstraint params)
    boost::property_tree::ptree pt;
    loadData::readPropertyTree(taskFile, pt);
    const std::string prefix = "contacts.contactWrenchConeSoftConstraint.";
    ContactWrenchConeConstraint::Config coneConfig;
    loadData::loadPtreeValue(pt, coneConfig.frictionCoefficient, absl::StrCat(prefix, "frictionCoefficient"), verbose_);
    loadData::loadPtreeValue(pt, coneConfig.torsionalFrictionCoefficient, absl::StrCat(prefix, "torsionalFrictionCoefficient"), verbose_);
    loadData::loadPtreeValue(pt, coneConfig.minNormalForce, absl::StrCat(prefix, "minNormalForce"), verbose_);
    loadData::loadPtreeValue(pt, coneConfig.gripperForce, absl::StrCat(prefix, "gripperForce"), verbose_);
    loadData::loadPtreeValue(pt, coneConfig.numBasisVectors, absl::StrCat(prefix, "numBasisVectors"), verbose_);

    // Build per-contact basis matrices
    std::array<ContactWrenchConeBasisMatrix, N_CONTACTS> basisMatrices = {
        ContactWrenchConeBasisMatrix(coneConfig, ContactRectangle::loadContactRectangle(taskFile, modelSettings_, 0, verbose_)),
        ContactWrenchConeBasisMatrix(coneConfig, ContactRectangle::loadContactRectangle(taskFile, modelSettings_, 1, verbose_))};

    const size_t numBasisPerFoot = basisMatrices[0].numBasis();
    LOG(INFO) << "[CentroidalMpcInterface] Basis vectors per foot: " << numBasisPerFoot
              << " (total basis input dim: " << numBasisPerFoot * N_CONTACTS + modelSettings_.mpc_joint_dim << ")";

    // Create decorator models (take ownership of cloned inner models). The decorators carry a pinocchio
    // interface so that they can rotate the local-frame basis wrench into the world frame for a given state.
    basisDecoratorPtr_ = std::make_unique<BasisInputsModelDecorator<scalar_t>>(
        std::unique_ptr<MpcRobotModelBase<scalar_t>>(mpcRobotModelPtr_->clone()), basisMatrices, *pinocchioInterfacePtr_);
    basisDecoratorADPtr_ = std::make_unique<BasisInputsModelDecorator<ad_scalar_t>>(
        std::unique_ptr<MpcRobotModelBase<ad_scalar_t>>(mpcRobotModelADPtr_->clone()), basisMatrices, pinocchioInterfacePtr_->toCppAd());

    effectiveMpcRobotModelPtr_ = basisDecoratorPtr_.get();
    effectiveMpcRobotModelADPtr_ = basisDecoratorADPtr_.get();

    // Constant local-frame map u_wrench_local = M * u_basis, shared by the input-cost transform, the online parameter
    // updater and the CppAD end-effector kinematics mapping (which only needs the joint-velocity block).
    basisToWrenchMap_ = basisDecoratorPtr_->getLocalBasisToWrenchMap();

    // Regularization on λ: M has a non-trivial null space, so M^T R M alone leaves the λ Hessian singular.
    // LINT.IfChange(basis_regularization_yaml_path)
    constexpr scalar_t kDefaultBasisScalingRegularization = 1e-4;
    basisScalingRegularization_ = kDefaultBasisScalingRegularization;
    loadData::loadPtreeValue(pt, basisScalingRegularization_, "contacts.basisScalingRegularization", verbose_);
    // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:basis_regularization_config)
    if (basisScalingRegularization_ < 0.0) {
      throw std::invalid_argument("[CentroidalMpcInterface] contacts.basisScalingRegularization must be non-negative");
    }
  } else {
    effectiveMpcRobotModelPtr_ = mpcRobotModelPtr_.get();
    effectiveMpcRobotModelADPtr_ = mpcRobotModelADPtr_.get();
  }

  // Swing trajectory planner
  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(
      new SwingTrajectoryPlanner(loadSwingTrajectorySettings(taskFile, "swing_trajectory_config", verbose_), N_CONTACTS));

  if (modelSettings_.useContactPlanning) {
    // Online mixed-integer contact planning replaces the periodic gait schedule. The gait schedule is still loaded: it is
    // used until the first plan arrives and whenever the planner has no valid plan.
    ContactPlanningConfig contactPlanningConfig = loadContactPlanningConfig(taskFile, "contact_planning.", verbose_);
    if (contactPlanningConfig.horizon() < mpcSettings_.timeHorizon_) {
      LOG(WARNING) << "[CentroidalMpcInterface] contact_planning horizon (" << contactPlanningConfig.horizon()
                   << " s) is shorter than the MPC horizon (" << mpcSettings_.timeHorizon_
                   << " s); the schedule beyond the planned horizon defaults to double support.";
    }
    auto planningReferenceManager = std::make_shared<ContactPlanningReferenceManager>(
        GaitSchedule::loadGaitSchedule(referenceFile, modelSettings_, verbose_), std::move(swingTrajectoryPlanner), *pinocchioInterfacePtr_,
        *effectiveMpcRobotModelPtr_, contactPlanningConfig);
    contactPlannerModulePtr_ = std::make_shared<ContactPlannerModule>(planningReferenceManager, contactPlanningConfig);
    referenceManagerPtr_ = planningReferenceManager;
    LOG(INFO) << "[CentroidalMpcInterface] Using mixed-integer contact planning (" << contactPlanningConfig.numNodes << " nodes x "
              << contactPlanningConfig.dt << " s, " << (contactPlanningConfig.runInBackgroundThread ? "background thread" : "synchronous")
              << ").";
  } else {
    referenceManagerPtr_ = std::make_shared<SwitchedModelReferenceManager>(
        GaitSchedule::loadGaitSchedule(referenceFile, modelSettings_, verbose_), std::move(swingTrajectoryPlanner), *pinocchioInterfacePtr_,
        *effectiveMpcRobotModelPtr_);
  }
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
      HumanoidCostConstraintFactory(taskFile_, referenceFile_, *referenceManagerPtr_, *pinocchioInterfacePtr_, *effectiveMpcRobotModelPtr_,
                                    *effectiveMpcRobotModelADPtr_, modelSettings_, verbose_);

  // Optimal control problem
  problemPtr_.reset(new OptimalControlProblem);

  // Dynamics
  std::unique_ptr<SystemDynamicsBase> dynamicsPtr;
  const std::string modelName = "dynamics";
  if (useContactBasisVectorInputs_) {
    // R costs are loaded in wrench dims and transformed into basis space: R_basis = M^T * R_wrench * M + reg
    factory.setBasisToWrenchMap(*basisToWrenchMap_, getWrenchInputDim(), getNumBasisInputs(), basisScalingRegularization_);

    // The dynamics rotate each local-frame basis wrench B_i * λ_i into the world frame inside the CppAD tape.
    dynamicsPtr.reset(new CentroidalDynamicsBasisInputsAD(*pinocchioInterfacePtr_, centroidalModelInfo_, modelName, modelSettings_,
                                                          basisDecoratorPtr_->getBasisMatrices()));
  } else {
    dynamicsPtr.reset(new CentroidalDynamicsAD(*pinocchioInterfacePtr_, centroidalModelInfo_, modelName, modelSettings_));
  }
  problemPtr_->dynamicsPtr = std::move(dynamicsPtr);

  // Load configured MPC formulation tasks
  ASSIGN_OR_RETURN(const MpcFormulationTasks formulationTasks, loadMpcFormulationTasks(taskFile_, verbose_));

  // Cost terms
  if (formulationTasks.hasCost(MpcCostType::StateInputQuadraticCost)) {
    problemPtr_->costPtr->add("stateInputQuadraticCost", factory.getStateInputQuadraticCost());
  }
  if (formulationTasks.hasCost(MpcCostType::StateQuadraticCost)) {
    problemPtr_->costPtr->add("stateQuadraticCost", factory.getStateQuadraticCost());

    if (modelSettings_.useComAndAcomTracking) {
      problemPtr_->stateCostPtr->add("comAndAcomTrackingCost", factory.getComAndAcomTrackingCost(centroidalModelInfo_));
    }
  }
  if (formulationTasks.hasCost(MpcCostType::InputQuadraticCost)) {
    problemPtr_->costPtr->add("inputQuadraticCost", factory.getInputQuadraticCost());
  }
  // Terminal cost: either the DCM viability cost (useDcmTerminalCost, or dcm_terminal_cost in the cost list) or the
  // quadratic Q_final cost. With the DCM cost enabled, Q_final / terminal_cost are ignored.
  const bool useDcmTerminalCost = modelSettings_.useDcmTerminalCost || formulationTasks.hasCost(MpcCostType::DcmTerminalCost);
  if (useDcmTerminalCost) {
    const DcmTerminalCost::Config dcmConfig = DcmTerminalCost::loadConfig(taskFile_, "dcm_terminal_cost.", verbose_);
    problemPtr_->finalCostPtr->add("dcmTerminalCost",
                                   std::make_unique<DcmTerminalCost>(*referenceManagerPtr_, dcmConfig, *pinocchioInterfacePtr_,
                                                                     *effectiveMpcRobotModelADPtr_, "dcmTerminalCost", modelSettings_));
    if (formulationTasks.hasCost(MpcCostType::TerminalCost)) {
      LOG(INFO) << "[CentroidalMpcInterface] useDcmTerminalCost is enabled: the quadratic terminal_cost (Q_final) is ignored.";
    }
  } else if (formulationTasks.hasCost(MpcCostType::TerminalCost)) {
    problemPtr_->finalCostPtr->add("terminalCost", factory.getTerminalCost());
  }

  const CentroidalModelInfoCppAd infoCppAd = centroidalModelInfo_.toCppAd();
  const CentroidalModelPinocchioMappingCppAd pinocchioMappingCppAdBase(infoCppAd);

  // When using basis-vector inputs, wrap the pinocchio mapping so that CppAD-compiled
  // EE kinematics (zero-velocity, normal-velocity constraints, foot tracking cost)
  // first convert u_basis → u_wrench = M * u_basis before extracting joint velocities at
  // wrench-space offsets. Only the joint-velocity block of M matters here (the kinematics
  // never read the contact block), so the local-frame map is sufficient.
  std::unique_ptr<PinocchioStateInputMapping<ad_scalar_t>> effectiveMappingPtr;
  if (useContactBasisVectorInputs_) {
    effectiveMappingPtr = std::make_unique<BasisInputsMappingDecorator<ad_scalar_t>>(
        std::unique_ptr<PinocchioStateInputMapping<ad_scalar_t>>(pinocchioMappingCppAdBase.clone()), *basisToWrenchMap_,
        getWrenchInputDim(), modelSettings_.mpc_joint_dim);
  } else {
    effectiveMappingPtr = std::unique_ptr<PinocchioStateInputMapping<ad_scalar_t>>(pinocchioMappingCppAdBase.clone());
  }
  const PinocchioStateInputMapping<ad_scalar_t>& pinocchioMappingCppAd = *effectiveMappingPtr;

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
                                                                *effectiveMpcRobotModelADPtr_, "icp_Cost", modelSettings_)));
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
  bool footCostActiveInStance = false;
  if (formulationTasks.hasCost(MpcCostType::TaskSpaceFootCost)) {
    footTrackingCostWeights = EndEffectorKinematicsWeights::getWeights(taskFile_, "task_space_foot_cost_weights.", verbose_);
    try {
      boost::property_tree::ptree pt;
      loadData::readPropertyTree(taskFile_, pt);
      loadData::loadPtreeValue(pt, footCostActiveInStance, "task_space_foot_cost_weights.activeInStance", verbose_);
    } catch (...) {
      footCostActiveInStance = false;
    }
  }

  for (size_t i = 0; i < N_CONTACTS; i++) {
    const std::string& footName = modelSettings_.contactNames[i];

    std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKinematicsPtr;
    bool needsEeKinematics = formulationTasks.hasHardConstraint(MpcHardConstraintType::ZeroVelocity) ||
                             formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ZeroVelocity) ||
                             formulationTasks.hasHardConstraint(MpcHardConstraintType::NormalVelocity);
    if (needsEeKinematics) {
      const size_t effectiveInputDim = effectiveMpcRobotModelPtr_->getInputDim();
      eeKinematicsPtr.reset(new PinocchioEndEffectorKinematicsCppAd(*pinocchioInterfacePtr_, pinocchioMappingCppAd, {footName},
                                                                    centroidalModelInfo_.stateDim, effectiveInputDim,
                                                                    velocityUpdateCallback, footName, modelSettings_.modelFolderCppAd,
                                                                    modelSettings_.recompileLibrariesCppAd, modelSettings_.verboseCppAd));
    }

    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactWrenchCone)) {
      if (useContactBasisVectorInputs_) {
        // In basis-vector mode the wrench cone is enforced structurally:
        // all basis vectors lie inside the cone, so λ ≥ 0 ⟹ W ∈ cone.
        // The ContactWrenchConeConstraint is wrench-space specific and
        // cannot operate on basis-vector inputs.
        LOG(INFO) << "[CentroidalMPC] Skipping contact_wrench_cone soft constraint for " << footName
                  << " (enforced structurally via basis-vector inputs).";

        // Add λ ≥ 0 non-negativity constraint as a barrier penalty.
        // This is the structural enforcement: all basis scalings must be non-negative
        // for the wrench to remain inside the friction/wrench cone.
        // Load barrier parameters from YAML; fall back to defaults if absent.
        constexpr scalar_t kDefaultLambdaBarrierMu = 1e-2;
        constexpr scalar_t kDefaultLambdaBarrierDelta = 1e-3;
        // LINT.IfChange(basis_barrier_yaml_path)
        const std::string barrierPrefix = "contacts.basisNonNegativityBarrier.";
        // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:basis_barrier_config)
        scalar_t lambdaBarrierMu = kDefaultLambdaBarrierMu;
        scalar_t lambdaBarrierDelta = kDefaultLambdaBarrierDelta;
        boost::property_tree::ptree barrierPt;
        loadData::readPropertyTree(taskFile_, barrierPt);
        loadData::loadPtreeValue(barrierPt, lambdaBarrierMu, absl::StrCat(barrierPrefix, "mu"), verbose_);
        loadData::loadPtreeValue(barrierPt, lambdaBarrierDelta, absl::StrCat(barrierPrefix, "delta"), verbose_);
        const PieceWisePolynomialBarrierPenalty::Config lambdaBarrierConfig(lambdaBarrierMu, lambdaBarrierDelta);
        const size_t lambdaStartIdx = basisDecoratorPtr_->getContactWrenchStartIndices(i);
        const size_t numBasis = basisDecoratorPtr_->getNumBasisPerFoot();

        problemPtr_->costPtr->add(
            absl::StrCat(footName, "_basisNonNegativity"),
            std::make_unique<BasisScalingNonNegativityConstraint>(*referenceManagerPtr_, i, lambdaStartIdx, numBasis, lambdaBarrierConfig));

        LOG(INFO) << "[CentroidalMPC] Added λ ≥ 0 non-negativity barrier for " << footName << " (" << numBasis
                  << " basis vectors, start idx " << lambdaStartIdx << ").";
      } else {
        problemPtr_->softConstraintPtr->add(absl::StrCat(footName, "_contactWrenchCone"), factory.getContactWrenchConeConstraint(i));
      }
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
      problemPtr_->costPtr->add(footTrackingCostName,
                                std::unique_ptr<StateInputCost>(new CentroidalMpcEndEffectorFootCost(
                                    *referenceManagerPtr_, footTrackingCostWeights, *pinocchioInterfacePtr_, *effectiveMpcRobotModelADPtr_,
                                    i, footTrackingCostName, modelSettings_, footCostActiveInStance)));
    }
    if (formulationTasks.hasCost(MpcCostType::ExternalTorqueCost)) {
      problemPtr_->costPtr->add(absl::StrCat(footName, "_ExternalTorqueQuadraticCost"), factory.getExternalTorqueQuadraticCost(i));
    }
  }

  // Pre-computation
  problemPtr_->preComputationPtr.reset(
      new HumanoidPreComputation(*pinocchioInterfacePtr_, *referenceManagerPtr_->getSwingTrajectoryPlanner(), *effectiveMpcRobotModelPtr_));

  // Rollout
  rolloutPtr_.reset(new TimeTriggeredRollout(*problemPtr_->dynamicsPtr, rolloutSettings_));

  // Initialization
  constexpr bool extendNormalizedMomentum = true;
  initializerPtr_.reset(new CentroidalWeightCompInitializer(centroidalModelInfo_, *referenceManagerPtr_, *effectiveMpcRobotModelPtr_,
                                                            extendNormalizedMomentum));

  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputConstraint> CentroidalMpcInterface::getStanceFootConstraint(const EndEffectorKinematics<scalar_t>& eeKinematics,
                                                                                      size_t contactPointIndex) {
  const auto& footConfig = modelSettings_.footConstraintConfig;
  const size_t numConstraints = footConfig.constrainOrientation ? 6 : 3;

  auto eeZeroVelConConfig = [numConstraints](const ModelSettings::FootConstraintConfig& footConfig) {
    EndEffectorKinematicsTwistConstraint::Config config;
    config.b.setZero(6);
    config.Ax.setZero(6, 6);
    config.Av.setZero(6, 6);

    // Position error gain: only z-axis (foot height tracking during stance)
    if (!numerics::almost_eq(footConfig.positionErrorGain_z, 0.0)) {
      config.Ax(2, 2) = footConfig.positionErrorGain_z;
    }
    // Orientation error gain: all 3 rotation axes (only effective when constrainOrientation is true)
    if (!numerics::almost_eq(footConfig.orientationErrorGain, 0.0)) {
      config.Ax.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * footConfig.orientationErrorGain;
    }

    // Linear velocity gains: xy and z separately
    config.Av(0, 0) = footConfig.linearVelocityErrorGain_xy;
    config.Av(1, 1) = footConfig.linearVelocityErrorGain_xy;
    config.Av(2, 2) = footConfig.linearVelocityErrorGain_z;

    // Angular velocity gain: all 3 rotation axes (only effective when constrainOrientation is true)
    config.Av(3, 3) = footConfig.angularVelocityErrorGain;
    config.Av(4, 4) = footConfig.angularVelocityErrorGain;
    config.Av(5, 5) = footConfig.angularVelocityErrorGain;

    return config;
  };

  return std::unique_ptr<StateInputConstraint>(new ZeroVelocityConstraintCppAd(*referenceManagerPtr_, eeKinematics, contactPointIndex,
                                                                               numConstraints, eeZeroVelConConfig(footConfig)));
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

  JointMimicKinematicConstraint::Config config(*effectiveMpcRobotModelPtr_, parentJointName, childJointName, multiplier, positionGain);

  return std::unique_ptr<StateInputConstraint>(new JointMimicKinematicConstraint(*effectiveMpcRobotModelPtr_, config));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void CentroidalMpcInterface::addTaskSpaceKinematicsCosts(
    const PinocchioStateInputMapping<ad_scalar_t>& pinocchioMappingCppAd,
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
                                                                  centroidalModelInfo_.stateDim, effectiveMpcRobotModelPtr_->getInputDim(),
                                                                  velocityUpdateCallback, linkName, modelSettings_.modelFolderCppAd,
                                                                  modelSettings_.recompileLibrariesCppAd, modelSettings_.verboseCppAd));

    EndEffectorKinematicsWeights weights =
        EndEffectorKinematicsWeights::getWeights(taskFile_, absl::StrCat("task_space_costs.", costName, ".weights."), verbose_);

    std::unique_ptr<StateInputCost> cost = std::make_unique<EndEffectorKinematicsQuadraticCost>(
        weights, *pinocchioInterfacePtr_, *eeKinematicsPtr, *effectiveMpcRobotModelADPtr_, linkName, modelSettings_);

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
