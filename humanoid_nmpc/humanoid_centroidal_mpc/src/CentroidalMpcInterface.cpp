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

#include "absl/log/check.h"
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
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>

#include <humanoid_common_mpc/HumanoidCostConstraintFactory.h>
#include <humanoid_common_mpc/HumanoidPreComputation.h>
#include <humanoid_common_mpc/common/MpcFormulationConfig.h>
#include <humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h>
#include <humanoid_common_mpc/constraint/ContactComplementarityConstraint.h>
#include <humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h>
#include <humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h>
#include <humanoid_common_mpc/constraint/GroundPenetrationConstraint.h>
#include <humanoid_common_mpc/contact/FootprintCornerHeights.h>
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
#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"

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

    // A conic combination is homogeneous, so the basis can represent neither the minimum normal force nor a gripper
    // adhesion force: both are affine offsets of the cone and lambda = 0 always yields the zero wrench.
    if (coneConfig.minNormalForce > 0.0 || coneConfig.gripperForce > 0.0) {
      LOG(WARNING) << "[CentroidalMpcInterface] contacts.contactWrenchConeSoftConstraint.minNormalForce (" << coneConfig.minNormalForce
                   << " N) and gripperForce (" << coneConfig.gripperForce
                   << " N) are NOT enforced with useContactBasisVectorInputs: true. The friction, CoP and torsional "
                      "limits are enforced structurally by the basis; these two affine offsets cannot be.";
    }

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
    // clang-format off
    // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:basis_regularization_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:basis_regularization_config)
    // clang-format on
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
    const std::string contactPlanningFile = resolveContactPlanningConfigFile(taskFile);
    LOG(INFO) << "[CentroidalMpcInterface] Loading contact planning config from " << contactPlanningFile;
    ContactPlanningConfig contactPlanningConfig =
        loadContactPlanningConfig(contactPlanningFile, "contact_planning.", verbose_, /*validate=*/false);
    {
      // Parameters left at 0 in the task file are derived from the robot model and from the ground parameters of the
      // wrench cone, so that the planner never assumes more friction, torque or footprint than the whole-body
      // constraints allow. The initial state of the task file is the nominal posture for the LIP height.
      vector_t nominalState = vector_t::Zero(centroidalModelInfo_.stateDim);
      loadData::loadEigenMatrix(taskFile, "initialState", nominalState);
      boost::property_tree::ptree conePt;
      loadData::readPropertyTree(taskFile, conePt);
      const std::string conePrefix = "contacts.contactWrenchConeSoftConstraint.";
      ContactPlanningGroundParameters ground;
      loadData::loadPtreeValue(conePt, ground.frictionCoefficient, conePrefix + "frictionCoefficient", verbose_);
      loadData::loadPtreeValue(conePt, ground.torsionalFrictionCoefficient, conePrefix + "torsionalFrictionCoefficient", verbose_);
      try {
        const ContactRectangle footprint = ContactRectangle::loadContactRectangle(taskFile, modelSettings_, 0, verbose_);
        ground.footprintHalfLengthX = 0.5 * (footprint.getBounds().x_max - footprint.getBounds().x_min);
        ground.footprintHalfWidthY = 0.5 * (footprint.getBounds().y_max - footprint.getBounds().y_min);
      } catch (const std::exception& e) {
        LOG(WARNING) << "[CentroidalMpcInterface] no footprint for the contact planner's ZMP box: " << e.what();
      }
      contactPlanningModelParameters_ = deriveContactPlanningModelParameters(
          *pinocchioInterfacePtr_, *effectiveMpcRobotModelPtr_, nominalState, modelSettings_.contactParentJointNames, ground,
          contactPlanningConfig.shared.gravity, contactPlanningConfig.stepWidth.nominalStepWidth);
      contactPlanningModelParameters_->applyTo(contactPlanningConfig);
      LOG(INFO) << "[CentroidalMpcInterface] contact planner model parameters: " << contactPlanningModelParameters_->summary();
    }
    contactPlanningConfig.validate();
    if (contactPlanningConfig.horizon() < mpcSettings_.timeHorizon_) {
      LOG(WARNING) << "[CentroidalMpcInterface] contact_planning horizon (" << contactPlanningConfig.horizon()
                   << " s) is shorter than the MPC horizon (" << mpcSettings_.timeHorizon_
                   << " s); the schedule beyond the planned horizon defaults to double support.";
    }
    auto planningReferenceManager = std::make_shared<ContactPlanningReferenceManager>(
        GaitSchedule::loadGaitSchedule(referenceFile, modelSettings_, verbose_), std::move(swingTrajectoryPlanner), *pinocchioInterfacePtr_,
        *effectiveMpcRobotModelPtr_, contactPlanningConfig);
    if (contactPlanningConfig.usesHeadingModel()) {
      try {
        planningReferenceManager->setAngularCenterOfMass(AngularCenterOfMass::createForRobot(modelSettings_.robotName));
        LOG(INFO) << "[CentroidalMpcInterface] contact planner heading model: angular centre of mass of '" << modelSettings_.robotName
                  << "'.";
      } catch (const std::exception& e) {
        LOG(WARNING) << "[CentroidalMpcInterface] contact planner heading model: no ACoM network for '" << modelSettings_.robotName << "' ("
                     << e.what() << "); the base yaw is the heading.";
      }
    }
    contactPlannerModulePtr_ = std::make_shared<ContactPlannerModule>(planningReferenceManager, contactPlanningConfig);
    contactPlannerModulePtr_->setModelParameters(*contactPlanningModelParameters_);
    referenceManagerPtr_ = planningReferenceManager;
    LOG(INFO) << "[CentroidalMpcInterface] Using mixed-integer contact planning (" << contactPlanningConfig.planner.numNodes << " nodes x "
              << contactPlanningConfig.planner.dt << " s, "
              << (contactPlanningConfig.planner.runInBackgroundThread ? "background thread" : "synchronous") << ").";
  } else {
    referenceManagerPtr_ = std::make_shared<SwitchedModelReferenceManager>(
        GaitSchedule::loadGaitSchedule(referenceFile, modelSettings_, verbose_), std::move(swingTrajectoryPlanner), *pinocchioInterfacePtr_,
        *effectiveMpcRobotModelPtr_);
  }
  // A legs-only robot omits model_settings.armJointNames and has no arm to swing.
  referenceManagerPtr_->setArmSwingReferenceActive(modelSettings_.hasArmSwingJoints);

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

absl::Status CentroidalMpcInterface::setupLocomotionHeuristics() {
  // The reference-shaping layer of Bledt's Regularized Predictive Control heuristics
  // (humanoid_nmpc/docs/locomotion_heuristics/README.md).
  //
  // Here rather than in the constructor because every failure it can have is a configuration error the operator has to
  // read - an unknown name, a name in the wrong list, a coefficient outside its range, a foothold heuristic under an
  // online contact planner - and this is the function that returns a Status. `initialState_` is already loaded by the
  // time this runs, which matters: the model constants the heuristics need (the hip positions and the nominal CoM
  // height) are properties of the nominal standing posture.
  //
  // Every robot ships with all three lists empty, so on all of them this builds an empty layer and every reference
  // downstream is bit for bit what it was before this subsystem existed.
  ASSIGN_OR_RETURN(const LocomotionHeuristicConfig heuristicConfig, loadLocomotionHeuristicConfig(taskFile_, verbose_));
  ASSIGN_OR_RETURN(const LocomotionHeuristicModelParameters heuristicModel,
                   deriveLocomotionHeuristicModelParameters(*pinocchioInterfacePtr_, *effectiveMpcRobotModelPtr_, initialState_));
  ASSIGN_OR_RETURN(std::unique_ptr<LocomotionHeuristicLayer> heuristicLayer,
                   LocomotionHeuristicLayer::Create(heuristicConfig, heuristicModel, modelSettings_.useContactPlanning,
                                                    useContactBasisVectorInputs_, verbose_));
  locomotionHeuristicLayerPtr_ = std::shared_ptr<LocomotionHeuristicLayer>(std::move(heuristicLayer));
  referenceManagerPtr_->setLocomotionHeuristicLayer(locomotionHeuristicLayerPtr_);
  if (!locomotionHeuristicLayerPtr_->empty()) {
    LOG(INFO) << "[CentroidalMpcInterface] locomotion heuristics active:\n" << locomotionHeuristicLayerPtr_->summary();
  }
  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status CentroidalMpcInterface::setupOptimalControlProblem() {
  RETURN_IF_ERROR(setupLocomotionHeuristics());

  // Loaded before the factory is built: the factory needs to know whether the contact cones it creates may gate
  // themselves on the mode schedule, and that follows the hard `zero_wrench` constraint.
  ASSIGN_OR_RETURN(const MpcFormulationTasks formulationTasks, loadMpcFormulationTasks(taskFile_, verbose_));
  const bool scheduleGatedContactConstraints = contactConstraintsAreScheduleGated(formulationTasks);

  // loadMpcFormulationTasks() has already insisted that SOME cone is listed once `zero_wrench` is gone. In
  // basis-vector mode that is not enough: it has to be `contact_wrench_cone` specifically, because that is the branch
  // below which builds the non-negativity barrier on the basis scalings, and lambda >= 0 is the whole of the cone
  // here. `friction_force_cone` is not a substitute - it bounds mu*Fz - |F_xy| on the assembled wrench and says
  // nothing about the individual scalings, so the centre-of-pressure and torsional limits would go unenforced and a
  // negative scaling (an adhesive, outside-the-cone generator) would still be free.
  //
  // While `zero_wrench` is listed a missing `contact_wrench_cone` is harmless, because the swing foot's scalings are
  // pinned to zero by that equality anyway, and the stance foot's are bounded by the gated barrier.
  if (useContactBasisVectorInputs_ && !scheduleGatedContactConstraints &&
      !formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactWrenchCone)) {
    return absl::InvalidArgumentError(
        "[CentroidalMpcInterface] with useContactBasisVectorInputs: true and the hard 'zero_wrench' constraint removed, "
        "'contact_wrench_cone' must be listed in soft_constraints: it is what builds the non-negativity barrier on the basis "
        "scalings, which is then the only bound keeping each contact wrench inside its friction cone. 'friction_force_cone' "
        "does not stand in for it - it bounds the assembled wrench, not the scalings "
        "(humanoid_nmpc/docs/contact_implicit_mpc/README.md).");
  }

  HumanoidCostConstraintFactory factory =
      HumanoidCostConstraintFactory(taskFile_, referenceFile_, *referenceManagerPtr_, *pinocchioInterfacePtr_, *effectiveMpcRobotModelPtr_,
                                    *effectiveMpcRobotModelADPtr_, modelSettings_, verbose_, scheduleGatedContactConstraints);

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

    // The kinematics of the CONTACT FRAME, i.e. the centre of the sole. `ground_penetration` is deliberately absent
    // from this list: it is evaluated at the footprint's corner frames instead and builds its own kinematics below, so
    // listing it here would generate a whole extra CppAD model per foot that nothing reads.
    std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKinematicsPtr;
    const bool needsEeKinematics = formulationTasks.hasHardConstraint(MpcHardConstraintType::ZeroVelocity) ||
                                   formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ZeroVelocity) ||
                                   formulationTasks.hasHardConstraint(MpcHardConstraintType::NormalVelocity) ||
                                   formulationTasks.hasSoftConstraint(MpcSoftConstraintType::NormalVelocity) ||
                                   formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ForceWeightedSlip);
    // Neither contact_complementarity nor ground_penetration appears here: both read the footprint CORNERS through a
    // FootprintCornerHeights of their own, not this contact-frame kinematics.
    if (needsEeKinematics) {
      const size_t effectiveInputDim = effectiveMpcRobotModelPtr_->getInputDim();
      eeKinematicsPtr.reset(new PinocchioEndEffectorKinematicsCppAd(*pinocchioInterfacePtr_, pinocchioMappingCppAd, {footName},
                                                                    centroidalModelInfo_.stateDim, effectiveInputDim,
                                                                    velocityUpdateCallback, footName, modelSettings_.modelFolderCppAd,
                                                                    modelSettings_.recompileLibrariesCppAd, modelSettings_.verboseCppAd));
    }

    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactWrenchCone)) {
      if (useContactBasisVectorInputs_) {
        // In basis-vector mode the friction, CoP and torsional limits are enforced structurally: every generator lies
        // inside the cone (ContactWrenchConeBasisMatrix verifies this against the constraint's own rows at
        // construction) and the cone is convex, so λ ≥ 0 ⟹ W ∈ cone. The ContactWrenchConeConstraint is wrench-space
        // specific and cannot operate on basis-vector inputs. The minimum normal force and the gripper force are the
        // exception and are not enforced; a warning is logged where the basis is built.
        LOG(INFO) << "[CentroidalMPC] Skipping contact_wrench_cone soft constraint for " << footName
                  << " (friction, CoP and torsional limits enforced structurally via basis-vector inputs).";

        // Add λ ≥ 0 non-negativity constraint as a barrier penalty.
        // This is the structural enforcement: all basis scalings must be non-negative
        // for the wrench to remain inside the friction/wrench cone.
        // Load barrier parameters from YAML; fall back to defaults if absent.
        constexpr scalar_t kDefaultLambdaBarrierMu = 1e-2;
        constexpr scalar_t kDefaultLambdaBarrierDelta = 1e-3;
        // LINT.IfChange(basis_barrier_yaml_path)
        const std::string barrierPrefix = "contacts.basisNonNegativityBarrier.";
        // clang-format off
        // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:basis_barrier_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:basis_barrier_config)
        // clang-format on
        scalar_t lambdaBarrierMu = kDefaultLambdaBarrierMu;
        scalar_t lambdaBarrierDelta = kDefaultLambdaBarrierDelta;
        boost::property_tree::ptree barrierPt;
        loadData::readPropertyTree(taskFile_, barrierPt);
        loadData::loadPtreeValue(barrierPt, lambdaBarrierMu, absl::StrCat(barrierPrefix, "mu"), verbose_);
        loadData::loadPtreeValue(barrierPt, lambdaBarrierDelta, absl::StrCat(barrierPrefix, "delta"), verbose_);
        const PieceWisePolynomialBarrierPenalty::Config lambdaBarrierConfig(lambdaBarrierMu, lambdaBarrierDelta);

        // WHAT `mu` IS NOW BEING ASKED TO DO. Gated, this term is a redundant bound: the swing foot's scalings are
        // pinned to zero by `zero_wrench` and the stance foot's are pulled positive by R's weight-compensating
        // nominal, so 0.01 was tuned as a regulariser. Un-gated it becomes the ONLY thing holding every contact
        // wrench inside its cone, at every node, which is the job `contactWrenchConeSoftConstraint.mu` does in the
        // wrench parameterization - and there it is 0.2. A scaling pair (+a, -a) costs only `mu * a^2` and leaves the
        // load indicator f_n = sum(lambda) at zero, so both contact-implicit products stay blind to it.
        //
        // Raising it is a closed-loop tuning decision and not one this loader may take on the operator's behalf, so
        // the mismatch is reported rather than patched. See humanoid_nmpc/docs/contact_implicit_mpc/README.md.
        if (!scheduleGatedContactConstraints) {
          scalar_t wrenchConeMu = 0.0;
          loadData::loadPtreeValue(barrierPt, wrenchConeMu, "contacts.contactWrenchConeSoftConstraint.mu", false);
          if (lambdaBarrierMu < wrenchConeMu) {
            LOG(WARNING) << "[CentroidalMpcInterface] " << footName << ": contacts.basisNonNegativityBarrier.mu = " << lambdaBarrierMu
                         << " is the ONLY bound on this contact's wrench once the schedule gate is off, and it is softer than the "
                         << "contacts.contactWrenchConeSoftConstraint.mu = " << wrenchConeMu
                         << " that does the same job in the wrench parameterization. Tune it against a foot in flight before "
                         << "trusting the contact-implicit formulation on hardware.";
          }
        }

        const size_t lambdaStartIdx = basisDecoratorPtr_->getContactWrenchStartIndices(i);
        const size_t numBasis = basisDecoratorPtr_->getNumBasisPerFoot();

        problemPtr_->costPtr->add(absl::StrCat(footName, "_basisNonNegativity"), std::make_unique<BasisScalingNonNegativityConstraint>(
                                                                                     *referenceManagerPtr_, i, lambdaStartIdx, numBasis,
                                                                                     lambdaBarrierConfig, scheduleGatedContactConstraints));

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

    // The relaxed complementarity conditions of rigid contact: with all three listed (and zero_wrench / zero_velocity
    // dropped, which loadMpcFormulationTasks enforces) the mode schedule no longer gates any contact constraint and
    // the solver decides where each foot carries load. See humanoid_nmpc/docs/contact_implicit_mpc/README.md.
    const ModelSettings::ContactImplicitConfig& contactImplicit = modelSettings_.contactImplicitConfig;
    // The force both residuals are measured in is the robot's own weight, read from the model rather than configured,
    // so it cannot fall out of step with the URDF. It is what makes the two weights mean the same thing on a 40 kg
    // robot and a 160 kg one.
    constexpr scalar_t kStandardGravity = 9.81;  // [m/s^2]
    const scalar_t forceReference = centroidalModelInfo_.robotMass * kStandardGravity;

    // Where the foot is, as far as contact is concerned: the CORNERS of the footprint, not the centre of the sole.
    // createPinocchioModel() already adds a frame at each point of the contact polygon. Both the penetration hinge and
    // the complementarity product are built on this one object, so they cannot end up measuring different heights -
    // they used to, and a foot rocked onto its heel then read a positive height while carrying the whole robot.
    // Constraining the sole centre alone would also leave the toe and the heel free to go through the floor, because
    // this formulation deliberately lets the foot rock (ForceWeightedSlipConstraint leaves the rocking rates free),
    // and on this robot the corners are 0.12 m fore and aft of the centre.
    std::unique_ptr<FootprintCornerHeights> cornerHeightsPtr;
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactComplementarity) ||
        formulationTasks.hasSoftConstraint(MpcSoftConstraintType::GroundPenetration)) {
      const ContactRectangle footprint = ContactRectangle::loadContactRectangle(taskFile_, modelSettings_, static_cast<int>(i), false);
      std::vector<std::string> cornerFrames;
      cornerFrames.reserve(footprint.getNumberOfContactPoints());
      for (size_t corner = 0; corner < footprint.getNumberOfContactPoints(); ++corner) {
        cornerFrames.push_back(footprint.getPolygonPointFrameName(static_cast<int>(corner)));
      }
      cornerHeightsPtr =
          std::make_unique<FootprintCornerHeights>(*pinocchioInterfacePtr_, *effectiveMpcRobotModelADPtr_, std::move(cornerFrames),
                                                   absl::StrCat(footName, "_footprintCorners"), modelSettings_);
    }

    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactComplementarity)) {
      std::unique_ptr<StateInputConstraint> complementarity = std::make_unique<ContactComplementarityConstraint>(
          *cornerHeightsPtr, *effectiveMpcRobotModelPtr_, i, modelSettings_.terrainHeight, forceReference, contactImplicit.heightReference,
          contactImplicit.gapSmoothing);
      auto penalty = std::make_unique<QuadraticPenalty>(contactImplicit.complementarityWeight);
      problemPtr_->softConstraintPtr->add(absl::StrCat(footName, "_contactComplementarity"),
                                          std::make_unique<StateInputSoftConstraint>(std::move(complementarity), std::move(penalty)));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ForceWeightedSlip)) {
      // Not `&& eeKinematicsPtr`, which is how the neighbouring terms guard themselves: this one is load bearing.
      // loadMpcFormulationTasks() refuses `contact_complementarity` without `force_weighted_slip`, because nothing
      // else holds a LOADED foot still once the schedule-gated zero-velocity constraint is gone. Silently skipping it
      // because `needsEeKinematics` above had drifted would defeat that guarantee and leave a foot carrying full body
      // weight free to slide.
      CHECK(eeKinematicsPtr != nullptr) << "[CentroidalMpcInterface] 'force_weighted_slip' needs the contact frame's kinematics; "
                                           "needsEeKinematics must list it.";
      std::unique_ptr<StateInputConstraint> slip =
          std::make_unique<ForceWeightedSlipConstraint>(*eeKinematicsPtr, *effectiveMpcRobotModelPtr_, i, forceReference,
                                                        contactImplicit.velocityReference, contactImplicit.angularVelocityReference);
      auto penalty = std::make_unique<QuadraticPenalty>(contactImplicit.slipWeight);
      problemPtr_->softConstraintPtr->add(absl::StrCat(footName, "_forceWeightedSlip"),
                                          std::make_unique<StateInputSoftConstraint>(std::move(slip), std::move(penalty)));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::GroundPenetration)) {
      std::unique_ptr<StateConstraint> penetration =
          std::make_unique<GroundPenetrationConstraint>(*cornerHeightsPtr, modelSettings_.terrainHeight);
      // A one-sided quadratic hinge, NOT a relaxed log barrier: the delta of 0 puts its zero exactly on the ground, so
      // the term is silent for a foot resting on the ground and only bites below it. A log barrier here pushed every
      // loaded foot into a hover; see ModelSettings::ContactImplicitConfig::penetrationWeight.
      auto penalty = std::make_unique<SquaredHingePenalty>(SquaredHingePenalty::Config(contactImplicit.penetrationWeight, 0.0));
      problemPtr_->stateSoftConstraintPtr->add(absl::StrCat(footName, "_groundPenetration"),
                                               std::make_unique<StateSoftConstraint>(std::move(penetration), std::move(penalty)));
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
    // The same row, priced instead of imposed. Imposed, it fixes the whole height profile of a scheduled swing and the
    // solver can neither land early nor late; priced, it shapes the swing and is overruled whenever anything else pays
    // more, which is what the contact-implicit formulation needs of a reduced-order plan's guidance.
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::NormalVelocity) && eeKinematicsPtr) {
      auto penalty = std::make_unique<QuadraticPenalty>(modelSettings_.footConstraintConfig.normalVelocitySoftConstraintWeight);
      problemPtr_->softConstraintPtr->add(
          absl::StrCat(footName, "_normalVelocitySoft"),
          std::make_unique<StateInputSoftConstraint>(getNormalVelocityConstraint(*eeKinematicsPtr, i), std::move(penalty)));
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

  auto constraint = std::make_unique<ZeroVelocityConstraintCppAd>(*referenceManagerPtr_, eeKinematics, contactPointIndex, numConstraints,
                                                                  eeZeroVelConConfig(footConfig));
  constraint->getTwistConstraint().setConstrainYawRateAboutNormal(footConfig.constrainYawRateAboutContactNormal);
  return constraint;
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
