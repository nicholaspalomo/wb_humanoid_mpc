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

#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

// Pinocchio forward declarations must be included first
#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_centroidal_model/FactoryFunctions.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_core/misc/Numerics.h"
#include "ocs2_core/penalties/Penalties.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"
#include "ocs2_oc/synchronized_module/SolverSynchronizedModule.h"
#include "ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/config/costs/DcmTerminalCostFromConfig.h"
#include "humanoid_centroidal_mpc/config/costs/IcpCostFromConfig.h"
#include "humanoid_centroidal_mpc/constraint/JointMimicKinematicConstraint.h"
#include "humanoid_centroidal_mpc/constraint/NormalVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/constraint/ZeroVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsAD.h"
#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsBasisInputsAD.h"
#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/HumanoidPreComputation.h"
#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/BasisInputsMappingDecorator.h"
#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/contact_planning/ContactPlanningFromConfig.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/config/costs/JointMimicFromConfig.h"
#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"
#include "humanoid_common_mpc/config/swing/LocomotionHeuristicsFromConfig.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h"
#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"
#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/contact/FootprintCornerHeights.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.pb.h"
#include "humanoid_mpc_config/contact_planning_file.pb.h"
#include "humanoid_mpc_config/reference_file.nproto.pb.h"
#include "humanoid_mpc_config/reference_file.pb.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {

namespace {

/**
 * NotFound naming `path` when it does not exist; `what` says which of the interface's inputs it is. Create() runs it on
 * every input before anything reads them: the model settings read the URDF as well as the task file, and a missing file
 * used to surface as whatever the first reader threw.
 */
absl::Status checkInputFileExists(absl::string_view what, const std::string& path) {
  if (!std::filesystem::exists(path)) {
    return absl::NotFoundError(absl::StrCat("[CentroidalMpcInterface] ", what, " not found: ", path));
  }
  LOG(INFO) << "[CentroidalMpcInterface] " << what << ": " << path;
  return absl::OkStatus();
}

/** checkInputFileExists() of the task and reference files the path forms of Create() read. */
absl::Status checkConfigFilesExist(const std::string& taskFile, const std::string& referenceFile) {
  RETURN_IF_ERROR(checkInputFileExists("task file", taskFile));
  return checkInputFileExists("reference file", referenceFile);
}

/**
 * Runs `steps`, the part of the set-up that builds the models and the problem through OCS2 and Pinocchio, and returns their
 * Status, or InvalidArgument with what was thrown out of them: OCS2's model builders (centroidal_model::createCentroidalModelInfo,
 * the CppAD code generation of the dynamics, the kinematics and the costs) report a model they cannot build by throwing,
 * and the set-up is where that becomes a Status. The configuration itself is typed and converted without exceptions.
 * Nothing here runs on the solver or realtime threads.
 */
absl::Status catchingModelBuilderExceptions(absl::FunctionRef<absl::Status()> steps) {
  // NOLINTNEXTLINE(exceptions): OCS2's model builders report a model they cannot build by throwing; converted here, once.
  try {
    return steps();
  } catch (const std::exception& error) {  // NOLINT(exceptions): the boundary of the try above.
    return absl::InvalidArgumentError(absl::StrCat("[CentroidalMpcInterface] the MPC was not set up: ", error.what()));
  }
}

/** Logs the files of `config` as the textprotos they were read as: the verbose start-up's record of what it runs with. */
void logConfig(const CentroidalMpcConfig& config) {
  humanoid_mpc_config::TaskFile task;
  mpc_config::ToProto(config.task, &task);
  LOG(INFO) << "[CentroidalMpcInterface] the task file:\n" << nproto::WriteTextproto(task);
  humanoid_mpc_config::ReferenceFile reference;
  mpc_config::ToProto(config.reference, &reference);
  LOG(INFO) << "[CentroidalMpcInterface] the reference file:\n" << nproto::WriteTextproto(reference);
  if (config.contactPlanning.has_value()) {
    humanoid_mpc_config::ContactPlanningFile contactPlanning;
    mpc_config::ToProto(*config.contactPlanning, &contactPlanning);
    LOG(INFO) << "[CentroidalMpcInterface] the contact planner's file:\n" << nproto::WriteTextproto(contactPlanning);
  }
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

CentroidalMpcInterface::CentroidalMpcInterface(CentroidalMpcConfig config,
                                               ModelSettings modelSettings,
                                               SolverSettings solverSettings,
                                               const std::string& urdfFile)
    : config_(std::move(config)),
      modelSettings_(std::move(modelSettings)),
      solverSettings_(std::move(solverSettings)),
      urdfFile_(urdfFile),
      verbose_(solverSettings_.verbose) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status CentroidalMpcInterface::setupModels() {
  // PinocchioInterface: a model whose actuated joints are not model_settings' in order is refused here, by name.
  ASSIGN_OR_RETURN(PinocchioInterface pinocchioInterface,
                   loadCustomPinocchioInterface(config_.task, urdfFile_, modelSettings_, /*scaleTotalMass=*/false));
  pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));

  // CentroidalModelInfo
  ASSIGN_OR_RETURN(const CentroidalModelType centroidalModelType, centroidalModelTypeFromConfig(config_.task));
  ASSIGN_OR_RETURN(const vector_t defaultJointState,
                   defaultJointStateFromConfig(config_.reference, modelSettings_.mpcModelJointNames, modelSettings_.fixedJointNames));
  centroidalModelInfo_ = centroidal_model::createCentroidalModelInfo(*pinocchioInterfacePtr_, centroidalModelType, defaultJointState,
                                                                     modelSettings_.contactNames3DoF, modelSettings_.contactNames6DoF);

  LOG(INFO) << "centroidalModelInfo_.numSixDofContacts: " << centroidalModelInfo_.numSixDofContacts;
  for (size_t i = 0; i < centroidalModelInfo_.numSixDofContacts; ++i) {
    LOG(INFO) << "frameIndices: " << centroidalModelInfo_.endEffectorFrameIndices[i];
  }

  // Setup Centroidal State Input Mapping
  mpcRobotModelPtr_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(modelSettings_, *pinocchioInterfacePtr_, centroidalModelInfo_);
  mpcRobotModelADPtr_ = std::make_unique<CentroidalMpcRobotModel<ad_scalar_t>>(modelSettings_, (*pinocchioInterfacePtr_).toCppAd(),
                                                                               centroidalModelInfo_.toCppAd());
  // The wrench-space models until setupContactInputParameterization() has read the task file's choice.
  effectiveMpcRobotModelPtr_ = mpcRobotModelPtr_.get();
  effectiveMpcRobotModelADPtr_ = mpcRobotModelADPtr_.get();

  // initial state, by name on the coordinates of the centroidal state (StateInputLayout)
  ASSIGN_OR_RETURN(initialState_,
                   stateValuesFromConfig(config_.task.initial_state, stateInputLayout(modelSettings_, StateInputLayout::Mpc::kCentroidal),
                                         "initial_state"));
  if (static_cast<size_t>(initialState_.size()) != centroidalModelInfo_.stateDim) {
    return absl::InternalError(absl::StrCat("[CentroidalMpcInterface] initial_state has ", initialState_.size(),
                                            " coordinates, but the centroidal state has ", centroidalModelInfo_.stateDim, "."));
  }

  // The robot's pendulum length at its nominal posture: what a dcm_terminal_cost or a contact_planning.textproto shared
  // block without com_height stands for.
  nominalComHeight_ =
      computeComHeightAboveFeet(mpcRobotModelPtr_->getGeneralizedCoordinates(initialState_), *pinocchioInterfacePtr_, *mpcRobotModelPtr_);
  LOG(INFO) << "[CentroidalMpcInterface] the model's center of mass is " << nominalComHeight_
            << " m above its feet at initial_state (the pendulum length a com_height left out stands for).";
  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> CentroidalMpcInterface::Create(const std::string& taskFile,
                                                                                       const std::string& urdfFile,
                                                                                       const std::string& referenceFile) {
  // The inputs exist before anything reads them.
  RETURN_IF_ERROR(checkConfigFilesExist(taskFile, referenceFile));
  ASSIGN_OR_RETURN(CentroidalMpcConfig config, loadCentroidalMpcConfig(taskFile, referenceFile));
  return Create(std::move(config), urdfFile);
}

absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> CentroidalMpcInterface::Create(CentroidalMpcConfig config,
                                                                                       const std::string& urdfFile) {
  return build(std::move(config), urdfFile, Scope::kMpc);
}

absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> CentroidalMpcInterface::CreateControllerModels(const std::string& taskFile,
                                                                                                       const std::string& urdfFile,
                                                                                                       const std::string& referenceFile) {
  RETURN_IF_ERROR(checkConfigFilesExist(taskFile, referenceFile));
  ASSIGN_OR_RETURN(CentroidalMpcConfig config, loadCentroidalMpcConfig(taskFile, referenceFile));
  return CreateControllerModels(std::move(config), urdfFile);
}

absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> CentroidalMpcInterface::CreateControllerModels(CentroidalMpcConfig config,
                                                                                                       const std::string& urdfFile) {
  return build(std::move(config), urdfFile, Scope::kControllerModels);
}

absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> CentroidalMpcInterface::build(CentroidalMpcConfig config,
                                                                                      const std::string& urdfFile,
                                                                                      Scope scope) {
  // The model settings read the URDF.
  RETURN_IF_ERROR(checkInputFileExists("URDF file", urdfFile));
  // The task file's interface.verbose first: it decides the logging of the model settings.
  ASSIGN_OR_RETURN(SolverSettings solverSettings, solverSettingsFromConfig(config.task));
  if (solverSettings.verbose) {
    logConfig(config);
  }
  ASSIGN_OR_RETURN(ModelSettings modelSettings, ModelSettings::Create(config.task, urdfFile, "centroidal_mpc_", solverSettings.verbose));
  std::unique_ptr<CentroidalMpcInterface> interface =
      absl::WrapUnique(new CentroidalMpcInterface(std::move(config), std::move(modelSettings), std::move(solverSettings), urdfFile));
  RETURN_IF_ERROR(catchingModelBuilderExceptions([&]() -> absl::Status {
    // In this order: the robot models first; the reference manager and the contact planner are built on the effective
    // robot model, which the contact input parameterization decides, and the problem on all three. The models of a robot
    // process whose MPC runs elsewhere are the first two steps: the reference manager and the problem, which tape and
    // load the CppAD libraries, are the MPC's.
    RETURN_IF_ERROR(interface->setupModels());
    RETURN_IF_ERROR(interface->setupContactInputParameterization());
    if (scope == Scope::kControllerModels) {
      return absl::OkStatus();
    }
    RETURN_IF_ERROR(interface->setupReferenceManager());
    return interface->setupOptimalControlProblem();
  }));
  return interface;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

namespace {

/**
 * Whether the horizontal position of a swing foot's landing target reaches the problem: task_space_foot_cost is listed
 * and task_space_foot_cost.weights gives pos_x or pos_y a non-zero weight. A foothold - planned by the contact planner
 * or computed by a locomotion heuristic - reaches the MPC only through that cost (CentroidalMpcEndEffectorFootCost),
 * which multiplies its xy residual by exactly these two weights.
 */
absl::StatusOr<bool> footPositionIsTracked(const mpc_config::TaskFile& task, const MpcFormulationTasks& formulationTasks) {
  if (!formulationTasks.hasCost(MpcCostType::kTaskSpaceFootCost)) return false;
  ASSIGN_OR_RETURN(const TaskSpaceFootCostSettings footCost, taskSpaceFootCostFromConfig(task.task_space_foot_cost));
  return footCost.weights.contactPositionErrorWeight(0) != 0.0 || footCost.weights.contactPositionErrorWeight(1) != 0.0;
}

}  // namespace

absl::Status CentroidalMpcInterface::setupContactInputParameterization() {
  ASSIGN_OR_RETURN(contactInputParameterization_, contactInputParameterizationFromConfig(config_.task));
  LOG(INFO) << "[CentroidalMpcInterface] contact_input_parameterization: "
            << contactInputParameterizationName(contactInputParameterization_);

  if (contactInputParameterization_ == ContactInputParameterization::kWrench) {
    effectiveMpcRobotModelPtr_ = mpcRobotModelPtr_.get();
    effectiveMpcRobotModelADPtr_ = mpcRobotModelADPtr_.get();
    keyCppAdModelFolder(kWrenchInputsLibraryKey);
    return absl::OkStatus();
  }

  // The bases of both feet: the generator set named by contacts.basis_generator_set, built from the
  // contacts.contact_wrench_cone_soft_constraint block (every geometry field required) and each foot's contact
  // rectangle. The set fixes the input dimension, so it is read here and never reloaded.
  const mpc_config::ContactsConfig& contacts = config_.task.contacts;
  ASSIGN_OR_RETURN(const feet_array_t<ContactWrenchConeBasisMatrix> basisMatrices,
                   contactWrenchConeBasesFromConfig(contacts, modelSettings_));
  basisGeneratorSet_ = basisMatrices[0].generatorSet();
  ASSIGN_OR_RETURN(const ContactWrenchConeConstraint::Config coneConfig, contactWrenchConeConfigFromConfig(contacts));

  // The regularization of the lambda block of the input cost: M has a non-trivial null space, so M^T R M alone leaves
  // the Hessian of the scalings singular. Its weight and its shape S are both read here; the online parameter updater
  // reads the same two fields on a hot reload.
  ASSIGN_OR_RETURN(const BasisRegularizationSettings regularization, basisRegularizationFromConfig(contacts));
  basisScalingRegularization_ = regularization.basisScalingRegularization;
  basisRegularization_ = regularization.basisRegularization;

  // A conic combination is homogeneous, so the basis can represent neither the minimum normal force nor a gripper
  // adhesion force: both are affine offsets of the cone and lambda = 0 always yields the zero wrench.
  if (coneConfig.minNormalForce > 0.0 || coneConfig.gripperForce > 0.0) {
    LOG(WARNING) << "[CentroidalMpcInterface] contacts.contact_wrench_cone_soft_constraint.min_normal_force (" << coneConfig.minNormalForce
                 << " N) and gripper_force (" << coneConfig.gripperForce
                 << " N) are NOT enforced with contact_input_parameterization: " << kBasisVectorsContactInputParameterization
                 << ". Every generator lies inside the cone, so its friction, center-of-pressure and torsional limits are never "
                    "exceeded; these two affine offsets cannot be represented by a conic combination at all.";
  }

  const size_t numBasisPerFoot = basisMatrices[0].numBasis();
  LOG(INFO) << "[CentroidalMpcInterface] basis-vector contact inputs, generator set '" << basisGeneratorSet_ << "': " << numBasisPerFoot
            << " basis vectors per foot (total input dim: " << numBasisPerFoot * kNumContacts + modelSettings_.mpc_joint_dim
            << "); lambda regularization '" << basisRegularization_ << "' with weight " << basisScalingRegularization_ << ".";

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

  // An unknown regularization name or a negative weight, named by its key.
  RETURN_IF_ERROR(validateBasisInputsCostTransformConfig(makeBasisInputsCostTransformConfig(*basisToWrenchMap_)));

  keyCppAdModelFolder(basisInputsLibraryKey(basisDecoratorPtr_->getBasisMatrices()));
  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void CentroidalMpcInterface::keyCppAdModelFolder(absl::string_view libraryKey) {
  // Every CppAD tape of this interface is built on the effective model or on the input mapping, so its domain - and,
  // for the external-torque cost and the contact-moment constraint, its body - depends on the contact input
  // parameterization and, under basis vectors, on the basis itself. OCS2 loads a cached library by name without
  // checking its domain, and every robot ships recompileLibrariesCppAd: false, so without this key a library built for
  // the other parameterization (or another basis) would be loaded and the process would exit on its first evaluation.
  // One folder per key keeps them apart, for every term at once.
  modelSettings_.modelFolderCppAd = absl::StrCat(modelSettings_.modelFolderCppAd, "/", libraryKey);
  LOG(INFO) << "[CentroidalMpcInterface] CppAD model folder: " << modelSettings_.modelFolderCppAd;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status CentroidalMpcInterface::setupReferenceManager() {
  // Swing trajectory planner
  ASSIGN_OR_RETURN(const SwingTrajectoryPlanner::Config swingConfig,
                   swingTrajectorySettingsFromConfig(config_.task.swing_trajectory_config));
  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner = std::make_unique<SwingTrajectoryPlanner>(swingConfig, kNumContacts);

  // Where the mode schedule and the footholds come from, by name.
  ASSIGN_OR_RETURN(contactScheduleSource_, contactScheduleSourceFromConfig(config_.task));
  LOG(INFO) << "[CentroidalMpcInterface] contact_schedule_source: " << contactScheduleSourceName(contactScheduleSource_);
  // Either source starts from the reference file's gait schedule.
  ASSIGN_OR_RETURN(std::shared_ptr<GaitSchedule> gaitSchedule, GaitSchedule::Create(config_.reference, modelSettings_, verbose_));

  if (contactScheduleSource_ == ContactScheduleSource::kGaitSchedule) {
    referenceManagerPtr_ = std::make_shared<SwitchedModelReferenceManager>(std::move(gaitSchedule), std::move(swingTrajectoryPlanner),
                                                                           *pinocchioInterfacePtr_, *effectiveMpcRobotModelPtr_);
    return absl::OkStatus();
  }

  // contact_planner: online contact planning (the planner planner.type of contact_planning.textproto names) replaces the
  // periodic gait schedule. The gait schedule is still loaded: it is used until the first plan arrives and whenever the
  // planner has no valid plan. A robot without the planner's file runs its library defaults.
  LOG(INFO) << "[CentroidalMpcInterface] the contact planner runs on "
            << (config_.contactPlanning.has_value() ? "the robot's contact_planning.textproto"
                                                    : "its library defaults: the robot has no contact_planning.textproto");
  // Not validated yet: the values a file may leave out (or the ZMP box at 0) to mean "from the model" are filled in below.
  ASSIGN_OR_RETURN(ContactPlanningConfig contactPlanningConfig,
                   contactPlanningConfigFromOptionalFile(config_.contactPlanning.has_value() ? &*config_.contactPlanning : nullptr,
                                                         ContactPlanningValidation::kDeferUntilModelParametersApplied));
  const std::optional<scalar_t> fileComHeight = contactPlanningConfig.shared.comHeight;

  // Parameters the planner's file leaves out (shared.com_height) or at 0 (the ZMP box) are derived from the robot model and from the ground
  // parameters of the wrench cone, so that the planner never assumes more friction, torque or footprint than the whole-body constraints
  // allow. The initial state of the task file is the nominal posture for the LIP height, the one every LIP consumer
  // shares. The same conversion as the wrench cone and the basis generators: a task file whose cone leaves out a ground
  // field is refused rather than planned against the library default of mu 0.7.
  ASSIGN_OR_RETURN(const ContactWrenchConeConstraint::Config coneConfig, contactWrenchConeConfigFromConfig(config_.task.contacts));
  ContactPlanningGroundParameters ground;
  ground.frictionCoefficient = coneConfig.frictionCoefficient;
  ground.torsionalFrictionCoefficient = coneConfig.torsionalFrictionCoefficient;
  // Without a footprint the planner derives no ZMP box from it.
  const absl::StatusOr<ContactRectangle> footprint = contactRectangleFromConfig(config_.task.contacts, modelSettings_, /*contactIndex=*/0);
  if (footprint.ok()) {
    ground.footprintHalfLengthX = 0.5 * (footprint->getBounds().x_max - footprint->getBounds().x_min);
    ground.footprintHalfWidthY = 0.5 * (footprint->getBounds().y_max - footprint->getBounds().y_min);
  } else {
    LOG(WARNING) << "[CentroidalMpcInterface] no footprint for the contact planner's ZMP box: " << footprint.status().message();
  }
  contactPlanningModelParameters_ = deriveContactPlanningModelParameters(
      *pinocchioInterfacePtr_, *effectiveMpcRobotModelPtr_, initialState_, modelSettings_.contactParentJointNames, ground,
      contactPlanningConfig.shared.gravity, contactPlanningConfig.stepWidth.nominalStepWidth);
  contactPlanningModelParameters_->applyTo(contactPlanningConfig);
  LOG(INFO) << "[CentroidalMpcInterface] contact planner model parameters: " << contactPlanningModelParameters_->summary();
  RETURN_IF_ERROR(contactPlanningConfig.validateStatus());
  LOG(INFO) << "[CentroidalMpcInterface] contact planner pendulum: shared.com_height " << contactPlanningConfig.pendulumHeight() << " m"
            << (fileComHeight.has_value() ? " (explicit)" : " (the model's center of mass above its feet at initial_state)") << ", omega "
            << contactPlanningConfig.omega() << " rad/s.";
  if (contactPlanningConfig.horizon() < mpcSettings().timeHorizon_) {
    LOG(WARNING) << "[CentroidalMpcInterface] contact_planning horizon (" << contactPlanningConfig.horizon()
                 << " s) is shorter than the MPC horizon (" << mpcSettings().timeHorizon_
                 << " s); the schedule beyond the planned horizon defaults to double support.";
  }
  ASSIGN_OR_RETURN(std::shared_ptr<ContactPlanningReferenceManager> planningReferenceManager,
                   ContactPlanningReferenceManager::Create(std::move(gaitSchedule), std::move(swingTrajectoryPlanner),
                                                           *pinocchioInterfacePtr_, *effectiveMpcRobotModelPtr_, contactPlanningConfig));
  // The heading model's ACoM evaluator is installed by setupOptimalControlProblem(), which returns a Status, through
  // ContactPlanningReferenceManager::loadHeadingModelEvaluator(), the same call that installs it on a hot reload.
  ASSIGN_OR_RETURN(contactPlannerModulePtr_,
                   ContactPlannerModule::Create(planningReferenceManager, contactPlanningConfig, contactPlanningModelParameters_));
  referenceManagerPtr_ = planningReferenceManager;
  LOG(INFO) << "[CentroidalMpcInterface] Using contact planning with planner.type '"
            << canonicalPlannerName(contactPlanningConfig.planner.type) << "' (" << contactPlanningConfig.planner.numNodes << " nodes x "
            << contactPlanningConfig.planner.dt << " s, "
            << (contactPlanningConfig.planner.runInBackgroundThread ? "background thread" : "synchronous") << ").";
  return absl::OkStatus();
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
  ASSIGN_OR_RETURN(const LocomotionHeuristicConfig heuristicConfig,
                   locomotionHeuristicConfigFromConfig(config_.task.locomotion_heuristics));
  const LocomotionHeuristicFormulation& formulation = heuristicConfig.formulation;
  if (verbose_ && !(formulation.basePose.empty() && formulation.foothold.empty() && formulation.wrench.empty())) {
    LOG(INFO) << "\n #### Locomotion Heuristics (Bledt RPC, Appendix C) of the task file's locomotion_heuristics";
    LOG(INFO) << "\n" << formulation.summary();
  }
  ASSIGN_OR_RETURN(const LocomotionHeuristicModelParameters heuristicModel,
                   deriveLocomotionHeuristicModelParameters(*pinocchioInterfacePtr_, *effectiveMpcRobotModelPtr_, initialState_));
  // What the rest of this task file says that decides whether a listed heuristic can act at all; see
  // LocomotionHeuristicEnvironment.
  LocomotionHeuristicEnvironment environment;
  environment.usesContactPlanning = contactScheduleSource_ == ContactScheduleSource::kContactPlanner;
  environment.usesContactBasisVectorInputs = usesContactBasisVectorInputs();
  environment.nominalStepWidth = modelSettings_.nominalFootholdConfig.stepWidth;
  ASSIGN_OR_RETURN(const MpcFormulationTasks formulationTasks, mpcFormulationTasksFromConfig(config_.task, FormulationLogging::kQuiet));
  environment.listsComAndAcomTrackingCost = formulationTasks.hasCost(MpcCostType::kComAndAcomTrackingCost);
  ASSIGN_OR_RETURN(const bool footPositionTracked, footPositionIsTracked(config_.task, formulationTasks));
  environment.footPositionIsUntracked = !footPositionTracked;
  ASSIGN_OR_RETURN(std::unique_ptr<LocomotionHeuristicLayer> heuristicLayer,
                   LocomotionHeuristicLayer::Create(heuristicConfig, heuristicModel, environment, verbose_));
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
  ASSIGN_OR_RETURN(const MpcFormulationTasks formulationTasks,
                   mpcFormulationTasksFromConfig(config_.task, formulationLoggingFor(verbose_)));
  const bool scheduleGatedContactConstraints = contactConstraintsAreScheduleGated(formulationTasks);
  const mpc_config::TaskFile& task = config_.task;

  // The contact_implicit block, before any term is built (a field the schema does not know - a renamed or misspelled
  // one - is refused by the strict parser). The values are divisors and penalty weights: refused with a Status naming the
  // field, rather than CHECK-aborting inside a term constructor or turning a penalty into a reward, whenever the
  // formulation that reads them is listed.
  if (usesContactImplicitFormulation(formulationTasks)) {
    RETURN_IF_ERROR(validateContactImplicitConfig(modelSettings_.contactImplicitConfig));
  }

  // The planner's footholds are its whole stabilizing mechanism - the H-LIP deadbeat step is a foot placement - and they
  // reach this problem only through the foot cost's xy weights. At 0 the planner runs, prints its formulation and
  // re-times the schedule while the MPC places the swing foot wherever the other costs prefer; say so rather than let
  // it look like a planner that does not work.
  ASSIGN_OR_RETURN(const bool footPositionTracked, footPositionIsTracked(task, formulationTasks));
  if (contactScheduleSource_ == ContactScheduleSource::kContactPlanner && !footPositionTracked) {
    LOG(WARNING) << "[CentroidalMpcInterface] contact_schedule_source is " << kContactPlannerContactScheduleSource
                 << ", but the planned footholds carry no weight: they reach the MPC only through task_space_foot_cost, "
                 << (formulationTasks.hasCost(MpcCostType::kTaskSpaceFootCost)
                         ? "whose task_space_foot_cost.weights.pos_x and pos_y are both 0. Raise task_space_foot_cost.weights.pos_x / "
                           "pos_y"
                         : "which `costs` does not list. List task_space_foot_cost and give task_space_foot_cost.weights.pos_x / pos_y "
                           "a weight")
                 << " before relying on the planner's foot placement.";
  }

  // The parameters of the lambda >= 0 barrier that the basis-vector parameterization builds for every contact below
  // (contacts.basis_non_negativity_barrier). They are converted here, before any CppAD model is built, so that a value
  // that is not finite is refused by its field at once rather than thrown out of Create() minutes later.
  PieceWisePolynomialBarrierPenalty::Config lambdaBarrierConfig;
  // The `mu` of the wrench parameterization's cone, which the un-gated barrier is compared against below; 0 when the
  // file leaves it out.
  scalar_t wrenchConeBarrierMu = 0.0;
  if (usesContactBasisVectorInputs()) {
    ASSIGN_OR_RETURN(lambdaBarrierConfig, basisNonNegativityBarrierFromConfig(task.contacts));
    wrenchConeBarrierMu = task.contacts.contact_wrench_cone_soft_constraint.mu.value_or(0.0);
  }

  // The swing-foot cost (task_space_foot_cost): its weights and whether it also weighs a foot in stance, converted here
  // for the same reason.
  TaskSpaceFootCostSettings footCost;
  if (formulationTasks.hasCost(MpcCostType::kTaskSpaceFootCost)) {
    ASSIGN_OR_RETURN(footCost, taskSpaceFootCostFromConfig(task.task_space_foot_cost));
  }

  // Under online contact planning with the heading model, the planner reads the whole-body heading off the robot's ACoM
  // network. A robot without one falls back to the base yaw with a warning; a network trained on other joints is an
  // error, because it would run on the wrong joint vector.
  const std::shared_ptr<ContactPlanningReferenceManager> planningReferenceManager =
      // NOLINTNEXTLINE(rtti): the reference manager is a ContactPlanningReferenceManager only under contact planning.
      std::dynamic_pointer_cast<ContactPlanningReferenceManager>(referenceManagerPtr_);
  if (planningReferenceManager != nullptr) {
    RETURN_IF_ERROR(planningReferenceManager->loadHeadingModelEvaluator());
  }

  // CoM + ACoM tracking is a cost of the `costs` list, and everything it implies follows from that one entry: the
  // ComAndAcomTrackingCost itself, the zeroed base-pose blocks of Q and Q_final (the factory), the terminal instance
  // beside terminal_cost, and the procedural arm swing switched off - the arm motion emerges from the ACoM cost, and a
  // generator driving the same joints would fight it. A legs-only robot omits model_settings.arm_joint_names and has no
  // arm to swing either way.
  const bool comAndAcomTracking = formulationTasks.hasCost(MpcCostType::kComAndAcomTrackingCost);
  referenceManagerPtr_->setArmSwingReferenceActive(modelSettings_.hasArmSwingJoints && !comAndAcomTracking);

  HumanoidCostConstraintFactory factory(&task, StateInputLayout::Mpc::kCentroidal, *referenceManagerPtr_, *pinocchioInterfacePtr_,
                                        *effectiveMpcRobotModelPtr_, *effectiveMpcRobotModelADPtr_, modelSettings_, verbose_,
                                        scheduleGatedContactConstraints);
  factory.setComAndAcomTrackingCostListed(comAndAcomTracking);

  // Optimal control problem
  problemPtr_ = std::make_unique<OptimalControlProblem>();

  // Dynamics
  std::unique_ptr<SystemDynamicsBase> dynamicsPtr;
  const std::string modelName = "dynamics";
  if (usesContactBasisVectorInputs()) {
    // R is written in wrench dimensions and transformed into basis space, R_basis = M^T R_wrench M + reg * blkdiag(S, 0),
    // through the same config the online parameter updater applies on a hot reload.
    const std::optional<BasisInputsCostTransformConfig> basisCostTransformConfig = getBasisInputsCostTransformConfig();
    if (!basisCostTransformConfig.has_value()) {
      return absl::InternalError("[CentroidalMpcInterface] basis-vector contact inputs without their basis: set up the models first");
    }
    const BasisInputsCostTransformConfig& basisCostTransform = *basisCostTransformConfig;
    if (formulationTasks.hasCost(MpcCostType::kInputQuadraticCost) || formulationTasks.hasCost(MpcCostType::kStateInputQuadraticCost)) {
      // The QP needs a unique input: the lambda block of R_basis has to be positive definite, which a zero weight, or
      // null_space with a contact wrench direction R does not weigh, would break.
      ASSIGN_OR_RETURN(const matrix_t R_wrench,
                       inputWeightsFromConfig(task.input_weights, stateInputLayout(modelSettings_, StateInputLayout::Mpc::kCentroidal),
                                              "input_weights"));
      if (static_cast<size_t>(R_wrench.rows()) != basisCostTransform.wrenchInputDim) {
        return absl::InternalError(absl::StrCat("[CentroidalMpcInterface] input_weights has ", R_wrench.rows(),
                                                " rows, but the wrench-space input has ", basisCostTransform.wrenchInputDim, "."));
      }
      RETURN_IF_ERROR(checkLambdaBlockPositiveDefinite(transformWrenchInputCostToBasisSpace(R_wrench, basisCostTransform),
                                                       basisCostTransform.numBasisInputs));
    }
    factory.setBasisInputsCostTransform(basisCostTransform);

    // The dynamics rotate each local-frame basis wrench B_i * λ_i into the world frame inside the CppAD tape.
    ASSIGN_OR_RETURN(std::unique_ptr<CentroidalDynamicsBasisInputsAD> basisDynamicsPtr,
                     CentroidalDynamicsBasisInputsAD::Create(*pinocchioInterfacePtr_, centroidalModelInfo_, modelName, modelSettings_,
                                                             basisDecoratorPtr_->getBasisMatrices()));
    dynamicsPtr = std::move(basisDynamicsPtr);
  } else {
    dynamicsPtr = std::make_unique<CentroidalDynamicsAD>(*pinocchioInterfacePtr_, centroidalModelInfo_, modelName, modelSettings_);
  }
  problemPtr_->dynamicsPtr = std::move(dynamicsPtr);

  // Cost terms
  if (formulationTasks.hasCost(MpcCostType::kStateInputQuadraticCost)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> stateInputQuadraticCost, factory.makeStateInputQuadraticCost());
    problemPtr_->costPtr->add(kStateInputQuadraticCostTerm, std::move(stateInputQuadraticCost));
  }
  if (formulationTasks.hasCost(MpcCostType::kStateQuadraticCost)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> stateQuadraticCost, factory.makeStateQuadraticCost());
    problemPtr_->costPtr->add(kStateQuadraticCostTerm, std::move(stateQuadraticCost));
  }
  // Beside whichever quadratic state cost is listed, whose base-pose block the factory has zeroed - or on its own.
  if (comAndAcomTracking) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateCost> comAndAcomTrackingCost, factory.makeComAndAcomTrackingCost(centroidalModelInfo_));
    problemPtr_->stateCostPtr->add(std::string(ComAndAcomTrackingCost::kRunningTermName), std::move(comAndAcomTrackingCost));
  }
  if (formulationTasks.hasCost(MpcCostType::kInputQuadraticCost)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> inputQuadraticCost, factory.makeInputQuadraticCost());
    problemPtr_->costPtr->add(kInputQuadraticCostTerm, std::move(inputQuadraticCost));
  }
  // Terminal cost: the DCM viability cost (dcm_terminal_cost) or the quadratic final_state_weights cost (terminal_cost),
  // whichever the costs list names; the conversion refuses a list that names both, so the two branches below are
  // exclusive.
  if (formulationTasks.hasCost(MpcCostType::kDcmTerminalCost)) {
    // A dcm_terminal_cost without com_height has the model's pendulum length, resolved here and, through the cost's own
    // setConfig(), on every hot reload of the block.
    ASSIGN_OR_RETURN(const DcmTerminalCost::Config dcmConfig, dcmTerminalCostConfigFromConfig(task.dcm_terminal_cost));
    ASSIGN_OR_RETURN(std::unique_ptr<DcmTerminalCost> dcmTerminalCost,
                     DcmTerminalCost::Create(*referenceManagerPtr_, dcmConfig, nominalComHeight_, *pinocchioInterfacePtr_,
                                             *effectiveMpcRobotModelADPtr_, DcmTerminalCost::kLibraryName, modelSettings_));
    LOG(INFO) << "[CentroidalMpcInterface] DCM terminal cost pendulum: dcm_terminal_cost.com_height "
              << dcmTerminalCost->getConfig().comHeight.value_or(std::numeric_limits<scalar_t>::quiet_NaN()) << " m"
              << (dcmConfig.comHeight.has_value() ? " (explicit)" : " (the model's center of mass above its feet at initial_state)")
              << ", omega " << dcmTerminalCost->getConfig().omega() << " rad/s.";
    problemPtr_->finalCostPtr->add(DcmTerminalCost::kTermName, std::move(dcmTerminalCost));
  } else if (formulationTasks.hasCost(MpcCostType::kTerminalCost)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateCost> terminalCost, factory.makeTerminalCost());
    problemPtr_->finalCostPtr->add(kTerminalCostTerm, std::move(terminalCost));
    // final_state_weights' base-pose block is zeroed under com_and_acom_tracking_cost; this is what regulates the terminal
    // CoM, height and orientation in its place, in the coordinates every other node uses.
    if (comAndAcomTracking) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateCost> terminalComAndAcomTrackingCost,
                       factory.makeTerminalComAndAcomTrackingCost(centroidalModelInfo_));
      problemPtr_->finalCostPtr->add(std::string(ComAndAcomTrackingCost::kTerminalTermName), std::move(terminalComAndAcomTrackingCost));
    }
  }

  const CentroidalModelInfoCppAd infoCppAd = centroidalModelInfo_.toCppAd();
  const CentroidalModelPinocchioMappingCppAd pinocchioMappingCppAdBase(infoCppAd);

  // When using basis-vector inputs, wrap the pinocchio mapping so that CppAD-compiled
  // EE kinematics (zero-velocity, normal-velocity constraints, foot tracking cost)
  // first convert u_basis → u_wrench = M * u_basis before extracting joint velocities at
  // wrench-space offsets. Only the joint-velocity block of M matters here (the kinematics
  // never read the contact block), so the local-frame map is sufficient.
  std::unique_ptr<PinocchioStateInputMapping<ad_scalar_t>> effectiveMappingPtr;
  if (usesContactBasisVectorInputs()) {
    if (!basisToWrenchMap_.has_value()) {
      return absl::InternalError("[CentroidalMpcInterface] basis-vector contact inputs without their basis: set up the models first");
    }
    effectiveMappingPtr = std::make_unique<BasisInputsMappingDecorator<ad_scalar_t>>(
        std::unique_ptr<PinocchioStateInputMapping<ad_scalar_t>>(pinocchioMappingCppAdBase.clone()), *basisToWrenchMap_,
        getWrenchInputDim(), modelSettings_.mpc_joint_dim);
  } else {
    effectiveMappingPtr = std::unique_ptr<PinocchioStateInputMapping<ad_scalar_t>>(pinocchioMappingCppAdBase.clone());
  }
  const PinocchioStateInputMapping<ad_scalar_t>& pinocchioMappingCppAd = *effectiveMappingPtr;

  const PinocchioEndEffectorKinematicsCppAd::update_pinocchio_interface_callback velocityUpdateCallback =
      [&infoCppAd](const ad_vector_t& state, PinocchioInterfaceCppAd& pinocchioInterfaceAd) {
        const ad_vector_t q = centroidal_model::getGeneralizedCoordinates(state, infoCppAd);
        updateCentroidalDynamics(pinocchioInterfaceAd, infoCppAd, q);
      };

  if (formulationTasks.hasCost(MpcCostType::kTaskSpaceTorsoCost)) {
    RETURN_IF_ERROR(addTaskSpaceKinematicsCosts(pinocchioMappingCppAd, velocityUpdateCallback));
  }

  if (formulationTasks.hasCost(MpcCostType::kIcpCost)) {
    ASSIGN_OR_RETURN(const vector2_t icpWeights, icpCostWeightsFromConfig(task.icp_cost_weights));
    problemPtr_->costPtr->add(
        kIcpCostTerm, std::make_unique<ICPCost>(*referenceManagerPtr_, icpWeights, *pinocchioInterfacePtr_, *effectiveMpcRobotModelADPtr_,
                                                ICPCost::kLibraryName, modelSettings_));
  }

  // Soft constraints
  if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kJointLimits)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateCost> jointLimits, factory.makeJointLimitsConstraint());
    problemPtr_->stateSoftConstraintPtr->add(kJointLimitsTerm, std::move(jointLimits));
  }
  if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kFootCollision)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateCost> footCollision, factory.makeFootCollisionConstraint());
    problemPtr_->stateSoftConstraintPtr->add(kFootCollisionTerm, std::move(footCollision));
  }

  // The knee mimic joints, in the order of the contacts, when the hard constraint is listed.
  feet_array_t<JointMimicSettings> kneeMimicJoints;
  if (formulationTasks.hasHardConstraint(MpcHardConstraintType::kKneeJointMimic)) {
    ASSIGN_OR_RETURN(kneeMimicJoints, kneeMimicKinematicJointsFromConfig(task.mimic_joints));
  }

  // Constraint terms
  for (size_t i = 0; i < kNumContacts; ++i) {
    const std::string& footName = modelSettings_.contactNames[i];

    // The kinematics of the CONTACT FRAME, i.e. the center of the sole. `ground_penetration` is deliberately absent
    // from this list: it is evaluated at the footprint's corner frames instead and builds its own kinematics below, so
    // listing it here would generate a whole extra CppAD model per foot that nothing reads.
    std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKinematicsPtr;
    const bool needsEeKinematics = formulationTasks.hasHardConstraint(MpcHardConstraintType::kZeroVelocity) ||
                                   formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kZeroVelocity) ||
                                   formulationTasks.hasHardConstraint(MpcHardConstraintType::kNormalVelocity) ||
                                   formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kNormalVelocity) ||
                                   formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kForceWeightedSlip);
    // Neither contact_complementarity nor ground_penetration appears here: both read the footprint CORNERS through a
    // FootprintCornerHeights of their own, not this contact-frame kinematics.
    if (needsEeKinematics) {
      const size_t effectiveInputDim = effectiveMpcRobotModelPtr_->getInputDim();
      eeKinematicsPtr = std::make_unique<PinocchioEndEffectorKinematicsCppAd>(
          *pinocchioInterfacePtr_, pinocchioMappingCppAd, std::vector<std::string>{footName}, centroidalModelInfo_.stateDim,
          effectiveInputDim, velocityUpdateCallback, footName, modelSettings_.modelFolderCppAd, modelSettings_.recompileLibrariesCppAd,
          modelSettings_.verboseCppAd);
    }

    if (usesContactBasisVectorInputs()) {
      // In basis-vector mode the friction, CoP and torsional limits are enforced structurally: every generator lies
      // inside the cone (ContactWrenchConeBasisMatrix verifies this against the constraint's own rows at construction)
      // and the cone is convex, so λ ≥ 0 ⟹ W ∈ cone. The ContactWrenchConeConstraint is wrench-space specific and cannot
      // operate on basis-vector inputs, so a listed `contact_wrench_cone` builds nothing here. The minimum normal force
      // and the gripper force are the exception and are not enforced; a warning is logged where the basis is built.
      if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kContactWrenchCone)) {
        LOG(INFO) << "[CentroidalMPC] Skipping contact_wrench_cone soft constraint for " << footName
                  << " (friction, CoP and torsional limits enforced structurally via basis-vector inputs).";
      }

      // The λ ≥ 0 barrier IS the contact wrench cone of this parameterization, so it is built for every contact whatever
      // the soft-constraint lists say, gated exactly as `zero_wrench` dictates. Nothing else bounds the individual
      // scalings - `friction_force_cone` bounds only the assembled force - and without it a negative scaling is
      // adhesion, a center of pressure outside the sole or unbounded torsion. (It used to be built only when
      // `contact_wrench_cone` was listed, so dropping that entry silently removed the cone.) Its parameters,
      // contacts.basis_non_negativity_barrier, were converted before the problem was built (lambdaBarrierConfig).

      // WHAT `mu` IS NOW BEING ASKED TO DO. Gated, this term is a redundant bound: the swing foot's scalings are
      // pinned to zero by `zero_wrench` and the stance foot's are pulled positive by R's weight-compensating
      // nominal, so 0.01 was tuned as a regularizer. Un-gated it becomes the ONLY thing holding every contact
      // wrench inside its cone, at every node, which is the job `contact_wrench_cone_soft_constraint.mu` does in the
      // wrench parameterization - and there it is 0.2. A scaling pair (+a, -a) costs only `mu * a^2` and leaves the
      // load indicator f_n = sum(lambda) at zero, so both contact-implicit products stay blind to it.
      //
      // Raising it is a closed-loop tuning decision and not one this loader may take on the operator's behalf, so
      // the mismatch is reported rather than patched. See humanoid_nmpc/docs/contact_implicit_mpc/README.md.
      if (!scheduleGatedContactConstraints && lambdaBarrierConfig.mu < wrenchConeBarrierMu) {
        LOG(WARNING) << "[CentroidalMpcInterface] " << footName << ": contacts.basis_non_negativity_barrier.mu = " << lambdaBarrierConfig.mu
                     << " is the ONLY bound on this contact's wrench once the schedule gate is off, and it is softer than the "
                     << "contacts.contact_wrench_cone_soft_constraint.mu = " << wrenchConeBarrierMu
                     << " that does the same job in the wrench parameterization. Tune it against a foot in flight before "
                     << "trusting the contact-implicit formulation on hardware.";
      }

      const size_t lambdaStartIdx = basisDecoratorPtr_->getContactWrenchStartIndices(i);
      const size_t numBasis = basisDecoratorPtr_->getNumBasisPerFoot();

      problemPtr_->costPtr->add(basisNonNegativityTermName(footName), std::make_unique<BasisScalingNonNegativityConstraint>(
                                                                          *referenceManagerPtr_, i, lambdaStartIdx, numBasis,
                                                                          lambdaBarrierConfig, scheduleGatedContactConstraints));

      LOG(INFO) << "[CentroidalMPC] Added λ ≥ 0 non-negativity barrier for " << footName << " (" << numBasis << " basis vectors, start idx "
                << lambdaStartIdx << ").";
    } else if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kContactWrenchCone)) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> contactWrenchCone, factory.makeContactWrenchConeConstraint(i));
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kContactWrenchCone), std::move(contactWrenchCone));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kFrictionForceCone)) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> frictionForceCone, factory.makeFrictionForceConeConstraint(i));
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kFrictionForceCone), std::move(frictionForceCone));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kContactMomentXy)) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> contactMomentXYConstraint,
                       factory.makeContactMomentXYConstraint(i, absl::StrCat(footName, "_contact_moment_XY_constraint")));
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kContactMomentXY),
                                          std::move(contactMomentXYConstraint));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kZeroVelocity) && eeKinematicsPtr) {
      std::unique_ptr<StateInputConstraint> stanceConstraint = getStanceFootConstraint(*eeKinematicsPtr, i);
      auto penalty = std::make_unique<QuadraticPenalty>(modelSettings_.footConstraintConfig.softConstraintWeight);
      problemPtr_->softConstraintPtr->add(zeroVelocityTermName(footName),
                                          std::make_unique<StateInputSoftConstraint>(std::move(stanceConstraint), std::move(penalty)));
    }

    // The relaxed complementarity conditions of rigid contact: with all three listed (and zero_wrench / zero_velocity
    // dropped, which checkMpcFormulationTasks enforces) the mode schedule no longer gates any contact constraint and
    // the solver decides where each foot carries load. See humanoid_nmpc/docs/contact_implicit_mpc/README.md.
    const ModelSettings::ContactImplicitConfig& contactImplicit = modelSettings_.contactImplicitConfig;
    // The force both residuals are measured in is the robot's own weight, read from the model rather than configured,
    // so it cannot fall out of step with the URDF. It is what makes the two weights mean the same thing on a 40 kg
    // robot and a 160 kg one.
    constexpr scalar_t kStandardGravity = 9.81;  // [m/s^2]
    const scalar_t forceReference = centroidalModelInfo_.robotMass * kStandardGravity;

    // Where the foot is, as far as contact is concerned: the CORNERS of the footprint, not the center of the sole.
    // createPinocchioModel() already adds a frame at each point of the contact polygon. Both the penetration hinge and
    // the complementarity product are built on this one object, so they cannot end up measuring different heights -
    // they used to, and a foot rocked onto its heel then read a positive height while carrying the whole robot.
    // Constraining the sole center alone would also leave the toe and the heel free to go through the floor, because
    // this formulation deliberately lets the foot rock (ForceWeightedSlipConstraint leaves the rocking rates free),
    // and on this robot the corners are 0.12 m fore and aft of the center.
    // The ground both terms are built on is the reference manager's, which owns it from here on and builds the swing
    // trajectories on it; the parameter updater (ContactImplicitApplier) keeps the terms on the ground it applies.
    const scalar_t terrainHeight = referenceManagerPtr_->getTerrainHeight();
    std::unique_ptr<FootprintCornerHeights> cornerHeightsPtr;
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kContactComplementarity) ||
        formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kGroundPenetration)) {
      ASSIGN_OR_RETURN(const ContactRectangle footprint, contactRectangleFromConfig(task.contacts, modelSettings_, static_cast<int>(i)));
      std::vector<std::string> cornerFrames;
      cornerFrames.reserve(footprint.getNumberOfContactPoints());
      for (size_t corner = 0; corner < footprint.getNumberOfContactPoints(); ++corner) {
        cornerFrames.push_back(footprint.getPolygonPointFrameName(static_cast<int>(corner)));
      }
      cornerHeightsPtr =
          std::make_unique<FootprintCornerHeights>(*pinocchioInterfacePtr_, *effectiveMpcRobotModelADPtr_, std::move(cornerFrames),
                                                   absl::StrCat(footName, "_footprintCorners"), modelSettings_);
    }

    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kContactComplementarity)) {
      std::unique_ptr<StateInputConstraint> complementarity =
          std::make_unique<ContactComplementarityConstraint>(*cornerHeightsPtr, *effectiveMpcRobotModelPtr_, i, terrainHeight,
                                                             forceReference, contactImplicit.heightReference, contactImplicit.gapSmoothing);
      auto penalty = std::make_unique<QuadraticPenalty>(contactImplicit.complementarityWeight);
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kContactComplementarity),
                                          std::make_unique<StateInputSoftConstraint>(std::move(complementarity), std::move(penalty)));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kForceWeightedSlip)) {
      // Not `&& eeKinematicsPtr`, which is how the neighboring terms guard themselves: this one is load bearing.
      // checkMpcFormulationTasks() refuses `contact_complementarity` without `force_weighted_slip`, because nothing
      // else holds a LOADED foot still once the schedule-gated zero-velocity constraint is gone. Silently skipping it
      // because `needsEeKinematics` above had drifted would defeat that guarantee and leave a foot carrying full body
      // weight free to slide.
      CHECK(eeKinematicsPtr != nullptr) << "[CentroidalMpcInterface] 'force_weighted_slip' needs the contact frame's kinematics; "
                                           "needsEeKinematics must list it.";
      std::unique_ptr<StateInputConstraint> slip =
          std::make_unique<ForceWeightedSlipConstraint>(*eeKinematicsPtr, *effectiveMpcRobotModelPtr_, i, forceReference,
                                                        contactImplicit.velocityReference, contactImplicit.angularVelocityReference);
      auto penalty = std::make_unique<QuadraticPenalty>(contactImplicit.slipWeight);
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kForceWeightedSlip),
                                          std::make_unique<StateInputSoftConstraint>(std::move(slip), std::move(penalty)));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kGroundPenetration)) {
      std::unique_ptr<StateConstraint> penetration = std::make_unique<GroundPenetrationConstraint>(*cornerHeightsPtr, terrainHeight);
      // A one-sided quadratic hinge, NOT a relaxed log barrier: the delta of 0 puts its zero exactly on the ground, so
      // the term is silent for a foot resting on the ground and only bites below it. A log barrier here pushed every
      // loaded foot into a hover; see ModelSettings::ContactImplicitConfig::penetrationWeight.
      auto penalty =
          std::make_unique<SquaredHingePenalty>(SquaredHingePenalty::Config(contactImplicit.penetrationWeight, /*deltaParam=*/0.0));
      problemPtr_->stateSoftConstraintPtr->add(contact_term::name(footName, contact_term::kGroundPenetration),
                                               std::make_unique<StateSoftConstraint>(std::move(penetration), std::move(penalty)));
    }

    if (formulationTasks.hasHardConstraint(MpcHardConstraintType::kZeroWrench)) {
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_zeroWrench"), factory.getZeroWrenchConstraint(i));
    }
    if (formulationTasks.hasHardConstraint(MpcHardConstraintType::kZeroVelocity) && eeKinematicsPtr) {
      problemPtr_->equalityConstraintPtr->add(zeroVelocityTermName(footName), getStanceFootConstraint(*eeKinematicsPtr, i));
    }
    if (formulationTasks.hasHardConstraint(MpcHardConstraintType::kNormalVelocity) && eeKinematicsPtr) {
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_normalVelocity"), getNormalVelocityConstraint(*eeKinematicsPtr, i));
    }
    // The same row, priced instead of imposed. Imposed, it fixes the whole height profile of a scheduled swing and the
    // solver can neither land early nor late; priced, it shapes the swing and is overruled whenever anything else pays
    // more, which is what the contact-implicit formulation needs of a reduced-order plan's guidance.
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kNormalVelocity) && eeKinematicsPtr) {
      auto penalty = std::make_unique<QuadraticPenalty>(modelSettings_.footConstraintConfig.normalVelocitySoftConstraintWeight);
      problemPtr_->softConstraintPtr->add(
          contact_term::name(footName, contact_term::kNormalVelocitySoft),
          std::make_unique<StateInputSoftConstraint>(getNormalVelocityConstraint(*eeKinematicsPtr, i), std::move(penalty)));
    }
    if (formulationTasks.hasHardConstraint(MpcHardConstraintType::kKneeJointMimic)) {
      const JointMimicSettings& mimic = kneeMimicJoints[i];
      ASSIGN_OR_RETURN(std::unique_ptr<JointMimicKinematicConstraint> kneeJointMimic,
                       JointMimicKinematicConstraint::Create(*effectiveMpcRobotModelPtr_, mimic.parentJointName, mimic.childJointName,
                                                             mimic.multiplier, mimic.positionGain));
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_kneeJointMimic"), std::move(kneeJointMimic));
    }

    if (formulationTasks.hasCost(MpcCostType::kTaskSpaceFootCost)) {
      // The collection name and the CppAD library name are the same string today, defined twice on purpose: a rename of
      // the term must not rename (and regenerate) the library.
      problemPtr_->costPtr->add(taskSpaceKinematicsCostName(footName),
                                std::make_unique<CentroidalMpcEndEffectorFootCost>(
                                    *referenceManagerPtr_, footCost.weights, *pinocchioInterfacePtr_, *effectiveMpcRobotModelADPtr_, i,
                                    CentroidalMpcEndEffectorFootCost::libraryName(footName), modelSettings_, footCost.activeInStance));
    }
    if (formulationTasks.hasCost(MpcCostType::kExternalTorqueCost)) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> externalTorqueCost, factory.makeExternalTorqueQuadraticCost(i));
      problemPtr_->costPtr->add(externalTorqueCostName(footName), std::move(externalTorqueCost));
    }
  }

  // Pre-computation
  problemPtr_->preComputationPtr = std::make_unique<HumanoidPreComputation>(
      *pinocchioInterfacePtr_, *referenceManagerPtr_->getSwingTrajectoryPlanner(), *effectiveMpcRobotModelPtr_);

  // Rollout
  rolloutPtr_ = std::make_unique<TimeTriggeredRollout>(*problemPtr_->dynamicsPtr, solverSettings_.rolloutSettings);

  // Initialization
  constexpr bool extendNormalizedMomentum = true;
  initializerPtr_ = std::make_unique<CentroidalWeightCompInitializer>(centroidalModelInfo_, *referenceManagerPtr_,
                                                                      *effectiveMpcRobotModelPtr_, extendNormalizedMomentum);

  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputConstraint> CentroidalMpcInterface::getStanceFootConstraint(const EndEffectorKinematics<scalar_t>& eeKinematics,
                                                                                      size_t contactPointIndex) {
  const ModelSettings::FootConstraintConfig& footConfig = modelSettings_.footConstraintConfig;
  const size_t numConstraints = footConfig.constrainOrientation ? 6 : 3;

  const std::function<EndEffectorKinematicsTwistConstraint::Config(const ModelSettings::FootConstraintConfig&)> eeZeroVelConConfig =
      [](const ModelSettings::FootConstraintConfig& footConfig) {
        EndEffectorKinematicsTwistConstraint::Config config;
        config.b.setZero(6);
        config.Ax.setZero(6, 6);
        config.Av.setZero(6, 6);

        // Position error gain: only z-axis (foot height tracking during stance)
        if (!numerics::almost_eq(footConfig.positionErrorGain_z, /*y=*/0.0)) {
          config.Ax(2, 2) = footConfig.positionErrorGain_z;
        }
        // Orientation error gain: all 3 rotation axes (only effective when constrainOrientation is true)
        if (!numerics::almost_eq(footConfig.orientationErrorGain, /*y=*/0.0)) {
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
  return std::make_unique<NormalVelocityConstraintCppAd>(*referenceManagerPtr_, eeKinematics, contactPointIndex);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
absl::Status CentroidalMpcInterface::addTaskSpaceKinematicsCosts(
    const PinocchioStateInputMapping<ad_scalar_t>& pinocchioMappingCppAd,
    const PinocchioEndEffectorKinematicsCppAd::update_pinocchio_interface_callback& velocityUpdateCallback) {
  // The cost task_space_torso_cost is the named link costs of task_space_costs; listed without any, it would build
  // nothing, so it is refused.
  if (config_.task.task_space_costs.empty()) {
    return absl::InvalidArgumentError(
        "[CentroidalMpcInterface] the costs list task_space_torso_cost, but task_space_costs names no link to track: add an entry "
        "task_space_costs { name: ... link_name: ... weights { ... } } or drop the cost.");
  }
  ASSIGN_OR_RETURN(const std::vector<TaskSpaceLinkCostSettings> linkCosts, taskSpaceLinkCostsFromConfig(config_.task.task_space_costs));
  for (const TaskSpaceLinkCostSettings& linkCost : linkCosts) {
    const std::string& linkName = linkCost.linkName;
    std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKinematicsPtr = std::make_unique<PinocchioEndEffectorKinematicsCppAd>(
        *pinocchioInterfacePtr_, pinocchioMappingCppAd, std::vector<std::string>{linkName}, centroidalModelInfo_.stateDim,
        effectiveMpcRobotModelPtr_->getInputDim(), velocityUpdateCallback, linkName, modelSettings_.modelFolderCppAd,
        modelSettings_.recompileLibrariesCppAd, modelSettings_.verboseCppAd);

    // With the reference manager, so that a torso that follows the base follows the heuristics' shaped base pose too.
    std::unique_ptr<StateInputCost> cost = std::make_unique<EndEffectorKinematicsQuadraticCost>(
        linkCost.weights, *pinocchioInterfacePtr_, *eeKinematicsPtr, *effectiveMpcRobotModelADPtr_, linkName, modelSettings_,
        referenceManagerPtr_.get());

    problemPtr_->costPtr->add(taskSpaceKinematicsCostName(linkCost.name), std::move(cost));

    LOG(INFO) << "Initialized Task Space Kinematics Cost for link: " << linkName;
  }
  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<std::string> CentroidalMpcInterface::getCostNames() const {
  std::vector<std::string> costNames;
  for (const std::pair<const std::string, size_t>& term : problemPtr_->costPtr->getTermNameMap()) {
    costNames.emplace_back(term.first);
  }
  return costNames;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<std::string> CentroidalMpcInterface::getTerminalCostNames() const {
  std::vector<std::string> terminalCostNames;
  for (const std::pair<const std::string, size_t>& term : problemPtr_->finalCostPtr->getTermNameMap()) {
    terminalCostNames.emplace_back(term.first);
  }
  return terminalCostNames;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<std::string> CentroidalMpcInterface::getStateSoftConstraintNames() const {
  std::vector<std::string> costNames;
  for (const std::pair<const std::string, size_t>& term : problemPtr_->stateSoftConstraintPtr->getTermNameMap()) {
    costNames.emplace_back(term.first);
  }
  return costNames;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<std::string> CentroidalMpcInterface::getSoftConstraintNames() const {
  std::vector<std::string> costNames;
  for (const std::pair<const std::string, size_t>& term : problemPtr_->softConstraintPtr->getTermNameMap()) {
    costNames.emplace_back(term.first);
  }
  return costNames;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<std::string> CentroidalMpcInterface::getEqualityConstraintNames() const {
  std::vector<std::string> costNames;
  for (const std::pair<const std::string, size_t>& term : problemPtr_->equalityConstraintPtr->getTermNameMap()) {
    costNames.emplace_back(term.first);
  }
  return costNames;
}

}  // namespace ocs2::humanoid
