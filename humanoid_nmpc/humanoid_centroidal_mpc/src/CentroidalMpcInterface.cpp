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

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

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
#include <humanoid_common_mpc/common/BasisInputsCostTransform.h>
#include <humanoid_common_mpc/common/ContactInputParameterization.h>
#include <humanoid_common_mpc/common/ContactTermNames.h>
#include <humanoid_common_mpc/common/MpcFormulationConfig.h>
#include <humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h>
#include <humanoid_common_mpc/constraint/ContactComplementarityConstraint.h>
#include <humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h>
#include <humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h>
#include <humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h>
#include <humanoid_common_mpc/constraint/GroundPenetrationConstraint.h>
#include <humanoid_common_mpc/contact/FootprintCornerHeights.h>
#include <humanoid_common_mpc/cost/ComAndAcomTrackingCost.h>
#include <humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h>
#include <humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h>
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
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
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
#include <boost/optional.hpp>
#include <boost/property_tree/ptree.hpp>

namespace ocs2::humanoid {

namespace {

/**
 * NotFound naming `path` when it does not exist; `what` says which of the interface's three inputs it is. Create() runs
 * it on all three before anything reads them: the model settings read the URDF as well as the task file, and a missing
 * file used to surface as whatever the first reader threw - or, after the model settings, as the std::invalid_argument
 * the constructor threw.
 */
absl::Status checkInputFileExists(absl::string_view what, const std::string& path) {
  if (!std::filesystem::exists(path)) {
    return absl::NotFoundError(absl::StrCat("[CentroidalMpcInterface] ", what, " not found: ", path));
  }
  LOG(INFO) << "[CentroidalMpcInterface] " << what << ": " << path;
  return absl::OkStatus();
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

CentroidalMpcInterface::CentroidalMpcInterface(const std::string& taskFile,
                                               const std::string& urdfFile,
                                               const std::string& referenceFile,
                                               bool verbose)
    : modelSettings_(taskFile, urdfFile, "centroidal_mpc_", verbose),
      taskFile_(taskFile),
      urdfFile_(urdfFile),
      referenceFile_(referenceFile),
      verbose_(verbose) {
  // The three files exist: Create() checked them before anything read them.
  // load setting from loading file
  ddpSettings_ = ddp::loadSettings(taskFile, "ddp", verbose_);
  mpcSettings_ = mpc::loadSettings(taskFile, "mpc", verbose_);
  rolloutSettings_ = rollout::loadSettings(taskFile, "rollout", verbose_);
  sqpSettings_ = sqp::loadSettings(taskFile, "multiple_shooting", verbose_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status CentroidalMpcInterface::setupModels() {
  // PinocchioInterface: a model whose actuated joints are not model_settings' in order is refused here, by name.
  ASSIGN_OR_RETURN(PinocchioInterface pinocchioInterface,
                   loadCustomPinocchioInterface(taskFile_, urdfFile_, modelSettings_, /*scaleTotalMass=*/false));
  pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));

  // CentroidalModelInfo
  centroidalModelInfo_ = centroidal_model::createCentroidalModelInfo(
      *pinocchioInterfacePtr_, centroidal_model::loadCentroidalType(taskFile_),
      centroidal_model::loadDefaultJointState(pinocchioInterfacePtr_->getModel().nq - 6, referenceFile_), modelSettings_.contactNames3DoF,
      modelSettings_.contactNames6DoF);

  LOG(INFO) << "centroidalModelInfo_.numSixDofContacts: " << centroidalModelInfo_.numSixDofContacts;
  for (int i = 0; i < centroidalModelInfo_.numSixDofContacts; i++) {
    LOG(INFO) << "frameIndices: " << centroidalModelInfo_.endEffectorFrameIndices[i];
  }

  // Setup Centroidal State Input Mapping
  mpcRobotModelPtr_.reset(new CentroidalMpcRobotModel<scalar_t>(modelSettings_, *pinocchioInterfacePtr_, centroidalModelInfo_));
  mpcRobotModelADPtr_.reset(
      new CentroidalMpcRobotModel<ad_scalar_t>(modelSettings_, (*pinocchioInterfacePtr_).toCppAd(), centroidalModelInfo_.toCppAd()));
  // The wrench-space models until setupContactInputParameterization() has read the task file's choice.
  effectiveMpcRobotModelPtr_ = mpcRobotModelPtr_.get();
  effectiveMpcRobotModelADPtr_ = mpcRobotModelADPtr_.get();

  // initial state
  initialState_.setZero(centroidalModelInfo_.stateDim);
  loadData::loadEigenMatrix(taskFile_, "initialState", initialState_);

  // The robot's pendulum length at its nominal posture: what a dcm_terminal_cost.comHeight or a contact_planning.yaml
  // shared.comHeight of 0 stands for.
  nominalComHeight_ =
      computeComHeightAboveFeet(mpcRobotModelPtr_->getGeneralizedCoordinates(initialState_), *pinocchioInterfacePtr_, *mpcRobotModelPtr_);
  LOG(INFO) << "[CentroidalMpcInterface] the model's center of mass is " << nominalComHeight_
            << " m above its feet at initialState (the pendulum length a comHeight of 0 stands for).";
  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> CentroidalMpcInterface::Create(const std::string& taskFile,
                                                                                       const std::string& urdfFile,
                                                                                       const std::string& referenceFile) {
  // The inputs exist before anything reads them; the constructor's model settings read the URDF as well.
  RETURN_IF_ERROR(checkInputFileExists("task file", taskFile));
  RETURN_IF_ERROR(checkInputFileExists("URDF file", urdfFile));
  RETURN_IF_ERROR(checkInputFileExists("reference file", referenceFile));
  // The task file's interface.verbose first: it decides the logging of the model settings the constructor loads.
  ASSIGN_OR_RETURN(const bool verbose, ModelSettings::loadInterfaceVerbose(taskFile));
  std::unique_ptr<CentroidalMpcInterface> interface(new CentroidalMpcInterface(taskFile, urdfFile, referenceFile, verbose));
  // In this order: the robot models first; the reference manager and the contact planner are built on the effective
  // robot model, which the contact input parameterization decides, and the problem on all three.
  RETURN_IF_ERROR(interface->setupModels());
  RETURN_IF_ERROR(interface->setupContactInputParameterization());
  RETURN_IF_ERROR(interface->setupReferenceManager());
  RETURN_IF_ERROR(interface->setupOptimalControlProblem());
  return interface;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

namespace {

/**
 * Reads the optional scalar `key` of `pt` into `value`, leaving it untouched when the key is absent. A value that does
 * not parse as a T is an InvalidArgument naming the key, where loadData::loadPtreeValue would throw.
 */
template <typename T>
absl::Status loadOptionalTaskFileValue(const boost::property_tree::ptree& pt, absl::string_view key, T& value, bool verbose) {
  const boost::optional<const boost::property_tree::ptree&> child = pt.get_child_optional(std::string(key));
  if (!child) {
    if (verbose) {
      LOG(INFO) << "[CentroidalMpcInterface] " << key << " = " << value << " (default)";
    }
    return absl::OkStatus();
  }
  const boost::optional<T> parsed = child->get_value_optional<T>();
  if (!parsed) {
    return absl::InvalidArgumentError(
        absl::StrCat("[CentroidalMpcInterface] ", key, " is '", child->data(), "', which is not a value of the expected type."));
  }
  value = *parsed;
  if (verbose) {
    LOG(INFO) << "[CentroidalMpcInterface] " << key << " = " << value;
  }
  return absl::OkStatus();
}

/**
 * Whether the horizontal position of a swing foot's landing target reaches the problem: task_space_foot_cost is listed
 * and task_space_foot_cost_weights gives pos_x or pos_y a non-zero weight. A foothold - planned by the contact planner
 * or computed by a locomotion heuristic - reaches the MPC only through that cost (CentroidalMpcEndEffectorFootCost),
 * which multiplies its xy residual by exactly these two weights.
 */
bool footPositionIsTracked(const std::string& taskFile, const MpcFormulationTasks& formulationTasks) {
  if (!formulationTasks.hasCost(MpcCostType::TaskSpaceFootCost)) return false;
  const EndEffectorKinematicsWeights footWeights =
      EndEffectorKinematicsWeights::getWeights(taskFile, "task_space_foot_cost_weights.", /*verbose=*/false);
  return footWeights.contactPositionErrorWeight(0) != 0.0 || footWeights.contactPositionErrorWeight(1) != 0.0;
}

}  // namespace

absl::Status CentroidalMpcInterface::setupContactInputParameterization() {
  ASSIGN_OR_RETURN(contactInputParameterization_, loadContactInputParameterization(taskFile_));
  LOG(INFO) << "[CentroidalMpcInterface] " << kContactInputParameterizationKey << ": "
            << contactInputParameterizationName(contactInputParameterization_);

  if (contactInputParameterization_ == ContactInputParameterization::kWrench) {
    effectiveMpcRobotModelPtr_ = mpcRobotModelPtr_.get();
    effectiveMpcRobotModelADPtr_ = mpcRobotModelADPtr_.get();
    keyCppAdModelFolder(kWrenchInputsLibraryKey);
    return absl::OkStatus();
  }

  // The bases of both feet: the generator set named by contacts.basisGeneratorSet, built from the
  // contacts.contactWrenchConeSoftConstraint block (every key required) and each foot's contact rectangle. The set fixes
  // the input dimension, so it is read here and never reloaded.
  ASSIGN_OR_RETURN(const feet_array_t<ContactWrenchConeBasisMatrix> basisMatrices,
                   loadContactWrenchConeBases(taskFile_, modelSettings_, verbose_));
  basisGeneratorSet_ = basisMatrices[0].generatorSet();
  ASSIGN_OR_RETURN(const ContactWrenchConeConstraint::Config coneConfig, ContactWrenchConeConstraint::loadConfig(taskFile_));

  boost::property_tree::ptree pt;
  loadData::readPropertyTree(taskFile_, pt);

  // The regularization of the lambda block of the input cost: M has a non-trivial null space, so M^T R M alone leaves
  // the Hessian of the scalings singular. Its weight and its shape S are both read here; the online parameter updater
  // reads the same two keys on a hot reload.
  // LINT.IfChange(basis_regularization_yaml_path)
  constexpr scalar_t kDefaultBasisScalingRegularization = 1e-4;
  basisScalingRegularization_ = kDefaultBasisScalingRegularization;
  RETURN_IF_ERROR(loadOptionalTaskFileValue(pt, kBasisScalingRegularizationKey, basisScalingRegularization_, verbose_));
  basisRegularization_ = std::string(kDefaultBasisRegularization);
  RETURN_IF_ERROR(loadOptionalTaskFileValue(pt, kBasisRegularizationKey, basisRegularization_, verbose_));
  // clang-format off
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:basis_regularization_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:basis_regularization_config)
  // clang-format on

  // A conic combination is homogeneous, so the basis can represent neither the minimum normal force nor a gripper
  // adhesion force: both are affine offsets of the cone and lambda = 0 always yields the zero wrench.
  if (coneConfig.minNormalForce > 0.0 || coneConfig.gripperForce > 0.0) {
    LOG(WARNING) << "[CentroidalMpcInterface] " << ContactWrenchConeConstraint::kConfigBlock << ".minNormalForce ("
                 << coneConfig.minNormalForce << " N) and gripperForce (" << coneConfig.gripperForce << " N) are NOT enforced with "
                 << kContactInputParameterizationKey << ": " << kBasisVectorsContactInputParameterization
                 << ". Every generator lies inside the cone, so its friction, center-of-pressure and torsional limits are never "
                    "exceeded; these two affine offsets cannot be represented by a conic combination at all.";
  }

  const size_t numBasisPerFoot = basisMatrices[0].numBasis();
  LOG(INFO) << "[CentroidalMpcInterface] basis-vector contact inputs, generator set '" << basisGeneratorSet_ << "': " << numBasisPerFoot
            << " basis vectors per foot (total input dim: " << numBasisPerFoot * N_CONTACTS + modelSettings_.mpc_joint_dim
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
  RETURN_IF_ERROR(validateBasisInputsCostTransformConfig(*getBasisInputsCostTransformConfig()));

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
  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(
      new SwingTrajectoryPlanner(loadSwingTrajectorySettings(taskFile_, "swing_trajectory_config", verbose_), N_CONTACTS));

  // Where the mode schedule and the footholds come from, by name; the retired useContactPlanning boolean is refused here.
  ASSIGN_OR_RETURN(contactScheduleSource_, loadContactScheduleSource(taskFile_));
  LOG(INFO) << "[CentroidalMpcInterface] " << kContactScheduleSourceKey << ": " << contactScheduleSourceName(contactScheduleSource_);

  if (contactScheduleSource_ == ContactScheduleSource::kGaitSchedule) {
    referenceManagerPtr_ = std::make_shared<SwitchedModelReferenceManager>(
        GaitSchedule::loadGaitSchedule(referenceFile_, modelSettings_, verbose_), std::move(swingTrajectoryPlanner),
        *pinocchioInterfacePtr_, *effectiveMpcRobotModelPtr_);
    return absl::OkStatus();
  }

  // contact_planner: online contact planning (the planner planner.type in contact_planning.yaml names) replaces the
  // periodic gait schedule. The gait schedule is still loaded: it is used until the first plan arrives and whenever the
  // planner has no valid plan.
  const std::string contactPlanningFile = resolveContactPlanningConfigFile(taskFile_);
  LOG(INFO) << "[CentroidalMpcInterface] Loading contact planning config from " << contactPlanningFile;
  // Not validated yet: the values a file may leave at 0 to mean "from the model" are filled in below.
  ASSIGN_OR_RETURN(ContactPlanningConfig contactPlanningConfig,
                   loadContactPlanningConfigStatus(contactPlanningFile, "contact_planning.", verbose_, /*validate=*/false));
  const scalar_t fileComHeight = contactPlanningConfig.shared.comHeight;

  // Parameters left at 0 in the task file are derived from the robot model and from the ground parameters of the wrench
  // cone, so that the planner never assumes more friction, torque or footprint than the whole-body constraints allow.
  // The initial state of the task file is the nominal posture for the LIP height, the one every LIP consumer shares.
  // The same loader as the wrench cone and the basis generators: a task file without the block is refused rather than
  // planned against the library default of mu 0.7.
  ASSIGN_OR_RETURN(const ContactWrenchConeConstraint::Config coneConfig,
                   ContactWrenchConeConstraint::loadConfig(taskFile_, /*verbose=*/false));
  ContactPlanningGroundParameters ground;
  ground.frictionCoefficient = coneConfig.frictionCoefficient;
  ground.torsionalFrictionCoefficient = coneConfig.torsionalFrictionCoefficient;
  try {
    const ContactRectangle footprint = ContactRectangle::loadContactRectangle(taskFile_, modelSettings_, /*contactIndex=*/0, verbose_);
    ground.footprintHalfLengthX = 0.5 * (footprint.getBounds().x_max - footprint.getBounds().x_min);
    ground.footprintHalfWidthY = 0.5 * (footprint.getBounds().y_max - footprint.getBounds().y_min);
  } catch (const std::exception& e) {
    LOG(WARNING) << "[CentroidalMpcInterface] no footprint for the contact planner's ZMP box: " << e.what();
  }
  contactPlanningModelParameters_ = deriveContactPlanningModelParameters(
      *pinocchioInterfacePtr_, *effectiveMpcRobotModelPtr_, initialState_, modelSettings_.contactParentJointNames, ground,
      contactPlanningConfig.shared.gravity, contactPlanningConfig.stepWidth.nominalStepWidth);
  contactPlanningModelParameters_->applyTo(contactPlanningConfig);
  LOG(INFO) << "[CentroidalMpcInterface] contact planner model parameters: " << contactPlanningModelParameters_->summary();
  RETURN_IF_ERROR(contactPlanningConfig.validateStatus());
  LOG(INFO) << "[CentroidalMpcInterface] contact planner pendulum: shared.comHeight " << contactPlanningConfig.shared.comHeight << " m"
            << (fileComHeight > 0.0 ? " (explicit)" : " (the model's center of mass above its feet at initialState)") << ", omega "
            << contactPlanningConfig.omega() << " rad/s.";
  if (contactPlanningConfig.horizon() < mpcSettings_.timeHorizon_) {
    LOG(WARNING) << "[CentroidalMpcInterface] contact_planning horizon (" << contactPlanningConfig.horizon()
                 << " s) is shorter than the MPC horizon (" << mpcSettings_.timeHorizon_
                 << " s); the schedule beyond the planned horizon defaults to double support.";
  }
  ASSIGN_OR_RETURN(std::shared_ptr<ContactPlanningReferenceManager> planningReferenceManager,
                   ContactPlanningReferenceManager::Create(GaitSchedule::loadGaitSchedule(referenceFile_, modelSettings_, verbose_),
                                                           std::move(swingTrajectoryPlanner), *pinocchioInterfacePtr_,
                                                           *effectiveMpcRobotModelPtr_, contactPlanningConfig));
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
  ASSIGN_OR_RETURN(const LocomotionHeuristicConfig heuristicConfig, loadLocomotionHeuristicConfig(taskFile_, verbose_));
  ASSIGN_OR_RETURN(const LocomotionHeuristicModelParameters heuristicModel,
                   deriveLocomotionHeuristicModelParameters(*pinocchioInterfacePtr_, *effectiveMpcRobotModelPtr_, initialState_));
  // What the rest of this task file says that decides whether a listed heuristic can act at all; see
  // LocomotionHeuristicEnvironment.
  LocomotionHeuristicEnvironment environment;
  environment.usesContactPlanning = contactScheduleSource_ == ContactScheduleSource::kContactPlanner;
  environment.usesContactBasisVectorInputs = usesContactBasisVectorInputs();
  environment.nominalStepWidth = modelSettings_.nominalFootholdConfig.stepWidth;
  ASSIGN_OR_RETURN(const MpcFormulationTasks formulationTasks, loadMpcFormulationTasks(taskFile_, /*verbose=*/false));
  environment.listsComAndAcomTrackingCost = formulationTasks.hasCost(MpcCostType::ComAndAcomTrackingCost);
  environment.footPositionIsUntracked = !footPositionIsTracked(taskFile_, formulationTasks);
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
  ASSIGN_OR_RETURN(const MpcFormulationTasks formulationTasks, loadMpcFormulationTasks(taskFile_, verbose_));
  const bool scheduleGatedContactConstraints = contactConstraintsAreScheduleGated(formulationTasks);

  // The contact_implicit block, before any term is built. A key nothing reads - a renamed or misspelled one - is refused
  // whether or not the formulation is listed, because the term it was meant for would otherwise run on its default and
  // its tuning slider reach nothing. The values are divisors and penalty weights: refused with a Status naming the key,
  // rather than CHECK-aborting inside a term constructor or turning a penalty into a reward, whenever the formulation
  // that reads them is listed.
  {
    boost::property_tree::ptree taskTree;
    loadData::readPropertyTree(taskFile_, taskTree);
    RETURN_IF_ERROR(checkContactImplicitBlockKeys(taskTree));
  }
  if (usesContactImplicitFormulation(formulationTasks)) {
    RETURN_IF_ERROR(validateContactImplicitConfig(modelSettings_.contactImplicitConfig));
  }

  // The planner's footholds are its whole stabilizing mechanism - the H-LIP deadbeat step is a foot placement - and they
  // reach this problem only through the foot cost's xy weights. At 0 the planner runs, prints its formulation and
  // re-times the schedule while the MPC places the swing foot wherever the other costs prefer; say so rather than let
  // it look like a planner that does not work.
  if (contactScheduleSource_ == ContactScheduleSource::kContactPlanner && !footPositionIsTracked(taskFile_, formulationTasks)) {
    LOG(WARNING) << "[CentroidalMpcInterface] " << kContactScheduleSourceKey << " is " << kContactPlannerContactScheduleSource
                 << ", but the planned footholds carry no weight: they reach the MPC only through task_space_foot_cost, "
                 << (formulationTasks.hasCost(MpcCostType::TaskSpaceFootCost)
                         ? "whose task_space_foot_cost_weights.pos_x and pos_y are both 0. Raise task_space_foot_cost_weights.pos_x / "
                           "pos_y"
                         : "which `costs` does not list. List task_space_foot_cost and give task_space_foot_cost_weights.pos_x / pos_y "
                           "a weight")
                 << " before relying on the planner's foot placement.";
  }

  // The parameters of the lambda >= 0 barrier that the basis-vector parameterization builds for every contact below.
  // They are read here, before any CppAD model is built, so that a value that is not a number is refused by its key at
  // once rather than thrown out of Create() minutes later; a file without the section gets the defaults.
  constexpr scalar_t kDefaultLambdaBarrierMu = 1e-2;
  constexpr scalar_t kDefaultLambdaBarrierDelta = 1e-3;
  PieceWisePolynomialBarrierPenalty::Config lambdaBarrierConfig(kDefaultLambdaBarrierMu, kDefaultLambdaBarrierDelta);
  // The `mu` of the wrench parameterization's cone, which the un-gated barrier is compared against below.
  scalar_t wrenchConeBarrierMu = 0.0;
  if (usesContactBasisVectorInputs()) {
    boost::property_tree::ptree barrierPt;
    loadData::readPropertyTree(taskFile_, barrierPt);
    // LINT.IfChange(basis_barrier_yaml_path)
    const std::string barrierPrefix = "contacts.basisNonNegativityBarrier.";
    // clang-format off
    // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:basis_barrier_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:basis_barrier_config)
    // clang-format on
    RETURN_IF_ERROR(loadOptionalTaskFileValue(barrierPt, absl::StrCat(barrierPrefix, "mu"), lambdaBarrierConfig.mu, verbose_));
    RETURN_IF_ERROR(loadOptionalTaskFileValue(barrierPt, absl::StrCat(barrierPrefix, "delta"), lambdaBarrierConfig.delta, verbose_));
    RETURN_IF_ERROR(loadOptionalTaskFileValue(barrierPt, absl::StrCat(ContactWrenchConeConstraint::kConfigBlock, ".mu"),
                                              wrenchConeBarrierMu, /*verbose=*/false));
  }

  // Under online contact planning with the heading model, the planner reads the whole-body heading off the robot's ACoM
  // network. A robot without one falls back to the base yaw with a warning; a network trained on other joints is an
  // error, because it would run on the wrong joint vector.
  const std::shared_ptr<ContactPlanningReferenceManager> planningReferenceManager =
      std::dynamic_pointer_cast<ContactPlanningReferenceManager>(referenceManagerPtr_);
  if (planningReferenceManager != nullptr) {
    RETURN_IF_ERROR(planningReferenceManager->loadHeadingModelEvaluator());
  }

  // CoM + ACoM tracking is a cost of the `costs` list, and everything it implies follows from that one entry: the
  // ComAndAcomTrackingCost itself, the zeroed base-pose blocks of Q and Q_final (the factory), the terminal instance
  // beside terminal_cost, and the procedural arm swing switched off - the arm motion emerges from the ACoM cost, and a
  // generator driving the same joints would fight it. A legs-only robot omits model_settings.armJointNames and has no
  // arm to swing either way.
  const bool comAndAcomTracking = formulationTasks.hasCost(MpcCostType::ComAndAcomTrackingCost);
  referenceManagerPtr_->setArmSwingReferenceActive(modelSettings_.hasArmSwingJoints && !comAndAcomTracking);

  HumanoidCostConstraintFactory factory =
      HumanoidCostConstraintFactory(taskFile_, referenceFile_, *referenceManagerPtr_, *pinocchioInterfacePtr_, *effectiveMpcRobotModelPtr_,
                                    *effectiveMpcRobotModelADPtr_, modelSettings_, verbose_, scheduleGatedContactConstraints);
  factory.setComAndAcomTrackingCostListed(comAndAcomTracking);

  // Optimal control problem
  problemPtr_.reset(new OptimalControlProblem);

  // Dynamics
  std::unique_ptr<SystemDynamicsBase> dynamicsPtr;
  const std::string modelName = "dynamics";
  if (usesContactBasisVectorInputs()) {
    // R is written in wrench dimensions and transformed into basis space, R_basis = M^T R_wrench M + reg * blkdiag(S, 0),
    // through the same config the online parameter updater applies on a hot reload.
    const BasisInputsCostTransformConfig basisCostTransform = *getBasisInputsCostTransformConfig();
    if (formulationTasks.hasCost(MpcCostType::InputQuadraticCost) || formulationTasks.hasCost(MpcCostType::StateInputQuadraticCost)) {
      // The QP needs a unique input: the lambda block of R_basis has to be positive definite, which a zero weight, or
      // null_space with a contact wrench direction R does not weigh, would break.
      matrix_t R_wrench(basisCostTransform.wrenchInputDim, basisCostTransform.wrenchInputDim);
      loadData::loadEigenMatrix(taskFile_, "R", R_wrench);
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
    dynamicsPtr.reset(new CentroidalDynamicsAD(*pinocchioInterfacePtr_, centroidalModelInfo_, modelName, modelSettings_));
  }
  problemPtr_->dynamicsPtr = std::move(dynamicsPtr);

  // Cost terms
  if (formulationTasks.hasCost(MpcCostType::StateInputQuadraticCost)) {
    problemPtr_->costPtr->add("stateInputQuadraticCost", factory.getStateInputQuadraticCost());
  }
  if (formulationTasks.hasCost(MpcCostType::StateQuadraticCost)) {
    problemPtr_->costPtr->add("stateQuadraticCost", factory.getStateQuadraticCost());
  }
  // Beside whichever quadratic state cost is listed, whose base-pose block the factory has zeroed - or on its own.
  if (comAndAcomTracking) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateCost> comAndAcomTrackingCost, factory.getComAndAcomTrackingCost(centroidalModelInfo_));
    problemPtr_->stateCostPtr->add(std::string(ComAndAcomTrackingCost::kRunningTermName), std::move(comAndAcomTrackingCost));
  }
  if (formulationTasks.hasCost(MpcCostType::InputQuadraticCost)) {
    problemPtr_->costPtr->add("inputQuadraticCost", factory.getInputQuadraticCost());
  }
  // Terminal cost: the DCM viability cost (dcm_terminal_cost) or the quadratic Q_final cost (terminal_cost), whichever
  // the costs list names; the loader refuses a list that names both, so the two branches below are exclusive.
  if (formulationTasks.hasCost(MpcCostType::DcmTerminalCost)) {
    // A dcm_terminal_cost.comHeight of 0 is the model's pendulum length, resolved here and, through the cost's own
    // setConfig(), on every hot reload of the block.
    ASSIGN_OR_RETURN(const DcmTerminalCost::Config dcmConfig,
                     DcmTerminalCost::loadConfig(taskFile_, DcmTerminalCost::kConfigPrefix, verbose_));
    ASSIGN_OR_RETURN(std::unique_ptr<DcmTerminalCost> dcmTerminalCost,
                     DcmTerminalCost::Create(*referenceManagerPtr_, dcmConfig, nominalComHeight_, *pinocchioInterfacePtr_,
                                             *effectiveMpcRobotModelADPtr_, DcmTerminalCost::kTermName, modelSettings_));
    LOG(INFO) << "[CentroidalMpcInterface] DCM terminal cost pendulum: " << DcmTerminalCost::kConfigPrefix << "comHeight "
              << dcmTerminalCost->getConfig().comHeight << " m"
              << (dcmConfig.comHeight > 0.0 ? " (explicit)" : " (the model's center of mass above its feet at initialState)") << ", omega "
              << dcmTerminalCost->getConfig().omega() << " rad/s.";
    problemPtr_->finalCostPtr->add(DcmTerminalCost::kTermName, std::move(dcmTerminalCost));
  } else if (formulationTasks.hasCost(MpcCostType::TerminalCost)) {
    // LINT.IfChange(quadratic_terminal_cost_term)
    problemPtr_->finalCostPtr->add("terminalCost", factory.getTerminalCost());
    // LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/src/mrt/MpcParameterUpdaterModule.cpp:terminal_cost_term_names)
    // Q_final's base-pose block is zeroed under com_and_acom_tracking_cost; this is what regulates the terminal CoM,
    // height and orientation in its place, in the coordinates every other node uses.
    if (comAndAcomTracking) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateCost> terminalComAndAcomTrackingCost,
                       factory.getTerminalComAndAcomTrackingCost(centroidalModelInfo_));
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

    // The kinematics of the CONTACT FRAME, i.e. the center of the sole. `ground_penetration` is deliberately absent
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

    if (usesContactBasisVectorInputs()) {
      // In basis-vector mode the friction, CoP and torsional limits are enforced structurally: every generator lies
      // inside the cone (ContactWrenchConeBasisMatrix verifies this against the constraint's own rows at construction)
      // and the cone is convex, so λ ≥ 0 ⟹ W ∈ cone. The ContactWrenchConeConstraint is wrench-space specific and cannot
      // operate on basis-vector inputs, so a listed `contact_wrench_cone` builds nothing here. The minimum normal force
      // and the gripper force are the exception and are not enforced; a warning is logged where the basis is built.
      if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactWrenchCone)) {
        LOG(INFO) << "[CentroidalMPC] Skipping contact_wrench_cone soft constraint for " << footName
                  << " (friction, CoP and torsional limits enforced structurally via basis-vector inputs).";
      }

      // The λ ≥ 0 barrier IS the contact wrench cone of this parameterization, so it is built for every contact whatever
      // the soft-constraint lists say, gated exactly as `zero_wrench` dictates. Nothing else bounds the individual
      // scalings - `friction_force_cone` bounds only the assembled force - and without it a negative scaling is
      // adhesion, a center of pressure outside the sole or unbounded torsion. (It used to be built only when
      // `contact_wrench_cone` was listed, so dropping that entry silently removed the cone.) Its parameters,
      // contacts.basisNonNegativityBarrier, were read before the problem was built (lambdaBarrierConfig).

      // WHAT `mu` IS NOW BEING ASKED TO DO. Gated, this term is a redundant bound: the swing foot's scalings are
      // pinned to zero by `zero_wrench` and the stance foot's are pulled positive by R's weight-compensating
      // nominal, so 0.01 was tuned as a regularizer. Un-gated it becomes the ONLY thing holding every contact
      // wrench inside its cone, at every node, which is the job `contactWrenchConeSoftConstraint.mu` does in the
      // wrench parameterization - and there it is 0.2. A scaling pair (+a, -a) costs only `mu * a^2` and leaves the
      // load indicator f_n = sum(lambda) at zero, so both contact-implicit products stay blind to it.
      //
      // Raising it is a closed-loop tuning decision and not one this loader may take on the operator's behalf, so
      // the mismatch is reported rather than patched. See humanoid_nmpc/docs/contact_implicit_mpc/README.md.
      if (!scheduleGatedContactConstraints && lambdaBarrierConfig.mu < wrenchConeBarrierMu) {
        LOG(WARNING) << "[CentroidalMpcInterface] " << footName << ": contacts.basisNonNegativityBarrier.mu = " << lambdaBarrierConfig.mu
                     << " is the ONLY bound on this contact's wrench once the schedule gate is off, and it is softer than the "
                     << ContactWrenchConeConstraint::kConfigBlock << ".mu = " << wrenchConeBarrierMu
                     << " that does the same job in the wrench parameterization. Tune it against a foot in flight before "
                     << "trusting the contact-implicit formulation on hardware.";
      }

      const size_t lambdaStartIdx = basisDecoratorPtr_->getContactWrenchStartIndices(i);
      const size_t numBasis = basisDecoratorPtr_->getNumBasisPerFoot();

      problemPtr_->costPtr->add(absl::StrCat(footName, "_basisNonNegativity"), std::make_unique<BasisScalingNonNegativityConstraint>(
                                                                                   *referenceManagerPtr_, i, lambdaStartIdx, numBasis,
                                                                                   lambdaBarrierConfig, scheduleGatedContactConstraints));

      LOG(INFO) << "[CentroidalMPC] Added λ ≥ 0 non-negativity barrier for " << footName << " (" << numBasis << " basis vectors, start idx "
                << lambdaStartIdx << ").";
    } else if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactWrenchCone)) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> contactWrenchCone, factory.getContactWrenchConeConstraint(i));
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kContactWrenchCone), std::move(contactWrenchCone));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::FrictionForceCone)) {
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kFrictionForceCone),
                                          factory.getFrictionForceConeConstraint(i));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactMomentXY)) {
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kContactMomentXY),
                                          factory.getContactMomentXYConstraint(i, absl::StrCat(footName, "_contact_moment_XY_constraint")));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ZeroVelocity) && eeKinematicsPtr) {
      std::unique_ptr<StateInputConstraint> stanceConstraint = getStanceFootConstraint(*eeKinematicsPtr, i);
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

    // Where the foot is, as far as contact is concerned: the CORNERS of the footprint, not the center of the sole.
    // createPinocchioModel() already adds a frame at each point of the contact polygon. Both the penetration hinge and
    // the complementarity product are built on this one object, so they cannot end up measuring different heights -
    // they used to, and a foot rocked onto its heel then read a positive height while carrying the whole robot.
    // Constraining the sole center alone would also leave the toe and the heel free to go through the floor, because
    // this formulation deliberately lets the foot rock (ForceWeightedSlipConstraint leaves the rocking rates free),
    // and on this robot the corners are 0.12 m fore and aft of the center.
    // The ground both terms are built on is the reference manager's, which owns it from here on and builds the swing
    // trajectories on it; MpcParameterUpdaterModule keeps the terms on the ground it applies.
    const scalar_t terrainHeight = referenceManagerPtr_->getTerrainHeight();
    std::unique_ptr<FootprintCornerHeights> cornerHeightsPtr;
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactComplementarity) ||
        formulationTasks.hasSoftConstraint(MpcSoftConstraintType::GroundPenetration)) {
      const ContactRectangle footprint =
          ContactRectangle::loadContactRectangle(taskFile_, modelSettings_, static_cast<int>(i), /*verbose=*/false);
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
      std::unique_ptr<StateInputConstraint> complementarity =
          std::make_unique<ContactComplementarityConstraint>(*cornerHeightsPtr, *effectiveMpcRobotModelPtr_, i, terrainHeight,
                                                             forceReference, contactImplicit.heightReference, contactImplicit.gapSmoothing);
      auto penalty = std::make_unique<QuadraticPenalty>(contactImplicit.complementarityWeight);
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kContactComplementarity),
                                          std::make_unique<StateInputSoftConstraint>(std::move(complementarity), std::move(penalty)));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ForceWeightedSlip)) {
      // Not `&& eeKinematicsPtr`, which is how the neighboring terms guard themselves: this one is load bearing.
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
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kForceWeightedSlip),
                                          std::make_unique<StateInputSoftConstraint>(std::move(slip), std::move(penalty)));
    }
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::GroundPenetration)) {
      std::unique_ptr<StateConstraint> penetration = std::make_unique<GroundPenetrationConstraint>(*cornerHeightsPtr, terrainHeight);
      // A one-sided quadratic hinge, NOT a relaxed log barrier: the delta of 0 puts its zero exactly on the ground, so
      // the term is silent for a foot resting on the ground and only bites below it. A log barrier here pushed every
      // loaded foot into a hover; see ModelSettings::ContactImplicitConfig::penetrationWeight.
      auto penalty =
          std::make_unique<SquaredHingePenalty>(SquaredHingePenalty::Config(contactImplicit.penetrationWeight, /*deltaParam=*/0.0));
      problemPtr_->stateSoftConstraintPtr->add(contact_term::name(footName, contact_term::kGroundPenetration),
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
          contact_term::name(footName, contact_term::kNormalVelocitySoft),
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

  for (const boost::property_tree::ptree::value_type& taskSpaceCost : task_space_costs_pt) {
    const std::string& costName = taskSpaceCost.first;
    std::string linkName;

    loadData::loadPtreeValue(task_space_costs_pt, linkName, absl::StrCat(costName, ".link_name"), verbose_);

    std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKinematicsPtr;

    eeKinematicsPtr.reset(new PinocchioEndEffectorKinematicsCppAd(*pinocchioInterfacePtr_, pinocchioMappingCppAd, {linkName},
                                                                  centroidalModelInfo_.stateDim, effectiveMpcRobotModelPtr_->getInputDim(),
                                                                  velocityUpdateCallback, linkName, modelSettings_.modelFolderCppAd,
                                                                  modelSettings_.recompileLibrariesCppAd, modelSettings_.verboseCppAd));

    EndEffectorKinematicsWeights weights =
        EndEffectorKinematicsWeights::getWeights(taskFile_, absl::StrCat("task_space_costs.", costName, ".weights."), verbose_);

    // With the reference manager, so that a torso that follows the base follows the heuristics' shaped base pose too.
    std::unique_ptr<StateInputCost> cost = std::make_unique<EndEffectorKinematicsQuadraticCost>(
        weights, *pinocchioInterfacePtr_, *eeKinematicsPtr, *effectiveMpcRobotModelADPtr_, linkName, modelSettings_,
        referenceManagerPtr_.get());

    problemPtr_->costPtr->add(absl::StrCat(costName, "_TaskSpaceKinematicsCost"), std::move(cost));

    LOG(INFO) << "Initialized Task Space Kinematics Cost for link: " << linkName;
  }
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
