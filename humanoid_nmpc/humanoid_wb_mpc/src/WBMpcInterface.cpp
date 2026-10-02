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

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

// Pinocchio forward declarations must be included first
#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_wb_mpc/WBMpcInterface.h"

#include <ocs2_core/misc/Display.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/Numerics.h>
#include <ocs2_core/misc/PropertyTree.h>
#include <ocs2_core/penalties/Penalties.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>

#include <humanoid_common_mpc/pinocchio_model/createPinocchioModel.h>
#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
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
    return absl::NotFoundError(absl::StrCat("[WBMpcInterface] ", what, " not found: ", path));
  }
  LOG(INFO) << "[WBMpcInterface] " << what << ": " << path;
  return absl::OkStatus();
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
WBMpcInterface::WBMpcInterface(const std::string& taskFile, const std::string& urdfFile, const std::string& referenceFile, bool verbose)
    : modelSettings_(taskFile, urdfFile, "wb_mpc_", verbose),
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
absl::Status WBMpcInterface::setupModels() {
  // PinocchioInterface: a model whose actuated joints are not model_settings' in order is refused here, by name.
  ASSIGN_OR_RETURN(PinocchioInterface pinocchioInterface, loadCustomPinocchioInterface(taskFile_, urdfFile_, modelSettings_));
  pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));

  // Setup WB State Input Mapping
  mpcRobotModelPtr_.reset(new WBAccelMpcRobotModel<scalar_t>(modelSettings_));
  mpcRobotModelADPtr_.reset(new WBAccelMpcRobotModel<ad_scalar_t>(modelSettings_));

  // Swing trajectory planner
  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(
      new SwingTrajectoryPlanner(loadSwingTrajectorySettings(taskFile_, "swing_trajectory_config", verbose_), N_CONTACTS));

  // Mode schedule manager
  referenceManagerPtr_ =
      std::make_shared<SwitchedModelReferenceManager>(GaitSchedule::loadGaitSchedule(referenceFile_, modelSettings_, verbose_),
                                                      std::move(swingTrajectoryPlanner), *pinocchioInterfacePtr_, *mpcRobotModelPtr_);
  // A legs-only robot omits model_settings.armJointNames and has no arm to swing.
  referenceManagerPtr_->setArmSwingReferenceActive(modelSettings_.hasArmSwingJoints);

  // initial state
  initialState_.setZero(mpcRobotModelPtr_->getStateDim());
  loadData::loadEigenMatrix(taskFile_, "initialState", initialState_);
  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::StatusOr<std::unique_ptr<WBMpcInterface>> WBMpcInterface::Create(const std::string& taskFile,
                                                                       const std::string& urdfFile,
                                                                       const std::string& referenceFile) {
  // The inputs exist before anything reads them; the constructor's model settings read the URDF as well.
  RETURN_IF_ERROR(checkInputFileExists("task file", taskFile));
  RETURN_IF_ERROR(checkInputFileExists("URDF file", urdfFile));
  RETURN_IF_ERROR(checkInputFileExists("reference file", referenceFile));
  // The task file's interface.verbose first: it decides the logging of the model settings the constructor loads.
  ASSIGN_OR_RETURN(const bool verbose, ModelSettings::loadInterfaceVerbose(taskFile));
  std::unique_ptr<WBMpcInterface> interface(new WBMpcInterface(taskFile, urdfFile, referenceFile, verbose));
  RETURN_IF_ERROR(interface->setupModels());
  RETURN_IF_ERROR(interface->setupOptimalControlProblem());
  return interface;
}

absl::StatusOr<std::unique_ptr<WBMpcInterface>> WBMpcInterface::CreateControllerModels(const std::string& taskFile,
                                                                                       const std::string& urdfFile,
                                                                                       const std::string& referenceFile) {
  RETURN_IF_ERROR(checkInputFileExists("task file", taskFile));
  RETURN_IF_ERROR(checkInputFileExists("URDF file", urdfFile));
  RETURN_IF_ERROR(checkInputFileExists("reference file", referenceFile));
  ASSIGN_OR_RETURN(const bool verbose, ModelSettings::loadInterfaceVerbose(taskFile));
  std::unique_ptr<WBMpcInterface> interface(new WBMpcInterface(taskFile, urdfFile, referenceFile, verbose));
  // The models only; the problem, which tapes and loads the CppAD libraries, is the MPC's.
  RETURN_IF_ERROR(interface->setupModels());
  return interface;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status WBMpcInterface::setupOptimalControlProblem() {
  // Loaded before the factory is built: the factory needs to know whether the contact cones it creates may gate
  // themselves on the mode schedule, and that follows the hard `zero_wrench` constraint.
  ASSIGN_OR_RETURN(const MpcFormulationTasks formulationTasks, loadMpcFormulationTasks(taskFile_, verbose_));

  // The contact-implicit formulation is implemented for the centroidal MPC only: this interface builds none of its
  // three terms. Accepting the combination here would silently give a whole-body task file the WORST half of it - the
  // loader has already forced `zero_wrench` out, so the swing foot's wrench would be unbounded - with none of the
  // complementarity conditions that are supposed to replace it.
  if (usesContactImplicitFormulation(formulationTasks)) {
    return absl::InvalidArgumentError(
        "[WBMpcInterface] the contact-implicit formulation (contact_complementarity / force_weighted_slip / "
        "ground_penetration) is implemented for the centroidal MPC only; this interface builds none of those terms, so listing "
        "them would remove the hard 'zero_wrench' constraint and put nothing in its place "
        "(humanoid_nmpc/docs/contact_implicit_mpc/README.md).");
  }
  // CoM + ACoM tracking is implemented for the centroidal MPC only. Here it would be worse than absent: the cost is not
  // built, yet the factory's base-pose indices (6..11) are the centroidal state's, which in this state are the first six
  // joint angles - so accepting it would silently drop the weights of one leg and put nothing in their place.
  if (formulationTasks.hasCost(MpcCostType::ComAndAcomTrackingCost)) {
    return absl::InvalidArgumentError(
        "[WBMpcInterface] 'com_and_acom_tracking_cost' is implemented for the centroidal MPC only: this interface does not build "
        "ComAndAcomTrackingCost, and the base-pose block it would zero in Q and Q_final is a block of joint weights in the "
        "whole-body state. Remove 'com_and_acom_tracking_cost' from costs (humanoid_learning/acom/README.md, section 4.1).");
  }
  // The basis-vector contact inputs are implemented for the centroidal MPC only: this interface has no decorated model,
  // no basis dynamics and no non-negativity barrier, so it would run the wrench parameterization while the file asks for
  // the other one. The loader also refuses the retired `useContactBasisVectorInputs` boolean, as the centroidal MPC does.
  ASSIGN_OR_RETURN(const ContactInputParameterization contactInputs, loadContactInputParameterization(taskFile_));
  if (contactInputs != ContactInputParameterization::kWrench) {
    return absl::InvalidArgumentError(
        absl::StrCat("[WBMpcInterface] ", kContactInputParameterizationKey, ": ", contactInputParameterizationName(contactInputs),
                     " is implemented for the centroidal MPC only; the whole-body MPC optimizes the contact wrenches directly. Set ",
                     kContactInputParameterizationKey, ": ", kWrenchContactInputParameterization,
                     ", or delete the key (humanoid_nmpc/docs/contact_basis_vectors/README.md)."));
  }
  // Nor is the online contact planner wired here: this interface builds a gait-schedule reference manager and no planner
  // module, so the name would be read and then ignored while the robot walks the periodic schedule. The loader also
  // refuses the retired `useContactPlanning` boolean, as the centroidal MPC does.
  ASSIGN_OR_RETURN(const ContactScheduleSource contactScheduleSource, loadContactScheduleSource(taskFile_));
  if (contactScheduleSource != ContactScheduleSource::kGaitSchedule) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[WBMpcInterface] ", kContactScheduleSourceKey, ": ", contactScheduleSourceName(contactScheduleSource),
        " is implemented for the centroidal MPC only; this interface builds no contact planner and would walk the periodic gait "
        "schedule instead. Set ",
        kContactScheduleSourceKey, ": ", kGaitScheduleContactScheduleSource,
        " in the whole-body task file, or delete the key (humanoid_nmpc/docs/README.md, section 2)."));
  }
  // The DCM terminal cost is built by the centroidal MPC only: this interface would end the horizon on no terminal cost
  // at all while the file asks for the capture point. The costs below are the ones this interface builds.
  if (formulationTasks.hasCost(MpcCostType::DcmTerminalCost)) {
    return absl::InvalidArgumentError(
        "[WBMpcInterface] 'dcm_terminal_cost' is implemented for the centroidal MPC only: this interface does not build "
        "DcmTerminalCost, so the horizon would end on no terminal cost at all. List 'terminal_cost' (Q_final) in costs "
        "instead (humanoid_nmpc/docs/README.md, section 1).");
  }

  HumanoidCostConstraintFactory factory =
      HumanoidCostConstraintFactory(taskFile_, referenceFile_, *referenceManagerPtr_, *pinocchioInterfacePtr_, *mpcRobotModelPtr_,
                                    *mpcRobotModelADPtr_, modelSettings_, verbose_, contactConstraintsAreScheduleGated(formulationTasks));

  // Optimal control problem
  problemPtr_.reset(new OptimalControlProblem);

  // Dynamics
  std::unique_ptr<SystemDynamicsBase> dynamicsPtr;
  const std::string modelName = "dynamics";
  dynamicsPtr.reset(new WBAccelDynamicsAD(*pinocchioInterfacePtr_, *mpcRobotModelADPtr_, modelName, modelSettings_));

  problemPtr_->dynamicsPtr = std::move(dynamicsPtr);

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
                           formulationTasks.hasSoftConstraint(MpcSoftConstraintType::NormalVelocity) ||
                           formulationTasks.hasCost(MpcCostType::TaskSpaceFootCost);
    if (needsEeDynamics) {
      eeDynamicsPtr.reset(new PinocchioEndEffectorDynamicsCppAd(*pinocchioInterfacePtr_, *mpcRobotModelADPtr_, {footName}, footName,
                                                                modelSettings_.modelFolderCppAd, modelSettings_.recompileLibrariesCppAd,
                                                                modelSettings_.verboseCppAd));
    }

    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactWrenchCone)) {
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
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ZeroVelocity) && eeDynamicsPtr) {
      std::unique_ptr<StateInputConstraint> stanceConstraint = getStanceFootConstraint(*eeDynamicsPtr, i);
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
    // The same row as a cost; see CentroidalMpcInterface for why the hard form cannot coexist with a solver that is
    // meant to choose its own touch-down.
    if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::NormalVelocity) && eeDynamicsPtr) {
      auto penalty = std::make_unique<QuadraticPenalty>(modelSettings_.footConstraintConfig.normalVelocitySoftConstraintWeight);
      problemPtr_->softConstraintPtr->add(
          contact_term::name(footName, contact_term::kNormalVelocitySoft),
          std::make_unique<StateInputSoftConstraint>(getNormalVelocityConstraint(*eeDynamicsPtr, i), std::move(penalty)));
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
  if (!numerics::almost_eq(footCfg.positionErrorGain_z, /*y=*/0.0)) {
    config.Ax(2, 2) = footCfg.positionErrorGain_z;
  }
  if (!numerics::almost_eq(footCfg.orientationErrorGain, /*y=*/0.0)) {
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
  PropertyTree pt;
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
