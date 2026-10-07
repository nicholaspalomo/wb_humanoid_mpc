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
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

// Pinocchio forward declarations must be included first
#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "ocs2_core/misc/Display.h"
#include "ocs2_core/penalties/Penalties.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_oc/synchronized_module/SolverSynchronizedModule.h"
#include "ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h"

#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/costs/JointMimicFromConfig.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/initialization/WeightCompInitializer.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/joint_weights.nproto.h"
#include "humanoid_mpc_config/mpc_settings_config.nproto.pb.h"
#include "humanoid_mpc_config/mpc_settings_config.pb.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/rollout_settings_config.nproto.pb.h"
#include "humanoid_mpc_config/rollout_settings_config.pb.h"
#include "humanoid_mpc_config/sqp_settings_config.nproto.pb.h"
#include "humanoid_mpc_config/sqp_settings_config.pb.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/WBMpcPreComputation.h"
#include "humanoid_wb_mpc/config/costs/EndEffectorDynamicsWeightsFromConfig.h"
#include "humanoid_wb_mpc/constraint/JointMimicDynamicsConstraint.h"
#include "humanoid_wb_mpc/constraint/SwingLegVerticalConstraintCppAd.h"
#include "humanoid_wb_mpc/constraint/ZeroAccelerationConstraintCppAd.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsCostHelpers.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsFootCost.h"
#include "humanoid_wb_mpc/cost/JointTorqueCostCppAd.h"
#include "humanoid_wb_mpc/dynamics/WBAccelDynamicsAD.h"
#include "humanoid_wb_mpc/end_effector/PinocchioEndEffectorDynamicsCppAd.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {

namespace {

/**
 * NotFound naming `path` when it does not exist; `what` says which of the interface's three inputs it is. The path forms
 * of the factories run it on all three before anything reads them: the model settings read the URDF as well as the task
 * file.
 */
absl::Status checkInputFileExists(absl::string_view what, const std::string& path) {
  std::error_code error;
  if (!std::filesystem::exists(path, error)) {
    return absl::NotFoundError(absl::StrCat("[WBMpcInterface] ", what, " not found: ", path));
  }
  LOG(INFO) << "[WBMpcInterface] " << what << ": " << path;
  return absl::OkStatus();
}

// The error of a typed file's conversion, prefixed with the file it is about ("<file>: <message>", withConfigFile()),
// or unchanged when there is no file to name (`source` empty: a typed file).
absl::Status fromFile(const absl::Status& status, absl::string_view source) {
  return source.empty() ? status : withConfigFile(status, source);
}
template <typename T>
absl::StatusOr<T> fromFile(absl::StatusOr<T> result, absl::string_view source) {
  if (result.ok()) return result;
  return fromFile(result.status(), source);
}

/**
 * Runs `steps`, the part of a factory that builds the models and the problem, and returns their Status, or
 * InvalidArgument with what was thrown out of them: OCS2's CppAD code generation reports a library it cannot build or
 * load by throwing (CppAdInterface::loadModels() on a stale library), and so do Pinocchio and OCS2's gait tiling for a
 * model or schedule they cannot set up. Nothing here runs on the solver or realtime threads.
 */
absl::Status catchingLibraryExceptions(absl::FunctionRef<absl::Status()> steps) {
  // NOLINTNEXTLINE(exceptions): OCS2's CppAD code generation and Pinocchio report failures by throwing; converted to a Status here, once.
  try {
    return steps();
  } catch (const std::exception& error) {  // NOLINT(exceptions): the boundary of the try above.
    return absl::InvalidArgumentError(absl::StrCat("[WBMpcInterface] the MPC was not set up: ", error.what()));
  }
}

/** Logs the solver settings of `taskFile` (multiple_shooting, rollout, mpc) as the task file writes them. */
void logSolverSettings(const mpc_config::TaskFile& taskFile) {
  humanoid_mpc_config::SqpSettingsConfig sqp;
  mpc_config::ToProto(taskFile.multiple_shooting, &sqp);
  humanoid_mpc_config::RolloutSettingsConfig rollout;
  mpc_config::ToProto(taskFile.rollout, &rollout);
  humanoid_mpc_config::MpcSettingsConfig mpc;
  mpc_config::ToProto(taskFile.mpc, &mpc);
  LOG(INFO) << "[WBMpcInterface] The solver settings of the task file:\nmultiple_shooting {\n"
            << nproto::WriteTextproto(sqp) << "}\nrollout {\n"
            << nproto::WriteTextproto(rollout) << "}\nmpc {\n"
            << nproto::WriteTextproto(mpc) << "}";
}

/**
 * Refuses the terms and selections of `tasks` and `taskFile` that only the centroidal MPC implements, each by name and
 * before any term is built: this interface builds none of them, and accepting one would run another formulation than the
 * file asks for.
 */
absl::Status checkWholeBodyFormulation(const MpcFormulationTasks& tasks, const mpc_config::TaskFile& taskFile) {
  // The contact-implicit formulation: this interface builds none of its three terms. Accepting the combination here
  // would silently give a whole-body task file the WORST half of it - the formulation has already forced `zero_wrench`
  // out, so the swing foot's wrench would be unbounded - with none of the complementarity conditions that are supposed
  // to replace it.
  if (usesContactImplicitFormulation(tasks)) {
    return absl::InvalidArgumentError(
        "[WBMpcInterface] the contact-implicit formulation (contact_complementarity / force_weighted_slip / "
        "ground_penetration) is implemented for the centroidal MPC only; this interface builds none of those terms, so listing "
        "them would remove the hard 'zero_wrench' constraint and put nothing in its place "
        "(humanoid_nmpc/docs/contact_implicit_mpc/README.md).");
  }
  // CoM + ACoM tracking would be worse than absent: the cost is not built, yet the factory's base-pose indices (6..11)
  // are the centroidal state's, which in this state are the first six joint angles - so accepting it would silently
  // drop the weights of one leg and put nothing in their place.
  if (tasks.hasCost(MpcCostType::kComAndAcomTrackingCost)) {
    return absl::InvalidArgumentError(
        "[WBMpcInterface] 'com_and_acom_tracking_cost' is implemented for the centroidal MPC only: this interface does not build "
        "ComAndAcomTrackingCost, and the base-pose block it would zero in state_weights and final_state_weights is a block of "
        "joint weights in the whole-body state. Remove 'com_and_acom_tracking_cost' from costs (humanoid_learning/acom/README.md, "
        "section 4.1).");
  }
  // The basis-vector contact inputs: this interface has no decorated model, no basis dynamics and no non-negativity
  // barrier, so it would run the wrench parameterization while the file asks for the other one.
  ASSIGN_OR_RETURN(const ContactInputParameterization contactInputs, contactInputParameterizationFromConfig(taskFile));
  if (contactInputs != ContactInputParameterization::kWrench) {
    return absl::InvalidArgumentError(
        absl::StrCat("[WBMpcInterface] contact_input_parameterization: \"", contactInputParameterizationName(contactInputs),
                     "\" is implemented for the centroidal MPC only; the whole-body MPC optimizes the contact wrenches directly. Set "
                     "contact_input_parameterization: \"",
                     kWrenchContactInputParameterization, "\", or delete the field (humanoid_nmpc/docs/contact_basis_vectors/README.md)."));
  }
  // The online contact planner: this interface builds a gait-schedule reference manager and no planner module, so the
  // name would be read and then ignored while the robot walks the periodic schedule.
  ASSIGN_OR_RETURN(const ContactScheduleSource contactScheduleSource, contactScheduleSourceFromConfig(taskFile));
  if (contactScheduleSource != ContactScheduleSource::kGaitSchedule) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[WBMpcInterface] contact_schedule_source: \"", contactScheduleSourceName(contactScheduleSource),
        "\" is implemented for the centroidal MPC only; this interface builds no contact planner and would walk the periodic gait "
        "schedule instead. Set contact_schedule_source: \"",
        kGaitScheduleContactScheduleSource,
        "\" in the whole-body task file, or delete the field (humanoid_nmpc/docs/README.md, section 2)."));
  }
  // The DCM terminal cost: this interface would end the horizon on no terminal cost at all while the file asks for the
  // capture point.
  if (tasks.hasCost(MpcCostType::kDcmTerminalCost)) {
    return absl::InvalidArgumentError(
        "[WBMpcInterface] 'dcm_terminal_cost' is implemented for the centroidal MPC only: this interface does not build "
        "DcmTerminalCost, so the horizon would end on no terminal cost at all. List 'terminal_cost' (final_state_weights) in costs "
        "instead (humanoid_nmpc/docs/README.md, section 1).");
  }
  return absl::OkStatus();
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
WBMpcInterface::WBMpcInterface(ModelSettings modelSettings, const SolverSettings& solverSettings)
    : modelSettings_(std::move(modelSettings)),
      mpcSettings_(solverSettings.mpcSettings),
      sqpSettings_(solverSettings.sqpSettings),
      rolloutSettings_(solverSettings.rolloutSettings),
      verbose_(solverSettings.verbose) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::StatusOr<std::unique_ptr<WBMpcInterface>> WBMpcInterface::Create(const mpc_config::TaskFile& taskFile,
                                                                       const std::string& urdfFile,
                                                                       const mpc_config::ReferenceFile& referenceFile) {
  return build(taskFile, urdfFile, referenceFile, ConfigSources{}, Scope::kMpc);
}

absl::StatusOr<std::unique_ptr<WBMpcInterface>> WBMpcInterface::Create(const std::string& taskFile,
                                                                       const std::string& urdfFile,
                                                                       const std::string& referenceFile) {
  return buildFromFiles(taskFile, urdfFile, referenceFile, Scope::kMpc);
}

absl::StatusOr<std::unique_ptr<WBMpcInterface>> WBMpcInterface::CreateControllerModels(const mpc_config::TaskFile& taskFile,
                                                                                       const std::string& urdfFile,
                                                                                       const mpc_config::ReferenceFile& referenceFile) {
  return build(taskFile, urdfFile, referenceFile, ConfigSources{}, Scope::kControllerModels);
}

absl::StatusOr<std::unique_ptr<WBMpcInterface>> WBMpcInterface::CreateControllerModels(const std::string& taskFile,
                                                                                       const std::string& urdfFile,
                                                                                       const std::string& referenceFile) {
  return buildFromFiles(taskFile, urdfFile, referenceFile, Scope::kControllerModels);
}

absl::StatusOr<std::unique_ptr<WBMpcInterface>> WBMpcInterface::buildFromFiles(const std::string& taskFile,
                                                                               const std::string& urdfFile,
                                                                               const std::string& referenceFile,
                                                                               Scope scope) {
  // The inputs exist before anything reads them; the model settings read the URDF as well.
  RETURN_IF_ERROR(checkInputFileExists("task file", taskFile));
  RETURN_IF_ERROR(checkInputFileExists("URDF file", urdfFile));
  RETURN_IF_ERROR(checkInputFileExists("reference file", referenceFile));
  ASSIGN_OR_RETURN(const mpc_config::TaskFile task, loadTaskFile(taskFile));
  ASSIGN_OR_RETURN(const mpc_config::ReferenceFile reference, loadReferenceFile(referenceFile));
  return build(task, urdfFile, reference, ConfigSources{.taskFile = taskFile, .referenceFile = referenceFile}, scope);
}

absl::StatusOr<std::unique_ptr<WBMpcInterface>> WBMpcInterface::build(const mpc_config::TaskFile& taskFile,
                                                                      const std::string& urdfFile,
                                                                      const mpc_config::ReferenceFile& referenceFile,
                                                                      const ConfigSources& sources,
                                                                      Scope scope) {
  std::unique_ptr<WBMpcInterface> interface;
  RETURN_IF_ERROR(catchingLibraryExceptions([&]() -> absl::Status {
    // The solver settings first: their interface.verbose decides the logging of everything after them.
    ASSIGN_OR_RETURN(const SolverSettings solverSettings, fromFile(solverSettingsFromConfig(taskFile), sources.taskFile));
    if (solverSettings.verbose) logSolverSettings(taskFile);
    ASSIGN_OR_RETURN(ModelSettings modelSettings,
                     fromFile(ModelSettings::Create(taskFile, urdfFile, "wb_mpc_", solverSettings.verbose), sources.taskFile));
    // The mode schedule, the reference file's part of the models.
    ASSIGN_OR_RETURN(std::shared_ptr<GaitSchedule> gaitSchedule,
                     fromFile(GaitSchedule::Create(referenceFile, modelSettings, solverSettings.verbose), sources.referenceFile));
    interface = absl::WrapUnique(new WBMpcInterface(std::move(modelSettings), solverSettings));
    interface->taskFile_ = taskFile;
    RETURN_IF_ERROR(fromFile(interface->setupModels(taskFile, urdfFile, std::move(gaitSchedule)), sources.taskFile));
    // The models only for the controller; the problem, which tapes and loads the CppAD libraries, is the MPC's.
    if (scope == Scope::kControllerModels) return absl::OkStatus();
    return fromFile(interface->setupOptimalControlProblem(taskFile), sources.taskFile);
  }));
  return interface;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
absl::Status WBMpcInterface::setupModels(const mpc_config::TaskFile& taskFile,
                                         const std::string& urdfFile,
                                         std::shared_ptr<GaitSchedule> gaitSchedule) {
  // PinocchioInterface: a model whose actuated joints are not model_settings' in order is refused here, by name.
  ASSIGN_OR_RETURN(PinocchioInterface pinocchioInterface, loadCustomPinocchioInterface(taskFile, urdfFile, modelSettings_));
  pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));

  // Setup WB State Input Mapping
  mpcRobotModelPtr_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(modelSettings_);
  mpcRobotModelADPtr_ = std::make_unique<WBAccelMpcRobotModel<ad_scalar_t>>(modelSettings_);

  // Swing trajectory planner
  ASSIGN_OR_RETURN(const SwingTrajectoryPlanner::Config swingConfig, swingTrajectorySettingsFromConfig(taskFile.swing_trajectory_config));
  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner = std::make_unique<SwingTrajectoryPlanner>(swingConfig, kNumContacts);

  // Mode schedule manager
  referenceManagerPtr_ = std::make_shared<SwitchedModelReferenceManager>(std::move(gaitSchedule), std::move(swingTrajectoryPlanner),
                                                                         *pinocchioInterfacePtr_, *mpcRobotModelPtr_);
  // A legs-only robot omits model_settings.arm_joint_names and has no arm to swing.
  referenceManagerPtr_->setArmSwingReferenceActive(modelSettings_.hasArmSwingJoints);

  // The initial state, by coordinate and joint name.
  ASSIGN_OR_RETURN(
      initialState_,
      stateValuesFromConfig(taskFile.initial_state, stateInputLayout(modelSettings_, StateInputLayout::Mpc::kWholeBody), "initial_state"));
  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status WBMpcInterface::setupOptimalControlProblem(const mpc_config::TaskFile& taskFile) {
  // Read before the factory is built: the factory needs to know whether the contact cones it creates may gate
  // themselves on the mode schedule, and that follows the hard `zero_wrench` constraint.
  ASSIGN_OR_RETURN(const MpcFormulationTasks formulationTasks, mpcFormulationTasksFromConfig(taskFile, formulationLoggingFor(verbose_)));
  RETURN_IF_ERROR(checkWholeBodyFormulation(formulationTasks, taskFile));

  const HumanoidCostConstraintFactory factory(&taskFile, StateInputLayout::Mpc::kWholeBody, *referenceManagerPtr_, *pinocchioInterfacePtr_,
                                              *mpcRobotModelPtr_, *mpcRobotModelADPtr_, modelSettings_, verbose_,
                                              contactConstraintsAreScheduleGated(formulationTasks));

  // Optimal control problem
  problemPtr_ = std::make_unique<OptimalControlProblem>();

  // Dynamics
  std::unique_ptr<SystemDynamicsBase> dynamicsPtr;
  const std::string modelName = "dynamics";
  dynamicsPtr = std::make_unique<WBAccelDynamicsAD>(*pinocchioInterfacePtr_, *mpcRobotModelADPtr_, modelName, modelSettings_);

  problemPtr_->dynamicsPtr = std::move(dynamicsPtr);

  RETURN_IF_ERROR(addCostsAndStateConstraints(formulationTasks, taskFile, factory));
  RETURN_IF_ERROR(addContactTerms(formulationTasks, taskFile, factory));

  // Pre-computation
  problemPtr_->preComputationPtr = std::make_unique<WBMpcPreComputation>(
      *pinocchioInterfacePtr_, *referenceManagerPtr_->getSwingTrajectoryPlanner(), *mpcRobotModelPtr_);

  // Rollout
  rolloutPtr_ = std::make_unique<TimeTriggeredRollout>(*problemPtr_->dynamicsPtr, rolloutSettings_);

  // Initialization
  initializerPtr_ = std::make_unique<WeightCompInitializer>(*pinocchioInterfacePtr_, *referenceManagerPtr_, *mpcRobotModelPtr_);

  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status WBMpcInterface::addCostsAndStateConstraints(const MpcFormulationTasks& tasks,
                                                         const mpc_config::TaskFile& taskFile,
                                                         const HumanoidCostConstraintFactory& factory) {
  // Cost terms
  if (tasks.hasCost(MpcCostType::kStateInputQuadraticCost)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> cost, factory.makeStateInputQuadraticCost());
    problemPtr_->costPtr->add(kStateInputQuadraticCostTerm, std::move(cost));
  }
  if (tasks.hasCost(MpcCostType::kStateQuadraticCost)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> cost, factory.makeStateQuadraticCost());
    problemPtr_->costPtr->add(kStateQuadraticCostTerm, std::move(cost));
  }
  if (tasks.hasCost(MpcCostType::kInputQuadraticCost)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> cost, factory.makeInputQuadraticCost());
    problemPtr_->costPtr->add(kInputQuadraticCostTerm, std::move(cost));
  }
  if (tasks.hasCost(MpcCostType::kJointTorqueCost)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> cost, makeJointTorqueCost(taskFile.joint_torque_weights));
    problemPtr_->costPtr->add(JointTorqueCostCppAd::kTermName, std::move(cost));
  }
  if (tasks.hasCost(MpcCostType::kTerminalCost)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateCost> cost, factory.makeTerminalCost());
    problemPtr_->finalCostPtr->add(kTerminalCostTerm, std::move(cost));
  }

  // Soft constraints
  if (tasks.hasSoftConstraint(MpcSoftConstraintType::kJointLimits)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateCost> jointLimits, factory.makeJointLimitsConstraint());
    problemPtr_->stateSoftConstraintPtr->add(kJointLimitsTerm, std::move(jointLimits));
  }
  if (tasks.hasSoftConstraint(MpcSoftConstraintType::kFootCollision)) {
    ASSIGN_OR_RETURN(std::unique_ptr<StateCost> footCollision, factory.makeFootCollisionConstraint());
    problemPtr_->stateSoftConstraintPtr->add(kFootCollisionTerm, std::move(footCollision));
  }
  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status WBMpcInterface::addContactTerms(const MpcFormulationTasks& tasks,
                                             const mpc_config::TaskFile& taskFile,
                                             const HumanoidCostConstraintFactory& factory) {
  // Foot tracking cost weights
  EndEffectorDynamicsWeights footTrackingCostWeights;
  if (tasks.hasCost(MpcCostType::kTaskSpaceFootCost)) {
    ASSIGN_OR_RETURN(footTrackingCostWeights, wholeBodyFootCostWeightsFromConfig(taskFile.task_space_foot_cost));
  }
  // The knee mimic joints of both legs, in the order of the contacts.
  feet_array_t<JointMimicSettings> kneeMimicJoints;
  if (tasks.hasHardConstraint(MpcHardConstraintType::kKneeJointMimic)) {
    ASSIGN_OR_RETURN(kneeMimicJoints, kneeMimicJointsFromConfig(taskFile.mimic_joints));
  }
  const bool needsEeDynamics =
      tasks.hasHardConstraint(MpcHardConstraintType::kZeroVelocity) || tasks.hasSoftConstraint(MpcSoftConstraintType::kZeroVelocity) ||
      tasks.hasHardConstraint(MpcHardConstraintType::kNormalVelocity) || tasks.hasSoftConstraint(MpcSoftConstraintType::kNormalVelocity) ||
      tasks.hasCost(MpcCostType::kTaskSpaceFootCost);

  for (size_t i = 0; i < kNumContacts; ++i) {
    const std::string& footName = modelSettings_.contactNames[i];

    std::unique_ptr<EndEffectorDynamics<scalar_t>> eeDynamicsPtr;
    if (needsEeDynamics) {
      eeDynamicsPtr = std::make_unique<PinocchioEndEffectorDynamicsCppAd>(
          *pinocchioInterfacePtr_, *mpcRobotModelADPtr_, std::vector<std::string>{footName}, footName, modelSettings_.modelFolderCppAd,
          modelSettings_.recompileLibrariesCppAd, modelSettings_.verboseCppAd);
    }

    if (tasks.hasSoftConstraint(MpcSoftConstraintType::kContactWrenchCone)) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> contactWrenchCone, factory.makeContactWrenchConeConstraint(i));
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kContactWrenchCone), std::move(contactWrenchCone));
    }
    if (tasks.hasSoftConstraint(MpcSoftConstraintType::kFrictionForceCone)) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> frictionForceCone, factory.makeFrictionForceConeConstraint(i));
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kFrictionForceCone), std::move(frictionForceCone));
    }
    if (tasks.hasSoftConstraint(MpcSoftConstraintType::kContactMomentXy)) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateInputCost> contactMomentXYConstraint,
                       factory.makeContactMomentXYConstraint(i, absl::StrCat(footName, "_contact_moment_XY_constraint")));
      problemPtr_->softConstraintPtr->add(contact_term::name(footName, contact_term::kContactMomentXY),
                                          std::move(contactMomentXYConstraint));
    }
    if (tasks.hasSoftConstraint(MpcSoftConstraintType::kZeroVelocity) && eeDynamicsPtr) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateInputConstraint> stanceConstraint, getStanceFootConstraint(*eeDynamicsPtr, i));
      std::unique_ptr<QuadraticPenalty> penalty =
          std::make_unique<QuadraticPenalty>(modelSettings_.footConstraintConfig.softConstraintWeight);
      problemPtr_->softConstraintPtr->add(zeroVelocityTermName(footName),
                                          std::make_unique<StateInputSoftConstraint>(std::move(stanceConstraint), std::move(penalty)));
    }

    if (tasks.hasHardConstraint(MpcHardConstraintType::kZeroWrench)) {
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_zeroWrench"), factory.getZeroWrenchConstraint(i));
    }
    if (tasks.hasHardConstraint(MpcHardConstraintType::kZeroVelocity) && eeDynamicsPtr) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateInputConstraint> stanceConstraint, getStanceFootConstraint(*eeDynamicsPtr, i));
      problemPtr_->equalityConstraintPtr->add(zeroVelocityTermName(footName), std::move(stanceConstraint));
    }
    if (tasks.hasHardConstraint(MpcHardConstraintType::kNormalVelocity) && eeDynamicsPtr) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateInputConstraint> normalVelocityConstraint, getNormalVelocityConstraint(*eeDynamicsPtr, i));
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_normalVelocity"), std::move(normalVelocityConstraint));
    }
    // The same row as a cost; see CentroidalMpcInterface for why the hard form cannot coexist with a solver that is
    // meant to choose its own touch-down.
    if (tasks.hasSoftConstraint(MpcSoftConstraintType::kNormalVelocity) && eeDynamicsPtr) {
      ASSIGN_OR_RETURN(std::unique_ptr<StateInputConstraint> normalVelocityConstraint, getNormalVelocityConstraint(*eeDynamicsPtr, i));
      std::unique_ptr<QuadraticPenalty> penalty =
          std::make_unique<QuadraticPenalty>(modelSettings_.footConstraintConfig.normalVelocitySoftConstraintWeight);
      problemPtr_->softConstraintPtr->add(
          contact_term::name(footName, contact_term::kNormalVelocitySoft),
          std::make_unique<StateInputSoftConstraint>(std::move(normalVelocityConstraint), std::move(penalty)));
    }
    if (tasks.hasHardConstraint(MpcHardConstraintType::kKneeJointMimic)) {
      // The loop runs over the contacts, so the index is valid by construction.
      const JointMimicSettings& mimic = kneeMimicJoints[i];
      ASSIGN_OR_RETURN(std::unique_ptr<JointMimicDynamicsConstraint> kneeJointMimic,
                       JointMimicDynamicsConstraint::Create(*mpcRobotModelPtr_, mimic.parentJointName, mimic.childJointName,
                                                            mimic.multiplier, mimic.positionGain, mimic.velocityGain));
      problemPtr_->equalityConstraintPtr->add(absl::StrCat(footName, "_kneeJointMimic"), std::move(kneeJointMimic));
    }

    if (tasks.hasCost(MpcCostType::kTaskSpaceFootCost) && eeDynamicsPtr) {
      problemPtr_->costPtr->add(EndEffectorDynamicsFootCost::termName(footName),
                                std::make_unique<EndEffectorDynamicsFootCost>(
                                    *referenceManagerPtr_, footTrackingCostWeights, *pinocchioInterfacePtr_, *eeDynamicsPtr,
                                    *mpcRobotModelADPtr_, i, EndEffectorDynamicsFootCost::libraryModelName(footName), modelSettings_));
    }
  }
  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::StatusOr<std::unique_ptr<StateInputConstraint>> WBMpcInterface::getStanceFootConstraint(
    const EndEffectorDynamics<scalar_t>& eeDynamics, size_t contactPointIndex) {
  ASSIGN_OR_RETURN(std::unique_ptr<ZeroAccelerationConstraintCppAd> constraint,
                   ZeroAccelerationConstraintCppAd::Create(*referenceManagerPtr_, eeDynamics, contactPointIndex,
                                                           stanceFootAccelerationConstraintConfig(modelSettings_.footConstraintConfig)));
  return constraint;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
absl::StatusOr<std::unique_ptr<StateInputConstraint>> WBMpcInterface::getNormalVelocityConstraint(
    const EndEffectorDynamics<scalar_t>& eeDynamics, size_t contactPointIndex) {
  ASSIGN_OR_RETURN(std::unique_ptr<SwingLegVerticalConstraintCppAd> constraint,
                   SwingLegVerticalConstraintCppAd::Create(*referenceManagerPtr_, eeDynamics, contactPointIndex));
  return constraint;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::StatusOr<std::unique_ptr<StateInputCost>> WBMpcInterface::makeJointTorqueCost(const mpc_config::JointWeights& weights) const {
  ASSIGN_OR_RETURN(
      const vector_t jointTorqueWeights,
      jointTorqueWeightsFromConfig(weights, stateInputLayout(modelSettings_, StateInputLayout::Mpc::kWholeBody), "joint_torque_weights"));
  return std::make_unique<JointTorqueCostCppAd>(jointTorqueWeights, *pinocchioInterfacePtr_, *mpcRobotModelADPtr_,
                                                JointTorqueCostCppAd::kLibraryCostName, modelSettings_);
}

}  // namespace ocs2::humanoid
