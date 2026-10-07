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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "tools/config_dump/StackDump.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/MRT_BASE.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/teleop/KeyboardVelocityCommand.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"
#include "robot_model/RobotDescription.h"
#include "tools/config_dump/ProblemDump.h"
#include "tools/config_dump/SettingsDump.h"
#include "tools/config_dump/ValueDump.h"

namespace ocs2::humanoid::config_dump {
namespace {

/**
 * An MpcLink with no MPC behind it, for an MRT joint controller whose PD gains are dumped: observations go nowhere and no
 * policy ever arrives, so the controller starts no solver and opens no bus.
 */
class NullMpcLink final : public MpcLink {
 public:
  void start(const SystemObservation& /*initialObservation*/) override {}
  void stop() override {}
  MRT_BASE& mrt() override { return mrt_; }
  const MRT_BASE& mrt() const override { return mrt_; }

 private:
  /** An MRT that discards what it is given. */
  class NullMrt final : public MRT_BASE {
   public:
    void resetMpcNode(const TargetTrajectories& /*initTargetTrajectories*/) override {}
    void setCurrentObservation(const SystemObservation& /*observation*/) override {}
  };

  NullMrt mrt_;
};

MpcLinkFactory nullMpcLinkFactory() {
  return [](const MpcLink::ResetTargetFunction& /*resetTarget*/) { return std::make_unique<NullMpcLink>(); };
}

/** The motion manager's target function, which the dump never calls. */
TargetTrajectories noTargets(const vector4_t& /*velocityTarget*/,
                             scalar_t /*initTime*/,
                             scalar_t /*finalTime*/,
                             const vector_t& /*initState*/) {
  return TargetTrajectories();
}

/** [v_x, v_y, delta height, yaw rate]: beyond every shipped limit, so that the limits clamp it. */
vector4_t saturatingVelocityCommand() {
  return vector4_t(5.0, 5.0, 1.0, 5.0);
}

/** [delta x, delta y, delta height, delta yaw] of a pose command. */
vector4_t poseCommand() {
  return vector4_t(1.0, 0.5, 0.1, 0.5);
}

void dumpTargets(ValueDump& dump, TargetTrajectoriesCalculatorBase& calculator, const vector_t& initialState) {
  dump.addSection("target_trajectories");
  dumpTargetTrajectories(dump, "target_trajectories.velocity_command",
                         calculator.commandedVelocityToTargetTrajectories(saturatingVelocityCommand(), /*initTime=*/0.0, initialState));
  dumpTargetTrajectories(dump, "target_trajectories.position_command",
                         calculator.commandedPositionToTargetTrajectories(poseCommand(), /*initTime=*/0.0, initialState));
}

absl::Status dumpMotionManager(ValueDump& dump,
                               const validation::RobotConfiguration& configuration,
                               std::shared_ptr<SwitchedModelReferenceManager> referenceManager,
                               const MpcRobotModelBase<scalar_t>& commandModel) {
  ASSIGN_OR_RETURN(std::unique_ptr<ProceduralMpcMotionManager> motionManager,
                   ProceduralMpcMotionManager::Create(configuration.gaitFile, configuration.referenceFile, std::move(referenceManager),
                                                      commandModel, noTargets));
  dump.addSection("motion_manager");
  dumpGaitMap(dump, "motion_manager.gaits", motionManager->gaitMap());
  dump.addDouble("motion_manager.maxLinearAcceleration", motionManager->getMaxLinearAcceleration());
  dump.addDouble("motion_manager.maxAngularAcceleration", motionManager->getMaxAngularAcceleration());
  dump.addDouble("motion_manager.velocityCommandFilterBreakFrequency", motionManager->getVelocityCommandFilterBreakFrequency());
  return absl::OkStatus();
}

/** The robot process's, the visualization's and the keyboard teleoperation's settings, which every formulation shares. */
absl::Status dumpApplications(ValueDump& dump, const validation::RobotConfiguration& configuration, const ModelSettings& modelSettings) {
  ASSIGN_OR_RETURN(const RobotProcessSettings robotProcess, loadRobotProcessSettings(configuration.taskFile));
  dump.addSection("robot_process");
  dumpRobotProcessSettings(dump, "robot_process", robotProcess);
  ASSIGN_OR_RETURN(const visualization::VisualizationConfig visualization,
                   visualization::loadVisualizationConfig(configuration.taskFile, modelSettings));
  dump.addSection("visualization");
  dumpVisualizationConfig(dump, "visualization", visualization);
  ASSIGN_OR_RETURN(const teleop::KeyboardCommandLimits keyboard, teleop::loadKeyboardCommandLimits(configuration.referenceFile));
  dump.addSection("keyboard_teleop");
  dumpKeyboardCommandLimits(dump, "keyboard_teleop", keyboard);
  return absl::OkStatus();
}

void dumpSolver(ValueDump& dump,
                const ModelSettings& modelSettings,
                const mpc::Settings& mpcSettings,
                const sqp::Settings& sqpSettings,
                const rollout::Settings& rolloutSettings) {
  dump.addSection("model_settings");
  dumpModelSettings(dump, "model_settings", modelSettings);
  dump.addSection("solver");
  dumpMpcSettings(dump, "mpc", mpcSettings);
  dumpSqpSettings(dump, "multiple_shooting", sqpSettings);
  dumpRolloutSettings(dump, "rollout", rolloutSettings);
}

void dumpReferenceManager(ValueDump& dump, const SwitchedModelReferenceManager& referenceManager) {
  dump.addSection("reference_manager");
  dumpSwingSettings(dump, "reference_manager.swing_trajectory", referenceManager.getSwingTrajectoryPlanner()->getConfig());
  dumpModeSchedule(dump, "reference_manager.initial_mode_schedule", referenceManager.getGaitSchedule()->getCurrentModeSchedule());
  dump.addDouble("reference_manager.terrainHeight", referenceManager.getTerrainHeight());
}

absl::Status dumpCentroidal(ValueDump& dump, const validation::RobotConfiguration& configuration) {
  ASSIGN_OR_RETURN(std::unique_ptr<CentroidalMpcInterface> interface,
                   CentroidalMpcInterface::Create(configuration.taskFile, configuration.urdfFile, configuration.referenceFile));
  dumpSolver(dump, interface->modelSettings(), interface->mpcSettings(), interface->sqpSettings(), interface->rolloutSettings());
  dump.addSection("interface");
  dump.addMatrix("interface.initialState", interface->getInitialState());
  dump.addDouble("interface.nominalComHeight", interface->getNominalComHeight());
  dump.addString("interface.contactInputParameterization", contactInputParameterizationName(interface->contactInputParameterization()));
  dump.addString("interface.contactScheduleSource", contactScheduleSourceName(interface->contactScheduleSource()));
  dump.addBool("interface.usesContactPlanning", interface->usesContactPlanning());
  dump.addInteger("interface.wrenchInputDim", static_cast<int64_t>(interface->getWrenchInputDim()));
  dump.addInteger("interface.numBasisInputs", static_cast<int64_t>(interface->getNumBasisInputs()));
  dump.addDouble("interface.basisScalingRegularization", interface->getBasisScalingRegularization());
  dump.addString("interface.basisGeneratorSet", interface->getBasisGeneratorSet());
  dump.addBool("interface.basisToWrenchMap.has_value", interface->getBasisToWrenchMap().has_value());
  if (interface->getBasisToWrenchMap().has_value()) {
    dump.addFingerprint("interface.basisToWrenchMap", *interface->getBasisToWrenchMap());
  }
  dump.addStrings("interface.costNames", interface->getCostNames());
  dump.addStrings("interface.terminalCostNames", interface->getTerminalCostNames());
  dump.addStrings("interface.softConstraintNames", interface->getSoftConstraintNames());
  dump.addStrings("interface.stateSoftConstraintNames", interface->getStateSoftConstraintNames());
  dump.addStrings("interface.equalityConstraintNames", interface->getEqualityConstraintNames());
  const std::shared_ptr<LocomotionHeuristicLayer>& heuristics = interface->getLocomotionHeuristicLayerPtr();
  dump.addBool("interface.locomotionHeuristics.present", heuristics != nullptr);
  if (heuristics != nullptr) {
    dump.addBool("interface.locomotionHeuristics.basePoseEmpty", heuristics->basePoseEmpty());
    dump.addBool("interface.locomotionHeuristics.footholdEmpty", heuristics->footholdEmpty());
    dump.addBool("interface.locomotionHeuristics.wrenchEmpty", heuristics->wrenchEmpty());
    dump.addString("interface.locomotionHeuristics.summary", heuristics->summary());
  }
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = interface->getSwitchedModelReferenceManagerPtr();
  dumpReferenceManager(dump, *referenceManager);
  const MpcRobotModelBase<scalar_t>& effectiveModel = interface->getEffectiveMpcRobotModel();

  // The MPC node's side: the target trajectories and the motion manager.
  ASSIGN_OR_RETURN(
      std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator> calculator,
      CentroidalMpcTargetTrajectoriesCalculator::Create(configuration.referenceFile, effectiveModel, interface->getPinocchioInterface(),
                                                        interface->getCentroidalModelInfo(), interface->mpcSettings().timeHorizon_));
  dumpTargets(dump, *calculator, interface->getInitialState());
  RETURN_IF_ERROR(dumpMotionManager(dump, configuration, referenceManager, effectiveModel));

  // The robot binary's side: the joint controller's PD gains.
  ASSIGN_OR_RETURN(const robot::model::RobotDescription description, robot::model::RobotDescription::Create(configuration.urdfFile));
  ASSIGN_OR_RETURN(
      std::unique_ptr<CentroidalMpcMrtJointController> controller,
      CentroidalMpcMrtJointController::Create(description, interface->modelSettings(), interface->getMpcRobotModel(), nullMpcLinkFactory(),
                                              interface->getPinocchioInterface(), configuration.pdGainsFile, &effectiveModel));
  dump.addSection("pd_gains");
  dumpJointPdGains(dump, "pd_gains", controller->getPdGains());
  RETURN_IF_ERROR(dumpApplications(dump, configuration, interface->modelSettings()));

  // Last, because it sets a walking schedule and target on the reference manager.
  dump.addSection("problem");
  dumpProblem(dump, "problem", interface->getOptimalControlProblem(), *referenceManager, interface->getInitialState(),
              ProblemLayout{.inputDim = effectiveModel.getInputDim(),
                            .basePositionIndex = static_cast<Eigen::Index>(effectiveModel.getBaseStartindex())});
  return absl::OkStatus();
}

absl::Status dumpWholeBody(ValueDump& dump, const validation::RobotConfiguration& configuration) {
  ASSIGN_OR_RETURN(std::unique_ptr<WBMpcInterface> interface,
                   WBMpcInterface::Create(configuration.taskFile, configuration.urdfFile, configuration.referenceFile));
  dumpSolver(dump, interface->modelSettings(), interface->mpcSettings(), interface->sqpSettings(), interface->rolloutSettings());
  dump.addSection("interface");
  dump.addMatrix("interface.initialState", interface->getInitialState());
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = interface->getSwitchedModelReferenceManagerPtr();
  dumpReferenceManager(dump, *referenceManager);
  const WBAccelMpcRobotModel<scalar_t>& model = interface->getMpcRobotModel();

  ASSIGN_OR_RETURN(std::unique_ptr<WBMpcTargetTrajectoriesCalculator> calculator,
                   WBMpcTargetTrajectoriesCalculator::Create(configuration.referenceFile, model, interface->mpcSettings().timeHorizon_));
  dumpTargets(dump, *calculator, interface->getInitialState());
  RETURN_IF_ERROR(dumpMotionManager(dump, configuration, referenceManager, model));

  ASSIGN_OR_RETURN(const robot::model::RobotDescription description, robot::model::RobotDescription::Create(configuration.urdfFile));
  ASSIGN_OR_RETURN(std::unique_ptr<WBMpcMrtJointController> controller,
                   WBMpcMrtJointController::Create(description, interface->modelSettings(), nullMpcLinkFactory(),
                                                   interface->getPinocchioInterface(), configuration.pdGainsFile));
  dump.addSection("pd_gains");
  dumpJointPdGains(dump, "pd_gains", controller->getPdGains());
  RETURN_IF_ERROR(dumpApplications(dump, configuration, interface->modelSettings()));

  dump.addSection("problem");
  dumpProblem(dump, "problem", interface->getOptimalControlProblem(), *referenceManager, interface->getInitialState(),
              ProblemLayout{.inputDim = model.getInputDim(), .basePositionIndex = static_cast<Eigen::Index>(model.getBaseStartindex())});
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::string> dumpConfiguration(const validation::RobotConfiguration& configuration) {
  ValueDump dump;
  dump.addSection("configuration");
  dump.addString("configuration.name", configuration.name);
  dump.addString("configuration.formulation", validation::formulationName(configuration.formulation));
  if (configuration.formulation == validation::MpcFormulation::kCentroidal) {
    RETURN_IF_ERROR(dumpCentroidal(dump, configuration));
  } else {
    RETURN_IF_ERROR(dumpWholeBody(dump, configuration));
  }
  return dump.text();
}

}  // namespace ocs2::humanoid::config_dump
