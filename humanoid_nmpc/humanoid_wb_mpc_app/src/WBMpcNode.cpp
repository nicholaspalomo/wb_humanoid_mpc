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

#include "humanoid_wb_mpc_app/WBMpcNode.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/parameter_update/CommandLimitsReloaders.h"
#include "humanoid_wb_mpc/mrt/WBMpcParameterUpdater.h"
#include "humanoid_wb_mpc/mrt/WBMpcResetTarget.h"

namespace ocs2::humanoid {

absl::StatusOr<std::unique_ptr<WBMpcNode>> WBMpcNode::Create(
    const node::MpcFiles& files,
    std::unique_ptr<robot::ipc::Bus> bus,
    Options options) {
  const std::optional<std::filesystem::file_time_type> taskFileReadAt =
      lastWriteTime(files.taskFile);
  const std::optional<std::filesystem::file_time_type> referenceFileReadAt =
      lastWriteTime(files.referenceFile);

  std::unique_ptr<WBMpcNode> node = absl::WrapUnique(new WBMpcNode());
  ASSIGN_OR_RETURN(node->interface_,
                   WBMpcInterface::Create(files.taskFile, files.urdfFile,
                                          files.referenceFile));
  WBMpcInterface& interface = *node->interface_;

  node->mpc_ = std::make_unique<SqpMpc>(
      interface.mpcSettings(), interface.sqpSettings(),
      interface.getOptimalControlProblem(), interface.getInitializer());

  // LINT.IfChange(mpc_wiring)
  // Reference and motion management for the procedural MPC.
  ASSIGN_OR_RETURN(node->targetCalculator_,
                   WBMpcTargetTrajectoriesCalculator::Create(
                       files.referenceFile, interface.getMpcRobotModel(),
                       interface.mpcSettings().timeHorizon_));
  WBMpcTargetTrajectoriesCalculator* absl_nonnull calculator =
      node->targetCalculator_.get();
  // The commanded base height stands on the ground the reference manager
  // applied in this solve, so it follows a hot reload of terrain_height
  // (TargetTrajectoriesCalculatorBase::setTerrainHeightSource).
  calculator->setTerrainHeightSource(
      [referenceManager = interface.getSwitchedModelReferenceManagerPtr()]() {
        return referenceManager->getAppliedTerrainHeight();
      });
  const ProceduralMpcMotionManager::VelocityTargetToTargetTrajectories
      targetTrajectoriesFunction =
          [calculator](const vector4_t& velocityTarget, scalar_t initTime,
                       scalar_t /*finalTime*/, const vector_t& initState,
                       const BasePlanarMotion& measuredMotion) {
            return calculator->velocityCommandToTargetTrajectories(
                velocityTarget, initTime, initState, measuredMotion);
          };
  ASSIGN_OR_RETURN(
      node->motionManager_,
      ProceduralMpcMotionManager::Create(
          files.gaitFile, files.referenceFile,
          interface.getSwitchedModelReferenceManagerPtr(),
          interface.getMpcRobotModel(), targetTrajectoriesFunction));
  // A reset of the MPC (MPC_BASE::reset()) resets the command path with it: the
  // motion manager itself, and through this hook the target calculator behind
  // targetTrajectoriesFunction, whose filters are its state.
  node->motionManager_->setResetHook([calculator]() { calculator->reset(); });

  SolverBase& solver = *node->mpc_->getSolverPtr();
  solver.setReferenceManager(interface.getReferenceManagerPtr());
  solver.addSynchronizedModule(node->motionManager_);

  // Hot reloading of the MPC parameters, wired by the one function every node
  // and driver that runs the whole-body MPC uses, as the centroidal node does:
  // it applies the task file's RELOAD_HOT fields before a solve, and hands the
  // command limits of a reloaded reference file to the calculator and the
  // motion manager built above.
  ASSIGN_OR_RETURN(
      node->parameterUpdater_,
      makeWholeBodyMpcParameterUpdater(
          node->mpc_.get(), interface, files.taskFile, files.referenceFile,
          makeCommandLimitsReloaders(calculator, node->motionManager_),
          taskFileReadAt, referenceFileReadAt));
  solver.addSynchronizedModule(node->parameterUpdater_);
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/WholeBodyClosedLoopDriver.cpp:whole_body_mpc_node_wiring, //humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopDriver.cpp:command_path)
  // clang-format on

  node::MpcNodeRuntime::Components components;
  components.mpc = node->mpc_.get();
  const WBMpcNode* absl_nonnull self = node.get();
  components.resetTargetTrajectories =
      [self](const SystemObservation& observation) {
        return self->resetTargetTrajectories(observation);
      };
  components.motionManager = node->motionManager_;
  const std::shared_ptr<MpcParameterUpdaterModule> parameterUpdater =
      node->parameterUpdater_;
  // The tuning GUI's whole task file, typed (operator/mpc_parameters,
  // humanoid_nmpc/humanoid_mpc_config/README.md, "Live updates"), applied by
  // the updater before the next solve.
  components.parameterUpdateSink =
      [parameterUpdater](const mpc_config::MpcParameterUpdate& update) {
        parameterUpdater->enqueueParameterUpdate(update);
      };
  // The visualization publisher, on the node's bus: the MPC's own models,
  // copied before the solver thread runs, and the state manifold the problem,
  // the rollout and the policies share, along which it interpolates the
  // policies' states (retained: interface_ outlives visualization_), the
  // nominal state of the DCM's pendulum, and the solver's own problem, whose
  // terms the post-solve observer evaluates on the solver thread (retained:
  // mpc_ outlives visualization_).
  components.attachVisualization =
      visualization::VisualizationPublisher::MakeBusAttacher(
          visualization::VisualizationModel{
              .taskFile = files.taskFile,
              .urdfFile = files.urdfFile,
              .pinocchioInterface = &interface.getPinocchioInterface(),
              .mpcRobotModel = &interface.getMpcRobotModel(),
              .stateManifold = interface.getStateManifold().get(),
              .nominalState = interface.getInitialState(),
              .solverProblem =
                  &node->mpc_->getSolverPtr()->getOcpDefinitions().front()},
          std::move(options.visualization), &node->visualization_);

  node::MpcNodeRuntime::Config config;
  config.dimensions = node->dimensions();
  config.robotName = interface.modelSettings().robotName;
  const std::string taskFileIdentity = configFileIdentity(files.taskFile);
  config.taskFileIdentity = taskFileIdentity;
  config.mpcDesiredFrequency = interface.mpcSettings().mpcDesiredFrequency_;
  config.solverThread = std::move(options.solverThread);
  config.walkingCommandTimeout = options.walkingCommandTimeout;
  // An observation of another layout (equal dimensions, other joints) is
  // refused and not solved.
  config.modelFingerprint = modelLayoutFingerprint(interface.modelSettings());
  ASSIGN_OR_RETURN(node->runtime_, node::MpcNodeRuntime::Create(
                                       std::move(bus), std::move(components),
                                       std::move(config)));
  const ipc::ModelDimensions dimensions = node->dimensions();
  LOG(INFO)
      << "[WBMpcNode] The whole-body MPC is built (" << dimensions.stateDim
      << " states, " << dimensions.inputDim
      << " inputs); its parameter updater applies the RELOAD_HOT fields of "
      << taskFileIdentity << " live.";
  return node;
}

WBMpcNode::~WBMpcNode() {
  stop();
}

absl::Status WBMpcNode::start() {
  RETURN_IF_ERROR(visualization_->start());
  return runtime_->start();
}

void WBMpcNode::stop() {
  // The visualization first: its thread publishes on the runtime's bus.
  if (visualization_ != nullptr) visualization_->stop();
  if (runtime_ != nullptr) runtime_->stop();
}

TargetTrajectories WBMpcNode::resetTargetTrajectories(
    const SystemObservation& observation) const {
  return wbMpcResetTargetTrajectories(observation,
                                      interface_->getMpcRobotModel(),
                                      interface_->getPinocchioInterface());
}

ipc::ModelDimensions WBMpcNode::dimensions() const {
  const WBAccelMpcRobotModel<scalar_t>& model = interface_->getMpcRobotModel();
  return {.stateDim = model.getStateDim(),
          .inputDim = model.getInputDim(),
          .numModes = kNumModeNumbers};
}

}  // namespace ocs2::humanoid
