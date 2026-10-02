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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_wb_mpc_app/WBMpcNode.h"

#include <utility>

#include "absl/log/log.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_wb_mpc/mrt/WBMpcResetTarget.h"

namespace ocs2::humanoid {

absl::StatusOr<std::unique_ptr<WBMpcNode>> WBMpcNode::Create(const node::MpcFiles& files,
                                                             std::unique_ptr<robot::ipc::Bus> bus,
                                                             Options options) {
  std::unique_ptr<WBMpcNode> node(new WBMpcNode());
  ASSIGN_OR_RETURN(node->interface_, WBMpcInterface::Create(files.taskFile, files.urdfFile, files.referenceFile));
  WBMpcInterface& interface = *node->interface_;

  node->mpc_ = std::make_unique<SqpMpc>(interface.mpcSettings(), interface.sqpSettings(), interface.getOptimalControlProblem(),
                                        interface.getInitializer());

  // LINT.IfChange(mpc_wiring)
  // Reference and motion management for the procedural MPC.
  node->targetCalculator_ = std::make_unique<WBMpcTargetTrajectoriesCalculator>(files.referenceFile, interface.getMpcRobotModel(),
                                                                                interface.mpcSettings().timeHorizon_);
  WBMpcTargetTrajectoriesCalculator* calculator = node->targetCalculator_.get();
  // The commanded base height stands on the ground the reference manager applied in this solve, so it follows a hot
  // reload of terrainHeight (TargetTrajectoriesCalculatorBase::setTerrainHeightSource).
  calculator->setTerrainHeightSource(
      [referenceManager = interface.getSwitchedModelReferenceManagerPtr()]() { return referenceManager->getAppliedTerrainHeight(); });
  const ProceduralMpcMotionManager::VelocityTargetToTargetTrajectories targetTrajectoriesFunction =
      [calculator](const vector4_t& velocityTarget, scalar_t initTime, scalar_t /*finalTime*/, const vector_t& initState) {
        return calculator->commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState);
      };
  node->motionManager_ =
      std::make_shared<ProceduralMpcMotionManager>(files.gaitFile, files.referenceFile, interface.getSwitchedModelReferenceManagerPtr(),
                                                   interface.getMpcRobotModel(), targetTrajectoriesFunction);
  // A reset of the MPC (MPC_BASE::reset()) resets the command path with it: the motion manager itself, and through this
  // hook the target calculator behind targetTrajectoriesFunction, whose filters are its state.
  node->motionManager_->setResetHook([calculator]() { calculator->reset(); });

  SolverBase& solver = *node->mpc_->getSolverPtr();
  solver.setReferenceManager(interface.getReferenceManagerPtr());
  solver.addSynchronizedModule(node->motionManager_);
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/WholeBodyClosedLoopDriver.cpp:whole_body_mpc_node_wiring, //humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopDriver.cpp:command_path)
  // clang-format on

  node::MpcNodeRuntime::Components components;
  components.mpc = node->mpc_.get();
  const WBMpcNode* self = node.get();
  components.resetTargetTrajectories = [self](const SystemObservation& observation) { return self->resetTargetTrajectories(observation); };
  components.motionManager = node->motionManager_;
  // The visualization publisher, on the node's bus: the MPC's own models, copied before the solver thread runs.
  components.attachVisualization = visualization::VisualizationPublisher::MakeBusAttacher(
      visualization::VisualizationModel{.taskFile = files.taskFile,
                                        .urdfFile = files.urdfFile,
                                        .pinocchioInterface = &interface.getPinocchioInterface(),
                                        .mpcRobotModel = &interface.getMpcRobotModel()},
      std::move(options.visualization), &node->visualization_);

  node::MpcNodeRuntime::Config config;
  config.dimensions = node->dimensions();
  config.mpcDesiredFrequency = interface.mpcSettings().mpcDesiredFrequency_;
  config.solverThread = std::move(options.solverThread);
  ASSIGN_OR_RETURN(node->runtime_, node::MpcNodeRuntime::Create(std::move(bus), std::move(components), std::move(config)));
  const ipc::ModelDimensions dimensions = node->dimensions();
  LOG(INFO) << "[WBMpcNode] The whole-body MPC is built (" << dimensions.stateDim << " states, " << dimensions.inputDim
            << " inputs). It has no MPC parameter updater: operator/mpc_parameters is not subscribed.";
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

TargetTrajectories WBMpcNode::resetTargetTrajectories(const SystemObservation& observation) const {
  return wbMpcResetTargetTrajectories(observation, interface_->getMpcRobotModel(), interface_->getPinocchioInterface());
}

ipc::ModelDimensions WBMpcNode::dimensions() const {
  const WBAccelMpcRobotModel<scalar_t>& model = interface_->getMpcRobotModel();
  return {.stateDim = model.getStateDim(), .inputDim = model.getInputDim(), .numModes = node::kNumHumanoidModes};
}

}  // namespace ocs2::humanoid
