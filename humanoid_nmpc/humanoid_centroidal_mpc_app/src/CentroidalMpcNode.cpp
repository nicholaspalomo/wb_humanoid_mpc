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

#include "humanoid_centroidal_mpc_app/CentroidalMpcNode.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcParameterUpdater.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcResetTarget.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/parameter_update/CommandLimitsReloaders.h"
#include "humanoid_mpc_config/mpc_parameter_update.nproto.h"

namespace ocs2::humanoid {

absl::StatusOr<std::unique_ptr<CentroidalMpcNode>> CentroidalMpcNode::Create(const node::MpcFiles& files,
                                                                             std::unique_ptr<robot::ipc::Bus> bus,
                                                                             Options options) {
  std::unique_ptr<CentroidalMpcNode> node = absl::WrapUnique(new CentroidalMpcNode());
  ASSIGN_OR_RETURN(node->interface_, CentroidalMpcInterface::Create(files.taskFile, files.urdfFile, files.referenceFile));
  CentroidalMpcInterface& interface = *node->interface_;

  node->mpc_ = std::make_unique<SqpMpc>(interface.mpcSettings(), interface.sqpSettings(), interface.getOptimalControlProblem(),
                                        interface.getInitializer());

  // Reference inputs must be laid out like the OCP input: in basis-vector mode that is [lambda, joint velocities], so the
  // effective model (the decorator when active) is used rather than the wrench-space model.
  const MpcRobotModelBase<scalar_t>& effectiveModel = interface.getEffectiveMpcRobotModel();

  // LINT.IfChange(mpc_wiring)
  // Reference and motion management for the procedural MPC.
  ASSIGN_OR_RETURN(node->targetCalculator_, CentroidalMpcTargetTrajectoriesCalculator::Create(
                                                files.referenceFile, effectiveModel, interface.getPinocchioInterface(),
                                                interface.getCentroidalModelInfo(), interface.mpcSettings().timeHorizon_));
  CentroidalMpcTargetTrajectoriesCalculator* absl_nonnull calculator = node->targetCalculator_.get();
  // The commanded base height stands on the ground the reference manager applied in this solve, so it follows a hot
  // reload of terrain_height (TargetTrajectoriesCalculatorBase::setTerrainHeightSource).
  calculator->setTerrainHeightSource(
      [referenceManager = interface.getSwitchedModelReferenceManagerPtr()]() { return referenceManager->getAppliedTerrainHeight(); });
  const ProceduralMpcMotionManager::VelocityTargetToTargetTrajectories targetTrajectoriesFunction =
      [calculator](const vector4_t& velocityTarget, scalar_t initTime, scalar_t /*finalTime*/, const vector_t& initState) {
        return calculator->commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState);
      };
  ASSIGN_OR_RETURN(node->motionManager_,
                   ProceduralMpcMotionManager::Create(files.gaitFile, files.referenceFile, interface.getSwitchedModelReferenceManagerPtr(),
                                                      effectiveModel, targetTrajectoriesFunction));
  // A reset of the MPC (MPC_BASE::reset()) resets the command path with it: the motion manager itself, and through this
  // hook the target calculator behind targetTrajectoriesFunction, whose filters are its state.
  node->motionManager_->setResetHook([calculator]() { calculator->reset(); });

  SolverBase& solver = *node->mpc_->getSolverPtr();
  solver.setReferenceManager(interface.getReferenceManagerPtr());
  solver.addSynchronizedModule(node->motionManager_);
  // Online contact planning (contact_schedule_source: "contact_planner"): the planner module feeds mode schedules and
  // footholds to the reference manager and, like the other synchronized modules, runs before every solve.
  if (const std::shared_ptr<ContactPlannerModule> contactPlannerModule = interface.getContactPlannerModulePtr()) {
    solver.addSynchronizedModule(contactPlannerModule);
  }

  // Hot reloading of the MPC parameters, wired by the one function every node that runs the centroidal MPC uses; the
  // reloaders hand the command limits of a reloaded reference file to their consumers built above, so that the Command
  // Limits tab of the remote control changes them on the running node.
  ASSIGN_OR_RETURN(node->parameterUpdater_,
                   makeCentroidalMpcParameterUpdater(node->mpc_.get(), interface, files.taskFile, files.referenceFile,
                                                     makeCommandLimitsReloaders(calculator, node->motionManager_)));
  solver.addSynchronizedModule(node->parameterUpdater_);
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/CentroidalClosedLoopDriver.cpp:centroidal_mpc_node_wiring, //humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopDriver.cpp:command_path)
  // clang-format on

  node::MpcNodeRuntime::Components components;
  components.mpc = node->mpc_.get();
  const CentroidalMpcNode* absl_nonnull self = node.get();
  components.resetTargetTrajectories = [self](const SystemObservation& observation) { return self->resetTargetTrajectories(observation); };
  components.motionManager = node->motionManager_;
  const std::shared_ptr<MpcParameterUpdaterModule> parameterUpdater = node->parameterUpdater_;
  // The tuning GUI's whole task file and contact planner's file, typed (operator/mpc_parameters,
  // humanoid_nmpc/humanoid_mpc_config/README.md, "Live updates"), applied by the updater before the next solve.
  components.parameterUpdateSink = [parameterUpdater](const mpc_config::MpcParameterUpdate& update) {
    parameterUpdater->enqueueParameterUpdate(update);
  };
  // The contact planner's reference manager under contact planning; null otherwise.
  components.contactPlanningReferenceManager =
      // NOLINTNEXTLINE(rtti): OCS2 hands out a ReferenceManagerInterface; only a ContactPlanningReferenceManager plans.
      std::dynamic_pointer_cast<const ContactPlanningReferenceManager>(interface.getReferenceManagerPtr());
  // The visualization publisher, on the node's bus: the MPC's own models (the effective one, whose inputs the policies
  // carry), copied before the solver thread runs.
  components.attachVisualization = visualization::VisualizationPublisher::MakeBusAttacher(
      visualization::VisualizationModel{.taskFile = files.taskFile,
                                        .urdfFile = files.urdfFile,
                                        .pinocchioInterface = &interface.getPinocchioInterface(),
                                        .mpcRobotModel = &effectiveModel},
      std::move(options.visualization), &node->visualization_);

  node::MpcNodeRuntime::Config config;
  config.dimensions = node->dimensions();
  config.robotName = interface.modelSettings().robotName;
  // The configuration a parameter update must be of: robot_name alone does not tell the centroidal G1 from the
  // whole-body one (checkMpcParameterUpdate()).
  config.taskFileIdentity = configFileIdentity(files.taskFile);
  config.mpcDesiredFrequency = interface.mpcSettings().mpcDesiredFrequency_;
  config.solverThread = std::move(options.solverThread);
  ASSIGN_OR_RETURN(node->runtime_, node::MpcNodeRuntime::Create(std::move(bus), std::move(components), std::move(config)));
  const ipc::ModelDimensions dimensions = node->dimensions();
  LOG(INFO) << "[CentroidalMpcNode] The centroidal MPC is built (" << dimensions.stateDim << " states, " << dimensions.inputDim
            << " inputs, " << (interface.usesContactPlanning() ? "contact planner" : "gait schedule") << ").";
  return node;
}

CentroidalMpcNode::~CentroidalMpcNode() {
  stop();
}

absl::Status CentroidalMpcNode::start() {
  RETURN_IF_ERROR(visualization_->start());
  return runtime_->start();
}

void CentroidalMpcNode::stop() {
  // The visualization first: its thread publishes on the runtime's bus.
  if (visualization_ != nullptr) visualization_->stop();
  if (runtime_ != nullptr) runtime_->stop();
}

TargetTrajectories CentroidalMpcNode::resetTargetTrajectories(const SystemObservation& observation) const {
  return centroidalMpcResetTargetTrajectories(observation, interface_->getCentroidalModelInfo(), interface_->getEffectiveMpcRobotModel(),
                                              interface_->getPinocchioInterface());
}

ipc::ModelDimensions CentroidalMpcNode::dimensions() const {
  const MpcRobotModelBase<scalar_t>& effectiveModel = interface_->getEffectiveMpcRobotModel();
  return {.stateDim = effectiveModel.getStateDim(), .inputDim = effectiveModel.getInputDim(), .numModes = node::kNumHumanoidModes};
}

}  // namespace ocs2::humanoid
