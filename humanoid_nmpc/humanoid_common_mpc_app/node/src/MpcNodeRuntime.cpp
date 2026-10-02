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

#include "humanoid_common_mpc_app/node/MpcNodeRuntime.h"

#include <functional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/ThreadAffinity.h"
#include "humanoid_common_mpc_app/node/ViewerAnnotations.h"
#include "humanoid_common_mpc_app/node/WalkingVelocityCommandConversions.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "humanoid_mpc_msgs/yaml_document.pb.h"
#include "robot_ipc/Delivery.h"

namespace ocs2::humanoid::node {
namespace {

constexpr double kLogPeriodSeconds = 5.0;

}  // namespace

robot::realtime::RealtimeThreadConfig defaultSolverThreadConfig(int realtimePriority) {
  robot::realtime::RealtimeThreadConfig config;
  config.name = "mpc_solver";
  config.priority = realtimePriority;
  config.cores = getDefaultCoreAllocation().mpcCores;
  config.memoryLock = robot::realtime::MemoryLock::kNone;
  return config;
}

absl::Status applyWalkingVelocityCommand(const humanoid_mpc_msgs::WalkingVelocityCommand& message,
                                         ProceduralMpcMotionManager& motionManager) {
  ASSIGN_OR_RETURN(const WalkingVelocityCommand command, walkingVelocityCommandFromProto(message));
  motionManager.setAndScaleVelocityCommand(command);
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<MpcNodeRuntime>> MpcNodeRuntime::Create(std::unique_ptr<robot::ipc::Bus> bus,
                                                                       Components components,
                                                                       Config config) {
  if (bus == nullptr) {
    return absl::InvalidArgumentError("MpcNodeRuntime: no bus");
  }
  if (components.mpc == nullptr) {
    return absl::InvalidArgumentError("MpcNodeRuntime: Components::mpc is null");
  }
  if (!components.resetTargetTrajectories) {
    return absl::InvalidArgumentError("MpcNodeRuntime: Components::resetTargetTrajectories is empty");
  }
  if (components.motionManager == nullptr) {
    return absl::InvalidArgumentError("MpcNodeRuntime: Components::motionManager is null");
  }
  std::unique_ptr<MpcNodeRuntime> runtime(new MpcNodeRuntime(std::move(bus), std::make_shared<Counters>()));
  RETURN_IF_ERROR(runtime->subscribeOperatorInputs(components));

  ipc::MpcServer::Hooks hooks;
  // On the solver thread, after the solve: the planner's targets and the command that solve started from.
  hooks.annotationsProvider = [motionManager = components.motionManager, planner = components.contactPlanningReferenceManager](
                                  const CommandData& /*command*/, const PrimalSolution& /*solution*/,
                                  humanoid_mpc_msgs::ViewerAnnotations* annotations) {
    const WalkingVelocityCommand scaledCommand = motionManager->getScaledWalkingVelocityCommand();
    if (planner != nullptr) {
      const feet_array_t<TargetContactPose> poses = planner->getTargetContactPoses();
      fillViewerAnnotations(&poses, scaledCommand, annotations);
    } else {
      fillViewerAnnotations(/*targetContactPoses=*/nullptr, scaledCommand, annotations);
    }
  };
  if (components.attachVisualization) {
    ASSIGN_OR_RETURN(hooks.postSolveObserver, components.attachVisualization(*runtime->bus_));
  }

  ipc::MpcServer::Config serverConfig;
  serverConfig.dimensions = config.dimensions;
  serverConfig.mpcDesiredFrequency = config.mpcDesiredFrequency;
  serverConfig.resetSupervisor = config.resetSupervisor;
  serverConfig.solverThread = config.solverThread;
  ASSIGN_OR_RETURN(runtime->server_, ipc::MpcServer::Create(*runtime->bus_, *components.mpc, std::move(components.resetTargetTrajectories),
                                                            std::move(serverConfig), std::move(hooks)));
  return runtime;
}

MpcNodeRuntime::MpcNodeRuntime(std::unique_ptr<robot::ipc::Bus> bus, std::shared_ptr<Counters> counters)
    : bus_(std::move(bus)), counters_(std::move(counters)) {}

MpcNodeRuntime::~MpcNodeRuntime() {
  stop();
}

absl::Status MpcNodeRuntime::subscribeOperatorInputs(const Components& components) {
  const std::function<void(const humanoid_mpc_msgs::WalkingVelocityCommand&)> onVelocityCommand =
      [motionManager = components.motionManager, counters = counters_](const humanoid_mpc_msgs::WalkingVelocityCommand& message) {
        counters->velocityCommandsReceived.fetch_add(1);
        const absl::Status applied = applyWalkingVelocityCommand(message, *motionManager);
        if (!applied.ok()) {
          counters->velocityCommandsRejected.fetch_add(1);
          LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds) << "[MpcNode] Ignoring a walking velocity command: " << applied.message();
        }
      };
  RETURN_IF_ERROR(bus_->subscribe<humanoid_mpc_msgs::WalkingVelocityCommand>(ipc::topics::kOperatorWalkingVelocityCommand,
                                                                             robot::ipc::Delivery::kLatest, onVelocityCommand));
  if (!components.parameterUpdateSink) {
    return absl::OkStatus();
  }
  const std::function<void(const humanoid_mpc_msgs::YamlDocument&)> onParameters =
      [sink = components.parameterUpdateSink, counters = counters_](const humanoid_mpc_msgs::YamlDocument& message) {
        counters->parameterUpdatesReceived.fetch_add(1);
        sink(message.yaml());
      };
  return bus_->subscribe<humanoid_mpc_msgs::YamlDocument>(ipc::topics::kOperatorMpcParameters, robot::ipc::Delivery::kLatest, onParameters);
}

absl::Status MpcNodeRuntime::start() {
  RETURN_IF_ERROR(bus_->start());
  return server_->start();
}

void MpcNodeRuntime::stop() {
  if (server_ != nullptr) server_->stop();
  if (bus_ != nullptr) bus_->stop();
}

MpcNodeRuntime::Statistics MpcNodeRuntime::statistics() const {
  Statistics statistics;
  statistics.velocityCommandsReceived = counters_->velocityCommandsReceived.load();
  statistics.velocityCommandsRejected = counters_->velocityCommandsRejected.load();
  statistics.parameterUpdatesReceived = counters_->parameterUpdatesReceived.load();
  statistics.server = server_->statistics();
  return statistics;
}

}  // namespace ocs2::humanoid::node
