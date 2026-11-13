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

#include "legged_common_mpc_app/node/MpcNodeRuntime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "legged_common_mpc/command/NormalizedWalkingVelocityCommand.h"
#include "legged_common_mpc/command/WalkingVelocityCommand.h"
#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc/common/ThreadAffinity.h"
#include "legged_common_mpc/config/OperatorPayloadChecks.h"
#include "legged_common_mpc/contact_planning/TargetContactPose.h"
#include "legged_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "legged_common_mpc_app/node/ViewerAnnotations.h"
#include "legged_common_mpc_app/node/WalkingVelocityCommandConversions.h"
#include "legged_mpc_config/mpc_parameter_update.nproto.h"
#include "legged_mpc_config/mpc_parameter_update.nproto.pb.h"
#include "legged_mpc_config/mpc_parameter_update.pb.h"
#include "legged_mpc_ipc/MpcServer.h"
#include "legged_mpc_ipc/Topics.h"
#include "legged_mpc_msgs/viewer_annotations.pb.h"
#include "legged_mpc_msgs/walking_velocity_command.pb.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/Delivery.h"
#include "robot_realtime/RealtimeThread.h"

namespace ocs2::legged::node {
namespace {

constexpr double kLogPeriodSeconds = 5.0;
// How many times per walking command timeout the watchdog looks at the age of
// the last command, and the shortest period it looks at: the stop command comes
// at most a quarter of the timeout late.
constexpr int64_t kWalkingCommandChecksPerTimeout = 4;
constexpr absl::Duration kMinWalkingCommandCheckPeriod = absl::Milliseconds(10);

}  // namespace

robot::realtime::RealtimeThreadConfig defaultSolverThreadConfig(
    int realtimePriority) {
  robot::realtime::RealtimeThreadConfig config;
  config.name = "mpc_solver";
  config.priority = realtimePriority;
  config.cores = getDefaultCoreAllocation().mpcCores;
  config.memoryLock = robot::realtime::MemoryLock::kNone;
  return config;
}

absl::Status applyWalkingVelocityCommand(
    const legged_mpc_msgs::WalkingVelocityCommand& message,
    ProceduralMpcMotionManager& motionManager) {
  ASSIGN_OR_RETURN(const NormalizedWalkingVelocityCommand command,
                   walkingVelocityCommandFromProto(message));
  motionManager.setAndScaleVelocityCommand(command);
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<MpcNodeRuntime>> MpcNodeRuntime::Create(
    std::unique_ptr<robot::ipc::Bus> bus,
    Components components,
    Config config) {
  if (bus == nullptr) {
    return absl::InvalidArgumentError("MpcNodeRuntime: no bus");
  }
  if (components.mpc == nullptr) {
    return absl::InvalidArgumentError(
        "MpcNodeRuntime: Components::mpc is null");
  }
  if (!components.resetTargetTrajectories) {
    return absl::InvalidArgumentError(
        "MpcNodeRuntime: Components::resetTargetTrajectories is empty");
  }
  if (components.motionManager == nullptr) {
    return absl::InvalidArgumentError(
        "MpcNodeRuntime: Components::motionManager is null");
  }
  if (config.walkingCommandTimeout < absl::ZeroDuration()) {
    return absl::InvalidArgumentError(
        "MpcNodeRuntime: Config::walkingCommandTimeout is negative");
  }
  std::unique_ptr<MpcNodeRuntime> runtime = absl::WrapUnique(
      new MpcNodeRuntime(std::move(bus), std::make_shared<Counters>()));
  RETURN_IF_ERROR(runtime->subscribeOperatorInputs(
      components.motionManager, std::move(components.parameterUpdateSink),
      std::move(config.robotName), std::move(config.taskFileIdentity)));
  if (config.walkingCommandTimeout > absl::ZeroDuration()) {
    RETURN_IF_ERROR(runtime->addWalkingCommandWatchdog(
        components.motionManager, config.walkingCommandTimeout));
  }

  ipc::MpcServer::Hooks hooks;
  // On the solver thread, after the solve: the planner's targets and the
  // command that solve started from.
  hooks.annotationsProvider =
      [motionManager = components.motionManager,
       planner = components.contactPlanningReferenceManager](
          const CommandData& /*command*/, const PrimalSolution& /*solution*/,
          legged_mpc_msgs::ViewerAnnotations* absl_nonnull annotations) {
        const WalkingVelocityCommand scaledCommand =
            motionManager->getScaledWalkingVelocityCommand();
        if (planner != nullptr) {
          const BipedFeetArray<TargetContactPose> desPoseFeetRtWorld =
              planner->getDesPoseFeetRtWorld();
          fillViewerAnnotations(&desPoseFeetRtWorld, scaledCommand,
                                annotations);
        } else {
          fillViewerAnnotations(/*desPoseFeetRtWorld=*/nullptr, scaledCommand,
                                annotations);
        }
        return absl::OkStatus();
      };
  if (components.attachVisualization) {
    ASSIGN_OR_RETURN(hooks.postSolveObserver,
                     components.attachVisualization(*runtime->bus_));
  }
  // On the solver thread, before the solve: the observation it starts from,
  // whose input carries the measured joint velocities the motion manager's
  // exact base twist needs (ProceduralMpcMotionManager::observe()).
  // LINT.IfChange(observation_hand_over)
  hooks.preSolveObserver = [motionManager = components.motionManager](
                               const SystemObservation& observation) {
    motionManager->observe(observation);
  };
  // LINT.ThenChange(//legged_nmpc/legged_mpc_validation/src/closed_loop/ClosedLoopDriver.cpp:observation_hand_over)

  ipc::MpcServer::Config serverConfig;
  serverConfig.dimensions = config.dimensions;
  serverConfig.mpcDesiredFrequency = config.mpcDesiredFrequency;
  serverConfig.resetSupervisor = config.resetSupervisor;
  serverConfig.solverThread = std::move(config.solverThread);
  serverConfig.modelFingerprint = config.modelFingerprint;
  ASSIGN_OR_RETURN(
      runtime->server_,
      ipc::MpcServer::Create(runtime->bus_.get(), components.mpc,
                             std::move(components.resetTargetTrajectories),
                             std::move(serverConfig), std::move(hooks)));
  return runtime;
}

MpcNodeRuntime::MpcNodeRuntime(std::unique_ptr<robot::ipc::Bus> bus,
                               std::shared_ptr<Counters> counters)
    : bus_(std::move(bus)), counters_(std::move(counters)) {}

MpcNodeRuntime::~MpcNodeRuntime() {
  stop();
}

absl::Status MpcNodeRuntime::subscribeOperatorInputs(
    std::shared_ptr<ProceduralMpcMotionManager> motionManager,
    ParameterUpdateSink parameterUpdateSink,
    std::string robotName,
    std::string taskFileIdentity) {
  RETURN_IF_ERROR(bus_->subscribe<legged_mpc_msgs::WalkingVelocityCommand>(
      ipc::topics::kOperatorWalkingVelocityCommand,
      robot::ipc::Delivery::kLatest,
      [motionManager = std::move(motionManager), counters = counters_](
          const legged_mpc_msgs::WalkingVelocityCommand& message) {
        counters->velocityCommandsReceived.fetch_add(1);
        const absl::Status applied =
            applyWalkingVelocityCommand(message, *motionManager);
        if (!applied.ok()) {
          counters->velocityCommandsRejected.fetch_add(1);
          LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
              << "[MpcNode] Ignoring a walking velocity command: "
              << applied.message();
          return;
        }
        counters->lastWalkingCommandTime = std::chrono::steady_clock::now();
        counters->prevCmdPosBaseRtSurfaceEwrtWorldZ =
            message.cmd_pos_base_rt_surface_ewrt_world_z();
        counters->walkingCommandTimedOut = false;
      }));
  if (parameterUpdateSink == nullptr) {
    return absl::OkStatus();
  }
  return bus_->subscribe<legged_mpc_config::MpcParameterUpdate>(
      ipc::topics::kOperatorMpcParameters, robot::ipc::Delivery::kLatest,
      [sink = std::move(parameterUpdateSink), counters = counters_,
       robotName = std::move(robotName),
       taskFileIdentity = std::move(taskFileIdentity)](
          const legged_mpc_config::MpcParameterUpdate& message) mutable {
        counters->parameterUpdatesReceived.fetch_add(1);
        if (const absl::Status refused =
                checkMpcParameterUpdate(message, robotName, taskFileIdentity);
            !refused.ok()) {
          counters->parameterUpdatesRejected.fetch_add(1);
          LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
              << "[MpcNode] Ignoring an MPC parameter update: "
              << refused.message() << "; nothing was applied.";
          return;
        }
        mpc_config::MpcParameterUpdate update;
        const absl::Status converted = mpc_config::FromProto(message, &update);
        if (!converted.ok()) {
          counters->parameterUpdatesRejected.fetch_add(1);
          LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
              << "[MpcNode] Ignoring an MPC parameter update that does not "
                 "convert: "
              << converted.message();
          return;
        }
        sink(update);
      });
}

absl::Status MpcNodeRuntime::addWalkingCommandWatchdog(
    std::shared_ptr<ProceduralMpcMotionManager> motionManager,
    absl::Duration timeout) {
  const absl::Duration period = std::max(
      timeout / kWalkingCommandChecksPerTimeout, kMinWalkingCommandCheckPeriod);
  const std::chrono::steady_clock::duration maxAge =
      absl::ToChronoNanoseconds(timeout);
  return bus_->addPeriodicCallback(
      period, [motionManager = std::move(motionManager), counters = counters_,
               maxAge, timeout]() {
        if (!counters->lastWalkingCommandTime.has_value() ||
            counters->walkingCommandTimedOut) {
          return;
        }
        if (std::chrono::steady_clock::now() -
                *counters->lastWalkingCommandTime <=
            maxAge) {
          return;
        }
        legged_mpc_msgs::WalkingVelocityCommand stop;
        stop.set_cmd_pos_base_rt_surface_ewrt_world_z(
            counters->prevCmdPosBaseRtSurfaceEwrtWorldZ);
        // The last command converted, so its pelvis height does; the zero
        // velocities always do.
        applyWalkingVelocityCommand(stop, *motionManager).IgnoreError();
        counters->walkingCommandTimedOut = true;
        counters->walkingCommandTimeouts.fetch_add(1);
        LOG(WARNING) << "[MpcNode] No walking command for " << timeout
                     << ": the robot is commanded to stop.";
      });
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
  statistics.velocityCommandsReceived =
      counters_->velocityCommandsReceived.load();
  statistics.velocityCommandsRejected =
      counters_->velocityCommandsRejected.load();
  statistics.parameterUpdatesReceived =
      counters_->parameterUpdatesReceived.load();
  statistics.parameterUpdatesRejected =
      counters_->parameterUpdatesRejected.load();
  statistics.walkingCommandTimeouts = counters_->walkingCommandTimeouts.load();
  statistics.server = server_->statistics();
  return statistics;
}

}  // namespace ocs2::legged::node
