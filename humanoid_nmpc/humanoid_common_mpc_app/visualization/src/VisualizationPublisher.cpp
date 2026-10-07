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

#include "humanoid_common_mpc_app/visualization/VisualizationPublisher.h"

#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.pb.h"
#include "robot_ipc/Delivery.h"
#include "robot_realtime/PeriodicTimer.h"
#include "robot_realtime/RealtimeThread.h"

namespace ocs2::humanoid::visualization {

namespace {

/** Raises the calling thread's nice value by `increment` (the thread alone: Linux applies setpriority() to a TID). */
absl::Status raiseCurrentThreadNiceValue(int increment) {
  const id_t threadId = static_cast<id_t>(::syscall(SYS_gettid));
  errno = 0;
  const int current = ::getpriority(PRIO_PROCESS, threadId);
  if (current == -1 && errno != 0) {
    return absl::ErrnoToStatus(errno, "getpriority()");
  }
  if (::setpriority(PRIO_PROCESS, threadId, current + increment) != 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("setpriority() to the nice value ", current + increment));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<VisualizationPublisher>> VisualizationPublisher::Create(const VisualizationModel& model,
                                                                                       PublishFunction publish,
                                                                                       Options options) {
  if (publish == nullptr) {
    return absl::InvalidArgumentError("the visualization publisher needs a publish function.");
  }
  if (options.robotStateQueueCapacity == 0) {
    return absl::InvalidArgumentError("the visualization publisher's robot/state queue must hold at least one sample.");
  }
  if (options.pollPeriod <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError("the visualization publisher's poll period must be positive.");
  }
  RETURN_IF_ERROR(checkVisualizationModel(model));
  ASSIGN_OR_RETURN(VisualizationConfig config, loadVisualizationConfig(model.taskFile, model.mpcRobotModel->modelSettings));
  ASSIGN_OR_RETURN(std::unique_ptr<SceneBuilder> sceneBuilder, SceneBuilder::Create(model, config));
  ASSIGN_OR_RETURN(std::unique_ptr<TelemetryBuilder> telemetryBuilder, TelemetryBuilder::Create(model, config));
  LOG(INFO) << "[VisualizationPublisher] " << describeVisualizationConfig(config);
  return absl::WrapUnique(new VisualizationPublisher(model, std::move(config), std::move(publish), std::move(options),
                                                     std::move(sceneBuilder), std::move(telemetryBuilder)));
}

absl::StatusOr<std::unique_ptr<VisualizationPublisher>> VisualizationPublisher::Create(const VisualizationModel& model,
                                                                                       robot::ipc::Bus& bus,
                                                                                       Options options) {
  return Create(
      model, [&bus](absl::string_view topic, const google::protobuf::Message& message) { return bus.publish(topic, message); },
      std::move(options));
}

VisualizationPublisher::BusAttacher VisualizationPublisher::MakeBusAttacher(
    VisualizationModel model, Options options, std::unique_ptr<VisualizationPublisher>* absl_nullable publisher) {
  return [model = std::move(model), options = std::move(options), publisher](robot::ipc::Bus& bus) -> absl::StatusOr<PostSolveObserver> {
    if (publisher == nullptr) {
      return absl::InvalidArgumentError("the visualization attacher needs a place to keep its publisher.");
    }
    ASSIGN_OR_RETURN(*publisher, Create(model, bus, options));
    RETURN_IF_ERROR((*publisher)->subscribeRobotState(bus));
    return (*publisher)->postSolveObserver();
  };
}

VisualizationPublisher::VisualizationPublisher(const VisualizationModel& model,
                                               VisualizationConfig config,
                                               PublishFunction publish,
                                               Options options,
                                               std::unique_ptr<SceneBuilder> sceneBuilder,
                                               std::unique_ptr<TelemetryBuilder> telemetryBuilder)
    : config_(std::move(config)),
      options_(std::move(options)),
      publish_(std::move(publish)),
      scenePeriod_(std::chrono::nanoseconds(static_cast<int64_t>(1.0e9 / config_.sceneFrequency))),
      stateDim_(model.mpcRobotModel->getStateDim()),
      inputDim_(model.mpcRobotModel->getInputDim()),
      guard_(std::make_shared<CallbackGuard>()),
      robotStateQueue_(options_.robotStateQueueCapacity),
      sceneBuilder_(std::move(sceneBuilder)),
      telemetryBuilder_(std::move(telemetryBuilder)),
      decoder_(model.mpcRobotModel->modelSettings) {
  absl::MutexLock lock(&guard_->mutex);
  guard_->publisher = this;
}

VisualizationPublisher::~VisualizationPublisher() {
  {
    absl::MutexLock lock(&guard_->mutex);
    guard_->publisher = nullptr;
  }
  stop();
}

absl::Status VisualizationPublisher::subscribeRobotState(robot::ipc::Bus& bus) {
  return bus.subscribe<humanoid_mpc_msgs::RobotStateSample>(ipc::topics::kRobotState, robot::ipc::Delivery::kAll,
                                                            [guard = guard_](const humanoid_mpc_msgs::RobotStateSample& sample) {
                                                              absl::MutexLock lock(&guard->mutex);
                                                              if (guard->publisher != nullptr) {
                                                                guard->publisher->pushRobotState(sample);
                                                              }
                                                            });
}

absl::Status VisualizationPublisher::start() {
  absl::MutexLock lock(&lifecycleMutex_);
  if (started_) {
    return absl::FailedPreconditionError("the visualization publisher was started already.");
  }
  started_ = true;
  thread_ = std::thread([this] { run(); });
  return absl::OkStatus();
}

void VisualizationPublisher::stop() {
  stopRequested_.store(true);
  absl::MutexLock lock(&lifecycleMutex_);
  if (thread_.joinable()) {
    thread_.join();
  }
}

void VisualizationPublisher::setObservation(const SystemObservation& observation) {
  observationMailbox_.writeSlot() = observation;
  observationMailbox_.publishWrite();
  ++observationsSet_;
}

void VisualizationPublisher::setPolicy(const CommandData& command, const PrimalSolution& solution) {
  policyMailbox_.writeSlot().assign(command, solution);
  policyMailbox_.publishWrite();
  ++policiesSet_;
}

bool VisualizationPublisher::pushRobotState(const humanoid_mpc_msgs::RobotStateSample& sample) {
  const bool queued =
      robotStateQueue_.tryPushInPlace([&sample](QueuedRobotState& slot) { slot.valid = msgs::FromProto(sample, &slot.sample).ok(); });
  if (queued) {
    ++robotStatesQueued_;
  }
  return queued;
}

VisualizationPublisher::PostSolveObserver VisualizationPublisher::postSolveObserver() {
  return [this](const CommandData& command, const PrimalSolution& solution, const PerformanceIndex& /*performance*/) {
    setPolicy(command, solution);
    setObservation(command.mpcInitObservation_);
    return absl::OkStatus();
  };
}

VisualizationPublisher::Statistics VisualizationPublisher::statistics() const {
  Statistics statistics;
  statistics.observationsSet = observationsSet_.load(std::memory_order_relaxed);
  statistics.policiesSet = policiesSet_.load(std::memory_order_relaxed);
  statistics.robotStatesQueued = robotStatesQueued_.load(std::memory_order_relaxed);
  statistics.robotStatesDropped = robotStateQueue_.droppedCount();
  statistics.robotStatesRejected = robotStatesRejected_.load(std::memory_order_relaxed);
  statistics.telemetryPublished = telemetryPublished_.load(std::memory_order_relaxed);
  statistics.scenesPublished = scenesPublished_.load(std::memory_order_relaxed);
  statistics.publishFailures = publishFailures_.load(std::memory_order_relaxed);
  statistics.buildFailures = buildFailures_.load(std::memory_order_relaxed);
  return statistics;
}

// ---------------------------------------------------------------------------------------------------------------------
// Visualization thread
// ---------------------------------------------------------------------------------------------------------------------

void VisualizationPublisher::run() {
  robot::realtime::RealtimeThreadConfig threadConfig;
  threadConfig.name = options_.threadName;
  threadConfig.cores = options_.cores;
  if (const absl::Status configured = robot::realtime::configureCurrentThread(threadConfig); !configured.ok()) {
    LOG(WARNING) << "[VisualizationPublisher] the visualization thread runs without part of its configuration: " << configured;
  }
  if (options_.niceIncrement != 0) {
    if (const absl::Status niced = raiseCurrentThreadNiceValue(options_.niceIncrement); !niced.ok()) {
      LOG(WARNING) << "[VisualizationPublisher] the visualization thread keeps its priority: " << niced;
    }
  }
  robot::realtime::PeriodicTimer timer(absl::ToChronoNanoseconds(options_.pollPeriod), robot::realtime::OverrunPolicy::kSkipMissedPeriods);
  timer.start();
  while (!stopRequested_.load(std::memory_order_acquire)) {
    timer.waitForNextPeriod();
    if (stopRequested_.load(std::memory_order_acquire)) {
      break;
    }
    poll();
  }
}

const SystemObservation* absl_nullable VisualizationPublisher::latestObservation() const {
  return hasObservation_ ? &observationMailbox_.readSlot() : nullptr;
}

const PolicySnapshot* absl_nullable VisualizationPublisher::latestPolicy() const {
  return hasPolicy_ ? &policyMailbox_.readSlot() : nullptr;
}

void VisualizationPublisher::poll() {
  if (observationMailbox_.acquireRead()) {
    hasObservation_ = static_cast<size_t>(observationMailbox_.readSlot().state.size()) == stateDim_;
    sceneInputsChanged_ = true;
  }
  if (policyMailbox_.acquireRead()) {
    hasPolicy_ = isConsistentPlan(policyMailbox_.readSlot(), stateDim_, inputDim_);
    ++policyVersion_;
    sceneInputsChanged_ = true;
  }
  while (robotStateQueue_.tryPopInPlace([this](const QueuedRobotState& queued) { processRobotState(queued); })) {
  }
  const std::chrono::nanoseconds now = robot::realtime::monotonicNow();
  if (sceneInputsChanged_ && now >= nextSceneTime_) {
    publishScene(now);
  }
}

void VisualizationPublisher::publishMessage(absl::string_view topic,
                                            const google::protobuf::Message& message,
                                            std::atomic<uint64_t>* absl_nonnull published) {
  const absl::Status status = publish_(topic, message);
  if (status.ok()) {
    ++*published;
    return;
  }
  ++publishFailures_;
  LOG_EVERY_N_SEC(WARNING, 5.0) << "[VisualizationPublisher] " << topic << " was not published: " << status;
}

void VisualizationPublisher::processRobotState(const QueuedRobotState& queued) {
  if (!queued.valid) {
    ++robotStatesRejected_;
    return;
  }
  if (const absl::Status decoded = decoder_.decode(queued.sample, &decoded_); !decoded.ok()) {
    ++robotStatesRejected_;
    LOG_EVERY_N_SEC(WARNING, 5.0) << "[VisualizationPublisher] a robot/state sample is not plotted: " << decoded;
    return;
  }
  latestRobotState_ = queued.sample;
  hasRobotState_ = true;
  latestRobotStateArrival_ = robot::realtime::monotonicNow();
  sceneInputsChanged_ = true;
  // The visualization thread's isolation point: the builders call Pinocchio's algorithms and OCS2, which throw for a
  // model or a state they do not accept, and a sample that cannot be drawn is counted and logged, never fatal.
  try {  // NOLINT(exceptions): Pinocchio and OCS2 throw; see the comment above.
    publishMessage(ipc::topics::kVizTelemetry, telemetryBuilder_->build(decoded_, latestObservation(), latestPolicy()),
                   &telemetryPublished_);
  } catch (const std::exception& e) {  // NOLINT(exceptions): the boundary of the try above.
    ++buildFailures_;
    LOG_EVERY_N_SEC(WARNING, 5.0) << "[VisualizationPublisher] the telemetry of a robot/state sample failed: " << e.what();
  }
}

void VisualizationPublisher::publishScene(std::chrono::nanoseconds now) {
  SceneInputs inputs;
  inputs.observation = latestObservation();
  inputs.policy = latestPolicy();
  inputs.policyVersion = policyVersion_;
  const bool robotStateIsRecent = hasRobotState_ && now - latestRobotStateArrival_ <= absl::ToChronoNanoseconds(options_.robotStateTimeout);
  inputs.robotState = robotStateIsRecent ? &latestRobotState_ : nullptr;
  if (inputs.observation == nullptr && inputs.robotState == nullptr) {
    return;
  }
  sceneInputsChanged_ = false;
  nextSceneTime_ += scenePeriod_;
  if (nextSceneTime_ <= now) {
    nextSceneTime_ = now + scenePeriod_;
  }
  // The isolation point of processRobotState(), for the scene.
  try {  // NOLINT(exceptions): Pinocchio and OCS2 throw; see processRobotState().
    const absl::Status built = sceneBuilder_->build(inputs, &scene_);
    if (!built.ok()) {
      ++buildFailures_;
      LOG_EVERY_N_SEC(WARNING, 5.0) << "[VisualizationPublisher] no scene: " << built;
      return;
    }
  } catch (const std::exception& e) {  // NOLINT(exceptions): the boundary of the try above.
    ++buildFailures_;
    LOG_EVERY_N_SEC(WARNING, 5.0) << "[VisualizationPublisher] the scene failed: " << e.what();
    return;
  }
  publishMessage(ipc::topics::kVizScene, scene_, &scenesPublished_);
}

}  // namespace ocs2::humanoid::visualization
