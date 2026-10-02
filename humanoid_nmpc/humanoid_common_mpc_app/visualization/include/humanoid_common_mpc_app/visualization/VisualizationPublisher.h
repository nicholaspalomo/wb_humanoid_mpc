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

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <google/protobuf/message.h>

#include <ocs2_mpc/CommandData.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_oc/oc_data/PerformanceIndex.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "humanoid_common_mpc_app/visualization/PolicySnapshot.h"
#include "humanoid_common_mpc_app/visualization/RobotStateDecoder.h"
#include "humanoid_common_mpc_app/visualization/SceneBuilder.h"
#include "humanoid_common_mpc_app/visualization/TelemetryBuilder.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_common_mpc_app/visualization/VisualizationModel.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"
#include "humanoid_mpc_msgs/visualization_scene.pb.h"
#include "robot_core/TripleBuffer.h"
#include "robot_ipc/Bus.h"
#include "robot_realtime/SpscQueue.h"

namespace ocs2::humanoid::visualization {

/** How a VisualizationPublisher runs; the defaults suit the MPC node. What the robot needs is in the task file instead. */
struct VisualizationPublisherOptions {
  /** robot/state samples waiting for the visualization thread; more are dropped. */
  size_t robotStateQueueCapacity = 64;
  /** How often the visualization thread wakes to take the samples and to see whether a scene is due. */
  absl::Duration pollPeriod = absl::Milliseconds(5);
  /** The thread's name and the CPU cores it may run on (empty: any). */
  std::string threadName = "mpc_visualize";
  std::vector<int> cores;
  /** Added to the thread's nice value (0 leaves it). */
  int niceIncrement = 10;
  /**
   * The measured robot is drawn from the latest robot/state sample while one has arrived within this long, and from
   * the latest observation's state otherwise (a robot process that publishes no samples, such as the dummy sim).
   */
  absl::Duration robotStateTimeout = absl::Milliseconds(500);
};

/** What a VisualizationPublisher has done; every counter only grows. */
struct VisualizationPublisherStatistics {
  uint64_t observationsSet = 0;
  uint64_t policiesSet = 0;
  /** pushRobotState() calls whose sample was queued. */
  uint64_t robotStatesQueued = 0;
  /** pushRobotState() calls whose sample was dropped because the queue was full. */
  uint64_t robotStatesDropped = 0;
  /** Samples that did not decode (RobotStateDecoder) and were not plotted. */
  uint64_t robotStatesRejected = 0;
  uint64_t telemetryPublished = 0;
  uint64_t scenesPublished = 0;
  /** Messages the publish function refused (for the bus: its queue to the IO thread was full). */
  uint64_t publishFailures = 0;
  /** Scenes or series whose computation threw. */
  uint64_t buildFailures = 0;
};

/**
 * The visualization publisher of the MPC process (humanoid_nmpc/docs/distributed_runtime/README.md, "Visualization with
 * Rerun"), which replaces HumanoidVisualizer, EquivalentContactCornerForcesVisualizer and PinocchioTelemetryPublisher.
 * It publishes, through the bus, what the Rerun bridge draws:
 *   - viz/scene (SceneBuilder) at the task file's rerunSceneFrequency at most, when something new arrived;
 *   - viz/telemetry (TelemetryBuilder), one message per robot/state sample.
 *
 * THREADS. Everything is computed on a thread of its own, start() to stop(), which runs on the time-sharing scheduler
 * with its nice value raised by Options::niceIncrement, so that it yields to the solver. It is fed through lock-free
 * mailboxes that never block the feeding thread and never wait for the visualization:
 *   - setObservation(): the latest observation (a triple buffer: a newer one replaces one not yet drawn);
 *   - setPolicy(): the latest solution and its command (a triple buffer);
 *   - pushRobotState(): every robot/state sample, through a bounded queue: when the visualization falls behind, new
 *     samples are dropped and counted, never queued without bound.
 * Each mailbox has exactly one feeding thread: setObservation() and setPolicy() are typically called from the solver
 * thread (MpcServer::Hooks::postSolveObserver), pushRobotState() from the bus's IO thread (subscribeRobotState()).
 * They copy their argument and return; the copy reuses the mailbox's storage, so it allocates only when the shape of
 * what is copied changes.
 *
 * Attaching it in the MPC node (see README.md of this package):
 *
 *   ASSIGN_OR_RETURN(std::unique_ptr<VisualizationPublisher> visualization,
 *                    VisualizationPublisher::Create({taskFile, urdfFile, &pinocchioInterface, &mpcRobotModel}, *bus));
 *   RETURN_IF_ERROR(visualization->subscribeRobotState(*bus));  // before bus->start()
 *   RETURN_IF_ERROR(visualization->start());
 *   hooks.postSolveObserver = visualization->postSolveObserver();
 *   ... MpcServer::Create(*bus, mpc, resetTarget, config, hooks); bus->start(); server->start();
 *   // Shutdown: server->stop(); visualization->stop(); bus->stop();
 */
class VisualizationPublisher {
 public:
  /** MpcServer::PostSolveObserver: called on the solver thread after every policy the MPC node publishes. */
  using PostSolveObserver =
      std::function<void(const CommandData& command, const PrimalSolution& solution, const PerformanceIndex& performance)>;

  /** Sends one message on a topic; Bus::publish() in the MPC node. Called on the visualization thread. */
  using PublishFunction = std::function<absl::Status(absl::string_view topic, const google::protobuf::Message& message)>;

  using Options = VisualizationPublisherOptions;
  using Statistics = VisualizationPublisherStatistics;

  /**
   * Loads the task file's visualization keys (logging them), builds the scene and telemetry builders and the
   * mailboxes. The thread starts with start().
   *
   * @return the errors of loadVisualizationConfig(), SceneBuilder::Create() and TelemetryBuilder::Create();
   *         InvalidArgument for an empty publish function, a zero queue capacity or a non-positive poll period.
   */
  static absl::StatusOr<std::unique_ptr<VisualizationPublisher>> Create(const VisualizationModel& model,
                                                                        PublishFunction publish,
                                                                        Options options = Options());

  /** Create() publishing on `bus`, which must outlive the publisher. */
  static absl::StatusOr<std::unique_ptr<VisualizationPublisher>> Create(const VisualizationModel& model,
                                                                        robot::ipc::Bus& bus,
                                                                        Options options = Options());

  /**
   * node::MpcNodeRuntime::VisualizationAttacher (humanoid_common_mpc_app/node): called with the node's bus before the
   * bus starts, it returns the observer the node's MpcServer calls after every policy it publishes.
   */
  using BusAttacher = std::function<absl::StatusOr<PostSolveObserver>(robot::ipc::Bus& bus)>;

  /**
   * The attacher of an MPC node's publisher: on the node's bus, Create() into `*publisher`, subscribeRobotState(), and
   * postSolveObserver(). The node starts the publisher with itself (start() before the runtime's), and stops it before
   * the runtime, which owns the bus. The objects `model` points to are read only while the attacher runs (in
   * MpcNodeRuntime::Create()). `*publisher` must outlive the node's MpcServer, whose observer reaches it: the node
   * declares it before its runtime, so that it is destroyed after it. The attacher fails with the errors of Create() and
   * subscribeRobotState(), and with InvalidArgument for a null `publisher`.
   */
  static BusAttacher MakeBusAttacher(VisualizationModel model, Options options, std::unique_ptr<VisualizationPublisher>* publisher);

  /** Stops the thread. */
  ~VisualizationPublisher();
  VisualizationPublisher(const VisualizationPublisher&) = delete;
  VisualizationPublisher& operator=(const VisualizationPublisher&) = delete;

  /**
   * Subscribes to robot/state on `bus` (every sample, kAll) and pushes each into the queue, on the bus's IO thread.
   * Call it before the bus starts. The subscription is safe to outlive the publisher: it then does nothing.
   */
  absl::Status subscribeRobotState(robot::ipc::Bus& bus);

  /** Starts the visualization thread. Once: FailedPrecondition after a start(). */
  absl::Status start();

  /** Stops and joins the visualization thread (within about a poll period). Idempotent. */
  void stop();

  // ------------------------------------------------------------------ feeding (one thread per mailbox, never blocks)

  /** The latest MPC observation. */
  void setObservation(const SystemObservation& observation);

  /** The latest MPC solution and the command it was solved for. The controller is not copied. */
  void setPolicy(const CommandData& command, const PrimalSolution& solution);

  /** Queues one robot/state sample; false when the queue is full (the sample is dropped and counted). */
  bool pushRobotState(const humanoid_mpc_msgs::RobotStateSample& sample);

  /**
   * The observer for MpcServer::Hooks::postSolveObserver (and the return value of MpcNodeRuntime's
   * VisualizationAttacher): setPolicy() with the published solution and setObservation() with the observation it was
   * solved from. Must not outlive the publisher.
   */
  PostSolveObserver postSolveObserver();

  // ------------------------------------------------------------------ any thread

  Statistics statistics() const;
  const VisualizationConfig& config() const { return config_; }

 private:
  /** A slot of the robot/state queue: `valid` is false when the message did not convert. */
  struct QueuedRobotState {
    msgs::RobotStateSample sample;
    bool valid = false;
  };

  /** What the bus callback reaches the publisher through, cleared by the destructor. */
  struct CallbackGuard {
    absl::Mutex mutex;
    VisualizationPublisher* publisher ABSL_GUARDED_BY(mutex) = nullptr;
  };

  VisualizationPublisher(const VisualizationModel& model,
                         VisualizationConfig config,
                         PublishFunction publish,
                         Options options,
                         std::unique_ptr<SceneBuilder> sceneBuilder,
                         std::unique_ptr<TelemetryBuilder> telemetryBuilder);

  // Visualization thread.
  void run();
  void poll();
  void processRobotState(const QueuedRobotState& queued);
  void publishScene(std::chrono::nanoseconds now);
  void publishMessage(absl::string_view topic, const google::protobuf::Message& message, std::atomic<uint64_t>* published);
  const SystemObservation* latestObservation() const;
  const PolicySnapshot* latestPolicy() const;

  const VisualizationConfig config_;
  const Options options_;
  const PublishFunction publish_;
  const std::chrono::nanoseconds scenePeriod_;
  const size_t stateDim_;
  const size_t inputDim_;
  std::shared_ptr<CallbackGuard> guard_;

  // ---- Mailboxes.
  robot::TripleBuffer<SystemObservation> observationMailbox_;
  robot::TripleBuffer<PolicySnapshot> policyMailbox_;
  robot::realtime::SpscQueue<QueuedRobotState> robotStateQueue_;

  // ---- Visualization thread.
  std::unique_ptr<SceneBuilder> sceneBuilder_;
  std::unique_ptr<TelemetryBuilder> telemetryBuilder_;
  RobotStateDecoder decoder_;
  DecodedRobotState decoded_;
  bool hasObservation_ = false;
  bool hasPolicy_ = false;
  uint64_t policyVersion_ = 0;
  msgs::RobotStateSample latestRobotState_;
  bool hasRobotState_ = false;
  std::chrono::nanoseconds latestRobotStateArrival_{0};
  bool sceneInputsChanged_ = false;
  std::chrono::nanoseconds nextSceneTime_{0};
  humanoid_mpc_msgs::VisualizationScene scene_;

  // ---- Lifecycle.
  std::atomic<bool> stopRequested_{false};
  absl::Mutex lifecycleMutex_;
  std::thread thread_ ABSL_GUARDED_BY(lifecycleMutex_);
  bool started_ ABSL_GUARDED_BY(lifecycleMutex_) = false;

  // ---- Statistics: each written on one thread, read anywhere.
  std::atomic<uint64_t> observationsSet_{0};
  std::atomic<uint64_t> policiesSet_{0};
  std::atomic<uint64_t> robotStatesQueued_{0};
  std::atomic<uint64_t> robotStatesRejected_{0};
  std::atomic<uint64_t> telemetryPublished_{0};
  std::atomic<uint64_t> scenesPublished_{0};
  std::atomic<uint64_t> publishFailures_{0};
  std::atomic<uint64_t> buildFailures_{0};
};

}  // namespace ocs2::humanoid::visualization
