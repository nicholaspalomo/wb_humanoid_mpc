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
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include <ocs2_mpc/MPC_BASE.h>

#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_ipc/MpcServer.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "robot_ipc/Bus.h"
#include "robot_realtime/RealtimeThread.h"

namespace ocs2::humanoid::node {

/** The modes of a humanoid MPC: the contact combinations of its two feet (MotionPhaseDefinition.h), 0 to STANCE. */
inline constexpr size_t kNumHumanoidModes = 4;

/**
 * The solver thread of an MPC node: named "mpc_solver", pinned to the MPC cores of ThreadAffinity.h
 * (getDefaultCoreAllocation().mpcCores), SCHED_FIFO at `realtimePriority` when it is positive (the --realtime_priority
 * of the node; 0 keeps it on the time-sharing scheduler), and without locking the memory of the process, which the
 * solver allocates from as it runs (MpcServer::Config::solverThread).
 */
robot::realtime::RealtimeThreadConfig defaultSolverThreadConfig(int realtimePriority);

/**
 * What the MPC node does with an operator/walking_velocity_command it receives: the command of
 * walkingVelocityCommandFromProto() goes to `motionManager`'s setAndScaleVelocityCommand(). A message that conversion
 * refuses (a value that is not finite) changes nothing and is returned as its error. The lockstep closed loop
 * (humanoid_nmpc/humanoid_mpc_validation) hands the GUI's messages to its motion manager through this too.
 */
absl::Status applyWalkingVelocityCommand(const humanoid_mpc_msgs::WalkingVelocityCommand& message,
                                         ProceduralMpcMotionManager& motionManager);

/**
 * The formulation-independent part of an MPC node (humanoid_centroidal_mpc_node, humanoid_wb_mpc_node; see
 * humanoid_nmpc/docs/distributed_runtime/README.md, "Processes"): the node's bus, the ipc::MpcServer that solves the
 * robot's observations on it, and the operator inputs the MPC takes from the bus:
 *
 *   - operator/walking_velocity_command (latest delivery) -> Components::motionManager's setAndScaleVelocityCommand(),
 *     on the bus's IO thread; a command with a value that is not finite is rejected, logged and counted;
 *   - operator/mpc_parameters (latest delivery: an edit the solver has not applied yet is replaced by a newer one) ->
 *     Components::parameterUpdateSink, on the IO thread; the MpcParameterUpdaterModule it feeds applies the document
 *     in the preSolverRun() of the next solve.
 *
 * With every policy the MpcServer publishes, it sends MpcPolicy.annotations (ViewerAnnotations.h): the contact planner's
 * target contact poses and the motion manager's scaled velocity command, and then calls the visualization publisher's
 * post-solve observer (Components::attachVisualization).
 *
 * THREADS. Create() and start() run on the caller's thread; the handlers above run on the bus's IO thread, the reset
 * target, the annotations and the post-solve observer on the solver thread. The MPC, its reference manager and its
 * synchronized modules are driven from the solver thread only, from start() to stop().
 *
 * LIFETIME. Everything Components names must outlive the runtime. The runtime owns the bus; stop() (and the destructor)
 * stops the solver thread first, then the bus.
 */
class MpcNodeRuntime {
 public:
  /**
   * SEAM for the visualization publisher (humanoid_nmpc/docs/distributed_runtime/README.md, "Visualization with
   * Rerun"), which is written separately: called once by Create() with the node's bus before the bus starts, so that
   * the publisher can subscribe to robot/state and publish viz/scene and viz/telemetry from a thread of its own. It
   * returns the observer the MpcServer calls after every policy it publishes, on the solver thread, which copies what
   * it needs and returns at once (MpcServer::PostSolveObserver); an empty observer is allowed. The observer is
   * destroyed with the server, before the bus. An error fails Create().
   */
  using VisualizationAttacher = std::function<absl::StatusOr<ipc::MpcServer::PostSolveObserver>(robot::ipc::Bus& bus)>;

  /** What the node is made of; see the class comment. */
  struct Components {
    /** The MPC (an SqpMpc with its reference manager and synchronized modules). Required. */
    MPC_BASE* mpc = nullptr;
    /** The formulation's reset target (centroidalMpcResetTargetTrajectories(), wbMpcResetTargetTrajectories()). Required. */
    ipc::MpcServer::ResetTargetTrajectoriesFunction resetTargetTrajectories;
    /** The motion manager the velocity commands go to, also a synchronized module of `mpc`. Required. */
    std::shared_ptr<ProceduralMpcMotionManager> motionManager;
    /** Where the documents of operator/mpc_parameters go; empty: the topic is not subscribed (no parameter updater). */
    std::function<void(std::string yamlText)> parameterUpdateSink;
    /** The contact planner's reference manager, for the target contact patches; null: no patches are sent. */
    std::shared_ptr<const ContactPlanningReferenceManager> contactPlanningReferenceManager;
    /** Optional; see VisualizationAttacher. */
    VisualizationAttacher attachVisualization;
  };

  struct Config {
    /** The MPC's model, against which every observation is checked (the effective model of the OCP). */
    ipc::ModelDimensions dimensions;
    /** [Hz] mpc.mpcDesiredFrequency; <= 0: one solve per new observation, as fast as they come. */
    scalar_t mpcDesiredFrequency = -1.0;
    /** The failure policy of the solves. */
    MpcResetSupervisor::Config resetSupervisor;
    /** See defaultSolverThreadConfig(). */
    robot::realtime::RealtimeThreadConfig solverThread = defaultSolverThreadConfig(/*realtimePriority=*/0);
  };

  /** What the node has done so far; every counter only grows. */
  struct Statistics {
    uint64_t velocityCommandsReceived = 0;
    /** Not finite, so not handed to the motion manager. */
    uint64_t velocityCommandsRejected = 0;
    uint64_t parameterUpdatesReceived = 0;
    ipc::MpcServer::Statistics server;
  };

  /**
   * The runtime on `bus`, which must not be running yet and must have a node name to publish the policies as (the MPC
   * node's is "mpc"). Registers the subscriptions and the MpcServer; start() starts them. InvalidArgument for a missing
   * required component, and the errors of MpcServer::Create() and of the visualization attacher.
   */
  static absl::StatusOr<std::unique_ptr<MpcNodeRuntime>> Create(std::unique_ptr<robot::ipc::Bus> bus, Components components, Config config);

  /** stop(). */
  ~MpcNodeRuntime();
  MpcNodeRuntime(const MpcNodeRuntime&) = delete;
  MpcNodeRuntime& operator=(const MpcNodeRuntime&) = delete;

  /** Starts the bus and the solver thread, which waits for the first observation. Once. */
  absl::Status start();

  /** Stops the solver thread after the attempt in progress, then the bus. Idempotent. */
  void stop();

  /** Thread-safe. */
  Statistics statistics() const;

  /** The node's bus, e.g. for a tool that publishes on it alongside the MPC. */
  robot::ipc::Bus& bus() { return *bus_; }

 private:
  /** The counters of the IO thread's handlers, shared with them so that they never reach a destroyed runtime. */
  struct Counters {
    std::atomic<uint64_t> velocityCommandsReceived{0};
    std::atomic<uint64_t> velocityCommandsRejected{0};
    std::atomic<uint64_t> parameterUpdatesReceived{0};
  };

  MpcNodeRuntime(std::unique_ptr<robot::ipc::Bus> bus, std::shared_ptr<Counters> counters);

  /** Subscribes the motion manager and the parameter sink to their topics. */
  absl::Status subscribeOperatorInputs(const Components& components);

  std::unique_ptr<robot::ipc::Bus> bus_;
  std::shared_ptr<Counters> counters_;
  std::unique_ptr<ipc::MpcServer> server_;
};

}  // namespace ocs2::humanoid::node
