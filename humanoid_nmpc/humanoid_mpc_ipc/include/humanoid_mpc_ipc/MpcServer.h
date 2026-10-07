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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_mpc/MPC_BASE.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_msgs/mpc_observation.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "humanoid_mpc_msgs/mpc_solver_status.pb.h"
#include "humanoid_mpc_msgs/mpc_status.pb.h"
#include "humanoid_mpc_msgs/viewer_annotations.pb.h"
#include "robot_ipc/Bus.h"
#include "robot_realtime/RealtimeThread.h"

namespace ocs2::humanoid::ipc {

/**
 * The MPC side of the network MPC link (humanoid_nmpc/docs/distributed_runtime/README.md, "The MPC link"): the solve
 * loop that the MRT joint controllers' solverWorker() runs in process, driven by the bus. It subscribes to
 * robot/mpc_observation (the newest one wins) and, on a thread of its own:
 *
 *   1. waits for an observation newer than the one it last solved from (after a failed attempt it retries at once on
 *      the newest one, as solverWorker() does);
 *   2. resets the MPC from that observation when the robot asks for it, fully when the robot's full-reset counter
 *      exceeds the full resets served, and when its own MpcResetSupervisor asks for it after a failed solve; the first
 *      reset, before the first solve, is a full one, and so is the reset after the robot process restarted (its
 *      sequence numbers or its counters went backwards). A reset is MPC_BASE::reset() or resetSolver(), then the
 *      reference manager's target trajectories from the caller's ResetTargetTrajectoriesFunction, as
 *      MPC_MRT_Interface::resetMpcNode() / resetMpcSolver() do;
 *   3. solves (MPC_BASE::run()) and reports the result to its supervisor, which backs off and declares the MPC unhealthy
 *      exactly as in process;
 *   4. on success, publishes mpc/policy: the solution as MPC_MRT_Interface::copyToBuffer() assembles it, cut to
 *      mpc.solution_time_window (trimToSolutionWindow()), stamped with the robot's reset counters it has served and its
 *      solver status, with the annotations of Hooks::annotationsProvider; then calls Hooks::postSolveObserver;
 *   5. after every attempt, publishes mpc/status;
 *   6. waits for the rest of the period when Config::mpcDesiredFrequency is positive.
 *
 * A failed solve publishes no policy, and the robot learns of it from mpc/status; the robot drops the policies solved
 * before a reset it asked for by their counters.
 *
 * THREADS. The MPC (its solver, reference manager and synchronized modules) is driven from the solver thread only, from
 * start() to stop(). The IO thread of the bus only decodes observations into a mailbox. Publishing goes through
 * Bus::publish(). No ROS.
 */
class MpcServer {
 public:
  /**
   * The target trajectories to reset the MPC to from an observation (the controllers' currentObservationToResetTrajectory()),
   * or why there are none: the reset then fails like a failed solve, and is retried.
   */
  using ResetTargetTrajectoriesFunction = std::function<absl::StatusOr<TargetTrajectories>(const SystemObservation& observation)>;
  /**
   * Fills MpcPolicy.annotations from the solution about to be sent, on the solver thread. Display only: when it returns
   * an error the policy goes out without annotations.
   */
  using AnnotationsProvider = std::function<absl::Status(
      const CommandData& command, const PrimalSolution& solution, humanoid_mpc_msgs::ViewerAnnotations* absl_nonnull annotations)>;
  /**
   * Called after each policy is published, on the solver thread; the visualization publisher copies what it needs. An
   * error it returns is logged.
   */
  using PostSolveObserver =
      std::function<absl::Status(const CommandData& command, const PrimalSolution& solution, const PerformanceIndex& performance)>;

  struct Config {
    /** The MPC's model: an observation of other dimensions or with a mode outside it is rejected. */
    ModelDimensions dimensions;
    /** [Hz] Solves per second at most; <= 0: one solve per new observation, as fast as they arrive. */
    scalar_t mpcDesiredFrequency = -1.0;
    /** The failure policy of the solves. */
    MpcResetSupervisor::Config resetSupervisor;
    /**
     * Name, CPU cores and SCHED_FIFO priority of the solver thread, set by robot::realtime::configureCurrentThread()
     * when the thread starts. A setting it cannot be given is skipped and the others are still applied; one warning
     * names every one skipped. The memory of the MPC process is not locked by default (MemoryLock::kNone): the solver
     * allocates as it runs, and mlockall()'s MCL_FUTURE would make those allocations fail at a finite RLIMIT_MEMLOCK.
     * MemoryLock::kLockProcess locks the memory of the whole process, every thread of it.
     */
    robot::realtime::RealtimeThreadConfig solverThread{.name = "mpc_solver", .cores = {}, .memoryLock = robot::realtime::MemoryLock::kNone};
  };

  /** Optional; empty ones are skipped. An error they return is logged and the solve loop goes on. */
  struct Hooks {
    AnnotationsProvider annotationsProvider;
    PostSolveObserver postSolveObserver;
  };

  /** What the server has done so far; every counter only grows. */
  struct Statistics {
    uint64_t observationsReceived = 0;
    /** Repeated sequence numbers. */
    uint64_t observationsSkipped = 0;
    /** Not decodable, or of other dimensions than Config::dimensions. */
    uint64_t observationsRejected = 0;
    uint64_t solveAttempts = 0;
    uint64_t failedAttempts = 0;
    uint64_t policiesPublished = 0;
    uint64_t statusesPublished = 0;
    uint64_t fullResets = 0;
    uint64_t solverResets = 0;
    /** Robot processes seen: 1 after the first observation, one more for each restart of the robot process. */
    uint64_t robotSessions = 0;
    /** The robot's counters the server has served (MpcPolicy.resets_served, full_resets_served). */
    uint64_t robotResetsServed = 0;
    uint64_t robotFullResetsServed = 0;
    bool healthy = true;
  };

  /**
   * A server on `bus`, which must not be running yet; the caller starts it, and start() starts the solver thread.
   * `mpc` must outlive the server and must not be used by anything else while it runs.
   */
  static absl::StatusOr<std::unique_ptr<MpcServer>> Create(
      robot::ipc::Bus& bus, MPC_BASE& mpc, ResetTargetTrajectoriesFunction resetTargetTrajectories, Config config, Hooks hooks = Hooks());

  /** Stops the solver thread. */
  ~MpcServer();
  MpcServer(const MpcServer&) = delete;
  MpcServer& operator=(const MpcServer&) = delete;

  /** Starts the solver thread, which waits for the first observation. Once: FailedPrecondition after a start(). */
  absl::Status start();

  /** Stops and joins the solver thread after the attempt in progress. Idempotent; the server cannot be started again. */
  void stop();

  /** Thread-safe. */
  Statistics statistics() const;

  /** The server's own supervisor: its solves, failures and back-off. */
  const MpcResetSupervisor& resetSupervisor() const { return supervisor_; }

  /** MpcSolverStatus.server_instance of everything this server publishes: random, nonzero, drawn once per server. */
  uint64_t serverInstance() const { return serverInstance_; }

 private:
  /** What the callbacks on the bus reach the server through, cleared by the destructor. */
  struct CallbackGuard {
    absl::Mutex mutex;
    MpcServer* absl_nullable server ABSL_GUARDED_BY(mutex) = nullptr;
  };

  /** The newest observation as the solver thread takes it from the mailbox. */
  struct Snapshot {
    SystemObservation observation;
    MpcResetSupervisor::ResetCounters requests;
    bool newSession = false;
    /** Restarts of the robot process seen when the observation was taken (mailboxSessions_). */
    uint64_t sessions = 0;
  };

  MpcServer(robot::ipc::Bus& bus, MPC_BASE& mpc, ResetTargetTrajectoriesFunction resetTargetTrajectories, Config config, Hooks hooks);
  absl::Status registerOnBus();

  // IO thread.
  void onObservation(const humanoid_mpc_msgs::MpcObservation& message);

  // Solver thread.
  void runSolverLoop();
  /** Blocks until there is an observation (a new one if `requireNew`) or stop(); false on stop(). */
  bool awaitObservation(bool requireNew);
  bool hasWork() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  /**
   * True after stop(), or when the robot has asked for a reset since the attempt's observation was taken: its counters
   * or its process (a restart) differ from the snapshot's. Ends the back-off wait early. A request the failed attempt
   * did not manage to serve is not a new one, and waits out the back-off with the failure.
   */
  bool robotRequestedResetSinceSnapshot() const;
  /**
   * Serves the resets the snapshot and the supervisor ask for. Internal when the reset cannot be made (the reset
   * function returned an error or the MPC threw); the requests then stay unserved, and the attempt fails.
   */
  absl::Status serveResets();
  absl::Status solve();
  absl::Status buildPolicy();
  void fillSolverStatus(const absl::Status& attempt, double solveTimeMs, humanoid_mpc_msgs::MpcSolverStatus* absl_nonnull status) const;
  void publishPolicy(const absl::Status& attempt, double solveTimeMs);
  void publishStatus(const absl::Status& attempt, double solveTimeMs);
  /** Sleeps until `deadline` or stop(). */
  void sleepUntil(absl::Time deadline);

  robot::ipc::Bus& bus_;
  MPC_BASE& mpc_;
  const ResetTargetTrajectoriesFunction resetTargetTrajectories_;
  const Config config_;
  const Hooks hooks_;
  MpcResetSupervisor supervisor_;
  // MpcSolverStatus.server_instance of every policy and status: this server, drawn once.
  const uint64_t serverInstance_;
  std::shared_ptr<CallbackGuard> guard_;

  // ---- IO thread -> solver thread.
  mutable absl::Mutex mutex_;
  SystemObservation mailboxObservation_ ABSL_GUARDED_BY(mutex_);
  MpcResetSupervisor::ResetCounters mailboxRequests_ ABSL_GUARDED_BY(mutex_);
  uint64_t mailboxSequence_ ABSL_GUARDED_BY(mutex_) = 0;
  uint64_t mailboxVersion_ ABSL_GUARDED_BY(mutex_) = 0;
  bool hasObservation_ ABSL_GUARDED_BY(mutex_) = false;
  bool newSession_ ABSL_GUARDED_BY(mutex_) = false;
  // Restarts of the robot process seen so far; only grows, unlike newSession_, which the solver thread takes.
  uint64_t mailboxSessions_ ABSL_GUARDED_BY(mutex_) = 0;
  uint64_t receivedSinceStatus_ ABSL_GUARDED_BY(mutex_) = 0;
  uint64_t skippedSinceStatus_ ABSL_GUARDED_BY(mutex_) = 0;
  bool stopRequested_ ABSL_GUARDED_BY(mutex_) = false;
  // Set by the solver thread for hasWork().
  bool requireNewObservation_ ABSL_GUARDED_BY(mutex_) = false;
  uint64_t takenVersion_ ABSL_GUARDED_BY(mutex_) = 0;

  // ---- Solver thread.
  absl::Mutex lifecycleMutex_;
  std::thread solverThread_ ABSL_GUARDED_BY(lifecycleMutex_);
  bool started_ ABSL_GUARDED_BY(lifecycleMutex_) = false;
  Snapshot snapshot_;
  bool firstResetDone_ = false;
  MpcResetSupervisor::ResetCounters robotServed_;
  uint64_t solveCount_ = 0;
  size_t consecutiveCrashes_ = 0;
  CommandData command_;
  PrimalSolution solution_;
  PerformanceIndex performance_;
  humanoid_mpc_msgs::MpcPolicy policyMessage_;
  humanoid_mpc_msgs::MpcStatus statusMessage_;

  // ---- Statistics: written on one thread, read anywhere.
  std::atomic<uint64_t> observationsReceived_{0};
  std::atomic<uint64_t> observationsSkipped_{0};
  std::atomic<uint64_t> observationsRejected_{0};
  std::atomic<uint64_t> solveAttempts_{0};
  std::atomic<uint64_t> failedAttempts_{0};
  std::atomic<uint64_t> policiesPublished_{0};
  std::atomic<uint64_t> statusesPublished_{0};
  std::atomic<uint64_t> fullResets_{0};
  std::atomic<uint64_t> solverResets_{0};
  std::atomic<uint64_t> robotSessions_{0};
  std::atomic<uint64_t> robotResetsServed_{0};
  std::atomic<uint64_t> robotFullResetsServed_{0};
};

}  // namespace ocs2::humanoid::ipc
