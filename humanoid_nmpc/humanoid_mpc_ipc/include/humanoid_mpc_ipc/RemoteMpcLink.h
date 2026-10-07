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
#include <memory>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/MRT_BASE.h"
#include "ocs2_mpc/SystemObservation.h"

#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_msgs/mpc_observation.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "humanoid_mpc_msgs/mpc_solver_status.pb.h"
#include "humanoid_mpc_msgs/mpc_status.pb.h"
#include "humanoid_mpc_msgs/viewer_annotations.nproto.h"
#include "robot_core/TripleBuffer.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"

namespace ocs2::humanoid::ipc {

/**
 * The robot side of the network MPC link (humanoid_nmpc/docs/distributed_runtime/README.md, "The MPC link"): an
 * ocs2::MRT_BASE whose solver is an MpcServer in another process, reached over the bus. The controller uses it exactly
 * as it uses MPC_MRT_Interface, and reads the same predicates with the same meaning:
 *
 *   - updatePolicy(), evaluatePolicy(), initialPolicyReceived() and isActivePolicyCurrent() are MRT_BASE's own;
 *   - hasOutstandingReset() and isHealthy() are those of the controller's MpcResetSupervisor, which this link serves.
 *
 * THREADS. The realtime thread calls setCurrentObservation(), updatePolicy() and the predicates, and evaluates the policy
 * in use (getPolicy()) with a RealtimePolicyEvaluator; nothing it calls here touches the bus, allocates or logs:
 * setCurrentObservation() copies the observation into a triple buffer and checks the deadline of the newest policy, and
 * the policy swap is MRT_BASE's try-lock. MRT_BASE::evaluatePolicy() is not for the realtime thread: it allocates
 * (OCS2's interpolation and controllers return by value) and logs once the time is past the plan. Everything else runs
 * on the bus's IO thread, every Config::pollPeriod and for every message: it takes the newest observation from the
 * triple buffer and publishes it on robot/mpc_observation with the supervisor's request counters, serves the reset
 * handshake, decodes mpc/policy into the policy buffer (MRT_BASE::moveToBuffer(), which frees the replaced policy on
 * that thread too), mirrors the MPC's health and checks the policy timeout. The realtime thread then reads the outcome
 * as atomics.
 *
 * RESETS. A reset the controller requests from its supervisor (requestReset(), resetMpcNode()) is seen by the IO thread,
 * which drops the buffered policy at once (MRT_BASE::discardBufferedPolicy(): the policy in use stops being current)
 * and keeps the ticket open. The MPC node sees the counters in the next observation it receives and resets from it. A
 * policy stamped with fewer resets served than the robot has asked for was solved before the reset and is dropped; the
 * first one that serves the ticket is moved to the buffer and completes it. From then on the controller sees
 * isActivePolicyCurrent() && !hasOutstandingReset(), exactly as with the in-process solver; in between, as there, at
 * least one of the two is false.
 *
 * HEALTH. isHealthy() of the supervisor reads false while
 *   - the MPC node reports its solver unhealthy (MpcSolverStatus, in every policy and in mpc/status, which it
 *     publishes after every attempt: a failing solver publishes no policy), or
 *   - the link is lost: no policy solved from an observation of the last Config::policyTimeout seconds of robot time
 *     has arrived, or the robot clock has reached the end of the newest policy's horizon. Only after a policy has been
 *     received; the robot clock is the time of the observations.
 * The realtime thread checks the second itself, in setCurrentObservation(), against the newest policy's deadline, which
 * the IO thread publishes when it accepts the policy: the controller holds the robot at the first cycle past the
 * deadline even when the IO thread is stalled (MpcResetSupervisor::setRemotePolicyExpired()). The IO thread finds the
 * same loss on its next poll, logs it, and requests a full reset, as the in-process supervisor does once it declares a
 * failing solver unhealthy, so that the hold ends on a policy solved from the robot as it is then. A lost link stays
 * lost until a policy that arrived afterwards and serves that reset is accepted; a clock that runs backwards does not
 * end it.
 * When the MPC node reports a failed solve, or the link is lost, the buffered policy is dropped as for a reset, as the
 * in-process solver drops it when it resets after a failure: the controller's hold then ends on the first policy that
 * arrives afterwards, not on the one it was holding against.
 *
 * A policy solved from an observation this link did not publish (later than the newest one, or older than the first
 * one since the clock last ran backwards) comes from another robot process and is dropped, and so is a policy that is
 * older than the policy timeout on arrival or whose horizon has ended by then. The clock runs backwards when an
 * observation is earlier than the newest one by more than the supervisor's clockRewindTolerance, as
 * MpcResetSupervisor::observeTime() counts it; a smaller step back is jitter and changes nothing.
 *
 * LIFETIME. Create() registers the subscriptions and the periodic callback on a bus that is not running yet; the caller
 * starts it. The callbacks hold the link only through a guard that the destructor clears, so a bus that outlives the
 * link calls nothing of it afterwards. The overload that takes BusOptions creates, starts and owns its bus.
 */
class RemoteMpcLink final : public MRT_BASE {
 public:
  struct Config {
    /** The model the controller evaluates the policies with; every observation sent and policy received must match. */
    ModelDimensions dimensions;
    /**
     * [s, robot clock] Link loss: once a policy has been received, the link reads unhealthy when the robot clock passes
     * the time of the observation the newest one was solved from by more than this. mpc_link.policy_timeout of the task
     * file. Longer than one TCP retransmission (Linux: at least 200 ms) with margin, so that a single lost packet on the
     * bus is not a lost link (RobotProcessSettings::mpcLinkPolicyTimeout).
     */
    // LINT.IfChange(policy_timeout_default)
    scalar_t policyTimeout = 0.5;
    // clang-format off
    // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/include/humanoid_common_mpc_app/robot/RobotProcessSettings.h:policy_timeout_default, //humanoid_nmpc/humanoid_mpc_config/mpc_link_config.proto:policy_timeout_default)
    // clang-format on
    /** How often the IO thread takes the newest observation, publishes it, and checks the policy timeout. */
    absl::Duration pollPeriod = absl::Milliseconds(1);
  };

  /**
   * What the link has done so far; every counter only grows. A message is counted once the link has handled it, and a
   * link loss once its reset request and health are in place.
   */
  struct Statistics {
    uint64_t observationsPublished = 0;
    uint64_t policiesReceived = 0;
    /** Moved to the policy buffer. */
    uint64_t policiesAccepted = 0;
    /** Solved before a reset the robot requested (MpcPolicy.resets_served below the request). */
    uint64_t stalePoliciesDropped = 0;
    /** Older than the policy timeout on arrival. */
    uint64_t latePoliciesDropped = 0;
    /** Solved from an observation this link did not publish. */
    uint64_t foreignPoliciesDropped = 0;
    /** Not decodable, or of other dimensions than Config::dimensions. */
    uint64_t invalidPoliciesRejected = 0;
    uint64_t statusesReceived = 0;
    /** Reset tickets the link has taken from the supervisor, and completed with a policy that serves them. */
    uint64_t resetTicketsTaken = 0;
    uint64_t resetTicketsCompleted = 0;
    /** Times the link was lost (policy timeout or end of horizon). */
    uint64_t linkLosses = 0;
    /** The MPC side's health as last reported, and the link's own. The supervisor's isHealthy() is both. */
    bool solverHealthy = true;
    uint64_t solverConsecutiveFailures = 0;
    bool linkHealthy = true;
    /** [s, robot clock] The newest observation taken from the triple buffer; NaN before the first. */
    scalar_t latestObservationTime = 0.0;
    /**
     * [s, robot clock] The observation the newest accepted policy was solved from; NaN before the first, and from a
     * rewind of the clock until the first policy of the new clock.
     */
    scalar_t newestPolicyInitTime = 0.0;
    /** [s] latestObservationTime - newestPolicyInitTime, as LoopTiming.policy_age_s; negative when there is none. */
    scalar_t policyAge = -1.0;
  };

  /** A link on `bus`, which must not be running yet; the caller starts it. `supervisor` must outlive the link. */
  static absl::StatusOr<std::unique_ptr<RemoteMpcLink>> Create(robot::ipc::Bus& bus, MpcResetSupervisor& supervisor, Config config);

  /** A link on a bus of its own, created from `busOptions` and started; it stops with the link. */
  static absl::StatusOr<std::unique_ptr<RemoteMpcLink>> Create(robot::ipc::BusOptions busOptions,
                                                               MpcResetSupervisor& supervisor,
                                                               Config config);

  ~RemoteMpcLink() override;
  RemoteMpcLink(const RemoteMpcLink&) = delete;
  RemoteMpcLink& operator=(const RemoteMpcLink&) = delete;

  /**
   * Requests a full reset through the supervisor and returns at once (MPC_MRT_Interface resets synchronously, the ROS
   * MRT blocked until the MPC node answered). The target trajectories are not sent: the MPC node resets to the
   * trajectories its own MpcServer::ResetTargetTrajectoriesFunction makes of the observation it resets from.
   */
  void resetMpcNode(const TargetTrajectories& initTargetTrajectories) override;

  /**
   * Realtime-safe: copies `observation` into the triple buffer's write slot and publishes the slot, then checks its
   * time against the deadline of the newest accepted policy and tells the supervisor when the verdict changes (see
   * HEALTH). Makes no heap allocation while the state and input keep the sizes of Config::dimensions, and takes no
   * lock. The IO thread sends the observation at its next poll; a newer observation written before that replaces it.
   */
  void setCurrentObservation(const SystemObservation& observation) override;

  /** Thread-safe. */
  Statistics statistics() const;

  /**
   * The viewer annotations (MpcPolicy.annotations: the contact planner's target contact patches and the scaled walking
   * command) of the newest accepted policy, when a policy was accepted since the last call. For ONE consumer thread
   * (the robot process's communication thread, which draws them in the MuJoCo viewer); lock-free, and allocation-free
   * once `annotations` holds as many patches as the policy carries. A robot::TripleBuffer that the IO thread fills
   * when it accepts a policy.
   */
  bool takeAnnotations(msgs::ViewerAnnotations& annotations);

 private:
  /** What the callbacks on the bus reach the link through, cleared by the destructor. */
  struct CallbackGuard {
    absl::Mutex mutex;
    RemoteMpcLink* absl_nullable link ABSL_GUARDED_BY(mutex) = nullptr;
  };

  /** One observation in the triple buffer: what the realtime thread wrote, and the how-many-th it was. */
  struct ObservationSlot {
    SystemObservation observation;
    uint64_t sequence = 0;
  };

  RemoteMpcLink(robot::ipc::Bus& bus, MpcResetSupervisor& supervisor, Config config);
  absl::Status registerOnBus();

  // ---- IO thread, under guard_->mutex.
  void onPoll();
  void onPolicy(const humanoid_mpc_msgs::MpcPolicy& message);
  void onStatus(const humanoid_mpc_msgs::MpcStatus& message);
  /** Takes the newest observation from the triple buffer, if there is a new one, and publishes it. */
  void takeAndPublishObservation();
  /** Takes the reset the supervisor has outstanding, if it is a new one: drops the buffered policy, keeps the ticket. */
  void serveResetRequests();
  /** False for a time the link did not publish an observation of (see the class comment). */
  bool isOwnObservationTime(scalar_t time) const;
  void mirrorSolverStatus(const humanoid_mpc_msgs::MpcSolverStatus& status, bool fromPolicy);
  /**
   * IO thread: a policy or status of another MPC server than the last one (a restarted MPC node, which counts its solves
   * from zero again) makes its solve counts the ones to compare with: newestPolicySolveCount_ starts over.
   */
  void observeServerInstance(uint64_t serverInstance);
  /** Checks the policy timeout and the horizon, and hands the combined health to the supervisor. */
  void updateHealth();
  /** Hands the deadline of newestPolicyInitTime_ and newestPolicyFinalTime_ to the realtime thread's watchdog. */
  void publishPolicyDeadline();
  /** Hands the annotations of the policy just accepted to the consumer of takeAnnotations(). */
  void publishAnnotations(const humanoid_mpc_msgs::ViewerAnnotations& message);
  void publishStatistics();

  robot::ipc::Bus& bus_;
  std::unique_ptr<robot::ipc::Bus> ownedBus_;
  MpcResetSupervisor& supervisor_;
  const Config config_;
  std::shared_ptr<CallbackGuard> guard_;

  // ---- Realtime thread (producer) -> IO thread (consumer).
  robot::TripleBuffer<ObservationSlot> observations_;

  // ---- IO thread (producer) -> the consumer of takeAnnotations().
  robot::TripleBuffer<msgs::ViewerAnnotations> annotations_;
  uint64_t observationsWritten_ = 0;  // realtime thread

  // ---- IO thread (writer) -> realtime thread (reader): the robot time after which the newest accepted policy has
  // expired, min(its init time + policyTimeout, the double just below its final time); +infinity before the first.
  std::atomic<scalar_t> policyDeadline_;
  bool policyExpired_ = false;  // realtime thread: the verdict last handed to the supervisor

  // ---- IO thread.
  humanoid_mpc_msgs::MpcObservation observationMessage_;
  bool haveObservation_ = false;
  scalar_t latestObservationTime_ = 0.0;
  // The first observation taken since the clock last ran backwards.
  scalar_t sessionStartTime_ = 0.0;
  // The newest reset ticket taken; open until a policy serving it arrives.
  MpcResetSupervisor::ResetTicket ticket_;
  bool ticketOpen_ = false;
  bool policyAccepted_ = false;
  scalar_t newestPolicyInitTime_ = 0.0;
  scalar_t newestPolicyFinalTime_ = 0.0;
  bool solverHealthy_ = true;
  uint64_t solverConsecutiveFailures_ = 0;
  // MpcSolverStatus.solve_count of the newest policy received: a status of an older attempt says nothing new.
  uint64_t newestPolicySolveCount_ = 0;
  // MpcSolverStatus.server_instance of the last policy or status received; 0 before the first.
  uint64_t serverInstance_ = 0;
  // The buffered policy was dropped for the failure the MPC node last reported; cleared by the next accepted policy.
  bool droppedForFailure_ = false;
  bool linkLost_ = false;

  // ---- Statistics: written on the IO thread, read anywhere.
  std::atomic<uint64_t> observationsPublished_{0};
  std::atomic<uint64_t> policiesReceived_{0};
  std::atomic<uint64_t> policiesAccepted_{0};
  std::atomic<uint64_t> stalePoliciesDropped_{0};
  std::atomic<uint64_t> latePoliciesDropped_{0};
  std::atomic<uint64_t> foreignPoliciesDropped_{0};
  std::atomic<uint64_t> invalidPoliciesRejected_{0};
  std::atomic<uint64_t> statusesReceived_{0};
  std::atomic<uint64_t> resetTicketsTaken_{0};
  std::atomic<uint64_t> resetTicketsCompleted_{0};
  std::atomic<uint64_t> linkLosses_{0};
  std::atomic<bool> solverHealthyStatistic_{true};
  std::atomic<uint64_t> solverConsecutiveFailuresStatistic_{0};
  std::atomic<bool> linkHealthyStatistic_{true};
  std::atomic<scalar_t> latestObservationTimeStatistic_;
  std::atomic<scalar_t> newestPolicyInitTimeStatistic_;
};

}  // namespace ocs2::humanoid::ipc
