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

#include "humanoid_mpc_ipc/RemoteMpcLink.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include <ocs2_mpc/CommandData.h>
#include <ocs2_oc/oc_data/PerformanceIndex.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/viewer_annotations.nproto.pb.h"
#include "robot_ipc/Delivery.h"

namespace ocs2::humanoid::ipc {
namespace {

constexpr double kLogPeriodSeconds = 5.0;
constexpr scalar_t kNoTime = std::numeric_limits<scalar_t>::quiet_NaN();
constexpr scalar_t kNoEnd = std::numeric_limits<scalar_t>::infinity();
static_assert(std::atomic<scalar_t>::is_always_lock_free, "the realtime thread reads the policy deadline without a lock");

// Adds one to a counter when it goes out of scope: a message is counted once the link has handled it, whichever way it
// went, so that whoever sees the count sees what handling it did.
class CountWhenHandled {
 public:
  explicit CountWhenHandled(std::atomic<uint64_t>& counter) : counter_(counter) {}
  ~CountWhenHandled() { counter_.fetch_add(1); }
  CountWhenHandled(const CountWhenHandled&) = delete;
  CountWhenHandled& operator=(const CountWhenHandled&) = delete;

 private:
  std::atomic<uint64_t>& counter_;
};

absl::Status validateConfig(const RemoteMpcLink::Config& config) {
  if (config.dimensions.stateDim == 0) {
    return absl::InvalidArgumentError("RemoteMpcLink: Config::dimensions.stateDim is 0; give the dimensions of the controller's model");
  }
  if (config.dimensions.numModes == 0) {
    return absl::InvalidArgumentError(
        "RemoteMpcLink: Config::dimensions.numModes is 0; give the number of modes of the controller's model");
  }
  if (!std::isfinite(config.policyTimeout) || config.policyTimeout <= 0.0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "RemoteMpcLink: Config::policyTimeout (mpcLink.policyTimeout) must be a positive number of seconds, got ", config.policyTimeout));
  }
  if (config.pollPeriod <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError(absl::StrCat("RemoteMpcLink: Config::pollPeriod must be positive, got ", config.pollPeriod));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<RemoteMpcLink>> RemoteMpcLink::Create(robot::ipc::Bus& bus, MpcResetSupervisor& supervisor, Config config) {
  RETURN_IF_ERROR(validateConfig(config));
  if (bus.isRunning()) {
    return absl::FailedPreconditionError("RemoteMpcLink: the bus is running; create the link before Bus::start()");
  }
  std::unique_ptr<RemoteMpcLink> link(new RemoteMpcLink(bus, supervisor, std::move(config)));
  RETURN_IF_ERROR(link->registerOnBus());
  return link;
}

absl::StatusOr<std::unique_ptr<RemoteMpcLink>> RemoteMpcLink::Create(robot::ipc::BusOptions busOptions,
                                                                     MpcResetSupervisor& supervisor,
                                                                     Config config) {
  RETURN_IF_ERROR(validateConfig(config));
  ASSIGN_OR_RETURN(std::unique_ptr<robot::ipc::Bus> bus, robot::ipc::Bus::Create(std::move(busOptions)));
  std::unique_ptr<RemoteMpcLink> link(new RemoteMpcLink(*bus, supervisor, std::move(config)));
  link->ownedBus_ = std::move(bus);
  RETURN_IF_ERROR(link->registerOnBus());
  RETURN_IF_ERROR(link->ownedBus_->start());
  return link;
}

RemoteMpcLink::RemoteMpcLink(robot::ipc::Bus& bus, MpcResetSupervisor& supervisor, Config config)
    : bus_(bus),
      supervisor_(supervisor),
      config_(std::move(config)),
      guard_(std::make_shared<CallbackGuard>()),
      observations_(
          ObservationSlot{.observation = SystemObservation{.mode = 0,
                                                           .time = 0.0,
                                                           .state = vector_t::Zero(static_cast<Eigen::Index>(config_.dimensions.stateDim)),
                                                           .input = vector_t::Zero(static_cast<Eigen::Index>(config_.dimensions.inputDim))},
                          .sequence = 0}),
      policyDeadline_(kNoEnd),
      latestObservationTimeStatistic_(kNoTime),
      newestPolicyInitTimeStatistic_(kNoTime) {
  // A reset served before the link existed counts as the last ticket: a policy that serves less was solved before it.
  ticket_.request = supervisor_.numResetsServed();
  ticket_.fullRequest = supervisor_.numFullResetsServed();
  absl::MutexLock lock(guard_->mutex);
  guard_->link = this;
}

RemoteMpcLink::~RemoteMpcLink() {
  if (ownedBus_ != nullptr) {
    ownedBus_->stop();
  }
  absl::MutexLock lock(guard_->mutex);
  guard_->link = nullptr;
}

absl::Status RemoteMpcLink::registerOnBus() {
  const std::shared_ptr<CallbackGuard> guard = guard_;
  const std::function<void(const humanoid_mpc_msgs::MpcPolicy&)> onPolicyMessage = [guard](const humanoid_mpc_msgs::MpcPolicy& message) {
    absl::MutexLock lock(guard->mutex);
    if (guard->link != nullptr) guard->link->onPolicy(message);
  };
  const std::function<void(const humanoid_mpc_msgs::MpcStatus&)> onStatusMessage = [guard](const humanoid_mpc_msgs::MpcStatus& message) {
    absl::MutexLock lock(guard->mutex);
    if (guard->link != nullptr) guard->link->onStatus(message);
  };
  const std::function<void()> onPollPeriod = [guard]() {
    absl::MutexLock lock(guard->mutex);
    if (guard->link != nullptr) guard->link->onPoll();
  };
  RETURN_IF_ERROR(bus_.subscribe<humanoid_mpc_msgs::MpcPolicy>(topics::kMpcPolicy, robot::ipc::Delivery::kLatest, onPolicyMessage));
  // After mpc/policy: the bus hands the newest message of each topic of one drain over in the order of the
  // subscriptions, so a status is handled after a policy of the same drain, which an earlier or the same attempt solved.
  RETURN_IF_ERROR(bus_.subscribe<humanoid_mpc_msgs::MpcStatus>(topics::kMpcStatus, robot::ipc::Delivery::kLatest, onStatusMessage));
  return bus_.addPeriodicCallback(config_.pollPeriod, onPollPeriod);
}

// ---------------------------------------------------------------------------------------------------------------------
// The controller's side
// ---------------------------------------------------------------------------------------------------------------------

void RemoteMpcLink::resetMpcNode(const TargetTrajectories& /*initTargetTrajectories*/) {
  supervisor_.requestReset(MpcResetSupervisor::ResetKind::kFull);
}

void RemoteMpcLink::setCurrentObservation(const SystemObservation& observation) {
  ObservationSlot& slot = observations_.writeSlot();
  slot.observation = observation;
  slot.sequence = ++observationsWritten_;
  observations_.publishWrite();

  // The watchdog of this thread, against the deadline of the newest policy the IO thread accepted. The IO thread checks
  // the same deadline, but only when it runs: a stalled or starved IO thread must not leave the controller executing a
  // plan past its end. False before the first policy (an infinite deadline) and for a NaN time.
  const bool expired = observation.time > policyDeadline_.load(std::memory_order_acquire);
  if (expired != policyExpired_) {
    policyExpired_ = expired;
    supervisor_.setRemotePolicyExpired(expired);
  }
}

bool RemoteMpcLink::takeAnnotations(msgs::ViewerAnnotations& annotations) {
  if (!annotations_.acquireRead()) {
    return false;
  }
  annotations = annotations_.readSlot();
  return true;
}

RemoteMpcLink::Statistics RemoteMpcLink::statistics() const {
  Statistics statistics;
  statistics.observationsPublished = observationsPublished_.load();
  statistics.policiesReceived = policiesReceived_.load();
  statistics.policiesAccepted = policiesAccepted_.load();
  statistics.stalePoliciesDropped = stalePoliciesDropped_.load();
  statistics.latePoliciesDropped = latePoliciesDropped_.load();
  statistics.foreignPoliciesDropped = foreignPoliciesDropped_.load();
  statistics.invalidPoliciesRejected = invalidPoliciesRejected_.load();
  statistics.statusesReceived = statusesReceived_.load();
  statistics.resetTicketsTaken = resetTicketsTaken_.load();
  statistics.resetTicketsCompleted = resetTicketsCompleted_.load();
  statistics.linkLosses = linkLosses_.load();
  statistics.solverHealthy = solverHealthyStatistic_.load();
  statistics.solverConsecutiveFailures = solverConsecutiveFailuresStatistic_.load();
  statistics.linkHealthy = linkHealthyStatistic_.load();
  statistics.latestObservationTime = latestObservationTimeStatistic_.load();
  statistics.newestPolicyInitTime = newestPolicyInitTimeStatistic_.load();
  const bool haveAge = std::isfinite(statistics.latestObservationTime) && std::isfinite(statistics.newestPolicyInitTime);
  statistics.policyAge = haveAge ? statistics.latestObservationTime - statistics.newestPolicyInitTime : -1.0;
  return statistics;
}

// ---------------------------------------------------------------------------------------------------------------------
// The IO thread
// ---------------------------------------------------------------------------------------------------------------------

void RemoteMpcLink::onPoll() {
  takeAndPublishObservation();
  serveResetRequests();
  updateHealth();
}

void RemoteMpcLink::takeAndPublishObservation() {
  if (!observations_.acquireRead()) {
    return;
  }
  const ObservationSlot& slot = observations_.readSlot();
  const scalar_t time = slot.observation.time;
  // A rewind is what the controller's supervisor counts as one (MpcResetSupervisor::observeTime()).
  if (!haveObservation_ || time < latestObservationTime_ - supervisor_.getConfig().clockRewindTolerance) {
    // The first observation, or a clock that ran backwards: a policy solved before this one is not of this clock.
    sessionStartTime_ = time;
    if (policyAccepted_) {
      // The newest policy is timed on the old clock; the timeout counts from here, and its horizon says nothing. A lost
      // link stays lost: only a policy that arrives ends that.
      newestPolicyInitTime_ = time;
      newestPolicyFinalTime_ = kNoEnd;
      publishPolicyDeadline();
      // No policy of this clock yet, so no policy age (statistics() reads the observation time first).
      newestPolicyInitTimeStatistic_.store(kNoTime);
    }
    latestObservationTime_ = time;
  } else {
    // A step back within the tolerance is jitter of the clock, not a new one: the newest time stays the newest.
    latestObservationTime_ = std::max(latestObservationTime_, time);
  }
  haveObservation_ = true;
  latestObservationTimeStatistic_.store(latestObservationTime_);

  toProto(slot.observation, observationMessage_.mutable_observation());
  const absl::Status dimensions = checkDimensions(observationMessage_.observation(), config_.dimensions);
  if (!dimensions.ok()) {
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "[RemoteMpcLink] Not sending the observation at t = " << time << ": "
                                              << dimensions.message();
    return;
  }
  const MpcResetSupervisor::ResetCounters counters = supervisor_.resetsRequested();
  observationMessage_.mutable_resets()->set_requested(counters.requested);
  observationMessage_.mutable_resets()->set_full_requested(counters.fullRequested);
  observationMessage_.set_sequence(slot.sequence);
  const absl::Status published = bus_.publishFromIoThread(topics::kRobotMpcObservation, observationMessage_);
  if (published.ok()) {
    observationsPublished_.fetch_add(1);
  } else {
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds) << "[RemoteMpcLink] Publishing the observation failed: " << published.message();
  }
}

void RemoteMpcLink::serveResetRequests() {
  const std::optional<MpcResetSupervisor::ResetTicket> ticket = supervisor_.takeResetRequest();
  if (!ticket.has_value()) {
    return;
  }
  if (ticketOpen_ && ticket->request == ticket_.request && ticket->fullRequest == ticket_.fullRequest) {
    return;  // Already waiting for the policy that serves it.
  }
  // The policy in use stops being current, and no policy solved before this request can be swapped in after it.
  discardBufferedPolicy();
  ticket_ = *ticket;
  ticketOpen_ = true;
  resetTicketsTaken_.fetch_add(1);
}

bool RemoteMpcLink::isOwnObservationTime(scalar_t time) const {
  return haveObservation_ && time >= sessionStartTime_ && time <= latestObservationTime_;
}

void RemoteMpcLink::onPolicy(const humanoid_mpc_msgs::MpcPolicy& message) {
  const CountWhenHandled received(policiesReceived_);
  // The newest observation, the reset requests and the clock as of now, before the policy is judged by them.
  onPoll();

  const scalar_t initTime = message.init_observation().time();
  if (!isOwnObservationTime(initTime)) {
    foreignPoliciesDropped_.fetch_add(1);
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
        << "[RemoteMpcLink] Dropping a policy solved from an observation at t = " << initTime
        << " that this robot did not send (its observations since the clock last started run from t = " << sessionStartTime_ << " to "
        << latestObservationTime_ << "): another robot process is on the bus, or the MPC node still answers a previous one.";
    return;
  }
  observeServerInstance(message.solver_status().server_instance());
  if (message.resets_served() < ticket_.request || message.full_resets_served() < ticket_.fullRequest) {
    // Solved before a reset this robot requested. Its solver status is current all the same.
    stalePoliciesDropped_.fetch_add(1);
    mirrorSolverStatus(message.solver_status(), /*fromPolicy=*/true);
    updateHealth();
    return;
  }
  if (latestObservationTime_ > initTime + config_.policyTimeout) {
    // Older than the timeout on arrival: it would not end a hold, nor keep the link healthy.
    latePoliciesDropped_.fetch_add(1);
    mirrorSolverStatus(message.solver_status(), /*fromPolicy=*/true);
    updateHealth();
    return;
  }

  std::unique_ptr<CommandData> command = std::make_unique<CommandData>();
  std::unique_ptr<PrimalSolution> solution = std::make_unique<PrimalSolution>();
  std::unique_ptr<PerformanceIndex> performance = std::make_unique<PerformanceIndex>();
  absl::Status decoded = checkDimensions(message, config_.dimensions);
  if (decoded.ok()) {
    decoded = policyFromProto(message, command.get(), solution.get(), performance.get());
  }
  if (!decoded.ok()) {
    invalidPoliciesRejected_.fetch_add(1);
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "[RemoteMpcLink] Rejecting a policy from the MPC node: " << decoded.message();
    return;
  }

  const scalar_t finalTime = solution->timeTrajectory_.back();
  if (latestObservationTime_ >= finalTime) {
    // Its horizon ended before it arrived: taking it would lose the link again at once.
    latePoliciesDropped_.fetch_add(1);
    mirrorSolverStatus(message.solver_status(), /*fromPolicy=*/true);
    updateHealth();
    return;
  }
  moveToBuffer(std::move(command), std::move(solution), std::move(performance));
  publishAnnotations(message.annotations());
  policiesAccepted_.fetch_add(1);
  policyAccepted_ = true;
  newestPolicyInitTime_ = initTime;
  newestPolicyFinalTime_ = finalTime;
  // After moveToBuffer(): the realtime thread's watchdog never sees the new deadline before the policy it belongs to.
  publishPolicyDeadline();
  newestPolicyInitTimeStatistic_.store(initTime);
  droppedForFailure_ = false;
  if (linkLost_) {
    // A policy that arrived after the loss, solved within the timeout and, as the loss requested a full reset, after it.
    linkLost_ = false;
    LOG(INFO) << "[RemoteMpcLink] A fresh MPC policy has arrived (solved from the observation at t = " << initTime
              << "); the link is healthy again.";
  }
  // Completed after the policy is in the buffer: the controller never sees the reset served without a policy of it.
  if (ticketOpen_) {
    supervisor_.completeReset(ticket_);
    ticketOpen_ = false;
    resetTicketsCompleted_.fetch_add(1);
  }
  mirrorSolverStatus(message.solver_status(), /*fromPolicy=*/true);
  updateHealth();
}

void RemoteMpcLink::onStatus(const humanoid_mpc_msgs::MpcStatus& message) {
  const CountWhenHandled received(statusesReceived_);
  if (!isOwnObservationTime(message.observation_time())) {
    return;
  }
  observeServerInstance(message.solver_status().server_instance());
  // The status of the attempt that made the newest policy, or of an older one, says nothing the policy did not.
  if (message.solver_status().solve_count() <= newestPolicySolveCount_) {
    return;
  }
  mirrorSolverStatus(message.solver_status(), /*fromPolicy=*/false);
  updateHealth();
}

void RemoteMpcLink::observeServerInstance(uint64_t serverInstance) {
  if (serverInstance == serverInstance_) return;
  if (serverInstance_ != 0) {
    LOG(INFO) << "[RemoteMpcLink] The MPC node restarted (another server instance): its solves are counted from zero again, "
              << "and its statuses are mirrored from its first one.";
  }
  serverInstance_ = serverInstance;
  newestPolicySolveCount_ = 0;
}

void RemoteMpcLink::mirrorSolverStatus(const humanoid_mpc_msgs::MpcSolverStatus& status, bool fromPolicy) {
  if (fromPolicy) {
    newestPolicySolveCount_ = status.solve_count();
  }
  solverHealthy_ = status.healthy();
  solverConsecutiveFailures_ = status.consecutive_failures();
  // The in-process solver resets after a failed solve and drops the buffered policy with it; so does the link, once per
  // run of failures, so that the policy in use is no longer current until one solved after the failure arrives.
  if (!fromPolicy && status.consecutive_failures() > 0 && !droppedForFailure_) {
    discardBufferedPolicy();
    droppedForFailure_ = true;
  }
}

void RemoteMpcLink::publishAnnotations(const humanoid_mpc_msgs::ViewerAnnotations& message) {
  // Into the slot's own storage: once it holds as many patches as the policies carry, this allocates nothing.
  msgs::ViewerAnnotations& slot = annotations_.writeSlot();
  const absl::Status converted = FromProto(message, &slot);
  if (!converted.ok()) {
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
        << "[RemoteMpcLink] Not showing the viewer annotations of a policy: " << converted.message();
    return;
  }
  annotations_.publishWrite();
}

void RemoteMpcLink::publishPolicyDeadline() {
  // The realtime thread's watchdog compares with `time > deadline`: the timeout is passed after init + policyTimeout,
  // the horizon is reached at its final time, which is the same as passing the double just below it.
  const scalar_t horizonEnd = std::nextafter(newestPolicyFinalTime_, -std::numeric_limits<scalar_t>::infinity());
  policyDeadline_.store(std::min(newestPolicyInitTime_ + config_.policyTimeout, horizonEnd), std::memory_order_release);
}

void RemoteMpcLink::updateHealth() {
  // Once lost, the link stays lost until onPolicy() accepts a policy that arrived afterwards; nothing else ends a loss.
  bool lostNow = false;
  if (policyAccepted_ && haveObservation_ && !linkLost_) {
    // The checks of the realtime thread's watchdog (publishPolicyDeadline()), on the newest observation time.
    const bool timedOut = latestObservationTime_ > newestPolicyInitTime_ + config_.policyTimeout;
    const bool pastHorizon = latestObservationTime_ >= newestPolicyFinalTime_;
    if (timedOut || pastHorizon) {
      linkLost_ = true;
      lostNow = true;
      // As the in-process supervisor does once a failing solver is declared unhealthy: a full reset, so that the hold
      // ends on a policy solved from the robot as it is now, with the MPC's references and gait schedule cleared,
      // rather than on one that resumes the schedule the MPC kept running through the loss. Taking the ticket at once
      // drops the buffered policy, and every policy in flight is stale for it: the hold ends on a policy that arrived
      // afterwards.
      supervisor_.requestReset(MpcResetSupervisor::ResetKind::kFull);
      serveResetRequests();
      if (timedOut) {
        LOG(WARNING) << "[RemoteMpcLink] No MPC policy solved within the last " << config_.policyTimeout
                     << " s of robot time has arrived (the newest is from t = " << newestPolicyInitTime_
                     << ", the robot is at t = " << latestObservationTime_
                     << "). The link reads unhealthy, WB_MPC holds the robot with the JOINT_PD action, and the MPC "
                     << "node is asked for a full reset, which the first policy that ends the hold serves.";
      } else {
        LOG(WARNING) << "[RemoteMpcLink] The robot clock (t = " << latestObservationTime_ << ") has reached the end of the newest "
                     << "MPC policy (t = " << newestPolicyFinalTime_ << "). The link reads unhealthy, WB_MPC holds the robot with "
                     << "the JOINT_PD action, and the MPC node is asked for a full reset, which the first policy that ends the "
                     << "hold serves.";
      }
    }
  }

  const bool healthy = solverHealthy_ && !linkLost_;
  if (solverHealthy_ != solverHealthyStatistic_.load()) {
    if (solverHealthy_) {
      LOG(INFO) << "[RemoteMpcLink] The MPC node reports its solver healthy again.";
    } else {
      LOG(ERROR) << "[RemoteMpcLink] The MPC node reports its solver unhealthy after " << solverConsecutiveFailures_
                 << " failed solves in a row; WB_MPC holds the robot with the JOINT_PD action until it recovers. The MPC node "
                 << "logs the cause.";
    }
  }
  supervisor_.setRemoteHealth(healthy, solverConsecutiveFailures_);
  publishStatistics();
  if (lostNow) {
    // Counted last: whoever sees the count sees the loss's reset request, dropped policy, health and statistics too.
    linkLosses_.fetch_add(1);
  }
}

void RemoteMpcLink::publishStatistics() {
  solverHealthyStatistic_.store(solverHealthy_);
  solverConsecutiveFailuresStatistic_.store(solverConsecutiveFailures_);
  linkHealthyStatistic_.store(!linkLost_);
}

}  // namespace ocs2::humanoid::ipc
