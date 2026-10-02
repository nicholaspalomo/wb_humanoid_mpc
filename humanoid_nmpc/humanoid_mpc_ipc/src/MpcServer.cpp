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

#include "humanoid_mpc_ipc/MpcServer.h"

#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"

#include <ocs2_oc/oc_solver/SolverBase.h>

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_mpc_ipc/SolutionTimeWindow.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "robot_ipc/Delivery.h"

namespace ocs2::humanoid::ipc {
namespace {

constexpr double kLogPeriodSeconds = 5.0;

/** A random nonzero number: the MpcSolverStatus.server_instance of one server. */
uint64_t drawServerInstance() {
  std::random_device device;
  const uint64_t instance = (static_cast<uint64_t>(device()) << 32) ^ static_cast<uint64_t>(device()) ^
                            static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
  return instance != 0 ? instance : 1;
}

}  // namespace

absl::StatusOr<std::unique_ptr<MpcServer>> MpcServer::Create(
    robot::ipc::Bus& bus, MPC_BASE& mpc, ResetTargetTrajectoriesFunction resetTargetTrajectories, Config config, Hooks hooks) {
  if (resetTargetTrajectories == nullptr) {
    return absl::InvalidArgumentError("MpcServer: the reset target trajectories function is empty");
  }
  if (config.dimensions.stateDim == 0) {
    return absl::InvalidArgumentError("MpcServer: Config::dimensions.stateDim is 0; give the dimensions of the MPC's model");
  }
  if (config.dimensions.numModes == 0) {
    return absl::InvalidArgumentError("MpcServer: Config::dimensions.numModes is 0; give the number of modes of the MPC's model");
  }
  if (!std::isfinite(config.mpcDesiredFrequency)) {
    return absl::InvalidArgumentError(absl::StrCat("MpcServer: Config::mpcDesiredFrequency is ", config.mpcDesiredFrequency));
  }
  if (bus.nodeName().empty()) {
    return absl::InvalidArgumentError("MpcServer: the bus has no node name, so it cannot publish the policies");
  }
  if (bus.isRunning()) {
    return absl::FailedPreconditionError("MpcServer: the bus is running; create the server before Bus::start()");
  }
  std::unique_ptr<MpcServer> server(new MpcServer(bus, mpc, std::move(resetTargetTrajectories), std::move(config), std::move(hooks)));
  RETURN_IF_ERROR(server->registerOnBus());
  return server;
}

MpcServer::MpcServer(
    robot::ipc::Bus& bus, MPC_BASE& mpc, ResetTargetTrajectoriesFunction resetTargetTrajectories, Config config, Hooks hooks)
    : bus_(bus),
      mpc_(mpc),
      resetTargetTrajectories_(std::move(resetTargetTrajectories)),
      config_(std::move(config)),
      hooks_(std::move(hooks)),
      supervisor_(config_.resetSupervisor),
      serverInstance_(drawServerInstance()),
      guard_(std::make_shared<CallbackGuard>()) {
  absl::MutexLock lock(guard_->mutex);
  guard_->server = this;
}

MpcServer::~MpcServer() {
  stop();
  absl::MutexLock lock(guard_->mutex);
  guard_->server = nullptr;
}

absl::Status MpcServer::registerOnBus() {
  const std::shared_ptr<CallbackGuard> guard = guard_;
  const std::function<void(const humanoid_mpc_msgs::MpcObservation&)> onObservationMessage =
      [guard](const humanoid_mpc_msgs::MpcObservation& message) {
        absl::MutexLock lock(guard->mutex);
        if (guard->server != nullptr) guard->server->onObservation(message);
      };
  return bus_.subscribe<humanoid_mpc_msgs::MpcObservation>(topics::kRobotMpcObservation, robot::ipc::Delivery::kLatest,
                                                           onObservationMessage);
}

absl::Status MpcServer::start() {
  absl::MutexLock lock(lifecycleMutex_);
  if (started_) {
    return absl::FailedPreconditionError("MpcServer::start(): the server was started (or stopped) before; it starts once");
  }
  started_ = true;
  solverThread_ = std::thread(&MpcServer::runSolverLoop, this);
  return absl::OkStatus();
}

void MpcServer::stop() {
  {
    absl::MutexLock lock(mutex_);
    stopRequested_ = true;
  }
  absl::MutexLock lock(lifecycleMutex_);
  started_ = true;
  if (solverThread_.joinable()) {
    solverThread_.join();
  }
}

MpcServer::Statistics MpcServer::statistics() const {
  Statistics statistics;
  statistics.observationsReceived = observationsReceived_.load();
  statistics.observationsSkipped = observationsSkipped_.load();
  statistics.observationsRejected = observationsRejected_.load();
  statistics.solveAttempts = solveAttempts_.load();
  statistics.failedAttempts = failedAttempts_.load();
  statistics.policiesPublished = policiesPublished_.load();
  statistics.statusesPublished = statusesPublished_.load();
  statistics.fullResets = fullResets_.load();
  statistics.solverResets = solverResets_.load();
  statistics.robotSessions = robotSessions_.load();
  statistics.robotResetsServed = robotResetsServed_.load();
  statistics.robotFullResetsServed = robotFullResetsServed_.load();
  statistics.healthy = supervisor_.isHealthy();
  return statistics;
}

// ---------------------------------------------------------------------------------------------------------------------
// The IO thread
// ---------------------------------------------------------------------------------------------------------------------

void MpcServer::onObservation(const humanoid_mpc_msgs::MpcObservation& message) {
  observationsReceived_.fetch_add(1);
  absl::Status valid = checkDimensions(message.observation(), config_.dimensions);
  absl::MutexLock lock(mutex_);
  ++receivedSinceStatus_;
  if (valid.ok() && hasObservation_ && message.sequence() == mailboxSequence_) {
    observationsSkipped_.fetch_add(1);
    ++skippedSinceStatus_;
    return;
  }
  if (valid.ok()) {
    // Checks the whole message before it writes anything: a rejected observation leaves the mailbox as it was.
    valid = fromProto(message.observation(), &mailboxObservation_);
  }
  if (!valid.ok()) {
    observationsRejected_.fetch_add(1);
    LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "[MpcServer] Rejecting an observation from the robot: " << valid.message();
    return;
  }
  if (hasObservation_ && message.sequence() < mailboxSequence_) {
    // The robot process counts its control cycles from its start: it has restarted.
    newSession_ = true;
    ++mailboxSessions_;
  }
  mailboxSequence_ = message.sequence();
  mailboxRequests_.requested = message.resets().requested();
  mailboxRequests_.fullRequested = message.resets().full_requested();
  ++mailboxVersion_;
  hasObservation_ = true;
}

// ---------------------------------------------------------------------------------------------------------------------
// The solver thread
// ---------------------------------------------------------------------------------------------------------------------

void MpcServer::runSolverLoop() {
  const absl::Status configured = robot::realtime::configureCurrentThread(config_.solverThread);
  if (!configured.ok()) {
    LOG(WARNING) << "[MpcServer] The solver thread runs without part of its configuration: " << configured.message();
  }
  const std::function<bool()> endBackOff = [this]() { return robotRequestedResetSinceSnapshot(); };

  // After a success the next solve waits for a newer observation; a failed attempt is retried on the newest one at once.
  bool lastAttemptSucceeded = true;
  while (awaitObservation(/*requireNew=*/lastAttemptSucceeded)) {
    const absl::Time iterationStart = absl::Now();
    ++solveCount_;
    solveAttempts_.fetch_add(1);

    absl::Status attempt = absl::OkStatus();
    try {
      serveResets();
    } catch (const std::exception& error) {
      attempt = absl::InternalError(absl::StrCat("Resetting the MPC failed: ", error.what()));
      if (snapshot_.newSession) {
        // Not served: the next attempt resets for the restarted robot process again.
        absl::MutexLock lock(mutex_);
        newSession_ = true;
      }
    }
    const absl::Time solveStart = absl::Now();
    if (attempt.ok()) attempt = solve();
    if (attempt.ok()) attempt = buildPolicy();
    const double solveTimeMs = absl::ToDoubleMilliseconds(absl::Now() - solveStart);

    // A failed attempt requests a reset and, once the failures persist, a pause before the next one; the supervisor
    // logs what happened, once. Its health goes out with the policy and the status.
    const std::chrono::duration<scalar_t> retryDelay = supervisor_.onSolveResult(attempt);
    if (attempt.ok()) {
      publishPolicy(attempt, solveTimeMs);
    } else {
      failedAttempts_.fetch_add(1);
    }
    publishStatus(attempt, solveTimeMs);
    lastAttemptSucceeded = attempt.ok();

    if (retryDelay.count() > 0.0) {
      supervisor_.waitBeforeRetry(retryDelay, endBackOff);
      continue;
    }
    if (config_.mpcDesiredFrequency > 0.0) {
      const absl::Time deadline = iterationStart + absl::Seconds(1.0 / config_.mpcDesiredFrequency);
      const absl::Duration overrun = absl::Now() - deadline;
      if (overrun > absl::Milliseconds(1)) {
        LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds) << "[MpcServer] The MPC loop runs slow by " << absl::ToDoubleMilliseconds(overrun)
                                                    << " ms of its " << 1e3 / config_.mpcDesiredFrequency << " ms period.";
      } else {
        sleepUntil(deadline);
      }
    }
  }
  LOG(INFO) << "[MpcServer] The solver thread stops.";
}

bool MpcServer::hasWork() const {
  return stopRequested_ || (hasObservation_ && (!requireNewObservation_ || mailboxVersion_ != takenVersion_));
}

bool MpcServer::awaitObservation(bool requireNew) {
  absl::MutexLock lock(mutex_);
  requireNewObservation_ = requireNew;
  mutex_.Await(absl::Condition(this, &MpcServer::hasWork));
  if (stopRequested_) {
    return false;
  }
  snapshot_.observation = mailboxObservation_;
  snapshot_.requests = mailboxRequests_;
  snapshot_.newSession = newSession_;
  snapshot_.sessions = mailboxSessions_;
  newSession_ = false;
  takenVersion_ = mailboxVersion_;
  return true;
}

bool MpcServer::robotRequestedResetSinceSnapshot() const {
  absl::MutexLock lock(mutex_);
  if (stopRequested_) return true;
  // Against the snapshot, not against what was served: a reset that threw left the robot's request unserved, and the
  // back-off of that failure must not end because of it, or the solver thread would retry it as fast as it can.
  return mailboxSessions_ != snapshot_.sessions || mailboxRequests_.requested != snapshot_.requests.requested ||
         mailboxRequests_.fullRequested != snapshot_.requests.fullRequested;
}

void MpcServer::sleepUntil(absl::Time deadline) {
  absl::MutexLock lock(mutex_);
  mutex_.AwaitWithDeadline(absl::Condition(&stopRequested_), deadline);
}

void MpcServer::serveResets() {
  const Snapshot& snapshot = snapshot_;
  bool reset = false;
  bool full = false;
  absl::string_view reason;
  if (!firstResetDone_) {
    reset = full = true;
    reason = "start-up";
    robotSessions_.store(1);
  } else if (snapshot.newSession || snapshot.requests.requested < robotServed_.requested ||
             snapshot.requests.fullRequested < robotServed_.fullRequested) {
    // A restarted robot process counts its control cycles and its requests from zero again.
    reset = full = true;
    reason = "the robot process restarted";
    robotSessions_.fetch_add(1);
  } else if (snapshot.requests.fullRequested > robotServed_.fullRequested) {
    reset = full = true;
    reason = "requested by the robot";
  } else if (snapshot.requests.requested > robotServed_.requested) {
    reset = true;
    reason = "requested by the robot";
  }
  const std::optional<MpcResetSupervisor::ResetTicket> ticket = supervisor_.takeResetRequest();
  if (ticket.has_value()) {
    if (!reset) reason = "after a failed solve";
    reset = true;
    full = full || ticket->full;
  }
  if (!reset) {
    return;
  }

  // As MPC_MRT_Interface::resetMpcNode() and resetMpcSolver(), but the robot drops the buffered policy, on its side.
  const TargetTrajectories targetTrajectories = resetTargetTrajectories_(snapshot.observation);
  if (full) {
    mpc_.reset();
  } else {
    mpc_.resetSolver();
  }
  mpc_.getSolverPtr()->getReferenceManager().setTargetTrajectories(targetTrajectories);
  if (ticket.has_value()) {
    supervisor_.completeReset(*ticket);
  }
  // Every request the observation counts is served by a reset from it.
  robotServed_.requested = snapshot.requests.requested;
  if (full) {
    robotServed_.fullRequested = snapshot.requests.fullRequested;
  }
  firstResetDone_ = true;
  (full ? fullResets_ : solverResets_).fetch_add(1);
  robotResetsServed_.store(robotServed_.requested);
  robotFullResetsServed_.store(robotServed_.fullRequested);
  // While the solver keeps failing it is reset before every attempt; the supervisor has said so once already.
  if (supervisor_.isHealthy()) {
    LOG(INFO) << "[MpcServer] " << (full ? "MPC reset" : "MPC solver reset") << " to the observation at t = " << snapshot.observation.time
              << " s (" << reason << ").";
  }
}

absl::Status MpcServer::solve() {
  const SystemObservation& observation = snapshot_.observation;
  bool controllerIsUpdated = false;
  try {
    controllerIsUpdated = mpc_.run(observation.time, observation.state, observation.mode);
  } catch (const std::exception& error) {
    // Dumped once per run of failures, as MPC_MRT_Interface::advanceMpc() does.
    if (consecutiveCrashes_++ == 0) {
      const vector_t& state = observation.state;
      LOG(WARNING) << "[MpcServer] MPC solver crashed at t = " << observation.time << ": " << error.what()
                   << "\nState: " << absl::StrJoin(state.data(), state.data() + state.size(), " ")
                   << "\nDesired trajectories: " << mpc_.getSolverPtr()->getReferenceManager().getTargetTrajectories();
    }
    return absl::InternalError(absl::StrCat("MPC solver crashed at t = ", observation.time, ": ", error.what()));
  }
  consecutiveCrashes_ = 0;
  if (!controllerIsUpdated) {
    return absl::FailedPreconditionError(absl::StrCat("MPC not run: the observation time ", observation.time, " is past the final time ",
                                                      mpc_.getSolverPtr()->getFinalTime(),
                                                      " of the previous solution; the MPC has to be reset."));
  }
  return absl::OkStatus();
}

absl::Status MpcServer::buildPolicy() {
  // As MPC_MRT_Interface::copyToBuffer(), cut to the solution time window whatever the solver does with it.
  const SystemObservation& observation = snapshot_.observation;
  SolverBase* solver = mpc_.getSolverPtr();
  const scalar_t window = mpc_.settings().solutionTimeWindow_;
  const scalar_t finalTime = window < 0.0 ? solver->getFinalTime() : observation.time + window;
  solution_.clear();
  solver->getPrimalSolution(finalTime, &solution_);
  trimToSolutionWindow(finalTime, &solution_);
  command_.mpcInitObservation_ = observation;
  command_.mpcTargetTrajectories_ = solver->getReferenceManager().getTargetTrajectories();
  performance_ = solver->getPerformanceIndeces();
  const absl::Status encoded = policyToProto(command_, solution_, performance_, &policyMessage_);
  if (!encoded.ok()) {
    return absl::InternalError(absl::StrCat("The MPC solution cannot be sent to the robot: ", encoded.message()));
  }
  return absl::OkStatus();
}

void MpcServer::fillSolverStatus(const absl::Status& attempt, double solveTimeMs, humanoid_mpc_msgs::MpcSolverStatus* status) const {
  status->set_healthy(supervisor_.isHealthy());
  status->set_consecutive_failures(supervisor_.numConsecutiveFailures());
  status->set_last_error(attempt.ok() ? std::string() : std::string(attempt.message()));
  status->set_solve_time_ms(solveTimeMs);
  status->set_solve_count(solveCount_);
  status->set_server_instance(serverInstance_);
}

void MpcServer::publishPolicy(const absl::Status& attempt, double solveTimeMs) {
  policyMessage_.set_resets_served(robotServed_.requested);
  policyMessage_.set_full_resets_served(robotServed_.fullRequested);
  fillSolverStatus(attempt, solveTimeMs, policyMessage_.mutable_solver_status());
  if (hooks_.annotationsProvider) {
    humanoid_mpc_msgs::ViewerAnnotations* annotations = policyMessage_.mutable_annotations();
    annotations->Clear();
    try {
      hooks_.annotationsProvider(command_, solution_, annotations);
    } catch (const std::exception& error) {
      annotations->Clear();
      LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "[MpcServer] The annotations provider threw: " << error.what();
    }
  }
  const absl::Status published = bus_.publish(topics::kMpcPolicy, policyMessage_);
  if (published.ok()) {
    policiesPublished_.fetch_add(1);
  } else {
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds) << "[MpcServer] Publishing the policy failed: " << published.message();
  }
  if (hooks_.postSolveObserver) {
    try {
      hooks_.postSolveObserver(command_, solution_, performance_);
    } catch (const std::exception& error) {
      LOG_EVERY_N_SEC(ERROR, kLogPeriodSeconds) << "[MpcServer] The post-solve observer threw: " << error.what();
    }
  }
}

void MpcServer::publishStatus(const absl::Status& attempt, double solveTimeMs) {
  fillSolverStatus(attempt, solveTimeMs, statusMessage_.mutable_solver_status());
  statusMessage_.set_observation_time(snapshot_.observation.time);
  statusMessage_.set_resets_served(robotServed_.requested);
  statusMessage_.set_full_resets_served(robotServed_.fullRequested);
  {
    absl::MutexLock lock(mutex_);
    statusMessage_.set_observations_received(receivedSinceStatus_);
    statusMessage_.set_observations_skipped(skippedSinceStatus_);
    receivedSinceStatus_ = 0;
    skippedSinceStatus_ = 0;
  }
  const absl::Status published = bus_.publish(topics::kMpcStatus, statusMessage_);
  if (published.ok()) {
    statusesPublished_.fetch_add(1);
  } else {
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds) << "[MpcServer] Publishing the status failed: " << published.message();
  }
}

}  // namespace ocs2::humanoid::ipc
