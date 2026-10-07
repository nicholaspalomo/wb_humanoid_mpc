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

#include "humanoid_common_mpc_app/node/DummySimLoop.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "robot_realtime/PeriodicTimer.h"

namespace ocs2::humanoid::node {
namespace {

constexpr double kLogPeriodSeconds = 5.0;
/** How long a wait for a policy sleeps between two looks at the policy buffer. */
constexpr absl::Duration kPolicyPollPeriod = absl::Microseconds(100);

}  // namespace

absl::StatusOr<std::unique_ptr<DummySimLoop>> DummySimLoop::Create(std::unique_ptr<robot::ipc::Bus> bus,
                                                                   const RolloutBase& rollout,
                                                                   Config config) {
  if (bus == nullptr) {
    return absl::InvalidArgumentError("DummySimLoop: no bus");
  }
  if (!std::isfinite(config.simulationFrequency) || config.simulationFrequency <= 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("DummySimLoop: the simulation frequency is ", config.simulationFrequency, " Hz; it must be a positive number"));
  }
  if (!std::isfinite(config.mpcDesiredFrequency)) {
    return absl::InvalidArgumentError(absl::StrCat("DummySimLoop: the MPC frequency is ", config.mpcDesiredFrequency, " Hz"));
  }
  std::unique_ptr<DummySimLoop> loop = absl::WrapUnique(new DummySimLoop(std::move(bus), config));
  ASSIGN_OR_RETURN(loop->link_, ipc::RemoteMpcLink::Create(*loop->bus_, loop->supervisor_, config.link));
  loop->link_->initRollout(&rollout);
  return loop;
}

DummySimLoop::DummySimLoop(std::unique_ptr<robot::ipc::Bus> bus, Config config) : bus_(std::move(bus)), config_(config) {}

DummySimLoop::~DummySimLoop() {
  // Before the link and the supervisor its callbacks reach are destroyed.
  bus_->stop();
}

absl::Status DummySimLoop::run(const SystemObservation& initialObservation, const std::function<bool()>& shouldStop) {
  if (ran_) {
    return absl::FailedPreconditionError("DummySimLoop::run(): the loop has run before; it runs once");
  }
  ran_ = true;
  publishLatest(initialObservation);
  RETURN_IF_ERROR(bus_->start());

  // As the ROS dummy did. The remote link sends the request in the observation's counters; the MPC node resets to the
  // target its own reset target function makes of the observation, not to this one.
  link_->resetMpcNode(TargetTrajectories({initialObservation.time}, {initialObservation.state}, {initialObservation.input}));
  LOG(INFO) << "[DummySim] Waiting for the initial policy of the MPC node ...";
  if (!awaitInitialPolicy(initialObservation, shouldStop)) {
    return absl::OkStatus();
  }
  LOG(INFO) << "[DummySim] Initial policy received.";

  const bool synchronized = config_.mpcDesiredFrequency > 0.0;
  // The plant steps per MPC update.
  const size_t mpcUpdateRatio =
      synchronized ? std::max<size_t>(static_cast<size_t>(config_.simulationFrequency / config_.mpcDesiredFrequency), 1) : 1;
  LOG(INFO) << "[DummySim] Rolling the plant out at " << config_.simulationFrequency << " Hz, "
            << (synchronized ? absl::StrCat("synchronized with the MPC: one update every ", mpcUpdateRatio, " steps")
                             : std::string("in real time"))
            << ".";

  const std::chrono::nanoseconds period(static_cast<int64_t>(std::llround(1.0e9 / config_.simulationFrequency)));
  robot::realtime::PeriodicTimer timer(period, robot::realtime::OverrunPolicy::kSkipMissedPeriods);
  SystemObservation observation = initialObservation;
  if (!synchronized) {
    // The first rollout needs a policy in use; the buffer holds one, and updatePolicy() takes it unless the IO thread
    // is moving a newer one in at this very moment.
    while (!link_->updatePolicy()) {
      if (shouldStop()) return absl::OkStatus();
      absl::SleepFor(kPolicyPollPeriod);
    }
    policyUpdates_.fetch_add(1);
  }
  timer.start();
  for (size_t loopCounter = 0; !shouldStop(); ++loopCounter) {
    if (synchronized) {
      if (loopCounter % mpcUpdateRatio == 0 && !awaitPolicyFor(observation.time, shouldStop)) {
        break;
      }
    } else if (link_->updatePolicy()) {
      policyUpdates_.fetch_add(1);
    }

    ASSIGN_OR_RETURN(observation, forwardSimulation(observation));

    // Synchronized, the MPC solves only from the observation of the step before its update.
    if (!synchronized || (loopCounter + 1) % mpcUpdateRatio == 0) {
      link_->setCurrentObservation(observation);
    }
    publishLatest(observation);
    steps_.fetch_add(1);
    LOG_EVERY_N_SEC(INFO, kLogPeriodSeconds) << "[DummySim] t = " << observation.time << " s, base state "
                                             << observation.state.head(std::min<Eigen::Index>(12, observation.state.size())).transpose();
    timer.waitForNextPeriod();
  }
  LOG(INFO) << "[DummySim] Stopped at t = " << observation.time << " s.";
  return absl::OkStatus();
}

bool DummySimLoop::awaitInitialPolicy(const SystemObservation& initialObservation, const std::function<bool()>& shouldStop) {
  const std::chrono::nanoseconds period(static_cast<int64_t>(std::llround(1.0e9 / config_.simulationFrequency)));
  robot::realtime::PeriodicTimer timer(period, robot::realtime::OverrunPolicy::kSkipMissedPeriods);
  timer.start();
  while (!link_->initialPolicyReceived()) {
    if (shouldStop()) return false;
    link_->setCurrentObservation(initialObservation);
    timer.waitForNextPeriod();
  }
  return true;
}

bool DummySimLoop::awaitPolicyFor(scalar_t time, const std::function<bool()>& shouldStop) {
  // A policy solved from the observation sent for this update starts at its time; an older one, still in flight when
  // the observation was sent, starts an MPC period or more earlier.
  const scalar_t tolerance = 0.1 / config_.mpcDesiredFrequency;
  absl::Time nextLog = absl::Now() + config_.waitLogPeriod;
  while (!shouldStop()) {
    if (link_->updatePolicy()) {
      policyUpdates_.fetch_add(1);
      if (std::abs(link_->getPolicy().timeTrajectory_.front() - time) < tolerance) {
        synchronizedPolicies_.fetch_add(1);
        return true;
      }
    }
    if (absl::Now() > nextLog) {
      LOG(INFO) << "[DummySim] Still waiting for the policy solved from the observation at t = " << time << " s.";
      nextLog = absl::Now() + config_.waitLogPeriod;
    }
    absl::SleepFor(kPolicyPollPeriod);
  }
  return false;
}

absl::StatusOr<SystemObservation> DummySimLoop::forwardSimulation(const SystemObservation& observation) {
  const scalar_t timeStep = 1.0 / config_.simulationFrequency;
  SystemObservation next;
  next.time = observation.time + timeStep;
  try {  // NOLINT(exceptions): MRT_BASE::rolloutPolicy() and OCS2's rollouts throw; their exception becomes a Status here.
    link_->rolloutPolicy(observation.time, observation.state, timeStep, next.state, next.input, next.mode);
  } catch (const std::exception& error) {  // NOLINT(exceptions): the boundary of the try above.
    return absl::InternalError(absl::StrCat("DummySimLoop: the rollout of the plant failed at t = ", observation.time, ": ", error.what()));
  }
  return next;
}

void DummySimLoop::publishLatest(const SystemObservation& observation) {
  absl::MutexLock lock(latestMutex_);
  latestObservation_ = observation;
}

SystemObservation DummySimLoop::latestObservation() const {
  absl::MutexLock lock(latestMutex_);
  return latestObservation_;
}

DummySimLoop::Statistics DummySimLoop::statistics() const {
  Statistics statistics;
  statistics.steps = steps_.load();
  statistics.synchronizedPolicies = synchronizedPolicies_.load();
  statistics.policyUpdates = policyUpdates_.load();
  statistics.link = link_->statistics();
  return statistics;
}

}  // namespace ocs2::humanoid::node
