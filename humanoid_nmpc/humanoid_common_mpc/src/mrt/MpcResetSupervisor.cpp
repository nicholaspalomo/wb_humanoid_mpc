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

#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"

#include <algorithm>
#include <cmath>

#include "absl/log/log.h"

namespace ocs2::humanoid {

void MpcResetSupervisor::requestReset(ResetKind kind) {
  {
    // Under the mutex the waiting solver thread evaluates its predicate with, so the notification cannot be lost.
    std::lock_guard<std::mutex> lock(waitMutex_);
    if (kind == ResetKind::kFull) fullResetsRequested_.fetch_add(1);
    resetsRequested_.fetch_add(1);
  }
  waitCondition_.notify_all();
}

bool MpcResetSupervisor::hasOutstandingReset() const {
  // Served first: it only ever catches up with requested, so reading it first can only err towards "outstanding".
  const uint64_t served = resetsServed_.load();
  return resetsRequested_.load() != served;
}

std::optional<MpcResetSupervisor::ResetTicket> MpcResetSupervisor::takeResetRequest() const {
  ResetTicket ticket;
  ticket.request = resetsRequested_.load();
  if (ticket.request == resetsServed_.load()) return std::nullopt;
  // Read after the request count: every full request counted there is counted here too.
  ticket.fullRequest = fullResetsRequested_.load();
  ticket.full = ticket.fullRequest != fullResetsServed_.load();
  return ticket;
}

void MpcResetSupervisor::completeReset(const ResetTicket& ticket) {
  if (ticket.full) fullResetsServed_.store(std::max(ticket.fullRequest, fullResetsServed_.load()));
  resetsServed_.store(std::max(ticket.request, resetsServed_.load()));
}

std::chrono::duration<scalar_t> MpcResetSupervisor::onSolveResult(const absl::Status& status) {
  if (status.ok()) {
    const size_t failures = consecutiveFailures_.exchange(0);
    if (!healthy_.exchange(true)) {
      LOG(INFO) << "[MpcResetSupervisor] The MPC solver recovered after " << failures
                << " failed solves; WB_MPC executes its policy again.";
    }
    return std::chrono::duration<scalar_t>(0.0);
  }

  const size_t failures = consecutiveFailures_.fetch_add(1) + 1;
  // The first failures reset the solver alone and a robot in mid-stride keeps its schedule; failures that persist
  // through that are taken to come from state the solves inherit, which only the full reset clears.
  requestReset(failures < config_.maxConsecutiveFailures ? ResetKind::kSolver : ResetKind::kFull);
  requestsAtLastFailure_ = resetsRequested_.load();

  if (failures < config_.maxConsecutiveFailures) {
    LOG(WARNING) << "[MpcResetSupervisor] MPC solve failed (" << failures << " of " << config_.maxConsecutiveFailures
                 << " in a row before the controller holds the robot): " << status.message() << " Resetting the solver and retrying.";
    return std::chrono::duration<scalar_t>(0.0);
  }
  if (failures == config_.maxConsecutiveFailures) {
    healthy_.store(false);
    LOG(ERROR) << "[MpcResetSupervisor] The MPC solver failed " << failures << " times in a row (last: " << status.message()
               << "). WB_MPC now holds the robot with the JOINT_PD action instead of the MPC policy, and the solver retries from a "
                  "full reset every "
               << config_.initialRetryInterval << " to " << config_.maxRetryInterval
               << " s without logging each attempt. It resumes on its own at the first solve that succeeds. To retry at once, "
                  "switch to JOINT_PD and back to WB_MPC; if it keeps failing, lock the gantry, or restart the controller.";
  }
  const scalar_t exponent = static_cast<scalar_t>(failures - config_.maxConsecutiveFailures);
  const scalar_t interval = std::min(config_.initialRetryInterval * std::pow(2.0, exponent), config_.maxRetryInterval);
  return std::chrono::duration<scalar_t>(std::max(interval, scalar_t(0.0)));
}

void MpcResetSupervisor::waitBeforeRetry(std::chrono::duration<scalar_t> duration, const std::function<bool()>& stop) {
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(duration);
  std::unique_lock<std::mutex> lock(waitMutex_);
  while (std::chrono::steady_clock::now() < deadline) {
    if (resetsRequested_.load() != requestsAtLastFailure_) return;
    if (stop && stop()) return;
    const std::chrono::steady_clock::time_point wakeUp =
        std::min(deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
    waitCondition_.wait_until(lock, wakeUp);
  }
}

scalar_t MpcResetSupervisor::observeTime(scalar_t time) {
  scalar_t rewind = 0.0;
  if (lastObservationTime_.has_value() && time < *lastObservationTime_ - config_.clockRewindTolerance) {
    rewind = *lastObservationTime_ - time;
    requestReset();
  }
  lastObservationTime_ = time;
  return rewind;
}

}  // namespace ocs2::humanoid
