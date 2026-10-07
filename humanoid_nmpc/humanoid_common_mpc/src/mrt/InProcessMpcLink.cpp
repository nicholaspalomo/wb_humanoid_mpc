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

#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"

#include <pthread.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ThreadAffinity.h"
#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"

namespace ocs2::humanoid {

InProcessMpcLink::InProcessMpcLink(MPC_BASE& mpc, ResetTargetFunction resetTarget, Config config)
    : mpcMrtInterface_(mpc),
      resetTarget_(std::move(resetTarget)),
      // Unused when realtime_; the frequency is not divided by then, so that no negative or infinite period is converted.
      mpcDeltaTMicroSeconds_(config.mpcDesiredFrequency > 0 ? static_cast<size_t>(1000000 / config.mpcDesiredFrequency) : 0),
      realtime_(config.mpcDesiredFrequency <= 0),
      solverThreadName_(std::move(config.solverThreadName)),
      solveObserver_(std::move(config.solveObserver)),
      execution_(config.execution) {
  ABSL_CHECK(resetTarget_ != nullptr) << "[InProcessMpcLink] a reset target function is required.";
}

InProcessMpcLink::~InProcessMpcLink() {
  stop();
}

MpcLinkFactory InProcessMpcLink::factory(MPC_BASE& mpc, Config config) {
  return [&mpc, config = std::move(config)](ResetTargetFunction resetTarget) -> std::unique_ptr<MpcLink> {
    return std::make_unique<InProcessMpcLink>(mpc, std::move(resetTarget), config);
  };
}

MpcLinkFactory InProcessMpcLink::factory(MPC_BASE& mpc, Config config, InProcessMpcLink* absl_nullable* absl_nullable created) {
  ABSL_CHECK(created != nullptr) << "[InProcessMpcLink] factory() needs a place to report the link it makes.";
  return [&mpc, config = std::move(config), created](ResetTargetFunction resetTarget) -> std::unique_ptr<MpcLink> {
    auto link = std::make_unique<InProcessMpcLink>(mpc, std::move(resetTarget), config);
    *created = link.get();
    return link;
  };
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void InProcessMpcLink::start(const SystemObservation& initialObservation) {
  if (started_.exchange(true)) {
    LOG(ERROR) << "[InProcessMpcLink] start() called on a link that was started before; ignored: a link starts once.";
    return;
  }
  // Set observation to MPC
  mpcMrtInterface_.setCurrentObservation(initialObservation);
  if (execution_ == Execution::kCaller) {
    // The start of the solver thread, on this thread; the caller runs the iterations (runSolverIteration()).
    resetMpcToCurrentObservation(/*full=*/true, "start-up");
    LOG(INFO) << "MPC is reset. NMPC solver started; the caller runs its iterations.";
    return;
  }
  solverWorker_ = std::jthread(&InProcessMpcLink::solverWorker, this);
}

void InProcessMpcLink::stop() {
  // Signal the solver thread to terminate
  terminateThread_.store(true);

  // Wait for the solver thread to finish if it's joinable
  if (solverWorker_.joinable()) {
    solverWorker_.join();
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void InProcessMpcLink::solverWorker() {
  const SystemCoreAllocation coreAlloc = getDefaultCoreAllocation();
  setThreadCpuAffinity(coreAlloc.mpcCores, pthread_self(), solverThreadName_);

  resetMpcToCurrentObservation(/*full=*/true, "start-up");
  LOG(INFO) << "MPC is reset. NMPC solver started!";

  size_t slowWarningCount = 0;
  while (!terminateThread_.load()) {
    const std::chrono::steady_clock::time_point targetTimeForNextIteration =
        std::chrono::steady_clock::now() + std::chrono::microseconds(mpcDeltaTMicroSeconds_);

    // A requested reset, then one solve; a failed solve may ask for a pause before the next.
    const SolverIterationResult iteration = solveOnce();
    if (iteration.retryDelay.count() > 0.0) {
      resetSupervisor().waitBeforeRetry(iteration.retryDelay, [this]() { return terminateThread_.load(); });
      continue;
    }

    if (!realtime_) {
      const std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
      if (currentTime > targetTimeForNextIteration) {
        const int64_t delay = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - targetTimeForNextIteration).count();
        if (delay > 1000 && (++slowWarningCount % 20 == 0)) {
          LOG(WARNING) << "MPC loop running slow by " << delay << " microseconds.";
        }
      } else {
        // Sleep in case sim loop is faster than specified
        std::this_thread::sleep_until(targetTimeForNextIteration);
      }
    }
  }
  LOG(INFO) << "Shutting down NMPC";
}

InProcessMpcLink::SolverIterationResult InProcessMpcLink::runSolverIteration() {
  SolverIterationResult refused;
  if (execution_ != Execution::kCaller) {
    refused.status = absl::FailedPreconditionError(
        "[InProcessMpcLink] runSolverIteration() on a link that runs its own solver thread (Execution::kSolverThread).");
    return refused;
  }
  if (!started_.load() || terminateThread_.load()) {
    refused.status = absl::FailedPreconditionError("[InProcessMpcLink] runSolverIteration() before start() or after stop().");
    return refused;
  }
  return solveOnce();
}

InProcessMpcLink::SolverIterationResult InProcessMpcLink::solveOnce() {
  // Serve a requested reset (a mode change, a discontinuity of the plant, a failed solve) before solving again.
  if (const std::optional<MpcResetSupervisor::ResetTicket> ticket = resetSupervisor().takeResetRequest()) {
    resetMpcToCurrentObservation(ticket->full, "requested");
    resetSupervisor().completeReset(*ticket);
  }

  // A failed solve requests a reset and, once the failures persist, a pause before the next attempt; the supervisor
  // logs what happened, once.
  SolverIterationResult result;
  result.status = mpcMrtInterface_.advanceMpc();
  result.retryDelay = resetSupervisor().onSolveResult(result.status);
  if (solveObserver_) {
    solveObserver_(result.status);
  }
  return result;
}

void InProcessMpcLink::resetMpcToCurrentObservation(bool full, absl::string_view reason) {
  const SystemObservation observation = mpcMrtInterface_.getCurrentObservation();
  if (full) {
    mpcMrtInterface_.resetMpcNode(resetTarget_(observation));
  } else {
    mpcMrtInterface_.resetMpcSolver(resetTarget_(observation));
  }
  // While the solver keeps failing it is reset before every attempt; the supervisor has said so once already.
  if (resetSupervisor().isHealthy()) {
    LOG(INFO) << (full ? "MPC reset" : "MPC solver reset") << " to the observation at t = " << observation.time << " s (" << reason << ").";
  }
}

}  // namespace ocs2::humanoid
