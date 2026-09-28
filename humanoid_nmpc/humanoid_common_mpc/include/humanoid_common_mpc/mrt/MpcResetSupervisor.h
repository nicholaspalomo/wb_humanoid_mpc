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
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>

#include "absl/status/status.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The reset and failure policy of the MRT joint controllers (CentroidalMpcMrtJointController, WBMpcMrtJointController),
 * which run the MPC on a solver thread of their own and the joint control on the caller's thread.
 *
 * RESETS. Any thread asks for one with requestReset(); the solver thread serves it between two solves
 * (takeResetRequest(), then the reset, then completeReset()). hasOutstandingReset() is true from the request until
 * the reset is complete, so a control thread that sees it false AND a policy solved after the last reset
 * (MRT_BASE::isActivePolicyCurrent()) knows it is executing a policy planned after every reset it asked for. The
 * counters make that race-free: a reset requested while another is being served is served next. A reset is FULL
 * (MPC_MRT_Interface::resetMpcNode(): the solver, the references and the modules) or of the SOLVER alone
 * (resetMpcSolver(): the schedule and the command state in execution are kept); when both are outstanding the full
 * one is served.
 *
 * FAILURES. The solver thread reports every solve (onSolveResult()). A failed solve requests a reset of the solver and
 * the next attempt follows at once: a robot in mid-stride keeps the schedule it is executing. After
 * maxConsecutiveFailures in a row the cause is taken to be state the solves inherit: the reset becomes a full one, the
 * MPC is declared unhealthy, one error is logged saying so and how to recover, the controllers hold the robot in
 * JOINT_PD instead of executing the stale policy, and the attempts back off exponentially instead of running at the
 * solve rate. A reset requested from outside (a mode change into WB_MPC) cuts the wait short. The first solve that
 * succeeds makes the MPC healthy again.
 *
 * CLOCK. observeTime() is the control thread's check that the observation time never runs backwards; when it does,
 * everything planned is timed on the old clock, so a reset is requested.
 */
class MpcResetSupervisor {
 public:
  struct Config {
    // LINT.IfChange(reset_supervisor_defaults)
    /// Consecutive failed solves after which the MPC is declared unhealthy.
    size_t maxConsecutiveFailures = 3;
    /// [s] Wait before the first attempt once unhealthy; doubled after every further failure, up to maxRetryInterval.
    scalar_t initialRetryInterval = 0.1;
    scalar_t maxRetryInterval = 2.0;
    // LINT.ThenChange(//humanoid_nmpc/docs/mpc_reset/README.md:controller_reset_events)
    /// [s] A time earlier than the previous one by more than this is a rewind of the clock.
    scalar_t clockRewindTolerance = 1e-6;
  };

  MpcResetSupervisor() : MpcResetSupervisor(Config()) {}
  explicit MpcResetSupervisor(Config config) : config_(config) {}

  MpcResetSupervisor(const MpcResetSupervisor&) = delete;
  MpcResetSupervisor& operator=(const MpcResetSupervisor&) = delete;

  const Config& getConfig() const { return config_; }

  /** What a reset clears; see the class comment. */
  enum class ResetKind {
    kFull,    ///< MPC_MRT_Interface::resetMpcNode(): the solver, the reference manager and the synchronized modules
    kSolver,  ///< MPC_MRT_Interface::resetMpcSolver(): the solver alone
  };

  /** A reset to serve: the request it serves, and whether it has to be a full one. */
  struct ResetTicket {
    uint64_t request = 0;
    uint64_t fullRequest = 0;
    bool full = false;
  };

  // ------------------------------------------------------------------ any thread

  /** Asks the solver thread for a reset of the MPC. */
  void requestReset(ResetKind kind = ResetKind::kFull);

  /** True from a requestReset() until the solver thread has completed a reset that serves it. */
  bool hasOutstandingReset() const;

  /** False from the maxConsecutiveFailures-th failed solve in a row until the next solve that succeeds. */
  bool isHealthy() const { return healthy_.load(); }

  /** Failed solves since the last one that succeeded. */
  size_t numConsecutiveFailures() const { return consecutiveFailures_.load(); }

  /** Resets requested and served so far, of any kind and full ones, for tests and diagnostics. */
  uint64_t numResetsServed() const { return resetsServed_.load(); }
  uint64_t numFullResetsServed() const { return fullResetsServed_.load(); }

  // ------------------------------------------------------------------ solver thread

  /**
   * The request to serve before the next solve, if one is outstanding: the solver thread resets the MPC - fully when
   * the ticket says so - and then passes the ticket to completeReset().
   */
  std::optional<ResetTicket> takeResetRequest() const;
  void completeReset(const ResetTicket& ticket);

  /**
   * Accounts for one solve. A failure requests a reset and returns how long to wait before the next attempt: a reset
   * of the solver and no wait for the first maxConsecutiveFailures - 1 failures in a row, then a full reset and
   * initialRetryInterval doubled per further failure up to maxRetryInterval. A success returns zero. Logs a warning for
   * each of the first failures, one error when the MPC becomes unhealthy, nothing while it stays so, and one line when
   * it recovers.
   */
  std::chrono::duration<scalar_t> onSolveResult(const absl::Status& status);

  /**
   * Waits up to `duration`. Returns early when a reset is requested after the failure that caused the wait (an operator
   * re-entering WB_MPC should not wait out the back-off) or when `stop()` becomes true (polled every 10 ms).
   */
  void waitBeforeRetry(std::chrono::duration<scalar_t> duration, const std::function<bool()>& stop);

  // ------------------------------------------------------------------ control thread

  /**
   * Checks that the observation clock never runs backwards. Returns the size of the rewind [s] (and requests a reset)
   * when `time` is earlier than the time of the previous call by more than clockRewindTolerance, and zero otherwise.
   */
  scalar_t observeTime(scalar_t time);

 private:
  const Config config_;

  // A full request increments fullResetsRequested_ before resetsRequested_, so a ticket that sees the second sees the first.
  std::atomic<uint64_t> resetsRequested_{0};
  std::atomic<uint64_t> resetsServed_{0};
  std::atomic<uint64_t> fullResetsRequested_{0};
  std::atomic<uint64_t> fullResetsServed_{0};

  std::atomic<bool> healthy_{true};
  std::atomic<size_t> consecutiveFailures_{0};
  uint64_t requestsAtLastFailure_ = 0;  // solver thread

  std::mutex waitMutex_;
  std::condition_variable waitCondition_;

  std::optional<scalar_t> lastObservationTime_;  // control thread
};

}  // namespace ocs2::humanoid
