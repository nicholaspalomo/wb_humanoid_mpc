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
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>
#include <ocs2_mpc/MRT_BASE.h>
#include <ocs2_mpc/SystemObservation.h>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/mrt/MpcLink.h"

namespace ocs2::humanoid {

/**
 * The MPC in this process: an ocs2::MPC_MRT_Interface over the caller's MPC_BASE, advanced by a solver thread of the
 * link's own (the thread the MRT joint controllers used to run themselves), or by the caller (Execution::kCaller).
 *
 * The solver thread is pinned to the MPC cores (ThreadAffinity.h) and, from start(), resets the MPC fully from the
 * initial observation, then loops until stop(). Steps 1 to 3 are the solver iteration, the same code that
 * runSolverIteration() runs:
 *  1. it serves a requested reset (MpcResetSupervisor::takeResetRequest()) from the observation current then, through
 *     the controller's reset target: MPC_MRT_Interface::resetMpcNode() for a full one, resetMpcSolver() otherwise;
 *  2. it solves once (MPC_MRT_Interface::advanceMpc()), which moves the policy to the MRT buffer;
 *  3. it reports the result to the supervisor (MpcResetSupervisor::onSolveResult()), then to the solve observer;
 *  4. a failure that asks for a pause waits it out (waitBeforeRetry(), cut short by a new reset request or by stop())
 *     and starts the next iteration;
 *  5. it paces itself at Config::mpcDesiredFrequency, sleeping out what is left of the period, and warns once every 20
 *     iterations that run more than a millisecond late. A frequency <= 0 runs the solves back to back ("realtime").
 *
 * With Execution::kCaller, start() serves the start-up reset on the calling thread and starts no thread; the caller runs
 * the iterations itself (runSolverIteration(): steps 1 to 3) and does steps 4 and 5 on a clock of its own, ending a pause
 * early when MpcResetSupervisor::resetRequestedSinceLastFailure() says what waitBeforeRetry() polls. That is how
 * the lockstep closed loop of humanoid_nmpc/humanoid_mpc_validation solves in simulation time, with the production
 * controller and the production reset and failure policy.
 *
 * setCurrentObservation() copies the observation under MPC_MRT_Interface's observation mutex, which the solver thread
 * holds only for the copy it takes at the start of a solve, into storage it keeps from solve to solve, so that the
 * control thread never waits for an allocation. A reset served (resetMpcToCurrentObservation()) copies it once more.
 * The lock itself stays: a link whose control thread must never wait at all is RemoteMpcLink (MpcLink.h).
 */
class InProcessMpcLink final : public MpcLink {
 public:
  /**
   * Called on the thread that solves after every solve attempt, with its status, once the supervisor has accounted for
   * it.
   */
  using SolveObserver = std::function<void(const absl::Status& solveStatus)>;

  /** Who runs the solver iterations. */
  enum class Execution {
    /// start() starts the link's solver thread, which runs them until stop() (the class comment). The default.
    kSolverThread,
    /// start() serves the start-up reset on the calling thread and starts no thread; the caller then calls
    /// runSolverIteration() wherever the solver thread would solve, and waits out its retry delays on its own clock,
    /// cut short by a reset requested after the failure (MpcResetSupervisor::resetRequestedSinceLastFailure()).
    kCaller,
  };

  struct Config {
    /// [Hz] The rate the solver thread runs at; <= 0 runs the solves back to back. Unused with Execution::kCaller.
    scalar_t mpcDesiredFrequency = -1.0;
    /// Name of the solver thread in the log of its CPU affinity.
    std::string solverThreadName = "MPC Solver Thread";
    /// Optional; see SolveObserver.
    SolveObserver solveObserver;
    /// See Execution.
    Execution execution = Execution::kSolverThread;
  };

  /** The outcome of one solver iteration (runSolverIteration()). */
  struct SolverIterationResult {
    /// Of the solve, MPC_MRT_Interface::advanceMpc(); FailedPrecondition when the link refused to run the iteration.
    absl::Status status;
    /// The pause before the next attempt (MpcResetSupervisor::onSolveResult()); zero: solve again at the next period.
    std::chrono::duration<scalar_t> retryDelay{0.0};
  };

  /**
   * @param mpc          The MPC to advance; it must outlive the link.
   * @param resetTarget  The target every reset restarts the MPC from (see MpcLink::ResetTargetFunction); must be set.
   */
  InProcessMpcLink(MPC_BASE& mpc, ResetTargetFunction resetTarget, Config config);

  /** Stops the solver thread, if it runs, and waits for it. */
  ~InProcessMpcLink() override;

  /** The MpcLinkFactory of the MRT joint controllers for an InProcessMpcLink over `mpc`, which must outlive the link. */
  static MpcLinkFactory factory(MPC_BASE& mpc, Config config);

  /**
   * factory(), and the factory points `*created` at every link it makes, for a caller that runs the solver iterations
   * of the controller's link itself (Execution::kCaller). The link is the controller's: `*created` is valid while the
   * controller is, and `created` must outlive the factory's calls.
   */
  static MpcLinkFactory factory(MPC_BASE& mpc, Config config, InProcessMpcLink** created);

  /**
   * Hands the MPC the initial observation and starts the solver thread, which begins with a full reset from it; with
   * Execution::kCaller, serves that reset here, on the calling thread, and starts no thread. A link starts once: a
   * second call, also after stop(), is refused with an error log.
   */
  void start(const SystemObservation& initialObservation) override;
  void stop() override;

  /**
   * One iteration of the solver on the calling thread, for Execution::kCaller (see the class comment): serves a
   * requested reset from the observation current now, solves once, and reports the result to the reset supervisor and
   * the solve observer; the solver thread runs exactly this between its waits. The caller waits out `retryDelay`
   * before the next call, as the thread would. Refused, with FailedPrecondition and nothing solved, before start(),
   * after stop() and when the link runs its own solver thread.
   */
  SolverIterationResult runSolverIteration();

  MRT_BASE& mrt() override { return mpcMrtInterface_; }
  const MRT_BASE& mrt() const override { return mpcMrtInterface_; }

 private:
  /** The solver thread (see the class comment). */
  void solverWorker();

  /** Steps 1 to 3 of the class comment on the calling thread: the solver thread's iteration and runSolverIteration(). */
  SolverIterationResult solveOnce();

  /**
   * The thread that solves: MPC_MRT_Interface::resetMpcNode() (`full`) or resetMpcSolver() from the observation current
   * now. Logged while the MPC is healthy.
   */
  void resetMpcToCurrentObservation(bool full, absl::string_view reason);

  MPC_MRT_Interface mpcMrtInterface_;
  const ResetTargetFunction resetTarget_;
  const size_t mpcDeltaTMicroSeconds_;
  const bool realtime_;  // True if MPC is to be run as fast as possible
  const std::string solverThreadName_;
  const SolveObserver solveObserver_;
  const Execution execution_;

  std::atomic_bool started_{false};
  std::atomic_bool terminateThread_{false};
  std::jthread solverWorker_;
};

}  // namespace ocs2::humanoid
