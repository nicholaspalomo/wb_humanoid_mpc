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

#include <cstddef>
#include <functional>
#include <memory>

#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_mpc/MRT_BASE.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"

namespace ocs2::humanoid {

/**
 * How an MRT joint controller (CentroidalMpcMrtJointController, WBMpcMrtJointController) reaches the MPC that plans for
 * it: an ocs2::MRT_BASE (mrt()) the controller sends observations to and executes policies from, with a solver behind
 * it, and the MpcResetSupervisor through which the controller asks for resets and learns whether the solver is healthy.
 * See humanoid_nmpc/docs/distributed_runtime/README.md, "The MPC link".
 *
 * THREADS. The control thread (the one that runs computeJointControlAction()) calls the MRT methods and observeTime();
 * requestReset(), hasOutstandingReset() and isHealthy() may be called from any thread. Everything the control thread
 * calls must stay realtime-safe in steady state: no blocking on a lock another thread holds for long, no I/O. start()
 * and stop() are called from a non-realtime thread, before the control loop runs and after it has stopped.
 *
 * RESETS. A reset is requested with requestReset() and served by whatever solves (the solver thread of
 * InProcessMpcLink, the MPC node behind a remote link); hasOutstandingReset() is true from the request until a reset
 * that serves it is complete, so a control thread that sees it false together with MRT_BASE::isActivePolicyCurrent()
 * knows it executes a policy planned after every reset it asked for (MpcResetSupervisor's contract).
 *
 * IMPLEMENTATIONS.
 *  - InProcessMpcLink (InProcessMpcLink.h): ocs2::MPC_MRT_Interface and a solver thread of this process.
 *  - RemoteMpcLink (humanoid_mpc_ipc, written separately): the MPC node, over the bus. It must provide
 *      - mrt(): an MRT_BASE whose setCurrentObservation() never blocks the control thread (a triple buffer the
 *        communication thread publishes from, not a mutex), and whose buffer the communication thread fills with
 *        MRT_BASE::moveToBuffer() for every policy received, after dropping any policy that was solved before a reset
 *        the robot requested (its served-reset counters below the requested ones);
 *      - start(): begins publishing observations and accepting policies; every observation carries the request
 *        counters of this class's MpcResetSupervisor, so a lost message loses no reset. stop(): ends both and joins the
 *        link's threads; it must be idempotent and must not throw;
 *      - resets: when the communication thread sees a new request count it calls MRT_BASE::discardBufferedPolicy(),
 *        and when the first policy that serves the requested counters arrives it completes the ticket
 *        (MpcResetSupervisor::takeResetRequest(), completeReset()) before moving that policy to the buffer, so that
 *        hasOutstandingReset() keeps the contract above;
 *      - isHealthy(): overridden to combine the health the MPC node reports with the link's policy timeout, so that the
 *        controller holds the robot in JOINT_PD on link loss as it does for a failing solver.
 *    It ignores the reset target the controller hands its factory: the MPC node computes the formulation's reset
 *    target itself, from the observation the reset is served at.
 */
class MpcLink {
 public:
  using ResetKind = MpcResetSupervisor::ResetKind;

  /**
   * The target an MPC restarts from after a reset, for the observation current when the reset is served. It is
   * formulation-specific, so the controller supplies it (for instance the observed pose held still and upright, with
   * the weight on both feet). Called on the thread that serves resets.
   */
  using ResetTargetFunction = std::function<TargetTrajectories(const SystemObservation& observation)>;

  virtual ~MpcLink();

  MpcLink(const MpcLink&) = delete;
  MpcLink& operator=(const MpcLink&) = delete;

  // ------------------------------------------------------------------ lifecycle (non-realtime thread)

  /** Starts serving policies, beginning with a full reset from `initialObservation`. Call once. */
  virtual void start(const SystemObservation& initialObservation) = 0;

  /** Stops serving policies and joins the link's threads. Idempotent; also called by the destructor of every link. */
  virtual void stop() = 0;

  // ------------------------------------------------------------------ the MRT (control thread)

  /** The MRT the controller executes policies from. */
  virtual MRT_BASE& mrt() = 0;
  virtual const MRT_BASE& mrt() const = 0;

  /** Hands the solver the observation of this control cycle. */
  void setCurrentObservation(const SystemObservation& observation) { mrt().setCurrentObservation(observation); }
  /** Takes the newest policy into use, if one has arrived (MRT_BASE::updatePolicy()). */
  bool updatePolicy() { return mrt().updatePolicy(); }
  bool initialPolicyReceived() const { return mrt().initialPolicyReceived(); }
  /** The policy in use was solved after the last reset the solver served (MRT_BASE::isActivePolicyCurrent()). */
  bool isActivePolicyCurrent() const { return mrt().isActivePolicyCurrent(); }
  void evaluatePolicy(scalar_t currentTime, const vector_t& currentState, vector_t& mpcState, vector_t& mpcInput, size_t& mode) {
    mrt().evaluatePolicy(currentTime, currentState, mpcState, mpcInput, mode);
  }
  const PrimalSolution& getPolicy() const { return mrt().getPolicy(); }
  const CommandData& getCommand() const { return mrt().getCommand(); }

  // ------------------------------------------------------------------ resets and health

  /** Asks for a reset of the MPC, served before the next solve. Any thread. */
  void requestReset(ResetKind kind = ResetKind::kFull) { resetSupervisor_.requestReset(kind); }

  /** True from a requestReset() until a reset that serves it is complete. Any thread. */
  bool hasOutstandingReset() const { return resetSupervisor_.hasOutstandingReset(); }

  /** False while the solver keeps failing (MpcResetSupervisor::isHealthy()); a remote link adds its policy timeout. */
  virtual bool isHealthy() const { return resetSupervisor_.isHealthy(); }

  /** The control thread's clock check (MpcResetSupervisor::observeTime()): the rewind [s], after requesting a reset. */
  scalar_t observeTime(scalar_t time) { return resetSupervisor_.observeTime(time); }

  /** The supervisor's counters, for tests and diagnostics. */
  const MpcResetSupervisor& getResetSupervisor() const { return resetSupervisor_; }

 protected:
  MpcLink() = default;
  explicit MpcLink(MpcResetSupervisor::Config resetSupervisorConfig) : resetSupervisor_(resetSupervisorConfig) {}

  /** The side of the supervisor that serves resets and accounts for solves, for the implementation. */
  MpcResetSupervisor& resetSupervisor() { return resetSupervisor_; }

 private:
  MpcResetSupervisor resetSupervisor_;
};

/**
 * Makes an MRT joint controller's link once the controller can say how to reset its MPC: it is called by the
 * controller's constructor with the controller's reset target, which InProcessMpcLink serves its resets from and a
 * remote link ignores (see MpcLink). Returning nullptr is an error.
 */
using MpcLinkFactory = std::function<std::unique_ptr<MpcLink>(MpcLink::ResetTargetFunction resetTarget)>;

}  // namespace ocs2::humanoid
