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
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include <ocs2_core/Types.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>

#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "ocs2_mpc_test/ScriptedMpc.h"
#include "robot_ipc/Bus.h"

/**
 * What the tests of the network MPC link share: a small deterministic MPC (OCS2's ScriptedMpc, whose plan depends on
 * the time and the state it solves from), buses on ephemeral loopback ports, and a gate that holds a policy "in flight"
 * on the solver thread for as long as a test needs.
 */
namespace ocs2::humanoid::ipc::test_support {

inline constexpr size_t kStateDim = 3;
inline constexpr size_t kInputDim = 2;
/** As many modes as a humanoid's two feet have contact combinations. */
inline constexpr size_t kNumModes = 4;
/**
 * How long a test waits for something that happens in milliseconds before it fails: generous for a loaded machine, and
 * short enough that a test with several waits reports its failure before Bazel's timeout for a small test.
 */
inline constexpr absl::Duration kTimeout = absl::Seconds(5);

ModelDimensions modelDimensions();

/** An observation at `time` with every state entry `value` and a zero input. */
SystemObservation observationAt(scalar_t time, scalar_t value);

/** The reset target trajectories of the tests: the observation held. */
TargetTrajectories resetTargetsFor(const SystemObservation& observation);

/** A one-second horizon over the whole of which the solution is sent, unless `solutionTimeWindow` is positive. */
mpc::Settings mpcSettings(scalar_t solutionTimeWindow = -1.0);

/**
 * A ScriptedMpc that plans x(t) = x0 + (t - t0) from the time t0 and the state x0 it solves from, so that every
 * observation gives a policy of its own.
 */
std::unique_ptr<mpc_test::ScriptedMpc> makeScriptedMpc(const mpc::Settings& settings = mpcSettings());

/** Polls `condition` every millisecond until it holds or `timeout` passes; returns whether it held. */
bool waitFor(const std::function<bool()>& condition, absl::Duration timeout = kTimeout);

/** A publishing bus of node `name` on an ephemeral loopback port, alone in its network. */
std::unique_ptr<robot::ipc::Bus> createNodeBus(const std::string& name);

/** A bus that only subscribes, to the publisher `nodeName` at `port` on the loopback interface. */
std::unique_ptr<robot::ipc::Bus> createSubscriberBus(const std::string& nodeName, int port);

/** Connects the subscriber of each of the two buses to the publisher of the other. Before either starts. */
void connectBoth(robot::ipc::Bus& first, robot::ipc::Bus& second);

/**
 * A policy message as an MPC node sends it: solved from the observation at `initTime` (state `value`), planned as
 * x(t) = value + (t - initTime) on `nodes` nodes up to `finalTime`, with a feedforward controller, the given counters
 * and a healthy solver status of `solveCount`.
 */
humanoid_mpc_msgs::MpcPolicy makePolicyMessage(scalar_t initTime,
                                               scalar_t finalTime,
                                               scalar_t value,
                                               uint64_t resetsServed,
                                               uint64_t fullResetsServed,
                                               uint64_t solveCount,
                                               size_t nodes = 11);

/**
 * Holds the threads that pass it while it is closed. The MPC server's annotations provider passes it, so that a test
 * can keep a solved policy from being published - in flight - and let it go when it wants.
 */
class Gate {
 public:
  /** Every pass() from now on waits, except as many as release() lets through. */
  void close();
  /** Lets every waiting and future pass() through. */
  void open();
  /** Lets `count` more pass() calls through while the gate is closed. */
  void release(size_t count);
  /** Counts the arrival, then waits while the gate is closed and nothing is released. */
  void pass();
  /** The pass() calls so far, including the one waiting. */
  size_t arrivals() const;

 private:
  bool canPass() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_;
  bool closed_ ABSL_GUARDED_BY(mutex_) = false;
  size_t permits_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t arrivals_ ABSL_GUARDED_BY(mutex_) = 0;
};

/** Counts the resets of the reference manager's companions: only a full reset (MPC_BASE::reset()) resets the modules. */
class CountingModule final : public SolverSynchronizedModule {
 public:
  void preSolverRun(scalar_t /*initTime*/,
                    scalar_t /*finalTime*/,
                    const vector_t& /*initState*/,
                    const ReferenceManagerInterface& /*referenceManager*/) override {}
  void postSolverRun(const PrimalSolution& /*primalSolution*/) override {}
  void reset() override { resets_.fetch_add(1); }
  uint64_t resets() const { return resets_.load(); }

 private:
  std::atomic<uint64_t> resets_{0};
};

}  // namespace ocs2::humanoid::ipc::test_support
