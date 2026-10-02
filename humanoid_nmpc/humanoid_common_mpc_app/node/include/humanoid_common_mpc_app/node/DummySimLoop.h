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
#include <cstdint>
#include <functional>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_oc/rollout/RolloutBase.h>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_mpc_ipc/RemoteMpcLink.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid::node {

/**
 * The dummy simulator of the laptop (humanoid_centroidal_mpc_dummy_sim, humanoid_wb_mpc_dummy_sim): the plant is the
 * MPC's own model, rolled out under the policy in use (the formulation's RolloutBase, MRT_BASE::rolloutPolicy()), and
 * the MPC is the MPC node, reached over the bus through an ipc::RemoteMpcLink exactly as the robot process reaches it.
 * It plays the robot (it publishes robot/mpc_observation as the bus node "robot"), so that it runs against the same MPC
 * node as the robot would. This is OCS2's MRT_ROS_Dummy_Loop without ROS:
 *
 *   1. it requests a full reset (the MPC node's first reset is a full one anyway) and sends the initial observation
 *      every step until the first policy has arrived;
 *   2. with a positive Config::mpcDesiredFrequency it runs synchronized with the MPC: every
 *      max(1, simulationFrequency / mpcDesiredFrequency) steps it waits for the policy solved from the observation it
 *      sent the step before (a policy that starts within 0.1 MPC periods of the plant's time), and it sends only the
 *      observation of the step before an MPC update. The plant's time therefore advances by exactly one MPC period per
 *      solve however long a solve takes, as the ROS dummy did;
 *   3. with mpcDesiredFrequency <= 0 it runs in real time: every step takes the newest policy, if one has arrived, and
 *      sends its observation;
 *   4. each step rolls the plant forward by 1 / simulationFrequency under the policy in use; the steps are paced on
 *      absolute deadlines (robot::realtime::PeriodicTimer, missed periods skipped).
 *
 * The RemoteMpcLink's health is not acted on: the plant keeps rolling the policy in use out, as the ROS dummy did.
 *
 * THREADS. run() runs the loop on the caller's thread; the bus's IO thread runs the link. latestObservation() and
 * statistics() may be called from any thread.
 */
class DummySimLoop {
 public:
  struct Config {
    /** [Hz] The plant's step rate (the ROS dummy's mrtDesiredFrequency, 100 Hz). Positive. */
    scalar_t simulationFrequency = 100.0;
    /** [Hz] The MPC's rate (mpc.mpcDesiredFrequency): positive runs the loop synchronized with it, <= 0 in real time. */
    scalar_t mpcDesiredFrequency = -1.0;
    /** The link to the MPC node: the model's dimensions, the policy timeout. */
    ipc::RemoteMpcLink::Config link;
    /** How often a synchronized wait for a policy that does not come logs that it is still waiting. */
    absl::Duration waitLogPeriod = absl::Seconds(5);
  };

  /** What the loop has done so far; every counter only grows. */
  struct Statistics {
    /** Plant steps rolled out. */
    uint64_t steps = 0;
    /** Synchronized waits that ended with the policy solved from the observation sent for them. */
    uint64_t synchronizedPolicies = 0;
    /** Policies taken into use (MRT_BASE::updatePolicy() swaps). */
    uint64_t policyUpdates = 0;
    ipc::RemoteMpcLink::Statistics link;
  };

  /**
   * A loop on `bus`, which must not be running yet and must publish as the robot (node name "robot" in the shipped
   * network), with `rollout` (cloned) as the plant. InvalidArgument for a frequency that is not positive or finite, and
   * the errors of RemoteMpcLink::Create().
   */
  static absl::StatusOr<std::unique_ptr<DummySimLoop>> Create(std::unique_ptr<robot::ipc::Bus> bus,
                                                              const RolloutBase& rollout,
                                                              Config config);

  /** Stops the bus. */
  ~DummySimLoop();
  DummySimLoop(const DummySimLoop&) = delete;
  DummySimLoop& operator=(const DummySimLoop&) = delete;

  /**
   * Starts the bus and runs the loop from `initialObservation` until `shouldStop()` returns true (checked every step
   * and while waiting). Returns OK when stopped, FailedPrecondition when called a second time, and Internal when the
   * rollout of the plant throws.
   */
  absl::Status run(const SystemObservation& initialObservation, const std::function<bool()>& shouldStop);

  /** The plant's newest observation (the initial one before the first step). Thread-safe. */
  SystemObservation latestObservation() const;

  /** Thread-safe. */
  Statistics statistics() const;

 private:
  DummySimLoop(std::unique_ptr<robot::ipc::Bus> bus, Config config);

  /** Steps until the first policy has arrived; false on stop. */
  bool awaitInitialPolicy(const SystemObservation& initialObservation, const std::function<bool()>& shouldStop);
  /** Waits for a policy that starts at `time` (synchronized loop); false on stop. */
  bool awaitPolicyFor(scalar_t time, const std::function<bool()>& shouldStop);
  /** One step of the plant under the policy in use. */
  SystemObservation forwardSimulation(const SystemObservation& observation);
  void publishLatest(const SystemObservation& observation);

  std::unique_ptr<robot::ipc::Bus> bus_;
  const Config config_;
  MpcResetSupervisor supervisor_;
  std::unique_ptr<ipc::RemoteMpcLink> link_;
  bool ran_ = false;

  mutable absl::Mutex latestMutex_;
  SystemObservation latestObservation_ ABSL_GUARDED_BY(latestMutex_);

  std::atomic<uint64_t> steps_{0};
  std::atomic<uint64_t> synchronizedPolicies_{0};
  std::atomic<uint64_t> policyUpdates_{0};
};

}  // namespace ocs2::humanoid::node
