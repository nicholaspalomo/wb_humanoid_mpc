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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "google/protobuf/message.h"
#include "ocs2_mpc/SystemObservation.h"

#include "humanoid_mpc_msgs/mpc_observation.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid::node::test_support {

/**
 * The robot process and the operator GUI as an MPC node's tests script them: two buses on ephemeral loopback ports
 * (nodes "robot" and "operator") that an MPC node's bus is connected to, the policies the robot receives, and helpers
 * that resend a message until its effect is seen (the first messages of a fresh ZeroMQ connection can be lost).
 */
class ScriptedRobot {
 public:
  ScriptedRobot();
  /** Stops both buses. */
  ~ScriptedRobot();
  ScriptedRobot(const ScriptedRobot&) = delete;
  ScriptedRobot& operator=(const ScriptedRobot&) = delete;

  /** Connects `mpcBus` to the robot and the operator, and the robot to `mpcBus`. Before any of them starts. */
  void connect(robot::ipc::Bus& mpcBus);

  /** Subscribes the robot to mpc/policy and starts both buses. */
  absl::Status start();

  /** The MpcObservation message of `observation`, with its sequence number and the robot's reset counters. */
  static humanoid_mpc_msgs::MpcObservation observationMessage(const SystemObservation& observation,
                                                              uint64_t sequence,
                                                              uint64_t requested = 0,
                                                              uint64_t fullRequested = 0);

  /**
   * Sends `observation` until the policy solved from it arrives (a repeated sequence number is not solved again) and
   * returns that policy; nullopt when none arrives within `timeoutSeconds`.
   */
  std::optional<humanoid_mpc_msgs::MpcPolicy> solve(const humanoid_mpc_msgs::MpcObservation& observation, double timeoutSeconds = 30.0);

  /** Publishes `message` on `topic` as the operator until `received()` holds; false when it does not within 30 s. */
  bool sendAsOperator(absl::string_view topic, const google::protobuf::Message& message, const std::function<bool()>& received);

  /** The newest policy solved from the observation at `time`. */
  std::optional<humanoid_mpc_msgs::MpcPolicy> policySolvedAt(double time) const;
  /** Every policy received so far. */
  size_t numPolicies() const;

 private:
  std::unique_ptr<robot::ipc::Bus> robot_;
  std::unique_ptr<robot::ipc::Bus> operator_;
  mutable absl::Mutex mutex_;
  std::vector<humanoid_mpc_msgs::MpcPolicy> policies_ ABSL_GUARDED_BY(mutex_);
};

/** Polls `condition` every millisecond until it holds or `timeoutSeconds` pass; returns whether it held. */
bool waitFor(const std::function<bool()>& condition, double timeoutSeconds = 30.0);

}  // namespace ocs2::humanoid::node::test_support
