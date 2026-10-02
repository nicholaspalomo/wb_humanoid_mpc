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
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "humanoid_mpc_msgs/fsm_state.pb.h"
#include "humanoid_mpc_msgs/loop_timing.pb.h"
#include "humanoid_mpc_msgs/mpc_status.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/NetworkConfig.h"

namespace ocs2::humanoid::test_support {

/** The lowest and highest base heights [m] of the robot/state samples received since ScriptedOperator::resetBaseHeightRange(). */
struct BaseHeightRange {
  double min = std::numeric_limits<double>::infinity();
  double max = -std::numeric_limits<double>::infinity();
  uint64_t samples = 0;
};

/**
 * The operator of an end-to-end test: the remote control GUI as the test scripts it, as the bus node "operator" of the
 * test's network file. It keeps the newest robot/fsm_state, robot/state, robot/loop_timing and mpc/status, counts the
 * viz/scene messages, and sends FSM and walking velocity commands as the GUI does.
 */
class ScriptedOperator {
 public:
  /** The operator's bus on `network` (which must name "operator") and its subscriptions; CHECK-fails on an error. */
  explicit ScriptedOperator(const robot::ipc::NetworkConfig& network);

  /** stop(). */
  ~ScriptedOperator();
  ScriptedOperator(const ScriptedOperator&) = delete;
  ScriptedOperator& operator=(const ScriptedOperator&) = delete;

  /** Starts the bus; CHECK-fails on an error. */
  void start();

  /** Stops the bus. Idempotent. */
  void stop();

  std::optional<humanoid_mpc_msgs::FsmState> fsmState() const;
  std::optional<humanoid_mpc_msgs::RobotStateSample> sample() const;
  std::optional<humanoid_mpc_msgs::LoopTiming> timing() const;
  std::optional<humanoid_mpc_msgs::MpcStatus> mpcStatus() const;
  uint64_t scenesReceived() const;

  /**
   * Sends the FSM command `command` with the next sequence number, again every 50 ms (the first messages of a fresh
   * ZeroMQ connection can be lost; the robot drops the repeats by their sequence), until the robot's FSM state
   * satisfies `reached`; false when it does not within `timeout`.
   */
  bool sendFsmCommand(const std::string& command,
                      const std::function<bool(const humanoid_mpc_msgs::FsmState&)>& reached,
                      absl::Duration timeout = absl::Seconds(10));

  /** sendFsmCommand(`mode`) until the robot reports that mode. */
  bool enterMode(const std::string& mode);

  /** Publishes `command` on operator/walking_velocity_command every 40 ms (25 Hz, as the GUI) for `duration`. */
  void sendVelocityCommand(const humanoid_mpc_msgs::WalkingVelocityCommand& command, absl::Duration duration);

  /** Starts a new BaseHeightRange. */
  void resetBaseHeightRange();
  BaseHeightRange baseHeightRange() const;

 private:
  std::unique_ptr<robot::ipc::Bus> bus_;
  uint64_t sequence_ = 0;
  mutable absl::Mutex mutex_;
  std::optional<humanoid_mpc_msgs::FsmState> fsmState_ ABSL_GUARDED_BY(mutex_);
  std::optional<humanoid_mpc_msgs::RobotStateSample> sample_ ABSL_GUARDED_BY(mutex_);
  std::optional<humanoid_mpc_msgs::LoopTiming> timing_ ABSL_GUARDED_BY(mutex_);
  std::optional<humanoid_mpc_msgs::MpcStatus> mpcStatus_ ABSL_GUARDED_BY(mutex_);
  uint64_t scenesReceived_ ABSL_GUARDED_BY(mutex_) = 0;
  BaseHeightRange baseHeightRange_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace ocs2::humanoid::test_support
