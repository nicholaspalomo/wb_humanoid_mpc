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
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid::teleop {

/** The rate of operator/walking_velocity_command (humanoid_nmpc/docs/distributed_runtime/README.md, "Topics"). */
inline constexpr absl::Duration kVelocityCommandPeriod = absl::Milliseconds(40);

/**
 * Publishes the operator's latest walking command on operator/walking_velocity_command at the topic's 25 Hz, from the
 * bus's IO thread, so that an MPC node that connects later, or a message ZeroMQ drops, still gets it (the subscribers
 * take the latest one). Nothing is published until the first setCommand().
 */
class VelocityCommandRepeater {
 public:
  /** Registers the periodic publication on `bus`, which must not be running yet and must outlive the repeater. */
  static absl::StatusOr<std::unique_ptr<VelocityCommandRepeater>> Create(robot::ipc::Bus& bus,
                                                                         absl::Duration period = kVelocityCommandPeriod);

  /** The command to publish from now on. Any thread. */
  void setCommand(const humanoid_mpc_msgs::WalkingVelocityCommand& command);

  /** Messages handed to the bus so far. Thread-safe. */
  uint64_t published() const { return state_->published.load(); }

 private:
  /** Shared with the bus's callback, so that a bus that outlives the repeater never reaches a destroyed one. */
  struct State {
    absl::Mutex mutex;
    std::optional<humanoid_mpc_msgs::WalkingVelocityCommand> command ABSL_GUARDED_BY(mutex);
    std::atomic<uint64_t> published{0};
  };

  explicit VelocityCommandRepeater(std::shared_ptr<State> state) : state_(std::move(state)) {}

  std::shared_ptr<State> state_;
};

}  // namespace ocs2::humanoid::teleop
