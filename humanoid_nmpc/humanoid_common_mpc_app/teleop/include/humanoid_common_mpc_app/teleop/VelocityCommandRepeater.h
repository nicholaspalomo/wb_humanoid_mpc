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
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/teleop/KeyboardVelocityCommand.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid::teleop {

/** The rate of operator/walking_velocity_command
 * (humanoid_nmpc/docs/distributed_runtime/README.md, "Topics"). */
inline constexpr absl::Duration kVelocityCommandPeriod = absl::Milliseconds(40);
/** How often the repeater looks at the reference file's modification time for
 * changed command limits. */
inline constexpr absl::Duration kCommandLimitsCheckPeriod = absl::Seconds(1);
/** How long publishStopCommand() waits for the bus to write the stop command,
 * and then for DDS to send it. */
inline constexpr absl::Duration kStopCommandSendTimeout = absl::Seconds(1);
inline constexpr absl::Duration kStopCommandGracePeriod =
    absl::Milliseconds(100);

/**
 * Publishes the operator's latest walking command on
 * operator/walking_velocity_command at the topic's 25 Hz, from the bus's IO
 * thread, so that an MPC node that DDS discovers later, or a sample lost on the
 * link (the topic is BEST_EFFORT), still gets it (the subscribers take the
 * latest one). Nothing is published until the first setCommand().
 *
 * The command is held in physical units, not as the normalized message: the MPC
 * scales the message back by its own copy of the command limits, which its
 * parameter updater reloads when the reference file changes. So the repeater
 * looks at the reference file about once every kCommandLimitsCheckPeriod and,
 * when it changed and still reads, rebuilds the message with the new limits;
 * the robot keeps walking at the speed the operator typed (clamped to a lowered
 * limit). A file that no longer reads keeps the previous limits, with a
 * warning.
 *
 * Thread-safe: setCommand() and publishStopCommand() may be called from any
 * thread while the bus runs.
 */
class VelocityCommandRepeater {
 public:
  /**
   * Registers the periodic publication on `bus`, which must not be running yet
   * and must outlive the repeater. `referenceFile` is the file `limits` were
   * read from (loadKeyboardCommandLimits()); it is read again on the bus's IO
   * thread when its modification time changes.
   */
  static absl::StatusOr<std::unique_ptr<VelocityCommandRepeater>> Create(
      robot::ipc::Bus& bus,
      std::string referenceFile,
      const KeyboardCommandLimits& limits,
      absl::Duration period = kVelocityCommandPeriod,
      absl::Duration limitsCheckPeriod = kCommandLimitsCheckPeriod);

  VelocityCommandRepeater(const VelocityCommandRepeater&) = delete;
  VelocityCommandRepeater& operator=(const VelocityCommandRepeater&) = delete;
  VelocityCommandRepeater(VelocityCommandRepeater&&) = delete;
  VelocityCommandRepeater& operator=(VelocityCommandRepeater&&) = delete;
  ~VelocityCommandRepeater() = default;

  /**
   * The command to publish from now on: [v_x m/s, v_y m/s, pelvis height change
   * m, yaw rate rad/s], as the operator's terminal showed it, and the limits it
   * was clamped with (which replace the repeater's). Any thread.
   */
  void setCommand(const vector4_t& command,
                  const KeyboardCommandLimits& limits);

  /**
   * Replaces the command with the stop command - zero velocities, the pelvis
   * height change kept - and waits until the bus has written it and
   * `gracePeriod` more, for DDS to send it before the bus deletes its writer.
   * The MPC keeps the last walking command it received, so a teleoperation that
   * ends without it leaves the robot walking. Blocks the caller for up to
   * `timeout` plus `gracePeriod`; never call it on the bus's IO thread.
   *
   * @return true once the bus sent the stop command; false when no command was
   * ever set (nothing to stop) or the bus did not send it within `timeout` (a
   * bus that is not running).
   */
  bool publishStopCommand(absl::Duration timeout = kStopCommandSendTimeout,
                          absl::Duration gracePeriod = kStopCommandGracePeriod);

  /** Messages handed to the bus so far. Thread-safe. */
  uint64_t published() const { return state_->published.load(); }

  /** Times the command limits were read again from a changed reference file.
   * Thread-safe. */
  uint64_t limitsReloads() const { return state_->limitsReloads.load(); }

 private:
  /** Shared with the bus's callback, so that a bus that outlives the repeater
   * never reaches a destroyed one. */
  struct State {
    absl::Mutex mutex;
    /** The held command in physical units; nullopt before the first
     * setCommand(). */
    std::optional<vector4_t> command ABSL_GUARDED_BY(mutex);
    KeyboardCommandLimits limits ABSL_GUARDED_BY(mutex);
    /** Incremented by every setCommand() and publishStopCommand(). */
    uint64_t commandGeneration ABSL_GUARDED_BY(mutex) = 0;
    /** The commandGeneration of the last message the bus accepted. */
    uint64_t publishedGeneration ABSL_GUARDED_BY(mutex) = 0;
    /** The reference file and, on the IO thread only, its modification time
     * when last read (nullopt: not yet). */
    std::string referenceFile;
    std::optional<std::filesystem::file_time_type> referenceFileTime;
    /** IO-thread callbacks until the next look at the reference file. */
    int64_t callbacksUntilLimitsCheck = 0;
    int64_t callbacksPerLimitsCheck = 1;
    std::atomic<uint64_t> published{0};
    std::atomic<uint64_t> limitsReloads{0};
  };

  explicit VelocityCommandRepeater(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  /** The bus's periodic callback: looks at the reference file when due, then
   * publishes the held command. */
  static void publishFromIoThread(State& state, robot::ipc::Bus& bus);
  /** Reads the reference file again when its modification time changed; IO
   * thread only. */
  static void reloadLimitsIfChanged(State& state);

  // Shared with the bus's periodic callback, which holds its own reference: the
  // bus has no way to remove a periodic callback, so the state lives as long as
  // the later of the repeater and the bus.
  std::shared_ptr<State> state_;
};

}  // namespace ocs2::humanoid::teleop
