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

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/mrt/ControllerEvent.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "humanoid_common_mpc_app/robot/FallRecoveryTypes.h"
#include "robot_realtime/SpscQueue.h"

namespace ocs2::humanoid {

/**
 * What the realtime thread of the robot process reports, for the communication thread to log. The realtime thread
 * never logs: a log line formats into a string and writes to a file descriptor, either of which can allocate or block.
 * It posts one of these instead, a few plain values, and formatRealtimeEvent() turns it into the line the old ROS sims
 * logged from their control loops.
 */
enum class RealtimeEventCode : std::uint8_t {
  kTorquesDisabled,         ///< an FSM command switched the torques off; text: the command
  kTorquesEnabled,          ///< an FSM command switched the torques on; text: the command
  kGantryLockCommanded,     ///< LOCK_GANTRY
  kGantryUnlockCommanded,   ///< UNLOCK_GANTRY
  kGantryUnlockedMpcReset,  ///< the gantry was released: the MPC is reset, the policy in use carries the robot
  /**
   * SimFallRecovery caught the robot and runs the settle sequence. detail: the DiscontinuityCause; count: the reset
   * epoch; values: the tilt [rad], simMaxBaseTiltAngle [rad] and simGantryCatchLift [m].
   */
  kCaughtAndSettling,
  /** SimFallRecovery put the robot in JOINT_PD and resets the controller; detail, count, values as kCaughtAndSettling. */
  kDiscontinuity,
  kSettleEndedByUnlock,          ///< the gantry was unlocked during the settle sequence; detail: the phase
  kMpcModeRefusedWhileSettling,  ///< an MPC mode was refused while the robot settles; text: the mode, detail: the phase
  kLiftedSettleTimedOut,         ///< the lifted robot did not come to rest; values[0]: the timeout [s]
  kOnFeetSettleTimedOut,         ///< the robot did not come to rest on its feet; values[0]: the timeout [s]
  kSettled,                      ///< the robot rests on its feet on the gantry; WB_MPC is accepted again
  kContactEstimatorSwapped,      ///< the controller's contact estimator was replaced; text: its name
  /** A report of the MRT joint controller (ControllerEvent.h). detail: the ControllerEventCode; source: its class. */
  kControllerEvent,
};

/** One report of the realtime thread: plain data, so that posting it copies a few words and allocates nothing. */
struct RealtimeEvent {
  RealtimeEventCode code = RealtimeEventCode::kTorquesDisabled;
  /** A code-specific enumerator: the DiscontinuityCause, or the SimFallRecovery::Phase. */
  std::int32_t detail = 0;
  std::array<double, 3> values{};
  std::uint64_t count = 0;
  /** A mode, command or estimator name, NUL-terminated and cut to fit. */
  std::array<char, 32> text{};
  /** kControllerEvent: the class that reported it, a string literal (ControllerEvent::controller). */
  const char* source = "";
};

/** The line the communication thread logs for `event`; the wording of the lines the ROS sims logged. */
std::string formatRealtimeEvent(const RealtimeEvent& event);

/** Whether formatRealtimeEvent(event) is logged as a warning (otherwise as information). */
bool isWarningEvent(const RealtimeEvent& event);

/**
 * The mailbox of the realtime thread's reports: a bounded SPSC queue that the realtime thread posts to and the
 * communication thread drains (and logs) at its own pace. A report posted while the queue is full is dropped and
 * counted, so the realtime thread never waits for the logger. It is the ControllerEventSink of the robot process's
 * controller too, so that the controller's reports take the same way off the realtime thread.
 */
class RealtimeEventLog final : public ControllerEventSink {
 public:
  explicit RealtimeEventLog(std::size_t capacity = 64);

  RealtimeEventLog(const RealtimeEventLog&) = delete;
  RealtimeEventLog& operator=(const RealtimeEventLog&) = delete;

  /** Realtime thread: copies the values into a slot. False when the queue was full (the report is dropped). */
  bool post(RealtimeEventCode code,
            std::int32_t detail = 0,
            absl::string_view text = {},
            double value0 = 0.0,
            double value1 = 0.0,
            double value2 = 0.0,
            std::uint64_t count = 0);

  /** Realtime thread: a report of the controller, as kControllerEvent. */
  bool post(const ControllerEvent& event) override;

  /** Communication thread: hands every report posted so far to `consumer`, oldest first; returns how many. */
  std::size_t drain(const std::function<void(const RealtimeEvent&)>& consumer);

  /** Communication thread: drain() into absl logging, each report at its severity. */
  std::size_t drainToLog();

  /** Reports dropped because the queue was full. Any thread. */
  std::uint64_t dropped() const { return queue_.droppedCount(); }

 private:
  robot::realtime::SpscQueue<RealtimeEvent> queue_;
};

/** The text of `event`, as a string_view of its NUL-terminated buffer. */
absl::string_view realtimeEventText(const RealtimeEvent& event);

}  // namespace ocs2::humanoid
