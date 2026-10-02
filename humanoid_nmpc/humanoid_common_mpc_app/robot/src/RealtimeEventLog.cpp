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

#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"

#include <algorithm>
#include <cstring>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {

absl::string_view realtimeEventText(const RealtimeEvent& event) {
  const std::size_t length = ::strnlen(event.text.data(), event.text.size());
  return absl::string_view(event.text.data(), length);
}

std::string formatRealtimeEvent(const RealtimeEvent& event) {
  const absl::string_view text = realtimeEventText(event);
  const SettlePhase phase = static_cast<SettlePhase>(event.detail);
  switch (event.code) {
    case RealtimeEventCode::kTorquesDisabled:
      return absl::StrCat("FSM command received: ", text, " - zero-torque mode.");
    case RealtimeEventCode::kTorquesEnabled:
      return absl::StrCat("FSM command received: ", text, " - enabling torques.");
    case RealtimeEventCode::kGantryLockCommanded:
      return "FSM command received: Locking gantry.";
    case RealtimeEventCode::kGantryUnlockCommanded:
      return "FSM command received: Unlocking gantry.";
    case RealtimeEventCode::kGantryUnlockedMpcReset:
      return "Gantry unlocked - resetting MPC.";
    case RealtimeEventCode::kCaughtAndSettling:
      return absl::StrCat("[SimFallRecovery] Caught on the gantry in JOINT_PD, resetting the controller: ",
                          discontinuityReason(static_cast<DiscontinuityCause>(event.detail), event.count, event.values[0], event.values[1]),
                          ". The gantry lifts the robot by ", event.values[2],
                          " m (simGantryCatchLift) until it rests at the nominal posture, then lowers it back; WB_MPC is accepted after "
                          "that.");
    case RealtimeEventCode::kDiscontinuity:
      return absl::StrCat("[SimFallRecovery] JOINT_PD, resetting the controller: ",
                          discontinuityReason(static_cast<DiscontinuityCause>(event.detail), event.count, event.values[0], event.values[1]),
                          ".");
    case RealtimeEventCode::kSettleEndedByUnlock:
      return absl::StrCat("[SimFallRecovery] The gantry was unlocked while ", settlePhaseDescription(phase),
                          "; the settle sequence ends here.");
    case RealtimeEventCode::kMpcModeRefusedWhileSettling:
      return absl::StrCat("[SimFallRecovery] ", text, " refused while ", settlePhaseDescription(phase),
                          "; the robot stays in JOINT_PD until it has settled on the gantry.");
    case RealtimeEventCode::kLiftedSettleTimedOut:
      return absl::StrCat("[SimFallRecovery] The lifted robot did not come to rest at the nominal posture within ", event.values[0],
                          " s; lowering it anyway.");
    case RealtimeEventCode::kOnFeetSettleTimedOut:
      return absl::StrCat("[SimFallRecovery] The robot did not come to rest on its feet within ", event.values[0],
                          " s; WB_MPC is accepted anyway.");
    case RealtimeEventCode::kSettled:
      return "[SimFallRecovery] The robot rests on its feet on the gantry; WB_MPC is accepted again.";
    case RealtimeEventCode::kContactEstimatorSwapped:
      return absl::StrCat("[RobotProcess] The measured contact state now comes from the contact estimator '", text, "'.");
    case RealtimeEventCode::kControllerEvent:
      return formatControllerEvent(
          makeControllerEvent(static_cast<ControllerEventCode>(event.detail), event.source, event.values[0], event.values[1], text));
  }
  return absl::StrCat("[RobotProcess] realtime event ", static_cast<int>(event.code));
}

bool isWarningEvent(const RealtimeEvent& event) {
  switch (event.code) {
    case RealtimeEventCode::kCaughtAndSettling:
    case RealtimeEventCode::kDiscontinuity:
    case RealtimeEventCode::kLiftedSettleTimedOut:
    case RealtimeEventCode::kOnFeetSettleTimedOut:
      return true;
    case RealtimeEventCode::kControllerEvent:
      return isWarningControllerEvent(makeControllerEvent(static_cast<ControllerEventCode>(event.detail), event.source));
    default:
      return false;
  }
}

RealtimeEventLog::RealtimeEventLog(std::size_t capacity) : queue_(capacity) {}

bool RealtimeEventLog::post(
    RealtimeEventCode code, std::int32_t detail, absl::string_view text, double value0, double value1, double value2, std::uint64_t count) {
  return queue_.tryPushInPlace([&](RealtimeEvent& slot) {
    slot.code = code;
    slot.detail = detail;
    slot.values = {value0, value1, value2};
    slot.count = count;
    const std::size_t length = std::min(text.size(), slot.text.size() - 1);
    std::memcpy(slot.text.data(), text.data(), length);
    slot.text[length] = '\0';
    slot.source = "";
  });
}

bool RealtimeEventLog::post(const ControllerEvent& event) {
  return queue_.tryPushInPlace([&](RealtimeEvent& slot) {
    slot.code = RealtimeEventCode::kControllerEvent;
    slot.detail = static_cast<std::int32_t>(event.code);
    slot.values = {event.values[0], event.values[1], 0.0};
    slot.count = 0;
    slot.text = event.text;
    slot.source = event.controller;
  });
}

std::size_t RealtimeEventLog::drain(const std::function<void(const RealtimeEvent&)>& consumer) {
  std::size_t drained = 0;
  while (queue_.tryPopInPlace([&](const RealtimeEvent& event) { consumer(event); })) {
    ++drained;
  }
  return drained;
}

std::size_t RealtimeEventLog::drainToLog() {
  return drain([](const RealtimeEvent& event) {
    if (isWarningEvent(event)) {
      LOG(WARNING) << formatRealtimeEvent(event);
    } else {
      LOG(INFO) << formatRealtimeEvent(event);
    }
  });
}

}  // namespace ocs2::humanoid
