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

#include "absl/strings/string_view.h"

#include "humanoid_mpc_msgs/fsm_state.nproto.h"
#include "robot_core/TripleBuffer.h"

namespace ocs2::humanoid {

/**
 * The FSM state on its way from the realtime thread to the communication thread, which publishes it on
 * robot/fsm_state: when it changes, and again at 2 Hz (RobotProcess). The realtime thread writes the newest state into
 * a robot::TripleBuffer of msgs::FsmState (latest wins: the state is a value, and the controller reset count it carries
 * cannot be lost between two writes); the communication thread takes it. Writing allocates nothing while the mode
 * names fit the small-string buffer of std::string, as every control mode name does.
 */
class FsmStateMailbox {
 public:
  FsmStateMailbox();

  FsmStateMailbox(const FsmStateMailbox&) = delete;
  FsmStateMailbox& operator=(const FsmStateMailbox&) = delete;

  /** Realtime thread. */
  void write(absl::string_view mode, bool gantryLocked, std::uint64_t controllerResets, bool mpcHealthy);

  /** Communication thread: the newest state, when one was written since the last call. */
  bool take(msgs::FsmState& state);

 private:
  robot::TripleBuffer<msgs::FsmState> states_;
};

}  // namespace ocs2::humanoid
