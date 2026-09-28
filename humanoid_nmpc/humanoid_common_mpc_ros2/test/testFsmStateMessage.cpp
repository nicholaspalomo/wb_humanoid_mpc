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

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>

#include "humanoid_common_mpc/mrt/ControlMode.h"
#include "humanoid_common_mpc_ros2/fsm/SimFsmBridge.h"

/**
 * The `/humanoid/fsm_state` message SimFsmBridge publishes, which the remote control parses (remote_control/fsm_state.py,
 * pinned by test/test_fsm_state.py on the same strings). Its last field counts the controller resets: the remote control
 * re-centers its joysticks on every change of it, which is the only thing that shows it a reset while the gantry was
 * already locked in JOINT_PD.
 */
namespace ocs2::humanoid {
namespace {

TEST(FsmStateMessage, TheModeTheGantryAndTheControllerResetsInThatOrder) {
  EXPECT_EQ(formatFsmState(control_mode::kJointPd, /*gantryLocked=*/true, /*controllerResets=*/3), "JOINT_PD,GANTRY_LOCKED,3");
  EXPECT_EQ(formatFsmState(control_mode::kWbMpc, /*gantryLocked=*/false, /*controllerResets=*/0), "WB_MPC,GANTRY_UNLOCKED,0");
}

TEST(FsmStateMessage, AResetChangesTheMessageWhenNothingElseDoes) {
  // The case the fix is for: the same mode and the same gantry before and after a simulator reset. Only the count can
  // tell the remote control that the controller started again.
  const std::string before = formatFsmState(control_mode::kJointPd, /*gantryLocked=*/true, /*controllerResets=*/1);
  const std::string after = formatFsmState(control_mode::kJointPd, /*gantryLocked=*/true, /*controllerResets=*/2);
  EXPECT_NE(before, after);
  // Positive control: the same state and count publish the same message, which the remote control must not act on.
  EXPECT_EQ(after, formatFsmState(control_mode::kJointPd, /*gantryLocked=*/true, /*controllerResets=*/2));
}

TEST(FsmStateMessage, TheCountIsWrittenInFullAsTheLastField) {
  const uint64_t largest = std::numeric_limits<uint64_t>::max();
  const std::string message = formatFsmState(control_mode::kSafety, /*gantryLocked=*/false, largest);
  EXPECT_EQ(message, "SAFETY,GANTRY_UNLOCKED," + std::to_string(largest));
  EXPECT_EQ(message.substr(message.rfind(',') + 1), std::to_string(largest));
}

}  // namespace
}  // namespace ocs2::humanoid
