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

#include <array>

#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/mrt/ControlMode.h"

/*
 * The control mode names the MRT joint controllers and the sim fall recovery share. Each of them used to spell the names
 * and the two families out on its own.
 */

namespace ocs2::humanoid {
namespace {

constexpr std::array<absl::string_view, 4> kPassiveModes{control_mode::kZeroTorque, control_mode::kJointPd, control_mode::kGravityComp,
                                                         control_mode::kSafety};
constexpr std::array<absl::string_view, 2> kMpcModes{control_mode::kWbMpc, control_mode::kMpcActive};

TEST(ControlMode, EveryModeIsInExactlyOneFamily) {
  for (absl::string_view mode : kPassiveModes) {
    EXPECT_TRUE(control_mode::isPassive(mode)) << mode;
    EXPECT_FALSE(control_mode::isMpc(mode)) << mode;
  }
  for (absl::string_view mode : kMpcModes) {
    EXPECT_TRUE(control_mode::isMpc(mode)) << mode;
    EXPECT_FALSE(control_mode::isPassive(mode)) << mode;
  }
}

TEST(ControlMode, ANameTheFsmDoesNotPublishIsInNeitherFamily) {
  // FSM commands that are not modes, a name in the wrong case, and nothing at all.
  for (absl::string_view name :
       {absl::string_view("ENABLE_TORQUES"), absl::string_view("LOCK_GANTRY"), absl::string_view("wb_mpc"), absl::string_view("")}) {
    EXPECT_FALSE(control_mode::isPassive(name)) << name;
    EXPECT_FALSE(control_mode::isMpc(name)) << name;
  }
}

TEST(ControlMode, TheNamesAreTheOnesTheFsmPublishes) {
  // The strings on operator/fsm_command and robot/fsm_state (humanoid_finite_state_machine.py ControlMode).
  EXPECT_EQ(control_mode::kZeroTorque, "ZERO_TORQUE");
  EXPECT_EQ(control_mode::kJointPd, "JOINT_PD");
  EXPECT_EQ(control_mode::kGravityComp, "GRAVITY_COMP");
  EXPECT_EQ(control_mode::kSafety, "SAFETY");
  EXPECT_EQ(control_mode::kWbMpc, "WB_MPC");
}

}  // namespace
}  // namespace ocs2::humanoid
