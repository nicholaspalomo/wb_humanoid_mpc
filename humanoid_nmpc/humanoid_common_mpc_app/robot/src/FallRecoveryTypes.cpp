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

#include "humanoid_common_mpc_app/robot/FallRecoveryTypes.h"

#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {

absl::string_view settlePhaseDescription(SettlePhase phase) {
  switch (phase) {
    case SettlePhase::kLifting:
      return "lifting the robot clear of the ground";
    case SettlePhase::kSettlingLifted:
      return "waiting for the lifted robot to come to rest at the nominal posture";
    case SettlePhase::kLowering:
      return "lowering the robot back onto its feet";
    case SettlePhase::kSettlingOnFeet:
      return "waiting for the robot to come to rest on its feet";
    case SettlePhase::kIdle:
    default:
      return "idle";
  }
}

std::string discontinuityReason(DiscontinuityCause cause, std::uint64_t resetEpoch, double tilt, double maxTilt) {
  switch (cause) {
    case DiscontinuityCause::kSimulatorReset:
      return absl::StrCat("the simulator put the robot back in its initial state (reset epoch ", resetEpoch, ")");
    case DiscontinuityCause::kGantryLocked:
      return "the gantry was locked";
    case DiscontinuityCause::kTiltCaught:
      return absl::StrCat("the base tilted ", tilt, " rad, past simMaxBaseTiltAngle ", maxTilt, " rad");
    case DiscontinuityCause::kNone:
    default:
      return "no discontinuity";
  }
}

}  // namespace ocs2::humanoid
