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
#include <string>

#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

/** The settle sequence of SimFallRecovery (see its class comment). */
enum class SettlePhase : std::int32_t {
  kIdle = 0,        ///< no sequence running: WB_MPC is accepted
  kLifting,         ///< raising the gantry by catchLift
  kSettlingLifted,  ///< feet clear of the ground, waiting for rest at the nominal posture
  kLowering,        ///< lowering the gantry back to the height of the catch
  kSettlingOnFeet,  ///< feet on the ground, waiting for rest
};

/** What the robot is waiting for in `phase`, for the log ("lifting the robot clear of the ground"). */
absl::string_view settlePhaseDescription(SettlePhase phase);

/** Why SimFallRecovery reset the controller (SimFallRecovery::Cycle::cause). */
enum class DiscontinuityCause : std::int32_t {
  kNone = 0,
  kSimulatorReset,  ///< the simulator put the robot back in its initial state (its reset epoch moved)
  kGantryLocked,    ///< the gantry was locked since the previous cycle
  kTiltCaught,      ///< the base tilted past simMaxBaseTiltAngle and was caught on the gantry
};

/**
 * The reason of a discontinuity as the log states it: "the simulator put the robot back in its initial state (reset
 * epoch 3)", "the gantry was locked", "the base tilted 1.2 rad, past simMaxBaseTiltAngle 1 rad". Allocates; not for the
 * realtime thread.
 */
std::string discontinuityReason(DiscontinuityCause cause, std::uint64_t resetEpoch, double tilt, double maxTilt);

}  // namespace ocs2::humanoid
