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
#include <vector>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/robot/FallRecoveryTypes.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {

/**
 * What the robot process does about a robot that fell, or whose plant jumped,
 * on its realtime thread after the cycle's FSM command
 * (FsmBridge::processCommands()). Selected with the FSM bridge by the backend's
 * supervisor (RobotSupervisorRegistry): SimFallRecovery catches a falling robot
 * on the simulator's gantry and notices the simulator's own resets;
 * HardwareFallRecovery switches the torques of a robot that tilts past its
 * limit off. Every call is for the realtime thread: no lock, no log line
 * (reports go to a RealtimeEventLog) and no allocation.
 */
class FallRecovery {
 public:
  /** What happened in one control cycle. */
  struct Cycle {
    /// The plant jumped since the previous cycle: the loop resets the
    /// controller and holds it (cause says why).
    bool discontinuity = false;
    DiscontinuityCause cause = DiscontinuityCause::kNone;
    /// The simulator's reset epoch this cycle (kSimulatorReset) and the tilt
    /// that was caught [rad] (kTiltCaught).
    uint64_t resetEpoch = 0;
    scalar_t tilt = 0.0;
    /// The gantry was released since the previous cycle.
    bool gantryUnlocked = false;
    /// update() changed the mode: the loop publishes the FSM state.
    bool modeChanged = false;
    /**
     * The backend moved the base by a means its sensors do not see (the
     * simulator put it back, or the gantry took it): a state estimator that
     * integrates the sensors starts again from the backend's measured state.
     */
    bool baseRelocated = false;
  };

  virtual ~FallRecovery() = default;

  /**
   * Once per control cycle, after the operator's commands were processed, with
   * the robot state of the cycle and the nominal posture of JOINT_PD (indexed
   * like the robot's joints). May switch the torques, move the gantry and
   * change `currentModeName`.
   */
  virtual Cycle update(const robot::model::RobotState& robotState,
                       const std::vector<scalar_t>& nominalJointPositions,
                       std::string& currentModeName) = 0;
};

}  // namespace ocs2::humanoid
