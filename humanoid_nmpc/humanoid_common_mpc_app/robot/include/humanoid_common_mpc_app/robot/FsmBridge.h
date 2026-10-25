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

#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The supervisory FSM of the robot process, on its realtime thread: it takes
 * the operator's commands from the OperatorCommandMailbox, switches the robot's
 * torques (and, in simulation, its gantry), and hands the FSM state to the
 * FsmStateMailbox, which the communication thread publishes on robot/fsm_state.
 * Selected with the fall recovery by the backend's supervisor
 * (RobotSupervisorRegistry): SimFsmBridge drives the simulator's torque switch,
 * gantry and dodgeballs; HardwareFsmBridge the backend's torque switch only.
 * Every call is for the realtime thread: no bus, no parsing, no log line
 * (reports go to a RealtimeEventLog) and no allocation.
 *
 * Every bridge keeps:
 * 1. the nominal posture of JOINT_PD, with the operator's joint targets merged
 * in (applyJointTargetUpdates());
 * 2. the FSM commands, one per cycle in the order they were sent
 * (processCommands());
 * 3. the FSM state: the mode, the gantry, the number of controller resets and
 * the MPC's health.
 */
class FsmBridge {
 public:
  virtual ~FsmBridge() = default;

  /** Takes the operator's newest joint targets into the nominal posture, if new
   * ones have arrived. */
  virtual void applyJointTargetUpdates() = 0;

  /** The nominal joint positions of JOINT_PD, indexed like the robot's joints.
   */
  virtual const std::vector<scalar_t>& getNominalJointPositions() const = 0;

  /**
   * Processes the oldest FSM command, if any, switching the robot as it asks
   * and updating `currentModeName`, and publishes the new FSM state. True if a
   * command changed something.
   */
  virtual bool processCommands(std::string& currentModeName) = 0;

  /** Whether a gantry holds the robot's base; false on a robot without one the
   * process drives. */
  virtual bool isGantryLocked() const = 0;

  /** Writes the FSM state: `modeName`, the gantry, the controller resets so far
   * and the MPC's health. */
  virtual void publishFsmState(absl::string_view modeName) = 0;

  /**
   * Counts one controller reset (a discontinuity after which the controller
   * starts again from where the robot is, FallRecovery::Cycle::discontinuity)
   * and publishes the state with the new count: the remote control re-centers
   * its joysticks on every change of the count.
   */
  virtual void publishControllerReset(absl::string_view modeName) = 0;

  /** Records the MPC's health (false: the controller holds JOINT_PD), and
   * publishes the state when it changed. */
  virtual void setMpcHealthy(bool mpcHealthy, absl::string_view modeName) = 0;

  /** The controller resets counted so far, the controller_resets of every state
   * published. */
  virtual uint64_t controllerResets() const = 0;
  virtual bool mpcHealthy() const = 0;
};

}  // namespace ocs2::humanoid
