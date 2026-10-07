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
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/robot/FallRecoveryTypes.h"
#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {

/**
 * The simulator's fall recovery, as the realtime loop of the robot process sees it (RobotProcess, with the MuJoCo
 * backend). It logs nothing itself: what it would say goes to a RealtimeEventLog, which the communication thread logs.
 *
 * DISCONTINUITIES. The plant jumps whenever the robot is caught on the gantry, and the controller has to start again
 * from where the robot is then. update() reports one discontinuity per event, whichever of these it was:
 *  - the simulator put the robot back in its initial state: its resetEpoch() moved (a reset below the floor limit, the
 *    recovery from an unstable step, an explicit reset). Both automatic resets also lock the gantry, on the SIMULATION
 *    thread, at any moment of the control cycle: the loops used to look for a lock only between two reads inside one
 *    cycle, which it almost never fell between, so these resets went unnoticed;
 *  - the gantry locked since the previous cycle (the operator's LOCK_GANTRY);
 *  - the base tilted past `maxBaseTiltAngle` with the gantry unlocked: update() catches the robot by locking the gantry.
 * State is compared with the PREVIOUS CYCLE, so an event on the simulation thread cannot fall between two reads. On a
 * discontinuity the robot is put in JOINT_PD (torques enabled), and the loop resets the controller and counts the reset
 * in the FSM state it publishes (SimFsmBridge::publishControllerReset()), which is what makes the remote control
 * re-center its joysticks - also after a reset that changes neither the mode nor the gantry.
 *
 * SETTLING. A robot caught after a fall hangs at the gantry height, which is its standing height: its feet stay loaded
 * where they landed, JOINT_PD cannot bring its legs back to the nominal posture, and the weld holds its base pitched. An
 * MPC entered from there plans against a posture it cannot move, and the robot fell again within two seconds of the
 * unlock in every headless trial. With a `catchLift` the catch runs a settle sequence: the gantry lifts the robot by
 * that much so that its feet clear the ground, waits until JOINT_PD has brought it to rest at the nominal posture,
 * lowers it back to the height it was caught at and waits for it to rest again on its feet. Until the sequence is over,
 * WB_MPC is refused (the mode stays JOINT_PD and one report says why). The operator unlocking the gantry ends the
 * sequence. An operator's LOCK_GANTRY of a robot that did not fall runs no sequence.
 *
 * REALTIME. update() runs on the realtime thread every cycle: it allocates nothing (the mode is a short string, which
 * stays in its small-string buffer), takes no lock of its own and logs nothing.
 */
class SimFallRecovery {
 public:
  struct Config {
    /// [rad] Tilt of the base past which the robot is caught (task.textproto `sim_max_base_tilt_angle`); <= 0 disables the catch.
    scalar_t maxBaseTiltAngle = 0.0;
    /// [m] How far the gantry lifts a caught robot to let it settle (task.textproto `sim_gantry_catch_lift`); <= 0: no sequence.
    scalar_t catchLift = 0.0;
    // LINT.IfChange(settle_defaults)
    /// [m/s] Rate the gantry height is moved at during the sequence, up and down.
    scalar_t gantryHeightRate = 0.25;
    /// At rest: base tilt [rad], base linear [m/s] and angular [rad/s] speed and joint error to nominal [rad] all below
    /// these, without a break, for settleHoldTime [s]. The joint velocities are deliberately not read: the simulator
    /// samples them at the control rate, and a joint that chatters at the step rate reads as a large constant velocity
    /// while it holds its position (see the Atlas ankles under JOINT_PD with an unloaded foot); a joint that really
    /// moves leaves the position band within the hold time.
    scalar_t settleTilt = 0.05;
    scalar_t settleLinearSpeed = 0.05;
    scalar_t settleAngularSpeed = 0.2;
    scalar_t settleJointError = 0.15;
    scalar_t settleHoldTime = 0.5;
    /// [s] A settle phase that does not come to rest within this moves on anyway, with a warning.
    scalar_t settleTimeout = 5.0;
    // LINT.ThenChange(//humanoid_nmpc/docs/mpc_reset/README.md:settle_sequence)
  };

  using Phase = SettlePhase;

  /** What happened in one control cycle. */
  struct Cycle {
    /// The plant jumped since the previous cycle: the loop resets the controller and holds it (cause says why).
    bool discontinuity = false;
    DiscontinuityCause cause = DiscontinuityCause::kNone;
    /// The simulator's reset epoch this cycle (kSimulatorReset) and the tilt that was caught [rad] (kTiltCaught).
    uint64_t resetEpoch = 0;
    scalar_t tilt = 0.0;
    /// The gantry was released since the previous cycle.
    bool gantryUnlocked = false;
    /// update() changed the mode (to JOINT_PD): the loop publishes the FSM state.
    bool modeChanged = false;
  };

  /**
   * Seeds the gantry state and the reset epoch of the previous cycle from `robotInterface`. `restJointIndices` are the
   * joints whose posture and speed decide whether the robot is at rest: the loops pass the joints of the MPC model, the
   * ones JOINT_PD brings to the nominal posture and the MPC starts from. `eventLog` receives what the recovery reports;
   * nullptr reports nothing.
   */
  SimFallRecovery(const Config& config,
                  const robot::mujoco_sim_interface::MujocoSimInterface& robotInterface,
                  std::vector<size_t> restJointIndices,
                  RealtimeEventLog* absl_nullable eventLog = nullptr);

  /**
   * Once per control cycle, after the operator's commands were processed (SimFsmBridge::processCommands()), with the
   * robot state read at the start of the cycle and the nominal posture of JOINT_PD (indexed like the robot's joints).
   * May lock the gantry, move its height, enable the torques and change `currentModeName`; see the class comment.
   */
  Cycle update(const robot::model::RobotState& robotState,
               const std::vector<scalar_t>& nominalJointPositions,
               robot::mujoco_sim_interface::MujocoSimInterface& robotInterface,
               std::string& currentModeName);

  Phase phase() const { return phase_; }
  bool isSettling() const { return phase_ != Phase::kIdle; }
  const Config& getConfig() const { return config_; }

  /** Why `cycle` was a discontinuity, as the log states it. Allocates; not for the realtime thread. */
  std::string describeDiscontinuity(const Cycle& cycle) const;

  /** The angle between the base's vertical and the world vertical, in radians (0 upright, pi upside down). */
  static scalar_t baseTiltAngle(const quaternion_t& baseRotationLocalToWorld);

  /**
   * Whether `robotState` is at rest at the nominal posture by the thresholds of `config`, judged on the joints
   * `jointIndices` (instantaneous; no hold).
   */
  static bool isAtRest(const robot::model::RobotState& robotState,
                       const std::vector<scalar_t>& nominalJointPositions,
                       const std::vector<size_t>& jointIndices,
                       const Config& config);

 private:
  /** True once the robot has been at rest for settleHoldTime; false (and the hold restarted) otherwise. */
  bool restedFor(const robot::model::RobotState& robotState, const std::vector<scalar_t>& nominalJointPositions);
  void enterPhase(Phase phase, scalar_t time);
  void stepSettleSequence(const robot::model::RobotState& robotState,
                          const std::vector<scalar_t>& nominalJointPositions,
                          robot::mujoco_sim_interface::MujocoSimInterface& robotInterface);
  void report(RealtimeEventCode code,
              int32_t detail = 0,
              absl::string_view text = {},
              double value0 = 0.0,
              double value1 = 0.0,
              double value2 = 0.0,
              uint64_t count = 0);

  const Config config_;
  const std::vector<size_t> restJointIndices_;
  RealtimeEventLog* absl_nullable const eventLog_;
  uint64_t lastResetEpoch_ = 0;
  bool lastGantryLocked_ = false;

  Phase phase_ = Phase::kIdle;
  scalar_t phaseStartTime_ = 0.0;
  scalar_t standingGantryHeight_ = 0.0;  ///< [m] gantry height at the catch, returned to at the end of the sequence
  scalar_t rampStartHeight_ = 0.0;       ///< [m] gantry height the current lift or descent started from
  std::optional<scalar_t> restingSince_;
  bool refusalReported_ = false;
};

}  // namespace ocs2::humanoid
