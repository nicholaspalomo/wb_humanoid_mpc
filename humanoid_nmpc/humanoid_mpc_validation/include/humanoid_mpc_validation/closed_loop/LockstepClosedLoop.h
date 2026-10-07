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

#include <cstddef>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopDriver.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetrics.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopScenario.h"
#include "humanoid_mpc_validation/closed_loop/RecordedRobotStates.h"
#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"
#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "humanoid_mpc_validation/io/JsonValue.h"
#include "robot_core/Types.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid::validation {

/** The operator's sequence before a scenario's commands, as an operator runs the MuJoCo sims. */
struct LockstepTimeline {
  double jointPdSettleTime = 1.0;  ///< [s] JOINT_PD on the locked gantry before WB_MPC is entered
  double entryMargin = 0.5;        ///< [s] after the entry into WB_MPC has held and blended, before the gantry is released
  double maxEntryTime = 10.0;      ///< [s] the entry must be over within this; a run whose entry is not fails
};

/** How a lockstep run is made. */
struct LockstepOptions {
  ClosedLoopDriverOptions driver;
  LockstepTimeline timeline;
  bool recordRobotStates = false;  ///< record the robot at the solves of the commands, for the solve benchmark
  double recordingDelay = 2.0;     ///< [s] after the first command, so that the recording is of steady walking
  size_t maxRecordedStates = 300;
  double timeSeriesRate = 50.0;  ///< [Hz] of LockstepResult::timeSeries
};

/** What a lockstep run produced. */
struct LockstepResult {
  JsonValue metrics;                   ///< the closed-loop metrics document (ClosedLoopMetricsSchema)
  RecordedRobotStates recordedStates;  ///< with LockstepOptions::recordRobotStates
  GoldenFile timeSeries;               ///< the base, its reference and each solve's rotation gap over the commands (TimeSeriesLabels.h)
  vector_t finalObservationState;      ///< the controller's last observation, for the determinism test
};

/**
 * The MuJoCo closed loop without the bus and without threads, in lockstep: the headless MujocoSimInterface, the
 * formulation's MRT joint controller as the robot process drives it, and the MPC's solver iterations called
 * synchronously (InProcessMpcLink::Execution::kCaller), all on the calling thread.
 *
 * Every control cycle (1 / mpc.mrt_desired_frequency of the task file, a whole number of simulation steps) runs in the order
 * of RobotProcess::cycle(): the runner reads the simulator's state, hands the controller the mode and the JOINT_PD
 * posture, computes the joint action, applies the controller-side settings of the task file, serves the operator's commands
 * of this cycle (which take effect from the next one) and applies the action last. A solve is due every
 * 1 / mpc.mpc_desired_frequency of simulation time; it runs in the cycle it falls in, after the action, on that cycle's
 * observation, and its policy is in use from the next cycle on. A failed solve's pause before the next attempt ends
 * when its delay has passed or a reset is requested after it, as on the solver thread (LockstepSolveSchedule). The
 * solve takes no simulation time: the closed loop measures the controller, not the machine or the bus, and so is
 * reproducible.
 *
 * The sequence (LockstepTimeline): the robot starts on the locked gantry in JOINT_PD with torques enabled, enters WB_MPC
 * once settled, has the gantry released as the robot process releases it (unlockGantry() and a reset of the MPC) once
 * the entry is over, stands for the scenario's standingTime, and then runs the scenario's commands, over which the
 * metrics are evaluated. The commands start at the top of the cycle in which the standing time is over: the GUI's
 * message is no FSM command of the robot process, so the operator's sequence does not delay it. A tilt beyond the task
 * file's sim_max_base_tilt_angle (1 rad without one), a base below half its initial height or a reset by the simulator
 * is a fall; the run ends there.
 */
class LockstepClosedLoop {
 public:
  LockstepClosedLoop(RobotConfiguration configuration, LockstepOptions options);

  /**
   * Runs `scenario` once on a fresh MPC and simulator. `info` names the run in the metrics document; its formulation,
   * robot, scenario and settings are filled in here. The error of a driver or simulator that cannot be built, or of an
   * entry into WB_MPC that does not end.
   */
  absl::StatusOr<LockstepResult> run(const ClosedLoopScenario& scenario, ClosedLoopRunInfo info) const;

 private:
  RobotConfiguration configuration_;
  LockstepOptions options_;
};

/**
 * The total feedback torque (JointAction::getTotalFeedbackTorque()) that `action` commands at `state` for each of
 * `joints`, in their order: what a control cycle of the run records. An internal error names the first joint `action`
 * holds no action for.
 */
absl::StatusOr<Eigen::VectorXd> jointFeedbackTorques(const robot::model::RobotJointAction& action,
                                                     const robot::model::RobotState& state,
                                                     absl::Span<const robot::joint_index_t> joints);

}  // namespace ocs2::humanoid::validation
