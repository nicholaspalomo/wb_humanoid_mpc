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

namespace ocs2::humanoid::validation::time_series {

/*
 * The matrices of a lockstep time series (LockstepResult::timeSeries, written
 * as <robot>_<scenario>_timeseries.txt into a run's outputs): what
 * LockstepClosedLoop writes, to plot or inspect a run beyond its metrics
 * document. Over the commands; the first five share their rows, one per sample
 * at LockstepOptions::timeSeriesRate.
 */

/** [s] Simulation time of the sample (1 column). */
inline constexpr char kTime[] = "time";
/** [m] World position of the base (3 columns). */
inline constexpr char kBasePosition[] = "base_position";
/** Base to world, coefficients (x, y, z, w) (4 columns). */
inline constexpr char kBaseQuaternion[] = "base_quaternion_xyzw";
/** World velocity of the base (x, y) [m/s] and its world yaw rate [rad/s] (3
 * columns). */
inline constexpr char kBaseVelocity[] = "base_velocity_world_xy_yaw_rate";
/** The reference velocity: (forward, lateral) in the heading frame [m/s] and
 * the yaw rate [rad/s] (3 columns). */
inline constexpr char kReferenceVelocity[] =
    "reference_velocity_heading_xy_yaw_rate";
/**
 * One row per solve of the commands that recorded an initial-state gap, with
 * rows of its own: the observation time of the solve [s] and the rotation part
 * of its gap [rad] (SolveSample::initialStateRotationGap) (2 columns). Its
 * largest value is the metrics document's initial_state_gap.max_rotation_rad
 * (test_lockstep_determinism checks it), which keeps only the largest.
 */
inline constexpr char kSolveRotationGap[] = "solve_initial_state_rotation_gap";
/**
 * One row per solve of the commands that recorded an SQP step
 * (SolveSample::sqpStep), with rows of its own: the observation time of the
 * solve [s], the step size, the codes (SqpStepKind: 0 unknown, 1 constraint, 2
 * dual, 3 cost, 4 zero) of the rule that accepted the step and of the one that
 * judged the full step, the full step's constraint violation and the constraint
 * violation after the step (kSolveSqpStepColumns columns). A Gauss-Newton SQP
 * cycling from one solve to the next shows as runs of full steps whose
 * violation grows, each ended by a full step refused above g_max (code 1) and a
 * reduced step.
 */
inline constexpr char kSolveSqpStep[] = "solve_sqp_step";
inline constexpr int kSolveSqpStepColumns = 6;
/**
 * One row per solve of the commands with an active hard swing-foot
 * normal-velocity row, with rows of its own: the observation time [s] and the
 * largest residual of those rows over the new solution's nodes, [m/s] on the
 * centroidal MPCs and [m/s^2] on the whole-body MPC
 * (SqpStepSample::swingNormalVelocityResidual) (kSolveSwingResidualColumns
 * columns).
 */
inline constexpr char kSolveSwingNormalVelocityResidual[] =
    "solve_swing_normal_velocity_residual";
inline constexpr int kSolveSwingResidualColumns = 2;

}  // namespace ocs2::humanoid::validation::time_series
