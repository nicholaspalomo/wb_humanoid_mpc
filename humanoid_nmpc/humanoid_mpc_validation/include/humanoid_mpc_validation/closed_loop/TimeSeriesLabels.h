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
 * The matrices of a lockstep time series (LockstepResult::timeSeries, written as <robot>_<scenario>_timeseries.txt):
 * what LockstepClosedLoop writes and the 720-degree turn exception of MetricBands.h reads. Over the commands; the first
 * five share their rows, one per sample at LockstepOptions::timeSeriesRate.
 */

/** [s] Simulation time of the sample (1 column). */
inline constexpr char kTime[] = "time";
/** [m] World position of the base (3 columns). */
inline constexpr char kBasePosition[] = "base_position";
/** Base to world, coefficients (x, y, z, w) (4 columns). */
inline constexpr char kBaseQuaternion[] = "base_quaternion_xyzw";
/** World velocity of the base (x, y) [m/s] and its world yaw rate [rad/s] (3 columns). */
inline constexpr char kBaseVelocity[] = "base_velocity_world_xy_yaw_rate";
/** The reference velocity: (forward, lateral) in the heading frame [m/s] and the yaw rate [rad/s] (3 columns). */
inline constexpr char kReferenceVelocity[] = "reference_velocity_heading_xy_yaw_rate";
/**
 * One row per solve of the commands that recorded an initial-state gap, with rows of its own: the observation time of
 * the solve [s] and the rotation part of its gap [rad] (SolveSample::initialStateRotationGap) (2 columns). Its largest
 * value is the metrics document's initial_state_gap.max_rotation_rad. The time series archived with M0 were recorded
 * before it was added and do not have it; only a candidate's series needs it.
 */
inline constexpr char kSolveRotationGap[] = "solve_initial_state_rotation_gap";

}  // namespace ocs2::humanoid::validation::time_series
