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

#include <optional>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid::visualization {

// The balance quantities the telemetry plots (humanoid_nmpc/humanoid_rerun_viewer/README.md, "The plots"): the linear
// inverted pendulum's natural frequency, the divergent component of motion (DCM, the instantaneous capture point ICP)
// and the zero moment point (ZMP) of the contact wrenches. Display values only, computed on the MPC node's
// visualization thread, never on the robot's realtime thread. Pure functions, safe to call from any thread.

/**
 * Returns the natural frequency omega = sqrt(g / z0) [1/s] of a linear inverted pendulum of height `pendulumHeight` (z0,
 * [m]) under `gravity` (g, [m/s^2]); nullopt unless both are finite and positive.
 */
std::optional<scalar_t> pendulumFrequency(scalar_t gravity, scalar_t pendulumHeight);

/**
 * Returns the horizontal divergent component of motion xi = c + c_dot / omega [m] (the instantaneous capture point) of
 * the center of mass at `com` moving at `comVelocity`, for the pendulum frequency `omega` (pendulumFrequency(), positive).
 * Only the x and y components of `com` and `comVelocity` are read.
 */
vector2_t divergentComponentOfMotion(const vector3_t& com, const vector3_t& comVelocity, scalar_t omega);

/**
 * Returns the zero moment point of the contact wrenches: the point p = (p_x, p_y, h) of the horizontal plane at the
 * height h about which the net contact moment has no horizontal component,
 *
 *   p_x = (h F_x - M_y) / F_z,   p_y = (h F_y + M_x) / F_z,
 *
 * with F = sum_i f_i the net force and M = sum_i (r_i x f_i + tau_i) the net moment about the world origin. The plane is
 * the normal-force-weighted height of the contacts, h = sum_i f_z,i r_z,i / F_z: on flat ground, where every contact is
 * at the same height, it is the ground, and p is the net center of pressure.
 *
 * @param positions  per contact, the world position r_i of the point the wrench is about (the contact frame's origin).
 * @param wrenches   per contact, the world-frame wrench [f_i; tau_i], tau_i the torque about r_i.
 * @param minNormalForce [N] below this net normal force F_z there is no ZMP.
 * @return nullopt when F_z is not above `minNormalForce` (a flight phase, or no wrenches).
 */
std::optional<vector3_t> zeroMomentPoint(const feet_array_t<vector3_t>& positions,
                                         const feet_array_t<vector6_t>& wrenches,
                                         scalar_t minNormalForce);

}  // namespace ocs2::humanoid::visualization
