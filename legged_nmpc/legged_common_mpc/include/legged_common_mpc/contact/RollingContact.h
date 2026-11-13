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

#include "legged_common_mpc/common/Types.h"

namespace ocs2::legged {

// The geometry of a rolling contact (ModelSettings::ContactKind::kRolling): a
// wheel of radius r whose center frame `wheel` the MPC model carries, touching
// the ground at its lowest point. Every consumer converts between the wheel
// center and the contact point with these, so that the swing planner, the
// footholds, the references, the telemetry and the metrics speak in contact
// points and the constraints and costs on the wheel-center frame add the
// offset (legged_nmpc/docs/rolling_contact/README.md).
//
// The ground normal n is the unit normal of the terrain the wheel rolls on.
// Rolling contacts assume flat ground in world z, n = normalTerrainEwrtWorld().
// Each helper is a template on the scalar, so that it records on a CppAD tape
// the same arithmetic it evaluates in double.

/** Returns the normal of the flat ground the rolling contacts roll on: world
 * z. */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> normalTerrainEwrtWorld() {
  return VECTOR3_T<SCALAR_T>::UnitZ();
}

/**
 * Returns the contact point relative to the wheel center, in world axes: the
 * wheel's lowest point, -r n for a wheel of radius `wheelRadius` [m] on ground
 * of unit normal `normalSurfaceEwrtWorld`. The wheel's own orientation does not
 * enter: the point is on the rim wherever the wheel has turned to.
 */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> posContactRtWheelEwrtWorld(
    const SCALAR_T& wheelRadius,
    const VECTOR3_T<SCALAR_T>& normalSurfaceEwrtWorld) {
  return -wheelRadius * normalSurfaceEwrtWorld;
}

/**
 * Returns the contact point of a wheel whose center is at
 * `posWheelRtWorldEwrtWorld` [m]: p_contact = p_wheel - r n.
 */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> posContactRtWorldEwrtWorld(
    const VECTOR3_T<SCALAR_T>& posWheelRtWorldEwrtWorld,
    const SCALAR_T& wheelRadius,
    const VECTOR3_T<SCALAR_T>& normalSurfaceEwrtWorld) {
  return posWheelRtWorldEwrtWorld +
         posContactRtWheelEwrtWorld(wheelRadius, normalSurfaceEwrtWorld);
}

/**
 * Returns the velocity of the wheel's material at the contact point, the
 * no-slip residual of a rolling contact: v_contact = v_wheel + w_wheel x (-r n)
 * = v_wheel + r (n x w_wheel), for a wheel center moving at
 * `linvelWheelRtWorldEwrtWorld` [m/s] and turning at
 * `angvelWheelRtWorldEwrtWorld` [rad/s] (the wheel link's angular velocity,
 * its spin about the axle included). It is zero when the wheel rolls without
 * slipping: the center then moves at w x (r n), along the wheel's heading.
 * It is not the velocity of the contact point itself, which moves with the
 * wheel center over the ground.
 */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> linvelContactRtWorldEwrtWorld(
    const VECTOR3_T<SCALAR_T>& linvelWheelRtWorldEwrtWorld,
    const VECTOR3_T<SCALAR_T>& angvelWheelRtWorldEwrtWorld,
    const SCALAR_T& wheelRadius,
    const VECTOR3_T<SCALAR_T>& normalSurfaceEwrtWorld) {
  return linvelWheelRtWorldEwrtWorld +
         angvelWheelRtWorldEwrtWorld.cross(
             posContactRtWheelEwrtWorld(wheelRadius, normalSurfaceEwrtWorld));
}

/**
 * Returns the moment about the wheel center of a force
 * `forceContactEwrtWorld` [N] the ground applies at the contact point of a
 * wheel of radius `wheelRadius` [m]: (-r n) x f. Together with f it is the
 * wrench at the wheel-center frame that the contact force is, whose moment
 * about the axle, r times the force along the rolling direction, is the
 * wheel joint's torque.
 */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> torqueWheelEwrtWorld(
    const VECTOR3_T<SCALAR_T>& forceContactEwrtWorld,
    const SCALAR_T& wheelRadius,
    const VECTOR3_T<SCALAR_T>& normalSurfaceEwrtWorld) {
  return posContactRtWheelEwrtWorld(wheelRadius, normalSurfaceEwrtWorld)
      .cross(forceContactEwrtWorld);
}

}  // namespace ocs2::legged
