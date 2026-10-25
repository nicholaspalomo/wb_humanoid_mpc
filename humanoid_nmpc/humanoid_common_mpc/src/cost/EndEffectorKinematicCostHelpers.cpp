/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"

#include "ocs2_core/manifold/UnitQuaternionMath.h"

namespace ocs2::humanoid {

vector12_t EndEffectorKinematicsWeights::toVector() const {
  vector12_t weightVector;
  weightVector << contactPositionErrorWeight, contactOrientationErrorWeight,
      contactLinearVelocityErrorWeight, contactAngularVelocityErrorWeight;
  return weightVector;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
VECTOR12_T<SCALAR_T> computeTaskSpaceErrors(
    const EndEffectorKinematicsCostElement<SCALAR_T>& current,
    const EndEffectorKinematicsCostElement<SCALAR_T>& reference) {
  // The rotation vector theta n of the rotation that takes the current
  // orientation to the reference, in world axes: Log(q_ref (x) q^-1), on the
  // shortest path (|theta| <= pi). It replaces the vector part of that error
  // quaternion, sin(theta / 2) n, which is half the angle to first order and
  // saturates at a half turn.
  const quaternion_coeffs_t<SCALAR_T> errorQuaternion =
      quaternionProduct<SCALAR_T>(
          reference.getOrientation().coeffs(),
          quaternionConjugate<SCALAR_T>(current.getOrientation().coeffs()));
  const VECTOR3_T<SCALAR_T> orientationError =
      quaternionLog<SCALAR_T>(errorQuaternion);

  VECTOR12_T<SCALAR_T> errors;
  errors << (current.getPosition() - reference.getPosition()), orientationError,
      (current.getLinearVelocity() - reference.getLinearVelocity()),
      (current.getAngularVelocity() - reference.getAngularVelocity());
  return errors;
}
template VECTOR12_T<scalar_t> computeTaskSpaceErrors(
    const EndEffectorKinematicsCostElement<scalar_t>& current,
    const EndEffectorKinematicsCostElement<scalar_t>& reference);
template VECTOR12_T<ad_scalar_t> computeTaskSpaceErrors(
    const EndEffectorKinematicsCostElement<ad_scalar_t>& current,
    const EndEffectorKinematicsCostElement<ad_scalar_t>& reference);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> tiltAngleToPlane(const MATRIX3_T<SCALAR_T>& rotation,
                                     const VECTOR3_T<SCALAR_T>& planeNormal) {
  const VECTOR3_T<SCALAR_T> endEffectorNormal = rotation.col(kZAxisIndex);
  // The shortest arc from v to n has the vector part (v x n) / sqrt(2 (1 + v .
  // n)), finite at v = -n (its clamp); Log takes it to theta (v x n) / |v x n|,
  // and the tilt turns n into v, the other way.
  const Eigen::Quaternion<SCALAR_T> shortestArc =
      getQuaternionFromUnitVectors<SCALAR_T>(endEffectorNormal, planeNormal);
  return -quaternionLog<SCALAR_T>(shortestArc.coeffs());
}
template VECTOR3_T<scalar_t> tiltAngleToPlane(
    const MATRIX3_T<scalar_t>& rotation,
    const VECTOR3_T<scalar_t>& planeNormal);
template VECTOR3_T<ad_scalar_t> tiltAngleToPlane(
    const MATRIX3_T<ad_scalar_t>& rotation,
    const VECTOR3_T<ad_scalar_t>& planeNormal);

}  // namespace ocs2::humanoid
