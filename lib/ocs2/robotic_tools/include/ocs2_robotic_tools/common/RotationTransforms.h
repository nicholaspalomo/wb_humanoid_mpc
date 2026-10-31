/******************************************************************************
Copyright (c) 2017, Farbod Farshidian. All rights reserved.

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

#include <array>
#include <cmath>

#include "ocs2_core/Dimensions.h"

#include <Eigen/Core>

// CppAD
#include <ocs2_core/automatic_differentiation/Types.h>

namespace ocs2 {

/**
 * ZYX Euler angles (z, y, x): the angles of the rotations about z, then about the new y and then about the new x
 * (getRotationMatrixFromZyxEulerAngles()), stored in that order.
 */
inline constexpr int kEulerAnglesZyxDim = 3;
inline constexpr int kEulerAnglesZyxZIndex = 0;
inline constexpr int kEulerAnglesZyxYIndex = 1;
inline constexpr int kEulerAnglesZyxXIndex = 2;

/**
 * Compute the quaternion distance measure
 *
 * @param [in] q: measured end effector quaternion.
 * @param [in] qRef: desired end effector quaternion.
 * @return A 3x1 vector representing the quaternion distance.
 * In particular, if Qd and Qe are the desired and the measured end-effector
 * quaternions, the measured and desired frames are aligned if this vector is 0.
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kRotationTangentDim, 1> quaternionDistance(const Eigen::Quaternion<SCALAR_T>& q,
                                                                   const Eigen::Quaternion<SCALAR_T>& qRef) {
  return q.w() * qRef.vec() - qRef.w() * q.vec() + q.vec().cross(qRef.vec());
}

/**
 * Compute the quaternion corresponding to euler angles zyx
 *
 * @param [in] eulerAnglesZyx
 * @return The corresponding quaternion
 */
template <typename SCALAR_T>
Eigen::Quaternion<SCALAR_T> getQuaternionFromEulerAnglesZyx(const Eigen::Matrix<SCALAR_T, kEulerAnglesZyxDim, 1>& eulerAnglesZyx) {
  // clang-format off
  return Eigen::AngleAxis<SCALAR_T>(eulerAnglesZyx(kEulerAnglesZyxZIndex), Eigen::Matrix<SCALAR_T, kPositionDim, 1>::UnitZ()) *
         Eigen::AngleAxis<SCALAR_T>(eulerAnglesZyx(kEulerAnglesZyxYIndex), Eigen::Matrix<SCALAR_T, kPositionDim, 1>::UnitY()) *
         Eigen::AngleAxis<SCALAR_T>(eulerAnglesZyx(kEulerAnglesZyxXIndex), Eigen::Matrix<SCALAR_T, kPositionDim, 1>::UnitX());
  // clang-format on
}

/**
 * The floor of the squared norm |a x b|^2 + (1 + a.b)^2 = 2 (1 + a.b) that getQuaternionFromUnitVectors() normalizes by:
 * twice the anti-parallel threshold 1 + a.b < 1e-6 of the humanoid MPC's orientation error with respect to a plane
 * (kPlaneNormalAntiParallelThreshold), below which the shortest arc is not differentiable.
 */
inline constexpr scalar_t kShortestArcSquaredNormFloor = 2.0e-6;

/**
 * Compute the quaternion corresponding to the shortest-arc rotation from unit vector a to unit vector b, for double and
 * for a CppAD tape.
 *
 * The quaternion is (1 + a.b, a x b) normalized by its norm sqrt(2 (1 + a.b)), which vanishes where a = -b: there the
 * shortest arc is not defined, and the division was 0 / 0. The squared norm is clamped from below at
 * kShortestArcSquaredNormFloor with a conditional expression (CppAD::CondExpLt, which tapes both branches), so that the
 * value and its derivatives are finite at every input. Within the clamp (1 + a.b < 1e-6, a within about 0.08 degrees of
 * -b) the result is the unnormalized quaternion divided by the floor's root: continuous at the clamp, no longer of unit
 * norm, and tending to zero at a = -b.
 *
 * @param [in] a Unit vector of initial orientation
 * @param [in] b Unit vector of final orientation
 * @return The corresponding quaternion
 */
template <typename SCALAR_T>
Eigen::Quaternion<SCALAR_T> getQuaternionFromUnitVectors(const Eigen::Matrix<SCALAR_T, kPositionDim, 1>& a,
                                                         const Eigen::Matrix<SCALAR_T, kPositionDim, 1>& b) {
  // q.vec() = a.cross(b);
  SCALAR_T x = a[kYAxisIndex] * b[kZAxisIndex] - a[kZAxisIndex] * b[kYAxisIndex];
  SCALAR_T y = a[kZAxisIndex] * b[kXAxisIndex] - a[kXAxisIndex] * b[kZAxisIndex];
  SCALAR_T z = a[kXAxisIndex] * b[kYAxisIndex] - a[kYAxisIndex] * b[kXAxisIndex];
  // q.w = sqrt(len(a)^2 * len(b)^2) + a.dot(b);
  SCALAR_T w = 1 + a[kXAxisIndex] * b[kXAxisIndex] + a[kYAxisIndex] * b[kYAxisIndex] + a[kZAxisIndex] * b[kZAxisIndex];
  const SCALAR_T squaredNorm = x * x + y * y + z * z + w * w;
  const SCALAR_T floor(kShortestArcSquaredNormFloor);
  const SCALAR_T norm = CppAD::sqrt(CppAD::CondExpLt(squaredNorm, floor, floor, squaredNorm));

  x /= norm;
  y /= norm;
  z /= norm;
  w /= norm;

  return Eigen::Quaternion<SCALAR_T>(w, x, y, z);
}

/**
 * Compute the rotation matrix corresponding to euler angles zyx
 *
 * @param [in] eulerAnglesZyx
 * @return The corresponding rotation matrix
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kPositionDim, kPositionDim> getRotationMatrixFromZyxEulerAngles(
    const Eigen::Matrix<SCALAR_T, kEulerAnglesZyxDim, 1>& eulerAngles) {
  const SCALAR_T z = eulerAngles(kEulerAnglesZyxZIndex);
  const SCALAR_T y = eulerAngles(kEulerAnglesZyxYIndex);
  const SCALAR_T x = eulerAngles(kEulerAnglesZyxXIndex);

  const SCALAR_T c1 = cos(z);
  const SCALAR_T c2 = cos(y);
  const SCALAR_T c3 = cos(x);
  const SCALAR_T s1 = sin(z);
  const SCALAR_T s2 = sin(y);
  const SCALAR_T s3 = sin(x);

  const SCALAR_T s2s3 = s2 * s3;
  const SCALAR_T s2c3 = s2 * c3;

  // clang-format off
  Eigen::Matrix<SCALAR_T, kPositionDim, kPositionDim> matrix;
  matrix << c1 * c2,      c1 * s2s3 - s1 * c3,       c1 * s2c3 + s1 * s3,
            s1 * c2,      s1 * s2s3 + c1 * c3,       s1 * s2c3 - c1 * s3,
                -s2,                  c2 * s3,                   c2 * c3;
  // clang-format on
  return matrix;
}

/**
 * Compute a quaternion from a matrix
 *
 * @param [in] R: Rotation Matrix.
 * @return A quaternion representing an equivalent rotation to R.
 */
template <typename SCALAR_T>
Eigen::Quaternion<SCALAR_T> matrixToQuaternion(const Eigen::Matrix<SCALAR_T, kPositionDim, kPositionDim>& R) {
  return Eigen::Quaternion<SCALAR_T>(R);
}

/**
 * Compute a quaternion from a matrix, specialized for CppAd (the default Eigen implementation uses branches)
 *
 * @param [in] R: Rotation Matrix templated on ad_scalar_t.
 * @return A quaternion representing an equivalent rotation to R.
 */
inline Eigen::Quaternion<ad_scalar_t> matrixToQuaternion(const Eigen::Matrix<ad_scalar_t, kPositionDim, kPositionDim>& R) {
  // The elements R(row, column) of the rotation matrix, named by the axes of their row and column.
  const ad_scalar_t& rxx = R(kXAxisIndex, kXAxisIndex);
  const ad_scalar_t& rxy = R(kXAxisIndex, kYAxisIndex);
  const ad_scalar_t& rxz = R(kXAxisIndex, kZAxisIndex);
  const ad_scalar_t& ryx = R(kYAxisIndex, kXAxisIndex);
  const ad_scalar_t& ryy = R(kYAxisIndex, kYAxisIndex);
  const ad_scalar_t& ryz = R(kYAxisIndex, kZAxisIndex);
  const ad_scalar_t& rzx = R(kZAxisIndex, kXAxisIndex);
  const ad_scalar_t& rzy = R(kZAxisIndex, kYAxisIndex);
  const ad_scalar_t& rzz = R(kZAxisIndex, kZAxisIndex);

  ad_scalar_t t1, t2, t;
  ad_scalar_t x1, x2, x;
  ad_scalar_t y1, y2, y;
  ad_scalar_t z1, z2, z;
  ad_scalar_t w1, w2, w;

  t1 = CppAD::CondExpGt(rxx, ryy, 1 + rxx - ryy - rzz, 1 - rxx + ryy - rzz);
  t2 = CppAD::CondExpLt(rxx, -ryy, 1 - rxx - ryy + rzz, 1 + rxx + ryy + rzz);
  t = CppAD::CondExpLt(rzz, ad_scalar_t(0.0), t1, t2);

  x1 = CppAD::CondExpGt(rxx, ryy, t, ryx + rxy);
  x2 = CppAD::CondExpLt(rxx, -ryy, rxz + rzx, rzy - ryz);
  x = CppAD::CondExpLt(rzz, ad_scalar_t(0.0), x1, x2);

  y1 = CppAD::CondExpGt(rxx, ryy, ryx + rxy, t);
  y2 = CppAD::CondExpLt(rxx, -ryy, rzy + ryz, rxz - rzx);
  y = CppAD::CondExpLt(rzz, ad_scalar_t(0.0), y1, y2);

  z1 = CppAD::CondExpGt(rxx, ryy, rxz + rzx, rzy + ryz);
  z2 = CppAD::CondExpLt(rxx, -ryy, t, ryx - rxy);
  z = CppAD::CondExpLt(rzz, ad_scalar_t(0.0), z1, z2);

  w1 = CppAD::CondExpGt(rxx, ryy, rzy - ryz, rxz - rzx);
  w2 = CppAD::CondExpLt(rxx, -ryy, ryx - rxy, t);
  w = CppAD::CondExpLt(rzz, ad_scalar_t(0.0), w1, w2);

  Eigen::Matrix<ad_scalar_t, kQuaternionCoefficients, 1> q({x, y, z, w});
  q *= 0.5 / sqrt(t);

  return Eigen::Quaternion<ad_scalar_t>(q(kQuaternionWIndex), q(kQuaternionXIndex), q(kQuaternionYIndex), q(kQuaternionZIndex));
}

/**
 * Find an angle that is closest to a reference angle.
 *
 * @param [in] x : The input angle.
 * @param [in] reference : The reference angle.
 * @return An angle (x + k*2*pi) with k such that the result is within [reference - pi, reference + pi].
 */
scalar_t moduloAngleWithReference(scalar_t x, scalar_t reference);

/**
 * Compute the quaternion distance measure to the closest orientation aligned with a reference plane:
 * e = -(v x n) / sqrt(2 (1 + v.n)), v = q e_z the end-effector normal and n the plane normal. Finite, with finite
 * derivatives, where v = -n (getQuaternionFromUnitVectors()).
 *
 * @param [in] q: measured end effector quaternion.
 * @param [in] planeNormal: Unit normal of the plane the orientation should be aligned with.
 *
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kRotationTangentDim, 1> quaternionDistanceToPlane(const Eigen::Quaternion<SCALAR_T>& q,
                                                                          const Eigen::Matrix<SCALAR_T, kPositionDim, 1>& planeNormal) {
  const Eigen::Matrix<SCALAR_T, kPositionDim, 1> z_axis(SCALAR_T(0.0), SCALAR_T(0.0), SCALAR_T(1.0));

  // Passive rotation projecting from end effector frame to the closest frame in plane.
  // Computed through the shortest arc rotation  from the end effector z axis to the plane normal (both expressed in world frame).
  const Eigen::Quaternion<SCALAR_T> shortestArc = getQuaternionFromUnitVectors<SCALAR_T>(q * z_axis, planeNormal);

  return quaternionDistance<SCALAR_T>(shortestArc, Eigen::Quaternion<SCALAR_T>::Identity());
}

/**
 * Compute the quaternion distance measure to the closest orientation aligned with a reference plane:
 * e = -(v x n) / sqrt(2 (1 + v.n)), v = R e_z the end-effector normal and n the plane normal. Finite, with finite
 * derivatives, where v = -n (getQuaternionFromUnitVectors()).
 *
 * @param [in] R: measured rotation matrix end effector to world.
 * @param [in] planeNormal: Unit normal of the plane the orientation should be aligned with.
 *
 */

template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kRotationTangentDim, 1> rotationMatrixDistanceToPlane(const Eigen::Matrix<SCALAR_T, kPositionDim, kPositionDim>& R,
                                                                              const Eigen::Matrix<SCALAR_T, kPositionDim, 1>& planeNormal) {
  const Eigen::Matrix<SCALAR_T, kPositionDim, 1> z_axis(SCALAR_T(0.0), SCALAR_T(0.0), SCALAR_T(1.0));

  // Passive rotation projecting from end effector frame to the closest frame in plane.
  // Computed through the shortest arc rotation  from the end effector z axis to the plane normal (both expressed in world frame).
  const Eigen::Quaternion<SCALAR_T> shortestArc = getQuaternionFromUnitVectors<SCALAR_T>(R * z_axis, planeNormal);

  return quaternionDistance<SCALAR_T>(shortestArc, Eigen::Quaternion<SCALAR_T>::Identity());
}

}  // namespace ocs2