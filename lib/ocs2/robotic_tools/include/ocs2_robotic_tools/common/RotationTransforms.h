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

#include <Eigen/Core>
#include <array>
#include <cmath>

// CppAD
#include <ocs2_core/automatic_differentiation/Types.h>

namespace ocs2 {

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
Eigen::Matrix<SCALAR_T, 3, 1> quaternionDistance(const Eigen::Quaternion<SCALAR_T>& q, const Eigen::Quaternion<SCALAR_T>& qRef) {
  return q.w() * qRef.vec() - qRef.w() * q.vec() + q.vec().cross(qRef.vec());
}

/**
 * Compute the quaternion corresponding to euler angles zyx
 *
 * @param [in] eulerAnglesZyx
 * @return The corresponding quaternion
 */
template <typename SCALAR_T>
Eigen::Quaternion<SCALAR_T> getQuaternionFromEulerAnglesZyx(const Eigen::Matrix<SCALAR_T, 3, 1>& eulerAnglesZyx) {
  // clang-format off
  return Eigen::AngleAxis<SCALAR_T>(eulerAnglesZyx(0), Eigen::Matrix<SCALAR_T, 3, 1>::UnitZ()) *
         Eigen::AngleAxis<SCALAR_T>(eulerAnglesZyx(1), Eigen::Matrix<SCALAR_T, 3, 1>::UnitY()) *
         Eigen::AngleAxis<SCALAR_T>(eulerAnglesZyx(2), Eigen::Matrix<SCALAR_T, 3, 1>::UnitX());
  // clang-format on
}

/**
 * Compute the quaternion corresponding to the rotation from unit vector a to unit vector b. This fkunction is not yet tested for non AD
 * types.
 *
 * @param [in] a Unit vector of initial orientation
 * @param [in] b Unit vector of final orientation
 * @return The corresponding quaternion
 */
template <typename SCALAR_T>
Eigen::Quaternion<SCALAR_T> getQuaternionFromUnitVectors(const Eigen::Matrix<SCALAR_T, 3, 1>& a, const Eigen::Matrix<SCALAR_T, 3, 1>& b) {
  // q.vec() = a.cross(b);
  SCALAR_T x = a[1] * b[2] - a[2] * b[1];
  SCALAR_T y = a[2] * b[0] - a[0] * b[2];
  SCALAR_T z = a[0] * b[1] - a[1] * b[0];
  // q.w = sqrt(len(a)^2 * len(b)^2) + a.dot(b);
  SCALAR_T w = 1 + a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const SCALAR_T norm = CppAD::sqrt(x * x + y * y + z * z + w * w);

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
Eigen::Matrix<SCALAR_T, 3, 3> getRotationMatrixFromZyxEulerAngles(const Eigen::Matrix<SCALAR_T, 3, 1>& eulerAngles) {
  const SCALAR_T z = eulerAngles(0);
  const SCALAR_T y = eulerAngles(1);
  const SCALAR_T x = eulerAngles(2);

  const SCALAR_T c1 = cos(z);
  const SCALAR_T c2 = cos(y);
  const SCALAR_T c3 = cos(x);
  const SCALAR_T s1 = sin(z);
  const SCALAR_T s2 = sin(y);
  const SCALAR_T s3 = sin(x);

  const SCALAR_T s2s3 = s2 * s3;
  const SCALAR_T s2c3 = s2 * c3;

  // clang-format off
  Eigen::Matrix<SCALAR_T, 3, 3> rotationMatrix;
  rotationMatrix << c1 * c2,      c1 * s2s3 - s1 * c3,       c1 * s2c3 + s1 * s3,
                    s1 * c2,      s1 * s2s3 + c1 * c3,       s1 * s2c3 - c1 * s3,
                        -s2,                  c2 * s3,                   c2 * c3;
  // clang-format on
  return rotationMatrix;
}

/**
 * Compute a quaternion from a matrix
 *
 * @param [in] R: Rotation Matrix.
 * @return A quaternion representing an equivalent rotation to R.
 */
template <typename SCALAR_T>
Eigen::Quaternion<SCALAR_T> matrixToQuaternion(const Eigen::Matrix<SCALAR_T, 3, 3>& R) {
  return Eigen::Quaternion<SCALAR_T>(R);
}

/**
 * Compute a quaternion from a matrix, specialized for CppAd (the default Eigen implementation uses branches)
 *
 * @param [in] R: Rotation Matrix templated on ad_scalar_t.
 * @return A quaternion representing an equivalent rotation to R.
 */
inline Eigen::Quaternion<ad_scalar_t> matrixToQuaternion(const Eigen::Matrix<ad_scalar_t, 3, 3>& R) {
  ad_scalar_t t1, t2, t;
  ad_scalar_t x1, x2, x;
  ad_scalar_t y1, y2, y;
  ad_scalar_t z1, z2, z;
  ad_scalar_t w1, w2, w;

  t1 = CppAD::CondExpGt(R(0, 0), R(1, 1), 1 + R(0, 0) - R(1, 1) - R(2, 2), 1 - R(0, 0) + R(1, 1) - R(2, 2));
  t2 = CppAD::CondExpLt(R(0, 0), -R(1, 1), 1 - R(0, 0) - R(1, 1) + R(2, 2), 1 + R(0, 0) + R(1, 1) + R(2, 2));
  t = CppAD::CondExpLt(R(2, 2), ad_scalar_t(0.0), t1, t2);

  x1 = CppAD::CondExpGt(R(0, 0), R(1, 1), t, R(1, 0) + R(0, 1));
  x2 = CppAD::CondExpLt(R(0, 0), -R(1, 1), R(0, 2) + R(2, 0), R(2, 1) - R(1, 2));
  x = CppAD::CondExpLt(R(2, 2), ad_scalar_t(0.0), x1, x2);

  y1 = CppAD::CondExpGt(R(0, 0), R(1, 1), R(1, 0) + R(0, 1), t);
  y2 = CppAD::CondExpLt(R(0, 0), -R(1, 1), R(2, 1) + R(1, 2), R(0, 2) - R(2, 0));
  y = CppAD::CondExpLt(R(2, 2), ad_scalar_t(0.0), y1, y2);

  z1 = CppAD::CondExpGt(R(0, 0), R(1, 1), R(0, 2) + R(2, 0), R(2, 1) + R(1, 2));
  z2 = CppAD::CondExpLt(R(0, 0), -R(1, 1), t, R(1, 0) - R(0, 1));
  z = CppAD::CondExpLt(R(2, 2), ad_scalar_t(0.0), z1, z2);

  w1 = CppAD::CondExpGt(R(0, 0), R(1, 1), R(2, 1) - R(1, 2), R(0, 2) - R(2, 0));
  w2 = CppAD::CondExpLt(R(0, 0), -R(1, 1), R(1, 0) - R(0, 1), t);
  w = CppAD::CondExpLt(R(2, 2), ad_scalar_t(0.0), w1, w2);

  Eigen::Matrix<ad_scalar_t, 4, 1> q({x, y, z, w});
  q *= 0.5 / sqrt(t);

  return Eigen::Quaternion<ad_scalar_t>(q(3), q(0), q(1), q(2));
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
 * Compute the quaternion distance measure to the closest orientation aligned with a reference plane.
 *
 * @param [in] q: measured end effector quaternion.
 * @param [in] planeNormal: desired end effector quaternion.Normal of the plane the orientation should be aligned with
 *
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 1> quaternionDistanceToPlane(const Eigen::Quaternion<SCALAR_T>& q,
                                                        const Eigen::Matrix<SCALAR_T, 3, 1>& planeNormal) {
  const Eigen::Matrix<SCALAR_T, 3, 1> z_axis(SCALAR_T(0.0), SCALAR_T(0.0), SCALAR_T(1.0));

  // Passive rotation projecting from end effector frame to the closest frame in plane.
  // Computed through the shortest arc rotation  from the end effector z axis to the plane normal (both expressed in world frame).
  const Eigen::Quaternion<SCALAR_T> quaternion_correction = getQuaternionFromUnitVectors<SCALAR_T>(q * z_axis, planeNormal);

  return quaternionDistance<SCALAR_T>(quaternion_correction, ad_quaternion_t::Identity());
}

/**
 * Compute the quaternion distance measure to the closest orientation aligned with a reference plane.
 *
 * @param [in] R: measured rotation matrix world to end effector.
 * @param [in] planeNormal: desired end effector quaternion.Normal of the plane the orientation should be aligned with
 *
 */

template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 1> rotationMatrixDistanceToPlane(const Eigen::Matrix<SCALAR_T, 3, 3>& R,
                                                            const Eigen::Matrix<SCALAR_T, 3, 1>& planeNormal) {
  const Eigen::Matrix<SCALAR_T, 3, 1> z_axis(SCALAR_T(0.0), SCALAR_T(0.0), SCALAR_T(1.0));

  // Passive rotation projecting from end effector frame to the closest frame in plane.
  // Computed through the shortest arc rotation  from the end effector z axis to the plane normal (both expressed in world frame).
  const Eigen::Quaternion<SCALAR_T> quaternion_correction = getQuaternionFromUnitVectors<SCALAR_T>(R * z_axis, planeNormal);

  return quaternionDistance<SCALAR_T>(quaternion_correction, Eigen::Quaternion<SCALAR_T>::Identity());
}

}  // namespace ocs2