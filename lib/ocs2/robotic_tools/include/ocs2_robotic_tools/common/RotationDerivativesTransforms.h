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
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include <Eigen/Core>

namespace ocs2 {

/**
 * Compute derivatives of ZYX-Euler angles from global angular velocities
 * The inverse of the map w = [0, -sz, cy*cz; 0, cz, cy*sz; 1, 0, -sy] * d/dt (z, y, x)
 *
 * This transformation is singular for y = +- pi / 2
 *
 * @param [in] eulerAngles: ZYX-Euler angles
 * @param [in] angularVelocity: angular velocity expressed in world frame
 * @return derivatives of ZYX-Euler angles
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kEulerAnglesZyxDim, 1> getEulerAnglesZyxDerivativesFromGlobalAngularVelocity(
    const Eigen::Matrix<SCALAR_T, kEulerAnglesZyxDim, 1>& eulerAngles,
    const Eigen::Matrix<SCALAR_T, kRotationTangentDim, 1>& angularVelocity) {
  const SCALAR_T sz = sin(eulerAngles(kEulerAnglesZyxZIndex));
  const SCALAR_T cz = cos(eulerAngles(kEulerAnglesZyxZIndex));
  const SCALAR_T sy = sin(eulerAngles(kEulerAnglesZyxYIndex));
  const SCALAR_T cy = cos(eulerAngles(kEulerAnglesZyxYIndex));
  const SCALAR_T wx = angularVelocity(kXAxisIndex);
  const SCALAR_T wy = angularVelocity(kYAxisIndex);
  const SCALAR_T wz = angularVelocity(kZAxisIndex);
  const SCALAR_T tmp = cz * wx / cy + sz * wy / cy;
  return {sy * tmp + wz, -sz * wx + cz * wy, tmp};
}

/**
 * Compute the matrix that maps derivatives of ZYX-Euler angles to local angular velocities
 *
 * @param [in] eulerAngles: ZYX-Euler angles
 * @return 3x3 matrix mapping
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kRotationTangentDim, kEulerAnglesZyxDim> getMappingFromEulerAnglesZyxDerivativeToLocalAngularVelocity(
    const Eigen::Matrix<SCALAR_T, kEulerAnglesZyxDim, 1>& eulerAngles) {
  const SCALAR_T sy = sin(eulerAngles(kEulerAnglesZyxYIndex));
  const SCALAR_T cy = cos(eulerAngles(kEulerAnglesZyxYIndex));
  const SCALAR_T sx = sin(eulerAngles(kEulerAnglesZyxXIndex));
  const SCALAR_T cx = cos(eulerAngles(kEulerAnglesZyxXIndex));
  Eigen::Matrix<SCALAR_T, kRotationTangentDim, kEulerAnglesZyxDim> transformationMatrix;
  // clang-format off
  transformationMatrix <<     -sy,  SCALAR_T(0.0),    SCALAR_T(1.0),
                          cy * sx,             cx,    SCALAR_T(0.0),
                          cx * cy,            -sx,    SCALAR_T(0.0);
  // clang-format on
  return transformationMatrix;
}

/**
 * Compute angular velocities expressed in the local frame from derivatives of ZYX-Euler angles
 *
 * @param [in] eulerAngles: ZYX-Euler angles
 * @param [in] derivativesEulerAngles: time-derivative of ZYX-Euler angles
 * @return angular velocity expressed in local frame
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kRotationTangentDim, 1> getLocalAngularVelocityFromEulerAnglesZyxDerivatives(
    const Eigen::Matrix<SCALAR_T, kEulerAnglesZyxDim, 1>& eulerAngles,
    const Eigen::Matrix<SCALAR_T, kEulerAnglesZyxDim, 1>& derivativesEulerAngles) {
  const SCALAR_T sy = sin(eulerAngles(kEulerAnglesZyxYIndex));
  const SCALAR_T cy = cos(eulerAngles(kEulerAnglesZyxYIndex));
  const SCALAR_T sx = sin(eulerAngles(kEulerAnglesZyxXIndex));
  const SCALAR_T cx = cos(eulerAngles(kEulerAnglesZyxXIndex));
  const SCALAR_T dz = derivativesEulerAngles(kEulerAnglesZyxZIndex);
  const SCALAR_T dy = derivativesEulerAngles(kEulerAnglesZyxYIndex);
  const SCALAR_T dx = derivativesEulerAngles(kEulerAnglesZyxXIndex);
  return {-sy * dz + dx, cy * sx * dz + cx * dy, cx * cy * dz - sx * dy};
}

/**
 * Compute derivatives of ZYX-Euler angles from local angular velocities
 * The inverse of getLocalAngularVelocityFromEulerAnglesZyxDerivatives
 *
 * This transformation is singular for y = +- pi / 2
 *
 * @param [in] eulerAngles: ZYX-Euler angles
 * @param [in] angularVelocity: angular velocity expressed in local frame
 * @return derivatives of ZYX-Euler angles
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kEulerAnglesZyxDim, 1> getEulerAnglesZyxDerivativesFromLocalAngularVelocity(
    const Eigen::Matrix<SCALAR_T, kEulerAnglesZyxDim, 1>& eulerAngles,
    const Eigen::Matrix<SCALAR_T, kRotationTangentDim, 1>& angularVelocity) {
  const SCALAR_T sy = sin(eulerAngles(kEulerAnglesZyxYIndex));
  const SCALAR_T cy = cos(eulerAngles(kEulerAnglesZyxYIndex));
  const SCALAR_T sx = sin(eulerAngles(kEulerAnglesZyxXIndex));
  const SCALAR_T cx = cos(eulerAngles(kEulerAnglesZyxXIndex));
  const SCALAR_T wx = angularVelocity(kXAxisIndex);
  const SCALAR_T wy = angularVelocity(kYAxisIndex);
  const SCALAR_T wz = angularVelocity(kZAxisIndex);
  const SCALAR_T tmp = sx * wy / cy + cx * wz / cy;
  return {tmp, cx * wy - sx * wz, wx + sy * tmp};
}

}  // namespace ocs2