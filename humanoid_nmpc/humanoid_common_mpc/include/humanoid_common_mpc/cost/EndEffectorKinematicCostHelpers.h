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

#pragma once

#include "absl/status/statusor.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The entries of an EndEffectorKinematicsCostElement: the pose's coefficients
 * (position and orientation quaternion) and the twist (linear and angular
 * velocity).
 */
inline constexpr int kEndEffectorCostElementDim = kPoseCoefficients + kTwistDim;

/**
 * The task-space error of an end effector, computeTaskSpaceErrors(), and with
 * it the weights of EndEffectorKinematicsWeights::toVector(): the pose's
 * tangent (position and orientation error) and the twist.
 */
inline constexpr int kEndEffectorTaskSpaceErrorDim =
    kPoseTangentDim + kTwistDim;

/** The task-space weights of an end-effector cost: position, orientation,
 * linear and angular velocity, per axis. */
struct EndEffectorKinematicsWeights {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // The default weight of every axis of each error.
  static constexpr scalar_t kDefaultPositionErrorWeight = 3.0;
  // Per rad^2 (computeTaskSpaceErrors(), tiltAngleToPlane()): 3 on the former
  // half-angle residuals.
  static constexpr scalar_t kDefaultOrientationErrorWeight = 0.75;
  static constexpr scalar_t kDefaultLinearVelocityErrorWeight = 0.01;
  static constexpr scalar_t kDefaultAngularVelocityErrorWeight = 0.01;

  // Cost weights [w_x,w_y,w_z]
  vector3_t contactPositionErrorWeight{kDefaultPositionErrorWeight,
                                       kDefaultPositionErrorWeight,
                                       kDefaultPositionErrorWeight};
  vector3_t contactOrientationErrorWeight{kDefaultOrientationErrorWeight,
                                          kDefaultOrientationErrorWeight,
                                          kDefaultOrientationErrorWeight};
  vector3_t contactLinearVelocityErrorWeight{kDefaultLinearVelocityErrorWeight,
                                             kDefaultLinearVelocityErrorWeight,
                                             kDefaultLinearVelocityErrorWeight};
  vector3_t contactAngularVelocityErrorWeight{
      kDefaultAngularVelocityErrorWeight, kDefaultAngularVelocityErrorWeight,
      kDefaultAngularVelocityErrorWeight};

  vector12_t toVector() const;
};

/**
 * The task-space state of an end effector packed in one 13-vector, [position
 * (3), orientation quaternion coefficients (4), linear velocity (3), angular
 * velocity (3)], with typed accessors; the parameter layout of the end-effector
 * costs.
 */
template <typename SCALAR_T>
struct EndEffectorKinematicsCostElement {
  // Where each block starts; the position is at the head and the angular
  // velocity at the tail.
  static constexpr int kOrientationStart = kPositionDim;
  static constexpr int kLinearVelocityStart = kPoseCoefficients;

  explicit EndEffectorKinematicsCostElement(
      const Eigen::Matrix<SCALAR_T, kEndEffectorCostElementDim, 1>& vector)
      : costElementVector(vector) {}
  EndEffectorKinematicsCostElement()
      : costElementVector(
            Eigen::Matrix<SCALAR_T, kEndEffectorCostElementDim, 1>::Zero()) {}
  Eigen::Matrix<SCALAR_T, kEndEffectorCostElementDim, 1> costElementVector;

  Eigen::Matrix<SCALAR_T, kEndEffectorCostElementDim, 1> getValues() {
    return costElementVector;
  }

  VECTOR3_T<SCALAR_T> getPosEeRtWorldEwrtWorld() const {
    return costElementVector.head(kPositionDim);
  }
  QUATERNION_T<SCALAR_T> getQuatEeRtWorld() const {
    return QUATERNION_T<SCALAR_T>(VECTOR4_T<SCALAR_T>(
        costElementVector.segment(kOrientationStart, kQuaternionCoefficients)));
  }
  VECTOR3_T<SCALAR_T> getLinvelEeRtWorldEwrtWorld() const {
    return costElementVector.segment(kLinearVelocityStart, kPositionDim);
  }
  VECTOR3_T<SCALAR_T> getAngvelEeRtWorldEwrtWorld() const {
    return costElementVector.tail(kRotationTangentDim);
  }

  void setPosEeRtWorldEwrtWorld(
      const VECTOR3_T<SCALAR_T>& posEeRtWorldEwrtWorld) {
    costElementVector.head(kPositionDim) = posEeRtWorldEwrtWorld;
  }
  void setQuatEeRtWorld(const VECTOR4_T<SCALAR_T>& quatEeRtWorld) {
    costElementVector.segment(kOrientationStart, kQuaternionCoefficients) =
        quatEeRtWorld;
  }
  void setQuatEeRtWorld(const QUATERNION_T<SCALAR_T>& quatEeRtWorld) {
    costElementVector.segment(kOrientationStart, kQuaternionCoefficients) =
        quatEeRtWorld.coeffs();
  }
  void setLinvelEeRtWorldEwrtWorld(
      const VECTOR3_T<SCALAR_T>& linvelEeRtWorldEwrtWorld) {
    costElementVector.segment(kLinearVelocityStart, kPositionDim) =
        linvelEeRtWorldEwrtWorld;
  }
  void setAngvelEeRtWorldEwrtWorld(
      const VECTOR3_T<SCALAR_T>& angvelEeRtWorldEwrtWorld) {
    costElementVector.tail(kRotationTangentDim) = angvelEeRtWorldEwrtWorld;
  }
};

/**
 * Returns the task-space error of `current` against `reference`, [position (3),
 * orientation (3), linear velocity (3), angular velocity (3)], every block in
 * world axes. The orientation error is the rotation vector theta n of the
 * rotation from the current orientation to the reference one, Log(q_ref (x)
 * q^-1) with |theta| <= pi, in world axes: a weight on its x, y or z entry is
 * per rad^2 of rotation about that world axis, so orientation_z weighs the
 * heading error and orientation_x / orientation_y the tilt. Either quaternion
 * may be of either sign and need not be unit. The orientation error used to be
 * the vector part sin(theta / 2) n of that error quaternion; a weight w on it
 * equals w / 4 on this one at small angles, which is how the shipped
 * task_space_costs weights were rescaled.
 */
template <typename SCALAR_T>
VECTOR12_T<SCALAR_T> computeTaskSpaceErrors(
    const EndEffectorKinematicsCostElement<SCALAR_T>& current,
    const EndEffectorKinematicsCostElement<SCALAR_T>& reference);

/**
 * Returns the tilt of an end effector whose orientation is `rotmatEeRtWorld`
 * (end effector to world) against the plane whose unit normal is `planeNormal`,
 * as a rotation vector in world axes: theta u, with v = R e_z the end-effector
 * normal, theta = atan2(|n x v|, n . v) the angle between v and n and u = (n x
 * v) / |n x v| the axis that turns n into v. A weight on its x, y or z entry is
 * per rad^2 of tilt about that world axis, as on the orientation error of
 * computeTaskSpaceErrors(); the rotation about the plane normal is not part of
 * it (u is orthogonal to n).
 *
 * It is -Log(q_c), q_c the shortest arc from v to n
 * (getQuaternionFromUnitVectors()), whose vector part, the half-angle error -(v
 * x n) / sqrt(2 (1 + v . n)) = -sin(theta / 2) u of
 * rotationMatrixDistanceToPlane(), it replaces: the two agree to first order up
 * to the factor 2, so a weight w on the half-angle error is w / 4 on this one
 * at small tilts. Value and derivatives are finite at every unit v, on a CppAD
 * tape too: at v = n through quaternionLog()'s series branch, at v = -n through
 * the shortest arc's clamp, where q_c tends to the zero quaternion and the
 * residual to 0 (the axis of a half turn away from the plane is not defined). A
 * template so that the tapes of the foot costs and the tests share it.
 */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> tiltAngleToPlane(const MATRIX3_T<SCALAR_T>& rotmatEeRtWorld,
                                     const VECTOR3_T<SCALAR_T>& planeNormal);

/**
 * The planar reference of an end effector packed in one 12-vector, [position
 * (3), plane normal (3), linear velocity (3), angular velocity (3)], with typed
 * accessors.
 */
template <typename SCALAR_T>
struct PlanarEndEffectorKinematicsPlanarReference {
  // Where each block starts; the position is at the head and the angular
  // velocity at the tail. The plane normal is a direction in space, with as
  // many entries as a position.
  static constexpr int kPlaneNormalStart = kPositionDim;
  static constexpr int kLinearVelocityStart = kPlaneNormalStart + kPositionDim;
  static constexpr int kAngularVelocityStart =
      kLinearVelocityStart + kPositionDim;
  /** The entries of the reference, the size of costElementVector. */
  static constexpr int kDim = kAngularVelocityStart + kRotationTangentDim;
  static_assert(kDim == VECTOR12_T<SCALAR_T>::RowsAtCompileTime,
                "the reference is one 12-vector");

  explicit PlanarEndEffectorKinematicsPlanarReference(
      const VECTOR12_T<SCALAR_T>& vector)
      : costElementVector(vector) {}
  PlanarEndEffectorKinematicsPlanarReference()
      : costElementVector(VECTOR12_T<SCALAR_T>::Zero()) {}
  VECTOR12_T<SCALAR_T> costElementVector;

  VECTOR12_T<SCALAR_T> getValues() { return costElementVector; }

  VECTOR3_T<SCALAR_T> getPosEeRtWorldEwrtWorld() const {
    return costElementVector.head(kPositionDim);
  }
  VECTOR3_T<SCALAR_T> getPlaneNormal() const {
    return costElementVector.segment(kPlaneNormalStart, kPositionDim);
  }
  VECTOR3_T<SCALAR_T> getLinvelEeRtWorldEwrtWorld() const {
    return costElementVector.segment(kLinearVelocityStart, kPositionDim);
  }
  VECTOR3_T<SCALAR_T> getAngvelEeRtWorldEwrtWorld() const {
    return costElementVector.tail(kRotationTangentDim);
  }

  void setPosEeRtWorldEwrtWorld(
      const VECTOR3_T<SCALAR_T>& posEeRtWorldEwrtWorld) {
    costElementVector.head(kPositionDim) = posEeRtWorldEwrtWorld;
  }
  void setLinvelEeRtWorldEwrtWorld(
      const VECTOR3_T<SCALAR_T>& linvelEeRtWorldEwrtWorld) {
    costElementVector.segment(kLinearVelocityStart, kPositionDim) =
        linvelEeRtWorldEwrtWorld;
  }
  void setAngvelEeRtWorldEwrtWorld(
      const VECTOR3_T<SCALAR_T>& angvelEeRtWorldEwrtWorld) {
    costElementVector.tail(kRotationTangentDim) = angvelEeRtWorldEwrtWorld;
  }
};

}  // namespace ocs2::humanoid
