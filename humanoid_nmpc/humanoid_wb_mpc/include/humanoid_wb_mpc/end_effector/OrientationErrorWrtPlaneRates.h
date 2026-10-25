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

#include <cmath>

#include "Eigen/Core"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h"

/*
 * The first and second time derivatives of the orientation error of an end
 * effector with respect to a plane, the error the stance foot's orientation
 * rows hold (rotationMatrixDistanceToPlane() of ocs2_robotic_tools):
 *
 *   e = -(v x n) / s,    s = sqrt(2 (1 + v.n)),
 *
 * v = R e_z the end-effector normal and n the unit plane normal, both in the
 * world frame. e is the vector part of the shortest-arc rotation from v to n,
 * sign flipped: the sine of half the tilt angle (about half the angle for a
 * small tilt), perpendicular to n. Along a motion with the world-frame angular
 * velocity w and angular acceleration alpha = dw/dt, n fixed,
 *
 *   dv/dt = w x v,    d2v/dt2 = alpha x v + w x (w x v),
 *
 * and with a = v x n and c = v.n (so that ds/dt = (dc/dt) / s), da/dt = (dv/dt)
 * x n, dc/dt = (dv/dt).n and so on:
 *
 *   de/dt   = -(da/dt) / s + a (dc/dt) / s^3,
 *   d2e/dt2 = -(d2a/dt2) / s + (2 (da/dt)(dc/dt) + a (d2c/dt2)) / s^3 - 3 a
 * (dc/dt)^2 / s^5.
 *
 * de/dt = M(v) w with the map M = de/dw of EndEffectorKinematicsTwistConstraint
 * (the centroidal MPC's), and its rank is two: e has no component along n, and
 * a rotation about v leaves v, and so e, unchanged. The rate the stance rows
 * hold therefore adds, in the direction of n, half the rotation rate about the
 * end-effector normal, v.w, with the half-angle scaling of the tilt rows
 * (EndEffectorKinematicsTwistConstraint's yaw-rate term 1/2 n v^T); its exact
 * time derivative is 1/2 v.alpha, since d/dt (v.w) = (w x v).w + v.alpha. At a
 * level end effector (v = n) the rate is w / 2 and its derivative alpha / 2 +
 * 1/2 (n.w)(n x w). Where the normal points away from the plane the shortest
 * arc, and with it e, is not differentiable, and both functions fall back to w
 * / 2 and alpha / 2, as EndEffectorKinematicsTwistConstraint does (design
 * README section 2.8.6,
 * humanoid_nmpc/docs/quaternion_base_orientation/README.md).
 *
 * Templates on the scalar, for double and for a CppAD tape
 * (PinocchioEndEffectorDynamicsCppAd records them): the guard is a conditional
 * expression on an AD scalar, and the branch it selects away is evaluated at 1
 * + v.n = 1, so that both branches and their derivatives are finite at every
 * input, the anti-parallel normal included.
 */
namespace ocs2::humanoid {

// The guard, kOrientationErrorWrtPlaneAntiParallelThreshold, and the half-angle
// scaling of the rates, kQuaternionHalfAngleFactor (the orientation error is
// half the rotation angle to first order, so its rate is half the angular
// velocity), are EndEffectorKinematicsTwistConstraint's, which falls back
// alike.

namespace internal {

/** s^2 = kSquaredRootPerOnePlusCosine (1 + v.n): s is the norm of v + n, of
 * square 2 (1 + v.n) for unit v and n. */
inline constexpr scalar_t kSquaredRootPerOnePlusCosine = 2.0;
/** The coefficient of (da/dt)(dc/dt) / s^3 in d2e/dt2 (the file comment): one
 * from each term of de/dt. */
inline constexpr scalar_t kAxisRateCosineRateCoefficient = 2.0;
/** The coefficient of a (dc/dt)^2 / s^5 in d2e/dt2: the power of s^-3 in de/dt,
 * as ds/dt = (dc/dt) / s. */
inline constexpr scalar_t kSquaredCosineRateCoefficient = 3.0;

/** left < right ? ifTrue : ifFalse, as a plain conditional for double. */
inline double ifLessThan(double left,
                         double right,
                         double ifTrue,
                         double ifFalse) {
  return left < right ? ifTrue : ifFalse;
}

/** left < right ? ifTrue : ifFalse, as a CppAD conditional expression for an AD
 * scalar, which tapes both branches. */
template <typename SCALAR_T>
SCALAR_T ifLessThan(const SCALAR_T& left,
                    const SCALAR_T& right,
                    const SCALAR_T& ifTrue,
                    const SCALAR_T& ifFalse) {
  return CppAD::CondExpLt(left, right, ifTrue, ifFalse);
}

/** What both derivatives of e read of the attitude. Passive data. */
template <typename SCALAR_T>
struct PlaneTilt {
  /** v = R e_z, the end-effector normal in the world frame. */
  VECTOR3_T<SCALAR_T> normal;
  /** a = v x n. */
  VECTOR3_T<SCALAR_T> axis;
  /** 1 + v.n; the anti-parallel guard applies below
   * kOrientationErrorWrtPlaneAntiParallelThreshold. */
  SCALAR_T onePlusCosine = SCALAR_T(0.0);
  /** s = sqrt(2 (1 + v.n)) outside the guard; sqrt(2) inside it, where only the
   * fallback is selected. */
  SCALAR_T root = SCALAR_T(0.0);
};

/** The PlaneTilt of the attitude `rotation` (end effector to world) against the
 * unit normal `planeNormal`. */
template <typename SCALAR_T>
PlaneTilt<SCALAR_T> planeTiltOf(const MATRIX3_T<SCALAR_T>& rotation,
                                const VECTOR3_T<SCALAR_T>& planeNormal) {
  using std::sqrt;
  PlaneTilt<SCALAR_T> tilt;
  tilt.normal = rotation.col(kZAxisIndex);
  tilt.axis = tilt.normal.cross(planeNormal);
  tilt.onePlusCosine = SCALAR_T(1.0) + tilt.normal.dot(planeNormal);
  // The guard is applied before the square root and the divisions: the branch
  // selected away sees 1 + v.n = 1.
  const SCALAR_T guarded =
      ifLessThan(tilt.onePlusCosine,
                 SCALAR_T(kOrientationErrorWrtPlaneAntiParallelThreshold),
                 SCALAR_T(1.0), tilt.onePlusCosine);
  tilt.root = sqrt(SCALAR_T(kSquaredRootPerOnePlusCosine) * guarded);
  return tilt;
}

/** `fallback` where 1 + v.n < kOrientationErrorWrtPlaneAntiParallelThreshold,
 * `regular` elsewhere, entry by entry. */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> selectOutsideAntiParallelGuard(
    const SCALAR_T& onePlusCosine,
    const VECTOR3_T<SCALAR_T>& fallback,
    const VECTOR3_T<SCALAR_T>& regular) {
  VECTOR3_T<SCALAR_T> selected;
  for (Eigen::Index i = 0; i < kRotationTangentDim; ++i) {
    selected(i) = ifLessThan(
        onePlusCosine, SCALAR_T(kOrientationErrorWrtPlaneAntiParallelThreshold),
        fallback(i), regular(i));
  }
  return selected;
}

}  // namespace internal

/**
 * Returns de/dt + 1/2 n (v.w) of the file comment: the rate of the orientation
 * error with respect to the plane of the unit normal `planeNormal`, completed
 * along the normal by half the rotation rate about the end-effector normal; w /
 * 2 where 1 + v.n < kOrientationErrorWrtPlaneAntiParallelThreshold. `rotation`
 * maps the end-effector frame to the world frame and `angularVelocity` is the
 * end effector's, in the world frame.
 */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> orientationErrorWrtPlaneRate(
    const MATRIX3_T<SCALAR_T>& rotation,
    const VECTOR3_T<SCALAR_T>& angularVelocity,
    const VECTOR3_T<SCALAR_T>& planeNormal) {
  const internal::PlaneTilt<SCALAR_T> tilt =
      internal::planeTiltOf(rotation, planeNormal);
  const VECTOR3_T<SCALAR_T> normalRate = angularVelocity.cross(tilt.normal);
  const VECTOR3_T<SCALAR_T> axisRate = normalRate.cross(planeNormal);
  const SCALAR_T cosineRate = normalRate.dot(planeNormal);
  const SCALAR_T squaredRoot = tilt.root * tilt.root;
  const VECTOR3_T<SCALAR_T> tiltRate =
      (tilt.axis * (cosineRate / squaredRoot) - axisRate) / tilt.root;
  const SCALAR_T normalSpinRate = tilt.normal.dot(angularVelocity);
  const VECTOR3_T<SCALAR_T> regular =
      tiltRate +
      planeNormal * (SCALAR_T(kQuaternionHalfAngleFactor) * normalSpinRate);
  const VECTOR3_T<SCALAR_T> fallback =
      angularVelocity * SCALAR_T(kQuaternionHalfAngleFactor);
  return internal::selectOutsideAntiParallelGuard(tilt.onePlusCosine, fallback,
                                                  regular);
}

/**
 * Returns the time derivative of orientationErrorWrtPlaneRate() along the
 * motion, d2e/dt2 + 1/2 n (v.alpha) of the file comment; alpha / 2 where 1 +
 * v.n < kOrientationErrorWrtPlaneAntiParallelThreshold. `angularAcceleration`
 * is the time derivative of `angularVelocity`, in the world frame (the angular
 * part of Pinocchio's frame acceleration, LOCAL_WORLD_ALIGNED).
 */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> orientationErrorWrtPlaneAcceleration(
    const MATRIX3_T<SCALAR_T>& rotation,
    const VECTOR3_T<SCALAR_T>& angularVelocity,
    const VECTOR3_T<SCALAR_T>& angularAcceleration,
    const VECTOR3_T<SCALAR_T>& planeNormal) {
  const internal::PlaneTilt<SCALAR_T> tilt =
      internal::planeTiltOf(rotation, planeNormal);
  const VECTOR3_T<SCALAR_T> normalRate = angularVelocity.cross(tilt.normal);
  const VECTOR3_T<SCALAR_T> normalAcceleration =
      angularAcceleration.cross(tilt.normal) +
      angularVelocity.cross(normalRate);
  const VECTOR3_T<SCALAR_T> axisRate = normalRate.cross(planeNormal);
  const VECTOR3_T<SCALAR_T> axisAcceleration =
      normalAcceleration.cross(planeNormal);
  const SCALAR_T cosineRate = normalRate.dot(planeNormal);
  const SCALAR_T cosineAcceleration = normalAcceleration.dot(planeNormal);
  const SCALAR_T squaredRoot = tilt.root * tilt.root;
  const VECTOR3_T<SCALAR_T> tiltAcceleration =
      ((axisRate *
            (SCALAR_T(internal::kAxisRateCosineRateCoefficient) * cosineRate) +
        tilt.axis * cosineAcceleration) /
           squaredRoot -
       tilt.axis * (SCALAR_T(internal::kSquaredCosineRateCoefficient) *
                    cosineRate * cosineRate / (squaredRoot * squaredRoot)) -
       axisAcceleration) /
      tilt.root;
  const SCALAR_T normalSpinAcceleration = tilt.normal.dot(angularAcceleration);
  const VECTOR3_T<SCALAR_T> regular =
      tiltAcceleration + planeNormal * (SCALAR_T(kQuaternionHalfAngleFactor) *
                                        normalSpinAcceleration);
  const VECTOR3_T<SCALAR_T> fallback =
      angularAcceleration * SCALAR_T(kQuaternionHalfAngleFactor);
  return internal::selectOutsideAntiParallelGuard(tilt.onePlusCosine, fallback,
                                                  regular);
}

}  // namespace ocs2::humanoid
