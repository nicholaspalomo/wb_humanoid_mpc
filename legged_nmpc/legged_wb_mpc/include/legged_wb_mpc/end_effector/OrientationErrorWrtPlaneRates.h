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
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"

#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h"

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
 * legged_nmpc/docs/quaternion_base_orientation/README.md).
 *
 * Templates on the scalar, for double and for a CppAD tape
 * (PinocchioEndEffectorDynamicsCppAd records them): the guard is a conditional
 * expression on an AD scalar, and the branch it selects away is evaluated at 1
 * + v.n = 1, so that both branches and their derivatives are finite at every
 * input, the anti-parallel normal included.
 */
namespace ocs2::legged {

// The guard, kPlaneNormalAntiParallelThreshold, and the half-angle
// scaling of the rates, kUnitQuaternionHalfAngleFactor (the orientation error
// is half the rotation angle to first order, so its rate is half the angular
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
  /** v = R e_z, the end effector's z axis in the world frame. */
  VECTOR3_T<SCALAR_T> zAxisEeEwrtWorld;
  /** a = v x n. */
  VECTOR3_T<SCALAR_T> vCrossN;
  /** 1 + v.n; the anti-parallel guard applies below
   * kPlaneNormalAntiParallelThreshold. */
  SCALAR_T onePlusCosine = SCALAR_T(0.0);
  /** s = sqrt(2 (1 + v.n)) outside the guard; sqrt(2) inside it, where only the
   * fallback is selected. */
  SCALAR_T root = SCALAR_T(0.0);
};

/** The PlaneTilt of the attitude `rotmatEeRtWorld` (end effector to world)
 * against the unit normal `normalSurfaceEwrtWorld`. */
template <typename SCALAR_T>
PlaneTilt<SCALAR_T> planeTiltOf(
    const MATRIX3_T<SCALAR_T>& rotmatEeRtWorld,
    const VECTOR3_T<SCALAR_T>& normalSurfaceEwrtWorld) {
  using std::sqrt;
  PlaneTilt<SCALAR_T> planeTilt;
  planeTilt.zAxisEeEwrtWorld = rotmatEeRtWorld.col(kZAxisIndex);
  planeTilt.vCrossN = planeTilt.zAxisEeEwrtWorld.cross(normalSurfaceEwrtWorld);
  planeTilt.onePlusCosine =
      SCALAR_T(1.0) + planeTilt.zAxisEeEwrtWorld.dot(normalSurfaceEwrtWorld);
  // The guard is applied before the square root and the divisions: the branch
  // selected away sees 1 + v.n = 1.
  const SCALAR_T guarded = ifLessThan(
      planeTilt.onePlusCosine, SCALAR_T(kPlaneNormalAntiParallelThreshold),
      SCALAR_T(1.0), planeTilt.onePlusCosine);
  planeTilt.root = sqrt(SCALAR_T(kSquaredRootPerOnePlusCosine) * guarded);
  return planeTilt;
}

/** `fallback` where 1 + v.n < kPlaneNormalAntiParallelThreshold,
 * `regular` elsewhere, entry by entry. */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> selectOutsideAntiParallelGuard(
    const SCALAR_T& onePlusCosine,
    const VECTOR3_T<SCALAR_T>& fallback,
    const VECTOR3_T<SCALAR_T>& regular) {
  VECTOR3_T<SCALAR_T> selected;
  for (Eigen::Index i = 0; i < kRotationTangentDim; ++i) {
    selected(i) =
        ifLessThan(onePlusCosine, SCALAR_T(kPlaneNormalAntiParallelThreshold),
                   fallback(i), regular(i));
  }
  return selected;
}

}  // namespace internal

/**
 * Returns de/dt + 1/2 n (v.w) of the file comment: the rate of the orientation
 * error with respect to the plane of the unit normal `normalSurfaceEwrtWorld`,
 * completed along the normal by half the rotation rate about the end effector's
 * z axis; w / 2 where 1 + v.n < kPlaneNormalAntiParallelThreshold.
 * `rotmatEeRtWorld` maps the end-effector frame to the world frame and
 * `angvelEeRtWorldEwrtWorld` is the end effector's, in the world frame.
 */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> planeTiltRate(
    const MATRIX3_T<SCALAR_T>& rotmatEeRtWorld,
    const VECTOR3_T<SCALAR_T>& angvelEeRtWorldEwrtWorld,
    const VECTOR3_T<SCALAR_T>& normalSurfaceEwrtWorld) {
  const internal::PlaneTilt<SCALAR_T> planeTilt =
      internal::planeTiltOf(rotmatEeRtWorld, normalSurfaceEwrtWorld);
  const VECTOR3_T<SCALAR_T> rateZAxisEeEwrtWorld =
      angvelEeRtWorldEwrtWorld.cross(planeTilt.zAxisEeEwrtWorld);
  const VECTOR3_T<SCALAR_T> rateVCrossN =
      rateZAxisEeEwrtWorld.cross(normalSurfaceEwrtWorld);
  const SCALAR_T cosineRate = rateZAxisEeEwrtWorld.dot(normalSurfaceEwrtWorld);
  const SCALAR_T squaredRoot = planeTilt.root * planeTilt.root;
  const VECTOR3_T<SCALAR_T> rateTiltEeRtSurfaceEwrtWorld =
      (planeTilt.vCrossN * (cosineRate / squaredRoot) - rateVCrossN) /
      planeTilt.root;
  const SCALAR_T angvelEeRtWorldEwrtEeZ =
      planeTilt.zAxisEeEwrtWorld.dot(angvelEeRtWorldEwrtWorld);
  const VECTOR3_T<SCALAR_T> regular =
      rateTiltEeRtSurfaceEwrtWorld +
      normalSurfaceEwrtWorld *
          (SCALAR_T(kUnitQuaternionHalfAngleFactor) * angvelEeRtWorldEwrtEeZ);
  const VECTOR3_T<SCALAR_T> fallback =
      angvelEeRtWorldEwrtWorld * SCALAR_T(kUnitQuaternionHalfAngleFactor);
  return internal::selectOutsideAntiParallelGuard(planeTilt.onePlusCosine,
                                                  fallback, regular);
}

/**
 * Returns the time derivative of planeTiltRate() along the
 * motion, d2e/dt2 + 1/2 n (v.alpha) of the file comment; alpha / 2 where 1 +
 * v.n < kPlaneNormalAntiParallelThreshold. `angaccEeRtWorldEwrtWorld`
 * is the time derivative of `angvelEeRtWorldEwrtWorld`, in the world frame (the
 * angular part of Pinocchio's frame acceleration, LOCAL_WORLD_ALIGNED).
 */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> planeTiltSecondRate(
    const MATRIX3_T<SCALAR_T>& rotmatEeRtWorld,
    const VECTOR3_T<SCALAR_T>& angvelEeRtWorldEwrtWorld,
    const VECTOR3_T<SCALAR_T>& angaccEeRtWorldEwrtWorld,
    const VECTOR3_T<SCALAR_T>& normalSurfaceEwrtWorld) {
  const internal::PlaneTilt<SCALAR_T> planeTilt =
      internal::planeTiltOf(rotmatEeRtWorld, normalSurfaceEwrtWorld);
  const VECTOR3_T<SCALAR_T> rateZAxisEeEwrtWorld =
      angvelEeRtWorldEwrtWorld.cross(planeTilt.zAxisEeEwrtWorld);
  const VECTOR3_T<SCALAR_T> secondRateZAxisEeEwrtWorld =
      angaccEeRtWorldEwrtWorld.cross(planeTilt.zAxisEeEwrtWorld) +
      angvelEeRtWorldEwrtWorld.cross(rateZAxisEeEwrtWorld);
  const VECTOR3_T<SCALAR_T> rateVCrossN =
      rateZAxisEeEwrtWorld.cross(normalSurfaceEwrtWorld);
  const VECTOR3_T<SCALAR_T> secondRateVCrossN =
      secondRateZAxisEeEwrtWorld.cross(normalSurfaceEwrtWorld);
  const SCALAR_T cosineRate = rateZAxisEeEwrtWorld.dot(normalSurfaceEwrtWorld);
  const SCALAR_T cosineSecondRate =
      secondRateZAxisEeEwrtWorld.dot(normalSurfaceEwrtWorld);
  const SCALAR_T squaredRoot = planeTilt.root * planeTilt.root;
  const VECTOR3_T<SCALAR_T> secondRateTiltEeRtSurfaceEwrtWorld =
      ((rateVCrossN *
            (SCALAR_T(internal::kAxisRateCosineRateCoefficient) * cosineRate) +
        planeTilt.vCrossN * cosineSecondRate) /
           squaredRoot -
       planeTilt.vCrossN *
           (SCALAR_T(internal::kSquaredCosineRateCoefficient) * cosineRate *
            cosineRate / (squaredRoot * squaredRoot)) -
       secondRateVCrossN) /
      planeTilt.root;
  const SCALAR_T angaccEeRtWorldEwrtEeZ =
      planeTilt.zAxisEeEwrtWorld.dot(angaccEeRtWorldEwrtWorld);
  const VECTOR3_T<SCALAR_T> regular =
      secondRateTiltEeRtSurfaceEwrtWorld +
      normalSurfaceEwrtWorld *
          (SCALAR_T(kUnitQuaternionHalfAngleFactor) * angaccEeRtWorldEwrtEeZ);
  const VECTOR3_T<SCALAR_T> fallback =
      angaccEeRtWorldEwrtWorld * SCALAR_T(kUnitQuaternionHalfAngleFactor);
  return internal::selectOutsideAntiParallelGuard(planeTilt.onePlusCosine,
                                                  fallback, regular);
}

}  // namespace ocs2::legged
