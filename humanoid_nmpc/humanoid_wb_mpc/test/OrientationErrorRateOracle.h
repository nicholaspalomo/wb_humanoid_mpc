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

#include "humanoid_common_mpc/common/Types.h"

/*
 * A numerical oracle for the time derivatives of the stance foot's orientation error, independent of the formulas of
 * humanoid_wb_mpc/end_effector/OrientationErrorWrtPlaneRates.h: an attitude is moved along a curve whose world-frame
 * angular velocity and acceleration at t = 0 are prescribed, and the orientation error of the production definition,
 * e(t) = rotationMatrixDistanceToPlane(R(t), n), is differenced in t. The row along n, where e has no component, carries
 * half the rotation rate about the foot normal, y(t) = 1/2 v(t).w(t), whose value is read at t = 0 and whose rate is
 * differenced in t too.
 */
namespace ocs2::humanoid::test_support {

/** The motion R(t) = Exp(w t + alpha t^2 / 2) R0 of an attitude (rotationAlong()). Passive data. */
struct AttitudeMotion {
  /** R0, from the end-effector frame to the world frame. */
  matrix3_t initialRotation = matrix3_t::Identity();
  /** [rad/s] w, the world-frame angular velocity at t = 0. */
  vector3_t angularVelocity = vector3_t::Zero();
  /** [rad/s^2] alpha, the world-frame angular acceleration at t = 0. */
  vector3_t angularAcceleration = vector3_t::Zero();
};

/** Returns [phi]x, the cross-product matrix of `vector`. */
matrix3_t crossProductMatrix(const vector3_t& vector);

/**
 * Returns the left Jacobian of SO(3) at the rotation vector `rotationVector` (phi, angle |phi|):
 *   J_l(phi) = I + (1 - cos|phi|) / |phi|^2 [phi]x + (|phi| - sin|phi|) / |phi|^3 [phi]x^2,
 * with the coefficients' Taylor series below |phi| = 1e-2, where the closed forms cancel. d/dt Exp(phi(t)) =
 * [J_l(phi) phi']x Exp(phi(t)): J_l maps the rate of a rotation vector to the world-frame angular velocity.
 */
matrix3_t leftJacobianSo3(const vector3_t& rotationVector);

/**
 * Returns R(t) = Exp(phi(t)) R0 with phi(t) = w t + alpha t^2 / 2. Its world-frame angular velocity
 * angularVelocityAlong() is J_l(phi(t)) phi'(t): w at t = 0, and alpha + 1/2 w x w = alpha is its derivative there (the
 * derivative of J_l at 0 is [phi']x / 2), so that the motion has the prescribed angular velocity and acceleration.
 */
matrix3_t rotationAlong(const AttitudeMotion& motion, scalar_t time);

/** Returns the world-frame angular velocity of rotationAlong() at `time`, J_l(phi(t)) phi'(t). */
vector3_t angularVelocityAlong(const AttitudeMotion& motion, scalar_t time);

/**
 * [1/s] Returns the natural rate of `motion` against the unit plane normal `planeNormal`, max(|w|, sqrt(psi |alpha|)) / psi
 * with psi = min(1, sqrt(2 (1 + v.n))), the angle over which e turns (near the anti-parallel normal its direction turns
 * once over psi): the k-th time derivative of e is of the order of its k-th power. Requires 1 + v.n > 0.
 */
scalar_t orientationErrorRateScale(const AttitudeMotion& motion, const vector3_t& planeNormal);

/** What differencedOrientationErrorRates() returns. Passive data. */
struct DifferencedOrientationErrorRates {
  /** de/dt + 1/2 n (v.w) at t = 0, de/dt by differences. */
  vector3_t rate = vector3_t::Zero();
  /** d2e/dt2 + 1/2 n d/dt (v.w) at t = 0, both derivatives by differences. */
  vector3_t acceleration = vector3_t::Zero();
  /** [1/s] orientationErrorRateScale() of the motion; the differences' largest step is 4e-2 / max(1, rateScale). */
  scalar_t rateScale = 0.0;
};

/**
 * Returns the rate and the acceleration of the orientation error of `motion` with respect to the plane of the unit normal
 * `planeNormal`, by central differences of e(t) and y(t) at the steps h, h / 2 and h / 4, h = 4e-2 / max(1, rateScale),
 * extrapolated twice (Richardson: (4 D(h / 2) - D(h)) / 3 and its like at h / 4, then (16 R(h / 4) - R(h / 2)) / 15),
 * so that their truncation is O(h^6). In units of the derivatives' scale max(1, rateScale^k) their error, measured
 * against the closed forms on 6000 random motions per regime of the distributions of testOrientationErrorWrtPlaneRates
 * (pure spins included), is
 *  - at most 6e-12 at tilts up to 80 degrees;
 *  - at most 3e-10 between 80 degrees and 1 + v.n = 1e-2, set by the truncation, which peaks near 1 + v.n = 1e-2;
 *  - at most 7e-10 for 1 + v.n from 1e-2 down to 1e-5, set by the round-off near 1 + v.n = 1e-5: e carries about
 *    eps / psi there (v carries eps, and e turns over psi), which the extrapolated second differences multiply by about
 *    100 / h^2, 100 eps / (psi 4e-2^2) = 1.5e-9 of the scale at 1 + v.n = 1e-5 (psi = 4.5e-3) where the round-offs add.
 * One extrapolation at h = 1e-2, as this oracle had, leaves a truncation of 5e-9 near 1 + v.n = 3e-3, and at h = 5e-3 a
 * round-off of 1e-8 at 1 + v.n = 1e-5. Requires 1 + v.n > 0: the normal turns by about 4e-2 psi over the step, so that
 * 1 + v.n changes by less than a tenth along the samples and e is smooth there.
 */
DifferencedOrientationErrorRates differencedOrientationErrorRates(const AttitudeMotion& motion, const vector3_t& planeNormal);

/**
 * The difference allowed between a rate and the oracle's, relative to max(1, rateScale) for the rate and to
 * max(1, rateScale^2) for the acceleration: fourteen times the oracle's largest error (7e-10, near the anti-parallel
 * normal), and more than a thousand times its error at a tilt (differencedOrientationErrorRates()).
 */
inline constexpr scalar_t kDifferencedRateTolerance = 1.0e-8;

}  // namespace ocs2::humanoid::test_support
