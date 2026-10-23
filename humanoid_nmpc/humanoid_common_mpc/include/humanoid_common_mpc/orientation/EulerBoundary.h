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

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"

/*
 * The Euler boundary: the only place in the humanoid MPCs that converts between ZYX Euler angles and the quaternion
 * base orientation (design section 4.2 of humanoid_nmpc/docs/quaternion_base_orientation/README.md). The formulation
 * never sees an Euler angle; the human-facing inputs and outputs keep them: the tuning layout of the task file
 * (initial_state, state_weights, final_state_weights), velocity and pose commands, the locomotion-heuristic seam,
 * telemetry and display.
 *
 * Conventions: Theta = (psi, theta, phi) = (yaw, pitch, roll) are ZYX Euler angles, R = R_z(psi) R_y(theta) R_x(phi); a
 * quaternion is a vector4_t of coefficients (x, y, z, w), as in orientation/BaseOrientation.h.
 */

namespace ocs2::humanoid {

/******************************************************************************************************/
/* Euler angles and quaternions                                                                       */
/******************************************************************************************************/

/**
 * xi = q_z(psi) (x) q_y(theta) (x) q_x(phi) for Theta = (psi, theta, phi), as coefficients (x, y, z, w). The same
 * quaternion, bit for bit, as ocs2::getQuaternionFromEulerAnglesZyx, which the human-facing code calls today.
 */
vector4_t quaternionFromEulerZyx(const vector3_t& eulerZyx);

/**
 * Below this cos(theta), within 0.06 degrees of a pitch of +-90 degrees, eulerZyxFromQuaternion reads pitch and roll
 * from R_z(psi)^T R. The asin and atan2 formulas lose about 2e-16 / cos(theta) of the rotation, so at this threshold they
 * still reproduce it to about 2e-13, and below it they would lose it entirely at gimbal lock.
 */
inline constexpr scalar_t kGimbalLockCosPitch = 1.0e-3;

/**
 * Theta = (psi, theta, phi) of a unit quaternion xi, with psi, phi in (-pi, pi] and theta in [-pi/2, pi/2]:
 *
 *   psi = atan2(2 (w z + x y), 1 - 2 (y^2 + z^2)),  theta = asin(clamp(2 (w y - z x), -1, 1)),
 *   phi = atan2(2 (w x + y z), 1 - 2 (x^2 + y^2)).
 *
 * The argument of asin is clamped, so a quaternion whose 2 (w y - z x) exceeds 1 by round-off gives theta = +-pi/2 and
 * not NaN. xi is read as given, not renormalized: pass a unit quaternion.
 *
 * At gimbal lock (cos(theta) below kGimbalLockCosPitch) only psi - phi (theta = 90 degrees) or psi + phi (theta = -90
 * degrees) is defined, and the formulas above split it by round-off. There psi keeps its formula, and theta and phi are
 * recomputed from R_z(psi)^T R, so quaternionFromEulerZyx(eulerZyxFromQuaternion(xi)) reproduces the rotation of xi at
 * every attitude, gimbal lock included.
 */
vector3_t eulerZyxFromQuaternion(const vector4_t& xi);

/**
 * T_B(Theta), the 3 x 3 matrix with w_B = T_B(Theta) dTheta/dt that maps ZYX Euler rates to the body angular velocity
 * (design section 2.1), ocs2::getMappingFromEulerAnglesZyxDerivativeToLocalAngularVelocity. T_B(0) = P_3 =
 * antidiag(1, 1, 1); it is singular at cos(theta) = 0. With the quaternion rate matrix G = ocs2::quaternionRateMatrix,
 * d quaternionFromEulerZyx(Theta) / d Theta = 1/2 G(xi) T_B(Theta) (design section 2.12).
 */
matrix3_t eulerZyxRateToLocalAngularVelocityMatrix(const vector3_t& eulerZyx);

/** w_B = T_B(Theta) dTheta/dt, ocs2::getLocalAngularVelocityFromEulerAnglesZyxDerivatives. */
vector3_t localAngularVelocityFromEulerZyxRates(const vector3_t& eulerZyx, const vector3_t& eulerZyxRates);

/**
 * dTheta/dt = T_B(Theta)^-1 w_B, the inverse of localAngularVelocityFromEulerZyxRates, bit for bit
 * ocs2::getEulerAnglesZyxDerivativesFromLocalAngularVelocity. It divides by cos(theta): the rates are not finite at a
 * pitch of +-90 degrees.
 */
vector3_t eulerZyxRatesFromLocalAngularVelocity(const vector3_t& eulerZyx, const vector3_t& localAngularVelocity);

/**
 * d w_B / d Theta of w_B = T_B(Theta) dTheta/dt at fixed rates dTheta/dt: the 3 x 3 matrix whose columns are the
 * derivatives along yaw (zero: T_B does not depend on the yaw), pitch and roll.
 */
matrix3_t localAngularVelocityEulerZyxJacobian(const vector3_t& eulerZyx, const vector3_t& eulerZyxRates);

/******************************************************************************************************/
/* The Euler tuning layout (design sections 2.3 and 2.10)                                             */
/******************************************************************************************************/

/**
 * What one block of the Euler tuning layout - a block of the task file's initial_state, state_weights and
 * final_state_weights - is in the state. The tuning layout has the size of the state tangent (design decision D6): the
 * ZYX Euler state, index for index; the state, which stores the base orientation as a quaternion, has one row more.
 */
enum class TuningSegmentKind {
  /** The same quantity in both, row for row: momentum, positions, joint angles, linear and joint velocities. */
  kEuclidean,
  /**
   * The base orientation: (yaw, pitch, roll) ZYX Euler angles in the tuning layout, the unit quaternion xi = (x, y, z, w)
   * on kQuaternionOrientationStateDim rows in the state.
   */
  kEulerZyxOrientation,
  /**
   * The whole-body base angular velocity: (w_z, w_y, w_x) in the tuning layout, the order of the Euler rates whose rows
   * it inherits, and w_B = (w_x, w_y, w_z) in the body frame in the state (design decision D18).
   */
  kZyxOrderedAngularVelocity,
};

/** The state rows of a kEulerZyxOrientation block, which the state stores as a quaternion (x, y, z, w). */
inline constexpr size_t kQuaternionOrientationStateDim = kQuaternionCoefficients;

/**
 * The tuning rows of a rotation block (not kEuclidean): the three ZYX Euler angles (yaw, pitch, roll) of a
 * kEulerZyxOrientation block, the three components (w_z, w_y, w_x) of a kZyxOrderedAngularVelocity block.
 */
inline constexpr size_t kRotationTuningDim = 3;

/** One block of a TuningLayout. */
struct TuningSegment {
  /** Names the block in error messages and in tuningSegmentStart(), e.g. "joint_positions". */
  std::string name;
  TuningSegmentKind kind = TuningSegmentKind::kEuclidean;
  /** Rows in the tuning layout: any size for kEuclidean, 3 for the two rotation kinds. */
  size_t tuningDim = 0;
  /** The block's first row in the state; 0 until placeTuningLayout() places the layout. */
  size_t stateStart = 0;
  /**
   * Rows in the state: tuningDim, except kEulerZyxOrientation's, which is kQuaternionOrientationStateDim. 0 until
   * placeTuningLayout() places the layout.
   */
  size_t stateDim = 0;
};

/**
 * A tuning layout and the state layout it maps to, as consecutive blocks. The robot models hold theirs, built by
 * tuningLayoutOf() (config/weights/StateInputLayout.h) and placed by placeTuningLayout() (design section 2.3): for
 * example the centroidal [normalized_linear_momentum (3), normalized_angular_momentum (3), base_position (3),
 * base_orientation, joint_positions (nj)]; a layout of kEuclidean blocks only maps a vector to itself.
 */
struct TuningLayout {
  /** Names the layout in error messages, e.g. "centroidal". */
  std::string name;
  std::vector<TuningSegment> segments;
};

/**
 * `layout` placed on a state: each segment's stateStart and stateDim set, the segments consecutive in the state in their
 * tuning order, every one as many state rows as tuning rows but a kEulerZyxOrientation segment, which gets the
 * kQuaternionOrientationStateDim rows of its quaternion.
 */
TuningLayout placeTuningLayout(TuningLayout layout);

/** Rows of the tuning layout: the sum of the segments' tuningDim. */
size_t getTuningLayoutDim(const TuningLayout& layout);

/** Rows of the state the tuning layout is placed on: the sum of the segments' stateDim (placeTuningLayout()). */
size_t getTuningLayoutStateDim(const TuningLayout& layout);

/**
 * The first tuning row of the segment named `name`: where the task file's block of that name sits in the tuning vector
 * and in Q. NotFound, naming the layout and its blocks, when no segment has that name.
 */
absl::StatusOr<size_t> tuningSegmentStart(const TuningLayout& layout, absl::string_view name);

/**
 * The layout as text for an error message, e.g.
 * "centroidal tuning layout (35 rows: h 6, p_W 3, base_orientation 3 (yaw, pitch, roll), q_j 23)".
 */
std::string describeTuningLayout(const TuningLayout& layout);

/**
 * The state of a tuning-layout vector, e.g. the task file's initial_state, on a layout that placeTuningLayout() placed:
 * kEuclidean rows are copied bit for bit, (yaw, pitch, roll) become quaternionFromEulerZyx, and (w_z, w_y, w_x) become
 * (w_x, w_y, w_z). An InvalidArgumentError, naming the layout with each block and its size, when the vector does not
 * have getTuningLayoutDim(layout) entries, a rotation block does not have 3 rows, or the layout is not where
 * placeTuningLayout() places it.
 */
absl::StatusOr<vector_t> stateFromTuningLayout(const vector_t& tuningVector, const TuningLayout& layout);

/**
 * The inverse of stateFromTuningLayout, for display and for writing a state back in the tuning layout: the quaternion
 * becomes eulerZyxFromQuaternion of its safe normalization. An exact inverse for yaw and roll in (-pi, pi] and a pitch
 * of less than 90 degrees. An InvalidArgumentError, naming the layout, when the state does not have
 * getTuningLayoutStateDim(layout) entries, a rotation block does not have 3 rows, or the layout is not where
 * placeTuningLayout() places it (as stateFromTuningLayout).
 */
absl::StatusOr<vector_t> tuningLayoutFromState(const vector_t& state, const TuningLayout& layout);

}  // namespace ocs2::humanoid
