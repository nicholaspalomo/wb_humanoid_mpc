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

#include "humanoid_common_mpc/common/Types.h"

/*
 * The Euler boundary: the only place in the humanoid MPCs that converts between ZYX Euler angles and the quaternion
 * base orientation (design section 4.2 of humanoid_nmpc/docs/quaternion_base_orientation/README.md). The formulation
 * never sees an Euler angle; the human-facing inputs and outputs keep them: the task.yaml tuning layout (initialState,
 * Q, Q_final), velocity and pose commands, the locomotion-heuristic seam, telemetry and display.
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
inline constexpr scalar_t kGimbalLockCosPitch = 1e-3;

/**
 * Theta = (psi, theta, phi) of a unit quaternion xi, with psi, phi in (-pi, pi] and theta in [-pi/2, pi/2]:
 *
 *   psi = atan2(2 (w z + x y), 1 - 2 (y^2 + z^2)),  theta = asin(clamp(2 (w y - z x), -1, 1)),
 *   phi = atan2(2 (w x + y z), 1 - 2 (x^2 + y^2)).
 *
 * The argument of asin is clamped, so a quaternion whose 2 (w y - z x) exceeds 1 by round-off gives theta = +-pi/2 and
 * not NaN. Away from a pitch of +-90 degrees this is, bit for bit, the conversion the MRTs and the telemetry used before
 * the switch (pinocchio_model/DynamicsHelperFunctions.h quaternionToEulerZYX, with the clamp), so a call site moved to it
 * reproduces its old output exactly. xi is read as given, not renormalized, for that reason: pass a unit quaternion.
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

/******************************************************************************************************/
/* The Euler tuning layout (design sections 2.3 and 2.10)                                             */
/******************************************************************************************************/

/**
 * What one block of the Euler tuning layout - a row block of task.yaml's initialState, Q and Q_final - is in the state.
 * The tuning layout has the size of the state tangent (design decision D6), which is the Euler state of before the
 * switch, index for index; the state stores the base orientation as a quaternion, one row more.
 */
enum class TuningSegmentKind {
  /** The same quantity in both, row for row: momentum, positions, joint angles, linear and joint velocities. */
  kEuclidean,
  /** The base orientation: (yaw, pitch, roll) ZYX Euler angles in the tuning layout, xi = (x, y, z, w) in the state. */
  kEulerZyxOrientation,
  /**
   * The whole-body base angular velocity: (w_z, w_y, w_x) in the tuning layout, the order of the Euler rates whose rows
   * it inherits, and w_B = (w_x, w_y, w_z) in the body frame in the state (design decision D18).
   */
  kZyxOrderedAngularVelocity,
};

/** One block of a TuningLayout. */
struct TuningSegment {
  /** Names the block in error messages, e.g. "q_j". */
  std::string name;
  TuningSegmentKind kind = TuningSegmentKind::kEuclidean;
  /** Rows in the tuning layout: any size for kEuclidean, 3 for the two rotation kinds. */
  size_t tuningDim = 0;
};

/**
 * A tuning layout and the state layout it maps to, as consecutive blocks. The robot models build theirs (design
 * section 2.3), for example the centroidal [h(6), p_W(3), base orientation, q_j(nj)]; a layout of kEuclidean blocks only
 * maps a vector to itself, the Euler state of before the switch.
 */
struct TuningLayout {
  /** Names the layout in error messages, e.g. "centroidal". */
  std::string name;
  std::vector<TuningSegment> segments;
};

/** Rows of the tuning layout: the sum of the segments' tuningDim. */
size_t getTuningLayoutDim(const TuningLayout& layout);

/** Rows of the state the tuning layout maps to: one more than the tuning rows for each kEulerZyxOrientation segment. */
size_t getTuningLayoutStateDim(const TuningLayout& layout);

/**
 * The layout as text for an error message, e.g.
 * "centroidal tuning layout (35 rows: h 6, p_W 3, base_orientation 3 (yaw, pitch, roll), q_j 23)".
 */
std::string describeTuningLayout(const TuningLayout& layout);

/**
 * The state of a tuning-layout vector, e.g. task.yaml's initialState: kEuclidean rows are copied bit for bit, (yaw,
 * pitch, roll) become quaternionFromEulerZyx, and (w_z, w_y, w_x) become (w_x, w_y, w_z). An InvalidArgumentError,
 * naming the layout with each block and its size, when the vector does not have getTuningLayoutDim(layout) entries or a
 * rotation block does not have 3 rows.
 */
absl::StatusOr<vector_t> stateFromTuningLayout(const vector_t& tuningVector, const TuningLayout& layout);

/**
 * The inverse of stateFromTuningLayout, for display and for writing a state back in the tuning layout: the quaternion
 * becomes eulerZyxFromQuaternion of its safe normalization. An exact inverse for yaw and roll in (-pi, pi] and a pitch
 * of less than 90 degrees. An InvalidArgumentError, naming the layout, when the state does not have
 * getTuningLayoutStateDim(layout) entries or a rotation block does not have 3 rows.
 */
absl::StatusOr<vector_t> tuningLayoutFromState(const vector_t& state, const TuningLayout& layout);

}  // namespace ocs2::humanoid
