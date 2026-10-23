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

/*
 * The dimensions and coefficient indices of three-dimensional geometry, shared by the OCS2 fork and the first-party code
 * in place of the literals 3, 4 and 6 (AGENTS.md, C++ "Magic numbers"). Only meanings that hold everywhere in the
 * repository live here; a dimension of one model or one formulation (a state layout, the number of feet) belongs to the
 * code that defines it.
 *
 * Every constant is an `int`, the type of the literal it replaces, so that arithmetic, overload resolution and template
 * arguments (`segment<kPositionDim>()`, `Eigen::Matrix<T, kPositionDim, 1>`, `std::array<T, kPositionDim>`) are what
 * they were with the literal. The header has no dependencies (//lib/ocs2:ocs2_dimensions), so code that uses neither
 * Eigen nor OCS2, such as the robot runtime, may include it too.
 */

namespace ocs2 {

/** A position in space, (x, y, z), and with it a linear velocity or acceleration and a rotation matrix's side. */
inline constexpr int kPositionDim = 3;

/** The horizontal components (x, y) of a position: a point of the ground plane. */
inline constexpr int kHorizontalPositionDim = 2;

/** The index of each axis in a position, a linear velocity or a force; z points up, against gravity. */
inline constexpr int kXAxisIndex = 0;
inline constexpr int kYAxisIndex = 1;
inline constexpr int kZAxisIndex = 2;

/** The tangent space of a rotation, so(3): an angular velocity, a rotation vector or an orientation error. */
inline constexpr int kRotationTangentDim = 3;

/**
 * The coefficients of a unit quaternion. The indices below are those of the storage order (x, y, z, w) of Eigen's
 * coeffs(), of Pinocchio's configuration and of the MPC's state (humanoid_nmpc/docs/quaternion_base_orientation/
 * README.md, section 2.1); MuJoCo stores (w, x, y, z), which they do not describe.
 */
inline constexpr int kQuaternionCoefficients = 4;
inline constexpr int kQuaternionXIndex = 0;
inline constexpr int kQuaternionYIndex = 1;
inline constexpr int kQuaternionZIndex = 2;
inline constexpr int kQuaternionWIndex = 3;

/** A force and a torque (a moment), each a vector of space. */
inline constexpr int kForceDim = 3;
inline constexpr int kTorqueDim = 3;

/** A wrench: a force and a torque. Which comes first is the layout's to say. */
inline constexpr int kWrenchDim = kForceDim + kTorqueDim;

/** A twist: a linear and an angular velocity. Which comes first is the layout's to say. */
inline constexpr int kTwistDim = kPositionDim + kRotationTangentDim;

/** A pose: its coefficients, the position and the quaternion, and its tangent, the position and the rotation tangent. */
inline constexpr int kPoseCoefficients = kPositionDim + kQuaternionCoefficients;
inline constexpr int kPoseTangentDim = kPositionDim + kRotationTangentDim;

}  // namespace ocs2
