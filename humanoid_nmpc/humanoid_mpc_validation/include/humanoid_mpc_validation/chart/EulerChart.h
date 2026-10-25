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
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"

/*
 * The chart between the ZYX Euler coordinates of a humanoid state and the
 * physical state of the quaternion formulation
 * (humanoid_nmpc/docs/quaternion_base_orientation/README.md, section 2.12):
 * what the equivalence suites (humanoid_common_mpc/test/equivalence) map the
 * Euler oracle's states and derivatives through.
 *
 *   Phi:  identity on momentum, positions, joint angles, linear and joint
 * velocities; xi = quaternionFromEulerZyx(Theta) on the base angles; whole body
 * w_B = T_B(Theta) dTheta/dt on the rates. dPhi / dx_e: identity, 1/2 G(xi)
 * T_B(Theta) on the angles, [d(T_B dTheta) / dTheta, T_B(Theta)] on the rates.
 *   P_e:  the same in the physical tangent (the body rotation delta of xi
 * instead of xi): T_B(Theta) on the angles.
 *
 * Every conversion goes through orientation/EulerBoundary.h, the one Euler
 * boundary of the MPCs.
 */

namespace ocs2::humanoid::validation {

/** What one block of an Euler-layout state is. */
enum class EulerStateSegmentKind {
  /** The same quantity in both coordinates, row for row: momentum, positions,
     joint angles, linear and joint velocities. */
  kEuclidean,
  /** The ZYX Euler angles (yaw, pitch, roll) of the base; the unit quaternion
     xi (x, y, z, w) in the physical state. */
  kEulerZyxAngles,
  /**
   * The rates of the ZYX Euler angles (yaw rate, pitch rate, roll rate) of the
   * whole-body base; the body angular velocity w_B (x, y, z) = T_B(Theta)
   * dTheta/dt in the physical state. Theta is the kEulerZyxAngles block before
   * it.
   */
  kEulerZyxRates,
};

/** One block of an EulerStateLayout. Passive data. */
struct EulerStateSegment {
  /** The task file's name of the block (base_orientation, joint_positions,
   * ...), for messages and labels. */
  std::string name;
  EulerStateSegmentKind kind = EulerStateSegmentKind::kEuclidean;
  /** Rows of the block in the Euler state: any size for kEuclidean, 3 for the
   * two rotation kinds. */
  size_t dim = 0;
};

/**
 * An Euler-layout state as consecutive blocks: the tuning layout of the task
 * file's initial_state row for row (StateInputLayout), which is the state of
 * the Euler oracle. Passive data.
 */
struct EulerStateLayout {
  /** "centroidal" or "whole_body", for messages. */
  std::string name;
  std::vector<EulerStateSegment> segments;
};

/**
 * The Euler state of the MPC `layout` describes: its tuning layout
 * (tuningLayoutOf()) block for block, the kEulerZyxOrientation block as
 * kEulerZyxAngles and the kZyxOrderedAngularVelocity block as kEulerZyxRates:
 *   centroidal   normalized_linear_momentum (3), normalized_angular_momentum
 * (3), base_position (3), base_orientation (3, angles), joint_positions (n)
 *   whole body   base_position (3), base_orientation (3, angles),
 * joint_positions (n), base_linear_velocity (3), base_angular_velocity (3,
 * rates), joint_velocities (n) testEulerChart pins the blocks against
 * stateValuesFromConfig(), which places the task file's values.
 */
EulerStateLayout eulerStateLayoutOf(const StateInputLayout& layout);

/** Rows of an Euler state of `layout`: the sum of its blocks. */
size_t getEulerStateDim(const EulerStateLayout& layout);

/** Rows of the physical state Phi maps an Euler state of `layout` to: one more
 * for each kEulerZyxAngles block. */
size_t getPhysicalStateDim(const EulerStateLayout& layout);

/**
 * Phi(x_e): the physical state of the Euler state `eulerState`. InvalidArgument
 * when the state does not have getEulerStateDim(layout) rows, a rotation block
 * does not have 3 rows, or a kEulerZyxRates block has no kEulerZyxAngles block
 * before it.
 */
absl::StatusOr<vector_t> physicalStateFromEulerState(
    const vector_t& eulerState, const EulerStateLayout& layout);

/**
 * Phi^-1(x): the Euler state of the physical state `physicalState`
 * (getPhysicalStateDim(layout) rows), the Euler view of a state of the
 * quaternion formulation. Identity on the Euclidean blocks; Theta =
 * eulerZyxFromQuaternion(xi / |xi|) on the angles, with yaw and roll in (-pi,
 * pi] and pitch in [-pi/2, pi/2] (so Phi^-1(Phi(x_e)) is x_e with its yaw and
 * roll wrapped); dTheta/dt = T_B(Theta)^-1 w_B on the whole-body rates, which
 * divides by cos(pitch) and is not finite at gimbal lock. InvalidArgument when
 * the state does not have getPhysicalStateDim(layout) rows, a quaternion block
 * is zero or not finite, or the layout is malformed (as
 * physicalStateFromEulerState()).
 */
absl::StatusOr<vector_t> eulerStateFromPhysicalState(
    const vector_t& physicalState, const EulerStateLayout& layout);

/**
 * dPhi / dx_e at `eulerState`, getPhysicalStateDim(layout) x
 * getEulerStateDim(layout): identity on the Euclidean blocks, d xi / d Theta =
 * 1/2 G(xi) T_B(Theta) on the angles, and on the rates d w_B / d Theta and d
 * w_B / d dTheta = T_B(Theta). Fails like physicalStateFromEulerState().
 */
absl::StatusOr<matrix_t> eulerChartJacobian(const vector_t& eulerState,
                                            const EulerStateLayout& layout);

/**
 * P_e(x_e), the physical-tangent Jacobian in Euler coordinates (design
 * sections 2.12 and 4.4), square of getEulerStateDim(layout): the change of the
 * physical state in its tangent coordinates - the body rotation delta of xi on
 * the angles, d xi = 1/2 G(xi) delta - per change of the Euler state. Identity
 * on the Euclidean blocks, T_B(Theta) on the angles, and on the rates d w_B / d
 * Theta and T_B(Theta). A term's Jacobian J_t in the tangent then gives the
 * Euler one as J_t P_e, a Gauss-Newton Hessian as P_e' H_t P_e. Fails like
 * physicalStateFromEulerState().
 */
absl::StatusOr<matrix_t> physicalTangentJacobian(
    const vector_t& eulerState, const EulerStateLayout& layout);

/**
 * d (T_B(Theta) dTheta) / d Theta at the angles `eulerZyx` and the rates
 * `eulerZyxRates` (3 x 3, columns yaw, pitch, roll): the coupling block of the
 * whole-body rates in eulerChartJacobian() and physicalTangentJacobian(). It is
 * EulerBoundary's localAngularVelocityEulerZyxJacobian(), bit for bit.
 */
matrix3_t localAngularVelocityEulerAngleDerivative(
    const vector3_t& eulerZyx, const vector3_t& eulerZyxRates);

}  // namespace ocs2::humanoid::validation
