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
#include <optional>
#include <vector>

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"

namespace ocs2::humanoid {

/**
 * A run of the Jacobian dr/dx of a tuning-layout residual that is the identity: the `size` residual rows from `tuningRow`
 * are the `size` state columns from `stateColumn`, entry for entry.
 */
struct TuningJacobianIdentityRun {
  size_t tuningRow = 0;
  size_t stateColumn = 0;
  size_t size = 0;
};

/** A dense block of dr/dx: the residual rows from `tuningRow` by the state columns from `stateColumn`, `value` in size. */
struct TuningJacobianDenseBlock {
  size_t tuningRow = 0;
  size_t stateColumn = 0;
  matrix_t value;
};

/**
 * The tuning-layout residual r(x, x_ref) at one state and its Jacobian J = dr/dx, block-sparse: J is the sum of its
 * identity runs and dense blocks, every other entry zero. Two blocks may share state columns (the whole-body angular
 * velocity rows depend on the orientation and the angular-velocity slots), never residual rows of an identity run.
 * TuningLayoutDeviation::linearize() fills the dense blocks in this order: the orientation rows' block (at the
 * orientation slots), then, on the whole body, the angular-velocity rows' block at the angular-velocity slots and their
 * block at the orientation slots.
 */
struct TuningDeviationApproximation {
  /** r, getTuningDim() rows: one per row of the task file's state_weights. */
  vector_t residual;
  std::vector<TuningJacobianIdentityRun> identityRuns;
  std::vector<TuningJacobianDenseBlock> denseBlocks;
};

/**
 * The deviation of a state from its reference in the rows of the tuning layout (design section 3 of
 * humanoid_nmpc/docs/quaternion_base_orientation/README.md and its section 2.8.2): the one evaluation of the
 * Euler-written weights' rows, which every quadratic state cost weighs with the task file's state_weights or
 * final_state_weights, at start-up and after a live reload alike. It walks the robot model's getTuningLayout(), block by
 * block:
 *   - kEuclidean (momentum, positions, joints, linear and joint velocities): the difference x - x_ref of the block's
 *     state slots; its Jacobian an identity run.
 *   - kEulerZyxOrientation, rows (yaw, pitch, roll): the heading-tilt residual e_HT(xi, xi_r) of the base orientations
 *     xi = getBaseOrientation(x) and xi_r = getBaseOrientation(x_ref) (orientation/BaseOrientation.h, design section
 *     2.8.1): the twist heading error wrapped to (-pi, pi] and the tilt error in the heading frame. Its Jacobian is the
 *     dense block J_HT(xi) getBaseRotationTangentJacobian(x) at the orientation slots.
 *   - kZyxOrderedAngularVelocity, the whole body's rows (z, y, x) (design decision D8):
 *     e_w = P_3 (w_B - R(xi)^T R(xi_r) w_B,r) of the body angular velocities w_B = getBaseAngularVelocityLocal(x) and
 *     w_B,r = getBaseAngularVelocityLocal(x_ref), P_3 = antidiag(1, 1, 1). Its Jacobian is two dense blocks:
 *     P_3 dw_B/d(angular-velocity slots) at those slots, and P_3 (dw_B/d(orientation slots) - [a]x
 *     getBaseRotationTangentJacobian(x)) at the orientation slots, with a = R(xi)^T R(xi_r) w_B,r
 *     (getBaseAngularVelocityLocalJacobian()).
 * It reads the orientation only through those four robot-model accessors; the slots hold xi and w_B, and the orientation
 * blocks are 3 x 4. The residual is invariant under xi -> -xi, except the orientation rows' tilt at an exactly
 * upside-down attitude (orientation/BaseOrientation.h). The 3 x 4 blocks are not: each is a tangent Jacobian times
 * E+(xi), which changes sign with xi, as the ambient derivative of a function even in xi must; pulled back to the
 * tangent (times E(xi)) they are invariant, and so are a cost's value and Gauss-Newton model in tangent coordinates.
 *
 * How far this is from the Euler formulation's differences x - x_ref, read through the Euler chart (design sections
 * 2.8.1 and 4.4), with the Jacobians below taken in the Euler coordinates:
 *   - The orientation rows' Jacobian is the identity at a level state (zero pitch and roll), whatever the reference
 *     (J_HT T_B(0) = P_3 P_3 = I), so their Gauss-Newton Hessian is Q there; at a tilt |tilt| of the state it deviates
 *     from the identity by about |tilt| / 2. Their value and gradient there equal the Euler differences' when the
 *     reference is tilted about at most one axis (and psi - psi_r is in (-pi, pi]: the yaw row is wrapped, the Euler
 *     difference is not); against a mixed tilt (theta_r, phi_r) the yaw row is shifted by the
 *     twist-vs-ZYX identity 2 atan(tan(theta_r / 2) tan(phi_r / 2)) and the tilt rows differ at third order.
 *   - The whole body's angular-velocity rows have the identity on the rate slots at a level state, and on the orientation
 *     slots the block P_3 (dw_B/dTheta - [a]x T_B(Theta)). For a reference whose Euler rate is a world yaw rate,
 *     dTheta_r = (yawRate, 0, 0) (every shipped reference), e_w = P_3 T_B(Theta) v exactly, v = dTheta - dTheta_r the
 *     Euler-rate error, at any attitude; the block is P_3 (dT_B/dTheta) v, zero at any attitude where v = 0, and at a
 *     level state [[0, 0, -v_pitch], [0, 0, v_yaw], [0, -v_yaw, 0]]. So J is the identity at a level state only when
 *     the yaw and pitch rate errors vanish (the roll-rate error does not enter the block): a state at rest against a
 *     reference turning at yawRate has the block [[0, 0, 0], [0, 0, -yawRate], [0, yawRate, 0]], and with the G1-WB
 *     weights (3, 3, 3) at 0.5 rad/s J' Q J - Q is +0.75 on the pitch and roll diagonal and couples the pitch angle to the
 *     roll rate by +1.5 and the roll angle to the pitch rate by -1.5.
 *
 * Use: a cost holds one by value, built from the robot model it evaluates, and calls evaluate() for a value and
 * linearize() for a Gauss-Newton approximation (cost/TuningGaussNewton.h). A copyable, immutable value type that keeps a
 * pointer to the robot model, which must outlive it and every copy. Its const functions are safe to call concurrently.
 */
class TuningLayoutDeviation {
 public:
  /**
   * The deviation of `mpcRobotModel`'s getTuningLayout(), which is retained. The layout is checked here, once, because
   * a malformed one is a defect of the robot model: every kEuclidean block has as many state rows as tuning rows, the
   * rotation blocks have 3 tuning rows, there is at most one block of each rotation kind, and an angular-velocity block
   * comes with an orientation block.
   */
  explicit TuningLayoutDeviation(const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel);

  TuningLayoutDeviation(const TuningLayoutDeviation& other) = default;
  TuningLayoutDeviation& operator=(const TuningLayoutDeviation& other) = default;
  TuningLayoutDeviation(TuningLayoutDeviation&& other) noexcept = default;
  TuningLayoutDeviation& operator=(TuningLayoutDeviation&& other) noexcept = default;
  ~TuningLayoutDeviation() = default;

  /** Rows of the residual: the tuning layout's, the model's getStateTangentDim(). */
  size_t getTuningDim() const { return tuningDim_; }

  /** Entries of the state the residual is a function of: the model's getStateDim(). */
  size_t getStateDim() const { return stateDim_; }

  /**
   * The residual r(state, stateReference), getTuningDim() rows. Both vectors have getStateDim() entries (a
   * precondition, unchecked: the solver calls this at every node). Bit for bit the residual of linearize(). Allocates
   * the result.
   */
  vector_t evaluate(const vector_t& state, const vector_t& stateReference) const;

  /**
   * evaluate() and the block-sparse Jacobian dr/dstate at `state`, under the same precondition. The cost of the
   * rotation blocks is a few 3 x 3 (3 x 4) products, independent of the state's size.
   */
  TuningDeviationApproximation linearize(const vector_t& state, const vector_t& stateReference) const;

 private:
  /** Where a rotation block of the tuning layout sits: its first residual row and its first state slot. */
  struct RotationBlock {
    size_t tuningRow = 0;
    size_t stateStart = 0;
  };

  /** Adds `segment`, whose rows start at `tuningRow`, to the identity runs or the rotation blocks; the constructor's. */
  void addSegment(const TuningSegment& segment, size_t tuningRow);

  /** The residual rows of the identity runs, x - x_ref; the rows of the rotation blocks are left for the caller. */
  vector_t differences(const vector_t& state, const vector_t& stateReference) const;

  const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel_;
  size_t tuningDim_ = 0;
  size_t stateDim_ = 0;
  // The identity runs of dr/dx, one per run of consecutive kEuclidean blocks whose tuning rows and state slots both
  // follow on from the previous block's: on the centroidal layout two, the momentum and base position, and the joints.
  std::vector<TuningJacobianIdentityRun> identityRuns_;
  // The kEulerZyxOrientation block; nullopt only for a layout without one, which no humanoid robot model has.
  std::optional<RotationBlock> orientation_;
  // The kZyxOrderedAngularVelocity block of the whole body; nullopt on the centroidal layout. Set only with orientation_.
  std::optional<RotationBlock> angularVelocity_;
};

}  // namespace ocs2::humanoid
