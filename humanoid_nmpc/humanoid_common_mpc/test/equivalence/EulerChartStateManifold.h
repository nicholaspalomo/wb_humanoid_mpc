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

#include <memory>

#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/StateManifold.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_validation/chart/EulerChart.h"

namespace ocs2::humanoid {

/**
 * The quaternion state space of an MPC seen through the Euler chart Phi (humanoid_mpc_validation/chart/EulerChart.h):
 * the state manifold on which OCS2's SQP performs the iteration of the Euler-coordinate formulation, with the MPC's own
 * terms. A tangent vector is d = P_e(x_e) dx_e, the physical-tangent image of an Euler-coordinate step dx_e at x_e =
 * Phi^-1(x) (the base rotation as T_B(Theta) dTheta, the whole body's angular velocity as d(T_B(Theta) dTheta/dt)):
 *
 *   retract(x, d, alpha)    Phi(x_e + alpha P_e(x_e)^-1 d)                     (an Euler step)
 *   difference(x0, x1)      P_e(x_e0) (x_e1 - x_e0)                            (an Euler difference)
 *   interpolate(x0, x1, a)  Phi((1 - a) x_e0 + a x_e1)
 *   E(x)                    dPhi(x_e) P_e(x_e)^-1, which is the quaternion manifold's E (dPhi = E P_e), so the terms'
 *                           derivatives pull back exactly as on the quaternion manifold;
 *   shooting gap            f = P_e(x_e,next) (Phi^-1(Pi(F)) - x_e,next), and its sensitivities
 *                           P_e(x_e,next) P_e(Phi^-1(Pi(F)))^-1 E+(Pi(F)) Pi'(F) dF/dx E(x): the Euler coordinates' defect,
 *                           where the quaternion manifold has Log(xi_next^-1 (x) Pi(F)) and Jr^-1 of it.
 *
 * So the QP of an iteration on this manifold is the Euler formulation's QP in the variables d = P_e dx_e, node by node
 * (with the MPC's own RK4 flow, O(dt^5) from the Euler RK4), and it differs from the quaternion manifold's QP only in the
 * shooting gaps' chart, whose sensitivity is P_e(x_e,next) P_e(x_e,F)^-1 here and Jr^-1(b) on the rotation rows (the
 * identity elsewhere) there:
 *   - the centroidal state: P_e depends on Theta only, so the two agree where no gap carries a rotation (b = 0), and
 *     differ at first order in b otherwise (T_B(Theta_next) T_B(Theta_F)^-1 and Jr^-1(b) are both I + O(|b|), with
 *     different first-order terms);
 *   - the whole body: P_e's (rotation, angular velocity) block is [[T_B, 0], [C(dTheta/dt), T_B]], with C(dTheta/dt) =
 *     d(T_B(Theta) dTheta/dt) / dTheta linear in the Euler rates (the yaw and pitch rates; the roll rate does not enter).
 *     At Theta_F = Theta_next the chart's sensitivity is [[I, 0], [C(dTheta_next/dt - dTheta_F/dt) T_B^-1, I]], so the
 *     two also differ at first order in a gap of the angular velocity (one with a yaw- or pitch-rate component), without
 *     any rotation in the gap. They agree where no gap carries a rotation and, on the whole body, no gap carries an
 *     angular velocity (testMpcSqpStepEquivalence (S2) and (S2w)).
 *
 * Valid away from the Euler singularity (|pitch| < 90 degrees) and from a yaw of +-pi between the states it relates. A
 * test-only manifold: it allocates and is not for the realtime path. Immutable; its const methods are safe to call
 * concurrently.
 */
class EulerChartStateManifold final : public StateManifold {
 public:
  /** The chart of `layout` on `quaternionManifold`, the MPC's own state manifold (its E and E+ are the chart's). */
  EulerChartStateManifold(std::shared_ptr<const StateManifold> quaternionManifold, validation::EulerStateLayout layout);

  EulerChartStateManifold(const EulerChartStateManifold&) = delete;
  EulerChartStateManifold& operator=(const EulerChartStateManifold&) = delete;
  EulerChartStateManifold(EulerChartStateManifold&&) = delete;
  EulerChartStateManifold& operator=(EulerChartStateManifold&&) = delete;
  ~EulerChartStateManifold() override = default;

  size_t getAmbientDim() const override { return quaternionManifold_->getAmbientDim(); }
  size_t getTangentDim() const override { return quaternionManifold_->getTangentDim(); }
  void retract(const vector_t& x, const vector_t& dx, scalar_t alpha, vector_t& xNew) const override;
  vector_t difference(const vector_t& x0, const vector_t& x1) const override;
  vector_t interpolate(const vector_t& x0, const vector_t& x1, scalar_t alpha) const override;
  void differenceInto(const vector_t& x0, const vector_t& x1, vector_t& dx) const override;
  void interpolateInto(const vector_t& x0, const vector_t& x1, scalar_t alpha, vector_t& x) const override;
  void project(vector_t& x) const override { quaternionManifold_->project(x); }
  scalar_t getMaximumRotationAngle(const vector_t& dx) const override { return quaternionManifold_->getMaximumRotationAngle(dx); }
  void pullBackStateColumns(const vector_t& x, matrix_t& J) const override { quaternionManifold_->pullBackStateColumns(x, J); }
  void pullBackStateRows(const vector_t& x, matrix_t& M) const override { quaternionManifold_->pullBackStateRows(x, M); }
  void pullBackGradient(const vector_t& x, vector_t& g) const override { quaternionManifold_->pullBackGradient(x, g); }
  void liftStateColumns(const vector_t& x, matrix_t& J) const override { quaternionManifold_->liftStateColumns(x, J); }
  void pushForwardDynamics(const vector_t& x, const vector_t& xNext, VectorFunctionLinearApproximation& dynamics) const override;

  /** Phi^-1(x): the Euler state of a quaternion state (its quaternion read through its own direction). */
  vector_t eulerStateOf(const vector_t& x) const;
  /** Phi(x_e). */
  vector_t stateOf(const vector_t& eulerState) const;
  /** P_e(x_e). */
  matrix_t physicalTangentJacobian(const vector_t& eulerState) const;

 private:
  // Shared because OCS2 shares a state manifold (OptimalControlProblem::stateManifoldPtr).
  std::shared_ptr<const StateManifold> quaternionManifold_;
  validation::EulerStateLayout layout_;
};

}  // namespace ocs2::humanoid
