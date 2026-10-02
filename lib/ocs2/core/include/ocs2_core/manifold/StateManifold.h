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

#include <ocs2_core/Types.h>

namespace ocs2 {

/**
 * The manifold a state lives on, for solvers that work in its tangent space.
 *
 * A state is stored in ambient coordinates x (size getAmbientDim(), e.g. a unit quaternion as four coefficients),
 * while every perturbation of it is a tangent vector dx (size getTangentDim(), e.g. a rotation as three numbers): the
 * QP, the Riccati gains, the line search, the shooting gaps and the feedback policy of the SQP work in the tangent, and
 * costs, constraints and dynamics keep returning derivatives with respect to the ambient state, which the solver pulls
 * back once per node with the maps of this class.
 *
 * The tangent at x is parameterized through the retraction x (+) dx = retract(x, dx), with
 *   E(x) = d retract(x, dx) / d dx at dx = 0          (ambient x tangent),
 *   E+(x), a left inverse of E(x) (E+ E = I)           (tangent x ambient).
 * Every function a solver evaluates on the state is expected to be invariant along the directions E does not span (for
 * a unit quaternion: to see the quaternion only through xi / |xi|), so its ambient gradient g satisfies g' x = 0 on the
 * rotation block and the pulled-back Gauss-Newton Hessian E' H E needs no retraction-curvature correction.
 *
 * A solver holds the manifold as std::shared_ptr<const StateManifold>: it is immutable and shared across worker threads,
 * so implementations keep no mutable state. A nullptr manifold means the flat state space, on which solvers keep their
 * original arithmetic exactly.
 *
 * All derivative maps work in place on the blocks that are not Euclidean and never form E as a dense matrix.
 */
class StateManifold {
 public:
  virtual ~StateManifold() = default;

  /** The size of a stored state x. */
  virtual size_t getAmbientDim() const = 0;

  /** The size of a state perturbation dx. */
  virtual size_t getTangentDim() const = 0;

  /** xNew = x (+) (alpha * dx), on the manifold. xNew is resized; it may not alias x. */
  virtual void retract(const vector_t& x, const vector_t& dx, scalar_t alpha, vector_t& xNew) const = 0;

  /**
   * The tangent vector d at x0 with x0 (+) d = x1, along the shortest path (for a rotation, |d| <= pi). It is a function
   * of the rotation x1 represents, so it is invariant under x1 -> -x1 on a quaternion block, and it does not require x1
   * to be normalized: difference(xNext, Phi) is the shooting gap of an ambient flow Phi.
   */
  virtual vector_t difference(const vector_t& x0, const vector_t& x1) const = 0;

  /** The shortest-path geodesic from x0 (alpha = 0) to x1 (alpha = 1); linear on Euclidean blocks. */
  virtual vector_t interpolate(const vector_t& x0, const vector_t& x1, scalar_t alpha) const = 0;

  /** Projects an ambient vector onto the manifold in place (normalizes every quaternion block). */
  virtual void project(vector_t& x) const = 0;

  /** The largest rotation angle [rad] any rotation block of the tangent vector dx describes; 0 without rotations. */
  virtual scalar_t getMaximumRotationAngle(const vector_t& dx) const = 0;

  /** J <- J E(x): the columns of a Jacobian (any rows x ambient) become tangent columns (rows x tangent). */
  virtual void pullBackStateColumns(const vector_t& x, matrix_t& J) const = 0;

  /** M <- E(x)' M: the rows of a matrix (ambient x any) become tangent rows (tangent x any). */
  virtual void pullBackStateRows(const vector_t& x, matrix_t& M) const = 0;

  /** g <- E(x)' g: an ambient gradient becomes a tangent gradient. */
  virtual void pullBackGradient(const vector_t& x, vector_t& g) const = 0;

  /** H <- E(x)' H E(x): an ambient (Gauss-Newton) Hessian becomes a tangent Hessian. */
  void pullBackHessian(const vector_t& x, matrix_t& H) const {
    pullBackStateColumns(x, H);
    pullBackStateRows(x, H);
  }

  /** J <- J E+(x): a Jacobian with tangent columns (rows x tangent) becomes one with ambient columns (rows x ambient). */
  virtual void liftStateColumns(const vector_t& x, matrix_t& J) const = 0;

  /**
   * Turns the ambient linearization of a discrete flow into the manifold linearization of its shooting gap.
   *
   * On input `dynamics` holds the flow Phi = Phi(x, u) and its sensitivities: f = Phi (ambient, not necessarily on the
   * manifold), dfdx = dPhi/dx (ambient x ambient) and dfdu = dPhi/du (ambient x inputs). On output it holds the
   * linearization of xNext (+) dxNext = Pi(Phi(x (+) dx, u + du)) around dx = du = 0, dxNext = f + dfdx dx + dfdu du:
   *   f    = difference(xNext, Phi)                             (tangent),
   *   dfdx = Jr^-1(f) E+(Pi(Phi)) Pi'(Phi) dPhi/dx E(x)        (tangent x tangent),
   *   dfdu = Jr^-1(f) E+(Pi(Phi)) Pi'(Phi) dPhi/du             (tangent x inputs),
   * with Pi the projection onto the manifold and Jr^-1 the inverse right Jacobian of each rotation block (identity on
   * Euclidean blocks). An empty dfdu (no rows) is left as it is.
   */
  virtual void pushForwardDynamics(const vector_t& x, const vector_t& xNext, VectorFunctionLinearApproximation& dynamics) const = 0;

  /**
   * The same for a jump map at an event: `dynamics` holds the post-event state f = Phi(x) and dfdx = dPhi/dx. On output f
   * is the tangent gap difference(xNext, Phi), dfdx the tangent sensitivity and dfdu the empty (tangent x 0) matrix.
   */
  void pushForwardJump(const vector_t& x, const vector_t& xNext, VectorFunctionLinearApproximation& jump) const {
    jump.dfdu.resize(0, 0);
    pushForwardDynamics(x, xNext, jump);
    jump.dfdu.setZero(getTangentDim(), 0);
  }
};

}  // namespace ocs2
