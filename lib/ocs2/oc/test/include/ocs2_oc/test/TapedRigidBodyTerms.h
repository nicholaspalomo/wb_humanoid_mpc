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

#include "absl/base/nullability.h"

#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_core/cost/StateCost.h>
#include <ocs2_core/cost/StateInputCost.h>
#include <ocs2_core/manifold/UnitQuaternionMath.h>

#include "ocs2_oc/test/TapedFunction.h"

namespace ocs2 {
namespace manifold_test {

/*
 * Nonlinear terms on the rigid body with position of RigidBodyAttitudeDynamics (x = [p, xi, omega], u = [v, tau]), each a
 * CppAD tape with exact derivatives. Every term sees the quaternion through xi / |xi| (the manifold contract), and
 * mixes it with the Euclidean entries, so the pull-backs are exercised on every block. These live in one header because
 * they are test fixtures, not terms of a formulation.
 */

/** The rotation matrix of the body in the world frame; its column k is the body's k axis written in world axes. */
inline Eigen::Matrix<TapedFunction::ad_t, 3, 3> getRotmatBaseRtWorld(const TapedFunction::ad_vector_t& x) {
  const quaternion_coeffs_t<TapedFunction::ad_t> quatBaseRtWorldXyzw = x.template segment<4>(3);
  return quaternionRotationMatrix(quaternionSafeNormalize(quatBaseRtWorldXyzw));
}

/** l(x, u) = 1/2 |R z - (0.1, 0.2, 0.97)|^2 + (R x)_y p_x + 1/2 |omega - (0.1, 0, 0.3)|^2 + 1/2 u' diag(1..6) u + omega_z u_0. */
class TapedStateInputCost final : public StateInputCost {
 public:
  TapedStateInputCost()
      : tape_(std::make_shared<TapedFunction>(
            [](const TapedFunction::ad_vector_t& z) {
              using ad_t = TapedFunction::ad_t;
              const TapedFunction::ad_vector_t x = z.head(10);
              const TapedFunction::ad_vector_t u = z.tail(6);
              // The reference for column 2 of the base rotation, the base z axis.
              rotation_vector_t<ad_t> refZAxisBaseEwrtWorld;
              refZAxisBaseEwrtWorld << ad_t(0.1), ad_t(0.2), ad_t(0.97);
              rotation_vector_t<ad_t> refAngvelBaseRtWorldEwrtBase;
              refAngvelBaseRtWorldEwrtBase << ad_t(0.1), ad_t(0.0), ad_t(0.3);
              const rotation_vector_t<ad_t> angvelBaseRtWorldEwrtBase = x.template segment<3>(7);
              ad_t cost = ad_t(0.5) * (getRotmatBaseRtWorld(x).col(2) - refZAxisBaseEwrtWorld).squaredNorm() + getRotmatBaseRtWorld(x).col(0)(1) * x(0) +
                          ad_t(0.5) * (angvelBaseRtWorldEwrtBase - refAngvelBaseRtWorldEwrtBase).squaredNorm() + angvelBaseRtWorldEwrtBase(2) * u(0);
              for (int i = 0; i < 6; ++i) {
                cost += ad_t(0.5 * (i + 1)) * u(i) * u(i);
              }
              TapedFunction::ad_vector_t y(1);
              y << cost;
              return y;
            },
            /*inputDim=*/16)) {}
  TapedStateInputCost* absl_nonnull clone() const override { return new TapedStateInputCost(*this); }

  scalar_t getValue(scalar_t /*time*/,
                    const vector_t& x,
                    const vector_t& u,
                    const TargetTrajectories& /*targets*/,
                    const PreComputation& /*preComp*/) const override {
    return tape_->value(TapedFunction::stack(x, u))(0);
  }

  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t /*time*/,
                                                                 const vector_t& x,
                                                                 const vector_t& u,
                                                                 const TargetTrajectories& /*targets*/,
                                                                 const PreComputation& /*preComp*/) const override {
    const vector_t z = TapedFunction::stack(x, u);
    const vector_t gradient = tape_->jacobian(z).row(0).transpose();
    const matrix_t hessian = tape_->hessian(z, /*outputIndex=*/0);
    ScalarFunctionQuadraticApproximation approximation;
    approximation.f = tape_->value(z)(0);
    approximation.dfdx = gradient.head(10);
    approximation.dfdu = gradient.tail(6);
    approximation.dfdxx = hessian.topLeftCorner(10, 10);
    approximation.dfdux = hessian.bottomLeftCorner(6, 10);
    approximation.dfduu = hessian.bottomRightCorner(6, 6);
    return approximation;
  }

 private:
  TapedStateInputCost(const TapedStateInputCost& other) = default;
  std::shared_ptr<TapedFunction> tape_;  // single-threaded tests only
};

/** l(x) = 1/2 |R x - (0.6, 0.8, 0)|^2 + p_z (R y)_z + 1/2 |omega|^2. */
class TapedStateCost final : public StateCost {
 public:
  TapedStateCost()
      : tape_(std::make_shared<TapedFunction>(
            [](const TapedFunction::ad_vector_t& x) {
              using ad_t = TapedFunction::ad_t;
              // The reference for column 0 of the base rotation, the base x axis.
              rotation_vector_t<ad_t> refXAxisBaseEwrtWorld;
              refXAxisBaseEwrtWorld << ad_t(0.6), ad_t(0.8), ad_t(0.0);
              TapedFunction::ad_vector_t y(1);
              y << ad_t(0.5) * (getRotmatBaseRtWorld(x).col(0) - refXAxisBaseEwrtWorld).squaredNorm() + x(2) * getRotmatBaseRtWorld(x).col(1)(2) +
                       ad_t(0.5) * x.template segment<3>(7).squaredNorm();
              return y;
            },
            /*inputDim=*/10)) {}
  TapedStateCost* absl_nonnull clone() const override { return new TapedStateCost(*this); }

  scalar_t getValue(scalar_t /*time*/,
                    const vector_t& x,
                    const TargetTrajectories& /*targets*/,
                    const PreComputation& /*preComp*/) const override {
    return tape_->value(x)(0);
  }

  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t /*time*/,
                                                                 const vector_t& x,
                                                                 const TargetTrajectories& /*targets*/,
                                                                 const PreComputation& /*preComp*/) const override {
    ScalarFunctionQuadraticApproximation approximation;
    approximation.f = tape_->value(x)(0);
    approximation.dfdx = tape_->jacobian(x).row(0).transpose();
    approximation.dfdxx = tape_->hessian(x, /*outputIndex=*/0);
    return approximation;
  }

 private:
  TapedStateCost(const TapedStateCost& other) = default;
  std::shared_ptr<TapedFunction> tape_;
};

/** g(x, u) = [(R x)_y + 0.2 v_x - 0.1; p_z + omega_x tau_y]. */
class TapedStateInputConstraint final : public StateInputConstraint {
 public:
  TapedStateInputConstraint()
      : StateInputConstraint(ConstraintOrder::Linear),
        tape_(std::make_shared<TapedFunction>(
            [](const TapedFunction::ad_vector_t& z) {
              using ad_t = TapedFunction::ad_t;
              TapedFunction::ad_vector_t g(2);
              g << getRotmatBaseRtWorld(z).col(0)(1) + ad_t(0.2) * z(10) - ad_t(0.1), z(2) + z(7) * z(14);
              return g;
            },
            /*inputDim=*/16)) {}
  TapedStateInputConstraint* absl_nonnull clone() const override { return new TapedStateInputConstraint(*this); }

  size_t getNumConstraints(scalar_t /*time*/) const override { return 2; }

  vector_t getValue(scalar_t /*time*/, const vector_t& x, const vector_t& u, const PreComputation& /*preComp*/) const override {
    return tape_->value(TapedFunction::stack(x, u));
  }

  VectorFunctionLinearApproximation getLinearApproximation(scalar_t /*time*/,
                                                           const vector_t& x,
                                                           const vector_t& u,
                                                           const PreComputation& /*preComp*/) const override {
    const vector_t z = TapedFunction::stack(x, u);
    const matrix_t jacobian = tape_->jacobian(z);
    VectorFunctionLinearApproximation approximation;
    approximation.f = tape_->value(z);
    approximation.dfdx = jacobian.leftCols(10);
    approximation.dfdu = jacobian.rightCols(6);
    return approximation;
  }

 private:
  TapedStateInputConstraint(const TapedStateInputConstraint& other) = default;
  std::shared_ptr<TapedFunction> tape_;
};

/** h(x) = [(R z)_x + p_y; omega_z (R y)_x]. */
class TapedStateConstraint final : public StateConstraint {
 public:
  TapedStateConstraint()
      : StateConstraint(ConstraintOrder::Linear),
        tape_(std::make_shared<TapedFunction>(
            [](const TapedFunction::ad_vector_t& x) {
              TapedFunction::ad_vector_t h(2);
              h << getRotmatBaseRtWorld(x).col(2)(0) + x(1), x(9) * getRotmatBaseRtWorld(x).col(1)(0);
              return h;
            },
            /*inputDim=*/10)) {}
  TapedStateConstraint* absl_nonnull clone() const override { return new TapedStateConstraint(*this); }

  size_t getNumConstraints(scalar_t /*time*/) const override { return 2; }

  vector_t getValue(scalar_t /*time*/, const vector_t& x, const PreComputation& /*preComp*/) const override { return tape_->value(x); }

  VectorFunctionLinearApproximation getLinearApproximation(scalar_t /*time*/,
                                                           const vector_t& x,
                                                           const PreComputation& /*preComp*/) const override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = tape_->value(x);
    approximation.dfdx = tape_->jacobian(x);
    return approximation;
  }

 private:
  TapedStateConstraint(const TapedStateConstraint& other) = default;
  std::shared_ptr<TapedFunction> tape_;
};

}  // namespace manifold_test
}  // namespace ocs2
