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

#include <cppad/cppad.hpp>

#include <memory>
#include <vector>

#include <ocs2_core/Types.h>
#include <ocs2_core/dynamics/SystemDynamicsBase.h>
#include <ocs2_core/manifold/ProductStateManifold.h>
#include <ocs2_core/manifold/UnitQuaternionMath.h>

namespace ocs2 {
namespace manifold_test {

/**
 * A free rigid body whose attitude is a unit quaternion, for the tests of the manifold machinery:
 *   state  x = [p (3, world), xi (4, x y z w), omega (3, body)]   (without the position: x = [xi, omega])
 *   input  u = [v (3, body velocity), tau (3, body torque)]       (without the position: u = tau)
 *   p' = R(xi / |xi|) v,  xi' = 0.5 G(xi) omega,  omega' = I^-1 (tau - omega x I omega).
 * As in the humanoid flow maps, every function sees the quaternion through xi / |xi| except the kinematic row, which
 * uses the raw xi; the vector field is then tangent to the sphere through xi.
 *
 * The jump map rotates the body by a fixed quaternion (xi+ = xi (x) q) and halves the angular velocity.
 *
 * Derivatives come from CppAD tapes (plain AD<double>, no code generation), recorded at x = u = ones, so the tapes
 * are branch-free and also exercise UnitQuaternionMath on AD scalars.
 */
class RigidBodyAttitudeDynamics final : public SystemDynamicsBase {
 public:
  using ad_t = CppAD::AD<scalar_t>;
  using ad_vector_t = Eigen::Matrix<ad_t, Eigen::Dynamic, 1>;

  explicit RigidBodyAttitudeDynamics(bool withPosition) : withPosition_(withPosition) { recordTapes(); }
  ~RigidBodyAttitudeDynamics() override = default;
  RigidBodyAttitudeDynamics* clone() const override { return new RigidBodyAttitudeDynamics(withPosition_); }

  size_t getStateDim() const { return withPosition_ ? 10 : 7; }
  size_t getInputDim() const { return withPosition_ ? 6 : 3; }
  size_t getQuaternionOffset() const { return withPosition_ ? 3 : 0; }

  /** The manifold [E(3), Q, E(3)] (with the position) or [Q, E(3)]. */
  std::shared_ptr<const ProductStateManifold> getStateManifold() const {
    std::vector<StateManifoldSegment> segments;
    if (withPosition_) {
      segments.push_back(StateManifoldSegment::euclidean(3));
    }
    segments.push_back(StateManifoldSegment::unitQuaternion());
    segments.push_back(StateManifoldSegment::euclidean(3));
    return ProductStateManifold::create(segments).value();
  }

  vector_t computeFlowMap(scalar_t /*t*/, const vector_t& x, const vector_t& u, const PreComputation& /*preComp*/) override {
    return toVector(flowTape_->Forward(/*q=*/0, stack(x, u)));
  }

  VectorFunctionLinearApproximation linearApproximation(scalar_t /*t*/,
                                                        const vector_t& x,
                                                        const vector_t& u,
                                                        const PreComputation& /*preComp*/) override {
    const std::vector<scalar_t> xu = stack(x, u);
    VectorFunctionLinearApproximation approximation;
    approximation.f = toVector(flowTape_->Forward(/*q=*/0, xu));
    const matrix_t jacobian = rowMajorJacobian(*flowTape_, xu, getStateDim());
    approximation.dfdx = jacobian.leftCols(getStateDim());
    approximation.dfdu = jacobian.rightCols(getInputDim());
    return approximation;
  }

  vector_t computeJumpMap(scalar_t /*t*/, const vector_t& x, const PreComputation& /*preComp*/) override {
    return toVector(jumpTape_->Forward(/*q=*/0, stack(x, vector_t())));
  }

  VectorFunctionLinearApproximation jumpMapLinearApproximation(scalar_t /*t*/,
                                                               const vector_t& x,
                                                               const PreComputation& /*preComp*/) override {
    VectorFunctionLinearApproximation approximation;
    const std::vector<scalar_t> point = stack(x, vector_t());
    approximation.f = toVector(jumpTape_->Forward(/*q=*/0, point));
    approximation.dfdx = rowMajorJacobian(*jumpTape_, point, getStateDim());
    approximation.dfdu.setZero(getStateDim(), 0);
    return approximation;
  }

  /** The flow map on AD scalars, x and u stacked. */
  ad_vector_t flowMap(const ad_vector_t& xu) const {
    const size_t q = getQuaternionOffset();
    const quaternion_coeffs_t<ad_t> xi = xu.template segment<4>(q);
    const rotation_vector_t<ad_t> omega = xu.template segment<3>(q + 4);
    const rotation_vector_t<ad_t> tau = xu.template tail<3>();
    const Eigen::Matrix<ad_t, 3, 3> inertia = inertiaMatrix().template cast<ad_t>();
    const Eigen::Matrix<ad_t, 3, 3> inertiaInverse = inertiaMatrix().inverse().template cast<ad_t>();

    ad_vector_t dxdt(getStateDim());
    if (withPosition_) {
      const rotation_vector_t<ad_t> velocity = xu.template segment<3>(getStateDim());
      dxdt.template head<3>() = quaternionRotationMatrix(quaternionSafeNormalize(xi)) * velocity;
    }
    dxdt.template segment<4>(q) = ad_t(0.5) * quaternionRateMatrix(xi) * omega;
    const rotation_vector_t<ad_t> angularMomentum = inertia * omega;
    dxdt.template segment<3>(q + 4) = inertiaInverse * (tau - omega.cross(angularMomentum));
    return dxdt;
  }

  /** The fixed rotation of the jump map. */
  static quaternion_coeffs_t<scalar_t> jumpRotation() {
    rotation_vector_t<scalar_t> rotationVector;
    rotationVector << 0.3, -0.2, 0.5;
    return quaternionExp(rotationVector);
  }

  static Eigen::Matrix<scalar_t, 3, 3> inertiaMatrix() {
    Eigen::Matrix<scalar_t, 3, 3> inertia;
    inertia << 0.8, 0.05, -0.02, 0.05, 1.1, 0.03, -0.02, 0.03, 1.4;
    return inertia;
  }

 private:
  RigidBodyAttitudeDynamics(const RigidBodyAttitudeDynamics& other) = delete;

  static std::vector<scalar_t> stack(const vector_t& x, const vector_t& u) {
    std::vector<scalar_t> xu(x.size() + u.size());
    Eigen::Map<vector_t>(xu.data(), x.size()) = x;
    Eigen::Map<vector_t>(xu.data() + x.size(), u.size()) = u;
    return xu;
  }

  static vector_t toVector(const std::vector<scalar_t>& values) { return Eigen::Map<const vector_t>(values.data(), values.size()); }

  static matrix_t rowMajorJacobian(CppAD::ADFun<scalar_t>& tape, const std::vector<scalar_t>& point, size_t rows) {
    const std::vector<scalar_t> values = tape.Jacobian(point);
    return Eigen::Map<const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, point.size());
  }

  void recordTapes() {
    const size_t nx = getStateDim();
    ad_vector_t xu = ad_vector_t::Ones(nx + getInputDim());
    CppAD::Independent(xu);
    ad_vector_t dxdt = flowMap(xu);
    flowTape_ = std::make_unique<CppAD::ADFun<scalar_t>>(xu, dxdt);

    ad_vector_t x = ad_vector_t::Ones(nx);
    CppAD::Independent(x);
    const size_t q = getQuaternionOffset();
    ad_vector_t xNext = x;
    const quaternion_coeffs_t<ad_t> xi = x.template segment<4>(q);
    xNext.template segment<4>(q) = quaternionProduct(xi, quaternion_coeffs_t<ad_t>(jumpRotation().template cast<ad_t>()));
    xNext.template segment<3>(q + 4) = ad_t(0.5) * x.template segment<3>(q + 4);
    jumpTape_ = std::make_unique<CppAD::ADFun<scalar_t>>(x, xNext);
  }

  bool withPosition_;
  std::unique_ptr<CppAD::ADFun<scalar_t>> flowTape_;
  std::unique_ptr<CppAD::ADFun<scalar_t>> jumpTape_;
};

}  // namespace manifold_test
}  // namespace ocs2
