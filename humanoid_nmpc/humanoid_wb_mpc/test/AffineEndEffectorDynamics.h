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

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_wb_mpc/end_effector/EndEffectorDynamics.h"

namespace ocs2::humanoid::test_support {

/** The state and input dimensions of AffineEndEffectorDynamics. */
inline constexpr Eigen::Index kAffineStateDim = 4;
inline constexpr Eigen::Index kAffineInputDim = 5;

/**
 * End-effector dynamics whose position, orientation error wrt the plane, twist and accelerations are fixed affine maps
 * of the state and the input, with exact linear approximations. The orientation error ignores the plane normal: only
 * the composition the constraint performs is under test.
 */
class AffineEndEffectorDynamics final : public EndEffectorDynamics<scalar_t> {
 public:
  /** `ids`: the end effectors it reports (getIds()); the maps are those of one, whatever their number. */
  explicit AffineEndEffectorDynamics(std::vector<std::string> ids = {"test_foot"}) : ids_(std::move(ids)) {
    // Fixed, dense and of no particular structure, so that every row and column of every block is exercised.
    positionDx_ = matrix_t::Zero(3, kAffineStateDim);
    positionDx_ << 1.0, 0.2, -0.3, 0.4, -0.5, 1.1, 0.6, -0.2, 0.3, -0.7, 0.9, 0.5;
    positionOffset_ = vector3_t(0.1, -0.2, 0.05);
    orientationDx_ = matrix_t::Zero(3, kAffineStateDim);
    orientationDx_ << 0.3, -0.1, 0.8, 0.2, 0.6, 0.4, -0.5, 0.1, -0.2, 0.7, 0.3, -0.9;
    twistDx_ = matrix_t::Zero(6, kAffineStateDim);
    twistDu_ = matrix_t::Zero(6, kAffineInputDim);
    accelerationDx_ = matrix_t::Zero(6, kAffineStateDim);
    accelerationDu_ = matrix_t::Zero(6, kAffineInputDim);
    for (Eigen::Index row = 0; row < 6; ++row) {
      for (Eigen::Index col = 0; col < kAffineStateDim; ++col) {
        twistDx_(row, col) = 0.1 * static_cast<scalar_t>((row + 2 * col) % 5) - 0.2;
        accelerationDx_(row, col) = 0.05 * static_cast<scalar_t>((3 * row + col) % 7) - 0.15;
      }
      for (Eigen::Index col = 0; col < kAffineInputDim; ++col) {
        twistDu_(row, col) = 0.2 * static_cast<scalar_t>((2 * row + col) % 3) - 0.1;
        accelerationDu_(row, col) = 0.3 * static_cast<scalar_t>((row + 3 * col) % 4) - 0.4;
      }
    }
  }
  ~AffineEndEffectorDynamics() override = default;
  AffineEndEffectorDynamics& operator=(const AffineEndEffectorDynamics&) = delete;
  AffineEndEffectorDynamics(AffineEndEffectorDynamics&&) = delete;
  AffineEndEffectorDynamics& operator=(AffineEndEffectorDynamics&&) = delete;
  AffineEndEffectorDynamics* absl_nonnull clone() const override { return new AffineEndEffectorDynamics(*this); }
  const std::vector<std::string>& getIds() const override { return ids_; }

  vector3_t position(const vector_t& state) const { return positionDx_ * state + positionOffset_; }
  vector3_t orientationErrorWrtPlane(const vector_t& state) const { return orientationDx_ * state; }
  vector6_t twist(const vector_t& state, const vector_t& input) const { return twistDx_ * state + twistDu_ * input; }
  vector6_t accelerations(const vector_t& state, const vector_t& input) const { return accelerationDx_ * state + accelerationDu_ * input; }

  std::vector<vector3_t> getPosition(const vector_t& state) const override { return {position(state)}; }
  std::vector<vector3_t> getOrientationErrorWrtPlane(const vector_t& state, const std::vector<vector3_t>& /*planeNormals*/) const override {
    return {orientationErrorWrtPlane(state)};
  }
  std::vector<vector6_t> getTwist(const vector_t& state, const vector_t& input) const override { return {twist(state, input)}; }
  std::vector<vector3_t> getVelocity(const vector_t& state, const vector_t& input) const override {
    return {twist(state, input).head<3>()};
  }
  std::vector<vector3_t> getAngularVelocity(const vector_t& state, const vector_t& input) const override {
    return {twist(state, input).tail<3>()};
  }
  std::vector<vector6_t> getAccelerations(const vector_t& state, const vector_t& input) const override {
    return {accelerations(state, input)};
  }
  std::vector<vector3_t> getLinearAcceleration(const vector_t& state, const vector_t& input) const override {
    return {accelerations(state, input).head<3>()};
  }
  std::vector<vector3_t> getAngularAcceleration(const vector_t& state, const vector_t& input) const override {
    return {accelerations(state, input).tail<3>()};
  }

  std::vector<VectorFunctionLinearApproximation> getPositionLinearApproximation(const vector_t& state) const override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = position(state);
    approximation.dfdx = positionDx_;
    return {approximation};
  }
  std::vector<VectorFunctionLinearApproximation> getOrientationErrorWrtPlaneLinearApproximation(
      const vector_t& state, const std::vector<vector3_t>& /*planeNormals*/) const override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = orientationErrorWrtPlane(state);
    approximation.dfdx = orientationDx_;
    return {approximation};
  }
  std::vector<VectorFunctionLinearApproximation> getTwistLinearApproximation(const vector_t& state, const vector_t& input) const override {
    return {affine(twist(state, input), twistDx_, twistDu_)};
  }
  std::vector<VectorFunctionLinearApproximation> getVelocityLinearApproximation(const vector_t& state,
                                                                                const vector_t& input) const override {
    return {affine(twist(state, input).head<3>(), twistDx_.topRows(3), twistDu_.topRows(3))};
  }
  std::vector<VectorFunctionLinearApproximation> getAngularVelocityLinearApproximation(const vector_t& state,
                                                                                       const vector_t& input) const override {
    return {affine(twist(state, input).tail<3>(), twistDx_.bottomRows(3), twistDu_.bottomRows(3))};
  }
  std::vector<VectorFunctionLinearApproximation> getAccelerationsLinearApproximation(const vector_t& state,
                                                                                     const vector_t& input) const override {
    return {affine(accelerations(state, input), accelerationDx_, accelerationDu_)};
  }
  std::vector<VectorFunctionLinearApproximation> getLinearAccelerationLinearApproximation(const vector_t& state,
                                                                                          const vector_t& input) const override {
    return {affine(accelerations(state, input).head<3>(), accelerationDx_.topRows(3), accelerationDu_.topRows(3))};
  }
  std::vector<VectorFunctionLinearApproximation> getAngularAccelerationLinearApproximation(const vector_t& state,
                                                                                           const vector_t& input) const override {
    return {affine(accelerations(state, input).tail<3>(), accelerationDx_.bottomRows(3), accelerationDu_.bottomRows(3))};
  }

  // The constraint never asks for an orientation, only for its error with respect to a plane.
  std::vector<quaternion_t> getOrientation(const vector_t& /*state*/) const override { throw std::logic_error("not used"); }
  std::vector<vector3_t> getOrientationError(const vector_t& /*state*/,
                                             const std::vector<quaternion_t>& /*referenceOrientations*/) const override {
    throw std::logic_error("not used");
  }
  std::vector<VectorFunctionLinearApproximation> getOrientationErrorLinearApproximation(
      const vector_t& /*state*/, const std::vector<quaternion_t>& /*referenceOrientations*/) const override {
    throw std::logic_error("not used");
  }

 private:
  AffineEndEffectorDynamics(const AffineEndEffectorDynamics&) = default;

  static VectorFunctionLinearApproximation affine(const vector_t& value, const matrix_t& dfdx, const matrix_t& dfdu) {
    VectorFunctionLinearApproximation approximation;
    approximation.f = value;
    approximation.dfdx = dfdx;
    approximation.dfdu = dfdu;
    return approximation;
  }

  std::vector<std::string> ids_;
  matrix_t positionDx_;
  vector3_t positionOffset_;
  matrix_t orientationDx_;
  matrix_t twistDx_;
  matrix_t twistDu_;
  matrix_t accelerationDx_;
  matrix_t accelerationDu_;
};

}  // namespace ocs2::humanoid::test_support
