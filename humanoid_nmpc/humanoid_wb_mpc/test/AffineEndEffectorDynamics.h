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
 * End-effector dynamics whose position, linear velocity and acceleration, orientation error wrt the plane and
 * pose-wrt-plane rate and acceleration are fixed affine maps of the state and the input, with exact linear
 * approximations. The pose rates are maps of their own, other than the linear velocity's and acceleration's: only the
 * composition a constraint performs is under test, and so it shows which of them a constraint reads. The orientation
 * error and the pose rates ignore the plane normal.
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
    velocityDx_ = matrix_t::Zero(3, kAffineStateDim);
    velocityDu_ = matrix_t::Zero(3, kAffineInputDim);
    linearAccelerationDx_ = matrix_t::Zero(3, kAffineStateDim);
    linearAccelerationDu_ = matrix_t::Zero(3, kAffineInputDim);
    poseRateDx_ = matrix_t::Zero(6, kAffineStateDim);
    poseRateDu_ = matrix_t::Zero(6, kAffineInputDim);
    poseAccelerationDx_ = matrix_t::Zero(6, kAffineStateDim);
    poseAccelerationDu_ = matrix_t::Zero(6, kAffineInputDim);
    for (Eigen::Index row = 0; row < 6; ++row) {
      for (Eigen::Index col = 0; col < kAffineStateDim; ++col) {
        if (row < 3) {
          velocityDx_(row, col) = 0.1 * static_cast<scalar_t>((row + 2 * col) % 5) - 0.2;
          linearAccelerationDx_(row, col) = 0.05 * static_cast<scalar_t>((3 * row + col) % 7) - 0.15;
        }
        poseRateDx_(row, col) = 0.07 * static_cast<scalar_t>((2 * row + 3 * col) % 6) - 0.13;
        poseAccelerationDx_(row, col) = 0.11 * static_cast<scalar_t>((row + 5 * col) % 4) - 0.17;
      }
      for (Eigen::Index col = 0; col < kAffineInputDim; ++col) {
        if (row < 3) {
          velocityDu_(row, col) = 0.2 * static_cast<scalar_t>((2 * row + col) % 3) - 0.1;
          linearAccelerationDu_(row, col) = 0.3 * static_cast<scalar_t>((row + 3 * col) % 4) - 0.4;
        }
        poseRateDu_(row, col) = 0.15 * static_cast<scalar_t>((3 * row + 2 * col) % 5) - 0.25;
        poseAccelerationDu_(row, col) = 0.09 * static_cast<scalar_t>((4 * row + col) % 7) - 0.3;
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
  vector3_t velocity(const vector_t& state, const vector_t& input) const { return velocityDx_ * state + velocityDu_ * input; }
  vector3_t linearAcceleration(const vector_t& state, const vector_t& input) const {
    return linearAccelerationDx_ * state + linearAccelerationDu_ * input;
  }
  vector3_t orientationErrorWrtPlane(const vector_t& state) const { return orientationDx_ * state; }
  vector6_t poseWrtPlaneRate(const vector_t& state, const vector_t& input) const { return poseRateDx_ * state + poseRateDu_ * input; }
  vector6_t poseWrtPlaneAcceleration(const vector_t& state, const vector_t& input) const {
    return poseAccelerationDx_ * state + poseAccelerationDu_ * input;
  }

  std::vector<vector3_t> getPosition(const vector_t& state) const override { return {position(state)}; }
  std::vector<vector3_t> getVelocity(const vector_t& state, const vector_t& input) const override { return {velocity(state, input)}; }
  std::vector<vector3_t> getLinearAcceleration(const vector_t& state, const vector_t& input) const override {
    return {linearAcceleration(state, input)};
  }
  std::vector<vector3_t> getOrientationErrorWrtPlane(const vector_t& state, const std::vector<vector3_t>& /*planeNormals*/) const override {
    return {orientationErrorWrtPlane(state)};
  }
  std::vector<vector6_t> getPoseWrtPlaneRate(const vector_t& state,
                                             const vector_t& input,
                                             const std::vector<vector3_t>& /*planeNormals*/) const override {
    return {poseWrtPlaneRate(state, input)};
  }
  std::vector<vector6_t> getPoseWrtPlaneAcceleration(const vector_t& state,
                                                     const vector_t& input,
                                                     const std::vector<vector3_t>& /*planeNormals*/) const override {
    return {poseWrtPlaneAcceleration(state, input)};
  }

  std::vector<VectorFunctionLinearApproximation> getPositionLinearApproximation(const vector_t& state) const override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = position(state);
    approximation.dfdx = positionDx_;
    return {approximation};
  }
  std::vector<VectorFunctionLinearApproximation> getVelocityLinearApproximation(const vector_t& state,
                                                                                const vector_t& input) const override {
    return {affine(velocity(state, input), velocityDx_, velocityDu_)};
  }
  std::vector<VectorFunctionLinearApproximation> getLinearAccelerationLinearApproximation(const vector_t& state,
                                                                                          const vector_t& input) const override {
    return {affine(linearAcceleration(state, input), linearAccelerationDx_, linearAccelerationDu_)};
  }
  std::vector<VectorFunctionLinearApproximation> getOrientationErrorWrtPlaneLinearApproximation(
      const vector_t& state, const std::vector<vector3_t>& /*planeNormals*/) const override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = orientationErrorWrtPlane(state);
    approximation.dfdx = orientationDx_;
    return {approximation};
  }
  std::vector<VectorFunctionLinearApproximation> getPoseWrtPlaneRateLinearApproximation(
      const vector_t& state, const vector_t& input, const std::vector<vector3_t>& /*planeNormals*/) const override {
    return {affine(poseWrtPlaneRate(state, input), poseRateDx_, poseRateDu_)};
  }
  std::vector<VectorFunctionLinearApproximation> getPoseWrtPlaneAccelerationLinearApproximation(
      const vector_t& state, const vector_t& input, const std::vector<vector3_t>& /*planeNormals*/) const override {
    return {affine(poseWrtPlaneAcceleration(state, input), poseAccelerationDx_, poseAccelerationDu_)};
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
  matrix_t velocityDx_;
  matrix_t velocityDu_;
  matrix_t linearAccelerationDx_;
  matrix_t linearAccelerationDu_;
  matrix_t poseRateDx_;
  matrix_t poseRateDu_;
  matrix_t poseAccelerationDx_;
  matrix_t poseAccelerationDu_;
};

}  // namespace ocs2::humanoid::test_support
