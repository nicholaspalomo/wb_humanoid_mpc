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
 * End-effector dynamics whose position, linear velocity and acceleration,
 * orientation error wrt the plane and pose-wrt-plane rate and acceleration are
 * fixed affine maps of the state and the input, with exact linear
 * approximations. The pose rates are maps of their own, other than the linear
 * velocity's and acceleration's: only the composition a constraint performs is
 * under test, and so it shows which of them a constraint reads. The orientation
 * error and the pose rates ignore the plane normal.
 */
class AffineEndEffectorDynamics final : public EndEffectorDynamics<scalar_t> {
 public:
  /** `ids`: the end effectors it reports (getIds()); the maps are those of one,
   * whatever their number. */
  explicit AffineEndEffectorDynamics(
      std::vector<std::string> ids = {"test_foot"})
      : ids_(std::move(ids)) {
    // Fixed, dense and of no particular structure, so that every row and column
    // of every block is exercised.
    dxPosEeRtWorldEwrtWorld_ = matrix_t::Zero(3, kAffineStateDim);
    dxPosEeRtWorldEwrtWorld_ << 1.0, 0.2, -0.3, 0.4, -0.5, 1.1, 0.6, -0.2, 0.3,
        -0.7, 0.9, 0.5;
    offsetPosEeRtWorldEwrtWorld_ = vector3_t(0.1, -0.2, 0.05);
    dxTiltEeRtSurfaceEwrtWorld_ = matrix_t::Zero(3, kAffineStateDim);
    dxTiltEeRtSurfaceEwrtWorld_ << 0.3, -0.1, 0.8, 0.2, 0.6, 0.4, -0.5, 0.1,
        -0.2, 0.7, 0.3, -0.9;
    dxLinvelEeRtWorldEwrtWorld_ = matrix_t::Zero(3, kAffineStateDim);
    duLinvelEeRtWorldEwrtWorld_ = matrix_t::Zero(3, kAffineInputDim);
    dxLinaccEeRtWorldEwrtWorld_ = matrix_t::Zero(3, kAffineStateDim);
    duLinaccEeRtWorldEwrtWorld_ = matrix_t::Zero(3, kAffineInputDim);
    dxXeeRate_ = matrix_t::Zero(6, kAffineStateDim);
    duXeeRate_ = matrix_t::Zero(6, kAffineInputDim);
    dxXeeSecondRate_ = matrix_t::Zero(6, kAffineStateDim);
    duXeeSecondRate_ = matrix_t::Zero(6, kAffineInputDim);
    for (Eigen::Index row = 0; row < 6; ++row) {
      for (Eigen::Index col = 0; col < kAffineStateDim; ++col) {
        if (row < 3) {
          dxLinvelEeRtWorldEwrtWorld_(row, col) =
              0.1 * static_cast<scalar_t>((row + 2 * col) % 5) - 0.2;
          dxLinaccEeRtWorldEwrtWorld_(row, col) =
              0.05 * static_cast<scalar_t>((3 * row + col) % 7) - 0.15;
        }
        dxXeeRate_(row, col) =
            0.07 * static_cast<scalar_t>((2 * row + 3 * col) % 6) - 0.13;
        dxXeeSecondRate_(row, col) =
            0.11 * static_cast<scalar_t>((row + 5 * col) % 4) - 0.17;
      }
      for (Eigen::Index col = 0; col < kAffineInputDim; ++col) {
        if (row < 3) {
          duLinvelEeRtWorldEwrtWorld_(row, col) =
              0.2 * static_cast<scalar_t>((2 * row + col) % 3) - 0.1;
          duLinaccEeRtWorldEwrtWorld_(row, col) =
              0.3 * static_cast<scalar_t>((row + 3 * col) % 4) - 0.4;
        }
        duXeeRate_(row, col) =
            0.15 * static_cast<scalar_t>((3 * row + 2 * col) % 5) - 0.25;
        duXeeSecondRate_(row, col) =
            0.09 * static_cast<scalar_t>((4 * row + col) % 7) - 0.3;
      }
    }
  }
  ~AffineEndEffectorDynamics() override = default;
  AffineEndEffectorDynamics& operator=(const AffineEndEffectorDynamics&) =
      delete;
  AffineEndEffectorDynamics(AffineEndEffectorDynamics&&) = delete;
  AffineEndEffectorDynamics& operator=(AffineEndEffectorDynamics&&) = delete;
  AffineEndEffectorDynamics* absl_nonnull clone() const override {
    return new AffineEndEffectorDynamics(*this);
  }
  const std::vector<std::string>& getIds() const override { return ids_; }

  vector3_t posEeRtWorldEwrtWorld(const vector_t& state) const {
    return dxPosEeRtWorldEwrtWorld_ * state + offsetPosEeRtWorldEwrtWorld_;
  }
  vector3_t linvelEeRtWorldEwrtWorld(const vector_t& state,
                                     const vector_t& input) const {
    return dxLinvelEeRtWorldEwrtWorld_ * state +
           duLinvelEeRtWorldEwrtWorld_ * input;
  }
  vector3_t linaccEeRtWorldEwrtWorld(const vector_t& state,
                                     const vector_t& input) const {
    return dxLinaccEeRtWorldEwrtWorld_ * state +
           duLinaccEeRtWorldEwrtWorld_ * input;
  }
  vector3_t tiltEeRtSurfaceEwrtWorld(const vector_t& state) const {
    return dxTiltEeRtSurfaceEwrtWorld_ * state;
  }
  vector6_t xeeRate(const vector_t& state, const vector_t& input) const {
    return dxXeeRate_ * state + duXeeRate_ * input;
  }
  vector6_t xeeSecondRate(const vector_t& state, const vector_t& input) const {
    return dxXeeSecondRate_ * state + duXeeSecondRate_ * input;
  }

  std::vector<vector3_t> getPosEesRtWorldEwrtWorld(
      const vector_t& state) const override {
    return {posEeRtWorldEwrtWorld(state)};
  }
  std::vector<vector3_t> getLinvelEesRtWorldEwrtWorld(
      const vector_t& state, const vector_t& input) const override {
    return {linvelEeRtWorldEwrtWorld(state, input)};
  }
  std::vector<vector3_t> getLinaccEesRtWorldEwrtWorld(
      const vector_t& state, const vector_t& input) const override {
    return {linaccEeRtWorldEwrtWorld(state, input)};
  }
  std::vector<vector3_t> getTiltEesRtSurfaceEwrtWorld(
      const vector_t& state,
      const std::vector<vector3_t>& /*normalSurfacesEwrtWorld*/)
      const override {
    return {tiltEeRtSurfaceEwrtWorld(state)};
  }
  std::vector<vector6_t> getXeeRate(
      const vector_t& state,
      const vector_t& input,
      const std::vector<vector3_t>& /*normalSurfacesEwrtWorld*/)
      const override {
    return {xeeRate(state, input)};
  }
  std::vector<vector6_t> getXeeSecondRate(
      const vector_t& state,
      const vector_t& input,
      const std::vector<vector3_t>& /*normalSurfacesEwrtWorld*/)
      const override {
    return {xeeSecondRate(state, input)};
  }

  std::vector<VectorFunctionLinearApproximation>
  getLinearApproximationPosEesRtWorldEwrtWorld(
      const vector_t& state) const override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = posEeRtWorldEwrtWorld(state);
    approximation.dfdx = dxPosEeRtWorldEwrtWorld_;
    return {approximation};
  }
  std::vector<VectorFunctionLinearApproximation>
  getLinearApproximationLinvelEesRtWorldEwrtWorld(
      const vector_t& state, const vector_t& input) const override {
    return {affine(linvelEeRtWorldEwrtWorld(state, input),
                   dxLinvelEeRtWorldEwrtWorld_, duLinvelEeRtWorldEwrtWorld_)};
  }
  std::vector<VectorFunctionLinearApproximation>
  getLinearApproximationLinaccEesRtWorldEwrtWorld(
      const vector_t& state, const vector_t& input) const override {
    return {affine(linaccEeRtWorldEwrtWorld(state, input),
                   dxLinaccEeRtWorldEwrtWorld_, duLinaccEeRtWorldEwrtWorld_)};
  }
  std::vector<VectorFunctionLinearApproximation>
  getLinearApproximationTiltEesRtSurfaceEwrtWorld(
      const vector_t& state,
      const std::vector<vector3_t>& /*normalSurfacesEwrtWorld*/)
      const override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = tiltEeRtSurfaceEwrtWorld(state);
    approximation.dfdx = dxTiltEeRtSurfaceEwrtWorld_;
    return {approximation};
  }
  std::vector<VectorFunctionLinearApproximation> getLinearApproximationXeeRate(
      const vector_t& state,
      const vector_t& input,
      const std::vector<vector3_t>& /*normalSurfacesEwrtWorld*/)
      const override {
    return {affine(xeeRate(state, input), dxXeeRate_, duXeeRate_)};
  }
  std::vector<VectorFunctionLinearApproximation>
  getLinearApproximationXeeSecondRate(
      const vector_t& state,
      const vector_t& input,
      const std::vector<vector3_t>& /*normalSurfacesEwrtWorld*/)
      const override {
    return {affine(xeeSecondRate(state, input), dxXeeSecondRate_,
                   duXeeSecondRate_)};
  }

 private:
  AffineEndEffectorDynamics(const AffineEndEffectorDynamics&) = default;

  static VectorFunctionLinearApproximation affine(const vector_t& value,
                                                  const matrix_t& dfdx,
                                                  const matrix_t& dfdu) {
    VectorFunctionLinearApproximation approximation;
    approximation.f = value;
    approximation.dfdx = dfdx;
    approximation.dfdu = dfdu;
    return approximation;
  }

  std::vector<std::string> ids_;
  matrix_t dxPosEeRtWorldEwrtWorld_;
  vector3_t offsetPosEeRtWorldEwrtWorld_;
  matrix_t dxTiltEeRtSurfaceEwrtWorld_;
  matrix_t dxLinvelEeRtWorldEwrtWorld_;
  matrix_t duLinvelEeRtWorldEwrtWorld_;
  matrix_t dxLinaccEeRtWorldEwrtWorld_;
  matrix_t duLinaccEeRtWorldEwrtWorld_;
  matrix_t dxXeeRate_;
  matrix_t duXeeRate_;
  matrix_t dxXeeSecondRate_;
  matrix_t duXeeSecondRate_;
};

}  // namespace ocs2::humanoid::test_support
