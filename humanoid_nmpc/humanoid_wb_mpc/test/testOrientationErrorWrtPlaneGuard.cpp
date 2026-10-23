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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/common/Types.h"

/*
 * The orientation error with respect to a plane, e = -(v x n) / sqrt(2 (1 + v.n)) (rotationMatrixDistanceToPlane() and
 * quaternionDistanceToPlane() of ocs2_robotic_tools), v the end-effector normal and n the plane's: the closed form on
 * random attitudes and planes; finite, with a finite taped Jacobian, where v = -n, where the shortest arc of
 * getQuaternionFromUnitVectors() used to divide 0 by 0; continuous across the clamp of its squared norm; and the taped
 * Jacobian equal to central differences outside it.
 */
namespace ocs2::humanoid {
namespace {

using ad_fun_t = CppAD::ADFun<ad_base_t>;

/** e = -(v x n) / sqrt(2 (1 + v.n)), the closed form, for v and n with 1 + v.n well above zero. */
vector3_t closedForm(const vector3_t& v, const vector3_t& n) {
  return -v.cross(n) / std::sqrt(2.0 * (1.0 + v.dot(n)));
}

/** A uniformly random unit quaternion. */
quaternion_t randomAttitude(std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal;
  const vector4_t coefficients(normal(generator), normal(generator), normal(generator), normal(generator));
  return quaternion_t(coefficients).normalized();
}

/** The attitude of an end effector tilted by `angle` about the world x axis: its normal is (0, -sin, cos). */
matrix3_t tiltedAboutX(scalar_t angle) {
  return Eigen::AngleAxisd(angle, vector3_t::UnitX()).toRotationMatrix();
}

/** Tapes e(R) = rotationMatrixDistanceToPlane(R, n) with the nine entries of R, column by column, the variables. */
ad_fun_t tapedError(const vector3_t& planeNormal) {
  const matrix3_t identity = matrix3_t::Identity();
  ad_vector_t x(9);
  for (Eigen::Index i = 0; i < x.size(); ++i) x(i) = ad_scalar_t(identity(i));
  CppAD::Independent(x);
  ad_matrix_t rotation(3, 3);
  for (Eigen::Index i = 0; i < x.size(); ++i) rotation(i) = x(i);
  const ad_vector3_t error = rotationMatrixDistanceToPlane<ad_scalar_t>(ad_matrix3_t(rotation), planeNormal.cast<ad_scalar_t>());
  ad_vector_t y = error;
  return ad_fun_t(x, y);
}

/** The taped Jacobian of `fun` at the attitude `rotation`, 3 x 9. */
matrix_t tapedJacobian(ad_fun_t& fun, const matrix3_t& rotation) {
  CppAD::vector<ad_base_t> at(9);
  for (Eigen::Index i = 0; i < 9; ++i) at[static_cast<size_t>(i)] = ad_base_t(rotation(i));
  const CppAD::vector<ad_base_t> jacobian = fun.Jacobian(at);
  matrix_t result(3, 9);
  for (Eigen::Index row = 0; row < 3; ++row) {
    for (Eigen::Index col = 0; col < 9; ++col) result(row, col) = jacobian[static_cast<size_t>(row * 9 + col)].getValue();
  }
  return result;
}

TEST(OrientationErrorWrtPlane, IsTheClosedFormOnRandomAttitudesAndPlanes) {
  std::mt19937 generator(/*sd=*/2026);
  int checked = 0;
  for (int sample = 0; sample < 200; ++sample) {
    const quaternion_t attitude = randomAttitude(generator);
    const vector3_t planeNormal = (randomAttitude(generator) * vector3_t::UnitZ()).normalized();
    const vector3_t v = attitude.toRotationMatrix().col(2);
    if (1.0 + v.dot(planeNormal) < 1.0e-3) continue;
    ++checked;
    const vector3_t expected = closedForm(v, planeNormal);
    EXPECT_LT((rotationMatrixDistanceToPlane<scalar_t>(attitude.toRotationMatrix(), planeNormal) - expected).norm(), 1.0e-12) << sample;
    EXPECT_LT((quaternionDistanceToPlane<scalar_t>(attitude, planeNormal) - expected).norm(), 1.0e-12) << sample;
  }
  EXPECT_GT(checked, 150);
}

TEST(OrientationErrorWrtPlane, IsFiniteWithAFiniteTapedJacobianWhereTheNormalPointsAwayFromThePlane) {
  const vector3_t planeNormal = vector3_t::UnitZ();
  ad_fun_t fun = tapedError(planeNormal);
  // Upside down (v = -n exactly), and within the clamp of the shortest arc's norm (1 + v.n = delta^2 / 2 < 1e-6).
  for (const scalar_t offset : {0.0, 1.0e-8, 1.0e-4, 1.0e-3}) {
    SCOPED_TRACE(absl::StrCat("tilt pi - ", offset));
    const matrix3_t rotation = tiltedAboutX(M_PI - offset);
    ASSERT_LT(1.0 + rotation.col(2).dot(planeNormal), 1.0e-6);
    const vector3_t error = rotationMatrixDistanceToPlane<scalar_t>(rotation, planeNormal);
    EXPECT_TRUE(error.allFinite()) << error.transpose();
    EXPECT_LE(error.norm(), 1.0 + 1.0e-12) << "the error is the sine of half the tilt, at most 1";
    EXPECT_TRUE(quaternionDistanceToPlane<scalar_t>(quaternion_t(rotation), planeNormal).allFinite());
    const matrix_t jacobian = tapedJacobian(fun, rotation);
    EXPECT_TRUE(jacobian.allFinite()) << jacobian;
  }
}

TEST(OrientationErrorWrtPlane, IsContinuousAcrossTheClampOfTheShortestArc) {
  // 1 + v.n = 1 - cos(delta) = 1e-6 at the clamp: the error there is (almost) of unit norm on both sides of it.
  const vector3_t planeNormal = vector3_t::UnitZ();
  const scalar_t clampOffset = std::acos(1.0 - 1.0e-6);
  const vector3_t outside = rotationMatrixDistanceToPlane<scalar_t>(tiltedAboutX(M_PI - clampOffset * (1.0 + 1.0e-6)), planeNormal);
  const vector3_t inside = rotationMatrixDistanceToPlane<scalar_t>(tiltedAboutX(M_PI - clampOffset * (1.0 - 1.0e-6)), planeNormal);
  EXPECT_NEAR(outside.norm(), 1.0, 1.0e-5);
  EXPECT_LT((outside - inside).norm(), 1.0e-5);
}

TEST(OrientationErrorWrtPlane, TheTapedJacobianIsTheCentralDifferenceOutsideTheClamp) {
  std::mt19937 generator(/*sd=*/7);
  const vector3_t planeNormal = vector3_t(0.2, -0.1, 1.0).normalized();
  ad_fun_t fun = tapedError(planeNormal);
  constexpr scalar_t kStep = 1.0e-6;
  for (int sample = 0; sample < 20; ++sample) {
    const matrix3_t rotation = randomAttitude(generator).toRotationMatrix();
    if (1.0 + rotation.col(2).dot(planeNormal) < 1.0e-2) continue;
    const matrix_t jacobian = tapedJacobian(fun, rotation);
    matrix_t differences(3, 9);
    for (Eigen::Index i = 0; i < 9; ++i) {
      matrix3_t plus = rotation;
      matrix3_t minus = rotation;
      plus(i) += kStep;
      minus(i) -= kStep;
      differences.col(i) =
          (rotationMatrixDistanceToPlane<scalar_t>(plus, planeNormal) - rotationMatrixDistanceToPlane<scalar_t>(minus, planeNormal)) /
          (2.0 * kStep);
    }
    EXPECT_LT((jacobian - differences).cwiseAbs().maxCoeff(), 1.0e-6 * std::max(1.0, differences.cwiseAbs().maxCoeff())) << sample;
  }
}

}  // namespace
}  // namespace ocs2::humanoid
