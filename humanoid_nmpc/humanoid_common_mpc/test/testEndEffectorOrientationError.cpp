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

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "gtest/gtest.h"
#include "ocs2_core/automatic_differentiation/CppAdInterface.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"

#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"

/*
 * The orientation error of the end-effector kinematic costs (computeTaskSpaceErrors()): the rotation vector
 * Log(q_ref (x) q^-1) in world axes, against an independent construction q_ref = AngleAxis(theta, n) q; invariant to the
 * sign and scale of either quaternion; its derivative along a world-frame perturbation q -> Exp(delta) q against
 * -Jr^-1(e) and central differences; and the CppAD tape the costs generate against the double overload, with no
 * comparison between variables recorded. It compiles one small library, so it runs alone (tags = ["exclusive"]).
 */
namespace ocs2::humanoid {
namespace {

constexpr Eigen::Index kOrientationErrorStart = kPositionDim;
constexpr scalar_t kTol = 1.0e-12;
constexpr scalar_t kStep = 1.0e-6;
constexpr scalar_t kFiniteDifferenceTolerance = 1.0e-8;

/** A random unit quaternion. */
Eigen::Quaterniond randomQuaternion(std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  const vector4_t coefficients(normal(generator), normal(generator), normal(generator), normal(generator));  // (x, y, z, w)
  return Eigen::Quaterniond(coefficients).normalized();
}

/** A random rotation vector of angle below pi - 0.1, away from the half turn where the shortest path is not unique. */
vector3_t randomRotationVector(std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  constexpr scalar_t kSmallestAngle = 1.0e-3;  // [rad]
  std::uniform_real_distribution<scalar_t> angle(kSmallestAngle, M_PI - 0.1);
  return angle(generator) * vector3_t(normal(generator), normal(generator), normal(generator)).normalized();
}

/** The orientation error of computeTaskSpaceErrors() of a link at `current` against `reference`, coefficients (x, y, z, w). */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> orientationError(const VECTOR4_T<SCALAR_T>& current, const VECTOR4_T<SCALAR_T>& reference) {
  EndEffectorKinematicsCostElement<SCALAR_T> currentElement;
  EndEffectorKinematicsCostElement<SCALAR_T> referenceElement;
  currentElement.setOrientation(current);
  referenceElement.setOrientation(reference);
  return computeTaskSpaceErrors<SCALAR_T>(currentElement, referenceElement).template segment<kRotationTangentDim>(kOrientationErrorStart);
}

Eigen::Quaterniond worldRotated(const vector3_t& rotationVector, const Eigen::Quaterniond& orientation) {
  const scalar_t angle = rotationVector.norm();
  if (angle == 0.0) return orientation;
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotationVector / angle)) * orientation;
}

}  // namespace

TEST(EndEffectorOrientationError, IsTheRotationVectorInWorldAxes) {
  std::mt19937 generator(13);
  std::uniform_real_distribution<scalar_t> scale(0.2, 5.0);
  for (int trial = 0; trial < 1000; ++trial) {
    const Eigen::Quaterniond current = randomQuaternion(generator);
    const vector3_t rotation = randomRotationVector(generator);
    const Eigen::Quaterniond reference = worldRotated(rotation, current);
    EXPECT_LT((orientationError<scalar_t>(current.coeffs(), reference.coeffs()) - rotation).norm(), kTol) << "trial " << trial;
    // Either quaternion of either sign and of any positive scale describes the same orientation.
    const vector4_t scaledCurrent = -scale(generator) * current.coeffs();
    const vector4_t scaledReference = scale(generator) * reference.coeffs();
    EXPECT_LT((orientationError<scalar_t>(scaledCurrent, scaledReference) - rotation).norm(), kTol) << "trial " << trial;
  }
  // At the reference the error is zero.
  const Eigen::Quaterniond orientation = randomQuaternion(generator);
  EXPECT_LT(orientationError<scalar_t>(orientation.coeffs(), orientation.coeffs()).norm(), kTol);
}

TEST(EndEffectorOrientationError, WeighsTheAngleNotItsHalfSine) {
  // The residual it replaced, the vector part sin(theta / 2) n of q_ref (x) q^-1, is half the angle at small angles: a
  // weight w on it is w / 4 on this residual (the rescaled task_space_costs weights).
  std::mt19937 generator(17);
  for (int trial = 0; trial < 100; ++trial) {
    const Eigen::Quaterniond current = randomQuaternion(generator);
    const vector3_t rotation = 1.0e-4 * randomRotationVector(generator);
    const Eigen::Quaterniond reference = worldRotated(rotation, current);
    const vector3_t halfSine = (reference * current.conjugate()).vec();
    // theta - 2 sin(theta / 2) = theta^3 / 24 + O(theta^5): the two agree to third order in the angle.
    const scalar_t angle = rotation.norm();
    EXPECT_LT((orientationError<scalar_t>(current.coeffs(), reference.coeffs()) - 2.0 * halfSine).norm(),
              angle * angle * angle / 12.0 + 1.0e-15);
  }
}

TEST(EndEffectorOrientationError, DerivativeAlongAWorldPerturbation) {
  // q(delta) = Exp(delta) q: e(delta) = Log(Exp(e0) Exp(-delta)) = e0 - Jr^-1(e0) delta + O(delta^2).
  std::mt19937 generator(19);
  for (int trial = 0; trial < 200; ++trial) {
    const Eigen::Quaterniond current = randomQuaternion(generator);
    const vector3_t rotation = randomRotationVector(generator);
    const Eigen::Quaterniond reference = worldRotated(rotation, current);
    matrix3_t finiteDifference;
    for (int axis = 0; axis < kRotationTangentDim; ++axis) {
      const vector3_t delta = kStep * vector3_t::Unit(axis);
      finiteDifference.col(axis) = (orientationError<scalar_t>(worldRotated(delta, current).coeffs(), reference.coeffs()) -
                                    orientationError<scalar_t>(worldRotated(-delta, current).coeffs(), reference.coeffs())) /
                                   (2.0 * kStep);
    }
    const matrix3_t expected = -so3RightJacobianInverse<scalar_t>(rotation);
    EXPECT_LT((finiteDifference - expected).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance) << "trial " << trial;
  }
}

TEST(EndEffectorOrientationError, TheTapeMatchesTheDoubleOverload) {
  // x = [current (4), reference (4)] -> the orientation error, taped and generated as the costs are.
  constexpr Eigen::Index kVariableDim = 2 * kQuaternionCoefficients;
  const CppAdInterface::ad_function_t taped = [](const ad_vector_t& x, ad_vector_t& y) {
    y = orientationError<ad_scalar_t>(ad_vector4_t(x.head<kQuaternionCoefficients>()), ad_vector4_t(x.tail<kQuaternionCoefficients>()));
  };
  const std::function<vector_t(const vector_t&)> reference = [](const vector_t& x) -> vector_t {
    return orientationError<scalar_t>(vector4_t(x.head<kQuaternionCoefficients>()), vector4_t(x.tail<kQuaternionCoefficients>()));
  };

  // No comparison between variables, which CppADCodeGen cannot generate: recording comparisons adds no operation.
  const std::function<size_t(bool)> tapeOperations = [&taped](bool recordCompare) {
    ad_vector_t x = ad_vector_t::Ones(kVariableDim);
    CppAD::Independent(x, /*abort_op_index=*/0, recordCompare);
    ad_vector_t y;
    taped(x, y);
    const CppAD::ADFun<ad_base_t> fun(x, y);
    return fun.size_op();
  };
  EXPECT_EQ(tapeOperations(/*recordCompare=*/true), tapeOperations(/*recordCompare=*/false));

  CppAdInterface library(taped, static_cast<size_t>(kVariableDim), "testEndEffectorOrientationError",
                         testing::TempDir() + "cppad_end_effector_orientation_error");
  library.createModels(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);

  std::mt19937 generator(23);
  std::vector<vector_t> points;
  for (int trial = 0; trial < 50; ++trial) {
    const Eigen::Quaterniond current = randomQuaternion(generator);
    const Eigen::Quaterniond target = worldRotated(randomRotationVector(generator), current);
    vector_t x(kVariableDim);
    x << current.coeffs(), target.coeffs();
    points.push_back(x);
    // At the reference itself (zero error, the series branch of the logarithm).
    x << current.coeffs(), current.coeffs();
    points.push_back(x);
  }
  for (const vector_t& x : points) {
    EXPECT_LT((library.getFunctionValue(x) - reference(x)).cwiseAbs().maxCoeff(), 1.0e-12) << x.transpose();
    matrix_t finiteDifference(kRotationTangentDim, kVariableDim);
    for (Eigen::Index i = 0; i < kVariableDim; ++i) {
      vector_t plus = x;
      vector_t minus = x;
      plus(i) += kStep;
      minus(i) -= kStep;
      finiteDifference.col(i) = (reference(plus) - reference(minus)) / (2.0 * kStep);
    }
    EXPECT_LT((library.getJacobian(x) - finiteDifference).cwiseAbs().maxCoeff(), 1.0e-7) << x.transpose();
  }
}

}  // namespace ocs2::humanoid
