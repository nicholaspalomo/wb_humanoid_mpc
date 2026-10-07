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
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/automatic_differentiation/CppAdInterface.h"

#include "humanoid_common_mpc/orientation/BaseOrientation.h"

/*
 * Step 5 of humanoid_nmpc/docs/quaternion_base_orientation/README.md: the two pieces of quaternion algebra the
 * humanoid CppAD tapes use, the safe normalization (BaseOrientation.h, on ocs2::quaternionSafeNorm) and the quaternion
 * rate matrix (ocs2::quaternionRateMatrix, the kinematic row of the flow maps), taped as CppAdInterface tapes every MPC
 * function (at x = ones) and code-generated:
 *   - the tape records no comparison between variables, which CppADCodeGen cannot generate (the reason Eigen's
 *     normalized() is never called on an AD scalar; it is the positive control here);
 *   - the generated values match the double-precision overloads, and the generated Jacobians match their central
 *     differences, at random quaternions, at x = ones, below epsilon and at the zero quaternion;
 *   - in particular the reverse-mode Jacobian, the mode CppADCodeGen generates when a function has fewer outputs than
 *     inputs (every cost), is finite at the zero quaternion. Guarding the norm instead of the squared norm makes it NaN
 *     there (Step 0, testQuaternionRootJointCppAd.cpp); that guard is the positive control.
 * It compiles three small libraries, so it runs alone (tags = ["exclusive"]).
 */

namespace ocs2::humanoid {
namespace {

using ad_fun_t = CppAD::ADFun<ad_base_t>;
using ReferenceFunction = std::function<vector_t(const vector_t&)>;

constexpr Eigen::Index kQuaternionDim = 4;
constexpr Eigen::Index kAngularVelocityDim = 3;
constexpr Eigen::Index kVariableDim = kQuaternionDim + kAngularVelocityDim;

/** Generated code (-O3 -ffast-math) against the double-precision overloads. */
constexpr scalar_t kValueTolerance = 1.0e-13;
constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
/** Below epsilon, so the finite differences at the zero quaternion stay on the branch they check. */
constexpr scalar_t kSmallFiniteDifferenceStep = 1.0e-9;
constexpr scalar_t kFiniteDifferenceTolerance = 1.0e-8;

/** Fixed weights that reduce the outputs to two, so that CppADCodeGen generates the Jacobian in reverse mode. */
vector4_t normalizedWeights() {
  return vector4_t(0.3, -0.5, 0.7, 0.2);
}
vector4_t rateWeights() {
  return vector4_t(-0.4, 0.1, 0.6, 0.9);
}

/**
 * x = [xi(4), w(3)] -> [xi_hat(4), n_safe(xi), G(xi) w(4), G(xi_hat) w(4)]: the safe normalization, and the rate matrix
 * on the raw quaternion (the kinematic row 1/2 G(xi) w_B of the flow maps) and on the normalized one. 13 outputs of 7
 * inputs: the Jacobian is generated in forward mode.
 */
template <typename SCALAR_T>
VECTOR_T<SCALAR_T> wideFunction(const VECTOR_T<SCALAR_T>& x) {
  const VECTOR4_T<SCALAR_T> xi = x.template head<kQuaternionDim>();
  const VECTOR3_T<SCALAR_T> omega = x.template tail<kAngularVelocityDim>();
  const VECTOR4_T<SCALAR_T> normalized = safelyNormalizedQuaternion(xi);
  VECTOR_T<SCALAR_T> y(13);
  y.template head<4>() = normalized;
  y(4) = safeQuaternionNorm(xi);
  y.template segment<4>(5) = quaternionRateMatrix<SCALAR_T>(xi) * omega;
  y.template tail<4>() = quaternionRateMatrix<SCALAR_T>(normalized) * omega;
  return y;
}

/** x = [xi(4), w(3)] -> [a^T xi_hat, b^T G(xi_hat) w]: 2 outputs of 7 inputs, a reverse-mode Jacobian. */
template <typename SCALAR_T>
VECTOR_T<SCALAR_T> narrowFunctionOf(const VECTOR4_T<SCALAR_T>& normalized, const VECTOR_T<SCALAR_T>& x) {
  const VECTOR3_T<SCALAR_T> omega = x.template tail<kAngularVelocityDim>();
  VECTOR_T<SCALAR_T> y(2);
  y(0) = normalizedWeights().cast<SCALAR_T>().dot(normalized);
  y(1) = rateWeights().cast<SCALAR_T>().dot(quaternionRateMatrix<SCALAR_T>(normalized) * omega);
  return y;
}

template <typename SCALAR_T>
VECTOR_T<SCALAR_T> narrowFunction(const VECTOR_T<SCALAR_T>& x) {
  return narrowFunctionOf<SCALAR_T>(safelyNormalizedQuaternion(VECTOR4_T<SCALAR_T>(x.template head<kQuaternionDim>())), x);
}

/** Design section 2.6 as first written: the guard on the norm, CondExpGt(|xi|, epsilon, |xi|, 1). */
ad_vector4_t normGuardedNormalization(const ad_vector4_t& xi) {
  const ad_scalar_t norm = CppAD::sqrt(xi.squaredNorm());
  return xi / CppAD::CondExpGt(norm, ad_scalar_t(kQuaternionNormGuard), norm, ad_scalar_t(1.0));
}

/** One taped function, its double-precision reference, and the name of its library. */
struct TapedCase {
  std::string name;
  CppAdInterface::ad_function_t taped;
  ReferenceFunction reference;
};

std::vector<TapedCase> tapedCases() {
  return {
      TapedCase{
          .name = "wide",
          .taped = [](const ad_vector_t& x, ad_vector_t& y) { y = wideFunction<ad_scalar_t>(x); },
          .reference = [](const vector_t& x) { return wideFunction<scalar_t>(x); },
      },
      TapedCase{
          .name = "narrow",
          .taped = [](const ad_vector_t& x, ad_vector_t& y) { y = narrowFunction<ad_scalar_t>(x); },
          .reference = [](const vector_t& x) { return narrowFunction<scalar_t>(x); },
      },
  };
}

/** The operations of the tape of `function` at x = ones, as CppAdInterface::createModels records it. */
size_t tapeOperations(const CppAdInterface::ad_function_t& function, Eigen::Index variableDim, bool recordCompare) {
  ad_vector_t x = ad_vector_t::Ones(variableDim);
  CppAD::Independent(x, /*abort_op_index=*/0, recordCompare);
  ad_vector_t y;
  function(x, y);
  const ad_fun_t fun(x, y);
  return fun.size_op();
}

/** The comparisons the tape records: the operations only there when comparisons are recorded. */
size_t recordedComparisons(const CppAdInterface::ad_function_t& function, Eigen::Index variableDim) {
  return tapeOperations(function, variableDim, /*recordCompare=*/true) - tapeOperations(function, variableDim, /*recordCompare=*/false);
}

/**
 * Runs the tape forward on CppADCodeGen variables, where code generation throws on a comparison between variables
 * ("... cannot be called for non-parameters"). Empty when it does not throw.
 */
std::string codeGenerationError(const CppAdInterface::ad_function_t& function, Eigen::Index variableDim) {
  ad_vector_t x = ad_vector_t::Ones(variableDim);
  CppAD::Independent(x);
  ad_vector_t y;
  function(x, y);
  ad_fun_t fun(x, y);
  CppAD::cg::CodeHandler<scalar_t> handler;
  CppAD::vector<ad_base_t> variables(static_cast<size_t>(variableDim));
  handler.makeVariables(variables);
  try {
    fun.Forward(/*q=*/0, variables);
  } catch (const std::exception& error) {
    return error.what();
  }
  return std::string();
}

std::unique_ptr<CppAdInterface> generate(const CppAdInterface::ad_function_t& function, absl::string_view name) {
  std::unique_ptr<CppAdInterface> library =
      std::make_unique<CppAdInterface>(function, static_cast<size_t>(kVariableDim), absl::StrCat("testBaseOrientationCppAd_", name),
                                       absl::StrCat(testing::TempDir(), "cppad_base_orientation"));
  library->createModels(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);
  return library;
}

matrix_t finiteDifferenceJacobian(const ReferenceFunction& function, const vector_t& x, scalar_t step) {
  const vector_t value = function(x);
  matrix_t jacobian(value.size(), x.size());
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    vector_t forward = x;
    vector_t backward = x;
    forward(i) += step;
    backward(i) -= step;
    jacobian.col(i) = (function(forward) - function(backward)) / (2.0 * step);
  }
  return jacobian;
}

vector_t variables(const vector4_t& xi, const vector3_t& omega) {
  vector_t x(kVariableDim);
  x << xi, omega;
  return x;
}

/** x = ones (the tape point, xi_hat = (1, 1, 1, 1) / 2), two chosen points and random ones. */
std::vector<vector_t> evaluationPoints() {
  std::vector<vector_t> points = {
      vector_t::Ones(kVariableDim),
      variables(vector4_t(0.1, -0.7, 0.3, 0.6), vector3_t(0.5, -1.0, 2.0)),
      variables(vector4_t(0.0, 0.0, 0.0, 2.5), vector3_t(-0.3, 0.2, 0.1)),
  };
  std::mt19937 generator(21);
  std::uniform_real_distribution<scalar_t> uniform(-2.0, 2.0);
  for (int i = 0; i < 8; ++i) {
    vector_t x(kVariableDim);
    for (Eigen::Index j = 0; j < kVariableDim; ++j) x(j) = uniform(generator);
    points.push_back(x);
  }
  return points;
}

/** The zero quaternion and one below epsilon, where n_safe = 1 is constant. */
std::vector<vector_t> pointsBelowTheGuard() {
  return {variables(vector4_t(vector4_t::Zero()), vector3_t(0.4, -0.9, 1.3)),
          variables(vector4_t(1.0e-7 * vector4_t(0.3, -0.5, 0.1, 0.8)), vector3_t(-1.1, 0.2, 0.6))};
}

TEST(BaseOrientationCppAd, TapesRecordNoComparisonBetweenVariables) {
  // Positive control: the detectors see the comparison Eigen's normalized() records on an AD scalar.
  const CppAdInterface::ad_function_t eigenNormalized = [](const ad_vector_t& x, ad_vector_t& y) {
    y = ad_vector4_t(x.head<kQuaternionDim>()).normalized();
  };
  ASSERT_GE(recordedComparisons(eigenNormalized, kQuaternionDim), 1u);
  const std::string eigenNormalizedError = codeGenerationError(eigenNormalized, kQuaternionDim);
  EXPECT_NE(eigenNormalizedError.find("non-parameters"), std::string::npos) << eigenNormalizedError;

  for (const TapedCase& tapedCase : tapedCases()) {
    EXPECT_EQ(recordedComparisons(tapedCase.taped, kVariableDim), 0u) << tapedCase.name;
    EXPECT_EQ(codeGenerationError(tapedCase.taped, kVariableDim), "") << tapedCase.name;
  }
}

TEST(BaseOrientationCppAd, GeneratedLibrariesMatchTheDoubleOverloads) {
  for (const TapedCase& tapedCase : tapedCases()) {
    const std::unique_ptr<CppAdInterface> library = generate(tapedCase.taped, tapedCase.name);
    for (const vector_t& x : evaluationPoints()) {
      EXPECT_LT((library->getFunctionValue(x) - tapedCase.reference(x)).cwiseAbs().maxCoeff(), kValueTolerance)
          << tapedCase.name << " at " << x.transpose();
      EXPECT_LT((library->getJacobian(x) - finiteDifferenceJacobian(tapedCase.reference, x, kFiniteDifferenceStep)).cwiseAbs().maxCoeff(),
                kFiniteDifferenceTolerance)
          << tapedCase.name << " at " << x.transpose();
    }
    // Below epsilon and at the zero quaternion: finite, and the derivatives of the branch n_safe = 1.
    for (const vector_t& x : pointsBelowTheGuard()) {
      const vector_t value = library->getFunctionValue(x);
      ASSERT_TRUE(value.allFinite()) << tapedCase.name << " at " << x.transpose();
      EXPECT_LT((value - tapedCase.reference(x)).cwiseAbs().maxCoeff(), kValueTolerance) << tapedCase.name;
      const matrix_t jacobian = library->getJacobian(x);
      ASSERT_TRUE(jacobian.allFinite()) << tapedCase.name << " at " << x.transpose() << "\n" << jacobian;
      EXPECT_LT((jacobian - finiteDifferenceJacobian(tapedCase.reference, x, kSmallFiniteDifferenceStep)).cwiseAbs().maxCoeff(),
                kFiniteDifferenceTolerance)
          << tapedCase.name << " at " << x.transpose() << "\n"
          << jacobian;
    }
  }
  // The wide function's value at the tape point, and d xi_hat / d xi = I, d n_safe / d xi = 0 below epsilon.
  const std::unique_ptr<CppAdInterface> wide = generate(tapedCases()[0].taped, tapedCases()[0].name);
  EXPECT_LT((wide->getFunctionValue(vector_t::Ones(kVariableDim)).head<4>() - 0.5 * vector4_t::Ones()).cwiseAbs().maxCoeff(),
            kValueTolerance);
  for (const vector_t& x : pointsBelowTheGuard()) {
    const matrix_t jacobian = wide->getJacobian(x);
    EXPECT_LT((jacobian.topLeftCorner<4, 4>() - matrix4_t::Identity()).cwiseAbs().maxCoeff(), kValueTolerance);
    EXPECT_LT(jacobian.row(4).cwiseAbs().maxCoeff(), kValueTolerance);
  }
}

TEST(BaseOrientationCppAd, ReverseModeJacobianIsFiniteAtTheZeroQuaternionOnlyWithTheSquaredNormGuard) {
  const vector_t zero = pointsBelowTheGuard()[0];
  const std::unique_ptr<CppAdInterface> squaredNormGuard = generate(tapedCases()[1].taped, "narrow_squared_norm_guard");
  EXPECT_TRUE(squaredNormGuard->getJacobian(zero).allFinite()) << squaredNormGuard->getJacobian(zero);

  // Positive control: the same narrow function on the guard of the norm. Its square root's reverse sweep multiplies the
  // zero partial of the unselected branch by 1 / (2 |xi|) = inf in the generated code, so the Jacobian is NaN. This is
  // what shows that the Jacobians above were generated in reverse mode.
  const CppAdInterface::ad_function_t normGuarded = [](const ad_vector_t& x, ad_vector_t& y) {
    y = narrowFunctionOf<ad_scalar_t>(normGuardedNormalization(ad_vector4_t(x.head<kQuaternionDim>())), x);
  };
  EXPECT_EQ(recordedComparisons(normGuarded, kVariableDim), 0u);
#ifdef NDEBUG
  // Only with assertions off: CppAdInterface::getJacobian asserts a finite Jacobian otherwise, and the control expects
  // exactly the non-finite one (as the positive control of testQuaternionRootJointCppAd.cpp).
  const std::unique_ptr<CppAdInterface> normGuard = generate(normGuarded, "narrow_norm_guard");
  EXPECT_TRUE(normGuard->getFunctionValue(zero).allFinite());
  EXPECT_FALSE(normGuard->getJacobian(zero).allFinite()) << normGuard->getJacobian(zero);
#endif
}

}  // namespace
}  // namespace ocs2::humanoid
