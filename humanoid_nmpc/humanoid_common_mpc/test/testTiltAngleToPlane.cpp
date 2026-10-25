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

#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "Eigen/Geometry"
#include "gtest/gtest.h"
#include "ocs2_core/automatic_differentiation/CppAdInterface.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"

/*
 * The tilt residual of the foot costs (tiltAngleToPlane()): the rotation vector
 * theta u in world axes that turns the plane normal n into the end-effector
 * normal v = R e_z, against an independent construction (AngleAxis(theta, u) n
 * = v, u orthogonal to n, theta the angle between the two); blind to the
 * rotation about the foot's own normal; equivariant under a world rotation;
 * twice the half-angle residual it replaced to third order in the tilt; and its
 * CppAD tape, the one the foot costs generate, against the double overload and
 * central differences along a world perturbation of R, at the plane (v = n, the
 * series branch of the logarithm, where the Jacobian is the projection onto the
 * plane), away from it and next to the half turn (v = -n, the shortest arc's
 * clamp), with no comparison between variables recorded. It compiles one small
 * library, so it runs alone (tags = ["exclusive"]).
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kTol = 1.0e-12;
constexpr scalar_t kStep = 1.0e-6;
constexpr scalar_t kFiniteDifferenceTolerance = 1.0e-7;
// The tape's variables (a world rotation vector applied to R0) and parameters
// (R0 column by column, then n).
constexpr Eigen::Index kVariableDim = kRotationTangentDim;
constexpr Eigen::Index kRotationEntries = kPositionDim * kPositionDim;
constexpr Eigen::Index kParameterDim = kRotationEntries + kPositionDim;

vector3_t randomUnitVector(std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  return vector3_t(normal(generator), normal(generator), normal(generator))
      .normalized();
}

matrix3_t randomRotation(std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  return Eigen::Quaterniond(vector4_t(normal(generator), normal(generator),
                                      normal(generator), normal(generator)))
      .normalized()
      .toRotationMatrix();
}

/** exp([phi]x) R0: R0 turned by the world rotation vector phi. */
template <typename SCALAR_T>
MATRIX3_T<SCALAR_T> worldRotated(const VECTOR3_T<SCALAR_T>& phi,
                                 const MATRIX3_T<SCALAR_T>& rotation) {
  return quaternionRotationMatrix<SCALAR_T>(quaternionExp<SCALAR_T>(phi)) *
         rotation;
}

/** The tilt of exp([x]x) R0 against n, with p = [R0 (column-major), n]: the
 * function the tape is recorded from. */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> perturbedTilt(
    const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& x,
    const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& p) {
  const MATRIX3_T<SCALAR_T> rotation =
      Eigen::Map<const MATRIX3_T<SCALAR_T>>(p.data());
  const VECTOR3_T<SCALAR_T> planeNormal =
      p.template segment<kPositionDim>(kRotationEntries);
  return tiltAngleToPlane<SCALAR_T>(
      worldRotated<SCALAR_T>(
          VECTOR3_T<SCALAR_T>(x.template head<kRotationTangentDim>()),
          rotation),
      planeNormal);
}

vector_t parametersOf(const matrix3_t& rotation, const vector3_t& planeNormal) {
  vector_t p(kParameterDim);
  p << Eigen::Map<const vector_t>(rotation.data(), kRotationEntries),
      planeNormal;
  return p;
}

/** A rotation whose z axis is `normal`: the foot standing flat on the plane of
 * that normal, at a random heading. */
matrix3_t standingOn(const vector3_t& normal, std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> heading(-M_PI, M_PI);
  const matrix3_t arc =
      Eigen::Quaterniond::FromTwoVectors(vector3_t::UnitZ(), normal)
          .toRotationMatrix();
  return arc * Eigen::AngleAxisd(heading(generator), vector3_t::UnitZ())
                   .toRotationMatrix();
}

}  // namespace

TEST(TiltAngleToPlane, IsTheAngleAboutTheAxisThatTurnsTheNormalIntoTheFoot) {
  std::mt19937 generator(29);
  for (int trial = 0; trial < 1000; ++trial) {
    const matrix3_t rotation = randomRotation(generator);
    const vector3_t planeNormal = randomUnitVector(generator);
    const vector3_t footNormal = rotation.col(2);
    const vector3_t tilt = tiltAngleToPlane<scalar_t>(rotation, planeNormal);
    const scalar_t angle = std::atan2(planeNormal.cross(footNormal).norm(),
                                      planeNormal.dot(footNormal));
    EXPECT_NEAR(tilt.norm(), angle, kTol) << "trial " << trial;
    EXPECT_NEAR(tilt.dot(planeNormal), 0.0, kTol)
        << "trial " << trial << ": the tilt has no part about the plane normal";
    const vector3_t turned =
        Eigen::AngleAxisd(tilt.norm(), tilt.normalized()) * planeNormal;
    EXPECT_LT((turned - footNormal).norm(), 1.0e-11)
        << "trial " << trial << ": exp([e]x) n is the foot normal";
  }
  // Standing flat on the plane, at any heading, the tilt is zero.
  for (int trial = 0; trial < 100; ++trial) {
    const vector3_t planeNormal = randomUnitVector(generator);
    EXPECT_LT(tiltAngleToPlane<scalar_t>(standingOn(planeNormal, generator),
                                         planeNormal)
                  .norm(),
              kTol);
  }
}

TEST(TiltAngleToPlane,
     IgnoresTheRotationAboutTheFootNormalAndFollowsAWorldRotation) {
  std::mt19937 generator(31);
  std::uniform_real_distribution<scalar_t> angle(-M_PI, M_PI);
  for (int trial = 0; trial < 500; ++trial) {
    const matrix3_t rotation = randomRotation(generator);
    const vector3_t planeNormal = randomUnitVector(generator);
    const vector3_t tilt = tiltAngleToPlane<scalar_t>(rotation, planeNormal);
    // A yaw of the foot about its own z axis leaves its normal, and so its
    // tilt, where they are.
    const matrix3_t yawed =
        rotation * Eigen::AngleAxisd(angle(generator), vector3_t::UnitZ())
                       .toRotationMatrix();
    EXPECT_LT((tiltAngleToPlane<scalar_t>(yawed, planeNormal) - tilt).norm(),
              kTol)
        << "trial " << trial;
    // The foot and the plane turned together by Q: the tilt turns with them.
    const matrix3_t world = randomRotation(generator);
    EXPECT_LT((tiltAngleToPlane<scalar_t>(matrix3_t(world * rotation),
                                          vector3_t(world * planeNormal)) -
               world * tilt)
                  .norm(),
              1.0e-11)
        << "trial " << trial;
  }
}

TEST(TiltAngleToPlane, WeighsTheAngleNotItsHalfSine) {
  // The residual it replaced, rotationMatrixDistanceToPlane() = sin(theta / 2)
  // u, is half the angle at small tilts: a weight w on it is w / 4 on this
  // residual (the rescaled task_space_foot_cost orientation weights).
  std::mt19937 generator(37);
  std::uniform_real_distribution<scalar_t> smallAngle(1.0e-6, 1.0e-2);
  for (int trial = 0; trial < 200; ++trial) {
    const vector3_t planeNormal = randomUnitVector(generator);
    const vector3_t axis = randomUnitVector(generator);
    const matrix3_t rotation =
        worldRotated<scalar_t>(vector3_t(smallAngle(generator) * axis),
                               standingOn(planeNormal, generator));
    const vector3_t tilt = tiltAngleToPlane<scalar_t>(rotation, planeNormal);
    const vector3_t halfSine =
        rotationMatrixDistanceToPlane<scalar_t>(rotation, planeNormal);
    const scalar_t theta = tilt.norm();
    // theta - 2 sin(theta / 2) = theta^3 / 24 + O(theta^5), along the same
    // axis.
    EXPECT_LT((tilt - 2.0 * halfSine).norm(),
              theta * theta * theta / 12.0 + 1.0e-15)
        << "trial " << trial;
  }
}

TEST(TiltAngleToPlane, TheTapeMatchesTheDoubleOverloadAndItsDerivative) {
  const CppAdInterface::ad_parameterized_function_t taped =
      [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
        y = perturbedTilt<ad_scalar_t>(x, p);
      };
  // No comparison between variables, which CppADCodeGen cannot generate:
  // recording comparisons adds no operation.
  const std::function<size_t(bool)> tapeOperations =
      [&taped](bool recordCompare) {
        ad_vector_t xp = ad_vector_t::Ones(kVariableDim + kParameterDim);
        CppAD::Independent(xp, /*abort_op_index=*/0, recordCompare);
        ad_vector_t y;
        taped(xp.head(kVariableDim), xp.tail(kParameterDim), y);
        const CppAD::ADFun<ad_base_t> fun(xp, y);
        return fun.size_op();
      };
  EXPECT_EQ(tapeOperations(/*recordCompare=*/true),
            tapeOperations(/*recordCompare=*/false));

  CppAdInterface library(taped, static_cast<size_t>(kVariableDim),
                         static_cast<size_t>(kParameterDim),
                         "testTiltAngleToPlane",
                         testing::TempDir() + "cppad_tilt_angle_to_plane");
  library.createModels(CppAdInterface::ApproximationOrder::First,
                       /*verbose=*/false);

  const std::function<matrix_t(const vector_t&, const vector_t&)>
      finiteDifference = [](const vector_t& x, const vector_t& p) {
        matrix_t jacobian(kRotationTangentDim, kVariableDim);
        for (Eigen::Index i = 0; i < kVariableDim; ++i) {
          vector_t plus = x;
          vector_t minus = x;
          plus(i) += kStep;
          minus(i) -= kStep;
          jacobian.col(i) = (perturbedTilt<scalar_t>(plus, p) -
                             perturbedTilt<scalar_t>(minus, p)) /
                            (2.0 * kStep);
        }
        return jacobian;
      };

  std::mt19937 generator(41);
  std::uniform_real_distribution<scalar_t> perturbation(-0.5, 0.5);
  const vector_t zero = vector_t::Zero(kVariableDim);
  for (int trial = 0; trial < 50; ++trial) {
    const vector3_t planeNormal = randomUnitVector(generator);
    // Anywhere, at the derivative's own point x.
    const vector_t anywhere =
        parametersOf(randomRotation(generator), planeNormal);
    const vector_t x = (vector_t(kVariableDim) << perturbation(generator),
                        perturbation(generator), perturbation(generator))
                           .finished();
    EXPECT_LT((library.getFunctionValue(x, anywhere) -
               perturbedTilt<scalar_t>(x, anywhere))
                  .cwiseAbs()
                  .maxCoeff(),
              kTol)
        << trial;
    EXPECT_LT((library.getJacobian(x, anywhere) - finiteDifference(x, anywhere))
                  .cwiseAbs()
                  .maxCoeff(),
              kFiniteDifferenceTolerance)
        << trial;

    // Flat on the plane: the series branch of the logarithm. To first order e =
    // n x (phi x n), the projection of phi onto the plane, so the Jacobian is I
    // - n n^T.
    const vector_t flat =
        parametersOf(standingOn(planeNormal, generator), planeNormal);
    EXPECT_LT(library.getFunctionValue(zero, flat).norm(), kTol);
    const matrix_t projection =
        matrix3_t::Identity() - planeNormal * planeNormal.transpose();
    EXPECT_LT(
        (library.getJacobian(zero, flat) - projection).cwiseAbs().maxCoeff(),
        1.0e-10)
        << trial;
    EXPECT_LT((finiteDifference(zero, flat) - projection).cwiseAbs().maxCoeff(),
              kFiniteDifferenceTolerance)
        << trial;

    // Upside down, v = -n, inside the shortest arc's clamp: value and
    // derivatives finite (the axis of the half turn is not defined, and the
    // residual is 0 exactly there), and just outside the clamp the tilt is
    // nearly pi.
    const vector_t upsideDown = parametersOf(
        standingOn(vector3_t(-planeNormal), generator), planeNormal);
    EXPECT_TRUE(library.getFunctionValue(zero, upsideDown).allFinite());
    EXPECT_TRUE(library.getJacobian(zero, upsideDown).allFinite());
    const vector3_t nearHalfTurnAxis = planeNormal.unitOrthogonal();
    const vector_t nearHalfTurn =
        (vector_t(kVariableDim) << 1.0e-2 * nearHalfTurnAxis).finished();
    EXPECT_NEAR(library.getFunctionValue(nearHalfTurn, upsideDown).norm(),
                M_PI - 1.0e-2, 1.0e-9)
        << trial;
    EXPECT_TRUE(library.getJacobian(nearHalfTurn, upsideDown).allFinite());
  }
}

}  // namespace ocs2::humanoid
