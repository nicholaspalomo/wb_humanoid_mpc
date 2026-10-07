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

#include <gtest/gtest.h>

#include <cppad/cppad.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Geometry>

#include <ocs2_core/manifold/EuclideanStateManifold.h>
#include <ocs2_core/manifold/ProductStateManifold.h>
#include <ocs2_core/manifold/UnitQuaternionMath.h>

namespace ocs2 {
namespace {

using quaternion_t = quaternion_coeffs_t<scalar_t>;
using rotation_t = rotation_vector_t<scalar_t>;

constexpr scalar_t kPi = 3.14159265358979323846;

quaternion_t randomUnitQuaternion() {
  return quaternion_t::Random().normalized();
}

rotation_t rotationVector(scalar_t x, scalar_t y, scalar_t z) {
  rotation_t phi;
  phi << x, y, z;
  return phi;
}

Eigen::Quaterniond eigenQuaternion(const quaternion_t& xi) {
  Eigen::Quaterniond q;
  q.coeffs() = xi;
  return q;
}

/** A rotation's distance, insensitive to the sign of the quaternion. */
scalar_t rotationDistance(const quaternion_t& a, const quaternion_t& b) {
  return quaternionLog(quaternionProduct(quaternionConjugate(a), b)).norm();
}

// ---------------------------------------------------------------------------------------------------------------------
// UnitQuaternionMath
// ---------------------------------------------------------------------------------------------------------------------

TEST(UnitQuaternionMath, RateMatrixIdentities) {
  for (int trial = 0; trial < 20; ++trial) {
    const quaternion_t xi = 1.7 * quaternion_t::Random();
    const rotation_t omega = rotation_t::Random();
    const Eigen::Matrix<scalar_t, 4, 3> G = quaternionRateMatrix(xi);
    quaternion_t pureOmega;
    pureOmega << omega, 0.0;
    EXPECT_TRUE((G * omega).isApprox(quaternionProduct(xi, pureOmega), 1e-14));
    EXPECT_TRUE((G.transpose() * G).isApprox(xi.squaredNorm() * Eigen::Matrix3d::Identity(), 1e-14));
    EXPECT_LT((G.transpose() * xi).norm(), 1e-14);
  }
}

TEST(UnitQuaternionMath, ProductAndRotationMatrixMatchEigen) {
  for (int trial = 0; trial < 20; ++trial) {
    const quaternion_t a = randomUnitQuaternion();
    const quaternion_t b = randomUnitQuaternion();
    EXPECT_TRUE(quaternionProduct(a, b).isApprox((eigenQuaternion(a) * eigenQuaternion(b)).coeffs(), 1e-14));
    EXPECT_TRUE(quaternionRotationMatrix(a).isApprox(eigenQuaternion(a).toRotationMatrix(), 1e-14));
    EXPECT_TRUE(quaternionProduct(a, quaternionConjugate(a)).isApprox(quaternion_t(0.0, 0.0, 0.0, 1.0), 1e-14));
  }
  EXPECT_TRUE(quaternionRotationMatrix(quaternion_t::Zero().eval()).isApprox(Eigen::Matrix3d::Identity()));
}

TEST(UnitQuaternionMath, ExpMatchesAngleAxisAndLogInvertsIt) {
  for (int trial = 0; trial < 50; ++trial) {
    const rotation_t phi = 0.99 * kPi * rotation_t::Random().normalized() * (0.5 + 0.5 * std::abs(std::sin(trial)));
    const quaternion_t xi = quaternionExp(phi);
    EXPECT_NEAR(xi.norm(), 1.0, 1e-15);
    const Eigen::AngleAxisd angleAxis(phi.norm(), phi.normalized());
    EXPECT_LT(rotationDistance(xi, Eigen::Quaterniond(angleAxis).coeffs()), 1e-14);
    EXPECT_TRUE(quaternionLog(xi).isApprox(phi, 1e-12));
    // Shortest path and double cover.
    EXPECT_TRUE(quaternionLog((-xi).eval()).isApprox(phi, 1e-12));
    // Homogeneous of degree 0.
    EXPECT_TRUE(quaternionLog((3.0 * xi).eval()).isApprox(phi, 1e-12));
  }
  // Beyond pi, Log returns the shorter way round.
  const rotation_t longWay = rotationVector(/*x=*/0.0, /*y=*/0.0, /*z=*/1.5 * kPi);
  EXPECT_TRUE(quaternionLog(quaternionExp(longWay)).isApprox(rotationVector(/*x=*/0.0, /*y=*/0.0, /*z=*/-0.5 * kPi), 1e-12));
  EXPECT_LE(quaternionLog(randomUnitQuaternion()).norm(), kPi + 1e-12);
}

TEST(UnitQuaternionMath, SmallAngleSeriesAreContinuous) {
  for (const scalar_t angle : {0.0, 1e-12, 1e-8, 0.999e-6, 1.001e-6, 1e-4, 0.0999, 0.1001, 0.3}) {
    const rotation_t phi = angle * rotationVector(/*x=*/0.6, /*y=*/-0.48, /*z=*/0.64);
    // Exp and Log
    const scalar_t halfAngle = 0.5 * angle;
    EXPECT_NEAR(quaternionExp(phi)(3), std::cos(halfAngle), 1e-15);
    EXPECT_NEAR(quaternionExp(phi).head<3>().norm(), std::sin(halfAngle), 1e-15);
    EXPECT_LT((quaternionLog(quaternionExp(phi)) - phi).norm(), 1e-15 + 1e-14 * angle);
    // Jr Jr^-1 = I on both sides of the series thresholds.
    EXPECT_TRUE((so3RightJacobian(phi) * so3RightJacobianInverse(phi)).isApprox(Eigen::Matrix3d::Identity(), 1e-13)) << angle;
  }
  // The zero quaternion maps to the zero rotation vector, not NaN.
  EXPECT_TRUE(quaternionLog(quaternion_t::Zero().eval()).isZero());
}

TEST(UnitQuaternionMath, RightJacobiansMatchFiniteDifferences) {
  const scalar_t eps = 1e-6;
  for (const rotation_t& phi : {rotationVector(/*x=*/0.3, /*y=*/-0.2, /*z=*/0.5), rotationVector(/*x=*/1.2, /*y=*/0.4, /*z=*/-2.1),
                                rotationVector(/*x=*/0.0, /*y=*/0.0, /*z=*/3.1), rotationVector(/*x=*/1e-3, /*y=*/-2e-3, /*z=*/5e-4),
                                rotationVector(/*x=*/0.05, /*y=*/0.02, /*z=*/-0.06)}) {
    Eigen::Matrix3d JrNumeric;
    Eigen::Matrix3d JrInverseNumeric;
    for (int i = 0; i < 3; ++i) {
      const rotation_t step = eps * rotation_t::Unit(i);
      // Exp(phi + d) = Exp(phi) Exp(Jr d)
      const rotation_t plus = quaternionLog(quaternionProduct(quaternionConjugate(quaternionExp(phi)), quaternionExp((phi + step).eval())));
      const rotation_t minus =
          quaternionLog(quaternionProduct(quaternionConjugate(quaternionExp(phi)), quaternionExp((phi - step).eval())));
      JrNumeric.col(i) = (plus - minus) / (2.0 * eps);
      // Log(Exp(phi) Exp(d)) = phi + Jr^-1 d
      JrInverseNumeric.col(i) = (quaternionLog(quaternionProduct(quaternionExp(phi), quaternionExp(step))) -
                                 quaternionLog(quaternionProduct(quaternionExp(phi), quaternionExp((-step).eval())))) /
                                (2.0 * eps);
    }
    EXPECT_LT((so3RightJacobian(phi) - JrNumeric).norm(), 1e-8) << phi.transpose();
    EXPECT_LT((so3RightJacobianInverse(phi) - JrInverseNumeric).norm(), 1e-8) << phi.transpose();
  }
  // Regular at a half turn, where (1 + cos a) / (2 a sin a) would be 0 / 0.
  const Eigen::Matrix3d atHalfTurn = so3RightJacobianInverse(rotationVector(/*x=*/0.0, /*y=*/kPi, /*z=*/0.0));
  EXPECT_TRUE(atHalfTurn.allFinite());
  EXPECT_NEAR(atHalfTurn(0, 0), 1.0 - kPi * kPi / (kPi * kPi), 1e-12);
}

TEST(UnitQuaternionMath, TangentMapsAreInverse) {
  for (int trial = 0; trial < 20; ++trial) {
    const quaternion_t xi = (0.5 + trial * 0.1) * randomUnitQuaternion();
    const Eigen::Matrix<scalar_t, 4, 3> E = quaternionTangentMap(xi);
    const Eigen::Matrix<scalar_t, 3, 4> EPlus = quaternionTangentMapPseudoInverse(xi);
    EXPECT_TRUE((EPlus * E).isApprox(Eigen::Matrix3d::Identity(), 1e-14));
    EXPECT_LT((EPlus * xi).norm(), 1e-14);
  }
}

TEST(UnitQuaternionMath, SlerpEndpointsAndAntipodes) {
  const quaternion_t xi0 = randomUnitQuaternion();
  const quaternion_t xi1 = quaternionProduct(xi0, quaternionExp(rotationVector(/*x=*/0.4, /*y=*/-0.9, /*z=*/1.3)));
  EXPECT_LT(rotationDistance(quaternionSlerp(xi0, xi1, /*alpha=*/0.0), xi0), 1e-15);
  EXPECT_LT(rotationDistance(quaternionSlerp(xi0, xi1, /*alpha=*/1.0), xi1), 1e-14);
  const scalar_t angle = rotationDistance(xi0, xi1);
  EXPECT_NEAR(rotationDistance(xi0, quaternionSlerp(xi0, xi1, /*alpha=*/0.25)), 0.25 * angle, 1e-14);
  EXPECT_NEAR(quaternionSlerp(xi0, xi1, /*alpha=*/0.25).norm(), 1.0, 1e-15);
  // -xi1 is the same rotation: the same path.
  EXPECT_LT(rotationDistance(quaternionSlerp(xi0, xi1, /*alpha=*/0.4), quaternionSlerp(xi0, (-xi1).eval(), /*alpha=*/0.4)), 1e-14);
  // Just short of and just past a half turn the path goes the short way, so it is never longer than pi / 2 at alpha 0.5.
  for (const scalar_t turn : {kPi - 1e-6, kPi + 1e-6}) {
    const quaternion_t far = quaternionProduct(xi0, quaternionExp(rotationVector(/*x=*/0.0, /*y=*/0.0, /*z=*/turn)));
    EXPECT_LE(rotationDistance(xi0, quaternionSlerp(xi0, far, /*alpha=*/0.5)), 0.5 * kPi + 1e-9);
  }
  // Rotations about one axis interpolate linearly in the angle.
  const quaternion_t yaw0 = quaternionExp(rotationVector(/*x=*/0.0, /*y=*/0.0, /*z=*/0.2));
  const quaternion_t yaw1 = quaternionExp(rotationVector(/*x=*/0.0, /*y=*/0.0, /*z=*/1.0));
  EXPECT_TRUE(quaternionLog(quaternionSlerp(yaw0, yaw1, /*alpha=*/0.3)).isApprox(rotationVector(/*x=*/0.0, /*y=*/0.0, /*z=*/0.44), 1e-14));
}

TEST(UnitQuaternionMath, SafeNormalization) {
  const quaternion_t xi = 2.5 * randomUnitQuaternion();
  EXPECT_NEAR(quaternionSafeNormalize(xi).norm(), 1.0, 1e-15);
  EXPECT_TRUE(quaternionSafeNormalize(quaternion_t::Zero().eval()).isZero());
  EXPECT_DOUBLE_EQ(quaternionSafeNorm(quaternion_t::Zero().eval()), 1.0);
}

/**
 * On a CppAD tape recorded at ones, every function must be valid at every input, also where its branches switch, and
 * its reverse-mode Jacobian must be finite there (the sqrt of a guarded squared norm, not a guarded sqrt).
 */
TEST(UnitQuaternionMath, TapedAtOnesIsValidEverywhere) {
  using ad_t = CppAD::AD<scalar_t>;
  using ad_vector_t = Eigen::Matrix<ad_t, Eigen::Dynamic, 1>;

  // y = [Log(xi); safeNormalize(xi); vec(Jr^-1(Log(xi))); vec(Jr(Log xi)); Exp(Log(xi)); vec(G(xi))]
  ad_vector_t xi = ad_vector_t::Ones(4);
  CppAD::Independent(xi);
  const quaternion_coeffs_t<ad_t> q = xi;
  const rotation_vector_t<ad_t> log = quaternionLog(q);
  const Eigen::Matrix<ad_t, 3, 3> jrInverse = so3RightJacobianInverse(log);
  const Eigen::Matrix<ad_t, 3, 3> jr = so3RightJacobian(log);
  const Eigen::Matrix<ad_t, 4, 3> rate = quaternionRateMatrix(q);
  ad_vector_t y(3 + 4 + 9 + 9 + 4 + 12);
  y << log, quaternionSafeNormalize(q), Eigen::Map<const Eigen::Matrix<ad_t, 9, 1>>(jrInverse.data()),
      Eigen::Map<const Eigen::Matrix<ad_t, 9, 1>>(jr.data()), quaternionExp(log), Eigen::Map<const Eigen::Matrix<ad_t, 12, 1>>(rate.data());
  CppAD::ADFun<scalar_t> tape(xi, y);

  const std::vector<quaternion_t> points = {randomUnitQuaternion(),
                                            (-randomUnitQuaternion()).eval(),
                                            quaternion_t(0.0, 0.0, 0.0, 1.0),
                                            quaternion_t(0.0, 0.0, 0.0, -1.0),
                                            quaternion_t(1e-9, -2e-9, 0.0, 1.0),
                                            quaternion_t(1.0, 0.0, 0.0, 0.0),
                                            quaternion_t::Zero(),
                                            quaternion_t(0.0, 0.03, -0.04, 0.998)};
  for (const quaternion_t& point : points) {
    const std::vector<scalar_t> input(point.data(), point.data() + 4);
    const std::vector<scalar_t> taped = tape.Forward(/*q=*/0, input);
    const rotation_t expectedLog = quaternionLog(point);
    const Eigen::Matrix3d expectedJrInverse = so3RightJacobianInverse(expectedLog);
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR(taped[i], expectedLog(i), 1e-14) << point.transpose();
    }
    for (int i = 0; i < 4; ++i) {
      EXPECT_NEAR(taped[3 + i], quaternionSafeNormalize(point)(i), 1e-14) << point.transpose();
    }
    for (int i = 0; i < 9; ++i) {
      EXPECT_NEAR(taped[7 + i], expectedJrInverse.data()[i], 1e-12) << point.transpose();
    }
    const std::vector<scalar_t> jacobian = tape.Jacobian(input);
    for (const scalar_t value : jacobian) {
      EXPECT_TRUE(std::isfinite(value)) << point.transpose();
    }
    // Reverse mode, which is how the costs are differentiated.
    tape.Forward(/*q=*/0, input);
    for (size_t row = 0; row < y.size(); ++row) {
      std::vector<scalar_t> weight(y.size(), 0.0);
      weight[row] = 1.0;
      for (const scalar_t value : tape.Reverse(/*q=*/1, weight)) {
        EXPECT_TRUE(std::isfinite(value)) << "row " << row << " at " << point.transpose();
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// ProductStateManifold
// ---------------------------------------------------------------------------------------------------------------------

/** [E(3), Q, E(2), Q, E(1)]: two quaternions, so offsets that differ between ambient and tangent are exercised. */
std::shared_ptr<const ProductStateManifold> twoQuaternionManifold() {
  return ProductStateManifold::create({StateManifoldSegment::euclidean(2), StateManifoldSegment::euclidean(1),
                                       StateManifoldSegment::unitQuaternion(), StateManifoldSegment::euclidean(2),
                                       StateManifoldSegment::unitQuaternion(), StateManifoldSegment::euclidean(1)})
      .value();
}

vector_t randomState(const ProductStateManifold& manifold) {
  vector_t x = vector_t::Random(manifold.getAmbientDim());
  manifold.project(x);
  return x;
}

/** The dense E(x) (ambient x tangent), from the quaternion tangent maps. */
matrix_t denseTangentMap(const ProductStateManifold& manifold, const vector_t& x) {
  matrix_t E = matrix_t::Zero(manifold.getAmbientDim(), manifold.getTangentDim());
  size_t ambient = 0;
  size_t tangent = 0;
  for (const StateManifoldSegment& segment : manifold.getSegments()) {
    if (segment.type == StateManifoldSegment::Type::kEuclidean) {
      E.block(ambient, tangent, segment.dimension, segment.dimension).setIdentity();
    } else {
      const quaternion_t xi = x.segment<4>(ambient);
      E.block<4, 3>(ambient, tangent) = quaternionTangentMap(xi);
    }
    ambient += segment.getAmbientDim();
    tangent += segment.getTangentDim();
  }
  return E;
}

matrix_t denseTangentMapPseudoInverse(const ProductStateManifold& manifold, const vector_t& x) {
  matrix_t EPlus = matrix_t::Zero(manifold.getTangentDim(), manifold.getAmbientDim());
  size_t ambient = 0;
  size_t tangent = 0;
  for (const StateManifoldSegment& segment : manifold.getSegments()) {
    if (segment.type == StateManifoldSegment::Type::kEuclidean) {
      EPlus.block(tangent, ambient, segment.dimension, segment.dimension).setIdentity();
    } else {
      const quaternion_t xi = x.segment<4>(ambient);
      EPlus.block<3, 4>(tangent, ambient) = quaternionTangentMapPseudoInverse(xi);
    }
    ambient += segment.getAmbientDim();
    tangent += segment.getTangentDim();
  }
  return EPlus;
}

TEST(ProductStateManifold, LayoutAndValidation) {
  const std::shared_ptr<const ProductStateManifold> manifold = twoQuaternionManifold();
  EXPECT_EQ(manifold->getAmbientDim(), 14u);
  EXPECT_EQ(manifold->getTangentDim(), 12u);
  EXPECT_EQ(manifold->getSegments().size(), 5u);  // the two leading Euclidean segments are merged
  EXPECT_EQ(manifold->getQuaternionAmbientOffsets(), (std::vector<size_t>{3, 9}));
  EXPECT_EQ(manifold->getQuaternionTangentOffsets(), (std::vector<size_t>{3, 8}));

  EXPECT_FALSE(ProductStateManifold::create({}).ok());
  EXPECT_FALSE(ProductStateManifold::create({StateManifoldSegment::euclidean(0)}).ok());
  EXPECT_FALSE(ProductStateManifold::create({StateManifoldSegment{StateManifoldSegment::Type::kUnitQuaternion, 3}}).ok());
}

TEST(ProductStateManifold, RetractAndDifferenceAreInverse) {
  const std::shared_ptr<const ProductStateManifold> manifold = twoQuaternionManifold();
  for (int trial = 0; trial < 20; ++trial) {
    const vector_t x = randomState(*manifold);
    vector_t dx = vector_t::Random(manifold->getTangentDim());
    dx.segment<3>(3) *= 2.5;  // large rotations, clipped below pi (the cut locus of the difference)
    for (const size_t offset : manifold->getQuaternionTangentOffsets()) {
      if (dx.segment<3>(offset).norm() > 0.95 * kPi) {
        dx.segment<3>(offset) *= 0.95 * kPi / dx.segment<3>(offset).norm();
      }
    }
    vector_t xNew;
    manifold->retract(x, dx, /*alpha=*/1.0, xNew);
    EXPECT_TRUE(manifold->difference(x, xNew).isApprox(dx, 1e-12));
    for (const size_t offset : manifold->getQuaternionAmbientOffsets()) {
      EXPECT_NEAR(xNew.segment<4>(offset).norm(), 1.0, 1e-15);
    }
    // x (+) (x1 (-) x) = x1
    const vector_t x1 = randomState(*manifold);
    vector_t x1Again;
    manifold->retract(x, manifold->difference(x, x1), /*alpha=*/1.0, x1Again);
    EXPECT_LT(manifold->difference(x1, x1Again).norm(), 1e-12);
    // alpha scales the step.
    vector_t xHalf;
    manifold->retract(x, dx, /*alpha=*/0.5, xHalf);
    EXPECT_TRUE(manifold->difference(x, xHalf).isApprox(0.5 * dx, 1e-12));
    EXPECT_NEAR(manifold->getMaximumRotationAngle(dx), std::max(dx.segment<3>(3).norm(), dx.segment<3>(8).norm()), 1e-15);
  }
}

TEST(ProductStateManifold, DifferenceIsInvariantUnderTheDoubleCover) {
  const std::shared_ptr<const ProductStateManifold> manifold = twoQuaternionManifold();
  const vector_t x0 = randomState(*manifold);
  const vector_t x1 = randomState(*manifold);
  vector_t x1Flipped = x1;
  x1Flipped.segment<4>(3) *= -1.0;
  vector_t x0Flipped = x0;
  x0Flipped.segment<4>(9) *= -1.0;
  EXPECT_EQ(manifold->difference(x0, x1), manifold->difference(x0, x1Flipped));
  EXPECT_EQ(manifold->difference(x0, x1), manifold->difference(x0Flipped, x1));
}

TEST(ProductStateManifold, InterpolateAndProject) {
  const std::shared_ptr<const ProductStateManifold> manifold = twoQuaternionManifold();
  const vector_t x0 = randomState(*manifold);
  vector_t dx = 0.5 * vector_t::Random(manifold->getTangentDim());
  vector_t x1;
  manifold->retract(x0, dx, /*alpha=*/1.0, x1);
  EXPECT_LT(manifold->difference(x0, manifold->interpolate(x0, x1, /*alpha=*/0.0)).norm(), 1e-14);
  EXPECT_LT(manifold->difference(x1, manifold->interpolate(x0, x1, /*alpha=*/1.0)).norm(), 1e-14);
  // On the geodesic: the midpoint is half the difference away.
  EXPECT_TRUE(manifold->difference(x0, manifold->interpolate(x0, x1, /*alpha=*/0.5)).isApprox(0.5 * dx, 1e-12));

  vector_t x = vector_t::Random(manifold->getAmbientDim());
  const vector_t before = x;
  manifold->project(x);
  EXPECT_NEAR(x.segment<4>(3).norm(), 1.0, 1e-15);
  EXPECT_NEAR(x.segment<4>(9).norm(), 1.0, 1e-15);
  EXPECT_EQ(x.head<3>(), before.head<3>());
  EXPECT_EQ(x.segment<2>(7), before.segment<2>(7));
}

TEST(ProductStateManifold, RetractionCurvatureIdentity) {
  // d^2 (xi (x) Exp(delta)) / d delta_i d delta_j = -xi / 4 delta_ij at delta = 0: the term that vanishes from the
  // pulled-back Hessian of a function that sees the quaternion only through xi / |xi|.
  const quaternion_t xi = randomUnitQuaternion();
  const scalar_t eps = 1e-4;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const rotation_t ei = eps * rotation_t::Unit(i);
      const rotation_t ej = eps * rotation_t::Unit(j);
      const quaternion_t secondDerivative =
          (quaternionProduct(xi, quaternionExp((ei + ej).eval())) - quaternionProduct(xi, quaternionExp((ei - ej).eval())) -
           quaternionProduct(xi, quaternionExp((ej - ei).eval())) + quaternionProduct(xi, quaternionExp((-ei - ej).eval()))) /
          (4.0 * eps * eps);
      const quaternion_t expected = (i == j) ? quaternion_t(-0.25 * xi) : quaternion_t(quaternion_t::Zero());
      EXPECT_LT((secondDerivative - expected).norm(), 1e-7) << i << ", " << j;
    }
  }
}

TEST(ProductStateManifold, InPlacePullBacksEqualDenseProducts) {
  const std::shared_ptr<const ProductStateManifold> manifold = twoQuaternionManifold();
  const size_t nx = manifold->getAmbientDim();
  const size_t ndx = manifold->getTangentDim();
  const vector_t x = randomState(*manifold);
  const matrix_t E = denseTangentMap(*manifold, x);
  const matrix_t EPlus = denseTangentMapPseudoInverse(*manifold, x);
  EXPECT_TRUE((EPlus * E).isApprox(matrix_t::Identity(ndx, ndx), 1e-14));

  for (const size_t rows : {size_t(0), size_t(1), size_t(5)}) {
    const matrix_t J = matrix_t::Random(rows, nx);
    matrix_t pulled = J;
    manifold->pullBackStateColumns(x, pulled);
    ASSERT_EQ(pulled.rows(), static_cast<Eigen::Index>(rows));
    ASSERT_EQ(pulled.cols(), static_cast<Eigen::Index>(ndx));
    EXPECT_TRUE(pulled.isApprox(J * E, 1e-14) || rows == 0);

    const matrix_t Jt = matrix_t::Random(rows, ndx);
    matrix_t lifted = Jt;
    manifold->liftStateColumns(x, lifted);
    ASSERT_EQ(lifted.cols(), static_cast<Eigen::Index>(nx));
    EXPECT_TRUE(lifted.isApprox(Jt * EPlus, 1e-14) || rows == 0);
    // Lifting then pulling back is the identity.
    manifold->pullBackStateColumns(x, lifted);
    EXPECT_TRUE(lifted.isApprox(Jt, 1e-14) || rows == 0);

    const matrix_t M = matrix_t::Random(nx, rows + 1);
    matrix_t pulledRows = M;
    manifold->pullBackStateRows(x, pulledRows);
    EXPECT_TRUE(pulledRows.isApprox(E.transpose() * M, 1e-14));
  }

  matrix_t H = matrix_t::Random(nx, nx);
  H = (H * H.transpose()).eval();
  matrix_t pulledHessian = H;
  manifold->pullBackHessian(x, pulledHessian);
  EXPECT_TRUE(pulledHessian.isApprox(E.transpose() * H * E, 1e-14));

  const vector_t g = vector_t::Random(nx);
  vector_t pulledGradient = g;
  manifold->pullBackGradient(x, pulledGradient);
  EXPECT_TRUE(pulledGradient.isApprox(E.transpose() * g, 1e-14));
}

TEST(ProductStateManifold, PushForwardEqualsTheDenseFormula) {
  const std::shared_ptr<const ProductStateManifold> manifold = twoQuaternionManifold();
  const size_t nx = manifold->getAmbientDim();
  const size_t nu = 2;
  const vector_t x = randomState(*manifold);
  const vector_t xNext = randomState(*manifold);
  VectorFunctionLinearApproximation dynamics;
  dynamics.f = vector_t::Random(nx);  // an ambient flow, not normalized
  dynamics.dfdx = matrix_t::Random(nx, nx);
  dynamics.dfdu = matrix_t::Random(nx, nu);

  // Dense: rows Jr^-1(f_q) E+(Phi) on each quaternion block, the identity elsewhere; columns E(x).
  const vector_t gap = manifold->difference(xNext, dynamics.f);
  matrix_t rowMap = denseTangentMapPseudoInverse(*manifold, dynamics.f);
  for (size_t k = 0; k < 2; ++k) {
    const size_t tangentOffset = manifold->getQuaternionTangentOffsets()[k];
    const size_t ambientOffset = manifold->getQuaternionAmbientOffsets()[k];
    const rotation_t blockGap = gap.segment<3>(tangentOffset);
    rowMap.block<3, 4>(tangentOffset, ambientOffset) = so3RightJacobianInverse(blockGap) * rowMap.block<3, 4>(tangentOffset, ambientOffset);
  }
  const matrix_t expectedA = rowMap * dynamics.dfdx * denseTangentMap(*manifold, x);
  const matrix_t expectedB = rowMap * dynamics.dfdu;

  VectorFunctionLinearApproximation pushed = dynamics;
  manifold->pushForwardDynamics(x, xNext, pushed);
  EXPECT_EQ(pushed.f, gap);
  EXPECT_TRUE(pushed.dfdx.isApprox(expectedA, 1e-13));
  EXPECT_TRUE(pushed.dfdu.isApprox(expectedB, 1e-13));

  VectorFunctionLinearApproximation jump = dynamics;
  manifold->pushForwardJump(x, xNext, jump);
  EXPECT_TRUE(jump.dfdx.isApprox(expectedA, 1e-13));
  EXPECT_EQ(jump.dfdu.rows(), static_cast<Eigen::Index>(manifold->getTangentDim()));
  EXPECT_EQ(jump.dfdu.cols(), 0);
}

// ---------------------------------------------------------------------------------------------------------------------
// EuclideanStateManifold
// ---------------------------------------------------------------------------------------------------------------------

TEST(EuclideanStateManifold, IsTheFlatSpace) {
  const EuclideanStateManifold manifold(4);
  const vector_t x = vector_t::Random(4);
  const vector_t dx = vector_t::Random(4);
  vector_t xNew;
  manifold.retract(x, dx, /*alpha=*/0.3, xNew);
  EXPECT_EQ(xNew, (x + 0.3 * dx).eval());
  EXPECT_EQ(manifold.difference(x, xNew), (xNew - x).eval());
  EXPECT_EQ(manifold.getMaximumRotationAngle(dx), 0.0);
  matrix_t J = matrix_t::Random(2, 4);
  const matrix_t before = J;
  manifold.pullBackStateColumns(x, J);
  manifold.liftStateColumns(x, J);
  EXPECT_EQ(J, before);
  VectorFunctionLinearApproximation dynamics;
  dynamics.f = vector_t::Random(4);
  dynamics.dfdx = matrix_t::Random(4, 4);
  dynamics.dfdu = matrix_t::Random(4, 1);
  VectorFunctionLinearApproximation pushed = dynamics;
  manifold.pushForwardDynamics(x, xNew, pushed);
  EXPECT_EQ(pushed.f, (dynamics.f - xNew).eval());
  EXPECT_EQ(pushed.dfdx, dynamics.dfdx);
  EXPECT_EQ(pushed.dfdu, dynamics.dfdu);
}

}  // namespace
}  // namespace ocs2
