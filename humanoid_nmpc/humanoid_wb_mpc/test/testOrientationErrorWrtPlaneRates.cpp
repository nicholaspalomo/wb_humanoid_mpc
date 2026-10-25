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
#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "ocs2_core/automatic_differentiation/CppAdInterface.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/OrientationErrorRateOracle.h"
#include "humanoid_wb_mpc/end_effector/OrientationErrorWrtPlaneRates.h"

/*
 * The math of the stance foot's orientation rates
 * (end_effector/OrientationErrorWrtPlaneRates.h), on the production functions
 * taped, generated and compiled as PinocchioEndEffectorDynamicsCppAd's
 * libraries are (CppAdInterface, taped at ones, optimized, -ffast-math),
 * against a numerical oracle that shares no formula with them
 * (OrientationErrorRateOracle.h): the attitude is moved along R(t) = Exp(w t +
 * alpha t^2 / 2) R0, whose world-frame angular velocity and acceleration at t =
 * 0 are w and alpha, and the production orientation error
 * rotationMatrixDistanceToPlane(R(t), n) and the normal row's integrand
 * v(t).w(t) / 2 are differenced in t. The cases, each regime about the ground
 * normal e_z and about random plane normals, with random w and alpha, pure
 * spins about the foot normal and about the plane normal among them:
 *  - level attitudes (v = n);
 *  - attitudes tilted up to 80 degrees;
 *  - steep ones, tilted from 80 degrees to 1 + v.n = 1e-2 (about 172 degrees),
 * across the oracle's psi = 1 at 120;
 *  - near the anti-parallel normal, 1 + v.n from 1e-2 down to 1e-5, the
 * oracle's limit;
 *  - near the guard, 1 + v.n from 1e-5 down to 1.1e-6, outside the guard but
 * closer to it than the oracle reaches;
 *  - inside the guard (1 + v.n < 1e-6, the anti-parallel normal included).
 * Held:
 *  - the taped rate and acceleration equal the oracle's within
 * kDifferencedRateTolerance of their natural scale, from the level attitudes to
 * 1 + v.n = 1e-5, and the plain double instantiation of the same functions
 * equals the tape in every regime;
 *  - at a level attitude the rate is w / 2 and the acceleration alpha / 2 + 1/2
 * (n.w)(n x w), the closed form the rescaled gains of the shipped task file
 * rest on;
 *  - inside the guard both are the fallback w / 2 and alpha / 2, and the tape's
 * Jacobian is finite and the fallback's: no division by zero is taped, at the
 * anti-parallel normal either; near the guard the Jacobian is finite;
 *  - from the level attitudes to 1 + v.n = 1e-5 the tape's Jacobian is the
 * derivative of its value, entry by entry against Richardson-extrapolated
 * differences in every variable, each entry within kJacobianTolerance of the
 *    natural scale of its block and of the Jacobian's largest entry, a
 * comparison that rejects the map blocks (the rate's by w, the acceleration's
 * by alpha) with their largest entry moved by 1e-6 of max(1, itself). The
 * oracle's own angular velocity J_l(phi) phi' is held to the differences of its
 * rotation.
 */
namespace ocs2::humanoid {
namespace {

using test_support::AttitudeMotion;
using test_support::DifferencedOrientationErrorRates;
using test_support::kDifferencedRateTolerance;

/** The variables of the taped functions: vec(R) (column-major), w and alpha;
 * the parameters: n. */
constexpr Eigen::Index kVariableDim = 15;
constexpr Eigen::Index kParameterDim = 3;
constexpr Eigen::Index kAngularVelocityIndex = 9;
constexpr Eigen::Index kAngularAccelerationIndex = 12;
/** [rad] The largest tilt of the tilted cases, the smallest of the steep ones.
 */
constexpr scalar_t kLargestTilt = 80.0 * std::numbers::pi / 180.0;
/** 1 + v.n where the steep cases end and the near-anti-parallel ones begin. */
constexpr scalar_t kSteepestOnePlusCosine = 1.0e-2;
/** The smallest 1 + v.n the oracle is compared at
 * (OrientationErrorRateOracle.h), where the near-guard cases begin. */
constexpr scalar_t kOracleSmallestOnePlusCosine = 1.0e-5;
/** The smallest 1 + v.n of the near-guard cases, a tenth above the guard's
 * threshold. */
constexpr scalar_t kNearGuardSmallestOnePlusCosine = 1.1e-6;
/** The number of random cases per regime and plane-normal kind. */
constexpr size_t kCasesPerRegime = 40;
/**
 * The difference allowed between the tape and the plain double instantiation of
 * the same formulas, relative to the scale: -ffast-math reassociation and
 * CppAD's optimizer change only the order of the operations, a few units of
 * 1e-16 per operation, and 1 + v.n near the anti-parallel normal carries an
 * absolute error of 1e-16 into 1 / s^5 (relative 1e-11 at 1 + v.n = 1e-5).
 * Below kOracleSmallestOnePlusCosine that error grows as 1 / (1 + v.n), and so
 * does the tolerance (tapeAgreementTolerance()).
 */
constexpr scalar_t kTapeAgreementTolerance = 1.0e-10;
/** The level closed form's tolerance, relative to its scale: a dozen operations
 * on numbers of the scale. */
constexpr scalar_t kClosedFormTolerance = 1.0e-13;
/**
 * The steps of the Jacobian's differences. In the attitude, vec(R), in units of
 * psi = min(1, sqrt(2 (1 + v.n))): the functions vary over psi^2 / 2 along the
 * normal (1 + v.n moves by the step), so the step is 2e-5 / psi of that scale
 * (4.4e-3 at 1 + v.n = 1e-5, 1e-5 at a tilt), and Richardson's truncation, of
 * the order of its fourth power, is at most about 4e-10 of the derivatives'
 * scale; the round-off, 3 delta / h of the values' error delta, is what the
 * tolerance below bounds. In w and alpha one step of 1 rad/s and rad/s^2: the
 * functions are quadratic in w and linear in alpha, so central differences have
 * no truncation there and a large step keeps their round-off small.
 */
constexpr scalar_t kAttitudeJacobianStep = 1.0e-5;
constexpr scalar_t kRateJacobianStep = 1.0;
/**
 * The difference allowed between a taped Jacobian entry and its differences,
 * relative to the smaller of the natural scale of its block (jacobianScales())
 * and max(1, the Jacobian's largest entry): both comparisons hold.
 */
constexpr scalar_t kJacobianTolerance = 1.0e-7;
/** The change of the largest entry of a map block, relative to max(1, the
 * entry), that the comparison must reject. */
constexpr scalar_t kJacobianMutation = 1.0e-6;

enum class Regime {
  kLevel,
  kTilted,
  kSteep,
  kNearAntiParallel,
  kNearGuard,
  kInsideGuard,
};

std::string regimeName(Regime regime) {
  switch (regime) {
    case Regime::kLevel:
      return "level";
    case Regime::kTilted:
      return "tilted";
    case Regime::kSteep:
      return "steep";
    case Regime::kNearAntiParallel:
      return "near_anti_parallel";
    case Regime::kNearGuard:
      return "near_guard";
    case Regime::kInsideGuard:
      return "inside_guard";
  }
  return "unknown";
}

/** Whether the oracle and the Jacobian's differences reach the regime: from the
 * level attitudes to 1 + v.n = 1e-5. */
bool comparedWithDifferences(Regime regime) {
  return regime != Regime::kNearGuard && regime != Regime::kInsideGuard;
}

/** One attitude, motion and plane. Passive data. */
struct RateCase {
  std::string name;
  Regime regime = Regime::kLevel;
  AttitudeMotion motion;
  vector3_t planeNormal = vector3_t::UnitZ();
};

/** The taped functions' values at a case: [rate; acceleration]. */
struct TapedRates {
  vector3_t rate = vector3_t::Zero();
  vector3_t acceleration = vector3_t::Zero();
};

/** The variables of the taped functions at `motion`. */
vector_t variablesOf(const AttitudeMotion& motion) {
  vector_t variables(kVariableDim);
  variables.head<9>() = Eigen::Map<const Eigen::Matrix<scalar_t, 9, 1>>(
      motion.initialRotation.data());
  variables.segment<3>(kAngularVelocityIndex) = motion.angularVelocity;
  variables.segment<3>(kAngularAccelerationIndex) = motion.angularAcceleration;
  return variables;
}

/** y = [orientationErrorWrtPlaneRate(); orientationErrorWrtPlaneAcceleration()]
 * of x = [vec(R); w; alpha], p = n. */
void ratesFunction(const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
  ad_matrix3_t rotation;
  rotation.col(0) = x.segment<3>(0);
  rotation.col(1) = x.segment<3>(3);
  rotation.col(2) = x.segment<3>(6);
  const ad_vector3_t angularVelocity = x.segment<3>(kAngularVelocityIndex);
  const ad_vector3_t angularAcceleration =
      x.segment<3>(kAngularAccelerationIndex);
  const ad_vector3_t planeNormal = p.head<3>();
  y.resize(6);
  y.head<3>() = orientationErrorWrtPlaneRate<ad_scalar_t>(
      rotation, angularVelocity, planeNormal);
  y.tail<3>() = orientationErrorWrtPlaneAcceleration<ad_scalar_t>(
      rotation, angularVelocity, angularAcceleration, planeNormal);
}

std::unique_ptr<CppAdInterface> makeTapedRates() {
  const std::string folder = (std::filesystem::path(testing::TempDir()) /
                              "orientation_error_wrt_plane_rates")
                                 .string();
  std::unique_ptr<CppAdInterface> library = std::make_unique<CppAdInterface>(
      &ratesFunction, kVariableDim, kParameterDim,
      "orientation_error_wrt_plane_rates", folder);
  library->createModels(CppAdInterface::ApproximationOrder::First,
                        /*verbose=*/false);
  return library;
}

/** The taped functions, generated once for the test program; never destroyed.
 */
const CppAdInterface& tapedRates() {
  static const CppAdInterface* absl_nonnull const kLibrary =
      makeTapedRates().release();
  return *kLibrary;
}

TapedRates tapedRatesAt(const RateCase& rateCase) {
  const vector_t values = tapedRates().getFunctionValue(
      variablesOf(rateCase.motion), rateCase.planeNormal);
  return TapedRates{.rate = values.head<3>(), .acceleration = values.tail<3>()};
}

/** A unit vector perpendicular to the unit vector `normal`, at the angle
 * `direction` about it. */
vector3_t perpendicularTo(const vector3_t& normal, scalar_t direction) {
  const vector3_t reference =
      std::abs(normal.x()) < 0.9 ? vector3_t::UnitX() : vector3_t::UnitY();
  const vector3_t first = normal.cross(reference).normalized();
  const vector3_t second = normal.cross(first);
  return std::cos(direction) * first + std::sin(direction) * second;
}

/**
 * The shortest-arc rotation from e_z to the unit vector `normal`: about e_z x n
 * by the angle between them, a half turn about e_x for n = -e_z.
 */
matrix3_t shortestArcFromUnitZ(const vector3_t& normal) {
  const vector3_t axis = vector3_t::UnitZ().cross(normal);
  const scalar_t sine = axis.norm();
  if (sine > 0.0)
    return Eigen::AngleAxisd(std::atan2(sine, normal.z()), axis / sine)
        .toRotationMatrix();
  if (normal.z() > 0.0) return matrix3_t::Identity();
  return Eigen::AngleAxisd(std::numbers::pi, vector3_t::UnitX())
      .toRotationMatrix();
}

/**
 * The attitude whose normal R e_z is the plane normal turned by `tiltAngle`
 * about a perpendicular axis at `tiltDirection`, turned by `yaw` about its own
 * normal: R = Exp(tilt u) R_n R_z(yaw), R_n the shortest arc from e_z to n.
 */
matrix3_t attitudeOf(const vector3_t& planeNormal,
                     scalar_t tiltAngle,
                     scalar_t tiltDirection,
                     scalar_t yaw) {
  return Eigen::AngleAxisd(tiltAngle,
                           perpendicularTo(planeNormal, tiltDirection))
             .toRotationMatrix() *
         shortestArcFromUnitZ(planeNormal) *
         Eigen::AngleAxisd(yaw, vector3_t::UnitZ()).toRotationMatrix();
}

/** The tilt angle pi - psi at which 1 + v.n = 1 - cos(psi) = `onePlusCosine`:
 * psi = 2 asin(sqrt(onePlusCosine / 2)). */
scalar_t tiltAngleAt(scalar_t onePlusCosine) {
  return std::numbers::pi - 2.0 * std::asin(std::sqrt(0.5 * onePlusCosine));
}

/** The cases of the file comment, from a fixed seed. */
std::vector<RateCase> rateCases() {
  std::mt19937 generator(20261012);
  std::uniform_real_distribution<scalar_t> unit(-1.0, 1.0);
  std::uniform_real_distribution<scalar_t> angle(-std::numbers::pi,
                                                 std::numbers::pi);
  // log10(1 + v.n) of the steep, the near-anti-parallel and the near-guard
  // cases.
  std::uniform_real_distribution<scalar_t> logSteep(
      std::log10(kSteepestOnePlusCosine),
      std::log10(1.0 + std::cos(kLargestTilt)));
  std::uniform_real_distribution<scalar_t> logDistance(
      std::log10(kOracleSmallestOnePlusCosine),
      std::log10(kSteepestOnePlusCosine));
  std::uniform_real_distribution<scalar_t> logNearGuard(
      std::log10(kNearGuardSmallestOnePlusCosine),
      std::log10(kOracleSmallestOnePlusCosine));
  const std::array<scalar_t, 4> insideGuard = {0.0, 1.0e-9, 1.0e-7, 5.0e-7};
  const std::array<Regime, 6> regimes = {
      Regime::kLevel,     Regime::kTilted,
      Regime::kSteep,     Regime::kNearAntiParallel,
      Regime::kNearGuard, Regime::kInsideGuard};
  std::vector<RateCase> cases;
  for (Regime regime : regimes) {
    for (size_t i = 0; i < 2 * kCasesPerRegime; ++i) {
      // The first half about the ground normal, the second about random plane
      // normals.
      const bool groundNormal = i < kCasesPerRegime;
      const vector3_t planeNormal =
          groundNormal
              ? vector3_t::UnitZ()
              : vector3_t(unit(generator), unit(generator), unit(generator))
                    .normalized();
      scalar_t tiltAngle = 0.0;
      switch (regime) {
        case Regime::kLevel:
          tiltAngle = 0.0;
          break;
        case Regime::kTilted:
          // Uniform in (0, 80 degrees].
          tiltAngle = kLargestTilt * 0.5 * (1.0 - unit(generator));
          break;
        case Regime::kSteep:
          tiltAngle = tiltAngleAt(std::pow(10.0, logSteep(generator)));
          break;
        case Regime::kNearAntiParallel:
          tiltAngle = tiltAngleAt(std::pow(10.0, logDistance(generator)));
          break;
        case Regime::kNearGuard:
          tiltAngle = tiltAngleAt(std::pow(10.0, logNearGuard(generator)));
          break;
        case Regime::kInsideGuard:
          tiltAngle = tiltAngleAt(insideGuard[i % insideGuard.size()]);
          break;
      }
      RateCase rateCase;
      rateCase.regime = regime;
      rateCase.planeNormal = planeNormal;
      rateCase.motion.initialRotation = attitudeOf(
          planeNormal, tiltAngle, angle(generator), angle(generator));
      rateCase.motion.angularVelocity =
          2.0 * vector3_t(unit(generator), unit(generator), unit(generator));
      rateCase.motion.angularAcceleration =
          15.0 * vector3_t(unit(generator), unit(generator), unit(generator));
      // Every tenth: a pure spin about the end-effector normal, then about the
      // plane normal.
      const vector3_t endEffectorNormal =
          rateCase.motion.initialRotation.col(2);
      if (i % 10 == 3)
        rateCase.motion.angularVelocity = 1.7 * endEffectorNormal;
      if (i % 10 == 7) rateCase.motion.angularVelocity = -1.3 * planeNormal;
      rateCase.name =
          absl::StrFormat("%s_%d (tilt %.6g rad, 1 + v.n = %.3g, |w| = %.3g)",
                          regimeName(regime), i, tiltAngle,
                          1.0 + endEffectorNormal.dot(planeNormal),
                          rateCase.motion.angularVelocity.norm());
      cases.push_back(std::move(rateCase));
    }
  }
  return cases;
}

/** max |a - b| relative to max(1, scale). */
scalar_t relativeDeviation(const vector3_t& a,
                           const vector3_t& b,
                           scalar_t scale) {
  return (a - b).cwiseAbs().maxCoeff() / std::max(1.0, scale);
}

/** 1 + v.n of the case's attitude and plane. */
scalar_t onePlusCosineOf(const RateCase& rateCase) {
  return 1.0 + rateCase.motion.initialRotation.col(2).dot(rateCase.planeNormal);
}

/**
 * kTapeAgreementTolerance, grown as 1 / (1 + v.n) below
 * kOracleSmallestOnePlusCosine (to 9e-10 at the near-guard cases' 1.1e-6),
 * where the absolute error of 1 + v.n weighs more; inside the guard the
 * fallback is exact anyway.
 */
scalar_t tapeAgreementTolerance(const RateCase& rateCase) {
  const scalar_t onePlusCosine =
      std::max(onePlusCosineOf(rateCase), kNearGuardSmallestOnePlusCosine);
  return kTapeAgreementTolerance *
         std::max(1.0, kOracleSmallestOnePlusCosine / onePlusCosine);
}

/**
 * The natural scale of each entry of the taped Jacobian, by block, for
 * kJacobianTolerance: with psi = min(1, sqrt(2 (1 + v.n))) and r = max(1,
 * orientationErrorRateScale()), the k-th time derivative of e is of the order
 * of r^k (OrientationErrorRateOracle.h) and e varies over psi^2 / 2 along the
 * normal, so a derivative of the rate (k = 1) or the acceleration (k = 2) by
 * the attitude is of the order of r^k / psi^2; the rate is M w and the
 * acceleration M alpha plus a quadratic form in w, M = de/dw of the order of 1
 * / psi, so their derivatives by w are of the order of 1 / psi and r / psi, and
 * the acceleration's by alpha of 1 / psi (the rate does not read alpha). Where
 * the entries cancel, as they do for a spin about the plane normal, the
 * round-off is that of their summands, which these scales are.
 */
matrix_t jacobianScales(const RateCase& rateCase) {
  const scalar_t psi =
      std::min(1.0, std::sqrt(2.0 * onePlusCosineOf(rateCase)));
  const scalar_t rate =
      std::max(1.0, test_support::orientationErrorRateScale(
                        rateCase.motion, rateCase.planeNormal));
  matrix_t scales(6, kVariableDim);
  scales.block(0, 0, 3, kAngularVelocityIndex).setConstant(rate / (psi * psi));
  scales.block(3, 0, 3, kAngularVelocityIndex)
      .setConstant(rate * rate / (psi * psi));
  scales.block<3, 3>(0, kAngularVelocityIndex).setConstant(1.0 / psi);
  scales.block<3, 3>(3, kAngularVelocityIndex).setConstant(rate / psi);
  scales.block<6, 3>(0, kAngularAccelerationIndex).setConstant(1.0 / psi);
  return scales;
}

/**
 * The Jacobian of the tape at the case by Richardson-extrapolated central
 * differences (4 D(h / 2) - D(h)) / 3 in every variable, at the steps of
 * kAttitudeJacobianStep and kRateJacobianStep.
 */
matrix_t differencedJacobian(const RateCase& rateCase) {
  const vector_t variables = variablesOf(rateCase.motion);
  // The functions turn over the angle psi near the anti-parallel normal
  // (OrientationErrorRateOracle.h).
  const scalar_t attitudeStep =
      kAttitudeJacobianStep *
      std::min(1.0, std::sqrt(2.0 * onePlusCosineOf(rateCase)));
  matrix_t differenced(6, kVariableDim);
  for (Eigen::Index i = 0; i < kVariableDim; ++i) {
    const scalar_t step =
        i < kAngularVelocityIndex ? attitudeStep : kRateJacobianStep;
    const std::array<scalar_t, 4> offsets = {step, -step, 0.5 * step,
                                             -0.5 * step};
    std::array<vector_t, 4> values;
    for (size_t s = 0; s < offsets.size(); ++s) {
      vector_t moved = variables;
      moved(i) += offsets[s];
      values[s] = tapedRates().getFunctionValue(moved, rateCase.planeNormal);
    }
    const vector_t full = (values[0] - values[1]) / (2.0 * step);
    const vector_t half = (values[2] - values[3]) / step;
    differenced.col(i) = (4.0 * half - full) / 3.0;
  }
  return differenced;
}

/**
 * The largest |jacobian - differenced| relative to its tolerance,
 * kJacobianTolerance of the smaller of the entry's natural scale (`scales`) and
 * max(1, the Jacobian's largest entry): above 1, the comparison rejects
 * `jacobian`.
 */
scalar_t largestDeviationToTolerance(const matrix_t& jacobian,
                                     const matrix_t& differenced,
                                     const matrix_t& scales) {
  const scalar_t largestEntry = std::max(1.0, jacobian.cwiseAbs().maxCoeff());
  const matrix_t tolerance = kJacobianTolerance * scales.cwiseMin(largestEntry);
  return (jacobian - differenced)
      .cwiseAbs()
      .cwiseQuotient(tolerance)
      .maxCoeff();
}

/** `jacobian` with the largest entry of its block at (`row`, `col`) of size 3 x
 * 3 moved by kJacobianMutation of max(1, it). */
matrix_t withLargestBlockEntryMoved(const matrix_t& jacobian,
                                    Eigen::Index row,
                                    Eigen::Index col) {
  Eigen::Index blockRow = 0;
  Eigen::Index blockCol = 0;
  const scalar_t largest =
      jacobian.block<3, 3>(row, col).cwiseAbs().maxCoeff(&blockRow, &blockCol);
  matrix_t moved = jacobian;
  moved(row + blockRow, col + blockCol) +=
      kJacobianMutation * std::max(1.0, largest);
  return moved;
}

TEST(OrientationErrorWrtPlaneRates,
     TheTapedRatesAreTheDerivativesOfTheOrientationError) {
  std::map<std::string, std::pair<scalar_t, scalar_t>>
      largestDeviation;  // per regime: rate, acceleration
  size_t compared = 0;
  for (const RateCase& rateCase : rateCases()) {
    if (!comparedWithDifferences(rateCase.regime)) continue;
    SCOPED_TRACE(rateCase.name);
    const TapedRates taped = tapedRatesAt(rateCase);
    const DifferencedOrientationErrorRates differenced =
        test_support::differencedOrientationErrorRates(rateCase.motion,
                                                       rateCase.planeNormal);
    const scalar_t rateDeviation =
        relativeDeviation(taped.rate, differenced.rate, differenced.rateScale);
    const scalar_t accelerationDeviation =
        relativeDeviation(taped.acceleration, differenced.acceleration,
                          differenced.rateScale * differenced.rateScale);
    EXPECT_LE(rateDeviation, kDifferencedRateTolerance)
        << taped.rate.transpose() << "\nvs\n"
        << differenced.rate.transpose();
    EXPECT_LE(accelerationDeviation, kDifferencedRateTolerance)
        << taped.acceleration.transpose() << "\nvs\n"
        << differenced.acceleration.transpose();
    std::pair<scalar_t, scalar_t>& largest =
        largestDeviation[regimeName(rateCase.regime)];
    largest.first = std::max(largest.first, rateDeviation);
    largest.second = std::max(largest.second, accelerationDeviation);
    ++compared;
  }
  // The level, tilted, steep and near-anti-parallel regimes.
  EXPECT_EQ(compared, 4 * 2 * kCasesPerRegime);
  for (const std::pair<const std::string, std::pair<scalar_t, scalar_t>>&
           entry : largestDeviation) {
    absl::PrintF(
        "[rates] %s: largest deviation from the differences, relative to the "
        "scale: rate %.3g, acceleration %.3g\n",
        entry.first, entry.second.first, entry.second.second);
  }
}

TEST(OrientationErrorWrtPlaneRates, TheTapeIsThePlainDoubleInstantiation) {
  size_t nearGuard = 0;
  for (const RateCase& rateCase : rateCases()) {
    SCOPED_TRACE(rateCase.name);
    const TapedRates taped = tapedRatesAt(rateCase);
    const AttitudeMotion& motion = rateCase.motion;
    const vector3_t rate = orientationErrorWrtPlaneRate<scalar_t>(
        motion.initialRotation, motion.angularVelocity, rateCase.planeNormal);
    const vector3_t acceleration =
        orientationErrorWrtPlaneAcceleration<scalar_t>(
            motion.initialRotation, motion.angularVelocity,
            motion.angularAcceleration, rateCase.planeNormal);
    // Inside the guard the rates are w / 2 and alpha / 2, of the scale of w and
    // alpha.
    const scalar_t scale = rateCase.regime == Regime::kInsideGuard
                               ? std::max(motion.angularVelocity.norm(),
                                          motion.angularAcceleration.norm())
                               : test_support::orientationErrorRateScale(
                                     motion, rateCase.planeNormal);
    const scalar_t tolerance = tapeAgreementTolerance(rateCase);
    EXPECT_LE(relativeDeviation(taped.rate, rate, scale), tolerance);
    EXPECT_LE(
        relativeDeviation(taped.acceleration, acceleration, scale * scale),
        tolerance);
    if (rateCase.regime == Regime::kNearGuard) ++nearGuard;
  }
  EXPECT_EQ(nearGuard, 2 * kCasesPerRegime);
}

TEST(OrientationErrorWrtPlaneRates,
     AtALevelAttitudeTheRatesAreHalfTheAngularRatesButForTheGyroscopicTerm) {
  scalar_t largestGyroscopicTerm = 0.0;
  for (const RateCase& rateCase : rateCases()) {
    if (rateCase.regime != Regime::kLevel) continue;
    SCOPED_TRACE(rateCase.name);
    const TapedRates taped = tapedRatesAt(rateCase);
    const vector3_t& n = rateCase.planeNormal;
    const vector3_t& w = rateCase.motion.angularVelocity;
    const vector3_t& alpha = rateCase.motion.angularAcceleration;
    const vector3_t gyroscopicTerm = 0.5 * n.dot(w) * n.cross(w);
    EXPECT_LE(relativeDeviation(taped.rate, 0.5 * w, w.norm()),
              kClosedFormTolerance);
    EXPECT_LE(
        relativeDeviation(taped.acceleration, 0.5 * alpha + gyroscopicTerm,
                          alpha.norm() + w.squaredNorm()),
        kClosedFormTolerance);
    // The term is perpendicular to n: the row along the plane normal is
    // (n.alpha) / 2 exactly.
    EXPECT_NEAR(n.dot(gyroscopicTerm), 0.0,
                /*abs_error=*/1.0e-15 * std::max(1.0, w.squaredNorm()));
    largestGyroscopicTerm =
        std::max(largestGyroscopicTerm,
                 gyroscopicTerm.norm() / std::max(1.0e-12, w.squaredNorm()));
  }
  // |1/2 (n.w)(n x w)| = 1/2 |w_n| |w_t| <= |w|^2 / 4: the bound is reached
  // where the spin and the tilt rate are equal.
  EXPECT_LE(largestGyroscopicTerm, 0.25 + 1.0e-12);
  EXPECT_GT(largestGyroscopicTerm, 0.05)
      << "the case under test: level attitudes that spin and tilt at once";
  absl::PrintF("[rates] level: largest |1/2 (n.w)(n x w)| / |w|^2 = %.3g\n",
               largestGyroscopicTerm);
}

TEST(OrientationErrorWrtPlaneRates,
     InsideTheAntiParallelGuardTheRatesFallBackWithAFiniteJacobian) {
  matrix_t fallbackJacobian = matrix_t::Zero(6, kVariableDim);
  fallbackJacobian.block<3, 3>(0, kAngularVelocityIndex) =
      0.5 * matrix3_t::Identity();
  fallbackJacobian.block<3, 3>(3, kAngularAccelerationIndex) =
      0.5 * matrix3_t::Identity();
  size_t guarded = 0;
  for (const RateCase& rateCase : rateCases()) {
    if (rateCase.regime != Regime::kInsideGuard) continue;
    SCOPED_TRACE(rateCase.name);
    ASSERT_LT(
        1.0 + rateCase.motion.initialRotation.col(2).dot(rateCase.planeNormal),
        kOrientationErrorWrtPlaneAntiParallelThreshold);
    const TapedRates taped = tapedRatesAt(rateCase);
    EXPECT_LE(
        relativeDeviation(taped.rate, 0.5 * rateCase.motion.angularVelocity,
                          rateCase.motion.angularVelocity.norm()),
        1.0e-15);
    EXPECT_LE(relativeDeviation(taped.acceleration,
                                0.5 * rateCase.motion.angularAcceleration,
                                rateCase.motion.angularAcceleration.norm()),
              1.0e-15);
    const matrix_t jacobian = tapedRates().getJacobian(
        variablesOf(rateCase.motion), rateCase.planeNormal);
    ASSERT_TRUE(jacobian.allFinite()) << jacobian;
    EXPECT_TRUE(jacobian == fallbackJacobian) << jacobian;
    ++guarded;
  }
  EXPECT_EQ(guarded, 2 * kCasesPerRegime);
}

TEST(OrientationErrorWrtPlaneRates,
     OutsideTheGuardTheTapedJacobianIsTheDerivative) {
  std::map<std::string, scalar_t>
      largestDeviation;  // per regime, relative to the tolerance
  scalar_t smallestRateMapRejection = 1.0e300;
  scalar_t smallestAccelerationMapRejection = 1.0e300;
  size_t nearGuard = 0;
  for (const RateCase& rateCase : rateCases()) {
    if (rateCase.regime == Regime::kInsideGuard) continue;
    SCOPED_TRACE(rateCase.name);
    const matrix_t jacobian = tapedRates().getJacobian(
        variablesOf(rateCase.motion), rateCase.planeNormal);
    ASSERT_TRUE(jacobian.allFinite()) << jacobian;
    if (!comparedWithDifferences(rateCase.regime)) {
      // Near the guard, outside it: finite, and not differenced, whose steps
      // the guard's threshold would bound.
      EXPECT_GE(onePlusCosineOf(rateCase),
                kOrientationErrorWrtPlaneAntiParallelThreshold);
      ++nearGuard;
      continue;
    }
    const matrix_t differenced = differencedJacobian(rateCase);
    const matrix_t scales = jacobianScales(rateCase);
    const scalar_t deviation =
        largestDeviationToTolerance(jacobian, differenced, scales);
    EXPECT_LE(deviation, 1.0) << "taped\n"
                              << jacobian << "\ndifferenced\n"
                              << differenced << "\nscales\n"
                              << scales;
    scalar_t& largest = largestDeviation[regimeName(rateCase.regime)];
    largest = std::max(largest, deviation);

    // The comparison resolves a change of 1e-6 of the maps: the rate's by w,
    // the acceleration's by alpha.
    const scalar_t rateMapRejection = largestDeviationToTolerance(
        withLargestBlockEntryMoved(jacobian, /*row=*/0,
                                   /*col=*/kAngularVelocityIndex),
        differenced, scales);
    const scalar_t accelerationMapRejection = largestDeviationToTolerance(
        withLargestBlockEntryMoved(jacobian, /*row=*/3,
                                   /*col=*/kAngularAccelerationIndex),
        differenced, scales);
    EXPECT_GT(rateMapRejection, 1.0)
        << "a 1e-6 change of the rate's map is not resolved";
    EXPECT_GT(accelerationMapRejection, 1.0)
        << "a 1e-6 change of the acceleration's map is not resolved";
    smallestRateMapRejection =
        std::min(smallestRateMapRejection, rateMapRejection);
    smallestAccelerationMapRejection =
        std::min(smallestAccelerationMapRejection, accelerationMapRejection);
  }
  EXPECT_EQ(nearGuard, 2 * kCasesPerRegime);
  for (const std::pair<const std::string, scalar_t>& entry : largestDeviation) {
    absl::PrintF(
        "[rates] %s: largest Jacobian deviation from the differences, relative "
        "to its tolerance: %.3g\n",
        entry.first, entry.second);
  }
  absl::PrintF(
      "[rates] the Jacobian's comparison rejects a 1e-6 change of the rate's "
      "map by w by a factor of %.3g at least, of the "
      "acceleration's by alpha by %.3g\n",
      smallestRateMapRejection, smallestAccelerationMapRejection);
}

TEST(OrientationErrorWrtPlaneRates,
     TheOraclesMotionHasThePrescribedAngularVelocityAndAcceleration) {
  // The oracle's rotation R(t) and its angular velocity J_l(phi) phi' agree:
  // [w(t)]x = dR/dt R', by differences at t and t = 0.3 s; and dw/dt at 0 is
  // alpha. Central differences at h = 1e-4 s with Richardson's extrapolation:
  // truncation O(h^4) (1e-16) and round-off 3 eps / h (1e-11) of the rates of
  // order 10.
  const scalar_t step = 1.0e-4;
  for (const RateCase& rateCase : rateCases()) {
    if (rateCase.regime != Regime::kTilted) continue;
    SCOPED_TRACE(rateCase.name);
    const AttitudeMotion& motion = rateCase.motion;
    for (scalar_t time : {0.0, 0.3}) {
      const matrix3_t full =
          (test_support::rotationAlong(motion, time + step) -
           test_support::rotationAlong(motion, time - step)) /
          (2.0 * step);
      const matrix3_t half =
          (test_support::rotationAlong(motion, time + 0.5 * step) -
           test_support::rotationAlong(motion, time - 0.5 * step)) /
          step;
      const matrix3_t rotationRate = (4.0 * half - full) / 3.0;
      const matrix3_t cross =
          rotationRate * test_support::rotationAlong(motion, time).transpose();
      const vector3_t differenced(cross(2, 1), cross(0, 2), cross(1, 0));
      EXPECT_LE((test_support::angularVelocityAlong(motion, time) - differenced)
                    .cwiseAbs()
                    .maxCoeff(),
                1.0e-9)
          << "t = " << time;
    }
    const vector3_t full = (test_support::angularVelocityAlong(motion, step) -
                            test_support::angularVelocityAlong(motion, -step)) /
                           (2.0 * step);
    const vector3_t half =
        (test_support::angularVelocityAlong(motion, 0.5 * step) -
         test_support::angularVelocityAlong(motion, -0.5 * step)) /
        step;
    EXPECT_LE(((4.0 * half - full) / 3.0 - motion.angularAcceleration)
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-9);
  }
}

}  // namespace
}  // namespace ocs2::humanoid
