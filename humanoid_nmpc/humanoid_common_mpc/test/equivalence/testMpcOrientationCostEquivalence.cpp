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

#include "pinocchio/fwd.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "Eigen/SVD"
#include "absl/base/nullability.h"
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateCost.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateInputCost.h"
#include "humanoid_common_mpc/orientation/TuningLayoutDeviation.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/MpcEquivalenceFixture.h"

/*
 * The quadratic state costs of the MPC on the quaternion layout against the Euler formulation's, under the task file's
 * Euler-written weights as the one mapping layer turns them into the costs' Q (design section 3 and 2.8.2): the Euler MPC
 * weighed the Euler state's difference, L_e = 1/2 (x_e - x_e,r)' Q (x_e - x_e,r); today's MPC weighs the tuning-layout
 * deviation r(x, x_r) of TuningLayoutDeviation with the same Q, L = 1/2 r' Q r. Q is read off the MPC's own running and
 * terminal costs.
 *
 * Rows. r has the Euler difference's Euclidean rows exactly. The orientation rows (yaw, pitch, roll), at ZYX angles
 * Theta = (psi, theta, phi) of the state and Theta_r of the reference, are closed forms of the Euler angles (derived
 * here, independent of the swing-twist code that computes them from xi):
 *   (R1) yaw   wrap(psi - psi_r + o(theta, phi) - o(theta_r, phi_r)),  o(theta, phi) = -2 atan(tan(theta/2) tan(phi/2)):
 *        the twist of q_y(theta) q_x(phi) about z is (0, 0, -sin(theta/2) sin(phi/2), cos(theta/2) cos(phi/2)), of angle o;
 *   (R2) tilt  tau(theta, phi) - tau(theta_r, phi_r) in (pitch, roll) = (tau_y, tau_x), the swing's rotation vector
 *        tau = alpha N / |N|, N = 1/2 (sin(phi) cos(theta), sin(theta)), alpha = 2 atan2(|N|, rho^2),
 *        rho^2 = cos^2(theta/2) cos^2(phi/2) + sin^2(theta/2) sin^2(phi/2): the swing q_t^-1 (x) q_y q_x has the vector part
 *        N / rho and the scalar part rho;
 *   (R3) whole body, the angular-velocity rows (z, y, x): P_3 (T_B(Theta) dTheta - R(Theta)' R(Theta_r) T_B(Theta_r) dTheta_r).
 *
 * Second order. With d = r - (x_e - x_e,r) (zero but on those rows), L - L_e = (x_e - x_e,r)' Q d + 1/2 d' Q d, and the
 * series of (R1)-(R3) bound d row by row (|t| = |(theta, phi)|):
 *   yaw    |o(theta, phi) + theta phi / 2| <= kYawRemainder |t|^4;
 *   pitch  |tau_y - theta - theta phi^2 / 6| <= kTiltRemainder |t|^5,   roll |tau_x - phi + phi theta^2 / 3| <= kTiltRemainder |t|^5
 *          (the cubic terms from expanding (R2): alpha^2 = theta^2 + phi^2 - theta^2 phi^2 / 3 + O(6),
 *          |N|^2 = (phi^2 + theta^2 - phi^4 / 3 - theta^2 phi^2 - theta^4 / 3) / 4 + O(6));
 *   rates  |P_3 T_B(Theta) - I|_2 <= |t| and |R' R_r - I|_2 <= |Delta psi| + |Delta theta| + |Delta phi|, |T_B|_2 <= sqrt(3)
 *          (design section 4.4).
 * The orientation and rate errors and the tilts are all O(eps) at an attitude scale eps, so L and L_e are O(eps^2) and
 * their difference O(eps^3): the costs agree to second order, which the scaling test asserts (halving eps divides the
 * difference by at least kMinimumThirdOrderRatio). At a level state the orientation rows' Jacobian through the chart is
 * the identity, J_HT(xi) T_B(0) = P_3 P_3 = I, so the Gauss-Newton Hessian is the Euler one; at a tilt |t| it differs by
 * at most |t| |Q| (design section 4.4).
 */

namespace ocs2::humanoid {
namespace {

/** Round-off of the closed forms against the production rows: a few dozen operations of trigonometric functions. */
constexpr scalar_t kClosedFormTolerance = 1.0e-12;

/**
 * The quintic remainder of the tilt rows' series and the quartic one of the yaw row's, for |t| <= kSeriesTiltLimit: the
 * series converge there, and their remainders, checked on a grid by SeriesBoundsOfTheHeadingTiltRows, are at most
 * 0.012 |t|^5 (roll), 0.0068 |t|^5 (pitch) and 0.021 |t|^4 (yaw); the constants leave a margin of 1.4 to 3.
 */
constexpr scalar_t kTiltRemainder = 0.02;
constexpr scalar_t kYawRemainder = 0.03;
constexpr scalar_t kSeriesTiltLimit = 0.7;

/**
 * A cubic difference a eps^3 (1 + x eps) shrinks by 8 (1 + x eps) / (1 + x eps / 2) when eps halves. The quartic term can
 * have the other sign and a large coefficient: on the whole body's terminal weights a fit gave x = -13, which takes the
 * ratio to 4.1 at eps = 0.05. At the scales of kScalingTestScales the ratio stays above kMinimumThirdOrderRatio for
 * |x| <= 40, and a quadratic term would give 4.
 */
constexpr scalar_t kMinimumThirdOrderRatio = 6.0;
constexpr std::array<scalar_t, 3> kScalingTestScales = {0.01, 0.005, 0.0025};

class MpcOrientationCostEquivalence : public MpcEquivalenceTest {};

/** o(theta, phi) of (R1). */
scalar_t headingOffset(scalar_t pitch, scalar_t roll) {
  return -2.0 * std::atan(std::tan(0.5 * pitch) * std::tan(0.5 * roll));
}

/** tau(theta, phi) = (tau_x, tau_y) of (R2). */
vector2_t zyxTilt(scalar_t pitch, scalar_t roll) {
  const vector2_t n(0.5 * std::sin(roll) * std::cos(pitch), 0.5 * std::sin(pitch));
  const scalar_t rhoSquared =
      std::pow(std::cos(0.5 * pitch) * std::cos(0.5 * roll), 2) + std::pow(std::sin(0.5 * pitch) * std::sin(0.5 * roll), 2);
  const scalar_t length = n.norm();
  if (length == 0.0) return vector2_t::Zero();
  return (2.0 * std::atan2(length, rhoSquared) / length) * n;
}

/** R_z(psi) R_y(theta) R_x(phi) from the elementary rotations. */
matrix3_t zyxRotation(const vector3_t& eulerZyx) {
  const matrix3_t yaw = Eigen::AngleAxisd(eulerZyx(0), vector3_t::UnitZ()).toRotationMatrix();
  const matrix3_t pitch = Eigen::AngleAxisd(eulerZyx(1), vector3_t::UnitY()).toRotationMatrix();
  const matrix3_t roll = Eigen::AngleAxisd(eulerZyx(2), vector3_t::UnitX()).toRotationMatrix();
  return yaw * pitch * roll;
}

/** T_B(Theta), the body angular velocity of the ZYX rates: omega_B = R' dR/dt, columns (yaw, pitch, roll) rates. */
matrix3_t zyxRateMap(const vector3_t& eulerZyx) {
  const scalar_t sinPitch = std::sin(eulerZyx(1));
  const scalar_t cosPitch = std::cos(eulerZyx(1));
  const scalar_t sinRoll = std::sin(eulerZyx(2));
  const scalar_t cosRoll = std::cos(eulerZyx(2));
  matrix3_t map;
  map << -sinPitch, 0.0, 1.0, cosPitch * sinRoll, cosRoll, 0.0, cosPitch * cosRoll, -sinRoll, 0.0;
  return map;
}

/** wrap to (-pi, pi]. */
scalar_t wrapped(scalar_t angle) {
  const scalar_t turn = 2.0 * M_PI;
  scalar_t result = std::fmod(angle + M_PI, turn);
  if (result <= 0.0) result += turn;
  return result - M_PI;
}

/** Where the orientation (and whole-body rate) rows of the tuning layout are: the same rows of the Euler state. Passive data. */
struct OrientationRows {
  Eigen::Index orientation = 0;
  /** The whole body's angular-velocity rows (z, y, x), on the Euler state the ZYX rates; nullopt on the centroidal MPC. */
  std::optional<Eigen::Index> rates;
};

OrientationRows orientationRowsOf(const MpcEquivalenceFixture& mpc) {
  OrientationRows rows;
  Eigen::Index row = 0;
  for (const validation::EulerStateSegment& segment : mpc.eulerLayout().segments) {
    if (segment.kind == validation::EulerStateSegmentKind::kEulerZyxAngles) rows.orientation = row;
    if (segment.kind == validation::EulerStateSegmentKind::kEulerZyxRates) rows.rates = row;
    row += static_cast<Eigen::Index>(segment.dim);
  }
  return rows;
}

/** The closed-form deviation of (R1)-(R3) at the Euler states `eulerState` and `eulerReference`. */
vector_t closedFormDeviation(const OrientationRows& rows, const vector_t& eulerState, const vector_t& eulerReference) {
  vector_t deviation = eulerState - eulerReference;
  const vector3_t angles = eulerState.segment<3>(rows.orientation);
  const vector3_t referenceAngles = eulerReference.segment<3>(rows.orientation);
  const vector2_t tilt = zyxTilt(angles(1), angles(2));
  const vector2_t referenceTilt = zyxTilt(referenceAngles(1), referenceAngles(2));
  deviation(rows.orientation) =
      wrapped(angles(0) - referenceAngles(0) + headingOffset(angles(1), angles(2)) - headingOffset(referenceAngles(1), referenceAngles(2)));
  deviation(rows.orientation + 1) = tilt(1) - referenceTilt(1);
  deviation(rows.orientation + 2) = tilt(0) - referenceTilt(0);
  if (rows.rates.has_value()) {
    const vector3_t rates = eulerState.segment<3>(*rows.rates);
    const vector3_t referenceRates = eulerReference.segment<3>(*rows.rates);
    const vector3_t transported =
        zyxRotation(angles).transpose() * zyxRotation(referenceAngles) * zyxRateMap(referenceAngles) * referenceRates;
    deviation.segment<3>(*rows.rates) = (zyxRateMap(angles) * rates - transported).reverse();
  }
  return deviation;
}

/** The Euler state's angles' tilt |(theta, phi)|. */
scalar_t tiltOf(const OrientationRows& rows, const vector_t& eulerState) {
  return eulerState.segment<2>(rows.orientation + 1).norm();
}

/** The bound of the second-order section on |d|, row by row. */
vector_t secondOrderRowBounds(const OrientationRows& rows, const vector_t& eulerState, const vector_t& eulerReference) {
  vector_t bound = vector_t::Zero(eulerState.size());
  const vector3_t angles = eulerState.segment<3>(rows.orientation);
  const vector3_t referenceAngles = eulerReference.segment<3>(rows.orientation);
  const scalar_t tilt = tiltOf(rows, eulerState);
  const scalar_t referenceTilt = tiltOf(rows, eulerReference);
  const scalar_t quintic = kTiltRemainder * (std::pow(tilt, 5) + std::pow(referenceTilt, 5));
  bound(rows.orientation) = 0.5 * std::abs(angles(1) * angles(2) - referenceAngles(1) * referenceAngles(2)) +
                            kYawRemainder * (std::pow(tilt, 4) + std::pow(referenceTilt, 4));
  bound(rows.orientation + 1) =
      (std::abs(angles(1)) * angles(2) * angles(2) + std::abs(referenceAngles(1)) * referenceAngles(2) * referenceAngles(2)) / 6.0 +
      quintic;
  bound(rows.orientation + 2) =
      (std::abs(angles(2)) * angles(1) * angles(1) + std::abs(referenceAngles(2)) * referenceAngles(1) * referenceAngles(1)) / 3.0 +
      quintic;
  if (rows.rates.has_value()) {
    const scalar_t attitudeError = (angles - referenceAngles).cwiseAbs().sum();
    bound.segment<3>(*rows.rates)
        .setConstant(tilt * eulerState.segment<3>(*rows.rates).norm() +
                     (referenceTilt + std::sqrt(3.0) * attitudeError) * eulerReference.segment<3>(*rows.rates).norm());
  }
  return bound;
}

/** One quadratic state cost's weights. Passive data. */
struct CostWeights {
  /** The cost term's name in the problem. */
  std::string term;
  matrix_t stateWeights;
};

/** The MPC's running and terminal state weights: the costs the one mapping layer sets. */
std::vector<CostWeights> stateWeightsOf(MpcEquivalenceFixture& mpc) {
  std::vector<CostWeights> weights;
  const OptimalControlProblem& problem = mpc.problem();
  size_t index = 0;
  for (const char* absl_nonnull name : {kStateInputQuadraticCostTerm, kStateQuadraticCostTerm}) {
    if (problem.costPtr->getTermIndex(name, index)) {
      weights.push_back(
          CostWeights{.term = name, .stateWeights = problem.costPtr->get<TuningQuadraticStateInputCost>(name).getStateWeights()});
    }
  }
  if (problem.finalCostPtr->getTermIndex(kTerminalCostTerm, index)) {
    weights.push_back(
        CostWeights{.term = kTerminalCostTerm,
                    .stateWeights = problem.finalCostPtr->get<TuningQuadraticStateCost>(kTerminalCostTerm).getStateWeights()});
  }
  return weights;
}

/** The dense dr/dx of a linearization: its identity runs and dense blocks summed. */
matrix_t denseJacobian(const TuningDeviationApproximation& approximation, Eigen::Index stateDim) {
  matrix_t jacobian = matrix_t::Zero(approximation.residual.size(), stateDim);
  for (const TuningJacobianIdentityRun& run : approximation.identityRuns) {
    for (size_t i = 0; i < run.size; ++i) {
      jacobian(static_cast<Eigen::Index>(run.tuningRow + i), static_cast<Eigen::Index>(run.stateColumn + i)) += 1.0;
    }
  }
  for (const TuningJacobianDenseBlock& block : approximation.denseBlocks) {
    jacobian.block(static_cast<Eigen::Index>(block.tuningRow), static_cast<Eigen::Index>(block.stateColumn), block.value.rows(),
                   block.value.cols()) += block.value;
  }
  return jacobian;
}

TEST(SeriesBoundsOfTheHeadingTiltRows, HoldOnAGridUpToTheTiltLimit) {
  for (int radius = 1; radius <= 70; ++radius) {
    const scalar_t tilt = kSeriesTiltLimit * radius / 70.0;
    for (int direction = 0; direction < 360; ++direction) {
      const scalar_t angle = 2.0 * M_PI * direction / 360.0;
      const scalar_t roll = tilt * std::cos(angle);
      const scalar_t pitch = tilt * std::sin(angle);
      const vector2_t tau = zyxTilt(pitch, roll);
      ASSERT_LE(std::abs(tau(0) - roll + roll * pitch * pitch / 3.0), kTiltRemainder * std::pow(tilt, 5)) << tilt << " " << angle;
      ASSERT_LE(std::abs(tau(1) - pitch - pitch * roll * roll / 6.0), kTiltRemainder * std::pow(tilt, 5)) << tilt << " " << angle;
      ASSERT_LE(std::abs(headingOffset(pitch, roll) + 0.5 * pitch * roll), kYawRemainder * std::pow(tilt, 4)) << tilt << " " << angle;
      const matrix3_t rateMapError = zyxRateMap(vector3_t(0.0, pitch, roll)).colwise().reverse() - matrix3_t::Identity();
      ASSERT_LE(Eigen::JacobiSVD<matrix3_t>(rateMapError).singularValues()(0), tilt * (1.0 + 1.0e-12)) << tilt << " " << angle;
    }
  }
}

TEST_P(MpcOrientationCostEquivalence, TheTuningRowsAreTheClosedFormsOfTheEulerAngles) {
  MpcEquivalenceFixture& mpc = this->mpc();
  const TuningLayoutDeviation deviation(&mpc.model());
  const OrientationRows rows = orientationRowsOf(mpc);
  std::mt19937 generator(/*sd=*/20261030);
  std::uniform_real_distribution<scalar_t> yawError(-2.5, 2.5);
  EulerSamplingBounds bounds;
  bounds.maxAbsPitch = 0.6;
  bounds.maxAbsRoll = 0.6;
  for (int sample = 0; sample < 50; ++sample) {
    SCOPED_TRACE(sample);
    const vector_t eulerState = mpc.randomEulerState(bounds, generator);
    vector_t eulerReference = mpc.randomEulerState(bounds, generator);
    eulerReference(rows.orientation) = eulerState(rows.orientation) + yawError(generator);
    const vector_t residual = deviation.evaluate(mpc.quaternionStateOf(eulerState), mpc.quaternionStateOf(eulerReference));
    EXPECT_TRUE(isCloseToScale(residual, closedFormDeviation(rows, eulerState, eulerReference), kClosedFormTolerance, /*floor=*/1.0));
  }
}

/** The quadratic cost of today's deviation and the Euler difference's, at one state and reference. Passive data. */
struct CostPair {
  scalar_t quaternion = 0.0;
  scalar_t euler = 0.0;
};

/** 1/2 r' Q r of the tuning-layout deviation at Phi(x_e), Phi(x_e,r), and 1/2 (x_e - x_e,r)' Q (x_e - x_e,r). */
CostPair costsAt(const TuningLayoutDeviation& deviation,
                 const MpcEquivalenceFixture& mpc,
                 const matrix_t& stateWeights,
                 const vector_t& eulerState,
                 const vector_t& eulerReference) {
  const vector_t residual = deviation.evaluate(mpc.quaternionStateOf(eulerState), mpc.quaternionStateOf(eulerReference));
  const vector_t eulerDifference = eulerState - eulerReference;
  return CostPair{.quaternion = 0.5 * residual.dot(stateWeights * residual),
                  .euler = 0.5 * eulerDifference.dot(stateWeights * eulerDifference)};
}

/** One direction of the scaling test: the state's and the reference's orientation (and rate) offsets per unit eps. Passive data. */
struct ScalingDirection {
  vector_t base;
  vector3_t stateAngles;
  vector3_t referenceAngles;
  vector3_t stateRates;
  vector3_t referenceRates;
};

/** The state (`reference` false) or reference of `direction` at scale eps: the base, its angles and rates offset by eps times the
 * direction's. */
vector_t scaledState(const OrientationRows& rows, const ScalingDirection& direction, scalar_t eps, bool reference) {
  vector_t state = direction.base;
  state.segment<3>(rows.orientation) += eps * (reference ? direction.referenceAngles : direction.stateAngles);
  if (rows.rates.has_value()) state.segment<3>(*rows.rates) = eps * (reference ? direction.referenceRates : direction.stateRates);
  return state;
}

TEST_P(MpcOrientationCostEquivalence, TheCostsAgreeToSecondOrderWithinTheDerivedBound) {
  MpcEquivalenceFixture& mpc = this->mpc();
  const TuningLayoutDeviation deviation(&mpc.model());
  const OrientationRows rows = orientationRowsOf(mpc);
  const std::vector<CostWeights> weights = stateWeightsOf(mpc);
  ASSERT_FALSE(weights.empty()) << "the MPC has no tuning-layout state cost";
  std::mt19937 generator(/*sd=*/20261031);
  std::uniform_real_distribution<scalar_t> unit(-1.0, 1.0);
  EulerSamplingBounds level;
  level.maxAbsPitch = 0.0;
  level.maxAbsRoll = 0.0;
  for (const CostWeights& weight : weights) {
    SCOPED_TRACE(weight.term);
    const matrix_t& stateWeights = weight.stateWeights;
    ASSERT_EQ(stateWeights.rows(), mpc.oracle().stateDim());
    for (int sample = 0; sample < 8; ++sample) {
      SCOPED_TRACE(sample);
      // A level base, every Euclidean entry shared by the state and the reference, so that only the orientation (and
      // rate) rows differ; the angles of both offset in random directions.
      ScalingDirection direction{.base = mpc.randomEulerState(level, generator),
                                 .stateAngles = vector3_t(unit(generator), unit(generator), unit(generator)),
                                 .referenceAngles = vector3_t(unit(generator), unit(generator), unit(generator)),
                                 .stateRates = vector3_t(unit(generator), unit(generator), unit(generator)),
                                 .referenceRates = vector3_t(unit(generator), unit(generator), unit(generator))};
      // The derived bound at attitude and rate scales eps up to 0.2.
      for (const scalar_t eps : {0.2, 0.1, 0.05}) {
        const vector_t eulerState = scaledState(rows, direction, eps, /*reference=*/false);
        const vector_t eulerReference = scaledState(rows, direction, eps, /*reference=*/true);
        ASSERT_LE(std::max(tiltOf(rows, eulerState), tiltOf(rows, eulerReference)), kSeriesTiltLimit);
        const CostPair costs = costsAt(deviation, mpc, stateWeights, eulerState, eulerReference);
        // |e' Q d + 1/2 d' Q d| <= |e|' |Q| b + 1/2 b' |Q| b, entry by entry.
        const vector_t rowBound = secondOrderRowBounds(rows, eulerState, eulerReference);
        const vector_t eulerDifference = eulerState - eulerReference;
        const matrix_t absoluteWeights = stateWeights.cwiseAbs();
        const scalar_t bound = eulerDifference.cwiseAbs().dot(absoluteWeights * rowBound) + 0.5 * rowBound.dot(absoluteWeights * rowBound) +
                               kClosedFormTolerance * std::max(1.0, std::abs(costs.euler));
        EXPECT_LE(std::abs(costs.quaternion - costs.euler), bound)
            << "eps " << eps << ": " << costs.quaternion << " against the Euler " << costs.euler;
      }
      // Third order: halving eps divides the difference by 8, where the cubic term dominates; round-off aside.
      std::vector<scalar_t> differences;
      for (const scalar_t eps : kScalingTestScales) {
        const CostPair costs = costsAt(deviation, mpc, stateWeights, scaledState(rows, direction, eps, /*reference=*/false),
                                       scaledState(rows, direction, eps, /*reference=*/true));
        differences.push_back(std::abs(costs.quaternion - costs.euler));
      }
      for (size_t i = 0; i + 1 < differences.size(); ++i) {
        if (differences[i] <= 1.0e3 * kClosedFormTolerance * kScalingTestScales[i] * kScalingTestScales[i]) continue;
        EXPECT_GE(differences[i] / differences[i + 1], kMinimumThirdOrderRatio) << differences[i] << " " << differences[i + 1];
      }
    }
  }
}

TEST_P(MpcOrientationCostEquivalence, TheGaussNewtonModelIsTheEulerOneAtALevelStateAndWithinTheTiltBoundElsewhere) {
  MpcEquivalenceFixture& mpc = this->mpc();
  const TuningLayoutDeviation deviation(&mpc.model());
  const OrientationRows rows = orientationRowsOf(mpc);
  const std::vector<CostWeights> weights = stateWeightsOf(mpc);
  ASSERT_FALSE(weights.empty());
  std::mt19937 generator(/*sd=*/20261032);
  std::uniform_real_distribution<scalar_t> unit(-1.0, 1.0);
  EulerSamplingBounds tilted;
  tilted.maxAbsPitch = 0.6;
  tilted.maxAbsRoll = 0.6;
  EulerSamplingBounds level;
  level.maxAbsPitch = 0.0;
  level.maxAbsRoll = 0.0;
  const Eigen::Index orientation = rows.orientation;
  for (int sample = 0; sample < 20; ++sample) {
    SCOPED_TRACE(sample);
    const bool isLevel = sample % 2 == 0;
    const vector_t eulerState = mpc.randomEulerState(isLevel ? level : tilted, generator);
    vector_t eulerReference = mpc.randomEulerState(tilted, generator);
    eulerReference(orientation) = eulerState(orientation) + 2.5 * unit(generator);
    if (isLevel) eulerReference(orientation + 1 + sample % 4 / 2) = 0.0;  // a reference tilted about one axis
    const vector_t state = mpc.quaternionStateOf(eulerState);
    const TuningDeviationApproximation approximation = deviation.linearize(state, mpc.quaternionStateOf(eulerReference));
    // (E2): the Euler-chart Jacobian of the residual.
    const matrix_t eulerJacobian = denseJacobian(approximation, state.size()) * mpc.chartJacobian(eulerState);
    const matrix3_t orientationJacobian = eulerJacobian.block<3, 3>(orientation, orientation);
    const scalar_t tilt = tiltOf(rows, eulerState);
    if (isLevel) {
      // J_HT T_B(0) = I; against a reference tilted about one axis the rows are the Euler differences (yaw wrapped).
      EXPECT_TRUE(isCloseToScale(orientationJacobian, matrix3_t::Identity(), kClosedFormTolerance, /*floor=*/1.0));
      vector3_t eulerOrientationDifference = (eulerState - eulerReference).segment<3>(orientation);
      eulerOrientationDifference(0) = wrapped(eulerOrientationDifference(0));
      EXPECT_TRUE(
          isCloseToScale(approximation.residual.segment<3>(orientation), eulerOrientationDifference, kClosedFormTolerance, /*floor=*/1.0));
    } else {
      // Design section 4.4: |J_e - I|_max <= 0.6 |tilt| (the coefficient is 1/2 to first order, 0.52 measured at 0.85 rad).
      EXPECT_LE((orientationJacobian - matrix3_t::Identity()).cwiseAbs().maxCoeff(), 0.6 * tilt);
    }
    for (const CostWeights& weight : weights) {
      const matrix3_t orientationWeights = weight.stateWeights.block<3, 3>(orientation, orientation);
      const matrix3_t hessian = orientationJacobian.transpose() * orientationWeights * orientationJacobian;
      // |J' Q J - Q|_max <= |tilt| |Q|_max (0.52 measured); zero at a level state.
      const scalar_t largestWeight = orientationWeights.cwiseAbs().maxCoeff();
      EXPECT_LE((hessian - orientationWeights).cwiseAbs().maxCoeff(),
                tilt * largestWeight + kClosedFormTolerance * std::max(1.0, largestWeight))
          << weight.term;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Shipped,
                         MpcOrientationCostEquivalence,
                         ::testing::ValuesIn(mpcEquivalenceConfigurationNames()),
                         mpcEquivalenceTestName);

}  // namespace
}  // namespace ocs2::humanoid
