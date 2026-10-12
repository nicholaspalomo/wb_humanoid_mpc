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

#include <memory>
#include <random>
#include <utility>

#include "Eigen/LU"
#include "Eigen/SVD"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_validation/chart/EulerChart.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/RootJointModelPair.h"

/*
 * The centroidal quantities through the two root joints (RootJointModelPair.h): the centroidal momentum h_G = A_G v,
 * the centroidal momentum matrix A_G, its time derivative, the momentum rate, the composite rigid-body inertias, and
 * the base velocity the centroidal MPC recovers from the momentum (design section 2.5) - the same physics in both
 * coordinates, with the velocities mapped through T_B (C2).
 */

namespace ocs2::humanoid {
namespace {

constexpr int kSamples = 200;

/**
 * Round-off bound (T1) of the kinematics (testRootJointKinematicsEquivalence.cpp, indexed in RootJointModelPair.h),
 * relative: A_G and h_G are sums of composite inertias times twists.
 */
constexpr scalar_t kKinematicsTolerance = 1.0e-12;

/**
 * Step and bound (T3) of the time derivative of A_G along a motion, by Richardson's extrapolation of central differences,
 * D = (4 D(h / 2) - D(h)) / 3 with D(h) = (A_G(q (+) h v) - A_G(q (+) -h v)) / 2h: its truncation error is h^4 / 480 times
 * the fifth derivative of A_G along the curve, of order |A_G| |v|^5 with |v| (the curve's speed in radians and meters
 * per second) at most about 15 for the sampled motions, so 2.4e-13 * 7.6e5 / 480 = 4e-10 at h = 1e-3, relative to
 * |A_G|; its round-off error eps |A_G| / h times (4 * 2 / 2 + 1 / 2) / 3 = 1.5, so 3.3e-13. 1e-8 is 25 times their sum,
 * and the chart's dJ_v/dt term it has to show is of order |A_G| |dTheta|. Measured when this was written: at most
 * 1.4e-10 relative.
 */
constexpr scalar_t kRichardsonStep = 1.0e-3;
constexpr scalar_t kMomentumMatrixRateTolerance = 1.0e-8;

class RootJointCentroidalEquivalence : public ::testing::TestWithParam<EquivalenceRobot> {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<RootJointModelPair>> models = RootJointModelPair::Create(GetParam());
    ASSERT_TRUE(models.ok()) << models.status();
    models_ = *std::move(models);
  }

  PinocchioMotion quaternionMotion(const PinocchioMotion& eulerMotion) const {
    absl::StatusOr<PinocchioMotion> motion = quaternionMotionOf(eulerMotion, models_->pinocchioStateLayout());
    EXPECT_TRUE(motion.ok()) << motion.status();
    return motion.ok() ? *std::move(motion) : PinocchioMotion{};
  }

  ChartVelocityMaps velocityMaps(const PinocchioMotion& eulerMotion) const {
    absl::StatusOr<ChartVelocityMaps> maps = chartVelocityMapsAt(eulerMotion, models_->pinocchioStateLayout());
    EXPECT_TRUE(maps.ok()) << maps.status();
    return maps.ok() ? *std::move(maps) : ChartVelocityMaps{};
  }

  std::unique_ptr<RootJointModelPair> models_;
};

/** A_G at q, from pinocchio::computeCentroidalMap. */
matrix_t momentumMatrixAt(const pinocchio::Model& model, const vector_t& q) {
  pinocchio::Data data(model);
  return pinocchio::computeCentroidalMap(model, data, q);
}

/** (A_G(q (+) h v) - A_G(q (+) -h v)) / 2h, with (+) pinocchio::integrate. */
matrix_t momentumMatrixCentralDifference(const pinocchio::Model& model, const vector_t& q, const vector_t& v, scalar_t step) {
  return (momentumMatrixAt(model, pinocchio::integrate(model, q, step * v)) -
          momentumMatrixAt(model, pinocchio::integrate(model, q, -step * v))) /
         (2.0 * step);
}

/**
 * d A_G / dt along the motion (q, v) of `model` (T3): Richardson's extrapolation of central differences of A_G along
 * pinocchio::integrate, which on the quaternion root moves along v in its tangent and on the Euler root is q + t v. Only
 * the curve's tangent v at t = 0 enters the first derivative, so the two models' different curves through the same
 * tangent give the same rate.
 */
matrix_t momentumMatrixRateAlong(const pinocchio::Model& model, const vector_t& q, const vector_t& v) {
  return (4.0 * momentumMatrixCentralDifference(model, q, v, kRichardsonStep / 2.0) -
          momentumMatrixCentralDifference(model, q, v, kRichardsonStep)) /
         3.0;
}

TEST_P(RootJointCentroidalEquivalence, MomentumMatrixMomentumAndCompositeInertiasAreTheSame) {
  // (C5): h_G = A_G v = A_G J_v v_e for every v_e, so A_G,e = A_G J_v and h_G is the same; the composite rigid-body
  // inertia of every subtree (pinocchio::ccrba's oYcrb, world axes) and the centroidal one (Ig) depend only on the
  // placements (C1).
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  std::mt19937 generator(201);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion = randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const PinocchioMotion motion = quaternionMotion(eulerMotion);
    const matrix_t eulerMap = pinocchio::ccrba(eulerModel, eulerData, eulerMotion.q, eulerMotion.v);
    const matrix_t quaternionMap = pinocchio::ccrba(quaternionModel, quaternionData, motion.q, motion.v);
    ASSERT_TRUE(isCloseRelative(quaternionMap * velocityMaps(eulerMotion).velocityMap, eulerMap, kKinematicsTolerance))
        << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(quaternionData.hg.toVector(), eulerData.hg.toVector(), kKinematicsTolerance)) << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(quaternionData.Ig.matrix(), eulerData.Ig.matrix(), kKinematicsTolerance)) << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(quaternionData.com[0], eulerData.com[0], kKinematicsTolerance)) << "sample " << sample;
    for (int joint = 1; joint < eulerModel.njoints; ++joint) {
      ASSERT_TRUE(isCloseRelative(quaternionData.oYcrb[joint].matrix(), eulerData.oYcrb[joint].matrix(), kKinematicsTolerance))
          << eulerModel.names[joint] << ", sample " << sample;
    }
  }
}

TEST_P(RootJointCentroidalEquivalence, MomentumRateIsTheSame) {
  // dh_G/dt of the same motion (C2, C3) from Pinocchio's forward kinematics with accelerations, which carries the Euler
  // side's dT_B/dt dTheta in SphericalZYX's bias and the quaternion side's in a = J_v a_e + (dJ_v/dt) v_e.
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  std::mt19937 generator(203);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion = randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const PinocchioMotion motion = quaternionMotion(eulerMotion);
    const vector6_t eulerRate =
        pinocchio::computeCentroidalMomentumTimeVariation(eulerModel, eulerData, eulerMotion.q, eulerMotion.v, eulerMotion.a).toVector();
    const vector6_t quaternionRate =
        pinocchio::computeCentroidalMomentumTimeVariation(quaternionModel, quaternionData, motion.q, motion.v, motion.a).toVector();
    ASSERT_TRUE(isCloseRelative(quaternionRate, eulerRate, kKinematicsTolerance)) << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(quaternionData.hg.toVector(), eulerData.hg.toVector(), kKinematicsTolerance)) << "sample " << sample;
  }
}

TEST_P(RootJointCentroidalEquivalence, MomentumMatrixTimeDerivativeMapsThroughTheChart) {
  // (C7): A_G,e(q_e) = A_G(Phi(q_e)) J_v(q_e), so d A_G,e / dt = (d A_G / dt) J_v + A_G dJ_v/dt along the same motion.
  // Each side's rate is the derivative of Pinocchio's own A_G along Pinocchio's own integrator (T3), and is anchored to
  // Pinocchio's exact momentum rate: dh_G/dt = A_G a + (d A_G / dt) v in both coordinates. pinocchio::dccrba is not
  // used: its dJ = v x J leaves out dS/dt, which neither root's motion subspace is free of (the composite's translation
  // columns turn with the rotation, SphericalZYX's S depends on the angles).
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  std::mt19937 generator(205);
  for (int sample = 0; sample < kSamples / 4; ++sample) {
    const PinocchioMotion eulerMotion = randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const PinocchioMotion motion = quaternionMotion(eulerMotion);
    const ChartVelocityMaps maps = velocityMaps(eulerMotion);
    const matrix_t eulerRate = momentumMatrixRateAlong(eulerModel, eulerMotion.q, eulerMotion.v);
    const matrix_t quaternionRate = momentumMatrixRateAlong(quaternionModel, motion.q, motion.v);
    const matrix_t quaternionMap = momentumMatrixAt(quaternionModel, motion.q);
    ASSERT_TRUE(
        isCloseRelative(quaternionRate * maps.velocityMap + quaternionMap * maps.velocityMapRate, eulerRate, kMomentumMatrixRateTolerance))
        << "sample " << sample;

    const vector6_t eulerMomentumRate =
        pinocchio::computeCentroidalMomentumTimeVariation(eulerModel, eulerData, eulerMotion.q, eulerMotion.v, eulerMotion.a).toVector();
    const vector6_t quaternionMomentumRate =
        pinocchio::computeCentroidalMomentumTimeVariation(quaternionModel, quaternionData, motion.q, motion.v, motion.a).toVector();
    const matrix_t eulerMap = momentumMatrixAt(eulerModel, eulerMotion.q);
    ASSERT_TRUE(isCloseRelative(eulerMap * eulerMotion.a + eulerRate * eulerMotion.v, eulerMomentumRate, kMomentumMatrixRateTolerance))
        << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(quaternionMap * motion.a + quaternionRate * motion.v, quaternionMomentumRate, kMomentumMatrixRateTolerance))
        << "sample " << sample;
  }
}

TEST_P(RootJointCentroidalEquivalence, BaseVelocityFromTheMomentumIsTheEulerRatesThroughTB) {
  // The centroidal MPC's base velocity (design section 2.5): v_b = A_b^-1 (h_G - A_j qd_j), with A_b the first six
  // columns of A_G and A_j the rest. By (C5) A_b,e = A_b blockdiag(I, T_B) and A_j,e = A_j, so the quaternion root
  // recovers [pd_W, w_B] = [pd_W, T_B dTheta] from the momentum where the Euler root recovers [pd_W, dTheta]. Both solves
  // lose their condition number times (T1).
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  const Eigen::Index nj = models_->numJoints();
  std::mt19937 generator(207);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion = randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const PinocchioMotion motion = quaternionMotion(eulerMotion);
    const matrix_t eulerMap = pinocchio::ccrba(eulerModel, eulerData, eulerMotion.q, eulerMotion.v);
    const matrix_t quaternionMap = pinocchio::ccrba(quaternionModel, quaternionData, motion.q, motion.v);
    const vector6_t momentum = quaternionData.hg.toVector();

    const matrix6_t eulerBaseBlock = eulerMap.leftCols<6>();
    const matrix6_t quaternionBaseBlock = quaternionMap.leftCols<6>();
    const vector6_t eulerBaseVelocity = eulerBaseBlock.partialPivLu().solve(momentum - eulerMap.rightCols(nj) * eulerMotion.v.tail(nj));
    const vector6_t quaternionBaseVelocity =
        quaternionBaseBlock.partialPivLu().solve(momentum - quaternionMap.rightCols(nj) * motion.v.tail(nj));

    const Eigen::JacobiSVD<matrix6_t> eulerSvd(eulerBaseBlock);
    const Eigen::JacobiSVD<matrix6_t> quaternionSvd(quaternionBaseBlock);
    const scalar_t eulerCondition = eulerSvd.singularValues()(0) / eulerSvd.singularValues()(5);
    const scalar_t quaternionCondition = quaternionSvd.singularValues()(0) / quaternionSvd.singularValues()(5);
    ASSERT_TRUE(isCloseRelative(eulerBaseVelocity, eulerMotion.v.head<6>(), kKinematicsTolerance * eulerCondition)) << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(quaternionBaseVelocity, motion.v.head<6>(), kKinematicsTolerance * quaternionCondition))
        << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(quaternionBaseVelocity.tail<3>(),
                                velocityMaps(eulerMotion).velocityMap.block<3, 3>(3, 3) * eulerBaseVelocity.tail<3>(),
                                kKinematicsTolerance * (eulerCondition + quaternionCondition)))
        << "sample " << sample;
  }
}

TEST_P(RootJointCentroidalEquivalence, CentroidalChartIsTheIdentityOnTheMomentum) {
  // EulerChart's centroidal layout [h / m, p_W, Theta, q_j] -> [h / m, p_W, xi, q_j]: identity on the normalized
  // momentum, because both roots give the same h_G for the same motion; and its configuration rows are the quaternion
  // root's q, the whole-body chart's (C1).
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  const Eigen::Index nq = quaternionModel.nq;
  const scalar_t mass = pinocchio::computeTotalMass(eulerModel);
  std::mt19937 generator(209);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion = randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const PinocchioMotion motion = quaternionMotion(eulerMotion);
    pinocchio::ccrba(eulerModel, eulerData, eulerMotion.q, eulerMotion.v);
    pinocchio::ccrba(quaternionModel, quaternionData, motion.q, motion.v);
    vector_t centroidalEulerState(6 + eulerModel.nq);
    centroidalEulerState << eulerData.hg.toVector() / mass, eulerMotion.q;
    absl::StatusOr<vector_t> centroidalState =
        validation::physicalStateFromEulerState(centroidalEulerState, models_->centroidalStateLayout());
    ASSERT_TRUE(centroidalState.ok()) << centroidalState.status();
    ASSERT_EQ(centroidalState->size(), 6 + nq);
    ASSERT_TRUE(isCloseRelative(centroidalState->head<6>(), quaternionData.hg.toVector() / mass, kKinematicsTolerance))
        << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(centroidalState->tail(nq), motion.q, /*relativeTolerance=*/0.0)) << "sample " << sample;
  }
}

INSTANTIATE_TEST_SUITE_P(EveryShippedRobot,
                         RootJointCentroidalEquivalence,
                         ::testing::ValuesIn(equivalenceRobots().begin(), equivalenceRobots().end()),
                         equivalenceRobotName);

}  // namespace
}  // namespace ocs2::humanoid
