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
#include <cmath>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "Eigen/Eigenvalues"
#include "Eigen/LU"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/energy.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/RootJointModelPair.h"

/*
 * The dynamics through the two root joints (RootJointModelPair.h): for the same physical motion, mapped from the Euler
 * root to the quaternion root by the chart (C1)-(C3), the mass matrix is congruent through J_v (C6), the nonlinear
 * effects and gravity map through J_v^T with the Euler side's dT_B/dt term (C6), inverse dynamics gives the same joint
 * torques and base force and maps the base moment through T_B^T (C4), with and without external wrenches, forward
 * dynamics gives the same accelerations after the chart (C3, C4), and the energies are the same scalars.
 */

namespace ocs2::humanoid {
namespace {

constexpr int kSamples = 200;

/**
 * Round-off bound (T1) (testRootJointKinematicsEquivalence.cpp, indexed in RootJointModelPair.h), relative to the compared
 * quantity (floor 1): the recursive algorithms (crba, rnea, nonLinearEffects) sum composite inertias times twists and
 * accelerations over the tree, the same operations in both models but for the root, so their results agree like the
 * kinematics they are built from.
 */
constexpr scalar_t kDynamicsTolerance = 1.0e-12;

/**
 * Round-off bounds (T4) of forward dynamics. pinocchio::aba solves M a = tau - n.
 *
 *   Forward error: the computed a is the exact solution of a problem perturbed by the residual's round-off, so its
 *   relative error is at most that round-off (T1) times the condition number kappa(M) = lambda_max / lambda_min, computed
 *   per sample; the Euler side solves with M_e = J_v^T M J_v, so |a - (J_v a_e + dJ_v/dt v_e)| <= kDynamicsTolerance *
 *   (kappa(M) + kappa(M_e)) * max(1, |a|). This is the worst case: the shipped robots' kappa is 2e4 to 1.3e5 (light
 *   wrists and ankles) and the measured forward error was at most 5.2e-15 relative when this was written.
 *
 *   Residual: the tight check. aba is backward stable, M_e a_e + n_e - tau_e = O(eps (|M_e| |a_e| + |n_e| + |tau_e|)),
 *   and (C6) maps that residual to the quaternion root through J_v^-T, so the mapped accelerations put back into the
 *   quaternion root's rnea give tau up to
 *   kDynamicsTolerance * |J_v^-1| * (| |M| |a| | + |n| + |tau|) (max norms, |M| |a| entrywise), which kappa does not
 *   enter. Any error of the chart's accelerations that M can see - an error in dJ_v/dt v_e, a T_B in place of its
 *   transpose - is far above it; the measured residual was at most 3.4e-4 of this bound when this was written.
 */
scalar_t forwardDynamicsTolerance(const matrix_t& quaternionMassMatrix, const matrix_t& eulerMassMatrix) {
  const vector_t quaternionEigenvalues = Eigen::SelfAdjointEigenSolver<matrix_t>(quaternionMassMatrix).eigenvalues();
  const vector_t eulerEigenvalues = Eigen::SelfAdjointEigenSolver<matrix_t>(eulerMassMatrix).eigenvalues();
  const scalar_t quaternionCondition = quaternionEigenvalues.maxCoeff() / quaternionEigenvalues.minCoeff();
  const scalar_t eulerCondition = eulerEigenvalues.maxCoeff() / eulerEigenvalues.minCoeff();
  return kDynamicsTolerance * (quaternionCondition + eulerCondition);
}

/** The residual bound of (T4), absolute, for the accelerations `acceleration` of the motion with velocity map J_v. */
scalar_t inverseDynamicsResidualBound(const matrix_t& massMatrix,
                                      const vector_t& acceleration,
                                      const vector_t& nonlinearEffects,
                                      const vector_t& torques,
                                      const matrix_t& velocityMap) {
  const scalar_t inverseMapNorm = velocityMap.inverse().cwiseAbs().rowwise().sum().maxCoeff();
  const scalar_t scale = (massMatrix.cwiseAbs() * acceleration.cwiseAbs()).maxCoeff() + maxAbs(nonlinearEffects) + maxAbs(torques);
  return kDynamicsTolerance * inverseMapNorm * std::max(1.0, scale);
}

class RootJointDynamicsEquivalence : public ::testing::TestWithParam<EquivalenceRobot> {
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

/** M(q), both triangles. */
matrix_t massMatrixAt(const pinocchio::Model& model, pinocchio::Data& data, const vector_t& q) {
  return symmetricFromUpper(pinocchio::crba(model, data, q));
}

/** Random wrenches on every joint, in the joint's own frame (pinocchio::rnea's fext), of up to 100 N and 20 N m. */
std::vector<pinocchio::Force> randomJointWrenches(const pinocchio::Model& model, std::mt19937& generator) {
  std::vector<pinocchio::Force> wrenches;
  wrenches.reserve(static_cast<size_t>(model.njoints));
  for (int joint = 0; joint < model.njoints; ++joint) {
    const vector3_t force = uniformVector(/*size=*/3, /*halfWidth=*/100.0, generator);
    const vector3_t moment = uniformVector(/*size=*/3, /*halfWidth=*/20.0, generator);
    wrenches.emplace_back(force, moment);
  }
  return wrenches;
}

TEST_P(RootJointDynamicsEquivalence, MassMatrixIsCongruentThroughTheVelocityMap) {
  // (C6): 1/2 v^T M v = 1/2 v_e^T J_v^T M J_v v_e is the same kinetic energy, so M_e = J_v^T M J_v.
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  std::mt19937 generator(301);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion = randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const matrix_t velocityMap = velocityMaps(eulerMotion).velocityMap;
    const matrix_t eulerMassMatrix = massMatrixAt(eulerModel, eulerData, eulerMotion.q);
    const matrix_t quaternionMassMatrix = massMatrixAt(quaternionModel, quaternionData, quaternionMotion(eulerMotion).q);
    ASSERT_TRUE(isCloseRelative(velocityMap.transpose() * quaternionMassMatrix * velocityMap, eulerMassMatrix, kDynamicsTolerance))
        << "sample " << sample;
  }
}

TEST_P(RootJointDynamicsEquivalence, NonlinearEffectsAndGravityMapThroughTheChart) {
  // (C6): M a + n = tau with a = J_v a_e + (dJ_v/dt) v_e and tau_e = J_v^T tau gives M_e a_e + n_e = tau_e with
  // n_e = J_v^T (n + M (dJ_v/dt) v_e): the Euler side's Coriolis term carries SphericalZYX's dS/dt dTheta, which is the
  // quaternion side's M (dJ_v/dt) v_e. Gravity has no velocity term: g_e = J_v^T g.
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  std::mt19937 generator(303);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion = randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const PinocchioMotion motion = quaternionMotion(eulerMotion);
    const ChartVelocityMaps maps = velocityMaps(eulerMotion);
    const matrix_t quaternionMassMatrix = massMatrixAt(quaternionModel, quaternionData, motion.q);
    const vector_t eulerEffects = pinocchio::nonLinearEffects(eulerModel, eulerData, eulerMotion.q, eulerMotion.v);
    const vector_t quaternionEffects = pinocchio::nonLinearEffects(quaternionModel, quaternionData, motion.q, motion.v);
    ASSERT_TRUE(
        isCloseRelative(maps.velocityMap.transpose() * (quaternionEffects + quaternionMassMatrix * maps.velocityMapRate * eulerMotion.v),
                        eulerEffects, kDynamicsTolerance))
        << "sample " << sample;

    const vector_t eulerGravity = pinocchio::computeGeneralizedGravity(eulerModel, eulerData, eulerMotion.q);
    const vector_t quaternionGravity = pinocchio::computeGeneralizedGravity(quaternionModel, quaternionData, motion.q);
    ASSERT_TRUE(isCloseRelative(maps.velocityMap.transpose() * quaternionGravity, eulerGravity, kDynamicsTolerance)) << "sample " << sample;
    // The gravity force on the base is -m g in both, its moment differs only by T_B^T.
    ASSERT_TRUE(isCloseRelative(quaternionGravity.head<3>(), eulerGravity.head<3>(), kDynamicsTolerance)) << "sample " << sample;
  }
}

TEST_P(RootJointDynamicsEquivalence, InverseDynamicsGivesTheSameTorquesForTheSameMotion) {
  // (C4): tau_e = J_v^T tau. The joint torques and the base force are the same numbers; the Euler base "torque" is
  // T_B^T times the body moment. With external wrenches on every body (in the bodies' own frames, which are the same
  // frames in both models), the same holds.
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  const Eigen::Index nj = models_->numJoints();
  std::mt19937 generator(305);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion = randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const PinocchioMotion motion = quaternionMotion(eulerMotion);
    const matrix_t velocityMap = velocityMaps(eulerMotion).velocityMap;

    const vector_t eulerTorques = pinocchio::rnea(eulerModel, eulerData, eulerMotion.q, eulerMotion.v, eulerMotion.a);
    const vector_t quaternionTorques = pinocchio::rnea(quaternionModel, quaternionData, motion.q, motion.v, motion.a);
    ASSERT_TRUE(isCloseRelative(velocityMap.transpose() * quaternionTorques, eulerTorques, kDynamicsTolerance)) << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(quaternionTorques.tail(nj), eulerTorques.tail(nj), kDynamicsTolerance)) << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(quaternionTorques.head<3>(), eulerTorques.head<3>(), kDynamicsTolerance)) << "sample " << sample;

    const std::vector<pinocchio::Force> wrenches = randomJointWrenches(eulerModel, generator);
    const vector_t eulerLoadedTorques = pinocchio::rnea(eulerModel, eulerData, eulerMotion.q, eulerMotion.v, eulerMotion.a, wrenches);
    const vector_t quaternionLoadedTorques = pinocchio::rnea(quaternionModel, quaternionData, motion.q, motion.v, motion.a, wrenches);
    ASSERT_TRUE(isCloseRelative(velocityMap.transpose() * quaternionLoadedTorques, eulerLoadedTorques, kDynamicsTolerance))
        << "sample " << sample;
  }
}

TEST_P(RootJointDynamicsEquivalence, ForwardDynamicsGivesTheSameAccelerationsAfterTheChart) {
  // (C3, C4): the same generalized force, tau_e = J_v^T tau, gives a = J_v a_e + (dJ_v/dt) v_e: aba on the quaternion
  // root against aba on the Euler root mapped by the chart, for a force on every coordinate and for an unactuated base
  // (tau = [0; tau_j], whose Euler image is also [0; tau_j]): within the forward bound of (T4), and, tightly, the mapped
  // accelerations reproduce the quaternion root's torques through rnea within the residual bound of (T4).
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  const Eigen::Index nj = models_->numJoints();
  std::mt19937 generator(307);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion = randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const PinocchioMotion motion = quaternionMotion(eulerMotion);
    const ChartVelocityMaps maps = velocityMaps(eulerMotion);
    const matrix_t quaternionMassMatrix = massMatrixAt(quaternionModel, quaternionData, motion.q);
    const scalar_t tolerance = forwardDynamicsTolerance(quaternionMassMatrix, massMatrixAt(eulerModel, eulerData, eulerMotion.q));
    const vector_t quaternionEffects = pinocchio::nonLinearEffects(quaternionModel, quaternionData, motion.q, motion.v);

    vector_t torques(quaternionModel.nv);
    torques << uniformVector(/*size=*/3, /*halfWidth=*/500.0, generator), uniformVector(/*size=*/3, /*halfWidth=*/100.0, generator),
        uniformVector(nj, /*halfWidth=*/50.0, generator);
    vector_t unactuatedBaseTorques = torques;
    unactuatedBaseTorques.head<6>().setZero();
    for (const vector_t& quaternionTorques : {torques, unactuatedBaseTorques}) {
      const vector_t eulerTorques = maps.velocityMap.transpose() * quaternionTorques;
      const vector_t eulerAcceleration = pinocchio::aba(eulerModel, eulerData, eulerMotion.q, eulerMotion.v, eulerTorques);
      const vector_t quaternionAcceleration = pinocchio::aba(quaternionModel, quaternionData, motion.q, motion.v, quaternionTorques);
      const vector_t mappedAcceleration = maps.velocityMap * eulerAcceleration + maps.velocityMapRate * eulerMotion.v;
      ASSERT_TRUE(isCloseRelative(quaternionAcceleration, mappedAcceleration, tolerance)) << "sample " << sample;
      const scalar_t residualBound =
          inverseDynamicsResidualBound(quaternionMassMatrix, mappedAcceleration, quaternionEffects, quaternionTorques, maps.velocityMap);
      ASSERT_TRUE(isCloseAbsolute(pinocchio::rnea(quaternionModel, quaternionData, motion.q, motion.v, mappedAcceleration),
                                  quaternionTorques, residualBound))
          << "sample " << sample;
    }
  }
}

TEST_P(RootJointDynamicsEquivalence, KineticAndPotentialEnergiesAreTheSame) {
  // The kinetic energy of the same motion (C2, C6) and the potential energy of the same placements (C1) are the same
  // scalars.
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  std::mt19937 generator(309);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion = randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const PinocchioMotion motion = quaternionMotion(eulerMotion);
    const scalar_t eulerKinetic = pinocchio::computeKineticEnergy(eulerModel, eulerData, eulerMotion.q, eulerMotion.v);
    const scalar_t quaternionKinetic = pinocchio::computeKineticEnergy(quaternionModel, quaternionData, motion.q, motion.v);
    EXPECT_LE(std::abs(quaternionKinetic - eulerKinetic), kDynamicsTolerance * std::max(1.0, std::abs(eulerKinetic)))
        << "sample " << sample;
    const scalar_t eulerPotential = pinocchio::computePotentialEnergy(eulerModel, eulerData, eulerMotion.q);
    const scalar_t quaternionPotential = pinocchio::computePotentialEnergy(quaternionModel, quaternionData, motion.q);
    EXPECT_LE(std::abs(quaternionPotential - eulerPotential), kDynamicsTolerance * std::max(1.0, std::abs(eulerPotential)))
        << "sample " << sample;
  }
}

INSTANTIATE_TEST_SUITE_P(EveryShippedRobot,
                         RootJointDynamicsEquivalence,
                         ::testing::ValuesIn(equivalenceRobots().begin(), equivalenceRobots().end()),
                         equivalenceRobotName);

}  // namespace
}  // namespace ocs2::humanoid
