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
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Eigenvalues"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/RootJointModelPair.h"

/*
 * What only the quaternion root handles (RootJointModelPair.h), checked against
 * Pinocchio:
 *
 *   - a pitch of +-90 degrees, where the Euler root's T_B is singular (det T_B
 * = -cos(pitch)): there the quaternion root's dynamics IS its level dynamics
 * under a rotated gravity (the covariance below), so it has no singularity; the
 * Euler root's base block loses rank exactly as cos(pitch), its positive
 * control;
 *   - paths that Pinocchio's own integrator drives through a pitch of 90
 * degrees and across a yaw of +-pi: the quaternion moves continuously while the
 * Euler boundary's angles jump, and the physics on the path is the Euler root's
 * at those angles;
 *   - -xi, the same rotation;
 *   - the Euler boundary's inverse (eulerZyxFromQuaternion) at every attitude,
 * gimbal lock included: the Euler root at its angles has the quaternion root's
 * placements, as the telemetry and the display rely on.
 *
 * The covariance (D): rotating the world by the constant W = R(xi)^T maps the
 * base pose (p_W, R) to (W p_W, I), the world velocities and accelerations
 * pd_W, pdd_W and the world-axes momentum to W times themselves, gravity g to W
 * g, and leaves every body-frame quantity (w_B, dw_B/dt, the joints, body
 * wrenches) unchanged; Newton-Euler is covariant under a constant rotation of
 * the inertial frame. With D = blockdiag(W, I_3, I_nj) on the quaternion root's
 * v, a and tau (the base force is a world vector, the base moment a body one):
 *
 *   aba(q, v, tau; g) = D^-1 aba(q', D v, D tau; W g),   rnea(q, v, a; g) =
 * D^-1 rnea(q', D v, D a; W g), M(q) = D^T M(q') D,   h_G(q, v) = blockdiag(W,
 * W)^-1 h_G(q', D v),   oMf(q) = (W, 0)^-1 oMf(q'),
 *
 * with q' = [W p_W, identity, q_j]. D is orthogonal, so M(q) has the
 * eigenvalues of M(q') at every attitude.
 */

namespace ocs2::humanoid {
namespace {

/**
 * Round-off bound (T1) (testRootJointKinematicsEquivalence.cpp, indexed in
 * RootJointModelPair.h), relative: the two sides of (D) differ by the rotations
 * by W.
 */
constexpr scalar_t kRoundOffTolerance = 1.0e-12;

/**
 * Bound (T5) of the Euler boundary's inverse on the placements:
 * eulerZyxFromQuaternion reproduces the rotation to about 2e-16 / cos(pitch)
 * radians, at most 2e-13 at its gimbal-lock threshold cos(pitch) = 1e-3
 * (EulerBoundary.h); at lever arms of at most 4 m that is 8e-13 m, plus (T1) on
 * the placements: 1e-11 is ten times the sum. Measured when this was written:
 * at most 1e-15.
 */
constexpr scalar_t kBoundaryInverseTolerance = 1.0e-11;

/** An attitude the quaternion root is checked at, by its quaternion and a
 * label. Passive data. */
struct TestAttitude {
  std::string label;
  vector4_t quaternion;
};

/** Pitch exactly and almost +-90 degrees at several yaws and rolls, and random
 * unit quaternions (uniform on S^3). */
std::vector<TestAttitude> testAttitudes(std::mt19937& generator) {
  std::vector<TestAttitude> attitudes;
  for (const scalar_t pitch : {M_PI / 2.0, -M_PI / 2.0, M_PI / 2.0 - 1.0e-9,
                               -M_PI / 2.0 + 1.0e-9, M_PI / 2.0 - 1.0e-4}) {
    for (const scalar_t yaw : {0.0, 2.5, -M_PI}) {
      for (const scalar_t roll : {0.0, -1.1, M_PI}) {
        attitudes.push_back({.label = absl::StrCat("yaw ", yaw, " pitch ",
                                                   pitch, " roll ", roll),
                             .quaternion = quaternionFromEulerZyx(
                                 vector3_t(yaw, pitch, roll))});
      }
    }
  }
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  for (int i = 0; i < 20; ++i) {
    const vector4_t coefficients(normal(generator), normal(generator),
                                 normal(generator), normal(generator));
    attitudes.push_back({.label = absl::StrCat("random attitude ", i),
                         .quaternion = coefficients.normalized()});
  }
  return attitudes;
}

/** R(xi) of coefficients (x, y, z, w), built from the coefficient vector
 * (design section 2.1's construction trap). */
matrix3_t rotationOf(const vector4_t& xi) {
  quaternion_t quaternion;
  quaternion.coeffs() = xi;
  return quaternion.toRotationMatrix();
}

/** D = blockdiag(W, I_3, I_nj) of (D). */
matrix_t covarianceMap(const matrix3_t& worldRotation, Eigen::Index nv) {
  matrix_t map = matrix_t::Identity(nv, nv);
  map.topLeftCorner<3, 3>() = worldRotation;
  return map;
}

class QuaternionRootOnlyCases
    : public ::testing::TestWithParam<EquivalenceRobot> {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<RootJointModelPair>> models =
        RootJointModelPair::Create(GetParam());
    ASSERT_TRUE(models.ok()) << models.status();
    models_ = *std::move(models);
  }

  /** A random quaternion-root configuration with the base attitude `xi`, a
   * random base position and random joints. */
  vector_t configurationAt(const vector4_t& xi, std::mt19937& generator) const {
    vector_t q = randomEulerMotion(models_->eulerModel(),
                                   MotionSamplingBounds{}, generator)
                     .q;
    vector_t quaternionQ(models_->quaternionModel().nq);
    quaternionQ << q.head<3>(), xi, q.tail(models_->numJoints());
    return quaternionQ;
  }

  /** v = [pd_W, w_B, qd_j] of the sampling bounds (the body rates as large as
   * the Euler rates). */
  vector_t velocity(std::mt19937& generator) const {
    return randomEulerMotion(models_->eulerModel(), MotionSamplingBounds{},
                             generator)
        .v;
  }

  std::unique_ptr<RootJointModelPair> models_;
};

TEST_P(QuaternionRootOnlyCases,
       DynamicsAtEveryAttitudeIsTheLevelDynamicsUnderARotatedGravity) {
  // (D) at a pitch of exactly and almost +-90 degrees and at random attitudes:
  // the quaternion root's forward and inverse dynamics, mass matrix, centroidal
  // momentum and frame placements equal those of the same robot at the identity
  // attitude with gravity W g - a computation with no attitude in it at all, so
  // nothing is singular at gimbal lock. The mass matrix keeps its eigenvalues.
  const pinocchio::Model& model = models_->quaternionModel();
  const Eigen::Index nv = model.nv;
  pinocchio::Data data(model);
  std::mt19937 generator(401);
  for (const TestAttitude& attitude : testAttitudes(generator)) {
    const vector_t q = configurationAt(attitude.quaternion, generator);
    const vector_t v = velocity(generator);
    const vector_t tau = uniformVector(nv, /*halfWidth=*/50.0, generator);
    const vector_t a = uniformVector(nv, /*halfWidth=*/10.0, generator);
    const matrix3_t worldRotation = rotationOf(attitude.quaternion).transpose();
    const matrix_t map = covarianceMap(worldRotation, nv);

    pinocchio::Model levelModel = model;
    levelModel.gravity.linear(worldRotation * model.gravity.linear());
    pinocchio::Data levelData(levelModel);
    vector_t levelQ = q;
    levelQ.head<3>() = worldRotation * q.head<3>();
    levelQ.segment<4>(3) = vector4_t(0.0, 0.0, 0.0, 1.0);
    const vector_t levelV = map * v;

    const vector_t acceleration = pinocchio::aba(model, data, q, v, tau);
    const vector_t levelAcceleration =
        pinocchio::aba(levelModel, levelData, levelQ, levelV, map * tau);
    ASSERT_TRUE(acceleration.allFinite()) << attitude.label;
    const matrix_t massMatrix =
        symmetricFromUpper(pinocchio::crba(model, data, q));
    const matrix_t levelMassMatrix =
        symmetricFromUpper(pinocchio::crba(levelModel, levelData, levelQ));
    const vector_t eigenvalues =
        Eigen::SelfAdjointEigenSolver<matrix_t>(massMatrix).eigenvalues();
    const vector_t levelEigenvalues =
        Eigen::SelfAdjointEigenSolver<matrix_t>(levelMassMatrix).eigenvalues();
    // (T4) of forward dynamics (testRootJointDynamicsEquivalence.cpp): the
    // residual's round-off times the condition number of M, the same at both
    // attitudes.
    const scalar_t condition =
        levelEigenvalues.maxCoeff() / levelEigenvalues.minCoeff();
    ASSERT_TRUE(isCloseRelative(map * acceleration, levelAcceleration,
                                kRoundOffTolerance * condition))
        << attitude.label;

    const vector_t torques = pinocchio::rnea(model, data, q, v, a);
    const vector_t levelTorques =
        pinocchio::rnea(levelModel, levelData, levelQ, levelV, map * a);
    ASSERT_TRUE(
        isCloseRelative(map * torques, levelTorques, kRoundOffTolerance))
        << attitude.label;

    ASSERT_TRUE(isCloseRelative(massMatrix,
                                map.transpose() * levelMassMatrix * map,
                                kRoundOffTolerance))
        << attitude.label;
    // Weyl: each eigenvalue moves by at most the 2-norm of the perturbation,
    // here (T1) of the largest.
    ASSERT_TRUE(
        isCloseAbsolute(eigenvalues, levelEigenvalues,
                        kRoundOffTolerance * levelEigenvalues.maxCoeff()))
        << attitude.label;

    pinocchio::ccrba(model, data, q, v);
    pinocchio::ccrba(levelModel, levelData, levelQ, levelV);
    ASSERT_TRUE(isCloseRelative(worldRotation * data.hg.linear(),
                                levelData.hg.linear(), kRoundOffTolerance))
        << attitude.label;
    ASSERT_TRUE(isCloseRelative(worldRotation * data.hg.angular(),
                                levelData.hg.angular(), kRoundOffTolerance))
        << attitude.label;

    pinocchio::framesForwardKinematics(model, data, q);
    pinocchio::framesForwardKinematics(levelModel, levelData, levelQ);
    for (int frame = 0; frame < model.nframes; ++frame) {
      // A frame of the universe stays where it is: only the robot is rotated.
      if (model.frames[frame].parentJoint == 0) continue;
      ASSERT_TRUE(isCloseRelative(worldRotation * data.oMf[frame].translation(),
                                  levelData.oMf[frame].translation(),
                                  kRoundOffTolerance))
          << model.frames[frame].name << ", " << attitude.label;
      ASSERT_TRUE(isCloseAbsolute(worldRotation * data.oMf[frame].rotation(),
                                  levelData.oMf[frame].rotation(),
                                  kRoundOffTolerance))
          << model.frames[frame].name << ", " << attitude.label;
    }
  }
}

TEST_P(QuaternionRootOnlyCases, TheEulerRootLosesRankAtGimbalLockAsCosPitch) {
  // The positive control of the test above: the Euler root's centroidal map has
  // the base rotation block I_G R T_B(Theta) where the quaternion root has I_G
  // R (C5), and det T_B = -cos(pitch), so det(A_G,e[3:6, 3:6]) / det(A_G[3:6,
  // 3:6]) = -cos(pitch), exactly 0 at gimbal lock; the quaternion root's block
  // keeps det I_G. Relative round-off of a 3 x 3 determinant: (T1).
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  std::mt19937 generator(403);
  for (const scalar_t pitch : {0.3, M_PI / 2.0 - 1.0e-1, M_PI / 2.0 - 1.0e-3,
                               M_PI / 2.0 - 1.0e-6, M_PI / 2.0, -M_PI / 2.0}) {
    PinocchioMotion eulerMotion =
        randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    eulerMotion.q(4) = pitch;
    vector_t q(quaternionModel.nq);
    q << eulerMotion.q.head<3>(),
        quaternionFromEulerZyx(eulerMotion.q.segment<3>(3)),
        eulerMotion.q.tail(models_->numJoints());
    const matrix3_t eulerBlock =
        pinocchio::ccrba(eulerModel, eulerData, eulerMotion.q, eulerMotion.v)
            .block<3, 3>(3, 3);
    const matrix3_t quaternionBlock =
        pinocchio::ccrba(quaternionModel, quaternionData, q, eulerMotion.v)
            .block<3, 3>(3, 3);
    const scalar_t compositeInertiaDeterminant =
        quaternionData.Ig.inertia().matrix().determinant();
    EXPECT_LE(
        std::abs(quaternionBlock.determinant() - compositeInertiaDeterminant),
        kRoundOffTolerance * compositeInertiaDeterminant)
        << "pitch " << pitch;
    EXPECT_LE(
        std::abs(eulerBlock.determinant() / quaternionBlock.determinant() +
                 std::cos(pitch)),
        kRoundOffTolerance)
        << "pitch " << pitch;
  }
}

TEST_P(QuaternionRootOnlyCases,
       PathsThroughGimbalLockAndAcrossYawPiAreContinuous) {
  // pinocchio::integrate drives the quaternion root at a constant body rate
  // through a pitch of 90 degrees (about body y, from level yaw and roll) and
  // across a yaw of +-pi (about body z, from level pitch and roll). Along each
  // path the quaternion moves continuously, |xi_k - xi_k-1| <= |w_B| dt / 2 (a
  // chord of the arc), Pinocchio's difference of consecutive configurations is
  // the step, and the Euler boundary's angles jump - the yaw by 2 pi across
  // +-pi, yaw and roll by pi through gimbal lock - while the Euler root at
  // those angles has the quaternion root's placements (T5).
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  constexpr scalar_t kRate = 1.0;
  constexpr scalar_t kStep = 0.01;
  constexpr int kSteps = 10;
  struct Path {
    std::string label;
    vector3_t startEulerZyx;
    vector3_t bodyRate;
  };
  const Path paths[] = {
      {.label = "through a pitch of 90 degrees",
       .startEulerZyx = vector3_t(0.0, M_PI / 2.0 - 0.05, 0.0),
       .bodyRate = vector3_t(0.0, kRate, 0.0)},
      {.label = "across a yaw of +-pi",
       .startEulerZyx = vector3_t(M_PI - 0.05, 0.0, 0.0),
       .bodyRate = vector3_t(0.0, 0.0, kRate)},
  };
  std::mt19937 generator(405);
  for (const Path& path : paths) {
    vector_t q =
        configurationAt(quaternionFromEulerZyx(path.startEulerZyx), generator);
    vector_t v = vector_t::Zero(quaternionModel.nv);
    v.segment<3>(3) = path.bodyRate;
    scalar_t largestAngleJump = 0.0;
    vector3_t previousAngles = path.startEulerZyx;
    for (int step = 1; step <= kSteps; ++step) {
      const vector_t next = pinocchio::integrate(quaternionModel, q, kStep * v);
      EXPECT_LE((next.segment<4>(3) - q.segment<4>(3)).norm(),
                kRate * kStep / 2.0 + kRoundOffTolerance)
          << path.label << ", step " << step;
      EXPECT_TRUE(
          isCloseAbsolute(pinocchio::difference(quaternionModel, q, next),
                          kStep * v, kRoundOffTolerance))
          << path.label << ", step " << step;
      q = next;

      const vector3_t angles = eulerZyxFromQuaternion(q.segment<4>(3));
      largestAngleJump = std::max(
          largestAngleJump, (angles - previousAngles).cwiseAbs().maxCoeff());
      previousAngles = angles;
      vector_t eulerQ(eulerModel.nq);
      eulerQ << q.head<3>(), angles, q.tail(models_->numJoints());
      pinocchio::framesForwardKinematics(quaternionModel, quaternionData, q);
      pinocchio::framesForwardKinematics(eulerModel, eulerData, eulerQ);
      for (int frame = 0; frame < quaternionModel.nframes; ++frame) {
        ASSERT_TRUE(isCloseRelative(quaternionData.oMf[frame].translation(),
                                    eulerData.oMf[frame].translation(),
                                    kBoundaryInverseTolerance))
            << quaternionModel.frames[frame].name << ", " << path.label
            << ", step " << step;
        ASSERT_TRUE(isCloseAbsolute(quaternionData.oMf[frame].rotation(),
                                    eulerData.oMf[frame].rotation(),
                                    kBoundaryInverseTolerance))
            << quaternionModel.frames[frame].name << ", " << path.label
            << ", step " << step;
      }
    }
    // The Euler coordinates jump by about pi or 2 pi where the quaternion moved
    // by a 0.01 rad step.
    EXPECT_GT(largestAngleJump, 3.0) << path.label;
  }
}

TEST_P(QuaternionRootOnlyCases, TheNegatedQuaternionIsTheSameRobot) {
  // xi and -xi are the same rotation: Pinocchio reads R(xi) quadratically, so
  // every quantity is the same (bit for bit: every product of two negated
  // coefficients is the product of the coefficients), and pinocchio::difference
  // between the two configurations is zero.
  const pinocchio::Model& model = models_->quaternionModel();
  pinocchio::Data data(model);
  pinocchio::Data negatedData(model);
  std::mt19937 generator(407);
  for (const TestAttitude& attitude : testAttitudes(generator)) {
    const vector_t q = configurationAt(attitude.quaternion, generator);
    vector_t negatedQ = q;
    negatedQ.segment<4>(3) = -q.segment<4>(3);
    const vector_t v = velocity(generator);
    const vector_t tau = uniformVector(model.nv, /*halfWidth=*/50.0, generator);
    EXPECT_TRUE(isCloseAbsolute(pinocchio::difference(model, q, negatedQ),
                                vector_t::Zero(model.nv), kRoundOffTolerance))
        << attitude.label;
    EXPECT_TRUE(
        isCloseAbsolute(pinocchio::aba(model, negatedData, negatedQ, v, tau),
                        pinocchio::aba(model, data, q, v, tau),
                        /*absoluteTolerance=*/0.0))
        << attitude.label;
    EXPECT_TRUE(isCloseAbsolute(
        pinocchio::ccrba(model, negatedData, negatedQ, v),
        pinocchio::ccrba(model, data, q, v), /*absoluteTolerance=*/0.0))
        << attitude.label;
    EXPECT_TRUE(isCloseAbsolute(pinocchio::crba(model, negatedData, negatedQ),
                                pinocchio::crba(model, data, q),
                                /*absoluteTolerance=*/0.0))
        << attitude.label;
    pinocchio::framesForwardKinematics(model, data, q);
    pinocchio::framesForwardKinematics(model, negatedData, negatedQ);
    for (int frame = 0; frame < model.nframes; ++frame) {
      ASSERT_TRUE(isCloseAbsolute(negatedData.oMf[frame].toHomogeneousMatrix(),
                                  data.oMf[frame].toHomogeneousMatrix(),
                                  /*absoluteTolerance=*/0.0))
          << model.frames[frame].name << ", " << attitude.label;
    }
  }
}

TEST_P(QuaternionRootOnlyCases,
       TheEulerBoundaryInverseReproducesThePlacementsAtEveryAttitude) {
  // The display and telemetry read the base orientation as
  // eulerZyxFromQuaternion(xi): at every attitude, gimbal lock included (where
  // it splits yaw and roll by its own rule), the Euler root at those angles has
  // the quaternion root's placements (T5).
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  std::mt19937 generator(409);
  for (const TestAttitude& attitude : testAttitudes(generator)) {
    const vector_t q = configurationAt(attitude.quaternion, generator);
    vector_t eulerQ(eulerModel.nq);
    eulerQ << q.head<3>(), eulerZyxFromQuaternion(attitude.quaternion),
        q.tail(models_->numJoints());
    pinocchio::framesForwardKinematics(quaternionModel, quaternionData, q);
    pinocchio::framesForwardKinematics(eulerModel, eulerData, eulerQ);
    for (int frame = 0; frame < quaternionModel.nframes; ++frame) {
      ASSERT_TRUE(isCloseRelative(quaternionData.oMf[frame].translation(),
                                  eulerData.oMf[frame].translation(),
                                  kBoundaryInverseTolerance))
          << quaternionModel.frames[frame].name << ", " << attitude.label;
      ASSERT_TRUE(isCloseAbsolute(quaternionData.oMf[frame].rotation(),
                                  eulerData.oMf[frame].rotation(),
                                  kBoundaryInverseTolerance))
          << quaternionModel.frames[frame].name << ", " << attitude.label;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(EveryShippedRobot,
                         QuaternionRootOnlyCases,
                         ::testing::ValuesIn(equivalenceRobots().begin(),
                                             equivalenceRobots().end()),
                         equivalenceRobotName);

}  // namespace
}  // namespace ocs2::humanoid
