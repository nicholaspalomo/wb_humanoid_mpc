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
#include <vector>

#include "Eigen/Eigenvalues"
#include "Eigen/LU"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/joint/joint-free-flyer.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/RootJointModelPair.h"

/*
 * The production root joint against Pinocchio's own floating base. The MPCs' models root at Composite(Translation,
 * Spherical); the suite's other root-joint tests hold it to the Euler root Composite(Translation, SphericalZYX), and the
 * two share the Translation sub-joint, so a defect common to both would cancel. Here the comparison is with a single
 * FreeFlyer joint, which shares nothing with it but the configuration's layout [p_W, xi, q_j]:
 *
 *   (F1) placements  the FreeFlyer's placement of [p_W, xi] is (R(xi), p_W), the composite's T(p_W) R(xi): every frame is
 *        placed alike at the same q;
 *   (F2) velocities  the FreeFlyer's velocity is the root's twist in its own frame, v_f = [R' pd_W; w_B; qd_j] = J_m v
 *        with J_m = blockdiag(R', I_3, I) and v = [pd_W, w_B, qd_j] the composite's;
 *   (F3) accelerations  a_f = J_m a + (dJ_m/dt) v, (dJ_m/dt) v = [-w_B x R' pd_W; 0; 0] (from d(R')/dt = -[w_B]x R');
 *   (F4) forces  power is invariant, so tau = J_m' tau_f: rnea, M = J_m' M_f J_m (crba), n = J_m' (n_f + M_f (dJ_m/dt) v)
 *        (nonLinearEffects), and aba solves the same equations;
 *   (F5) momentum  h_G = A_G,f v_f, so A_G = A_G,f J_m (computeCentroidalMap) and the momenta agree.
 * Also the structure of the base block of A_G that the centroidal MPC's closed-form inverse assumes (design section
 * 2.5), A_b = [[m I, -m [c - p_W]x R], [0, I_G R]], on every robot, at a pitch of +-90 degrees, a yaw of pi and the
 * negated quaternion (bit for bit the same, R(-xi) = R(xi) coefficient for coefficient).
 *
 * Tolerances: (T1) and (T4) (indexed in RootJointModelPair.h): the two roots form R from the same quaternion and the
 * rest of each tree from the same joints, so the kinematics agree like two evaluations of one chain of transforms.
 */

namespace ocs2::humanoid {
namespace {

constexpr int kSamples = 100;

/** (T1) (testRootJointKinematicsEquivalence.cpp, kKinematicsTolerance), relative to the compared quantity (floor 1). */
constexpr scalar_t kRoundOffTolerance = 1.0e-12;

/** `model` with its root joint replaced by a FreeFlyer: every other joint, inertia and frame copied in its order. */
pinocchio::Model freeFlyerTwinOf(const pinocchio::Model& model) {
  pinocchio::Model twin;
  twin.name = model.name;
  twin.gravity = model.gravity;
  for (pinocchio::JointIndex joint = 1; joint < static_cast<pinocchio::JointIndex>(model.njoints); ++joint) {
    if (joint == 1) {
      twin.addJoint(/*parent=*/0, pinocchio::JointModel(pinocchio::JointModelFreeFlyer()), model.jointPlacements[joint],
                    model.names[joint]);
    } else {
      twin.addJoint(model.parents[joint], model.joints[joint], model.jointPlacements[joint], model.names[joint]);
    }
    twin.inertias[joint] = model.inertias[joint];
  }
  for (pinocchio::FrameIndex frame = 1; frame < static_cast<pinocchio::FrameIndex>(model.nframes); ++frame) {
    // The joints' inertias already hold every body's: the frames add none.
    twin.addFrame(model.frames[frame], /*append_inertia=*/false);
  }
  twin.armature = model.armature;
  return twin;
}

/** J_m and (dJ_m/dt) v of a motion of the composite root: (F2) and (F3). Passive data. */
struct FreeFlyerMap {
  matrix_t map;
  vector_t mapRateTimesVelocity;
};

FreeFlyerMap freeFlyerMapOf(const PinocchioMotion& motion) {
  const Eigen::Index nv = motion.v.size();
  const matrix3_t rotation = quaternionRotationMatrix<scalar_t>(vector4_t(motion.q.segment<4>(3)));
  FreeFlyerMap map{.map = matrix_t::Identity(nv, nv), .mapRateTimesVelocity = vector_t::Zero(nv)};
  map.map.topLeftCorner<3, 3>() = rotation.transpose();
  const vector3_t bodyLinearVelocity = rotation.transpose() * motion.v.head<3>();
  map.mapRateTimesVelocity.head<3>() = -vector3_t(motion.v.segment<3>(3)).cross(bodyLinearVelocity);
  return map;
}

class RootJointFreeFlyerEquivalence : public ::testing::TestWithParam<EquivalenceRobot> {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<RootJointModelPair>> models = RootJointModelPair::Create(GetParam());
    ASSERT_TRUE(models.ok()) << models.status();
    models_ = *std::move(models);
    freeFlyer_ = std::make_unique<pinocchio::Model>(freeFlyerTwinOf(models_->quaternionModel()));
    ASSERT_EQ(freeFlyer_->nq, models_->quaternionModel().nq);
    ASSERT_EQ(freeFlyer_->nv, models_->quaternionModel().nv);
    ASSERT_EQ(freeFlyer_->nframes, models_->quaternionModel().nframes);
  }

  /**
   * Random motions of the composite root: through the chart from random Euler motions (yaw and roll in [-pi, pi], pitch
   * within 80 degrees), and with the quaternion replaced by the attitudes the Euler root cannot reach - a pitch of
   * exactly +-90 degrees at a yaw of pi - and by its negation.
   */
  std::vector<PinocchioMotion> motions() const {
    std::mt19937 generator(/*sd=*/20261011);
    std::vector<PinocchioMotion> result;
    const std::vector<vector4_t> special = {quaternionFromEulerZyx(vector3_t(M_PI, 0.5 * M_PI, 0.0)),
                                            quaternionFromEulerZyx(vector3_t(M_PI, -0.5 * M_PI, 0.3)),
                                            quaternionFromEulerZyx(vector3_t(-M_PI, 0.5 * M_PI, -2.0))};
    for (int sample = 0; sample < kSamples; ++sample) {
      const PinocchioMotion euler = randomEulerMotion(models_->eulerModel(), MotionSamplingBounds{}, generator);
      absl::StatusOr<PinocchioMotion> motion = quaternionMotionOf(euler, models_->pinocchioStateLayout());
      EXPECT_TRUE(motion.ok()) << motion.status();
      if (!motion.ok()) continue;
      if (sample < static_cast<int>(special.size())) motion->q.segment<4>(3) = special[static_cast<size_t>(sample)];
      result.push_back(*motion);
      PinocchioMotion negated = *motion;
      negated.q.segment<4>(3) = -motion->q.segment<4>(3);
      result.push_back(negated);
    }
    return result;
  }

  std::unique_ptr<RootJointModelPair> models_;
  std::unique_ptr<pinocchio::Model> freeFlyer_;
};

TEST_P(RootJointFreeFlyerEquivalence, PlacementsTwistsAndAccelerationsAreTheFreeFlyers) {
  const pinocchio::Model& model = models_->quaternionModel();
  pinocchio::Data data(model);
  pinocchio::Data freeFlyerData(*freeFlyer_);
  for (const PinocchioMotion& motion : motions()) {
    const FreeFlyerMap map = freeFlyerMapOf(motion);
    const vector_t freeFlyerVelocity = map.map * motion.v;
    const vector_t freeFlyerAcceleration = map.map * motion.a + map.mapRateTimesVelocity;
    pinocchio::forwardKinematics(model, data, motion.q, motion.v, motion.a);
    pinocchio::updateFramePlacements(model, data);
    pinocchio::forwardKinematics(*freeFlyer_, freeFlyerData, motion.q, freeFlyerVelocity, freeFlyerAcceleration);
    pinocchio::updateFramePlacements(*freeFlyer_, freeFlyerData);
    for (pinocchio::FrameIndex frame = 1; frame < static_cast<pinocchio::FrameIndex>(model.nframes); ++frame) {
      SCOPED_TRACE(model.frames[frame].name);
      // (F1)
      ASSERT_TRUE(isCloseRelative(data.oMf[frame].translation(), freeFlyerData.oMf[frame].translation(), kRoundOffTolerance));
      ASSERT_TRUE(isCloseRelative(data.oMf[frame].rotation(), freeFlyerData.oMf[frame].rotation(), kRoundOffTolerance));
      // (F2), in the frame and in the world-aligned frame at its origin.
      for (const pinocchio::ReferenceFrame reference : {pinocchio::LOCAL, pinocchio::LOCAL_WORLD_ALIGNED}) {
        ASSERT_TRUE(isCloseRelative(pinocchio::getFrameVelocity(model, data, frame, reference).toVector(),
                                    pinocchio::getFrameVelocity(*freeFlyer_, freeFlyerData, frame, reference).toVector(),
                                    kRoundOffTolerance));
        // (F3): the classical acceleration of the frame's origin.
        ASSERT_TRUE(isCloseRelative(pinocchio::getFrameClassicalAcceleration(model, data, frame, reference).toVector(),
                                    pinocchio::getFrameClassicalAcceleration(*freeFlyer_, freeFlyerData, frame, reference).toVector(),
                                    kRoundOffTolerance));
      }
    }
  }
}

TEST_P(RootJointFreeFlyerEquivalence, TheEquationsOfMotionAreTheFreeFlyersThroughThePowerBalance) {
  const pinocchio::Model& model = models_->quaternionModel();
  pinocchio::Data data(model);
  pinocchio::Data freeFlyerData(*freeFlyer_);
  std::mt19937 generator(/*sd=*/7);
  for (const PinocchioMotion& motion : motions()) {
    const FreeFlyerMap map = freeFlyerMapOf(motion);
    const vector_t freeFlyerVelocity = map.map * motion.v;
    const vector_t freeFlyerAcceleration = map.map * motion.a + map.mapRateTimesVelocity;
    // (F4) rnea: tau = J_m' tau_f.
    const vector_t torques = pinocchio::rnea(model, data, motion.q, motion.v, motion.a);
    const vector_t freeFlyerTorques = pinocchio::rnea(*freeFlyer_, freeFlyerData, motion.q, freeFlyerVelocity, freeFlyerAcceleration);
    ASSERT_TRUE(isCloseRelative(torques, map.map.transpose() * freeFlyerTorques, kRoundOffTolerance));
    // crba: M = J_m' M_f J_m.
    const matrix_t massMatrix = symmetricFromUpper(pinocchio::crba(model, data, motion.q));
    const matrix_t freeFlyerMassMatrix = symmetricFromUpper(pinocchio::crba(*freeFlyer_, freeFlyerData, motion.q));
    ASSERT_TRUE(isCloseRelative(massMatrix, map.map.transpose() * freeFlyerMassMatrix * map.map, kRoundOffTolerance));
    // nonLinearEffects: n = J_m' (n_f + M_f (dJ_m/dt) v).
    const vector_t effects = pinocchio::nonLinearEffects(model, data, motion.q, motion.v);
    const vector_t freeFlyerEffects = pinocchio::nonLinearEffects(*freeFlyer_, freeFlyerData, motion.q, freeFlyerVelocity);
    ASSERT_TRUE(isCloseRelative(effects, map.map.transpose() * (freeFlyerEffects + freeFlyerMassMatrix * map.mapRateTimesVelocity),
                                kRoundOffTolerance));
    // aba: the accelerations of a torque, a_f = J_m a + (dJ_m/dt) v with tau_f = J_m^-T tau, within (T4): the round-off of
    // the solve times the condition number of M.
    const vector_t torque = uniformVector(model.nv, /*halfWidth=*/50.0, generator);
    const vector_t acceleration = pinocchio::aba(model, data, motion.q, motion.v, torque);
    const vector_t freeFlyerTorque = map.map.transpose().partialPivLu().solve(torque);
    const vector_t freeFlyerFromAba = pinocchio::aba(*freeFlyer_, freeFlyerData, motion.q, freeFlyerVelocity, freeFlyerTorque);
    const vector_t eigenvalues = Eigen::SelfAdjointEigenSolver<matrix_t>(massMatrix).eigenvalues();
    const scalar_t condition = eigenvalues.maxCoeff() / eigenvalues.minCoeff();
    ASSERT_TRUE(isCloseRelative(map.map * acceleration + map.mapRateTimesVelocity, freeFlyerFromAba, kRoundOffTolerance * condition));
  }
}

TEST_P(RootJointFreeFlyerEquivalence, TheCentroidalMapIsTheFreeFlyersAndItsBaseBlockHasTheClosedFormStructure) {
  const pinocchio::Model& model = models_->quaternionModel();
  pinocchio::Data data(model);
  pinocchio::Data freeFlyerData(*freeFlyer_);
  const scalar_t mass = pinocchio::computeTotalMass(model);
  for (const PinocchioMotion& motion : motions()) {
    const FreeFlyerMap map = freeFlyerMapOf(motion);
    // (F5)
    const matrix_t momentumMatrix = pinocchio::computeCentroidalMap(model, data, motion.q);
    const matrix_t freeFlyerMomentumMatrix = pinocchio::computeCentroidalMap(*freeFlyer_, freeFlyerData, motion.q);
    ASSERT_TRUE(isCloseRelative(momentumMatrix, freeFlyerMomentumMatrix * map.map, kRoundOffTolerance));
    ASSERT_TRUE(isCloseRelative(momentumMatrix * motion.v, freeFlyerMomentumMatrix * (map.map * motion.v), kRoundOffTolerance));

    // A_b = [[m I, -m [c - p_W]x R], [0, I_G R]]: linear momentum m c_dot = m (pd_W + R w_B x (c - p_W)), and the
    // angular momentum about c of a turn of the whole robot, I_G w_W = I_G R w_B. I_G from ccrba, the centroidal
    // composite inertia (world-aligned, about c).
    pinocchio::ccrba(model, data, motion.q, motion.v);
    const matrix_t baseBlock = data.Ag.leftCols<6>();
    const matrix3_t rotation = quaternionRotationMatrix<scalar_t>(vector4_t(motion.q.segment<4>(3)));
    const vector3_t offset = data.com[0] - motion.q.head<3>();
    matrix3_t offsetCross;
    offsetCross << 0.0, -offset.z(), offset.y(), offset.z(), 0.0, -offset.x(), -offset.y(), offset.x(), 0.0;
    matrix_t closedForm = matrix_t::Zero(6, 6);
    closedForm.topLeftCorner<3, 3>() = mass * matrix3_t::Identity();
    closedForm.topRightCorner<3, 3>() = -mass * offsetCross * rotation;
    closedForm.bottomRightCorner<3, 3>() = data.Ig.inertia().matrix() * rotation;
    ASSERT_TRUE(isCloseRelative(baseBlock, closedForm, kRoundOffTolerance));
  }
  // The negated quaternion: the same rotation coefficient for coefficient, so the same matrix bit for bit.
  for (const PinocchioMotion& motion : motions()) {
    PinocchioMotion negated = motion;
    negated.q.segment<4>(3) = -motion.q.segment<4>(3);
    const matrix_t momentumMatrix = pinocchio::computeCentroidalMap(model, data, motion.q);
    pinocchio::Data negatedData(model);
    EXPECT_EQ(pinocchio::computeCentroidalMap(model, negatedData, negated.q), momentumMatrix);
  }
}

INSTANTIATE_TEST_SUITE_P(Shipped, RootJointFreeFlyerEquivalence, ::testing::ValuesIn(equivalenceRobots()), equivalenceRobotName);

}  // namespace
}  // namespace ocs2::humanoid
