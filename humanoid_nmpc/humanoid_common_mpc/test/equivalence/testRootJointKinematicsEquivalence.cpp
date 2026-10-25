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
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/RootJointModelPair.h"

/*
 * Kinematics through the two root joints (RootJointModelPair.h): at seeded
 * random motions of every shipped robot, mapped from the Euler root to the
 * quaternion root by the chart (C1)-(C3), every frame placement, frame
 * Jacobian, frame twist and frame acceleration and the center of mass with its
 * velocity, acceleration and Jacobian are the same; and the chart is
 * Pinocchio's own: its velocity is the root body's velocity Pinocchio computes
 * on both roots, and its configuration map moves as pinocchio::integrate moves
 * the quaternion root.
 */

namespace ocs2::humanoid {
namespace {

/** Seeded random motions per robot and test: hundreds, the size of the suite
 * the plan asks for. */
constexpr int kSamples = 200;

/**
 * Round-off bound (T1) on two evaluations of the same chain of rigid-body
 * transforms whose root rotations agree to round-off, relative to the compared
 * quantity (floor 1). R(xi(Theta)) (the quaternion's quadratic form) and R_z
 * R_y R_x (Pinocchio's trigonometric products) agree entry by entry to about 4
 * eps, eps = 2.2e-16, xi(Theta) itself carrying 2 eps; a frame at depth d below
 * the root accumulates about d such roundings, d <= 12 in the shipped trees, on
 * lengths of at most 4 m (|p_W| <= 3.3 plus the limbs), so placements agree to
 * about 12 * 4 * eps * 4 = 4e-14 and Jacobians, twists and accelerations (the
 * same products times velocities of order 10) to about 4e-13 / 10 relative.
 * 1e-12 is 25 times that; the measured difference was at most 3e-15 relative
 * over the five robots when this was written.
 */
constexpr scalar_t kKinematicsTolerance = 1.0e-12;

/**
 * Step and bound (T2) of the central difference of the configuration map along
 * a motion. The difference log(Phi(q_e - h v_e)^-1 Phi(q_e + h v_e)) / 2h
 * equals the tangent velocity up to h^2 / 6 times the third derivative of the
 * curve, of order |v_e|^3 <= (3 sqrt(3))^3 = 140 for the sampled Euler rates:
 * 1e-10 * 140 / 6 = 2.3e-9 at h = 1e-5, plus round-off eps * |q| / h <= 2.2e-16
 * * 4 / 1e-5 = 9e-11. 1e-7 is 40 times their sum, and a wrong rate map (T_B^T,
 * or the Euler rates themselves) is off by order 1. Measured when this was
 * written: at most 3.3e-10.
 */
constexpr scalar_t kIntegratorStep = 1.0e-5;
constexpr scalar_t kIntegratorTolerance = 1.0e-7;

class RootJointKinematicsEquivalence
    : public ::testing::TestWithParam<EquivalenceRobot> {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<RootJointModelPair>> models =
        RootJointModelPair::Create(GetParam());
    ASSERT_TRUE(models.ok()) << models.status();
    models_ = *std::move(models);
  }

  /** The motion `eulerMotion` on the quaternion root; fails the test when the
   * chart refuses it. */
  PinocchioMotion quaternionMotion(const PinocchioMotion& eulerMotion) const {
    absl::StatusOr<PinocchioMotion> motion =
        quaternionMotionOf(eulerMotion, models_->pinocchioStateLayout());
    EXPECT_TRUE(motion.ok()) << motion.status();
    return motion.ok() ? *std::move(motion) : PinocchioMotion{};
  }

  ChartVelocityMaps velocityMaps(const PinocchioMotion& eulerMotion) const {
    absl::StatusOr<ChartVelocityMaps> maps =
        chartVelocityMapsAt(eulerMotion, models_->pinocchioStateLayout());
    EXPECT_TRUE(maps.ok()) << maps.status();
    return maps.ok() ? *std::move(maps) : ChartVelocityMaps{};
  }

  std::unique_ptr<RootJointModelPair> models_;
};

TEST_P(RootJointKinematicsEquivalence, ModelsDifferOnlyInTheRootJoint) {
  // The premise of every other test: the same bodies, inertias, joint
  // placements and frames, index for index, so that a difference anywhere else
  // can only come from the root joint.
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  EXPECT_EQ(quaternionModel.nq, 7 + models_->numJoints());
  EXPECT_EQ(eulerModel.nq, 6 + models_->numJoints());
  EXPECT_EQ(quaternionModel.nv, eulerModel.nv);
  EXPECT_EQ(quaternionModel.joints[1].nq(), 7);
  EXPECT_EQ(eulerModel.joints[1].nq(), 6);
  for (int joint = 0; joint < eulerModel.njoints; ++joint) {
    EXPECT_EQ(quaternionModel.inertias[joint], eulerModel.inertias[joint])
        << eulerModel.names[joint];
    EXPECT_EQ(quaternionModel.jointPlacements[joint],
              eulerModel.jointPlacements[joint])
        << eulerModel.names[joint];
    EXPECT_EQ(quaternionModel.parents[joint], eulerModel.parents[joint])
        << eulerModel.names[joint];
  }
  for (int frame = 0; frame < eulerModel.nframes; ++frame) {
    EXPECT_EQ(quaternionModel.frames[frame].placement,
              eulerModel.frames[frame].placement)
        << eulerModel.frames[frame].name;
    EXPECT_EQ(quaternionModel.frames[frame].parentJoint,
              eulerModel.frames[frame].parentJoint)
        << eulerModel.frames[frame].name;
  }
  EXPECT_EQ(quaternionModel.lowerPositionLimit.tail(models_->numJoints()),
            eulerModel.lowerPositionLimit.tail(models_->numJoints()));
  EXPECT_EQ(quaternionModel.upperPositionLimit.tail(models_->numJoints()),
            eulerModel.upperPositionLimit.tail(models_->numJoints()));
}

TEST_P(RootJointKinematicsEquivalence,
       ChartVelocityIsTheRootBodyVelocityOfBothJoints) {
  // (C2) against Pinocchio's joint models: the root body's spatial velocity in
  // its own frame, [R^T pd_W; w_B], is the same on both roots, and the chart's
  // w_B = T_B(Theta) dTheta/dt is what SphericalZYX's motion subspace gives.
  // The top block row of P_e [v_e; a_e] repeats Phi's velocity (C3).
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  std::mt19937 generator(101);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion =
        randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const PinocchioMotion motion = quaternionMotion(eulerMotion);
    pinocchio::forwardKinematics(eulerModel, eulerData, eulerMotion.q,
                                 eulerMotion.v);
    pinocchio::forwardKinematics(quaternionModel, quaternionData, motion.q,
                                 motion.v);
    const vector6_t eulerRootVelocity = eulerData.v[1].toVector();
    const vector6_t quaternionRootVelocity = quaternionData.v[1].toVector();
    ASSERT_TRUE(isCloseRelative(quaternionRootVelocity, eulerRootVelocity,
                                kKinematicsTolerance))
        << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(motion.v.segment<3>(3),
                                eulerData.v[1].angular(), kKinematicsTolerance))
        << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(motion.v.head<3>(), eulerMotion.v.head<3>(),
                                /*relativeTolerance=*/0.0))
        << "sample " << sample;
    ASSERT_TRUE(
        isCloseRelative(velocityMaps(eulerMotion).velocityMap * eulerMotion.v,
                        motion.v, kKinematicsTolerance))
        << "sample " << sample;
  }
}

TEST_P(RootJointKinematicsEquivalence,
       ChartFollowsPinocchiosConfigurationIntegrator) {
  // (C1) and (C2) against pinocchio::integrate / difference on the quaternion
  // root: moving the Euler configuration along v_e moves Phi(q_e) along v = J_v
  // v_e in Pinocchio's tangent (translation in the world frame, rotation as a
  // body right perturbation), at every sampled attitude.
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  std::mt19937 generator(103);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion =
        randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    PinocchioMotion forward = eulerMotion;
    forward.q += kIntegratorStep * eulerMotion.v;
    PinocchioMotion backward = eulerMotion;
    backward.q -= kIntegratorStep * eulerMotion.v;
    const vector_t tangentRate =
        pinocchio::difference(quaternionModel, quaternionMotion(backward).q,
                              quaternionMotion(forward).q) /
        (2.0 * kIntegratorStep);
    ASSERT_TRUE(isCloseRelative(tangentRate, quaternionMotion(eulerMotion).v,
                                kIntegratorTolerance))
        << "sample " << sample;
  }
}

TEST_P(RootJointKinematicsEquivalence, EveryFramePlacementIsTheSame) {
  // (C1): R(xi(Theta)) = R_z R_y R_x, so every body and frame below the root
  // has the same placement (T1).
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  std::mt19937 generator(107);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion =
        randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    pinocchio::framesForwardKinematics(eulerModel, eulerData, eulerMotion.q);
    pinocchio::framesForwardKinematics(quaternionModel, quaternionData,
                                       quaternionMotion(eulerMotion).q);
    for (int frame = 0; frame < eulerModel.nframes; ++frame) {
      ASSERT_TRUE(isCloseRelative(quaternionData.oMf[frame].translation(),
                                  eulerData.oMf[frame].translation(),
                                  kKinematicsTolerance))
          << eulerModel.frames[frame].name << ", sample " << sample;
      ASSERT_TRUE(isCloseAbsolute(quaternionData.oMf[frame].rotation(),
                                  eulerData.oMf[frame].rotation(),
                                  kKinematicsTolerance))
          << eulerModel.frames[frame].name << ", sample " << sample;
    }
  }
}

TEST_P(RootJointKinematicsEquivalence, FrameJacobiansMapThroughTheVelocityMap) {
  // (C5): a frame's twist is J(q) v = J(q) J_v v_e for every v_e, so the Euler
  // root's frame Jacobian is the quaternion root's times J_v, in the frame's
  // own axes and in world-aligned axes alike.
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  std::mt19937 generator(109);
  for (int sample = 0; sample < kSamples / 4; ++sample) {
    const PinocchioMotion eulerMotion =
        randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const matrix_t velocityMap = velocityMaps(eulerMotion).velocityMap;
    pinocchio::computeJointJacobians(eulerModel, eulerData, eulerMotion.q);
    pinocchio::computeJointJacobians(quaternionModel, quaternionData,
                                     quaternionMotion(eulerMotion).q);
    for (const pinocchio::ReferenceFrame referenceFrame :
         {pinocchio::LOCAL, pinocchio::LOCAL_WORLD_ALIGNED}) {
      for (int frame = 0; frame < eulerModel.nframes; ++frame) {
        matrix_t eulerJacobian = matrix_t::Zero(6, eulerModel.nv);
        matrix_t quaternionJacobian = matrix_t::Zero(6, quaternionModel.nv);
        pinocchio::getFrameJacobian(eulerModel, eulerData, frame,
                                    referenceFrame, eulerJacobian);
        pinocchio::getFrameJacobian(quaternionModel, quaternionData, frame,
                                    referenceFrame, quaternionJacobian);
        ASSERT_TRUE(isCloseRelative(quaternionJacobian * velocityMap,
                                    eulerJacobian, kKinematicsTolerance))
            << eulerModel.frames[frame].name << ", reference frame "
            << referenceFrame << ", sample " << sample;
      }
    }
  }
}

TEST_P(RootJointKinematicsEquivalence, FrameTwistsAndAccelerationsAreTheSame) {
  // (C2) and (C3) through Pinocchio's forward kinematics: the same motion gives
  // every frame the same twist, spatial acceleration and classical
  // acceleration. The Euler side's dT_B/dt dTheta term enters through
  // SphericalZYX's bias c = dS/dt dTheta, the quaternion side's through the
  // chart's a = J_v a_e + (dJ_v/dt) v_e; dropping the latter (the control
  // below) breaks the equality, so the test sees the term.
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  pinocchio::Data controlData(quaternionModel);
  std::mt19937 generator(113);
  int controlsThatDiffer = 0;
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion =
        randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const PinocchioMotion motion = quaternionMotion(eulerMotion);
    pinocchio::forwardKinematics(eulerModel, eulerData, eulerMotion.q,
                                 eulerMotion.v, eulerMotion.a);
    pinocchio::forwardKinematics(quaternionModel, quaternionData, motion.q,
                                 motion.v, motion.a);
    pinocchio::updateFramePlacements(eulerModel, eulerData);
    pinocchio::updateFramePlacements(quaternionModel, quaternionData);
    for (int frame = 0; frame < eulerModel.nframes; ++frame) {
      const std::string label =
          absl::StrCat(eulerModel.frames[frame].name, ", sample ", sample);
      ASSERT_TRUE(isCloseRelative(
          pinocchio::getFrameVelocity(quaternionModel, quaternionData, frame,
                                      pinocchio::LOCAL_WORLD_ALIGNED)
              .toVector(),
          pinocchio::getFrameVelocity(eulerModel, eulerData, frame,
                                      pinocchio::LOCAL_WORLD_ALIGNED)
              .toVector(),
          kKinematicsTolerance))
          << label;
      ASSERT_TRUE(isCloseRelative(
          pinocchio::getFrameAcceleration(quaternionModel, quaternionData,
                                          frame, pinocchio::LOCAL)
              .toVector(),
          pinocchio::getFrameAcceleration(eulerModel, eulerData, frame,
                                          pinocchio::LOCAL)
              .toVector(),
          kKinematicsTolerance))
          << label;
      ASSERT_TRUE(isCloseRelative(
          pinocchio::getFrameClassicalAcceleration(
              quaternionModel, quaternionData, frame,
              pinocchio::LOCAL_WORLD_ALIGNED)
              .toVector(),
          pinocchio::getFrameClassicalAcceleration(
              eulerModel, eulerData, frame, pinocchio::LOCAL_WORLD_ALIGNED)
              .toVector(),
          kKinematicsTolerance))
          << label;
    }

    // The control: J_v a_e alone, without (dJ_v/dt) v_e.
    const vector_t accelerationWithoutTheRateTerm =
        velocityMaps(eulerMotion).velocityMap * eulerMotion.a;
    pinocchio::forwardKinematics(quaternionModel, controlData, motion.q,
                                 motion.v, accelerationWithoutTheRateTerm);
    if (!isCloseRelative(controlData.a[1].toVector(), eulerData.a[1].toVector(),
                         kKinematicsTolerance))
      ++controlsThatDiffer;
  }
  EXPECT_EQ(controlsThatDiffer, kSamples)
      << "the chart's dT_B/dt term does not show in the root acceleration";
}

TEST_P(RootJointKinematicsEquivalence,
       CenterOfMassItsRatesAndItsJacobianAreTheSame) {
  // The center of mass is a function of the frame placements (C1), its velocity
  // and acceleration of the motion (C2, C3), and its Jacobian maps through J_v
  // (C5).
  const pinocchio::Model& quaternionModel = models_->quaternionModel();
  const pinocchio::Model& eulerModel = models_->eulerModel();
  pinocchio::Data quaternionData(quaternionModel);
  pinocchio::Data eulerData(eulerModel);
  std::mt19937 generator(127);
  for (int sample = 0; sample < kSamples; ++sample) {
    const PinocchioMotion eulerMotion =
        randomEulerMotion(eulerModel, MotionSamplingBounds{}, generator);
    const PinocchioMotion motion = quaternionMotion(eulerMotion);
    pinocchio::centerOfMass(eulerModel, eulerData, eulerMotion.q, eulerMotion.v,
                            eulerMotion.a);
    pinocchio::centerOfMass(quaternionModel, quaternionData, motion.q, motion.v,
                            motion.a);
    ASSERT_TRUE(isCloseRelative(quaternionData.com[0], eulerData.com[0],
                                kKinematicsTolerance))
        << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(quaternionData.vcom[0], eulerData.vcom[0],
                                kKinematicsTolerance))
        << "sample " << sample;
    ASSERT_TRUE(isCloseRelative(quaternionData.acom[0], eulerData.acom[0],
                                kKinematicsTolerance))
        << "sample " << sample;

    const matrix_t eulerJacobian =
        pinocchio::jacobianCenterOfMass(eulerModel, eulerData, eulerMotion.q);
    const matrix_t quaternionJacobian = pinocchio::jacobianCenterOfMass(
        quaternionModel, quaternionData, motion.q);
    ASSERT_TRUE(isCloseRelative(
        quaternionJacobian * velocityMaps(eulerMotion).velocityMap,
        eulerJacobian, kKinematicsTolerance))
        << "sample " << sample;
  }
}

INSTANTIATE_TEST_SUITE_P(EveryShippedRobot,
                         RootJointKinematicsEquivalence,
                         ::testing::ValuesIn(equivalenceRobots().begin(),
                                             equivalenceRobots().end()),
                         equivalenceRobotName);

}  // namespace
}  // namespace ocs2::humanoid
