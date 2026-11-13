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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "legged_centroidal_mpc/CentroidalMpcConfig.h"
#include "legged_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"
#include "legged_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "legged_common_mpc/acom/AngularCenterOfMass.h"
#include "legged_common_mpc/command/BasePlanarMotion.h"
#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/config/ConfigFiles.h"
#include "legged_common_mpc/config/contact_planning/ContactPlanningFromConfig.h"
#include "legged_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "legged_common_mpc/contact_planning/ContactPlan.h"
#include "legged_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "legged_common_mpc/contact_planning/ContactPlanningModelParameters.h"
#include "legged_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "legged_common_mpc/gait/BipedModes.h"
#include "legged_common_mpc/gait/GaitSchedule.h"
#include "legged_common_mpc/orientation/BaseOrientation.h"
#include "legged_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "legged_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "legged_mpc_config/reference_file.nproto.h"
#include "legged_nmpc/legged_common_mpc/test/support/StatusTestMacros.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::legged {

namespace {

constexpr size_t kNumFeet = static_cast<size_t>(kBipedFootCount);

/// The calls a target is built over. The calculator passes the command through
/// unfiltered (it once low-pass filtered it per call), so every call gives the
/// same target; repeating it checks that the target does not depend on the
/// calls before it.
constexpr int kFilterSettlingCalls = 120;

/// Index of the body z angular velocity within the Pinocchio v vector [dp_W,
/// w_B, dq_j] of the quaternion root: for a level base the yaw rate.
constexpr Eigen::Index kBaseYawRateIndex = 5;

}  // namespace

/**
 * Tests the physical momentum of the velocity command mapping used by the DRC
 * Atlas walking command path.
 *
 * The centroidal state carries the normalized momentum h = [p / m, L / m]. A
 * commanded base twist must land in it as the momentum that twist actually
 * produces on the whole body, or the MPC is asked for a different motion than
 * the one the operator commanded. Every expectation below is computed
 * independently from Pinocchio's centroidal dynamics of the same reduced model
 * the MPC uses, never from the code under test. The last test is the
 * calculator's Create() on a reference file that does not load.
 */
class YawCommandDynamicsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const CentroidalRobotFiles files = atlasFiles();
    referenceFile_ = files.referenceFile;
    urdfFile_ = files.urdfFile;
    absl::StatusOr<CentroidalMpcConfig> config = loadConfigOf(files);
    ASSERT_TRUE(config.ok()) << config.status();
    atlas_ = *std::move(config);

    modelSettings_ = std::make_unique<ModelSettings>(checkedValue(
        ModelSettings::Create(atlas_.task, urdfFile_, "testYawCommandDynamics",
                              /*verbose=*/false)));
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(checkedValue(
        loadCustomPinocchioInterface(atlas_.task, urdfFile_, *modelSettings_,
                                     /*scaleTotalMass=*/false)));
    LEGGED_ASSERT_OK_AND_ASSIGN(
        info_,
        centroidalModelInfoOf(atlas_, *pinocchioInterface_, *modelSettings_));
    robotModel_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(
        *modelSettings_, *pinocchioInterface_, info_);

    LEGGED_ASSERT_OK_AND_ASSIGN(initialState_,
                                initialStateOnModel(atlas_.task, *robotModel_));
    mass_ = pinocchio::computeTotalMass(pinocchioInterface_->getModel());

    // Constructed exactly as the MPC node (CentroidalMpcNode) does for the
    // walking command path.
    LEGGED_ASSERT_OK_AND_ASSIGN(
        calculator_, CentroidalMpcTargetTrajectoriesCalculator::Create(
                         referenceFile_, *robotModel_, *pinocchioInterface_,
                         /*mpcHorizon=*/1.0));
  }

  /** The target the calculator emits for `command` = [v_x, v_y, dz, yaw_rate]
   * after kFilterSettlingCalls calls. */
  TargetTrajectories settledTarget(const vector4_t& command,
                                   const vector_t& state) {
    TargetTrajectories target;
    for (int i = 0; i < kFilterSettlingCalls; ++i) {
      target = calculator_->velocityCommandToTargetTrajectories(
          command, /*initTime=*/0.0, state,
          /*measuredMotion=*/BasePlanarMotion{});
    }
    EXPECT_FALSE(target.stateTrajectory.empty());
    return target;
  }

  /** Normalized momentum of that settled target. */
  vector6_t getSettledRefMnormCmomComRtWorldEwrtWorld(const vector4_t& command,
                                                      const vector_t& state) {
    return settledTarget(command, state).stateTrajectory.front().head<6>();
  }

  /** A contact-planning reference manager with the heading model, built the way
   * the interface builds it. */
  std::shared_ptr<ContactPlanningReferenceManager>
  makeHeadingReferenceManager() {
    absl::StatusOr<ContactPlanningConfig> loaded =
        contactPlanningConfigFromOptionalFile(
            atlas_.contactPlanning.has_value() ? &*atlas_.contactPlanning
                                               : nullptr,
            ContactPlanningValidation::kDeferUntilModelParametersApplied);
    EXPECT_TRUE(loaded.ok()) << loaded.status();
    if (!loaded.ok()) return nullptr;
    ContactPlanningConfig config = *loaded;
    config.setHeadingModel(true);
    ContactPlanningGroundParameters ground;
    ground.frictionCoefficient = 0.5;
    ground.torsionalFrictionCoefficient = 0.05;
    PinocchioInterface pinocchioForDerivation(*pinocchioInterface_);
    deriveContactPlanningModelParameters(
        pinocchioForDerivation, *robotModel_, initialState_,
        modelSettings_->contactParentJointNames, modelSettings_->numContacts(),
        ground)
        .applyTo(config);
    std::unique_ptr<SwingTrajectoryPlanner> swingPlanner(
        new SwingTrajectoryPlanner(
            checkedValue(swingTrajectorySettingsFromConfig(
                atlas_.task.swing_trajectory_config)),
            kNumFeet));
    absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>>
        referenceManager = ContactPlanningReferenceManager::Create(
            checkedValue(GaitSchedule::Create(atlas_.reference, *modelSettings_,
                                              /*verbose=*/false)),
            std::move(swingPlanner), *pinocchioInterface_, *robotModel_,
            config);
    EXPECT_TRUE(referenceManager.ok()) << referenceManager.status();
    if (!referenceManager.ok()) return nullptr;
    absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
        AngularCenterOfMass::Create(modelSettings_->robotName,
                                    modelSettings_->mpcModelJointNames);
    EXPECT_TRUE(acom.ok()) << acom.status();
    if (acom.ok()) {
      (*referenceManager)->setAngularCenterOfMass(*std::move(acom));
    }
    return *referenceManager;
  }

  /** Whole-body (locked) inertia about the CoM in the world frame, from
   * Pinocchio's composite rigid body algorithm. */
  matrix3_t computeInertiaComEwrtWorld(const vector_t& state) {
    PinocchioInterface pinocchio(*pinocchioInterface_);
    const vector_t q = robotModel_->getGeneralizedCoordinates(state);
    pinocchio::ccrba(pinocchio.getModel(), pinocchio.getData(), q,
                     vector_t::Zero(pinocchio.getModel().nv));
    return pinocchio.getData().Ig.inertia().matrix();
  }

  /** Whole-body inertia about the vertical through the CoM. */
  scalar_t computeInertiaComEwrtWorldZz(const vector_t& state) {
    return computeInertiaComEwrtWorld(state)(2, 2);
  }

  /** Position of the CoM relative to the base origin, in the world frame. */
  vector3_t getPosComRtBaseEwrtWorld(const vector_t& state) {
    PinocchioInterface pinocchio(*pinocchioInterface_);
    const vector_t q = robotModel_->getGeneralizedCoordinates(state);
    pinocchio::centerOfMass(pinocchio.getModel(), pinocchio.getData(), q,
                            /*computeSubtreeComs=*/false);
    return vector3_t(pinocchio.getData().com[0] - q.head<3>());
  }

  /** The base rows [linvelBaseRtWorldEwrtWorld; angvelBaseRtWorldEwrtBase] of
   * Pinocchio's v that produce the normalized momentum `h` with the joints at
   * rest. */
  vector6_t baseGeneralizedVelocityOf(const vector6_t& h,
                                      const vector_t& state) {
    PinocchioInterface pinocchio(*pinocchioInterface_);
    const vector_t q = robotModel_->getGeneralizedCoordinates(state);
    updateCentroidalDynamics(pinocchio, info_, q);
    const Eigen::Matrix<scalar_t, 6, 6> Ab =
        getCentroidalMomentumMatrix(pinocchio).leftCols<6>();
    return computeFloatingBaseCentroidalMomentumMatrixInverse(Ab) * (mass_ * h);
  }

  std::string referenceFile_, urdfFile_;
  // The typed DRC Atlas files.
  CentroidalMpcConfig atlas_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  CentroidalModelInfo info_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> robotModel_;
  std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator> calculator_;
  vector_t initialState_;
  scalar_t mass_ = 0.0;
};

/**
 * A commanded yaw rate must appear in the target as the angular momentum that
 * rate produces on the whole body, L_z / m = I_zz * yaw_rate / m. The mapping
 * used to be yaw_rate / m, which dropped the inertia altogether.
 */
TEST_F(YawCommandDynamicsTest, YawRateCommandMapsToWholeBodyAngularMomentum) {
  const scalar_t cmdRateYawBaseRtWorld = 0.4;
  const vector6_t h = getSettledRefMnormCmomComRtWorldEwrtWorld(
      vector4_t(0.0, 0.0, 0.0, cmdRateYawBaseRtWorld), initialState_);

  const matrix3_t inertiaComEwrtWorld =
      computeInertiaComEwrtWorld(initialState_);
  ASSERT_GT(inertiaComEwrtWorld(2, 2), 1.0)
      << "Atlas has several kg m^2 of yaw inertia; a tiny "
         "value means the model or the state is wrong.";
  ASSERT_GT(mass_, 100.0);

  // L = I_G * (0, 0, yaw_rate): the yaw column of the locked inertia, products
  // of inertia included.
  const vector3_t expected =
      inertiaComEwrtWorld.col(2) * cmdRateYawBaseRtWorld / mass_;
  EXPECT_NEAR((h.tail<3>() - expected).norm(), 0.0, 1.0e-9);
  EXPECT_NEAR(h(5), inertiaComEwrtWorld(2, 2) * cmdRateYawBaseRtWorld / mass_,
              1.0e-9);
  // The old mapping is a factor of I_zz away from the right one; make sure the
  // test would have caught it.
  EXPECT_GT(std::abs(h(5) - cmdRateYawBaseRtWorld / mass_), 1.0e-3);
  // A pure turn commands no linear momentum.
  EXPECT_NEAR(h.head<3>().norm(), 0.0, 1.0e-9);
}

/**
 * Closing the loop through the dynamics, exactly. The momentum target must
 * describe the whole body spinning about the vertical through its CoM at the
 * commanded rate while the CoM moves at the commanded velocity:
 *   - inverting the locked inertia on the angular target must give back the
 * angular velocity (0, 0, yaw_rate), and
 *   - the pelvis twist that Pinocchio's centroidal momentum matrix associates
 * with the target (joints at rest) must be that of a point of the same rigid
 * body: v_base = v_com + omega x (p_base - p_com), yaw rate = yaw_rate. Both
 * are computed from Pinocchio, not from the code under test.
 */
TEST_F(YawCommandDynamicsTest, MomentumTargetIsARigidSpinAboutTheCenterOfMass) {
  const scalar_t cmdRateYawBaseRtWorld = 0.3;
  const scalar_t cmdLinvelBaseRtWorldEwrtHeadingX = 0.5;
  const vector6_t h = getSettledRefMnormCmomComRtWorldEwrtWorld(
      vector4_t(cmdLinvelBaseRtWorldEwrtHeadingX, 0.0, 0.0,
                cmdRateYawBaseRtWorld),
      initialState_);

  // Angular velocity from the angular momentum.
  const matrix3_t inertiaComEwrtWorld =
      computeInertiaComEwrtWorld(initialState_);
  const vector3_t angvelCrbRtWorldEwrtWorld =
      inertiaComEwrtWorld.ldlt().solve(vector3_t(mass_ * h.tail<3>()));
  EXPECT_NEAR(angvelCrbRtWorldEwrtWorld(0), 0.0, 1.0e-9) << "no roll rate";
  EXPECT_NEAR(angvelCrbRtWorldEwrtWorld(1), 0.0, 1.0e-9) << "no pitch rate";
  EXPECT_NEAR(angvelCrbRtWorldEwrtWorld(2), cmdRateYawBaseRtWorld, 1.0e-9)
      << "the commanded yaw rate";

  // CoM velocity from the linear momentum.
  EXPECT_NEAR(h(0), cmdLinvelBaseRtWorldEwrtHeadingX, 1.0e-9);
  EXPECT_NEAR(h(1), 0.0, 1.0e-9);
  EXPECT_NEAR(h(2), 0.0, 1.0e-9);

  // Pelvis twist [dp_W, w_B] through the centroidal momentum matrix. The pelvis
  // is not at the CoM, so a spin about the CoM moves it sideways by omega x r;
  // with the base level the body z rate is the angular velocity about the
  // vertical.
  const vector6_t baseGeneralizedVelocity =
      baseGeneralizedVelocityOf(h, initialState_);
  const vector3_t posBaseRtComEwrtWorld =
      -getPosComRtBaseEwrtWorld(initialState_);  // p_base - p_com
  const vector3_t expectedLinvelBaseRtWorldEwrtWorld =
      vector3_t(cmdLinvelBaseRtWorldEwrtHeadingX, 0.0, 0.0) +
      vector3_t(0.0, 0.0, cmdRateYawBaseRtWorld).cross(posBaseRtComEwrtWorld);
  EXPECT_NEAR(
      (baseGeneralizedVelocity.head<3>() - expectedLinvelBaseRtWorldEwrtWorld)
          .norm(),
      0.0, 1.0e-6)
      << "recovered " << baseGeneralizedVelocity.head<3>().transpose()
      << " expected " << expectedLinvelBaseRtWorldEwrtWorld.transpose();
  EXPECT_NEAR(baseGeneralizedVelocity(kBaseYawRateIndex), cmdRateYawBaseRtWorld,
              1.0e-6)
      << "recovered yaw rate";
  EXPECT_NEAR(baseGeneralizedVelocity(3), 0.0, 1.0e-6) << "roll rate, body x";
  EXPECT_NEAR(baseGeneralizedVelocity(4), 0.0, 1.0e-6) << "pitch rate, body y";
  // And the sideways component is not a rounding artifact: the pelvis really is
  // offset from the CoM.
  EXPECT_GT(posBaseRtComEwrtWorld.norm(), 1.0e-3);
}

/**
 * The linear entries are the CoM velocity, so a velocity command must land in
 * them unscaled. Guards against the inertia fix being applied to the wrong
 * entries.
 */
TEST_F(YawCommandDynamicsTest, LinearVelocityCommandIsCenterOfMassVelocity) {
  const vector6_t h = getSettledRefMnormCmomComRtWorldEwrtWorld(
      vector4_t(0.7, -0.2, 0.0, 0.0), initialState_);
  EXPECT_NEAR(h(0), 0.7, 1.0e-6);
  EXPECT_NEAR(h(1), -0.2, 1.0e-6);
  EXPECT_NEAR(h.tail<4>().norm(), 0.0, 1.0e-9);
}

/**
 * The yaw rate command is given in the pelvis frame and integrated in the world
 * frame. With the base turned, the angular momentum about the world vertical is
 * unchanged by that rotation, so the emitted momentum entry must not depend on
 * the base yaw.
 */
TEST_F(YawCommandDynamicsTest, YawMomentumTargetIsInvariantToBaseYaw) {
  const scalar_t cmdRateYawBaseRtWorld = 0.25;
  const vector6_t reference = getSettledRefMnormCmomComRtWorldEwrtWorld(
      vector4_t(0.0, 0.0, 0.0, cmdRateYawBaseRtWorld), initialState_);

  for (const scalar_t eulerzyxBaseRtWorldYaw : {0.7, -1.9, 2.8}) {
    vector_t turned = initialState_;
    robotModel_->setEulerzyxBaseRtWorld(
        turned, vector3_t(eulerzyxBaseRtWorldYaw, 0.0, 0.0));
    const vector6_t h = getSettledRefMnormCmomComRtWorldEwrtWorld(
        vector4_t(0.0, 0.0, 0.0, cmdRateYawBaseRtWorld), turned);
    EXPECT_NEAR(h(5), reference(5), 1.0e-9)
        << "base yaw " << eulerzyxBaseRtWorldYaw;
    // The products of inertia I_xz, I_yz rotate with the body; only their
    // magnitude is invariant.
    EXPECT_NEAR(h.segment<2>(3).norm(), reference.segment<2>(3).norm(), 1.0e-9)
        << "base yaw " << eulerzyxBaseRtWorldYaw;
    EXPECT_NEAR(computeInertiaComEwrtWorldZz(turned),
                computeInertiaComEwrtWorldZz(initialState_), 1.0e-9)
        << "I_zz must not change under a rotation about z";
  }
}

/**
 * The inertia the mapping relies on is a property of the multibody model, not
 * of the base placement: rotating the base about the vertical leaves it
 * unchanged, and it agrees with the yaw column of the centroidal momentum
 * matrix, which is the momentum a unit yaw rate produces with the joints
 * locked.
 */
TEST_F(YawCommandDynamicsTest, YawInertiaMatchesTheCentroidalMomentumMatrix) {
  PinocchioInterface pinocchio(*pinocchioInterface_);
  const vector_t q = robotModel_->getGeneralizedCoordinates(initialState_);
  updateCentroidalDynamics(pinocchio, info_, q);
  // Column 5 of the CMM is the momentum of a unit body z angular velocity; with
  // the base level that is a unit angular velocity about the vertical, so its z
  // component is I_zz.
  const scalar_t fromCmm =
      getCentroidalMomentumMatrix(pinocchio)(5, kBaseYawRateIndex);
  EXPECT_NEAR(fromCmm, computeInertiaComEwrtWorldZz(initialState_), 1.0e-9);
  EXPECT_GT(fromCmm, 1.0);
}

/**
 * The ACoM-LIP heading model is fed from the whole-body state by the reference
 * manager: the heading is the twist heading of the angular center of mass's
 * attitude, its rate is the normalized angular momentum about the vertical
 * scaled by mass over the locked yaw inertia, and the foot yaws are the contact
 * frames' yaws unwrapped near the heading. Each is checked against an
 * independent evaluation of the same quantity from the robot model.
 */
TEST_F(YawCommandDynamicsTest,
       PlannerInputCarriesTheAcomHeadingAndItsRateFromAngularMomentum) {
  const std::shared_ptr<ContactPlanningReferenceManager> referenceManager =
      makeHeadingReferenceManager();
  ASSERT_NE(referenceManager, nullptr);
  absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> created =
      AngularCenterOfMass::Create(modelSettings_->robotName,
                                  modelSettings_->mpcModelJointNames);
  ASSERT_TRUE(created.ok()) << created.status();
  const std::shared_ptr<AngularCenterOfMass> acom = *std::move(created);

  // A turned base spinning about the vertical at a known rate, expressed as the
  // normalized angular momentum the state carries.
  const scalar_t cmdRateYawBaseRtWorld = 0.3;
  vector_t state = initialState_;
  robotModel_->setEulerzyxBaseRtWorld(state, vector3_t(0.7, 0.0, 0.0));
  const matrix3_t inertiaComEwrtWorld = computeInertiaComEwrtWorld(state);
  state.segment<3>(3) = inertiaComEwrtWorld.col(2) *
                        (cmdRateYawBaseRtWorld /
                         mass_);  // L / m for a rigid spin about the vertical

  const ContactPlannerInput input = referenceManager->makePlannerInput(
      /*initTime=*/0.0, state, vector2_t::Zero());

  // Heading: the ACoM heading of the state - the twist heading of the base
  // orientation composed with the learned joint rotation - and the planning
  // frame is the heading.
  const vector_t q = robotModel_->getGeneralizedCoordinates(state);
  const scalar_t yawAcomRtWorld = yawOf(acom->computeQuatAcomRtWorldXyzw(
      robotModel_->getQuatBaseRtWorldXyzw(state),
      robotModel_->getJointAngles(state)));
  EXPECT_NEAR(input.initYawHeadingRtWorld, yawAcomRtWorld, 1.0e-12);
  EXPECT_NEAR(input.yawHeadingRtWorld, input.initYawHeadingRtWorld, 1.0e-12);
  EXPECT_GT(std::abs(yawAcomRtWorld - 0.7), 1.0e-4)
      << "the joint offset must actually contribute, or this test would not "
         "tell the ACoM yaw from the base yaw";

  // Heading rate: L_z / I_zz recovers the spin the momentum was built from,
  // with the inertia the manager derives itself.
  EXPECT_NEAR(input.inertiaComEwrtWorldZz, inertiaComEwrtWorld(2, 2), 1.0e-9);
  EXPECT_NEAR(input.initRateYawHeadingRtWorld, cmdRateYawBaseRtWorld, 1.0e-9);

  // Foot yaws: the contact frames' yaws from forward kinematics, unwrapped to
  // within pi of the heading.
  PinocchioInterface pinocchio(*pinocchioInterface_);
  pinocchio::forwardKinematics(pinocchio.getModel(), pinocchio.getData(), q);
  pinocchio::updateFramePlacements(pinocchio.getModel(), pinocchio.getData());
  for (size_t foot = 0; foot < kNumFeet; ++foot) {
    const pinocchio::FrameIndex frameId =
        pinocchio.getModel().getFrameId(modelSettings_->contactNames6DoF[foot]);
    const matrix3_t rotmatFootRtWorld =
        pinocchio.getData().oMf[frameId].rotation();
    const scalar_t eulerzyxFootRtWorldYaw =
        std::atan2(rotmatFootRtWorld(1, 0), rotmatFootRtWorld(0, 0));
    EXPECT_NEAR(std::remainder(
                    input.eulerzyxFeetRtWorldYaw[foot] - eulerzyxFootRtWorldYaw,
                    2.0 * std::numbers::pi),
                0.0, 1.0e-9)
        << "foot " << foot;
    EXPECT_LE(std::abs(input.eulerzyxFeetRtWorldYaw[foot] -
                       input.initYawHeadingRtWorld),
              std::numbers::pi + 1.0e-9)
        << "unwrapped near the heading, foot " << foot;
  }
}

/**
 * The yaw rate command handed to the planner is the operator's, read off the
 * momentum channel of the calculator's target where the command is carried as
 * h_z = I_zz omega / m. The target's base yaw cannot serve as the source: its
 * first stretch integrates the average of the measured and the commanded yaw
 * rate, so a robot at rest commanded 0.5 rad/s was planned at 0.25 rad/s, and a
 * robot that under-tracked its turn was asked for less and less.
 */
TEST_F(YawCommandDynamicsTest,
       PlannerYawRateCommandIsTheOperatorsNotTheBlendedTargetYaw) {
  const std::shared_ptr<ContactPlanningReferenceManager> referenceManager =
      makeHeadingReferenceManager();
  ASSERT_NE(referenceManager, nullptr);
  const scalar_t cmdRateYawBaseRtWorld = 0.5;
  const scalar_t horizon = 1.0;
  const TargetTrajectories target = settledTarget(
      vector4_t(0.0, 0.0, 0.0, cmdRateYawBaseRtWorld), initialState_);

  // What differentiating the target's base yaw over its first stretch gives for
  // a robot at rest: half the command.
  const scalar_t yaw0 = robotModel_->getEulerzyxBaseRtWorld(
      robotModel_->getDesiredState(target, /*time=*/0.0))(0);
  const scalar_t yaw1 = robotModel_->getEulerzyxBaseRtWorld(
      robotModel_->getDesiredState(target, /*time=*/0.2))(0);
  EXPECT_NEAR((yaw1 - yaw0) / 0.2, 0.5 * cmdRateYawBaseRtWorld, 1.0e-6)
      << "the blended ramp is why the base yaw is not the command source";

  referenceManager->setTargetTrajectories(target);
  referenceManager->preSolverRun(/*initTime=*/0.0, horizon, initialState_,
                                 kBipedModeStance);
  EXPECT_NEAR(referenceManager->cmdRateYawBaseRtWorld(), cmdRateYawBaseRtWorld,
              1.0e-6);
  const ContactPlannerInput input = referenceManager->makePlannerInput(
      /*initTime=*/0.0, initialState_, vector2_t::Zero());
  EXPECT_NEAR(input.cmdRateYawBaseRtWorld, cmdRateYawBaseRtWorld, 1.0e-6);

  // The same command read with the robot already turning at the commanded rate,
  // and with a turned base: the momentum channel is a base-frame-invariant
  // quantity and the inertia the manager derives matches the calculator's.
  vector_t turning = initialState_;
  robotModel_->setEulerzyxBaseRtWorld(turning, vector3_t(1.1, 0.0, 0.0));
  turning.segment<3>(3) = computeInertiaComEwrtWorld(turning).col(2) *
                          (cmdRateYawBaseRtWorld / mass_);
  referenceManager->setTargetTrajectories(
      settledTarget(vector4_t(0.0, 0.0, 0.0, cmdRateYawBaseRtWorld), turning));
  referenceManager->preSolverRun(/*initTime=*/0.02, 0.02 + horizon, turning,
                                 kBipedModeStance);
  EXPECT_NEAR(referenceManager->cmdRateYawBaseRtWorld(), cmdRateYawBaseRtWorld,
              1.0e-6);

  // No yaw command, or a linear command only: no yaw rate is asked of the
  // planner.
  referenceManager->setTargetTrajectories(
      settledTarget(vector4_t(0.5, 0.1, 0.0, 0.0), initialState_));
  referenceManager->preSolverRun(/*initTime=*/0.04, 0.04 + horizon,
                                 initialState_, kBipedModeStance);
  EXPECT_NEAR(referenceManager->cmdRateYawBaseRtWorld(), 0.0, 1.0e-9);
  // An empty target leaves nothing to command.
  referenceManager->setTargetTrajectories(TargetTrajectories());
  referenceManager->preSolverRun(/*initTime=*/0.06, 0.06 + horizon,
                                 initialState_, kBipedModeStance);
  EXPECT_NEAR(referenceManager->cmdRateYawBaseRtWorld(), 0.0, 1.0e-9);
}

namespace {

/**
 * The shipped reference file with its
 * target_joint_state_interpolation_time_constant made unparsable, in the test's
 * temporary directory; empty (and a test failure) when the shipped file does
 * not give one.
 */
std::string writeReferenceWithAnUnparsableTimeConstant(
    absl::string_view referenceFile) {
  const absl::StatusOr<mpc_config::ReferenceFile> shipped =
      loadReferenceFile(referenceFile);
  if (!shipped.ok()) {
    ADD_FAILURE() << shipped.status();
    return std::string();
  }
  std::string reference = referenceFileText(*shipped);
  const std::string key = "target_joint_state_interpolation_time_constant: ";
  const size_t value = reference.find(key);
  if (value == std::string::npos) {
    ADD_FAILURE() << "the shipped reference file no longer gives "
                     "target_joint_state_interpolation_time_constant";
    return std::string();
  }
  const size_t valueStart = value + key.size();
  reference.replace(valueStart,
                    reference.find_first_of(" \n", valueStart) - valueStart,
                    "slowly");
  const std::string brokenReference =
      (std::filesystem::path(testing::TempDir()) /
       "testYawCommandDynamics_reference.textproto")
          .string();
  EXPECT_TRUE(writeTextFile(brokenReference, reference).ok());
  return brokenReference;
}

}  // namespace

TEST_F(YawCommandDynamicsTest,
       CreateRefusesAJointStateTimeConstantThatDoesNotParse) {
  // A value the strict parser cannot read is refused with the file, the line
  // and the column.
  const std::string brokenReference =
      writeReferenceWithAnUnparsableTimeConstant(referenceFile_);
  ASSERT_FALSE(brokenReference.empty());

  const absl::StatusOr<
      std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator>>
      created = CentroidalMpcTargetTrajectoriesCalculator::Create(
          brokenReference, *robotModel_, *pinocchioInterface_,
          /*mpcHorizon=*/1.0);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument)
      << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(),
                                absl::StrCat(brokenReference, ":")))
      << created.status();
}

TEST_F(YawCommandDynamicsTest, CreateRequiresAPositiveJointStateTimeConstant) {
  // The centroidal calculator's own field, which the whole-body MPC does not
  // read: the conversion leaves it absent when the file has none, and the
  // centroidal calculator refuses that, and a time constant that would make the
  // joint-state filter grow rather than decay.
  mpc_config::ReferenceFile reference = atlas_.reference;
  for (const std::optional<double> timeConstant :
       {std::optional<double>(), std::optional<double>(0.0),
        std::optional<double>(-0.5)}) {
    reference.target_joint_state_interpolation_time_constant = timeConstant;
    const absl::StatusOr<
        std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator>>
        created = CentroidalMpcTargetTrajectoriesCalculator::Create(
            reference, *robotModel_, *pinocchioInterface_, /*mpcHorizon=*/1.0);
    EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument)
        << created.status();
    EXPECT_TRUE(
        absl::StrContains(created.status().message(),
                          "target_joint_state_interpolation_time_constant"))
        << created.status();
  }
}

namespace {

/** The centroidal robot model of the Atlas, counting its live instances, clones
 * included. */
class CountingRobotModel final : public CentroidalMpcRobotModel<scalar_t> {
 public:
  CountingRobotModel(const ModelSettings& modelSettings,
                     const PinocchioInterface& pinocchioInterface,
                     const CentroidalModelInfo& info)
      : CentroidalMpcRobotModel<scalar_t>(
            modelSettings, pinocchioInterface, info),
        modelSettings_(modelSettings),
        pinocchioInterface_(pinocchioInterface),
        info_(info) {
    ++liveInstances;
  }
  ~CountingRobotModel() override { --liveInstances; }
  CountingRobotModel(const CountingRobotModel&) = delete;
  CountingRobotModel& operator=(const CountingRobotModel&) = delete;
  CountingRobotModel* absl_nonnull clone() const override {
    return new CountingRobotModel(modelSettings_, pinocchioInterface_, info_);
  }

  /** The instances alive in the test program. */
  inline static int liveInstances = 0;

 private:
  // The arguments of the model, which outlive it, for clone(): the base's copy
  // constructor is private.
  const ModelSettings& modelSettings_;
  const PinocchioInterface& pinocchioInterface_;
  const CentroidalModelInfo& info_;
};

}  // namespace

TEST_F(YawCommandDynamicsTest,
       TheCalculatorReleasesTheRobotModelItClonesAlsoWhenCreateFails) {
  // TargetTrajectoriesCalculatorBase clones the model it is given; it used to
  // keep the clone in a raw pointer that nothing deleted, so every calculator,
  // and every Create() refused by a loader, leaked a model.
  {
    const CountingRobotModel model(*modelSettings_, *pinocchioInterface_,
                                   info_);
    ASSERT_EQ(CountingRobotModel::liveInstances, 1);
    {
      absl::StatusOr<std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator>>
          created = CentroidalMpcTargetTrajectoriesCalculator::Create(
              referenceFile_, model, *pinocchioInterface_, /*mpcHorizon=*/1.0);
      ASSERT_TRUE(created.ok()) << created.status();
      EXPECT_EQ(CountingRobotModel::liveInstances, 2)
          << "the calculator holds a copy of the model of its own";
    }
    EXPECT_EQ(CountingRobotModel::liveInstances, 1)
        << "the calculator released its model";

    const std::string brokenReference =
        writeReferenceWithAnUnparsableTimeConstant(referenceFile_);
    ASSERT_FALSE(brokenReference.empty());
    const absl::StatusOr<
        std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator>>
        refused = CentroidalMpcTargetTrajectoriesCalculator::Create(
            brokenReference, model, *pinocchioInterface_, /*mpcHorizon=*/1.0);
    ASSERT_FALSE(refused.ok());
    EXPECT_EQ(CountingRobotModel::liveInstances, 1)
        << "a Create() its loader refused released the model it had cloned";
  }
  EXPECT_EQ(CountingRobotModel::liveInstances, 0);
}

}  // namespace ocs2::legged
