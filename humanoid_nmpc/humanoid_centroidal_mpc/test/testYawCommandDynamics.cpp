/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>

#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_core/misc/LoadData.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

namespace ocs2::humanoid {

namespace {

/// The command filter is first order with coefficient 0.8 per call, so its state settles on the command as 0.8^N. After
/// this many calls the residual is 0.8^120, about 2e-12 of the command, well below the 1e-9 the exact identities below
/// are held to.
constexpr int kFilterSettlingCalls = 120;

/// Index of the first base orientation coordinate (ZYX Euler, yaw first) within the Pinocchio q vector.
constexpr Eigen::Index kBaseYawIndex = 3;

}  // namespace

/**
 * Tests the physics of the velocity command mapping used by the DRC Atlas walking command path.
 *
 * The centroidal state carries the normalized momentum h = [p / m, L / m]. A commanded base twist must land in it as
 * the momentum that twist actually produces on the whole body, or the MPC is asked for a different motion than the one
 * the operator commanded. Every expectation below is computed independently from Pinocchio's centroidal dynamics of the
 * same reduced model the MPC uses, never from the code under test.
 */
class YawCommandDynamicsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string configDir = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
    const std::string descriptionDir = ament_index_cpp::get_package_share_directory("drc_atlas_description");
    taskFile_ = configDir + "/config/mpc/task.yaml";
    referenceFile_ = configDir + "/config/command/reference.yaml";
    urdfFile_ = descriptionDir + "/urdf/atlas.urdf";

    modelSettings_ = std::make_unique<ModelSettings>(taskFile_, urdfFile_, "testYawCommandDynamics", false);
    pinocchioInterface_ =
        std::make_unique<PinocchioInterface>(createCustomPinocchioInterface(taskFile_, urdfFile_, *modelSettings_, false));
    info_ = centroidal_model::createCentroidalModelInfo(
        *pinocchioInterface_, centroidal_model::loadCentroidalType(taskFile_),
        centroidal_model::loadDefaultJointState(pinocchioInterface_->getModel().nq - 6, referenceFile_), modelSettings_->contactNames3DoF,
        modelSettings_->contactNames6DoF);
    robotModel_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, info_);

    initialState_.setZero(info_.stateDim);
    loadData::loadEigenMatrix(taskFile_, "initialState", initialState_);
    mass_ = pinocchio::computeTotalMass(pinocchioInterface_->getModel());

    // Constructed exactly as CentroidalMpcRobotSim does for the walking command path.
    calculator_ = std::make_unique<CentroidalMpcTargetTrajectoriesCalculator>(referenceFile_, *robotModel_, *pinocchioInterface_, info_,
                                                                              /*mpcHorizon=*/1.0);
  }

  /** The target the calculator emits once its command filter has settled on `command` = [v_x, v_y, dz, yaw_rate]. */
  TargetTrajectories settledTarget(const vector4_t& command, const vector_t& state) {
    TargetTrajectories target;
    for (int i = 0; i < kFilterSettlingCalls; ++i) {
      target = calculator_->commandedVelocityToTargetTrajectories(command, /*initTime=*/0.0, state);
    }
    EXPECT_FALSE(target.stateTrajectory.empty());
    return target;
  }

  /** Normalized momentum of that settled target. */
  vector6_t settledMomentumTarget(const vector4_t& command, const vector_t& state) {
    return settledTarget(command, state).stateTrajectory.front().head<6>();
  }

  /** A contact-planning reference manager with the heading model, built the way the interface builds it. */
  std::unique_ptr<ContactPlanningReferenceManager> makeHeadingReferenceManager() {
    ContactPlanningConfig config = loadContactPlanningConfig(resolveContactPlanningConfigFile(taskFile_), "contact_planning.", false);
    config.setHeadingModel(true);
    ContactPlanningGroundParameters ground;
    ground.frictionCoefficient = 0.5;
    ground.torsionalFrictionCoefficient = 0.05;
    PinocchioInterface pinocchioForDerivation(*pinocchioInterface_);
    deriveContactPlanningModelParameters(pinocchioForDerivation, *robotModel_, initialState_, modelSettings_->contactParentJointNames,
                                         ground, config.shared.gravity, config.stepWidth.nominalStepWidth)
        .applyTo(config);
    config.validate();
    std::unique_ptr<SwingTrajectoryPlanner> swingPlanner(
        new SwingTrajectoryPlanner(loadSwingTrajectorySettings(taskFile_, "swing_trajectory_config", false), N_CONTACTS));
    auto gaitSchedule = GaitSchedule::loadGaitSchedule(referenceFile_, *modelSettings_, false);
    auto referenceManager = std::make_unique<ContactPlanningReferenceManager>(std::move(gaitSchedule), std::move(swingPlanner),
                                                                              *pinocchioInterface_, *robotModel_, config);
    referenceManager->setAngularCenterOfMass(AngularCenterOfMass::createForRobot(modelSettings_->robotName));
    return referenceManager;
  }

  /** Whole-body (locked) inertia about the CoM in the world frame, from Pinocchio's composite rigid body algorithm. */
  matrix3_t lockedInertia(const vector_t& state) {
    PinocchioInterface pinocchio(*pinocchioInterface_);
    const vector_t q = robotModel_->getGeneralizedCoordinates(state);
    pinocchio::ccrba(pinocchio.getModel(), pinocchio.getData(), q, vector_t::Zero(pinocchio.getModel().nv));
    return pinocchio.getData().Ig.inertia().matrix();
  }

  /** Whole-body inertia about the vertical through the CoM. */
  scalar_t yawInertia(const vector_t& state) { return lockedInertia(state)(2, 2); }

  /** Position of the CoM relative to the base origin, in the world frame. */
  vector3_t comFromBase(const vector_t& state) {
    PinocchioInterface pinocchio(*pinocchioInterface_);
    const vector_t q = robotModel_->getGeneralizedCoordinates(state);
    pinocchio::centerOfMass(pinocchio.getModel(), pinocchio.getData(), q, false);
    return vector3_t(pinocchio.getData().com[0] - q.head<3>());
  }

  /** Base twist [v_base, euler_zyx_rate] that produces the normalized momentum `h` with the joints at rest. */
  vector6_t baseTwistFromMomentum(const vector6_t& h, const vector_t& state) {
    PinocchioInterface pinocchio(*pinocchioInterface_);
    const vector_t q = robotModel_->getGeneralizedCoordinates(state);
    updateCentroidalDynamics(pinocchio, info_, q);
    const Eigen::Matrix<scalar_t, 6, 6> Ab = getCentroidalMomentumMatrix(pinocchio).leftCols<6>();
    return computeFloatingBaseCentroidalMomentumMatrixInverse(Ab) * (mass_ * h);
  }

  std::string taskFile_, referenceFile_, urdfFile_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  CentroidalModelInfo info_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> robotModel_;
  std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator> calculator_;
  vector_t initialState_;
  scalar_t mass_ = 0.0;
};

/**
 * A commanded yaw rate must appear in the target as the angular momentum that rate produces on the whole body,
 * L_z / m = I_zz * yaw_rate / m. The mapping used to be yaw_rate / m, which dropped the inertia altogether.
 */
TEST_F(YawCommandDynamicsTest, YawRateCommandMapsToWholeBodyAngularMomentum) {
  const scalar_t yawRate = 0.4;
  const vector6_t h = settledMomentumTarget(vector4_t(0.0, 0.0, 0.0, yawRate), initialState_);

  const matrix3_t Ig = lockedInertia(initialState_);
  ASSERT_GT(Ig(2, 2), 1.0) << "Atlas has several kg m^2 of yaw inertia; a tiny value means the model or the state is wrong.";
  ASSERT_GT(mass_, 100.0);

  // L = I_G * (0, 0, yaw_rate): the yaw column of the locked inertia, products of inertia included.
  const vector3_t expected = Ig.col(2) * yawRate / mass_;
  EXPECT_NEAR((h.tail<3>() - expected).norm(), 0.0, 1e-9);
  EXPECT_NEAR(h(5), Ig(2, 2) * yawRate / mass_, 1e-9);
  // The old mapping is a factor of I_zz away from the right one; make sure the test would have caught it.
  EXPECT_GT(std::abs(h(5) - yawRate / mass_), 1e-3);
  // A pure turn commands no linear momentum.
  EXPECT_NEAR(h.head<3>().norm(), 0.0, 1e-9);
}

/**
 * Closing the loop through the dynamics, exactly. The momentum target must describe the whole body spinning about the
 * vertical through its CoM at the commanded rate while the CoM moves at the commanded velocity:
 *   - inverting the locked inertia on the angular target must give back the angular velocity (0, 0, yaw_rate), and
 *   - the pelvis twist that Pinocchio's centroidal momentum matrix associates with the target (joints at rest) must be
 *     that of a point of the same rigid body: v_base = v_com + omega x (p_base - p_com), yaw rate = yaw_rate.
 * Both are computed from Pinocchio, not from the code under test.
 */
TEST_F(YawCommandDynamicsTest, MomentumTargetIsARigidSpinAboutTheCentreOfMass) {
  const scalar_t yawRate = 0.3;
  const scalar_t forwardVelocity = 0.5;
  const vector6_t h = settledMomentumTarget(vector4_t(forwardVelocity, 0.0, 0.0, yawRate), initialState_);

  // Angular velocity from the angular momentum.
  const matrix3_t Ig = lockedInertia(initialState_);
  const vector3_t omega = Ig.ldlt().solve(vector3_t(mass_ * h.tail<3>()));
  EXPECT_NEAR(omega(0), 0.0, 1e-9) << "no roll rate";
  EXPECT_NEAR(omega(1), 0.0, 1e-9) << "no pitch rate";
  EXPECT_NEAR(omega(2), yawRate, 1e-9) << "the commanded yaw rate";

  // CoM velocity from the linear momentum.
  EXPECT_NEAR(h(0), forwardVelocity, 1e-9);
  EXPECT_NEAR(h(1), 0.0, 1e-9);
  EXPECT_NEAR(h(2), 0.0, 1e-9);

  // Pelvis twist through the centroidal momentum matrix. The pelvis is not at the CoM, so a spin about the CoM moves it
  // sideways by omega x r; with the base level the ZYX yaw rate is the angular velocity about the vertical.
  const vector6_t twist = baseTwistFromMomentum(h, initialState_);
  const vector3_t r = -comFromBase(initialState_);  // p_base - p_com
  const vector3_t expectedBaseVelocity = vector3_t(forwardVelocity, 0.0, 0.0) + vector3_t(0.0, 0.0, yawRate).cross(r);
  EXPECT_NEAR((twist.head<3>() - expectedBaseVelocity).norm(), 0.0, 1e-6)
      << "recovered " << twist.head<3>().transpose() << " expected " << expectedBaseVelocity.transpose();
  EXPECT_NEAR(twist(3), yawRate, 1e-6) << "recovered yaw rate";
  EXPECT_NEAR(twist(4), 0.0, 1e-6) << "pitch rate";
  EXPECT_NEAR(twist(5), 0.0, 1e-6) << "roll rate";
  // And the sideways component is not a rounding artefact: the pelvis really is offset from the CoM.
  EXPECT_GT(r.norm(), 1e-3);
}

/**
 * The linear entries are the CoM velocity, so a velocity command must land in them unscaled. Guards against the
 * inertia fix being applied to the wrong entries.
 */
TEST_F(YawCommandDynamicsTest, LinearVelocityCommandIsCentreOfMassVelocity) {
  const vector6_t h = settledMomentumTarget(vector4_t(0.7, -0.2, 0.0, 0.0), initialState_);
  EXPECT_NEAR(h(0), 0.7, 1e-6);
  EXPECT_NEAR(h(1), -0.2, 1e-6);
  EXPECT_NEAR(h.tail<4>().norm(), 0.0, 1e-9);
}

/**
 * The yaw rate command is given in the pelvis frame and integrated in the world frame. With the base turned, the
 * angular momentum about the world vertical is unchanged by that rotation, so the emitted momentum entry must not
 * depend on the base yaw.
 */
TEST_F(YawCommandDynamicsTest, YawMomentumTargetIsInvariantToBaseYaw) {
  const scalar_t yawRate = 0.25;
  const vector6_t reference = settledMomentumTarget(vector4_t(0.0, 0.0, 0.0, yawRate), initialState_);

  for (const scalar_t baseYaw : {0.7, -1.9, 2.8}) {
    vector_t turned = initialState_;
    robotModel_->setBaseOrientationEulerZYX(turned, vector3_t(baseYaw, 0.0, 0.0));
    const vector6_t h = settledMomentumTarget(vector4_t(0.0, 0.0, 0.0, yawRate), turned);
    EXPECT_NEAR(h(5), reference(5), 1e-9) << "base yaw " << baseYaw;
    // The products of inertia I_xz, I_yz rotate with the body; only their magnitude is invariant.
    EXPECT_NEAR(h.segment<2>(3).norm(), reference.segment<2>(3).norm(), 1e-9) << "base yaw " << baseYaw;
    EXPECT_NEAR(yawInertia(turned), yawInertia(initialState_), 1e-9) << "I_zz must not change under a rotation about z";
  }
}

/**
 * The inertia the mapping relies on is a property of the multibody model, not of the base placement: rotating the base
 * about the vertical leaves it unchanged, and it agrees with the yaw column of the centroidal momentum matrix, which is
 * the momentum a unit yaw rate produces with the joints locked.
 */
TEST_F(YawCommandDynamicsTest, YawInertiaMatchesTheCentroidalMomentumMatrix) {
  PinocchioInterface pinocchio(*pinocchioInterface_);
  const vector_t q = robotModel_->getGeneralizedCoordinates(initialState_);
  updateCentroidalDynamics(pinocchio, info_, q);
  // Column 3 of the CMM is the momentum of a unit ZYX yaw rate; with the base level that is a unit angular velocity
  // about the vertical, so its z component is I_zz.
  const scalar_t fromCmm = getCentroidalMomentumMatrix(pinocchio)(5, kBaseYawIndex);
  EXPECT_NEAR(fromCmm, yawInertia(initialState_), 1e-9);
  EXPECT_GT(fromCmm, 1.0);
}

/**
 * The ACoM-LIP heading model is fed from the whole-body state by the reference manager: the heading is the yaw of the
 * angular centre of mass, its rate is the normalized angular momentum about the vertical scaled by mass over the locked
 * yaw inertia, and the foot yaws are the contact frames' yaws unwrapped near the heading. Each is checked against an
 * independent evaluation of the same quantity from the robot model.
 */
TEST_F(YawCommandDynamicsTest, PlannerInputCarriesTheAcomHeadingAndItsRateFromAngularMomentum) {
  std::unique_ptr<ContactPlanningReferenceManager> referenceManager = makeHeadingReferenceManager();
  std::shared_ptr<AngularCenterOfMass> acom = AngularCenterOfMass::createForRobot(modelSettings_->robotName);

  // A turned base spinning about the vertical at a known rate, expressed as the normalized angular momentum the state carries.
  const scalar_t yawRate = 0.3;
  vector_t state = initialState_;
  robotModel_->setBaseOrientationEulerZYX(state, vector3_t(0.7, 0.0, 0.0));
  const matrix3_t Ig = lockedInertia(state);
  state.segment<3>(3) = Ig.col(2) * (yawRate / mass_);  // L / m for a rigid spin about the vertical

  const ContactPlannerInput input = referenceManager->makePlannerInput(0.0, state, vector2_t::Zero());

  // Heading: the ACoM yaw of the state (base yaw plus the learned joint offset), and the planning frame is the heading.
  const vector_t q = robotModel_->getGeneralizedCoordinates(state);
  const scalar_t acomYaw = acom->computeAcomOrientation(q)(0);
  EXPECT_NEAR(input.heading, acomYaw, 1e-12);
  EXPECT_NEAR(input.yaw, input.heading, 1e-12);
  EXPECT_GT(std::abs(acomYaw - 0.7), 1e-4)
      << "the joint offset must actually contribute, or this test would not tell the ACoM yaw from the base yaw";

  // Heading rate: L_z / I_zz recovers the spin the momentum was built from, with the inertia the manager derives itself.
  EXPECT_NEAR(input.yawInertia, Ig(2, 2), 1e-9);
  EXPECT_NEAR(input.headingRate, yawRate, 1e-9);

  // Foot yaws: the contact frames' yaws from forward kinematics, unwrapped to within pi of the heading.
  PinocchioInterface pinocchio(*pinocchioInterface_);
  pinocchio::forwardKinematics(pinocchio.getModel(), pinocchio.getData(), q);
  pinocchio::updateFramePlacements(pinocchio.getModel(), pinocchio.getData());
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const auto frameId = pinocchio.getModel().getFrameId(modelSettings_->contactNames6DoF[foot]);
    const matrix3_t R = pinocchio.getData().oMf[frameId].rotation();
    const scalar_t frameYaw = std::atan2(R(1, 0), R(0, 0));
    EXPECT_NEAR(std::remainder(input.footYaws[foot] - frameYaw, 2.0 * M_PI), 0.0, 1e-9) << "foot " << foot;
    EXPECT_LE(std::abs(input.footYaws[foot] - input.heading), M_PI + 1e-9) << "unwrapped near the heading, foot " << foot;
  }
}

/**
 * The yaw rate command handed to the planner is the operator's, read off the momentum channel of the calculator's
 * target where the command is carried as h_z = I_zz omega / m. The target's base yaw cannot serve as the source: its
 * first stretch integrates the average of the measured and the commanded yaw rate, so a robot at rest commanded
 * 0.5 rad/s was planned at 0.25 rad/s, and a robot that under-tracked its turn was asked for less and less.
 */
TEST_F(YawCommandDynamicsTest, PlannerYawRateCommandIsTheOperatorsNotTheBlendedTargetYaw) {
  std::unique_ptr<ContactPlanningReferenceManager> referenceManager = makeHeadingReferenceManager();
  const scalar_t yawRate = 0.5;
  const scalar_t horizon = 1.0;
  const TargetTrajectories target = settledTarget(vector4_t(0.0, 0.0, 0.0, yawRate), initialState_);

  // What differentiating the target's base yaw over its first stretch gives for a robot at rest: half the command.
  const scalar_t yaw0 = robotModel_->getBaseOrientationEulerZYX(target.getDesiredState(0.0))(0);
  const scalar_t yaw1 = robotModel_->getBaseOrientationEulerZYX(target.getDesiredState(0.2))(0);
  EXPECT_NEAR((yaw1 - yaw0) / 0.2, 0.5 * yawRate, 1e-6) << "the blended ramp is why the base yaw is not the command source";

  referenceManager->setTargetTrajectories(target);
  referenceManager->preSolverRun(0.0, horizon, initialState_, ModeNumber::STANCE);
  EXPECT_NEAR(referenceManager->commandedYawRate(), yawRate, 1e-6);
  const ContactPlannerInput input = referenceManager->makePlannerInput(0.0, initialState_, vector2_t::Zero());
  EXPECT_NEAR(input.headingRateCommand, yawRate, 1e-6);

  // The same command read with the robot already turning at the commanded rate, and with a turned base: the momentum
  // channel is a base-frame-invariant quantity and the inertia the manager derives matches the calculator's.
  vector_t turning = initialState_;
  robotModel_->setBaseOrientationEulerZYX(turning, vector3_t(1.1, 0.0, 0.0));
  turning.segment<3>(3) = lockedInertia(turning).col(2) * (yawRate / mass_);
  referenceManager->setTargetTrajectories(settledTarget(vector4_t(0.0, 0.0, 0.0, yawRate), turning));
  referenceManager->preSolverRun(0.02, 0.02 + horizon, turning, ModeNumber::STANCE);
  EXPECT_NEAR(referenceManager->commandedYawRate(), yawRate, 1e-6);

  // No yaw command, or a linear command only: no yaw rate is asked of the planner.
  referenceManager->setTargetTrajectories(settledTarget(vector4_t(0.5, 0.1, 0.0, 0.0), initialState_));
  referenceManager->preSolverRun(0.04, 0.04 + horizon, initialState_, ModeNumber::STANCE);
  EXPECT_NEAR(referenceManager->commandedYawRate(), 0.0, 1e-9);
  // An empty target leaves nothing to command.
  referenceManager->setTargetTrajectories(TargetTrajectories());
  referenceManager->preSolverRun(0.06, 0.06 + horizon, initialState_, ModeNumber::STANCE);
  EXPECT_NEAR(referenceManager->commandedYawRate(), 0.0, 1e-9);
}

}  // namespace ocs2::humanoid
