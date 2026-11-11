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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/LegJointReference.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/**
 * The foot_motion leg joint reference (LegJointReference.h): its name
 * registry, the leg maps of FootMotionLegReference on ANYbotics ANYmal B's four
 * point feet at its default posture, their refusals, and the reference
 * manager's state and input references with it and without it.
 */
namespace ocs2::humanoid {
namespace {

constexpr char kTaskFile[] =
    "robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/config/mpc/"
    "task.textproto";
constexpr char kReferenceFile[] =
    "robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/config/command/"
    "reference.textproto";
constexpr char kUrdfFile[] =
    "robot_models/anybotics_anymal_b/anymal_b_description/urdf/anymal_b.urdf";
constexpr char kAtlasTaskFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr char kAtlasUrdfFile[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

constexpr scalar_t kTolerance = 1.0e-9;
constexpr size_t kJointsPerLeg = 3;
// [m] The base height of the nominal state.
constexpr scalar_t kNomPosBaseRtWorldEwrtWorldZ = 0.51;
// An entry setLegJointVelocities() must not write.
constexpr scalar_t kSentinel = 7.0;

/** ANYmal B loaded for the reference: its settings, model and posture. Not
 * copyable: the model keeps a reference to the settings. */
struct LoadedRobot {
  std::string referenceFile;
  std::unique_ptr<ModelSettings> settings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;
  std::unique_ptr<test::QuaternionLayoutRobotModel> model;
  vector_t defaultJointState;
};

absl::StatusOr<mpc_config::TaskFile> loadTask(absl::string_view taskPath) {
  ASSIGN_OR_RETURN(const std::string taskFile,
                   robot::resolveResourcePath(taskPath));
  return loadTaskFile(taskFile);
}

absl::StatusOr<LoadedRobot> loadRobot(const mpc_config::TaskFile& task,
                                      absl::string_view urdfPath) {
  LoadedRobot loaded;
  ASSIGN_OR_RETURN(const std::string urdfFile,
                   robot::resolveResourcePath(urdfPath));
  ASSIGN_OR_RETURN(
      ModelSettings settings,
      ModelSettings::Create(task, urdfFile, "testLegJointReference",
                            /*verbose=*/false));
  loaded.settings = std::make_unique<ModelSettings>(std::move(settings));
  ASSIGN_OR_RETURN(
      PinocchioInterface pinocchioInterface,
      loadCustomPinocchioInterface(task, urdfFile, *loaded.settings));
  loaded.pinocchioInterface =
      std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));
  loaded.model = std::make_unique<test::QuaternionLayoutRobotModel>(
      *loaded.settings, StateInputLayout::Mpc::kCentroidal);
  return loaded;
}

absl::StatusOr<LoadedRobot> loadAnymal() {
  ASSIGN_OR_RETURN(const mpc_config::TaskFile task, loadTask(kTaskFile));
  ASSIGN_OR_RETURN(LoadedRobot loaded, loadRobot(task, kUrdfFile));
  ASSIGN_OR_RETURN(loaded.referenceFile,
                   robot::resolveResourcePath(kReferenceFile));
  ASSIGN_OR_RETURN(const mpc_config::ReferenceFile reference,
                   loadReferenceFile(loaded.referenceFile));
  ASSIGN_OR_RETURN(loaded.defaultJointState,
                   defaultJointStateFromConfig(
                       reference, loaded.settings->mpcModelJointNames,
                       loaded.settings->fixedJointNames));
  return loaded;
}

/** The nominal state of `robot`: its default posture, the base up and level.
 */
vector_t nominalState(const LoadedRobot& robot) {
  vector_t state = robot.model->getNeutralState();
  robot.model->setPosBaseRtWorldEwrtWorld(
      state, vector3_t(0.0, 0.0, kNomPosBaseRtWorldEwrtWorldZ));
  robot.model->setJointAngles(state, robot.defaultJointState);
  return state;
}

/** [m/s per rad/s] The joint columns of contact `contact`'s position Jacobian
 * at `state`, in world axes, computed here. */
matrix_t jointJacobian(const LoadedRobot& robot,
                       const vector_t& state,
                       size_t contact) {
  PinocchioInterface pinocchioInterface = *robot.pinocchioInterface;
  const pinocchio::Model& model = pinocchioInterface.getModel();
  matrix_t jacobian = matrix_t::Zero(kTwistDim, model.nv);
  pinocchio::computeFrameJacobian(
      model, pinocchioInterface.getData(),
      robot.model->getGeneralizedCoordinates(state),
      model.getFrameId(robot.settings->contactNames[contact]),
      pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
  return jacobian.block(0, kTwistDim, kPositionDim, model.nv - kTwistDim);
}

TEST(LegJointReferenceNames, RoundTripAndRefuseOthers) {
  for (const std::string& name : legJointReferenceNames()) {
    const absl::StatusOr<LegJointReference> reference =
        legJointReferenceFromName(name);
    ASSERT_TRUE(reference.ok()) << reference.status();
    EXPECT_EQ(legJointReferenceName(*reference), name);
  }
  EXPECT_EQ(*legJointReferenceFromName("default_posture"),
            LegJointReference::kDefaultPosture);
  EXPECT_EQ(*legJointReferenceFromName("foot_motion"),
            LegJointReference::kFootMotion);
  EXPECT_EQ(legJointReferenceFromName("Foot_Motion").status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(FootMotionLegReference, MovesEachContactAtTheRequestedVelocity) {
  const absl::StatusOr<LoadedRobot> robot = loadAnymal();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const vector_t state = nominalState(*robot);
  const absl::StatusOr<FootMotionLegReference> legs =
      FootMotionLegReference::Create(*robot->pinocchioInterface, *robot->model,
                                     state);
  ASSERT_TRUE(legs.ok()) << legs.status();
  ASSERT_EQ(legs->numContacts(), robot->settings->contactNames.size());

  const vector3_t linvelContactRtBaseEwrtHeading(0.3, -0.1, 0.2);
  const size_t jointDim = robot->model->getJointDim();
  std::vector<bool> written(jointDim, false);
  for (size_t contact = 0; contact < legs->numContacts(); ++contact) {
    vector_t jointVelocities =
        vector_t::Constant(static_cast<Eigen::Index>(jointDim), kSentinel);
    legs->setLegJointVelocities(
        contact, legs->getNomPosContactRtBaseEwrtHeading(contact),
        linvelContactRtBaseEwrtHeading, jointVelocities);
    // Only the contact's own leg is written, and no joint by two legs.
    vector_t legJointVelocities =
        vector_t::Zero(static_cast<Eigen::Index>(jointDim));
    size_t legJoints = 0;
    for (size_t joint = 0; joint < jointDim; ++joint) {
      const Eigen::Index entry = static_cast<Eigen::Index>(joint);
      if (jointVelocities(entry) == kSentinel) continue;
      EXPECT_FALSE(written[joint]) << "joint " << joint;
      written[joint] = true;
      legJointVelocities(entry) = jointVelocities(entry);
      ++legJoints;
    }
    EXPECT_EQ(legJoints, kJointsPerLeg) << "contact " << contact;
    // The nominal yaw is 0, so the heading frame is the world's.
    const vector3_t linvel =
        jointJacobian(*robot, state, contact) * legJointVelocities;
    EXPECT_TRUE(linvel.isApprox(linvelContactRtBaseEwrtHeading, kTolerance))
        << "contact " << contact << ": " << linvel.transpose();
  }
}

/** [m] Where contact `contact` of `robot` stands relative to the base, in
 * world axes, with the joints `jointPositions` and the base of `state`. */
vector3_t computePosContactRtBaseEwrtWorld(const LoadedRobot& robot,
                                           const vector_t& state,
                                           const vector_t& jointPositions,
                                           size_t contact) {
  vector_t posed = state;
  robot.model->setJointAngles(posed, jointPositions);
  PinocchioInterface pinocchioInterface = *robot.pinocchioInterface;
  return computePosContactsRtWorldEwrtWorld<scalar_t>(
             robot.model->getGeneralizedCoordinates(posed), pinocchioInterface,
             *robot.model)[contact] -
         robot.model->getPosBaseRtWorldEwrtWorld(state);
}

// [m] How far the leg map may miss a contact displaced by a few centimeters:
// its error is third order in the displacement.
constexpr scalar_t kSecondOrderTolerance = 1.0e-3;
// The relative error of the joint rates at a displaced posture, whose Jacobian
// the map updates to first order.
constexpr scalar_t kRateTolerance = 0.05;

TEST(FootMotionLegReference, PlacesAndMovesEachContactToSecondOrder) {
  const absl::StatusOr<LoadedRobot> robot = loadAnymal();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const vector_t state = nominalState(*robot);
  const absl::StatusOr<FootMotionLegReference> legs =
      FootMotionLegReference::Create(*robot->pinocchioInterface, *robot->model,
                                     state);
  ASSERT_TRUE(legs.ok()) << legs.status();
  const vector3_t errPosContactRtBaseEwrtHeading(0.03, -0.02, 0.04);
  const vector3_t linvelContactRtBaseEwrtHeading(-0.5, 0.1, 0.2);
  for (size_t contact = 0; contact < legs->numContacts(); ++contact) {
    const vector3_t& nominal = legs->getNomPosContactRtBaseEwrtHeading(contact);
    // At its nominal place a contact's leg is at the nominal joints.
    vector_t jointPositions = robot->defaultJointState;
    legs->setLegJointPositions(contact, nominal, jointPositions);
    EXPECT_TRUE(jointPositions.isApprox(robot->defaultJointState, kTolerance))
        << "contact " << contact;
    // Displaced, the forward kinematics put it where it was asked to be.
    const vector3_t target = nominal + errPosContactRtBaseEwrtHeading;
    legs->setLegJointPositions(contact, target, jointPositions);
    const vector3_t placed = computePosContactRtBaseEwrtWorld(
        *robot, state, jointPositions, contact);
    EXPECT_LT((placed - target).norm(), kSecondOrderTolerance)
        << "contact " << contact << ": " << placed.transpose() << " against "
        << target.transpose();
    // And the rates there move it at the requested velocity.
    vector_t jointVelocities = vector_t::Zero(jointPositions.size());
    legs->setLegJointVelocities(contact, target, linvelContactRtBaseEwrtHeading,
                                jointVelocities);
    vector_t posed = state;
    robot->model->setJointAngles(posed, jointPositions);
    const vector3_t linvel =
        jointJacobian(*robot, posed, contact) * jointVelocities;
    EXPECT_LT((linvel - linvelContactRtBaseEwrtHeading).norm(),
              kRateTolerance * linvelContactRtBaseEwrtHeading.norm())
        << "contact " << contact << ": " << linvel.transpose();
  }
}

TEST(FootMotionLegReference, NominalContactPositionIsTheForwardKinematics) {
  const absl::StatusOr<LoadedRobot> robot = loadAnymal();
  ASSERT_TRUE(robot.ok()) << robot.status();
  const vector_t state = nominalState(*robot);
  const absl::StatusOr<FootMotionLegReference> legs =
      FootMotionLegReference::Create(*robot->pinocchioInterface, *robot->model,
                                     state);
  ASSERT_TRUE(legs.ok()) << legs.status();
  PinocchioInterface pinocchioInterface = *robot->pinocchioInterface;
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(
          robot->model->getGeneralizedCoordinates(state), pinocchioInterface,
          *robot->model);
  for (size_t contact = 0; contact < legs->numContacts(); ++contact) {
    const vector3_t expected =
        posContactsRtWorldEwrtWorld[contact] -
        vector3_t(0.0, 0.0, kNomPosBaseRtWorldEwrtWorldZ);
    EXPECT_TRUE(legs->getNomPosContactRtBaseEwrtHeading(contact).isApprox(
        expected, kTolerance))
        << "contact " << contact;
  }
}

TEST(FootMotionLegReference, StanceVelocityCancelsTheBaseTwist) {
  const vector2_t cmdLinvelBaseRtWorldEwrtHeadingXy(0.5, 0.1);
  const scalar_t cmdRateYawBaseRtWorld = 0.4;
  const vector3_t posContactRtBaseEwrtHeading(0.3, 0.2, -0.5);
  const vector3_t linvel =
      FootMotionLegReference::computeStanceLinvelContactRtBaseEwrtHeading(
          cmdLinvelBaseRtWorldEwrtHeadingXy, cmdRateYawBaseRtWorld,
          posContactRtBaseEwrtHeading);
  // The world velocity of a point fixed to the base at r, v + omega e_z x r,
  // plus the contact's own velocity relative to the base, is zero.
  const vector3_t linvelContactRtWorldEwrtHeading =
      vector3_t(cmdLinvelBaseRtWorldEwrtHeadingXy.x(),
                cmdLinvelBaseRtWorldEwrtHeadingXy.y(), 0.0) +
      vector3_t(0.0, 0.0, cmdRateYawBaseRtWorld)
          .cross(posContactRtBaseEwrtHeading);
  EXPECT_TRUE((linvelContactRtWorldEwrtHeading + linvel).isZero(kTolerance))
      << linvel.transpose();
}

/** A 3-DoF contact on the Atlas sole of the ankle `ankle`. */
mpc_config::ModelSettingsConfig::PointContact soleCorner(
    absl::string_view name,
    absl::string_view ankle,
    scalar_t posContactRtLinkEwrtLinkX) {
  mpc_config::ModelSettingsConfig::PointContact contact;
  contact.name = std::string(name);
  contact.parent_joint_name = std::string(ankle);
  contact.pos_contact_rt_link_ewrt_link.x = posContactRtLinkEwrtLinkX;
  contact.pos_contact_rt_link_ewrt_link.z = -0.082;
  return contact;
}

TEST(FootMotionLegReference, RefusesContactsThatShareALeg) {
  // The Atlas on a toe and a heel corner of each sole: two point contacts per
  // leg.
  absl::StatusOr<mpc_config::TaskFile> task = loadTask(kAtlasTaskFile);
  ASSERT_TRUE(task.ok()) << task.status();
  task->model_settings.contacts_3dof = {
      soleCorner("sole_l_toe_contact", "l_leg_akx",
                 /*posContactRtLinkEwrtLinkX=*/0.16),
      soleCorner("sole_l_heel_contact", "l_leg_akx",
                 /*posContactRtLinkEwrtLinkX=*/-0.08),
      soleCorner("sole_r_toe_contact", "r_leg_akx",
                 /*posContactRtLinkEwrtLinkX=*/0.16),
      soleCorner("sole_r_heel_contact", "r_leg_akx",
                 /*posContactRtLinkEwrtLinkX=*/-0.08),
  };
  task->model_settings.contact_names_6dof = {};
  task->model_settings.contact_parent_joint_names = {};
  const absl::StatusOr<LoadedRobot> robot = loadRobot(*task, kAtlasUrdfFile);
  ASSERT_TRUE(robot.ok()) << robot.status();
  const absl::StatusOr<FootMotionLegReference> legs =
      FootMotionLegReference::Create(*robot->pinocchioInterface, *robot->model,
                                     robot->model->getNeutralState());
  EXPECT_EQ(legs.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(FootMotionLegReference, RefusesSixDofFeet) {
  const absl::StatusOr<mpc_config::TaskFile> task = loadTask(kAtlasTaskFile);
  ASSERT_TRUE(task.ok()) << task.status();
  const absl::StatusOr<LoadedRobot> robot = loadRobot(*task, kAtlasUrdfFile);
  ASSERT_TRUE(robot.ok()) << robot.status();
  const absl::StatusOr<FootMotionLegReference> legs =
      FootMotionLegReference::Create(*robot->pinocchioInterface, *robot->model,
                                     robot->model->getNeutralState());
  EXPECT_EQ(legs.status().code(), absl::StatusCode::kInvalidArgument);
}

/** A reference manager on ANYmal B after a solve at t = 0 on the nominal state
 * at rest: all four contacts in stance until kLiftOffTime, then a trot whose
 * first diagonal, RF and LH, swings until kTrotSwitchTime. */
struct TrotSetup {
  std::unique_ptr<SwitchedModelReferenceManager> manager;
  TargetTrajectories target;
  vector_t state;
};

// [s] The trot's events, and the time the tests read, in the first swing.
constexpr scalar_t kLiftOffTime = 0.05;
constexpr scalar_t kTrotSwitchTime = 0.4;
constexpr scalar_t kTrotEndTime = 0.75;
constexpr scalar_t kQueryTime = 0.15;

absl::StatusOr<TrotSetup> makeTrot(const LoadedRobot& robot) {
  TrotSetup setup;
  ASSIGN_OR_RETURN(std::shared_ptr<GaitSchedule> gaitSchedule,
                   GaitSchedule::Create(robot.referenceFile, *robot.settings));
  setup.manager = std::make_unique<SwitchedModelReferenceManager>(
      std::move(gaitSchedule),
      std::make_unique<SwingTrajectoryPlanner>(
          SwingTrajectoryPlanner::Config(),
          /*numFeet=*/robot.settings->contactNames.size()),
      *robot.pinocchioInterface, *robot.model);
  setup.manager->configureNominalStance(nominalState(robot));
  setup.state = nominalState(robot);
  const vector_t zeroInput =
      vector_t::Zero(static_cast<Eigen::Index>(robot.model->getInputDim()));
  setup.target = TargetTrajectories({-100.0, 100.0}, {setup.state, setup.state},
                                    {zeroInput, zeroInput});
  const ModeSchedule trot(
      /*eventTimesInput=*/{kLiftOffTime, kTrotSwitchTime, kTrotEndTime},
      /*modeSequenceInput=*/{0b1111, 0b1001, 0b0110, 0b1001});
  setup.manager->getGaitSchedule()->updateModeSchedule(trot);
  setup.manager->setTargetTrajectories(setup.target);
  setup.manager->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, setup.state,
                              trot.modeAtTime(0.0));
  return setup;
}

TEST(SwitchedModelReferenceManagerLegJoints, ZeroKeepsTheContactInput) {
  const absl::StatusOr<LoadedRobot> robot = loadAnymal();
  ASSERT_TRUE(robot.ok()) << robot.status();
  absl::StatusOr<TrotSetup> trot = makeTrot(*robot);
  ASSERT_TRUE(trot.ok()) << trot.status();
  const vector_t expected = weightCompensatingInput(
      *robot->pinocchioInterface, trot->manager->getContactFlags(kQueryTime),
      *robot->model);
  EXPECT_EQ(
      trot->manager->getDesiredInput(trot->target, trot->state, kQueryTime),
      expected);
  ASSERT_TRUE(trot->manager
                  ->configureLegJointReference(
                      LegJointReference::kDefaultPosture, trot->state)
                  .ok());
  EXPECT_EQ(
      trot->manager->getDesiredInput(trot->target, trot->state, kQueryTime),
      expected);
}

TEST(SwitchedModelReferenceManagerLegJoints,
     FootMotionMovesEachSwingLegAlongItsReference) {
  const absl::StatusOr<LoadedRobot> robot = loadAnymal();
  ASSERT_TRUE(robot.ok()) << robot.status();
  absl::StatusOr<TrotSetup> trot = makeTrot(*robot);
  ASSERT_TRUE(trot.ok()) << trot.status();
  const absl::Status configured = trot->manager->configureLegJointReference(
      LegJointReference::kFootMotion, trot->state);
  ASSERT_TRUE(configured.ok()) << configured;

  const vector_t input =
      trot->manager->getDesiredInput(trot->target, trot->state, kQueryTime);
  const vector_t contactInput = weightCompensatingInput(
      *robot->pinocchioInterface, trot->manager->getContactFlags(kQueryTime),
      *robot->model);
  const Eigen::Index jointStart =
      static_cast<Eigen::Index>(robot->model->getJointVelocitiesStartindex());
  const Eigen::Index jointDim =
      static_cast<Eigen::Index>(robot->model->getJointDim());
  // The contact forces are the weight compensation's.
  EXPECT_EQ(input.head(jointStart), contactInput.head(jointStart));
  const vector_t jointVelocities = input.segment(jointStart, jointDim);

  // The rates are those of the reference posture of the node.
  const vector_t referenceState =
      trot->manager->getDesiredState(trot->target, trot->state, kQueryTime);
  const ContactMask contacts = trot->manager->getContactFlags(kQueryTime);
  size_t swingContacts = 0;
  for (size_t contact = 0; contact < contacts.size(); ++contact) {
    const vector3_t linvel =
        jointJacobian(*robot, referenceState, contact) * jointVelocities;
    if (contacts[contact]) {
      // At rest a stance contact stays where it is.
      EXPECT_TRUE(linvel.isZero(kTolerance))
          << "contact " << contact << ": " << linvel.transpose();
      continue;
    }
    ++swingContacts;
    const std::optional<SwingFootReference> swing =
        trot->manager->getSwingFootReference(contact, kQueryTime);
    if (!swing.has_value()) {
      ADD_FAILURE() << "contact " << contact << " has no swing reference";
      continue;
    }
    EXPECT_GT(swing->desLinvelFootRtWorldEwrtWorld.norm(), kTolerance);
    EXPECT_LT((linvel - swing->desLinvelFootRtWorldEwrtWorld).norm(),
              kRateTolerance * swing->desLinvelFootRtWorldEwrtWorld.norm())
        << "contact " << contact << ": " << linvel.transpose() << " against "
        << swing->desLinvelFootRtWorldEwrtWorld.transpose();
  }
  EXPECT_EQ(swingContacts, 2U);
}

TEST(SwitchedModelReferenceManagerLegJoints,
     DefaultPostureKeepsTheTargetJoints) {
  const absl::StatusOr<LoadedRobot> robot = loadAnymal();
  ASSERT_TRUE(robot.ok()) << robot.status();
  absl::StatusOr<TrotSetup> trot = makeTrot(*robot);
  ASSERT_TRUE(trot.ok()) << trot.status();
  EXPECT_EQ(
      trot->manager->getDesiredState(trot->target, trot->state, kQueryTime),
      robot->model->getDesiredState(trot->target, kQueryTime));
}

TEST(SwitchedModelReferenceManagerLegJoints,
     FootMotionLiftsEachSwingLegAlongItsReference) {
  const absl::StatusOr<LoadedRobot> robot = loadAnymal();
  ASSERT_TRUE(robot.ok()) << robot.status();
  absl::StatusOr<TrotSetup> trot = makeTrot(*robot);
  ASSERT_TRUE(trot.ok()) << trot.status();
  const absl::Status configured = trot->manager->configureLegJointReference(
      LegJointReference::kFootMotion, trot->state);
  ASSERT_TRUE(configured.ok()) << configured;

  const vector_t desiredState =
      trot->manager->getDesiredState(trot->target, trot->state, kQueryTime);
  const vector_t errJointPositions =
      robot->model->getJointAngles(desiredState) - robot->defaultJointState;
  const ContactMask contacts = trot->manager->getContactFlags(kQueryTime);
  size_t swingContacts = 0;
  for (size_t contact = 0; contact < contacts.size(); ++contact) {
    if (contacts[contact]) {
      // At rest a stance contact stays where it stood, at the nominal posture.
      const vector3_t errPosContactRtBaseEwrtHeading =
          jointJacobian(*robot, trot->state, contact) * errJointPositions;
      EXPECT_TRUE(errPosContactRtBaseEwrtHeading.isZero(kTolerance))
          << "contact " << contact << ": "
          << errPosContactRtBaseEwrtHeading.transpose();
      continue;
    }
    ++swingContacts;
    const std::optional<SwingFootReference> swing =
        trot->manager->getSwingFootReference(contact, kQueryTime);
    if (!swing.has_value()) {
      ADD_FAILURE() << "contact " << contact << " has no swing reference";
      continue;
    }
    // Lifted by the swing's height above the ground, in place.
    EXPECT_GT(swing->desPosFootRtWorldEwrtWorld.z(), kTolerance);
    const vector3_t placed = computePosContactRtBaseEwrtWorld(
        *robot, trot->state, robot->model->getJointAngles(desiredState),
        contact);
    const vector3_t target =
        computePosContactRtBaseEwrtWorld(*robot, trot->state,
                                         robot->defaultJointState, contact) +
        vector3_t(0.0, 0.0, swing->desPosFootRtWorldEwrtWorld.z());
    EXPECT_LT((placed - target).norm(), kSecondOrderTolerance)
        << "contact " << contact << ": " << placed.transpose() << " against "
        << target.transpose();
  }
  EXPECT_EQ(swingContacts, 2U);
}

}  // namespace
}  // namespace ocs2::humanoid
