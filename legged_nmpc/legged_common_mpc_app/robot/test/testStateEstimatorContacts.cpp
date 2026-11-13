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

// The kalman state estimator's contact points as the robot process builds
// them (stateEstimatorContacts()): the MPC's contact points on the URDF's
// links, so that a robot standing with its soles on the ground keeps its true
// base height, and a pad on the base is on the URDF's root link.

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <algorithm>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/config/ConfigFiles.h"
#include "legged_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "legged_common_mpc_app/robot/RobotProcessSettings.h"
#include "legged_common_mpc_app/robot/RobotSetUpSteps.h"
#include "legged_common_mpc_app/robot/config/robot/RobotProcessSettingsFromConfig.h"
#include "legged_mpc_config/joint_value.nproto.h"
#include "legged_mpc_config/reference_file.nproto.h"
#include "legged_mpc_config/task_file.nproto.h"
#include "legged_state_estimator/ContactEstimationInputs.h"
#include "legged_state_estimator/ContactKinematics.h"
#include "legged_state_estimator/KalmanBaseEstimator.h"
#include "legged_state_estimator/StateEstimator.h"
#include "legged_state_estimator/StateEstimatorNames.h"
#include "legged_state_estimator/StateEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

namespace ocs2::legged {
namespace {

constexpr char kG1Task[] =
    "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto";
constexpr char kG1Reference[] =
    "robot_models/unitree_g1/g1_centroidal_mpc/config/command/"
    "reference.textproto";
constexpr char kG1Urdf[] =
    "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr char kAtlasMultiContactTask[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/"
    "task.textproto";
constexpr char kAtlasUrdf[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
constexpr char kTron1Task[] =
    "robot_models/limx_tron1/tron1_centroidal_mpc/config/mpc/task.textproto";
constexpr char kTron1Urdf[] =
    "robot_models/limx_tron1/tron1_description/urdf/tron1.urdf";

/** [s] The robot process's control period, 500 Hz. */
constexpr scalar_t kDt = 0.002;
/** [s] How long the standing robot is estimated. */
constexpr scalar_t kStandingDuration = 2.0;
/** [m/s^2] The gravity of the URDF models (Pinocchio's default). */
constexpr scalar_t kGravityMagnitude = 9.81;

/** The robot's files and settings, read once. */
struct Robot {
  mpc_config::TaskFile task;
  ModelSettings modelSettings;
  robot::model::RobotDescription description;
};

Robot loadRobot(absl::string_view taskFile, absl::string_view urdfFile) {
  mpc_config::TaskFile task = loadTaskFile(taskFile).value();
  ModelSettings modelSettings =
      ModelSettings::Create(task, urdfFile, "state_estimator_contacts_test_",
                            /*verbose=*/false)
          .value();
  return Robot{
      .task = std::move(task),
      .modelSettings = std::move(modelSettings),
      .description = robot::model::RobotDescription::Create(urdfFile).value()};
}

/** The G1 at the nominal posture of its reference file: the joints it names,
 * by name; every other joint at zero. */
robot::model::RobotState nominalG1State(
    const robot::model::RobotDescription& description) {
  const mpc_config::ReferenceFile reference =
      loadReferenceFile(kG1Reference).value();
  robot::model::RobotState state(description,
                                 /*contactSize=*/2);
  for (const mpc_config::JointValue& joint : reference.default_joint_state) {
    state.setJointPosition(description.getJointIndex(joint.joint), joint.value);
  }
  return state;
}

/**
 * [m] The base height at which the MPC's contact frames foot_l_contact and
 * foot_r_contact (the sole centers) are on z = 0, for a level base and the
 * joints of `state`: the mean of their heights below the base in the MPC's
 * own Pinocchio model, which places them from the task file independently of
 * the estimator's contact points.
 */
scalar_t solesOnGroundBaseHeight(const Robot& robot,
                                 const robot::model::RobotState& state) {
  const PinocchioInterface interface =
      loadCustomPinocchioInterface(robot.task, kG1Urdf, robot.modelSettings)
          .value();
  const pinocchio::Model& model = interface.getModel();
  pinocchio::Data data(model);
  vector_t q = vector_t::Zero(model.nq);
  q[kJointConfigurationOffset - 1] = 1.0;  // the quaternion's w
  const std::vector<std::string>& joints =
      robot.modelSettings.mpcModelJointNames;
  for (size_t joint = 0; joint < joints.size(); ++joint) {
    q[static_cast<Eigen::Index>(kJointConfigurationOffset + joint)] =
        state.getJointPosition(robot.description.getJointIndex(joints[joint]));
  }
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  scalar_t posContactsRtBaseEwrtBaseZ = 0.0;
  for (const std::string& sole : robot.modelSettings.contactNames6DoF) {
    posContactsRtBaseEwrtBaseZ +=
        data.oMf[model.getFrameId(sole)].translation().z();
  }
  return -posContactsRtBaseEwrtBaseZ /
         static_cast<scalar_t>(robot.modelSettings.contactNames6DoF.size());
}

/** The contact points the robot process gives the kalman estimator. */
std::vector<estimation::ContactPointSpec> contactPointsOf(
    const Robot& robot, absl::string_view urdfFile) {
  return stateEstimatorContacts(robot.modelSettings, robot.task.contacts,
                                urdfFile)
      .value();
}

// Regression (kalman-contact-point-is-ankle-origin-not-sole): a 6-DoF contact
// was at its ankle joint's origin, so the contact height row put the ankle on
// z = 0 and the base settled low by the ankle's height above the sole.
TEST(StateEstimatorContactsTest, RegressionStandingRobotKeepsTrueBaseHeight) {
  const Robot robot = loadRobot(kG1Task, kG1Urdf);
  robot::model::RobotState measured = nominalG1State(robot.description);
  const scalar_t posBaseRtWorldEwrtWorldZ =
      solesOnGroundBaseHeight(robot, measured);
  ASSERT_GT(posBaseRtWorldEwrtWorldZ, 0.0);
  measured.setPosBaseRtWorldEwrtWorld(
      vector3_t(0.0, 0.0, posBaseRtWorldEwrtWorldZ));
  measured.setQuatBaseRtWorld(quaternion_t::Identity());
  measured.setSpecforceBaseRtWorldEwrtBase(
      vector3_t(0.0, 0.0, kGravityMagnitude));
  measured.setContactFlag(/*index=*/0, /*contactFlag=*/true);
  measured.setContactFlag(/*index=*/1, /*contactFlag=*/true);
  measured.setTime(0.0);

  const estimation::StateEstimatorContext context{
      .urdfFile = kG1Urdf,
      .contacts = contactPointsOf(robot, kG1Urdf),
      .description = &robot.description,
      .kalman = estimation::KalmanBaseEstimator::Config()};
  absl::StatusOr<std::unique_ptr<estimation::StateEstimator>> estimator =
      estimation::StateEstimatorRegistry().create(
          estimation::kKalmanStateEstimatorName, context);
  ASSERT_TRUE(estimator.ok()) << estimator.status();
  (*estimator)->reset(measured);
  // Flat ground at z = 0, as the robot process gives it while walking.
  const estimation::ContactEstimationInputs flatGround(/*numContacts=*/2);
  const int steps = static_cast<int>(kStandingDuration / kDt);
  scalar_t estPosBaseRtWorldEwrtWorldZ = 0.0;
  for (int step = 1; step <= steps; ++step) {
    measured.setTime(static_cast<scalar_t>(step) * kDt);
    estPosBaseRtWorldEwrtWorldZ = (*estimator)
                                      ->estimate(measured, flatGround)
                                      .getPosBaseRtWorldEwrtWorld()
                                      .z();
  }
  EXPECT_NEAR(estPosBaseRtWorldEwrtWorldZ, posBaseRtWorldEwrtWorldZ, 1.0e-3)
      << "the true base height puts the sole centers on z = 0";
}

TEST(StateEstimatorContactsTest, TheSolesAreOnTheAnkleLinksAtTheContactOffset) {
  const Robot robot = loadRobot(kG1Task, kG1Urdf);
  const std::vector<estimation::ContactPointSpec> contacts =
      contactPointsOf(robot, kG1Urdf);
  ASSERT_EQ(contacts.size(), 2u);
  EXPECT_EQ(contacts[0].linkName, "left_ankle_roll_link");
  EXPECT_EQ(contacts[1].linkName, "right_ankle_roll_link");
  for (const estimation::ContactPointSpec& contact : contacts) {
    // contacts.pos_contact_rt_link_ewrt_link of the task file.
    EXPECT_EQ(contact.posContactRtLinkEwrtLink, vector3_t(0.035, 0.0, -0.035))
        << contact.linkName;
  }
  // The MPC's free-flyer, which a contact on the base names as its parent, is
  // kMpcRootJointName.
  const PinocchioInterface interface =
      loadCustomPinocchioInterface(robot.task, kG1Urdf, robot.modelSettings)
          .value();
  ASSERT_GT(interface.getModel().njoints, 1);
  EXPECT_EQ(interface.getModel().names[1], estimation::kMpcRootJointName);
}

TEST(StateEstimatorContactsTest, ASeatPadOnTheRootJointIsOnTheUrdfsRootLink) {
  const Robot robot = loadRobot(kAtlasMultiContactTask, kAtlasUrdf);
  RobotProcessSettings settings =
      robotProcessSettingsFromConfig(robot.task).value();
  settings.stateEstimator = std::string(estimation::kKalmanStateEstimatorName);
  const std::vector<estimation::ContactPointSpec> contacts =
      contactPointsOf(robot, kAtlasUrdf);
  ASSERT_EQ(contacts.size(), robot.modelSettings.numContacts());
  const estimation::StateEstimatorContext context{
      .urdfFile = kAtlasUrdf,
      .contacts = contacts,
      .description = &robot.description,
      .kalman = settings.kalmanStateEstimator};
  const absl::StatusOr<std::unique_ptr<estimation::StateEstimator>> estimator =
      estimation::StateEstimatorRegistry().create(settings.stateEstimator,
                                                  context);
  ASSERT_TRUE(estimator.ok()) << estimator.status();

  const std::vector<std::string>& names = robot.modelSettings.contactNames;
  const size_t seat = static_cast<size_t>(
      std::find(names.begin(), names.end(), "seat_contact") - names.begin());
  ASSERT_LT(seat, names.size());
  EXPECT_EQ(contacts[seat].linkName, "pelvis");
  absl::StatusOr<std::unique_ptr<estimation::ContactKinematics>> kinematics =
      estimation::ContactKinematics::Create(kAtlasUrdf, contacts,
                                            robot.description.getJointNames());
  ASSERT_TRUE(kinematics.ok()) << kinematics.status();
  const Eigen::Index numJoints =
      static_cast<Eigen::Index>(robot.description.getNumJoints());
  std::mt19937 generator(/*seed=*/5);
  std::uniform_real_distribution<scalar_t> uniform(-0.5, 0.5);
  vector_t q(numJoints);
  vector_t v(numJoints);
  for (Eigen::Index joint = 0; joint < numJoints; ++joint) {
    q[joint] = uniform(generator);
    v[joint] = uniform(generator);
  }
  (*kinematics)->update(q, v);
  // contacts.surface_pads' pos_pad_rt_link_ewrt_link of seat_contact.
  EXPECT_TRUE((*kinematics)
                  ->posContactRtBaseEwrtBase(seat)
                  .isApprox(vector3_t(-0.02, 0.0, -0.143), 1.0e-12))
      << (*kinematics)->posContactRtBaseEwrtBase(seat).transpose();
  EXPECT_LT((*kinematics)->linvelContactRtBaseEwrtBase(seat).norm(), 1.0e-12);
}

TEST(StateEstimatorContactsTest,
     TheWheelsAreRollingContactsWhichOnlyTheGroundTruthTakes) {
  const Robot robot = loadRobot(kTron1Task, kTron1Urdf);
  const std::vector<estimation::ContactPointSpec> contacts =
      contactPointsOf(robot, kTron1Urdf);
  ASSERT_EQ(contacts.size(), 2u);
  const std::vector<std::string> links = {"wheel_L_Link", "wheel_R_Link"};
  for (size_t contact = 0; contact < contacts.size(); ++contact) {
    // The wheel link, the wheel center on it (the task file's
    // pos_wheel_rt_link_ewrt_link) and the radius.
    EXPECT_EQ(contacts[contact].linkName, links[contact]);
    EXPECT_EQ(contacts[contact].kind, estimation::ContactPointKind::kRolling);
    EXPECT_EQ(contacts[contact].wheelRadius,
              robot.modelSettings.wheelRadius(contact));
    EXPECT_EQ(
        contacts[contact].posContactRtLinkEwrtLink,
        robot.modelSettings.pointContacts[contact].posContactRtLinkEwrtLink);
    EXPECT_EQ(contacts[contact].contactName,
              robot.modelSettings.contactNames[contact]);
  }
  EXPECT_EQ(contacts[0].wheelRadius, 0.127);
  // The G1's soles stay fixed points.
  for (const estimation::ContactPointSpec& sole :
       contactPointsOf(loadRobot(kG1Task, kG1Urdf), kG1Urdf)) {
    EXPECT_EQ(sole.kind, estimation::ContactPointKind::kFixed);
    EXPECT_EQ(sole.wheelRadius, 0.0);
  }

  const estimation::StateEstimatorContext context{
      .urdfFile = kTron1Urdf,
      .contacts = contacts,
      .description = &robot.description,
      .kalman = estimation::KalmanBaseEstimator::Config()};
  const absl::StatusOr<std::unique_ptr<estimation::StateEstimator>> kalman =
      estimation::StateEstimatorRegistry().create(
          estimation::kKalmanStateEstimatorName, context);
  ASSERT_EQ(kalman.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(kalman.status().message(), "wheel_L_contact"))
      << kalman.status();
  EXPECT_TRUE(absl::StrContains(kalman.status().message(),
                                "models contact points fixed on their links "
                                "only"))
      << kalman.status();
  EXPECT_TRUE(estimation::StateEstimatorRegistry()
                  .create(estimation::kGroundTruthStateEstimatorName, context)
                  .ok());
}

TEST(StateEstimatorContactsTest, RefusesASurfaceContactWithoutAParentJoint) {
  Robot robot = loadRobot(kG1Task, kG1Urdf);
  robot.modelSettings.contactParentJointNames.pop_back();
  const absl::StatusOr<std::vector<estimation::ContactPointSpec>> contacts =
      stateEstimatorContacts(robot.modelSettings, robot.task.contacts, kG1Urdf);
  ASSERT_EQ(contacts.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(contacts.status().message(), "foot_r_contact"))
      << contacts.status();
}

TEST(StateEstimatorContactsTest, RefusesAParentJointTheUrdfDoesNotHave) {
  Robot robot = loadRobot(kG1Task, kG1Urdf);
  robot.modelSettings.contactParentJointNames[0] = "no_such_joint";
  const absl::StatusOr<std::vector<estimation::ContactPointSpec>> contacts =
      stateEstimatorContacts(robot.modelSettings, robot.task.contacts, kG1Urdf);
  ASSERT_EQ(contacts.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(contacts.status().message(), "no_such_joint"))
      << contacts.status();
}

}  // namespace
}  // namespace ocs2::legged
