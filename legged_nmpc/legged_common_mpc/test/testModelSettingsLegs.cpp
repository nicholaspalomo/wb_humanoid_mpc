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

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"  // IWYU pragma: keep
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/multibody/model.hpp"  // IWYU pragma: keep

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/config/ConfigFiles.h"
#include "legged_common_mpc/config/weights/StateInputLayout.h"
#include "legged_common_mpc/gait/BipedModes.h"
#include "legged_common_mpc/gait/GaitSchedule.h"
#include "legged_common_mpc/gait/MotionPhaseDefinition.h"
#include "legged_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "legged_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "legged_common_mpc/pinocchio_model/pinocchioUtils.h"
#include "legged_mpc_config/model_settings_config.nproto.h"
#include "legged_mpc_config/task_file.nproto.h"
#include "legged_mpc_config/task_file.nproto.pb.h"
#include "legged_mpc_config/task_file.pb.h"
#include "legged_nmpc/legged_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "nproto/Textproto.h"
#include "robot_core/ResourcePaths.h"

/**
 * The legs of ModelSettings (model_settings.legs): the default left and right
 * legs of every shipped biped, with the hips the locomotion heuristics walk to;
 * no legs on the multi-contact Atlas; the four legs of the Atlas sole-corner
 * set with their derived sides; the four legs of ANYmal B, the two wheeled
 * legs of Tron1 and the four wheeled legs of the wheeled ANYmal C; what a leg
 * description may not say; and the pendulum length above `feetmean`, the mean
 * over the legs' contacts.
 */
namespace ocs2::legged {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

/** A shipped robot configuration, from the repository root (string literals).
 */
struct Robot {
  const char* absl_nonnull taskFile;
  const char* absl_nonnull urdfFile;
};

constexpr char kAtlasTaskFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr char kAtlasUrdfFile[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
constexpr char kAtlasMultiContactTaskFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/"
    "task.textproto";

// LINT.IfChange(shipped_robots)
constexpr std::array<Robot, 5> kRobots = {{
    {.taskFile = kAtlasTaskFile, .urdfFile = kAtlasUrdfFile},
    {.taskFile = "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/"
                 "config/mpc/task.textproto",
     .urdfFile = "robot_models/engineai_sa01/engineai_sa01_description/urdf/"
                 "zq_sa01.urdf"},
    {.taskFile =
         "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"},
    {.taskFile = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"},
    {.taskFile = "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/"
                 "task.textproto",
     .urdfFile = "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf"},
}};

// The shipped quadruped, on its four point feet with one leg each.
constexpr Robot kAnymalB = {
    .taskFile =
        "robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/"
        "config/mpc/task.textproto",
    .urdfFile =
        "robot_models/anybotics_anymal_b/anymal_b_description/urdf/"
        "anymal_b.urdf"};

// The shipped wheeled biped, on one rolling contact per leg.
constexpr Robot kTron1 = {
    .taskFile =
        "robot_models/limx_tron1/tron1_centroidal_mpc/config/mpc/"
        "task.textproto",
    .urdfFile = "robot_models/limx_tron1/tron1_description/urdf/tron1.urdf"};

// The shipped wheeled quadruped, on one rolling contact per leg.
constexpr Robot kAnymalCWheels = {
    .taskFile =
        "robot_models/anybotics_anymal_c_wheels/"
        "anymal_c_wheels_centroidal_mpc/config/mpc/task.textproto",
    .urdfFile =
        "robot_models/anybotics_anymal_c_wheels/"
        "anymal_c_wheels_description/urdf/anymal_c_wheels.urdf"};
// LINT.ThenChange(//legged_nmpc/legged_common_mpc/BUILD.bazel:config_test_data)

// The four sole corners of the Atlas, a quadruped's leg order (as in
// legged_centroidal_mpc/test/testPointContactQuadrupedLikeSet.cpp).
constexpr char kFrontLeft[] = "sole_l_toe_contact";
constexpr char kFrontRight[] = "sole_r_toe_contact";
constexpr char kHindLeft[] = "sole_l_heel_contact";
constexpr char kHindRight[] = "sole_r_heel_contact";

constexpr scalar_t kTolerance = 1.0e-12;

/** A loaded robot: its task file, its URDF path and its model settings. */
struct LoadedRobot {
  mpc_config::TaskFile task;
  std::string urdfFile;
};

absl::StatusOr<LoadedRobot> loadRobot(absl::string_view taskFile,
                                      absl::string_view urdfFile) {
  LoadedRobot loaded;
  ASSIGN_OR_RETURN(const std::string taskPath,
                   robot::resolveResourcePath(taskFile));
  ASSIGN_OR_RETURN(loaded.urdfFile, robot::resolveResourcePath(urdfFile));
  ASSIGN_OR_RETURN(loaded.task, loadTaskFile(taskPath));
  return loaded;
}

absl::StatusOr<ModelSettings> settingsOf(const LoadedRobot& robot) {
  return ModelSettings::Create(robot.task, robot.urdfFile, "testLegs_",
                               /*verbose=*/false);
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

mpc_config::ModelSettingsConfig::Leg leg(absl::string_view name,
                                         std::vector<std::string> contacts,
                                         absl::string_view hipFrameName = "") {
  mpc_config::ModelSettingsConfig::Leg entry;
  entry.name = std::string(name);
  entry.contacts = std::move(contacts);
  entry.hip_frame_name = std::string(hipFrameName);
  return entry;
}

/** The Atlas on its four sole corners, with the legs LF, RF, LH and RH. */
absl::StatusOr<LoadedRobot> soleCornerAtlas() {
  ASSIGN_OR_RETURN(LoadedRobot atlas,
                   loadRobot(kAtlasTaskFile, kAtlasUrdfFile));
  mpc_config::ModelSettingsConfig& settings = atlas.task.model_settings;
  settings.contacts_3dof = {
      soleCorner(kFrontLeft, "l_leg_akx", /*posContactRtLinkEwrtLinkX=*/0.16),
      soleCorner(kFrontRight, "r_leg_akx", /*posContactRtLinkEwrtLinkX=*/0.16),
      soleCorner(kHindLeft, "l_leg_akx", /*posContactRtLinkEwrtLinkX=*/-0.08),
      soleCorner(kHindRight, "r_leg_akx", /*posContactRtLinkEwrtLinkX=*/-0.08),
  };
  settings.contact_names_6dof = {};
  settings.contact_parent_joint_names = {};
  settings.legs = {leg("LF", {kFrontLeft}), leg("RF", {kFrontRight}),
                   leg("LH", {kHindLeft}), leg("RH", {kHindRight})};
  return atlas;
}

/** The MPC model of a robot and a robot model of its quaternion layout, which
 * keeps a reference to the settings it is built with. */
struct MpcModel {
  MpcModel(PinocchioInterface interface, const ModelSettings& settings)
      : pinocchioInterface(std::move(interface)),
        robotModel(settings, StateInputLayout::Mpc::kCentroidal) {}

  PinocchioInterface pinocchioInterface;
  test::QuaternionLayoutRobotModel robotModel;
};

/** The MPC model of `robot` and `settings`, which must outlive it. */
absl::StatusOr<std::unique_ptr<MpcModel>> loadMpcModel(
    const LoadedRobot& robot, const ModelSettings& settings) {
  ASSIGN_OR_RETURN(
      PinocchioInterface pinocchioInterface,
      loadCustomPinocchioInterface(robot.task, robot.urdfFile, settings));
  return std::make_unique<MpcModel>(std::move(pinocchioInterface), settings);
}

/** Generalized coordinates of `model` standing 0.9 m up with a bent pose: its
 * joints at a fixed spread of angles. */
vector_t bentCoordinates(const test::QuaternionLayoutRobotModel& model,
                         size_t numJoints) {
  vector_t state = model.getNeutralState();
  model.setPosBaseRtWorldEwrtWorld(state, vector3_t(0.0, 0.0, 0.9));
  vector_t q = model.getGeneralizedCoordinates(state);
  for (size_t joint = 0; joint < numJoints; ++joint) {
    q(q.size() - numJoints + joint) =
        0.05 * static_cast<scalar_t>(joint % 5) - 0.1;
  }
  return q;
}

TEST(ModelSettingsLegs, EveryShippedBipedGetsALeftAndARightLeg) {
  for (const Robot& robot : kRobots) {
    SCOPED_TRACE(robot.taskFile);
    const absl::StatusOr<LoadedRobot> loaded =
        loadRobot(robot.taskFile, robot.urdfFile);
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    ASSERT_TRUE(loaded->task.model_settings.legs.empty())
        << "a shipped biped names no legs";
    const absl::StatusOr<ModelSettings> settings = settingsOf(*loaded);
    ASSERT_TRUE(settings.ok()) << settings.status();
    ASSERT_TRUE(settings->isBipedContactSet());
    ASSERT_EQ(settings->legs.size(), 2U);
    EXPECT_TRUE(settings->isLeggedContactSet());
    const ModelSettings::Leg& left = settings->legs[0];
    const ModelSettings::Leg& right = settings->legs[1];
    EXPECT_EQ(left.name, "left");
    EXPECT_EQ(right.name, "right");
    EXPECT_THAT(left.contactIndices, ElementsAre(kContactLeftIndex));
    EXPECT_THAT(right.contactIndices, ElementsAre(kContactRightIndex));
    EXPECT_EQ(left.lateralSign, 1.0);
    EXPECT_EQ(right.lateralSign, -1.0);
    EXPECT_EQ(left.longitudinal,
              ModelSettings::LegLongitudinalPosition::kCenter);
    EXPECT_EQ(right.longitudinal,
              ModelSettings::LegLongitudinalPosition::kCenter);
    EXPECT_EQ(settings->legOfContact(kContactLeftIndex), 0U);
    EXPECT_EQ(settings->legOfContact(kContactRightIndex), 1U);

    // The hip the locomotion heuristics walk to on the MPC model, at its
    // neutral configuration.
    const absl::StatusOr<std::unique_ptr<MpcModel>> loadedModel =
        loadMpcModel(*loaded, *settings);
    ASSERT_TRUE(loadedModel.ok()) << loadedModel.status();
    MpcModel& model = **loadedModel;
    const pinocchio::Model& mpcModel = model.pinocchioInterface.getModel();
    pinocchio::Data data(mpcModel);
    pinocchio::forwardKinematics(mpcModel, data, pinocchio::neutral(mpcModel));
    for (const ModelSettings::Leg& entry : settings->legs) {
      const std::string& parent =
          settings->contactParentJointNames[entry.contactIndices[0]];
      const pinocchio::JointIndex hip =
          hipJointOf(mpcModel, mpcModel.getJointId(parent));
      EXPECT_EQ(entry.hipFrameName, mpcModel.names[hip]) << entry.name;
      EXPECT_LT(
          (entry.posHipRtBaseEwrtBase - data.oMi[hip].translation()).norm(),
          kTolerance)
          << entry.name;
    }
  }
}

TEST(ModelSettingsLegs, TheBipedPendulumIsTheMeanOverItsTwoFeet) {
  for (const Robot& robot : kRobots) {
    SCOPED_TRACE(robot.taskFile);
    const absl::StatusOr<LoadedRobot> loaded =
        loadRobot(robot.taskFile, robot.urdfFile);
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    const absl::StatusOr<ModelSettings> settings = settingsOf(*loaded);
    ASSERT_TRUE(settings.ok()) << settings.status();
    const absl::StatusOr<std::unique_ptr<MpcModel>> loadedModel =
        loadMpcModel(*loaded, *settings);
    ASSERT_TRUE(loadedModel.ok()) << loadedModel.status();
    MpcModel& model = **loadedModel;
    const vector_t q =
        bentCoordinates(model.robotModel, settings->mpc_joint_dim);
    EXPECT_EQ(computeProxyPosComRtFeetmeanEwrtWorldZ(
                  q, model.pinocchioInterface, model.robotModel),
              computePosComRtFeetmeanEwrtWorldZ(q, model.pinocchioInterface,
                                                model.robotModel));
  }
}

TEST(ModelSettingsLegs, TheMultiContactAtlasHasNoLegsAndKeepsTheLowestContact) {
  const absl::StatusOr<LoadedRobot> loaded =
      loadRobot(kAtlasMultiContactTaskFile, kAtlasUrdfFile);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  const absl::StatusOr<ModelSettings> settings = settingsOf(*loaded);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_TRUE(settings->legs.empty());
  EXPECT_FALSE(settings->isLeggedContactSet());
  EXPECT_FALSE(settings->legOfContact(0).has_value());
  const absl::StatusOr<std::unique_ptr<MpcModel>> loadedModel =
      loadMpcModel(*loaded, *settings);
  ASSERT_TRUE(loadedModel.ok()) << loadedModel.status();
  MpcModel& model = **loadedModel;
  const vector_t q = bentCoordinates(model.robotModel, settings->mpc_joint_dim);
  EXPECT_EQ(computeProxyPosComRtFeetmeanEwrtWorldZ(q, model.pinocchioInterface,
                                                   model.robotModel),
            computePosComRtLowcontactEwrtWorldZ(q, model.pinocchioInterface,
                                                model.robotModel));
}

TEST(ModelSettingsLegs, TheSoleCornersAreFourLegsWithDerivedSides) {
  const absl::StatusOr<LoadedRobot> atlas = soleCornerAtlas();
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  const absl::StatusOr<ModelSettings> settings = settingsOf(*atlas);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_FALSE(settings->isBipedContactSet());
  EXPECT_TRUE(settings->isLeggedContactSet());
  ASSERT_EQ(settings->legs.size(), 4U);
  const std::array<scalar_t, 4> lateralSigns = {1.0, -1.0, 1.0, -1.0};
  const std::array<const char* absl_nonnull, 4> hips = {
      "l_leg_hpz", "r_leg_hpz", "l_leg_hpz", "r_leg_hpz"};
  for (size_t i = 0; i < settings->legs.size(); ++i) {
    const ModelSettings::Leg& entry = settings->legs[i];
    EXPECT_THAT(entry.contactIndices, ElementsAre(i)) << entry.name;
    EXPECT_EQ(entry.lateralSign, lateralSigns[i]) << entry.name;
    EXPECT_EQ(entry.hipFrameName, hips[i]) << entry.name;
    // A toe and a heel share their hip, so no leg is in front of another.
    EXPECT_EQ(entry.longitudinal,
              ModelSettings::LegLongitudinalPosition::kCenter)
        << entry.name;
    EXPECT_EQ(settings->legOfContact(i), i);
  }
}

TEST(ModelSettingsLegs, TheShippedAnymalBStandsOnFourLegsOfOneFootEach) {
  const absl::StatusOr<LoadedRobot> loaded =
      loadRobot(kAnymalB.taskFile, kAnymalB.urdfFile);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  const absl::StatusOr<ModelSettings> settings = settingsOf(*loaded);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_FALSE(settings->isBipedContactSet());
  EXPECT_FALSE(settings->hasBipedStance());
  EXPECT_TRUE(settings->isLeggedContactSet());
  EXPECT_THAT(settings->contactNames,
              ElementsAre("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"));
  ASSERT_EQ(settings->legs.size(), 4U);
  const std::array<const char* absl_nonnull, 4> names = {"LF", "RF", "LH",
                                                         "RH"};
  const std::array<scalar_t, 4> lateralSigns = {1.0, -1.0, 1.0, -1.0};
  const std::array<ModelSettings::LegLongitudinalPosition, 4> longitudinal = {
      ModelSettings::LegLongitudinalPosition::kFront,
      ModelSettings::LegLongitudinalPosition::kFront,
      ModelSettings::LegLongitudinalPosition::kHind,
      ModelSettings::LegLongitudinalPosition::kHind};
  for (size_t i = 0; i < settings->legs.size(); ++i) {
    const ModelSettings::Leg& entry = settings->legs[i];
    EXPECT_EQ(entry.name, names[i]);
    EXPECT_THAT(entry.contactIndices, ElementsAre(i)) << entry.name;
    EXPECT_EQ(entry.lateralSign, lateralSigns[i]) << entry.name;
    EXPECT_EQ(entry.longitudinal, longitudinal[i]) << entry.name;
    // The derived hip is the abduction joint, the first below the base.
    EXPECT_EQ(entry.hipFrameName, absl::StrCat(names[i], "_HAA")) << entry.name;
    EXPECT_EQ(settings->legOfContact(i), i);
  }
}

TEST(ModelSettingsLegs, TheShippedTron1StandsOnTwoLegsOfOneWheelEach) {
  const absl::StatusOr<LoadedRobot> loaded =
      loadRobot(kTron1.taskFile, kTron1.urdfFile);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  const absl::StatusOr<ModelSettings> settings = settingsOf(*loaded);
  ASSERT_TRUE(settings.ok()) << settings.status();
  // Two rolling contacts are not the biped set of two 6-DoF feet.
  EXPECT_FALSE(settings->isBipedContactSet());
  EXPECT_TRUE(settings->isLeggedContactSet());
  EXPECT_TRUE(settings->hasRollingContacts());
  EXPECT_THAT(settings->contactNames,
              ElementsAre("wheel_L_contact", "wheel_R_contact"));
  ASSERT_EQ(settings->legs.size(), 2U);
  const std::array<const char* absl_nonnull, 2> names = {"left", "right"};
  const std::array<const char* absl_nonnull, 2> hips = {"abad_L_Joint",
                                                        "abad_R_Joint"};
  const std::array<scalar_t, 2> lateralSigns = {1.0, -1.0};
  for (size_t i = 0; i < settings->legs.size(); ++i) {
    const ModelSettings::Leg& entry = settings->legs[i];
    EXPECT_EQ(entry.name, names[i]);
    EXPECT_THAT(entry.contactIndices, ElementsAre(i)) << entry.name;
    EXPECT_TRUE(settings->isRollingContact(i)) << entry.name;
    EXPECT_EQ(entry.lateralSign, lateralSigns[i]) << entry.name;
    // Both hips are abreast, so neither leg is in front of the other.
    EXPECT_EQ(entry.longitudinal,
              ModelSettings::LegLongitudinalPosition::kCenter)
        << entry.name;
    EXPECT_EQ(entry.hipFrameName, hips[i]) << entry.name;
    EXPECT_EQ(settings->legOfContact(i), i);
  }
}

TEST(ModelSettingsLegs, TheShippedAnymalCWheelsStandsOnFourLegsOfOneWheelEach) {
  const absl::StatusOr<LoadedRobot> loaded =
      loadRobot(kAnymalCWheels.taskFile, kAnymalCWheels.urdfFile);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  const absl::StatusOr<ModelSettings> settings = settingsOf(*loaded);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_FALSE(settings->isBipedContactSet());
  EXPECT_TRUE(settings->isLeggedContactSet());
  EXPECT_TRUE(settings->hasRollingContacts());
  EXPECT_THAT(settings->contactNames,
              ElementsAre("LF_wheel_contact", "RF_wheel_contact",
                          "LH_wheel_contact", "RH_wheel_contact"));
  ASSERT_EQ(settings->legs.size(), 4U);
  const std::array<const char* absl_nonnull, 4> names = {"LF", "RF", "LH",
                                                         "RH"};
  const std::array<scalar_t, 4> lateralSigns = {1.0, -1.0, 1.0, -1.0};
  const std::array<ModelSettings::LegLongitudinalPosition, 4> longitudinal = {
      ModelSettings::LegLongitudinalPosition::kFront,
      ModelSettings::LegLongitudinalPosition::kFront,
      ModelSettings::LegLongitudinalPosition::kHind,
      ModelSettings::LegLongitudinalPosition::kHind};
  for (size_t i = 0; i < settings->legs.size(); ++i) {
    const ModelSettings::Leg& entry = settings->legs[i];
    EXPECT_EQ(entry.name, names[i]);
    EXPECT_THAT(entry.contactIndices, ElementsAre(i)) << entry.name;
    EXPECT_TRUE(settings->isRollingContact(i)) << entry.name;
    EXPECT_EQ(entry.lateralSign, lateralSigns[i]) << entry.name;
    EXPECT_EQ(entry.longitudinal, longitudinal[i]) << entry.name;
    // The derived hip is the abduction joint, the first below the base.
    EXPECT_EQ(entry.hipFrameName, absl::StrCat(names[i], "_HAA")) << entry.name;
    EXPECT_EQ(settings->legOfContact(i), i);
  }
}

TEST(ModelSettingsLegs, AnExplicitHipSetsTheFrontAndHindPositions) {
  absl::StatusOr<LoadedRobot> atlas = soleCornerAtlas();
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  // The pitch joints of the hips, 0.05 m ahead of the yaw joints, stand in for
  // the front hips; the hind legs keep the derived yaw joints.
  atlas->task.model_settings.legs[0].hip_frame_name = "l_leg_hpy";
  atlas->task.model_settings.legs[1].hip_frame_name = "r_leg_hpy";
  const absl::StatusOr<ModelSettings> settings = settingsOf(*atlas);
  ASSERT_TRUE(settings.ok()) << settings.status();
  ASSERT_EQ(settings->legs.size(), 4U);
  const scalar_t frontX = settings->legs[0].posHipRtBaseEwrtBase.x();
  const scalar_t hindX = settings->legs[2].posHipRtBaseEwrtBase.x();
  ASSERT_GT(frontX - hindX, 0.02)
      << "the hip pitch joint is not ahead of the hip yaw joint";
  EXPECT_EQ(settings->legs[0].hipFrameName, "l_leg_hpy");
  EXPECT_EQ(settings->legs[0].longitudinal,
            ModelSettings::LegLongitudinalPosition::kFront);
  EXPECT_EQ(settings->legs[1].longitudinal,
            ModelSettings::LegLongitudinalPosition::kFront);
  EXPECT_EQ(settings->legs[2].longitudinal,
            ModelSettings::LegLongitudinalPosition::kHind);
  EXPECT_EQ(settings->legs[3].longitudinal,
            ModelSettings::LegLongitudinalPosition::kHind);
}

TEST(ModelSettingsLegs, TheLeggedPendulumIsTheMeanOverTheLegsContacts) {
  const absl::StatusOr<LoadedRobot> atlas = soleCornerAtlas();
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  const absl::StatusOr<ModelSettings> settings = settingsOf(*atlas);
  ASSERT_TRUE(settings.ok()) << settings.status();
  const absl::StatusOr<std::unique_ptr<MpcModel>> loadedModel =
      loadMpcModel(*atlas, *settings);
  ASSERT_TRUE(loadedModel.ok()) << loadedModel.status();
  MpcModel& model = **loadedModel;
  const vector_t q = bentCoordinates(model.robotModel, settings->mpc_joint_dim);
  const scalar_t posComRtFeetmeanEwrtWorldZ =
      computeProxyPosComRtFeetmeanEwrtWorldZ(q, model.pinocchioInterface,
                                             model.robotModel);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(q, model.pinocchioInterface,
                                                   model.robotModel);
  ASSERT_EQ(posContactsRtWorldEwrtWorld.size(), 4U);
  scalar_t posFeetmeanRtWorldEwrtWorldZ = 0.0;
  for (const vector3_t& posContactRtWorldEwrtWorld :
       posContactsRtWorldEwrtWorld) {
    posFeetmeanRtWorldEwrtWorldZ += posContactRtWorldEwrtWorld.z() / 4.0;
  }
  const scalar_t posComRtWorldEwrtWorldZ =
      model.pinocchioInterface.getData().com[0].z();
  EXPECT_NEAR(posComRtFeetmeanEwrtWorldZ,
              posComRtWorldEwrtWorldZ - posFeetmeanRtWorldEwrtWorldZ,
              kTolerance);
  // The bent pose puts the corners at different heights, so the lowest one is
  // another anchor.
  EXPECT_GT(std::abs(posComRtFeetmeanEwrtWorldZ -
                     computePosComRtLowcontactEwrtWorldZ(
                         q, model.pinocchioInterface, model.robotModel)),
            1.0e-6);
}

/** A change to the four legs of the sole-corner Atlas, and what the refusal
 * says. */
struct LegCase {
  const char* absl_nonnull description;
  std::vector<mpc_config::ModelSettingsConfig::Leg> legs;
  std::vector<std::string> messageParts;
};

TEST(ModelSettingsLegs, ALegDescriptionThatDoesNotHoldIsRefused) {
  const std::vector<LegCase> cases = {
      {.description = "duplicate name",
       .legs = {leg("LF", {kFrontLeft}), leg("LF", {kFrontRight})},
       .messageParts = {"model_settings.legs[1].name is 'LF'",
                        "another leg has"}},
      {.description = "underscore",
       .legs = {leg("L_F", {kFrontLeft})},
       .messageParts = {"model_settings.legs[0].name is 'L_F'",
                        "letters and digits"}},
      {.description = "empty name",
       .legs = {leg(/*name=*/"", {kFrontLeft})},
       .messageParts = {"model_settings.legs[0].name is ''"}},
      {.description = "digits only",
       .legs = {leg("12", {kFrontLeft})},
       .messageParts = {"model_settings.legs[0].name is '12'",
                        "not all digits"}},
      {.description = "a mode name",
       .legs = {leg("STANCE", {kFrontLeft})},
       .messageParts = {"model_settings.legs[0].name is 'STANCE'",
                        "mode name"}},
      {.description = "no contact",
       .legs = {leg("LF", /*contacts=*/{})},
       .messageParts = {"model_settings.legs[0] ('LF') names no contact"}},
      {.description = "unknown contact",
       .legs = {leg("LF", {"no_such_contact"})},
       .messageParts = {"model_settings.legs[0] ('LF') names contact "
                        "'no_such_contact', which is not a contact"}},
      {.description = "a contact in two legs",
       .legs = {leg("LF", {kFrontLeft}), leg("RF", {kFrontRight, kFrontLeft})},
       .messageParts = {"model_settings.legs[1] ('RF') names contact "
                        "'sole_l_toe_contact', which model_settings.legs[0] "
                        "names too"}},
      {.description = "unknown hip frame",
       .legs = {leg("LF", {kFrontLeft}, /*hipFrameName=*/"no_such_hip")},
       .messageParts = {"model_settings.legs[0].hip_frame_name is "
                        "'no_such_hip', which is not a frame of the URDF"}},
      {.description = "a hip on the centerline",
       .legs = {leg("LF", {kFrontLeft}, /*hipFrameName=*/"pelvis")},
       .messageParts = {"model_settings.legs[0] ('LF') has its hip pelvis",
                        "of the base's x-z plane"}},
  };
  for (const LegCase& entry : cases) {
    SCOPED_TRACE(entry.description);
    absl::StatusOr<LoadedRobot> atlas = soleCornerAtlas();
    ASSERT_TRUE(atlas.ok()) << atlas.status();
    atlas->task.model_settings.legs = entry.legs;
    const absl::StatusOr<ModelSettings> settings = settingsOf(*atlas);
    ASSERT_FALSE(settings.ok());
    EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument);
    for (const std::string& part : entry.messageParts) {
      EXPECT_THAT(settings.status().message(), HasSubstr(part));
    }
  }
}

TEST(ModelSettingsLegs, ABipedWhoseFeetAreSwappedIsRefused) {
  absl::StatusOr<LoadedRobot> atlas = loadRobot(kAtlasTaskFile, kAtlasUrdfFile);
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  std::vector<std::string>& parents =
      atlas->task.model_settings.contact_parent_joint_names;
  ASSERT_EQ(parents.size(), 2U);
  std::swap(parents[0], parents[1]);
  const absl::StatusOr<ModelSettings> settings = settingsOf(*atlas);
  ASSERT_FALSE(settings.ok());
  EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(settings.status().message(),
              HasSubstr("the default biped leg 'left' on contact "
                        "'foot_l_contact' has its hip r_leg_hpz"));

  // Named legs override the default.
  atlas->task.model_settings.legs = {leg("right", {"foot_l_contact"}),
                                     leg("left", {"foot_r_contact"})};
  const absl::StatusOr<ModelSettings> named = settingsOf(*atlas);
  ASSERT_TRUE(named.ok()) << named.status();
  EXPECT_EQ(named->legs[0].lateralSign, -1.0);
  EXPECT_EQ(named->legs[1].lateralSign, 1.0);
}

/** The Atlas on four 6-DoF soles, the legs LF, RF, LH and RH: the feet are the
 * front legs, the forearms stand in for the hind ones. Each leg names its hip,
 * so that no hip is derived from an arm. */
absl::StatusOr<LoadedRobot> fourSoleAtlas() {
  ASSIGN_OR_RETURN(LoadedRobot atlas,
                   loadRobot(kAtlasTaskFile, kAtlasUrdfFile));
  mpc_config::ModelSettingsConfig& settings = atlas.task.model_settings;
  settings.contacts_3dof = {};
  settings.contact_names_6dof = {"sole_lf", "sole_rf", "sole_lh", "sole_rh"};
  settings.contact_parent_joint_names = {"l_leg_akx", "r_leg_akx", "l_arm_elx",
                                         "r_arm_elx"};
  settings.legs = {leg("LF", {"sole_lf"}, /*hipFrameName=*/"l_leg_hpz"),
                   leg("RF", {"sole_rf"}, /*hipFrameName=*/"r_leg_hpz"),
                   leg("LH", {"sole_lh"}, /*hipFrameName=*/"l_arm_shz"),
                   leg("RH", {"sole_rh"}, /*hipFrameName=*/"r_arm_shz")};
  return atlas;
}

TEST(ModelSettingsLegs, ABipedWhoseLegsHaveTheFootAliasNamesIsRefused) {
  absl::StatusOr<LoadedRobot> atlas = loadRobot(kAtlasTaskFile, kAtlasUrdfFile);
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  // On the biped's stance a gait file's "RF" is the right foot, so a left leg
  // named RF would be read as the other foot.
  atlas->task.model_settings.legs = {leg("RF", {"foot_l_contact"}),
                                     leg("LF", {"foot_r_contact"})};
  const absl::StatusOr<ModelSettings> swapped = settingsOf(*atlas);
  ASSERT_FALSE(swapped.ok());
  EXPECT_EQ(swapped.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(swapped.status().message(),
              HasSubstr("model_settings.legs[0].name is 'RF', which is the "
                        "foot alias 'RF'"));

  // The same names on the side they alias are refused too: the leg name would
  // never be read.
  atlas->task.model_settings.legs = {leg("left", {"foot_l_contact"}),
                                     leg("RF", {"foot_r_contact"})};
  const absl::StatusOr<ModelSettings> aligned = settingsOf(*atlas);
  ASSERT_FALSE(aligned.ok());
  EXPECT_THAT(aligned.status().message(),
              HasSubstr("model_settings.legs[1].name is 'RF'"));

  // A quadruped's legs may be named LF and RF: its stance is all its contacts.
  const absl::StatusOr<LoadedRobot> quadruped = fourSoleAtlas();
  ASSERT_TRUE(quadruped.ok()) << quadruped.status();
  const absl::StatusOr<ModelSettings> quadrupedSettings =
      settingsOf(*quadruped);
  EXPECT_TRUE(quadrupedSettings.ok()) << quadrupedSettings.status();
}

TEST(ModelSettingsLegs, FourLeggedSixDofSolesStandOnAllOfThem) {
  const absl::StatusOr<LoadedRobot> atlas = fourSoleAtlas();
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  const absl::StatusOr<ModelSettings> settings = settingsOf(*atlas);
  ASSERT_TRUE(settings.ok()) << settings.status();
  ASSERT_TRUE(settings->isLeggedContactSet());
  ASSERT_FALSE(settings->isBipedContactSet());
  const GaitModeLayout layout = gaitModeLayoutOf(*settings);
  constexpr size_t kAllFour = 0b1111;
  EXPECT_EQ(layout.stanceMode, kAllFour)
      << "STANCE is every sole, not the two hind ones";
  const absl::StatusOr<size_t> stance =
      parseModeNumber(kModeNameStance, layout);
  ASSERT_TRUE(stance.ok()) << stance.status();
  EXPECT_EQ(*stance, kAllFour);
  const absl::StatusOr<size_t> leftFront =
      parseModeNumber(kModeNameLeftFoot, layout);
  ASSERT_TRUE(leftFront.ok()) << leftFront.status();
  EXPECT_EQ(*leftFront, 0b1000U) << "LF names the leg, not contact 2";
  const absl::StatusOr<size_t> rightFront =
      parseModeNumber(kModeNameRightFoot, layout);
  ASSERT_TRUE(rightFront.ok()) << rightFront.status();
  EXPECT_EQ(*rightFront, 0b0100U);
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/0b1000, layout),
            kModeNameLeftFoot);
  EXPECT_FALSE(defaultCycleStartMode(layout).has_value());
}

TEST(ModelSettingsLegs, TheBipedAndTheMultiContactAtlasKeepTheFeetStance) {
  const absl::StatusOr<LoadedRobot> biped =
      loadRobot(kAtlasTaskFile, kAtlasUrdfFile);
  ASSERT_TRUE(biped.ok()) << biped.status();
  const absl::StatusOr<ModelSettings> bipedSettings = settingsOf(*biped);
  ASSERT_TRUE(bipedSettings.ok()) << bipedSettings.status();
  EXPECT_EQ(gaitModeLayoutOf(*bipedSettings).stanceMode, kBipedModeStance);
  const absl::StatusOr<LoadedRobot> multiContact =
      loadRobot(kAtlasMultiContactTaskFile, kAtlasUrdfFile);
  ASSERT_TRUE(multiContact.ok()) << multiContact.status();
  const absl::StatusOr<ModelSettings> multiContactSettings =
      settingsOf(*multiContact);
  ASSERT_TRUE(multiContactSettings.ok()) << multiContactSettings.status();
  EXPECT_EQ(gaitModeLayoutOf(*multiContactSettings).stanceMode,
            kBipedModeStance);
}

TEST(ModelSettingsLegs, TheLegsParseFromTheTaskFile) {
  const absl::StatusOr<legged_mpc_config::TaskFile> message =
      nproto::ParseTextproto<legged_mpc_config::TaskFile>(
          "model_settings {\n"
          "  legs { name: \"LF\" contacts: \"a\" contacts: \"b\" "
          "hip_frame_name: \"hip\" }\n"
          "  legs { name: \"RF\" contacts: \"c\" }\n"
          "}\n",
          "task.textproto");
  ASSERT_TRUE(message.ok()) << message.status();
  mpc_config::TaskFile task;
  ASSERT_TRUE(mpc_config::FromProto(*message, &task).ok());
  ASSERT_EQ(task.model_settings.legs.size(), 2U);
  EXPECT_EQ(task.model_settings.legs[0].name, "LF");
  EXPECT_THAT(task.model_settings.legs[0].contacts, ElementsAre("a", "b"));
  EXPECT_EQ(task.model_settings.legs[0].hip_frame_name, "hip");
  EXPECT_EQ(task.model_settings.legs[1].hip_frame_name, "");
}

}  // namespace
}  // namespace ocs2::legged
