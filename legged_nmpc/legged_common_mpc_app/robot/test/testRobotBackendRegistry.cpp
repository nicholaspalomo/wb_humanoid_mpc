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

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/ContactArray.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/config/ConfigFiles.h"
#include "legged_common_mpc_app/robot/MujocoRobotBackend.h"
#include "legged_common_mpc_app/robot/RobotBackend.h"
#include "legged_common_mpc_app/robot/RobotBackendRegistry.h"
#include "legged_mpc_config/joint_value.nproto.h"
#include "legged_mpc_config/reference_file.nproto.h"
#include "legged_nmpc/legged_common_mpc_app/robot/test/RobotTestSupport.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/visualization/VisualizationRegistry.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

/*
 * The robot backends by name: `mujoco` builds the simulator from the options
 * the robot process gathers (without a viewer here, on the Atlas scene), an
 * unknown name is refused with the available ones, and a new backend is one
 * add(). A set of point contacts only starts it too, on the ANYmal B scene,
 * and so do sets of rolling contacts, on the LimX Tron1 scene and the wheeled
 * ANYmal C scene.
 */

namespace ocs2::legged {
namespace {

/** A shipped robot's files, from the repository root (string literals). */
struct ShippedRobotFiles {
  const char* absl_nonnull name;
  const char* absl_nonnull urdfFile;
  const char* absl_nonnull sceneFile;
  const char* absl_nonnull taskFile;
  const char* absl_nonnull referenceFile;
};

constexpr ShippedRobotFiles kAnymalB = {
    .name = "anymal_b",
    .urdfFile =
        "robot_models/anybotics_anymal_b/anymal_b_description/urdf/"
        "anymal_b.urdf",
    .sceneFile =
        "robot_models/anybotics_anymal_b/anymal_b_description/urdf/"
        "anymal_b.xml",
    .taskFile =
        "robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/"
        "config/mpc/task.textproto",
    .referenceFile =
        "robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/"
        "config/command/reference.textproto"};

constexpr ShippedRobotFiles kTron1 = {
    .name = "tron1",
    .urdfFile = "robot_models/limx_tron1/tron1_description/urdf/tron1.urdf",
    .sceneFile = "robot_models/limx_tron1/tron1_description/urdf/tron1.xml",
    .taskFile =
        "robot_models/limx_tron1/tron1_centroidal_mpc/config/mpc/"
        "task.textproto",
    .referenceFile =
        "robot_models/limx_tron1/tron1_centroidal_mpc/config/"
        "command/reference.textproto"};

constexpr ShippedRobotFiles kAnymalCWheels = {
    .name = "anymal_c_wheels",
    .urdfFile =
        "robot_models/anybotics_anymal_c_wheels/anymal_c_wheels_description/"
        "urdf/anymal_c_wheels.urdf",
    .sceneFile =
        "robot_models/anybotics_anymal_c_wheels/anymal_c_wheels_description/"
        "urdf/anymal_c_wheels.xml",
    .taskFile =
        "robot_models/anybotics_anymal_c_wheels/"
        "anymal_c_wheels_centroidal_mpc/config/mpc/task.textproto",
    .referenceFile =
        "robot_models/anybotics_anymal_c_wheels/"
        "anymal_c_wheels_centroidal_mpc/config/command/reference.textproto"};

RobotBackendOptions atlasOptions() {
  const robot::model::RobotDescription description =
      robot_test::atlasDescription();
  const ModelSettings modelSettings =
      ModelSettings::Create(robot_test::kAtlasTask, robot_test::kAtlasUrdf,
                            "centroidal_mpc_", /*verbose=*/false)
          .value();
  RobotBackendOptions options;
  options.robotName = "drc_atlas";
  options.urdfFile = robot_test::kAtlasUrdf;
  options.mjcfFile = robot_test::kAtlasScene;
  options.initialState.emplace(description);
  options.initialState->setPosBaseRtWorldEwrtWorld(vector3_t(0.0, 0.0, 0.95));
  options.contactFrameNames = modelSettings.contactNames;
  options.contactParentJointNames =
      modelSettings.contactParentJointNamesInContactOrder();
  options.numPointContacts = modelSettings.contactNames3DoF.size();
  options.simulator.minForceMjcontactEwrtMjcontactX = 7.0;
  options.simulator.visualizations = std::vector<std::string>{};
  options.viewer.names = {robot::mujoco_sim_interface::kNoViewerName};
  return options;
}

/** Returns the options of the mujoco backend on the scene of `files`, the robot
 * in the default stance of its reference file (default_joint_state, every other
 * joint at zero) with the base level at nom_pos_base_rt_surface_ewrt_world_z
 * (which the file must set), its contacts in the task file's order, and no
 * viewer. */
RobotBackendOptions standingOptions(const ShippedRobotFiles& files) {
  const robot::model::RobotDescription description =
      robot::model::RobotDescription::Create(files.urdfFile).value();
  const ModelSettings modelSettings =
      ModelSettings::Create(files.taskFile, files.urdfFile, "centroidal_mpc_",
                            /*verbose=*/false)
          .value();
  const mpc_config::ReferenceFile reference =
      loadReferenceFile(files.referenceFile).value();
  ABSL_CHECK(reference.nom_pos_base_rt_surface_ewrt_world_z.has_value())
      << files.referenceFile << " sets no nom_pos_base_rt_surface_ewrt_world_z";
  RobotBackendOptions options;
  options.robotName = files.name;
  options.urdfFile = files.urdfFile;
  options.mjcfFile = files.sceneFile;
  options.initialState.emplace(description);
  options.initialState->setPosBaseRtWorldEwrtWorld(
      vector3_t(0.0, 0.0, *reference.nom_pos_base_rt_surface_ewrt_world_z));
  for (const mpc_config::JointValue& joint : reference.default_joint_state) {
    options.initialState->setJointPosition(
        description.getJointIndex(joint.joint), joint.value);
  }
  options.contactFrameNames = modelSettings.contactNames;
  options.contactParentJointNames =
      modelSettings.contactParentJointNamesInContactOrder();
  options.numPointContacts = modelSettings.contactNames3DoF.size();
  options.simulator.visualizations = std::vector<std::string>{};
  options.viewer.names = {robot::mujoco_sim_interface::kNoViewerName};
  return options;
}

TEST(RobotBackendRegistry, TheMujocoBackendIsBuiltByName) {
  const RobotBackendRegistry registry;
  EXPECT_TRUE(registry.has("mujoco"));
  EXPECT_EQ(registry.names(), std::vector<std::string>{"mujoco"});
  absl::StatusOr<std::unique_ptr<RobotBackend>> backend =
      registry.create("mujoco", atlasOptions());
  ASSERT_TRUE(backend.ok()) << backend.status();
  EXPECT_EQ((*backend)->name(), "mujoco");
  ASSERT_NE((*backend)->simulator(), nullptr);
  EXPECT_TRUE((*backend)->simulator()->isGantryLocked())
      << "the robot starts on the gantry";
  EXPECT_FALSE((*backend)->acceptsJointAction()) << "and with its torques off";
  const absl::Status initialized = (*backend)->initialize();
  ASSERT_TRUE(initialized.ok()) << initialized;
  EXPECT_EQ((*backend)->simulator()->activeViewerName(),
            robot::mujoco_sim_interface::kNoViewerName);
  (*backend)->hardware().updateInterfaceStateFromRobot();
  EXPECT_GT(
      (*backend)->hardware().getRobotState().getPosBaseRtWorldEwrtWorld().z(),
      0.5);

  // The contact estimator only a simulator has.
  robot::model::ContactEstimatorRegistry estimators;
  EXPECT_FALSE(estimators.has("cheater_sim"));
  (*backend)->registerContactEstimators(estimators);
  EXPECT_TRUE(estimators.has("cheater_sim"));
  // The two feet carry the two force sensors; an entry past them has none.
  const vector3_t unset =
      vector3_t::Constant(std::numeric_limits<scalar_t>::quiet_NaN());
  ContactArray<vector3_t> measForceContactsEwrtWorld(
      /*numContacts=*/static_cast<size_t>(kBipedFootCount) + 1, unset);
  (*backend)->readMeasForceContactsEwrtWorld(measForceContactsEwrtWorld);
  EXPECT_EQ(measForceContactsEwrtWorld.size(),
            static_cast<size_t>(kBipedFootCount) + 1);
  EXPECT_TRUE(measForceContactsEwrtWorld[kContactLeftIndex].allFinite() &&
              measForceContactsEwrtWorld[kContactRightIndex].allFinite());
  EXPECT_TRUE(measForceContactsEwrtWorld[static_cast<size_t>(kBipedFootCount)]
                  .isZero());
}

// A contact set without exactly two 6-DoF contacts has no contact the foot
// sensors measure: every contact reads zero.
TEST(RobotBackendRegistry, TheMujocoFootSensorsNeedTwoSixDofContacts) {
  RobotBackendOptions options = atlasOptions();
  // The first foot counted as a point contact leaves one 6-DoF contact.
  options.numPointContacts = 1;
  const RobotBackendRegistry registry;
  absl::StatusOr<std::unique_ptr<RobotBackend>> backend =
      registry.create("mujoco", options);
  ASSERT_TRUE(backend.ok()) << backend.status();
  ASSERT_TRUE((*backend)->initialize().ok());
  ContactArray<vector3_t> measForceContactsEwrtWorld(
      /*numContacts=*/static_cast<size_t>(kBipedFootCount), vector3_t::Ones());
  (*backend)->readMeasForceContactsEwrtWorld(measForceContactsEwrtWorld);
  EXPECT_TRUE(measForceContactsEwrtWorld[0].isZero());
  EXPECT_TRUE(measForceContactsEwrtWorld[1].isZero());
}

// A set of point contacts only (ANYmal B's four feet, contacts_3dof, no 6-DoF
// contact) starts the simulator: every contact resolves to its foot's body
// through its parent joint, the ground truth reads all four feet on the floor
// once the robot stands on them, and no contact has a foot force sensor.
TEST(RobotBackendRegistry, TheMujocoBackendStartsOnFourPointContacts) {
  const RobotBackendOptions options = standingOptions(kAnymalB);
  ASSERT_EQ(options.contactFrameNames.size(), 4u);
  ASSERT_EQ(options.numPointContacts, 4u) << "no 6-DoF contact";

  const RobotBackendRegistry registry;
  absl::StatusOr<std::unique_ptr<RobotBackend>> backend =
      registry.create("mujoco", options);
  ASSERT_TRUE(backend.ok()) << backend.status();
  ASSERT_TRUE((*backend)->initialize().ok());
  robot::mujoco_sim_interface::MujocoSimInterface* absl_nullable const sim =
      (*backend)->simulator();
  ASSERT_NE(sim, nullptr);
  EXPECT_TRUE(sim->hasContactDetection());
  EXPECT_EQ(sim->getNumDetectedContactPoints(), 4u);
  EXPECT_EQ(sim->getUnresolvedContactMask(), 0u);

  sim->unlockGantry();
  for (int step = 0; step < 50; ++step) sim->simulationStep();
  EXPECT_EQ(sim->getGroundTruthContactMask(), 0b1111u);

  ContactArray<vector3_t> measForceContactsEwrtWorld(/*numContacts=*/4,
                                                     vector3_t::Ones());
  (*backend)->readMeasForceContactsEwrtWorld(measForceContactsEwrtWorld);
  for (size_t contact = 0; contact < 4; ++contact) {
    EXPECT_TRUE(measForceContactsEwrtWorld[contact].isZero()) << contact;
  }
}

// A set of rolling contacts (Tron1's two wheels, contacts_rolling) starts the
// simulator: each contact resolves to its wheel's body through the wheel joint,
// the ground truth reads both wheels on the floor, and no contact has a foot
// force sensor.
TEST(RobotBackendRegistry, TheMujocoBackendStartsOnTwoRollingContacts) {
  const RobotBackendOptions options = standingOptions(kTron1);
  ASSERT_EQ(options.contactFrameNames.size(), 2u);
  ASSERT_EQ(options.numPointContacts, 2u) << "no 6-DoF contact";

  const RobotBackendRegistry registry;
  absl::StatusOr<std::unique_ptr<RobotBackend>> backend =
      registry.create("mujoco", options);
  ASSERT_TRUE(backend.ok()) << backend.status();
  ASSERT_TRUE((*backend)->initialize().ok());
  robot::mujoco_sim_interface::MujocoSimInterface* absl_nullable const sim =
      (*backend)->simulator();
  ASSERT_NE(sim, nullptr);
  EXPECT_TRUE(sim->hasContactDetection());
  EXPECT_EQ(sim->getNumDetectedContactPoints(), 2u);
  EXPECT_EQ(sim->getUnresolvedContactMask(), 0u);

  // Released at its standing height, the robot rests on both wheels before it
  // tips over (it balances only under the MPC).
  sim->unlockGantry();
  for (int step = 0; step < 50; ++step) sim->simulationStep();
  EXPECT_EQ(sim->getGroundTruthContactMask(), 0b11u);

  ContactArray<vector3_t> measForceContactsEwrtWorld(/*numContacts=*/2,
                                                     vector3_t::Ones());
  (*backend)->readMeasForceContactsEwrtWorld(measForceContactsEwrtWorld);
  for (size_t contact = 0; contact < 2; ++contact) {
    EXPECT_TRUE(measForceContactsEwrtWorld[contact].isZero()) << contact;
  }
}

// Four rolling contacts (the wheeled ANYmal C's wheels, contacts_rolling) start
// the simulator: each contact resolves to its wheel's body, and the robot
// stands on its four wheels, which the ground truth reads on the floor.
TEST(RobotBackendRegistry, TheMujocoBackendStartsOnFourRollingContacts) {
  const RobotBackendOptions options = standingOptions(kAnymalCWheels);
  ASSERT_EQ(options.contactFrameNames.size(), 4u);
  ASSERT_EQ(options.numPointContacts, 4u) << "no 6-DoF contact";

  const RobotBackendRegistry registry;
  absl::StatusOr<std::unique_ptr<RobotBackend>> backend =
      registry.create("mujoco", options);
  ASSERT_TRUE(backend.ok()) << backend.status();
  ASSERT_TRUE((*backend)->initialize().ok());
  robot::mujoco_sim_interface::MujocoSimInterface* absl_nullable const sim =
      (*backend)->simulator();
  ASSERT_NE(sim, nullptr);
  EXPECT_TRUE(sim->hasContactDetection());
  EXPECT_EQ(sim->getNumDetectedContactPoints(), 4u);
  EXPECT_EQ(sim->getUnresolvedContactMask(), 0u);

  sim->unlockGantry();
  for (int step = 0; step < 50; ++step) sim->simulationStep();
  EXPECT_EQ(sim->getGroundTruthContactMask(), 0b1111u);
}

TEST(RobotBackendRegistry, AnUnknownNameListsTheAvailableOnes) {
  const RobotBackendRegistry registry;
  const absl::StatusOr<std::unique_ptr<RobotBackend>> backend =
      registry.create("hardware_v2", atlasOptions());
  EXPECT_EQ(backend.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(backend.status().message().find("hardware_v2"), std::string::npos);
  EXPECT_NE(backend.status().message().find("mujoco"), std::string::npos)
      << backend.status().message();
}

TEST(RobotBackendRegistry, ANewBackendIsOneAdd) {
  RobotBackendRegistry registry;
  bool built = false;
  registry.add("replay", "a recorded robot",
               [&](const RobotBackendOptions&)
                   -> absl::StatusOr<std::unique_ptr<RobotBackend>> {
                 built = true;
                 return absl::UnimplementedError("a test backend");
               });
  EXPECT_TRUE(registry.has("replay"));
  EXPECT_NE(registry.availableNames().find("replay (a recorded robot)"),
            std::string::npos);
  EXPECT_EQ(registry.create("replay", atlasOptions()).status().code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_TRUE(built);
}

TEST(MujocoRobotBackend, ASimulatorThatDoesNotStartIsRefusedWithItsScene) {
  RobotBackendOptions options = atlasOptions();
  options.mjcfFile =
      "robot_models/drc_atlas/drc_atlas_description/urdf/no_such_scene.xml";
  const absl::StatusOr<std::unique_ptr<MujocoRobotBackend>> backend =
      MujocoRobotBackend::Create(options);
  EXPECT_EQ(backend.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(backend.status().message().find(
                "the MuJoCo simulator did not start on " + options.mjcfFile),
            std::string::npos)
      << backend.status().message();

  // A URDF the robot description refuses keeps the code
  // RobotDescription::Create() gave it.
  options = atlasOptions();
  options.urdfFile =
      "robot_models/drc_atlas/drc_atlas_description/urdf/no_such_robot.urdf";
  EXPECT_EQ(MujocoRobotBackend::Create(options).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(MujocoRobotBackend, TheSimulatorConfigurationComesFromTheOptions) {
  RobotBackendOptions options = atlasOptions();
  options.simulator.gantryHold = "kinematic_teleport";
  options.simulator.projectile = "dodgeball";
  options.simulator.ragdollJointDamping = 12.5;
  options.simulator.autoResetMinPosBaseRtWorldEwrtWorldZ = 0.3;
  options.controlPeriod = 0.01;
  absl::StatusOr<robot::mujoco_sim_interface::MujocoSimConfig> config =
      MujocoRobotBackend::makeConfig(options);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->scenePath, robot_test::kAtlasScene);
  EXPECT_EQ(
      config->viewer.names,
      std::vector<std::string>{robot::mujoco_sim_interface::kNoViewerName});
  EXPECT_DOUBLE_EQ(config->minForceMjcontactEwrtMjcontactX, 7.0);
  EXPECT_EQ(config->gantryHold, "kinematic_teleport");
  EXPECT_EQ(config->projectile, "dodgeball");
  EXPECT_DOUBLE_EQ(config->ragdollJointDamping, 12.5);
  EXPECT_DOUBLE_EQ(config->autoResetMinPosBaseRtWorldEwrtWorldZ, 0.3);
  // The accelerometer reads the mean over the robot process's control period.
  EXPECT_DOUBLE_EQ(config->accelerometerWindow, 0.01);
  EXPECT_TRUE(config->visualizations.empty());
  if (!config->initialState.has_value()) {
    FAIL() << "the task file sets an initial state";
  }
  EXPECT_DOUBLE_EQ(config->initialState->getPosBaseRtWorldEwrtWorld().z(),
                   0.95);

  // Without visualizations named, the viewer's default set.
  options.simulator.visualizations.reset();
  EXPECT_EQ(MujocoRobotBackend::makeConfig(options)->visualizations,
            robot::mujoco_sim_interface::defaultVisualizationNames());

  options.simulator.gantryHold = "rope";
  EXPECT_FALSE(MujocoRobotBackend::makeConfig(options).ok())
      << "an unknown gantry hold";
  options = atlasOptions();
  options.mjcfFile.clear();
  EXPECT_EQ(MujocoRobotBackend::makeConfig(options).status().code(),
            absl::StatusCode::kInvalidArgument);
  options = atlasOptions();
  options.initialState.reset();
  EXPECT_EQ(MujocoRobotBackend::makeConfig(options).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(MujocoRobotBackend, TheViewerSettingsAreCopiedAndChecked) {
  RobotBackendOptions options = atlasOptions();
  options.viewer.names = {robot::mujoco_sim_interface::kWebViewerName,
                          robot::mujoco_sim_interface::kNoViewerName};
  options.viewer.web.port = 0;
  options.viewer.web.allowedHosts = {"robot.lan"};
  options.viewer.web.cores = {2, 3};
  options.viewer.web.eglDeviceId = 1;
  absl::StatusOr<robot::mujoco_sim_interface::MujocoSimConfig> config =
      MujocoRobotBackend::makeConfig(options);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->viewer.names, (std::vector<std::string>{"web", "none"}));
  EXPECT_EQ(config->viewer.web.port, 0);
  EXPECT_EQ(config->viewer.web.allowedHosts,
            std::vector<std::string>{"robot.lan"});
  EXPECT_EQ(config->viewer.web.cores, (std::vector<int>{2, 3}));
  EXPECT_EQ(config->viewer.web.eglDeviceId, 1);

  // A list the registry refuses is refused before anything is built, and so is
  // a web option out of its range.
  options = atlasOptions();
  options.viewer.names = {"vnc"};
  const absl::Status unknown = MujocoRobotBackend::checkOptions(options);
  EXPECT_EQ(unknown.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(unknown.message().find("glfw, web, none"), std::string::npos)
      << unknown;
  options.viewer.names = {"glfw", "none"};
  EXPECT_EQ(MujocoRobotBackend::checkOptions(options).code(),
            absl::StatusCode::kInvalidArgument)
      << "nothing after glfw";
  options = atlasOptions();
  options.viewer.web.fps = 0;
  EXPECT_EQ(MujocoRobotBackend::checkOptions(options).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::legged
