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

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc_app/robot/MujocoRobotBackend.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

/*
 * The robot backends by name: `mujoco` builds the simulator from the options the robot process gathers (headless here,
 * on the Atlas scene), an unknown name is refused with the available ones, and a new backend is one add().
 */

namespace ocs2::humanoid {
namespace {

RobotBackendOptions atlasOptions() {
  const robot::model::RobotDescription description = robot_test::atlasDescription();
  const ModelSettings modelSettings =
      ModelSettings::Create(robot_test::kAtlasTask, robot_test::kAtlasUrdf, "centroidal_mpc_", /*verbose=*/false).value();
  RobotBackendOptions options;
  options.robotName = "drc_atlas";
  options.urdfFile = robot_test::kAtlasUrdf;
  options.mjcfFile = robot_test::kAtlasScene;
  options.initialState.emplace(description);
  options.initialState->setRootPositionInWorldFrame(vector3_t(0.0, 0.0, 0.95));
  options.contactFrameNames = modelSettings.contactNames;
  options.contactParentJointNames = modelSettings.contactParentJointNames;
  options.simulator.contactForceThreshold = 7.0;
  options.simulator.visualizations = std::vector<std::string>{};
  options.headless = true;
  return options;
}

TEST(RobotBackendRegistry, TheMujocoBackendIsBuiltByName) {
  const RobotBackendRegistry registry;
  EXPECT_TRUE(registry.has("mujoco"));
  EXPECT_EQ(registry.names(), std::vector<std::string>{"mujoco"});
  absl::StatusOr<std::unique_ptr<RobotBackend>> backend = registry.create("mujoco", atlasOptions());
  ASSERT_TRUE(backend.ok()) << backend.status();
  EXPECT_EQ((*backend)->name(), "mujoco");
  ASSERT_NE((*backend)->simulator(), nullptr);
  EXPECT_TRUE((*backend)->simulator()->isGantryLocked()) << "the robot starts on the gantry";
  EXPECT_FALSE((*backend)->acceptsJointAction()) << "and with its torques off";
  ASSERT_TRUE((*backend)->initialize().ok());
  (*backend)->hardware().updateInterfaceStateFromRobot();
  EXPECT_GT((*backend)->hardware().getRobotState().getRootPositionInWorldFrame().z(), 0.5);

  // The contact estimator only a simulator has.
  robot::model::ContactEstimatorRegistry estimators;
  EXPECT_FALSE(estimators.has("cheater_sim"));
  (*backend)->registerContactEstimators(estimators);
  EXPECT_TRUE(estimators.has("cheater_sim"));
  std::array<vector3_t, kNumContacts> forces;
  (*backend)->readMeasuredContactForces(forces);
  EXPECT_TRUE(forces[0].allFinite() && forces[1].allFinite());
}

TEST(RobotBackendRegistry, AnUnknownNameListsTheAvailableOnes) {
  const RobotBackendRegistry registry;
  const absl::StatusOr<std::unique_ptr<RobotBackend>> backend = registry.create("hardware_v2", atlasOptions());
  EXPECT_EQ(backend.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(backend.status().message().find("hardware_v2"), std::string::npos);
  EXPECT_NE(backend.status().message().find("mujoco"), std::string::npos) << backend.status().message();
}

TEST(RobotBackendRegistry, ANewBackendIsOneAdd) {
  RobotBackendRegistry registry;
  bool built = false;
  registry.add("replay", "a recorded robot", [&](const RobotBackendOptions&) -> absl::StatusOr<std::unique_ptr<RobotBackend>> {
    built = true;
    return absl::UnimplementedError("a test backend");
  });
  EXPECT_TRUE(registry.has("replay"));
  EXPECT_NE(registry.availableNames().find("replay (a recorded robot)"), std::string::npos);
  EXPECT_EQ(registry.create("replay", atlasOptions()).status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_TRUE(built);
}

TEST(MujocoRobotBackend, ASimulatorThatDoesNotStartIsRefusedWithItsScene) {
  RobotBackendOptions options = atlasOptions();
  options.mjcfFile = "robot_models/drc_atlas/drc_atlas_description/urdf/no_such_scene.xml";
  const absl::StatusOr<std::unique_ptr<MujocoRobotBackend>> backend = MujocoRobotBackend::Create(options);
  EXPECT_EQ(backend.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(backend.status().message().find("the MuJoCo simulator did not start on " + options.mjcfFile), std::string::npos)
      << backend.status().message();

  // A URDF the robot description refuses keeps the code RobotDescription::Create() gave it.
  options = atlasOptions();
  options.urdfFile = "robot_models/drc_atlas/drc_atlas_description/urdf/no_such_robot.urdf";
  EXPECT_EQ(MujocoRobotBackend::Create(options).status().code(), absl::StatusCode::kNotFound);
}

TEST(MujocoRobotBackend, TheSimulatorConfigurationComesFromTheOptions) {
  RobotBackendOptions options = atlasOptions();
  options.simulator.gantryHold = "kinematic_teleport";
  options.simulator.projectile = "dodgeball";
  absl::StatusOr<robot::mujoco_sim_interface::MujocoSimConfig> config = MujocoRobotBackend::makeConfig(options);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->scenePath, robot_test::kAtlasScene);
  EXPECT_TRUE(config->headless);
  EXPECT_DOUBLE_EQ(config->contactForceThreshold, 7.0);
  EXPECT_EQ(config->gantryHold, "kinematic_teleport");
  EXPECT_EQ(config->projectile, "dodgeball");
  EXPECT_TRUE(config->visualizations.empty());
  ASSERT_NE(config->initStatePtr_, nullptr);
  EXPECT_DOUBLE_EQ(config->initStatePtr_->getRootPositionInWorldFrame().z(), 0.95);

  // Without visualizations named, the viewer's default set.
  options.simulator.visualizations.reset();
  EXPECT_EQ(MujocoRobotBackend::makeConfig(options)->visualizations, robot::mujoco_sim_interface::defaultVisualizationNames());

  options.simulator.gantryHold = "rope";
  EXPECT_FALSE(MujocoRobotBackend::makeConfig(options).ok()) << "an unknown gantry hold";
  options = atlasOptions();
  options.mjcfFile.clear();
  EXPECT_EQ(MujocoRobotBackend::makeConfig(options).status().code(), absl::StatusCode::kInvalidArgument);
  options = atlasOptions();
  options.initialState.reset();
  EXPECT_EQ(MujocoRobotBackend::makeConfig(options).status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid
