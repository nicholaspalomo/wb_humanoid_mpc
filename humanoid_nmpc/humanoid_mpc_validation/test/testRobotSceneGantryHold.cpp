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

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "humanoid_common_mpc_app/robot/MujocoRobotBackend.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/MujocoUtils.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

/*
 * Every robot configuration's MuJoCo scene against the gantry hold its task file names, as the robot process starts
 * them: the task file read by loadRobotProcessSettings(), the simulator created by the mujoco backend. A task file that
 * names weld_constraint over a scene without the gantry weld starts the simulator with an ERROR on kinematic_teleport,
 * as both Unitree G1 configurations did until g1_29dof.xml declared the weld, and the Unitree R1 until R1.xml did.
 */

namespace ocs2::humanoid::validation {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::robot::mujoco_sim_interface::checkSceneSupportsGantryHold;
using ::robot::mujoco_sim_interface::GantryHold;
using ::robot::mujoco_sim_interface::MjModelPtr;
using ::robot::mujoco_sim_interface::MjSpecPtr;
using ::testing::HasSubstr;

constexpr int kErrorSize = 1000;

/** The hold a configuration runs on. A change of one changes its closed-loop runs (data/closed_loop/M0_main). */
struct ExpectedHold {
  std::string configuration;
  GantryHold hold = GantryHold::kWeldConstraint;
};

std::vector<ExpectedHold> expectedHolds() {
  return {
      {.configuration = "drc_atlas", .hold = GantryHold::kWeldConstraint},
      {.configuration = "engineai_sa01", .hold = GantryHold::kWeldConstraint},
      {.configuration = "unitree_g1", .hold = GantryHold::kWeldConstraint},
      {.configuration = "unitree_r1", .hold = GantryHold::kWeldConstraint},
      {.configuration = "unitree_g1_wb", .hold = GantryHold::kWeldConstraint},
  };
}

/** The task-file name of `hold`, for comparisons that print readably. */
std::string holdName(GantryHold hold) {
  switch (hold) {
    case GantryHold::kWeldConstraint:
      return robot::mujoco_sim_interface::kWeldConstraintGantryHoldName;
    case GantryHold::kKinematicTeleport:
      return robot::mujoco_sim_interface::kKinematicTeleportGantryHoldName;
  }
  return "an unknown gantry hold";
}

/** The gantry hold the task file of `configuration` names, as the robot process reads it. */
absl::StatusOr<GantryHold> taskFileHold(const RobotConfiguration& configuration) {
  const absl::StatusOr<RobotProcessSettings> settings = loadRobotProcessSettings(configuration.taskFile);
  if (!settings.ok()) return settings.status();
  return robot::mujoco_sim_interface::gantryHoldFromName(settings->simulator.gantryHold);
}

TEST(RobotSceneGantryHold, EveryConfigurationsTaskFileNamesTheHoldItRunsOn) {
  ASSERT_EQ(robotConfigurations().size(), expectedHolds().size()) << "a configuration without an expected hold";
  for (const ExpectedHold& expected : expectedHolds()) {
    const absl::StatusOr<RobotConfiguration> configuration = findRobotConfiguration(expected.configuration);
    ASSERT_TRUE(configuration.ok()) << configuration.status();
    const absl::StatusOr<GantryHold> hold = taskFileHold(*configuration);
    ASSERT_TRUE(hold.ok()) << configuration->taskFile << ": " << hold.status();
    EXPECT_EQ(holdName(*hold), holdName(expected.hold)) << configuration->taskFile;
  }
}

TEST(RobotSceneGantryHold, EveryConfigurationsSceneSupportsTheHoldItsTaskFileNames) {
  for (const RobotConfiguration& configuration : robotConfigurations()) {
    const absl::StatusOr<GantryHold> hold = taskFileHold(configuration);
    ASSERT_TRUE(hold.ok()) << configuration.taskFile << ": " << hold.status();
    char error[kErrorSize] = "";
    const MjModelPtr model(mj_loadXML(configuration.sceneFile.c_str(), /*vfs=*/nullptr, error, kErrorSize));
    ASSERT_NE(model, nullptr) << configuration.sceneFile << ": " << error;
    EXPECT_THAT(checkSceneSupportsGantryHold(model.get(), *hold), IsOk())
        << configuration.name << ": " << configuration.sceneFile << " cannot hold the robot by the " << holdName(*hold) << " that "
        << configuration.taskFile << " names";
  }
}

TEST(RobotSceneGantryHold, TheRobotProcessesSimulatorRunsOnTheHoldTheTaskFileNames) {
  // No fallback: the mujoco backend the robot binaries create runs every configuration on its task file's hold.
  for (const RobotConfiguration& configuration : robotConfigurations()) {
    const absl::StatusOr<RobotProcessSettings> settings = loadRobotProcessSettings(configuration.taskFile);
    ASSERT_TRUE(settings.ok()) << configuration.taskFile << ": " << settings.status();
    const absl::StatusOr<robot::model::RobotDescription> description = robot::model::RobotDescription::Create(configuration.urdfFile);
    ASSERT_TRUE(description.ok()) << configuration.urdfFile << ": " << description.status();
    RobotBackendOptions options;
    options.robotName = configuration.name;
    options.urdfFile = configuration.urdfFile;
    options.mjcfFile = configuration.sceneFile;
    options.initialState.emplace(*description);
    options.simulator = settings->simulator;
    options.headless = true;
    const absl::StatusOr<std::unique_ptr<MujocoRobotBackend>> backend = MujocoRobotBackend::Create(options);
    ASSERT_TRUE(backend.ok()) << configuration.name << ": " << backend.status();
    const robot::mujoco_sim_interface::MujocoSimInterface* absl_nullable simulator = (*backend)->simulator();
    ASSERT_NE(simulator, nullptr) << configuration.name;
    EXPECT_EQ(holdName(simulator->gantryHold()), settings->simulator.gantryHold)
        << configuration.name << ": the simulator fell back from the task file's gantry_hold on " << configuration.sceneFile;
  }
}

TEST(RobotSceneGantryHold, EachWeldedSceneFailsTheCheckWithItsWeldTakenOut) {
  int welded = 0;
  for (const ExpectedHold& expected : expectedHolds()) {
    if (expected.hold != GantryHold::kWeldConstraint) continue;
    const absl::StatusOr<RobotConfiguration> configuration = findRobotConfiguration(expected.configuration);
    ASSERT_TRUE(configuration.ok()) << configuration.status();
    char error[kErrorSize] = "";
    const MjSpecPtr spec(mj_parseXML(configuration->sceneFile.c_str(), /*vfs=*/nullptr, error, kErrorSize));
    ASSERT_NE(spec, nullptr) << configuration->sceneFile << ": " << error;
    mjsElement* absl_nullable weld = mjs_findElement(spec.get(), mjOBJ_EQUALITY, "gantry");
    ASSERT_NE(weld, nullptr) << configuration->sceneFile << " declares no 'gantry' weld for " << configuration->taskFile;
    ++welded;
    ASSERT_EQ(mjs_delete(spec.get(), weld), 0) << configuration->sceneFile;
    const MjModelPtr model(mj_compile(spec.get(), /*vfs=*/nullptr));
    ASSERT_NE(model, nullptr) << configuration->sceneFile << ": " << mjs_getError(spec.get());
    EXPECT_THAT(checkSceneSupportsGantryHold(model.get(), GantryHold::kWeldConstraint),
                StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("no equality named 'gantry'")))
        << configuration->sceneFile;
  }
  EXPECT_EQ(welded, 5) << "every configuration runs on the weld: Atlas, SA01, R1 and both G1 configurations";
}

}  // namespace
}  // namespace ocs2::humanoid::validation
