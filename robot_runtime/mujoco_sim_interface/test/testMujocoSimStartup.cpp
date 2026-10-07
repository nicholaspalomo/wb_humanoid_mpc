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

#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoRenderer.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/MujocoUtils.h"

/*
 * How the simulator and its viewer fail to start: MujocoSimInterface::Create() reports every failure as a Status, and
 * the viewer, which used to end the whole process when there was no display, logs why and lets the simulation run on.
 * Which scenes can hold the robot by the gantry weld, and the hold a scene without one falls back to.
 */

namespace robot::mujoco_sim_interface {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

/** A floating base with an arm on two hinges (the robot description needs at least two joints). */
constexpr char kHingeUrdf[] = R"(
<robot name="hinge_robot">
  <link name="base"/>
  <link name="arm"/>
  <link name="hand"/>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="wrist" type="revolute">
    <parent link="arm"/><child link="hand"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
)";

/** The same robot in MuJoCo. */
constexpr char kHingeScene[] = R"(
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="arm" pos="0 0 -0.2">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.1"/>
        <body name="hand" pos="0.3 0 0">
          <joint name="wrist" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.03" mass="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge_motor" joint="hinge"/>
    <motor name="wrist_motor" joint="wrist"/>
  </actuator>
</mujoco>
)";

/** The scene with a body that already carries the name the simulator gives its projectile: it cannot compile with one. */
constexpr char kSceneWithAProjectileName[] = R"(
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="sim_projectile" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

std::string writeTempFile(const std::string& name, const char* absl_nonnull content) {
  const std::string path = testing::TempDir() + "/" + name;
  std::ofstream(path) << content;
  return path;
}

/** A headless simulator configuration of the hinge robot; the scene has no gantry weld. */
MujocoSimConfig hingeConfig() {
  MujocoSimConfig config;
  config.scenePath = writeTempFile("startup_hinge_scene.xml", kHingeScene);
  config.headless = true;
  config.enableGantry = false;
  config.gantryHold = "kinematic_teleport";
  return config;
}

std::string hingeUrdf() {
  return writeTempFile("startup_hinge_robot.urdf", kHingeUrdf);
}

TEST(MujocoSimInterfaceCreate, BuildsASimulatorOfTheScene) {
  const absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim = MujocoSimInterface::Create(hingeConfig(), hingeUrdf());
  ASSERT_TRUE(sim.ok()) << sim.status();
  ASSERT_NE(*sim, nullptr);
  EXPECT_EQ((*sim)->getModel()->nu, 2);
  EXPECT_EQ((*sim)->projectileBodyId(), -1) << "no projectile was named";
}

TEST(MujocoSimInterfaceCreate, AMissingSceneIsAnInvalidArgumentThatNamesIt) {
  MujocoSimConfig config = hingeConfig();
  config.scenePath = testing::TempDir() + "/no_such_scene.xml";
  EXPECT_THAT(MujocoSimInterface::Create(config, hingeUrdf()).status(),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("no_such_scene.xml")));
  // With a projectile the scene is parsed rather than loaded, and that fails alike.
  config.projectile = "dodgeball";
  EXPECT_THAT(MujocoSimInterface::Create(config, hingeUrdf()).status(),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Could not parse the MuJoCo scene")));
}

TEST(MujocoSimInterfaceCreate, AnUnknownProjectileIsRefusedWithTheValidNames) {
  MujocoSimConfig config = hingeConfig();
  config.projectile = "medicine_ball";
  EXPECT_THAT(MujocoSimInterface::Create(config, hingeUrdf()).status(),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("dodgeball")));
}

TEST(MujocoSimInterfaceCreate, AnUnknownGantryHoldIsRefusedWithTheValidNames) {
  MujocoSimConfig config = hingeConfig();
  config.gantryHold = "crane";
  EXPECT_THAT(MujocoSimInterface::Create(config, hingeUrdf()).status(),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("weld_constraint")));
}

/** The hinge scene with `equality` (the contents of an MJCF <equality> element) declared, written to `name`. */
std::string hingeSceneWithEquality(const std::string& name, absl::string_view equality) {
  const std::string scene =
      absl::StrReplaceAll(kHingeScene, {{"</mujoco>", absl::StrCat("<equality>", equality, "</equality>\n</mujoco>")}});
  return writeTempFile(name, scene.c_str());
}

/** The compiled scene at `path`; null, with a test failure saying why, when it does not load. */
MjModelPtr loadModel(const std::string& path) {
  constexpr int kErrorSize = 1000;
  char error[kErrorSize] = "";
  MjModelPtr model(mj_loadXML(path.c_str(), /*vfs=*/nullptr, error, kErrorSize));
  EXPECT_NE(model, nullptr) << path << ": " << error;
  return model;
}

constexpr char kGantryWeld[] = R"(<weld name="gantry" body1="world" body2="base" relpose="0 0 1 1 0 0 0" active="false"/>)";

TEST(GantryHoldScene, AWeldFromTheWorldToTheFreeJointsBodyHoldsTheRobot) {
  const MjModelPtr model = loadModel(hingeSceneWithEquality("gantry_weld_scene.xml", kGantryWeld));
  ASSERT_NE(model, nullptr);
  EXPECT_THAT(checkSceneSupportsGantryHold(model.get(), GantryHold::kWeldConstraint), IsOk());
  EXPECT_THAT(checkSceneSupportsGantryHold(model.get(), GantryHold::kKinematicTeleport), IsOk());
}

TEST(GantryHoldScene, ASceneWithoutTheWeldCannotHoldItAndSaysHowToDeclareIt) {
  const MjModelPtr model = loadModel(hingeConfig().scenePath);
  ASSERT_NE(model, nullptr);
  EXPECT_THAT(checkSceneSupportsGantryHold(model.get(), GantryHold::kWeldConstraint),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       AllOf(HasSubstr("no equality named 'gantry'"), HasSubstr(R"(body1="world" body2="base")"))));
  // The legacy hold needs nothing of the scene.
  EXPECT_THAT(checkSceneSupportsGantryHold(model.get(), GantryHold::kKinematicTeleport), IsOk());
}

TEST(GantryHoldScene, AGantryEqualityThatDoesNotHoldTheBaseFromTheWorldIsRefused) {
  struct Wrong {
    std::string equality;
    std::string message;
  };
  const Wrong wrong[] = {
      {.equality = R"(<weld name="gantry" body1="base" body2="world"/>)", .message = "joins body1 'base' to body2 'world'"},
      {.equality = R"(<weld name="gantry" body1="world" body2="arm"/>)", .message = "joins body1 'world' to body2 'arm'"},
      {.equality = R"(<weld name="gantry" body1="base"/>)", .message = "joins body1 'base' to body2 'world'"},
      {.equality = R"(<connect name="gantry" body1="world" body2="base" anchor="0 0 0"/>)", .message = "is not a weld between two bodies"},
  };
  for (const Wrong& scene : wrong) {
    const MjModelPtr model = loadModel(hingeSceneWithEquality("gantry_wrong_scene.xml", scene.equality));
    ASSERT_NE(model, nullptr) << scene.equality;
    EXPECT_THAT(checkSceneSupportsGantryHold(model.get(), GantryHold::kWeldConstraint),
                StatusIs(absl::StatusCode::kFailedPrecondition, AllOf(HasSubstr(scene.message), HasSubstr(R"(body2="base")"))))
        << scene.equality;
  }
}

TEST(GantryHoldScene, TheSimulatorHoldsByTheWeldOnlyWhereTheSceneDeclaresIt) {
  MujocoSimConfig config = hingeConfig();
  config.gantryHold = "weld_constraint";
  config.scenePath = hingeSceneWithEquality("gantry_weld_sim_scene.xml", kGantryWeld);
  const absl::StatusOr<std::unique_ptr<MujocoSimInterface>> welded = MujocoSimInterface::Create(config, hingeUrdf());
  ASSERT_TRUE(welded.ok()) << welded.status();
  EXPECT_EQ((*welded)->gantryHold(), GantryHold::kWeldConstraint);

  // Without the weld the simulator still starts, on the legacy hold, and logs an ERROR saying why.
  config.scenePath = hingeConfig().scenePath;
  const absl::StatusOr<std::unique_ptr<MujocoSimInterface>> fallback = MujocoSimInterface::Create(config, hingeUrdf());
  ASSERT_TRUE(fallback.ok()) << fallback.status();
  EXPECT_EQ((*fallback)->gantryHold(), GantryHold::kKinematicTeleport);

  config.gantryHold = "kinematic_teleport";
  config.scenePath = hingeSceneWithEquality("gantry_weld_sim_scene.xml", kGantryWeld);
  const absl::StatusOr<std::unique_ptr<MujocoSimInterface>> teleport = MujocoSimInterface::Create(config, hingeUrdf());
  ASSERT_TRUE(teleport.ok()) << teleport.status();
  EXPECT_EQ((*teleport)->gantryHold(), GantryHold::kKinematicTeleport) << "the named hold, even where the weld exists";
}

TEST(MujocoSimInterfaceCreate, ASceneThatDoesNotCompileWithTheProjectileIsAnInvalidArgument) {
  MujocoSimConfig config = hingeConfig();
  config.scenePath = writeTempFile("startup_projectile_name_scene.xml", kSceneWithAProjectileName);
  config.projectile = "dodgeball";
  EXPECT_THAT(MujocoSimInterface::Create(config, hingeUrdf()).status(),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Could not compile the MuJoCo scene")));
}

TEST(MujocoSimInterfaceCreate, AMissingUrdfIsANotFoundThatNamesIt) {
  // The robot description used to be read by a constructor that threw; Create() now returns its Status.
  EXPECT_THAT(MujocoSimInterface::Create(hingeConfig(), testing::TempDir() + "/no_such_robot.urdf").status(),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("no_such_robot.urdf")));
}

TEST(MujocoSimInterfaceCreate, AUrdfWithTooFewJointsIsAnInvalidArgument) {
  constexpr char kOneJointUrdf[] = R"(
<robot name="one_joint_robot">
  <link name="base"/>
  <link name="arm"/>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
)";
  EXPECT_THAT(MujocoSimInterface::Create(hingeConfig(), writeTempFile("startup_one_joint_robot.urdf", kOneJointUrdf)).status(),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("startup_one_joint_robot.urdf")));
}

/** True when GLFW could find a display here, in which case the viewer would really start. */
bool hasDisplay() {
  return std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr;
}

TEST(MujocoRenderer, WithoutADisplayTheViewerStopsAndSaysSo) {
  if (hasDisplay()) GTEST_SKIP() << "a display is available, so the viewer would start";
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim = MujocoSimInterface::Create(hingeConfig(), hingeUrdf());
  ASSERT_TRUE(sim.ok()) << sim.status();
  {
    MujocoRenderer renderer(sim->get());
    EXPECT_TRUE(renderer.ok());
    renderer.launchRenderThread();
    renderer.waitForInit();
    EXPECT_FALSE(renderer.ok()) << "the viewer reported a window it could not create";
  }  // The destructor joins the render thread without touching the window that was never created.
  (*sim)->simulationStep();
}

TEST(MujocoRenderer, ASimulatorWithAViewerButNoDisplayRunsOn) {
  if (hasDisplay()) GTEST_SKIP() << "a display is available, so the viewer would start";
  MujocoSimConfig config = hingeConfig();
  config.headless = false;
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim = MujocoSimInterface::Create(config, hingeUrdf());
  ASSERT_TRUE(sim.ok()) << sim.status();
  // This used to end the process: the viewer's GLFW start-up failure was a MuJoCo fatal error.
  (*sim)->initSim();
  for (int step = 0; step < 10; ++step) (*sim)->simulationStep();
  (*sim)->updateInterfaceStateFromRobot();
  EXPECT_GT((*sim)->getRobotState().getTime(), 0.0) << "the simulation clock advanced";
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
