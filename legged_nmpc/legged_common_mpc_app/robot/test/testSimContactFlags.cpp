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
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "legged_common_mpc/common/ContactMask.h"
#include "legged_common_mpc/mrt/ContactEstimateIntake.h"
#include "legged_common_mpc/mrt/ControllerEvent.h"
#include "legged_common_mpc/mrt/ControllerEventSink.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "robot_model/RobotState.h"
#include "robot_model/RobotStateContactEstimator.h"

/*
 * The contact flags the simulator hands the controller: one per contact point
 * of the controller's set, so a controller of a set other than the biped's two
 * feet takes the simulator's estimate. Headless, on a four-footed scene.
 */

namespace ocs2::legged {
namespace {

/** A floating base with two arm hinges (the robot description needs at least
 * two joints) and four feet on fixed joints. */
constexpr char kFourFootUrdf[] = R"(
<robot name="four_foot_robot">
  <link name="base"/>
  <link name="arm"/>
  <link name="hand"/>
  <link name="foot_lf"/>
  <link name="foot_rf"/>
  <link name="foot_lh"/>
  <link name="foot_rh"/>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="wrist" type="revolute">
    <parent link="arm"/><child link="hand"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="foot_lf_fixed" type="fixed"><parent link="base"/><child link="foot_lf"/></joint>
  <joint name="foot_rf_fixed" type="fixed"><parent link="base"/><child link="foot_rf"/></joint>
  <joint name="foot_lh_fixed" type="fixed"><parent link="base"/><child link="foot_lh"/></joint>
  <joint name="foot_rh_fixed" type="fixed"><parent link="base"/><child link="foot_rh"/></joint>
</robot>
)";

/**
 * The robot standing on its four feet on the floor, their spheres pressed a
 * millimeter into it at the simulator's default initial state (the base 1 m up,
 * every joint at zero).
 */
constexpr char kFourFootScene[] = R"(
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size="0.2 0.1 0.05" mass="4"/>
      <body name="foot_lf" pos="0.2 0.1 -0.951"><geom type="sphere" size="0.05" mass="0.1"/></body>
      <body name="foot_rf" pos="0.2 -0.1 -0.951"><geom type="sphere" size="0.05" mass="0.1"/></body>
      <body name="foot_lh" pos="-0.2 0.1 -0.951"><geom type="sphere" size="0.05" mass="0.1"/></body>
      <body name="foot_rh" pos="-0.2 -0.1 -0.951"><geom type="sphere" size="0.05" mass="0.1"/></body>
      <body name="arm" pos="0 0 0.1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.05"/>
        <body name="hand" pos="0.1 0 0">
          <joint name="wrist" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.01" mass="0.05"/>
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

constexpr double kTimestep = 0.0005;
/** The four feet, in the controller's contact order. */
constexpr size_t kFourContacts = 4;
/** Physics steps before the flags are read: 0.1 s, for the contacts to carry
 * the robot's weight. */
constexpr int kSettleSteps = 200;

std::string writeTempFile(absl::string_view name,
                          const char* absl_nonnull content) {
  const std::string path = absl::StrCat(testing::TempDir(), "/", name);
  std::ofstream(path) << content;
  return path;
}

/** A sink that counts what it is handed. */
class CountingSink final : public ControllerEventSink {
 public:
  bool post(const ControllerEvent& /*event*/) override {
    ++numPosted;
    return true;
  }
  int numPosted = 0;
};

/** The simulator of kFourFootScene with `contactFrameNames` as its contact
 * points, headless and without a gantry. */
std::unique_ptr<robot::mujoco_sim_interface::MujocoSimInterface> makeSim(
    std::vector<std::string> contactFrameNames) {
  robot::mujoco_sim_interface::MujocoSimConfig config;
  config.scenePath = writeTempFile("four_foot_scene.xml", kFourFootScene);
  config.viewer.names = {robot::mujoco_sim_interface::kNoViewerName};
  config.enableGantry = false;
  config.gantryHold = "kinematic_teleport";
  config.dt = kTimestep;
  config.renderFrequencyHz = 1.0 / kTimestep;
  config.ragdollJointDamping = 0.0;
  config.contactFrameNames = std::move(contactFrameNames);
  absl::StatusOr<
      std::unique_ptr<robot::mujoco_sim_interface::MujocoSimInterface>>
      sim = robot::mujoco_sim_interface::MujocoSimInterface::Create(
          config, writeTempFile("four_foot_robot.urdf", kFourFootUrdf));
  EXPECT_TRUE(sim.ok()) << sim.status();
  return sim.ok() ? *std::move(sim) : nullptr;
}

TEST(SimContactFlags, AFourContactSetGetsFourFlagsTheControllerTakes) {
  const std::unique_ptr<robot::mujoco_sim_interface::MujocoSimInterface> sim =
      makeSim({"foot_lf", "foot_rf", "foot_lh", "foot_rh"});
  ASSERT_NE(sim, nullptr);
  ASSERT_TRUE(sim->hasContactDetection());
  for (int step = 0; step < kSettleSteps; ++step) sim->simulationStep();
  sim->updateInterfaceStateFromRobot();
  const robot::model::RobotState& state = sim->getRobotState();
  ASSERT_EQ(state.getContactFlags().size(), kFourContacts);
  for (size_t contact = 0; contact < kFourContacts; ++contact) {
    EXPECT_TRUE(state.getContactFlags()[contact])
        << "foot " << contact << " stands on the floor";
  }

  // The controller's estimate of the simulator's flags, taken by the intake of
  // a four-contact controller.
  robot::model::RobotStateContactEstimator estimator;
  std::vector<bool> estimated;
  estimator.estimateContactFlags(state, estimated);
  ContactEstimateIntake intake("Controller", kFourContacts);
  ContactMask measured(kFourContacts);
  CountingSink sink;
  EXPECT_TRUE(intake.take(estimated, measured, sink));
  EXPECT_EQ(intake.numRefused(), 0U);
  EXPECT_EQ(sink.numPosted, 0);
  EXPECT_EQ(measured.modeNumber(), 0b1111U) << "every foot in stance";
}

TEST(SimContactFlags, NoContactPointsKeepTheTwoDefaultFlags) {
  const std::unique_ptr<robot::mujoco_sim_interface::MujocoSimInterface> sim =
      makeSim(/*contactFrameNames=*/{});
  ASSERT_NE(sim, nullptr);
  sim->simulationStep();
  sim->updateInterfaceStateFromRobot();
  EXPECT_EQ(sim->getRobotState().getContactFlags().size(),
            robot::model::RobotState::kDefaultContactSize);
}

}  // namespace
}  // namespace ocs2::legged
