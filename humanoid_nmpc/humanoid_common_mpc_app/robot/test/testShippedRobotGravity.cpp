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
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"
#include "ocs2_core/PhysicalConstants.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "mujoco_sim_interface/MujocoUtils.h"

/*
 * Every shipped robot runs one gravity: the MuJoCo scene the simulator steps
 * (mjModel::opt.gravity), the Pinocchio model of the MPC
 * (pinocchio::Model::gravity, which the whole-body dynamics, the DCM terminal
 * cost, the contact planner and the capture point read through modelGravity()),
 * the centroidal dynamics' kGravitationalAcceleration, and the gravity the GUI
 * aims the dodgeball with. None of them is a configuration key, so a scene or a
 * URDF is the only place one could change, and this is where it would show.
 */

namespace ocs2::humanoid {
namespace {

// LINT.IfChange(shipped_gravity)
// [m/s^2] The gravity remote_control/tk_app/dodgeball.py GRAVITY aims the
// dodgeball with.
constexpr scalar_t kGuiGravity = 9.81;
// LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/tk_app/dodgeball.py:dodgeball_gravity)

struct ShippedRobot {
  const char* absl_nonnull name;
  const char* absl_nonnull taskFile;
  const char* absl_nonnull urdfFile;
  const char* absl_nonnull sceneFile;
};

// LINT.IfChange(shipped_robots)
constexpr std::array<ShippedRobot, 5> kShippedRobots = {{
    {.name = "drc_atlas",
     .taskFile = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/"
                 "task.textproto",
     .urdfFile = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
     .sceneFile =
         "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml"},
    {.name = "engineai_sa01",
     .taskFile = "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/"
                 "config/mpc/task.textproto",
     .urdfFile = "robot_models/engineai_sa01/engineai_sa01_description/urdf/"
                 "zq_sa01.urdf",
     .sceneFile = "robot_models/engineai_sa01/engineai_sa01_description/urdf/"
                  "zq_sa01.xml"},
    {.name = "unitree_g1",
     .taskFile =
         "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
     .sceneFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml"},
    {.name = "unitree_g1_wb",
     .taskFile = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
     .sceneFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml"},
    {.name = "unitree_r1",
     .taskFile = "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/"
                 "task.textproto",
     .urdfFile = "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf",
     .sceneFile = "robot_models/unitree_r1/unitree_r1_description/urdf/R1.xml"},
}};
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/BUILD.bazel:shipped_task_files)

class ShippedRobotGravity : public ::testing::TestWithParam<ShippedRobot> {};

TEST_P(ShippedRobotGravity, TheSimulatorTheMpcAndTheGuiRunOneGravity) {
  const ShippedRobot& robot = GetParam();
  constexpr int kErrorSize = 1000;
  char error[kErrorSize] = "";
  const robot::mujoco_sim_interface::MjModelPtr scene(
      mj_loadXML(robot.sceneFile, /*vfs=*/nullptr, error, kErrorSize));
  ASSERT_NE(scene, nullptr) << robot.name << ": " << error;
  const vector3_t simulated(scene->opt.gravity[0], scene->opt.gravity[1],
                            scene->opt.gravity[2]);

  const absl::StatusOr<ModelSettings> modelSettings =
      ModelSettings::Create(robot.taskFile, robot.urdfFile,
                            "testShippedRobotGravity_", /*verbose=*/false);
  ASSERT_TRUE(modelSettings.ok())
      << robot.name << ": " << modelSettings.status();
  const absl::StatusOr<PinocchioInterface> pinocchioInterface =
      loadCustomPinocchioInterface(robot.taskFile, robot.urdfFile,
                                   *modelSettings);
  ASSERT_TRUE(pinocchioInterface.ok())
      << robot.name << ": " << pinocchioInterface.status();
  const vector3_t modeled = pinocchioInterface->getModel().gravity.linear();

  EXPECT_EQ(simulated, modeled)
      << robot.name
      << ": the simulator and the MPC's model fall under different gravities";
  EXPECT_EQ(modeled, vector3_t(0.0, 0.0, -kGravitationalAcceleration))
      << robot.name
      << ": the Pinocchio model and the centroidal dynamics use different "
         "gravities";
  const absl::StatusOr<scalar_t> pendulumGravity =
      modelGravity(pinocchioInterface->getModel());
  ASSERT_TRUE(pendulumGravity.ok())
      << robot.name << ": " << pendulumGravity.status();
  EXPECT_EQ(*pendulumGravity, kGravitationalAcceleration)
      << robot.name << ": the MPC's pendulums run on another gravity";
  EXPECT_EQ(*pendulumGravity, kGuiGravity)
      << robot.name << ": the GUI aims the dodgeball under another gravity";
}

// googletest's macro defines a static function and reads
// std::tuple_size<...>::value.
// NOLINTNEXTLINE(misc-use-anonymous-namespace): expanded from INSTANTIATE_TEST_SUITE_P.
INSTANTIATE_TEST_SUITE_P(
    ShippedRobots,
    ShippedRobotGravity,
    ::testing::ValuesIn(kShippedRobots),
    [](const ::testing::TestParamInfo<ShippedRobot>& info) {
      return std::string(info.param.name);
    });

TEST(ModelGravityTest, IsTheMagnitudeOfADownwardGravityAndRefusesAnyOther) {
  pinocchio::ModelTpl<scalar_t> model;
  EXPECT_EQ(modelGravity(model).value_or(0.0), kGravitationalAcceleration)
      << "Pinocchio's default is the dynamics' gravity";
  constexpr scalar_t kMarsGravity = 3.71;  // [m/s^2]
  model.gravity.linear(vector3_t(0.0, 0.0, -kMarsGravity));
  EXPECT_EQ(modelGravity(model).value_or(0.0), kMarsGravity);
  for (const vector3_t& refused :
       {vector3_t(0.0, 0.0, kGravitationalAcceleration),
        vector3_t(0.0, 0.0, 0.0),
        vector3_t(1.0, 0.0, -kGravitationalAcceleration),
        vector3_t(0.0, 0.0, std::nan(""))}) {
    model.gravity.linear(refused);
    EXPECT_EQ(modelGravity(model).status().code(),
              absl::StatusCode::kInvalidArgument)
        << refused.transpose();
  }
}

}  // namespace
}  // namespace ocs2::humanoid
