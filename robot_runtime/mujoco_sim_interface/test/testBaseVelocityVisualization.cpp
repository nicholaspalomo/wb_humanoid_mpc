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

#include <cmath>
#include <numbers>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/visualization/BaseVelocityVisualization.h"

namespace robot::mujoco_sim_interface {
namespace {

// A floating base with a leg, over nothing: the robot of BaseVelocityVisualization (body 1 on the first free joint).
constexpr char kScene[] = R"(
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="10"/>
      <body name="leg" pos="0 0 -0.4">
        <joint name="hip" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.03 0.2" mass="2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

/** An mjvScene of up to `maxGeoms` geoms, freed with the object. Not copyable. */
class OwnedScene {
 public:
  OwnedScene(const mjModel* absl_nonnull model, int maxGeoms) {
    mjv_defaultScene(&scene_);
    mjv_makeScene(model, &scene_, maxGeoms);
  }
  OwnedScene(const OwnedScene&) = delete;
  OwnedScene& operator=(const OwnedScene&) = delete;
  ~OwnedScene() { mjv_freeScene(&scene_); }
  mjvScene& get() { return scene_; }

 private:
  mjvScene scene_ = {};
};

/** The world direction an arrow of mjv_connector() points along: the z axis of its frame. */
void expectPointsAlong(const mjvGeom& arrow, double x, double y, absl::string_view which) {
  // mjvGeom::mat is row-major, so its third column is the arrow's axis.
  EXPECT_NEAR(arrow.mat[2], x, 1.0e-5) << which;
  EXPECT_NEAR(arrow.mat[5], y, 1.0e-5) << which;
  EXPECT_NEAR(arrow.mat[8], 0.0, 1.0e-5) << which;
}

// The measured velocity is a world-frame quantity. It used to be read from the base's row of mjData::cvel, which is in
// world-aligned axes already, and rotated by the base orientation once more: with the base yawed by 90 degrees and the
// robot walking forward exactly as commanded, the red (measured) arrow pointed along -x while the green (commanded) one
// pointed along +y - an error as large as the heading.
TEST(BaseVelocityVisualization, AYawedRobotTrackingItsCommandDrawsBothArrowsAlongTheSameWorldDirection) {
  char error[1000] = "";
  const MjSpecPtr spec(mj_parseXMLString(kScene, /*vfs=*/nullptr, error, sizeof(error)));
  ASSERT_NE(spec, nullptr) << "mj_parseXMLString: " << error;
  const MjModelPtr model(mj_compile(spec.get(), /*vfs=*/nullptr));
  ASSERT_NE(model, nullptr);
  absl::StatusOr<MjDataPtr> dataOrStatus = makeMjData(model.get());
  ASSERT_TRUE(dataOrStatus.ok()) << dataOrStatus.status();
  mjData& data = **dataOrStatus;

  // Yawed by 90 degrees: the base's forward axis is the world's +y. A free joint's linear velocity is in world axes.
  const double halfYaw = std::numbers::pi / 4.0;
  data.qpos[3] = std::cos(halfYaw);
  data.qpos[4] = 0.0;
  data.qpos[5] = 0.0;
  data.qpos[6] = std::sin(halfYaw);
  const double speed = 0.5;
  data.qvel[0] = 0.0;
  data.qvel[1] = speed;
  data.qvel[2] = 0.0;
  mj_forward(model.get(), &data);

  OwnedScene scene(model.get(), /*maxGeoms=*/16);
  BaseVelocityVisualization::addArrows(model.get(), &data, BaseVelocityCommand{.velocityX = speed}, scene.get());

  ASSERT_EQ(scene.get().ngeom, 2) << "a measured and a commanded velocity arrow, and no yaw-rate arrow";
  const mjvGeom& measured = scene.get().geoms[0];
  const mjvGeom& commanded = scene.get().geoms[1];
  ASSERT_EQ(measured.rgba[0], 1.0f) << "the first arrow is the red, measured one";
  ASSERT_EQ(commanded.rgba[1], 1.0f) << "the second arrow is the green, commanded one";
  expectPointsAlong(commanded, /*x=*/0.0, /*y=*/1.0, "commanded");
  expectPointsAlong(measured, /*x=*/0.0, /*y=*/1.0, "measured");
}

// The command is a velocity along the ground in the heading frame. It used to be turned by the whole base attitude, so a
// pitched base drew it tilted into the ground, at its horizontal length, and the yaw-rate arrows along the tilted base z.
TEST(BaseVelocityVisualization, APitchedBaseDrawsTheCommandAlongTheGroundAndTheYawRatesAlongTheVertical) {
  char error[1000] = "";
  const MjSpecPtr spec(mj_parseXMLString(kScene, /*vfs=*/nullptr, error, sizeof(error)));
  ASSERT_NE(spec, nullptr) << "mj_parseXMLString: " << error;
  const MjModelPtr model(mj_compile(spec.get(), /*vfs=*/nullptr));
  ASSERT_NE(model, nullptr);
  absl::StatusOr<MjDataPtr> dataOrStatus = makeMjData(model.get());
  ASSERT_TRUE(dataOrStatus.ok()) << dataOrStatus.status();
  mjData& data = **dataOrStatus;

  // Heading 90 degrees, then pitched by 0.3 rad about the base's y axis: q = q_z(pi/2) (x) q_y(0.3), w first.
  const double heading = std::numbers::pi / 2.0;
  const double pitch = 0.3;
  const mjtNum yawQuaternion[4] = {std::cos(heading / 2.0), 0.0, 0.0, std::sin(heading / 2.0)};
  const mjtNum pitchQuaternion[4] = {std::cos(pitch / 2.0), 0.0, std::sin(pitch / 2.0), 0.0};
  mju_mulQuat(&data.qpos[3], yawQuaternion, pitchQuaternion);
  data.qvel[5] = 0.4;  // a measured angular velocity about the base's own z axis (MuJoCo's free-joint qvel is local)
  mj_forward(model.get(), &data);

  OwnedScene scene(model.get(), /*maxGeoms=*/16);
  const double speed = 0.5;
  BaseVelocityVisualization::addArrows(model.get(), &data, BaseVelocityCommand{.velocityX = speed, .yawRate = 0.7}, scene.get());

  ASSERT_EQ(scene.get().ngeom, 3) << "the commanded velocity arrow and the two yaw-rate arrows (the base does not translate)";
  const mjvGeom& commanded = scene.get().geoms[0];
  ASSERT_EQ(commanded.rgba[1], 1.0f) << "the first arrow is the green, commanded one";
  expectPointsAlong(commanded, /*x=*/0.0, /*y=*/1.0, "commanded, turned by the heading only");
  EXPECT_NEAR(commanded.size[2], speed, 1.0e-6) << "an arrow as long as the commanded speed";
  for (const int index : {1, 2}) {
    const mjvGeom& yawArrow = scene.get().geoms[index];
    EXPECT_NEAR(yawArrow.mat[2], 0.0, 1.0e-6) << "yaw-rate arrow " << index;
    EXPECT_NEAR(yawArrow.mat[5], 0.0, 1.0e-6) << "yaw-rate arrow " << index;
    EXPECT_NEAR(yawArrow.mat[8], 1.0, 1.0e-6) << "yaw-rate arrow " << index << " points up the world's vertical";
  }
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
