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
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"
#include "robot_runtime/mujoco_sim_interface/test/ViewerTestSupport.h"

/*
 * The web viewer's offscreen rendering, without its server: an EGL context (llvmpipe in the dev container), a private
 * model copy and the MujocoFrameRenderer the GLFW window shares, on the hinge scene and on the shipped Atlas scene. A
 * frame has the size asked for, shows the robot, is the same for the same state, and follows the camera and the
 * visualizations.
 */

namespace robot::mujoco_sim_interface {
namespace {

std::unique_ptr<MujocoSimInterface> makeSim(const MujocoSimConfig& config, const std::string& urdfPath) {
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim = MujocoSimInterface::Create(config, urdfPath);
  EXPECT_TRUE(sim.ok()) << sim.status();
  return sim.ok() ? *std::move(sim) : nullptr;
}

std::unique_ptr<OffscreenTestRenderer> makeRenderer(const MujocoSimInterface* absl_nonnull sim, const WebViewerOptions& options) {
  absl::StatusOr<std::unique_ptr<OffscreenTestRenderer>> renderer = OffscreenTestRenderer::Create(sim, options);
  EXPECT_TRUE(renderer.ok()) << renderer.status();
  return renderer.ok() ? *std::move(renderer) : nullptr;
}

/** Atlas, stepped long enough for the contact timeline to have samples, with markers that draw no time-varying text. */
std::unique_ptr<MujocoSimInterface> makeSteppedAtlas() {
  std::unique_ptr<MujocoSimInterface> sim = makeSim(
      atlasSimConfig({kNoViewerName}, {"contact_forces", "base_velocity", "contact_timeline", "target_contact_patches"}), kAtlasUrdfPath);
  if (sim == nullptr) return nullptr;
  sim->initSim();
  for (int step = 0; step < 400; ++step) sim->simulationStep();
  return sim;
}

/** The share of red pixels: the hinge robot's base is a red box. */
double redPixelFraction(const std::vector<uint8_t>& rgb) {
  int64_t redPixels = 0;
  for (size_t pixel = 0; pixel + 2 < rgb.size(); pixel += 3) {
    const int red = rgb[pixel];
    if (red > 60 && red > rgb[pixel + 1] + 40 && red > rgb[pixel + 2] + 40) ++redPixels;
  }
  return static_cast<double>(redPixels) / static_cast<double>(rgb.size() / 3);
}

/** Rows [firstRow, lastRow) of a bottom-up RGB image of `width` pixels: row 0 is the bottom of the picture. */
std::vector<uint8_t> rows(const std::vector<uint8_t>& rgb, int width, int firstRow, int lastRow) {
  const size_t rowBytes = static_cast<size_t>(width) * 3;
  return std::vector<uint8_t>(rgb.begin() + static_cast<ptrdiff_t>(firstRow * rowBytes),
                              rgb.begin() + static_cast<ptrdiff_t>(lastRow * rowBytes));
}

TEST(OffscreenRender, AFrameHasTheSizeAskedForAlsoAfterAResize) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(hingeSimConfig({kNoViewerName}), hingeUrdfPath());
  ASSERT_NE(sim, nullptr);
  const std::unique_ptr<OffscreenTestRenderer> renderer = makeRenderer(sim.get(), WebViewerOptions{});
  ASSERT_NE(renderer, nullptr);
  struct Size {
    int width = 0;
    int height = 0;
  };
  for (const Size& size : std::vector<Size>{
           {.width = 640, .height = 480}, {.width = 320, .height = 200}, {.width = 1920, .height = 1080}, {.width = 160, .height = 120}}) {
    const absl::StatusOr<std::vector<uint8_t>> frame = renderer->render(size.width, size.height);
    ASSERT_TRUE(frame.ok()) << frame.status();
    EXPECT_EQ(frame->size(), static_cast<size_t>(size.width) * size.height * 3) << size.width << "x" << size.height;
  }
}

TEST(OffscreenRender, AFrameShowsTheRobot) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(hingeSimConfig({kNoViewerName}), hingeUrdfPath());
  ASSERT_NE(sim, nullptr);
  const std::unique_ptr<OffscreenTestRenderer> renderer = makeRenderer(sim.get(), WebViewerOptions{});
  ASSERT_NE(renderer, nullptr);
  const absl::StatusOr<std::vector<uint8_t>> frame = renderer->render(/*width=*/640, /*height=*/480);
  ASSERT_TRUE(frame.ok()) << frame.status();
  EXPECT_GT(byteVariance(*frame), 100.0) << "a uniform frame";
  EXPECT_GT(redPixelFraction(*frame), 0.001) << "the red base of the robot is not in the picture";
  if (HasFailure()) writeTestImage("hinge_robot", *frame, /*width=*/640, /*height=*/480);
}

TEST(OffscreenRender, TwoFramesOfOneStateAreByteIdentical) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSteppedAtlas();
  ASSERT_NE(sim, nullptr);
  const std::unique_ptr<OffscreenTestRenderer> renderer = makeRenderer(sim.get(), WebViewerOptions{});
  ASSERT_NE(renderer, nullptr);
  // The free camera stays where it is; the tracking one would ease towards the robot from frame to frame.
  renderer->frameRenderer().setCameraType(mjCAMERA_FREE);
  const absl::StatusOr<std::vector<uint8_t>> first = renderer->render(/*width=*/640, /*height=*/480);
  const absl::StatusOr<std::vector<uint8_t>> second = renderer->render(/*width=*/640, /*height=*/480);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_GT(byteVariance(*first), 100.0);
  EXPECT_EQ(*first, *second) << differingPixelFraction(*first, *second) * 100.0 << "% of the pixels differ";
}

TEST(OffscreenRender, MovingTheCameraChangesTheFrame) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(hingeSimConfig({kNoViewerName}), hingeUrdfPath());
  ASSERT_NE(sim, nullptr);
  const std::unique_ptr<OffscreenTestRenderer> renderer = makeRenderer(sim.get(), WebViewerOptions{});
  ASSERT_NE(renderer, nullptr);
  renderer->frameRenderer().setCameraType(mjCAMERA_FREE);
  const absl::StatusOr<std::vector<uint8_t>> before = renderer->render(/*width=*/640, /*height=*/480);
  ASSERT_TRUE(before.ok()) << before.status();
  const double azimuth = renderer->frameRenderer().camera().azimuth;
  renderer->frameRenderer().moveCamera(mjMOUSE_ROTATE_V, /*relativeDx=*/0.2, /*relativeDy=*/0.0);
  EXPECT_NE(renderer->frameRenderer().camera().azimuth, azimuth);
  const absl::StatusOr<std::vector<uint8_t>> after = renderer->render(/*width=*/640, /*height=*/480);
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_GT(differingPixelFraction(*before, *after), 0.01) << "the camera turned and the picture did not";
}

// The wheel's sign, which the web route and the GLFW window share: positive notches (the wheel turned away from the
// user) zoom out, as in MuJoCo's own viewers. The route's documentation used to say the opposite.
TEST(OffscreenRender, PositiveScrollNotchesZoomOut) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(hingeSimConfig({kNoViewerName}), hingeUrdfPath());
  ASSERT_NE(sim, nullptr);
  const std::unique_ptr<OffscreenTestRenderer> renderer = makeRenderer(sim.get(), WebViewerOptions{});
  ASSERT_NE(renderer, nullptr);
  ASSERT_TRUE(renderer->render(/*width=*/320, /*height=*/240).ok());
  renderer->frameRenderer().setCameraType(mjCAMERA_FREE);
  const double distance = renderer->frameRenderer().camera().distance;
  renderer->frameRenderer().scroll(/*notches=*/1.0);
  EXPECT_GT(renderer->frameRenderer().camera().distance, distance);
  renderer->frameRenderer().scroll(/*notches=*/-2.0);
  EXPECT_LT(renderer->frameRenderer().camera().distance, distance);
}

TEST(OffscreenRender, TogglingTheContactTimelineChangesOnlyTheBottomBand) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSteppedAtlas();
  ASSERT_NE(sim, nullptr);
  const std::unique_ptr<OffscreenTestRenderer> renderer = makeRenderer(sim.get(), WebViewerOptions{});
  ASSERT_NE(renderer, nullptr);
  renderer->frameRenderer().setCameraType(mjCAMERA_FREE);
  constexpr int kWidth = 800;
  constexpr int kHeight = 600;
  const absl::StatusOr<std::vector<uint8_t>> with = renderer->render(kWidth, kHeight);
  ASSERT_TRUE(with.ok()) << with.status();
  ASSERT_TRUE(renderer->frameRenderer().toggleVisualization("contact_timeline"));
  const absl::StatusOr<std::vector<uint8_t>> without = renderer->render(kWidth, kHeight);
  ASSERT_TRUE(without.ok()) << without.status();
  EXPECT_GT(differingPixelFraction(rows(*with, kWidth, /*firstRow=*/0, kHeight / 4), rows(*without, kWidth, /*firstRow=*/0, kHeight / 4)),
            0.05)
      << "the timeline panel along the bottom edge did not go";
  EXPECT_EQ(differingPixelFraction(rows(*with, kWidth, kHeight * 3 / 4, kHeight), rows(*without, kWidth, kHeight * 3 / 4, kHeight)), 0.0)
      << "the top of the picture changed with the timeline";
  EXPECT_FALSE(renderer->frameRenderer().toggleVisualization("no_such_visualization"));
}

/** True when the arrays of two models of one layout (one a copy of the other) hold the same bytes. */
bool sameModelArrays(const mjModel& first, const mjModel& second) {
  return first.nbuffer == second.nbuffer && std::memcmp(first.buffer, second.buffer, static_cast<size_t>(first.nbuffer)) == 0;
}

TEST(OffscreenRender, ThePrivateModelFollowsWhatTheSimulatorChangesWhileItRuns) {
  // The renderer's model is a copy taken before the physics starts. Active control lowers the ragdoll damping, the
  // gantry weld follows the gantry's height and a throw arms the ball with the operator's mass, all in the simulator's
  // model; each frame writes them into the copy, which then has the simulator's model arrays byte for byte.
  MujocoSimConfig config = atlasSimConfig({kNoViewerName}, {"contact_forces"});
  config.projectile = "dodgeball";
  config.isGantryLocked = true;
  config.renderFrequencyHz = 1.0e6;  // a snapshot after every step
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(config, kAtlasUrdfPath);
  ASSERT_NE(sim, nullptr);
  ASSERT_GE(sim->projectileBodyId(), 0);
  sim->initSim();
  const std::unique_ptr<OffscreenTestRenderer> renderer = makeRenderer(sim.get(), WebViewerOptions{});
  ASSERT_NE(renderer, nullptr);
  ASSERT_TRUE(sameModelArrays(renderer->model(), *sim->getModel()));

  sim->enableTorques();
  sim->setGantryHeight(sim->getGantryHeight() + 0.05);
  MujocoSimInterface::DodgeballThrow command;
  command.spawnOffset[0] = 3.0;
  command.spawnOffset[2] = 0.5;
  command.flightTime = 0.4;
  command.launchVelocity[0] = -command.spawnOffset[0] / command.flightTime;
  command.launchVelocity[2] = -command.spawnOffset[2] / command.flightTime + 0.5 * 9.81 * command.flightTime;
  command.mass = 2.0;
  sim->throwDodgeball(command);
  for (int step = 0; step < 50; ++step) sim->simulationStep();
  ASSERT_FALSE(sameModelArrays(renderer->model(), *sim->getModel())) << "the simulator's model did not change";

  const absl::StatusOr<std::vector<uint8_t>> frame = renderer->render(/*width=*/320, /*height=*/240);
  ASSERT_TRUE(frame.ok()) << frame.status();
  EXPECT_TRUE(sameModelArrays(renderer->model(), *sim->getModel()))
      << "a model array the simulator writes while it runs is missing from MjRuntimeModelFields";
}

TEST(OffscreenRender, RendersWithAndWithoutMultisampling) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(hingeSimConfig({kNoViewerName}), hingeUrdfPath());
  ASSERT_NE(sim, nullptr);
  for (const int samples : {0, 4}) {
    WebViewerOptions options;
    options.msaaSamples = samples;
    const std::unique_ptr<OffscreenTestRenderer> renderer = makeRenderer(sim.get(), options);
    ASSERT_NE(renderer, nullptr);
    const absl::StatusOr<std::vector<uint8_t>> frame = renderer->render(/*width=*/640, /*height=*/480);
    ASSERT_TRUE(frame.ok()) << frame.status();
    EXPECT_GT(redPixelFraction(*frame), 0.001) << samples << " samples";
  }
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
