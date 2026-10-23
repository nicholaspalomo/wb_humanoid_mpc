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

#include "robot_runtime/mujoco_sim_interface/test/ViewerTestSupport.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/viewer/EglContext.h"
#include "mujoco_sim_interface/viewer/EglLibrary.h"
#include "mujoco_sim_interface/viewer/MujocoFrameRenderer.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"

namespace robot::mujoco_sim_interface {
namespace {

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

/** The hinge robot in MuJoCo, colored so that every part of it stands out from the floor and the sky. */
constexpr char kHingeScene[] = R"(
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1" rgba="0.9 0.2 0.2 1"/>
      <body name="arm" pos="0 0 -0.2">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.1" rgba="0.2 0.9 0.2 1"/>
        <body name="hand" pos="0.3 0 0">
          <joint name="wrist" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.03" mass="0.05" rgba="0.2 0.2 0.9 1"/>
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

}  // namespace

std::string writeTestFile(absl::string_view name, absl::string_view content) {
  const std::string path = absl::StrCat(testing::TempDir(), "/", name);
  std::ofstream(path) << content;
  return path;
}

std::string hingeUrdfPath() {
  return writeTestFile("viewer_hinge_robot.urdf", kHingeUrdf);
}

MujocoSimConfig hingeSimConfig(std::vector<std::string> viewerNames) {
  MujocoSimConfig config;
  config.scenePath = writeTestFile("viewer_hinge_scene.xml", kHingeScene);
  config.viewer.names = std::move(viewerNames);
  config.enableGantry = false;
  config.gantryHold = "kinematic_teleport";
  config.visualizations = {};
  return config;
}

MujocoSimConfig atlasSimConfig(std::vector<std::string> viewerNames, std::vector<std::string> visualizations) {
  MujocoSimConfig config;
  config.scenePath = kAtlasScenePath;
  config.viewer.names = std::move(viewerNames);
  config.gantryHold = "weld_constraint";
  config.contactFrameNames = {"l_foot", "r_foot"};
  config.visualizations = std::move(visualizations);
  return config;
}

std::vector<int> processThreadIds() {
  std::vector<int> ids;
  std::error_code error;
  for (std::filesystem::directory_iterator entry("/proc/self/task", error); !error && entry != std::filesystem::directory_iterator();
       entry.increment(error)) {
    int id = 0;
    if (absl::SimpleAtoi(entry->path().filename().string(), &id)) ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

double differingPixelFraction(absl::Span<const uint8_t> first, absl::Span<const uint8_t> second) {
  if (first.size() != second.size() || first.empty()) return 1.0;
  int64_t differing = 0;
  for (size_t pixel = 0; pixel + 2 < first.size(); pixel += 3) {
    if (first[pixel] != second[pixel] || first[pixel + 1] != second[pixel + 1] || first[pixel + 2] != second[pixel + 2]) ++differing;
  }
  return static_cast<double>(differing) / static_cast<double>(first.size() / 3);
}

double byteVariance(absl::Span<const uint8_t> rgb) {
  if (rgb.empty()) return 0.0;
  double sum = 0.0;
  double sumOfSquares = 0.0;
  for (const uint8_t value : rgb) {
    sum += value;
    sumOfSquares += static_cast<double>(value) * value;
  }
  const double mean = sum / static_cast<double>(rgb.size());
  return sumOfSquares / static_cast<double>(rgb.size()) - mean * mean;
}

void writeTestImage(absl::string_view name, absl::Span<const uint8_t> rgb, int width, int height) {
  const char* absl_nullable directory = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (directory == nullptr || width <= 0 || height <= 0 || rgb.size() != static_cast<size_t>(width) * height * 3) return;
  std::ofstream file(absl::StrCat(directory, "/", name, ".ppm"), std::ios::binary);
  file << absl::StrCat("P6\n", width, " ", height, "\n255\n");
  const size_t rowBytes = static_cast<size_t>(width) * 3;
  for (int row = height - 1; row >= 0; --row) {
    file.write(reinterpret_cast<const char*>(rgb.data() + static_cast<size_t>(row) * rowBytes), static_cast<std::streamsize>(rowBytes));
  }
}

absl::StatusOr<std::unique_ptr<OffscreenTestRenderer>> OffscreenTestRenderer::Create(const MujocoSimInterface* absl_nonnull sim,
                                                                                     const WebViewerOptions& options) {
  absl::StatusOr<const EglLibrary* absl_nonnull> library = EglLibrary::Get(options.eglLibrary);
  if (!library.ok()) return library.status();
  absl::StatusOr<std::unique_ptr<EglContext>> context = EglContext::Create(*library, EglContext::Options{.deviceId = options.eglDeviceId});
  if (!context.ok()) return context.status();
  if (absl::Status current = (*context)->makeCurrent(); !current.ok()) {
    return current;
  }
  absl::StatusOr<MjModelPtr> model = copyModelForOffscreenRendering(sim->getModel(), options);
  if (!model.ok()) return model.status();
  std::unique_ptr<OffscreenTestRenderer> renderer = absl::WrapUnique(new OffscreenTestRenderer(*std::move(context), *std::move(model)));
  absl::StatusOr<std::unique_ptr<MujocoFrameRenderer>> frameRenderer = MujocoFrameRenderer::Create(sim, renderer->model_.get());
  if (!frameRenderer.ok()) return frameRenderer.status();
  renderer->frameRenderer_ = *std::move(frameRenderer);
  renderer->frameRenderer_->useOffscreenBuffer();
  return renderer;
}

OffscreenTestRenderer::OffscreenTestRenderer(std::unique_ptr<EglContext> context, MjModelPtr model)
    : context_(std::move(context)), model_(std::move(model)) {}

OffscreenTestRenderer::~OffscreenTestRenderer() {
  // MuJoCo frees its GPU objects in whichever context is current, which another renderer may have made its own.
  if (const absl::Status current = context_->makeCurrent(); !current.ok()) {
    ADD_FAILURE() << "the renderer's EGL context could not be made current to free it: " << current;
    return;
  }
  if (frameRenderer_ != nullptr) frameRenderer_->release();
  EXPECT_TRUE(context_->release().ok());
}

absl::StatusOr<std::vector<uint8_t>> OffscreenTestRenderer::render(int width, int height) {
  if (absl::Status current = context_->makeCurrent(); !current.ok()) {
    return current;
  }
  frameRenderer_->resizeOffscreen(width, height);
  const mjrRect viewport = {.left = 0, .bottom = 0, .width = width, .height = height};
  frameRenderer_->drawFrame(viewport, /*elapsedRealTime=*/0.0);
  std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
  if (absl::Status read = frameRenderer_->readPixels(viewport, absl::MakeSpan(rgb)); !read.ok()) {
    return read;
  }
  return rgb;
}

}  // namespace robot::mujoco_sim_interface
