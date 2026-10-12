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

// A web viewer without a simulator, for trying the page in a browser: it serves synthetic frames that respond to the
// page's input like the MuJoCo viewer's camera (a drag shifts the gradient, the wheel zooms it, the keys 0-5 and the
// settings apply). It runs the render loop the MuJoCo viewer runs: wait for demand, apply the input, draw, encode,
// publish the frame and the status.
//
//   bazel run //robot_runtime/web_viewer:web_viewer_demo -- --port=6091
//
// then open http://localhost:6091.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "web_viewer/EncodedFrame.h"
#include "web_viewer/JpegEncoder.h"
#include "web_viewer/ViewerCameraMode.h"
#include "web_viewer/ViewerFrameMailbox.h"
#include "web_viewer/ViewerInput.h"
#include "web_viewer/ViewerInputQueue.h"
#include "web_viewer/ViewerStatus.h"
#include "web_viewer/ViewerThreadPriority.h"
#include "web_viewer/WebViewerServer.h"
#include "web_viewer/WebViewerServerOptions.h"

ABSL_FLAG(std::string, address, "127.0.0.1", "the address to bind");
ABSL_FLAG(int, port, 6091, "the port to bind");
ABSL_FLAG(int, seconds, 0, "stop after this many seconds; 0 serves until the process is killed");

namespace robot::web_viewer {
namespace {

// The demo's stand-in for the MuJoCo scene and camera: what the input changes.
struct DemoScene {
  ViewerStatus status;
  double offsetX = 0.0;
  double offsetY = 0.0;
  double zoom = 1.0;
};

void applyInput(const ViewerInput& input, DemoScene& scene) {
  switch (input.type) {
    case ViewerInputType::kDrag:
      scene.offsetX += input.dx * 255.0;
      scene.offsetY += input.dy * 255.0;
      scene.status.camera.azimuth += input.dx * 180.0;
      scene.status.camera.elevation += input.dy * 90.0;
      return;
    case ViewerInputType::kScroll:
      scene.zoom = std::clamp(scene.zoom * std::exp(0.05 * input.notches), 0.1, 10.0);
      scene.status.camera.distance = 3.0 / scene.zoom;
      return;
    case ViewerInputType::kKey:
      if (input.key >= '0' && input.key <= '5') {
        const size_t group = static_cast<size_t>(input.key - '0');
        scene.status.geomGroups[group] = !scene.status.geomGroups[group];
      }
      if (input.key == 'k') {
        scene.status.camera.mode =
            scene.status.camera.mode == ViewerCameraMode::kTracking ? ViewerCameraMode::kFree : ViewerCameraMode::kTracking;
      }
      for (ViewerStatus::Visualization& visualization : scene.status.visualizations) {
        if (visualization.hotkey == input.key) visualization.enabled = !visualization.enabled;
      }
      return;
    case ViewerInputType::kResetCamera:
      scene.offsetX = 0.0;
      scene.offsetY = 0.0;
      scene.zoom = 1.0;
      scene.status.camera = ViewerStatus::Camera{};
      return;
    case ViewerInputType::kCameraMode:
      scene.status.camera.mode = input.cameraMode;
      return;
    case ViewerInputType::kToggleVisualization:
      for (ViewerStatus::Visualization& visualization : scene.status.visualizations) {
        if (visualization.name == input.visualizationName) visualization.enabled = !visualization.enabled;
      }
      return;
    case ViewerInputType::kSettings:
      if (input.width.has_value()) scene.status.render.width = std::min(*input.width, 1920);
      if (input.height.has_value()) scene.status.render.height = std::min(*input.height, 1080);
      if (input.fps.has_value()) scene.status.render.targetFps = *input.fps;
      if (input.quality.has_value()) scene.status.render.quality = *input.quality;
      return;
  }
  LOG(ERROR) << "unknown input type " << static_cast<int>(input.type);
}

std::vector<uint8_t> drawScene(const DemoScene& scene, int frame) {
  const int width = scene.status.render.width;
  const int height = scene.status.render.height;
  std::vector<uint8_t> rgb(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
  for (int row = 0; row < height; ++row) {
    for (int column = 0; column < width; ++column) {
      const size_t pixel = (static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(column)) * 3;
      const double x = (column - width / 2) / scene.zoom + scene.offsetX;
      const double y = (row - height / 2) / scene.zoom + scene.offsetY;
      const bool checker = (static_cast<int64_t>(std::floor(x / 40.0)) + static_cast<int64_t>(std::floor(y / 40.0))) % 2 == 0;
      rgb[pixel] = static_cast<uint8_t>(static_cast<int64_t>(std::fabs(x) + frame) % 256);
      rgb[pixel + 1] = static_cast<uint8_t>(static_cast<int64_t>(std::fabs(y)) % 256);
      rgb[pixel + 2] = checker && scene.status.geomGroups[0] ? 200 : 60;
    }
  }
  return rgb;
}

int runDemo() {
  ViewerFrameMailbox mailbox(/*maxClients=*/4);
  ViewerInputQueue inputs;
  absl::StatusOr<std::unique_ptr<WebViewerServer>> server =
      WebViewerServer::Create(WebViewerServerOptions{.address = absl::GetFlag(FLAGS_address),
                                                     .port = absl::GetFlag(FLAGS_port),
                                                     .allowedHosts = {},
                                                     .maxClients = 4,
                                                     .visualizationNames = {"contact_timeline", "zmp"},
                                                     .cores = {},
                                                     .threadPriority = ViewerThreadPriority::kInherited},
                              &mailbox, &inputs);
  if (!server.ok()) {
    LOG(ERROR) << server.status();
    return 1;
  }
  absl::StatusOr<std::unique_ptr<JpegEncoder>> encoder = JpegEncoder::Create();
  if (!encoder.ok()) {
    LOG(ERROR) << encoder.status();
    return 1;
  }
  DemoScene scene;
  scene.status.camera.distance = 3.0;
  scene.status.render = {.width = 1280, .height = 720, .targetFps = 30, .measuredFps = 0.0, .quality = 70};
  scene.status.geomGroups = {true, true, false, false, false, false};
  scene.status.visualizations = {
      {.name = "contact_timeline", .hotkey = 'c', .enabled = true, .description = "the feet's contacts over time"},
      {.name = "zmp", .hotkey = 'z', .enabled = false, .description = "the zero-moment point"}};
  LOG(INFO) << "open " << (*server)->url();
  const absl::Time end =
      absl::GetFlag(FLAGS_seconds) > 0 ? absl::Now() + absl::Seconds(absl::GetFlag(FLAGS_seconds)) : absl::InfiniteFuture();
  std::vector<absl::Time> recentFrames;
  for (int frame = 0; absl::Now() < end;) {
    if (!mailbox.waitForDemand(absl::Milliseconds(100))) continue;
    const absl::Time start = absl::Now();
    for (const ViewerInput& input : inputs.drain()) applyInput(input, scene);
    absl::StatusOr<std::string> jpeg = (*encoder)->encode(drawScene(scene, frame), scene.status.render.width, scene.status.render.height,
                                                          scene.status.render.quality, RowOrder::kTopDown);
    if (!jpeg.ok()) {
      LOG(ERROR) << jpeg.status();
      return 1;
    }
    mailbox.publish(EncodedFrame{.sequence = 0,
                                 .width = scene.status.render.width,
                                 .height = scene.status.render.height,
                                 .simTime = 0.033 * frame,
                                 .publishTime = absl::UnixEpoch(),
                                 .jpeg = *std::move(jpeg)});
    recentFrames.push_back(start);
    recentFrames.erase(recentFrames.begin(), std::find_if(recentFrames.begin(), recentFrames.end(),
                                                          [&start](absl::Time time) { return start - time < absl::Seconds(1); }));
    scene.status.render.measuredFps = static_cast<double>(recentFrames.size());
    mailbox.publishStatus(scene.status);
    ++frame;
    absl::SleepFor(start + absl::Seconds(1) / std::max(scene.status.render.targetFps, 1) - absl::Now());
  }
  mailbox.close();
  return 0;
}

}  // namespace
}  // namespace robot::web_viewer

int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  return robot::web_viewer::runDemo();
}
