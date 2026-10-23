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

#include "mujoco_sim_interface/viewer/WebMujocoViewer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mujoco/mujoco.h"
#include "ocs2_core/Dimensions.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/viewer/EglContext.h"
#include "mujoco_sim_interface/viewer/EglLibrary.h"
#include "mujoco_sim_interface/viewer/GlCapabilities.h"
#include "mujoco_sim_interface/viewer/MujocoFrameRenderer.h"
#include "mujoco_sim_interface/viewer/ViewerSettings.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"
#include "mujoco_sim_interface/visualization/VisualizationRegistry.h"
#include "web_viewer/EncodedFrame.h"
#include "web_viewer/JpegEncoder.h"
#include "web_viewer/ThreadPlacement.h"
#include "web_viewer/ViewerCameraMode.h"
#include "web_viewer/ViewerInput.h"
#include "web_viewer/ViewerStatus.h"
#include "web_viewer/WebViewerServer.h"
#include "web_viewer/WebViewerServerOptions.h"

namespace robot::mujoco_sim_interface {

using ::ocs2::kXAxisIndex;
using ::ocs2::kYAxisIndex;
using ::ocs2::kZAxisIndex;
namespace {

/// How long the render loop waits for a browser before it looks at the input queue again.
constexpr absl::Duration kIdlePoll = absl::Milliseconds(100);
/// How often start() logs that it is still waiting for the render thread.
constexpr absl::Duration kStartProgressPeriod = absl::Seconds(2);
/// How often initializeRendering()'s test delay looks for a stop request, [ms].
constexpr int kInitDelayPollMilliseconds = 10;

/** MuJoCo's camera motion for a drag with `button`, as the GLFW viewer maps them: left rotates, right moves, middle zooms. */
mjtMouse mouseAction(web_viewer::ViewerMouseButton button, bool shift) {
  switch (button) {
    case web_viewer::ViewerMouseButton::kLeft:
      return shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    case web_viewer::ViewerMouseButton::kRight:
      return shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    case web_viewer::ViewerMouseButton::kMiddle:
      return mjMOUSE_ZOOM;
  }
  return mjMOUSE_NONE;  // not a button of ViewerMouseButton
}

web_viewer::WebViewerServerOptions serverOptions(const WebViewerOptions& options,
                                                 const std::vector<web_viewer::ViewerStatus::Visualization>& visualizations) {
  web_viewer::WebViewerServerOptions server;
  server.address = options.address;
  server.port = options.port;
  server.allowedHosts = options.allowedHosts;
  server.maxClients = options.maxClients;
  for (const web_viewer::ViewerStatus::Visualization& visualization : visualizations) {
    server.visualizationNames.push_back(visualization.name);
  }
  server.cores = options.cores;
  server.threadPriority = options.threadPriority;
  return server;
}

// The monotonic time since `begin`, which no step of the wall clock moves.
absl::Duration steadySince(std::chrono::steady_clock::time_point begin) {
  return absl::FromChrono(std::chrono::steady_clock::now() - begin);
}

/** The visualizations `sim`'s configuration lists, as the page shows them before the render thread is set up. */
std::vector<web_viewer::ViewerStatus::Visualization> describeVisualizations(const MujocoSimInterface* absl_nonnull sim) {
  std::vector<web_viewer::ViewerStatus::Visualization> described;
  for (const std::unique_ptr<MujocoVisualization>& visualization :
       createVisualizations(sim->getConfig().visualizations, /*errors=*/nullptr)) {
    described.push_back({.name = visualization->name(),
                         .hotkey = visualization->hotkey(),
                         .enabled = visualization->enabled(),
                         .description = visualization->description()});
  }
  return described;
}

absl::Status stoppedDuringSetUp() {
  return absl::CancelledError("the web viewer was stopped during its set-up");  // NOLINT(american-spelling): Abseil's code name
}

}  // namespace

absl::StatusOr<std::unique_ptr<WebMujocoViewer>> WebMujocoViewer::Create(const MujocoSimInterface* absl_nonnull sim,
                                                                         const WebViewerOptions& options) {
  absl::StatusOr<const EglLibrary* absl_nonnull> egl = EglLibrary::Get(options.eglLibrary);
  if (!egl.ok()) return egl.status();
  // Here, on the simulator's thread before its physics starts: the render thread never reads the simulator's model.
  absl::StatusOr<MjModelPtr> model = copyModelForOffscreenRendering(sim->getModel(), options);
  if (!model.ok()) return model.status();
  std::unique_ptr<WebMujocoViewer> viewer = absl::WrapUnique(new WebMujocoViewer(sim, options, *std::move(model), *egl));
  absl::StatusOr<std::unique_ptr<web_viewer::WebViewerServer>> server =
      web_viewer::WebViewerServer::Create(serverOptions(options, viewer->configuredVisualizations_), &viewer->mailbox_, &viewer->inputs_);
  if (!server.ok()) return server.status();
  viewer->server_ = *std::move(server);
  // A page that opens during the render thread's set-up (up to startTimeout) sees the configured size, rate and markers.
  viewer->mailbox_.publishStatus(viewer->setUpStatus());
  if (std::getenv("LP_NUM_THREADS") == nullptr) {
    LOG(WARNING) << "[WebMujocoViewer] LP_NUM_THREADS is unset, so llvmpipe (software OpenGL) starts a rasterizer thread per CPU; set "
                    "LP_NUM_THREADS="
                 << softwareRasterizerThreads(options) << " (the viewer's cores) to bound them.";
  }
  return viewer;
}

WebMujocoViewer::WebMujocoViewer(const MujocoSimInterface* absl_nonnull sim,
                                 const WebViewerOptions& options,
                                 MjModelPtr model,
                                 const EglLibrary* absl_nonnull egl)
    : sim_(sim),
      options_(options),
      model_(std::move(model)),
      egl_(egl),
      mailbox_(options.maxClients),
      configuredVisualizations_(describeVisualizations(sim)),
      width_(options.width),
      height_(options.height),
      fps_(options.fps),
      quality_(options.quality) {}

WebMujocoViewer::~WebMujocoViewer() {
  requestStop();
  mailbox_.close();  // waiting requests and the render loop's wait return at once
  server_.reset();   // stops accepting and joins its threads
  if (renderThread_.joinable()) renderThread_.join();
}

absl::Status WebMujocoViewer::start() {
  if (renderThread_.joinable() || initialized_.HasBeenNotified()) {
    return absl::FailedPreconditionError("WebMujocoViewer::start() was called before; it starts once");
  }
  renderThread_ = std::thread(&WebMujocoViewer::renderLoop, this);
  // Relative waits on the monotonic clock: an absolute deadline would wait on the wall clock.
  const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  while (!initialized_.WaitForNotificationWithTimeout(std::min(kStartProgressPeriod, options_.startTimeout - steadySince(begin)))) {
    if (steadySince(begin) >= options_.startTimeout) {
      stopAfterStartUpTimeout();
      return absl::DeadlineExceededError(absl::StrCat("the web viewer's render thread did not set up EGL and OpenGL within ",
                                                      absl::FormatDuration(options_.startTimeout)));
    }
    LOG(INFO) << "[WebMujocoViewer] waiting for the render thread to set up EGL and OpenGL (" << absl::FormatDuration(steadySince(begin))
              << ")";
  }
  absl::Status status;
  std::string description;
  {
    absl::MutexLock lock(initMutex_);
    status = initStatus_;
    description = renderingDescription_;
  }
  if (!status.ok()) {
    renderThread_.join();
    return status;
  }
  LOG(INFO) << "[WebMujocoViewer] the MuJoCo viewer is at " << url() << " (" << description << ")";
  return absl::OkStatus();
}

void WebMujocoViewer::stopAfterStartUpTimeout() {
  // The render thread looks for the stop between the steps of its set-up, so the step running now (an EGL, GL or
  // MuJoCo call) and the release of what the set-up made end first. The thread holds `this`: it is joined, never left.
  requestStop();
  LOG(INFO) << "[WebMujocoViewer] stopping the render thread after the start-up timeout (" << absl::FormatDuration(options_.startTimeout)
            << "); it ends once its current set-up step does";
  const absl::Time begin = absl::Now();
  while (!renderLoopEnded_.WaitForNotificationWithTimeout(kStartProgressPeriod)) {
    LOG(INFO) << "[WebMujocoViewer] still waiting for the render thread's set-up step to end (" << absl::FormatDuration(absl::Now() - begin)
              << ")";
  }
  renderThread_.join();
}

void WebMujocoViewer::requestStop() {
  stopRequested_.store(true, std::memory_order_release);
  if (!stopNotification_.HasBeenNotified()) stopNotification_.Notify();
}

bool WebMujocoViewer::ok() const {
  return !stopped_.load(std::memory_order_acquire);
}

std::optional<int> WebMujocoViewer::port() const {
  return server_ != nullptr ? std::optional<int>(server_->port()) : std::nullopt;
}

std::string WebMujocoViewer::url() const {
  return server_ != nullptr ? server_->url() : std::string();
}

void WebMujocoViewer::renderLoop() {
  // First of all: llvmpipe starts its rasterizer threads while EGL sets up the display and the context, and they
  // inherit this thread's cores and scheduling.
  if (const absl::Status placed = web_viewer::placeCurrentThread(options_.cores, options_.threadPriority); !placed.ok()) {
    LOG(WARNING) << "[WebMujocoViewer] the render thread runs unplaced: " << placed;
  }
  absl::Status initialized = initializeRendering();
  if (!initialized.ok()) releaseRendering();
  {
    absl::MutexLock lock(initMutex_);
    initStatus_ = initialized;
  }
  initialized_.Notify();
  if (!initialized.ok()) {
    stopped_.store(true, std::memory_order_release);
    renderLoopEnded_.Notify();
    return;
  }

  // Paced on the monotonic clock, so that a step of the wall clock neither stretches a pause nor freezes the frames.
  const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  while (!stopRequested() && !mailbox_.closed()) {
    const bool demand = mailbox_.waitForDemand(kIdlePoll);
    if (stopRequested() || mailbox_.closed()) break;
    const std::chrono::steady_clock::time_point frameStart = std::chrono::steady_clock::now();
    const bool hadInput = applyInputs();
    if (!demand) {
      // Nobody watches, so nothing is drawn; /state still follows the input.
      if (hadInput) mailbox_.publishStatus(currentStatus());
      continue;
    }
    if (const absl::Status rendered = renderAndPublish(absl::ToDoubleSeconds(absl::FromChrono(frameStart - begin))); !rendered.ok()) {
      stopServingAfterRenderFailure(rendered);
      break;
    }
    // No catch-up bursts: a late frame only delays the next one. The pause ends at once on a stop.
    stopNotification_.WaitForNotificationWithTimeout(absl::Seconds(1) / fps_ - steadySince(frameStart));
  }
  releaseRendering();
  stopped_.store(true, std::memory_order_release);
  renderLoopEnded_.Notify();
}

absl::Status WebMujocoViewer::initializeRendering() {
  // The test hook of WebMujocoViewerTestPeer, in steps so that a stop request is seen.
  for (const absl::Time until = absl::Now() + initDelayForTesting_; absl::Now() < until && !stopRequested();) {
    absl::SleepFor(absl::Milliseconds(kInitDelayPollMilliseconds));
  }
  absl::SleepFor(uninterruptibleInitDelayForTesting_);
  if (stopRequested()) return stoppedDuringSetUp();
  absl::StatusOr<std::unique_ptr<EglContext>> context = EglContext::Create(egl_, EglContext::Options{.deviceId = options_.eglDeviceId});
  if (!context.ok()) return context.status();
  context_ = *std::move(context);
  if (absl::Status current = context_->makeCurrent(); !current.ok()) {
    return current;
  }
  const absl::StatusOr<GlCapabilities> capabilities = context_->queryGlCapabilities();
  if (!capabilities.ok()) return capabilities.status();
  // Everything MuJoCo would end the process for while it creates or resizes its offscreen buffer or its shadow map.
  if (absl::Status capable = checkGlCapabilities(*capabilities, options_); !capable.ok()) {
    return capable;
  }
  if (stopRequested()) return stoppedDuringSetUp();
  {
    absl::MutexLock lock(initMutex_);
    renderingDescription_ = absl::StrCat(capabilities->renderer, ", OpenGL ", capabilities->version, ", on ", context_->description());
  }

  absl::StatusOr<std::unique_ptr<MujocoFrameRenderer>> frameRenderer = MujocoFrameRenderer::Create(sim_, model_.get());
  if (!frameRenderer.ok()) return frameRenderer.status();
  frameRenderer_ = *std::move(frameRenderer);
  // A context without a window has no window framebuffer: every frame is drawn into the offscreen one.
  frameRenderer_->useOffscreenBuffer();
  frameRenderer_->resizeOffscreen(width_, height_);
  rgb_.assign(static_cast<size_t>(width_) * height_ * web_viewer::kRgbBytesPerPixel, 0);
  absl::StatusOr<std::unique_ptr<web_viewer::JpegEncoder>> encoder = web_viewer::JpegEncoder::Create();
  if (!encoder.ok()) return encoder.status();
  encoder_ = *std::move(encoder);
  mailbox_.publishStatus(currentStatus());
  return absl::OkStatus();
}

void WebMujocoViewer::stopServingAfterRenderFailure(const absl::Status& failure) {
  LOG(ERROR) << "[WebMujocoViewer] the viewer stops: " << failure;
  web_viewer::ViewerStatus status = currentStatus();
  status.render.measuredFps = 0.0;
  mailbox_.publishStatus(std::move(status));
  // Every waiting and later frame request answers 503 'the viewer has stopped', which the page shows, instead of a 204
  // every 2 s from a viewer that will never draw again.
  mailbox_.close();
}

void WebMujocoViewer::releaseRendering() {
  if (context_ == nullptr) return;
  if (frameRenderer_ != nullptr) frameRenderer_->release();
  frameRenderer_.reset();
  if (const absl::Status released = context_->release(); !released.ok()) {
    LOG(WARNING) << "[WebMujocoViewer] " << released;
  }
  context_.reset();
  encoder_.reset();
}

bool WebMujocoViewer::applyInputs() {
  const std::vector<web_viewer::ViewerInput> inputs = inputs_.drain();
  for (const web_viewer::ViewerInput& input : inputs) applyInput(input);
  return !inputs.empty();
}

void WebMujocoViewer::applyInput(const web_viewer::ViewerInput& input) {
  if (frameRenderer_ == nullptr) return;  // inputs are applied once the render thread is set up
  switch (input.type) {
    case web_viewer::ViewerInputType::kDrag:
      frameRenderer_->moveCamera(mouseAction(input.button, input.shift), input.dx, input.dy);
      return;
    case web_viewer::ViewerInputType::kScroll:
      frameRenderer_->scroll(input.notches);
      return;
    case web_viewer::ViewerInputType::kKey:
      frameRenderer_->handleKey(input.key);
      return;
    case web_viewer::ViewerInputType::kResetCamera:
      frameRenderer_->resetCamera();
      return;
    case web_viewer::ViewerInputType::kCameraMode:
      frameRenderer_->setCameraType(input.cameraMode == web_viewer::ViewerCameraMode::kTracking ? mjCAMERA_TRACKING : mjCAMERA_FREE);
      return;
    case web_viewer::ViewerInputType::kToggleVisualization:
      frameRenderer_->toggleVisualization(input.visualizationName);
      return;
    case web_viewer::ViewerInputType::kSettings:
      applySettings(input);
      return;
  }
  LOG(ERROR) << "[WebMujocoViewer] ignored an input of unknown type " << static_cast<int>(input.type);
}

void WebMujocoViewer::applySettings(const web_viewer::ViewerInput& settings) {
  // Within the offscreen buffer the GL preflight checked, so that no resize can reach mju_error.
  const int width = std::clamp(settings.width.value_or(width_), kMinWebViewerWidth, options_.maxWidth);
  const int height = std::clamp(settings.height.value_or(height_), kMinWebViewerHeight, options_.maxHeight);
  fps_ = std::clamp(settings.fps.value_or(fps_), kMinWebViewerFps, kMaxWebViewerFps);
  quality_ = std::clamp(settings.quality.value_or(quality_), kMinWebViewerQuality, kMaxWebViewerQuality);
  if (width == width_ && height == height_) return;
  width_ = width;
  height_ = height;
  frameRenderer_->resizeOffscreen(width_, height_);
  rgb_.assign(static_cast<size_t>(width_) * height_ * web_viewer::kRgbBytesPerPixel, 0);
}

absl::Status WebMujocoViewer::renderAndPublish(double elapsedRealTime) {
  if (failNextFrameForTesting_.exchange(false)) return absl::InternalError("the test failed this frame (WebMujocoViewerTestPeer)");
  const mjrRect viewport = {.left = 0, .bottom = 0, .width = width_, .height = height_};
  frameRenderer_->drawFrame(viewport, elapsedRealTime);
  if (absl::Status read = frameRenderer_->readPixels(viewport, absl::MakeSpan(rgb_)); !read.ok()) {
    return read;
  }
  absl::StatusOr<std::string> jpeg = encoder_->encode(rgb_, width_, height_, quality_, web_viewer::RowOrder::kBottomUp);
  if (!jpeg.ok()) return jpeg.status();
  web_viewer::EncodedFrame frame;
  frame.width = width_;
  frame.height = height_;
  frame.simTime = frameRenderer_->snapshotTime();
  frame.jpeg = *std::move(jpeg);
  mailbox_.publish(std::move(frame));
  mailbox_.publishStatus(currentStatus());
  return absl::OkStatus();
}

web_viewer::ViewerStatus WebMujocoViewer::setUpStatus() const {
  web_viewer::ViewerStatus status;
  status.render.width = width_;
  status.render.height = height_;
  status.render.targetFps = fps_;
  status.render.quality = quality_;
  status.visualizations = configuredVisualizations_;
  return status;
}

web_viewer::ViewerStatus WebMujocoViewer::currentStatus() const {
  if (frameRenderer_ == nullptr) return setUpStatus();
  web_viewer::ViewerStatus status;
  const mjvCamera& camera = frameRenderer_->camera();
  status.camera.mode = camera.type == mjCAMERA_TRACKING ? web_viewer::ViewerCameraMode::kTracking : web_viewer::ViewerCameraMode::kFree;
  status.camera.azimuth = camera.azimuth;
  status.camera.elevation = camera.elevation;
  status.camera.distance = camera.distance;
  status.camera.lookat = {camera.lookat[kXAxisIndex], camera.lookat[kYAxisIndex], camera.lookat[kZAxisIndex]};
  status.render.width = width_;
  status.render.height = height_;
  status.render.targetFps = fps_;
  status.render.measuredFps = frameRenderer_->fps();
  status.render.quality = quality_;
  for (int group = 0; group < kNumToggledGeomGroups; ++group) status.geomGroups.push_back(frameRenderer_->options().geomgroup[group] != 0);
  for (const std::unique_ptr<MujocoVisualization>& visualization : frameRenderer_->visualizations()) {
    status.visualizations.push_back({.name = visualization->name(),
                                     .hotkey = visualization->hotkey(),
                                     .enabled = visualization->enabled(),
                                     .description = visualization->description()});
  }
  return status;
}

}  // namespace robot::mujoco_sim_interface
