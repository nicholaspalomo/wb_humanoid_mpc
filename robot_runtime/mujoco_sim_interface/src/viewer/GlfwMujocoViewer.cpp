/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include "mujoco_sim_interface/viewer/GlfwMujocoViewer.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "GL/glew.h"
#include "GLFW/glfw3.h"  // for creating the OpenGL context
#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mujoco/mujoco.h"
#include "ocs2_core/PhysicalConstants.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "web_viewer/JpegEncoder.h"

namespace robot::mujoco_sim_interface {

namespace {
constexpr int kInitPollPeriodMilliseconds = 10;  // how often waitForInit() looks whether the render thread is set up
}  // namespace

std::optional<char> viewerKeyFromGlfwKey(int glfwKey) {
  if (glfwKey >= GLFW_KEY_0 && glfwKey < GLFW_KEY_0 + kNumToggledGeomGroups) return static_cast<char>('0' + (glfwKey - GLFW_KEY_0));
  // GLFW letter keys equal the upper-case ASCII code.
  if (glfwKey >= GLFW_KEY_A && glfwKey <= GLFW_KEY_Z) return static_cast<char>('a' + (glfwKey - GLFW_KEY_A));
  return std::nullopt;
}

CameraDragFractions cameraDragFractions(double dxPixels, double dyPixels, int windowHeight) {
  const double height = std::max(windowHeight, 1);
  return CameraDragFractions{.dx = dxPixels / height, .dy = dyPixels / height};
}

/// GLFW callbacks

void GlfwMujocoViewer::keyboard(GLFWwindow* absl_nonnull window, int key, int /*unused*/, int act, int /*unused*/) {
  GlfwMujocoViewer* absl_nonnull viewer = static_cast<GlfwMujocoViewer*>(glfwGetWindowUserPointer(window));
  if (act != GLFW_PRESS) return;
  if (const std::optional<char> viewerKey = viewerKeyFromGlfwKey(key); viewerKey.has_value()) {
    viewer->frameRenderer_->handleKey(*viewerKey);
  }
}

void GlfwMujocoViewer::mouseButton(GLFWwindow* absl_nonnull window, int /*unused*/, int /*unused*/, int /*unused*/) {
  GlfwMujocoViewer* absl_nonnull viewer = static_cast<GlfwMujocoViewer*>(glfwGetWindowUserPointer(window));
  // update button state
  viewer->buttonLeft_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  viewer->buttonMiddle_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  viewer->buttonRight_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &(viewer->lastX_), &(viewer->lastY_));
}

void GlfwMujocoViewer::mouseMove(GLFWwindow* absl_nonnull window, double xpos, double ypos) {
  GlfwMujocoViewer* absl_nonnull viewer = static_cast<GlfwMujocoViewer*>(glfwGetWindowUserPointer(window));
  // no buttons down: nothing to do
  if (!viewer->buttonLeft_ && !viewer->buttonMiddle_ && !viewer->buttonRight_) return;

  // compute mouse displacement, save
  const double dx = xpos - viewer->lastX_;
  const double dy = ypos - viewer->lastY_;
  viewer->lastX_ = xpos;
  viewer->lastY_ = ypos;

  // get current window size: a drag is in fractions of its height on both axes (cameraDragFractions())
  int height = 0;
  glfwGetWindowSize(window, /*width=*/nullptr, &height);

  // get shift key state
  const bool modShift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  // determine action based on mouse button: the middle one zooms
  mjtMouse action = mjMOUSE_ZOOM;
  if (viewer->buttonRight_) {
    action = modShift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (viewer->buttonLeft_) {
    action = modShift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  }

  const CameraDragFractions fractions = cameraDragFractions(dx, dy, height);
  viewer->frameRenderer_->moveCamera(action, fractions.dx, fractions.dy);
}

void GlfwMujocoViewer::scroll(GLFWwindow* absl_nonnull window, double /*unused*/, double yoffset) {
  GlfwMujocoViewer* absl_nonnull viewer = static_cast<GlfwMujocoViewer*>(glfwGetWindowUserPointer(window));
  viewer->frameRenderer_->scroll(yoffset);
}

//// Public

absl::StatusOr<std::unique_ptr<GlfwMujocoViewer>> GlfwMujocoViewer::Create(const MujocoSimInterface* absl_nonnull sim) {
  MjModelPtr model(mj_copyModel(/*dest=*/nullptr, sim->getModel()));
  if (model == nullptr) return absl::InternalError("the GLFW viewer could not copy the simulator's model (mj_copyModel)");
  return absl::WrapUnique(new GlfwMujocoViewer(sim, std::move(model)));
}

GlfwMujocoViewer::GlfwMujocoViewer(const MujocoSimInterface* absl_nonnull sim, MjModelPtr model)
    : sim_(sim),
      model_(std::move(model)),
      timeStepMicro_(static_cast<size_t>(::ocs2::kMicrosecondsPerSecond / sim->getConfig().renderFrequencyHz)) {}

GlfwMujocoViewer::~GlfwMujocoViewer() {
  LOG(INFO) << "Cleaning up renderer ...";
  stopRequested_.store(true);
  if (renderThread_.joinable()) renderThread_.join();
}

bool GlfwMujocoViewer::ok() const {
  return !windowClosed_.load(std::memory_order_acquire);
}

absl::Status GlfwMujocoViewer::start() {
  renderThread_ = std::thread(&GlfwMujocoViewer::renderLoop, this);
  return absl::OkStatus();
}

void GlfwMujocoViewer::waitForInit() const {
  while (!initComplete_.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kInitPollPeriodMilliseconds));
  }
}

void GlfwMujocoViewer::renderLoop() {
  const absl::Status initialized = initialize();
  if (!initialized.ok()) {
    LOG(ERROR) << "[GlfwMujocoViewer] the viewer did not start, so the simulation runs without it: " << initialized;
    windowClosed_.store(true);
    initComplete_.store(true);
    return;
  }
  initComplete_.store(true);

  const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

  while (!glfwWindowShouldClose(window_) && !stopRequested_.load()) {
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    glfwGetFramebufferSize(window_, &viewport_.width, &viewport_.height);
    frameRenderer_->drawFrame(viewport_, std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count());
    if (captureRequested_.load(std::memory_order_relaxed)) serveCapture();

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window_);
    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();

    // Sleep in case render loop is faster than specified sim rate.
    std::this_thread::sleep_until(start + std::chrono::microseconds(timeStepMicro_));
  }

  LOG(INFO) << "Exited Mujoco renderLoop.";

  windowClosed_.store(true);
  cleanup();
}

void GlfwMujocoViewer::serveCapture() {
  WindowCapture capture;
  capture.width = viewport_.width;
  capture.height = viewport_.height;
  capture.rgb.resize(static_cast<size_t>(capture.width) * capture.height * web_viewer::kRgbBytesPerPixel);
  if (const absl::Status read = frameRenderer_->readPixels(viewport_, absl::MakeSpan(capture.rgb)); !read.ok()) {
    LOG(ERROR) << "[GlfwMujocoViewer] window capture: " << read;
    capture.rgb.clear();
  }
  absl::MutexLock lock(captureMutex_);
  capture_ = std::move(capture);
  captureRequested_.store(false, std::memory_order_relaxed);
}

absl::Status GlfwMujocoViewer::initialize() {
  // init GLFW
  if (glfwInit() != GLFW_TRUE) return absl::UnavailableError("GLFW could not initialize (is there a display?)");

  // create window, make OpenGL context current, request v-sync
  window_ = glfwCreateWindow(viewportWidth_, viewportHeight_, "Mujoco Robot Sim", /*monitor=*/nullptr, /*share=*/nullptr);
  if (window_ == nullptr) {
    glfwTerminate();
    return absl::UnavailableError("GLFW could not create the viewer's window");
  }
  glfwMakeContextCurrent(window_);

  // init glew
  if (glewInit() != GLEW_OK) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
    glfwTerminate();
    return absl::UnavailableError("GLEW could not initialize the viewer's OpenGL context");
  }

  glfwSwapInterval(1);

  // The camera, the options, the scene and the GPU context, drawn with the private model copy.
  absl::StatusOr<std::unique_ptr<MujocoFrameRenderer>> frameRenderer = MujocoFrameRenderer::Create(sim_, model_.get());
  if (!frameRenderer.ok()) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
    glfwTerminate();
    return frameRenderer.status();
  }
  frameRenderer_ = *std::move(frameRenderer);

  glfwSetWindowUserPointer(window_, this);

  // install GLFW mouse and keyboard callbacks
  glfwSetKeyCallback(window_, keyboard);
  glfwSetCursorPosCallback(window_, mouseMove);
  glfwSetMouseButtonCallback(window_, mouseButton);
  glfwSetScrollCallback(window_, scroll);

  // get framebuffer viewport
  glfwGetFramebufferSize(window_, &viewport_.width, &viewport_.height);

  frameRenderer_->renderScene(viewport_);
  // swap OpenGL buffers (blocking call due to v-sync)
  glfwSwapBuffers(window_);
  // process pending GUI events, call GLFW callbacks
  glfwPollEvents();

  frameRenderer_->printHotkeys();
  return absl::OkStatus();
}

void GlfwMujocoViewer::cleanup() {
  // Cleanup mujoco stuff.
  frameRenderer_->release();
  frameRenderer_.reset();

  glfwDestroyWindow(window_);
  glfwTerminate();
}

}  // namespace robot::mujoco_sim_interface
