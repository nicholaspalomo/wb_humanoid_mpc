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

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/viewer/MujocoFrameRenderer.h"
#include "mujoco_sim_interface/viewer/MujocoViewer.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"

struct GLFWwindow;  // GLFW/glfw3.h, which only GlfwMujocoViewer.cpp includes

namespace robot::mujoco_sim_interface {

class MujocoSimInterface;

/** The viewer key (MujocoFrameRenderer::handleKey) of a GLFW key code: '0' to
 * '5' and 'a' to 'z'; none for other keys. */
std::optional<char> viewerKeyFromGlfwKey(int glfwKey);

/** A cursor motion as MuJoCo's camera takes it
 * (MujocoFrameRenderer::moveCamera()). */
struct CameraDragFractions {
  double dx = 0.0;
  double dy = 0.0;
};

/**
 * Returns the cursor motion `dxPixels`, `dyPixels` in a window `windowHeight`
 * pixels high as fractions of that height on both axes: mjv_moveCamera measures
 * both in units of the viewport's height, so a 100 pixel drag turns or moves
 * the camera as far horizontally as vertically. A height below one pixel counts
 * as one.
 */
CameraDragFractions cameraDragFractions(double dxPixels,
                                        double dyPixels,
                                        int windowHeight);

/**
 * The `glfw` viewer: an interactive MuJoCo window on DISPLAY (the Xvfb/VNC
 * desktop of the containers). Everything drawn on top of the model is a
 * MujocoVisualization (visualization/), drawn by its MujocoFrameRenderer.
 *
 * It draws with a private copy of the simulator's model, made by Create() on
 * the simulator's thread before the physics starts, which each frame brings up
 * to date with the snapshot's run-time model arrays (MjState::modelFields), as
 * the web viewer does: the physics thread writes the simulator's own model
 * while it runs (the gantry's weld, the torque switch's damping, the
 * projectile), which the render thread must not read.
 *
 * start() launches the render thread and returns without waiting: the window,
 * the OpenGL context and every MuJoCo rendering structure live on that thread.
 * Where the viewer cannot start - no display, no window, no OpenGL - it logs
 * why and stops, ok() turns false, and the simulation runs on without it. The
 * other methods may be called from any thread.
 */
class GlfwMujocoViewer final : public MujocoViewer {
 public:
  /**
   * Copies `sim`'s model for the render thread; call it on the simulator's
   * thread before the physics starts. `sim` must outlive the viewer; the render
   * thread reads its state snapshots. Internal when the model cannot be copied.
   */
  static absl::StatusOr<std::unique_ptr<GlfwMujocoViewer>> Create(
      const MujocoSimInterface* absl_nonnull sim);

  GlfwMujocoViewer(const GlfwMujocoViewer&) = delete;
  GlfwMujocoViewer& operator=(const GlfwMujocoViewer&) = delete;
  GlfwMujocoViewer(GlfwMujocoViewer&&) = delete;
  GlfwMujocoViewer& operator=(GlfwMujocoViewer&&) = delete;

  /** Stops the render thread and waits for it, which frees the window and the
   * rendering structures. */
  ~GlfwMujocoViewer() override;

  absl::string_view name() const override { return kGlfwViewerName; }

  /** Launches the render thread; always OK, see the class comment. Call it
   * once. */
  absl::Status start() override;

  /** False once the window was closed, or once the viewer failed to start (see
   * waitForInit()). */
  bool ok() const override;

  /** Waits until the render thread has started the viewer or failed to; ok()
   * then tells which. */
  void waitForInit() const;

 private:
  friend class GlfwMujocoViewerTestPeer;

  GlfwMujocoViewer(const MujocoSimInterface* absl_nonnull sim,
                   MjModelPtr model);

  /** One capture of the window's back buffer, for GlfwMujocoViewerTestPeer:
   * RGB, rows bottom-up. */
  struct WindowCapture {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
  };

  // The size the window opens with, [px].
  static constexpr int kInitialWindowWidth = 1920;
  static constexpr int kInitialWindowHeight = 1024;

  // GLFW's callbacks are plain functions; they reach the viewer through the
  // window's user pointer.
  static void keyboard(GLFWwindow* absl_nonnull window,
                       int key,
                       int scancode,
                       int act,
                       int mods);
  static void mouseButton(GLFWwindow* absl_nonnull window,
                          int button,
                          int act,
                          int mods);
  static void mouseMove(GLFWwindow* absl_nonnull window,
                        double xpos,
                        double ypos);
  static void scroll(GLFWwindow* absl_nonnull window,
                     double xoffset,
                     double yoffset);

  void renderLoop();

  // Init must occur in the same thread that uses the opengl context. Fails,
  // having released what it created, when GLFW, the window or GLEW cannot be
  // set up.
  absl::Status initialize();

  // Cleanup must occur in same thread that owns the opengl context.
  void cleanup();

  /** Serves a capture requested through captureRequested_; render thread, after
   * drawing and before the swap. */
  void serveCapture();

  const MujocoSimInterface* absl_nonnull sim_;
  /// The private model copy the render thread draws with; declared before
  /// frameRenderer_, which points at it.
  MjModelPtr model_;
  /// Created by initialize() on the render thread, and reset by cleanup(); null
  /// before and after, and when initialize() failed. The GLFW callbacks that
  /// use it are installed after it is created and run on the render thread.
  std::unique_ptr<MujocoFrameRenderer> frameRenderer_;

  std::thread renderThread_;

  // Created by initialize() on the render thread and only used there; null
  // until then, and when GLFW could not create the window.
  GLFWwindow* absl_nullable window_ = nullptr;
  mjrRect viewport_ = {.left = 0, .bottom = 0, .width = 0, .height = 0};

  int viewportWidth_ = kInitialWindowWidth;
  int viewportHeight_ = kInitialWindowHeight;

  // mouse interaction
  bool buttonLeft_ = false;
  bool buttonMiddle_ = false;
  bool buttonRight_ = false;

  double lastX_ = 0.0;
  double lastY_ = 0.0;

  size_t timeStepMicro_ = 0;

  std::atomic<bool> windowClosed_ = false;
  std::atomic<bool> initComplete_ = false;
  // Set by the destructor: the render thread closes the viewer at its next
  // frame. A flag rather than glfwSetWindowShouldClose, so that no other thread
  // touches the window, which may never have been created.
  std::atomic<bool> stopRequested_ = false;

  // A window capture, for tests only: requested from any thread, served once by
  // the render loop (one relaxed load per frame while none is requested).
  std::atomic<bool> captureRequested_ = false;
  absl::Mutex captureMutex_;
  std::optional<WindowCapture> capture_ ABSL_GUARDED_BY(captureMutex_);
};

}  // namespace robot::mujoco_sim_interface
