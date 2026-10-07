/******************************************************************************
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
#include <memory>
#include <thread>
#include <vector>

#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/visualization/MujocoVisualization.h"
#include "robot_core/FPSTracker.h"

namespace robot::mujoco_sim_interface {

class MujocoSimInterface;

/**
 * Interactive MuJoCo viewer of the simulator. Everything drawn on top of the model is a MujocoVisualization
 * (visualization/): the set is built at start-up from the names in MujocoSimConfig::visualizations (task file
 * `sim_visualizations`, see VisualizationRegistry.h for the names) and every frame runs their hooks in order.
 *
 * The window, the OpenGL context and every MuJoCo rendering structure live on the render thread
 * (launchRenderThread()); the other methods may be called from any thread. Where the viewer cannot start - no
 * display, no window, no OpenGL - it logs why and stops, and the simulation runs on without it.
 */
class MujocoRenderer {
 public:
  /** `simInterface` must outlive the renderer; the render thread reads its model and state. */
  explicit MujocoRenderer(const MujocoSimInterface* absl_nonnull simInterface);

  MujocoRenderer(const MujocoRenderer&) = delete;
  MujocoRenderer& operator=(const MujocoRenderer&) = delete;
  MujocoRenderer(MujocoRenderer&&) = delete;
  MujocoRenderer& operator=(MujocoRenderer&&) = delete;

  /** Stops the render thread and waits for it, which frees the window and the rendering structures. */
  ~MujocoRenderer();

  /** False once the window was closed, or once the viewer failed to start (see waitForInit()). */
  bool ok() const;

  void launchRenderThread();

  /** Waits until the render thread has started the viewer or failed to; ok() then tells which. */
  void waitForInit() const;

  /** The visualizations of this viewer, in drawing order (render thread only, for tests and the cheatsheet). */
  const std::vector<std::unique_ptr<MujocoVisualization>>& visualizations() const { return visualizations_; }

 private:
  /// These callbacks are required to be static by glfw3 and are hence not part of the visualizer class. They have access to the visualizer
  /// through the window user pointer.

  /**
   * @brief GLFW keyboard callback for interactive MuJoCo 3D viewer viewport controls.
   *
   * Keys of the viewer itself:
   *  - '0'-'5' : Toggle geometry groups in mujocoOptions_.geomgroup:
   *                '0' -> Floor / ground plane geometries
   *                '1' -> Visual meshes (high-resolution STL/OBJ surface meshes)
   *                '2' -> Collision primitives (capsules, boxes, cylinders, spheres)
   *                '3'-'5' -> Auxiliary/sensor geometry groups
   *  - 'k'     : Toggle Camera Tracking Mode
   *                Switches between mjCAMERA_TRACKING (locks camera view to translate with the
   *                robot base/pelvis) and mjCAMERA_FREE (stationary manual free-look camera).
   *  - 'p'     : Print Hotkeys Cheatsheet
   *                Prints all supported hotkeys and mouse bindings to the console, including the
   *                hotkey and state of every visualization.
   * Every other letter toggles the visualization that declares it as its hotkey (MujocoVisualization::hotkey), e.g.
   * 'b' for the contact timeline, 'g' for the target contact patches, 'c' / 'f' / 'm' / 'i' / 'h' / 't' for MuJoCo's own
   * contact points, contact forces, centers of mass, inertia ellipsoids, convex hulls and the model transparency.
   */
  static void keyboard(GLFWwindow* absl_nonnull window, int key, int scancode, int act, int mods);

  // mouse button callback
  static void mouse_button(GLFWwindow* absl_nonnull window, int button, int act, int mods);

  // mouse move callback
  static void mouse_move(GLFWwindow* absl_nonnull window, double xpos, double ypos);

  // scroll callback
  static void scroll(GLFWwindow* absl_nonnull window, double xoffset, double yoffset);

  ///

  void renderLoop();

  void printHotkeys() const;

  void toggleCameraTracking();
  void setupCamera();

  // Init must occur in the same thread that uses the opengl context. Fails, having released what it created, when GLFW,
  // the window or GLEW cannot be set up.
  absl::Status initialize();

  // Cleanup must occur in same thread that owns the opengl context.
  void cleanup();

  const MujocoSimInterface* absl_nonnull simInterface_;
  MjState simState_;

  std::vector<std::unique_ptr<MujocoVisualization>> visualizations_;

  std::thread render_thread_;

  // Created by initialize() on the render thread and only used there; null until then, and when GLFW could not create
  // the window.
  GLFWwindow* absl_nullable window_ = nullptr;
  mjrRect viewport_ = {0, 0, 0, 0};

  int viewportWidth_ = 1920;
  int viewportHeight_ = 1024;

  // mouse interaction
  bool button_left_ = false;
  bool button_middle_ = false;
  bool button_right_ = false;

  double lastx_ = 0;
  double lasty_ = 0;

  // Mujoco visualization structures, zeroed until initialize() fills them in.
  mjvCamera mujocoCam_{};       // abstract camera
  mjvOption mujocoOptions_{};   // visualization options
  mjvScene mujocoScene_{};      // abstract scene
  mjrContext mujocoContext_{};  // custom GPU context

  size_t timeStepMicro_ = 0;

  std::atomic<bool> window_closed_{false};
  std::atomic<bool> init_complete_{false};
  // Set by the destructor: the render thread closes the viewer at its next frame. A flag rather than
  // glfwSetWindowShouldClose, so that no other thread touches the window, which may never have been created.
  std::atomic<bool> stopRequested_{false};

  FPSTracker rendererFps_;
};

}  // namespace robot::mujoco_sim_interface
