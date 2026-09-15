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

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/visualization/MujocoVisualization.h"
#include "robot_core/FPSTracker.h"

namespace robot::mujoco_sim_interface {

class MujocoSimInterface;

class MjState;

/**
 * Interactive MuJoCo viewer of the simulator. Everything drawn on top of the model is a MujocoVisualization
 * (visualization/): the set is built at start-up from the names in MujocoSimConfig::visualizations (task file
 * `simVisualizations`, see VisualizationRegistry.h for the names) and every frame runs their hooks in order.
 */
class MujocoRenderer {
 public:
  MujocoRenderer(const MujocoSimInterface* simInterface);

  ~MujocoRenderer();

  bool ok() const;

  void launchRenderThread();

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
   * contact points, contact forces, centres of mass, inertia ellipsoids, convex hulls and the model transparency.
   */
  static void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);

  // mouse button callback
  static void mouse_button(GLFWwindow* window, int button, int act, int mods);

  // mouse move callback
  static void mouse_move(GLFWwindow* window, double xpos, double ypos);

  // scroll callback
  static void scroll(GLFWwindow* window, double xoffset, double yoffset);

  ///

  void renderLoop();

  void printHotkeys() const;

  void toggleCameraTracking();
  void setupCamera();

  // Init must occur in the same thread that uses the opengl context.
  void initialize();

  // Cleanup must occur in same thread that owns the opengl context.
  void cleanup();

  const MujocoSimInterface* simInterface_;
  MjState simState_;

  std::vector<std::unique_ptr<MujocoVisualization>> visualizations_;

  std::thread render_thread_;

  GLFWwindow* window_;
  mjrRect viewport_ = {0, 0, 0, 0};

  int viewportWidth{1920};
  int viewportHeight{1024};

  // mouse interaction
  bool button_left = false;
  bool button_middle = false;
  bool button_right = false;

  double lastx = 0;
  double lasty = 0;

  double lastclicktm = 0;

  // Mujoco visualization structures
  mjvCamera mujocoCam_;       // abstract camera
  mjvOption mujocoOptions_;   // visualization options
  mjvScene mujocoScene_;      // abstract scene
  mjrContext mujocoContext_;  // custom GPU context

  size_t timeStepMicro_;

  std::atomic<bool> window_closed_{false};
  std::atomic<bool> init_complete_{false};

  FPSTracker rendererFps_{"renderer"};
};

}  // namespace robot::mujoco_sim_interface
