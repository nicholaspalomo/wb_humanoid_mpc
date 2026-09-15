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

#include "mujoco_sim_interface/MujocoRenderer.h"

#include <GLFW/glfw3.h>  // for creating the OpenGL context
#include <mujoco/mujoco.h>

#include <cctype>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/visualization/VisualizationRegistry.h"

namespace robot::mujoco_sim_interface {

/// GLFW callbacks

// keyboard callback
void MujocoRenderer::keyboard(GLFWwindow* window, int key, int, int act, int) {
  auto* renderer = static_cast<MujocoRenderer*>(glfwGetWindowUserPointer(window));
  if (act != GLFW_PRESS) return;

  // Number keys '0'-'5': toggle geom groups
  // Effect: Group 0 = Floor/ground plane, Group 1 = Visual meshes, Group 2 = Collision primitives, Groups 3-5 = Aux
  if (key >= GLFW_KEY_0 && key <= GLFW_KEY_5) {
    const int group = key - GLFW_KEY_0;
    renderer->mujocoOptions_.geomgroup[group] = !renderer->mujocoOptions_.geomgroup[group];
  }

  // 'k' key: toggle camera tracking mode (tracking robot vs free camera)
  if (key == GLFW_KEY_K) renderer->toggleCameraTracking();

  // 'p' key: print hotkeys cheatsheet
  if (key == GLFW_KEY_P) renderer->printHotkeys();

  // Every other letter belongs to the visualization that declares it (GLFW letter keys equal the upper-case ASCII code).
  for (const std::unique_ptr<MujocoVisualization>& visualization : renderer->visualizations_) {
    const char hotkey = visualization->hotkey();
    if (hotkey != 0 && key == std::toupper(static_cast<unsigned char>(hotkey))) {
      visualization->toggle();
      std::cerr << "[MujocoRenderer] " << visualization->name() << (visualization->enabled() ? " on" : " off") << std::endl;
    }
  }
}

// mouse button callback
void MujocoRenderer::mouse_button(GLFWwindow* window, int, int, int) {
  auto* renderer = static_cast<MujocoRenderer*>(glfwGetWindowUserPointer(window));
  // update button state
  renderer->button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  renderer->button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  renderer->button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &(renderer->lastx), &(renderer->lasty));
}

// mouse move callback
void MujocoRenderer::mouse_move(GLFWwindow* window, double xpos, double ypos) {
  auto* renderer = static_cast<MujocoRenderer*>(glfwGetWindowUserPointer(window));
  // no buttons down: nothing to do
  if (!renderer->button_left && !renderer->button_middle && !renderer->button_right) return;

  // compute mouse displacement, save
  double dx = xpos - renderer->lastx;
  double dy = ypos - renderer->lasty;
  renderer->lastx = xpos;
  renderer->lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (renderer->button_right)
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  else if (renderer->button_left)
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  else
    action = mjMOUSE_ZOOM;

  // move camera
  mjv_moveCamera(renderer->simInterface_->getModel(), action, dx / width, dy / height, &renderer->mujocoScene_, &renderer->mujocoCam_);
}

// scroll callback
void MujocoRenderer::scroll(GLFWwindow* window, double, double yoffset) {
  auto* renderer = static_cast<MujocoRenderer*>(glfwGetWindowUserPointer(window));
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(renderer->simInterface_->getModel(), mjMOUSE_ZOOM, 0, -0.05 * yoffset, &renderer->mujocoScene_, &renderer->mujocoCam_);
}

//// Public

MujocoRenderer::MujocoRenderer(const MujocoSimInterface* simInterface)
    : simInterface_(simInterface),
      simState_(simInterface_->getModel()),
      timeStepMicro_(1e6 / simInterface_->getConfig().renderFrequencyHz) {
  mujocoScene_.flags[mjRND_SHADOW] = 1;
  mujocoScene_.flags[mjRND_REFLECTION] = 1;

  std::vector<std::string> errors;
  visualizations_ = createVisualizations(simInterface_->getConfig().visualizations, &errors);
  for (const std::string& error : errors) {
    std::cerr << "[MujocoRenderer] simVisualizations: " << error << std::endl;
  }
}

MujocoRenderer::~MujocoRenderer() {
  std::cerr << "Cleaning up renderer ..." << std::endl;
  if (render_thread_.joinable()) {
    glfwSetWindowShouldClose(window_, GLFW_TRUE);
    render_thread_.join();
  }
}

bool MujocoRenderer::ok() const {
  return !window_closed_.load(std::memory_order_acquire);
}

void MujocoRenderer::launchRenderThread() {
  render_thread_ = std::thread(&MujocoRenderer::renderLoop, this);
}

void MujocoRenderer::waitForInit() const {
  while (!init_complete_.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void MujocoRenderer::printHotkeys() const {
  std::cerr << "\n\n==========================================================="
            << "\nMuJoCo 3D Viewer Hotkeys & Controls\n===========================================================\n"
            << "  0-5 => toggle geom groups (0: Floor, 1: Visual Mesh, 2: Collision, 3-5: Aux)\n"
            << "  k   => toggle camera tracking mode (mjCAMERA_TRACKING vs mjCAMERA_FREE)\n"
            << "  p   => print this hotkey cheatsheet\n"
            << "-----------------------------------------------------------\n"
            << "Visualizations (task file simVisualizations; [x] on, [ ] off):\n";
  for (const std::unique_ptr<MujocoVisualization>& visualization : visualizations_) {
    const char hotkey = visualization->hotkey();
    std::cerr << "  " << (hotkey != 0 ? hotkey : ' ') << "   " << (visualization->enabled() ? "[x] " : "[ ] ") << visualization->name()
              << ": " << visualization->description() << "\n";
  }
  std::cerr << "  (not listed in the task file:";
  for (const VisualizationInfo& info : availableVisualizations()) {
    bool listed = false;
    for (const std::unique_ptr<MujocoVisualization>& visualization : visualizations_) listed = listed || visualization->name() == info.name;
    if (!listed) std::cerr << " " << info.name;
  }
  std::cerr << ")\n"
            << "-----------------------------------------------------------\n"
            << "Mouse Controls:\n"
            << "  Left Drag        => rotate / orbit camera around focal point\n"
            << "  Right Drag       => pan / translate camera horizontally & vertically\n"
            << "  Scroll / MidDrag => zoom camera in / out\n"
            << "  Shift + Drag     => constrain orbit / pan motion to horizontal plane\n"
            << "===========================================================\n\n";
}

void MujocoRenderer::renderLoop() {
  initialize();
  init_complete_.store(true);

  const auto start_time = std::chrono::steady_clock::now();

  while (!glfwWindowShouldClose(window_)) {
    auto start = std::chrono::steady_clock::now();

    glfwGetFramebufferSize(window_, &viewport_.width, &viewport_.height);

    // Copy physics data to render data
    simInterface_->readLatestMjState(simState_);
    mj_forward(simInterface_->getModel(), simState_.data);

    VisualizationFrame frame;
    frame.sim = simInterface_;
    frame.state = &simState_;
    frame.options = &mujocoOptions_;
    frame.scene = &mujocoScene_;
    frame.context = &mujocoContext_;
    frame.viewport = viewport_;
    frame.renderFps = rendererFps_.fps();
    frame.elapsedRealTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

    // Options and model tweaks, for every visualization so that a disabled one can undo its effect.
    for (const std::unique_ptr<MujocoVisualization>& visualization : visualizations_) visualization->beforeSceneUpdate(frame);

    mjv_updateScene(simInterface_->getModel(), simState_.data, &mujocoOptions_, nullptr, &mujocoCam_, mjCAT_ALL, &mujocoScene_);

    // Decor geoms of the enabled visualizations, added to the scene before it is rendered.
    for (const std::unique_ptr<MujocoVisualization>& visualization : visualizations_) {
      if (visualization->enabled()) visualization->addSceneGeoms(frame);
    }

    // render to glfw window
    mjv_updateCamera(simInterface_->getModel(), simState_.data, &mujocoCam_, &mujocoScene_);
    mjr_render(viewport_, &mujocoScene_, &mujocoContext_);

    // 2D overlays and text of the enabled visualizations, on top of the rendered scene.
    for (const std::unique_ptr<MujocoVisualization>& visualization : visualizations_) {
      if (visualization->enabled()) visualization->renderOverlay(frame);
    }

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window_);
    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();

    // Sleep in case render loop is faster than specified sim rate.
    std::this_thread::sleep_until(start + std::chrono::microseconds(timeStepMicro_));

    rendererFps_.tick();
  }

  std::cerr << "Exited Mujoco renderLoop." << std::endl;

  window_closed_.store(true);
  cleanup();
}

void MujocoRenderer::toggleCameraTracking() {
  if (mujocoCam_.type == mjCAMERA_TRACKING) {
    mujocoCam_.type = mjCAMERA_FREE;
    std::cerr << "Camera mode: FREE (manual)" << std::endl;
  } else {
    mujocoCam_.type = mjCAMERA_TRACKING;
    std::cerr << "Camera mode: TRACKING (following robot body " << mujocoCam_.trackbodyid << ")" << std::endl;
  }
}

void MujocoRenderer::setupCamera() {
  // Setup Tracking Camera (follows the robot)
  mujocoCam_.type = mjCAMERA_TRACKING;
  int trackbodyid = 1;
  const mjModel* m = simInterface_->getModel();
  if (m && m->nbody > 1) {
    int pelvis_id = mj_name2id(m, mjOBJ_BODY, "pelvis");
    if (pelvis_id > 0) {
      trackbodyid = pelvis_id;
    } else {
      int pelvis_link_id = mj_name2id(m, mjOBJ_BODY, "pelvis_link");
      if (pelvis_link_id > 0) {
        trackbodyid = pelvis_link_id;
      } else {
        int torso_id = mj_name2id(m, mjOBJ_BODY, "torso");
        if (torso_id > 0) {
          trackbodyid = torso_id;
        }
      }
    }
  }
  mujocoCam_.trackbodyid = trackbodyid;
  mujocoCam_.fixedcamid = -1;

  double arr_view[] = {89.608063, -5.588379, 3, 0.000000, 0.000000, 0.500000};  // view the left side (for ll, lh, left_side)
  mujocoCam_.azimuth = arr_view[0];
  mujocoCam_.elevation = arr_view[1];
  mujocoCam_.distance = arr_view[2];
  mujocoCam_.lookat[0] = arr_view[3];
  mujocoCam_.lookat[1] = arr_view[4];
  mujocoCam_.lookat[2] = arr_view[5];
}

void MujocoRenderer::initialize() {
  // init GLFW
  if (!glfwInit()) mju_error("Could not initialize GLFW");

  // create window, make OpenGL context current, request v-sync
  window_ = glfwCreateWindow(viewportWidth, viewportHeight, "Mujoco Robot Sim", nullptr, nullptr);
  glfwMakeContextCurrent(window_);

  // init glew
  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    return;
  }

  glfwSwapInterval(1);

  // initialize visualization data structures
  mjv_defaultCamera(&mujocoCam_);
  mjv_defaultOption(&mujocoOptions_);
  mjv_defaultScene(&mujocoScene_);
  mjr_defaultContext(&mujocoContext_);
  mjv_makeScene(simInterface_->getModel(), &mujocoScene_, 2000);                 // space for 2000 objects
  mjr_makeContext(simInterface_->getModel(), &mujocoContext_, mjFONTSCALE_150);  // model-specific context

  // MuJoCo's own markers start off; the mj_* visualizations of the task file switch them on (MujocoOptionFlagVisualization).
  mujocoOptions_.flags[mjVIS_CONTACTPOINT] = 0;
  mujocoOptions_.flags[mjVIS_CONTACTFORCE] = 0;
  mujocoOptions_.flags[mjVIS_COM] = 0;
  mujocoOptions_.flags[mjVIS_INERTIA] = 0;

  glfwSetWindowUserPointer(window_, this);

  // install GLFW mouse and keyboard callbacks
  glfwSetKeyCallback(window_, keyboard);
  glfwSetCursorPosCallback(window_, mouse_move);
  glfwSetMouseButtonCallback(window_, mouse_button);
  glfwSetScrollCallback(window_, scroll);

  setupCamera();

  simInterface_->readLatestMjState(simState_);

  // get framebuffer viewport
  // We don't want to step the actual simulation here, as it screws up the initialization of the IMU's
  mj_step(simInterface_->getModel(), simState_.data);  // populate state info
  glfwGetFramebufferSize(window_, &viewport_.width, &viewport_.height);

  mjv_updateScene(simInterface_->getModel(), simState_.data, &mujocoOptions_, nullptr, &mujocoCam_, mjCAT_ALL, &mujocoScene_);

  mjr_render(viewport_, &mujocoScene_, &mujocoContext_);
  // swap OpenGL buffers (blocking call due to v-sync)
  glfwSwapBuffers(window_);
  // process pending GUI events, call GLFW callbacks
  glfwPollEvents();

  printHotkeys();
}

void MujocoRenderer::cleanup() {
  // Cleanup mujoco stuff.
  mjv_freeScene(&mujocoScene_);
  mjr_freeContext(&mujocoContext_);

  glfwDestroyWindow(window_);
  glfwTerminate();
}

}  // namespace robot::mujoco_sim_interface
