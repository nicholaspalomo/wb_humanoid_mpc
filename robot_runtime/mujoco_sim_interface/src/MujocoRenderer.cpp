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

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <regex>
#include <string>
#include <thread>

#include "mujoco_sim_interface/MujocoSimInterface.h"

namespace robot::mujoco_sim_interface {

/// GLFW callbacks

// keyboard callback
void MujocoRenderer::keyboard(GLFWwindow* window, int key, int, int act, int mods) {
  auto* renderer = static_cast<MujocoRenderer*>(glfwGetWindowUserPointer(window));

  // 'c' key: toggle contact point visualization
  // Effect: Renders small colored spheres at active physical collision contact points
  if (act == GLFW_PRESS && key == GLFW_KEY_C) {
    renderer->mujocoOptions_.flags[mjVIS_CONTACTPOINT] = !renderer->mujocoOptions_.flags[mjVIS_CONTACTPOINT];
  }

  // 'f' key: toggle contact force visualization
  // Effect: Renders 3D arrows depicting normal and friction forces at contacts
  if (act == GLFW_PRESS && key == GLFW_KEY_F) {
    renderer->mujocoOptions_.flags[mjVIS_CONTACTFORCE] = !renderer->mujocoOptions_.flags[mjVIS_CONTACTFORCE];
  }

  // 'm' key: toggle center of mass (CoM) visualization
  // Effect: Renders CoM indicator spheres for kinematic bodies/links
  if (act == GLFW_PRESS && key == GLFW_KEY_M) {
    renderer->mujocoOptions_.flags[mjVIS_COM] = !renderer->mujocoOptions_.flags[mjVIS_COM];
  }

  // 't' key: toggle model transparency
  // Effect: Switches between 30% alpha (x-ray mode for internal joint/geom inspection) and 100% opaque
  if (act == GLFW_PRESS && key == GLFW_KEY_T) {
    renderer->model_transparent = !renderer->model_transparent;
    if (renderer->model_transparent) {
      renderer->setTransparency(0.3f);
    } else {
      renderer->setTransparency(1.0f);
    }
  }

  // 'i' key: toggle inertia visualization
  // Effect: Renders equivalent inertia ellipsoids depicting principal moments of inertia
  if (act == GLFW_PRESS && key == GLFW_KEY_I) {
    renderer->mujocoOptions_.flags[mjVIS_INERTIA] = !renderer->mujocoOptions_.flags[mjVIS_INERTIA];
  }

  // 'h' key: toggle convex hull visualization
  // Effect: Renders computed convex hulls enclosing the link meshes
  if (act == GLFW_PRESS && key == GLFW_KEY_H) {
    renderer->mujocoOptions_.flags[mjVIS_CONVEXHULL] = !renderer->mujocoOptions_.flags[mjVIS_CONVEXHULL];
  }

  // Number keys '0'-'5': toggle geom groups
  // Effect: Group 0 = Floor/ground plane, Group 1 = Visual meshes, Group 2 = Collision primitives, Groups 3-5 = Aux
  if (act == GLFW_PRESS && key >= GLFW_KEY_0 && key <= GLFW_KEY_5) {
    int group = key - GLFW_KEY_0;
    renderer->mujocoOptions_.geomgroup[group] = !renderer->mujocoOptions_.geomgroup[group];
  }

  // 'k' key: toggle camera tracking mode (tracking robot vs free camera)
  // Effect: Switches between mjCAMERA_TRACKING (locks camera to follow robot base/pelvis) and mjCAMERA_FREE
  if (act == GLFW_PRESS && key == GLFW_KEY_K) {
    renderer->toggleCameraTracking();
  }

  // 'b' key: toggle the contact timeline overlay
  // Effect: Shows/hides the barcode of planned vs ground-truth contact state per contact point
  if (act == GLFW_PRESS && key == GLFW_KEY_B) {
    renderer->showContactTimeline_ = !renderer->showContactTimeline_;
  }

  // 'p' key: print hotkeys cheatsheet
  // Effect: Prints all interactive viewer hotkeys, toggle states, and mouse gestures to console
  if (act == GLFW_PRESS && key == GLFW_KEY_P) {
    std::cerr << "\n\n==========================================================="
              << "\nMuJoCo 3D Viewer Hotkeys & Controls\n===========================================================\n"
              << "  0-5 => toggle geom groups (0: Floor, 1: Visual Mesh, 2: Collision, 3-5: Aux)\n"
              << "  c   => toggle contact point visualization (spheres at collision contacts)\n"
              << "  f   => toggle contact force vectors (3D normal & friction force arrows)\n"
              << "  m   => toggle center of mass (CoM) indicators\n"
              << "  t   => toggle model transparency (30% x-ray mode vs 100% opaque)\n"
              << "  i   => toggle link inertia ellipsoids (principal moments of inertia)\n"
              << "  h   => toggle convex hull visualization\n"
              << "  k   => toggle camera tracking mode (mjCAMERA_TRACKING vs mjCAMERA_FREE)\n"
              << "  b   => toggle contact timeline (planned vs ground-truth contact per contact point)\n"
              << "  p   => print this hotkey cheatsheet\n"
              << "-----------------------------------------------------------\n"
              << "Mouse Controls:\n"
              << "  Left Drag        => rotate / orbit camera around focal point\n"
              << "  Right Drag       => pan / translate camera horizontally & vertically\n"
              << "  Scroll / MidDrag => zoom camera in / out\n"
              << "  Shift + Drag     => constrain orbit / pan motion to horizontal plane\n"
              << "===========================================================\n\n";
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
}

MujocoRenderer::~MujocoRenderer() {
  std::cerr << "Cleaning up renderer ..." << std::endl;
  ;
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

void MujocoRenderer::setTransparency(float transparency) const {
  for (int i = 0; i < simInterface_->getModel()->ngeom; i++) {
    simInterface_->getModel()->geom_rgba[4 * i + 3] = transparency;
  }
}

namespace {
void renderMetrics(const mjrContext* con, const mjrRect& viewport, const MjState& state, double fpsRender, double elapsed_time) {
  std::ostringstream metrics;

  // FPS (Simulation & Renderer)
  metrics << "Render FPS: " << static_cast<int>(fpsRender) << "\n";
  metrics << "Sim FPS: " << static_cast<int>(state.metrics.fpsSim) << "\n";

  // The actual amount of time elapsed in simulation.
  metrics << "Real Time[s]: " << std::fixed << std::setprecision(3) << elapsed_time << "\n";
  metrics << "Sim  Time[s]: " << std::fixed << std::setprecision(3) << state.data->time << "\n\n";

  // Real-time tracking
  metrics << "RTF: " << std::fixed << std::setprecision(3) << state.metrics.rtfSmoothed << "\n";
  metrics << "Drift[ms]: " << std::fixed << std::setprecision(3) << state.metrics.driftTick * 1e3 << "\n";
  metrics << "Cummulative Drift[ms]: " << std::fixed << std::setprecision(3) << state.metrics.driftCumulative * 1e3;

  mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, metrics.str().c_str(), nullptr, con);
}
}  // namespace

void MujocoRenderer::renderExternalForces() {
  auto* model = simInterface_->getModel();
  auto* data = simState_.data;

  // Safety check
  if (!model || !data) {
    return;
  }

  for (int body_id = 0; body_id < model->nbody; ++body_id) {
    const double* force = &data->xfrc_applied[6 * body_id];
    double fx = force[0];
    double fy = force[1];
    double fz = force[2];

    double magnitude = std::sqrt(fx * fx + fy * fy + fz * fz);
    if (magnitude < 1e-6) {
      continue;
    }

    const double* xpos = &data->xpos[3 * body_id];

    // Create arrow geom
    mjvGeom* arrow = nullptr;
    if (mujocoScene_.ngeom < mujocoScene_.maxgeom) {
      arrow = &mujocoScene_.geoms[mujocoScene_.ngeom];
      mujocoScene_.ngeom++;
    } else {
      continue;  // Skip if we're out of geom space
    }

    // Clear the geom
    std::memset(arrow, 0, sizeof(mjvGeom));

    // Scale factor that grows with force magnitude
    // Adjust these constants to tune the visualization
    const double base_scale = 0.1;    // Minimum arrow length
    const double force_scale = 0.05;  // How much to scale with force
    const double scale = base_scale + force_scale * magnitude;

    // Calculate arrow end point using normalized force direction and scale
    double end_x = xpos[0] + scale * (fx / magnitude);
    double end_y = xpos[1] + scale * (fy / magnitude);
    double end_z = xpos[2] + scale * (fz / magnitude);

    mjtNum from[3] = {xpos[0], xpos[1], xpos[2]};
    mjtNum to[3] = {end_x, end_y, end_z};

    mjv_connector(arrow,         // geom to write to
                  mjGEOM_ARROW,  // type (arrow)
                  0.005,         // width (thin arrows)
                  from,          // from position
                  to             // to position
    );

    // Set color
    arrow->rgba[0] = 0.5f;
    arrow->rgba[1] = 1.0f;
    arrow->rgba[2] = 0.0f;
    arrow->rgba[3] = 1.0f;

    // Set additional properties
    arrow->category = mjCAT_DECOR;
    arrow->emission = 1.0f;  // Makes it glow/bright
  }
}

namespace {
struct Rgba {
  float r, g, b, a;
};
constexpr Rgba kPanelBackground{0.0f, 0.0f, 0.0f, 0.55f};
constexpr Rgba kPlanContact{0.35f, 0.70f, 1.0f, 1.0f};  // the plan has the point in contact
constexpr Rgba kSwing{0.16f, 0.16f, 0.20f, 1.0f};       // in the air (plan or physics, in agreement)
constexpr Rgba kUnknown{0.40f, 0.40f, 0.40f, 1.0f};     // no plan yet, or no MuJoCo body for the contact point
constexpr Rgba kSimContact{0.35f, 0.85f, 0.35f, 1.0f};  // physics agrees: touching
constexpr Rgba kSimEarly{0.95f, 0.25f, 0.25f, 1.0f};    // touching while the plan says swing (early touch-down, scuff)
constexpr Rgba kSimLate{1.0f, 0.62f, 0.10f, 1.0f};      // in the air while the plan says contact (late touch-down, slip)
constexpr Rgba kTick{1.0f, 1.0f, 1.0f, 0.25f};
constexpr Rgba kText{0.92f, 0.92f, 0.92f, 1.0f};

void fillRect(int left, int bottom, int width, int height, const Rgba& color) {
  if (width <= 0 || height <= 0) return;
  mjr_rectangle(mjrRect{left, bottom, width, height}, color.r, color.g, color.b, color.a);
}

/// Width in pixels of `text` in the normal font of `con`. A label box narrower or shorter than its text leaves the
/// string's raster position outside the box, and OpenGL then drops the whole string, so boxes are sized from this.
int textWidth(const char* text, const mjrContext* con) {
  int width = 0;
  for (; *text != '\0'; ++text) width += con->charWidth[static_cast<unsigned char>(*text) & 127];
  return width;
}

void drawLabel(int left, int bottom, int width, int height, const char* text, const mjrContext* con) {
  if (width <= 0 || height <= 0) return;
  mjr_label(mjrRect{left, bottom, width, height}, mjFONT_NORMAL, text, 0.0f, 0.0f, 0.0f, 0.0f, kText.r, kText.g, kText.b, con);
}

/// Combined state of one contact point in one sample: bit 0 physics touching, bit 1 plan contact, bit 2 plan known.
int contactStateCode(const ContactTimelineSample& sample, int contact) {
  return static_cast<int>((sample.actual >> contact) & 1u) | (static_cast<int>((sample.target >> contact) & 1u) << 1) |
         (sample.targetKnown ? 4 : 0);
}
}  // namespace

void MujocoRenderer::renderContactTimeline() {
  if (!showContactTimeline_ || !simInterface_->hasContactDetection()) return;
  simInterface_->copyContactTimeline(contactTimelineScratch_);
  const std::vector<ContactTimelineSample>& samples = contactTimelineScratch_;
  if (samples.empty()) return;

  const std::vector<std::string>& names = simInterface_->getContactNames();
  const int numContacts = static_cast<int>(names.size());
  const double window = simInterface_->getContactTimelineWindow();
  const double now = samples.back().time;
  const double start = now - window;
  const uint32_t unresolved = simInterface_->getUnresolvedContactMask();

  // Layout in framebuffer pixels (origin bottom-left): a translucent panel along the bottom edge, the label columns on
  // the left, the time axis running left (oldest) to right (now). Every box that holds text is sized from the font.
  const mjrContext* con = &mujocoContext_;
  const int fontH = std::max(12, con->charHeight);
  constexpr int margin = 12, pad = 8, stripGap = 2, groupGap = 8;
  const int stripH = fontH + 4;  // tall enough for its "plan" / "sim" tag
  const int headerH = fontH + 6;
  const int axisH = fontH + 6;
  const int tagW = std::max(textWidth("plan", con), textWidth("sim", con)) + 12;
  int nameTextW = 0;
  for (const std::string& name : names) nameTextW = std::max(nameTextW, textWidth(name.c_str(), con));
  const int nameW = std::clamp(nameTextW + 12, 60, 400);
  const int groupH = 2 * stripH + stripGap;
  const int panelH = 2 * pad + headerH + numContacts * (groupH + groupGap) + axisH;
  const int panelW = viewport_.width - 2 * margin;
  const int panelL = margin;
  const int panelB = margin;
  const int x0 = panelL + pad + nameW + tagW;
  const int x1 = panelL + panelW - pad;
  const int stripW = x1 - x0;
  if (stripW < 120 || panelH > viewport_.height / 2) return;
  fillRect(panelL, panelB, panelW, panelH, kPanelBackground);

  const auto xOf = [&](double time) {
    const double fraction = std::clamp((time - start) / window, 0.0, 1.0);
    return x0 + static_cast<int>(std::lround(fraction * stripW));
  };

  // Header: title and legend.
  const int headerB = panelB + panelH - pad - headerH;
  const char* title = "contact timeline [b]";
  drawLabel(panelL + pad, headerB, std::max(nameW + tagW, textWidth(title, con) + 8), headerH, title, con);
  {
    int lx = x0;
    const auto legend = [&](const Rgba& color, const char* text) {
      const int textW = textWidth(text, con) + 8;
      if (lx + 18 + textW > x1) return;  // no room: the rest of the legend is dropped
      fillRect(lx, headerB + 4, fontH - 4, headerH - 8, color);
      lx += fontH;
      drawLabel(lx, headerB, textW, headerH, text, con);
      lx += textW + 14;
    };
    legend(kPlanContact, "plan: contact");
    legend(kSimContact, "sim: contact");
    legend(kSimEarly, "sim touching, plan swing");
    legend(kSimLate, "sim in air, plan contact");
  }

  // One group of two strips per contact point: the plan on top, the physics below.
  const int stripsTop = headerB - groupGap;
  int groupTop = stripsTop;
  for (int contact = 0; contact < numContacts; ++contact) {
    const int targetB = groupTop - stripH;
    const int actualB = targetB - stripGap - stripH;
    drawLabel(panelL + pad, actualB, nameW, groupH, names[contact].c_str(), con);
    drawLabel(panelL + pad + nameW, targetB, tagW, stripH, "plan", con);
    drawLabel(panelL + pad + nameW, actualB, tagW, stripH, "sim", con);

    if (((unresolved >> contact) & 1u) != 0u) {
      fillRect(x0, actualB, stripW, groupH, kUnknown);
      drawLabel(x0, actualB, stripW, groupH, "no MuJoCo body resolved for this contact frame", con);
      groupTop = actualB - groupGap;
      continue;
    }

    // Runs of equal state are drawn as single bars.
    size_t runStart = 0;
    for (size_t k = 1; k <= samples.size(); ++k) {
      const int code = contactStateCode(samples[runStart], contact);
      if (k < samples.size() && contactStateCode(samples[k], contact) == code) continue;
      const double runEnd = (k < samples.size()) ? samples[k].time : now;
      if (runEnd >= start) {
        const int xa = xOf(samples[runStart].time);
        const int xb = std::max(xOf(runEnd), xa + 1);
        const bool touching = (code & 1) != 0;
        const bool planContact = (code & 2) != 0;
        const bool planKnown = (code & 4) != 0;
        fillRect(xa, targetB, xb - xa, stripH, !planKnown ? kUnknown : (planContact ? kPlanContact : kSwing));
        const Rgba& simColor = !planKnown ? (touching ? kSimContact : kSwing)
                               : touching ? (planContact ? kSimContact : kSimEarly)
                                          : (planContact ? kSimLate : kSwing);
        fillRect(xa, actualB, xb - xa, stripH, simColor);
      }
      runStart = k;
    }
    groupTop = actualB - groupGap;
  }

  // Time axis: one tick per second, relative to now, at fixed positions.
  const int axisB = panelB + pad;
  const int ticksH = stripsTop - (axisB + axisH);
  for (int secondsAgo = 0; secondsAgo <= static_cast<int>(window); ++secondsAgo) {
    const int x = xOf(now - secondsAgo);
    fillRect(x, axisB + axisH, 1, ticksH, kTick);
    char text[16];
    if (secondsAgo == 0) {
      std::snprintf(text, sizeof(text), "now");
    } else {
      std::snprintf(text, sizeof(text), "-%d s", secondsAgo);
    }
    const int labelW = textWidth(text, con) + 8;
    drawLabel(std::clamp(x - labelW / 2, panelL, x1 - labelW), axisB, labelW, axisH, text, con);
  }
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

    mjv_updateScene(simInterface_->getModel(), simState_.data, &mujocoOptions_, nullptr, &mujocoCam_, mjCAT_ALL, &mujocoScene_);

    renderExternalForces();

    // render to glfw window
    mjv_updateCamera(simInterface_->getModel(), simState_.data, &mujocoCam_, &mujocoScene_);
    mjr_render(viewport_, &mujocoScene_, &mujocoContext_);

    // render text overlay
    const auto current_time = std::chrono::steady_clock::now();
    const auto elapsed_time = std::chrono::duration<double>(current_time - start_time).count();
    renderMetrics(&mujocoContext_, viewport_, simState_, rendererFps_.fps(), elapsed_time);
    renderContactTimeline();

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

  // Set mujoco option
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
}

void MujocoRenderer::cleanup() {
  // Cleanup mujoco stuff.
  mjv_freeScene(&mujocoScene_);
  mjr_freeContext(&mujocoContext_);

  glfwDestroyWindow(window_);
  glfwTerminate();
}

}  // namespace robot::mujoco_sim_interface
