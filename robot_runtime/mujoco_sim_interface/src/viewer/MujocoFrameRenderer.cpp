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

#include "mujoco_sim_interface/viewer/MujocoFrameRenderer.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mujoco/mujoco.h"
#include "ocs2_core/Dimensions.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/visualization/VisualizationRegistry.h"
#include "web_viewer/JpegEncoder.h"

namespace robot::mujoco_sim_interface {

using ::ocs2::kZAxisIndex;

namespace {
constexpr int kMaxSceneGeoms = 2000;  // the objects the scene has space for
// A scroll notch emulates a vertical mouse motion of this fraction of the
// window's height.
constexpr double kScrollZoomPerNotch = 0.05;
// The view resetCamera() starts from, of the robot's left side: MuJoCo's camera
// azimuth and elevation [deg], its distance [m] and the point it looks at [m].
constexpr double kLeftSideViewAzimuth = 89.608063;
constexpr double kLeftSideViewElevation = -5.588379;
constexpr double kLeftSideViewDistance = 3.0;
constexpr double kLeftSideViewLookatX = 0.0;
constexpr double kLeftSideViewLookatY = 0.0;
constexpr double kLeftSideViewLookatZ = 0.5;
}  // namespace

absl::StatusOr<std::unique_ptr<MujocoFrameRenderer>>
MujocoFrameRenderer::Create(const MujocoSimInterface* absl_nonnull sim,
                            mjModel* absl_nonnull model) {
  std::unique_ptr<MujocoFrameRenderer> renderer =
      absl::WrapUnique(new MujocoFrameRenderer(sim, model));
  if (absl::Status setUp = renderer->setUp(); !setUp.ok()) {
    renderer->release();
    return setUp;
  }
  return renderer;
}

MujocoFrameRenderer::MujocoFrameRenderer(
    const MujocoSimInterface* absl_nonnull sim, mjModel* absl_nonnull model)
    : sim_(sim), model_(model), state_(sim->getModel()) {
  std::vector<std::string> errors;
  visualizations_ =
      createVisualizations(sim_->getConfig().visualizations, &errors);
  for (const std::string& error : errors) {
    LOG(INFO) << "[MujocoFrameRenderer] sim_visualizations: " << error;
  }
}

MujocoFrameRenderer::~MujocoFrameRenderer() {
  if (sceneAllocated_) mjv_freeScene(&scene_);
}

absl::Status MujocoFrameRenderer::setUp() {
  mjModel* absl_nonnull model = ABSL_DIE_IF_NULL(model_);
  mjv_defaultCamera(&camera_);
  mjv_defaultOption(&options_);
  mjv_defaultScene(&scene_);
  mjr_defaultContext(&context_);
  mjv_makeScene(model, &scene_, kMaxSceneGeoms);
  sceneAllocated_ = true;
  mjr_makeContext(model, &context_, mjFONTSCALE_150);  // model-specific context
  if (context_.glInitialized == 0) {
    return absl::FailedPreconditionError(
        "MuJoCo could not make its GPU context: no OpenGL context is current "
        "on the render thread");
  }

  // MuJoCo's own markers start off; the mj_* visualizations of the task file
  // switch them on (MujocoOptionFlagVisualization).
  options_.flags[mjVIS_CONTACTPOINT] = 0;
  options_.flags[mjVIS_CONTACTFORCE] = 0;
  options_.flags[mjVIS_COM] = 0;
  options_.flags[mjVIS_INERTIA] = 0;

  resetCamera();

  sim_->readLatestMjState(state_);
  state_.modelFields.applyTo(model);
  // The snapshot is stepped, not the simulation: stepping the simulation here
  // would upset the IMU's initialization.
  mj_step(model, state_.data);  // populate state info
  mjv_updateScene(model, state_.data, &options_, /*pert=*/nullptr, &camera_,
                  mjCAT_ALL, &scene_);
  return absl::OkStatus();
}

void MujocoFrameRenderer::drawFrame(const mjrRect& viewport,
                                    double elapsedRealTime) {
  if (model_ == nullptr) return;  // released
  mjModel* absl_nonnull model = model_;
  // Copy physics data to render data, and the model arrays the physics changed
  // into the private model copy.
  sim_->readLatestMjState(state_);
  state_.modelFields.applyTo(model);
  mj_forward(model, state_.data);

  VisualizationFrame frame;
  frame.sim = sim_;
  frame.model = model;
  frame.state = &state_;
  frame.options = &options_;
  frame.scene = &scene_;
  frame.context = &context_;
  frame.viewport = viewport;
  frame.renderFps = fps_.fps();
  frame.elapsedRealTime = elapsedRealTime;

  // Options and model tweaks, for every visualization so that a disabled one
  // can undo its effect.
  for (const std::unique_ptr<MujocoVisualization>& visualization :
       visualizations_)
    visualization->beforeSceneUpdate(frame);

  mjv_updateScene(model, state_.data, &options_, /*pert=*/nullptr, &camera_,
                  mjCAT_ALL, &scene_);

  // Decor geoms of the enabled visualizations, added to the scene before it is
  // rendered.
  for (const std::unique_ptr<MujocoVisualization>& visualization :
       visualizations_) {
    if (visualization->enabled()) visualization->addSceneGeoms(frame);
  }

  mjv_updateCamera(model, state_.data, &camera_, &scene_);
  mjr_render(viewport, &scene_, &context_);

  // 2D overlays and text of the enabled visualizations, on top of the rendered
  // scene.
  for (const std::unique_ptr<MujocoVisualization>& visualization :
       visualizations_) {
    if (visualization->enabled()) visualization->renderOverlay(frame);
  }
  fps_.tick();
}

void MujocoFrameRenderer::renderScene(const mjrRect& viewport) {
  mjr_render(viewport, &scene_, &context_);
}

void MujocoFrameRenderer::moveCamera(mjtMouse action,
                                     double relativeDx,
                                     double relativeDy) {
  mjv_moveCamera(model_, action, relativeDx, relativeDy, &scene_, &camera_);
}

void MujocoFrameRenderer::scroll(double notches) {
  mjv_moveCamera(model_, mjMOUSE_ZOOM, /*reldx=*/0.0,
                 -kScrollZoomPerNotch * notches, &scene_, &camera_);
}

void MujocoFrameRenderer::handleKey(char key) {
  // Number keys '0'-'5': toggle geom groups
  // Effect: Group 0 = Floor/ground plane, Group 1 = Visual meshes, Group 2 =
  // Collision primitives, Groups 3-5 = Aux
  if (key >= '0' && key < '0' + kNumToggledGeomGroups) {
    const int group = key - '0';
    options_.geomgroup[group] = !options_.geomgroup[group];
    return;
  }
  if (key < 'a' || key > 'z') return;

  // 'k' key: toggle camera tracking mode (tracking robot vs free camera)
  if (key == 'k') toggleCameraTracking();

  // 'p' key: print hotkeys cheatsheet
  if (key == 'p') printHotkeys();

  // Every other letter belongs to the visualization that declares it.
  for (const std::unique_ptr<MujocoVisualization>& visualization :
       visualizations_) {
    if (visualization->hotkey() == key) {
      visualization->toggle();
      LOG(INFO) << "[MujocoViewer] " << visualization->name()
                << (visualization->enabled() ? " on" : " off");
    }
  }
}

bool MujocoFrameRenderer::toggleVisualization(absl::string_view name) {
  for (const std::unique_ptr<MujocoVisualization>& visualization :
       visualizations_) {
    if (visualization->name() != name) continue;
    visualization->toggle();
    LOG(INFO) << "[MujocoViewer] " << visualization->name()
              << (visualization->enabled() ? " on" : " off");
    return true;
  }
  return false;
}

void MujocoFrameRenderer::toggleCameraTracking() {
  setCameraType(camera_.type == mjCAMERA_TRACKING ? mjCAMERA_FREE
                                                  : mjCAMERA_TRACKING);
}

void MujocoFrameRenderer::setCameraType(mjtCamera type) {
  if (type == mjCAMERA_FREE) {
    camera_.type = mjCAMERA_FREE;
    LOG(INFO) << "Camera mode: FREE (manual)";
  } else if (type == mjCAMERA_TRACKING) {
    camera_.type = mjCAMERA_TRACKING;
    LOG(INFO) << "Camera mode: TRACKING (following robot body "
              << camera_.trackbodyid << ")";
  }
}

void MujocoFrameRenderer::resetCamera() {
  // Setup Tracking Camera (follows the robot)
  camera_.type = mjCAMERA_TRACKING;
  // The body the camera follows, by the names the robots of this repository
  // give their floating base. Body 1 is the fallback: it is the first child of
  // the world and therefore the floating base of every model here, which is
  // what a robot naming its base something else entirely still gets.
  static constexpr std::array<const char* absl_nonnull, 4>
      kFloatingBaseCandidates = {"pelvis", "pelvis_link", "torso", "base_link"};
  int trackbodyid = 1;
  const mjModel* absl_nonnull m = model_ != nullptr ? model_ : sim_->getModel();
  if (m->nbody > 1) {
    for (const char* absl_nonnull candidate : kFloatingBaseCandidates) {
      const int bodyId = mj_name2id(m, mjOBJ_BODY, candidate);
      if (bodyId > 0) {
        trackbodyid = bodyId;
        break;
      }
    }
  }
  camera_.trackbodyid = trackbodyid;
  camera_.fixedcamid = -1;

  camera_.azimuth = kLeftSideViewAzimuth;
  camera_.elevation = kLeftSideViewElevation;
  camera_.distance = kLeftSideViewDistance;
  camera_.lookat[0] = kLeftSideViewLookatX;
  camera_.lookat[1] = kLeftSideViewLookatY;
  camera_.lookat[kZAxisIndex] = kLeftSideViewLookatZ;
}

void MujocoFrameRenderer::printHotkeys() const {
  LOG(INFO)
      << "\n\n==========================================================="
      << "\nMuJoCo 3D Viewer Hotkeys & "
         "Controls\n==========================================================="
         "\n"
      << "  0-5 => toggle geom groups (0: Floor, 1: Visual Mesh, 2: Collision, "
         "3-5: Aux)\n"
      << "  k   => toggle camera tracking mode (mjCAMERA_TRACKING vs "
         "mjCAMERA_FREE)\n"
      << "  p   => print this hotkey cheatsheet\n"
      << "-----------------------------------------------------------\n"
      << "Visualizations (task file sim_visualizations; [x] on, [ ] off):\n";
  for (const std::unique_ptr<MujocoVisualization>& visualization :
       visualizations_) {
    const char hotkey = visualization->hotkey();
    LOG(INFO) << "  " << (hotkey != 0 ? hotkey : ' ') << "   "
              << (visualization->enabled() ? "[x] " : "[ ] ")
              << visualization->name() << ": " << visualization->description()
              << "\n";
  }
  LOG(INFO) << "  (not listed in the task file:";
  for (const VisualizationInfo& info : availableVisualizations()) {
    bool listed = false;
    for (const std::unique_ptr<MujocoVisualization>& visualization :
         visualizations_)
      listed = listed || visualization->name() == info.name;
    if (!listed) LOG(INFO) << " " << info.name;
  }
  LOG(INFO)
      << ")\n"
      << "-----------------------------------------------------------\n"
      << "Mouse Controls:\n"
      << "  Left Drag        => rotate / orbit camera around focal point\n"
      << "  Right Drag       => pan / translate camera horizontally & "
         "vertically\n"
      << "  Scroll / MidDrag => zoom camera in / out\n"
      << "  Shift + Drag     => constrain orbit / pan motion to horizontal "
         "plane\n"
      << "===========================================================\n\n";
}

absl::Status MujocoFrameRenderer::readPixels(const mjrRect& viewport,
                                             absl::Span<uint8_t> rgb) const {
  const int64_t expected = static_cast<int64_t>(viewport.width) *
                           viewport.height * web_viewer::kRgbBytesPerPixel;
  if (viewport.width <= 0 || viewport.height <= 0 ||
      static_cast<int64_t>(rgb.size()) != expected) {
    return absl::InvalidArgumentError(
        absl::StrCat("readPixels: a ", viewport.width, "x", viewport.height,
                     " viewport needs ", expected,
                     " bytes of RGB, the buffer holds ", rgb.size()));
  }
  mjr_readPixels(rgb.data(), /*depth=*/nullptr, viewport, &context_);
  return absl::OkStatus();
}

void MujocoFrameRenderer::useOffscreenBuffer() {
  mjr_setBuffer(mjFB_OFFSCREEN, &context_);
}

void MujocoFrameRenderer::resizeOffscreen(int width, int height) {
  if (context_.offWidth == width && context_.offHeight == height) return;
  mjr_resizeOffscreen(width, height, &context_);
  mjr_setBuffer(mjFB_OFFSCREEN, &context_);
}

void MujocoFrameRenderer::release() {
  if (model_ == nullptr) return;
  if (sceneAllocated_) mjv_freeScene(&scene_);
  sceneAllocated_ = false;
  mjr_freeContext(&context_);
  model_ = nullptr;
}

double MujocoFrameRenderer::snapshotTime() const {
  return state_.data != nullptr ? state_.data->time : 0.0;
}

absl::StatusOr<MjModelPtr> copyModelForOffscreenRendering(
    const mjModel* absl_nonnull model, const WebViewerOptions& options) {
  MjModelPtr privateModel(mj_copyModel(/*dest=*/nullptr, model));
  if (privateModel == nullptr)
    return absl::InternalError(
        "MuJoCo could not copy the model for the viewer's render thread");
  privateModel->vis.global.offwidth = options.maxWidth;
  privateModel->vis.global.offheight = options.maxHeight;
  privateModel->vis.quality.offsamples = options.msaaSamples;
  privateModel->vis.quality.shadowsize = options.shadowSize;
  return privateModel;
}

}  // namespace robot::mujoco_sim_interface
