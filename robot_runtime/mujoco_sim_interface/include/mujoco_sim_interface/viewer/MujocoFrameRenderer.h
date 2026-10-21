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

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"
#include "mujoco_sim_interface/visualization/MujocoVisualization.h"
#include "robot_core/FPSTracker.h"

namespace robot::mujoco_sim_interface {

class MujocoSimInterface;

/** The geometry groups the number keys toggle: '0' to '5' (mjNGROUP of them). */
inline constexpr int kNumToggledGeomGroups = mjNGROUP;

/**
 * One frame of the MuJoCo viewer, drawn into whichever OpenGL framebuffer is current: the code the GLFW window
 * (GlfwMujocoViewer) and the browser viewer (WebMujocoViewer) share. It owns the camera, the visualization options, the
 * abstract scene, the MuJoCo GPU context, the state snapshot it draws from and the visualizations of the simulator's
 * `sim_visualizations` (MujocoVisualization), and it carries out the viewer's keys and mouse moves.
 *
 * Usage, on one thread that has an OpenGL context current: Create(), then drawFrame() per frame with input in between,
 * then release() before the context goes. Not thread-safe: everything runs on that thread.
 */
class MujocoFrameRenderer {
 public:
  /**
   * Builds the renderer of `sim`'s visualizations and sets up the camera, the options, the scene and the GPU context for
   * `model`, takes the first snapshot and computes it. Requires an OpenGL context current on this thread, and fails with
   * FailedPrecondition when MuJoCo could not make its GPU context in it (no context current). `model` is a private copy
   * of the simulator's model (the web viewer's copyModelForOffscreenRendering(), the GLFW viewer's mj_copyModel()) that
   * only this renderer's thread touches, and the one every frame draws with: before computing a snapshot, every frame
   * writes the model arrays the simulator changed while it ran into it (MjState::modelFields), so the contacts and
   * forces it draws are the physics'. `sim` and `model` must outlive the renderer: every frame reads `sim`'s latest state
   * snapshot (MujocoSimInterface::readLatestMjState) and its visualizations read it.
   */
  static absl::StatusOr<std::unique_ptr<MujocoFrameRenderer>> Create(const MujocoSimInterface* absl_nonnull sim,
                                                                     mjModel* absl_nonnull model);

  MujocoFrameRenderer(const MujocoFrameRenderer&) = delete;
  MujocoFrameRenderer& operator=(const MujocoFrameRenderer&) = delete;
  MujocoFrameRenderer(MujocoFrameRenderer&&) = delete;
  MujocoFrameRenderer& operator=(MujocoFrameRenderer&&) = delete;

  /** Frees the abstract scene if release() did not; the GPU context is release()'s, on the context's thread. */
  ~MujocoFrameRenderer();

  /**
   * Draws the newest snapshot of the simulator into `viewport` of the current framebuffer: the scene, the visualizations'
   * decor geoms and their overlays. `elapsedRealTime` [s] is the wall-clock time since the viewer started.
   */
  void drawFrame(const mjrRect& viewport, double elapsedRealTime);

  /** Renders the abstract scene as it stands into `viewport`, without a new snapshot or the overlays (the first frame). */
  void renderScene(const mjrRect& viewport);

  /** Moves the camera as MuJoCo's mouse `action`, by fractions of the viewport's width and height (mjv_moveCamera). */
  void moveCamera(mjtMouse action, double relativeDx, double relativeDy);

  /**
   * Zooms as a scroll wheel turned by `notches` (positive away from the user, as GLFW's yoffset), 5% per notch:
   * positive notches zoom out (the camera distance grows), as in MuJoCo's own viewers.
   */
  void scroll(double notches);

  /**
   * Carries out a viewer key, by its lower-case character: '0' to '5' toggle the geometry groups, 'k' the camera
   * tracking, 'p' prints the cheatsheet, and every letter toggles the visualization that declares it as its hotkey.
   * Any other character is ignored.
   */
  void handleKey(char key);

  /** Toggles the visualization registered as `name`; false when this viewer has none of that name. */
  bool toggleVisualization(absl::string_view name);

  /** Switches between following the robot's floating base (mjCAMERA_TRACKING) and the free camera (mjCAMERA_FREE). */
  void toggleCameraTracking();

  /** Sets the camera to `type`, mjCAMERA_TRACKING or mjCAMERA_FREE; any other type is ignored. */
  void setCameraType(mjtCamera type);

  /** Puts the camera back where Create() put it: tracking the floating base from the robot's left. */
  void resetCamera();

  /** Logs the hotkeys, the mouse bindings and the state of every visualization. */
  void printHotkeys() const;

  /**
   * Reads `viewport` of the current framebuffer into `rgb`: 3 bytes per pixel, rows bottom-up (mjr_readPixels). Fails
   * with InvalidArgument when `rgb` does not hold exactly width x height x 3 bytes.
   */
  absl::Status readPixels(const mjrRect& viewport, absl::Span<uint8_t> rgb) const;

  /** Makes the offscreen framebuffer of the GPU context current (mjr_setBuffer), as a context without a window needs. */
  void useOffscreenBuffer();

  /**
   * Resizes the offscreen framebuffer to `width` x `height` (mjr_resizeOffscreen) and makes it current; nothing when it
   * has that size. The caller keeps the size within the limits the GL preflight checked (checkGlCapabilities()), or
   * MuJoCo ends the process.
   */
  void resizeOffscreen(int width, int height);

  /**
   * Frees the GPU context and the scene, after which the renderer draws nothing more. Requires the OpenGL context of
   * Create() current on this thread; nothing the second time.
   */
  void release();

  const mjvCamera& camera() const { return camera_; }
  const mjvOption& options() const { return options_; }
  const std::vector<std::unique_ptr<MujocoVisualization>>& visualizations() const { return visualizations_; }
  /** [s] Simulation time of the snapshot the last frame drew. */
  double snapshotTime() const;
  /** Frames per second of drawFrame(), smoothed. */
  double fps() const { return fps_.fps(); }

 private:
  /** Builds the visualizations of `sim`'s configuration; does no OpenGL work (Create()). */
  MujocoFrameRenderer(const MujocoSimInterface* absl_nonnull sim, mjModel* absl_nonnull model);

  /** The OpenGL set-up of Create(): FailedPrecondition when MuJoCo made no GPU context. */
  absl::Status setUp();

  const MujocoSimInterface* absl_nonnull sim_;
  /// The model of Create(), writable: every snapshot's model fields are written into it; null after release().
  mjModel* absl_nullable model_;
  MjState state_;
  std::vector<std::unique_ptr<MujocoVisualization>> visualizations_;

  // MuJoCo's visualization structures, zeroed until Create() fills them in.
  mjvCamera camera_{};
  mjvOption options_{};
  mjvScene scene_{};
  mjrContext context_{};
  bool sceneAllocated_ = false;

  FPSTracker fps_;
};

/**
 * A copy of `model` for a viewer that renders offscreen, so that its render thread never reads the simulator's model
 * while the physics thread writes it: the offscreen buffer sized `options.maxWidth` x `options.maxHeight`, with
 * `options.msaaSamples` samples and shadow maps of `options.shadowSize`. Fails with Internal when MuJoCo cannot copy it.
 */
absl::StatusOr<MjModelPtr> copyModelForOffscreenRendering(const mjModel* absl_nonnull model, const WebViewerOptions& options);

}  // namespace robot::mujoco_sim_interface
