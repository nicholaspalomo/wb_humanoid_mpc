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

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/viewer/EglContext.h"
#include "mujoco_sim_interface/viewer/MujocoFrameRenderer.h"
#include "mujoco_sim_interface/viewer/MujocoViewer.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"
#include "web_viewer/JpegEncoder.h"
#include "web_viewer/ViewerFrameMailbox.h"
#include "web_viewer/ViewerInput.h"
#include "web_viewer/ViewerInputQueue.h"
#include "web_viewer/ViewerStatus.h"
#include "web_viewer/WebViewerServer.h"

namespace robot::mujoco_sim_interface {

class EglLibrary;
class MujocoSimInterface;

/**
 * The `web` viewer: the simulation in a browser tab. A render thread of its own
 * draws the simulator's newest snapshot offscreen through EGL (no X server:
 * Mesa's llvmpipe, or a GPU), encodes it as JPEG and hands it to an HTTP server
 * (robot_runtime/web_viewer) that browsers pull frames from and send their
 * camera input to, at http://<options.address>:<options.port>/. Nothing a
 * browser sends reaches the physics, the controller or the gantry.
 *
 * Rendering is demand-driven: with no browser watching, nothing is drawn. The
 * render thread reads the simulator's state snapshots only
 * (MujocoSimInterface::readLatestMjState) and draws with a private copy of its
 * model, which the physics thread never writes and each frame brings up to date
 * with the snapshot's run-time model arrays (MjState::modelFields), and places
 * itself and every thread it starts on `options.cores` at
 * `options.threadPriority` (SCHED_IDLE: a physics or MPC thread waking on those
 * cores preempts them at once), before any EGL call (llvmpipe's threads inherit
 * both), as the server does with its own. One Mesa thread places itself
 * instead: the shader disk cache's queue (`disk$0`) runs on every core, those
 * of the physics and the MPC included, at nice 19.
 *
 * Lifecycle: Create() (on the simulator's thread, before physics starts),
 * start() once, destroy. Thread-safe: the const methods may be called from any
 * thread.
 */
class WebMujocoViewer final : public MujocoViewer {
 public:
  /**
   * Copies `sim`'s model, loads EGL and starts serving (the server binds
   * `options.address`:`options.port` and listens), without drawing yet. `sim`
   * must outlive the viewer. Fails with Unavailable when EGL cannot be loaded
   * or the port cannot be bound, with InvalidArgument for options the server
   * refuses, and with Internal when the model cannot be copied.
   */
  static absl::StatusOr<std::unique_ptr<WebMujocoViewer>> Create(
      const MujocoSimInterface* absl_nonnull sim,
      const WebViewerOptions& options);

  WebMujocoViewer(const WebMujocoViewer&) = delete;
  WebMujocoViewer& operator=(const WebMujocoViewer&) = delete;
  WebMujocoViewer(WebMujocoViewer&&) = delete;
  WebMujocoViewer& operator=(WebMujocoViewer&&) = delete;

  /** Closes the mailbox (waiting requests return), stops and joins the server,
   * then the render thread. */
  ~WebMujocoViewer() override;

  absl::string_view name() const override { return kWebViewerName; }

  /**
   * Starts the render thread and waits up to options.startTimeout for its EGL
   * and OpenGL set-up to succeed. Fails with what the set-up returned (no EGL
   * display, FailedPrecondition from the GL preflight checkGlCapabilities()),
   * or with DeadlineExceeded on the timeout; in every failure the render thread
   * has ended before start() returns. The set-up looks for the stop only
   * between its steps, so on the timeout start() logs that it stops the render
   * thread and returns once the step running at the deadline and the release of
   * what the set-up made have ended: later than options.startTimeout by that
   * step, which a loaded machine can stretch to seconds. Call it once.
   */
  absl::Status start() override;

  /** False once the render thread has stopped: its set-up failed, or it could
   * not draw. */
  bool ok() const override;

  /** The port the server bound. */
  std::optional<int> port() const override;

  /** The address to open the viewer at, such as "http://127.0.0.1:6090/". */
  std::string url() const;

 private:
  friend class WebMujocoViewerTestPeer;

  WebMujocoViewer(const MujocoSimInterface* absl_nonnull sim,
                  const WebViewerOptions& options,
                  MjModelPtr model,
                  const EglLibrary* absl_nonnull egl);

  void renderLoop();
  /** Asks the render thread to stop after a start-up timeout and joins it,
   * logging while it waits. */
  void stopAfterStartUpTimeout();
  /** Asks the render thread to stop, and ends its frame-period pause at once.
   */
  void requestStop();
  /** Logs `failure`, publishes a status that renders nothing, and closes the
   * mailbox, so frame requests answer 503. */
  void stopServingAfterRenderFailure(const absl::Status& failure);
  /** The render thread's set-up: EGL context, GL preflight, MuJoCo's GPU
   * context and the JPEG encoder. */
  absl::Status initializeRendering();
  /** Frees what initializeRendering() made, on the render thread. */
  void releaseRendering();
  /** Applies the queued browser input; true when there was any. */
  bool applyInputs();
  void applyInput(const web_viewer::ViewerInput& input);
  void applySettings(const web_viewer::ViewerInput& settings);
  /** Draws, reads back, encodes and publishes one frame. */
  absl::Status renderAndPublish(double elapsedRealTime);
  web_viewer::ViewerStatus currentStatus() const;
  /**
   * The status before the render thread's set-up: the configured size, rate and
   * quality, the markers, and the default (tracking) camera. currentStatus()
   * reads MuJoCo's camera and options, which the set-up initializes.
   */
  web_viewer::ViewerStatus setUpStatus() const;
  bool stopRequested() const {
    return stopRequested_.load(std::memory_order_acquire);
  }

  const MujocoSimInterface* absl_nonnull sim_;
  const WebViewerOptions options_;
  /// The private model copy the render thread draws with
  /// (copyModelForOffscreenRendering()), and writes each snapshot's run-time
  /// model arrays into; no other thread touches it after Create().
  MjModelPtr model_;
  const EglLibrary* absl_nonnull egl_;

  web_viewer::ViewerFrameMailbox mailbox_;
  web_viewer::ViewerInputQueue inputs_;
  /// Null until Create() has started it; destroyed before mailbox_ and inputs_,
  /// which it reads.
  std::unique_ptr<web_viewer::WebViewerServer> server_;

  std::thread renderThread_;
  std::atomic<bool> stopRequested_ = false;
  std::atomic<bool> stopped_ = false;  // see ok()
  absl::Notification initialized_;
  absl::Notification renderLoopEnded_;   // the render thread's last act
  absl::Notification stopNotification_;  // notified once by requestStop(); the
                                         // render loop's pause waits on it
  absl::Mutex initMutex_;
  absl::Status initStatus_ ABSL_GUARDED_BY(initMutex_);
  std::string renderingDescription_ ABSL_GUARDED_BY(
      initMutex_);  // the renderer and the EGL display, for the log
  /// For WebMujocoViewerTestPeer: how long the render thread waits before its
  /// set-up.
  absl::Duration initDelayForTesting_ = absl::ZeroDuration();
  /// For WebMujocoViewerTestPeer: how long the render thread sleeps before its
  /// set-up, without looking for a stop.
  absl::Duration uninterruptibleInitDelayForTesting_ = absl::ZeroDuration();
  /// For WebMujocoViewerTestPeer: makes the next renderAndPublish() fail.
  std::atomic<bool> failNextFrameForTesting_ = false;

  /// The visualizations the configuration lists, as they start: for the server
  /// and the status before the set-up.
  const std::vector<web_viewer::ViewerStatus::Visualization>
      configuredVisualizations_;
  // Render thread only, from initializeRendering() to releaseRendering(); null
  // before and after.
  std::unique_ptr<MujocoFrameRenderer> frameRenderer_;
  std::unique_ptr<EglContext> context_;
  std::unique_ptr<web_viewer::JpegEncoder> encoder_;
  std::vector<uint8_t> rgb_;  // width_ x height_ x 3, rows bottom-up
  int width_ = 0;
  int height_ = 0;
  int fps_ = 0;
  int quality_ = 0;
};

}  // namespace robot::mujoco_sim_interface
