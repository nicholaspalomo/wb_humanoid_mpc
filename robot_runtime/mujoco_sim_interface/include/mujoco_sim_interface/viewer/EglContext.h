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

#include <memory>
#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "mujoco_sim_interface/viewer/GlCapabilities.h"

namespace robot::mujoco_sim_interface {

class EglLibrary;

/**
 * A desktop OpenGL context without a window or an X server, made through EGL for offscreen rendering
 * (WebMujocoViewer). Create() picks the display: an EGL device (a GPU, or the one `deviceId` names), else Mesa's
 * surfaceless platform (llvmpipe); EGL_DEFAULT_DISPLAY is never used, because it needs DISPLAY or falls back to X.
 *
 * Not thread-safe. The context is current on at most one thread at a time: makeCurrent() and release() may run on any
 * thread, and each binds the OpenGL API first, since EGL's bound API is per-thread state. The destructor destroys the
 * context but never terminates the display, which EGL shares between every context on it in the process.
 */
class EglContext {
 public:
  struct Options {
    std::optional<int> deviceId;  // an index of eglQueryDevicesEXT (MUJOCO_EGL_DEVICE_ID); unset picks a GPU, else surfaceless
  };

  /**
   * Initializes the display and creates a context on it, not yet current. `library` must outlive the context (every
   * EglLibrary does). Fails with InvalidArgument when `options.deviceId` is past the devices, naming their count, and
   * with Unavailable naming each display tried and its EGL error when none gives a desktop OpenGL context.
   */
  static absl::StatusOr<std::unique_ptr<EglContext>> Create(const EglLibrary* absl_nonnull library, const Options& options);

  EglContext(const EglContext&) = delete;
  EglContext& operator=(const EglContext&) = delete;
  EglContext(EglContext&&) = delete;
  EglContext& operator=(EglContext&&) = delete;

  /** Destroys the context (and its pbuffer, if any). Call release() first on the thread it is current on. */
  ~EglContext();

  /** Makes the context current on this thread, without a surface where EGL allows it (else on a 1x1 pbuffer). */
  absl::Status makeCurrent();

  /** Makes no context current on this thread. */
  absl::Status release();

  /**
   * What the context offers (glGetString, glGetIntegerv), for checkGlCapabilities(). Requires the context current on
   * this thread; fails with Unavailable when the library cannot resolve the OpenGL queries.
   */
  absl::StatusOr<GlCapabilities> queryGlCapabilities() const;

  /** The display and its EGL implementation, e.g. "surfaceless (Mesa Project, EGL 1.5)", for the start-up log. */
  const std::string& description() const { return description_; }

 private:
  EglContext(const EglLibrary* absl_nonnull library,
             void* absl_nonnull display,
             void* absl_nonnull context,
             void* absl_nullable surface,
             std::string description);

  const EglLibrary* absl_nonnull library_;
  // EGL's handles (EGLDisplay, EGLContext, EGLSurface are void*), kept as void* so that no viewer header includes EGL.
  void* absl_nonnull display_;
  void* absl_nonnull context_;
  void* absl_nullable surface_;  // a 1x1 pbuffer where the display lacks EGL_KHR_surfaceless_context; null otherwise
  std::string description_;
};

}  // namespace robot::mujoco_sim_interface
