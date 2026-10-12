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

#include <string>
#include <utility>

#include "EGL/egl.h"
#include "EGL/eglext.h"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace robot::mujoco_sim_interface {

/**
 * The EGL library the web viewer renders through, loaded at run time with dlopen() rather than linked, so that the
 * robot binary needs no EGL (and no Mesa) to start and a machine without EGL gets a Status that names the fix.
 *
 * Get() loads a library once per name and keeps it for the life of the process: it is never dlclose()d, because glvnd
 * and Mesa keep destructors and thread-local state that are unsafe to unload, and MuJoCo caches the OpenGL entry
 * points process-wide. Thread-safe: Get() may be called from any thread, and a library is immutable once loaded.
 *
 * Only the EGL code of the web viewer (EglContext.cpp) and its tests include this header; the EGL headers come with
 * EGL_NO_X11 (the egl_headers target), so that no translation unit sees Xlib through them.
 */
class EglLibrary {
 public:
  /**
   * The EGL entry points the viewer uses, resolved through eglGetProcAddress. The core ones are never null in a loaded
   * library (Get() fails instead); the extensions are null where the library has none. (The PFN types are pointer
   * typedefs, which carry no nullability annotation.)
   */
  struct Functions {
    PFNEGLGETPROCADDRESSPROC getProcAddress;
    PFNEGLGETERRORPROC getError;
    PFNEGLQUERYSTRINGPROC queryString;
    PFNEGLINITIALIZEPROC initialize;
    PFNEGLBINDAPIPROC bindApi;
    PFNEGLCHOOSECONFIGPROC chooseConfig;
    PFNEGLCREATECONTEXTPROC createContext;
    PFNEGLDESTROYCONTEXTPROC destroyContext;
    PFNEGLCREATEPBUFFERSURFACEPROC createPbufferSurface;
    PFNEGLDESTROYSURFACEPROC destroySurface;
    PFNEGLMAKECURRENTPROC makeCurrent;
    // Extensions (EGL_EXT_platform_base, EGL_EXT_device_enumeration, EGL_EXT_device_query).
    PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatformDisplay;
    PFNEGLQUERYDEVICESEXTPROC queryDevices;
    PFNEGLQUERYDEVICESTRINGEXTPROC queryDeviceString;
  };

  /**
   * The EGL library `soname` (`libEGL.so.1`, WebViewerOptions::eglLibrary), loaded on the first call for that name and
   * the same object on every later one. Fails with Unavailable when it cannot be loaded or lacks a core entry point,
   * saying how to install it or run without the web viewer; a failure is not remembered, so a later call tries again.
   */
  static absl::StatusOr<const EglLibrary* absl_nonnull> Get(absl::string_view soname);

  EglLibrary(const EglLibrary&) = delete;
  EglLibrary& operator=(const EglLibrary&) = delete;
  EglLibrary(EglLibrary&&) = delete;
  EglLibrary& operator=(EglLibrary&&) = delete;
  ~EglLibrary() = default;

  const Functions& functions() const { return functions_; }
  const std::string& soname() const { return soname_; }

 private:
  EglLibrary(std::string soname, const Functions& functions) : soname_(std::move(soname)), functions_(functions) {}

  /** Loads `soname` and resolves its entry points (Get()). */
  static absl::StatusOr<Functions> load(const std::string& soname);

  std::string soname_;
  Functions functions_;
};

}  // namespace robot::mujoco_sim_interface
