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

#include "mujoco_sim_interface/viewer/EglLibrary.h"

#include <dlfcn.h>

#include <memory>
#include <string>
#include <utility>

#include "EGL/egl.h"
#include "EGL/eglext.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "mujoco_sim_interface/viewer/ViewerNames.h"

namespace robot::mujoco_sim_interface {
namespace {

using LibraryTable =
    absl::flat_hash_map<std::string, std::unique_ptr<const EglLibrary>>;

/** Every EGL library loaded so far, by name; never shrinks (see the class
 * comment). */
struct LoadedLibraries {
  absl::Mutex mutex;
  LibraryTable byName ABSL_GUARDED_BY(mutex);
};

LoadedLibraries& loadedLibraries() {
  static absl::NoDestructor<LoadedLibraries> libraries;
  return *libraries;
}

/** What to do when the web viewer has no EGL, appended to every failure of
 * Get(). */
std::string howToGetEgl() {
  return absl::StrCat(
      "install libegl1 and libegl-mesa0 (the dev and robot-sim images have "
      "them), use a GPU container with NVIDIA's EGL, "
      "or choose --viewer=",
      kNoViewerName, " (or list ", kNoViewerName, " after ", kWebViewerName,
      ", --viewer=", kWebViewerName, ",", kNoViewerName, ", to fall back)");
}

/** The entry point `name` of the library, as the function pointer type
 * `Function`; null when it has none. */
template <typename Function>
Function resolve(PFNEGLGETPROCADDRESSPROC getProcAddress,
                 const char* absl_nonnull name) {
  return reinterpret_cast<Function>(getProcAddress(name));
}

}  // namespace

absl::StatusOr<EglLibrary::Functions> EglLibrary::load(
    const std::string& soname) {
  // RTLD_LOCAL: the library's symbols stay out of the global namespace, so that
  // they cannot satisfy another library's lookups. MuJoCo still finds the GL of
  // the context made current through it (dl_iterate_phdr sees local objects).
  void* absl_nullable handle = dlopen(soname.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* absl_nullable error = dlerror();
    return absl::UnavailableError(absl::StrCat(
        "the web viewer renders through EGL, and ", soname,
        " could not be loaded (", error != nullptr ? error : "no reason given",
        "): ", howToGetEgl()));
  }
  PFNEGLGETPROCADDRESSPROC getProcAddress =
      reinterpret_cast<PFNEGLGETPROCADDRESSPROC>(
          dlsym(handle, "eglGetProcAddress"));
  if (getProcAddress == nullptr) {
    return absl::UnavailableError(absl::StrCat(
        soname, " exports no eglGetProcAddress, so it is no EGL library: ",
        howToGetEgl()));
  }
  PFNEGLGETERRORPROC getError =
      resolve<PFNEGLGETERRORPROC>(getProcAddress, "eglGetError");
  PFNEGLQUERYSTRINGPROC queryString =
      resolve<PFNEGLQUERYSTRINGPROC>(getProcAddress, "eglQueryString");
  PFNEGLINITIALIZEPROC initialize =
      resolve<PFNEGLINITIALIZEPROC>(getProcAddress, "eglInitialize");
  PFNEGLBINDAPIPROC bindApi =
      resolve<PFNEGLBINDAPIPROC>(getProcAddress, "eglBindAPI");
  PFNEGLCHOOSECONFIGPROC chooseConfig =
      resolve<PFNEGLCHOOSECONFIGPROC>(getProcAddress, "eglChooseConfig");
  PFNEGLCREATECONTEXTPROC createContext =
      resolve<PFNEGLCREATECONTEXTPROC>(getProcAddress, "eglCreateContext");
  PFNEGLDESTROYCONTEXTPROC destroyContext =
      resolve<PFNEGLDESTROYCONTEXTPROC>(getProcAddress, "eglDestroyContext");
  PFNEGLCREATEPBUFFERSURFACEPROC createPbufferSurface =
      resolve<PFNEGLCREATEPBUFFERSURFACEPROC>(getProcAddress,
                                              "eglCreatePbufferSurface");
  PFNEGLDESTROYSURFACEPROC destroySurface =
      resolve<PFNEGLDESTROYSURFACEPROC>(getProcAddress, "eglDestroySurface");
  PFNEGLMAKECURRENTPROC makeCurrent =
      resolve<PFNEGLMAKECURRENTPROC>(getProcAddress, "eglMakeCurrent");
  if (getError == nullptr || queryString == nullptr || initialize == nullptr ||
      bindApi == nullptr || chooseConfig == nullptr ||
      createContext == nullptr || destroyContext == nullptr ||
      createPbufferSurface == nullptr || destroySurface == nullptr ||
      makeCurrent == nullptr) {
    return absl::UnavailableError(absl::StrCat(
        soname, " lacks a core EGL 1.4 entry point: ", howToGetEgl()));
  }
  return Functions{
      .getProcAddress = getProcAddress,
      .getError = getError,
      .queryString = queryString,
      .initialize = initialize,
      .bindApi = bindApi,
      .chooseConfig = chooseConfig,
      .createContext = createContext,
      .destroyContext = destroyContext,
      .createPbufferSurface = createPbufferSurface,
      .destroySurface = destroySurface,
      .makeCurrent = makeCurrent,
      .getPlatformDisplay = resolve<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
          getProcAddress, "eglGetPlatformDisplayEXT"),
      .queryDevices = resolve<PFNEGLQUERYDEVICESEXTPROC>(getProcAddress,
                                                         "eglQueryDevicesEXT"),
      .queryDeviceString = resolve<PFNEGLQUERYDEVICESTRINGEXTPROC>(
          getProcAddress, "eglQueryDeviceStringEXT"),
  };
}

absl::StatusOr<const EglLibrary* absl_nonnull> EglLibrary::Get(
    absl::string_view soname) {
  LoadedLibraries& libraries = loadedLibraries();
  absl::MutexLock lock(libraries.mutex);
  const std::pair<LibraryTable::iterator, bool> entry =
      libraries.byName.try_emplace(soname);
  if (!entry.second) return entry.first->second.get();
  absl::StatusOr<Functions> functions = load(entry.first->first);
  if (!functions.ok()) {
    libraries.byName.erase(
        entry.first);  // not remembered: a later call tries again
    return functions.status();
  }
  entry.first->second =
      absl::WrapUnique(new EglLibrary(entry.first->first, *functions));
  return entry.first->second.get();
}

}  // namespace robot::mujoco_sim_interface
