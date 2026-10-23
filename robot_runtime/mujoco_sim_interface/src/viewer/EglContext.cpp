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

#include "mujoco_sim_interface/viewer/EglContext.h"

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "EGL/egl.h"
#include "EGL/eglext.h"
#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

#include "mujoco_sim_interface/viewer/EglLibrary.h"
#include "mujoco_sim_interface/viewer/GlCapabilities.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"

namespace robot::mujoco_sim_interface {
namespace {

// The members keep EGL's handles as void*, which they are.
static_assert(std::is_same_v<EGLDisplay, void*> && std::is_same_v<EGLContext, void*> && std::is_same_v<EGLSurface, void*>);

// The OpenGL queries of queryGlCapabilities(), resolved through eglGetProcAddress so that this file needs no GL header.
using GlGetStringFunction = const unsigned char* absl_nullable (*absl_nullable)(unsigned int name);
using GlGetIntegervFunction = void (*absl_nullable)(unsigned int name, int* absl_nonnull data);
using GlGetErrorFunction = unsigned int (*absl_nullable)();
// The OpenGL enumerants they take (the values of GL/gl.h and glext.h).
constexpr unsigned int kGlVendor = 0x1F00;
constexpr unsigned int kGlRenderer = 0x1F01;
constexpr unsigned int kGlVersion = 0x1F02;
constexpr unsigned int kGlExtensions = 0x1F03;
constexpr unsigned int kGlMaxRenderbufferSize = 0x84E8;
constexpr unsigned int kGlMaxSamples = 0x8D57;
constexpr unsigned int kGlMaxTextureSize = 0x0D33;
// The most OpenGL errors queryGlCapabilities() clears after its queries, so that MuJoCo's first error check sees none.
constexpr int kMaxClearedGlErrors = 16;

// The framebuffer of the EGL config: 8-bit red, green, blue and alpha, a 24-bit depth and an 8-bit stencil buffer.
constexpr EGLint kColorChannelBits = 8;
constexpr EGLint kDepthBufferBits = 24;
constexpr EGLint kStencilBufferBits = 8;
// The attribute lists below: key-value pairs, then EGL_NONE.
constexpr int kConfigAttributeListSize = 17;
constexpr int kPbufferAttributeListSize = 5;

std::string eglErrorName(EGLint error) {
  switch (error) {
    case EGL_SUCCESS:
      return "EGL_SUCCESS";
    case EGL_NOT_INITIALIZED:
      return "EGL_NOT_INITIALIZED";
    case EGL_BAD_ACCESS:
      return "EGL_BAD_ACCESS";
    case EGL_BAD_ALLOC:
      return "EGL_BAD_ALLOC";
    case EGL_BAD_ATTRIBUTE:
      return "EGL_BAD_ATTRIBUTE";
    case EGL_BAD_CONFIG:
      return "EGL_BAD_CONFIG";
    case EGL_BAD_CONTEXT:
      return "EGL_BAD_CONTEXT";
    case EGL_BAD_CURRENT_SURFACE:
      return "EGL_BAD_CURRENT_SURFACE";
    case EGL_BAD_DISPLAY:
      return "EGL_BAD_DISPLAY";
    case EGL_BAD_MATCH:
      return "EGL_BAD_MATCH";
    case EGL_BAD_PARAMETER:
      return "EGL_BAD_PARAMETER";
    case EGL_BAD_SURFACE:
      return "EGL_BAD_SURFACE";
    case EGL_CONTEXT_LOST:
      return "EGL_CONTEXT_LOST";
    case EGL_BAD_DEVICE_EXT:
      return "EGL_BAD_DEVICE_EXT";
    default:
      return absl::StrFormat("EGL error 0x%04x", error);
  }
}

bool hasExtension(const char* absl_nullable extensions, absl::string_view name) {
  if (extensions == nullptr) return false;
  for (const absl::string_view extension : absl::StrSplit(extensions, ' ', absl::SkipEmpty())) {
    if (extension == name) return true;
  }
  return false;
}

/** The OpenGL string `name`, through `getString` (glGetString, never null here); empty where the context has none. */
std::string glString(GlGetStringFunction getString, unsigned int name) {
  const unsigned char* absl_nullable value = getString(name);
  return value != nullptr ? std::string(reinterpret_cast<const char*>(value)) : std::string();
}

std::string orUnknown(const char* absl_nullable text) {
  return text != nullptr ? std::string(text) : std::string("unknown");
}

/** A display EGL has initialized, and what it is. */
struct InitializedDisplay {
  void* absl_nonnull display;
  std::string description;
};

/** Initializes `display`, found on `platform`; appends why to `failures` and returns none when it fails. */
std::optional<InitializedDisplay> initializeDisplay(const EglLibrary::Functions& egl,
                                                    EGLDisplay display,
                                                    absl::string_view platform,
                                                    std::vector<std::string>& failures) {
  if (display == EGL_NO_DISPLAY) {
    failures.push_back(absl::StrCat(platform, ": no display (", eglErrorName(egl.getError()), ")"));
    return std::nullopt;
  }
  EGLint major = 0;
  EGLint minor = 0;
  if (egl.initialize(display, &major, &minor) != EGL_TRUE) {
    failures.push_back(absl::StrCat(platform, ": eglInitialize failed (", eglErrorName(egl.getError()), ")"));
    return std::nullopt;
  }
  return InitializedDisplay{
      .display = display,
      .description = absl::StrCat(platform, " (", orUnknown(egl.queryString(display, EGL_VENDOR)), ", EGL ", major, ".", minor, ")")};
}

/**
 * The display of an EGL device: the one `deviceId` names, else the first that is not Mesa's software device. None, with
 * the reason appended to `failures`, when there is no such device; InvalidArgument when `deviceId` is past the devices.
 */
absl::StatusOr<std::optional<InitializedDisplay>> deviceDisplay(const EglLibrary::Functions& egl,
                                                                std::optional<int> deviceId,
                                                                std::vector<std::string>& failures) {
  const char* absl_nullable clientExtensions = egl.queryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
  if (egl.getPlatformDisplay == nullptr || egl.queryDevices == nullptr || egl.queryDeviceString == nullptr ||
      !hasExtension(clientExtensions, "EGL_EXT_platform_device")) {
    if (deviceId.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat("EGL device ", *deviceId,
                                                     " does not exist: EGL reports 0 device(s), since it offers no device platform "
                                                     "(EGL_EXT_platform_device); unset MUJOCO_EGL_DEVICE_ID to let the viewer choose"));
    }
    failures.push_back("device platform: not offered (EGL_EXT_platform_device, EGL_EXT_device_enumeration)");
    return std::nullopt;
  }
  EGLint count = 0;
  if (egl.queryDevices(/*max_devices=*/0, /*devices=*/nullptr, &count) != EGL_TRUE) count = 0;
  std::vector<EGLDeviceEXT> devices(static_cast<size_t>(std::max<EGLint>(count, 0)));
  if (count > 0 && egl.queryDevices(count, devices.data(), &count) != EGL_TRUE) count = 0;
  devices.resize(static_cast<size_t>(std::max<EGLint>(count, 0)));
  if (deviceId.has_value() && *deviceId >= static_cast<int>(devices.size())) {
    return absl::InvalidArgumentError(absl::StrCat("EGL device ", *deviceId, " does not exist: EGL reports ", devices.size(),
                                                   " device(s); unset MUJOCO_EGL_DEVICE_ID to let the viewer choose"));
  }
  std::optional<int> chosen = deviceId;
  for (int index = 0; !chosen.has_value() && index < static_cast<int>(devices.size()); ++index) {
    if (!hasExtension(egl.queryDeviceString(devices[index], EGL_EXTENSIONS), "EGL_MESA_device_software")) chosen = index;
  }
  if (!chosen.has_value()) {
    failures.push_back(absl::StrCat("device platform: ", devices.size(), " device(s), none of them a GPU"));
    return std::nullopt;
  }
  const std::string platform = absl::StrCat("EGL device ", *chosen);
  return initializeDisplay(egl, egl.getPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devices[*chosen], /*attrib_list=*/nullptr), platform,
                           failures);
}

/** The display of Mesa's surfaceless platform (llvmpipe, or a render node); none, with the reason in `failures`. */
std::optional<InitializedDisplay> surfacelessDisplay(const EglLibrary::Functions& egl, std::vector<std::string>& failures) {
  if (egl.getPlatformDisplay == nullptr ||
      !hasExtension(egl.queryString(EGL_NO_DISPLAY, EGL_EXTENSIONS), "EGL_MESA_platform_surfaceless")) {
    failures.push_back("surfaceless platform: not offered (EGL_MESA_platform_surfaceless)");
    return std::nullopt;
  }
  return initializeDisplay(egl, egl.getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, /*attrib_list=*/nullptr),
                           "surfaceless", failures);
}

std::string howToRunWithoutEgl() {
  return absl::StrCat("use a machine with Mesa's EGL (libegl-mesa0) or a GPU, or choose --viewer=", kNoViewerName,
                      " (or --viewer=", kWebViewerName, ",", kNoViewerName, " to fall back)");
}

}  // namespace

absl::StatusOr<std::unique_ptr<EglContext>> EglContext::Create(const EglLibrary* absl_nonnull library, const Options& options) {
  const EglLibrary::Functions& egl = library->functions();
  std::vector<std::string> failures;
  absl::StatusOr<std::optional<InitializedDisplay>> device = deviceDisplay(egl, options.deviceId, failures);
  if (!device.ok()) return device.status();
  std::optional<InitializedDisplay> display = *std::move(device);
  // An explicit device is that device or nothing; without one, llvmpipe is the fallback.
  if (!display.has_value() && !options.deviceId.has_value()) display = surfacelessDisplay(egl, failures);
  if (!display.has_value()) {
    return absl::UnavailableError(
        absl::StrCat("EGL offers no display to render on (", absl::StrJoin(failures, "; "), "); ", howToRunWithoutEgl()));
  }
  const InitializedDisplay& initialized = *display;

  if (egl.bindApi(EGL_OPENGL_API) != EGL_TRUE) {
    return absl::UnavailableError(absl::StrCat("EGL on ", initialized.description, " offers no desktop OpenGL (",
                                               eglErrorName(egl.getError()), "); ", howToRunWithoutEgl()));
  }
  const std::array<EGLint, kConfigAttributeListSize> attributes = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,   EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,     EGL_RED_SIZE,   kColorChannelBits,
      EGL_GREEN_SIZE,   kColorChannelBits, EGL_BLUE_SIZE,       kColorChannelBits,  EGL_ALPHA_SIZE, kColorChannelBits,
      EGL_DEPTH_SIZE,   kDepthBufferBits,  EGL_STENCIL_SIZE,    kStencilBufferBits, EGL_NONE};
  EGLConfig config = nullptr;
  EGLint numConfigs = 0;
  if (egl.chooseConfig(initialized.display, attributes.data(), &config, /*config_size=*/1, &numConfigs) != EGL_TRUE || numConfigs < 1) {
    return absl::UnavailableError(absl::StrCat("EGL on ", initialized.description, " has no RGBA8 / depth 24 / stencil 8 OpenGL config (",
                                               eglErrorName(egl.getError()), "); ", howToRunWithoutEgl()));
  }
  // No attributes: MuJoCo needs the legacy entry points of a compatibility-profile context.
  void* absl_nullable context = egl.createContext(initialized.display, config, EGL_NO_CONTEXT, /*attrib_list=*/nullptr);
  if (context == EGL_NO_CONTEXT) {
    return absl::UnavailableError(absl::StrCat("EGL on ", initialized.description, " could not create an OpenGL context (",
                                               eglErrorName(egl.getError()), "); ", howToRunWithoutEgl()));
  }
  void* absl_nullable surface = EGL_NO_SURFACE;
  if (!hasExtension(egl.queryString(initialized.display, EGL_EXTENSIONS), "EGL_KHR_surfaceless_context")) {
    const std::array<EGLint, kPbufferAttributeListSize> pbuffer = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};  // 1 x 1 px
    surface = egl.createPbufferSurface(initialized.display, config, pbuffer.data());
    if (surface == EGL_NO_SURFACE) {
      const std::string error = eglErrorName(egl.getError());
      egl.destroyContext(initialized.display, context);
      return absl::UnavailableError(absl::StrCat("EGL on ", initialized.description,
                                                 " has neither surfaceless contexts nor a pbuffer to make one current (", error, "); ",
                                                 howToRunWithoutEgl()));
    }
  }
  return absl::WrapUnique(new EglContext(library, initialized.display, context, surface, initialized.description));
}

EglContext::EglContext(const EglLibrary* absl_nonnull library,
                       void* absl_nonnull display,
                       void* absl_nonnull context,
                       void* absl_nullable surface,
                       std::string description)
    : library_(library), display_(display), context_(context), surface_(surface), description_(std::move(description)) {}

EglContext::~EglContext() {
  const EglLibrary::Functions& egl = library_->functions();
  if (surface_ != nullptr) egl.destroySurface(display_, surface_);
  egl.destroyContext(display_, context_);
  // No eglTerminate: EGL returns this same display to every context of the platform in the process (see the class).
}

absl::Status EglContext::makeCurrent() {
  const EglLibrary::Functions& egl = library_->functions();
  // The bound API is per thread, and this thread may never have bound desktop OpenGL.
  if (egl.bindApi(EGL_OPENGL_API) != EGL_TRUE) {
    return absl::UnavailableError(absl::StrCat("eglBindAPI(EGL_OPENGL_API) failed (", eglErrorName(egl.getError()), ")"));
  }
  if (egl.makeCurrent(display_, surface_, surface_, context_) != EGL_TRUE) {
    return absl::UnavailableError(
        absl::StrCat("eglMakeCurrent failed on ", description_, " (", eglErrorName(egl.getError()), "); is it current on another thread?"));
  }
  return absl::OkStatus();
}

absl::Status EglContext::release() {
  const EglLibrary::Functions& egl = library_->functions();
  if (egl.bindApi(EGL_OPENGL_API) != EGL_TRUE) {
    return absl::UnavailableError(absl::StrCat("eglBindAPI(EGL_OPENGL_API) failed (", eglErrorName(egl.getError()), ")"));
  }
  if (egl.makeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT) != EGL_TRUE) {
    return absl::UnavailableError(absl::StrCat("eglMakeCurrent(EGL_NO_CONTEXT) failed (", eglErrorName(egl.getError()), ")"));
  }
  return absl::OkStatus();
}

absl::StatusOr<GlCapabilities> EglContext::queryGlCapabilities() const {
  const EglLibrary::Functions& egl = library_->functions();
  const GlGetStringFunction getString = reinterpret_cast<GlGetStringFunction>(egl.getProcAddress("glGetString"));
  const GlGetIntegervFunction getIntegerv = reinterpret_cast<GlGetIntegervFunction>(egl.getProcAddress("glGetIntegerv"));
  const GlGetErrorFunction getError = reinterpret_cast<GlGetErrorFunction>(egl.getProcAddress("glGetError"));
  if (getString == nullptr || getIntegerv == nullptr || getError == nullptr) {
    return absl::UnavailableError(absl::StrCat(library_->soname(), " resolves no glGetString / glGetIntegerv / glGetError"));
  }
  GlCapabilities capabilities;
  capabilities.vendor = glString(getString, kGlVendor);
  capabilities.renderer = glString(getString, kGlRenderer);
  capabilities.version = glString(getString, kGlVersion);
  capabilities.extensions = glString(getString, kGlExtensions);
  getIntegerv(kGlMaxRenderbufferSize, &capabilities.maxRenderbufferSize);
  getIntegerv(kGlMaxSamples, &capabilities.maxSamples);  // GL_INVALID_ENUM, and 0 left, before OpenGL 3.0
  getIntegerv(kGlMaxTextureSize, &capabilities.maxTextureSize);
  // Clear what the queries raised, so that MuJoCo's first error check does not report it as its own.
  for (int pending = 0; pending < kMaxClearedGlErrors; ++pending) {
    if (getError() == 0) break;
  }
  if (capabilities.version.empty()) {
    return absl::UnavailableError(absl::StrCat("glGetString(GL_VERSION) returned nothing on ", description_, "; is the context current?"));
  }
  return capabilities;
}

}  // namespace robot::mujoco_sim_interface
