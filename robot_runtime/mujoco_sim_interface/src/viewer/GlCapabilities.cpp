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

#include "mujoco_sim_interface/viewer/GlCapabilities.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"

namespace robot::mujoco_sim_interface {
namespace {

/** An OpenGL version as GL_VERSION begins with it: "major.minor[.release] [vendor information]". */
struct GlVersionNumber {
  int major = 0;
  int minor = 0;
};

/** The version number GL_VERSION starts with; 0.0 when it starts with none. */
GlVersionNumber parseGlVersionNumber(absl::string_view version) {
  const absl::string_view stripped = absl::StripAsciiWhitespace(version);
  const absl::string_view number = stripped.substr(0, stripped.find(' '));
  const std::vector<absl::string_view> parts = absl::StrSplit(number, '.');
  GlVersionNumber parsed;
  if (parts.size() < 2 || !absl::SimpleAtoi(parts[0], &parsed.major) || !absl::SimpleAtoi(parts[1], &parsed.minor)) return {};
  return parsed;
}

bool atLeast(const GlVersionNumber& version, int major, int minor) {
  return version.major > major || (version.major == major && version.minor >= minor);
}

bool hasExtension(absl::string_view extensions, absl::string_view name) {
  for (const absl::string_view extension : absl::StrSplit(extensions, ' ', absl::SkipEmpty())) {
    if (extension == name) return true;
  }
  return false;
}

/** The end of every refusal: the context it is about and how to run without the web viewer. */
std::string refusalSuffix(const GlCapabilities& capabilities) {
  return absl::StrCat(" (OpenGL '", capabilities.version, "' of '", capabilities.renderer, "'); choose --viewer=", kNoViewerName,
                      ", or list ", kNoViewerName, " after ", kWebViewerName, " (--viewer=", kWebViewerName, ",", kNoViewerName,
                      ") to fall back");
}

/** OK when the offscreen buffer's side `pixels`, set by the option `option`, fits GL_MAX_RENDERBUFFER_SIZE. */
absl::Status checkRenderbufferSide(absl::string_view option, int pixels, const GlCapabilities& capabilities) {
  if (pixels <= capabilities.maxRenderbufferSize) return absl::OkStatus();
  return absl::FailedPreconditionError(absl::StrCat("the web viewer's ", option, " of ", pixels,
                                                    " px exceeds the context's GL_MAX_RENDERBUFFER_SIZE of ",
                                                    capabilities.maxRenderbufferSize, " px; lower it", refusalSuffix(capabilities)));
}

}  // namespace

absl::Status checkGlCapabilities(const GlCapabilities& capabilities, const WebViewerOptions& options) {
  const GlVersionNumber version = parseGlVersionNumber(capabilities.version);
  if (!atLeast(version, /*major=*/1, /*minor=*/5)) {
    return absl::FailedPreconditionError(absl::StrCat("MuJoCo's offscreen rendering needs OpenGL 1.5 or newer, and the context offers ",
                                                      version.major, ".", version.minor, refusalSuffix(capabilities)));
  }
  if (!atLeast(version, /*major=*/3, /*minor=*/0) && !hasExtension(capabilities.extensions, "GL_ARB_framebuffer_object")) {
    return absl::FailedPreconditionError(
        absl::StrCat("MuJoCo's offscreen rendering needs framebuffer objects (OpenGL 3.0, or GL_ARB_framebuffer_object), and the "
                     "context has neither",
                     refusalSuffix(capabilities)));
  }
  if (absl::Status width = checkRenderbufferSide("maxWidth", options.maxWidth, capabilities); !width.ok()) {
    return width;
  }
  if (absl::Status height = checkRenderbufferSide("maxHeight", options.maxHeight, capabilities); !height.ok()) {
    return height;
  }
  if (options.msaaSamples > 0 && options.msaaSamples > capabilities.maxSamples) {
    return absl::FailedPreconditionError(absl::StrCat("the web viewer's msaaSamples of ", options.msaaSamples,
                                                      " exceeds the context's GL_MAX_SAMPLES of ", capabilities.maxSamples, "; lower it",
                                                      refusalSuffix(capabilities)));
  }
  return absl::OkStatus();
}

}  // namespace robot::mujoco_sim_interface
