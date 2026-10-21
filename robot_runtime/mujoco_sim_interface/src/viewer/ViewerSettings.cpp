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

#include "mujoco_sim_interface/viewer/ViewerSettings.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"
#include "web_viewer/ViewerThreadPriority.h"

namespace robot::mujoco_sim_interface {
namespace {

// The fewest llvmpipe rasterizer threads softwareRasterizerThreads() asks for, however few cores the viewer has.
constexpr int kMinSoftwareRasterizerThreads = 2;

bool isKnownViewer(absl::string_view name) {
  for (const ViewerInfo& viewer : availableViewers()) {
    if (viewer.name == name) return true;
  }
  return false;
}

/** OK when `value` of the web viewer option `field` lies in [minimum, maximum], else InvalidArgument naming both. */
absl::Status checkRange(absl::string_view field, int value, int minimum, int maximum) {
  if (value >= minimum && value <= maximum) return absl::OkStatus();
  return absl::InvalidArgumentError(
      absl::StrCat("the web viewer's ", field, " is ", value, "; it must lie in [", minimum, ", ", maximum, "]"));
}

absl::Status checkSizes(const WebViewerOptions& web) {
  if (absl::Status status = checkRange("maxWidth", web.maxWidth, kMinWebViewerWidth, kMaxWebViewerWidth); !status.ok()) {
    return status;
  }
  if (absl::Status status = checkRange("maxHeight", web.maxHeight, kMinWebViewerHeight, kMaxWebViewerHeight); !status.ok()) {
    return status;
  }
  if (absl::Status status = checkRange("width", web.width, kMinWebViewerWidth, web.maxWidth); !status.ok()) {
    return status;
  }
  return checkRange("height", web.height, kMinWebViewerHeight, web.maxHeight);
}

absl::Status checkRendering(const WebViewerOptions& web) {
  if (absl::Status status = checkRange("fps", web.fps, kMinWebViewerFps, kMaxWebViewerFps); !status.ok()) {
    return status;
  }
  if (absl::Status status = checkRange("quality", web.quality, kMinWebViewerQuality, kMaxWebViewerQuality); !status.ok()) {
    return status;
  }
  if (absl::Status status = checkRange("msaaSamples", web.msaaSamples, /*minimum=*/0, kMaxWebViewerMsaaSamples); !status.ok()) {
    return status;
  }
  return checkRange("shadowSize", web.shadowSize, /*minimum=*/0, kMaxWebViewerShadowSize);
}

absl::Status checkServing(const WebViewerOptions& web) {
  if (web.address.empty()) return absl::InvalidArgumentError("the web viewer's address is empty; 127.0.0.1 serves this machine only");
  if (absl::Status status = checkRange("port", web.port, /*minimum=*/0, WebViewerOptions::kMaxPort); !status.ok()) {
    return status;
  }
  for (const std::string& host : web.allowedHosts) {
    if (host.empty()) return absl::InvalidArgumentError("the web viewer's allowed hosts list an empty name");
  }
  return checkRange("maxClients", web.maxClients, /*minimum=*/1, kMaxWebViewerClients);
}

bool isThreadPriority(web_viewer::ViewerThreadPriority priority) {
  switch (priority) {
    case web_viewer::ViewerThreadPriority::kIdle:
    case web_viewer::ViewerThreadPriority::kInherited:
      return true;
  }
  return false;  // a value cast from an integer
}

absl::Status checkPlacementAndEgl(const WebViewerOptions& web) {
  for (const int core : web.cores) {
    if (core < 0) return absl::InvalidArgumentError(absl::StrCat("the web viewer's cores list ", core, ", which is no CPU core"));
  }
  if (!isThreadPriority(web.threadPriority)) {
    return absl::InvalidArgumentError(absl::StrCat("the web viewer's threadPriority ", static_cast<int>(web.threadPriority),
                                                   " is none of ViewerThreadPriority's (kIdle, kInherited)"));
  }
  if (web.eglDeviceId.has_value() && *web.eglDeviceId < 0) {
    return absl::InvalidArgumentError(absl::StrCat("the web viewer's EGL device id is ", *web.eglDeviceId, "; device ids start at 0"));
  }
  if (web.eglLibrary.empty()) return absl::InvalidArgumentError("the web viewer's EGL library name is empty; the default is libEGL.so.1");
  if (web.startTimeout <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError(absl::StrCat("the web viewer's start timeout is ", absl::FormatDuration(web.startTimeout)));
  }
  return absl::OkStatus();
}

}  // namespace

// The rules of a viewer list, which a plot configuration's mujoco_viewer follows too (checked in Python by the Rerun
// bridge, for the same file).
// LINT.IfChange(viewer_list_rules)
absl::Status checkViewerNames(absl::Span<const std::string> names) {
  const std::string list = absl::StrJoin(names, ",");
  if (names.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("no viewer is named; name one or more of ", availableViewerNames(), " (", kNoViewerName, " runs without one)"));
  }
  absl::flat_hash_set<std::string> seen;
  for (size_t index = 0; index < names.size(); ++index) {
    const std::string& name = names[index];
    if (!isKnownViewer(name)) {
      return absl::InvalidArgumentError(
          absl::StrCat("unknown viewer '", name, "' in '", list, "'; the viewers are ", availableViewerNames()));
    }
    if (!seen.insert(name).second) {
      return absl::InvalidArgumentError(absl::StrCat("the viewer '", name, "' is listed twice in '", list, "'"));
    }
    if (index + 1 == names.size()) continue;
    if (name == kNoViewerName) {
      return absl::InvalidArgumentError(
          absl::StrCat("'", kNoViewerName, "' always starts, so nothing may follow it in '", list, "'; list it last"));
    }
    if (name == kGlfwViewerName) {
      return absl::InvalidArgumentError(absl::StrCat("'", kGlfwViewerName,
                                                     "' cannot fail at start-up (a window it cannot open is logged and the simulation runs "
                                                     "on without it), so nothing may follow it in '",
                                                     list, "'"));
    }
  }
  return absl::OkStatus();
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/plot_config.py:viewer_list_rules)

absl::StatusOr<std::vector<std::string>> parseViewerNames(absl::string_view list) {
  std::vector<std::string> names;
  if (absl::StripAsciiWhitespace(list).empty()) {
    if (absl::Status status = checkViewerNames(names); !status.ok()) {
      return status;
    }
  }
  for (const absl::string_view name : absl::StrSplit(list, ',')) {
    const absl::string_view stripped = absl::StripAsciiWhitespace(name);
    if (stripped.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("the viewer list '", list, "' has an empty entry; the viewers are ", availableViewerNames()));
    }
    names.emplace_back(stripped);
  }
  if (absl::Status status = checkViewerNames(names); !status.ok()) {
    return status;
  }
  return names;
}

absl::Status validateViewerSettings(const ViewerSettings& settings) {
  if (absl::Status status = checkViewerNames(settings.names); !status.ok()) {
    return status;
  }
  if (absl::Status status = checkServing(settings.web); !status.ok()) {
    return status;
  }
  if (absl::Status status = checkSizes(settings.web); !status.ok()) {
    return status;
  }
  if (absl::Status status = checkRendering(settings.web); !status.ok()) {
    return status;
  }
  return checkPlacementAndEgl(settings.web);
}

int softwareRasterizerThreads(const WebViewerOptions& web) {
  return std::max(static_cast<int>(web.cores.size()), kMinSoftwareRasterizerThreads);
}

}  // namespace robot::mujoco_sim_interface
