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

#include "humanoid_common_mpc_app/robot/RobotAppOptions.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_rerun_viewer_proto/plot_config.nproto.h"
#include "humanoid_rerun_viewer_proto/plot_config.nproto.pb.h"
#include "humanoid_rerun_viewer_proto/plot_config.pb.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/ViewerSettings.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {
namespace {

/**
 * The mujoco_viewer of the plot configuration at `path` (--plot_config), read strictly and checked; kDefaultViewerName
 * when the file lists none. Errors are prefixed with the flag.
 */
absl::StatusOr<std::vector<std::string>> viewerNamesFromPlotConfig(absl::string_view path) {
  absl::StatusOr<rerun::PlotConfig> config = nproto::LoadTextprotoFile<rerun::PlotConfig, humanoid_rerun_viewer_proto::PlotConfig>(path);
  if (!config.ok()) return absl::Status(config.status().code(), absl::StrCat("--plot_config=", path, ": ", config.status().message()));
  if (config->mujoco_viewer.empty()) {
    LOG(INFO) << "[robot] MuJoCo viewer: " << kDefaultViewerName << ", the default (--plot_config=" << path
              << " lists no mujoco_viewer and --viewer is not given)";
    return std::vector<std::string>{std::string(kDefaultViewerName)};
  }
  if (absl::Status status = robot::mujoco_sim_interface::checkViewerNames(config->mujoco_viewer); !status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat("--plot_config=", path, ": mujoco_viewer: ", status.message()));
  }
  LOG(INFO) << "[robot] MuJoCo viewer: " << absl::StrJoin(config->mujoco_viewer, ",") << ", the mujoco_viewer of --plot_config=" << path
            << " (--viewer is not given)";
  return std::move(config->mujoco_viewer);
}

/** The web viewer's address, port, allowed hosts, cores and EGL device from the flags; InvalidArgument naming the flag. */
absl::StatusOr<robot::mujoco_sim_interface::WebViewerOptions> webViewerOptionsFromFlags(const ViewerFlagValues& flags,
                                                                                        const std::vector<int>& defaultViewerCores) {
  robot::mujoco_sim_interface::WebViewerOptions web;
  web.address = std::string(absl::StripAsciiWhitespace(flags.webViewerAddress));
  if (web.address.empty()) {
    return absl::InvalidArgumentError("--web_viewer_address is empty: 127.0.0.1 serves this machine only, 0.0.0.0 every interface");
  }
  if (flags.webViewerPort < 0 || flags.webViewerPort > robot::mujoco_sim_interface::WebViewerOptions::kMaxPort) {
    return absl::InvalidArgumentError(absl::StrCat("--web_viewer_port=", flags.webViewerPort, ": give a TCP port of 1-",
                                                   robot::mujoco_sim_interface::WebViewerOptions::kMaxPort, ", or 0 for any free one"));
  }
  web.port = flags.webViewerPort;
  for (const absl::string_view host : absl::StrSplit(flags.webViewerAllowedHosts, ',', absl::SkipWhitespace())) {
    web.allowedHosts.emplace_back(absl::StripAsciiWhitespace(host));
  }
  absl::StatusOr<std::vector<int>> cores = parseCoreList(flags.viewerCores, defaultViewerCores);
  if (!cores.ok()) return absl::InvalidArgumentError(absl::StrCat("--viewer_cores=", flags.viewerCores, ": ", cores.status().message()));
  web.cores = *std::move(cores);
  if (flags.eglDeviceId.has_value()) {
    int device = -1;
    if (!absl::SimpleAtoi(*flags.eglDeviceId, &device) || device < 0) {
      return absl::InvalidArgumentError(absl::StrCat("MUJOCO_EGL_DEVICE_ID=", *flags.eglDeviceId,
                                                     ": give the index of an EGL device (0, 1, ...), or unset it to render on the "
                                                     "first GPU, else on Mesa's llvmpipe"));
    }
    web.eglDeviceId = device;
  }
  return web;
}

}  // namespace

absl::StatusOr<std::vector<int>> parseCoreList(absl::string_view value, const std::vector<int>& defaultCores) {
  const absl::string_view trimmed = absl::StripAsciiWhitespace(value);
  if (trimmed == "default") return defaultCores;
  if (trimmed.empty() || trimmed == "none") return std::vector<int>();
  std::vector<int> cores;
  for (absl::string_view part : absl::StrSplit(trimmed, ',')) {
    int core = -1;
    if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(part), &core) || core < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("'", value, "' names no cores: give `default`, `none` or a comma-separated list of CPU numbers such as 4,5"));
    }
    cores.push_back(core);
  }
  return cores;
}

absl::StatusOr<std::vector<std::string>> viewerNamesFromFlags(const ViewerFlagValues& flags) {
  if (!absl::StripAsciiWhitespace(flags.viewer).empty()) {
    absl::StatusOr<std::vector<std::string>> names = robot::mujoco_sim_interface::parseViewerNames(flags.viewer);
    if (!names.ok()) return absl::InvalidArgumentError(absl::StrCat("--viewer=", flags.viewer, ": ", names.status().message()));
    return names;
  }
  const absl::string_view plotConfig = absl::StripAsciiWhitespace(flags.plotConfig);
  if (!plotConfig.empty()) return viewerNamesFromPlotConfig(plotConfig);
  LOG(INFO) << "[robot] MuJoCo viewer: " << kDefaultViewerName << ", the default (neither --viewer nor --plot_config is given)";
  return std::vector<std::string>{std::string(kDefaultViewerName)};
}

absl::StatusOr<robot::mujoco_sim_interface::ViewerSettings> viewerSettingsFromFlags(const ViewerFlagValues& flags,
                                                                                    const std::vector<int>& defaultViewerCores) {
  robot::mujoco_sim_interface::ViewerSettings settings;
  ASSIGN_OR_RETURN(settings.names, viewerNamesFromFlags(flags));
  ASSIGN_OR_RETURN(settings.web, webViewerOptionsFromFlags(flags, defaultViewerCores));
  RETURN_IF_ERROR(robot::mujoco_sim_interface::validateViewerSettings(settings));
  return settings;
}

std::optional<std::string> softwareRasterizerThreadsToSet(const robot::mujoco_sim_interface::ViewerSettings& viewer,
                                                          const char* absl_nullable current) {
  if (current != nullptr && *current != '\0') return std::nullopt;
  if (!absl::c_linear_search(viewer.names, robot::mujoco_sim_interface::kWebViewerName)) return std::nullopt;
  return absl::StrCat(robot::mujoco_sim_interface::softwareRasterizerThreads(viewer.web));
}

std::string describeSoftwareRasterizerThreads(const robot::mujoco_sim_interface::ViewerSettings& viewer, absl::string_view threads) {
  const std::string cores =
      viewer.web.cores.empty() ? " (viewer cores unrestricted)" : absl::StrCat(", for ", viewer.web.cores.size(), " viewer core(s)");
  return absl::StrCat(kSoftwareRasterizerThreadsVariable, "=", threads, ": llvmpipe's rasterizer threads", cores, ".");
}

}  // namespace ocs2::humanoid
