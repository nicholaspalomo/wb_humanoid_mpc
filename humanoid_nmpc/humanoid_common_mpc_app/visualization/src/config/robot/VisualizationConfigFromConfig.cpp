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

#include "humanoid_common_mpc_app/visualization/config/robot/VisualizationConfigFromConfig.h"

#include <cmath>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid::visualization {
namespace {

/** True for a name the Rerun bridge accepts as one part of an entity path: letters, digits, '_', '-' and '.'. */
bool isEntityPathName(absl::string_view name) {
  if (name.empty() || name == "." || name == "..") {
    return false;
  }
  for (const char character : name) {
    if (!absl::ascii_isalnum(static_cast<unsigned char>(character)) && character != '_' && character != '-' && character != '.') {
      return false;
    }
  }
  return true;
}

/** The frames of the list `field`, or `fallback` when it is empty; InvalidArgument naming `field` for a bad list. */
absl::StatusOr<std::vector<std::string>> frameList(const std::vector<std::string>& frames,
                                                   absl::string_view field,
                                                   const std::vector<std::string>& fallback) {
  if (frames.empty()) {
    return fallback;
  }
  absl::flat_hash_set<absl::string_view> seen;
  for (const std::string& frame : frames) {
    if (!isEntityPathName(frame)) {
      return absl::InvalidArgumentError(absl::StrCat(field, " names the frame '", frame,
                                                     "'; a frame name of the plots and the 3D scene may only contain letters, "
                                                     "digits, '_', '-' and '.'."));
    }
    if (!seen.insert(frame).second) {
      return absl::InvalidArgumentError(absl::StrCat(field, " names the frame '", frame, "' twice."));
    }
  }
  return frames;
}

}  // namespace

absl::StatusOr<VisualizationConfig> visualizationConfigFromConfig(const mpc_config::TaskFile& task,
                                                                  const std::vector<std::string>& contactFrameNames) {
  VisualizationConfig config;
  if (task.rerun_scene_frequency.has_value()) {
    const double frequency = *task.rerun_scene_frequency;
    if (!std::isfinite(frequency) || frequency <= 0.0) {
      return absl::InvalidArgumentError(
          absl::StrCat(kRerunSceneFrequencyField, " is ", frequency, "; it must be a positive number of Hz."));
    }
    config.sceneFrequency = frequency;
    config.sceneFrequencyIsDefault = false;
  }
  ASSIGN_OR_RETURN(config.telemetryFrames, frameList(task.telemetry_frames, kTelemetryFramesField, contactFrameNames));
  ASSIGN_OR_RETURN(config.planFrames, frameList(task.rerun_plan_frames, kRerunPlanFramesField, contactFrameNames));
  return config;
}

}  // namespace ocs2::humanoid::visualization
