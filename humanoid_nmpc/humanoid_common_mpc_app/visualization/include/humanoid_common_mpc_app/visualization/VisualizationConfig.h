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
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid::visualization {

// The fields of the robot's task file (config/mpc/task.textproto, humanoid_mpc_config.TaskFile) the visualization
// publisher reads, as its messages name them. Every field is optional.
// LINT.IfChange(visualization_task_fields)
/** [Hz] How often viz/scene is published at most: a positive number. */
inline constexpr absl::string_view kRerunSceneFrequencyField = "rerun_scene_frequency";
/** The frames of the frames/<kind>/<frame>/<source> telemetry groups: a list of frame names. */
inline constexpr absl::string_view kTelemetryFramesField = "telemetry_frames";
/** The frames whose planned paths world/plan/end_effectors draws, one line strip each: a list of frame names. */
inline constexpr absl::string_view kRerunPlanFramesField = "rerun_plan_frames";
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto:visualization_task_keys)

/** [Hz] rerun_scene_frequency when the task file does not set it. */
inline constexpr scalar_t kDefaultRerunSceneFrequency = 30.0;

/** What the visualization publisher takes from the task file. */
struct VisualizationConfig {
  /** [Hz] The most viz/scene messages per second (rerun_scene_frequency). */
  scalar_t sceneFrequency = kDefaultRerunSceneFrequency;
  /** True when the task file does not set rerun_scene_frequency and the default applies. */
  bool sceneFrequencyIsDefault = true;
  /** The frames of the frame telemetry groups (telemetry_frames); the contact frames when the file lists none. */
  std::vector<std::string> telemetryFrames;
  /** The frames whose planned paths are drawn (rerun_plan_frames); the contact frames when the file lists none. */
  std::vector<std::string> planFrames;
};

/**
 * What the visualization publisher takes from the task file `taskFile` (a robot's config/mpc/task.textproto, read
 * strictly: loadTaskFile(), then visualizationConfigFromConfig() over the contact frames of `modelSettings`).
 *
 * @return The errors of loadTaskFile(), which name the file, the line and the column, and those of
 *         visualizationConfigFromConfig() prefixed with the file.
 */
absl::StatusOr<VisualizationConfig> loadVisualizationConfig(const std::string& taskFile, const ModelSettings& modelSettings);

/** One line for the start-up log: the scene rate (and whether it is the default) and the frame lists. */
std::string describeVisualizationConfig(const VisualizationConfig& config);

}  // namespace ocs2::humanoid::visualization
