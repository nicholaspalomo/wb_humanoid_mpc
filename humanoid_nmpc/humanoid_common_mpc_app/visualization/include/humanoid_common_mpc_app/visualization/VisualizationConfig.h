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

// The keys of the robot's task file (config/mpc/task.yaml) the visualization publisher reads. Every key is optional.
// LINT.IfChange(visualization_task_keys)
/** [Hz] How often viz/scene is published at most: a positive number. */
inline constexpr absl::string_view kRerunSceneFrequencyKey = "rerunSceneFrequency";
/** The frames of the frames/<kind>/<frame>/<source> telemetry groups: a list of frame names. */
inline constexpr absl::string_view kTelemetryFramesKey = "telemetryFrames";
/** The frames whose planned paths world/plan/end_effectors draws, one line strip each: a list of frame names. */
inline constexpr absl::string_view kRerunPlanFramesKey = "rerunPlanFrames";
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:visualization_keys, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:visualization_keys, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml:visualization_keys, //robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml:visualization_keys, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml:visualization_keys, //humanoid_nmpc/humanoid_common_mpc_app/visualization/README.md:task_keys)
// clang-format on

/** [Hz] rerunSceneFrequency when the task file does not set it. */
inline constexpr scalar_t kDefaultRerunSceneFrequency = 30.0;

/** What the visualization publisher takes from the task file. */
struct VisualizationConfig {
  /** [Hz] The most viz/scene messages per second (rerunSceneFrequency). */
  scalar_t sceneFrequency = kDefaultRerunSceneFrequency;
  /** True when the task file does not set rerunSceneFrequency and the default applies. */
  bool sceneFrequencyIsDefault = true;
  /** The frames of the frame telemetry groups (telemetryFrames); the contact frames when the file lists none. */
  std::vector<std::string> telemetryFrames;
  /** The frames whose planned paths are drawn (rerunPlanFrames); the contact frames when the file lists none. */
  std::vector<std::string> planFrames;
};

/**
 * Reads the visualization keys from the YAML text of a task file. A key the text does not carry takes its default.
 *
 * @param yamlText The task file's contents.
 * @param source Names the text in error messages, usually the file's path.
 * @param modelSettings The robot's model settings: their contact frames are the default frame lists.
 * @return InvalidArgument naming the source and the key when the text does not parse, when rerunSceneFrequency is not
 *         a positive finite number, or when a frame list is not a list of distinct names made of letters, digits, '_',
 *         '-' and '.' (the characters of an entity path of the Rerun bridge).
 */
absl::StatusOr<VisualizationConfig> parseVisualizationConfig(absl::string_view yamlText,
                                                             absl::string_view source,
                                                             const ModelSettings& modelSettings);

/** parseVisualizationConfig() of the file at `taskFile`; NotFound when it cannot be read. */
absl::StatusOr<VisualizationConfig> loadVisualizationConfig(const std::string& taskFile, const ModelSettings& modelSettings);

/** One line for the start-up log: the scene rate (and whether it is the default) and the frame lists. */
std::string describeVisualizationConfig(const VisualizationConfig& config);

}  // namespace ocs2::humanoid::visualization
