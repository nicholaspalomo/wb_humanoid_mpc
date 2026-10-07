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

#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc_app/visualization/config/robot/VisualizationConfigFromConfig.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid::visualization {

absl::StatusOr<VisualizationConfig> loadVisualizationConfig(const std::string& taskFile, const ModelSettings& modelSettings) {
  ASSIGN_OR_RETURN(const mpc_config::TaskFile task, loadTaskFile(taskFile));
  absl::StatusOr<VisualizationConfig> config = visualizationConfigFromConfig(task, modelSettings.contactNames);
  if (!config.ok()) {
    return withConfigFile(config.status(), taskFile);
  }
  return config;
}

std::string describeVisualizationConfig(const VisualizationConfig& config) {
  return absl::StrCat("viz/scene at ", config.sceneFrequency, " Hz (", kRerunSceneFrequencyField,
                      config.sceneFrequencyIsDefault ? ", the default" : "", "), viz/telemetry once per robot/state sample; ",
                      kTelemetryFramesField, ": ", absl::StrJoin(config.telemetryFrames, ", "), "; ", kRerunPlanFramesField, ": ",
                      absl::StrJoin(config.planFrames, ", "));
}

}  // namespace ocs2::humanoid::visualization
