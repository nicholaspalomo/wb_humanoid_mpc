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

#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid::visualization {

/**
 * What the visualization publisher takes from the robot's typed task file. rerun_scene_frequency absent is the default
 * rate, reported as the default; a frame list the file leaves out, or lists empty, is `contactFrameNames`
 * (ModelSettings::contactNames).
 *
 * @return InvalidArgument naming the field when rerun_scene_frequency is not a positive finite number, or when a frame
 *         list names a frame twice or a frame whose name is not made of letters, digits, '_', '-' and '.' (the
 *         characters of an entity path of the Rerun bridge).
 */
absl::StatusOr<VisualizationConfig> visualizationConfigFromConfig(const mpc_config::TaskFile& task,
                                                                  const std::vector<std::string>& contactFrameNames);

}  // namespace ocs2::humanoid::visualization
