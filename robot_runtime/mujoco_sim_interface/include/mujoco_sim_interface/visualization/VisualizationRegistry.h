/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include <memory>
#include <string>
#include <vector>

#include "mujoco_sim_interface/visualization/MujocoVisualization.h"

namespace robot::mujoco_sim_interface {

/** One entry of the registry: what a name creates. */
struct VisualizationInfo {
  std::string name;
  std::string description;
  char hotkey{0};
};

/** Every visualization the viewer knows, in the order they are drawn. */
std::vector<VisualizationInfo> availableVisualizations();

/** Names enabled when the task file does not list any (`simVisualizations` absent): the viewer's historical set. */
std::vector<std::string> defaultVisualizationNames();

/** Creates the visualization registered under `name`, or nullptr if there is none. */
std::unique_ptr<MujocoVisualization> createVisualization(const std::string& name);

/**
 * Creates the visualizations listed in `names` (task file `simVisualizations`), in registry order regardless of the
 * order of the list, each enabled. Unknown and repeated names are skipped with a message appended to `errors`.
 */
std::vector<std::unique_ptr<MujocoVisualization>> createVisualizations(const std::vector<std::string>& names,
                                                                       std::vector<std::string>* errors);

}  // namespace robot::mujoco_sim_interface
