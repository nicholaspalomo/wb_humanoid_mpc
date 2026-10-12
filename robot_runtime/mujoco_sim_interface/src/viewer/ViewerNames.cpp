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

#include "mujoco_sim_interface/viewer/ViewerNames.h"

#include <string>
#include <vector>

#include "absl/strings/str_join.h"

namespace robot::mujoco_sim_interface {

std::vector<ViewerInfo> availableViewers() {
  // LINT.IfChange(viewer_descriptions)
  return {
      {.name = kGlfwViewerName, .description = "a GLFW window on DISPLAY (the X11/VNC desktop)"},
      {.name = kWebViewerName, .description = "a browser tab: offscreen EGL frames served over HTTP (default http://localhost:6090)"},
      {.name = kNoViewerName, .description = "no viewer"},
  };
  // LINT.ThenChange(//robot_runtime/mujoco_sim_interface/include/mujoco_sim_interface/viewer/ViewerNames.h:viewer_names)
}

std::string availableViewerNames() {
  std::vector<std::string> names;
  for (const ViewerInfo& viewer : availableViewers()) names.push_back(viewer.name);
  return absl::StrJoin(names, ", ");
}

}  // namespace robot::mujoco_sim_interface
