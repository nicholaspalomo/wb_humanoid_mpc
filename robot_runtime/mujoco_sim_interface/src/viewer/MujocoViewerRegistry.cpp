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

#include "mujoco_sim_interface/viewer/MujocoViewerRegistry.h"

#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/viewer/GlfwMujocoViewer.h"
#include "mujoco_sim_interface/viewer/MujocoViewer.h"
#include "mujoco_sim_interface/viewer/NoMujocoViewer.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/WebMujocoViewer.h"

namespace robot::mujoco_sim_interface {

absl::StatusOr<std::unique_ptr<MujocoViewer>> createViewer(absl::string_view name, const MujocoSimInterface* absl_nonnull sim) {
  // LINT.IfChange(viewer_factories)
  if (name == kGlfwViewerName) {
    absl::StatusOr<std::unique_ptr<GlfwMujocoViewer>> viewer = GlfwMujocoViewer::Create(sim);
    if (!viewer.ok()) return viewer.status();
    return std::unique_ptr<MujocoViewer>(*std::move(viewer));
  }
  if (name == kWebViewerName) {
    absl::StatusOr<std::unique_ptr<WebMujocoViewer>> viewer = WebMujocoViewer::Create(sim, sim->getConfig().viewer.web);
    if (!viewer.ok()) return viewer.status();
    return std::unique_ptr<MujocoViewer>(*std::move(viewer));
  }
  if (name == kNoViewerName) return std::make_unique<NoMujocoViewer>();
  // LINT.ThenChange(//robot_runtime/mujoco_sim_interface/include/mujoco_sim_interface/viewer/ViewerNames.h:viewer_names)
  return absl::InvalidArgumentError(absl::StrCat("unknown viewer '", name, "'; the viewers are ", availableViewerNames()));
}

}  // namespace robot::mujoco_sim_interface
