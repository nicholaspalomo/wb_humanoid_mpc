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

#include "mujoco_sim_interface/visualization/VisualizationRegistry.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/btree_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "mujoco_sim_interface/visualization/BaseVelocityVisualization.h"
#include "mujoco_sim_interface/visualization/CenterOfMassVisualization.h"
#include "mujoco_sim_interface/visualization/ContactForceVisualization.h"
#include "mujoco_sim_interface/visualization/ContactTimelineVisualization.h"
#include "mujoco_sim_interface/visualization/DcmVisualization.h"
#include "mujoco_sim_interface/visualization/ExternalForceVisualization.h"
#include "mujoco_sim_interface/visualization/MetricsOverlay.h"
#include "mujoco_sim_interface/visualization/MujocoOptionFlagVisualization.h"
#include "mujoco_sim_interface/visualization/MujocoVisualization.h"
#include "mujoco_sim_interface/visualization/TargetContactPatchVisualization.h"
#include "mujoco_sim_interface/visualization/ZmpVisualization.h"

namespace robot::mujoco_sim_interface {

namespace {
/// One row of the visualization registry: how to create the visualization and
/// whether it is on by default.
struct Entry {
  /// A capture-less factory, so that the table can be built from an
  /// initializer list (a move-only callable could not be copied out of one).
  std::unique_ptr<MujocoVisualization> (*absl_nonnull create)();
  bool enabledByDefault;
};

/// Creates a default-constructed T: the factory of a visualization class.
template <typename T>
std::unique_ptr<MujocoVisualization> makeVisualization() {
  return std::make_unique<T>();
}

/// The registry, in drawing order. A new visualization is added here and
/// nowhere else; the task file refers to the name it reports. The MuJoCo option
/// flags come last so that they never shift the order of the custom markers.
// LINT.IfChange(visualization_names)
const std::vector<Entry>& registry() {
  static const absl::NoDestructor<std::vector<Entry>> kEntries(
      std::vector<Entry>{
          {&makeVisualization<MetricsOverlay>, true},
          {&makeVisualization<ExternalForceVisualization>, true},
          {&makeVisualization<ContactForceVisualization>, true},
          {&makeVisualization<BaseVelocityVisualization>, true},
          {&makeVisualization<ContactTimelineVisualization>, true},
          {&makeVisualization<TargetContactPatchVisualization>, true},
          // Centroidal markers on the ground: off unless listed, like MuJoCo's
          // own markers.
          {&makeVisualization<CenterOfMassVisualization>, false},
          {&makeVisualization<ZmpVisualization>, false},
          {&makeVisualization<DcmVisualization>, false},
          {[] {
             return std::unique_ptr<MujocoVisualization>(
                 MujocoOptionFlagVisualization::contactPoints());
           },
           false},
          {[] {
             return std::unique_ptr<MujocoVisualization>(
                 MujocoOptionFlagVisualization::contactForceMarkers());
           },
           false},
          {[] {
             return std::unique_ptr<MujocoVisualization>(
                 MujocoOptionFlagVisualization::centerOfMass());
           },
           false},
          {[] {
             return std::unique_ptr<MujocoVisualization>(
                 MujocoOptionFlagVisualization::inertia());
           },
           false},
          {[] {
             return std::unique_ptr<MujocoVisualization>(
                 MujocoOptionFlagVisualization::convexHull());
           },
           false},
          {[] {
             return std::unique_ptr<MujocoVisualization>(
                 MujocoOptionFlagVisualization::transparency());
           },
           false},
      });
  return *kEntries;
}
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto:sim_visualizations, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto:sim_visualizations, //robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto:sim_visualizations, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto:sim_visualizations, //robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/config/mpc/task.textproto:sim_visualizations, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto:sim_visualizations, //humanoid_nmpc/humanoid_mpc_config/task_file.proto:sim_visualizations)
// clang-format on
}  // namespace

std::vector<VisualizationInfo> availableVisualizations() {
  std::vector<VisualizationInfo> infos;
  for (const Entry& entry : registry()) {
    const std::unique_ptr<MujocoVisualization> visualization = entry.create();
    infos.push_back({visualization->name(), visualization->description(),
                     visualization->hotkey()});
  }
  return infos;
}

std::vector<std::string> defaultVisualizationNames() {
  std::vector<std::string> names;
  for (const Entry& entry : registry()) {
    if (entry.enabledByDefault) names.push_back(entry.create()->name());
  }
  return names;
}

std::unique_ptr<MujocoVisualization> createVisualization(
    absl::string_view name) {
  for (const Entry& entry : registry()) {
    std::unique_ptr<MujocoVisualization> visualization = entry.create();
    if (visualization->name() == name) return visualization;
  }
  return nullptr;
}

std::vector<std::unique_ptr<MujocoVisualization>> createVisualizations(
    absl::Span<const std::string> names,
    std::vector<std::string>* absl_nullable errors) {
  absl::btree_set<std::string> known;
  for (const VisualizationInfo& info : availableVisualizations()) {
    known.insert(info.name);
  }
  absl::btree_set<std::string> requested;
  for (const std::string& name : names) {
    if (!known.contains(name)) {
      if (errors != nullptr) {
        errors->push_back(
            absl::StrCat("unknown visualization '", name,
                         "'; available: ", absl::StrJoin(known, " ")));
      }
      continue;
    }
    if (!requested.insert(name).second && errors != nullptr) {
      errors->push_back(
          absl::StrCat("visualization '", name, "' is listed more than once"));
    }
  }
  std::vector<std::unique_ptr<MujocoVisualization>> visualizations;
  for (const Entry& entry : registry()) {
    std::unique_ptr<MujocoVisualization> visualization = entry.create();
    if (!requested.contains(visualization->name())) continue;
    visualization->setEnabled(
        true);  // listed means on at start-up, also for MuJoCo's own markers
                // that default to off
    visualizations.push_back(std::move(visualization));
  }
  return visualizations;
}

}  // namespace robot::mujoco_sim_interface
