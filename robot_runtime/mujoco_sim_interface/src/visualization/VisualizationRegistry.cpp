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

#include <functional>
#include <set>
#include <sstream>

#include "mujoco_sim_interface/visualization/BaseVelocityVisualization.h"
#include "mujoco_sim_interface/visualization/CenterOfMassVisualization.h"
#include "mujoco_sim_interface/visualization/ContactForceVisualization.h"
#include "mujoco_sim_interface/visualization/ContactTimelineVisualization.h"
#include "mujoco_sim_interface/visualization/DcmVisualization.h"
#include "mujoco_sim_interface/visualization/ExternalForceVisualization.h"
#include "mujoco_sim_interface/visualization/MetricsOverlay.h"
#include "mujoco_sim_interface/visualization/MujocoOptionFlagVisualization.h"
#include "mujoco_sim_interface/visualization/TargetContactPatchVisualization.h"
#include "mujoco_sim_interface/visualization/ZmpVisualization.h"

namespace robot::mujoco_sim_interface {

namespace {
struct Entry {
  std::function<std::unique_ptr<MujocoVisualization>()> create;
  bool enabledByDefault;
};

/// The registry, in drawing order. A new visualization is added here and nowhere else; the task file refers to the
/// name it reports. The MuJoCo option flags come last so that they never shift the order of the custom markers.
// LINT.IfChange(visualization_names)
const std::vector<Entry>& registry() {
  static const std::vector<Entry> entries = {
      {[] { return std::make_unique<MetricsOverlay>(); }, true},
      {[] { return std::make_unique<ExternalForceVisualization>(); }, true},
      {[] { return std::make_unique<ContactForceVisualization>(); }, true},
      {[] { return std::make_unique<BaseVelocityVisualization>(); }, true},
      {[] { return std::make_unique<ContactTimelineVisualization>(); }, true},
      {[] { return std::make_unique<TargetContactPatchVisualization>(); }, true},
      // Centroidal markers on the ground: off unless listed, like MuJoCo's own markers.
      {[] { return std::make_unique<CenterOfMassVisualization>(); }, false},
      {[] { return std::make_unique<ZmpVisualization>(); }, false},
      {[] { return std::make_unique<DcmVisualization>(); }, false},
      {[] { return MujocoOptionFlagVisualization::contactPoints(); }, false},
      {[] { return MujocoOptionFlagVisualization::contactForces(); }, false},
      {[] { return MujocoOptionFlagVisualization::centerOfMass(); }, false},
      {[] { return MujocoOptionFlagVisualization::inertia(); }, false},
      {[] { return MujocoOptionFlagVisualization::convexHull(); }, false},
      {[] { return MujocoOptionFlagVisualization::transparency(); }, false},
  };
  return entries;
}
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:sim_visualizations, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml:sim_visualizations, //robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml:sim_visualizations, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml:sim_visualizations, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:sim_visualizations)
// clang-format on
}  // namespace

std::vector<VisualizationInfo> availableVisualizations() {
  std::vector<VisualizationInfo> infos;
  for (const Entry& entry : registry()) {
    const std::unique_ptr<MujocoVisualization> visualization = entry.create();
    infos.push_back({visualization->name(), visualization->description(), visualization->hotkey()});
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

std::unique_ptr<MujocoVisualization> createVisualization(const std::string& name) {
  for (const Entry& entry : registry()) {
    std::unique_ptr<MujocoVisualization> visualization = entry.create();
    if (visualization->name() == name) return visualization;
  }
  return nullptr;
}

std::vector<std::unique_ptr<MujocoVisualization>> createVisualizations(const std::vector<std::string>& names,
                                                                       std::vector<std::string>* errors) {
  std::set<std::string> known;
  for (const VisualizationInfo& info : availableVisualizations()) known.insert(info.name);
  std::set<std::string> requested;
  for (const std::string& name : names) {
    if (known.count(name) == 0) {
      if (errors != nullptr) {
        std::ostringstream message;
        message << "unknown visualization '" << name << "'; available:";
        for (const std::string& option : known) message << " " << option;
        errors->push_back(message.str());
      }
      continue;
    }
    if (!requested.insert(name).second && errors != nullptr) {
      errors->push_back("visualization '" + name + "' is listed more than once");
    }
  }
  std::vector<std::unique_ptr<MujocoVisualization>> visualizations;
  for (const Entry& entry : registry()) {
    std::unique_ptr<MujocoVisualization> visualization = entry.create();
    if (requested.count(visualization->name()) == 0) continue;
    visualization->setEnabled(true);  // listed means on at start-up, also for MuJoCo's own markers that default to off
    visualizations.push_back(std::move(visualization));
  }
  return visualizations;
}

}  // namespace robot::mujoco_sim_interface
