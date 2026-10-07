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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc_app/robot/RobotBackend.h"
#include "mujoco_sim_interface/MujocoContactPatch.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {

/** The simulator fields of the task file (RobotProcessSettings), for a simulated backend. */
struct SimulatorSettings {
  /** [N] Normal force above which a contact point counts as touching (`sim_contact_force_threshold`). */
  double contactForceThreshold = 5.0;
  /** [s] Window of the viewer's contact timeline (`sim_contact_timeline_window`). */
  double contactTimelineWindow = 5.0;
  /** Viewer visualizations by name (`sim_visualizations`); nullopt: the viewer's default set. */
  std::optional<std::vector<std::string>> visualizations;
  /** How the gantry holds the base (`gantry_hold`). */
  std::string gantryHold = "weld_constraint";
  /** The ball the Dodgeball tab throws (`sim_projectile`); empty: none. */
  std::string projectile;
};

/** What every backend is built from. */
struct RobotBackendOptions {
  std::string robotName;
  std::string urdfFile;
  /** The MuJoCo scene (--mjcf_file); a hardware backend ignores it. */
  std::string mjcfFile;
  /** The state the robot starts in: in simulation, where it is put (createInitialSimState()). */
  std::optional<robot::model::RobotState> initialState;
  /** The MPC's contact frames and their parent joints (model_settings), in contact order. */
  std::vector<std::string> contactFrameNames;
  std::vector<std::string> contactParentJointNames;
  /** The target contact patches the viewer draws, per contact; empty: none (no contact planner). */
  std::vector<robot::mujoco_sim_interface::ContactPatchCorners> contactPatchCorners;
  SimulatorSettings simulator;
  /** No viewer window (--headless): the robot-runtime image has no GL, and the tests have no display. */
  bool headless = false;
};

// LINT.IfChange(backend_names)
/** The MuJoCo simulator, robot_runtime/mujoco_sim_interface (MujocoRobotBackend). */
inline constexpr absl::string_view kMujocoBackendName = "mujoco";
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:backend_names)

/**
 * The robot backends by name (--backend): the only place a backend is added. An unknown name is refused with the list
 * of the available ones.
 */
class RobotBackendRegistry {
 public:
  using Factory = std::function<absl::StatusOr<std::unique_ptr<RobotBackend>>(const RobotBackendOptions& options)>;
  /**
   * What the backend would refuse in `options` without building it (the task file's simulator settings, the files it
   * needs): OK or the refusal create() would give. For a save the robot checks before it stores the file.
   */
  using OptionsCheck = std::function<absl::Status(const RobotBackendOptions& options)>;

  /** A registry with the built-in backends (`mujoco`). */
  RobotBackendRegistry();

  /** Adds the backend `name`; an empty `checkOptions` accepts any options (checkOptions()). */
  void add(const std::string& name, const std::string& description, Factory factory, OptionsCheck checkOptions = nullptr);
  bool has(absl::string_view name) const;
  std::vector<std::string> names() const;
  /** "mujoco (the MuJoCo simulator), ...". */
  std::string availableNames() const;

  /** The backend `name` names, built from `options`; InvalidArgument listing the available names for an unknown one. */
  absl::StatusOr<std::unique_ptr<RobotBackend>> create(absl::string_view name, const RobotBackendOptions& options) const;

  /**
   * The backend `name`'s check of `options` (OptionsCheck), which builds nothing; InvalidArgument listing the available
   * names for an unknown one.
   */
  absl::Status checkOptions(absl::string_view name, const RobotBackendOptions& options) const;

 private:
  struct Entry {
    std::string name;
    std::string description;
    Factory factory;
    OptionsCheck checkOptions;
  };
  /** The entry `name` names; nullptr for an unknown one. */
  const Entry* absl_nullable find(absl::string_view name) const;
  /** InvalidArgument naming `name` and the available backends. */
  absl::Status unknownBackend(absl::string_view name) const;

  std::vector<Entry> entries_;
};

}  // namespace ocs2::humanoid
