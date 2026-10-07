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

#include "humanoid_common_mpc_app/robot/MujocoRobotBackend.h"

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/ThreadAffinity.h"
#include "mujoco_sim_interface/CheaterSimContactEstimator.h"
#include "mujoco_sim_interface/Projectile.h"

namespace ocs2::humanoid {

absl::StatusOr<robot::mujoco_sim_interface::MujocoSimConfig> MujocoRobotBackend::makeConfig(const RobotBackendOptions& options) {
  if (options.mjcfFile.empty()) {
    return absl::InvalidArgumentError("the mujoco backend needs the robot's MuJoCo scene (--mjcf_file)");
  }
  if (!options.initialState.has_value()) {
    return absl::InvalidArgumentError("the mujoco backend needs the state the robot starts in");
  }
  const absl::StatusOr<robot::mujoco_sim_interface::GantryHold> gantryHold =
      robot::mujoco_sim_interface::gantryHoldFromName(options.simulator.gantryHold);
  if (!gantryHold.ok()) {
    return gantryHold.status();
  }
  robot::mujoco_sim_interface::MujocoSimConfig config;
  config.scenePath = options.mjcfFile;
  config.verbose = true;
  config.headless = options.headless;
  config.initStatePtr_ = std::make_shared<robot::model::RobotState>(*options.initialState);
  config.contactFrameNames = options.contactFrameNames;
  config.contactParentJointNames = options.contactParentJointNames;
  config.contactForceThreshold = options.simulator.contactForceThreshold;
  config.contactTimelineWindow = options.simulator.contactTimelineWindow;
  if (options.simulator.visualizations.has_value()) {
    config.visualizations = *options.simulator.visualizations;
  }
  config.gantryHold = options.simulator.gantryHold;
  config.projectile = options.simulator.projectile;
  config.contactPatchCorners = options.contactPatchCorners;
  return config;
}

absl::Status MujocoRobotBackend::checkOptions(const RobotBackendOptions& options) {
  if (const absl::StatusOr<robot::mujoco_sim_interface::MujocoSimConfig> config = makeConfig(options); !config.ok()) {
    return config.status();
  }
  if (options.simulator.projectile.empty()) return absl::OkStatus();
  return robot::mujoco_sim_interface::projectileFromName(options.simulator.projectile).status();
}

absl::StatusOr<std::unique_ptr<MujocoRobotBackend>> MujocoRobotBackend::Create(const RobotBackendOptions& options) {
  absl::StatusOr<robot::mujoco_sim_interface::MujocoSimConfig> config = makeConfig(options);
  if (!config.ok()) {
    return config.status();
  }
  absl::StatusOr<std::unique_ptr<robot::mujoco_sim_interface::MujocoSimInterface>> simulator =
      robot::mujoco_sim_interface::MujocoSimInterface::Create(*config, options.urdfFile);
  if (!simulator.ok()) {
    return absl::Status(simulator.status().code(),
                        absl::StrCat("the MuJoCo simulator did not start on ", options.mjcfFile, ": ", simulator.status().message()));
  }
  return absl::WrapUnique(new MujocoRobotBackend(*std::move(simulator)));
}

MujocoRobotBackend::MujocoRobotBackend(std::unique_ptr<robot::mujoco_sim_interface::MujocoSimInterface> simulator)
    : simulator_(std::move(simulator)) {}

absl::string_view MujocoRobotBackend::name() const {
  return kMujocoBackendName;
}

absl::Status MujocoRobotBackend::initialize() {
  simulator_->initSim();
  return absl::OkStatus();
}

absl::Status MujocoRobotBackend::start(const std::vector<int>& cores) {
  simulator_->startSim();
  if (!cores.empty() &&
      !setThreadCpuAffinity(cores, simulator_->getSimulationThread().native_handle(), /*threadName=*/"MuJoCo Simulation")) {
    return absl::FailedPreconditionError("the MuJoCo physics thread could not be pinned to its cores; it runs unpinned");
  }
  return absl::OkStatus();
}

void MujocoRobotBackend::readMeasuredContactForces(std::array<vector3_t, kNumContacts>& forces) {
  static_assert(kNumContacts == 2, "the simulator has a force sensor in each of two feet");
  simulator_->takeMeasuredFootForces(forces[0], forces[1]);
}

void MujocoRobotBackend::registerContactEstimators(robot::model::ContactEstimatorRegistry& registry) const {
  robot::mujoco_sim_interface::registerCheaterSimContactEstimator(registry, *simulator_);
}

}  // namespace ocs2::humanoid
