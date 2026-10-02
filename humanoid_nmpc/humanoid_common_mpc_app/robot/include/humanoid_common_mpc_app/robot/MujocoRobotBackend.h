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

#include <array>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include <mujoco_sim_interface/MujocoSimInterface.h>

#include "humanoid_common_mpc_app/robot/RobotBackend.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"

namespace ocs2::humanoid {

/**
 * The `mujoco` backend: robot_runtime/mujoco_sim_interface's MujocoSimInterface, with the configuration the ROS sims
 * built for it (the scene, the initial state, the contact frames, the simulator keys of the task file and the viewer's
 * target patches). initialize() is initSim(), start() is startSim() with the physics thread pinned to the cores given.
 */
class MujocoRobotBackend final : public RobotBackend {
 public:
  /** InvalidArgument when the options lack the scene or the initial state, or name an unknown gantry hold. */
  static absl::StatusOr<std::unique_ptr<MujocoRobotBackend>> Create(const RobotBackendOptions& options);

  absl::string_view name() const override;
  robot::model::RobotHWInterfaceBase& hardware() override { return *simulator_; }
  const robot::model::RobotHWInterfaceBase& hardware() const override { return *simulator_; }
  absl::Status initialize() override;
  absl::Status start(const std::vector<int>& cores) override;
  bool acceptsJointAction() const override { return !simulator_->isZeroTorqueMode(); }
  void readMeasuredContactForces(std::array<vector3_t, N_CONTACTS>& forces) override;
  /** Zero torque (disableTorques()): the start-up state of the simulator. */
  void enterSafeState() override { simulator_->disableTorques(); }
  void registerContactEstimators(robot::model::ContactEstimatorRegistry& registry) const override;
  robot::mujoco_sim_interface::MujocoSimInterface* simulator() override { return simulator_.get(); }

  /** The MujocoSimConfig the options make (for tests). */
  static absl::StatusOr<robot::mujoco_sim_interface::MujocoSimConfig> makeConfig(const RobotBackendOptions& options);

 private:
  explicit MujocoRobotBackend(std::unique_ptr<robot::mujoco_sim_interface::MujocoSimInterface> simulator);

  std::unique_ptr<robot::mujoco_sim_interface::MujocoSimInterface> simulator_;
};

}  // namespace ocs2::humanoid
