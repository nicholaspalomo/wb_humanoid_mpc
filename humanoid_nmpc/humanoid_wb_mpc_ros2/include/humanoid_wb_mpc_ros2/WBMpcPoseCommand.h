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

#include <memory>
#include <string>

#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_mpc/SystemObservation.h>

#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"

namespace ocs2::humanoid {

/**
 * The target the whole-body pose command node (WBMpcPoseCommandNode) publishes for a keyboard pose command, built by
 * the whole-body target calculator exactly as the centroidal node builds its own with CentroidalMpcTargetTrajectoriesCalculator.
 *
 * The node used to write the target by hand, with a copy of the calculator's time estimate, and put the base at
 * `defaultBaseHeight` in world coordinates: on a ground raised by the task file's `terrainHeight` it asked the robot to
 * crouch by the height of the ground. The calculator stands `defaultBaseHeight` on that ground, measures the xy
 * displacement in the pelvis frame and clamps the height change to `maxDeltaPelvisHeight`, as every other command path
 * of the whole-body MPC does.
 *
 * Only the robot model is built - the task file's model settings, no Pinocchio model and no CppAD library - so a keyboard
 * node started beside the MPC node never compiles the model libraries the MPC node is compiling.
 */
class WBMpcPoseCommand {
 public:
  /**
   * Loads the model settings of `taskFile` and `urdfFile` and the command limits of `referenceFile`.
   *
   * @return NotFound naming the path when one of the three files does not exist; InvalidArgument when a file cannot be
   *         loaded.
   */
  static absl::StatusOr<std::unique_ptr<WBMpcPoseCommand>> Create(const std::string& taskFile,
                                                                  const std::string& urdfFile,
                                                                  const std::string& referenceFile);

  WBMpcPoseCommand(const WBMpcPoseCommand&) = delete;
  WBMpcPoseCommand& operator=(const WBMpcPoseCommand&) = delete;

  /**
   * The target for `commandLineTarget` = [deltaX, deltaY, deltaZ, deltaYaw]: x and y in meters in the pelvis frame, z in
   * meters relative to `defaultBaseHeight` above the ground, yaw in degrees; see
   * WBMpcTargetTrajectoriesCalculator::commandedPositionToTargetTrajectories.
   */
  TargetTrajectories toTargetTrajectories(const vector4_t& commandLineTarget, const SystemObservation& observation);

  const ModelSettings& modelSettings() const { return *modelSettings_; }
  const WBAccelMpcRobotModel<scalar_t>& mpcRobotModel() const { return *mpcRobotModel_; }

 private:
  WBMpcPoseCommand(std::unique_ptr<ModelSettings> modelSettings, const std::string& referenceFile, scalar_t mpcHorizon);

  // Declared in the order they are built: the robot model refers to the model settings, and the calculator copies the
  // robot model, whose copy refers to the same settings.
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> mpcRobotModel_;
  std::unique_ptr<WBMpcTargetTrajectoriesCalculator> calculator_;
};

}  // namespace ocs2::humanoid
