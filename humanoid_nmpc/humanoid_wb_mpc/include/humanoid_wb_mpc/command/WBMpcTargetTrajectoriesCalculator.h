/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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
#include <string>

#include "absl/status/statusor.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid {

/**
 * The target trajectories of the whole-body MPC for a commanded pose or velocity in the pelvis frame: the base pose of
 * the whole-body state moved over the horizon, the joints held at the reference file's default posture. Not
 * thread-safe.
 */
class WBMpcTargetTrajectoriesCalculator : public TargetTrajectoriesCalculatorBase {
 public:
  // Create() comes in two forms with the same meaning: of the typed reference file (humanoid_mpc_config.ReferenceFile),
  // and of the file at a path - a root of the MPC's configuration - which loads it strictly (loadReferenceFile()) and
  // builds the typed form, prefixing its errors with the file.

  /**
   * The calculator of the reference file `referenceFile`, on the MPC `mpcRobotModel` (cloned) with the horizon
   * `mpcHorizon` [s]: its command limits (referenceSettingsFromConfig()) and its default joint state, in the model's
   * joint order (defaultJointStateFromConfig()). Nothing of the file is retained.
   *
   * @return The calculator; InvalidArgument naming a command limit that is absent or out of range, or listing the joints
   *         of default_joint_state that are missing, named twice, fixed or not joints of the model.
   */
  static absl::StatusOr<std::unique_ptr<WBMpcTargetTrajectoriesCalculator>> Create(const mpc_config::ReferenceFile& referenceFile,
                                                                                   const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                                   scalar_t mpcHorizon);

  /**
   * Create() of the reference file at `referenceFile`: loadReferenceFile()'s error for a file that cannot be read or
   * does not parse (naming its line and column), and the typed form's errors, prefixed with the file.
   */
  static absl::StatusOr<std::unique_ptr<WBMpcTargetTrajectoriesCalculator>> Create(const std::string& referenceFile,
                                                                                   const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                                   scalar_t mpcHorizon);

  /**
   * Converts command line to TargetTrajectories.
   * @param [in] commadLineTarget : [deltaX, deltaY, deltaZ, deltaYaw] defined in pelvis frame
   * @param [in] observation : the current observation
   */
  TargetTrajectories commandedPositionToTargetTrajectories(const vector4_t& commadLinePoseTarget,
                                                           scalar_t initTime,
                                                           const vector_t& initState) override;

  /**
   * Converts desired velocities to TargetTrajectories.
   * @param [in] commandedVelocities : [v_x, v_y, v_yaw] defined in pelvis frame
   * @param [in] observation : the current observation
   */
  TargetTrajectories commandedVelocityToTargetTrajectories(const vector4_t& commandedVelocities,
                                                           scalar_t initTime,
                                                           const vector_t& initState) override;

 private:
  /** The calculator of the checked `referenceSettings` and `defaultJointState` (one entry per joint of the model). */
  WBMpcTargetTrajectoriesCalculator(const ReferenceSettings& referenceSettings,
                                    const vector_t& defaultJointState,
                                    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                    scalar_t mpcHorizon);
};

}  // namespace ocs2::humanoid
