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
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
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
 * Turns the operator's position and velocity commands into target trajectories of the centroidal state
 * (TargetTrajectoriesCalculatorBase), with the joint-state target filtered towards the reference file's default joint
 * state. Thread-safe only as far as TargetTrajectoriesCalculatorBase says for its command limits; call the conversions
 * from one thread.
 */
class CentroidalMpcTargetTrajectoriesCalculator : public TargetTrajectoriesCalculatorBase {
 public:
  /**
   * The calculator of a robot's reference file: its default_joint_state (on the joints of `mpcRobotModel`'s model
   * settings), its command limits and the joint-state filter's target_joint_state_interpolation_time_constant, which this
   * calculator requires. The file is given typed, or by its path (loadReferenceFile(): the path form of a root of the
   * MPC's configuration). `info` must outlive the calculator.
   *
   * @return The loader's errors for the path form; InvalidArgument naming the field of a file whose command limits or
   *         default posture do not convert (referenceSettingsFromConfig(), defaultJointStateFromConfig()), and naming
   *         target_joint_state_interpolation_time_constant when the file does not give it or it is not positive. The
   *         path form prefixes the conversions' errors with the file.
   */
  static absl::StatusOr<std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator>> Create(const std::string& referenceFile,
                                                                                           const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                                           PinocchioInterface pinocchioInterface,
                                                                                           const CentroidalModelInfo& info,
                                                                                           scalar_t mpcHorizon);
  static absl::StatusOr<std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator>> Create(const mpc_config::ReferenceFile& referenceFile,
                                                                                           const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                                           PinocchioInterface pinocchioInterface,
                                                                                           const CentroidalModelInfo& info,
                                                                                           scalar_t mpcHorizon);

  ~CentroidalMpcTargetTrajectoriesCalculator() override = default;
  CentroidalMpcTargetTrajectoriesCalculator(const CentroidalMpcTargetTrajectoriesCalculator& rhs) = delete;
  CentroidalMpcTargetTrajectoriesCalculator& operator=(const CentroidalMpcTargetTrajectoriesCalculator&) = delete;

  /** The base class's filter, and here the joint-state filter and its clock: see TargetTrajectoriesCalculatorBase::reset(). */
  void reset() override;

  /**
   * Converts command line to TargetTrajectories.
   * @param [in] commandLinePoseTarget : [deltaX, deltaY, deltaZ, deltaYaw] defined in pelvis frame
   * @param [in] observation : the current observation
   */
  TargetTrajectories commandedPositionToTargetTrajectories(const vector4_t& commandLinePoseTarget,
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
  /** Private: Create() converts and checks the reference file's settings, which this only stores. */
  CentroidalMpcTargetTrajectoriesCalculator(const ReferenceSettings& referenceSettings,
                                            const vector_t& defaultJointState,
                                            scalar_t targetJointStateInterpolationTimeConstant,
                                            const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                            PinocchioInterface pinocchioInterface,
                                            const CentroidalModelInfo& info,
                                            scalar_t mpcHorizon);

  PinocchioInterface pinocchioInterface_;
  const CentroidalModelInfo& info_;
  const scalar_t mass_;

  // [s] The reference file's target_joint_state_interpolation_time_constant, positive.
  const scalar_t targetJointStateInterpolationTimeConstant_;
  scalar_t lastTime_ = 0.0;
  vector_t filteredJointState_;
};

}  // namespace ocs2::humanoid
