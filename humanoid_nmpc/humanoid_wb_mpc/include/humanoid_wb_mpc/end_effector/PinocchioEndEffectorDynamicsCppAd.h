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

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/automatic_differentiation/CppAdInterface.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/end_effector/EndEffectorDynamics.h"

namespace ocs2::humanoid {

/**
 * EndEffectorDynamics on Pinocchio, taped with CppAD for the whole-body MPC's state and input (WBAccelMpcRobotModel), the
 * accelerations those of the whole-body dynamics' generalized acceleration (computeGeneralizedAccelerations()). Each
 * function is a generated library of its own in the model folder, which its value and its linear approximation both
 * read: `<modelName>_position`, `_velocity`, `_linear_acceleration`, `_orientation_wrt_plane` (the plane normals its
 * parameters), `_pose_wrt_plane_rate` and `_pose_wrt_plane_acceleration` (the same). The swing foot's normal-motion
 * constraint reads the first three, the stance foot's constraint the position and the last three.
 *
 * The taped functions hold copies of what they read - the end-effector frames, a Pinocchio interface on AD scalars that
 * the functions of one object and of all its copies share, and a pointer to the robot model - so that a copy, and a
 * taping after construction (CppAdInterface::getTapeOperationCount()), never reads the object it was copied from. Not
 * thread-safe; the terms clone it.
 */
class PinocchioEndEffectorDynamicsCppAd final : public EndEffectorDynamics<scalar_t> {
 public:
  using EndEffectorDynamics<scalar_t>::vector3_t;
  using EndEffectorDynamics<scalar_t>::vector6_t;
  using EndEffectorDynamics<scalar_t>::vector_t;

  /**
   * Generates the libraries of the end effectors `endEffectorIds` (frames of `pinocchioInterface`) under `modelName` in
   * `modelFolder`, or, unless `recompileLibraries`, loads those of them that are there. `mpcRobotModel` maps the state
   * and the input to Pinocchio's; the taped functions keep a pointer to it, so it must outlive the object and every copy
   * of it. `verbose` prints the libraries' compilation.
   */
  PinocchioEndEffectorDynamicsCppAd(const PinocchioInterface& pinocchioInterface,
                                    WBAccelMpcRobotModel<ad_scalar_t>& mpcRobotModel,
                                    std::vector<std::string> endEffectorIds,
                                    const std::string& modelName,
                                    const std::string& modelFolder = "/tmp/ocs2",
                                    bool recompileLibraries = true,
                                    bool verbose = false);

  ~PinocchioEndEffectorDynamicsCppAd() override = default;
  PinocchioEndEffectorDynamicsCppAd(PinocchioEndEffectorDynamicsCppAd&&) = delete;
  PinocchioEndEffectorDynamicsCppAd& operator=(PinocchioEndEffectorDynamicsCppAd&&) = delete;
  PinocchioEndEffectorDynamicsCppAd* absl_nonnull clone() const override;
  PinocchioEndEffectorDynamicsCppAd& operator=(const PinocchioEndEffectorDynamicsCppAd&) = delete;

  const std::vector<std::string>& getIds() const override;

  std::vector<vector3_t> getPosition(const vector_t& state) const override;
  std::vector<vector3_t> getVelocity(const vector_t& state, const vector_t& input) const override;
  std::vector<vector3_t> getLinearAcceleration(const vector_t& state, const vector_t& input) const override;
  std::vector<vector3_t> getOrientationErrorWrtPlane(const vector_t& state, const std::vector<vector3_t>& planeNormals) const override;
  std::vector<vector6_t> getPoseWrtPlaneRate(const vector_t& state,
                                             const vector_t& input,
                                             const std::vector<vector3_t>& planeNormals) const override;
  std::vector<vector6_t> getPoseWrtPlaneAcceleration(const vector_t& state,
                                                     const vector_t& input,
                                                     const std::vector<vector3_t>& planeNormals) const override;

  std::vector<VectorFunctionLinearApproximation> getPositionLinearApproximation(const vector_t& state) const override;
  std::vector<VectorFunctionLinearApproximation> getVelocityLinearApproximation(const vector_t& state,
                                                                                const vector_t& input) const override;
  std::vector<VectorFunctionLinearApproximation> getLinearAccelerationLinearApproximation(const vector_t& state,
                                                                                          const vector_t& input) const override;
  std::vector<VectorFunctionLinearApproximation> getOrientationErrorWrtPlaneLinearApproximation(
      const vector_t& state, const std::vector<vector3_t>& planeNormals) const override;
  std::vector<VectorFunctionLinearApproximation> getPoseWrtPlaneRateLinearApproximation(
      const vector_t& state, const vector_t& input, const std::vector<vector3_t>& planeNormals) const override;
  std::vector<VectorFunctionLinearApproximation> getPoseWrtPlaneAccelerationLinearApproximation(
      const vector_t& state, const vector_t& input, const std::vector<vector3_t>& planeNormals) const override;

 private:
  PinocchioEndEffectorDynamicsCppAd(const PinocchioEndEffectorDynamicsCppAd& rhs);

  std::unique_ptr<CppAdInterface> positionCppAdInterfacePtr_;
  std::unique_ptr<CppAdInterface> velocityCppAdInterfacePtr_;
  std::unique_ptr<CppAdInterface> linearAccelerationCppAdInterfacePtr_;
  std::unique_ptr<CppAdInterface> orientationErrorWrtPlaneCppAdInterfacePtr_;
  std::unique_ptr<CppAdInterface> poseWrtPlaneRateCppAdInterfacePtr_;
  std::unique_ptr<CppAdInterface> poseWrtPlaneAccelerationCppAdInterfacePtr_;

  const std::vector<std::string> endEffectorIds_;
};

}  // namespace ocs2::humanoid
