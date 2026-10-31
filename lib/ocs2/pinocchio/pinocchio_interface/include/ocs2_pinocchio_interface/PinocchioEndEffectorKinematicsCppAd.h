/******************************************************************************
Copyright (c) 2021, Farbod Farshidian. All rights reserved.

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
#include <vector>

#include "absl/base/nullability.h"

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <ocs2_core/automatic_differentiation/CppAdInterface.h>

namespace ocs2 {

/**
 * This class provides the CppAD implementation the end-effector Kinematics based on pinocchio. No pre-computation
 * is required. Its "updateCallback" makes the extra update calls on PinocchioInterface that a PinocchioStateInputMapping
 * may require before the kinematics are evaluated, such as those of the centroidal model mapping (refer to
 * CentroidalModelPinocchioMapping).
 *
 * The taped functions hold copies of what they read - the end-effector frames, the dimensions, the callback, and a
 * Pinocchio interface on AD scalars with a clone of the mapping set to it, which the functions of one object and of all
 * its copies share - so that a copy, and a taping after construction (CppAdInterface::getTapeOperationCount()), never
 * reads the constructor's arguments or the object it was copied from. The callback must therefore not refer to anything
 * that dies before the object and its copies.
 */
class PinocchioEndEffectorKinematicsCppAd final : public EndEffectorKinematics<scalar_t> {
 public:
  using EndEffectorKinematics<scalar_t>::vector3_t;
  using EndEffectorKinematics<scalar_t>::matrix3x_t;
  using EndEffectorKinematics<scalar_t>::quaternion_t;
  using update_pinocchio_interface_callback =
      std::function<void(const ad_vector_t& state, PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface)>;

  /** Constructor
   * @param [in] pinocchioInterface pinocchio interface.
   * @param [in] mapping mapping from OCS2 to pinocchio state.
   * @param [in] endEffectorIds array of end effector names.
   * @param [in] stateDim : size of state vector
   * @param [in] inputDim : size of input vector
   * @param [in] updateCallback : In the cases that PinocchioStateInputMapping requires some additional update calls on PinocchioInterface,
   *                              use this callback.
   * @param [in] modelName : name of the generate model library
   * @param [in] modelFolder : folder to save the model library files to
   * @param [in] recompileLibraries : If true, the model library will be newly compiled. If false, an existing library will be loaded if
   *                                  available.
   * @param [in] verbose : print information.
   */
  PinocchioEndEffectorKinematicsCppAd(const PinocchioInterface& pinocchioInterface,
                                      const PinocchioStateInputMapping<ad_scalar_t>& mapping,
                                      std::vector<std::string> endEffectorIds,
                                      size_t stateDim,
                                      size_t inputDim,
                                      update_pinocchio_interface_callback updateCallback,
                                      const std::string& modelName,
                                      const std::string& modelFolder = "/tmp/ocs2",
                                      bool recompileLibraries = true,
                                      bool verbose = false);

  ~PinocchioEndEffectorKinematicsCppAd() override = default;
  PinocchioEndEffectorKinematicsCppAd* absl_nonnull clone() const override;
  PinocchioEndEffectorKinematicsCppAd& operator=(const PinocchioEndEffectorKinematicsCppAd&) = delete;

  const std::vector<std::string>& getIds() const override;

  std::vector<vector3_t> getPosEesRtWorldEwrtWorld(const vector_t& state) const override;
  std::vector<vector3_t> getLinvelEesRtWorldEwrtWorld(const vector_t& state, const vector_t& input) const override;

  // This function contains no linear approximation due to the non-minimal quaternion representation of SO(3)
  // Use getOrientationErrorWrtPlaneLinearApproximation if a linear approximation is needed.
  std::vector<quaternion_t> getQuatEesRtWorld(const vector_t& state) const override;
  // NOLINTNEXTLINE(frame-naming): the plane orientation error has no quantity word yet (TODO in EndEffectorKinematics.h)
  std::vector<vector3_t> getOrientationErrorWrtPlane(const vector_t& state, const std::vector<vector3_t>& planeNormals) const override;
  std::vector<vector3_t> getAngvelEesRtWorldEwrtWorld(const vector_t& state, const vector_t& input) const override;
  std::vector<vector6_t> getTwistEesRtWorldEwrtWorld(const vector_t& state, const vector_t& input) const override;

  std::vector<VectorFunctionLinearApproximation> getLinearApproximationPosEesRtWorldEwrtWorld(const vector_t& state) const override;
  std::vector<VectorFunctionLinearApproximation> getLinearApproximationLinvelEesRtWorldEwrtWorld(const vector_t& state,
                                                                                const vector_t& input) const override;
  // NOLINTNEXTLINE(frame-naming): the plane orientation error has no quantity word yet (TODO in EndEffectorKinematics.h)
  std::vector<VectorFunctionLinearApproximation> getOrientationErrorWrtPlaneLinearApproximation(
      const vector_t& state, const std::vector<vector3_t>& planeNormals) const override;
  std::vector<VectorFunctionLinearApproximation> getLinearApproximationAngvelEesRtWorldEwrtWorld(const vector_t& state,
                                                                                       const vector_t& input) const override;
  std::vector<VectorFunctionLinearApproximation> getLinearApproximationTwistEesRtWorldEwrtWorld(const vector_t& state, const vector_t& input) const override;

 private:
  PinocchioEndEffectorKinematicsCppAd(const PinocchioEndEffectorKinematicsCppAd& rhs);

  std::unique_ptr<CppAdInterface> positionCppAdInterfacePtr_;
  std::unique_ptr<CppAdInterface> velocityCppAdInterfacePtr_;
  std::unique_ptr<CppAdInterface> orientationCppAdInterfacePtr_;
  std::unique_ptr<CppAdInterface> orientationErrorWrtPlaneCppAdInterfacePtr_;
  std::unique_ptr<CppAdInterface> angularVelocityCppAdInterfacePtr_;
  std::unique_ptr<CppAdInterface> twistCppAdInterfacePtr_;

  const std::vector<std::string> endEffectorIds_;
};

}  // namespace ocs2
