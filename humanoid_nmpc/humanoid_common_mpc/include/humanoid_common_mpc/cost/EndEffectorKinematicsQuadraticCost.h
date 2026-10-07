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

#include "absl/base/nullability.h"
#include "ocs2_core/cost/StateInputGaussNewtonCostAd.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/end_effector/EndEffectorKinematics.h"
#include "pinocchio/algorithm/frames.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * A Gauss-Newton cost on the task-space error of one link - position, orientation, linear and angular velocity - against
 * the link pose the reference state implies, generated with CppAD.
 *
 * The robot model and the reference manager, when given, must outlive it. setWeights() retunes it between solves. Like
 * every OCS2 term it is cloned for each solver thread, and a single instance is not thread-safe.
 */
class EndEffectorKinematicsQuadraticCost : public ocs2::StateInputCostGaussNewtonAd {
 public:
  /**
   * `referenceManager`, when given, shapes the reference state's base pose with the locomotion heuristics before the
   * link's reference pose is computed from it (SwitchedModelReferenceManager::shapeBasePose). Pass it for a link whose
   * orientation follows the base - the torso - or its term regularizes the tilt towards the unshaped target while the
   * base-pose costs regularize it towards the shaped one, and the stiffer of the two cancels the heuristic. Null leaves
   * the reference exactly as the target trajectory has it.
   */
  EndEffectorKinematicsQuadraticCost(const EndEffectorKinematicsWeights& weights,
                                     const PinocchioInterface& pinocchioInterface,
                                     const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
                                     const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                                     const std::string& endEffectorName,
                                     const ModelSettings& modelSettings,
                                     const SwitchedModelReferenceManager* absl_nullable referenceManager = nullptr);

  ~EndEffectorKinematicsQuadraticCost() override = default;
  EndEffectorKinematicsQuadraticCost& operator=(const EndEffectorKinematicsQuadraticCost&) = delete;
  EndEffectorKinematicsQuadraticCost(EndEffectorKinematicsQuadraticCost&&) = delete;
  EndEffectorKinematicsQuadraticCost& operator=(EndEffectorKinematicsQuadraticCost&&) = delete;
  EndEffectorKinematicsQuadraticCost* absl_nonnull clone() const override { return new EndEffectorKinematicsQuadraticCost(*this); }

  vector_t getParameters(scalar_t time, const TargetTrajectories& targetTrajectories, const PreComputation& preComputation) const override;

  bool isActive(scalar_t /*time*/) const override { return isActive_; }
  void setActive(bool active) { isActive_ = active; }
  bool getActive() const { return isActive_; }

  void getWeights(vector12_t& weights) const { weights = sqrtWeights_.cwiseProduct(sqrtWeights_); }
  void setWeights(const vector12_t& weights) { sqrtWeights_ = weights.cwiseSqrt(); }

 protected:
  EndEffectorKinematicsQuadraticCost(const EndEffectorKinematicsQuadraticCost& other);

  static EndEffectorKinematicsCostElement<scalar_t> getReferenceCostElement(const vector_t& state,
                                                                            const vector_t& input,
                                                                            const EndEffectorKinematics<scalar_t>& endEffectorKinematics);

  ad_vector_t costVectorFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& input,
                                 const ad_vector_t& parameters) override;

  vector12_t sqrtWeights_;
  size_t n_parameters_ = 25;
  pinocchio::FrameIndex frameID_;
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  const std::unique_ptr<EndEffectorKinematics<scalar_t>> endEffectorKinematicsPtr_;
  std::unique_ptr<MpcRobotModelBase<ad_scalar_t>> mpcRobotModelADPtr_;
  bool isActive_ = true;
  /// Shapes the reference base pose when set; see the constructor. Shared, read-only, across the solver's clones.
  const SwitchedModelReferenceManager* absl_nullable referenceManagerPtr_ = nullptr;
};

}  // namespace ocs2::humanoid
