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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/cost/StateInputGaussNewtonCostAd.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_pinocchio_interface/PinocchioStateInputMapping.h"
#include "pinocchio/algorithm/frames.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsCostHelpers.h"

namespace ocs2::humanoid {

/**
 * The swing-foot cost of the whole-body MPC for one contact: a Gauss-Newton
 * cost, taped with CppAD, on the foot's tilt from the ground plane
 * (tiltAngleToPlane(), in radians: the orientation weights are per rad^2), its
 * twist and its accelerations against flat ground at rest, weighted by the task
 * file's task_space_foot_cost.weights and scaled by the swing trajectory
 * planner's impact proximity. Active while the reference manager has the foot
 * in swing. The weights enter the taped function as parameters, so setWeights()
 * retunes a running cost without taping or compiling anything. Not thread-safe;
 * the solver clones one per worker thread, and each clone owns its own copy of
 * the robot model.
 *
 * The impact proximity s multiplies the weighted error vector, so the cost is
 * s^2 sum_i w_i e_i^2: the configured
 * swing_trajectory_config.impact_proximity_factor_mid_point_value scales the
 * cost at mid-swing by its square (0.005 is a factor of 2.5e-5), as it scales
 * the swing velocity rows of the centroidal foot cost.
 */
class EndEffectorDynamicsFootCost final : public StateInputCostGaussNewtonAd {
 public:
  /**
   * Tapes, or loads, the cost's CppAD library `modelName` under
   * modelSettings.modelFolderCppAd; the MPC passes libraryModelName() of the
   * foot. `referenceManager` must outlive the cost and its clones; the robot
   * model is cloned.
   *
   * @return InvalidArgument when a weight is negative or not finite (the cost
   * weighs each error by the square root of its weight) or `contactIndex` is
   * not a contact of modelSettings; NotFound when the contact's frame,
   *         modelSettings.contactNames[contactIndex], is not a frame of
   * `pinocchioInterface`'s model.
   */
  static absl::StatusOr<std::unique_ptr<EndEffectorDynamicsFootCost>> Create(
      const SwitchedModelReferenceManager& referenceManager,
      const EndEffectorDynamicsWeights& weights,
      const PinocchioInterface& pinocchioInterface,
      const WBAccelMpcRobotModel<ad_scalar_t>& mpcRobotModel,
      size_t contactIndex,
      const std::string& modelName,
      const ModelSettings& modelSettings);

  /** Returns the name WBMpcInterface adds the cost of the contact `footName`
   * under in the problem's cost collection. */
  static std::string termName(absl::string_view footName);

  /**
   * Returns the CppAD model name the MPC compiles and caches the library of the
   * contact `footName`'s cost under. It is spelled apart from termName() so
   * that renaming the term never renames - and so regenerates - the library,
   * and it ends in kLibraryVersionSuffix.
   */
  static std::string libraryModelName(absl::string_view footName);

  /**
   * The version of the residual in libraryModelName(), changed with every
   * change of it, so that a library taped from an earlier residual is never
   * loaded in its place (the robots ship recompile_libraries_cpp_ad: false). No
   * suffix was the half-angle tilt -sin(theta / 2) u
   * (quaternionDistanceToPlane()); "_tiltAngle" is the tilt angle theta u
   * (tiltAngleToPlane()).
   */
  static constexpr char kLibraryVersionSuffix[] = "_tiltAngle";

  ~EndEffectorDynamicsFootCost() override = default;
  EndEffectorDynamicsFootCost& operator=(const EndEffectorDynamicsFootCost&) =
      delete;
  EndEffectorDynamicsFootCost(EndEffectorDynamicsFootCost&&) = delete;
  EndEffectorDynamicsFootCost& operator=(EndEffectorDynamicsFootCost&&) =
      delete;
  EndEffectorDynamicsFootCost* absl_nonnull clone() const override {
    return new EndEffectorDynamicsFootCost(*this);
  }

  vector_t getParameters(scalar_t time,
                         const TargetTrajectories& targetTrajectories,
                         const PreComputation& preComputation) const override;

  bool isActive(scalar_t time) const override {
    return !referenceManagerPtr_->isInContact(time, contactIndex_);
  }

  /**
   * Weighs the errors by `weights` from the next evaluation on, as a cost built
   * with them would: they are parameters of the taped function. For the
   * parameter updater, between solves.
   *
   * @return InvalidArgument naming the first weight that is negative or not
   * finite; the cost then keeps its weights.
   */
  absl::Status setWeights(const EndEffectorDynamicsWeights& weights);

 private:
  EndEffectorDynamicsFootCost(
      const SwitchedModelReferenceManager& referenceManager,
      const EndEffectorDynamicsWeights& weights,
      const PinocchioInterface& pinocchioInterface,
      const WBAccelMpcRobotModel<ad_scalar_t>& mpcRobotModel,
      size_t contactIndex,
      const std::string& modelName,
      const ModelSettings& modelSettings);
  EndEffectorDynamicsFootCost(const EndEffectorDynamicsFootCost& other);

  ad_vector_t costVectorFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& input,
                                 const ad_vector_t& parameters) override;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;

  Eigen::Matrix<scalar_t, kFootCostDim, 1> sqrtWeights_;

  size_t contactIndex_;
  const pinocchio::FrameIndex frameID_;
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  // The cost's own copy of the robot model, cloned by the constructor and by
  // every copy; never null.
  std::unique_ptr<WBAccelMpcRobotModel<ad_scalar_t>> mpcRobotModelPtr_;
};

}  // namespace ocs2::humanoid
