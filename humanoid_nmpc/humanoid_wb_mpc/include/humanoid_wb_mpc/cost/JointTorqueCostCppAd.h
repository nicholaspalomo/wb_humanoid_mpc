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
#include "absl/strings/string_view.h"
#include "ocs2_core/cost/StateInputGaussNewtonCostAd.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"

namespace ocs2::humanoid {

/**
 * The joint-torque cost of the whole-body MPC: a Gauss-Newton cost, taped with CppAD, on the joint torques of the
 * inverse dynamics (computeJointTorques, the full mass matrix), each weighted by the square root of its entry of
 * `weights`. The weights are the parameters of the taped function, so setWeights() retunes a running cost without
 * taping or compiling anything. Not thread-safe; the solver clones one per worker thread, and each clone owns its own
 * copy of the robot model.
 */
class JointTorqueCostCppAd final : public StateInputCostGaussNewtonAd {
 public:
  // LINT.IfChange(joint_torque_cost_names)
  /** The name WBMpcInterface adds the cost under in the problem's cost collection. */
  static constexpr char kTermName[] = "jointTorqueCost";
  /**
   * The cost name WBMpcInterface builds its CppAD library name from (libraryName(kLibraryCostName)). It is spelled apart
   * from kTermName so that renaming the term never renames - and so regenerates - the library.
   */
  static constexpr char kLibraryCostName[] = "jointTorqueCost";
  // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/test/testJointTorqueCostLibraryName.cpp:term_and_library_names)

  /**
   * Tapes, or loads, the library libraryName(costName) under modelSettings.modelFolderCppAd. `weights` has one entry
   * per joint of `mpcRobotModel` (checked: a mismatch is a programming error); the robot model is cloned.
   */
  JointTorqueCostCppAd(const vector_t& weights,
                       const PinocchioInterface& pinocchioInterface,
                       const WBAccelMpcRobotModel<ad_scalar_t>& mpcRobotModel,
                       const std::string& costName,
                       const ModelSettings& modelSettings);

  ~JointTorqueCostCppAd() override = default;
  JointTorqueCostCppAd& operator=(const JointTorqueCostCppAd&) = delete;
  JointTorqueCostCppAd(JointTorqueCostCppAd&&) = delete;
  JointTorqueCostCppAd& operator=(JointTorqueCostCppAd&&) = delete;
  JointTorqueCostCppAd* absl_nonnull clone() const override { return new JointTorqueCostCppAd(*this); }

  vector_t getParameters(scalar_t /*time*/,
                         const TargetTrajectories& /*targetTrajectories*/,
                         const PreComputation& /*preComputation*/) const override {
    return sqrtWeights_;
  }

  /**
   * The name the CppAD library of the cost called `costName` is compiled and cached under. It carries the version of the
   * taped inverse dynamics, so that a library taped from an earlier computeJointTorques - the one that left out the base
   * coupling of the mass matrix - is never loaded from the cache in its place (the robots ship
   * model_settings.recompile_libraries_cpp_ad: false).
   */
  static std::string libraryName(absl::string_view costName);

  /**
   * Weighs the joint torques by `weights` from the next evaluation on, as a cost built with them would: they are the
   * parameters of the taped function. For the parameter updater, between solves.
   *
   * @return InvalidArgument when `weights` does not have one entry per joint of the cost, which then keeps its weights.
   */
  absl::Status setWeights(const vector_t& weights);

 private:
  JointTorqueCostCppAd(const JointTorqueCostCppAd& other);

  ad_vector_t costVectorFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& input,
                                 const ad_vector_t& parameters) override;

  vector_t sqrtWeights_;

  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  // The cost's own copy of the robot model, cloned by the constructor and by every copy; never null.
  std::unique_ptr<WBAccelMpcRobotModel<ad_scalar_t>> mpcRobotModelPtr_;
};

}  // namespace ocs2::humanoid
