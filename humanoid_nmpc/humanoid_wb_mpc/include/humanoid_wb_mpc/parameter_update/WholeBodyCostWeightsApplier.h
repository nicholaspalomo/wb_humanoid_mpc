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

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"

namespace ocs2::humanoid {

/**
 * Applies the weights of the whole-body MPC's CppAD-taped costs, which enter the generated code as parameters, so that
 * an update is a parameter write and nothing is taped again: task_space_foot_cost to the swing-foot cost of every foot
 * (EndEffectorDynamicsFootCost::setWeights(), converted with wholeBodyFootCostWeightsFromConfig(), which refuses what
 * the start-up refuses, such as active_phases "swing_and_stance"), and joint_torque_weights to the joint torque cost
 * (JointTorqueCostCppAd::setWeights(), converted with jointTorqueWeightsFromConfig() on the whole-body layout). A block
 * whose term the running problem does not carry is not converted.
 *
 * Not thread-safe (HotFieldApplier).
 */
class WholeBodyCostWeightsApplier final : public HotFieldApplier {
 public:
  /** Returns the fields it applies: task_space_foot_cost and joint_torque_weights. */
  static absl::Span<const absl::string_view> staticFields();

  WholeBodyCostWeightsApplier() = default;

  absl::string_view name() const override { return "WholeBodyCostWeightsApplier"; }
  absl::Span<const absl::string_view> fields() const final { return staticFields(); }
  void apply(HotUpdateTarget& target) override;

 private:
  /** Writes task_space_foot_cost into the foot cost of every foot of every worker's problem. */
  static void applyFootCostWeights(HotUpdateTarget& target);
  /** Writes joint_torque_weights into the joint torque cost of every worker's problem. */
  static void applyJointTorqueWeights(HotUpdateTarget& target);
};

}  // namespace ocs2::humanoid
