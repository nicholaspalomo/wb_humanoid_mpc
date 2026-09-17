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

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionRule.h"

namespace ocs2::humanoid {

/**
 * `planned_height_override`: the plan's vertical CoM trajectory (vertical_double_integrator) reaches the whole-body MPC
 * as a change of the base height reference, so that a hop or a running flight has a push-off and a landing to track
 * instead of a constant height.
 *
 * What it writes is the rise and fall the plan predicts from where the robot is now, z_plan(t) - z_plan(t_now), not the
 * deviation from the pendulum height: the pendulum height is a parameter of the reduced model and any standing mismatch
 * between it and the real centre of mass would otherwise become a permanent command to change height. And it writes
 * nothing at all for a plan that keeps a foot on the ground throughout, so walking and standing keep exactly the height
 * reference they have without this rule. The reference is left alone where the plan has no height (no vertical block,
 * or a time outside the plan). Lives with the reference manager, which owns the robot model it needs.
 *
 * The rule is idempotent, which it has to be: the reference manager hands out the same TargetTrajectories object on
 * every solve that brought no new command (BufferedValue keeps the active value until one is published), and the model
 * adds to the base height rather than setting it, so applying the deviation again would stack a second hop on the first
 * and the reference would run away at the solver rate. Every call therefore undoes the offsets the previous one left in
 * this same, unchanged trajectory before it writes the current ones. The heading override has no such state because
 * assigning a yaw is already idempotent.
 */
class PlannedHeightOverride final : public ExecutionRule {
 public:
  explicit PlannedHeightOverride(const MpcRobotModelBase<scalar_t>& mpcRobotModel) : mpcRobotModel_(&mpcRobotModel) {}
  std::string describe() const override;
  std::vector<std::string> requiredBlocks() const override { return {term::kVerticalDoubleIntegrator}; }
  void configure(const ContactPlanningConfig& /*config*/) override {}
  void overrideTarget(const ExecutionContext& ctx, TargetTrajectories& targetTrajectories) const override;

  /** The offsets the last call wrote, for the test. */
  const scalar_array_t& appliedOffsets() const { return appliedOffsets_; }

 private:
  const MpcRobotModelBase<scalar_t>* mpcRobotModel_;
  // What the last call left behind: the times it wrote at, the offset it added to each, and the base height that came
  // out. A trajectory that still matches all three is the one this rule wrote into and nothing has replaced since.
  mutable scalar_array_t appliedTimes_;
  mutable scalar_array_t appliedOffsets_;
  mutable scalar_array_t appliedHeights_;
};

}  // namespace ocs2::humanoid
