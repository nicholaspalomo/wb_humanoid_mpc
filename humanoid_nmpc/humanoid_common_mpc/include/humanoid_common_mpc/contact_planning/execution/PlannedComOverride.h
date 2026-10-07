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

#include <string>
#include <vector>

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionRule.h"

namespace ocs2::humanoid {

/**
 * `planned_com_override`: the reduced model's center-of-mass trajectory replaces the horizontal center-of-mass
 * reference of the MPC's target trajectory.
 *
 * This is the reference plumbing of arXiv:2502.15630, not a correction heuristic: in the paper the reduced-order
 * solution is embedded into the whole-body reference the controller tracks (its equations 11-14), and the contact
 * sequence is only ever a reference. Without it the two layers optimize for contradictory motion, and the lateral
 * direction is where that shows first. The H-LIP's period-two orbit requires the center of mass to be *falling
 * towards the swing foot* at the pre-impact instant - on the Atlas's 1.0805 m pendulum (its model's center of mass
 * above its soles) and a 0.25 m step width, 0.132 m/s for the Atlas cadence (0.25 s single support, 0.05 s double
 * support), 0.18 m/s for a 0.35 s instantaneous exchange.
 * The target trajectory built from the operator's command asks for the opposite: a straight line with zero lateral
 * velocity. The whole-body MPC then holds the center of mass laterally still, the planner reads that state back at the
 * next cycle, concludes that no lateral step is needed and narrows the step towards `minStepWidth`; the support
 * narrows, the next cycle starts further from the orbit still, and the robot sidesteps and falls.
 *
 * The rule writes the plan into the reference the same way the planner produced it: the reference horizontal center of
 * mass becomes the planned one, and the reference linear momentum - which in the centroidal state *is* the center of
 * mass velocity - becomes the planned center of mass velocity. The reference is expressed through the base pose, so
 * the planned position is written as the base position that puts the reference center of mass where the plan wants it,
 * using the measured offset between the two at this cycle. Heights, orientations and joints are left alone.
 *
 * Only the knots inside the plan's horizon are rewritten (planCoversTime). Past the plan's end the target keeps the
 * operator's own knots, which the reference manager appends for exactly that stretch: the plan's lookups clamp to its
 * last node, so rewriting them froze the tail at the last planned position while asking for the last planned velocity.
 */
class PlannedComOverride final : public ExecutionRule {
 public:
  explicit PlannedComOverride(const MpcRobotModelBase<scalar_t>& mpcRobotModel) : mpcRobotModel_(&mpcRobotModel) {}

  std::string describe() const override;
  std::vector<std::string> requiredBlocks() const override { return {}; }
  void configure(const ContactPlanningConfig& /*config*/) override {}
  bool needsComState() const override { return true; }
  void overrideTarget(const ExecutionContext& ctx, TargetTrajectories& targetTrajectories) const override;
  bool rewritesTarget() const override { return true; }

 private:
  const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel_;
};

}  // namespace ocs2::humanoid
