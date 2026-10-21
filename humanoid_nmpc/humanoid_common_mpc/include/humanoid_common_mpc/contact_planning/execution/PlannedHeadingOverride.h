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

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionRule.h"

namespace ocs2::humanoid {

/**
 * Computes the heading the joints `jointAngles` add to a level base: the twist heading psi_h(Exp(Delta_theta(q_j)))
 * (orientation/BaseOrientation.h) of the ACoM attitude of the identity base, in (-pi, pi]. A rotation about the world z
 * axis adds to the twist, so for a level base at yaw psi the ACoM heading psi_h(q_z(psi) (x) Exp(Delta_theta)) is
 * psi plus this offset, modulo 2 pi, exactly; for a tilted base it differs by the tilt's coupling into the twist.
 * `jointAngles` as for AngularCenterOfMass::acomAttitude().
 */
scalar_t acomHeadingOffset(const AngularCenterOfMass& acom, const vector_t& jointAngles);

/**
 * `planned_heading_override`: the plan's heading replaces the commanded base yaw of the MPC's target trajectory, so
 * that the whole-body controller is asked for the turn the ground can support rather than the raw command. With an
 * ACoM evaluator the reference base yaw is set so that the reference ACoM heading equals the planned heading (the
 * ACoM cost compares the ACoM of the state with the ACoM of the reference state): the knot's ZYX yaw becomes the planned
 * heading minus acomHeadingOffset() of the knot's joints, re-branched next to its yaw (moduloAngleWithReference), with
 * its pitch and roll kept. That is exact for the level knots the target calculators write. Lives with the reference
 * manager, which owns the robot model and the evaluator it needs.
 *
 * Only the knots inside the plan's horizon are rewritten (planCoversTime): past the plan's end the plan's heading
 * lookup clamps to its last node, which froze the yaw reference there instead of leaving the operator's.
 *
 * Not thread-safe, like every ContactPlanningTerm: the planner that owns it configures and evaluates it from one thread at a time.
 */
class PlannedHeadingOverride final : public ExecutionRule {
 public:
  /**
   * `acom` is the manager's evaluator slot, null for none: it may be filled after construction and is read at every
   * cycle, so the slot, like `mpcRobotModel`, must outlive the rule.
   */
  PlannedHeadingOverride(const MpcRobotModelBase<scalar_t>& mpcRobotModel, const std::unique_ptr<AngularCenterOfMass>* absl_nullable acom)
      : mpcRobotModel_(&mpcRobotModel), acom_(acom) {}

  std::string describe() const override;
  std::vector<std::string> requiredBlocks() const override { return {term::kHeadingDoubleIntegrator}; }
  void configure(const ContactPlanningConfig& /*config*/) override {}
  void overrideTarget(const ExecutionContext& ctx, TargetTrajectories& targetTrajectories) const override;
  bool rewritesTarget() const override { return true; }

 private:
  const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel_;
  const std::unique_ptr<AngularCenterOfMass>* absl_nullable acom_;
};

}  // namespace ocs2::humanoid
