/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include "humanoid_common_mpc/contact_planning/execution/PlannedComOverride.h"

#include <optional>

namespace ocs2::humanoid {

std::string PlannedComOverride::describe() const {
  return "the plan's centre of mass replaces the horizontal centre-of-mass reference of the MPC target trajectory";
}

void PlannedComOverride::overrideTarget(const ExecutionContext& ctx, TargetTrajectories& targetTrajectories) const {
  if (!ctx.hasPlan()) return;
  const ContactPlan& plan = *ctx.activePlan;
  if (plan.comPosition.empty()) return;

  // The reference carries the base pose, the plan carries the centre of mass. The horizontal offset between them is a
  // property of the posture, so the one measured at this cycle is the right one to carry the plan across.
  const vector2_t comOffsetFromBase = ctx.com - ctx.basePosition;

  const size_t numPoints = targetTrajectories.timeTrajectory.size();
  for (size_t i = 0; i < numPoints; ++i) {
    const scalar_t time = targetTrajectories.timeTrajectory[i];
    const std::optional<vector2_t> plannedPosition = plan.comPositionAtTime(time);
    const std::optional<vector2_t> plannedVelocity = plan.comVelocityAtTime(time);
    if (!plannedPosition.has_value() || !plannedVelocity.has_value()) continue;

    vector_t& stateRef = targetTrajectories.stateTrajectory[i];
    vector3_t basePosition = mpcRobotModel_->getBasePosition(stateRef);
    basePosition.head<2>() = *plannedPosition - comOffsetFromBase;
    mpcRobotModel_->setBasePosition(stateRef, basePosition);

    vector3_t comVelocity = mpcRobotModel_->getBaseComLinearVelocity(stateRef);
    comVelocity.head<2>() = *plannedVelocity;
    mpcRobotModel_->setBaseComLinearVelocity(stateRef, comVelocity);
  }
}

}  // namespace ocs2::humanoid
