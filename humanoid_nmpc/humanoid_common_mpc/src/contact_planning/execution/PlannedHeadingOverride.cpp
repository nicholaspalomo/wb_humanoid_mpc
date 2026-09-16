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

#include "humanoid_common_mpc/contact_planning/execution/PlannedHeadingOverride.h"

#include <ocs2_robotic_tools/common/RotationTransforms.h>

namespace ocs2::humanoid {

std::string PlannedHeadingOverride::describe() const {
  return "the plan's heading replaces the commanded base yaw of the MPC target trajectory";
}

void PlannedHeadingOverride::overrideTarget(const ExecutionContext& ctx, TargetTrajectories& targetTrajectories) const {
  if (!ctx.hasPlan() || !ctx.activePlan->hasHeading()) return;
  const ContactPlan& plan = *ctx.activePlan;
  const AngularCenterOfMass* acom = (acom_ != nullptr) ? acom_->get() : nullptr;
  const size_t n = targetTrajectories.timeTrajectory.size();
  for (size_t i = 0; i < n; ++i) {
    vector_t& stateRef = targetTrajectories.stateTrajectory[i];
    vector3_t euler = mpcRobotModel_->getBaseOrientationEulerZYX(stateRef);
    const std::optional<scalar_t> heading = plan.headingAtTime(targetTrajectories.timeTrajectory[i]);
    if (heading.has_value()) {
      scalar_t offset = 0.0;
      if (acom != nullptr) offset = acomXyzToZyx(acom->computeJointOrientationOffset(mpcRobotModel_->getJointAngles(stateRef)))(0);
      euler(0) = moduloAngleWithReference(*heading - offset, euler(0));
      mpcRobotModel_->setBaseOrientationEulerZYX(stateRef, euler);
    }
  }
}

}  // namespace ocs2::humanoid
