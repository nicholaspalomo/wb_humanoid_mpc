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

#include "legged_common_mpc/contact_planning/execution/PlannedComOverride.h"

#include <cstddef>
#include <optional>
#include <string>

#include "Eigen/Geometry"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/contact_planning/ContactPlan.h"
#include "legged_common_mpc/contact_planning/execution/ExecutionContext.h"
#include "legged_common_mpc/contact_planning/execution/PlanCoverage.h"

namespace ocs2::legged {

std::string PlannedComOverride::describe() const {
  return "the plan's center of mass replaces the horizontal center-of-mass "
         "reference of the MPC target trajectory";
}

void PlannedComOverride::overrideTarget(
    const ExecutionContext& ctx, TargetTrajectories& targetTrajectories) const {
  if (!ctx.hasPlan()) return;
  const ContactPlan& plan = *ctx.activePlan;
  if (plan.posComRtWorldEwrtWorldXyTraj.empty()) return;

  // The reference carries the base pose, the plan carries the center of mass.
  // The horizontal offset between them is measured at this cycle, in world
  // axes at the measured heading. It is fixed in the base for a fixed posture,
  // so a knot whose reference heading differs from the measured one by dpsi
  // carries it turned by dpsi.
  const vector2_t measPosComRtBaseEwrtWorldXy =
      ctx.posComRtWorldEwrtWorldXy - ctx.posBaseRtWorldEwrtWorldXy;

  const size_t numPoints = targetTrajectories.timeTrajectory.size();
  for (size_t i = 0; i < numPoints; ++i) {
    const scalar_t time = targetTrajectories.timeTrajectory[i];
    // Only the knots the plan covers. ContactPlan's lookups clamp to the first
    // and the last node instead of saying that the query left the horizon, so a
    // knot past the plan's end - the operator's own knots, which the reference
    // manager appends exactly so that the reference tail follows the command
    // once the plan runs out - came back frozen at the plan's last position
    // while its velocity went on asking for the plan's last, non-zero velocity.
    if (!planCoversTime(plan, time)) continue;
    const std::optional<vector2_t> plannedPosComRtWorldEwrtWorldXy =
        plan.evaluatePosComRtWorldEwrtWorldXy(time);
    const std::optional<vector2_t> plannedLinvelComRtWorldEwrtWorldXy =
        plan.evaluateLinvelComRtWorldEwrtWorldXy(time);
    if (!plannedPosComRtWorldEwrtWorldXy.has_value() ||
        !plannedLinvelComRtWorldEwrtWorldXy.has_value()) {
      continue;
    }

    vector_t& stateRef = targetTrajectories.stateTrajectory[i];
    // No turn is exactly the identity, so a knot at the measured heading gets
    // the measured offset bit for bit.
    const vector2_t refPosComRtBaseEwrtWorldXy =
        Eigen::Rotation2D<scalar_t>(
            mpcRobotModel_->getYawBaseRtWorld(stateRef) - ctx.yawBaseRtWorld) *
        measPosComRtBaseEwrtWorldXy;
    vector3_t posBaseRtWorldEwrtWorld =
        mpcRobotModel_->getPosBaseRtWorldEwrtWorld(stateRef);
    posBaseRtWorldEwrtWorld.head<kHorizontalPositionDim>() =
        *plannedPosComRtWorldEwrtWorldXy - refPosComRtBaseEwrtWorldXy;
    mpcRobotModel_->setPosBaseRtWorldEwrtWorld(stateRef,
                                               posBaseRtWorldEwrtWorld);

    // The plan's CoM velocity becomes the knot's CoM velocity, or on a
    // whole-body knot its base velocity, which stands in for it.
    vector3_t refLinvelComRtWorldEwrtWorld =
        getProxyLinvelComRtWorldEwrtWorld(*mpcRobotModel_, stateRef);
    refLinvelComRtWorldEwrtWorld.head<kHorizontalPositionDim>() =
        *plannedLinvelComRtWorldEwrtWorldXy;
    setProxyLinvelComRtWorldEwrtWorld(*mpcRobotModel_, stateRef,
                                      refLinvelComRtWorldEwrtWorld);
  }
}

}  // namespace ocs2::legged
