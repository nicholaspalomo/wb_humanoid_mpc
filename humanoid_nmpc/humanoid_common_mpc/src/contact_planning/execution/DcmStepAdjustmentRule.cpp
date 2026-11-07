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

#include "humanoid_common_mpc/contact_planning/execution/DcmStepAdjustmentRule.h"

#include <cstddef>
#include <optional>
#include <string>

#include "absl/strings/str_cat.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionContext.h"

namespace ocs2::humanoid {

std::string DcmStepAdjustmentRule::describe() const {
  return absl::StrCat("landing target += ", params_.gain,
                      " * DCM error propagated to touch-down, |offset| <= ",
                      params_.maxNormErrPosFootRtWorldEwrtWorldXy,
                      " m, clipped to the reachable region");
}

void DcmStepAdjustmentRule::configure(const ContactPlanningConfig& config) {
  params_ = config.dcmStepAdjustment;
}

BipedFeetArray<vector2_t> DcmStepAdjustmentRule::correctFootholds(
    const ExecutionContext& ctx, const ModeSchedule& schedule) const {
  BipedFeetArray<vector2_t> errPosFeetRtWorldEwrtWorldXy =
      makeBipedFeetArray(vector2_t(vector2_t::Zero()));
  if (!ctx.hasPlan() || !ctx.hasPredictedComState) {
    return errPosFeetRtWorldEwrtWorldXy;
  }
  const scalar_t omega = ctx.omega();
  const ContactPlan& plan = *ctx.activePlan;
  // Deviation from what the whole-body controller itself predicted for now,
  // measured minus predicted; the anchor velocity stands in for the CoM's, as
  // in the planner's LIP
  // (ContactPlannerInput::proxyLinvelComRtWorldEwrtWorldXy). It is zero while
  // the robot does what the NMPC expects, however far the planner's reduced
  // model has drifted, and non-zero only under a real disturbance.
  const vector2_t errPosDcmRtWorldEwrtWorldXy =
      computePosDcmRtWorldEwrtWorldXy(ctx.posComRtWorldEwrtWorldXy,
                                      ctx.proxyLinvelComRtWorldEwrtWorldXy,
                                      omega) -
      computePosDcmRtWorldEwrtWorldXy(
          ctx.predictedPosComRtWorldEwrtWorldXy,
          ctx.predictedProxyLinvelComRtWorldEwrtWorldXy, omega);

  for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
    const std::optional<SwingWindow> phase =
        swingPhaseAtTime(schedule, foot, ctx.time);
    if (!phase.has_value()) continue;
    const scalar_t touchDownTime = phase->touchDown;
    const std::optional<vector2_t> plannedDesPosFootRtWorldEwrtWorldXy =
        plan.getDesPosFootRtWorldEwrtWorldXy(foot, touchDownTime);
    if (!plannedDesPosFootRtWorldEwrtWorldXy.has_value()) continue;
    const vector2_t errPosFootRtWorldEwrtWorldXy =
        computeErrPosFootRtWorldEwrtWorldXy(
            errPosDcmRtWorldEwrtWorldXy, omega, touchDownTime - ctx.time,
            params_.gain, params_.maxNormErrPosFootRtWorldEwrtWorldXy);
    const std::optional<LipState> atTouchDown =
        lipReferenceState(plan, omega, touchDownTime);
    const vector2_t touchdownPosComRtWorldEwrtWorldXy =
        atTouchDown.has_value() ? atTouchDown->posComRtWorldEwrtWorldXy
                                : plan.posComRtWorldEwrtWorldXyTraj.back();
    // The reachable region is the planner's, in the frame the planner placed
    // that foothold in: the planning heading at touch-down. Without the heading
    // model that is the snapshot heading turned by the plan's rate up to the
    // touch-down, not the snapshot heading itself.
    const scalar_t touchdownYawHeadingRtWorld =
        plan.getYawHeadingRtWorld(touchDownTime);
    const vector2_t desPosFootRtWorldEwrtWorldXy = clipFootholdToReach(
        *plannedDesPosFootRtWorldEwrtWorldXy + errPosFootRtWorldEwrtWorldXy,
        foot, touchdownPosComRtWorldEwrtWorldXy, touchdownYawHeadingRtWorld,
        *ctx.config);
    // The adjusted foothold minus the planned one.
    errPosFeetRtWorldEwrtWorldXy[foot] =
        desPosFootRtWorldEwrtWorldXy - *plannedDesPosFootRtWorldEwrtWorldXy;
  }
  return errPosFeetRtWorldEwrtWorldXy;
}

}  // namespace ocs2::humanoid
