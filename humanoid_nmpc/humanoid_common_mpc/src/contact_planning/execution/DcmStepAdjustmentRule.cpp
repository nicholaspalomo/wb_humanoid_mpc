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

#include "humanoid_common_mpc/contact_planning/execution/DcmStepAdjustmentRule.h"

#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

std::string DcmStepAdjustmentRule::describe() const {
  std::ostringstream out;
  out << "landing target += " << params_.gain << " * DCM error propagated to touch-down, |offset| <= " << params_.maxOffset
      << " m, clipped to the reachable region";
  return out.str();
}

void DcmStepAdjustmentRule::configure(const ContactPlanningConfig& config) {
  if (config.dcmStepAdjustment.gain < 0.0) throw std::invalid_argument("[dcm_step_adjustment] gain must be >= 0");
  if (config.dcmStepAdjustment.maxOffset < 0.0) throw std::invalid_argument("[dcm_step_adjustment] maxOffset must be >= 0");
  params_ = config.dcmStepAdjustment;
}

feet_array_t<vector2_t> DcmStepAdjustmentRule::correctFootholds(const ExecutionContext& ctx, const ModeSchedule& schedule) const {
  feet_array_t<vector2_t> adjustments = makeFeetArray(vector2_t(vector2_t::Zero()));
  if (!ctx.hasPlan() || !ctx.hasPredictedComState) return adjustments;
  const scalar_t omega = ctx.omega();
  const ContactPlan& plan = *ctx.activePlan;
  // Deviation from what the whole-body controller itself predicted for now. It is zero while the robot does what the
  // NMPC expects, however far the planner's reduced model has drifted, and non-zero only under a real disturbance.
  const vector2_t dcmError = computeDcm(ctx.com, ctx.comVelocity, omega) - computeDcm(ctx.predictedCom, ctx.predictedComVelocity, omega);

  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const auto phase = swingPhaseAtTime(schedule, foot, ctx.time);
    if (!phase.has_value()) continue;
    const scalar_t touchDownTime = phase->second;
    const std::optional<vector2_t> landing = plan.footholdAtTime(foot, touchDownTime);
    if (!landing.has_value()) continue;
    const vector2_t adjustment = dcmStepAdjustment(dcmError, omega, touchDownTime - ctx.time, params_.gain, params_.maxOffset);
    const std::optional<LipState> atTouchDown = lipReferenceState(plan, omega, touchDownTime);
    const vector2_t comAtTouchDown = atTouchDown.has_value() ? atTouchDown->com : plan.comPosition.back();
    // The reachable region is the planner's, in the frame the planner wrote it in for that node: the planned heading at
    // touch-down (the heading at the snapshot, plan.yaw, is the same thing without the heading model).
    const scalar_t yawAtTouchDown = plan.headingAtTime(touchDownTime).value_or(plan.yaw);
    const vector2_t clipped = clipFootholdToReach(*landing + adjustment, foot, comAtTouchDown, yawAtTouchDown, *ctx.config);
    adjustments[foot] = clipped - *landing;
  }
  return adjustments;
}

}  // namespace ocs2::humanoid
