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

#include "humanoid_common_mpc/contact_planning/TargetContactPose.h"

#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"

namespace ocs2::humanoid {

namespace {
TargetContactPose stancePose(size_t foot, const TargetContactPoseInputs& inputs) {
  TargetContactPose pose;
  pose.valid = true;
  pose.kind = TargetContactPose::Kind::STANCE;
  pose.position = inputs.footPositions[foot].head<2>();
  pose.height = inputs.footPositions[foot](2);
  pose.yaw = inputs.footYaws[foot];
  pose.yawPlanned = false;
  pose.touchDownTime = inputs.time;
  return pose;
}
}  // namespace

feet_array_t<TargetContactPose> computeTargetContactPoses(const ContactPlan& plan,
                                                          const ModeSchedule& schedule,
                                                          const TargetContactPoseInputs& inputs) {
  feet_array_t<TargetContactPose> poses = makeFeetArray(TargetContactPose{});
  if (!plan.valid) return poses;

  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    // The swing in flight, or else the foot's next swing: an event at the query time counts as passed in the schedule
    // queries, so the phase lookup at the next lift-off returns that swing.
    std::optional<std::pair<scalar_t, scalar_t>> swing = swingPhaseAtTime(schedule, foot, inputs.time);
    const bool inFlight = swing.has_value();
    if (!inFlight) {
      const std::optional<scalar_t> liftOff = currentOrNextLiftOffTime(schedule, foot, inputs.time);
      if (liftOff.has_value()) swing = swingPhaseAtTime(schedule, foot, *liftOff);
    }
    std::optional<vector2_t> landing;
    if (swing.has_value()) landing = plan.footholdAtTime(foot, swing->second);
    if (!landing.has_value()) {
      poses[foot] = stancePose(foot, inputs);
      continue;
    }

    TargetContactPose& pose = poses[foot];
    pose.valid = true;
    pose.kind = inFlight ? TargetContactPose::Kind::SWING_IN_FLIGHT : TargetContactPose::Kind::NEXT_SWING;
    pose.position = *landing + inputs.dcmStepAdjustment[foot];
    pose.height = inputs.terrainHeight;
    pose.touchDownTime = swing->second;
    const std::optional<scalar_t> plannedYaw = plan.footYawAtTime(foot, swing->second);
    pose.yawPlanned = plannedYaw.has_value();
    pose.yaw = plannedYaw.value_or(inputs.footYaws[foot]);
  }
  return poses;
}

}  // namespace ocs2::humanoid
