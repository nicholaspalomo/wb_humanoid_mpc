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

#include "humanoid_common_mpc/contact_planning/TargetContactPose.h"

#include <cstddef>
#include <optional>

#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"

namespace ocs2::humanoid {

namespace {
TargetContactPose makeStanceDesPoseFootRtWorld(
    size_t foot, const TargetContactPoseInputs& inputs) {
  TargetContactPose desPoseFootRtWorld;
  desPoseFootRtWorld.valid = true;
  desPoseFootRtWorld.kind = TargetContactPose::Kind::kStance;
  desPoseFootRtWorld.posFootRtWorldEwrtWorldXy =
      inputs.posFeetRtWorldEwrtWorld[foot].head<kHorizontalPositionDim>();
  desPoseFootRtWorld.posFootRtWorldEwrtWorldZ =
      inputs.posFeetRtWorldEwrtWorld[foot](kZAxisIndex);
  desPoseFootRtWorld.eulerzyxFootRtWorldYaw =
      inputs.eulerzyxFeetRtWorldYaw[foot];
  desPoseFootRtWorld.yawPlanned = false;
  desPoseFootRtWorld.touchDownTime = inputs.time;
  return desPoseFootRtWorld;
}
}  // namespace

BipedFeetArray<TargetContactPose> computeDesPoseFeetRtWorld(
    const ContactPlan& plan,
    const ModeSchedule& schedule,
    const TargetContactPoseInputs& inputs) {
  BipedFeetArray<TargetContactPose> desPoseFeetRtWorld =
      makeBipedFeetArray(TargetContactPose{});
  if (!plan.valid) return desPoseFeetRtWorld;

  for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
    // The swing in flight, or else the foot's next swing: an event at the query
    // time counts as passed in the schedule queries, so the phase lookup at the
    // next lift-off returns that swing.
    std::optional<SwingWindow> swing =
        swingPhaseAtTime(schedule, foot, inputs.time);
    const bool inFlight = swing.has_value();
    if (!inFlight) {
      const std::optional<scalar_t> liftOff =
          currentOrNextLiftOffTime(schedule, foot, inputs.time);
      if (liftOff.has_value()) {
        swing = swingPhaseAtTime(schedule, foot, *liftOff);
      }
    }
    const std::optional<vector2_t> touchdownDesPosFootRtWorldEwrtWorldXy =
        swing.has_value()
            ? plan.getDesPosFootRtWorldEwrtWorldXy(foot, swing->touchDown)
            : std::nullopt;
    if (!swing.has_value() ||
        !touchdownDesPosFootRtWorldEwrtWorldXy.has_value()) {
      desPoseFeetRtWorld[foot] = makeStanceDesPoseFootRtWorld(foot, inputs);
      continue;
    }

    TargetContactPose& desPoseFootRtWorld = desPoseFeetRtWorld[foot];
    desPoseFootRtWorld.valid = true;
    desPoseFootRtWorld.kind = inFlight ? TargetContactPose::Kind::kSwingInFlight
                                       : TargetContactPose::Kind::kNextSwing;
    desPoseFootRtWorld.posFootRtWorldEwrtWorldXy =
        *touchdownDesPosFootRtWorldEwrtWorldXy +
        inputs.errPosFeetRtWorldEwrtWorldXy[foot];
    desPoseFootRtWorld.posFootRtWorldEwrtWorldZ =
        inputs.posSurfaceRtWorldEwrtWorldZ;
    desPoseFootRtWorld.touchDownTime = swing->touchDown;
    const std::optional<scalar_t> plannedDesEulerzyxFootRtWorldYaw =
        plan.getDesEulerzyxFootRtWorldYaw(foot, swing->touchDown);
    desPoseFootRtWorld.yawPlanned =
        plannedDesEulerzyxFootRtWorldYaw.has_value();
    desPoseFootRtWorld.eulerzyxFootRtWorldYaw =
        plannedDesEulerzyxFootRtWorldYaw.value_or(
            inputs.eulerzyxFeetRtWorldYaw[foot]);
  }
  return desPoseFeetRtWorld;
}

}  // namespace ocs2::humanoid
