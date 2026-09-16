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

#include "humanoid_common_mpc/contact_planning/execution/EnergyCadenceModulationRule.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

namespace {
constexpr scalar_t kMinTimeShift = 1e-6;       // [s] smaller event shifts are not applied
constexpr scalar_t kMinRemainingSwing = 0.02;  // [s] a re-timed touch-down stays at least this far in the future
}  // namespace

std::string EnergyCadenceModulationRule::describe() const {
  std::ostringstream out;
  out << "touch-down shift = -" << params_.gain << " s/J * (E - E_pred) beyond a " << params_.deadband
      << " J deadband, within the swing duration limits";
  return out.str();
}

void EnergyCadenceModulationRule::configure(const ContactPlanningConfig& config) {
  if (config.energyCadenceModulation.gain < 0.0) throw std::invalid_argument("[energy_cadence_modulation] gain must be >= 0");
  if (config.energyCadenceModulation.deadband < 0.0) throw std::invalid_argument("[energy_cadence_modulation] deadband must be >= 0");
  params_ = config.energyCadenceModulation;
  limits_ = config.shared.gaitLimits;
}

void EnergyCadenceModulationRule::beginCycle(ExecutionContext& ctx) const {
  ctx.cadenceTouchDownShift.fill(0.0);
  if (!ctx.hasPlan() || !ctx.hasPredictedComState) return;
  const scalar_t omega = ctx.omega();
  const ContactPlan& plan = *ctx.activePlan;
  // The planned support point (ZMP) is the reference point of the orbital energy; the energy itself is compared between
  // the measured state and what the whole-body controller predicted for now.
  const std::optional<LipState> reference = lipReferenceState(plan, omega, ctx.time);
  if (!reference.has_value()) return;
  // Orbital energy along the planned heading at the current time: a CoM that carries more energy than the controller
  // expected passes over the support earlier and the step is brought forward, less energy delays it.
  const scalar_t yaw = plan.headingAtTime(ctx.time).value_or(plan.yaw);
  const vector2_t heading(std::cos(yaw), std::sin(yaw));
  const scalar_t measured = lipOrbitalEnergy(heading.dot(ctx.com - reference->zmp), heading.dot(ctx.comVelocity), omega, ctx.totalMass);
  const scalar_t predicted =
      lipOrbitalEnergy(heading.dot(ctx.predictedCom - reference->zmp), heading.dot(ctx.predictedComVelocity), omega, ctx.totalMass);
  // Deviations within the deadband re-time nothing; beyond it the shift is measured from the band's edge, so that the
  // touch-down moves continuously with the deviation instead of jumping at the threshold.
  const scalar_t deviation = measured - predicted;
  const scalar_t beyondDeadband = std::max(0.0, std::abs(deviation) - params_.deadband);
  ctx.cadenceTouchDownShift.fill(-params_.gain * std::copysign(beyondDeadband, deviation));
}

bool EnergyCadenceModulationRule::adaptSwingingFoot(const ExecutionContext& ctx,
                                                    size_t foot,
                                                    scalar_t liftOff,
                                                    scalar_t touchDown,
                                                    ModeSchedule& schedule,
                                                    SwingTimingLatch& latch,
                                                    ContactEventReport& report) const {
  // Not applied while the swing is being extended past its planned touch-down (late touch-down search).
  if (latch.lateExtension > 0.0) return false;
  const scalar_t time = ctx.time;
  const auto tdIndex = touchDownEventIndex(schedule, foot, time);
  if (!tdIndex.has_value()) return false;
  scalar_t target = latch.nominalTouchDownTime + ctx.cadenceTouchDownShift[foot];
  target = std::clamp(target, liftOff + limits_.minSwingDuration, liftOff + limits_.maxSwingDuration);
  const scalar_t previousEvent = (*tdIndex > 0) ? schedule.eventTimes[*tdIndex - 1] : time;
  const scalar_t earliest = std::max(time, previousEvent) + kMinRemainingSwing;
  // A request earlier than what is still feasible brings the touch-down forward as far as allowed, but never pushes a
  // touch-down that is already imminent further out: flooring at "now + margin" on every cycle would drag the touch-down
  // along with the clock and the foot would never land.
  if (target < earliest) target = std::min(earliest, touchDown);
  const scalar_t shift = target - touchDown;
  if (std::abs(shift) > kMinTimeShift && shiftEventsFrom(schedule, *tdIndex, shift)) {
    latch.cadenceShift = target - latch.nominalTouchDownTime;
    report.type = ContactEventReport::Type::CADENCE_SHIFT;
    report.touchDownTime = target;
    report.timeShift = shift;
    return true;
  }
  return false;
}

}  // namespace ocs2::humanoid
