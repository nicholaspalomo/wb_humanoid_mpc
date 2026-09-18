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

#pragma once

#include <optional>

#include <ocs2_core/reference/ModeSchedule.h>
#include <ocs2_core/reference/TargetTrajectories.h>

#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionContext.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningTerm.h"

namespace ocs2::humanoid {

/**
 * A heuristic applied between plans by the reference manager. The schedule hooks are called per foot, in the order of
 * the `execution` list, until a rule reports an event for that foot (the rules then do not stack on one foot in one
 * cycle); the foothold and target hooks run once per cycle.
 */
class ExecutionRule : public ContactPlanningTerm {
 public:
  /** True when the rule compares the measured centre of mass with the NMPC's predicted trajectory. */
  virtual bool needsPredictedTrajectory() const { return false; }
  /** True when the rule reads the measured centre of mass or base position of the cycle (implied by the above). */
  virtual bool needsComState() const { return needsPredictedTrajectory(); }
  /** Once per cycle, before the schedule is adapted (the cadence rule computes its shifts here). */
  virtual void beginCycle(ExecutionContext& /*ctx*/) const {}
  /**
   * The foot is scheduled to swing at ctx.time, between `liftOff` and `touchDown`; the latch identifies the swing.
   * Returns true when the rule changed the schedule and reported it.
   */
  virtual bool adaptSwingingFoot(const ExecutionContext& /*ctx*/,
                                 size_t /*foot*/,
                                 scalar_t /*liftOff*/,
                                 scalar_t /*touchDown*/,
                                 ModeSchedule& /*schedule*/,
                                 SwingTimingLatch& /*latch*/,
                                 ContactEventReport& /*report*/) const {
    return false;
  }
  /**
   * The foot is scheduled to be in contact at ctx.time but is not measured in contact, and the latch holds the swing
   * that just ended. Returns true when the rule extended the swing and reported it; otherwise the latch is cleared.
   */
  virtual bool adaptContactFoot(const ExecutionContext& /*ctx*/,
                                size_t /*foot*/,
                                ModeSchedule& /*schedule*/,
                                SwingTimingLatch& /*latch*/,
                                ContactEventReport& /*report*/) const {
    return false;
  }
  /** A ground search for the swing height reference of a foot (late touch-down), when the rule has one. */
  virtual std::optional<GroundSearchRequest> groundSearch(const ExecutionContext& /*ctx*/,
                                                          size_t /*foot*/,
                                                          const ModeSchedule& /*schedule*/,
                                                          const SwingTimingLatch& /*latch*/) const {
    return std::nullopt;
  }
  /** Per-foot landing target offsets (xy) of the swings in flight; zero for feet the rule does not move. */
  virtual feet_array_t<vector2_t> correctFootholds(const ExecutionContext& /*ctx*/, const ModeSchedule& /*schedule*/) const {
    return makeFeetArray(vector2_t(vector2_t::Zero()));
  }
  /** Rewrites the MPC's target trajectory (the planned heading override). */
  virtual void overrideTarget(const ExecutionContext& /*ctx*/, TargetTrajectories& /*targetTrajectories*/) const {}
};

}  // namespace ocs2::humanoid
