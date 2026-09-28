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

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"

namespace ocs2::humanoid {

/** [s] slack on the plan's ends, so that a knot sampled on the plan's own grid is inside it despite rounding. */
inline constexpr scalar_t kPlanCoverageTolerance = 1e-9;

/**
 * Whether `time` lies inside the horizon the plan was computed for, [startTime, endTime()].
 *
 * Every per-node lookup of ContactPlan (comPositionAtTime, comVelocityAtTime, headingAtTime, ...) CLAMPS to the first
 * or the last node rather than reporting that the query left the horizon. A reference written from such a lookup past
 * the plan's end is therefore frozen at the last node - a constant position that still asks for the last node's
 * velocity - so every rule that writes a reference from the plan, and every accessor that hands a planned quantity to
 * a cost, asks this first and leaves the operator's reference alone outside it.
 */
inline bool planCoversTime(const ContactPlan& plan, scalar_t time) {
  return time >= plan.startTime - kPlanCoverageTolerance && time <= plan.endTime() + kPlanCoverageTolerance;
}

}  // namespace ocs2::humanoid
