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
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"

namespace ocs2::humanoid {

/**
 * What the execution rules of the reference manager read at one solver cycle: the measured contact state, the active
 * plan, the measured and predicted centre of mass, and the per-foot touch-down shifts the cadence rule computes for
 * the schedule adaptation. Filled by ContactPlanningReferenceManager::modifyReferences() (and, for the tests, by
 * adaptScheduleToContactEvents()).
 */
struct ExecutionContext {
  scalar_t time = 0.0;
  contact_flag_t measuredContact = makeFeetArray(true);
  const ContactPlan* activePlan = nullptr;  // a valid plan, or null while none is active
  const ContactPlanningConfig* config = nullptr;
  bool hasPredictedComState = false;
  vector2_t com = vector2_t::Zero();  // measured CoM position / velocity at this cycle
  vector2_t comVelocity = vector2_t::Zero();
  // Measured base position at this cycle. A rule that wants to command a centre-of-mass motion has to write it into a
  // reference state that carries the base pose, and com - basePosition is the horizontal offset between the two.
  vector2_t basePosition = vector2_t::Zero();
  vector2_t predictedCom = vector2_t::Zero();  // the NMPC's own prediction for this cycle
  vector2_t predictedComVelocity = vector2_t::Zero();
  scalar_t totalMass = 0.0;
  feet_array_t<scalar_t> cadenceTouchDownShift = makeFeetArray(0.0);  // [s] per foot, set by energy_cadence_modulation

  scalar_t omega() const { return config->omega(); }
  bool hasPlan() const { return activePlan != nullptr && activePlan->valid; }
};

/** A late touch-down search: the foot height reference continues its planned swing as a straight descent. */
struct GroundSearchRequest {
  scalar_t liftOffTime = 0.0;
  scalar_t plannedTouchDownTime = 0.0;
  scalar_t searchVelocity = 0.0;  // [m/s]
};

}  // namespace ocs2::humanoid
