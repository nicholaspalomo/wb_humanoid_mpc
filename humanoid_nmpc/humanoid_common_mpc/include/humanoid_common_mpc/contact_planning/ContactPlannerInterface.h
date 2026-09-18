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

#include <string>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"

namespace ocs2::humanoid {

/**
 * What ContactPlannerModule needs of a contact planner, so that the implementation can be chosen by name in the
 * configuration (ContactPlannerFactory, `planner.type`).
 *
 * Two implementations exist: HlipContactPlanner, the closed-form reduced-order stepper of arXiv:2502.15630, and
 * LipContactPlanner, the mixed-integer program. Everything downstream of a plan - the reference manager, the merge
 * with the executed schedule, the swing trajectories, the target contact poses - reads a ContactPlan and does not
 * care which of the two produced it.
 *
 * A planner is used from one thread at a time (the module's worker, or the solver thread in synchronous mode) but not
 * from both at once; implementations need no locking of their own.
 */
class ContactPlannerInterface {
 public:
  virtual ~ContactPlannerInterface() = default;

  /**
   * Plans from the given input. Must not throw on a failure to plan: an invalid ContactPlan is returned instead, and
   * the reference manager then keeps the schedule it is executing.
   */
  virtual ContactPlan plan(const ContactPlannerInput& input) = 0;

  /** Replaces the configuration. Called between plans, never during one. */
  virtual void setConfig(const ContactPlanningConfig& config) = 0;

  /** Drops whatever the planner carries over between plans (a warm start, the previous plan). */
  virtual void reset() = 0;

  /** The assembled formulation, for the start-up print: what this planner is and what it was configured with. */
  virtual std::string getFormulationSummary() const = 0;
};

}  // namespace ocs2::humanoid
