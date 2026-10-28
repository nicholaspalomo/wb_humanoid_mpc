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

#pragma once

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"

namespace ocs2::humanoid::multi_contact {

/**
 * A multi-contact planner (design sections 4 and 5): from a scenario, a
 * certified plan that takes the robot from its initial pose to the goal.
 * Planners are built by name by MultiContactPlannerRegistry.
 *
 * Not thread-safe: plan() may keep state between calls, such as caches.
 */
class MultiContactPlanner {
 public:
  virtual ~MultiContactPlanner() = default;

  /**
   * Plans `loaded`: a plan whose hashes are its contact set's and terrain's,
   * certified by the scenario's StabilityCertifier, with the report in the
   * plan. An error names what failed: NotFound when no plan is found (the
   * node, edge and condition that failed), FailedPrecondition when a plan is
   * found but the certifier refuses it (the knot and condition).
   */
  virtual absl::StatusOr<config::MultiContactPlan> plan(
      const LoadedScenario& loaded) = 0;

  /** The planner's registry name. */
  virtual absl::string_view name() const = 0;
};

}  // namespace ocs2::humanoid::multi_contact
