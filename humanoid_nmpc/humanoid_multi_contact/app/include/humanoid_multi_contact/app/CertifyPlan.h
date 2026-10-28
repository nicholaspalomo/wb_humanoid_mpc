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

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/stability/certifier_report.nproto.h"

namespace ocs2::humanoid::multi_contact {

/**
 * Loads the scenario at `scenarioPath` (loadScenario()) and the plan at
 * `planPath` made for it (loadPlanFile(), which refuses a plan of another
 * contact set or scene), certifies every knot of the plan with the scenario's
 * margins (design section 6), and returns the plan with the report in its
 * `report`. With `samplesBetweenKnots` > 0 it certifies
 * planBetweenKnots(plan, samplesBetweenKnots) instead, and returns that. The
 * errors of the loaders and of the certifier; InvalidArgument for a negative
 * `samplesBetweenKnots`.
 */
absl::StatusOr<config::MultiContactPlan> certifyPlanFile(
    absl::string_view scenarioPath,
    absl::string_view planPath,
    int samplesBetweenKnots);

/**
 * `plan` with `samples` knots between every two consecutive ones, at equal
 * fractions of the time between them, as the MPC's reference interpolates a
 * plan (MpcRobotModelBase::interpolateStateTrajectory()): the base position,
 * the joints, the momenta and the wrenches linearly (the wrenches only when
 * both knots have the same contacts, else those of the first), the base's
 * orientation slerped along the shorter way, and the contacts of the first
 * knot, whose mode the MPC holds until the next. Without its report.
 */
config::MultiContactPlan planBetweenKnots(const config::MultiContactPlan& plan,
                                          int samples);

/** The report as text: a verdict, then one line per knot with its stance,
 * margins and failures. */
std::string formatReport(const config::CertifierReport& report);

}  // namespace ocs2::humanoid::multi_contact
