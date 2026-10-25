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

#include "humanoid_common_mpc/contact_planning/constraint/HipYawRangeConstraint.h"

#include <cmath>
#include <string>

#include "absl/strings/str_cat.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

namespace ocs2::humanoid {

std::string HipYawRangeConstraint::describe() const {
  std::string out = "psi_i - theta - (theta_base - theta at planning time) in";
  for (size_t foot = 0; foot < kNumContacts; ++foot)
    absl::StrAppend(&out, " [", params_.lower[foot], ", ", params_.upper[foot],
                    "]");
  absl::StrAppend(&out, " rad (hip yaw limits from the model) at every node, ",
                  penaltyText());
  return out;
}

void HipYawRangeConstraint::configure(const ContactPlanningConfig& config) {
  params_ = config.hipYawRange;
  configurePenalty(config, config.hipYawRange.slack);
}

scalar_t HipYawRangeConstraint::baseYawOffset(
    const ContactPlannerInput& input) {
  if (!input.baseYaw.has_value()) return 0.0;
  return moduloAngleWithReference(*input.baseYaw, input.heading) -
         input.heading;
}

void HipYawRangeConstraint::addRows(const ContactPlanningContext& ctx,
                                    int /*node*/,
                                    RowBuilder& rows) const {
  // The hip yaw joint turns the foot against the pelvis: lower_i <= psi_i -
  // theta_base <= upper_i. The planner's heading theta is the whole-body
  // heading (the ACoM's twist heading when the robot has an ACoM evaluator),
  // and the pelvis sits delta = theta_base - theta away from it. The planner
  // has no model of how delta evolves - it is the yaw the joints add to the
  // ACoM (acomHeadingOffset), which the whole-body MPC decides - so it is held
  // at its value at planning time over the horizon: lower_i + delta <= psi_i -
  // theta <= upper_i + delta. Without an ACoM (or with the base yaw unset)
  // delta is zero and the bound is the joint range itself. What is left out is
  // the change of delta over the horizon, which the whole-body MPC still sees
  // against the joint's own limits.
  const scalar_t delta = baseYawOffset(*ctx.input);
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    rows.addSoft({{idx_.footYaw[foot], 1.0}, {idx_.heading, -1.0}},
                 /*uCoefficients=*/{}, params_.lower[foot] + delta,
                 params_.upper[foot] + delta, penalty_);
  }
}

}  // namespace ocs2::humanoid
