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

#include <array>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::multi_contact {

/**
 * Which of the URDF's joint limits a multi-contact plan respects (the
 * scenario's `joint_limits`, design section 11, decision 6). The joint
 * position limits always hold; the effort limits bind the quasi-static joint
 * torques g(q) - sum J' w of every planned configuration (design section 2)
 * only under kPositionAndEffort. No planner and no certifier reads the URDF's
 * joint velocity limits under either: planner B's timing law moves every
 * joint at most 0.5 rad/s, far below them, and planner A's solves carry no
 * velocity limit.
 */
enum class JointLimits {
  // The position limits and, within the scenario's torque margin, the effort
  // limits: the stance graph's statics check, solve 3's torque constraint and
  // the certifier's condition 1 judge the torques.
  kPositionAndEffort,
  // The position limits alone: the certifier reports the torque utilization
  // for information and never fails a knot on it, and no planner checks it.
  kPositionOnly,
  // As kPositionOnly, with the position limits widened by the scenario's
  // joint_limit_relaxation on both sides for the stance graph planner's
  // configurations: a dummy sim's plan may bend a joint a little past the
  // URDF's range (design section 11, decision 8).
  kRelaxedPosition,
};

/** The names of JointLimits in the scenario, in the order of its values. */
// LINT.IfChange(joint_limits_names)
inline constexpr std::array<absl::string_view, 3> kJointLimitsNames = {
    "position_and_effort", "position_only", "relaxed_position"};
// LINT.ThenChange(//humanoid_nmpc/humanoid_multi_contact/scenarios/scenario.proto:joint_limits_names, //humanoid_nmpc/humanoid_multi_contact/stability/certifier_report.proto:joint_limits_names, //humanoid_nmpc/docs/multi_contact_planning/README.md:joint_limits_names)

/** The JointLimits named `name`; InvalidArgument, listing the known names, for
 * any other. */
absl::StatusOr<JointLimits> jointLimitsOf(absl::string_view name);

/** The name of `limits` (kJointLimitsNames). */
absl::string_view jointLimitsName(JointLimits limits);

/** Whether `limits` bind the joint torques to the effort limits. */
bool enforcesEffortLimits(JointLimits limits);

/** Whether `limits` widen the joint position limits by the scenario's
 * relaxation. */
bool relaxesPositionLimits(JointLimits limits);

/** [rad] The largest relaxation of the position limits a scenario may ask
 * for. */
inline constexpr double kMaxJointLimitRelaxation = 0.3;

}  // namespace ocs2::humanoid::multi_contact
