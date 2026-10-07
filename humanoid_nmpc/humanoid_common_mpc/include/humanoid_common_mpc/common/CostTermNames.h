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

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

/**
 * The names the MPC interfaces register their cost and soft-constraint terms under in the optimal control problem's
 * collections, and that the hot-field appliers of the parameter updater (parameter_update/) find them again by.
 *
 * An applier treats an absent term as the normal case - most terms are only built for some task lists - so it could
 * never tell a term that is not listed from one whose name had drifted: a rename on one side turned every hot reload of
 * that term into a silent no-op. With one definition a rename is a rename on both sides. The per-foot contact terms are
 * in ContactTermNames.h.
 *
 * These are COLLECTION names only. A CppAD term that used to take its code-generation model name from its collection
 * name has a library name of its own, with its own literal, beside its class, so that a change here can never rename a
 * generated library (test/parameter_update/testCostTermNames.cpp pins the strings).
 */
// LINT.IfChange(cost_term_names)
// The quadratic costs of the state and input weights.
inline constexpr char kStateInputQuadraticCostTerm[] = "stateInputQuadraticCost";
inline constexpr char kStateQuadraticCostTerm[] = "stateQuadraticCost";
inline constexpr char kInputQuadraticCostTerm[] = "inputQuadraticCost";
// The quadratic terminal cost of final_state_weights.
inline constexpr char kTerminalCostTerm[] = "terminalCost";
// The state soft constraints.
inline constexpr char kJointLimitsTerm[] = "jointLimits";
inline constexpr char kFootCollisionTerm[] = "FootCollisionSoftConstraint";
// The instantaneous capture point cost of the centroidal MPC.
inline constexpr char kIcpCostTerm[] = "icp_Cost";

/** The task-space kinematics cost of the link or foot `name` (a task_space_costs entry's name, or a contact's). */
inline std::string taskSpaceKinematicsCostName(absl::string_view name) {
  return absl::StrCat(name, "_TaskSpaceKinematicsCost");
}

/** The external torque cost of the leg of the contact `footName`. */
inline std::string externalTorqueCostName(absl::string_view footName) {
  return absl::StrCat(footName, "_ExternalTorqueQuadraticCost");
}

// The suffix of the stance-foot (zero velocity) constraint after the contact's name (zeroVelocityTermName()).
inline constexpr char kZeroVelocityTermSuffix[] = "_zeroVelocity";

/** The stance-foot (zero velocity) constraint of the contact `footName`, hard or soft. */
inline std::string zeroVelocityTermName(absl::string_view footName) {
  return absl::StrCat(footName, kZeroVelocityTermSuffix);
}

/** The lambda >= 0 barrier of the basis-vector contact inputs of the contact `footName`. */
inline std::string basisNonNegativityTermName(absl::string_view footName) {
  return absl::StrCat(footName, "_basisNonNegativity");
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/test/parameter_update/testCostTermNames.cpp:pinned_cost_term_names)

}  // namespace ocs2::humanoid
