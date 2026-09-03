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

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

/**
 * @brief Enumeration of available MPC cost terms.
 */
enum class MpcCostType {
  StateInputQuadraticCost,
  StateQuadraticCost,
  InputQuadraticCost,
  TerminalCost,
  IcpCost,
  TaskSpaceFootCost,
  TaskSpaceTorsoCost,
  ExternalTorqueCost,
  JointTorqueCost,
};

/**
 * @brief Enumeration of available MPC soft constraint terms (barriers / penalties).
 */
enum class MpcSoftConstraintType {
  JointLimits,
  FootCollision,
  FrictionForceCone,
  ContactMomentXY,
  ContactWrenchCone,
};

/**
 * @brief Enumeration of available MPC hard constraint terms (equality / inequality constraints).
 */
enum class MpcHardConstraintType {
  ZeroWrench,
  ZeroVelocity,
  NormalVelocity,
  KneeJointMimic,
};

/**
 * @brief Container holding the active MPC formulation tasks parsed from YAML.
 */
struct MpcFormulationTasks {
  absl::flat_hash_set<MpcCostType> costs;
  absl::flat_hash_set<MpcSoftConstraintType> softConstraints;
  absl::flat_hash_set<MpcHardConstraintType> hardConstraints;

  bool hasCost(MpcCostType type) const {
    return costs.contains(type);
  }

  bool hasSoftConstraint(MpcSoftConstraintType type) const {
    return softConstraints.contains(type);
  }

  bool hasHardConstraint(MpcHardConstraintType type) const {
    return hardConstraints.contains(type);
  }
};

// String to Enum conversions (supports snake_case and camelCase, case-insensitive)
absl::StatusOr<MpcCostType> stringToMpcCostType(absl::string_view name);
absl::StatusOr<std::string> mpcCostTypeToString(MpcCostType type);

absl::StatusOr<MpcSoftConstraintType> stringToMpcSoftConstraintType(absl::string_view name);
absl::StatusOr<std::string> mpcSoftConstraintTypeToString(MpcSoftConstraintType type);

absl::StatusOr<MpcHardConstraintType> stringToMpcHardConstraintType(absl::string_view name);
absl::StatusOr<std::string> mpcHardConstraintTypeToString(MpcHardConstraintType type);

/**
 * @brief Loads the active MPC formulation tasks from the specified YAML configuration file.
 *
 * Reads from "hard_constraints", "soft_constraints", and "costs" keys (or nested under "tasks" / "mpc_tasks").
 *
 * @param taskFile Path to the task.yaml configuration file.
 * @param verbose If true, logs the loaded tasks via LOG(INFO).
 * @return absl::StatusOr<MpcFormulationTasks> The parsed task configuration, or error status.
 */
absl::StatusOr<MpcFormulationTasks> loadMpcFormulationTasks(absl::string_view taskFile, bool verbose = false);

}  // namespace ocs2::humanoid
