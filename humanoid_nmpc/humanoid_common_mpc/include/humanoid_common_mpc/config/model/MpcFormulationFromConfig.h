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
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"

#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_mpc_config/task_file.nproto.h"

/**
 * The formulation the typed task file (humanoid_nmpc/humanoid_mpc_config) selects: the terms of its costs,
 * soft_constraints and hard_constraints lists, its contact schedule source, its contact input parameterization and its
 * centroidal model, each resolved by the registry that owns the names; their messages name the textproto's fields.
 *
 * The retired fields (use_dcm_terminal_cost, use_contact_planning, the mpc_tasks nesting, ...) cannot reach these
 * functions: the strict parser refuses them with what replaced them (the retired fields of TaskFile).
 */
namespace ocs2::humanoid {

/** Whether mpcFormulationTasksFromConfig() logs the terms it reads. */
enum class FormulationLogging {
  kQuiet,
  /** One line per list: what the formulation is made of (an interface with interface.verbose). */
  kLogSummary,
};

/** kLogSummary for a verbose interface (interface.verbose), kQuiet otherwise. */
inline FormulationLogging formulationLoggingFor(bool verbose) {
  return verbose ? FormulationLogging::kLogSummary : FormulationLogging::kQuiet;
}

/**
 * The terms the task file lists, by name: costs, soft_constraints and hard_constraints. A name may use either
 * spelling the registries accept (snake_case, camelCase); a name listed twice is one term.
 *
 * It refuses the combinations that are not a formulation (checkMpcFormulationTasks()).
 *
 * @param taskFile The typed task file.
 * @param logging Whether the terms are logged (the interface's interface.verbose).
 * @return The terms, or InvalidArgument naming the list entry (`costs[2]`) of an unknown name or the entries of a
 *         refused combination.
 */
absl::StatusOr<MpcFormulationTasks> mpcFormulationTasksFromConfig(const mpc_config::TaskFile& taskFile, FormulationLogging logging);

/**
 * Refuses the combinations of terms that are not a formulation, each with a message naming the entries to change:
 * both terminal costs; `zero_velocity` or `normal_velocity` both hard and soft; some but not all of the three
 * contact-implicit terms; `contact_complementarity` beside the hard `zero_wrench`; no `zero_wrench` and no cone;
 * `force_weighted_slip` beside either `zero_velocity`; and the contact-implicit formulation beside the hard
 * `normal_velocity` or without the soft one (humanoid_nmpc/docs/contact_implicit_mpc/README.md, section 3).
 *
 * mpcFormulationTasksFromConfig() runs them.
 *
 * @return OK, or InvalidArgument for the first refused combination, in the order above.
 */
absl::Status checkMpcFormulationTasks(const MpcFormulationTasks& tasks);

/**
 * The task file's contact_schedule_source, resolved by name (contactScheduleSourceNames()).
 *
 * @return InvalidArgument naming contact_schedule_source and listing the valid names, for an unknown name.
 */
absl::StatusOr<ContactScheduleSource> contactScheduleSourceFromConfig(const mpc_config::TaskFile& taskFile);

/**
 * The task file's contact_input_parameterization, resolved by name (contactInputParameterizationNames()).
 *
 * @return InvalidArgument naming contact_input_parameterization and listing the valid names, for an unknown name.
 */
absl::StatusOr<ContactInputParameterization> contactInputParameterizationFromConfig(const mpc_config::TaskFile& taskFile);

// The names of the task file's centroidal_model, one per OCS2 CentroidalModelType.
// LINT.IfChange(centroidal_model_names)
inline constexpr absl::string_view kFullCentroidalDynamicsModel = "full_centroidal_dynamics";
inline constexpr absl::string_view kSingleRigidBodyDynamicsModel = "single_rigid_body_dynamics";
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto:centroidal_model)

/** Every name centroidal_model accepts, in the order of CentroidalModelType. */
std::vector<std::string> centroidalModelNames();

/** The centroidal_model name of `type`. */
absl::string_view centroidalModelName(CentroidalModelType type);

/** The centroidal model named `name`; InvalidArgument naming centroidal_model and listing the names otherwise. */
absl::StatusOr<CentroidalModelType> centroidalModelTypeFromName(absl::string_view name);

/**
 * The task file's centroidal_model, which the centroidal MPC requires (centroidalModelTypeFromName()). It replaced
 * the int code centroidal_model_type, which the parser refuses with the names (a retired field of TaskFile).
 *
 * @return InvalidArgument naming centroidal_model when the file does not set it or names no centroidal model.
 */
absl::StatusOr<CentroidalModelType> centroidalModelTypeFromConfig(const mpc_config::TaskFile& taskFile);

}  // namespace ocs2::humanoid
