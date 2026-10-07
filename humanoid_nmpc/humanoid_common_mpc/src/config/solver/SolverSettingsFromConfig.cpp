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

#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/integration/Integrator.h"
#include "ocs2_core/integration/SensitivityIntegrator.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_oc/rollout/RolloutSettings.h"
#include "ocs2_oc/rollout/RootFinderType.h"
#include "ocs2_sqp/SqpSettings.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_mpc_config/mpc_settings_config.nproto.h"
#include "humanoid_mpc_config/rollout_settings_config.nproto.h"
#include "humanoid_mpc_config/sqp_settings_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {
namespace {

// The integrators a multiple_shooting block can name, each by its OCS2 name (sensitivity_integrator::toString()).
// LINT.IfChange(sensitivity_integrators)
constexpr std::array<SensitivityIntegratorType, 3> kSensitivityIntegrators = {
    SensitivityIntegratorType::EULER, SensitivityIntegratorType::RK2, SensitivityIntegratorType::RK4};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/sqp_settings_config.proto:integrator_names)

// The integrators a rollout block can name, each by its OCS2 name (integrator_type::toString()).
// LINT.IfChange(rollout_integrators)
constexpr std::array<IntegratorType, 5> kRolloutIntegrators = {IntegratorType::EULER, IntegratorType::ODE45, IntegratorType::ODE45_OCS2,
                                                               IntegratorType::MODIFIED_MIDPOINT, IntegratorType::RK4};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/rollout_settings_config.proto:integrator_names)

/** The sensitivity integrator named `name`; InvalidArgument naming the field and listing the integrators otherwise. */
absl::StatusOr<SensitivityIntegratorType> sensitivityIntegratorNamed(absl::string_view name) {
  for (const SensitivityIntegratorType type : kSensitivityIntegrators) {
    if (sensitivity_integrator::toString(type) == name) {
      return type;
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("multiple_shooting.integrator_type: '", name, "' is not an integrator of the SQP solver; the integrators are ",
                   absl::StrJoin(kSensitivityIntegrators, ", ", [](std::string* absl_nonnull out, SensitivityIntegratorType type) {
                     absl::StrAppend(out, sensitivity_integrator::toString(type));
                   })));
}

/** The rollout integrator named `name`; InvalidArgument naming the field and listing the integrators otherwise. */
absl::StatusOr<IntegratorType> rolloutIntegratorNamed(absl::string_view name) {
  for (const IntegratorType type : kRolloutIntegrators) {
    if (integrator_type::toString(type) == name) {
      return type;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "rollout.integrator_type: '", name, "' is not an integrator of the rollouts; the integrators are ",
      absl::StrJoin(kRolloutIntegrators, ", ",
                    [](std::string* absl_nonnull out, IntegratorType type) { absl::StrAppend(out, integrator_type::toString(type)); })));
}

/**
 * The name a rollout block gives the root finder `type`: its enumerator's (OCS2 has no toString() for RootFinderType).
 * -Werror=switch refuses this switch when OCS2 adds an algorithm that it does not name.
 */
absl::string_view rootFinderName(RootFinderType type) {
  switch (type) {
    case RootFinderType::ANDERSON_BJORCK:
      return "ANDERSON_BJORCK";
    case RootFinderType::PEGASUS:
      return "PEGASUS";
    case RootFinderType::ILLINOIS:
      return "ILLINOIS";
    case RootFinderType::REGULA_FALSI:
      return "REGULA_FALSI";
  }
  return "unknown";
}

// The root finders a rollout block can name, each by rootFinderName().
// LINT.IfChange(root_finders)
constexpr std::array<RootFinderType, 4> kRootFinders = {RootFinderType::ANDERSON_BJORCK, RootFinderType::PEGASUS, RootFinderType::ILLINOIS,
                                                        RootFinderType::REGULA_FALSI};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/rollout_settings_config.proto:root_finder_names)

/** The root finder named `name`; InvalidArgument naming the field and listing the root finders otherwise. */
absl::StatusOr<RootFinderType> rootFinderNamed(absl::string_view name) {
  for (const RootFinderType type : kRootFinders) {
    if (rootFinderName(type) == name) return type;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("rollout.root_finder_type: '", name, "' is not a root-finding algorithm of the rollouts; the root finders are ",
                   absl::StrJoin(kRootFinders, ", ",
                                 [](std::string* absl_nonnull out, RootFinderType type) { absl::StrAppend(out, rootFinderName(type)); })));
}

/** `value` as a count; InvalidArgument naming `field` when it is negative. */
absl::StatusOr<size_t> countOf(absl::string_view field, int32_t value) {
  if (value < 0) {
    return absl::InvalidArgumentError(absl::StrCat(field, ": ", value, " is negative; a count is 0 or more"));
  }
  return static_cast<size_t>(value);
}

}  // namespace

std::vector<std::string> sensitivityIntegratorNames() {
  std::vector<std::string> names;
  for (const SensitivityIntegratorType type : kSensitivityIntegrators) names.push_back(sensitivity_integrator::toString(type));
  return names;
}

std::vector<std::string> rolloutIntegratorNames() {
  std::vector<std::string> names;
  for (const IntegratorType type : kRolloutIntegrators) names.push_back(integrator_type::toString(type));
  return names;
}

std::vector<std::string> rootFinderNames() {
  std::vector<std::string> names;
  for (const RootFinderType type : kRootFinders) names.emplace_back(rootFinderName(type));
  return names;
}

absl::StatusOr<sqp::Settings> toSqpSettings(const mpc_config::SqpSettingsConfig& config) {
  sqp::Settings settings;
  ASSIGN_OR_RETURN(settings.sqpIteration, countOf("multiple_shooting.sqp_iteration", config.sqp_iteration));
  settings.deltaTol = config.delta_tol;
  settings.costTol = config.cost_tol;
  settings.alpha_decay = config.alpha_decay;
  settings.alpha_min = config.alpha_min;
  settings.g_max = config.g_max;
  settings.g_min = config.g_min;
  settings.armijoFactor = config.armijo_factor;
  settings.gamma_c = config.gamma_c;
  settings.useFeedbackPolicy = config.use_feedback_policy;
  settings.createValueFunction = config.create_value_function;
  settings.dt = config.dt;
  ASSIGN_OR_RETURN(settings.integratorType, sensitivityIntegratorNamed(config.integrator_type));
  settings.projectStateInputEqualityConstraints = config.project_state_input_equality_constraints;
  settings.extractProjectionMultiplier = config.extract_projection_multiplier;
  settings.printSolverStatus = config.print_solver_status;
  settings.printSolverStatistics = config.print_solver_statistics;
  settings.printLinesearch = config.print_linesearch;
  settings.enableLogging = config.enable_logging;
  ASSIGN_OR_RETURN(settings.logSize, countOf("multiple_shooting.log_size", config.log_size));
  settings.logFilePath = config.log_file_path;
  ASSIGN_OR_RETURN(settings.nThreads, countOf("multiple_shooting.n_threads", config.n_threads));
  settings.threadPriority = config.thread_priority;
  return settings;
}

absl::StatusOr<rollout::Settings> toRolloutSettings(const mpc_config::RolloutSettingsConfig& config) {
  rollout::Settings settings;
  settings.absTolODE = config.abs_tol_ode;
  settings.relTolODE = config.rel_tol_ode;
  ASSIGN_OR_RETURN(settings.maxNumStepsPerSecond, countOf("rollout.max_num_steps_per_second", config.max_num_steps_per_second));
  settings.timeStep = config.time_step;
  ASSIGN_OR_RETURN(settings.integratorType, rolloutIntegratorNamed(config.integrator_type));
  settings.checkNumericalStability = config.check_numerical_stability;
  settings.reconstructInputTrajectory = config.reconstruct_input_trajectory;
  ASSIGN_OR_RETURN(settings.rootFindingAlgorithm, rootFinderNamed(config.root_finder_type));
  settings.maxSingleEventIterations = config.max_single_event_iterations;
  settings.useTrajectorySpreadingController = config.use_trajectory_spreading_controller;
  return settings;
}

mpc::Settings toMpcSettings(const mpc_config::MpcSettingsConfig& config) {
  mpc::Settings settings;
  settings.timeHorizon_ = config.time_horizon;
  settings.solutionTimeWindow_ = config.solution_time_window;
  settings.debugPrint_ = config.debug_print;
  settings.coldStart_ = config.cold_start;
  settings.mpcDesiredFrequency_ = config.mpc_desired_frequency;
  settings.mrtDesiredFrequency_ = config.mrt_desired_frequency;
  return settings;
}

absl::StatusOr<SolverSettings> solverSettingsFromConfig(const mpc_config::TaskFile& task) {
  SolverSettings settings;
  ASSIGN_OR_RETURN(settings.sqpSettings, toSqpSettings(task.multiple_shooting));
  ASSIGN_OR_RETURN(settings.rolloutSettings, toRolloutSettings(task.rollout));
  settings.mpcSettings = toMpcSettings(task.mpc);
  settings.verbose = task.interface.verbose;
  return settings;
}

}  // namespace ocs2::humanoid
