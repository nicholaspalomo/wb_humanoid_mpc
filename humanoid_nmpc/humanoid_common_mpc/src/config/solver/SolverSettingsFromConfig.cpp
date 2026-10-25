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
#include "absl/time/time.h"
#include "ocs2_core/integration/Integrator.h"
#include "ocs2_core/integration/SensitivityIntegrator.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_oc/rollout/RolloutSettings.h"
#include "ocs2_sqp/SqpSettings.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigValueChecks.h"
#include "humanoid_mpc_config/mpc_settings_config.nproto.h"
#include "humanoid_mpc_config/rollout_settings_config.nproto.h"
#include "humanoid_mpc_config/sqp_settings_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {
namespace {

// The integrators a multiple_shooting block can name, each by its OCS2 name
// (sensitivity_integrator::toString()).
// LINT.IfChange(sensitivity_integrators)
constexpr std::array<SensitivityIntegratorType, 3> kSensitivityIntegrators = {
    SensitivityIntegratorType::EULER, SensitivityIntegratorType::RK2,
    SensitivityIntegratorType::RK4};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/sqp_settings_config.proto:integrator_names)

// The integrators a rollout block can name, each by its OCS2 name
// (integrator_type::toString()).
// LINT.IfChange(rollout_integrators)
constexpr std::array<IntegratorType, 5> kRolloutIntegrators = {
    IntegratorType::EULER, IntegratorType::ODE45, IntegratorType::ODE45_OCS2,
    IntegratorType::MODIFIED_MIDPOINT, IntegratorType::RK4};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/rollout_settings_config.proto:integrator_names)

/** The sensitivity integrator named `name`; InvalidArgument naming the field
 * and listing the integrators otherwise. */
absl::StatusOr<SensitivityIntegratorType> sensitivityIntegratorNamed(
    absl::string_view name) {
  for (const SensitivityIntegratorType type : kSensitivityIntegrators) {
    if (sensitivity_integrator::toString(type) == name) {
      return type;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "multiple_shooting.integrator_type: '", name,
      "' is not an integrator of the SQP solver; the integrators are ",
      absl::StrJoin(
          kSensitivityIntegrators, ", ",
          [](std::string* absl_nonnull out, SensitivityIntegratorType type) {
            absl::StrAppend(out, sensitivity_integrator::toString(type));
          })));
}

/** The rollout integrator named `name`; InvalidArgument naming the field and
 * listing the integrators otherwise. */
absl::StatusOr<IntegratorType> rolloutIntegratorNamed(absl::string_view name) {
  for (const IntegratorType type : kRolloutIntegrators) {
    if (integrator_type::toString(type) == name) {
      return type;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "rollout.integrator_type: '", name,
      "' is not an integrator of the rollouts; the integrators are ",
      absl::StrJoin(kRolloutIntegrators, ", ",
                    [](std::string* absl_nonnull out, IntegratorType type) {
                      absl::StrAppend(out, integrator_type::toString(type));
                    })));
}

// The highest SCHED_FIFO priority on Linux; 0 keeps a thread on the
// time-sharing scheduler.
constexpr int32_t kMaxRealtimePriority = 99;

/** `value` as a count; InvalidArgument naming `field` when it is negative. */
absl::StatusOr<size_t> countOf(absl::string_view field, int32_t value) {
  if (value < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, ": ", value, " is negative; a count is 0 or more"));
  }
  return static_cast<size_t>(value);
}

/**
 * OK when the real-valued fields of a multiple_shooting block are in the ranges
 * the SQP solver terminates with: dt > 0 (a non-positive step never reaches the
 * end of the horizon, TimeDiscretization.cpp), 0 < alpha_decay < 1 and 0 <
 * alpha_min <= 1 (otherwise the line search never shrinks the step below
 * alpha_min, SqpSolver.cpp), positive tolerances, 0 <= g_min <= g_max with
 * g_max > 0, and armijo_factor and gamma_c in [0, 1) (the filter line search's
 * sufficient-decrease factors). NaN fails every check.
 */
absl::Status checkSqpRanges(const mpc_config::SqpSettingsConfig& config) {
  RETURN_IF_ERROR(requirePositive("multiple_shooting.dt", config.dt));
  RETURN_IF_ERROR(requireInOpenInterval("multiple_shooting.alpha_decay",
                                        config.alpha_decay, /*low=*/0.0,
                                        /*high=*/1.0));
  RETURN_IF_ERROR(requireInLeftOpenInterval("multiple_shooting.alpha_min",
                                            config.alpha_min, /*low=*/0.0,
                                            /*high=*/1.0));
  RETURN_IF_ERROR(
      requirePositive("multiple_shooting.delta_tol", config.delta_tol));
  RETURN_IF_ERROR(
      requirePositive("multiple_shooting.cost_tol", config.cost_tol));
  RETURN_IF_ERROR(requirePositive("multiple_shooting.g_max", config.g_max));
  RETURN_IF_ERROR(requireNonNegative("multiple_shooting.g_min", config.g_min));
  RETURN_IF_ERROR(requireOrdered("multiple_shooting.g_min", config.g_min,
                                 "multiple_shooting.g_max", config.g_max));
  RETURN_IF_ERROR(requireInHalfOpenInterval("multiple_shooting.armijo_factor",
                                            config.armijo_factor, /*low=*/0.0,
                                            /*high=*/1.0));
  RETURN_IF_ERROR(requireInHalfOpenInterval(
      "multiple_shooting.gamma_c", config.gamma_c, /*low=*/0.0, /*high=*/1.0));
  if (config.thread_priority < 0 ||
      config.thread_priority > kMaxRealtimePriority) {
    return absl::InvalidArgumentError(absl::StrCat(
        "multiple_shooting.thread_priority: ", config.thread_priority,
        " is not in 0-", kMaxRealtimePriority,
        " (SCHED_FIFO's priorities, or 0 for the time-sharing scheduler)"));
  }
  return absl::OkStatus();
}

}  // namespace

std::vector<std::string> sensitivityIntegratorNames() {
  std::vector<std::string> names;
  for (const SensitivityIntegratorType type : kSensitivityIntegrators)
    names.push_back(sensitivity_integrator::toString(type));
  return names;
}

std::vector<std::string> rolloutIntegratorNames() {
  std::vector<std::string> names;
  for (const IntegratorType type : kRolloutIntegrators)
    names.push_back(integrator_type::toString(type));
  return names;
}

absl::StatusOr<sqp::Settings> toSqpSettings(
    const mpc_config::SqpSettingsConfig& config) {
  RETURN_IF_ERROR(checkSqpRanges(config));
  sqp::Settings settings;
  ASSIGN_OR_RETURN(
      settings.sqpIteration,
      countOf("multiple_shooting.sqp_iteration", config.sqp_iteration));
  settings.deltaTol = config.delta_tol;
  settings.costTol = config.cost_tol;
  settings.alpha_decay = config.alpha_decay;
  settings.alpha_min = config.alpha_min;
  settings.g_max = config.g_max;
  settings.g_min = config.g_min;
  settings.armijoFactor = config.armijo_factor;
  settings.gamma_c = config.gamma_c;
  settings.useFeedbackPolicy = config.use_feedback_policy;
  settings.dt = config.dt;
  ASSIGN_OR_RETURN(settings.integratorType,
                   sensitivityIntegratorNamed(config.integrator_type));
  settings.projectStateInputEqualityConstraints =
      config.project_state_input_equality_constraints;
  settings.printSolverStatus = config.print_solver_status;
  settings.printSolverStatistics = config.print_solver_statistics;
  settings.printLinesearch = config.print_linesearch;
  settings.enableLogging = config.enable_logging;
  ASSIGN_OR_RETURN(settings.logSize,
                   countOf("multiple_shooting.log_size", config.log_size));
  settings.logFilePath = config.log_file_path;
  ASSIGN_OR_RETURN(settings.nThreads,
                   countOf("multiple_shooting.n_threads", config.n_threads));
  settings.threadPriority = config.thread_priority;
  return settings;
}

absl::StatusOr<rollout::Settings> toRolloutSettings(
    const mpc_config::RolloutSettingsConfig& config) {
  RETURN_IF_ERROR(requirePositive("rollout.abs_tol_ode", config.abs_tol_ode));
  RETURN_IF_ERROR(requirePositive("rollout.rel_tol_ode", config.rel_tol_ode));
  RETURN_IF_ERROR(requirePositive("rollout.time_step", config.time_step));
  rollout::Settings settings;
  settings.absTolODE = config.abs_tol_ode;
  settings.relTolODE = config.rel_tol_ode;
  ASSIGN_OR_RETURN(settings.maxNumStepsPerSecond,
                   countOf("rollout.max_num_steps_per_second",
                           config.max_num_steps_per_second));
  settings.timeStep = config.time_step;
  ASSIGN_OR_RETURN(settings.integratorType,
                   rolloutIntegratorNamed(config.integrator_type));
  settings.checkNumericalStability = config.check_numerical_stability;
  settings.reconstructInputTrajectory = config.reconstruct_input_trajectory;
  return settings;
}

absl::StatusOr<mpc::Settings> toMpcSettings(
    const mpc_config::MpcSettingsConfig& config) {
  RETURN_IF_ERROR(requirePositive("mpc.time_horizon", config.time_horizon));
  RETURN_IF_ERROR(requirePositive("mpc.mrt_desired_frequency",
                                  config.mrt_desired_frequency));
  mpc::Settings settings;
  settings.timeHorizon_ = config.time_horizon;
  // Absent: OCS2's negative sentinel, the whole horizon
  // (mpc::Settings::solutionTimeWindow_).
  if (config.solution_time_window.has_value()) {
    RETURN_IF_ERROR(requirePositive("mpc.solution_time_window",
                                    *config.solution_time_window));
    settings.solutionTimeWindow_ = *config.solution_time_window;
  }
  settings.debugPrint_ = config.debug_print;
  settings.coldStart_ = config.cold_start;
  // Absent: OCS2's negative sentinel, one solve per new observation as fast as
  // they come (mpc::Settings::mpcDesiredFrequency_).
  if (config.mpc_desired_frequency.has_value()) {
    RETURN_IF_ERROR(requirePositive("mpc.mpc_desired_frequency",
                                    *config.mpc_desired_frequency));
    settings.mpcDesiredFrequency_ = *config.mpc_desired_frequency;
  }
  settings.mrtDesiredFrequency_ = config.mrt_desired_frequency;
  return settings;
}

absl::StatusOr<SolverSettings> solverSettingsFromConfig(
    const mpc_config::TaskFile& task) {
  SolverSettings settings;
  ASSIGN_OR_RETURN(settings.sqpSettings, toSqpSettings(task.multiple_shooting));
  ASSIGN_OR_RETURN(settings.rolloutSettings, toRolloutSettings(task.rollout));
  ASSIGN_OR_RETURN(settings.mpcSettings, toMpcSettings(task.mpc));
  settings.verbose = task.interface.verbose;
  return settings;
}

absl::StatusOr<MpcNodeSettings> mpcNodeSettingsFromConfig(
    const mpc_config::TaskFile& task) {
  RETURN_IF_ERROR(checkSqpRanges(task.multiple_shooting));
  // A timeout of 0 would keep a stale walking command forever: a publisher that
  // is killed, or a link that is lost, would leave the robot walking.
  RETURN_IF_ERROR(requirePositive("interface.walking_command_timeout",
                                  task.interface.walking_command_timeout));
  return MpcNodeSettings{
      .solverThreadPriority = task.multiple_shooting.thread_priority,
      .walkingCommandTimeout =
          absl::Seconds(task.interface.walking_command_timeout)};
}

}  // namespace ocs2::humanoid
