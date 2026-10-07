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

// The task file's solver blocks become OCS2's settings: an empty block gives the settings structs' defaults, every
// field reaches its setting, integrators are named as OCS2 names them, and what the settings cannot hold is refused
// naming the field. The schema refuses the keys it retired, naming what to do.

#include <array>
#include <cstddef>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/integration/Integrator.h"
#include "ocs2_core/integration/SensitivityIntegrator.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_oc/rollout/RolloutSettings.h"
#include "ocs2_oc/rollout/RootFinderType.h"
#include "ocs2_sqp/SqpSettings.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"
#include "humanoid_mpc_config/mpc_settings_config.nproto.h"
#include "humanoid_mpc_config/rollout_settings_config.nproto.h"
#include "humanoid_mpc_config/sqp_settings_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {
namespace {

/** The task file of the textproto `text`, parsed strictly. */
absl::StatusOr<mpc_config::TaskFile> parseTaskFile(absl::string_view text) {
  absl::StatusOr<humanoid_mpc_config::TaskFile> message = nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(text, "task.textproto");
  if (!message.ok()) {
    return message.status();
  }
  mpc_config::TaskFile task;
  RETURN_IF_ERROR(mpc_config::FromProto(*message, &task));
  return task;
}

void expectSameSqpSettings(const sqp::Settings& actual, const sqp::Settings& expected) {
  EXPECT_EQ(actual.sqpIteration, expected.sqpIteration);
  EXPECT_EQ(actual.deltaTol, expected.deltaTol);
  EXPECT_EQ(actual.costTol, expected.costTol);
  EXPECT_EQ(actual.alpha_decay, expected.alpha_decay);
  EXPECT_EQ(actual.alpha_min, expected.alpha_min);
  EXPECT_EQ(actual.g_max, expected.g_max);
  EXPECT_EQ(actual.g_min, expected.g_min);
  EXPECT_EQ(actual.armijoFactor, expected.armijoFactor);
  EXPECT_EQ(actual.gamma_c, expected.gamma_c);
  EXPECT_EQ(actual.useFeedbackPolicy, expected.useFeedbackPolicy);
  EXPECT_EQ(actual.createValueFunction, expected.createValueFunction);
  EXPECT_EQ(actual.dt, expected.dt);
  EXPECT_EQ(actual.integratorType, expected.integratorType);
  EXPECT_EQ(actual.projectStateInputEqualityConstraints, expected.projectStateInputEqualityConstraints);
  EXPECT_EQ(actual.extractProjectionMultiplier, expected.extractProjectionMultiplier);
  EXPECT_EQ(actual.printSolverStatus, expected.printSolverStatus);
  EXPECT_EQ(actual.printSolverStatistics, expected.printSolverStatistics);
  EXPECT_EQ(actual.printLinesearch, expected.printLinesearch);
  EXPECT_EQ(actual.enableLogging, expected.enableLogging);
  EXPECT_EQ(actual.logSize, expected.logSize);
  EXPECT_EQ(actual.logFilePath, expected.logFilePath);
  EXPECT_EQ(actual.nThreads, expected.nThreads);
  EXPECT_EQ(actual.threadPriority, expected.threadPriority);
  // No file sets the QP solver's settings.
  EXPECT_EQ(actual.hpipmSettings.iter_max, expected.hpipmSettings.iter_max);
  EXPECT_EQ(actual.hpipmSettings.tol_stat, expected.hpipmSettings.tol_stat);
}

void expectSameRolloutSettings(const rollout::Settings& actual, const rollout::Settings& expected) {
  EXPECT_EQ(actual.absTolODE, expected.absTolODE);
  EXPECT_EQ(actual.relTolODE, expected.relTolODE);
  EXPECT_EQ(actual.maxNumStepsPerSecond, expected.maxNumStepsPerSecond);
  EXPECT_EQ(actual.timeStep, expected.timeStep);
  EXPECT_EQ(actual.integratorType, expected.integratorType);
  EXPECT_EQ(actual.checkNumericalStability, expected.checkNumericalStability);
  EXPECT_EQ(actual.reconstructInputTrajectory, expected.reconstructInputTrajectory);
  EXPECT_EQ(actual.rootFindingAlgorithm, expected.rootFindingAlgorithm);
  EXPECT_EQ(actual.maxSingleEventIterations, expected.maxSingleEventIterations);
  EXPECT_EQ(actual.useTrajectorySpreadingController, expected.useTrajectorySpreadingController);
}

void expectSameMpcSettings(const mpc::Settings& actual, const mpc::Settings& expected) {
  EXPECT_EQ(actual.timeHorizon_, expected.timeHorizon_);
  EXPECT_EQ(actual.solutionTimeWindow_, expected.solutionTimeWindow_);
  EXPECT_EQ(actual.debugPrint_, expected.debugPrint_);
  EXPECT_EQ(actual.coldStart_, expected.coldStart_);
  EXPECT_EQ(actual.mpcDesiredFrequency_, expected.mpcDesiredFrequency_);
  EXPECT_EQ(actual.mrtDesiredFrequency_, expected.mrtDesiredFrequency_);
}

/** `status` is InvalidArgument and says `expected`. */
void expectRefused(const absl::Status& status, absl::string_view expected) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_TRUE(absl::StrContains(status.message(), expected)) << status << " does not say " << expected;
}

TEST(SolverSettingsFromConfigTest, AnEmptyBlockGivesTheDefaultsOfTheSettingsStructs) {
  const absl::StatusOr<sqp::Settings> sqpSettings = toSqpSettings(mpc_config::SqpSettingsConfig{});
  ASSERT_TRUE(sqpSettings.ok()) << sqpSettings.status();
  expectSameSqpSettings(*sqpSettings, sqp::Settings{});
  const absl::StatusOr<rollout::Settings> rolloutSettings = toRolloutSettings(mpc_config::RolloutSettingsConfig{});
  ASSERT_TRUE(rolloutSettings.ok()) << rolloutSettings.status();
  expectSameRolloutSettings(*rolloutSettings, rollout::Settings{});
  expectSameMpcSettings(toMpcSettings(mpc_config::MpcSettingsConfig{}), mpc::Settings{});
}

TEST(SolverSettingsFromConfigTest, AnEmptyTaskFileGivesTheDefaults) {
  const absl::StatusOr<mpc_config::TaskFile> task = parseTaskFile(/*text=*/"");
  ASSERT_TRUE(task.ok()) << task.status();
  const absl::StatusOr<SolverSettings> settings = solverSettingsFromConfig(*task);
  ASSERT_TRUE(settings.ok()) << settings.status();
  expectSameSqpSettings(settings->sqpSettings, sqp::Settings{});
  expectSameRolloutSettings(settings->rolloutSettings, rollout::Settings{});
  expectSameMpcSettings(settings->mpcSettings, mpc::Settings{});
  EXPECT_FALSE(settings->verbose);
}

TEST(SolverSettingsFromConfigTest, EveryFieldReachesItsSetting) {
  const absl::StatusOr<mpc_config::TaskFile> task = parseTaskFile(R"pb(
    interface { verbose: true }
    multiple_shooting {
      sqp_iteration: 3
      delta_tol: 2.5e-5
      cost_tol: 5e-3
      alpha_decay: 0.25
      alpha_min: 1e-3
      g_max: 0.5
      g_min: 3e-7
      armijo_factor: 2e-4
      gamma_c: 2e-6
      use_feedback_policy: false
      create_value_function: true
      dt: 0.025
      integrator_type: "RK4"
      project_state_input_equality_constraints: false
      extract_projection_multiplier: true
      print_solver_status: true
      print_solver_statistics: true
      print_linesearch: true
      enable_logging: false
      log_size: 50
      log_file_path: "/tmp/solver_log/"
      n_threads: 2
      thread_priority: 70
    }
    rollout {
      abs_tol_ode: 5e-5
      rel_tol_ode: 5e-3
      max_num_steps_per_second: 20000
      time_step: 0.015
      integrator_type: "RK4"
      check_numerical_stability: true
      reconstruct_input_trajectory: false
      root_finder_type: "ILLINOIS"
      max_single_event_iterations: 7
      use_trajectory_spreading_controller: true
    }
    mpc {
      time_horizon: 1.25
      solution_time_window: 0.3
      debug_print: true
      cold_start: true
      mpc_desired_frequency: 80
      mrt_desired_frequency: 400
    }
  )pb");
  ASSERT_TRUE(task.ok()) << task.status();
  const absl::StatusOr<SolverSettings> settings = solverSettingsFromConfig(*task);
  ASSERT_TRUE(settings.ok()) << settings.status();

  sqp::Settings sqpSettings;
  sqpSettings.sqpIteration = 3;
  sqpSettings.deltaTol = 2.5e-5;
  sqpSettings.costTol = 5.0e-3;
  sqpSettings.alpha_decay = 0.25;
  sqpSettings.alpha_min = 1.0e-3;
  sqpSettings.g_max = 0.5;
  sqpSettings.g_min = 3.0e-7;
  sqpSettings.armijoFactor = 2.0e-4;
  sqpSettings.gamma_c = 2.0e-6;
  sqpSettings.useFeedbackPolicy = false;
  sqpSettings.createValueFunction = true;
  sqpSettings.dt = 0.025;
  sqpSettings.integratorType = SensitivityIntegratorType::RK4;
  sqpSettings.projectStateInputEqualityConstraints = false;
  sqpSettings.extractProjectionMultiplier = true;
  sqpSettings.printSolverStatus = true;
  sqpSettings.printSolverStatistics = true;
  sqpSettings.printLinesearch = true;
  sqpSettings.enableLogging = false;
  sqpSettings.logSize = 50;
  sqpSettings.logFilePath = "/tmp/solver_log/";
  sqpSettings.nThreads = 2;
  sqpSettings.threadPriority = 70;
  expectSameSqpSettings(settings->sqpSettings, sqpSettings);

  rollout::Settings rolloutSettings;
  rolloutSettings.absTolODE = 5.0e-5;
  rolloutSettings.relTolODE = 5.0e-3;
  rolloutSettings.maxNumStepsPerSecond = 20000;
  rolloutSettings.timeStep = 0.015;
  rolloutSettings.integratorType = IntegratorType::RK4;
  rolloutSettings.checkNumericalStability = true;
  rolloutSettings.reconstructInputTrajectory = false;
  rolloutSettings.rootFindingAlgorithm = RootFinderType::ILLINOIS;
  rolloutSettings.maxSingleEventIterations = 7;
  rolloutSettings.useTrajectorySpreadingController = true;
  expectSameRolloutSettings(settings->rolloutSettings, rolloutSettings);

  mpc::Settings mpcSettings;
  mpcSettings.timeHorizon_ = 1.25;
  mpcSettings.solutionTimeWindow_ = 0.3;
  mpcSettings.debugPrint_ = true;
  mpcSettings.coldStart_ = true;
  mpcSettings.mpcDesiredFrequency_ = 80.0;
  mpcSettings.mrtDesiredFrequency_ = 400.0;
  expectSameMpcSettings(settings->mpcSettings, mpcSettings);
  EXPECT_TRUE(settings->verbose);
}

TEST(SolverSettingsFromConfigTest, TheIntegratorsAreNamedAsOcs2NamesThem) {
  for (const absl::string_view name : {"EULER", "RK2", "RK4"}) {
    mpc_config::SqpSettingsConfig config;
    config.integrator_type = std::string(name);
    const absl::StatusOr<sqp::Settings> settings = toSqpSettings(config);
    ASSERT_TRUE(settings.ok()) << settings.status();
    EXPECT_EQ(settings->integratorType, sensitivity_integrator::fromString(std::string(name))) << name;
  }
  for (const absl::string_view name : {"EULER", "ODE45", "ODE45_OCS2", "MODIFIED_MIDPOINT", "RK4"}) {
    mpc_config::RolloutSettingsConfig config;
    config.integrator_type = std::string(name);
    const absl::StatusOr<rollout::Settings> settings = toRolloutSettings(config);
    ASSERT_TRUE(settings.ok()) << settings.status();
    EXPECT_EQ(settings->integratorType, integrator_type::fromString(std::string(name))) << name;
  }
}

TEST(SolverSettingsFromConfigTest, AnUnknownIntegratorIsRefusedListingTheIntegrators) {
  mpc_config::SqpSettingsConfig sqpConfig;
  sqpConfig.integrator_type = "RK3";
  const absl::Status sqpStatus = toSqpSettings(sqpConfig).status();
  expectRefused(sqpStatus, "multiple_shooting.integrator_type: 'RK3'");
  expectRefused(sqpStatus, "the integrators are EULER, RK2, RK4");

  mpc_config::RolloutSettingsConfig rolloutConfig;
  rolloutConfig.integrator_type = "ode45";
  const absl::Status rolloutStatus = toRolloutSettings(rolloutConfig).status();
  expectRefused(rolloutStatus, "rollout.integrator_type: 'ode45'");
  expectRefused(rolloutStatus, "the integrators are EULER, ODE45, ODE45_OCS2, MODIFIED_MIDPOINT, RK4");
}

TEST(SolverSettingsFromConfigTest, ANegativeCountIsRefusedNamingTheField) {
  mpc_config::SqpSettingsConfig iterations;
  iterations.sqp_iteration = -1;
  expectRefused(toSqpSettings(iterations).status(), "multiple_shooting.sqp_iteration: -1");
  mpc_config::SqpSettingsConfig logSize;
  logSize.log_size = -1;
  expectRefused(toSqpSettings(logSize).status(), "multiple_shooting.log_size: -1");
  mpc_config::SqpSettingsConfig threads;
  threads.n_threads = -2;
  expectRefused(toSqpSettings(threads).status(), "multiple_shooting.n_threads: -2");
  mpc_config::RolloutSettingsConfig steps;
  steps.max_num_steps_per_second = -1;
  expectRefused(toRolloutSettings(steps).status(), "rollout.max_num_steps_per_second: -1");

  // 0 is a count.
  mpc_config::SqpSettingsConfig none;
  none.sqp_iteration = 0;
  EXPECT_TRUE(toSqpSettings(none).ok());
}

/** A root finder of OCS2 and the name a rollout block gives it. */
struct NamedRootFinder {
  const char* absl_nonnull name;
  RootFinderType type;
};

TEST(SolverSettingsFromConfigTest, TheRootFindersAreNamedAsTheirEnumerators) {
  // Each name is the RootFinderType enumerator of the same spelling; the int code it replaced was the enumerator's
  // number (0 ANDERSON_BJORCK, 1 PEGASUS, 2 ILLINOIS, 3 REGULA_FALSI), which the order below pins.
  const std::array<NamedRootFinder, 4> kNames = {{{.name = "ANDERSON_BJORCK", .type = RootFinderType::ANDERSON_BJORCK},
                                                  {.name = "PEGASUS", .type = RootFinderType::PEGASUS},
                                                  {.name = "ILLINOIS", .type = RootFinderType::ILLINOIS},
                                                  {.name = "REGULA_FALSI", .type = RootFinderType::REGULA_FALSI}}};
  for (size_t code = 0; code < kNames.size(); ++code) {
    mpc_config::RolloutSettingsConfig config;
    config.root_finder_type = kNames[code].name;
    const absl::StatusOr<rollout::Settings> settings = toRolloutSettings(config);
    ASSERT_TRUE(settings.ok()) << settings.status();
    EXPECT_EQ(settings->rootFindingAlgorithm, kNames[code].type);
    EXPECT_EQ(static_cast<size_t>(settings->rootFindingAlgorithm), code);
  }
  EXPECT_EQ(mpc_config::RolloutSettingsConfig{}.root_finder_type, "ANDERSON_BJORCK") << "the default of rollout::Settings";
  for (const absl::string_view name : {"0", "anderson_bjorck", "BRENT"}) {
    mpc_config::RolloutSettingsConfig config;
    config.root_finder_type = std::string(name);
    const absl::Status status = toRolloutSettings(config).status();
    expectRefused(status, absl::StrCat("rollout.root_finder_type: '", name, "' is not a root-finding algorithm"));
    expectRefused(status, "ANDERSON_BJORCK, PEGASUS, ILLINOIS, REGULA_FALSI");
  }
}

TEST(SolverSettingsFromConfigTest, AnErrorOfABlockIsTheErrorOfTheTaskFile) {
  const absl::StatusOr<mpc_config::TaskFile> task = parseTaskFile(R"pb(rollout { integrator_type: "RK45" })pb");
  ASSERT_TRUE(task.ok()) << task.status();
  expectRefused(solverSettingsFromConfig(*task).status(), "rollout.integrator_type: 'RK45'");
}

TEST(SolverSettingsSchemaTest, TheRetiredKeysAreRefusedSayingWhatToDo) {
  for (const absl::string_view text :
       {"interface { useAnalyticalGradientsDynamics: false }", "interface { use_analytical_gradients_constraints: false }"}) {
    const absl::Status status = parseTaskFile(text).status();
    expectRefused(status, "is retired");
    expectRefused(status, "delete the key");
  }
  const absl::Status ddp = parseTaskFile("ddp { }").status();
  expectRefused(ddp, "'ddp' is retired");
  expectRefused(ddp, "multiple_shooting");
  const absl::Status rootFinder = parseTaskFile("rollout {\n  rootFindingAlgorithm: 1\n}\n").status();
  expectRefused(rootFinder, "task.textproto:2:3: 'rootFindingAlgorithm' is retired");
  expectRefused(rootFinder, R"(root_finder_type: "ANDERSON_BJORCK" where it was 0, "PEGASUS" where it was 1)");
}

TEST(SolverSettingsSchemaTest, AnOldKeyOfALiveFieldIsRefusedNamingTheField) {
  const absl::Status status = parseTaskFile("multiple_shooting {\n  nThreads: 4\n}\n").status();
  expectRefused(status, "task.textproto:2:");
  expectRefused(status, "n_threads");
}

}  // namespace
}  // namespace ocs2::humanoid
