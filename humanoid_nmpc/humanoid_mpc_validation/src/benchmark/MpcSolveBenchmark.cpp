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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_mpc_validation/benchmark/MpcSolveBenchmark.h"

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"
#include "ocs2_core/automatic_differentiation/CppAdInterface.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/mrt/ControlMode.h"
#include "humanoid_common_mpc_app/robot/RobotController.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetrics.h"

namespace ocs2::humanoid::validation {
namespace {

/** The tape operation count of every library made ready while it is alive, once per library folder. */
class TapeOperationCounter {
 public:
  TapeOperationCounter() {
    CppAdInterface::setLibraryObserver([this](const CppAdInterface& library) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (counts_.contains(library.getLibraryFolder())) return;  // a copy reloads a library already counted
      counts_.emplace(library.getLibraryFolder(), library.getTapeOperationCount());
    });
  }
  ~TapeOperationCounter() { CppAdInterface::setLibraryObserver(CppAdInterface::LibraryObserver()); }

  TapeOperationCounter(const TapeOperationCounter&) = delete;
  TapeOperationCounter& operator=(const TapeOperationCounter&) = delete;

  /** The counts by library folder, sorted. */
  std::map<std::string, size_t> counts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return counts_;
  }

 private:
  mutable std::mutex mutex_;
  std::map<std::string, size_t> counts_;
};

}  // namespace

std::string libraryNameOfFolder(absl::string_view libraryFolder) {
  absl::string_view name = libraryFolder;
  absl::ConsumePrefix(&name, "cppad_code_gen/");
  absl::ConsumeSuffix(&name, "/cppad_generated");
  return std::string(name);
}

absl::StatusOr<JsonValue> runSolveBenchmark(const RobotConfiguration& configuration,
                                            const RecordedRobotStates& states,
                                            const SolveBenchmarkOptions& options,
                                            const std::string& label,
                                            JsonValue provenance) {
  if (states.records.size() <= options.warmupSolves) {
    return absl::InvalidArgumentError(absl::StrCat("[runSolveBenchmark] ", states.records.size(),
                                                   " recorded states leave nothing to time after ", options.warmupSolves,
                                                   " warm-up solves"));
  }
  const TapeOperationCounter tapeOperations;
  ASSIGN_OR_RETURN(std::unique_ptr<ClosedLoopDriver> driver, createClosedLoopDriver(configuration, options.driver));

  std::vector<robot::model::RobotState> robotStates;
  robotStates.reserve(states.records.size());
  for (const RobotStateRecord& record : states.records) {
    ASSIGN_OR_RETURN(robot::model::RobotState robotState, toRobotState(record, states.jointNames, driver->robotDescription()));
    robotStates.push_back(std::move(robotState));
  }

  std::vector<double> wallTimes;
  std::vector<double> lqApproximation;
  std::vector<double> solveQp;
  std::vector<double> linesearch;
  std::vector<double> computeController;
  size_t failedSolves = 0;
  robot::model::RobotJointAction action(driver->robotDescription());
  // The JOINT_PD posture, which WB_MPC holds while it waits for a policy solved after a reset.
  std::vector<scalar_t> nominalPosture(driver->robotDescription().getNumJoints(), 0.0);
  for (size_t joint = 0; joint < nominalPosture.size(); ++joint) {
    nominalPosture[joint] = driver->initialRobotState().getCheckedJointPosition(joint);
  }
  RobotController& controller = driver->robotController();
  for (size_t pass = 0; pass < options.repeats; ++pass) {
    // Each pass starts afresh from the first record, in WB_MPC (the controller's mode after construction): the first
    // pass starts the MPC from it, every later one resets the MPC fully at its first solve.
    driver->startMpc(robotStates.front());
    for (size_t k = 0; k < robotStates.size(); ++k) {
      driver->setGuiVelocityCommand(states.records[k].guiCommand);
      controller.prepareCycle(control_mode::kWbMpc, nominalPosture);
      controller.computeJointControlAction(robotStates[k], action);
      const DriverSolveOutcome outcome = driver->solve();
      if (k < options.warmupSolves) continue;
      if (!outcome.status.ok()) {
        ++failedSolves;
        continue;
      }
      wallTimes.push_back(outcome.wallTimeMs);
      lqApproximation.push_back(outcome.lqApproximationMs);
      solveQp.push_back(outcome.solveQpMs);
      linesearch.push_back(outcome.linesearchMs);
      computeController.push_back(outcome.computeControllerMs);
    }
  }

  JsonValue document = JsonValue::object();
  document.set("schema", JsonValue::string(std::string(kSolveBenchmarkSchemaName)));
  document.set("label", JsonValue::string(label));
  document.set("robot", JsonValue::string(configuration.name));
  document.set("formulation", JsonValue::string(formulationName(configuration.formulation)));
  document.set("provenance", std::move(provenance));

  JsonValue& settings = document.set("settings", JsonValue::object());
  settings.set("solver_threads", JsonValue::number(static_cast<double>(driver->solverThreads())));
  settings.set("mpc_frequency_hz", JsonValue::number(driver->mpcFrequency()));
  settings.set("recorded_states", JsonValue::number(static_cast<double>(states.records.size())));
  settings.set("warmup_solves", JsonValue::number(static_cast<double>(options.warmupSolves)));
  settings.set("repeats", JsonValue::number(static_cast<double>(options.repeats)));
  settings.set("timed_solves", JsonValue::number(static_cast<double>(wallTimes.size())));
  settings.set("failed_solves", JsonValue::number(static_cast<double>(failedSolves)));

  JsonValue& times = document.set("solve_time_ms", JsonValue::object());
  times.set("total", summarizeTimes(wallTimes));
  times.set("lq_approximation", summarizeTimes(lqApproximation));
  times.set("solve_qp", summarizeTimes(solveQp));
  times.set("linesearch", summarizeTimes(linesearch));
  times.set("compute_controller", summarizeTimes(computeController));
  const JsonValue* absl_nullable p99 = times.findPath("total.p99");
  const double periodMs = 1000.0 / driver->mpcFrequency();
  JsonValue& gate = document.set("real_time", JsonValue::object());
  gate.set("mpc_period_ms", JsonValue::number(periodMs));
  gate.set("p99_fraction_of_period",
           JsonValue::optionalNumber(p99 != nullptr && p99->isNumber() ? std::optional<double>(p99->asNumber() / periodMs) : std::nullopt));

  JsonValue& libraries = document.set("tape_operation_counts", JsonValue::object());
  size_t totalOperations = 0;
  for (const std::pair<const std::string, size_t>& library : tapeOperations.counts()) {
    libraries.set(libraryNameOfFolder(library.first), JsonValue::number(static_cast<double>(library.second)));
    totalOperations += library.second;
  }
  document.set("total_tape_operations", JsonValue::number(static_cast<double>(totalOperations)));
  return document;
}

}  // namespace ocs2::humanoid::validation
